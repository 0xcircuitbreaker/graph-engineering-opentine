"""Burr / CrewAI / LlamaIndex / pydantic-graph adapters with synthetic payloads."""

import asyncio
from types import SimpleNamespace

from opentine import Run

from graphforge import GraphRunCapture
from graphforge.adapters import (
    TineBurrHook,
    TineFlowListener,
    record_checkpoints,
    record_graph_run,
)


def _capture(tmp_path, framework):
    return GraphRunCapture(f"{framework} run", framework=framework, runs_dir=tmp_path, strict=True)


# --- Burr ---------------------------------------------------------------------


class _BurrState(dict):
    def serialize(self):
        return dict(self)


def test_burr_hook_records_steps_and_topology(tmp_path):
    capture = _capture(tmp_path, "burr")
    hook = TineBurrHook(capture)
    graph = SimpleNamespace(
        actions=[SimpleNamespace(name="plan"), SimpleNamespace(name="act")],
        transitions=[
            SimpleNamespace(
                from_=SimpleNamespace(name="plan"),
                to=SimpleNamespace(name="act"),
                condition=SimpleNamespace(name="default"),
            )
        ],
    )
    hook.post_application_create(
        app_id="app-1",
        partition_key="pk",
        state=_BurrState(q="hello"),
        application_graph=graph,
        parent_pointer=SimpleNamespace(app_id="parent-app", sequence_id=4),
    )
    hook.pre_run_step(
        sequence_id=0, state=_BurrState(q="hello"), action=SimpleNamespace(name="plan")
    )
    hook.post_run_step(
        sequence_id=0,
        state=_BurrState(q="hello", plan="steps"),
        action=SimpleNamespace(name="plan"),
        result={"plan": "steps"},
        exception=None,
    )
    hook.pre_run_step(
        sequence_id=1,
        state=_BurrState(q="hello", plan="steps"),
        action=SimpleNamespace(name="act"),
    )
    hook.post_run_step(
        sequence_id=1,
        state=_BurrState(q="hello", plan="steps", done=True),
        action=SimpleNamespace(name="act"),
        result=None,
        exception=ValueError("act broke"),
    )
    artifact = capture.end()
    run = Run.load(artifact)
    manifest_graph = run.manifest["graph"]
    assert manifest_graph["forked_from"]["app_id"] == "parent-app"
    assert {n["id"] for n in manifest_graph["nodes"]} == {"plan", "act"}
    steps = list(run.graph.steps.values())
    plan = next(s for s in steps if s.inputs.get("node") == "plan")
    act = next(s for s in steps if s.inputs.get("node") == "act")
    assert act.inputs["edge"]["source"] == "plan"
    assert plan.id in act.parent_ids
    assert "act broke" in str(act.outputs)
    assert act.outputs["state_after"]["done"] is True


# --- CrewAI -------------------------------------------------------------------


def _flow_event(method=None, state_id="flow-1", **extra):
    return SimpleNamespace(
        flow_name="demo_flow",
        method_name=method,
        state={"id": state_id, **extra.pop("state", {})},
        **extra,
    )


def test_crewai_listener_records_per_flow(tmp_path):
    listener = TineFlowListener(runs_dir=tmp_path)
    listener.on_flow_started(_flow_event(state={"topic": "x"}))
    listener.on_method_started(_flow_event(method="generate"))
    listener.on_llm_call_completed(
        SimpleNamespace(
            state={"id": "flow-1"},
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "hi"}],
            response="hello",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            cost=0.001,
        )
    )
    listener.on_method_finished(
        _flow_event(method="generate", result="success", state={"draft": "v1"})
    )
    listener.on_method_started(_flow_event(method="publish"))
    listener.on_method_finished(_flow_event(method="publish", result=None))
    listener.on_flow_finished(_flow_event(result="published", state={"done": True}))

    assert len(listener.artifacts) == 1
    run = Run.load(listener.artifacts[0])
    assert run.manifest["framework"] == "crewai"
    steps = list(run.graph.steps.values())
    generate = next(s for s in steps if s.inputs.get("node") == "generate")
    publish = next(s for s in steps if s.inputs.get("node") == "publish")
    # router label recorded on the edge into the next method
    assert publish.inputs["edge"]["label"] == "success"
    assert publish.inputs["edge"]["source"] == "generate"
    model = next(s for s in steps if s.inputs.get("model") == "claude-sonnet-5")
    assert model.usage == {"input": 10, "output": 5}
    assert generate.id in model.parent_ids


def test_crewai_listener_isolates_interleaved_flows(tmp_path):
    listener = TineFlowListener(runs_dir=tmp_path)
    listener.on_flow_started(_flow_event(state_id="A"))
    listener.on_flow_started(_flow_event(state_id="B"))
    listener.on_method_started(_flow_event(method="step", state_id="A"))
    listener.on_method_started(_flow_event(method="step", state_id="B"))
    listener.on_method_finished(_flow_event(method="step", state_id="A", result="a-result"))
    listener.on_method_finished(_flow_event(method="step", state_id="B", result="b-result"))
    listener.on_flow_finished(_flow_event(state_id="A", result="done-a"))
    listener.on_flow_finished(_flow_event(state_id="B", result="done-b"))
    assert len(listener.artifacts) == 2
    contents = [str(Run.load(p).to_dict()) for p in listener.artifacts]
    assert any("a-result" in c and "b-result" not in c for c in contents)
    assert any("b-result" in c and "a-result" not in c for c in contents)


# --- LlamaIndex Workflows -----------------------------------------------------


class _Ev:
    def __init__(self, **payload):
        self.__dict__.update(payload)


class _StartEvent(_Ev):
    pass


class _DraftEvent(_Ev):
    pass


class _StopEvent(_Ev):
    pass


def test_llamaindex_checkpoints_reconstruct_edges(tmp_path):
    capture = _capture(tmp_path, "llamaindex")
    checkpoints = [
        SimpleNamespace(
            id="c0",
            last_completed_step="draft",
            input_event=_StartEvent(topic="x"),
            output_event=_DraftEvent(text="v1"),
            ctx_state={"state": {"topic": "x", "draft": "v1"}},
        ),
        SimpleNamespace(
            id="c1",
            last_completed_step="review",
            input_event=_DraftEvent(text="v1"),
            output_event=_StopEvent(result="ok"),
            ctx_state={"state": {"topic": "x", "draft": "v1", "review": "ok"}},
        ),
    ]
    artifact = record_checkpoints(capture, checkpoints)
    run = Run.load(artifact)
    steps = list(run.graph.steps.values())
    review = next(
        s for s in steps if s.inputs.get("node") == "review" and "edge" in s.inputs
    )
    assert review.inputs["edge"]["source"] == "draft"
    assert review.inputs["edge"]["label"] == "_DraftEvent"
    draft = next(s for s in steps if s.inputs.get("node") == "draft" and "edge" in s.inputs)
    assert draft.id in review.parent_ids
    markers = [s for s in steps if s.inputs.get("checkpoint")]
    assert {m.inputs["checkpoint"] for m in markers} == {"c0", "c1"}
    assert markers[-1].outputs["state_after"]["review"] == "ok"


def test_llamaindex_accepts_checkpointer_shaped_object(tmp_path):
    capture = _capture(tmp_path, "llamaindex")
    checkpointer = SimpleNamespace(
        checkpoints={
            "run-1": [
                SimpleNamespace(
                    id="only",
                    last_completed_step="solo",
                    input_event=_StartEvent(),
                    output_event=_StopEvent(),
                    ctx_state={"state": {"k": 1}},
                )
            ]
        }
    )
    artifact = record_checkpoints(capture, checkpointer)
    run = Run.load(artifact)
    assert any(s.inputs.get("node") == "solo" for s in run.graph.steps.values())


# --- pydantic-graph -----------------------------------------------------------


class _FakeRun:
    def __init__(self, state, batches):
        self.state = state
        self._batches = batches
        self.next_task = [SimpleNamespace(node_id="__start__", inputs=None, fork_stack=())]
        self.output = None

    def __aiter__(self):
        self._iter = iter(self._batches)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


class _FakeIterCtx:
    def __init__(self, run):
        self._run = run

    async def __aenter__(self):
        return self._run

    async def __aexit__(self, *exc):
        return False


class _EndMarker:
    def __init__(self, value):
        self.value = value


class _FakeGraph:
    def __init__(self, run):
        self._run = run

    def iter(self, *, state=None, deps=None, inputs=None):
        return _FakeIterCtx(self._run)

    def render(self):
        return "stateDiagram-v2\n  a --> b"


def test_pydantic_graph_driver_records_tasks(tmp_path):
    capture = _capture(tmp_path, "pydantic-graph")
    fork_item = SimpleNamespace(fork_id="fork:a", node_run_id="run:1", thread_index=0)
    run = _FakeRun(
        state={"count": 0},
        batches=[
            [SimpleNamespace(node_id="NodeA", task_id="task:1", inputs={"n": 1}, fork_stack=())],
            [
                SimpleNamespace(
                    node_id="NodeB",
                    task_id="task:2",
                    inputs=None,
                    fork_stack=(fork_item,),
                )
            ],
            _EndMarker("final-output"),
        ],
    )
    output, artifact = asyncio.run(
        record_graph_run(capture, _FakeGraph(run), state={"count": 0})
    )
    assert output == "final-output"
    loaded = Run.load(artifact)
    assert loaded.manifest["graph"]["framework"] == "pydantic-graph"
    assert "stateDiagram" in loaded.manifest["graph"]["mermaid"]
    steps = list(loaded.graph.steps.values())
    node_b = next(
        s for s in steps if s.inputs.get("node") == "NodeB" and "edge" in s.inputs
    )
    assert node_b.outputs["outputs"]["fork_stack"][0]["fork_id"] == "fork:a"
    assert node_b.inputs["edge"]["source"] == "NodeA"
