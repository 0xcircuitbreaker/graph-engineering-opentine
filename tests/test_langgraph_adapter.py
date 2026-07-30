"""Adapter tests with synthetic LangGraph/LangChain payloads (no langgraph install)."""

from types import SimpleNamespace
from uuid import uuid4

from opentine import Run

from graphforge import GraphRunCapture
from graphforge.adapters import RecordingCallbackHandler, RecordingCheckpointer


def _capture(tmp_path):
    return GraphRunCapture("langgraph run", framework="langgraph", runs_dir=tmp_path, strict=True)


def test_callback_handler_records_nodes_and_models(tmp_path):
    capture = _capture(tmp_path)
    handler = RecordingCallbackHandler(capture, price_table={"gpt-x": (1.0, 2.0)})

    node_run = uuid4()
    handler.on_chain_start(
        {"name": "agent"},
        {"messages": ["hi"]},
        run_id=node_run,
        parent_run_id=None,
        metadata={
            "langgraph_node": "agent",
            "langgraph_step": 1,
            "langgraph_triggers": ["branch:to:agent"],
        },
    )
    llm_run = uuid4()
    handler.on_chat_model_start(
        {"name": "ChatModel"},
        [["hi"]],
        run_id=llm_run,
        parent_run_id=node_run,
        metadata={"ls_model_name": "gpt-x"},
    )
    response = SimpleNamespace(
        generations=[
            [
                SimpleNamespace(
                    text="hello!",
                    message=SimpleNamespace(
                        usage_metadata={"input_tokens": 1000, "output_tokens": 500}
                    ),
                )
            ]
        ],
        llm_output={},
    )
    handler.on_llm_end(response, run_id=llm_run)
    handler.on_chain_end({"messages": ["hello!"]}, run_id=node_run)
    artifact = capture.end()

    run = Run.load(artifact)
    steps = [run.graph.steps[sid] for sid in run.graph.order]
    node_step = next(s for s in steps if s.inputs.get("node") == "agent" and "edge" in s.inputs)
    assert node_step.inputs["edge"]["source"] == "branch:to:agent"
    model_step = next(s for s in steps if s.inputs.get("model") == "gpt-x")
    assert model_step.usage == {"input": 1000, "output": 500}
    assert abs(model_step.cost - (1000 * 1.0 + 500 * 2.0) / 1_000_000) < 1e-12
    assert node_step.id in model_step.parent_ids


def test_callback_handler_node_error(tmp_path):
    capture = _capture(tmp_path)
    handler = RecordingCallbackHandler(capture)
    node_run = uuid4()
    handler.on_chain_start(
        {}, {}, run_id=node_run, metadata={"langgraph_node": "worker"}
    )
    handler.on_chain_error(ValueError("node blew up"), run_id=node_run)
    artifact = capture.end(status="failed")
    run = Run.load(artifact)
    worker = next(
        s
        for s in run.graph.steps.values()
        if s.inputs.get("node") == "worker" and "edge" in s.inputs
    )
    assert "node blew up" in str(worker.outputs)


def test_callback_handler_ignores_non_node_chains(tmp_path):
    capture = _capture(tmp_path)
    handler = RecordingCallbackHandler(capture)
    run_id = uuid4()
    handler.on_chain_start({}, {}, run_id=run_id, metadata={})
    handler.on_chain_end({}, run_id=run_id)
    assert len(capture.recorder.run.graph.order) == 0


def test_checkpointer_records_checkpoints_and_delegates(tmp_path):
    capture = _capture(tmp_path)

    class FakeInner:
        def __init__(self):
            self.puts = []

        def put(self, config, checkpoint, metadata, new_versions):
            self.puts.append(checkpoint["id"])
            return config

        def put_writes(self, config, writes, task_id, task_path=""):
            return None

        def get_tuple(self, config):
            return "TUPLE"

        def list(self, config, **kwargs):
            return iter(["A"])

    inner = FakeInner()
    saver = RecordingCheckpointer(capture, inner=inner)
    config = {"configurable": {"thread_id": "t1", "checkpoint_id": "ckpt-0"}}
    checkpoint = {"id": "ckpt-1", "channel_values": {"messages": ["hi"], "count": 2}}
    saver.put(config, checkpoint, {"source": "loop", "step": 1}, {})
    assert inner.puts == ["ckpt-1"]
    assert saver.get_tuple(config) == "TUPLE"
    assert list(saver.list(config)) == ["A"]

    run = capture.recorder.run
    step = run.graph.steps[run.graph.order[-1]]
    assert step.inputs["checkpoint"] == "ckpt-1"
    assert step.inputs["parent_checkpoint"] == "ckpt-0"
    assert step.inputs["source"] == "loop"
    assert step.outputs["state_after"] == {"messages": ["hi"], "count": 2}


def test_nested_runnables_are_not_duplicate_node_steps(tmp_path):
    """Runnables inside a node inherit langgraph_node metadata; only the
    outermost run is the node execution."""
    capture = _capture(tmp_path)
    handler = RecordingCallbackHandler(capture)
    node_run, inner_run, llm_run = uuid4(), uuid4(), uuid4()
    handler.on_chain_start(
        {}, {"q": 1}, run_id=node_run, metadata={"langgraph_node": "agent"}
    )
    handler.on_chain_start(
        {},
        {"q": 1},
        run_id=inner_run,
        parent_run_id=node_run,
        metadata={"langgraph_node": "agent"},  # inherited
    )
    handler.on_chat_model_start(
        {"name": "chat"}, [["hi"]], run_id=llm_run, parent_run_id=inner_run, metadata={}
    )
    response = SimpleNamespace(generations=[[SimpleNamespace(text="ok", message=None)]])
    handler.on_llm_end(response, run_id=llm_run)
    handler.on_chain_end({}, run_id=inner_run)
    handler.on_chain_end({"out": 1}, run_id=node_run)
    artifact = capture.end()
    run = Run.load(artifact)
    node_steps = [
        s
        for s in run.graph.steps.values()
        if s.inputs.get("node") == "agent" and "edge" in s.inputs
    ]
    assert len(node_steps) == 1
    # nested model call still attributed to the enclosing node step
    model_step = next(s for s in run.graph.steps.values() if "model" in s.inputs)
    assert node_steps[0].id in model_step.parent_ids


def test_usage_not_multiplied_across_candidate_generations(tmp_path):
    capture = _capture(tmp_path)
    handler = RecordingCallbackHandler(capture)
    llm_run = uuid4()
    handler.on_chat_model_start({"name": "chat"}, [["hi"]], run_id=llm_run, metadata={})
    meta = {"input_tokens": 1000, "output_tokens": 500}
    response = SimpleNamespace(
        generations=[
            [
                SimpleNamespace(text="a", message=SimpleNamespace(usage_metadata=dict(meta))),
                SimpleNamespace(text="b", message=SimpleNamespace(usage_metadata=dict(meta))),
            ]
        ],
        llm_output={},
    )
    handler.on_llm_end(response, run_id=llm_run)
    run = capture.recorder.run
    model_step = next(s for s in run.graph.steps.values() if "model" in s.inputs)
    assert model_step.usage == {"input": 1000, "output": 500}  # not 2000/1000


def test_send_fanout_model_calls_attributed_per_task(tmp_path):
    """Two concurrent executions of the same node (Send API): nested calls
    must land on their own task's step, keyed by the node run id."""
    capture = _capture(tmp_path)
    handler = RecordingCallbackHandler(capture)
    task1, task2 = uuid4(), uuid4()
    llm1, llm2 = uuid4(), uuid4()
    handler.on_chain_start(
        {}, {"item": 1}, run_id=task1, metadata={"langgraph_node": "worker"}
    )
    handler.on_chain_start(
        {}, {"item": 2}, run_id=task2, metadata={"langgraph_node": "worker"}
    )
    handler.on_chat_model_start({}, [["1"]], run_id=llm1, parent_run_id=task1, metadata={})
    handler.on_chat_model_start({}, [["2"]], run_id=llm2, parent_run_id=task2, metadata={})
    resp = lambda text: SimpleNamespace(  # noqa: E731
        generations=[[SimpleNamespace(text=text, message=None)]], llm_output={}
    )
    handler.on_llm_end(resp("one"), run_id=llm1)
    handler.on_llm_end(resp("two"), run_id=llm2)
    handler.on_chain_end({"out": 1}, run_id=task1)
    handler.on_chain_end({"out": 2}, run_id=task2)
    artifact = capture.end()
    run = Run.load(artifact)
    model_steps = [s for s in run.graph.steps.values() if "model" in s.inputs]
    assert len(model_steps) == 2
    responses = {str(s.outputs.get("response")) for s in model_steps}
    assert responses == {"one", "two"}
    # each model call has a distinct node-step parent
    parents = {p for s in model_steps for p in s.parent_ids}
    assert len(parents) == 2


def test_checkpointer_works_without_inner(tmp_path):
    capture = _capture(tmp_path)
    saver = RecordingCheckpointer(capture)
    config = {"configurable": {"thread_id": "t1"}}
    saver.put(config, {"id": "c1", "channel_values": {"x": 1}}, {"source": "input", "step": -1}, {})
    assert saver.get_tuple(config) is None
    assert list(saver.list(config)) == []
    assert len(capture.recorder.run.graph.order) == 1
