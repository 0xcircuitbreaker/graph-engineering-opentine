"""Regression tests for the confirmed findings of the adversarial review."""

import json

import pytest
from opentine import Run, RunStatus, StepKind

from graphforge import (
    END,
    GraphEngine,
    GraphExecutionError,
    GraphNodeResult,
    GraphPolicy,
    GraphRecorder,
    GraphRunCapture,
    GraphSpec,
    extract_json,
)
from graphforge.examples import refinement_graph
from graphforge.state import MAX_SAFE_INT, sane_usage, to_jsonable


def _mark(name, update=None, **result_kwargs):
    def fn(ctx):
        return GraphNodeResult(update=update or {name: True}, **result_kwargs)

    return fn


# --- engine: join semantics -------------------------------------------------


def test_chained_joins_fire_upstream_first(tmp_path):
    """j2 must wait for j1 even when j2's first arrival parks earlier."""
    graph = (
        GraphSpec()
        .add_node("start", _mark("start"))
        .add_node("a", _mark("a"))
        .add_node("b", _mark("b"))
        .add_node("c", _mark("c"))
        .add_node("j1", _mark("j1"), join=True)
        .add_node("j2", _mark("j2"), join=True)
        .add_edge("start", "a")
        .add_edge("start", "b")
        .add_edge("start", "c")
        .add_edge("a", "j2")
        .add_edge("b", "j1")
        .add_edge("c", "j1")
        .add_edge("j1", "j2")
        .add_edge("j2", END)
        .set_entry("start")
    )
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal="chained", initial_state={})
    j2_hops = [h for h in result.path if h["node"] == "j2"]
    assert len(j2_hops) == 1, f"j2 executed {len(j2_hops)} times: {result.path}"
    # full merge: every branch's key present
    assert all(k in result.final_state for k in ("start", "a", "b", "c", "j1", "j2"))
    run = Run.load(result.artifact)
    j2_step = next(s for s in run.graph.steps.values() if s.inputs.get("node") == "j2")
    assert len(j2_step.parent_ids) == 2


def test_early_stop_with_pending_join_is_loud(tmp_path):
    def stopper(ctx):
        return GraphNodeResult(update={"b2": True}, stop=True)

    graph = (
        GraphSpec()
        .add_node("start", _mark("start"))
        .add_node("a", _mark("a"))
        .add_node("b1", _mark("b1"))
        .add_node("b2", stopper)
        .add_node("j", _mark("j"), join=True)
        .add_edge("start", "a")
        .add_edge("start", "b1")
        .add_edge("a", "j")
        .add_edge("b1", "b2")
        .add_edge("j", END)
        .set_entry("start")
    )
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal="early stop", initial_state={})
    assert result.stop_reason == "stop"
    assert "graph:unresolved_join" in Run.load(result.artifact).tags
    # parked branch state preserved, final_state stays the stopping branch's
    assert any("a" in s for s in result.final_states)
    assert result.final_state.get("b2") is True


# --- engine: user-callback failures never lose the artifact ------------------


def test_nan_score_does_not_crash_run(tmp_path):
    def scorer(ctx):
        return GraphNodeResult(score=float("nan"), update={"done": True}, stop=True)

    graph = GraphSpec().add_node("s", scorer).set_entry("s")
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal="nan", initial_state={})
    assert result.status == "completed"
    assert result.artifact.exists()
    assert result.best is None  # NaN never becomes best
    run = Run.load(result.artifact)
    step = next(s for s in run.graph.steps.values() if s.inputs.get("node") == "s")
    assert "score" not in step.outputs


def test_nan_score_never_locks_best(tmp_path):
    def first(ctx):
        return GraphNodeResult(score=float("nan"))

    def second(ctx):
        return GraphNodeResult(score=0.4, stop=True)

    graph = (
        GraphSpec()
        .add_node("first", first)
        .add_node("second", second)
        .add_edge("first", "second")
        .set_entry("first")
    )
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal="best", initial_state={})
    assert result.best is not None and result.best.score == 0.4


def test_router_exception_records_error_and_saves(tmp_path):
    def bad_router(state):
        raise KeyError("router blew up")

    graph = (
        GraphSpec()
        .add_node("a", _mark("a"))
        .add_router("a", bad_router)
        .set_entry("a")
    )
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal="router", initial_state={})
    assert result.status == "failed"
    assert result.stop_reason == "route_error"
    assert result.artifact.exists()
    run = Run.load(result.artifact)
    assert getattr(run.status, "value", str(run.status)) == "failed"


def test_join_reducer_exception_records_error_and_saves(tmp_path):
    def bad_reducer(states):
        raise ValueError("reducer blew up")

    graph = (
        GraphSpec()
        .add_node("start", _mark("start"))
        .add_node("x", _mark("x"))
        .add_node("y", _mark("y"))
        .add_node("j", _mark("j"), join=True, join_reducer=bad_reducer)
        .add_edge("start", "x")
        .add_edge("start", "y")
        .add_edge("x", "j")
        .add_edge("y", "j")
        .set_entry("start")
    )
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal="reducer", initial_state={})
    assert result.status == "failed"
    assert result.stop_reason == "join_error"
    assert result.artifact.exists()


# --- canonical-JSON safety ---------------------------------------------------


def test_huge_int_state_survives_save_and_load(tmp_path):
    def big(ctx):
        return GraphNodeResult(update={"wei": 2**60, "ok": 1}, stop=True)

    graph = GraphSpec().add_node("big", big).set_entry("big")
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal="big int", initial_state={})
    assert result.status == "completed"
    run = Run.load(result.artifact)
    step = next(s for s in run.graph.steps.values() if s.inputs.get("node") == "big")
    assert step.outputs["state_after"]["wei"] == str(2**60)  # encoded as string


def test_sane_usage_clamps_to_canonical_range():
    assert sane_usage({"input": 2**60})["input"] == MAX_SAFE_INT
    assert to_jsonable(2**60) == str(2**60)
    assert to_jsonable(-(2**60)) == str(-(2**60))
    assert to_jsonable(MAX_SAFE_INT) == MAX_SAFE_INT


# --- resume hardening ---------------------------------------------------------


def test_resume_continues_visit_numbering(tmp_path):
    engine = GraphEngine(refinement_graph(42.0), runs_dir=tmp_path)
    original = engine.run(goal="42", initial_state={"x": 0.0})
    run = Run.load(original.artifact)
    evaluate_steps = [
        sid for sid in run.graph.order if run.graph.steps[sid].inputs.get("node") == "evaluate"
    ]
    resumed = engine.resume(
        original.artifact, evaluate_steps[-1], state_update={"x": -50.0, "distance": 92.0}
    )
    forked = Run.load(resumed.artifact)
    pairs = [
        (s.inputs["node"], s.inputs["visit"])
        for s in forked.graph.steps.values()
        if isinstance(s.inputs, dict) and "visit" in s.inputs and s.inputs.get("node")
    ]
    assert len(pairs) == len(set(pairs)), f"duplicate (node, visit) pairs: {sorted(pairs)}"


def test_resumed_manifest_describes_executing_graph(tmp_path):
    engine = GraphEngine(refinement_graph(8.0), runs_dir=tmp_path)
    original = engine.run(goal="eight", initial_state={"x": 0.0})
    run = Run.load(original.artifact)
    node_step = next(
        sid for sid in run.graph.order if run.graph.steps[sid].inputs.get("node") == "evaluate"
    )
    drifted_graph = refinement_graph(8.0).add_node(
        "audit", lambda ctx: GraphNodeResult(observation="audit")
    )
    drifted = GraphEngine(drifted_graph, runs_dir=tmp_path)
    resumed = drifted.resume(original.artifact, node_step)
    manifest = Run.load(resumed.artifact).manifest
    assert manifest["graph"]["topology_digest"] == drifted_graph.topology_digest()
    assert manifest["graph_at_record"]["topology_digest"] != drifted_graph.topology_digest()


def test_resume_rejects_redacted_snapshot(tmp_path):
    engine = GraphEngine(refinement_graph(5.0), runs_dir=tmp_path)
    original = engine.run(goal="five", initial_state={"x": 0.0, "session_key": "hunter2secret"})
    run = Run.load(original.artifact)
    node_step = next(
        sid for sid in run.graph.order if run.graph.steps[sid].inputs.get("node") == "evaluate"
    )
    # Simulate opentine's save-time redaction on the artifact.
    payload = json.loads(original.artifact.read_text())

    replaced = []

    def redact(value):
        if isinstance(value, dict):
            for key, inner in value.items():
                if key == "state_after" and isinstance(inner, dict) and "session_key" in inner:
                    inner["session_key"] = "[REDACTED]"
                    replaced.append(True)
                else:
                    redact(inner)
        elif isinstance(value, list):
            for item in value:
                redact(item)

    redact(payload)
    assert replaced, "test setup: no state_after with session_key found"
    tampered = tmp_path / "tampered.tine"
    tampered.write_text(json.dumps(payload))

    with pytest.raises(GraphExecutionError, match="redact"):
        engine.resume(tampered, node_step)
    # supplying the redacted key explicitly is allowed and tagged
    resumed = engine.resume(tampered, node_step, state_update={"session_key": "hunter2secret"})
    assert "graph:snapshot-redacted" in Run.load(resumed.artifact).tags


# --- recorder ------------------------------------------------------------------


def test_checkpoint_state_not_truncated(tmp_path):
    recorder = GraphRecorder("big ckpt", runs_dir=tmp_path)
    big_value = "x" * 20_000
    step_id = recorder.record_checkpoint("c1", state={"blob": big_value})
    stored = recorder.run.graph.steps[step_id].outputs["state_after"]["blob"]
    assert stored == big_value


def test_identical_repeated_calls_get_distinct_steps(tmp_path):
    recorder = GraphRecorder("dupes", runs_dir=tmp_path)
    parent = recorder.record_node("n")
    first = recorder.record_model_call(model="m", prompt="same", response="same", parent_id=parent)
    second = recorder.record_model_call(model="m", prompt="same", response="same", parent_id=parent)
    assert first != second
    t1 = recorder.record_tool_call("t", args={"a": 1}, result="r", parent_id=parent)
    t2 = recorder.record_tool_call("t", args={"a": 1}, result="r", parent_id=parent)
    assert t1 != t2


def test_two_forks_at_same_step_do_not_collide(tmp_path):
    recorder = GraphRecorder("parent", runs_dir=tmp_path)
    a = recorder.record_node("a", state_after={"x": 1})
    recorder.record_done("done")
    fork1 = GraphRecorder.from_fork(recorder.run, a, branch="one", runs_dir=tmp_path)
    fork2 = GraphRecorder.from_fork(recorder.run, a, branch="one", runs_dir=tmp_path)
    assert fork1.run_id != fork2.run_id


def test_save_refuses_directory_target(tmp_path):
    recorder = GraphRecorder("dir", runs_dir=tmp_path)
    recorder.record_done("done")
    with pytest.raises(ValueError, match="directory"):
        recorder.save(tmp_path)


# --- capture --------------------------------------------------------------------


def test_capture_end_flushes_open_nodes(tmp_path):
    capture = GraphRunCapture("flush", runs_dir=tmp_path, strict=True)
    capture.node_start("open_node", state={})
    capture.model_call(model="m", prompt="p", response="r", node_id="open_node")
    artifact = capture.end(summary="host crashed mid-node")
    run = Run.load(artifact)
    node_step = next(
        s for s in run.graph.steps.values() if s.inputs.get("node") == "open_node"
        and "edge" in s.inputs
    )
    assert node_step.outputs["outputs"]["incomplete"] is True
    model_step = next(s for s in run.graph.steps.values() if s.inputs.get("model") == "m")
    assert node_step.id in model_step.parent_ids


def test_capture_reopen_closes_previous_execution(tmp_path):
    capture = GraphRunCapture("reopen", runs_dir=tmp_path, strict=True)
    capture.node_start("n", state={})
    capture.model_call(model="m1", prompt="p1", response="r1", node_id="n")
    capture.node_start("n", state={})  # re-open before close
    capture.node_end("n")
    capture.end()
    run = capture.recorder.run
    n_steps = [
        s for s in run.graph.steps.values() if s.inputs.get("node") == "n" and "edge" in s.inputs
    ]
    assert len(n_steps) == 2  # both executions recorded
    assert any((s.outputs.get("outputs") or {}).get("incomplete") for s in n_steps)
    # buffered call from the first execution survived
    assert any(s.inputs.get("model") == "m1" for s in run.graph.steps.values())


# --- surfaces --------------------------------------------------------------------


def test_build_server_smoke(tmp_path):
    import asyncio

    from graphforge.mcp_server import build_server

    server = build_server(str(tmp_path))
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "list_graph_runs",
        "show_graph_run",
        "show_graph_topology",
        "diff_graph_runs",
        "fork_graph_run",
    }


def test_mcp_fork_twice_distinct_artifacts(tmp_path):
    from graphforge.mcp_server import fork_run

    engine = GraphEngine(refinement_graph(5.0), runs_dir=tmp_path)
    original = engine.run(goal="five", initial_state={"x": 0.0})
    run = Run.load(original.artifact)
    node_step = next(sid for sid in run.graph.order if run.graph.steps[sid].inputs.get("node"))
    # Full run id: fork artifacts are named {parent_prefix}-{branch}-{uuid},
    # so short prefixes of the parent become ambiguous by design.
    one = fork_run(original.run_id, node_step[:12], "same-branch", str(tmp_path))
    two = fork_run(original.run_id, node_step[:12], "same-branch", str(tmp_path))
    assert one["run_id"] != two["run_id"]
    assert one["path"] != two["path"]


def test_foreign_artifact_does_not_crash_surfaces(tmp_path):
    from graphforge.mcp_server import list_runs, show_graph, show_run

    foreign = Run(manifest={"kind": "other"}, status=RunStatus.completed, user_prompt="p")
    foreign.add_step(
        StepKind.think, inputs={"node": 42, "edge": "not-a-dict"}, outputs={"x": 1}
    )
    foreign.add_step(StepKind.done, inputs={"weird": object.__class__.__name__})
    foreign.save(tmp_path / "foreign.tine")
    assert list_runs(str(tmp_path))
    shown = show_run("foreign", str(tmp_path))
    assert shown["step_count"] == 2
    graph_view = show_graph("foreign", str(tmp_path))
    assert "flowchart TD" in graph_view["mermaid_trail"]


def test_extract_json_survives_trailing_stray_brace():
    text = 'result: {"a": 1, "note": "ok"} and later a stray }'
    assert extract_json(text) == {"a": 1, "note": "ok"}
    nested = 'x {"outer": {"inner": 2}} trailing } noise'
    assert extract_json(nested) == {"outer": {"inner": 2}}


def test_static_adapter_reset():
    from graphforge import StaticModelAdapter

    adapter = StaticModelAdapter(script=["one", "two"])
    assert adapter.complete("p").text == "one"
    adapter.reset()
    assert adapter.complete("p").text == "one"


def test_policy_gate_artifact_still_saved_under_run_wide_budget(tmp_path):
    graph = (
        GraphSpec()
        .add_node("spin", lambda ctx: GraphNodeResult(goto="spin", cost=0.1))
        .set_entry("spin")
    )
    engine = GraphEngine(
        graph,
        policy=GraphPolicy(max_total_cost=0.25, max_visits_per_node=None),
        runs_dir=tmp_path,
    )
    result = engine.run(goal="budget", initial_state={})
    assert result.stop_reason == "max_total_cost"
    assert result.artifact.exists()
