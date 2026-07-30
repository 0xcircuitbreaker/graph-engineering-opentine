from opentine import Run

from graphforge import GraphRunCapture


def test_capture_lifecycle_produces_artifact(tmp_path):
    capture = GraphRunCapture(
        "framework run", framework="customlang", runs_dir=tmp_path, strict=True
    )
    assert capture.is_active
    capture.begin(graph={"nodes": ["a", "b"]}, state={"q": "hello"})
    capture.node_start("a", state={"q": "hello"})
    capture.model_call(model="m1", prompt="p", response="r", cost=0.02, usage={"input": 5})
    capture.node_end("a", update={"draft": 1}, kind="model")
    capture.edge("a", "b", label="next")
    capture.node_start("b")
    capture.tool_call("search", args={"q": "x"}, result=["hit"])
    capture.node_end("b", update={"final": True})
    artifact = capture.end(summary="all done")
    assert artifact is not None and artifact.exists()

    run = Run.load(artifact)
    assert run.manifest["framework"] == "customlang"
    steps = [run.graph.steps[sid] for sid in run.graph.order]
    node_steps = [s for s in steps if s.inputs.get("node") and "model" not in s.inputs]
    a_step = next(s for s in steps if s.inputs.get("node") == "a" and "edge" in s.inputs)
    b_step = next(s for s in steps if s.inputs.get("node") == "b" and "edge" in s.inputs)
    assert b_step.inputs["edge"]["source"] == "a"
    assert b_step.inputs["edge"]["label"] == "next"
    assert a_step.id in b_step.parent_ids
    # nested model call hangs off the node step
    model_step = next(s for s in steps if s.inputs.get("model") == "m1")
    assert a_step.id in model_step.parent_ids
    tool_step = next(s for s in steps if s.inputs.get("tool") == "search")
    assert b_step.id in tool_step.parent_ids
    assert len(node_steps) >= 2
    assert getattr(run.status, "value", str(run.status)) == "completed"


def test_capture_state_flows_between_nodes(tmp_path):
    capture = GraphRunCapture("state flow", runs_dir=tmp_path, strict=True)
    capture.node_start("a", state={"n": 1})
    capture.node_end("a", update={"n": 2})
    capture.node_start("b")  # inherits last state
    step_id = capture.node_end("b", update={"m": 3})
    run = capture.recorder.run
    outputs = run.graph.steps[step_id].outputs
    assert outputs["state_after"] == {"n": 2, "m": 3}
    assert outputs["delta"]["added"] == ["m"]


def test_capture_multi_parent_join(tmp_path):
    capture = GraphRunCapture("join", runs_dir=tmp_path, strict=True)
    capture.node_start("a", state={})
    a_id = capture.node_end("a", update={"a": 1})
    capture.node_start("b", state={})
    b_id = capture.node_end("b", update={"b": 1})
    capture.node_start("merge")
    merge_id = capture.node_end("merge", parents=[a_id, b_id])
    step = capture.recorder.run.graph.steps[merge_id]
    assert set(step.parent_ids) == {a_id, b_id}


def test_capture_never_throws_and_counts_drops(tmp_path):
    capture = GraphRunCapture("hostile", runs_dir=tmp_path, strict=False)
    # node_end for a node never started with junk metrics and unserializable state
    capture.node_end(
        "ghost",
        state={"weird": object()},
        cost=float("nan"),
        duration=-3,
        usage={"input": "many"},
    )
    capture.error("late failure")
    artifact = capture.end(status="failed", summary="host still alive")
    # whatever happened internally, the host never saw an exception
    assert capture.dropped_events >= 0
    assert artifact is None or artifact.exists()


def test_capture_checkpoint_marker(tmp_path):
    capture = GraphRunCapture("ckpt", runs_dir=tmp_path, strict=True)
    capture.checkpoint("c1", state={"k": 1}, parent_checkpoint_id="c0", source="loop", step=2)
    run = capture.recorder.run
    step = run.graph.steps[run.graph.order[-1]]
    assert step.inputs["checkpoint"] == "c1"
    assert step.outputs["state_after"] == {"k": 1}


def test_capture_error_status(tmp_path):
    capture = GraphRunCapture("err", runs_dir=tmp_path, strict=True)
    capture.node_start("a", state={})
    capture.node_end("a")
    capture.error(RuntimeError("boom"), node_id="a")
    artifact = capture.end(status="failed")
    run = Run.load(artifact)
    assert getattr(run.status, "value", str(run.status)) == "failed"
