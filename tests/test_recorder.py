from opentine import Run

from graphforge import GraphPolicy, GraphRecorder


def test_roundtrip_save_and_verify(tmp_path):
    recorder = GraphRecorder("goal", context="ctx", runs_dir=tmp_path)
    first = recorder.record_node("a", state_before={}, state_after={"k": 1})
    recorder.record_node("b", edge={"kind": "static", "source": "a"}, parent_id=first)
    recorder.record_done("done")
    path = recorder.save()
    assert path.exists()
    assert Run.verify_integrity(path).ok
    loaded = Run.load(path)
    assert loaded.manifest["kind"] == "graphforge"
    assert loaded.manifest["goal"] == "goal"
    assert getattr(loaded.status, "value", str(loaded.status)) == "completed"


def test_metrics_sanitized_for_strict_opentine(tmp_path):
    recorder = GraphRecorder("metrics", runs_dir=tmp_path)
    step_id = recorder.record_node(
        "n",
        cost=-5.0,
        duration=float("inf"),
        usage={"input": 3.7, "output": True, "bad": -1, "reasoning": 2},
    )
    step = recorder.run.graph.steps[step_id]
    assert step.cost == 0.0
    assert step.duration == 0.0
    assert step.usage == {"input": 3, "reasoning": 2}


def test_fork_preserves_ancestry_and_policy(tmp_path):
    recorder = GraphRecorder("orig", policy=GraphPolicy(max_node_executions=5), runs_dir=tmp_path)
    a = recorder.record_node("a", state_after={"x": 1})
    recorder.record_node("b", parent_id=a, state_after={"x": 2})
    forked = GraphRecorder.from_fork(
        recorder.run, a, branch="alt", policy=GraphPolicy(max_node_executions=9), runs_dir=tmp_path
    )
    assert forked.run_id != recorder.run_id
    assert a in forked.run.graph.steps
    assert forked.run.metadata.get("forked_from") == recorder.run_id
    assert forked.run.policies["graph"]["max_node_executions"] == 9
    assert forked.tip == a


def test_multi_parent_join_step(tmp_path):
    recorder = GraphRecorder("join", runs_dir=tmp_path)
    root = recorder.record_node("root")
    a = recorder.record_node("a", parent_id=root)
    b = recorder.record_node("b", parent_id=root)
    join = recorder.record_node("join", parent_ids=[a, b])
    step = recorder.run.graph.steps[join]
    assert set(step.parent_ids) == {a, b}


def test_error_sets_failed_status(tmp_path):
    recorder = GraphRecorder("err", runs_dir=tmp_path)
    recorder.record_node("a")
    recorder.record_error(ValueError("blew up"))
    assert recorder.finished
    path = recorder.save()
    loaded = Run.load(path)
    assert getattr(loaded.status, "value", str(loaded.status)) == "failed"


def test_gate_records_tag_once(tmp_path):
    recorder = GraphRecorder("gate", runs_dir=tmp_path)
    recorder.record_gate("max_total_cost", {"total_cost": 1.0})
    recorder.record_gate("max_total_cost", {"total_cost": 1.1})
    assert recorder.run.tags.count("gate:max_total_cost") == 1


def test_state_snapshot_optional(tmp_path):
    recorder = GraphRecorder("nosnap", runs_dir=tmp_path, state_snapshots=False)
    step_id = recorder.record_node("a", state_after={"secret": "big"})
    outputs = recorder.run.graph.steps[step_id].outputs
    assert "state_after" not in outputs
    assert outputs["state_digest"]


def test_checkpoint_marker(tmp_path):
    recorder = GraphRecorder("ckpt", runs_dir=tmp_path)
    step_id = recorder.record_checkpoint(
        "ckpt-1", state={"k": 1}, parent_checkpoint_id="ckpt-0", source="loop", step=3
    )
    step = recorder.run.graph.steps[step_id]
    assert step.inputs["checkpoint"] == "ckpt-1"
    assert step.inputs["parent_checkpoint"] == "ckpt-0"
    assert step.outputs["state_after"] == {"k": 1}
