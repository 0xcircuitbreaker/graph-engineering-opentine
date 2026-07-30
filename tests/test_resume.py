import pytest
from opentine import Run

from graphforge import GraphEngine, GraphExecutionError, GraphNodeResult, GraphSpec
from graphforge.examples import refinement_graph


def test_resume_forks_and_continues(tmp_path):
    engine = GraphEngine(refinement_graph(42.0), runs_dir=tmp_path)
    original = engine.run(goal="hit 42", initial_state={"x": 0.0})
    assert original.status == "completed"

    # Fork at the first evaluate step and push the state off-course.
    run = Run.load(original.artifact)
    evaluate_steps = [
        sid for sid in run.graph.order if run.graph.steps[sid].inputs.get("node") == "evaluate"
    ]
    fork_point = evaluate_steps[0]
    resumed = engine.resume(
        original.artifact, fork_point, state_update={"x": -100.0, "distance": 142.0}
    )
    assert resumed.status == "completed"
    assert abs(resumed.final_state["x"] - 42.0) <= 0.5
    forked_run = Run.load(resumed.artifact)
    assert forked_run.metadata.get("forked_from") == original.run_id
    assert fork_point in forked_run.graph.steps


def test_resume_accepts_step_prefix_and_start_node(tmp_path):
    engine = GraphEngine(refinement_graph(10.0), runs_dir=tmp_path)
    original = engine.run(goal="ten", initial_state={"x": 0.0})
    run = Run.load(original.artifact)
    node_steps = [sid for sid in run.graph.order if run.graph.steps[sid].inputs.get("node")]
    prefix = node_steps[0][:16]
    resumed = engine.resume(original.artifact, prefix, start_node="evaluate")
    assert resumed.status == "completed"
    assert resumed.path[0]["node"] == "evaluate"


def test_resume_without_snapshot_errors(tmp_path):
    graph = GraphSpec().add_node(
        "a", lambda ctx: GraphNodeResult(update={"k": 1}, stop=True)
    ).set_entry("a")
    engine = GraphEngine(graph, runs_dir=tmp_path, state_snapshots=False)
    result = engine.run(goal="nosnap", initial_state={})
    step_id = result.path[0]["step_id"]
    with pytest.raises(GraphExecutionError, match="state_after"):
        engine.resume(result.artifact, step_id)


def test_resume_unknown_step_errors(tmp_path):
    engine = GraphEngine(refinement_graph(5.0), runs_dir=tmp_path)
    result = engine.run(goal="five", initial_state={"x": 0.0})
    with pytest.raises(GraphExecutionError, match="not found"):
        engine.resume(result.artifact, "zzzzzz")


def test_resume_flags_topology_drift(tmp_path):
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
    assert "graph:topology-drift" in Run.load(resumed.artifact).tags
