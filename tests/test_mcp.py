import pytest

from graphforge import GraphEngine
from graphforge.examples import fanout_graph, refinement_graph
from graphforge.mcp_server import diff_runs, fork_run, list_runs, show_graph, show_run


@pytest.fixture()
def runs(tmp_path):
    refine = GraphEngine(refinement_graph(10.0), runs_dir=tmp_path).run(
        goal="ten", initial_state={"x": 0.0}
    )
    fanout = GraphEngine(fanout_graph(), runs_dir=tmp_path).run(
        goal="q", initial_state={"question": "q"}
    )
    return tmp_path, refine, fanout


def test_list_runs(runs):
    tmp_path, refine, fanout = runs
    entries = list_runs(str(tmp_path))
    assert len(entries) == 2
    ids = {e["run_id"] for e in entries}
    assert {refine.run_id, fanout.run_id} <= ids
    assert all(e["kind"] == "graphforge" for e in entries)


def test_show_run_by_prefix(runs):
    tmp_path, refine, _ = runs
    shown = show_run(refine.run_id[:10], str(tmp_path))
    assert shown["run_id"] == refine.run_id
    assert shown["graph"]["entry"] == "propose"
    assert any(s["node"] == "evaluate" for s in shown["steps"])


def test_show_graph_trail(runs):
    tmp_path, refine, _ = runs
    result = show_graph(refine.run_id[:10], str(tmp_path))
    assert result["topology"]["name"].startswith("refine-to")
    assert any(hop["node"] == "propose" for hop in result["trail"])
    assert "flowchart TD" in result["mermaid_trail"]
    assert "propose" in result["mermaid_trail"]


def test_show_graph_renders_join_fan_in(runs):
    tmp_path, _, fanout = runs
    mermaid = show_graph(fanout.run_id[:10], str(tmp_path))["mermaid_trail"]
    assert "research_a -- join --> merge" in mermaid
    assert "research_b -- join --> merge" in mermaid


def test_diff_runs_shape(runs):
    tmp_path, refine, fanout = runs
    diff = diff_runs(refine.run_id[:10], fanout.run_id[:10], str(tmp_path))
    assert diff["left"] == refine.run_id
    assert diff["right"] == fanout.run_id
    # regression pin on opentine FieldDelta/StepChange attribute names
    for change in diff["changed"]:
        assert set(change) == {"step_a", "step_b", "fields"}
        for field in change["fields"]:
            assert set(field) == {"name", "before", "after", "changed_keys"}


def test_fork_run_produces_new_artifact(runs):
    tmp_path, refine, _ = runs
    from opentine import Run

    run = Run.load(refine.artifact)
    node_step = next(
        sid for sid in run.graph.order if run.graph.steps[sid].inputs.get("node")
    )
    result = fork_run(refine.run_id[:10], node_step[:12], "test-fork", str(tmp_path))
    assert result["fork_point"] == node_step
    forked = Run.load(result["path"])
    assert forked.metadata.get("forked_from") == refine.run_id
    assert node_step in forked.graph.steps


def test_ambiguous_and_missing_refs(runs):
    tmp_path, _, _ = runs
    with pytest.raises(FileNotFoundError):
        show_run("does-not-exist", str(tmp_path))


def test_opentine_surface_regression_pins(tmp_path):
    """Pin the exact opentine attributes graphforge relies on (0.2.x and 0.3.x)."""
    from opentine import Run, RunStatus, StepKind

    run = Run(manifest={"kind": "pin"}, status=RunStatus.running, user_prompt="p", policies={})
    step = run.add_step(
        StepKind.think,
        inputs={"node": "a"},
        outputs={"observation": "o"},
        parent_id=None,
        duration=0.0,
        cost=0.0,
    )
    assert step.id
    child = run.add_step(StepKind.done, inputs={}, parent_ids=[step.id])
    assert step.id in child.parent_ids
    forked = run.fork(from_step_id=step.id, branch="b")
    assert forked.run_id != run.run_id
    diff = run.diff(forked)
    assert hasattr(diff, "common_ancestor")
    assert hasattr(diff, "only_a") and hasattr(diff, "only_b") and hasattr(diff, "changed")
    path = run.save(tmp_path / "pin.tine")
    integrity = Run.verify_integrity(path)
    assert integrity.ok is True
    assert integrity.expected
    breakdown = run.cost_breakdown()
    for attr in ("total_cost", "total_tokens", "by_model", "by_kind"):
        assert hasattr(breakdown, attr)
