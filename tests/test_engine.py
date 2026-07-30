from opentine import Run

from graphforge import (
    END,
    GraphEngine,
    GraphNodeResult,
    GraphPolicy,
    GraphSpec,
)
from graphforge.examples import fanout_graph, refinement_graph


def test_linear_graph_records_every_node(tmp_path):
    order = []

    def make(name):
        def fn(ctx):
            order.append(name)
            return GraphNodeResult(observation=name, update={name: True})

        return fn

    graph = (
        GraphSpec()
        .add_node("a", make("a"))
        .add_node("b", make("b"))
        .add_node("c", make("c"))
        .add_edge("a", "b")
        .add_edge("b", "c")
        .add_edge("c", END)
        .set_entry("a")
    )
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal="linear", initial_state={})
    assert order == ["a", "b", "c"]
    assert result.status == "completed"
    assert result.final_state == {"a": True, "b": True, "c": True}
    run = Run.load(result.artifact)
    nodes = [s.inputs.get("node") for s in run.graph.steps.values() if s.inputs.get("node")]
    assert set(nodes) == {"a", "b", "c"}


def test_edges_taken_recorded_first_class(tmp_path):
    result = GraphEngine(refinement_graph(10.0), runs_dir=tmp_path).run(
        goal="refine", initial_state={"x": 0.0}
    )
    run = Run.load(result.artifact)
    edges = [
        step.inputs["edge"]
        for step in run.graph.steps.values()
        if step.inputs.get("node") and step.inputs.get("edge")
    ]
    kinds = {edge.get("kind") for edge in edges}
    assert "entry" in kinds
    assert "static" in kinds  # propose -> evaluate
    assert "router" in kinds  # evaluate -> propose (cycle)


def test_refinement_cycle_converges(tmp_path):
    result = GraphEngine(refinement_graph(42.0), runs_dir=tmp_path).run(
        goal="hit 42", initial_state={"x": 0.0}
    )
    assert result.status == "completed"
    assert abs(result.final_state["x"] - 42.0) <= 0.5
    assert result.best is not None
    assert result.best.score == max(h["score"] for h in result.path if h["score"] is not None)


def test_fanout_join_gets_multi_parent_lineage(tmp_path):
    result = GraphEngine(fanout_graph(), runs_dir=tmp_path).run(
        goal="question", initial_state={"question": "q"}
    )
    assert result.status == "completed"
    assert result.final_state["notes"] == [
        "graphs declare architecture",
        "loops defer architecture",
    ]
    run = Run.load(result.artifact)
    merge_steps = [s for s in run.graph.steps.values() if s.inputs.get("node") == "merge"]
    assert len(merge_steps) == 1
    assert len(merge_steps[0].parent_ids) == 2  # fan-in recorded as multi-parent
    assert merge_steps[0].inputs["edge"]["kind"] == "join"


def test_goto_overrides_router_and_static_edges(tmp_path):
    def a(ctx):
        return GraphNodeResult(goto="c")

    def other(name):
        def fn(ctx):
            return GraphNodeResult(update={name: True})

        return fn

    graph = (
        GraphSpec()
        .add_node("a", a)
        .add_node("b", other("b"))
        .add_node("c", other("c"))
        .add_edge("a", "b")
        .set_entry("a")
    )
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal="goto", initial_state={})
    assert result.final_state == {"c": True}
    assert [h["node"] for h in result.path] == ["a", "c"]


def test_unknown_route_target_fails_run_but_keeps_artifact(tmp_path):
    def a(ctx):
        return GraphNodeResult(goto="nope")

    graph = GraphSpec().add_node("a", a).set_entry("a")
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal="bad", initial_state={})
    assert result.status == "failed"
    assert result.stop_reason == "route_error"
    assert result.artifact is not None and result.artifact.exists()
    run = Run.load(result.artifact)
    assert any("nope" in str(s.inputs) for s in run.graph.steps.values())


def test_node_exception_records_error_and_fails(tmp_path):
    def boom(ctx):
        raise ValueError("kaput")

    graph = GraphSpec().add_node("a", boom).set_entry("a")
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal="err", initial_state={})
    assert result.status == "failed"
    run = Run.load(result.artifact)
    assert getattr(run.status, "value", str(run.status)) == "failed"
    error_steps = [s for s in run.graph.steps.values() if "kaput" in str(s.inputs)]
    assert error_steps


def test_initial_state_not_aliased(tmp_path):
    initial = {"nested": {"n": 1}}

    def mutate(ctx):
        ctx.state["nested"]["n"] = 99
        return GraphNodeResult(update={"done": True})

    graph = GraphSpec().add_node("a", mutate).set_entry("a")
    GraphEngine(graph, runs_dir=tmp_path).run(goal="alias", initial_state=initial)
    assert initial == {"nested": {"n": 1}}


def test_cycle_guard_stops_infinite_loop(tmp_path):
    def spin(ctx):
        return GraphNodeResult(goto="spin")

    graph = GraphSpec().add_node("spin", spin).set_entry("spin")
    engine = GraphEngine(
        graph, policy=GraphPolicy(max_visits_per_node=5), runs_dir=tmp_path
    )
    result = engine.run(goal="spin", initial_state={})
    assert result.stop_reason == "max_visits_per_node"
    assert result.status == "stopped:max_visits_per_node"
    run = Run.load(result.artifact)
    assert "gate:max_visits_per_node" in run.tags
