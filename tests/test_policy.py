from opentine import Run

from graphforge import END, GraphEngine, GraphNodeResult, GraphPolicy, GraphSpec


def _looping_graph(cost=0.0, score=None):
    def node(ctx):
        return GraphNodeResult(observation="tick", cost=cost, score=score, goto="tick")

    return GraphSpec().add_node("tick", node).set_entry("tick")


def test_max_node_executions_is_run_wide(tmp_path):
    engine = GraphEngine(
        _looping_graph(),
        policy=GraphPolicy(max_node_executions=7, max_visits_per_node=None),
        runs_dir=tmp_path,
    )
    result = engine.run(goal="cap", initial_state={})
    assert result.stop_reason == "max_node_executions"
    assert len(result.path) == 7


def test_max_total_cost_is_run_wide(tmp_path):
    engine = GraphEngine(
        _looping_graph(cost=0.05),
        policy=GraphPolicy(max_total_cost=0.2, max_visits_per_node=None),
        runs_dir=tmp_path,
    )
    result = engine.run(goal="spend", initial_state={})
    assert result.stop_reason == "max_total_cost"
    # 4 * 0.05 hits the 0.2 ceiling; the 5th node is gated before running.
    assert len(result.path) == 4
    run = Run.load(result.artifact)
    assert "gate:max_total_cost" in run.tags


def test_min_score_stops_successfully(tmp_path):
    engine = GraphEngine(
        _looping_graph(score=0.99),
        policy=GraphPolicy(min_score=0.95, max_visits_per_node=None),
        runs_dir=tmp_path,
    )
    result = engine.run(goal="score", initial_state={})
    assert result.stop_reason == "min_score"
    assert result.status == "completed"
    assert len(result.path) == 1


def test_policy_embedded_and_roundtrips(tmp_path):
    policy = GraphPolicy(max_node_executions=9, max_total_cost=1.5, min_score=0.8)
    graph = (
        GraphSpec()
        .add_node("a", lambda ctx: GraphNodeResult(stop=True))
        .add_edge("a", END)
        .set_entry("a")
    )
    result = GraphEngine(graph, policy=policy, runs_dir=tmp_path).run(
        goal="embed", initial_state={}
    )
    run = Run.load(result.artifact)
    stored = GraphPolicy.from_dict(run.policies["graph"])
    assert stored.max_node_executions == 9
    assert stored.max_total_cost == 1.5
    assert stored.min_score == 0.8


def test_policy_dict_roundtrip():
    policy = GraphPolicy(max_duration_seconds=12.5, max_visits_per_node=3)
    assert GraphPolicy.from_dict(policy.to_dict()) == policy
    assert GraphPolicy.from_dict(None) == GraphPolicy()
