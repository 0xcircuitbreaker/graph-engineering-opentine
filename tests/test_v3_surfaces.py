"""Graph-layer diff, priced billing, signing, budget, and step attribution.

The attribution/signing/budget tests run on both opentine 0.2.0 and 0.3.0;
the billing ones need 0.3.0's pricing catalog.
"""


import pytest
from opentine import Run

from graphforge import (
    GraphEngine,
    GraphNodeResult,
    GraphPolicy,
    GraphRecorder,
    GraphSpec,
    graph_diff,
)
from graphforge.examples import fanout_graph, refinement_graph
from graphforge.pricing import billing_available, to_usage
from graphforge.recorder import read_artifact_document, write_artifact_document

HMAC_KEY = b"a-test-key-at-least-16-bytes-long"


# --- graph-layer diff ---------------------------------------------------------


def test_graph_diff_calls_identical_runs_identical(tmp_path):
    """v3 event ids embed a timestamp, so opentine's own diff cannot do this."""
    engine = GraphEngine(refinement_graph(42.0), runs_dir=tmp_path)
    left = Run.load(engine.run(goal="42", initial_state={"x": 0.0}).artifact)
    right = Run.load(engine.run(goal="42", initial_state={"x": 0.0}).artifact)
    diff = graph_diff(left, right)
    assert diff["identical_path"] is True
    assert diff["first_divergence"] is None
    assert diff["visit_delta"] == {}
    assert diff["edges_only_left"] == [] and diff["edges_only_right"] == []
    assert diff["topology_match"] is True


def test_graph_diff_pinpoints_divergence(tmp_path):
    engine = GraphEngine(refinement_graph(42.0), runs_dir=tmp_path)
    left = Run.load(engine.run(goal="42", initial_state={"x": 0.0}).artifact)
    right = Run.load(engine.run(goal="42", initial_state={"x": 30.0}).artifact)
    diff = graph_diff(left, right)
    assert diff["identical_path"] is False
    assert diff["first_divergence"]["reason"] == "same node, different state"
    assert diff["first_divergence"]["index"] == 0
    assert set(diff["visit_delta"]) <= {"propose", "evaluate"}
    assert "evaluate" in diff["score_drift"]


def test_graph_diff_detects_topology_change_and_edges(tmp_path):
    refine = GraphEngine(refinement_graph(10.0), runs_dir=tmp_path).run(
        goal="ten", initial_state={"x": 0.0}
    )
    fan = GraphEngine(fanout_graph(), runs_dir=tmp_path).run(
        goal="q", initial_state={"question": "q"}
    )
    diff = graph_diff(Run.load(refine.artifact), Run.load(fan.artifact))
    assert diff["topology_match"] is False
    assert diff["identical_path"] is False
    assert any("merge" in edge for edge in diff["edges_only_right"])


# --- step attribution ----------------------------------------------------------


def test_model_and_tool_attribution_reaches_opentine(tmp_path):
    recorder = GraphRecorder("attribution", runs_dir=tmp_path)
    model_step = recorder.record_model_call(
        model="claude-sonnet-5", prompt="p", response="r", cost=0.5
    )
    tool_step = recorder.record_tool_call("search", args={"q": "x"}, result="hit")
    run = recorder.run
    assert run.graph.steps[model_step].model_info == "claude-sonnet-5"
    assert run.graph.steps[tool_step].tool_info == {"name": "search"}
    # the whole point: cost is attributed, not dumped in one anonymous bucket
    assert run.cost_breakdown().by_model["claude-sonnet-5"] == 0.5


def test_error_step_carries_first_class_error(tmp_path):
    recorder = GraphRecorder("errors", runs_dir=tmp_path)
    step_id = recorder.record_error(ValueError("kaput"))
    step = recorder.run.graph.steps[step_id]
    assert step.error == {"type": "ValueError", "message": "kaput"}
    assert step.inputs["error_type"] == "ValueError"  # kept for compatibility


def test_usage_aliases_produce_real_token_totals(tmp_path):
    recorder = GraphRecorder("usage", runs_dir=tmp_path)
    recorder.record_model_call(
        model="m",
        usage={"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
    )
    # before aliasing these names were passed through verbatim and opentine
    # counted zero tokens for the whole run
    assert recorder.run.total_tokens == 1200
    breakdown = recorder.run.cost_breakdown()
    assert breakdown.total_tokens == 1200


# --- budget --------------------------------------------------------------------


def test_policy_is_declared_as_an_opentine_budget(tmp_path):
    policy = GraphPolicy(
        max_node_executions=9, max_total_cost=1.5, max_duration_seconds=30.0
    )
    recorder = GraphRecorder("budget", policy=policy, runs_dir=tmp_path)
    budget = recorder.run.budget()
    assert budget is not None
    assert budget.max_cost == 1.5
    assert budget.max_duration == 30.0
    # max_node_executions is deliberately NOT mapped onto opentine's
    # max_steps: opentine counts gate/terminal/nested steps too, so the run
    # would self-report as over budget. It stays in policies["graph"].
    assert budget.max_steps is None
    recorder.record_done("done")
    reloaded = Run.load(recorder.save())
    assert reloaded.manifest.get("budget") is not None
    # the graphforge-native policy is still there too
    assert reloaded.policies["graph"]["max_node_executions"] == 9


def test_float_policy_values_are_coerced(tmp_path):
    """A policy loaded from JSON can hold 3.0; 0.3.0 rejects a float max_steps."""
    policy = GraphPolicy.from_dict(
        {"max_node_executions": 3.0, "max_total_cost": 2, "max_duration_seconds": 5}
    )
    recorder = GraphRecorder("coerce", policy=policy, runs_dir=tmp_path)
    budget = recorder.run.budget()
    assert budget is not None
    assert budget.max_cost == 2.0
    assert isinstance(budget.max_duration, float)


# --- signing -------------------------------------------------------------------


def test_signed_artifact_verifies_and_detects_tampering(tmp_path):
    recorder = GraphRecorder("signing", runs_dir=tmp_path)
    recorder.record_node("a", state_after={"k": 1})
    recorder.record_done("done")
    path = recorder.save(sign_key=HMAC_KEY, signer="ci", key_id="k1")
    good = Run.verify_signature(path, hmac_key=HMAC_KEY)
    assert good.ok and good.state == "verified"
    assert good.signer == "ci"

    payload = read_artifact_document(path)
    if "objects" in payload:  # v4: rewrite one content object's bytes
        blob = next(o for o in payload["objects"].values() if o.get("type") == "blob")
        blob["data"] = "eyJyZXdyaXR0ZW4iOnRydWV9"
    else:
        payload["user_prompt"] = "rewritten"
    tampered = tmp_path / "tampered.tine"
    write_artifact_document(tampered, payload)
    assert not Run.verify_signature(tampered, hmac_key=HMAC_KEY).ok


def test_gates_are_recorded_in_the_signed_body_not_only_in_tags(tmp_path):
    """Tags sit outside the digest AND outside the signature allowlist."""
    graph = (
        GraphSpec()
        .add_node("spin", lambda ctx: GraphNodeResult(goto="spin"))
        .set_entry("spin")
    )
    engine = GraphEngine(
        graph, policy=GraphPolicy(max_visits_per_node=3), runs_dir=tmp_path
    )
    result = engine.run(goal="gated", initial_state={})
    run = Run.load(result.artifact)
    assert "gate:max_visits_per_node" in run.tags
    # the same fact inside a step body, which the digest and signature cover
    gate_steps = [
        s for s in run.graph.steps.values() if (s.inputs or {}).get("gate")
    ]
    assert any(s.inputs["gate"] == "max_visits_per_node" for s in gate_steps)


def test_unresolved_join_is_recorded_in_the_body(tmp_path):
    def stopper(ctx):
        return GraphNodeResult(update={"b2": True}, stop=True)

    graph = (
        GraphSpec()
        .add_node("start", lambda ctx: GraphNodeResult(update={"s": 1}))
        .add_node("a", lambda ctx: GraphNodeResult(update={"a": 1}))
        .add_node("b1", lambda ctx: GraphNodeResult(update={"b1": 1}))
        .add_node("b2", stopper)
        .add_node("j", lambda ctx: GraphNodeResult(), join=True)
        .add_edge("start", "a")
        .add_edge("start", "b1")
        .add_edge("a", "j")
        .add_edge("b1", "b2")
        .set_entry("start")
    )
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal="join", initial_state={})
    run = Run.load(result.artifact)
    assert "graph:unresolved_join" in run.tags
    assert any(
        (s.inputs or {}).get("gate") == "unresolved_join" for s in run.graph.steps.values()
    )


# --- pause ----------------------------------------------------------------------


def test_pause_marks_the_run_paused(tmp_path):
    recorder = GraphRecorder("pause", runs_dir=tmp_path)
    recorder.record_checkpoint("awaiting-human", state={"k": 1})
    path = recorder.pause()
    reloaded = Run.load(path)
    assert getattr(reloaded.status, "value", str(reloaded.status)) == "paused"


# --- priced billing (0.3.0 only) --------------------------------------------------


@pytest.mark.skipif(not billing_available(), reason="pricing needs opentine>=0.3.0")
def test_priced_model_call_carries_provenance(tmp_path):
    recorder = GraphRecorder("billing", runs_dir=tmp_path)
    step_id = recorder.record_model_call(
        model="claude-sonnet-5",
        provider="anthropic",
        usage={"prompt_tokens": 1000, "completion_tokens": 500},
    )
    step = recorder.run.graph.steps[step_id]
    assert step.billing["status"] in {"complete", "partial"}
    assert step.billing["catalog_id"]
    assert step.cost > 0
    assert recorder.run.cost_breakdown().by_model["claude-sonnet-5"] == step.cost


@pytest.mark.skipif(not billing_available(), reason="pricing needs opentine>=0.3.0")
def test_pricing_never_breaks_recording(tmp_path):
    recorder = GraphRecorder("unknown model", runs_dir=tmp_path)
    step_id = recorder.record_model_call(
        model="not-a-real-model-xyz", provider="nobody", usage={"input": 5}, cost=0.25
    )
    step = recorder.run.graph.steps[step_id]
    # unpriceable: the caller's own cost survives rather than being zeroed by
    # an unknown billing record (opentine's total_cost prefers the subtotal)
    assert step.cost == 0.25
    assert not step.billing
    assert step.outputs["pricing"]["status"] == "unknown"
    assert recorder.run.total_cost == 0.25


@pytest.mark.skipif(not billing_available(), reason="pricing needs opentine>=0.3.0")
def test_to_usage_maps_provider_names():
    usage = to_usage({"prompt_tokens": 10, "completion_tokens": 4, "custom_units": 2.5})
    assert usage is not None
    assert usage.input == 10 and usage.output == 4
    assert to_usage({}) is None
    assert to_usage("nonsense") is None
