"""v3 ``.tine/`` repository backend.

Every test here is skipped on opentine 0.2.0, which has no ``Repo``; the
rest of the suite must stay green there.
"""

import pytest
from opentine import Run

from graphforge import GraphEngine, GraphRecorder, GraphSpec, repo_available
from graphforge import repo_backend as rb
from graphforge.engine import GraphNodeResult
from graphforge.examples import fanout_graph, refinement_graph

pytestmark = pytest.mark.skipif(
    not repo_available(), reason="v3 repositories need opentine>=0.3.0"
)


@pytest.fixture()
def repo(tmp_path):
    return rb.open_repo(tmp_path / "wt", create=True)


def _run(tmp_path, graph=None, **kwargs):
    engine = GraphEngine(graph or fanout_graph(), runs_dir=tmp_path / "files")
    return engine.run(
        goal=kwargs.pop("goal", "q"),
        initial_state=kwargs.pop("state", {"question": "q"}),
    )


def test_round_trip_preserves_graph_layer(repo, tmp_path):
    result = _run(tmp_path)
    stored = rb.save_to_repo(result.recorder, repo, ref="experiments/demo")
    assert stored["run_id"].startswith("run:sha256:")
    assert stored["ref"] == "experiments/demo"
    # every graphforge step id maps to a content-addressed event
    assert len(stored["event_map"]) == len(result.recorder.run.graph.order)
    assert all(v.startswith("event:sha256:") for v in stored["event_map"].values())

    loaded = rb.load_from_repo(repo, "experiments/demo")
    merge = [s for s in loaded.graph.steps.values() if (s.inputs or {}).get("node") == "merge"]
    assert len(merge) == 1
    assert len(merge[0].parent_ids) == 2  # fan-in lineage survives
    assert loaded.manifest["graph"]["name"] == "fanout-research"
    assert loaded.manifest["kind"] == "graphforge"
    assert repo.fsck(deep=True).ok


def test_resume_from_repo_loaded_run(repo, tmp_path):
    """v3 canonicalizes 21.0 -> 21; the digest must survive that."""
    engine = GraphEngine(refinement_graph(42.0), runs_dir=tmp_path / "files")
    original = engine.run(goal="42", initial_state={"x": 0.0})
    rb.save_to_repo(original.recorder, repo, ref="experiments/r1")
    loaded = rb.load_from_repo(repo, "experiments/r1")
    evaluate_steps = [
        sid
        for sid in loaded.graph.order
        if (loaded.graph.steps[sid].inputs or {}).get("node") == "evaluate"
    ]
    resumed = engine.resume(
        loaded, evaluate_steps[0], state_update={"x": -50.0, "distance": 92.0}
    )
    assert resumed.status == "completed"
    assert abs(resumed.final_state["x"] - 42.0) <= 0.5


def test_ref_writes_are_confined(repo, tmp_path):
    result = _run(tmp_path)
    for forbidden in ("heads/main", "promotions/prod", "annotations/abc"):
        with pytest.raises(ValueError, match="may only write"):
            rb.save_to_repo(result.recorder, repo, ref=forbidden)
    assert rb.assert_writable("refs/experiments/ok") == "experiments/ok"


def test_ref_cas_semantics(repo, tmp_path):
    first = _run(tmp_path)
    stored = rb.save_to_repo(first.recorder, repo, ref="experiments/x", expected_old=None)
    second = _run(tmp_path, goal="other")
    # expected_old=None means "must not exist" — the ref exists now
    with pytest.raises(ValueError):
        rb.save_to_repo(second.recorder, repo, ref="experiments/x", expected_old=None)
    # a correct CAS advances it
    moved = rb.save_to_repo(
        second.recorder, repo, ref="experiments/x", expected_old=stored["run_id"]
    )
    assert repo.read_ref("experiments/x") == moved["run_id"]


def test_preflight_rejects_before_writing(repo, tmp_path):
    result = _run(tmp_path)
    result.recorder.run.tags.append(object())  # bypass add_tag on purpose
    before = len(repo.iter_oids())
    with pytest.raises(ValueError, match="tags must all be strings"):
        rb.save_to_repo(result.recorder, repo)
    # put_run validates tags only after writing every event; preflight must
    # not have written anything.
    assert len(repo.iter_oids()) == before


def test_preflight_rejects_oversize_run(repo, tmp_path, monkeypatch):
    result = _run(tmp_path)
    monkeypatch.setattr(rb, "MAX_REPO_STEPS", 2)
    with pytest.raises(ValueError, match="at most 2"):
        rb.save_to_repo(result.recorder, repo)


def test_evaluate_makes_run_searchable(repo, tmp_path):
    result = _run(tmp_path)
    stored = rb.save_to_repo(result.recorder, repo, ref="experiments/e")
    assert rb.candidates(repo, min_score=0.5) == []  # unscored runs are invisible
    rb.evaluate(repo, stored["run_id"], {"gate": 0.93}, signer="ci")
    found = rb.candidates(repo, min_score=0.9)
    assert [c.run_id for c in found] == [stored["run_id"]]
    assert found[0].score == pytest.approx(0.93)
    assert rb.candidates(repo, min_score=0.99) == []


def test_evaluate_rejects_unusable_scores(repo, tmp_path):
    result = _run(tmp_path)
    stored = rb.save_to_repo(result.recorder, repo)
    for bad in ({}, {"a": float("nan")}, {"a": True}, {"a": "high"}):
        with pytest.raises(ValueError, match="finite numeric score"):
            rb.evaluate(repo, stored["run_id"], bad, signer="ci")


def test_promote_is_cas_and_idempotent(repo, tmp_path):
    first = rb.save_to_repo(_run(tmp_path).recorder, repo)
    second = rb.save_to_repo(_run(tmp_path, goal="other").recorder, repo)
    moved = rb.promote(repo, first["run_id"], "prod", signer="ci")
    assert moved["changed"] is True
    assert moved["attestation"] is not None
    assert rb.promote(repo, first["run_id"], "prod")["changed"] is False  # idempotent
    with pytest.raises(ValueError, match="already points at"):
        rb.promote(repo, second["run_id"], "prod")
    assert rb.promote(repo, second["run_id"], "prod", force=True)["changed"] is True
    assert repo.read_ref("promotions/prod") == second["run_id"]


def test_repo_status_reports_refs_and_fsck(repo, tmp_path):
    stored = rb.save_to_repo(_run(tmp_path).recorder, repo, ref="experiments/s")
    status = rb.repo_status(repo)
    assert status["ok"] is True
    assert status["refs"]["experiments/s"] == stored["run_id"]
    assert status["objects"] > 0
    assert status["shallow"] == []


def test_model_info_makes_runs_findable_by_model(repo, tmp_path):
    def model_node(ctx):
        return GraphNodeResult(observation="called", stop=True)

    graph = GraphSpec().add_node("gen", model_node, kind="model").set_entry("gen")
    engine = GraphEngine(graph, runs_dir=tmp_path / "files")
    result = engine.run(goal="model run", initial_state={})
    result.recorder.record_model_call(model="claude-sonnet-5", prompt="p", response="r")
    stored = rb.save_to_repo(result.recorder, repo)
    rb.evaluate(repo, stored["run_id"], {"gate": 1.0}, signer="ci")
    found = rb.candidates(repo, model="claude-sonnet-5")
    assert [c.run_id for c in found] == [stored["run_id"]]
    assert "claude-sonnet-5" in found[0].models


def test_save_to_repo_accepts_plain_run_and_path(tmp_path):
    recorder = GraphRecorder("plain", runs_dir=tmp_path)
    recorder.record_node("a", state_after={"k": 1})
    recorder.record_done("done")
    worktree = tmp_path / "wt2"
    rb.open_repo(worktree, create=True)
    stored = rb.save_to_repo(recorder.run, worktree, ref="experiments/plain")
    assert Run.load  # sanity: still the v2 API too
    assert stored["run_id"].startswith("run:sha256:")
