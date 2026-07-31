"""Regressions for the confirmed findings of the v3 adversarial review."""

import json

import pytest
from opentine import Run

from graphforge import (
    GraphEngine,
    GraphNodeResult,
    GraphPolicy,
    GraphRecorder,
    GraphRunCapture,
    GraphSpec,
    repo_available,
)
from graphforge import repo_backend as rb
from graphforge.examples import fanout_graph
from graphforge.state import sane_usage, slug

requires_repo = pytest.mark.skipif(
    not repo_available(), reason="v3 repositories need opentine>=0.3.0"
)

LONE_HIGH = chr(0xD83D)


def _dup_join_graph():
    """One node fanning into the same join twice — duplicate parent ids."""
    return (
        GraphSpec(name="dupjoin")
        .add_node("a", lambda ctx: GraphNodeResult(update={"a": 1}))
        .add_node("j", lambda ctx: GraphNodeResult(update={"j": 1}), join=True)
        .add_edge("a", "j", label="left")
        .add_edge("a", "j", label="right")
        .set_entry("a")
    )


# --- critical: duplicate parent ids -------------------------------------------


def test_join_fed_twice_from_one_parent_records_unique_parents(tmp_path):
    result = GraphEngine(_dup_join_graph(), runs_dir=tmp_path).run(
        goal="dup", initial_state={}
    )
    run = Run.load(result.artifact)
    join = next(s for s in run.graph.steps.values() if (s.inputs or {}).get("node") == "j")
    assert len(join.parent_ids) == len(set(join.parent_ids))


def test_goto_listing_the_same_target_twice_is_deduped(tmp_path):
    graph = (
        GraphSpec()
        .add_node("a", lambda ctx: GraphNodeResult(goto=["j", "j"]))
        .add_node("j", lambda ctx: GraphNodeResult(update={"j": 1}), join=True)
        .set_entry("a")
    )
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal="dupgoto", initial_state={})
    run = Run.load(result.artifact)
    join = next(s for s in run.graph.steps.values() if (s.inputs or {}).get("node") == "j")
    assert len(join.parent_ids) == len(set(join.parent_ids))


def test_capture_dedupes_repeated_parents(tmp_path):
    capture = GraphRunCapture("dup", runs_dir=tmp_path, strict=True)
    capture.node_start("a", state={})
    a_id = capture.node_end("a", update={"a": 1})
    capture.node_start("j")
    join_id = capture.node_end("j", parents=[a_id, a_id])
    step = capture.recorder.run.graph.steps[join_id]
    assert len(step.parent_ids) == len(set(step.parent_ids))


@requires_repo
def test_duplicate_parent_run_can_still_enter_a_repository(tmp_path):
    """The whole point: such a run used to be permanently unstorable."""
    repo = rb.open_repo(tmp_path / "wt", create=True)
    result = GraphEngine(_dup_join_graph(), runs_dir=tmp_path / "files").run(
        goal="dup", initial_state={}
    )
    stored = rb.save_to_repo(result.recorder, repo, ref="experiments/dup")
    assert stored["run_id"].startswith("run:sha256:")
    assert repo.fsck(deep=True).ok


@requires_repo
def test_preflight_catches_a_hand_built_duplicate_parent(tmp_path):
    from opentine import StepKind

    repo = rb.open_repo(tmp_path / "wt", create=True)
    recorder = GraphRecorder("hand", runs_dir=tmp_path)
    first = recorder.record_node("a")
    recorder.run.add_step(StepKind.think, inputs={"node": "j"}, parent_ids=[first, first])
    before = len(repo.iter_oids())
    with pytest.raises(ValueError, match="repeats a parent id"):
        rb.save_to_repo(recorder, repo)
    assert len(repo.iter_oids()) == before


# --- major: rejected ref must not write ----------------------------------------


@requires_repo
def test_rejected_ref_writes_nothing(tmp_path):
    repo = rb.open_repo(tmp_path / "wt", create=True)
    result = GraphEngine(fanout_graph(), runs_dir=tmp_path / "files").run(
        goal="q", initial_state={"question": "q"}
    )
    before_objects = len(repo.iter_oids())
    before_refs = dict(repo.list_refs())
    for forbidden in ("promotions/prod", "heads/main"):
        with pytest.raises(ValueError, match="may only write"):
            rb.save_to_repo(result.recorder, repo, ref=forbidden)
    # Objects are immutable and there is no gc: a rejected ref must not
    # leave the run's events (or an annotations/* ref) behind forever.
    assert len(repo.iter_oids()) == before_objects
    assert dict(repo.list_refs()) == before_refs


# --- major: branch is a filesystem path component --------------------------------


def test_slug_neutralizes_path_traversal():
    assert "/" not in slug("../../etc/passwd")
    assert ".." not in slug("..")
    assert slug("") == "fork"
    assert slug("...") == "fork"
    assert LONE_HIGH not in slug(f"branch{LONE_HIGH}")
    assert slug("feature/my-branch") == "feature-my-branch"


def test_resume_with_hostile_branch_stays_in_runs_dir(tmp_path):
    runs = tmp_path / "runs"
    outside = tmp_path / "outside"
    outside.mkdir()
    engine = GraphEngine(fanout_graph(), runs_dir=runs)
    original = engine.run(goal="q", initial_state={"question": "q"})
    step_id = original.path[0]["step_id"]
    resumed = engine.resume(
        original.artifact, step_id, branch="../../outside/pwned", start_node="research_a"
    )
    assert resumed.artifact.resolve().parent == runs.resolve()
    assert list(outside.iterdir()) == []


def test_mcp_fork_with_hostile_branch_stays_in_runs_dir(tmp_path):
    from graphforge.mcp_server import fork_run

    runs = tmp_path / "runs"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    result = GraphEngine(fanout_graph(), runs_dir=runs).run(
        goal="q", initial_state={"question": "q"}
    )
    run = Run.load(result.artifact)
    step = next(sid for sid in run.graph.order if (run.graph.steps[sid].inputs or {}).get("node"))
    forked = fork_run(result.run_id, step[:12], "../../../outside/pwned", str(runs))
    from pathlib import Path

    assert Path(forked["path"]).resolve().parent == runs.resolve()
    assert sorted(p.name for p in outside.iterdir()) == ["secret.txt"]


# --- critical: usage double-counting ---------------------------------------------


def test_openai_inclusive_usage_is_not_double_counted():
    """prompt_tokens INCLUDES cached_tokens; opentine's dimensions exclude it."""
    usage = sane_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "cached_tokens": 800,
            "reasoning_tokens": 400,
        }
    )
    assert usage == {"input": 200, "output": 100, "cache_read": 800, "reasoning": 400}
    assert sum(usage.values()) == 1500  # not 2700


def test_anthropic_exclusive_usage_is_left_alone():
    """input_tokens EXCLUDES cache; only output includes thinking."""
    usage = sane_usage(
        {
            "input_tokens": 200,
            "output_tokens": 500,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 50,
        }
    )
    assert usage["input"] == 200
    assert usage["cache_read"] == 800
    assert usage["cache_write_5m"] == 50


def test_subtraction_never_goes_negative():
    # Over-subtraction clamps at zero, and a zero canonical dimension is
    # dropped entirely — matching opentine's own compact serialization.
    usage = sane_usage({"prompt_tokens": 10, "cached_tokens": 999})
    assert usage.get("input", 0) == 0
    assert all(v >= 0 for v in usage.values())


def test_usage_dimension_names_are_cleaned():
    usage = sane_usage({f"weird{LONE_HIGH}": 5, "": 2, "input": 7})
    assert usage is not None
    assert "" not in usage
    assert all(LONE_HIGH not in name for name in usage)
    assert json.dumps(usage)  # encodable, so the artifact will save


def test_surrogate_usage_key_still_saves(tmp_path):
    recorder = GraphRecorder("usage names", runs_dir=tmp_path)
    recorder.record_model_call(model="m", usage={f"dim{LONE_HIGH}": 3, "input": 5})
    recorder.record_done("done")
    path = recorder.save()
    assert LONE_HIGH not in json.dumps(Run.load(path).to_dict())


# --- major: budget units ------------------------------------------------------


def test_budget_does_not_claim_a_step_ceiling(tmp_path):
    """opentine counts every step; graphforge counts node executions."""
    policy = GraphPolicy(max_node_executions=3, max_total_cost=1.0)
    engine = GraphEngine(fanout_graph(), policy=policy, runs_dir=tmp_path)
    result = engine.run(goal="q", initial_state={"question": "q"})
    run = Run.load(result.artifact)
    budget = run.budget()
    assert budget is None or budget.max_steps is None
    assert run.policies["graph"]["max_node_executions"] == 3


# --- opentine 0.4.0: per-act fork identity ------------------------------------


def _native_fork_identity() -> bool:
    from graphforge.recorder import _fork_mints_unique_ids

    return _fork_mints_unique_ids()


def test_forks_are_unique_on_every_opentine_version(tmp_path):
    """Before 0.4.0 the id was a pure function of (parent, fork point), so the
    second fork's save silently destroyed the first."""
    from graphforge.mcp_server import fork_run as mcp_fork

    result = GraphEngine(fanout_graph(), runs_dir=tmp_path).run(
        goal="q", initial_state={"question": "q"}
    )
    run = Run.load(result.artifact)
    step = next(sid for sid in run.graph.order if (run.graph.steps[sid].inputs or {}).get("node"))
    first = mcp_fork(result.run_id, step[:12], "same-branch", str(tmp_path))
    second = mcp_fork(result.run_id, step[:12], "same-branch", str(tmp_path))
    assert first["run_id"] != second["run_id"]
    assert first["path"] != second["path"]
    assert Run.load(first["path"]) and Run.load(second["path"])


@pytest.mark.skipif(
    not _native_fork_identity(), reason="per-act fork identity needs opentine>=0.4.0"
)
def test_forks_carry_a_verifiable_basis_on_0_4_0(tmp_path):
    """Supplying our own new_run_id would discard the fork basis, so the
    workaround must NOT be used where opentine mints a verifiable id."""
    from graphforge import verify_fork_identity
    from graphforge.mcp_server import fork_run as mcp_fork

    result = GraphEngine(fanout_graph(), runs_dir=tmp_path).run(
        goal="q", initial_state={"question": "q"}
    )
    run = Run.load(result.artifact)
    step = next(sid for sid in run.graph.order if (run.graph.steps[sid].inputs or {}).get("node"))
    forked = Run.load(mcp_fork(result.run_id, step[:12], "b", str(tmp_path))["path"])
    assert verify_fork_identity(forked) is True
    assert (forked.metadata.get("fork") or {}).get("intent")


@pytest.mark.skipif(
    not _native_fork_identity(), reason="per-act fork identity needs opentine>=0.4.0"
)
def test_engine_resume_fork_is_verifiable(tmp_path):
    from graphforge import verify_fork_identity

    engine = GraphEngine(fanout_graph(), runs_dir=tmp_path)
    original = engine.run(goal="q", initial_state={"question": "q"})
    resumed = engine.resume(
        original.artifact, original.path[0]["step_id"], start_node="research_a"
    )
    assert verify_fork_identity(Run.load(resumed.artifact)) is True


# --- minor: run handles accept a ref, as documented ----------------------------


@requires_repo
def test_evaluate_and_promote_accept_a_ref_name(tmp_path):
    """load_run takes "OID or ref"; these used to raise a raw KernelError."""
    repo = rb.open_repo(tmp_path / "wt", create=True)
    result = GraphEngine(fanout_graph(), runs_dir=tmp_path / "files").run(
        goal="q", initial_state={"question": "q"}
    )
    stored = rb.save_to_repo(result.recorder, repo, ref="experiments/e1")
    rb.evaluate(repo, "experiments/e1", {"gate": 0.9}, signer="ci")
    assert [c.run_id for c in rb.candidates(repo, min_score=0.8)] == [stored["run_id"]]
    moved = rb.promote(repo, "experiments/e1", "prod", signer="ci")
    assert moved["run_id"] == stored["run_id"]
    # idempotency compares OID vs OID, so the same ref is a no-op
    assert rb.promote(repo, "experiments/e1", "prod")["changed"] is False


@requires_repo
def test_unknown_run_handle_raises_value_error(tmp_path):
    repo = rb.open_repo(tmp_path / "wt", create=True)
    for bad in ("experiments/missing", "", "not-an-oid"):
        with pytest.raises(ValueError):
            rb.evaluate(repo, bad, {"gate": 1.0}, signer="ci")


# --- major: verify fails closed -----------------------------------------------


def test_verify_rejects_an_empty_key(tmp_path):
    typer = pytest.importorskip("typer")
    from typer.testing import CliRunner

    from graphforge.cli import app

    recorder = GraphRecorder("verify", runs_dir=tmp_path)
    recorder.record_done("done")
    path = recorder.save(sign_key=b"a-test-key-at-least-16-bytes", signer="ci")
    runner = CliRunner()
    empty = runner.invoke(app, ["verify", str(path), "--hmac-key", ""])
    assert empty.exit_code != 0, "an empty key must not silently skip verification"

    good = runner.invoke(
        app, ["verify", str(path), "--hmac-key", "a-test-key-at-least-16-bytes"]
    )
    assert good.exit_code == 0
    assert '"signature_present": true' in good.stdout
    assert typer  # keep the import meaningful


def test_verify_reports_an_unchecked_signature(tmp_path):
    pytest.importorskip("typer")
    from typer.testing import CliRunner

    from graphforge.cli import app

    recorder = GraphRecorder("unchecked", runs_dir=tmp_path)
    recorder.record_done("done")
    path = recorder.save(sign_key=b"a-test-key-at-least-16-bytes", signer="ci")
    result = CliRunner().invoke(app, ["verify", str(path)])
    payload = json.loads(result.stdout)
    # signed-but-unverified must be greppable, not identical to unsigned
    assert payload["signature_present"] is True
    assert payload["signature_checked"] is False


# --- opentine 0.4.0: fork provenance is reachable without private imports ------


def test_verify_fork_identity_is_a_public_helper(tmp_path):
    """opentine keeps verify_fork_id private, so graphforge wraps it: callers
    should never have to import opentine._fork_identity themselves."""
    from graphforge import verify_fork_identity

    engine = GraphEngine(fanout_graph(), runs_dir=tmp_path)
    original = engine.run(goal="q", initial_state={"question": "q"})
    # a run that was never forked has no verdict to give
    assert verify_fork_identity(Run.load(original.artifact)) is None

    resumed = engine.resume(
        original.artifact, original.path[0]["step_id"], start_node="research_a"
    )
    verdict = verify_fork_identity(Run.load(resumed.artifact))
    # True on opentine >= 0.4.0; None on older, where the check does not exist
    assert verdict in (True, None)


def test_verify_fork_identity_never_raises():
    """A provenance check must be total on hostile input."""
    from graphforge import verify_fork_identity

    class NotARun:
        metadata = "not-a-dict"

    assert verify_fork_identity(NotARun()) is None
