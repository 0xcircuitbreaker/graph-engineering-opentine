"""v3 ``.tine/`` repository backend — the only module that imports ``Repo``.

opentine 0.3.0 added a Git-shaped, content-addressed object store alongside
portable ``.tine`` files. For graph runs it supplies exactly what a file
cannot: deduplicated storage (a fork re-stores only new events), a real
on-disk DAG with validated typed links so fan-in lineage survives, CAS refs
so "which run is the candidate/release" is race-safe rather than a filename
convention, and ``attest``/``promote`` as a tamper-evident evaluation gate.

Every import here is lazy and guarded: opentine 0.2.0 has no ``Repo``, and
graphforge supports both (``opentine>=0.2.0,<0.4``). Call
:func:`repo_available` before offering any repository feature.

Deliberate deviations from the obvious implementation, each verified:

- ``put_run(ref=...)`` is a read-then-swap, not a caller-controlled CAS, so
  it can silently clobber a head written a moment earlier. This module always
  passes no ref and does its own :func:`update_ref` with ``expected_old``.
- ``put_run`` validates tags only *after* writing every event object, so a
  non-string tag at step 400 wastes the whole write. Preflight instead.
- ``repo.attest`` accepts a non-dict claim and thereafter ``repo.diff``
  raises forever on that repository — objects are immutable and there is no
  undo. Claims are validated before they are written.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

#: Ref namespaces graphforge may write. ``heads/main`` is the operator's
#: mainline and ``promotions/*`` is a release gate — neither is a side effect
#: of recording. ``annotations/*`` is owned by ``put_run``.
WRITABLE_REF_PREFIXES = ("experiments/", "heads/graphforge/", "tags/")

#: A run object plus its events must fit one pack (10,000 objects), and
#: ``put_run`` costs ~25-35 ms/step, so refuse early with a clear message.
MAX_REPO_STEPS = 3300

_UNSET = object()


class RepoUnavailableError(RuntimeError):
    """Raised when a v3 repository feature is used without opentine >= 0.3.0."""


def repo_available() -> bool:
    """Whether the installed opentine exposes the v3 repository."""
    try:
        from opentine import Repo  # noqa: F401
    except ImportError:
        return False
    return True


def _repo_class() -> Any:
    try:
        from opentine import Repo
    except ImportError as exc:  # pragma: no cover - exercised on 0.2.0 only
        raise RepoUnavailableError(
            "v3 repositories need opentine>=0.3.0; the installed opentine has no Repo"
        ) from exc
    return Repo


def open_repo(path: str | Path = ".", *, create: bool = False) -> Any:
    """Open (or optionally create) a ``.tine`` repository."""
    repo_cls = _repo_class()
    target = Path(path).expanduser()
    if create:
        return repo_cls.init(target)
    return repo_cls.open(target)


def normalize_ref(ref: str) -> str:
    """Canonical ref name, via opentine's own normalizer when available."""
    try:
        from opentine.repository._refs import normalize_ref as _normalize
    except ImportError:  # pragma: no cover - private module may move
        return ref[len("refs/") :] if ref.startswith("refs/") else ref
    return _normalize(ref)


def assert_writable(ref: str) -> str:
    """Confine graphforge's ref writes; returns the canonical name.

    Checks the *canonical* name so ``refs/experiments/x`` is accepted and
    casing tricks are not.
    """
    name = normalize_ref(ref)
    if not name.startswith(WRITABLE_REF_PREFIXES):
        raise ValueError(
            f"graphforge may only write {WRITABLE_REF_PREFIXES}, not {name!r}; "
            "promotions and mainline heads are explicit operator actions"
        )
    return name


def _preflight(run: Any) -> None:
    """Refuse what ``put_run`` would only reject after writing everything."""
    metadata = getattr(run, "metadata", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"run.metadata must be a dict, not {type(metadata).__name__}")
    tags = list(getattr(run, "tags", []) or [])
    bad = [t for t in tags if not isinstance(t, str)]
    if bad:
        raise ValueError(f"run tags must all be strings; got {bad!r}")
    order = list(getattr(run.graph, "order", []) or [])
    if len(order) > MAX_REPO_STEPS:
        raise ValueError(
            f"run has {len(order)} steps; a v3 repository accepts at most {MAX_REPO_STEPS} "
            "per run (one pack). Save it as a .tine file, or record fewer steps."
        )
    # put_run rejects repeated parent_ids, but only after writing every event
    # object, so a duplicate would leave the store permanently littered.
    for step_id in order:
        step = run.graph.steps[step_id]
        parents = list(getattr(step, "parent_ids", []) or [])
        if len(set(parents)) != len(parents):
            raise ValueError(
                f"step {step_id[:12]} repeats a parent id; a v3 repository requires "
                "unique parent event ids (a join fed twice from the same node?)"
            )


def save_to_repo(
    run: Any,
    repo: Any | str | Path,
    *,
    ref: str | None = None,
    expected_old: Any = _UNSET,
    create: bool = False,
) -> dict[str, Any]:
    """Store a run as v3 objects; optionally advance a ref under CAS.

    ``run`` is an opentine ``Run`` or a :class:`~graphforge.GraphRecorder`.
    Returns ``{"run_id", "event_map", "annotation_id", "ref"}`` where
    ``event_map`` maps each graphforge step id to its ``event:sha256:`` OID.

    ``expected_old`` is passed straight to ``update_ref``: omit it for a
    blind write, pass ``None`` to require the ref not exist, or pass the OID
    you read to compare-and-swap.
    """
    run_obj = getattr(run, "run", run)  # accept a GraphRecorder
    repo_obj = repo if hasattr(repo, "put_run") else open_repo(repo, create=create)
    _preflight(run_obj)
    # Validate the ref BEFORE writing: objects are immutable and there is no
    # gc, so a rejected ref must not leave the run's events in the store.
    written_ref = assert_writable(ref) if ref is not None else None
    # Deliberately no ref= to put_run: its ref write is a read-then-swap.
    result = repo_obj.put_run(run_obj)
    if written_ref is not None:
        if expected_old is _UNSET:
            repo_obj.update_ref(written_ref, result.run_id)
        else:
            repo_obj.update_ref(written_ref, result.run_id, expected_old)
    return {
        "run_id": result.run_id,
        "event_map": dict(result.event_map),
        "annotation_id": result.annotation_id,
        "ref": written_ref,
    }


def load_from_repo(repo: Any | str | Path, oid_or_ref: str) -> Any:
    """Load a ``Run`` back out of a repository by OID or ref name."""
    repo_obj = repo if hasattr(repo, "load_run") else open_repo(repo)
    return repo_obj.load_run(oid_or_ref)


def resolve_run(repo_obj: Any, handle: str) -> str:
    """Accept a run OID or a ref name and return the run OID.

    ``load_run`` takes either, so every run-taking function here does too —
    otherwise a ref name reaches ``parse_oid`` and surfaces as a raw
    ``KernelError`` traceback.
    """
    if not isinstance(handle, str) or not handle.strip():
        raise ValueError("run handle must be a non-empty string")
    if handle.startswith("run:"):
        return handle
    oid = repo_obj.read_ref(normalize_ref(handle))
    if not oid:
        raise ValueError(f"no such run or ref: {handle!r}")
    if not oid.startswith("run:"):
        raise ValueError(f"ref {handle!r} points at {oid.split(':', 1)[0]}, not a run")
    return oid


def clean_scores(scores: dict[str, Any]) -> dict[str, float]:
    """Keep only finite, non-bool numeric scores, with clean names.

    ``float()`` on a very large int raises ``OverflowError``, which is not a
    ValueError and would escape the caller's error handling, so conversion is
    guarded rather than filtered on.
    """
    from .state import clean_text

    out: dict[str, float] = {}
    for key, value in (scores or {}).items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        try:
            number = float(value)
        except (OverflowError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        name = clean_text(str(key)).strip()
        if name:
            out[name] = number
    return out


def evaluate(
    repo: Any | str | Path,
    run_oid: str,
    scores: dict[str, Any],
    *,
    signer: str,
    evidence: list[str] | None = None,
) -> str:
    """Attach an evaluation attestation, making the run searchable by score.

    opentine's ``search`` reads only attestations whose claim ``kind`` is
    ``"evaluation"``, and scores a claim by the MEAN of its ``scores`` dict
    (then the max across attestations). Mixing a 0-1 pass rate with a raw
    cost in one claim therefore produces a meaningless number — emit one
    claim per gate, or a single claim holding the number you actually gate on.

    The signer label is self-asserted unless a real signature is attached;
    it is provenance, not authenticated identity.
    """
    cleaned = clean_scores(scores)
    if not cleaned:
        raise ValueError("an evaluation needs at least one finite numeric score")
    repo_obj = repo if hasattr(repo, "attest") else open_repo(repo)
    target = resolve_run(repo_obj, run_oid)
    return repo_obj.attest(
        target,
        {"kind": "evaluation", "scores": cleaned},
        signer=str(signer),
        evidence_ids=list(evidence or []),
    )


def candidates(
    repo: Any | str | Path,
    *,
    min_score: float | None = None,
    model: str | None = None,
    query: str = "",
    limit: int = 50,
) -> list[Any]:
    """Search evaluated runs. Requires :func:`evaluate` to have scored them."""
    repo_obj = repo if hasattr(repo, "search") else open_repo(repo)
    return repo_obj.search(
        query, successful_only=True, min_score=min_score, model=model, limit=limit
    )


def promote(
    repo: Any | str | Path,
    run_oid: str,
    name: str,
    *,
    force: bool = False,
    signer: str | None = None,
) -> dict[str, Any]:
    """Move a release gate to a run, always under compare-and-swap.

    ``promote(expected_old=None)`` in opentine means "the ref must not
    exist", so advancing a live gate requires reading it first — which makes
    an accidental release move impossible. Pass ``force=True`` to advance a
    gate that already points somewhere else.
    """
    repo_obj = repo if hasattr(repo, "promote") else open_repo(repo)
    run_oid = resolve_run(repo_obj, run_oid)  # so the compare below is OID-vs-OID
    ref = f"promotions/{name}"
    current = repo_obj.read_ref(ref)
    if current == run_oid:
        return {"ref": ref, "run_id": run_oid, "changed": False}
    if current is not None and not force:
        raise ValueError(
            f"{ref} already points at {current}; pass force=True to advance the gate"
        )
    repo_obj.promote(run_oid, name, expected_old=current)
    attestation = None
    if signer:
        attestation = repo_obj.attest(
            run_oid,
            {"kind": "approval", "approved": True, "release": str(name)},
            signer=str(signer),
            evidence_ids=[],
        )
    return {
        "ref": ref,
        "run_id": run_oid,
        "changed": True,
        "previous": current,
        "attestation": attestation,
    }


def repo_status(repo: Any | str | Path, *, deep: bool = True) -> dict[str, Any]:
    """Refs, object count, shallow boundary, and an fsck verdict."""
    repo_obj = repo if hasattr(repo, "fsck") else open_repo(repo)
    result = repo_obj.fsck(deep=deep)
    shallow = set()
    getter = getattr(repo_obj, "shallow_oids", None)
    if callable(getter):
        shallow = set(getter() or ())
    return {
        "worktree": str(getattr(repo_obj, "worktree", "")),
        "refs": dict(repo_obj.list_refs()),
        "ok": result.ok,
        "objects": result.objects,
        "errors": list(result.errors),
        # log/diff/context_slice silently stop at a shallow boundary, so a
        # caller reading aggregates deserves to know one exists.
        "shallow": sorted(shallow),
    }
