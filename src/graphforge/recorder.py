"""GraphRecorder — the single opentine integration point.

Every module that WRITES opentine runs goes through this file (the CLI and
MCP server only read), so an opentine version bump touches one place. The
recorder maps graph execution onto the portable ``Run`` surface that every
supported opentine exposes (0.2.x through the v4-format 0.2x/0.3x releases;
opentine chooses the on-disk format, graphforge never depends on it):

- one node execution  -> one step (kind mapped from the node kind), with the
  edge taken recorded first-class on the destination step,
- fan-in (join nodes) -> multi-parent steps via ``parent_ids``,
- policy gates        -> tagged think steps,
- fork               -> ``run.fork(from_step_id)`` preserving causal ancestry.

Metrics are sanitized before ``add_step`` because opentine 0.3.0 validates
cost/duration/usage strictly (ValueError on negatives, non-finite values,
bools) where 0.2.0 accepted anything.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from opentine import Run, RunStatus, StepKind

from ._version import __version__
from .policy import GraphPolicy
from .state import (
    clean_text,
    safe_json,
    sane_metric,
    sane_score,
    sane_usage,
    slug,
    state_digest,
    to_jsonable,
)

#: Where artifacts land by default. ``GRAPHFORGE_RUNS_DIR`` overrides it, so a
#: CI job or a project can keep runs out of the shared user directory.
DEFAULT_RUNS_DIR = Path(
    os.environ.get("GRAPHFORGE_RUNS_DIR") or "~/.local/share/graphforge"
).expanduser()


def _fork_mints_unique_ids() -> bool:
    """Whether opentine's own ``Run.fork`` yields a distinct id per fork act.

    opentine 0.4.0 derives a fork id from the source lineage, the retained
    slice, the branch, a declared intent, and a random nonce, and records
    that basis so the fork can prove its own id. Before 0.4.0 the id was a
    pure function of (parent, fork point), so two forks at one point
    collided — and saving the second silently destroyed the first.
    """
    import inspect

    try:
        return "nonce" in inspect.signature(Run.fork).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False


def verify_fork_identity(run: Run) -> bool | None:
    """Does a fork's id match the basis the artifact records?

    ``True``  — the id is exactly what ``metadata["fork"]`` re-derives.
    ``False`` — a basis is recorded but does not produce the id (edited).
    ``None``  — no verdict: not a fork, a pre-0.4.0 fork, an explicitly named
    ``new_run_id``, or an opentine too old to have the check at all.

    This is provenance, not authorization: an artifact's author can always
    make their own artifact self-consistent. It exists here so callers do not
    have to import opentine's private module, which is where the check still
    lives as of 0.4.0.
    """
    try:
        from opentine._fork_identity import verify_fork_id
    except ImportError:  # opentine < 0.4.0
        return None
    try:
        return verify_fork_id(run)
    except Exception:  # noqa: BLE001 - a provenance check must not raise
        return None


def fork_run(
    parent_run: Run,
    from_step_id: str,
    *,
    branch: str = "fork",
    intent: dict[str, Any] | None = None,
) -> Run:
    """Fork a run so the result is always uniquely identified.

    On opentine 0.4.0+ this defers to opentine, which mints a unique id AND
    records a verifiable fork basis. On older versions it supplies its own
    unique id, because the native id would collide.
    """
    if _fork_mints_unique_ids():
        declared = {"tool": "graphforge", "branch": clean_text(branch)}
        if intent:
            declared.update({str(k): v for k, v in intent.items()})
        return parent_run.fork(
            from_step_id=from_step_id, branch=branch, intent=to_jsonable(declared)
        )
    # slug(): the id becomes a file name and branch is caller/MCP-supplied.
    unique = f"{parent_run.run_id[:12]}-{slug(branch)}-{uuid4().hex[:8]}"
    return parent_run.fork(from_step_id=from_step_id, new_run_id=unique, branch=branch)


#: Signature schemes whose signed view covers the run's tags. ``tine-sig/1``
#: (opentine 0.3.0-0.7.0) signed a frozen metadata allowlist without tags;
#: ``tine-sig/2`` (0.7.1+) signs every metadata key but the integrity block,
#: and ``tine-sig/3`` (the v4 format, default since 0.26) binds the tag
#: annotation chain.
TAG_COVERING_SCHEMES = frozenset({"tine-sig/2", "tine-sig/3"})

#: Hard ceiling for the fallback reader on opentine builds that predate the
#: bounded ``_artifact_io`` reader. Mirrors opentine's own 256 MiB limit.
_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024


def read_artifact_document(path: str | Path) -> Any:
    """Parse a ``.tine`` file into its raw JSON document, whatever its form.

    opentine 0.11+ writes compressed artifacts and 0.26+ writes the v4
    object format by default, so ``json.loads(path.read_text())`` no longer
    works. This defers to opentine's bounded reader (size cap, inflate cap,
    duplicate-key and non-finite-number refusal) and only falls back to a
    size-capped plain JSON read on builds too old to have it.
    """
    try:
        from opentine._artifact_io import read_artifact_json
    except ImportError:  # pragma: no cover - opentine < 0.11
        import json

        source = Path(path)
        if source.stat().st_size > _MAX_ARTIFACT_BYTES:
            raise ValueError(".tine artifact exceeds the 256 MiB size limit") from None
        return json.loads(source.read_text(encoding="utf-8"))
    return read_artifact_json(path)


def write_artifact_document(path: str | Path, document: Any) -> Path:
    """Write a raw artifact document back as plain JSON (every reader accepts it).

    Intended for tooling and tests that edit an artifact by hand; normal
    recording always goes through :meth:`GraphRecorder.save`.
    """
    import json

    target = Path(path)
    target.write_text(json.dumps(document), encoding="utf-8")
    return target


def artifact_signature_info(document: Any) -> dict[str, Any]:
    """Describe the signature(s) an artifact document carries.

    Handles the legacy v2 layout (``metadata.integrity.signature``) and the
    v4 layout (top-level ``signatures`` list). Never raises: a malformed
    document simply reports no signature.
    """
    info: dict[str, Any] = {
        "format_version": None,
        "signature_present": False,
        "signature_schemes": [],
        "tags_covered_by_signature": False,
    }
    if not isinstance(document, dict):
        return info
    version = document.get("format_version")
    info["format_version"] = version if type(version) is int else None
    blocks: list[Any] = []
    signatures = document.get("signatures")
    if isinstance(signatures, list):
        blocks.extend(signatures)
    metadata = document.get("metadata")
    if isinstance(metadata, dict):
        integrity = metadata.get("integrity")
        if isinstance(integrity, dict) and integrity.get("signature"):
            blocks.append(integrity["signature"])
    schemes = sorted(
        {
            str(block.get("scheme") or "tine-sig/1")
            for block in blocks
            if isinstance(block, dict)
        }
    )
    info["signature_present"] = bool(schemes)
    info["signature_schemes"] = schemes
    info["tags_covered_by_signature"] = bool(schemes) and all(
        scheme in TAG_COVERING_SCHEMES for scheme in schemes
    )
    return info


def min_hmac_key_bytes() -> int:
    """The shortest HMAC key the installed opentine accepts (0 = no floor)."""
    try:
        from opentine.signing import MIN_HMAC_KEY_BYTES
    except ImportError:  # pragma: no cover - older opentine had no floor
        return 0
    return int(MIN_HMAC_KEY_BYTES)


def check_hmac_key(key: Any, algorithm: str) -> None:
    """Refuse an HMAC key shorter than opentine's floor before anything is written."""
    if not algorithm.startswith("hmac") or not isinstance(key, bytes | str):
        return
    size = len(key.encode("utf-8") if isinstance(key, str) else key)
    floor = min_hmac_key_bytes()
    if size < floor:
        raise ValueError(f"HMAC signing key is {size} bytes; opentine requires at least {floor}")


_KIND_MAP = {
    "model": StepKind.model,
    "tool": StepKind.tool,
    "think": StepKind.think,
}


class GraphRecorder:
    """Records one graph run into one opentine ``Run``."""

    def __init__(
        self,
        goal: str,
        *,
        context: str = "",
        policy: GraphPolicy | None = None,
        graph: dict[str, Any] | None = None,
        runs_dir: str | Path | None = None,
        state_snapshots: bool = True,
        run: Run | None = None,
    ) -> None:
        self.goal = goal
        self.context = context
        self.policy = policy
        self.state_snapshots = state_snapshots
        self.runs_dir = Path(runs_dir).expanduser() if runs_dir else DEFAULT_RUNS_DIR
        self.path: Path | None = None
        self._tip: str | None = None
        self._finished = False
        self._call_counter = 0
        if run is not None:
            self.run = run
        else:
            manifest: dict[str, Any] = {
                "kind": "graphforge",
                "goal": clean_text(goal),
                "context": clean_text(context),
                "version": __version__,
            }
            if graph is not None:
                manifest["graph"] = to_jsonable(graph)
            policies = {"graph": policy.to_dict()} if policy else {}
            self.run = Run(
                manifest=manifest,
                status=RunStatus.running,
                user_prompt=clean_text(goal),
                policies=policies,
            )
            if policy is not None:
                self._apply_budget(policy)

    # ------------------------------------------------------------------ fork

    @classmethod
    def from_fork(
        cls,
        parent_run: Run,
        from_step_id: str,
        *,
        branch: str = "fork",
        goal: str | None = None,
        context: str = "",
        policy: GraphPolicy | None = None,
        runs_dir: str | Path | None = None,
        state_snapshots: bool = True,
    ) -> GraphRecorder:
        """Fork an existing run at a step and return a recorder over the fork.

        The fork is always uniquely identified — see :func:`fork_run` for how
        that is achieved on each opentine version.
        """
        manifest = parent_run.manifest if isinstance(parent_run.manifest, dict) else {}
        forked = fork_run(parent_run, from_step_id, branch=branch)
        forked.status = RunStatus.running
        recorder = cls(
            goal=goal or str(manifest.get("goal", "")),
            context=context or str(manifest.get("context", "")),
            policy=policy,
            runs_dir=runs_dir,
            state_snapshots=state_snapshots,
            run=forked,
        )
        if policy is not None:
            forked.policies["graph"] = policy.to_dict()
        recorder._tip = from_step_id
        return recorder

    # ----------------------------------------------------------------- steps

    @property
    def run_id(self) -> str:
        return self.run.run_id

    @property
    def tip(self) -> str | None:
        return self._tip

    def record_node(
        self,
        node_id: str,
        *,
        kind: str = "think",
        visit: int = 0,
        edge: dict[str, Any] | None = None,
        state_before: dict[str, Any] | None = None,
        state_after: dict[str, Any] | None = None,
        delta: dict[str, Any] | None = None,
        observation: str = "",
        score: float | None = None,
        outputs: dict[str, Any] | None = None,
        parent_id: str | None = None,
        parent_ids: list[str] | None = None,
        duration: float = 0.0,
        cost: float = 0.0,
        usage: dict[str, Any] | None = None,
    ) -> str:
        """Record one node execution as one step; returns the step id."""
        inputs: dict[str, Any] = {
            "node": clean_text(node_id),
            "visit": visit,
            "edge": safe_json(edge) if edge else {"kind": "entry"},
        }
        if state_before is not None:
            inputs["state_digest_before"] = state_digest(state_before)
        outs: dict[str, Any] = {"observation": safe_json(observation)}
        clean_score = sane_score(score)
        if clean_score is not None:
            outs["score"] = clean_score
        if state_after is not None:
            outs["state_digest"] = state_digest(state_after)
            if delta is not None:
                outs["delta"] = delta
            if self.state_snapshots:
                # Structural snapshot for resume/fork. Caveat: opentine
                # redacts credential-named keys at save time, so such values
                # do not round-trip — resume verifies against state_digest.
                outs["state_after"] = to_jsonable(state_after)
        if outputs:
            outs["outputs"] = safe_json(outputs)
        return self._add(
            _KIND_MAP.get(kind, StepKind.think),
            inputs=inputs,
            outputs=outs,
            parent_id=parent_id,
            parent_ids=parent_ids,
            duration=duration,
            cost=cost,
            usage=usage,
        )

    def record_model_call(
        self,
        *,
        model: str = "",
        prompt: Any = None,
        response: Any = None,
        cost: float = 0.0,
        duration: float = 0.0,
        usage: dict[str, Any] | None = None,
        parent_id: str | None = None,
        node_id: str | None = None,
        provider: str | None = None,
        effective_at: Any = None,
        service_tier: str | None = None,
    ) -> str:
        """Record a model invocation (nested under a node step when given).

        When ``provider`` is given and the installed opentine can price the
        call, the step carries a full billing record — catalog id, rate-card
        id, and an explicit complete/partial/unknown status — instead of an
        unattributed cost float. Pricing never fails the recording.
        """
        self._call_counter += 1
        # call_index keeps identical repeated calls content-address-unique.
        inputs: dict[str, Any] = {
            "model": clean_text(model),
            "prompt": safe_json(prompt),
            "call_index": self._call_counter,
        }
        if node_id:
            inputs["node"] = clean_text(node_id)
        priced: dict[str, Any] = {}
        outputs: dict[str, Any] = {"response": safe_json(response)}
        if provider:
            from .pricing import billing_fields, price_call, pricing_note

            result = price_call(
                provider,
                model,
                usage,
                effective_at=effective_at,
                service_tier=service_tier,
            )
            priced = billing_fields(result)
            note = pricing_note(result)
            if note is not None:
                # Pricing was attempted and came back unknown; say so rather
                # than letting a zero subtotal overwrite the caller's cost.
                outputs["pricing"] = note
        return self._add(
            StepKind.model,
            inputs=inputs,
            outputs=outputs,
            parent_id=parent_id,
            duration=duration,
            cost=priced.get("cost", cost),
            usage=usage,
            model_info=clean_text(model) or None,
            billing=priced.get("billing"),
        )

    def record_tool_call(
        self,
        name: str,
        *,
        args: Any = None,
        result: Any = None,
        error: str | None = None,
        cost: float = 0.0,
        duration: float = 0.0,
        parent_id: str | None = None,
        node_id: str | None = None,
    ) -> str:
        self._call_counter += 1
        inputs: dict[str, Any] = {
            "tool": clean_text(name),
            "args": safe_json(args),
            "call_index": self._call_counter,
        }
        if node_id:
            inputs["node"] = clean_text(node_id)
        outputs: dict[str, Any] = {"result": safe_json(result)}
        if error:
            outputs["error"] = safe_json(error)
        return self._add(
            StepKind.tool,
            inputs=inputs,
            outputs=outputs,
            parent_id=parent_id,
            duration=duration,
            cost=cost,
            tool_info={"name": clean_text(name)},
        )

    def record_checkpoint(
        self,
        checkpoint_id: str,
        *,
        state: dict[str, Any] | None = None,
        parent_checkpoint_id: str | None = None,
        source: str = "",
        step: int | None = None,
        parent_id: str | None = None,
    ) -> str:
        """Record an externally-created checkpoint as a forkable marker step."""
        inputs: dict[str, Any] = {
            "checkpoint": clean_text(checkpoint_id),
            "source": clean_text(source),
        }
        if parent_checkpoint_id:
            inputs["parent_checkpoint"] = clean_text(parent_checkpoint_id)
        if step is not None:
            inputs["step"] = step
        outputs: dict[str, Any] | None = None
        if state is not None:
            # to_jsonable (not safe_json): checkpoint snapshots are fork
            # points and must not be truncated.
            outputs = {"state_digest": state_digest(state), "state_after": to_jsonable(state)}
        return self._add(StepKind.think, inputs=inputs, outputs=outputs, parent_id=parent_id)

    def record_gate(
        self, gate: str, detail: dict[str, Any] | None = None, *, parent_id: str | None = None
    ) -> str:
        """Record a policy gate (budget breach, prune, cycle guard) honestly.

        The gate also lands in the step body, which the integrity digest and
        any signature cover — tags are outside both and can be stripped.
        """
        self.add_tag(f"gate:{gate}")
        return self._add(
            StepKind.think,
            inputs={"gate": clean_text(str(gate)), "detail": safe_json(detail or {})},
            outputs=None,
            parent_id=parent_id,
        )

    def record_done(self, summary: str, *, parent_id: str | None = None) -> str:
        step_id = self._add(
            StepKind.done,
            inputs={"summary": safe_json(summary)},
            outputs=None,
            parent_id=parent_id,
        )
        self.run.status = RunStatus.completed
        self._finished = True
        return step_id

    def record_error(self, error: BaseException | str, *, parent_id: str | None = None) -> str:
        if isinstance(error, BaseException):
            message = str(error) or error.__class__.__name__
            kind_name = error.__class__.__name__
        else:
            message = str(error)
            kind_name = "error"
        step_id = self._add(
            StepKind.error,
            inputs={"error": safe_json(message), "error_type": kind_name},
            outputs=None,
            parent_id=parent_id,
            # Step.error is first-class in both 0.2.0 and 0.3.0; without it
            # opentine's tooling cannot tell a failure from a think step.
            error={"type": kind_name, "message": safe_json(message)},
        )
        self.run.status = RunStatus.failed
        self._finished = True
        return step_id

    @property
    def finished(self) -> bool:
        return self._finished

    def add_tag(self, tag: str) -> None:
        """Add a run tag, preferring opentine's normalizing ``add_tag``.

        The v3 repository's annotation writer rejects non-string tags *after*
        writing every event object, so normalization matters. Tags live
        outside the integrity digest and outside the signature's metadata
        allowlist — anything that must be tamper-evident also goes in a step
        body, never in a tag alone.
        """
        cleaned = clean_text(str(tag))
        adder = getattr(self.run, "add_tag", None)
        if callable(adder):
            adder(cleaned)
            return
        if cleaned not in self.run.tags:  # pragma: no cover - 0.2.0 fallback
            self.run.tags.append(cleaned)

    def _apply_budget(self, policy: GraphPolicy) -> None:
        """Declare the policy's ceilings as an opentine Budget.

        graphforge enforces its own budgets, but recording them in
        ``manifest.budget`` (inside the digest) is what lets ``tine cost``,
        the run index, and any v3 consumer see that the run was governed.
        """
        setter = getattr(self.run, "set_budget", None)
        if not callable(setter):  # pragma: no cover - defensive
            return
        kwargs: dict[str, Any] = {}
        # Explicit coercion: 0.3.0 validates budget dimensions that 0.2.0
        # accepted, and GraphPolicy fields are unvalidated (JSON may hold 3.0).
        if policy.max_total_cost is not None:
            kwargs["max_cost"] = float(policy.max_total_cost)
        # Deliberately NOT mapping max_node_executions onto max_steps:
        # opentine counts every recorded step (gate steps, the terminal step,
        # nested model/tool calls) while graphforge counts node executions
        # only, so the run would self-report as over budget. The node cap
        # stays in run.policies["graph"], where a graph-aware reader finds it.
        if policy.max_duration_seconds is not None:
            kwargs["max_duration"] = float(policy.max_duration_seconds)
        if not kwargs:
            return
        try:
            setter(on_breach="stop", **kwargs)
        except (ValueError, TypeError):
            # A nonsensical policy must not stop the run from being recorded.
            pass

    def _add(
        self,
        kind: StepKind,
        *,
        inputs: dict[str, Any],
        outputs: dict[str, Any] | None,
        parent_id: str | None = None,
        parent_ids: list[str] | None = None,
        duration: float = 0.0,
        cost: float = 0.0,
        usage: dict[str, Any] | None = None,
        model_info: str | None = None,
        tool_info: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        billing: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "inputs": inputs,
            "outputs": outputs,
            "duration": sane_metric(duration),
            "cost": sane_metric(cost),
        }
        clean_usage = sane_usage(usage)
        if clean_usage:
            kwargs["usage"] = clean_usage
        # model_info/tool_info drive opentine's cost-by-model breakdown, its
        # run index, and v3 search; without them every step is one anonymous
        # bucket. Both kwargs exist in 0.2.0 and 0.3.0.
        if model_info:
            kwargs["model_info"] = model_info
        if tool_info:
            kwargs["tool_info"] = tool_info
        if error:
            kwargs["error"] = error
        if billing:
            # add_step(billing=) is 0.3.0-only; on 0.2.0 it is a TypeError,
            # and the pricing layer never produces a record there anyway.
            kwargs["billing"] = billing
        if parent_ids:
            kwargs["parent_ids"] = parent_ids
        else:
            kwargs["parent_id"] = parent_id if parent_id is not None else self._tip
        step = self.run.add_step(kind, **kwargs)
        self._tip = step.id
        return step.id

    # ------------------------------------------------------------------ save

    def _target(self, path: str | Path | None) -> Path:
        target = Path(path).expanduser() if path else self.runs_dir / f"{self.run_id}.tine"
        if target.is_dir():
            # opentine silently switches to v3 repository mode for directory
            # targets (and would advance the repo's heads/main); require an
            # explicit file path so that never happens by accident. Use
            # graphforge.repo_backend to write into a repository on purpose.
            raise ValueError(
                f"save target {target} is a directory; pass a .tine file path "
                "(use graphforge.repo_backend.save_to_repo for a v3 repository)"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def save(
        self,
        path: str | Path | None = None,
        *,
        sign_key: bytes | str | None = None,
        sign_algorithm: str = "hmac-sha256",
        key_id: str | None = None,
        signer: str | None = None,
    ) -> Path:
        """Write the run as a portable ``.tine`` file, optionally signed.

        Signing covers the run body plus opentine's metadata allowlist,
        whose membership is version-dependent (it grew in 0.4.0). Tags are
        NOT covered in any version — anything that must be tamper-evident is
        recorded in a step body as well.
        """
        target = self._target(path)
        kwargs: dict[str, Any] = {}
        if sign_key is not None:
            check_hmac_key(sign_key, sign_algorithm)
            kwargs["sign_key"] = sign_key
            kwargs["sign_algorithm"] = sign_algorithm
            if key_id is not None:
                kwargs["key_id"] = key_id
            if signer is not None:
                kwargs["signer"] = signer
        self.path = self.run.save(target, **kwargs)
        return self.path

    def pause(self, path: str | Path | None = None) -> Path:
        """Park the run as paused so opentine tooling can resume it.

        A graph waiting on a human-in-the-loop checkpoint is otherwise
        indistinguishable from a completed one.
        """
        target = self._target(path)
        self.run.status = RunStatus.paused
        self._finished = True
        pauser = getattr(self.run, "pause", None)
        self.path = pauser(target) if callable(pauser) else self.run.save(target)
        return self.path


# ------------------------------------------------------------------ signing


def load_ed25519_private_key(path: str | Path) -> Any:
    """Load an Ed25519 private key file the way opentine writes them."""
    from opentine.signing import ed25519_private_from_file

    return ed25519_private_from_file(Path(path).expanduser())


def load_ed25519_public_key(path: str | Path) -> Any:
    """Load an Ed25519 public key file, in any encoding opentine accepts.

    opentine's own key files are hex text; passing their raw bytes to
    ``Run.verify_signature`` fails as "malformed", so go through opentine's
    loader when it exists and only fall back to raw bytes on old builds.
    """
    source = Path(path).expanduser()
    try:
        from opentine.signing import ed25519_public_from_file
    except ImportError:  # pragma: no cover - opentine without Ed25519 helpers
        return source.read_bytes()
    return ed25519_public_from_file(source)


def sign_artifact_file(
    source: str | Path,
    *,
    key: Any,
    algorithm: str = "hmac-sha256",
    signer: str | None = None,
    key_id: str | None = None,
    output: str | Path | None = None,
) -> Path:
    """Sign an existing finished artifact, in place or into ``output``.

    The run is loaded and re-saved through opentine, so the signature is
    always computed over exactly what opentine would verify. opentine
    refuses to sign a run that is still running.
    """
    src = Path(source).expanduser()
    target = Path(output).expanduser() if output else src
    if target.is_dir():
        raise ValueError(f"output {target} is a directory; pass a .tine file path")
    check_hmac_key(key, algorithm)
    run = Run.load(src)
    kwargs: dict[str, Any] = {"sign_key": key, "sign_algorithm": algorithm}
    if signer is not None:
        kwargs["signer"] = clean_text(signer)
    if key_id is not None:
        kwargs["key_id"] = clean_text(key_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    return run.save(target, **kwargs)
