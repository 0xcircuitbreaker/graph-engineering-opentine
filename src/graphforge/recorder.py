"""GraphRecorder — the single opentine integration point.

Every module that WRITES opentine runs goes through this file (the CLI and
MCP server only read), so an opentine version bump touches one place. The
recorder maps graph execution onto the portable v2 ``.tine`` surface that is
identical across opentine 0.2.x and 0.3.x:

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

from pathlib import Path
from typing import Any
from uuid import uuid4

from opentine import Run, RunStatus, StepKind

from ._version import __version__
from .policy import GraphPolicy
from .state import safe_json, sane_metric, sane_score, sane_usage, state_digest, to_jsonable

DEFAULT_RUNS_DIR = Path("~/.local/share/graphforge").expanduser()

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
                "goal": goal,
                "context": context,
                "version": __version__,
            }
            if graph is not None:
                manifest["graph"] = graph
            policies = {"graph": policy.to_dict()} if policy else {}
            self.run = Run(
                manifest=manifest,
                status=RunStatus.running,
                user_prompt=goal,
                policies=policies,
            )

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

        The fork gets a unique run id (opentine's default is deterministic
        from parent + fork point, so two forks at the same step would
        silently overwrite each other's artifact).
        """
        manifest = parent_run.manifest if isinstance(parent_run.manifest, dict) else {}
        new_run_id = f"{parent_run.run_id[:12]}-{branch}-{uuid4().hex[:8]}"
        forked = parent_run.fork(
            from_step_id=from_step_id, new_run_id=new_run_id, branch=branch
        )
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
            "node": node_id,
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
    ) -> str:
        """Record a model invocation (nested under a node step when given)."""
        self._call_counter += 1
        # call_index keeps identical repeated calls content-address-unique.
        inputs: dict[str, Any] = {
            "model": model,
            "prompt": safe_json(prompt),
            "call_index": self._call_counter,
        }
        if node_id:
            inputs["node"] = node_id
        return self._add(
            StepKind.model,
            inputs=inputs,
            outputs={"response": safe_json(response)},
            parent_id=parent_id,
            duration=duration,
            cost=cost,
            usage=usage,
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
            "tool": name,
            "args": safe_json(args),
            "call_index": self._call_counter,
        }
        if node_id:
            inputs["node"] = node_id
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
        inputs: dict[str, Any] = {"checkpoint": checkpoint_id, "source": source}
        if parent_checkpoint_id:
            inputs["parent_checkpoint"] = parent_checkpoint_id
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
        """Record a policy gate (budget breach, prune, cycle guard) honestly."""
        tag = f"gate:{gate}"
        if tag not in self.run.tags:
            self.run.tags.append(tag)
        return self._add(
            StepKind.think,
            inputs={"gate": gate, "detail": safe_json(detail or {})},
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
        )
        self.run.status = RunStatus.failed
        self._finished = True
        return step_id

    @property
    def finished(self) -> bool:
        return self._finished

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
        if parent_ids:
            kwargs["parent_ids"] = parent_ids
        else:
            kwargs["parent_id"] = parent_id if parent_id is not None else self._tip
        step = self.run.add_step(kind, **kwargs)
        self._tip = step.id
        return step.id

    # ------------------------------------------------------------------ save

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path).expanduser() if path else self.runs_dir / f"{self.run_id}.tine"
        if target.is_dir():
            # opentine silently switches to v3 repository mode for directory
            # targets (and would advance the repo's heads/main); require an
            # explicit file path so that never happens by accident.
            raise ValueError(
                f"save target {target} is a directory; pass a .tine file path "
                "(use opentine Repo APIs to write into a repository)"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        self.path = self.run.save(target)
        return self.path
