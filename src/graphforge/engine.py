"""GraphEngine — executes a GraphSpec and records every node into opentine.

Execution model: a frontier of tasks over ONE recorded run. A node function
receives a :class:`GraphNodeContext` and returns a :class:`GraphNodeResult`
whose ``update`` dict is merged into that branch's state (LangGraph-style
partial updates). Routing priority: ``result.goto`` > registered router >
static edges > branch terminates. Multiple targets fan out (each branch gets
a deep copy of state); join nodes park arrivals and fire — with multi-parent
step lineage — once the rest of the frontier drains. Cycles are legal and
bounded by ``GraphPolicy.max_visits_per_node``.

Every recorded step stores the edge taken and a lossless post-node state
snapshot, which is what makes :meth:`GraphEngine.resume` possible: fork any
artifact at any step, mutate state, and continue the graph from there.
"""

from __future__ import annotations

import copy
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opentine import Run

from .policy import GraphPolicy
from .recorder import GraphRecorder
from .spec import END, GraphSpec, NodeSpec
from .state import sane_score, state_delta, state_digest


class GraphExecutionError(RuntimeError):
    """Raised for unrecoverable engine-level failures (not node exceptions)."""


@dataclass
class GraphNodeContext:
    """Read-only view handed to node functions."""

    node_id: str
    visit: int
    iteration: int
    state: dict[str, Any]
    goal: str
    run_id: str
    branch: str
    edge: dict[str, Any] | None
    elapsed_seconds: float
    accumulated_cost: float
    best_score: float | None


@dataclass
class GraphNodeResult:
    """What a node returns: state update, routing, score, metrics."""

    observation: str = ""
    update: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    stop: bool = False
    goto: str | Sequence[str] | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    duration: float = 0.0
    usage: dict[str, Any] | None = None


@dataclass(frozen=True)
class BestSnapshot:
    """Immutable snapshot taken the moment a best score is observed."""

    node_id: str
    step_id: str
    score: float
    state: dict[str, Any]
    visit: int


@dataclass
class _Task:
    node_id: str
    state: dict[str, Any]
    parent_step_id: str | None
    edge: dict[str, Any] | None
    branch: str


@dataclass
class _JoinArrival:
    state: dict[str, Any]
    parent_step_id: str | None
    source: str


@dataclass
class GraphExecutionResult:
    status: str
    stop_reason: str | None
    final_states: list[dict[str, Any]]
    best: BestSnapshot | None
    path: list[dict[str, Any]]
    artifact: Path | None
    run_id: str
    recorder: GraphRecorder

    @property
    def final_state(self) -> dict[str, Any]:
        return self.final_states[-1] if self.final_states else {}


class GraphEngine:
    def __init__(
        self,
        graph: GraphSpec,
        *,
        policy: GraphPolicy | None = None,
        runs_dir: str | Path | None = None,
        state_snapshots: bool = True,
        max_node_executions: int = 200,
    ) -> None:
        graph.validate()
        if max_node_executions < 1:
            raise ValueError("max_node_executions must be >= 1")
        self.graph = graph
        self.runs_dir = runs_dir
        self.state_snapshots = state_snapshots
        policy = policy or GraphPolicy()
        if policy.max_node_executions is None:
            policy = GraphPolicy(**{**policy.to_dict(), "max_node_executions": max_node_executions})
        self.policy = policy

    # ------------------------------------------------------------------- run

    def run(
        self,
        *,
        goal: str,
        initial_state: dict[str, Any] | None = None,
        context: str = "",
    ) -> GraphExecutionResult:
        recorder = GraphRecorder(
            goal,
            context=context,
            policy=self.policy,
            graph=self.graph.describe(),
            runs_dir=self.runs_dir,
            state_snapshots=self.state_snapshots,
        )
        entry = self.graph.entry
        assert entry is not None  # validate() guarantees it
        frontier: deque[_Task] = deque(
            [
                _Task(
                    node_id=entry,
                    state=copy.deepcopy(initial_state or {}),
                    parent_step_id=None,
                    edge={"kind": "entry"},
                    branch="main",
                )
            ]
        )
        return self._execute(recorder, frontier, goal=goal)

    # ---------------------------------------------------------------- resume

    def resume(
        self,
        source: Run | str | Path,
        from_step: str,
        *,
        goal: str | None = None,
        state_update: dict[str, Any] | None = None,
        start_node: str | None = None,
        branch: str = "fork",
    ) -> GraphExecutionResult:
        """Fork a recorded artifact at a step and continue the graph from there.

        State is rebuilt from the step's recorded ``state_after`` snapshot,
        optionally mutated with ``state_update``. The next node is recomputed
        from the forked node's routing against the (possibly mutated) state,
        or forced with ``start_node``.
        """
        run = source if isinstance(source, Run) else Run.load(Path(source).expanduser())
        step = run.graph.steps.get(from_step)
        if step is None:
            matches = [sid for sid in run.graph.order if sid.startswith(from_step)]
            if len(matches) != 1:
                raise GraphExecutionError(
                    f"step {from_step!r} not found (or ambiguous) in run {run.run_id}"
                )
            from_step = matches[0]
            step = run.graph.steps[from_step]
        outputs = step.outputs or {}
        inputs = step.inputs or {}
        node_id = inputs.get("node")
        if node_id is None or node_id not in self.graph.nodes:
            raise GraphExecutionError(
                f"step {from_step[:12]} did not record a node known to this graph"
            )
        snapshot = outputs.get("state_after")
        if snapshot is None:
            raise GraphExecutionError(
                "step has no state_after snapshot (recorded with state_snapshots=False); "
                "resume needs an explicit state via state_update covering the full state"
            )
        snapshot_ok = True
        recorded_digest = outputs.get("state_digest")
        if isinstance(snapshot, dict) and recorded_digest:
            if state_digest(snapshot) != recorded_digest:
                # opentine redacts credential-named keys at save time, so a
                # snapshot may not round-trip. Loud, never silent.
                redacted = _keys_with_redaction(snapshot)
                uncovered = [k for k in redacted if k not in (state_update or {})]
                if uncovered or not redacted:
                    detail = (
                        f"keys {uncovered} were redacted at save time "
                        "(opentine blanks credential-named keys); supply their "
                        "values via state_update"
                        if redacted
                        else "the snapshot does not match its recorded digest"
                    )
                    raise GraphExecutionError(
                        f"cannot resume from step {from_step[:12]}: {detail}"
                    )
                snapshot_ok = False
        state = copy.deepcopy(snapshot) if isinstance(snapshot, dict) else {}
        if state_update:
            state.update(copy.deepcopy(state_update))

        manifest = run.manifest if isinstance(run.manifest, dict) else {}
        recorded_graph = manifest.get("graph") or {}
        recorder = GraphRecorder.from_fork(
            run,
            from_step,
            branch=branch,
            goal=goal or str(manifest.get("goal", "")),
            policy=self.policy,
            runs_dir=self.runs_dir,
            state_snapshots=self.state_snapshots,
        )
        if (
            recorded_graph.get("topology_digest")
            and recorded_graph["topology_digest"] != self.graph.topology_digest()
        ):
            tag = "graph:topology-drift"
            if tag not in recorder.run.tags:
                recorder.run.tags.append(tag)
        if not snapshot_ok and "graph:snapshot-redacted" not in recorder.run.tags:
            recorder.run.tags.append("graph:snapshot-redacted")
        # The artifact must describe the graph that actually executes the
        # continuation, not the stale parent topology.
        fork_manifest = recorder.run.manifest
        if isinstance(fork_manifest, dict):
            if recorded_graph:
                fork_manifest["graph_at_record"] = recorded_graph
            fork_manifest["graph"] = self.graph.describe()
        # Continue visit counting where the retained lineage left off so one
        # artifact never contains duplicate (node, visit) pairs.
        initial_visits: dict[str, int] = {}
        for sid in recorder.run.graph.order:
            step_inputs = recorder.run.graph.steps[sid].inputs or {}
            if isinstance(step_inputs, dict) and "visit" in step_inputs:
                seen_node = step_inputs.get("node")
                if seen_node:
                    initial_visits[seen_node] = initial_visits.get(seen_node, 0) + 1

        if start_node is not None:
            if start_node not in self.graph.nodes:
                raise GraphExecutionError(f"start_node {start_node!r} is not in the graph")
            targets = [(start_node, {"kind": "resume", "source": node_id})]
        else:
            resolved = self._resolve_targets(self.graph.nodes[node_id], None, state)
            targets = [
                (t, {"kind": "resume", "source": node_id, "label": label})
                for t, label in resolved
            ]
        if not targets:
            raise GraphExecutionError(
                f"node {node_id!r} routes nowhere from the recorded state; "
                "pass start_node to choose where to resume"
            )
        frontier: deque[_Task] = deque(
            _Task(
                node_id=t,
                state=copy.deepcopy(state),
                parent_step_id=from_step,
                edge=edge,
                branch=branch,
            )
            for t, edge in targets
        )
        return self._execute(
            recorder,
            frontier,
            goal=goal or str(manifest.get("goal", "")),
            initial_visits=initial_visits,
        )

    # ----------------------------------------------------------------- core

    def _execute(
        self,
        recorder: GraphRecorder,
        frontier: deque[_Task],
        *,
        goal: str,
        initial_visits: dict[str, int] | None = None,
    ) -> GraphExecutionResult:
        started = time.monotonic()
        visits: dict[str, int] = dict(initial_visits or {})
        joins: dict[str, list[_JoinArrival]] = {}
        path: list[dict[str, Any]] = []
        terminal_states: list[dict[str, Any]] = []
        best: BestSnapshot | None = None
        total_cost = 0.0
        executed = 0
        stop_reason: str | None = None
        status = "completed"
        last_step_id: str | None = None

        while frontier or joins:
            if frontier:
                task = frontier.popleft()
                node = self.graph.nodes[task.node_id]
                if node.join:
                    joins.setdefault(task.node_id, []).append(
                        _JoinArrival(
                            state=task.state,
                            parent_step_id=task.parent_step_id,
                            source=(task.edge or {}).get("source", ""),
                        )
                    )
                    continue
                parent_ids = None
                parent_id = task.parent_step_id
            else:
                # Frontier drained: fire a pending join that no OTHER pending
                # join can still feed (upstream joins first), so downstream
                # joins never fire with partial arrivals.
                node_id = self._pick_ready_join(joins)
                arrivals = joins.pop(node_id)
                node = self.graph.nodes[node_id]
                try:
                    merged = self._reduce_join(node, arrivals)
                except Exception as exc:  # noqa: BLE001 - reducer failures must not lose the trace
                    parents = [a.parent_step_id for a in arrivals if a.parent_step_id]
                    recorder.record_error(exc, parent_id=parents[-1] if parents else last_step_id)
                    stop_reason = "join_error"
                    status = "failed"
                    terminal_states.extend(a.state for a in arrivals)
                    break
                parents = [a.parent_step_id for a in arrivals if a.parent_step_id]
                task = _Task(
                    node_id=node_id,
                    state=merged,
                    parent_step_id=parents[-1] if parents else None,
                    edge={
                        "kind": "join",
                        "sources": [a.source for a in arrivals if a.source],
                    },
                    branch="join",
                )
                parent_ids = parents if len(parents) > 1 else None
                parent_id = None if parent_ids else task.parent_step_id

            elapsed = time.monotonic() - started
            visit = visits.get(task.node_id, 0) + 1

            gate = self._gate(executed, total_cost, elapsed, task.node_id, visit)
            if gate is not None:
                recorder.record_gate(gate["gate"], gate, parent_id=parent_id or last_step_id)
                stop_reason = gate["gate"]
                status = f"stopped:{gate['gate']}"
                terminal_states.append(task.state)
                break

            visits[task.node_id] = visit
            ctx = GraphNodeContext(
                node_id=task.node_id,
                visit=visit,
                iteration=executed,
                state=task.state,
                goal=goal,
                run_id=recorder.run_id,
                branch=task.branch,
                edge=task.edge,
                elapsed_seconds=elapsed,
                accumulated_cost=total_cost,
                best_score=best.score if best else None,
            )
            state_before = copy.deepcopy(task.state)
            node_started = time.monotonic()
            try:
                result = node.fn(ctx)
            except Exception as exc:  # noqa: BLE001 - node failures end the run honestly
                recorder.record_error(exc, parent_id=parent_id or last_step_id)
                stop_reason = "node_error"
                status = "failed"
                terminal_states.append(task.state)
                break
            if result is None:
                result = GraphNodeResult()
            if not isinstance(result, GraphNodeResult):
                recorder.record_error(
                    TypeError(
                        f"node {task.node_id!r} returned {type(result).__name__}, "
                        "expected GraphNodeResult"
                    ),
                    parent_id=parent_id or last_step_id,
                )
                stop_reason = "node_error"
                status = "failed"
                terminal_states.append(task.state)
                break
            node_duration = result.duration or (time.monotonic() - node_started)

            new_state = dict(task.state)
            if result.update:
                try:
                    new_state.update(copy.deepcopy(result.update))
                except Exception as exc:  # noqa: BLE001 - pathological updates must not lose the trace
                    recorder.record_error(exc, parent_id=parent_id or last_step_id)
                    stop_reason = "state_error"
                    status = "failed"
                    terminal_states.append(task.state)
                    break

            step_id = recorder.record_node(
                task.node_id,
                kind=node.kind,
                visit=visit,
                edge=task.edge,
                state_before=state_before,
                state_after=new_state,
                delta=state_delta(state_before, new_state),
                observation=result.observation,
                score=result.score,
                outputs=result.outputs,
                parent_id=parent_id,
                parent_ids=parent_ids,
                duration=node_duration,
                cost=result.cost,
                usage=result.usage,
            )
            executed += 1
            total_cost += max(float(result.cost or 0.0), 0.0)
            last_step_id = step_id
            path.append(
                {
                    "node": task.node_id,
                    "step_id": step_id,
                    "visit": visit,
                    "branch": task.branch,
                    "score": result.score,
                }
            )

            clean_score = sane_score(result.score)
            if clean_score is not None and (best is None or clean_score > best.score):
                best = BestSnapshot(
                    node_id=task.node_id,
                    step_id=step_id,
                    score=clean_score,
                    state=copy.deepcopy(new_state),
                    visit=visit,
                )

            # Post-node gates: wall clock (catches slow nodes immediately),
            # success threshold, explicit stop.
            elapsed = time.monotonic() - started
            if (
                self.policy.max_duration_seconds is not None
                and elapsed > self.policy.max_duration_seconds
            ):
                recorder.record_gate(
                    "max_duration_seconds",
                    {"elapsed_seconds": round(elapsed, 3)},
                    parent_id=step_id,
                )
                stop_reason = "max_duration_seconds"
                status = "stopped:max_duration_seconds"
                terminal_states.append(new_state)
                break
            if (
                self.policy.min_score is not None
                and result.score is not None
                and result.score >= self.policy.min_score
            ):
                stop_reason = "min_score"
                terminal_states.append(new_state)
                break
            if result.stop:
                stop_reason = "stop"
                terminal_states.append(new_state)
                break

            try:
                targets = self._resolve_targets(node, result, new_state)
            except Exception as exc:  # noqa: BLE001 - router failures must not lose the trace
                recorder.record_error(exc, parent_id=step_id)
                stop_reason = "route_error"
                status = "failed"
                terminal_states.append(new_state)
                break
            if not targets:
                terminal_states.append(new_state)
                continue
            fan_out = len(targets) > 1
            for index, (target, label) in enumerate(targets):
                edge_kind = (
                    "goto"
                    if result.goto is not None
                    else ("router" if node.node_id in self.graph.routers else "static")
                )
                frontier.append(
                    _Task(
                        node_id=target,
                        state=copy.deepcopy(new_state) if fan_out else new_state,
                        parent_step_id=step_id,
                        edge={"kind": edge_kind, "source": node.node_id, "label": label},
                        branch=f"{task.branch}.{index}" if fan_out else task.branch,
                    )
                )

        if joins:
            # Joins that never fired because something stopped the run first.
            # Keep the real stop reason but surface the parked arrivals: tag
            # the run and prepend their states so nothing silently vanishes
            # (final_state stays the state of the branch that stopped the run).
            tag = "graph:unresolved_join"
            if tag not in recorder.run.tags:
                recorder.run.tags.append(tag)
            parked = [a.state for arrivals in joins.values() for a in arrivals]
            terminal_states[:0] = parked

        if not recorder.finished:
            recorder.record_done(stop_reason or "completed", parent_id=last_step_id)
        artifact = recorder.save()
        return GraphExecutionResult(
            status=status,
            stop_reason=stop_reason,
            final_states=terminal_states,
            best=best,
            path=path,
            artifact=artifact,
            run_id=recorder.run_id,
            recorder=recorder,
        )

    # -------------------------------------------------------------- helpers

    def _pick_ready_join(self, joins: dict[str, list[_JoinArrival]]) -> str:
        """Pick a pending join that no other pending join can still feed.

        Uses static edge reachability; a router anywhere on a path makes
        reachability unknowable, treated conservatively as "reaches
        everything" (``goto`` routing is not considered). Falls back to
        first-inserted when every pending join is potentially fed by another
        (join cycles), so drain never deadlocks.
        """
        pending = list(joins)
        if len(pending) == 1:
            return pending[0]
        reach = {node_id: self._reachable_from(node_id) for node_id in pending}
        for candidate in pending:
            fed = any(
                reach[other] is None or candidate in reach[other]
                for other in pending
                if other != candidate
            )
            if not fed:
                return candidate
        return pending[0]

    def _reachable_from(self, start: str) -> set[str] | None:
        """Statically reachable nodes from ``start``; None = unbounded (router)."""
        seen: set[str] = set()
        stack = [start]
        while stack:
            node_id = stack.pop()
            if node_id in self.graph.routers:
                return None
            for edge in self.graph.outgoing(node_id):
                if edge.target != END and edge.target not in seen:
                    seen.add(edge.target)
                    stack.append(edge.target)
        return seen

    def _gate(
        self,
        executed: int,
        total_cost: float,
        elapsed: float,
        node_id: str,
        visit: int,
    ) -> dict[str, Any] | None:
        policy = self.policy
        if policy.max_node_executions is not None and executed >= policy.max_node_executions:
            return {"gate": "max_node_executions", "executed": executed}
        if policy.max_total_cost is not None and total_cost >= policy.max_total_cost:
            return {"gate": "max_total_cost", "total_cost": round(total_cost, 6)}
        if policy.max_duration_seconds is not None and elapsed > policy.max_duration_seconds:
            return {"gate": "max_duration_seconds", "elapsed_seconds": round(elapsed, 3)}
        if policy.max_visits_per_node is not None and visit > policy.max_visits_per_node:
            return {"gate": "max_visits_per_node", "node": node_id, "visit": visit}
        return None

    def _resolve_targets(
        self,
        node: NodeSpec,
        result: GraphNodeResult | None,
        state: dict[str, Any],
    ) -> list[tuple[str, str]]:
        """Resolve next targets as (node_id, label) pairs. END is filtered out."""
        raw: list[tuple[str, str]]
        if result is not None and result.goto is not None:
            goto = [result.goto] if isinstance(result.goto, str) else list(result.goto)
            raw = [(t, "goto") for t in goto]
        elif node.node_id in self.graph.routers:
            routed = self.graph.routers[node.node_id](state)
            if routed is None:
                routed = []
            elif isinstance(routed, str):
                routed = [routed]
            raw = [(t, "router") for t in routed]
        else:
            raw = [(e.target, e.label) for e in self.graph.outgoing(node.node_id)]
        targets: list[tuple[str, str]] = []
        for target, label in raw:
            if target == END:
                continue
            if target not in self.graph.nodes:
                raise GraphExecutionError(
                    f"node {node.node_id!r} routed to unknown node {target!r}"
                )
            targets.append((target, label))
        return targets

    @staticmethod
    def _reduce_join(node: NodeSpec, arrivals: list[_JoinArrival]) -> dict[str, Any]:
        states = [a.state for a in arrivals]
        if node.join_reducer is not None:
            return node.join_reducer(states)
        merged: dict[str, Any] = {}
        for state in states:
            merged.update(state)
        return merged


def _keys_with_redaction(state: dict[str, Any]) -> list[str]:
    """Top-level state keys whose subtree contains an opentine redaction marker."""

    def contains(value: Any) -> bool:
        if isinstance(value, str):
            return "[REDACTED]" in value
        if isinstance(value, dict):
            return any(contains(v) for v in value.values())
        if isinstance(value, list):
            return any(contains(v) for v in value)
        return False

    return sorted(key for key, value in state.items() if contains(value))
