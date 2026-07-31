"""GraphRunCapture — record ANY graph framework's execution into opentine.

This is the universal, push-based adapter surface: if a framework exposes
callbacks, lifecycle hooks, an event bus, or a checkpointer (LangGraph,
Burr, CrewAI Flows, LlamaIndex Workflows, pydantic-graph, homegrown
engines), map its events onto ``node_start`` / ``node_end`` /
``model_call`` / ``tool_call`` / ``edge`` / ``end`` and every execution
becomes a forkable, diffable, verifiable ``.tine`` artifact.

Discipline rules (learned from production recorder integrations):

- **Never throw.** Recorder failures must never break the host framework's
  execution path. Every public method is guarded; failures are counted and
  visible via ``dropped_events``, and ``strict=True`` re-raises for tests.
- **DAG-aware parents.** Steps chain per node lineage (edge events and the
  last step of the source node), not just linearly; fan-in accepts explicit
  ``parents`` for multi-parent join steps.
- **Bounded payloads.** Everything is truncated via ``safe_json`` before
  storage.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .policy import GraphPolicy
from .recorder import GraphRecorder
from .state import clean_text, safe_json, state_delta

logger = logging.getLogger(__name__)


class _OpenNode:
    __slots__ = (
        "node_id",
        "state_before",
        "edge",
        "parent_step",
        "started",
        "model_calls",
        "tool_calls",
    )

    def __init__(
        self,
        node_id: str,
        state_before: dict[str, Any] | None,
        edge: dict[str, Any] | None,
        parent_step: str | None,
    ) -> None:
        self.node_id = node_id
        self.state_before = state_before
        self.edge = edge
        # Parent resolved at START time, so a step recorded later (parallel
        # siblings) is never mistaken for this node's cause.
        self.parent_step = parent_step
        self.started = time.monotonic()
        self.model_calls: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []


class GraphRunCapture:
    """Push-based recorder for external graph frameworks."""

    def __init__(
        self,
        goal: str,
        *,
        context: str = "",
        framework: str = "",
        graph: dict[str, Any] | None = None,
        policy: GraphPolicy | None = None,
        runs_dir: str | Path | None = None,
        state_snapshots: bool = True,
        strict: bool = False,
    ) -> None:
        self.strict = strict
        self.dropped_events = 0
        self._recorder: GraphRecorder | None = None
        self._open: dict[str, _OpenNode] = {}
        self._node_tips: dict[str, str] = {}
        self._pending_edges: dict[str, dict[str, Any]] = {}
        self._last_state: dict[str, Any] | None = None
        self._saved: Path | None = None
        try:
            self._recorder = GraphRecorder(
                goal,
                context=context or (f"framework:{framework}" if framework else ""),
                policy=policy,
                graph=dict(graph, framework=framework) if graph and framework else graph,
                runs_dir=runs_dir,
                state_snapshots=state_snapshots,
            )
            if framework:
                manifest = self._recorder.run.manifest
                if isinstance(manifest, dict):
                    manifest["framework"] = clean_text(framework)
        except Exception:
            if strict:
                raise
            logger.debug("graphforge capture disabled: recorder init failed", exc_info=True)
            self._recorder = None

    # ------------------------------------------------------------ properties

    @property
    def is_active(self) -> bool:
        return self._recorder is not None

    @property
    def run_id(self) -> str | None:
        return self._recorder.run_id if self._recorder else None

    @property
    def recorder(self) -> GraphRecorder | None:
        return self._recorder

    @property
    def artifact(self) -> Path | None:
        return self._saved

    # ---------------------------------------------------------------- events

    def begin(
        self, *, graph: dict[str, Any] | None = None, state: dict[str, Any] | None = None
    ) -> None:
        """Optionally attach a late topology and the initial state."""

        def _do() -> None:
            assert self._recorder is not None
            if graph is not None:
                manifest = self._recorder.run.manifest
                if isinstance(manifest, dict):
                    manifest["graph"] = safe_json(graph)
            if state is not None:
                self._last_state = dict(state)

        self._guard(_do)

    def edge(self, source: str, target: str, *, label: str = "") -> None:
        """Declare that an edge was taken; applied to the next start of ``target``.

        Target ``"*"`` declares a wildcard edge (e.g. a router whose label is
        known before its destination): it attaches to whichever node starts
        next.
        """

        def _do() -> None:
            self._pending_edges[target] = {
                "kind": "event",
                "source": source,
                "label": label,
            }

        self._guard(_do)

    def node_start(
        self,
        node_id: str,
        *,
        state: dict[str, Any] | None = None,
        edge: dict[str, Any] | None = None,
        key: str | None = None,
    ) -> None:
        """Open a node execution. ``key`` disambiguates concurrent executions."""

        def _do() -> None:
            # Consume any declared edge for this node even when an explicit
            # edge is given, so a stale declaration never leaks onto a later
            # unrelated execution. A "*" wildcard edge attaches to whichever
            # node starts next.
            pending = self._pending_edges.pop(node_id, None)
            if pending is None:
                pending = self._pending_edges.pop("*", None)
            resolved_edge = edge or pending
            if resolved_edge is None and self._node_tips:
                # Fall back to "last completed node fed this one".
                last_node = next(reversed(self._node_tips))
                resolved_edge = {"kind": "inferred", "source": last_node}
            open_key = key or node_id
            if open_key in self._open:
                # Same node re-opened before closing: close the previous
                # execution instead of silently dropping its buffered calls.
                self._close_open(open_key, incomplete=True)
            parent_step = None
            if resolved_edge and resolved_edge.get("source") in self._node_tips:
                parent_step = self._node_tips[resolved_edge["source"]]
            self._open[open_key] = _OpenNode(
                node_id=node_id,
                state_before=dict(state) if state is not None else self._last_state,
                edge=resolved_edge,
                parent_step=parent_step,
            )

        self._guard(_do)

    def model_call(
        self,
        *,
        model: str = "",
        prompt: Any = None,
        response: Any = None,
        cost: float = 0.0,
        duration: float = 0.0,
        usage: dict[str, Any] | None = None,
        node_id: str | None = None,
        key: str | None = None,
    ) -> None:
        """Record a model invocation, nested under the open node when one exists."""

        def _do() -> None:
            assert self._recorder is not None
            open_node = self._resolve_open(node_id, key)
            call = {
                "model": model,
                "prompt": prompt,
                "response": response,
                "cost": cost,
                "duration": duration,
                "usage": usage,
            }
            if open_node is not None:
                open_node.model_calls.append(call)
            else:
                self._recorder.record_model_call(
                    model=model,
                    prompt=prompt,
                    response=response,
                    cost=cost,
                    duration=duration,
                    usage=usage,
                    parent_id=self._node_tips.get(node_id) if node_id else None,
                    node_id=node_id,
                )

        self._guard(_do)

    def tool_call(
        self,
        name: str,
        *,
        args: Any = None,
        result: Any = None,
        error: str | None = None,
        duration: float = 0.0,
        node_id: str | None = None,
        key: str | None = None,
    ) -> None:
        def _do() -> None:
            assert self._recorder is not None
            open_node = self._resolve_open(node_id, key)
            call = {
                "tool": name,
                "args": args,
                "result": result,
                "error": error,
                "duration": duration,
            }
            if open_node is not None:
                open_node.tool_calls.append(call)
            else:
                self._recorder.record_tool_call(
                    name,
                    args=args,
                    result=result,
                    error=error,
                    duration=duration,
                    parent_id=self._node_tips.get(node_id) if node_id else None,
                    node_id=node_id,
                )

        self._guard(_do)

    def node_end(
        self,
        node_id: str,
        *,
        state: dict[str, Any] | None = None,
        update: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        observation: str = "",
        score: float | None = None,
        kind: str = "think",
        cost: float = 0.0,
        duration: float | None = None,
        usage: dict[str, Any] | None = None,
        error: str | None = None,
        key: str | None = None,
        parents: list[str] | None = None,
        visit: int = 0,
    ) -> str | None:
        """Close a node execution and record it (plus buffered nested calls)."""
        holder: list[str | None] = [None]

        def _do() -> None:
            assert self._recorder is not None
            open_node = self._open.pop(key or node_id, None)
            state_before = open_node.state_before if open_node else self._last_state
            state_after = state
            if state_after is None and update is not None and state_before is not None:
                state_after = {**state_before, **update}
            elif state_after is None:
                state_after = update if update is not None else state_before
            elapsed = duration
            if elapsed is None:
                elapsed = time.monotonic() - open_node.started if open_node else 0.0
            edge = open_node.edge if open_node else self._pending_edges.pop(node_id, None)
            # Dedupe: opentine's v3 store rejects repeated parent_ids, and
            # a framework can legitimately report the same parent twice.
            # A new name, not a rebind: `parents` is a closure variable.
            unique_parents = list(dict.fromkeys(parents)) if parents else []
            parent_ids = unique_parents if len(unique_parents) > 1 else None
            parent_id = None
            if not parent_ids:
                if unique_parents:
                    parent_id = unique_parents[0]
                elif open_node is not None and open_node.parent_step:
                    parent_id = open_node.parent_step
                elif edge and edge.get("source") in self._node_tips:
                    parent_id = self._node_tips[edge["source"]]
            merged_outputs = dict(outputs or {})
            if error:
                merged_outputs["error"] = error
            step_id = self._recorder.record_node(
                node_id,
                kind=kind,
                visit=visit,
                edge=edge,
                state_before=state_before,
                state_after=state_after if isinstance(state_after, dict) else None,
                delta=(
                    state_delta(state_before, state_after)
                    if isinstance(state_after, dict)
                    else None
                ),
                observation=observation,
                score=score,
                outputs=merged_outputs,
                parent_id=parent_id,
                parent_ids=parent_ids,
                duration=elapsed,
                cost=cost,
                usage=usage,
            )
            if open_node:
                self._flush_calls(open_node, step_id)
            self._node_tips.pop(node_id, None)
            self._node_tips[node_id] = step_id  # re-insert to keep recency order
            if isinstance(state_after, dict):
                self._last_state = dict(state_after)
            holder[0] = step_id

        self._guard(_do)
        return holder[0]

    def checkpoint(
        self,
        checkpoint_id: str,
        *,
        state: dict[str, Any] | None = None,
        parent_checkpoint_id: str | None = None,
        source: str = "",
        step: int | None = None,
    ) -> None:
        """Record an externally-created checkpoint as a forkable marker step."""

        def _do() -> None:
            assert self._recorder is not None
            self._recorder.record_checkpoint(
                checkpoint_id,
                state=state,
                parent_checkpoint_id=parent_checkpoint_id,
                source=source,
                step=step,
            )

        self._guard(_do)

    def error(self, error: BaseException | str, *, node_id: str | None = None) -> None:
        def _do() -> None:
            assert self._recorder is not None
            parent = self._node_tips.get(node_id) if node_id else None
            self._recorder.record_error(error, parent_id=parent)

        self._guard(_do)

    def end(
        self,
        *,
        status: str = "completed",
        summary: str = "",
        final_state: dict[str, Any] | None = None,
    ) -> Path | None:
        """Finish and save the artifact; returns the saved path (or None)."""

        def _do() -> None:
            assert self._recorder is not None
            for open_key in list(self._open):
                # Nodes still open at run end: record them (with their
                # buffered model/tool calls) instead of dropping them.
                self._close_open(open_key, incomplete=True)
            if final_state is not None:
                self._last_state = dict(final_state)
            if not self._recorder.finished:
                if status == "failed":
                    self._recorder.record_error(summary or "failed")
                else:
                    self._recorder.record_done(summary or status)
            self._saved = self._recorder.save()

        self._guard(_do)
        return self._saved

    # --------------------------------------------------------------- helpers

    def _flush_calls(self, open_node: _OpenNode, step_id: str) -> None:
        assert self._recorder is not None
        for call in open_node.model_calls:
            self._recorder.record_model_call(
                model=call["model"],
                prompt=call["prompt"],
                response=call["response"],
                cost=call["cost"],
                duration=call["duration"],
                usage=call["usage"],
                parent_id=step_id,
                node_id=open_node.node_id,
            )
        for call in open_node.tool_calls:
            self._recorder.record_tool_call(
                call["tool"],
                args=call["args"],
                result=call["result"],
                error=call["error"],
                duration=call["duration"],
                parent_id=step_id,
                node_id=open_node.node_id,
            )

    def _close_open(self, open_key: str, *, incomplete: bool = False) -> str | None:
        open_node = self._open.pop(open_key, None)
        if open_node is None or self._recorder is None:
            return None
        step_id = self._recorder.record_node(
            open_node.node_id,
            edge=open_node.edge,
            state_before=open_node.state_before,
            observation="node execution never completed" if incomplete else "",
            outputs={"incomplete": True} if incomplete else None,
            parent_id=open_node.parent_step,
            duration=time.monotonic() - open_node.started,
        )
        self._flush_calls(open_node, step_id)
        self._node_tips.pop(open_node.node_id, None)
        self._node_tips[open_node.node_id] = step_id
        return step_id

    def _resolve_open(self, node_id: str | None, key: str | None) -> _OpenNode | None:
        if key and key in self._open:
            return self._open[key]
        if node_id:
            if node_id in self._open:
                return self._open[node_id]
            for open_node in self._open.values():
                if open_node.node_id == node_id:
                    return open_node
        if len(self._open) == 1:
            return next(iter(self._open.values()))
        return None

    def _guard(self, fn: Any) -> None:
        if self._recorder is None:
            return
        try:
            fn()
        except Exception:
            self.dropped_events += 1
            if self.strict:
                raise
            logger.debug("graphforge capture event dropped", exc_info=True)
