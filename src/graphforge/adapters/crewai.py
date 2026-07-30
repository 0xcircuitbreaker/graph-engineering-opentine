"""CrewAI Flows adapter: record flow executions into opentine.

CrewAI exposes a process-wide event bus; a single listener observes every
flow with zero changes to user flow code. Because the bus is a singleton
(concurrent flows interleave), this adapter keeps ONE capture per flow,
keyed by the flow's state id, and correlates every event to its flow.

The event-handling methods are plain functions over event-shaped objects
(``method_name`` / ``flow_name`` / ``state`` / ``result`` attributes read
defensively), so they are fully testable without crewai installed;
``setup_listeners`` wires them onto the real bus when crewai is present.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..capture import GraphRunCapture
from ..state import safe_json

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only with crewai installed
    from crewai.events import BaseEventListener as _ListenerBase

    HAS_CREWAI = True
except ImportError:  # pragma: no cover
    _ListenerBase = object
    HAS_CREWAI = False


def _flow_key(event: Any) -> str:
    """Correlate an event to its flow (the bus is a process singleton)."""
    state = getattr(event, "state", None)
    if isinstance(state, dict) and state.get("id"):
        return str(state["id"])
    state_id = getattr(state, "id", None)
    if state_id:
        return str(state_id)
    flow = getattr(event, "flow", None)
    flow_state = getattr(flow, "state", None)
    for candidate in (getattr(flow_state, "id", None), getattr(event, "flow_name", None)):
        if candidate:
            return str(candidate)
    return "default"


def _state_dict(event: Any) -> dict[str, Any] | None:
    state = getattr(event, "state", None)
    if isinstance(state, dict):
        return state
    dump = getattr(state, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # noqa: BLE001
            return None
    return None


class TineFlowListener(_ListenerBase):  # type: ignore[misc,valid-type]
    """Records every CrewAI Flow on the event bus into per-flow captures."""

    def __init__(
        self,
        capture_factory: Callable[[str], GraphRunCapture] | None = None,
        *,
        runs_dir: str | Path | None = None,
    ) -> None:
        self._factory = capture_factory or (
            lambda flow_name: GraphRunCapture(
                flow_name or "crewai flow", framework="crewai", runs_dir=runs_dir
            )
        )
        self.captures: dict[str, GraphRunCapture] = {}
        self.artifacts: list[Path] = []
        if HAS_CREWAI:  # pragma: no cover - BaseEventListener registers on init
            try:
                super().__init__()
            except Exception:  # noqa: BLE001
                logger.debug("graphforge crewai listener base init failed", exc_info=True)

    def _capture_for(self, event: Any) -> GraphRunCapture:
        key = _flow_key(event)
        if key not in self.captures:
            flow_name = str(getattr(event, "flow_name", "") or key)
            self.captures[key] = self._factory(flow_name)
        return self.captures[key]

    # -- flow lifecycle -----------------------------------------------------

    def on_flow_started(self, event: Any) -> None:
        try:
            capture = self._capture_for(event)
            capture.begin(state=_state_dict(event))
        except Exception:  # noqa: BLE001
            logger.debug("graphforge crewai on_flow_started failed", exc_info=True)

    def on_flow_finished(self, event: Any) -> None:
        try:
            key = _flow_key(event)
            capture = self.captures.pop(key, None)
            if capture is None:
                return
            artifact = capture.end(
                summary=safe_json(getattr(event, "result", "")) or "flow finished",
                final_state=_state_dict(event),
            )
            if artifact is not None:
                self.artifacts.append(artifact)
        except Exception:  # noqa: BLE001
            logger.debug("graphforge crewai on_flow_finished failed", exc_info=True)

    # -- method (node) lifecycle --------------------------------------------

    def on_method_started(self, event: Any) -> None:
        try:
            capture = self._capture_for(event)
            method = str(getattr(event, "method_name", "method"))
            capture.node_start(method, state=_state_dict(event), key=self._node_key(event))
        except Exception:  # noqa: BLE001
            logger.debug("graphforge crewai on_method_started failed", exc_info=True)

    def on_method_finished(self, event: Any) -> None:
        try:
            capture = self._capture_for(event)
            method = str(getattr(event, "method_name", "method"))
            result = getattr(event, "result", None)
            outputs = {"result": safe_json(result)} if result is not None else None
            # A @router method's return value is the label of the edge it
            # chose; declare it so the next started method records the edge.
            if isinstance(result, str) and result:
                capture.edge(method, "*", label=result)
            capture.node_end(
                method,
                state=_state_dict(event),
                outputs=outputs,
                key=self._node_key(event),
            )
        except Exception:  # noqa: BLE001
            logger.debug("graphforge crewai on_method_finished failed", exc_info=True)

    def on_method_failed(self, event: Any) -> None:
        try:
            capture = self._capture_for(event)
            method = str(getattr(event, "method_name", "method"))
            capture.node_end(
                method,
                error=str(getattr(event, "error", "failed")),
                key=self._node_key(event),
            )
        except Exception:  # noqa: BLE001
            logger.debug("graphforge crewai on_method_failed failed", exc_info=True)

    # -- model calls ---------------------------------------------------------

    def on_llm_call_completed(self, event: Any) -> None:
        try:
            capture = self._capture_for(event)
            usage = None
            raw_usage = getattr(event, "usage", None) or getattr(event, "token_usage", None)
            if isinstance(raw_usage, dict):
                usage = {
                    "input": raw_usage.get("prompt_tokens", raw_usage.get("input", 0)) or 0,
                    "output": raw_usage.get("completion_tokens", raw_usage.get("output", 0))
                    or 0,
                }
            capture.model_call(
                model=str(getattr(event, "model", "") or ""),
                prompt=safe_json(getattr(event, "messages", None)),
                response=safe_json(getattr(event, "response", None)),
                usage=usage,
                cost=float(getattr(event, "cost", 0.0) or 0.0),
            )
        except Exception:  # noqa: BLE001
            logger.debug("graphforge crewai on_llm_call_completed failed", exc_info=True)

    def _node_key(self, event: Any) -> str:
        return f"{_flow_key(event)}:{getattr(event, 'method_name', '')}"

    # -- bus wiring (only with crewai installed) -----------------------------

    def setup_listeners(self, crewai_event_bus: Any) -> None:  # pragma: no cover
        try:
            from crewai.events import (
                FlowFinishedEvent,
                FlowStartedEvent,
                LLMCallCompletedEvent,
                MethodExecutionFailedEvent,
                MethodExecutionFinishedEvent,
                MethodExecutionStartedEvent,
            )
        except ImportError:
            logger.debug("graphforge crewai event classes unavailable")
            return

        @crewai_event_bus.on(FlowStartedEvent)
        def _started(source: Any, event: Any) -> None:
            self.on_flow_started(event)

        @crewai_event_bus.on(FlowFinishedEvent)
        def _finished(source: Any, event: Any) -> None:
            self.on_flow_finished(event)

        @crewai_event_bus.on(MethodExecutionStartedEvent)
        def _method_started(source: Any, event: Any) -> None:
            self.on_method_started(event)

        @crewai_event_bus.on(MethodExecutionFinishedEvent)
        def _method_finished(source: Any, event: Any) -> None:
            self.on_method_finished(event)

        @crewai_event_bus.on(MethodExecutionFailedEvent)
        def _method_failed(source: Any, event: Any) -> None:
            self.on_method_failed(event)

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def _llm_done(source: Any, event: Any) -> None:
            self.on_llm_call_completed(event)
