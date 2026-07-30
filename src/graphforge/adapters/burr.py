"""Burr adapter: record Apache Burr applications into opentine.

One hook class, registered with ``ApplicationBuilder().with_hooks(...)`` —
zero changes to user actions. Burr's lifecycle surface maps directly:

- ``post_application_create`` -> run topology (actions + transitions) and
  fork lineage (``parent_pointer`` from ``initialize_from(fork_from_...)``)
- ``pre_run_step`` / ``post_run_step`` -> one recorded node per executed
  action, with state snapshots and the edge from the previous action

Import-safe without burr installed (the class degrades to a plain object
base), so the module never forces the dependency.
"""

from __future__ import annotations

import logging
from typing import Any

from ..capture import GraphRunCapture
from ..state import safe_json

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only with burr installed
    from burr.lifecycle import (
        PostApplicationCreateHook,
        PostRunStepHook,
        PreRunStepHook,
    )

    _BASES: tuple[type, ...] = (PostApplicationCreateHook, PreRunStepHook, PostRunStepHook)
    HAS_BURR = True
except ImportError:  # pragma: no cover
    _BASES = (object,)
    HAS_BURR = False


def _state_dict(state: Any) -> dict[str, Any] | None:
    """Serialize a Burr State (or anything state-shaped) defensively."""
    if state is None:
        return None
    for attr in ("serialize", "get_all"):
        method = getattr(state, attr, None)
        if callable(method):
            try:
                value = method()
                if isinstance(value, dict):
                    return value
            except Exception:  # noqa: BLE001
                continue
    if isinstance(state, dict):
        return state
    try:
        return dict(state)
    except Exception:  # noqa: BLE001
        return {"state": safe_json(state)}


def _action_name(action: Any) -> str:
    name = getattr(action, "name", None)
    return str(name) if name else str(action)


def _serialize_graph(application_graph: Any) -> dict[str, Any] | None:
    if application_graph is None:
        return None
    try:
        actions = [
            {"id": _action_name(a)} for a in getattr(application_graph, "actions", []) or []
        ]
        transitions = []
        for t in getattr(application_graph, "transitions", []) or []:
            transitions.append(
                {
                    "source": _action_name(getattr(t, "from_", getattr(t, "from_state", ""))),
                    "target": _action_name(getattr(t, "to", "")),
                    "label": str(getattr(getattr(t, "condition", None), "name", "") or ""),
                }
            )
        return {"framework": "burr", "nodes": actions, "edges": transitions}
    except Exception:  # noqa: BLE001
        logger.debug("graphforge burr graph serialization failed", exc_info=True)
        return None


class TineBurrHook(*_BASES):  # type: ignore[misc]
    """Burr lifecycle hook that records every step into a GraphRunCapture."""

    def __init__(self, capture: GraphRunCapture) -> None:
        self.capture = capture
        self._previous_action: str | None = None

    # -- application lifecycle ---------------------------------------------

    def post_application_create(
        self,
        *,
        app_id: Any = None,
        partition_key: Any = None,
        state: Any = None,
        application_graph: Any = None,
        parent_pointer: Any = None,
        **kwargs: Any,
    ) -> None:
        try:
            graph = _serialize_graph(application_graph) or {"framework": "burr"}
            graph["app_id"] = str(app_id) if app_id else ""
            if partition_key:
                graph["partition_key"] = str(partition_key)
            if parent_pointer is not None:
                graph["forked_from"] = {
                    "app_id": str(getattr(parent_pointer, "app_id", "")),
                    "sequence_id": getattr(parent_pointer, "sequence_id", None),
                }
            self.capture.begin(graph=graph, state=_state_dict(state))
        except Exception:  # noqa: BLE001
            logger.debug("graphforge burr post_application_create failed", exc_info=True)

    # -- step lifecycle -----------------------------------------------------

    def pre_run_step(
        self,
        *,
        sequence_id: Any = None,
        state: Any = None,
        action: Any = None,
        inputs: Any = None,
        **kwargs: Any,
    ) -> None:
        try:
            name = _action_name(action)
            if self._previous_action:
                self.capture.edge(self._previous_action, name)
            self.capture.node_start(
                name, state=_state_dict(state), key=f"seq-{sequence_id}"
            )
        except Exception:  # noqa: BLE001
            logger.debug("graphforge burr pre_run_step failed", exc_info=True)

    def post_run_step(
        self,
        *,
        sequence_id: Any = None,
        state: Any = None,
        action: Any = None,
        result: Any = None,
        exception: Any = None,
        **kwargs: Any,
    ) -> None:
        try:
            name = _action_name(action)
            self.capture.node_end(
                name,
                state=_state_dict(state),
                outputs={"result": safe_json(result)} if result is not None else None,
                error=str(exception) if exception is not None else None,
                key=f"seq-{sequence_id}",
            )
            self._previous_action = name
        except Exception:  # noqa: BLE001
            logger.debug("graphforge burr post_run_step failed", exc_info=True)
