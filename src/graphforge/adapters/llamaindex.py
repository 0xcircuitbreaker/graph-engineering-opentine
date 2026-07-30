"""LlamaIndex Workflows adapter: record checkpointed workflow runs.

LlamaIndex Workflows are event-driven: there is no explicit edge list, the
effective edge is the (producing step, event type, consuming step) triple.
``WorkflowCheckpointer`` materializes a ``Checkpoint(last_completed_step,
input_event, output_event, ctx_state)`` after every step — and stores them
in memory only, so persisting them is the recorder's job.

``record_checkpoints`` converts a checkpoint sequence into a recorded run:
one node per checkpoint, edges reconstructed by chaining each checkpoint's
``output_event`` type to the next checkpoint whose ``input_event`` matches,
and full ``ctx_state`` snapshots as forkable checkpoint markers.

Duck-typed on attribute access, so it works with real Checkpoint objects or
anything checkpoint-shaped — no llama-index import required.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..capture import GraphRunCapture
from ..state import safe_json

logger = logging.getLogger(__name__)


def _event_type(event: Any) -> str:
    if event is None:
        return ""
    return type(event).__name__ if not isinstance(event, str) else event


def _event_payload(event: Any) -> Any:
    dump = getattr(event, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # noqa: BLE001
            pass
    return safe_json(event)


def _ctx_state_dict(ctx_state: Any) -> dict[str, Any] | None:
    """Pull the user state store out of a serialized Context, defensively."""
    if not isinstance(ctx_state, dict):
        return None
    for key in ("state", "globals", "store"):
        inner = ctx_state.get(key)
        if isinstance(inner, dict):
            return inner
    return ctx_state


def record_checkpoints(
    capture: GraphRunCapture,
    checkpoints: Any,
    *,
    run_id: str | None = None,
    finish: bool = True,
) -> Path | None:
    """Record a WorkflowCheckpointer's checkpoints as a graph run.

    ``checkpoints`` is either a sequence of Checkpoint-shaped objects or a
    WorkflowCheckpointer (its ``.checkpoints`` dict is used; ``run_id``
    selects one run, defaulting to the only/last one). Returns the saved
    artifact path when ``finish`` is True.
    """
    try:
        store = getattr(checkpoints, "checkpoints", None)
        if isinstance(store, dict) and store:
            if run_id is None:
                run_id = next(reversed(store))
            checkpoints = store.get(run_id, [])
        sequence = list(checkpoints or [])
    except Exception:  # noqa: BLE001
        logger.debug("graphforge llamaindex checkpoint extraction failed", exc_info=True)
        sequence = []

    previous_step: str | None = None
    previous_output: str = ""
    for index, ckpt in enumerate(sequence):
        try:
            step_name = str(getattr(ckpt, "last_completed_step", f"step-{index}"))
            input_type = _event_type(getattr(ckpt, "input_event", None))
            output_type = _event_type(getattr(ckpt, "output_event", None))
            state = _ctx_state_dict(getattr(ckpt, "ctx_state", None))
            if previous_step is not None and input_type and input_type == previous_output:
                capture.edge(previous_step, step_name, label=input_type)
            capture.node_start(step_name, state=state, key=f"ckpt-{index}")
            capture.node_end(
                step_name,
                state=state,
                outputs={
                    "input_event": input_type,
                    "output_event": output_type,
                    "output_payload": _event_payload(getattr(ckpt, "output_event", None)),
                },
                key=f"ckpt-{index}",
            )
            if isinstance(state, dict):
                checkpoint_id = str(getattr(ckpt, "id", "") or f"{step_name}-{index}")
                capture.checkpoint(checkpoint_id, state=state, source=step_name, step=index)
            previous_step = step_name
            previous_output = output_type
        except Exception:  # noqa: BLE001
            logger.debug("graphforge llamaindex checkpoint %s failed", index, exc_info=True)
    if finish:
        return capture.end(summary=f"{len(sequence)} workflow checkpoints recorded")
    return None
