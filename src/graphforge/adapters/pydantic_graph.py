"""pydantic-graph (PydanticAI V2) adapter: record graph.iter() executions.

pydantic-graph V2 removed native persistence; its ``graph.iter()`` API
yields scheduled-task batches (``Sequence[GraphTask]``) as execution
progresses, ending with an ``EndMarker``. ``record_graph_run`` drives a
graph through ``iter()`` and records every scheduled task with its
``fork_stack`` lineage (the exact parallel-branch ancestry pydantic-graph
maintains), plus per-event state snapshots.

Duck-typed: anything exposing the same async iteration contract records the
same way — no pydantic-graph import required.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ..capture import GraphRunCapture
from ..state import safe_json

logger = logging.getLogger(__name__)


def _dump_state(state: Any) -> dict[str, Any] | None:
    if state is None:
        return None
    if isinstance(state, dict):
        return state
    if is_dataclass(state) and not isinstance(state, type):
        try:
            return asdict(state)
        except Exception:  # noqa: BLE001
            pass
    dump = getattr(state, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # noqa: BLE001
            pass
    attrs = getattr(state, "__dict__", None)
    if isinstance(attrs, dict):
        return dict(attrs)
    return {"state": safe_json(state)}


def _fork_stack(task: Any) -> list[dict[str, Any]]:
    stack = []
    for item in getattr(task, "fork_stack", ()) or ():
        stack.append(
            {
                "fork_id": str(getattr(item, "fork_id", "")),
                "node_run_id": str(getattr(item, "node_run_id", "")),
                "thread_index": getattr(item, "thread_index", None),
            }
        )
    return stack


async def record_graph_run(
    capture: GraphRunCapture,
    graph: Any,
    *,
    state: Any = None,
    deps: Any = None,
    inputs: Any = None,
    finish: bool = True,
) -> tuple[Any, Path | None]:
    """Drive ``graph.iter()`` and record every scheduled task.

    Returns ``(output, artifact_path)``. The graph's nodes run exactly as
    they would without recording; only observation is added.
    """
    output: Any = None
    artifact: Path | None = None
    try:
        renderer = getattr(graph, "render", None)
        if callable(renderer):
            try:
                capture.begin(graph={"framework": "pydantic-graph", "mermaid": str(renderer())})
            except Exception:  # noqa: BLE001
                capture.begin(graph={"framework": "pydantic-graph"})
        else:
            capture.begin(graph={"framework": "pydantic-graph"})

        async with graph.iter(state=state, deps=deps, inputs=inputs) as run:
            previous_nodes: list[str] = []
            for task in list(getattr(run, "next_task", None) or []):
                _record_task(capture, run, task, previous_nodes)
            async for event in run:
                if _is_end(event):
                    output = getattr(event, "value", None)
                    break
                if _is_error(event):
                    capture.error(str(getattr(event, "error", "graph error")))
                    continue
                batch = list(event) if _is_sequence(event) else []
                completed = previous_nodes
                previous_nodes = []
                for task in batch:
                    _record_task(capture, run, task, completed)
                    previous_nodes.append(str(getattr(task, "node_id", "")))
            if output is None:
                output = getattr(run, "output", None)
    except Exception as exc:  # noqa: BLE001 - recording must not swallow the run error silently
        capture.error(exc)
        if finish:
            artifact = capture.end(status="failed", summary=str(exc))
        raise
    if finish:
        artifact = capture.end(
            summary="graph run complete", final_state=_dump_state(state)
        )
    return output, artifact


def _record_task(
    capture: GraphRunCapture, run: Any, task: Any, completed: list[str]
) -> None:
    try:
        node_id = str(getattr(task, "node_id", "task"))
        task_id = str(getattr(task, "task_id", ""))
        edge = None
        if completed:
            edge = {"kind": "scheduled", "source": completed[-1]}
        capture.node_start(node_id, state=_dump_state(getattr(run, "state", None)), edge=edge)
        capture.node_end(
            node_id,
            state=_dump_state(getattr(run, "state", None)),
            outputs={
                "task_id": task_id,
                "inputs": safe_json(getattr(task, "inputs", None)),
                "fork_stack": _fork_stack(task),
            },
        )
    except Exception:  # noqa: BLE001
        logger.debug("graphforge pydantic-graph task record failed", exc_info=True)


def _is_end(event: Any) -> bool:
    return type(event).__name__ == "EndMarker" or hasattr(event, "value") and not _is_sequence(
        event
    )


def _is_error(event: Any) -> bool:
    return type(event).__name__ == "ErrorMarker"


def _is_sequence(event: Any) -> bool:
    return isinstance(event, list | tuple)
