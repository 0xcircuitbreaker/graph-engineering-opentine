"""Framework-agnostic ingestion: turn existing telemetry into graph runs.

Two zero-glue paths for setups graphforge has no dedicated adapter for:

- :func:`ingest_otel_spans` — any framework instrumented with the
  OpenTelemetry GenAI semantic conventions (``invoke_agent`` /
  ``execute_tool`` / ``chat`` operations, ``gen_ai.*`` attributes) can dump
  its spans and get a forkable, diffable ``.tine`` artifact.
- :func:`ingest_jsonl` — the lowest common denominator: any system that can
  append JSON lines shaped ``{"type": "node_start" | "node_end" |
  "model_call" | "tool_call" | "edge" | "error" | "end", ...}`` gets full
  recording with no Python integration at all.

Both build on :class:`graphforge.GraphRunCapture`, so all its guarantees
(bounded payloads, sanitized metrics, DAG parents) apply.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .capture import GraphRunCapture
from .policy import GraphPolicy

_NODE_OPS = {"invoke_agent", "invoke_workflow", "create_agent"}
_MODEL_OPS = {"chat", "text_completion", "generate_content", "embeddings"}
_TOOL_OPS = {"execute_tool"}


def _attr(span: dict[str, Any], *names: str, default: Any = None) -> Any:
    attributes = span.get("attributes") or {}
    for name in names:
        if name in attributes:
            return attributes[name]
        if name in span:
            return span[name]
    return default


def _span_duration(span: dict[str, Any]) -> float:
    start, end = span.get("start_time"), span.get("end_time")
    if isinstance(start, int | float) and isinstance(end, int | float) and end >= start:
        # OTLP times are nanoseconds; small values are already seconds.
        scale = 1e9 if end > 1e12 else 1.0
        return (end - start) / scale
    return 0.0


def ingest_otel_spans(
    goal: str,
    spans: Iterable[dict[str, Any]],
    *,
    runs_dir: str | Path | None = None,
    framework: str = "otel-genai",
    policy: GraphPolicy | None = None,
) -> Path | None:
    """Record a list of OTel GenAI span dicts as one graph run.

    Node identity: ``invoke_agent``/``invoke_workflow`` spans become nodes
    (named by ``gen_ai.agent.name`` falling back to the span name); ``chat``
    and friends become model calls and ``execute_tool`` spans tool calls,
    attributed to their nearest ancestor node span via ``parent_span_id``.
    Spans are processed in start-time order.
    """
    span_list = sorted(
        [s for s in spans if isinstance(s, dict)],
        key=lambda s: s.get("start_time") or 0,
    )
    by_id = {s.get("span_id"): s for s in span_list if s.get("span_id")}

    def node_ancestor(span: dict[str, Any]) -> str | None:
        cursor = span.get("parent_span_id")
        for _ in range(64):
            if cursor is None or cursor not in by_id:
                return None
            parent = by_id[cursor]
            op = _attr(parent, "gen_ai.operation.name", default="")
            if op in _NODE_OPS:
                return str(parent.get("span_id"))
            cursor = parent.get("parent_span_id")
        return None

    capture = GraphRunCapture(goal, framework=framework, runs_dir=runs_dir, policy=policy)
    previous_node_span: str | None = None
    node_names: dict[str, str] = {}
    failed = False
    for span in span_list:
        op = str(_attr(span, "gen_ai.operation.name", default="") or "")
        span_id = str(span.get("span_id", ""))
        if op in _NODE_OPS:
            name = str(
                _attr(span, "gen_ai.agent.name", "gen_ai.agent.id", default="")
                or span.get("name", "node")
            )
            node_names[span_id] = name
            parent_node = node_ancestor(span)
            edge = None
            if parent_node and parent_node in node_names:
                edge = {"kind": "span", "source": node_names[parent_node]}
            elif previous_node_span and previous_node_span in node_names:
                edge = {"kind": "sequence", "source": node_names[previous_node_span]}
            capture.node_start(name, edge=edge, key=span_id)
            status = span.get("status") or {}
            error = None
            if isinstance(status, dict) and str(status.get("status_code", "")).upper() in (
                "ERROR",
                "STATUS_CODE_ERROR",
            ):
                error = str(status.get("description", "error"))
                failed = True
            capture.node_end(
                name,
                outputs={
                    "operation": op,
                    "conversation": _attr(span, "gen_ai.conversation.id", default=""),
                },
                error=error,
                duration=_span_duration(span),
                key=span_id,
            )
            previous_node_span = span_id
        elif op in _MODEL_OPS:
            usage = {}
            input_tokens = _attr(span, "gen_ai.usage.input_tokens")
            output_tokens = _attr(span, "gen_ai.usage.output_tokens")
            if input_tokens is not None:
                usage["input"] = input_tokens
            if output_tokens is not None:
                usage["output"] = output_tokens
            ancestor = node_ancestor(span)
            capture.model_call(
                model=str(
                    _attr(span, "gen_ai.response.model", "gen_ai.request.model", default="")
                ),
                prompt=_attr(span, "gen_ai.input.messages"),
                response=_attr(span, "gen_ai.output.messages"),
                usage=usage or None,
                duration=_span_duration(span),
                node_id=node_names.get(ancestor or ""),
                key=ancestor,
            )
        elif op in _TOOL_OPS:
            ancestor = node_ancestor(span)
            capture.tool_call(
                str(_attr(span, "gen_ai.tool.name", default="tool")),
                args=_attr(span, "gen_ai.tool.call.arguments"),
                result=_attr(span, "gen_ai.tool.call.result"),
                duration=_span_duration(span),
                node_id=node_names.get(ancestor or ""),
                key=ancestor,
            )
    return capture.end(
        status="failed" if failed else "completed",
        summary=f"ingested {len(span_list)} otel spans",
    )


def ingest_jsonl(
    goal: str,
    source: str | Path | Iterable[str],
    *,
    runs_dir: str | Path | None = None,
    framework: str = "jsonl",
    policy: GraphPolicy | None = None,
) -> Path | None:
    """Replay a JSONL event stream into a recorded graph run.

    ``source`` is a file path or an iterable of JSON lines. Event shapes::

        {"type": "begin",      "graph": {...}, "state": {...}}
        {"type": "node_start", "node": "plan", "state": {...}, "key": "..."}
        {"type": "node_end",   "node": "plan", "update": {...}, "state": {...},
         "outputs": {...}, "score": 0.9, "kind": "model", "cost": 0.0,
         "duration": 0.0, "usage": {...}, "error": "...", "key": "..."}
        {"type": "model_call", "model": "...", "prompt": ..., "response": ...,
         "cost": 0.0, "usage": {...}, "node": "plan"}
        {"type": "tool_call",  "tool": "search", "args": ..., "result": ...}
        {"type": "edge",       "source": "plan", "target": "act", "label": ""}
        {"type": "checkpoint", "id": "c1", "state": {...}, "source": "..."}
        {"type": "error",      "error": "...", "node": "plan"}
        {"type": "end",        "status": "completed", "summary": "..."}

    Unknown types and malformed lines are skipped (never-throw discipline);
    a final ``end`` event is implied if the stream lacks one.
    """
    if isinstance(source, str | Path):
        lines: Iterable[str] = Path(source).expanduser().read_text().splitlines()
    else:
        lines = source
    capture = GraphRunCapture(goal, framework=framework, runs_dir=runs_dir, policy=policy)
    ended = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "begin":
            capture.begin(graph=event.get("graph"), state=event.get("state"))
        elif kind == "node_start":
            capture.node_start(
                str(event.get("node", "node")),
                state=event.get("state"),
                edge=event.get("edge"),
                key=event.get("key"),
            )
        elif kind == "node_end":
            capture.node_end(
                str(event.get("node", "node")),
                state=event.get("state"),
                update=event.get("update"),
                outputs=event.get("outputs"),
                observation=str(event.get("observation", "")),
                score=event.get("score"),
                kind=str(event.get("kind", "think")),
                cost=event.get("cost", 0.0),
                duration=event.get("duration"),
                usage=event.get("usage"),
                error=event.get("error"),
                key=event.get("key"),
                parents=event.get("parents"),
            )
        elif kind == "model_call":
            capture.model_call(
                model=str(event.get("model", "")),
                prompt=event.get("prompt"),
                response=event.get("response"),
                cost=event.get("cost", 0.0),
                duration=event.get("duration", 0.0),
                usage=event.get("usage"),
                node_id=event.get("node"),
                key=event.get("key"),
            )
        elif kind == "tool_call":
            capture.tool_call(
                str(event.get("tool", "tool")),
                args=event.get("args"),
                result=event.get("result"),
                error=event.get("error"),
                duration=event.get("duration", 0.0),
                node_id=event.get("node"),
                key=event.get("key"),
            )
        elif kind == "edge":
            capture.edge(
                str(event.get("source", "")),
                str(event.get("target", "*") or "*"),
                label=str(event.get("label", "")),
            )
        elif kind == "checkpoint":
            capture.checkpoint(
                str(event.get("id", "checkpoint")),
                state=event.get("state"),
                parent_checkpoint_id=event.get("parent"),
                source=str(event.get("source", "")),
                step=event.get("step"),
            )
        elif kind == "error":
            capture.error(str(event.get("error", "error")), node_id=event.get("node"))
        elif kind == "end":
            ended = True
            return capture.end(
                status=str(event.get("status", "completed")),
                summary=str(event.get("summary", "")),
                final_state=event.get("state"),
            )
    if not ended:
        return capture.end(summary="jsonl stream ended without end event")
    return None  # pragma: no cover - unreachable
