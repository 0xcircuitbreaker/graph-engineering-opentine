"""Framework-agnostic ingestion: turn existing telemetry into graph runs.

Four zero-glue paths for setups graphforge has no dedicated adapter for:

- :func:`ingest_otel_spans` — any framework instrumented with the
  OpenTelemetry GenAI semantic conventions (``invoke_agent`` /
  ``execute_tool`` / ``chat`` operations, ``gen_ai.*`` attributes) can dump
  its spans — an extracted span list *or* a raw OTLP/JSON export — and get a
  forkable, diffable ``.tine`` artifact.
- :func:`ingest_trace_events` — opentine's own normalized ``TraceEvent``
  stream (whatever produced it: native runs, JSONL, OTLP, framework
  callbacks) replayed as a graph, including multi-parent fan-in.
- :func:`ingest_framework` — raw LangChain / LlamaIndex / AutoGen / CrewAI /
  OpenAI-Agents callback records, via opentine's own importers.
- :func:`ingest_jsonl` — the lowest common denominator: any system that can
  append JSON lines shaped ``{"type": "node_start" | "node_end" |
  "model_call" | "tool_call" | "edge" | "error" | "end", ...}`` gets full
  recording with no Python integration at all.

All build on :class:`graphforge.GraphRunCapture`, so all its guarantees
(bounded payloads, sanitized metrics, DAG parents) apply.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from .capture import GraphRunCapture
from .policy import GraphPolicy
from .state import sane_metric, sane_usage

# opentine ships battle-tested OTLP traversal/decoding, but only from 0.3.0
# and only in PRIVATE modules (leading underscore, absent from every
# __all__), so a patch release may move or rename them. Import defensively
# and fall back to the pure-Python implementations below.
#
# What is guaranteed across both paths (and across opentine 0.2.0, where
# none of these modules exist): the set of spans found in a document, their
# ids/timing/status, the per-attribute decoding rules mirrored below, and
# graphforge's own input-record cap. What is NOT guaranteed is byte
# equality on pathological AnyValue records — so a value neither side can
# read is dropped per attribute and reported, never per span.
try:  # pragma: no cover - exercised by whichever opentine is installed
    from opentine.trace._import_helpers import (
        first as _opentine_first,
    )
    from opentine.trace._import_helpers import (
        integer as _opentine_integer,
    )
    from opentine.trace._import_helpers import (
        otel_spans as _opentine_otel_spans,
    )
except ImportError:  # pragma: no cover - opentine < 0.3.0 or private move
    _opentine_first = None
    _opentine_integer = None
    _opentine_otel_spans = None

try:  # pragma: no cover - exercised by whichever opentine is installed
    from opentine.trace._otel_values import any_value as _opentine_any_value
    from opentine.trace._otel_values import attributes as _opentine_attributes
except ImportError:  # pragma: no cover - opentine < 0.3.0 or private move
    _opentine_any_value = None
    _opentine_attributes = None

try:  # pragma: no cover - public API, but 0.3.0-only
    from opentine.trace import TraceEvent as _TraceEvent
except ImportError:  # pragma: no cover - opentine 0.2.0 has no trace schema
    _TraceEvent = None

try:  # pragma: no cover - public API, but 0.3.0-only
    from opentine.trace import framework_events as _opentine_framework_events
except ImportError:  # pragma: no cover - opentine 0.2.0 has no importers
    _opentine_framework_events = None

_NODE_OPS = {"invoke_agent", "invoke_workflow", "create_agent"}
_MODEL_OPS = {"chat", "text_completion", "generate_content", "embeddings"}
_TOOL_OPS = {"execute_tool"}

#: ``opentine.trace.schema.TRACE_KINDS`` in full. The four below describe a
#: unit of work with its own identity (an agent turn, a human decision, a
#: policy verdict), which is what a graph node is; "model"/"tool" are leaf
#: calls and "error" is a failure marker.
_TRACE_KINDS = {"model", "tool", "human", "policy", "approval", "subagent", "error"}
_NODE_KINDS = {"subagent", "human", "policy", "approval"}

#: Frameworks understood by ``opentine.trace.framework_events`` (its private
#: ``_FRAMEWORK_FIELDS`` table); each maps (id, parent, actor) record keys.
FRAMEWORK_DIALECTS = ("langchain", "llamaindex", "autogen", "crewai", "openai-agents")

_ERROR_CODES = {"ERROR", "STATUS_CODE_ERROR", "2"}
_STATUS_CODES = {0: "STATUS_CODE_UNSET", 1: "STATUS_CODE_OK", 2: "STATUS_CODE_ERROR"}
_USAGE_PREFIX = "gen_ai.usage."
_MAX_SPANS = 100_000
_MAX_ANY_VALUE_DEPTH = 100
_MAX_ANCESTOR_HOPS = 64
_MAX_CORRELATED_SPANS = 32
#: Protobuf-JSON stringifies ints past this; so does opentine's decoder.
_MAX_SAFE_INTEGER = (1 << 53) - 1
#: JSON integers are unbounded but timestamps are not: past 2**64 (~584
#: years of nanoseconds) a value is garbage, and converting it to a float
#: raises OverflowError, which would abort the whole import.
_MAX_TIME_VALUE = 2**64

#: OTLP spellings that only :func:`normalize_otlp` translates. A batch
#: carrying any of them needs normalizing even when its attributes were
#: already decoded, or every id, parent link and timestamp is silently lost.
_CAMEL_OTLP_KEYS = frozenset(
    {
        "spanId",
        "parentSpanId",
        "traceId",
        "startTimeUnixNano",
        "endTimeUnixNano",
        "startTime",
        "endTime",
    }
)

#: Raw OTLP spellings replaced by their snake_case form in :func:`normalize_otlp`;
#: every other key on the source span is copied through untouched.
_OTLP_KEYS = frozenset(
    {
        "attributes",
        "spanId",
        "span_id",
        "traceId",
        "trace_id",
        "parentSpanId",
        "parent_span_id",
        "startTimeUnixNano",
        "start_time_unix_nano",
        "startTime",
        "start_time",
        "endTimeUnixNano",
        "end_time_unix_nano",
        "endTime",
        "end_time",
        "status",
        "links",
        "name",
    }
)


# --------------------------------------------------------------- opentine glue


def _first(source: dict[str, Any], *names: str, default: Any = None) -> Any:
    """First present, non-empty value among ``names`` (opentine's semantics)."""
    if _opentine_first is not None:
        return _opentine_first(source, *names, default=default)
    for name in names:
        value = source.get(name)
        if value not in (None, ""):
            return value
    return default


def _integer(value: Any) -> int:
    """Coerce to ``int``, never raising (opentine's semantics)."""
    if _opentine_integer is not None:
        return _opentine_integer(value)
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _otel_spans(source: Iterable[dict[str, Any]] | dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield spans from a span list *or* a complete OTLP/JSON document.

    The record cap is counted here, on yielded spans, rather than inside
    either traversal: opentine 0.3.0 applies its own limit per list while
    the fallback walks the whole document, so counting at the seam is what
    makes the same export succeed or fail identically on both versions.
    """
    produced = (
        _opentine_otel_spans(source)
        if _opentine_otel_spans is not None
        else _fallback_otel_spans(source)
    )
    for count, span in enumerate(produced, start=1):
        if count > _MAX_SPANS:
            raise ValueError("trace import exceeds maximum input-record count")
        yield span


def _fallback_otel_spans(
    source: Iterable[dict[str, Any]] | dict[str, Any],
) -> Iterator[dict[str, Any]]:
    def dictionaries(values: Any) -> Iterator[dict[str, Any]]:
        for value in values if isinstance(values, list | tuple) else ():
            if isinstance(value, dict):
                yield value

    if isinstance(source, dict):
        resources = _first(source, "resourceSpans", "resource_spans")
        if isinstance(resources, list):
            for resource in dictionaries(resources):
                scopes = _first(resource, "scopeSpans", "scope_spans", default=[])
                for scope in dictionaries(scopes):
                    yield from dictionaries(scope.get("spans") or [])
            return
        spans = source.get("spans")
        if isinstance(spans, list):
            yield from dictionaries(spans)
        elif any(name in source for name in ("traceId", "trace_id", "spanId", "span_id")):
            yield source
        return
    for item in source:
        if isinstance(item, dict):
            yield from _fallback_otel_spans(item)


def _otel_attributes(span: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Decode a span's attributes; returns the decoded map and dropped keys.

    Degradation is per attribute, never per span: opentine's decoder raises
    on a malformed AnyValue (a container in a scalar slot, a cyclic value, a
    container used as a key), and losing the whole map to one sloppy
    attribute would delete the span's ``gen_ai.operation.name`` — and with
    it the node, the model call, or the tool call it describes.
    """
    if _opentine_attributes is not None:
        try:
            return _opentine_attributes(span), []
        except (RecursionError, ValueError):
            pass  # fall through and salvage attribute by attribute
    raw = span.get("attributes") or {}
    if isinstance(raw, list):
        pairs: list[tuple[Any, Any]] = [
            (item.get("key"), item.get("value"))
            for item in raw
            if isinstance(item, dict) and item.get("key") is not None
        ]
    elif isinstance(raw, dict):
        pairs = list(raw.items())
    else:
        return {}, []
    decoded: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in pairs:
        try:
            decoded[_any_key(key)] = _decode_any_value(value)
        except (RecursionError, ValueError):
            dropped.append(_any_key(key) if not isinstance(key, dict | list | tuple) else "?")
    return decoded, dropped


def _decode_any_value(value: Any) -> Any:
    """One attribute value, decoded by opentine's rules where available."""
    if _opentine_any_value is not None:
        return _opentine_any_value(value)
    return _any_value(value, 0, set())


def _any_value(value: Any, depth: int, active: set[int]) -> Any:
    """Unwrap an OTLP ``AnyValue`` with explicit cycle and depth bounds."""
    if depth > _MAX_ANY_VALUE_DEPTH:
        raise ValueError("OTLP AnyValue exceeds maximum nesting depth")
    if not isinstance(value, dict | list | tuple):
        return value
    identity = id(value)
    if identity in active:
        raise ValueError("OTLP AnyValue contains a cyclic container")
    active.add(identity)
    try:
        if isinstance(value, list | tuple):
            return [_any_value(item, depth + 1, active) for item in value]
        return _any_value_dict(value, depth, active)
    finally:
        active.discard(identity)


def _any_value_dict(value: dict[str, Any], depth: int, active: set[int]) -> Any:
    # Mirrors opentine 0.3.0's _otel_values rules — including *raising* on a
    # container in a scalar slot and stringifying ints past 2**53-1 — so an
    # attribute is decoded, or rejected, the same way on either version.
    for name in ("stringValue", "string_value", "bytesValue", "bytes_value"):
        if name in value:
            return str(_any_scalar(value[name]))
    for name in ("intValue", "int_value"):
        if name in value:
            raw = _any_scalar(value[name])
            number = _integer(raw)
            return number if abs(number) <= _MAX_SAFE_INTEGER else str(raw)
    for name in ("doubleValue", "double_value"):
        if name in value:
            return _any_scalar(value[name])
    for name in ("boolValue", "bool_value"):
        if name in value:
            raw = _any_scalar(value[name])
            return raw if isinstance(raw, bool) else str(raw).casefold() == "true"
    array = value.get("arrayValue", value.get("array_value"))
    if isinstance(array, dict):
        values = array.get("values")
        return (
            [_any_value(item, depth + 1, active) for item in values]
            if isinstance(values, list)
            else []
        )
    pairs = value.get("kvlistValue", value.get("kvlist_value"))
    if isinstance(pairs, dict) and isinstance(pairs.get("values"), list):
        return {
            _any_key(item["key"]): _any_value(item.get("value"), depth + 1, active)
            for item in pairs["values"]
            if isinstance(item, dict) and item.get("key") is not None
        }
    return {_any_key(key): _any_value(item, depth + 1, active) for key, item in value.items()}


def _any_scalar(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        raise ValueError("OTLP AnyValue scalar contains a container")
    return value


def _any_key(value: Any) -> str:
    if isinstance(value, dict | list | tuple):
        raise ValueError("OTLP AnyValue key contains a container")
    return str(value)


# ------------------------------------------------------------- span normalizing


def normalize_otlp(source: Iterable[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten an OTLP/JSON export into flat spans this module can read.

    Accepts a complete document (``resourceSpans[].scopeSpans[].spans[]``), a
    ``{"spans": [...]}`` envelope, a single span, or an already-flat iterable,
    and returns spans with a plain-dict ``attributes`` (AnyValue records
    decoded), snake_case ``span_id`` / ``trace_id`` / ``parent_span_id`` /
    ``start_time`` / ``end_time``, a ``{"status_code", "description"}`` status
    and ``links`` reduced to span references. Unrecognized keys are copied
    through, so custom fields survive the round trip. Attributes that cannot
    be decoded are dropped individually and listed under
    ``unreadable_attributes``; the span itself is always kept.
    """
    spans: list[dict[str, Any]] = []
    for raw in _otel_spans(source):
        attributes, unreadable = _otel_attributes(raw)
        span = {key: value for key, value in raw.items() if key not in _OTLP_KEYS}
        parent = _first(raw, "parentSpanId", "parent_span_id")
        start = _time_value(
            _first(raw, "startTimeUnixNano", "start_time_unix_nano", "startTime", "start_time")
        )
        span["attributes"] = attributes
        span["span_id"] = str(_first(raw, "spanId", "span_id", default="") or "")
        span["trace_id"] = str(_first(raw, "traceId", "trace_id", default="") or "")
        span["parent_span_id"] = str(parent) if parent not in (None, "") else None
        span["name"] = str(raw.get("name") or "")
        span["start_time"] = start
        span["end_time"] = _time_value(
            _first(
                raw,
                "endTimeUnixNano",
                "end_time_unix_nano",
                "endTime",
                "end_time",
                default=start,
            )
        )
        span["status"] = _normalized_status(raw.get("status"))
        span["links"] = _normalized_links(raw.get("links"))
        if unreadable:
            span["unreadable_attributes"] = unreadable
        spans.append(span)
    return spans


def _usable_time(raw: Any) -> int | float | None:
    """A timestamp as a number, or ``None`` when it is unusable.

    Out of range is unusable rather than clamped: a JSON integer is
    unbounded, and feeding one to float arithmetic raises OverflowError,
    which on this path (unlike the capture path) has no guard above it and
    would abort the whole import over one malformed span.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if abs(raw) <= _MAX_TIME_VALUE else None
    if isinstance(raw, float):
        return raw if math.isfinite(raw) and abs(raw) <= _MAX_TIME_VALUE else None
    if isinstance(raw, str):
        text = raw.strip()
        # Protobuf JSON encodes 64-bit fields as strings; keeping integral
        # ones as int avoids losing ~250ns of precision to float53.
        if text.lstrip("+-").isdigit():
            number: int | float = _integer(text)
        else:
            try:
                number = float(text)
            except ValueError:
                return None
        if isinstance(number, float) and not math.isfinite(number):
            return None
        return number if abs(number) <= _MAX_TIME_VALUE else None
    return None


def _time_value(raw: Any) -> int | float:
    """Coerce an OTLP timestamp; unusable values become 0."""
    value = _usable_time(raw)
    return value if value is not None else 0


def _status_code(code: Any) -> str:
    """Normalize a span status code across every encoding in the wild."""
    if isinstance(code, bool):
        return ""
    # Protobuf JSON emits the enum as a number or its full name; the OTel
    # Python SDK's to_json emits the short name ("ERROR").
    if isinstance(code, int):
        return _STATUS_CODES.get(code, str(code))
    return str(code or "")


def _normalized_status(status: Any) -> dict[str, Any]:
    if not isinstance(status, dict):
        return {}
    return {
        "status_code": _status_code(
            _first(status, "status_code", "statusCode", "code", default="")
        ),
        "description": str(_first(status, "description", "message", default="") or ""),
    }


def _normalized_links(links: Any) -> list[dict[str, str]]:
    if not isinstance(links, list):
        return []
    normalized: list[dict[str, str]] = []
    for link in links:
        if not isinstance(link, dict):
            continue
        span_id = _first(link, "spanId", "span_id")
        if span_id in (None, ""):
            continue
        normalized.append(
            {
                "span_id": str(span_id),
                "trace_id": str(_first(link, "traceId", "trace_id", default="") or ""),
            }
        )
    return normalized


def _attr(span: dict[str, Any], *names: str, default: Any = None) -> Any:
    attributes = span.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}
    for name in names:
        if name in attributes:
            return attributes[name]
        if name in span:
            return span[name]
    return default


def _time_scale(magnitude: float) -> float:
    """Infer a timestamp's unit from its magnitude.

    Epoch values are ~1.7e18 ns, ~1.7e15 us, ~1.7e12 ms and ~1.7e9 s, so the
    bands are three orders of magnitude apart and unambiguous for any date
    from 1973 to 2286. Anything smaller is a relative offset already in
    seconds (and is what hand-written fixtures use).
    """
    if magnitude > 1e17:
        return 1e9
    if magnitude > 1e14:
        return 1e6
    if magnitude > 1e11:
        return 1e3
    return 1.0


def _span_duration(span: dict[str, Any]) -> float:
    # Both endpoints go through _usable_time even on the un-normalized path,
    # where a raw span dict reaches this function untouched.
    start = _usable_time(span.get("start_time"))
    end = _usable_time(span.get("end_time"))
    if start is None or end is None or end < start:
        return 0.0
    return (end - start) / _time_scale(max(abs(start), abs(end)))


def _span_error(span: dict[str, Any]) -> str | None:
    """Return a span's failure message, or ``None`` when it succeeded."""
    status = span.get("status")
    status = status if isinstance(status, dict) else {}
    code = _status_code(_first(status, "status_code", "statusCode", "code", default=""))
    description = str(_first(status, "description", "message", default="") or "")
    if code.upper() in _ERROR_CODES:
        return description or "span reported error status"
    # semconv marks a failed GenAI operation with error.type even when the
    # exporter leaves the span status unset.
    error_type = _attr(span, "error.type")
    if error_type not in (None, "", False):
        return description or f"error.type={error_type}"
    return None


def _span_usage(span: dict[str, Any]) -> dict[str, int | float] | None:
    """Collect every ``gen_ai.usage.*`` dimension, canonicalized by state.sane_usage."""
    attributes = span.get("attributes")
    source: dict[str, Any] = dict(span)
    if isinstance(attributes, dict):
        source.update(attributes)
    return sane_usage(
        {
            name[len(_USAGE_PREFIX) :]: value
            for name, value in source.items()
            if isinstance(name, str) and name.startswith(_USAGE_PREFIX)
        }
    )


# ---------------------------------------------------------- normalized replay


@dataclass
class _Unit:
    """One unit of recorded work, the shape every ingest path converges on.

    ``role`` "skip" keeps a span in the tree without recording it, so a model
    call nested under a non-GenAI span (an HTTP client span, say) still finds
    its owning node when the ancestor walk passes through.
    """

    key: str
    role: str
    name: str = ""
    parent: str | None = None
    causal: tuple[str, ...] = ()
    model: str = ""
    inputs: Any = None
    outputs: Any = None
    node_outputs: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] | None = None
    cost: float = 0.0
    duration: float = 0.0
    error: str | None = None


def _ordered(units: list[_Unit]) -> list[_Unit]:
    """Stable order with every parent before its children (cycle-safe)."""
    by_key: dict[str, _Unit] = {}
    for unit in units:
        by_key.setdefault(unit.key, unit)
    ordered: list[_Unit] = []
    placed: set[str] = set()
    for unit in units:
        if unit.key in placed:
            continue
        chain: list[_Unit] = []
        seen: set[str] = set()
        cursor: _Unit | None = unit
        while cursor is not None and cursor.key not in placed and cursor.key not in seen:
            chain.append(cursor)
            seen.add(cursor.key)
            cursor = by_key.get(cursor.parent or "")
        for item in reversed(chain):
            placed.add(item.key)
            ordered.append(item)
    return ordered


def _node_ancestor(unit: _Unit, by_key: dict[str, _Unit]) -> str | None:
    """Key of the nearest enclosing node unit, if any."""
    cursor = unit.parent
    for _ in range(_MAX_ANCESTOR_HOPS):
        if cursor is None or cursor not in by_key:
            return None
        parent = by_key[cursor]
        if parent.role == "node":
            return parent.key
        cursor = parent.parent
    return None


def _correlate_children(units: list[_Unit]) -> None:
    """List each node's nested call ids on the node step.

    Model and tool steps have no free-form outputs of their own, so their
    backend span ids would be lost; naming them on the owning node keeps an
    ingested run navigable back to the tracing UI.
    """
    by_key = {unit.key: unit for unit in units}
    for unit in units:
        if unit.role not in ("model", "tool"):
            continue
        ancestor = _node_ancestor(unit, by_key)
        owner = by_key.get(ancestor or "")
        if owner is None:
            continue
        children = owner.node_outputs.setdefault("child_span_ids", [])
        if isinstance(children, list) and len(children) < _MAX_CORRELATED_SPANS:
            children.append(unit.key)


def _record_units(capture: GraphRunCapture, units: list[_Unit]) -> list[str]:
    """Replay normalized units onto a capture; returns the failure messages."""
    by_key = {unit.key: unit for unit in units}
    node_names: dict[str, str] = {}
    node_steps: dict[str, str] = {}
    previous_node: str | None = None
    errors: list[str] = []
    for unit in units:
        ancestor = _node_ancestor(unit, by_key)
        node_id = node_names.get(ancestor or "")
        if unit.role == "skip":
            continue
        if unit.error:
            errors.append(unit.error)
        if unit.role == "node":
            node_names[unit.key] = unit.name
            edge = None
            if ancestor and ancestor in node_names:
                edge = {"kind": "span", "source": node_names[ancestor]}
            elif previous_node and previous_node in node_names:
                edge = {"kind": "sequence", "source": node_names[previous_node]}
            capture.node_start(unit.name, edge=edge, key=unit.key)
            # Span links / causal refs are fan-in: naming every already
            # recorded parent step keeps multi-parent lineage instead of
            # flattening a join into one chain.
            parents = [
                node_steps[key] for key in (ancestor or "", *unit.causal) if key in node_steps
            ]
            step_id = capture.node_end(
                unit.name,
                outputs=unit.node_outputs or None,
                error=unit.error,
                duration=unit.duration,
                cost=unit.cost,
                usage=unit.usage,
                key=unit.key,
                parents=parents or None,
            )
            if step_id:
                node_steps[unit.key] = step_id
            previous_node = unit.key
        elif unit.role == "model":
            capture.model_call(
                model=unit.model,
                prompt=unit.inputs,
                response=unit.outputs,
                cost=unit.cost,
                duration=unit.duration,
                usage=unit.usage,
                node_id=node_id,
                key=ancestor,
            )
            if unit.error:
                # model_call has no error channel, so a failed call is
                # recorded as its own error step attributed to the node.
                capture.error(unit.error, node_id=node_id)
        elif unit.role == "tool":
            capture.tool_call(
                unit.name or "tool",
                args=unit.inputs,
                result=unit.outputs,
                error=unit.error,
                duration=unit.duration,
                node_id=node_id,
                key=ancestor,
            )
        elif unit.role == "error":
            capture.error(unit.error or unit.name or "error", node_id=node_id)
    return errors


def _finish(capture: GraphRunCapture, errors: list[str], summary: str) -> Path | None:
    """End the run, surfacing the first failure in the summary."""
    if errors:
        summary = f"{summary}; first error: {errors[0]}"
    return capture.end(status="failed" if errors else "completed", summary=summary)


def _begin_correlated(
    capture: GraphRunCapture,
    source: str,
    trace_ids: Iterable[str],
    notes: dict[str, Any] | None = None,
) -> None:
    """Stamp the backing trace ids on the run before any step is recorded."""
    identifiers = sorted({str(value) for value in trace_ids if value})[:_MAX_CORRELATED_SPANS]
    if identifiers or notes:
        capture.begin(graph={"source": source, "trace_ids": identifiers, **(notes or {})})


# ------------------------------------------------------------------ OTLP spans


def _prepared_spans(spans: Iterable[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    """Accept extracted span lists *and* raw OTLP/JSON, self-healing either way.

    A real OTLP export nests spans under ``resourceSpans`` and encodes
    attributes as a list of AnyValue records, which reads as "no attributes"
    to every ``gen_ai.*`` lookup — an empty run with no error. Normalizing is
    also what maps ``spanId``/``parentSpanId``/``startTimeUnixNano`` onto the
    snake_case names read downstream, so decoded attributes alone are not
    evidence that a batch is flat: a span keeping any camelCase OTLP spelling
    goes through :func:`normalize_otlp` too.

    The short-circuit cannot simply be dropped, because the OTLP walk yields
    only spans carrying an id: a hand-built flat span with no ``span_id``
    would vanish from its own run.
    """
    if isinstance(spans, dict):
        return normalize_otlp(spans)
    items = [span for span in spans if isinstance(span, dict)]
    if items and all(
        isinstance(span.get("attributes"), dict) and not (_CAMEL_OTLP_KEYS & span.keys())
        for span in items
    ):
        return items
    return normalize_otlp(items)


def _unreadable(span: dict[str, Any]) -> dict[str, list[str]]:
    """Attributes dropped while decoding, so the loss is recorded, not hidden."""
    dropped = span.get("unreadable_attributes")
    return {"unreadable_attributes": [str(key) for key in dropped]} if dropped else {}


def _span_unit(span: dict[str, Any], index: int) -> _Unit:
    operation = str(_attr(span, "gen_ai.operation.name", default="") or "")
    key = str(span.get("span_id") or "") or f"span-{index}"
    parent = span.get("parent_span_id")
    causal = tuple(
        str(identifier)
        for link in span.get("links") or []
        if isinstance(link, dict)
        and (identifier := _first(link, "span_id", "spanId")) not in (None, "")
    )
    shared = {
        "key": key,
        "parent": str(parent) if parent not in (None, "") else None,
        "causal": causal,
        "duration": _span_duration(span),
        "error": _span_error(span),
    }
    if operation in _NODE_OPS:
        return _Unit(
            role="node",
            name=str(
                _attr(span, "gen_ai.agent.name", "gen_ai.agent.id", default="")
                or span.get("name")
                or "node"
            ),
            usage=_span_usage(span),
            node_outputs={
                "operation": operation,
                "conversation": _attr(span, "gen_ai.conversation.id", default=""),
                "span_id": key,
                "trace_id": str(span.get("trace_id") or ""),
                **({"link_span_ids": list(causal)} if causal else {}),
                **_unreadable(span),
            },
            **shared,
        )
    if operation in _MODEL_OPS:
        return _Unit(
            role="model",
            name=operation,
            model=str(_attr(span, "gen_ai.response.model", "gen_ai.request.model", default="")),
            inputs=_attr(span, "gen_ai.input.messages"),
            outputs=_attr(span, "gen_ai.output.messages"),
            usage=_span_usage(span),
            **shared,
        )
    if operation in _TOOL_OPS:
        return _Unit(
            role="tool",
            name=str(_attr(span, "gen_ai.tool.name", default="") or "tool"),
            inputs=_attr(span, "gen_ai.tool.call.arguments"),
            outputs=_attr(span, "gen_ai.tool.call.result"),
            **shared,
        )
    # Not a GenAI span: kept only so the ancestor walk can pass through it.
    return _Unit(key=key, role="skip", parent=shared["parent"])


def ingest_otel_spans(
    goal: str,
    spans: Iterable[dict[str, Any]] | dict[str, Any],
    *,
    runs_dir: str | Path | None = None,
    framework: str = "otel-genai",
    policy: GraphPolicy | None = None,
) -> Path | None:
    """Record OTel GenAI spans — or a whole OTLP/JSON export — as one graph run.

    Node identity: ``invoke_agent``/``invoke_workflow`` spans become nodes
    (named by ``gen_ai.agent.name`` falling back to the span name); ``chat``
    and friends become model calls and ``execute_tool`` spans tool calls,
    attributed to their nearest ancestor node span via ``parent_span_id``.
    Span links become extra node parents (fan-in), span ids are recorded on
    the steps they produced, and an error status on *any* span — not just an
    agent span — fails the run. Spans are processed in start-time order.
    """
    span_list = _prepared_spans(spans)
    # _time_value, not the raw field: an un-normalized batch can mix string
    # and numeric timestamps, and sorting those against each other raises.
    span_list.sort(key=lambda span: _time_value(span.get("start_time")))
    units = _ordered([_span_unit(span, index) for index, span in enumerate(span_list)])
    _correlate_children(units)
    unreadable = {
        str(span.get("span_id") or f"span-{index}"): [str(key) for key in dropped]
        for index, span in enumerate(span_list)
        if (dropped := span.get("unreadable_attributes"))
    }
    capture = GraphRunCapture(goal, framework=framework, runs_dir=runs_dir, policy=policy)
    _begin_correlated(
        capture,
        "otel",
        (span.get("trace_id") for span in span_list),
        {"unreadable_attributes": unreadable} if unreadable else None,
    )
    errors = _record_units(capture, units)
    summary = f"ingested {len(span_list)} otel spans"
    if unreadable:
        # Silent attribute loss is how a whole node disappears unnoticed.
        summary = f"{summary}; {sum(len(v) for v in unreadable.values())} unreadable attributes"
    return _finish(capture, errors, summary)


# ---------------------------------------------------------------- trace events


def _event_mapping(event: Any) -> dict[str, Any] | None:
    """Accept a ``TraceEvent``, any dataclass of that shape, or a plain dict."""
    if isinstance(event, dict):
        return event
    if _TraceEvent is not None and isinstance(event, _TraceEvent):
        return event.to_dict()
    to_dict = getattr(event, "to_dict", None)
    if callable(to_dict):
        try:
            value = to_dict()
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None
    if is_dataclass(event) and not isinstance(event, type):
        return {item.name: getattr(event, item.name) for item in fields(event)}
    return None


def _event_unit(event: dict[str, Any], index: int, container: bool) -> _Unit:
    attributes = event.get("attributes")
    attributes = attributes if isinstance(attributes, dict) else {}
    outputs = event.get("outputs")
    kind = str(event.get("kind") or "model").casefold()
    kind = kind if kind in _TRACE_KINDS else "model"
    operation = str(attributes.get("gen_ai.operation.name") or "")
    actor = str(event.get("actor") or "")
    model = str(event.get("model") or "")
    key = str(event.get("span_id") or "") or f"event-{index}"
    parent = event.get("parent_span_id")
    causal = event.get("causal_span_ids")
    error = _event_error(event, kind, attributes, outputs)
    shared = {
        "key": key,
        "parent": str(parent) if parent not in (None, "") else None,
        "causal": tuple(str(value) for value in causal if value not in (None, ""))
        if isinstance(causal, list | tuple)
        else (),
        "inputs": event.get("inputs"),
        "outputs": outputs,
        "usage": sane_usage(event.get("usage")),
        "cost": sane_metric(event.get("cost")),
        "duration": sane_metric(event.get("duration")),
        "error": error,
    }
    # A model-kind event that is the parent of other events and names no
    # model is a container (a LangChain chain, a workflow step), not a leaf
    # call — importers have no dedicated kind for it.
    is_node = kind in _NODE_KINDS or operation in _NODE_OPS
    is_node = is_node or (kind == "model" and container and not model and not operation)
    if is_node:
        return _Unit(
            role="node",
            name=str(attributes.get("gen_ai.agent.name") or actor or kind),
            node_outputs={
                "kind": kind,
                "actor": actor,
                "span_id": key,
                "trace_id": str(event.get("trace_id") or ""),
                **({"causal_span_ids": list(shared["causal"])} if shared["causal"] else {}),
            },
            **shared,
        )
    if kind == "tool" or operation in _TOOL_OPS:
        return _Unit(role="tool", name=actor or "tool", **shared)
    if kind == "error":
        return _Unit(role="error", name=actor or "error", **shared)
    return _Unit(role="model", name=actor or operation or "model", model=model, **shared)


def _event_error(
    event: dict[str, Any], kind: str, attributes: dict[str, Any], outputs: Any
) -> str | None:
    payload = outputs if isinstance(outputs, dict) else {}
    message = str(
        payload.get("error")
        or payload.get("message")
        or attributes.get("error.type")
        or attributes.get("error")
        or ""
    )
    if kind == "error":
        return message or str(event.get("actor") or "") or "trace event reported an error"
    return message or None


def ingest_trace_events(
    goal: str,
    events: Iterable[Any],
    *,
    runs_dir: str | Path | None = None,
    framework: str = "opentine-trace",
    policy: GraphPolicy | None = None,
) -> Path | None:
    """Record an opentine ``TraceEvent`` stream (or dicts of that shape) as a run.

    Every importer in :mod:`opentine.trace` yields ``TraceEvent`` lists, so
    this is the one entry point that turns *any* of them — native runs,
    JSONL, OTLP, framework callbacks — into a graph artifact. Events are
    ordered by timestamp and then topologically by ``parent_span_id`` so a
    parent is always recorded before its children; ``parent_span_id`` plus
    ``causal_span_ids`` become the node's parent steps, which is how a
    multi-parent join survives the import.

    ``kind`` (opentine's own vocabulary: model, tool, human, policy,
    approval, subagent, error) picks the recording: subagent/human/policy/
    approval are node executions, model and tool are the calls beneath them,
    and error marks the run failed.
    """
    mappings = [mapping for mapping in map(_event_mapping, events) if mapping is not None]
    mappings.sort(key=lambda event: sane_metric(event.get("timestamp")))
    parents = {
        str(event.get("parent_span_id"))
        for event in mappings
        if event.get("parent_span_id") not in (None, "")
    }
    units = _ordered(
        [
            _event_unit(event, index, str(event.get("span_id") or "") in parents)
            for index, event in enumerate(mappings)
        ]
    )
    _correlate_children(units)
    capture = GraphRunCapture(goal, framework=framework, runs_dir=runs_dir, policy=policy)
    _begin_correlated(capture, framework, (event.get("trace_id") for event in mappings))
    errors = _record_units(capture, units)
    return _finish(capture, errors, f"ingested {len(mappings)} trace events")


# ------------------------------------------------------------ framework traces


def _jsonable_record(record: Any) -> dict[str, Any] | None:
    """Round-trip a live framework record into plain JSON types.

    LangChain hands back ``UUID`` run ids and ``datetime`` timestamps, and
    opentine's importer refuses any payload type it cannot store — so without
    this the very first real record raises ValueError and the whole ingest
    dies.
    """
    if not isinstance(record, dict):
        return None
    try:
        encoded = json.dumps(record, default=str)
    except TypeError:
        # A non-string mapping key (rare, but UUID-keyed metadata happens):
        # drop that key rather than the whole record.
        try:
            encoded = json.dumps(record, default=str, skipkeys=True)
        except (TypeError, ValueError, RecursionError):
            return None
    except (ValueError, RecursionError):
        return None
    try:
        decoded = json.loads(encoded)
    except (ValueError, RecursionError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _import_framework_records(records: Iterable[Any], framework: str) -> list[Any]:
    """Import records dialect-by-dialect, dropping only the records that fail."""
    if _opentine_framework_events is None:  # pragma: no cover - opentine 0.2.0
        raise RuntimeError("framework trace import requires opentine>=0.3.0")
    # Probe the dialect on an empty batch: an unknown framework name is a
    # caller bug and must raise, while a bad record must not.
    _opentine_framework_events([], framework)
    sanitized = [record for record in map(_jsonable_record, records) if record is not None]
    try:
        return list(_opentine_framework_events(sanitized, framework))
    except (ValueError, TypeError, RecursionError, OverflowError):
        pass
    events: list[Any] = []
    for record in sanitized:
        try:
            events.extend(_opentine_framework_events([record], framework))
        except (ValueError, TypeError, RecursionError, OverflowError):
            continue
    return events


def ingest_framework(
    goal: str,
    records: Iterable[Any],
    framework: str,
    *,
    runs_dir: str | Path | None = None,
    policy: GraphPolicy | None = None,
) -> Path | None:
    """Record a framework's own callback records as a graph run.

    ``framework`` is one of :data:`FRAMEWORK_DIALECTS` (``langchain``,
    ``llamaindex``, ``autogen``, ``crewai``, ``openai-agents``); each names
    the record keys opentine reads for identity, parentage and actor —
    LangChain's ``run_id``/``parent_run_id``/``name``, LlamaIndex's
    ``id_``/``parent_id``/``event_type``, and so on. Records are JSON
    round-tripped first (live records carry ``UUID``s and ``datetime``s that
    the importer rejects) and a record the importer still refuses is dropped
    on its own rather than failing the ingest.

    Raises ``ValueError`` for an unsupported dialect and ``RuntimeError``
    when the installed opentine predates the importers (0.2.0).
    """
    events = _import_framework_records(records, framework)
    return ingest_trace_events(goal, events, runs_dir=runs_dir, framework=framework, policy=policy)


# ------------------------------------------------------------------ JSONL path


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
        lines: Iterable[str] = read_bounded_text(source).splitlines()
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


#: Largest input file the file-based ingest entry points will read.
MAX_INGEST_BYTES = 256 * 1024 * 1024


def read_bounded_text(source: str | Path, *, limit: int = MAX_INGEST_BYTES) -> str:
    """Read a regular UTF-8 file, refusing anything larger than ``limit``.

    The check is on bytes actually read, not just ``stat``, so a FIFO or a
    file that grows while being read cannot get past it.
    """
    path = Path(source).expanduser()
    if not path.is_file():
        raise ValueError(f"{path} is not a regular file")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{path} exceeds the {limit}-byte ingest limit")
    return data.decode("utf-8")
