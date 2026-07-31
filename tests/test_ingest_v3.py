"""Ingestion of real OTLP exports, opentine trace events and framework records.

Everything here that needs opentine's 0.3.0 trace surface is gated on a
feature probe, so the suite stays green on opentine 0.2.0 where the
``opentine.trace`` importers do not exist.
"""

from __future__ import annotations

import builtins
import importlib
import json
import uuid
from datetime import datetime
from typing import Any

import pytest
from opentine import Run

from graphforge import ingest as ingest_module
from graphforge.ingest import (
    _span_duration,
    ingest_framework,
    ingest_otel_spans,
    ingest_trace_events,
    normalize_otlp,
)

try:  # opentine 0.2.0 has no normalized trace schema or importers
    from opentine.trace import TraceEvent, framework_events
except ImportError:  # pragma: no cover - depends on the installed opentine
    TraceEvent = None
    framework_events = None

requires_trace = pytest.mark.skipif(
    TraceEvent is None, reason="requires opentine>=0.3.0 trace schema"
)
requires_importers = pytest.mark.skipif(
    framework_events is None, reason="requires opentine>=0.3.0 trace importers"
)

SECOND = 1_000_000_000
BASE = 1_700_000_000 * SECOND


def _value(raw: Any) -> dict[str, Any]:
    if isinstance(raw, bool):
        return {"boolValue": raw}
    if isinstance(raw, int):
        return {"intValue": str(raw)}  # protobuf JSON encodes 64-bit ints as strings
    if isinstance(raw, float):
        return {"doubleValue": raw}
    if isinstance(raw, list):
        return {"arrayValue": {"values": [_value(item) for item in raw]}}
    if isinstance(raw, dict):
        return {
            "kvlistValue": {
                "values": [{"key": key, "value": _value(item)} for key, item in raw.items()]
            }
        }
    return {"stringValue": str(raw)}


def _otlp_span(
    span_id: str,
    parent: str,
    name: str,
    attributes: dict[str, Any],
    *,
    start: int,
    end: int,
    status: dict[str, Any] | None = None,
    links: list[str] | None = None,
) -> dict[str, Any]:
    """A span exactly as an OTLP/JSON exporter emits it."""
    span = {
        "traceId": "4bf92f3577b34da6a3ce929d0e0e4736",
        "spanId": span_id,
        "parentSpanId": parent,
        "name": name,
        "kind": "SPAN_KIND_INTERNAL",
        "startTimeUnixNano": str(start),
        "endTimeUnixNano": str(end),
        "attributes": [{"key": key, "value": _value(item)} for key, item in attributes.items()],
    }
    if status is not None:
        span["status"] = status
    if links:
        span["links"] = [
            {"traceId": "4bf92f3577b34da6a3ce929d0e0e4736", "spanId": link} for link in links
        ]
    return span


def _otlp_document(spans: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [{"key": "service.name", "value": _value("agent")}]},
                "scopeSpans": [
                    {
                        "scope": {"name": "opentelemetry.instrumentation.anthropic"},
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def _agent_document() -> dict[str, Any]:
    return _otlp_document(
        [
            _otlp_span(
                "a1",
                "",
                "invoke_agent planner",
                {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "planner"},
                start=BASE,
                end=BASE + 2 * SECOND,
            ),
            _otlp_span(
                "b1",
                "a1",
                "chat claude-sonnet-5",
                {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": "claude-sonnet-5",
                    "gen_ai.usage.input_tokens": 900,
                    "gen_ai.usage.output_tokens": 100,
                    "gen_ai.input.messages": ["plan the trip"],
                },
                start=BASE + SECOND // 2,
                end=BASE + SECOND,
            ),
            _otlp_span(
                "c1",
                "b1",
                "execute_tool search",
                {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": "search",
                    "gen_ai.tool.call.arguments": '{"q": "x"}',
                },
                start=BASE + SECOND + SECOND // 4,
                end=BASE + SECOND + SECOND // 2,
            ),
            _otlp_span(
                "a2",
                "",
                "invoke_agent executor",
                {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "executor"},
                start=BASE + 3 * SECOND,
                end=BASE + 4 * SECOND,
            ),
        ]
    )


def _steps(artifact) -> list[Any]:
    return list(Run.load(artifact).graph.steps.values())


def _node(steps: list[Any], name: str) -> Any:
    return next(s for s in steps if s.inputs.get("node") == name and "edge" in s.inputs)


def _custom(step: Any) -> dict[str, Any]:
    """A node step's caller-supplied outputs (recorder nests them)."""
    return step.outputs.get("outputs") or {}


# ------------------------------------------------------------- normalize_otlp


def test_normalize_otlp_flattens_a_real_export():
    spans = normalize_otlp(_agent_document())
    assert [span["span_id"] for span in spans] == ["a1", "b1", "c1", "a2"]
    agent, chat = spans[0], spans[1]
    assert agent["parent_span_id"] is None  # empty parentSpanId is a root, not ""
    assert agent["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert agent["attributes"]["gen_ai.agent.name"] == "planner"
    assert agent["start_time"] == BASE and agent["end_time"] == BASE + 2 * SECOND
    assert chat["parent_span_id"] == "a1"
    assert chat["attributes"]["gen_ai.usage.input_tokens"] == 900  # intValue decoded to int
    assert chat["attributes"]["gen_ai.input.messages"] == ["plan the trip"]
    assert spans[0]["name"] == "invoke_agent planner"


def test_normalize_otlp_decodes_every_any_value_shape():
    document = _otlp_document(
        [
            _otlp_span(
                "s1",
                "",
                "chat",
                {
                    "gen_ai.operation.name": "chat",
                    "text": "hello",
                    "count": 7,
                    "ratio": 0.5,
                    "flag": True,
                    "list": ["a", 2],
                    "map": {"inner": "v"},
                },
                start=BASE,
                end=BASE,
            )
        ]
    )
    attributes = normalize_otlp(document)[0]["attributes"]
    assert attributes["text"] == "hello"
    assert attributes["count"] == 7
    assert attributes["ratio"] == 0.5
    assert attributes["flag"] is True
    assert attributes["list"] == ["a", 2]
    assert attributes["map"] == {"inner": "v"}


def test_normalize_otlp_accepts_lists_envelopes_and_single_spans():
    spans = _agent_document()["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(normalize_otlp(spans)) == 4
    assert len(normalize_otlp({"spans": spans})) == 4
    assert normalize_otlp(spans[0])[0]["span_id"] == "a1"


def test_normalize_otlp_keeps_flat_snake_case_spans_intact():
    flat = [
        {
            "span_id": "s1",
            "parent_span_id": None,
            "name": "n",
            "start_time": 1,
            "end_time": 4,
            "attributes": {"gen_ai.operation.name": "chat"},
            "custom": {"keep": True},
        }
    ]
    span = normalize_otlp(flat)[0]
    assert span["span_id"] == "s1"
    assert span["start_time"] == 1 and span["end_time"] == 4
    assert span["attributes"] == {"gen_ai.operation.name": "chat"}
    assert span["custom"] == {"keep": True}  # unknown keys survive the round trip


# ---------------------------------------------------- opentine 0.2.0 fallbacks


def _without_opentine_helpers(monkeypatch) -> None:
    """Pretend opentine's private OTLP helpers are missing (0.2.0, or moved)."""
    for name in (
        "_opentine_first",
        "_opentine_integer",
        "_opentine_otel_spans",
        "_opentine_attributes",
        "_opentine_any_value",
    ):
        monkeypatch.setattr(ingest_module, name, None)


def test_pure_python_fallback_normalizes_identically(monkeypatch):
    document = _agent_document()
    expected = normalize_otlp(document)
    _without_opentine_helpers(monkeypatch)
    assert normalize_otlp(document) == expected


def test_pure_python_fallback_ingests_identically(tmp_path, monkeypatch):
    _without_opentine_helpers(monkeypatch)
    steps = _steps(ingest_otel_spans("fallback", _agent_document(), runs_dir=tmp_path))
    planner = _node(steps, "planner")
    assert planner.duration == 2.0
    assert _custom(planner)["span_id"] == "a1"
    assert any(s.inputs.get("model") == "claude-sonnet-5" for s in steps)


def test_module_imports_without_the_opentine_trace_package(monkeypatch):
    # The whole point of the guards: graphforge must still import (and the
    # OTLP path still work) against an opentine that has no trace package.
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("opentine.trace"):
            raise ImportError(f"no module named {name}")
        return real_import(name, *args, **kwargs)

    try:
        monkeypatch.setattr(builtins, "__import__", blocked)
        reloaded = importlib.reload(ingest_module)
        monkeypatch.undo()
        assert reloaded._opentine_otel_spans is None
        assert reloaded._TraceEvent is None
        assert len(reloaded.normalize_otlp(_agent_document())) == 4
        with pytest.raises(RuntimeError):
            reloaded.ingest_framework("no importers", [{"run_id": "a"}], "langchain")
    finally:
        importlib.reload(ingest_module)


# ------------------------------------------------------------ ingest_otel_spans


def test_ingest_otel_spans_reads_a_real_otlp_document(tmp_path):
    artifact = ingest_otel_spans("otlp run", _agent_document(), runs_dir=tmp_path)
    run = Run.load(artifact)
    steps = list(run.graph.steps.values())
    planner = _node(steps, "planner")
    executor = _node(steps, "executor")
    assert executor.inputs["edge"]["source"] == "planner"
    assert planner.duration == 2.0  # nanosecond epoch, not 2e-9
    assert _custom(planner)["span_id"] == "a1"
    assert _custom(planner)["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert set(_custom(planner)["child_span_ids"]) == {"b1", "c1"}
    model = next(s for s in steps if s.inputs.get("model") == "claude-sonnet-5")
    assert model.usage == {"input": 900, "output": 100}
    assert model.duration == 0.5
    assert planner.id in model.parent_ids
    tool = next(s for s in steps if s.inputs.get("tool") == "search")
    assert planner.id in tool.parent_ids  # nested under chat, owned by the agent
    assert getattr(run.status, "value", str(run.status)) == "completed"
    assert "ingested 4 otel spans" in str(run.to_dict())


def test_ingest_otel_spans_error_on_chat_span_fails_the_run(tmp_path):
    document = _otlp_document(
        [
            _otlp_span(
                "a1",
                "",
                "invoke_agent",
                {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "worker"},
                start=BASE,
                end=BASE + SECOND,
            ),
            _otlp_span(
                "b1",
                "a1",
                "chat",
                {"gen_ai.operation.name": "chat", "gen_ai.request.model": "m"},
                start=BASE,
                end=BASE + SECOND,
                status={"code": "STATUS_CODE_ERROR", "message": "model overloaded"},
            ),
        ]
    )
    artifact = ingest_otel_spans("otlp err", document, runs_dir=tmp_path)
    run = Run.load(artifact)
    assert getattr(run.status, "value", str(run.status)) == "failed"
    assert "model overloaded" in str(run.to_dict())


def test_ingest_otel_spans_error_on_tool_span_fails_the_run(tmp_path):
    document = _otlp_document(
        [
            _otlp_span(
                "a1",
                "",
                "invoke_agent",
                {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "worker"},
                start=BASE,
                end=BASE + SECOND,
            ),
            _otlp_span(
                "t1",
                "a1",
                "execute_tool",
                {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "search"},
                start=BASE,
                end=BASE + SECOND,
                status={"code": 2, "message": "search timed out"},
            ),
        ]
    )
    run = Run.load(ingest_otel_spans("tool err", document, runs_dir=tmp_path))
    assert getattr(run.status, "value", str(run.status)) == "failed"
    tool = next(s for s in run.graph.steps.values() if s.inputs.get("tool") == "search")
    assert "search timed out" in str(tool.outputs)


def test_ingest_otel_spans_links_become_multi_parent_fan_in(tmp_path):
    def agent(span_id: str, name: str, start: int, links: list[str] | None = None):
        return _otlp_span(
            span_id,
            "",
            f"invoke_agent {name}",
            {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": name},
            start=start,
            end=start + SECOND,
            links=links,
        )

    document = _otlp_document(
        [
            agent("a1", "left", BASE),
            agent("a2", "right", BASE + SECOND),
            agent("a3", "join", BASE + 2 * SECOND, links=["a1", "a2"]),
        ]
    )
    steps = _steps(ingest_otel_spans("fan-in", document, runs_dir=tmp_path))
    left, right, join = _node(steps, "left"), _node(steps, "right"), _node(steps, "join")
    assert {left.id, right.id} <= set(join.parent_ids)
    assert _custom(join)["link_span_ids"] == ["a1", "a2"]


def test_ingest_otel_spans_walks_through_non_genai_spans(tmp_path):
    document = _otlp_document(
        [
            _otlp_span(
                "a1",
                "",
                "invoke_agent",
                {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "planner"},
                start=BASE,
                end=BASE + 2 * SECOND,
            ),
            # An HTTP client span between the agent and its model call: it is
            # not recorded, but it must not orphan the model call either.
            _otlp_span(
                "h1", "a1", "POST /v1/messages", {"http.method": "POST"}, start=BASE, end=BASE
            ),
            _otlp_span(
                "b1",
                "h1",
                "chat",
                {"gen_ai.operation.name": "chat", "gen_ai.request.model": "m"},
                start=BASE,
                end=BASE + SECOND,
            ),
        ]
    )
    steps = _steps(ingest_otel_spans("nested", document, runs_dir=tmp_path))
    planner = _node(steps, "planner")
    model = next(s for s in steps if s.inputs.get("model") == "m")
    assert planner.id in model.parent_ids


def test_ingest_otel_spans_still_accepts_flat_span_dicts(tmp_path):
    flat = [
        {
            "span_id": "s1",
            "parent_span_id": None,
            "name": "agent",
            "start_time": 0,
            "end_time": 3,
            "attributes": {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "solo"},
        }
    ]
    steps = _steps(ingest_otel_spans("flat", flat, runs_dir=tmp_path))
    assert _node(steps, "solo").duration == 3.0


# ------------------------------- pre-decoded batches, bad attributes, overflow


def _pre_decoded_spans() -> list[dict[str, Any]]:
    """Attributes already decoded by a pre-processing step; ids still OTLP."""
    return [
        {
            "traceId": "t1",
            "spanId": "a1",
            "parentSpanId": "",
            "name": "invoke_agent planner",
            "startTimeUnixNano": str(BASE),
            "endTimeUnixNano": str(BASE + 2 * SECOND),
            "attributes": {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "planner"},
        },
        {
            "traceId": "t1",
            "spanId": "a2",
            "parentSpanId": "",
            "name": "invoke_agent executor",
            "startTimeUnixNano": str(BASE + SECOND),
            "endTimeUnixNano": str(BASE + 2 * SECOND),
            "attributes": {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "executor",
            },
        },
        {
            "traceId": "t1",
            "spanId": "b1",
            "parentSpanId": "a1",
            "name": "chat",
            "startTimeUnixNano": str(BASE + 3 * SECOND),
            "endTimeUnixNano": str(BASE + 4 * SECOND),
            "attributes": {"gen_ai.operation.name": "chat", "gen_ai.request.model": "m"},
        },
    ]


def test_ingest_otel_spans_normalizes_pre_decoded_camel_case_spans(tmp_path):
    # Decoded attributes are not evidence that a batch is flat: skipping
    # normalization here loses every id, parent and timestamp, which shows up
    # as a zero duration and a model call attributed to the wrong node.
    steps = _steps(ingest_otel_spans("pre-decoded", _pre_decoded_spans(), runs_dir=tmp_path))
    planner, executor = _node(steps, "planner"), _node(steps, "executor")
    model = next(s for s in steps if s.inputs.get("model") == "m")
    assert planner.duration == 2.0
    assert _custom(planner)["span_id"] == "a1"
    assert planner.id in model.parent_ids  # its real parent, not the last step
    assert executor.id not in model.parent_ids


def test_ingest_otel_spans_records_flat_spans_that_carry_no_ids(tmp_path):
    # Why the normalization short-circuit has to stay: the OTLP walk yields
    # only spans with an id, so an id-less flat span would vanish entirely.
    spans = [
        {
            "name": "agent",
            "start_time": 0,
            "end_time": 2,
            "attributes": {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "solo"},
        }
    ]
    assert normalize_otlp(spans) == []
    steps = _steps(ingest_otel_spans("id-less", spans, runs_dir=tmp_path))
    assert _node(steps, "solo").duration == 2.0


def _sloppy_attribute_document() -> dict[str, Any]:
    document = _otlp_document(
        [
            _otlp_span(
                "a1",
                "",
                "invoke_agent planner",
                {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "planner"},
                start=BASE,
                end=BASE + SECOND,
            )
        ]
    )
    span = document["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    # A container where protobuf JSON promises a scalar: opentine's decoder
    # raises on this one attribute.
    span["attributes"].append({"key": "sloppy", "value": {"stringValue": {"nested": "x"}}})
    span["attributes"].append({"key": "big", "value": {"intValue": str(2**53 + 1)}})
    return document


def test_one_unreadable_attribute_does_not_delete_the_span():
    span = normalize_otlp(_sloppy_attribute_document())[0]
    assert span["attributes"]["gen_ai.operation.name"] == "invoke_agent"
    assert span["attributes"]["gen_ai.agent.name"] == "planner"
    assert span["unreadable_attributes"] == ["sloppy"]
    assert span["attributes"]["big"] == str(2**53 + 1)  # past 2**53, kept as text


def test_unreadable_attribute_is_recorded_not_hidden(tmp_path):
    artifact = ingest_otel_spans("sloppy", _sloppy_attribute_document(), runs_dir=tmp_path)
    run = Run.load(artifact)
    steps = list(run.graph.steps.values())
    planner = _node(steps, "planner")  # the node survives its bad neighbour
    assert _custom(planner)["unreadable_attributes"] == ["sloppy"]
    assert "1 unreadable attributes" in str(run.to_dict())
    assert run.manifest["graph"]["unreadable_attributes"] == {"a1": ["sloppy"]}


def test_unreadable_attribute_handling_matches_without_opentine_helpers(monkeypatch):
    document = _sloppy_attribute_document()
    expected = normalize_otlp(document)
    _without_opentine_helpers(monkeypatch)
    assert normalize_otlp(document) == expected


needs_opentine_decoder = pytest.mark.skipif(
    ingest_module._opentine_attributes is None, reason="requires opentine>=0.3.0 AnyValue decoder"
)


def _decoded_both_ways(monkeypatch, value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    span = {"span_id": "s", "attributes": [{"key": "k", "value": value}]}
    native = normalize_otlp([span])[0]
    _without_opentine_helpers(monkeypatch)
    return native, normalize_otlp([span])[0]


@needs_opentine_decoder
@pytest.mark.parametrize(
    "value",
    [
        {"stringValue": "x"},
        {"string_value": "x"},
        {"bytesValue": "aGk="},
        {"stringValue": {"nested": "container"}},  # scalar slot holding a container
        {"intValue": "5"},
        {"intValue": str(2**53 + 1)},  # past float53, kept as text
        {"intValue": ["x"]},
        {"intValue": "not a number"},
        {"doubleValue": 1.5},
        {"boolValue": True},
        {"boolValue": "true"},
        {"arrayValue": {"values": [{"stringValue": "a"}, {"intValue": "2"}]}},
        {"arrayValue": {}},
        {"kvlistValue": {"values": [{"key": "in", "value": {"stringValue": "v"}}]}},
        {"kvlistValue": {"values": [{"key": ["bad"], "value": {"stringValue": "v"}}]}},
        {"unknown": 1},
        {},
        "bare string",
        7,
    ],
)
def test_fallback_decoder_agrees_with_opentine_per_attribute(monkeypatch, value):
    native, fallback = _decoded_both_ways(monkeypatch, value)
    assert fallback["attributes"] == native["attributes"]
    assert fallback.get("unreadable_attributes") == native.get("unreadable_attributes")


@needs_opentine_decoder
def test_fallback_decoder_agrees_on_depth_and_cycle_bombs(monkeypatch):
    deep: Any = {"stringValue": "bottom"}
    for _ in range(120):
        deep = {"arrayValue": {"values": [deep]}}
    cyclic: dict[str, Any] = {"arrayValue": {"values": []}}
    cyclic["arrayValue"]["values"].append(cyclic)
    for value in (deep, cyclic):
        native, fallback = _decoded_both_ways(monkeypatch, value)
        assert native["attributes"] == {}
        assert native.get("unreadable_attributes") == ["k"]
        assert fallback["attributes"] == native["attributes"]
        assert fallback.get("unreadable_attributes") == native.get("unreadable_attributes")
        monkeypatch.undo()


@pytest.mark.parametrize("encode", [int, str])
def test_hostile_timestamp_still_produces_an_artifact(tmp_path, encode):
    huge = int("9" * 400)  # a JSON integer is unbounded; a float is not
    spans = [
        {
            "span_id": "s1",
            "parent_span_id": None,
            "name": "n",
            "start_time": encode(0),
            "end_time": encode(huge),
            "attributes": {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "worker"},
        }
    ]
    steps = _steps(ingest_otel_spans("hostile", spans, runs_dir=tmp_path))
    assert _node(steps, "worker").duration == 0.0


def test_hostile_timestamp_survives_normalization_too(tmp_path):
    huge = "9" * 400
    document = _otlp_document(
        [
            _otlp_span(
                "s1",
                "",
                "invoke_agent",
                {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "worker"},
                start=0,
                end=0,
            )
        ]
    )
    span = document["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    span["startTimeUnixNano"] = "0"
    span["endTimeUnixNano"] = huge
    assert normalize_otlp(document)[0]["end_time"] == 0  # unusable, not clamped high
    steps = _steps(ingest_otel_spans("hostile doc", document, runs_dir=tmp_path))
    assert _node(steps, "worker").duration == 0.0


def test_span_duration_survives_an_unbounded_integer():
    assert _span_duration({"start_time": 0, "end_time": 10**400}) == 0.0
    assert _span_duration({"start_time": -(10**400), "end_time": 10**400}) == 0.0


def _multi_list_document(lists: int, per_list: int) -> dict[str, Any]:
    """Spans spread over several scope lists, none of them individually large."""
    return {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            _otlp_span(
                                f"s{outer}-{inner}",
                                "",
                                "chat",
                                {"gen_ai.operation.name": "chat"},
                                start=BASE,
                                end=BASE,
                            )
                            for inner in range(per_list)
                        ]
                    }
                ]
            }
            for outer in range(lists)
        ]
    }


def test_input_record_cap_is_enforced_identically_on_both_implementations(monkeypatch):
    # opentine 0.3.0 caps per list, the fallback walks the whole document —
    # so graphforge counts yielded spans and both versions agree.
    document = _multi_list_document(lists=3, per_list=2)
    monkeypatch.setattr(ingest_module, "_MAX_SPANS", 6)
    assert len(normalize_otlp(document)) == 6
    monkeypatch.setattr(ingest_module, "_MAX_SPANS", 5)
    with pytest.raises(ValueError):
        normalize_otlp(document)
    _without_opentine_helpers(monkeypatch)
    with pytest.raises(ValueError):
        normalize_otlp(document)
    monkeypatch.setattr(ingest_module, "_MAX_SPANS", 6)
    assert len(normalize_otlp(document)) == 6


# ------------------------------------------------------------ duration scaling


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (1_700_000_000 * 1_000_000_000, 1_700_000_002 * 1_000_000_000),  # nanoseconds
        (1_700_000_000 * 1_000_000, 1_700_000_002 * 1_000_000),  # microseconds
        (1_700_000_000 * 1_000, 1_700_000_002 * 1_000),  # milliseconds
        (1_700_000_000, 1_700_000_002),  # seconds
        (0, 2),  # relative offsets, already seconds
    ],
)
def test_span_duration_detects_the_timestamp_scale(start, end):
    assert _span_duration({"start_time": start, "end_time": end}) == pytest.approx(2.0)


def test_span_duration_ignores_unusable_timestamps():
    assert _span_duration({}) == 0.0
    assert _span_duration({"start_time": 5, "end_time": 1}) == 0.0
    assert _span_duration({"start_time": "a", "end_time": "b"}) == 0.0
    assert _span_duration({"start_time": True, "end_time": True}) == 0.0


# ----------------------------------------------------------- ingest_trace_events


def test_ingest_trace_events_from_dicts_records_nodes_calls_and_fan_in(tmp_path):
    events = [
        {"kind": "subagent", "timestamp": 1.0, "trace_id": "t", "span_id": "n1", "actor": "plan"},
        {
            "kind": "model",
            "timestamp": 1.1,
            "trace_id": "t",
            "span_id": "m1",
            "parent_span_id": "n1",
            "model": "claude-sonnet-5",
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            "duration": 0.25,
            "inputs": {"messages": ["hi"]},
            "outputs": {"text": "ok"},
        },
        {
            "kind": "tool",
            "timestamp": 1.2,
            "trace_id": "t",
            "span_id": "k1",
            "parent_span_id": "n1",
            "actor": "search",
            "inputs": {"q": "x"},
        },
        {"kind": "subagent", "timestamp": 1.3, "trace_id": "t", "span_id": "n2", "actor": "fetch"},
        {
            "kind": "subagent",
            "timestamp": 1.4,
            "trace_id": "t",
            "span_id": "n3",
            "parent_span_id": "n1",
            "causal_span_ids": ["n2"],
            "actor": "join",
        },
    ]
    steps = _steps(ingest_trace_events("trace run", events, runs_dir=tmp_path))
    plan, fetch, join = _node(steps, "plan"), _node(steps, "fetch"), _node(steps, "join")
    assert {plan.id, fetch.id} <= set(join.parent_ids)  # parent + causal fan-in
    assert _custom(plan)["span_id"] == "n1"
    model = next(s for s in steps if s.inputs.get("model") == "claude-sonnet-5")
    assert model.usage == {"input": 12, "output": 3}  # provider names normalized
    assert plan.id in model.parent_ids
    tool = next(s for s in steps if s.inputs.get("tool") == "search")
    assert plan.id in tool.parent_ids


def test_ingest_trace_events_orders_children_after_late_parents(tmp_path):
    # Parent recorded with a later timestamp than its child: the topological
    # pass must still open the node before attributing the call to it.
    events = [
        {
            "kind": "model",
            "timestamp": 1.0,
            "span_id": "m1",
            "parent_span_id": "n1",
            "model": "m",
        },
        {"kind": "subagent", "timestamp": 9.0, "span_id": "n1", "actor": "late"},
    ]
    steps = _steps(ingest_trace_events("late parent", events, runs_dir=tmp_path))
    late = _node(steps, "late")
    model = next(s for s in steps if s.inputs.get("model") == "m")
    assert late.id in model.parent_ids


def test_ingest_trace_events_error_kind_fails_the_run(tmp_path):
    events = [
        {"kind": "subagent", "timestamp": 1.0, "span_id": "n1", "actor": "worker"},
        {
            "kind": "error",
            "timestamp": 2.0,
            "span_id": "e1",
            "parent_span_id": "n1",
            "outputs": {"error": "downstream exploded"},
        },
    ]
    run = Run.load(ingest_trace_events("trace err", events, runs_dir=tmp_path))
    assert getattr(run.status, "value", str(run.status)) == "failed"
    assert "downstream exploded" in str(run.to_dict())


def test_ingest_trace_events_tolerates_cyclic_and_unusable_entries(tmp_path):
    events = [
        "not an event",
        None,
        {"kind": "subagent", "timestamp": 1.0, "span_id": "x", "parent_span_id": "y", "actor": "a"},
        {"kind": "subagent", "timestamp": 2.0, "span_id": "y", "parent_span_id": "x", "actor": "b"},
    ]
    run = Run.load(ingest_trace_events("cyclic", events, runs_dir=tmp_path))
    nodes = {s.inputs.get("node") for s in run.graph.steps.values()}
    assert {"a", "b"} <= nodes


@requires_trace
def test_ingest_trace_events_accepts_trace_event_objects(tmp_path):
    events = [
        TraceEvent(kind="subagent", timestamp=1.0, trace_id="t", span_id="n1", actor="planner"),
        TraceEvent(
            kind="model",
            timestamp=1.5,
            trace_id="t",
            span_id="m1",
            parent_span_id="n1",
            model="claude-sonnet-5",
            usage={"input": 7, "output": 2},
            duration=0.5,
        ),
    ]
    steps = _steps(ingest_trace_events("objects", events, runs_dir=tmp_path))
    planner = _node(steps, "planner")
    model = next(s for s in steps if s.inputs.get("model") == "claude-sonnet-5")
    assert planner.id in model.parent_ids
    assert model.usage == {"input": 7, "output": 2}


@requires_trace
def test_ingest_trace_events_consumes_opentine_otel_import(tmp_path):
    from opentine.trace import otel_genai_events

    steps = _steps(
        ingest_trace_events(
            "via opentine importer",
            otel_genai_events(_agent_document()),
            runs_dir=tmp_path,
        )
    )
    # opentine's importer flattens every GenAI span to kind "model"; the
    # gen_ai.operation.name attribute is what still makes the agent a node.
    planner = _node(steps, "planner")
    model = next(s for s in steps if s.inputs.get("model") == "claude-sonnet-5")
    assert planner.id in model.parent_ids
    assert any(s.inputs.get("tool") for s in steps)


# -------------------------------------------------------------- ingest_framework


def _langchain_records() -> list[dict[str, Any]]:
    chain_id, llm_id, tool_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    return [
        {
            "run_id": chain_id,
            "parent_run_id": None,
            "name": "AgentExecutor",
            "type": "chain",
            "start_time": datetime(2026, 7, 30, 12, 0, 0),
            "inputs": {"question": "where to?"},
        },
        {
            "run_id": llm_id,
            "parent_run_id": chain_id,
            "name": "ChatAnthropic",
            "type": "llm",
            "model": "claude-sonnet-5",
            "usage": {"prompt_tokens": 40, "completion_tokens": 9},
            "start_time": datetime(2026, 7, 30, 12, 0, 1),
        },
        {
            "run_id": tool_id,
            "parent_run_id": chain_id,
            "name": "search",
            "type": "tool",
            "inputs": {"q": "flights"},
            "start_time": datetime(2026, 7, 30, 12, 0, 2),
        },
    ]


@requires_importers
def test_framework_events_rejects_a_live_langchain_record():
    # The defect this ingest works around: opentine refuses UUID/datetime
    # payloads, which every real LangChain callback record carries.
    with pytest.raises(ValueError):
        framework_events(_langchain_records()[:1], "langchain")


@requires_importers
def test_ingest_framework_langchain_records_with_uuid_and_datetime(tmp_path):
    steps = _steps(ingest_framework("lc run", _langchain_records(), "langchain", runs_dir=tmp_path))
    chain = _node(steps, "AgentExecutor")
    model = next(s for s in steps if s.inputs.get("model") == "claude-sonnet-5")
    tool = next(s for s in steps if s.inputs.get("tool") == "search")
    assert chain.id in model.parent_ids
    assert chain.id in tool.parent_ids
    assert model.usage == {"input": 40, "output": 9}


@requires_importers
def test_ingest_framework_drops_only_the_records_that_fail(tmp_path):
    records: list[Any] = ["junk", None, 42]
    records.append({"run_id": "bad", "name": "boom", "type": "llm", "cost": float("nan")})
    circular: dict[str, Any] = {"run_id": "loop", "name": "loop", "type": "llm"}
    circular["self"] = circular
    records.append(circular)
    records.extend(_langchain_records())
    steps = _steps(ingest_framework("dirty", records, "langchain", runs_dir=tmp_path))
    assert _node(steps, "AgentExecutor")
    assert any(s.inputs.get("model") == "claude-sonnet-5" for s in steps)
    assert not any(s.inputs.get("node") == "boom" for s in steps)


@requires_importers
@pytest.mark.parametrize(
    "framework", ["langchain", "llamaindex", "autogen", "crewai", "openai-agents"]
)
def test_ingest_framework_supports_every_opentine_dialect(tmp_path, framework):
    records = [
        {
            "id": "p1",
            "id_": "p1",
            "run_id": "p1",
            "span_id": "p1",
            "type": "chain",
            "event_type": "chain",
            "name": "root",
            "sender": "root",
            "agent": "root",
        },
        {
            "id": "c1",
            "id_": "c1",
            "run_id": "c1",
            "span_id": "c1",
            "parent_id": "p1",
            "parent_run_id": "p1",
            "type": "tool",
            "event_type": "tool",
            "name": "search",
            "sender": "search",
            "agent": "search",
        },
    ]
    steps = _steps(ingest_framework(f"{framework} run", records, framework, runs_dir=tmp_path))
    assert any(s.inputs.get("tool") for s in steps)


@requires_importers
def test_ingest_framework_rejects_an_unknown_dialect(tmp_path):
    with pytest.raises(ValueError):
        ingest_framework("nope", [{"id": "a"}], "not-a-framework", runs_dir=tmp_path)


@requires_importers
def test_ingest_framework_records_are_json_round_tripped(tmp_path):
    # Guards the pre-step itself: the record still holds live objects when it
    # reaches ingest_framework, and the run must come out complete anyway.
    records = _langchain_records()
    assert isinstance(records[0]["run_id"], uuid.UUID)
    run = Run.load(ingest_framework("rt", records, "langchain", runs_dir=tmp_path))
    assert getattr(run.status, "value", str(run.status)) == "completed"
    assert isinstance(records[0]["start_time"], datetime)  # inputs are not mutated
    assert json.loads(json.dumps(run.manifest, default=str))["framework"] == "langchain"
