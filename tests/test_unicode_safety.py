"""Unpaired UTF-16 surrogates must never destroy a recorded trace.

opentine 0.3.0 refuses to write an artifact containing a surrogate (no UTF-8
spelling exists) and names the offending field. Any JavaScript or Java
producer emits one for a truncated emoji, and graph state comes from
arbitrary frameworks, so graphforge repairs them at its own boundary.
"""

import json

from opentine import Run

from graphforge import (
    GraphEngine,
    GraphNodeResult,
    GraphRunCapture,
    GraphSpec,
    ingest_jsonl,
    ingest_otel_spans,
)
from graphforge.state import clean_text, safe_json, state_delta, state_digest, to_jsonable

LONE_HIGH = chr(0xD83D)  # what a mid-emoji slice leaves behind
LONE_LOW = chr(0xDC00)
BAD = f"truncated {LONE_HIGH}"


def test_clean_text_replaces_only_surrogates():
    assert clean_text("plain") == "plain"
    assert clean_text("emoji 🎉 ok") == "emoji 🎉 ok"
    cleaned = clean_text(BAD)
    assert LONE_HIGH not in cleaned
    assert cleaned.startswith("truncated ")
    assert clean_text(LONE_LOW).encode("utf-8")  # encodable afterwards


def test_to_jsonable_cleans_values_and_keys():
    payload = to_jsonable({BAD: BAD, "nested": [BAD, {"k": BAD}]})
    encoded = json.dumps(payload)  # would raise on a surrogate
    assert LONE_HIGH not in encoded
    assert safe_json({"x": BAD})["x"].encode("utf-8")
    assert state_digest({BAD: BAD})  # digest path must not raise


def test_state_delta_cleans_key_names():
    delta = state_delta({}, {BAD: 1})
    assert json.dumps(delta)
    assert LONE_HIGH not in json.dumps(delta)


def test_engine_run_with_surrogates_still_saves(tmp_path):
    def node(ctx):
        return GraphNodeResult(
            observation=BAD, update={"text": BAD, BAD: "key too"}, stop=True
        )

    graph = GraphSpec().add_node(f"n{LONE_LOW}", node).set_entry(f"n{LONE_LOW}")
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal=f"goal {BAD}", initial_state={})
    assert result.status == "completed"
    assert result.artifact.exists()
    run = Run.load(result.artifact)
    assert LONE_HIGH not in json.dumps(run.to_dict())
    assert run.manifest["goal"].startswith("goal truncated ")


def test_capture_with_surrogates_everywhere_still_saves(tmp_path):
    capture = GraphRunCapture(
        f"cap {BAD}", framework=f"js{BAD}", runs_dir=tmp_path, strict=True
    )
    capture.node_start(f"n{BAD}", state={"x": BAD})
    capture.model_call(model=f"m{BAD}", prompt=BAD, response=BAD)
    capture.tool_call(f"t{BAD}", args={"a": BAD}, result=BAD)
    capture.node_end(f"n{BAD}", update={"y": BAD})
    capture.edge(f"a{BAD}", f"b{BAD}", label=f"L{BAD}")
    capture.checkpoint(f"c{BAD}", state={"k": BAD}, source=f"s{BAD}")
    artifact = capture.end(summary=f"done {BAD}")
    assert artifact is not None and artifact.exists()
    assert capture.dropped_events == 0
    assert LONE_HIGH not in json.dumps(Run.load(artifact).to_dict())


def test_jsonl_ingest_of_javascript_producer_survives(tmp_path):
    """The realistic path: a JS producer writes a truncated \\udXXX escape."""
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "node_start", "node": "plan"}),
                '{"type": "node_end", "node": "plan", "update": {"emoji": "\\ud83d"}}',
                json.dumps({"type": "end", "summary": "js run"}),
            ]
        )
    )
    artifact = ingest_jsonl("js stream", path, runs_dir=tmp_path)
    assert artifact is not None and artifact.exists()
    assert LONE_HIGH not in json.dumps(Run.load(artifact).to_dict())


def test_otel_ingest_with_surrogate_attributes_survives(tmp_path):
    spans = [
        {
            "span_id": "s1",
            "parent_span_id": None,
            "name": "agent",
            "attributes": {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": f"planner{BAD}",
            },
        },
        {
            "span_id": "s2",
            "parent_span_id": "s1",
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": f"model{BAD}",
                "gen_ai.output.messages": BAD,
            },
        },
    ]
    artifact = ingest_otel_spans("otel surrogate", spans, runs_dir=tmp_path)
    assert artifact is not None and artifact.exists()
    assert LONE_HIGH not in json.dumps(Run.load(artifact).to_dict())


def test_resume_from_surrogate_bearing_run(tmp_path):
    """A cleaned snapshot must still round-trip through resume."""

    def first(ctx):
        return GraphNodeResult(update={"note": BAD}, score=0.1)

    def second(ctx):
        return GraphNodeResult(update={"done": True}, stop=True)

    graph = (
        GraphSpec()
        .add_node("first", first)
        .add_node("second", second)
        .add_edge("first", "second")
        .set_entry("first")
    )
    engine = GraphEngine(graph, runs_dir=tmp_path)
    original = engine.run(goal="surrogate resume", initial_state={})
    step_id = original.path[0]["step_id"]
    resumed = engine.resume(original.artifact, step_id)
    assert resumed.status == "completed"
    assert resumed.artifact.exists()
