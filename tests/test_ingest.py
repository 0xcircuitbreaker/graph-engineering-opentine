import json

from opentine import Run

from graphforge import ingest_jsonl, ingest_otel_spans


def _span(span_id, parent, op, name="", start=0, end=0, attrs=None, status=None):
    attributes = {"gen_ai.operation.name": op}
    attributes.update(attrs or {})
    return {
        "span_id": span_id,
        "parent_span_id": parent,
        "name": name or op,
        "start_time": start,
        "end_time": end,
        "attributes": attributes,
        "status": status or {},
    }


def test_ingest_otel_spans_maps_agents_models_tools(tmp_path):
    spans = [
        _span(
            "s1",
            None,
            "invoke_agent",
            start=1,
            end=10,
            attrs={"gen_ai.agent.name": "planner"},
        ),
        _span(
            "s2",
            "s1",
            "chat",
            start=2,
            end=5,
            attrs={
                "gen_ai.request.model": "claude-sonnet-5",
                "gen_ai.usage.input_tokens": 900,
                "gen_ai.usage.output_tokens": 100,
            },
        ),
        _span(
            "s3",
            "s1",
            "execute_tool",
            start=6,
            end=7,
            attrs={"gen_ai.tool.name": "search", "gen_ai.tool.call.arguments": '{"q": "x"}'},
        ),
        _span(
            "s4",
            None,
            "invoke_agent",
            start=11,
            end=20,
            attrs={"gen_ai.agent.name": "executor"},
        ),
    ]
    artifact = ingest_otel_spans("otel run", spans, runs_dir=tmp_path)
    run = Run.load(artifact)
    steps = list(run.graph.steps.values())
    planner = next(s for s in steps if s.inputs.get("node") == "planner" and "edge" in s.inputs)
    executor = next(s for s in steps if s.inputs.get("node") == "executor" and "edge" in s.inputs)
    assert executor.inputs["edge"]["source"] == "planner"  # sequence edge
    model = next(s for s in steps if s.inputs.get("model") == "claude-sonnet-5")
    assert model.usage == {"input": 900, "output": 100}
    assert planner.id in model.parent_ids
    tool = next(s for s in steps if s.inputs.get("tool") == "search")
    assert planner.id in tool.parent_ids
    assert getattr(run.status, "value", str(run.status)) == "completed"


def test_ingest_otel_error_status_fails_run(tmp_path):
    spans = [
        _span(
            "s1",
            None,
            "invoke_agent",
            attrs={"gen_ai.agent.name": "worker"},
            status={"status_code": "ERROR", "description": "agent exploded"},
        )
    ]
    artifact = ingest_otel_spans("otel err", spans, runs_dir=tmp_path)
    run = Run.load(artifact)
    assert getattr(run.status, "value", str(run.status)) == "failed"
    assert "agent exploded" in str(run.to_dict())


def test_ingest_jsonl_full_stream(tmp_path):
    events = [
        {"type": "begin", "graph": {"nodes": ["plan", "act"]}, "state": {"q": "hi"}},
        {"type": "node_start", "node": "plan", "state": {"q": "hi"}},
        {
            "type": "model_call",
            "model": "m1",
            "prompt": "p",
            "response": "r",
            "usage": {"input": 5, "output": 2},
            "node": "plan",
        },
        {"type": "node_end", "node": "plan", "update": {"plan": "steps"}, "kind": "model"},
        {"type": "edge", "source": "plan", "target": "act", "label": "next"},
        {"type": "node_start", "node": "act"},
        {"type": "node_end", "node": "act", "update": {"done": True}, "score": 0.9},
        {"type": "checkpoint", "id": "c1", "state": {"done": True}, "source": "act"},
        {"type": "end", "status": "completed", "summary": "stream done"},
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events))
    artifact = ingest_jsonl("jsonl run", path, runs_dir=tmp_path)
    run = Run.load(artifact)
    steps = list(run.graph.steps.values())
    act = next(s for s in steps if s.inputs.get("node") == "act" and "edge" in s.inputs)
    assert act.inputs["edge"] == {"kind": "event", "source": "plan", "label": "next"}
    assert act.outputs["score"] == 0.9
    plan = next(s for s in steps if s.inputs.get("node") == "plan" and "edge" in s.inputs)
    assert plan.id in act.parent_ids
    model = next(s for s in steps if s.inputs.get("model") == "m1")
    assert plan.id in model.parent_ids
    assert any(s.inputs.get("checkpoint") == "c1" for s in steps)


def test_ingest_jsonl_tolerates_garbage_and_missing_end(tmp_path):
    lines = [
        "not json at all",
        '{"type": "mystery_event"}',
        '{"type": "node_start", "node": "only"}',
        '{"type": "node_end", "node": "only", "update": {"ok": 1}}',
        "[1, 2, 3]",
    ]
    artifact = ingest_jsonl("dirty stream", lines, runs_dir=tmp_path)
    run = Run.load(artifact)
    assert any(s.inputs.get("node") == "only" for s in run.graph.steps.values())
    assert getattr(run.status, "value", str(run.status)) == "completed"
