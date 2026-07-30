import pytest

from graphforge import (
    END,
    GraphEngine,
    GraphSpec,
    OpenAICompatAdapter,
    StaticModelAdapter,
    build_json_model_node,
    extract_json,
)


def test_extract_json_handles_json_fence():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_handles_prose_plus_fence():
    text = 'Here you go:\n```json\n{"observation": "ok", "score": 0.5}\n```\nDone.'
    assert extract_json(text)["score"] == 0.5


def test_extract_json_handles_bare_fence_and_raw():
    assert extract_json('```\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('{"a": 3}') == {"a": 3}
    assert extract_json('prefix {"a": 4} suffix') == {"a": 4}


def test_extract_json_rejects_garbage():
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json("")
    with pytest.raises(ValueError):
        extract_json("[1, 2, 3]")  # list, not an object


def test_static_adapter_script_sequences():
    adapter = StaticModelAdapter(script=["one", "two"])
    assert adapter.complete("p").text == "one"
    assert adapter.complete("p").text == "two"
    assert adapter.complete("p").text == "two"  # sticks at last


def test_openai_adapter_constructs_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = OpenAICompatAdapter(model="m")  # must not raise
    with pytest.raises(RuntimeError, match="API key"):
        adapter.complete("hello")


def test_json_model_node_end_to_end(tmp_path):
    adapter = StaticModelAdapter(
        text='```json\n{"observation": "drafted", "update": {"draft": "v1"}, '
        '"score": 0.9, "stop": true}\n```',
        cost=0.01,
    )
    node = build_json_model_node(adapter=adapter, prompt_fn=lambda ctx: ctx.goal)
    graph = (
        GraphSpec().add_node("gen", node, kind="model").add_edge("gen", END).set_entry("gen")
    )
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal="draft it", initial_state={})
    assert result.status == "completed"
    assert result.final_state["draft"] == "v1"
    assert result.best.score == 0.9


def test_json_model_node_fallback_on_parse_failure(tmp_path):
    node = build_json_model_node(
        adapter=StaticModelAdapter(text="not json at all"),
        prompt_fn=lambda ctx: "p",
    )
    graph = GraphSpec().add_node("gen", node, kind="model").set_entry("gen")
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal="fallback", initial_state={})
    assert result.status == "completed"
    assert result.stop_reason == "stop"
    assert result.path[0]["node"] == "gen"


def test_json_model_node_ignores_bool_score(tmp_path):
    node = build_json_model_node(
        adapter=StaticModelAdapter(text='{"observation": "x", "score": true, "stop": true}'),
        prompt_fn=lambda ctx: "p",
    )
    graph = GraphSpec().add_node("gen", node, kind="model").set_entry("gen")
    result = GraphEngine(graph, runs_dir=tmp_path).run(goal="bool", initial_state={})
    assert result.best is None  # bool score must not count as numeric
