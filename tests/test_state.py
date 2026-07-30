from graphforge.state import (
    safe_json,
    sane_metric,
    sane_usage,
    state_delta,
    state_digest,
    to_jsonable,
)


def test_to_jsonable_handles_weird_values():
    class Thing:
        def __repr__(self):
            return "<Thing>"

    payload = to_jsonable({"a": {1, 2}, "b": Thing(), "c": float("inf"), 4: "x"})
    assert payload["b"] == "<Thing>"
    assert payload["c"] == "inf"
    assert payload["4"] == "x"
    assert sorted(payload["a"]) == [1, 2]


def test_safe_json_truncates():
    long = "x" * 10_000
    out = safe_json({"text": long, "items": list(range(500))})
    assert len(out["text"]) < 5_000
    assert "truncated" in out["text"]
    assert len(out["items"]) == 101  # 100 items + truncation note


def test_state_digest_deterministic_and_order_insensitive():
    assert state_digest({"a": 1, "b": 2}) == state_digest({"b": 2, "a": 1})
    assert state_digest({"a": 1}) != state_digest({"a": 2})


def test_state_delta():
    delta = state_delta({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 9, "d": 4})
    assert delta == {"added": ["d"], "removed": ["c"], "changed": ["b"]}
    assert state_delta(None, None) == {"added": [], "removed": [], "changed": []}


def test_sane_metric():
    assert sane_metric(1.5) == 1.5
    assert sane_metric(-1) == 0.0
    assert sane_metric(float("nan")) == 0.0
    assert sane_metric(float("inf")) == 0.0
    assert sane_metric(True) == 0.0
    assert sane_metric("free") == 0.0


def test_sane_usage():
    messy = {"input": 3.9, "output": 2, "flag": True, "neg": -1, "inf": float("inf")}
    assert sane_usage(messy) == {"input": 3, "output": 2}
    assert sane_usage("nope") is None
    assert sane_usage({"flag": True}) is None
