import pytest

from graphforge import END, GraphSpec, GraphValidationError
from graphforge.engine import GraphNodeResult


def _noop(ctx):
    return GraphNodeResult()


def test_validate_requires_entry_and_known_edges():
    graph = GraphSpec().add_node("a", _noop)
    with pytest.raises(GraphValidationError, match="entry"):
        graph.validate()
    graph.set_entry("a").add_edge("a", "missing")
    with pytest.raises(GraphValidationError, match="missing"):
        graph.validate()


def test_reserved_and_duplicate_nodes_rejected():
    graph = GraphSpec()
    with pytest.raises(GraphValidationError):
        graph.add_node(END, _noop)
    graph.add_node("a", _noop)
    with pytest.raises(GraphValidationError, match="duplicate"):
        graph.add_node("a", _noop)
    with pytest.raises(GraphValidationError, match="kind"):
        graph.add_node("b", _noop, kind="nonsense")


def test_entry_cannot_be_join():
    graph = GraphSpec().add_node("j", _noop, join=True).set_entry("j")
    with pytest.raises(GraphValidationError, match="join"):
        graph.validate()


def test_topology_digest_stable_and_sensitive():
    def build():
        return (
            GraphSpec(name="g")
            .add_node("a", _noop)
            .add_node("b", _noop)
            .add_edge("a", "b")
            .set_entry("a")
        )

    assert build().topology_digest() == build().topology_digest()
    changed = build().add_edge("b", END)
    assert changed.topology_digest() != build().topology_digest()


def test_describe_and_mermaid():
    graph = (
        GraphSpec(name="demo")
        .add_node("a", _noop, kind="model")
        .add_node("b", _noop, join=True)
        .add_edge("a", "b", label="go")
        .add_edge("b", END)
        .set_entry("a")
    )
    body = graph.describe()
    assert body["entry"] == "a"
    assert {n["id"] for n in body["nodes"]} == {"a", "b"}
    assert body["topology_digest"]
    diagram = graph.mermaid()
    assert "flowchart TD" in diagram
    assert "a -- go --> b" in diagram
    assert "__start__" in diagram
