import re

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
    # Piped label form: `-- go -->` would let a label containing `---`
    # terminate the link and declare a node that never ran.
    assert 'a -->|"go"| b' in diagram
    assert "__start__" in diagram


def test_mermaid_label_cannot_terminate_a_link():
    graph = (
        GraphSpec()
        .add_node("a", _noop)
        .add_node("b", _noop)
        .add_edge("a", "b", label='ok --- injected["x"] |y|')
        .set_entry("a")
    )
    diagram = graph.mermaid()
    assert "injected" not in _identifiers(diagram)
    assert "#quot;" in diagram and "#124;" in diagram


def test_mermaid_slugifies_hostile_node_ids():
    hostile = 'evil"] --> owned["pwned'
    graph = (
        GraphSpec()
        .add_node(hostile, _noop)
        .add_node("end", _noop)  # a mermaid reserved word
        .add_edge(hostile, "end")
        .set_entry(hostile)
    )
    diagram = graph.mermaid()
    idents = _identifiers(diagram)
    assert "owned" not in idents
    # bare `end` as an identifier aborts the whole flowchart
    assert "end" not in idents
    assert all(ident.replace("_", "").isalnum() or not ident for ident in idents)


def _identifiers(diagram: str) -> set[str]:
    """Every token in an identifier position (outside quotes)."""
    found: set[str] = set()
    for line in diagram.splitlines()[1:]:
        outside = re.sub(r'"[^"]*"', "", line)
        found.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", outside))
    return found
