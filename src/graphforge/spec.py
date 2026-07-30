"""Framework-agnostic graph definitions.

A :class:`GraphSpec` declares nodes (pure functions over a context), static
edges, routers (conditional edges as functions of state), and join nodes
(fan-in with a reducer). The spec is what the engine executes and what gets
embedded — with a topology hash — into every recorded artifact, so structure
drift between two runs is diffable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

END = "__end__"

NODE_KINDS = ("think", "model", "tool")

Router = Callable[[dict[str, Any]], "str | Sequence[str] | None"]
JoinReducer = Callable[[list[dict[str, Any]]], dict[str, Any]]


class GraphValidationError(ValueError):
    """Raised when a GraphSpec is structurally invalid."""


@dataclass
class NodeSpec:
    node_id: str
    fn: Callable[..., Any]
    kind: str = "think"
    join: bool = False
    join_reducer: JoinReducer | None = None


@dataclass(frozen=True)
class EdgeSpec:
    source: str
    target: str
    label: str = ""


@dataclass
class GraphSpec:
    """A declared graph: nodes, edges, routers, entry point."""

    name: str = "graph"
    nodes: dict[str, NodeSpec] = field(default_factory=dict)
    edges: list[EdgeSpec] = field(default_factory=list)
    routers: dict[str, Router] = field(default_factory=dict)
    entry: str | None = None

    def add_node(
        self,
        node_id: str,
        fn: Callable[..., Any],
        *,
        kind: str = "think",
        join: bool = False,
        join_reducer: JoinReducer | None = None,
    ) -> GraphSpec:
        if node_id in (END,):
            raise GraphValidationError(f"{END!r} is reserved")
        if node_id in self.nodes:
            raise GraphValidationError(f"duplicate node {node_id!r}")
        if kind not in NODE_KINDS:
            raise GraphValidationError(f"unknown node kind {kind!r} (use one of {NODE_KINDS})")
        self.nodes[node_id] = NodeSpec(
            node_id=node_id, fn=fn, kind=kind, join=join, join_reducer=join_reducer
        )
        return self

    def add_edge(self, source: str, target: str, *, label: str = "") -> GraphSpec:
        self.edges.append(EdgeSpec(source=source, target=target, label=label))
        return self

    def add_router(self, source: str, router: Router) -> GraphSpec:
        if source in self.routers:
            raise GraphValidationError(f"node {source!r} already has a router")
        self.routers[source] = router
        return self

    def set_entry(self, node_id: str) -> GraphSpec:
        self.entry = node_id
        return self

    def outgoing(self, node_id: str) -> list[EdgeSpec]:
        return [e for e in self.edges if e.source == node_id]

    def validate(self) -> None:
        if not self.nodes:
            raise GraphValidationError("graph has no nodes")
        if self.entry is None:
            raise GraphValidationError("graph has no entry node (call set_entry)")
        if self.entry not in self.nodes:
            raise GraphValidationError(f"entry node {self.entry!r} is not defined")
        if self.nodes[self.entry].join:
            raise GraphValidationError("entry node cannot be a join node")
        for edge in self.edges:
            if edge.source not in self.nodes:
                raise GraphValidationError(f"edge source {edge.source!r} is not a node")
            if edge.target != END and edge.target not in self.nodes:
                raise GraphValidationError(f"edge target {edge.target!r} is not a node")
        for source in self.routers:
            if source not in self.nodes:
                raise GraphValidationError(f"router source {source!r} is not a node")

    def describe(self) -> dict[str, Any]:
        """Serializable topology summary embedded into run manifests."""
        body = {
            "name": self.name,
            "entry": self.entry,
            "nodes": [
                {"id": n.node_id, "kind": n.kind, "join": n.join}
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": e.source, "target": e.target, "label": e.label} for e in self.edges
            ],
            "routers": sorted(self.routers),
        }
        body["topology_digest"] = _topology_digest(body)
        return body

    def topology_digest(self) -> str:
        return self.describe()["topology_digest"]

    def mermaid(self) -> str:
        """Mermaid flowchart of the static topology."""
        lines = ["flowchart TD"]
        for node in self.nodes.values():
            shape = ("[[", "]]") if node.join else ("[", "]")
            suffix = f" ({node.kind})" if node.kind != "think" else ""
            lines.append(f'    {node.node_id}{shape[0]}"{node.node_id}{suffix}"{shape[1]}')
        if any(e.target == END for e in self.edges):
            lines.append(f'    {END}(("end"))')
        for edge in self.edges:
            arrow = f"-- {edge.label} -->" if edge.label else "-->"
            lines.append(f"    {edge.source} {arrow} {edge.target}")
        for source in sorted(self.routers):
            lines.append(f"    {source} -. router .-> {source}_route{{route}}")
        if self.entry:
            lines.insert(1, f'    __start__(("start")) --> {self.entry}')
        return "\n".join(lines)


def _topology_digest(body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
