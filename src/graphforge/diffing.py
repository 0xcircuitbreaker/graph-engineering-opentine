"""Graph-layer diff between two recorded runs.

opentine compares runs at the step layer: ``Run.diff`` aligns by lineage
position, and the v3 ``repo.diff`` aligns by sequence position over events
whose identity includes a timestamp — so two independently recorded but
behaviourally identical graph runs share zero events and everything reads as
changed.

This module compares what actually distinguishes two *graph* runs: which
nodes ran, in what order, over which edges, how many times, and how their
scores moved. It works on any ``Run`` graphforge recorded, whether it came
from a ``.tine`` file or a v3 repository.
"""

from __future__ import annotations

from typing import Any

from .state import sane_score


def _node_steps(run: Any) -> list[dict[str, Any]]:
    """The graph-layer trail: one entry per recorded node execution."""
    trail: list[dict[str, Any]] = []
    for step_id in run.graph.order:
        step = run.graph.steps[step_id]
        inputs = step.inputs if isinstance(step.inputs, dict) else {}
        node = inputs.get("node")
        # A model/tool call also carries "node" (its attribution); only a
        # node execution carries the edge that fired it.
        if not node or "edge" not in inputs:
            continue
        outputs = step.outputs if isinstance(step.outputs, dict) else {}
        edge = inputs.get("edge") if isinstance(inputs.get("edge"), dict) else {}
        sources = edge.get("sources") or ([edge["source"]] if edge.get("source") else [])
        if not isinstance(sources, list):
            sources = [sources]
        trail.append(
            {
                "node": str(node),
                "visit": inputs.get("visit", 0),
                "step_id": step.id,
                "edge_kind": edge.get("kind", ""),
                "edge_label": edge.get("label", ""),
                "sources": [str(s) for s in sources],
                "score": sane_score(outputs.get("score")),
                "state_digest": outputs.get("state_digest"),
                "cost": step.cost,
            }
        )
    return trail


def _edges(trail: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    return {
        (source, hop["node"], str(hop["edge_label"] or hop["edge_kind"]))
        for hop in trail
        for source in hop["sources"]
    }


def _topology(run: Any) -> dict[str, Any]:
    manifest = run.manifest if isinstance(run.manifest, dict) else {}
    graph = manifest.get("graph")
    return graph if isinstance(graph, dict) else {}


def graph_diff(left: Any, right: Any) -> dict[str, Any]:
    """Compare two recorded graph runs at the graph layer.

    Returns the node path of each side, the first point they diverge, edges
    taken by only one side, per-node visit-count deltas, and per-node score
    drift — plus whether the two runs even declare the same topology.
    """
    left_trail, right_trail = _node_steps(left), _node_steps(right)
    left_path = [f"{h['node']}#{h['visit']}" for h in left_trail]
    right_path = [f"{h['node']}#{h['visit']}" for h in right_trail]

    first_divergence: dict[str, Any] | None = None
    for index in range(max(len(left_trail), len(right_trail))):
        lhs = left_trail[index] if index < len(left_trail) else None
        rhs = right_trail[index] if index < len(right_trail) else None
        same = (
            lhs is not None
            and rhs is not None
            and lhs["node"] == rhs["node"]
            and lhs["visit"] == rhs["visit"]
            and lhs["state_digest"] == rhs["state_digest"]
        )
        if not same:
            first_divergence = {
                "index": index,
                "left": None if lhs is None else f"{lhs['node']}#{lhs['visit']}",
                "right": None if rhs is None else f"{rhs['node']}#{rhs['visit']}",
                "reason": (
                    "left ended"
                    if lhs is None
                    else "right ended"
                    if rhs is None
                    else "different node"
                    if lhs["node"] != rhs["node"] or lhs["visit"] != rhs["visit"]
                    else "same node, different state"
                ),
            }
            break

    left_visits: dict[str, int] = {}
    right_visits: dict[str, int] = {}
    for hop in left_trail:
        left_visits[hop["node"]] = left_visits.get(hop["node"], 0) + 1
    for hop in right_trail:
        right_visits[hop["node"]] = right_visits.get(hop["node"], 0) + 1
    visit_delta = {
        node: {"left": left_visits.get(node, 0), "right": right_visits.get(node, 0)}
        for node in sorted(set(left_visits) | set(right_visits))
        if left_visits.get(node, 0) != right_visits.get(node, 0)
    }

    def best_scores(trail: list[dict[str, Any]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for hop in trail:
            if hop["score"] is not None:
                out[hop["node"]] = max(out.get(hop["node"], hop["score"]), hop["score"])
        return out

    left_scores, right_scores = best_scores(left_trail), best_scores(right_trail)
    score_drift = {
        node: {"left": left_scores.get(node), "right": right_scores.get(node)}
        for node in sorted(set(left_scores) | set(right_scores))
        if left_scores.get(node) != right_scores.get(node)
    }

    left_edges, right_edges = _edges(left_trail), _edges(right_trail)
    left_topology, right_topology = _topology(left), _topology(right)
    return {
        "left_run": left.run_id,
        "right_run": right.run_id,
        "identical_path": left_path == right_path,
        "topology_match": (
            left_topology.get("topology_digest") == right_topology.get("topology_digest")
            if left_topology and right_topology
            else None
        ),
        "left_path": left_path,
        "right_path": right_path,
        "first_divergence": first_divergence,
        "edges_only_left": sorted(left_edges - right_edges),
        "edges_only_right": sorted(right_edges - left_edges),
        "visit_delta": visit_delta,
        "score_drift": score_drift,
        "cost": {
            "left": round(sum(h["cost"] for h in left_trail), 8),
            "right": round(sum(h["cost"] for h in right_trail), 8),
        },
    }
