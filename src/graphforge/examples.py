"""Deterministic demo graphs — fully offline, used by the CLI and tests."""

from __future__ import annotations

from typing import Any

from .engine import GraphNodeContext, GraphNodeResult
from .spec import END, GraphSpec


def refinement_graph(target: float = 42.0, *, tolerance: float = 0.5) -> GraphSpec:
    """Cyclic propose -> evaluate graph converging a number toward a target.

    Demonstrates: routers (conditional edges), cycles bounded by the policy
    cycle guard, per-node scores, and stop-on-threshold.
    """

    def propose(ctx: GraphNodeContext) -> GraphNodeResult:
        x = float(ctx.state.get("x", 0.0))
        step = (target - x) / 2 or tolerance
        candidate = x + step
        return GraphNodeResult(
            observation=f"propose x={candidate:.4f} (from {x:.4f})",
            update={"x": candidate},
        )

    def evaluate(ctx: GraphNodeContext) -> GraphNodeResult:
        x = float(ctx.state.get("x", 0.0))
        distance = abs(target - x)
        score = 1.0 / (1.0 + distance)
        return GraphNodeResult(
            observation=f"distance={distance:.4f} score={score:.4f}",
            update={"distance": distance, "last_score": score},
            score=score,
        )

    def route(state: dict[str, Any]) -> str:
        return END if float(state.get("distance", float("inf"))) <= tolerance else "propose"

    return (
        GraphSpec(name=f"refine-to-{target:g}")
        .add_node("propose", propose)
        .add_node("evaluate", evaluate)
        .add_edge("propose", "evaluate")
        .add_router("evaluate", route)
        .set_entry("propose")
    )


def fanout_graph() -> GraphSpec:
    """plan -> [research_a, research_b] -> merge (join) -> summarize.

    Demonstrates: static fan-out, join nodes with a reducer, multi-parent
    step lineage in the recorded artifact.
    """

    def plan(ctx: GraphNodeContext) -> GraphNodeResult:
        return GraphNodeResult(
            observation="split the question into two research tracks",
            update={"question": ctx.state.get("question", ctx.goal), "notes": []},
        )

    def research(track: str, finding: str):
        def fn(ctx: GraphNodeContext) -> GraphNodeResult:
            return GraphNodeResult(
                observation=f"track {track} found: {finding}",
                update={f"notes_{track}": [finding]},
            )

        return fn

    def merge_reducer(states: list[dict[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        notes: list[str] = []
        for state in states:
            merged.update(state)
            for key, value in state.items():
                if key.startswith("notes_") and isinstance(value, list):
                    notes.extend(value)
        merged["notes"] = sorted(set(notes))
        return merged

    def merge(ctx: GraphNodeContext) -> GraphNodeResult:
        return GraphNodeResult(
            observation=f"merged {len(ctx.state.get('notes', []))} notes",
        )

    def summarize(ctx: GraphNodeContext) -> GraphNodeResult:
        notes = ctx.state.get("notes", [])
        summary = "; ".join(str(n) for n in notes) or "nothing found"
        return GraphNodeResult(
            observation=f"summary: {summary}",
            update={"summary": summary},
            score=1.0 if notes else 0.0,
            stop=True,
        )

    return (
        GraphSpec(name="fanout-research")
        .add_node("plan", plan)
        .add_node("research_a", research("a", "graphs declare architecture"))
        .add_node("research_b", research("b", "loops defer architecture"))
        .add_node("merge", merge, join=True, join_reducer=merge_reducer)
        .add_node("summarize", summarize)
        .add_edge("plan", "research_a")
        .add_edge("plan", "research_b")
        .add_edge("research_a", "merge")
        .add_edge("research_b", "merge")
        .add_edge("merge", "summarize")
        .add_edge("summarize", END)
        .set_entry("plan")
    )
