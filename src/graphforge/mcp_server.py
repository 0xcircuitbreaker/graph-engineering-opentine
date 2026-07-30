"""MCP server exposing graph run artifacts to editors and agents.

Plain functions first (usable and testable without the ``mcp`` package),
FastMCP registration on top. Run refs accept a path, a bare file name, or a
run-id prefix.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from uuid import uuid4

from opentine import Run

from .recorder import DEFAULT_RUNS_DIR
from .state import safe_json

_MAX_LIST = 200
_MAX_STEPS = 25


def _normalize_ref(ref: str, runs_dir: Path) -> Path:
    """Resolve a path | file name | run-id prefix to an artifact path.

    Prefix resolution matches file stems only (graphforge saves artifacts as
    ``{run_id}.tine``), so it never loads artifacts just to resolve a name.
    """
    candidate = Path(ref).expanduser()
    if candidate.is_file():
        return candidate
    named = runs_dir / ref
    if named.is_file():
        return named
    if not ref.endswith(".tine"):
        named = runs_dir / f"{ref}.tine"
        if named.is_file():
            return named
    matches = [path for path in sorted(runs_dir.glob("*.tine")) if path.stem.startswith(ref)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(f"ambiguous run ref {ref!r}: {[p.name for p in matches]}")
    raise FileNotFoundError(f"no run artifact matches {ref!r} under {runs_dir}")


def _run_metadata(run: Run) -> dict[str, Any]:
    manifest = run.manifest if isinstance(run.manifest, dict) else {}
    breakdown = run.cost_breakdown()
    return {
        "run_id": run.run_id,
        "status": getattr(run.status, "value", str(run.status)),
        "goal": manifest.get("goal", ""),
        "kind": manifest.get("kind", ""),
        "step_count": len(run.graph.order),
        "created_at": run.created_at,
        "cost": {
            "total_cost": breakdown.total_cost,
            "total_usage": breakdown.total_tokens,
            "by_model": breakdown.by_model,
            "by_kind": breakdown.by_kind,
        },
        "policies": run.policies,
        "tags": list(run.tags),
    }


def list_runs(runs_dir: str = str(DEFAULT_RUNS_DIR)) -> list[dict[str, Any]]:
    """List graph run artifacts (newest first)."""
    base = Path(runs_dir).expanduser()
    if not base.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(base.glob("*.tine"), key=lambda p: p.stat().st_mtime, reverse=True):
        if len(entries) >= _MAX_LIST:
            break
        try:
            run = Run.load(path)
        except Exception as exc:  # noqa: BLE001
            entries.append({"path": str(path), "error": str(exc)})
            continue
        meta = _run_metadata(run)
        meta["path"] = str(path)
        entries.append(meta)
    return entries


def show_run(
    ref: str, runs_dir: str = str(DEFAULT_RUNS_DIR), *, steps: int = _MAX_STEPS
) -> dict[str, Any]:
    """Show a run's metadata, graph topology, and the path of nodes executed."""
    path = _normalize_ref(ref, Path(runs_dir).expanduser())
    run = Run.load(path)
    manifest = run.manifest if isinstance(run.manifest, dict) else {}
    result = _run_metadata(run)
    result["path"] = str(path)
    result["graph"] = manifest.get("graph", {})
    rendered = []
    for step_id in run.graph.order[:steps]:
        step = run.graph.steps[step_id]
        inputs = step.inputs or {}
        outputs = step.outputs or {}
        rendered.append(
            {
                "id": step.id,
                "kind": getattr(step.kind, "value", str(step.kind)),
                "node": inputs.get("node"),
                "edge": inputs.get("edge"),
                "observation": outputs.get("observation"),
                "score": outputs.get("score"),
                "duration": step.duration,
                "cost": step.cost,
            }
        )
    result["steps"] = rendered
    result["step_count_total"] = len(run.graph.order)
    return result


def show_graph(ref: str, runs_dir: str = str(DEFAULT_RUNS_DIR)) -> dict[str, Any]:
    """Topology + execution trail (edges actually taken) + mermaid rendering."""
    path = _normalize_ref(ref, Path(runs_dir).expanduser())
    run = Run.load(path)
    manifest = run.manifest if isinstance(run.manifest, dict) else {}
    topology = manifest.get("graph", {})
    trail: list[dict[str, Any]] = []
    for step_id in run.graph.order:
        step = run.graph.steps[step_id]
        inputs = step.inputs if isinstance(step.inputs, dict) else {}
        node = inputs.get("node")
        if not node:
            continue
        edge = inputs.get("edge")
        trail.append(
            {
                "node": str(node),
                "edge": edge if isinstance(edge, dict) else None,
                "step_id": step.id,
            }
        )
    lines = ["flowchart TD"]
    seen_nodes: set[str] = set()
    for hop in trail:
        seen_nodes.add(hop["node"])
    for node in sorted(seen_nodes):
        lines.append(f'    {node}["{node}"]')
    for hop in trail:
        edge = hop.get("edge") or {}
        sources = edge.get("sources") or ([edge["source"]] if edge.get("source") else [])
        if not isinstance(sources, list):
            sources = [sources]
        for source in sources:
            label = edge.get("label") or edge.get("kind", "")
            arrow = f"-- {label} -->" if label else "-->"
            lines.append(f"    {source} {arrow} {hop['node']}")
    return {
        "run_id": run.run_id,
        "topology": safe_json(topology),
        "trail": trail,
        "mermaid_trail": "\n".join(lines),
    }


def diff_runs(
    left: str, right: str, runs_dir: str = str(DEFAULT_RUNS_DIR)
) -> dict[str, Any]:
    """Semantic diff between two run artifacts (opentine lineage alignment)."""
    base = Path(runs_dir).expanduser()
    run_a = Run.load(_normalize_ref(left, base))
    run_b = Run.load(_normalize_ref(right, base))
    diff = run_a.diff(run_b)
    return {
        "left": run_a.run_id,
        "right": run_b.run_id,
        "common_ancestor": diff.common_ancestor,
        "only_left": [step.id for step in diff.only_a],
        "only_right": [step.id for step in diff.only_b],
        "changed": [
            {
                "step_a": getattr(change.step_a, "id", None),
                "step_b": getattr(change.step_b, "id", None),
                "fields": [
                    {
                        "name": delta.name,
                        "before": safe_json(delta.before),
                        "after": safe_json(delta.after),
                        "changed_keys": list(delta.changed_keys),
                    }
                    for delta in getattr(change, "fields", [])
                ],
            }
            for change in diff.changed
        ],
    }


def fork_run(
    ref: str,
    from_step: str,
    branch: str = "operator-fork",
    runs_dir: str = str(DEFAULT_RUNS_DIR),
) -> dict[str, Any]:
    """Fork a run artifact at a step (full or prefix step id); saves a new .tine."""
    base = Path(runs_dir).expanduser()
    path = _normalize_ref(ref, base)
    run = Run.load(path)
    matches = [sid for sid in run.graph.order if sid.startswith(from_step)]
    if len(matches) != 1:
        raise ValueError(
            f"step {from_step!r} is {'ambiguous' if matches else 'not found'} in {run.run_id}"
        )
    # Unique run id: opentine's default fork id is deterministic from
    # (parent, fork point), so two forks at the same step would overwrite
    # each other's artifact.
    new_run_id = f"{run.run_id[:12]}-{branch}-{uuid4().hex[:8]}"
    forked = run.fork(from_step_id=matches[0], new_run_id=new_run_id, branch=branch)
    target = base / f"{forked.run_id}.tine"
    saved = forked.save(target)
    return {"run_id": forked.run_id, "path": str(saved), "fork_point": matches[0]}


def build_server(runs_dir: str = str(DEFAULT_RUNS_DIR)):
    """Register the plain functions as MCP tools (mcp 1.x and 2.x)."""
    try:
        from mcp.server import MCPServer as _ServerClass  # mcp >= 2.0
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as _ServerClass  # mcp 1.x
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "MCP support needs the mcp package: "
                'pip install "opentine-graph-engineering[mcp]"'
            ) from exc
    server = _ServerClass("graphforge")

    @server.tool()
    def list_graph_runs() -> list[dict[str, Any]]:
        """List recorded graph runs (newest first)."""
        return list_runs(runs_dir)

    @server.tool()
    def show_graph_run(run: str) -> dict[str, Any]:
        """Show one run: metadata, topology, and executed steps."""
        return show_run(run, runs_dir)

    @server.tool()
    def show_graph_topology(run: str) -> dict[str, Any]:
        """Show a run's graph topology and the trail of edges actually taken."""
        return show_graph(run, runs_dir)

    @server.tool()
    def diff_graph_runs(left: str, right: str) -> dict[str, Any]:
        """Diff two runs by lineage position (drift + divergence)."""
        return diff_runs(left, right, runs_dir)

    @server.tool()
    def fork_graph_run(run: str, from_step: str, branch: str = "mcp-fork") -> dict[str, Any]:
        """Fork a run at a step into a new artifact."""
        return fork_run(run, from_step, branch, runs_dir)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="graphforge MCP server (stdio)")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    args = parser.parse_args()
    server = build_server(runs_dir=args.runs_dir)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
