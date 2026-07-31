"""MCP server exposing graph run artifacts to editors and agents.

Plain functions first (usable and testable without the ``mcp`` package),
server registration on top. Run refs accept a path, a bare file name, or a
run-id prefix -- always resolved *inside* the runs directory.

Everything a tool reads here is attacker-reachable: a run ref comes from a
model, and node ids/labels can come from an ingested third-party trace. So the
read path is confined (no escaping ``runs_dir``), capped (no unbounded
``Run.load`` and no response over ``_MAX_TEXT_CHARS``, whichever field is
oversized), and mermaid is emitted with slugified identifiers plus label text
that only ever appears inside a quoted position, entity-escaped.

The v3 repository tools are *not* reimplemented: when a repository is
available, opentine's own ``register_repository_tools`` is composed onto the
same server, and graphforge adds only what opentine has no concept of -- the
graph layer (topology, edges taken, visit counts).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, is_dataclass
from itertools import islice
from pathlib import Path
from typing import Any
from uuid import uuid4

from opentine import Run

from . import repo_backend
from .diffing import graph_diff
from .mermaid import mermaid_ids as _mermaid_ids
from .mermaid import mermaid_label as _mermaid_label
from .recorder import DEFAULT_RUNS_DIR
from .state import safe_json, slug

_LOG = logging.getLogger(__name__)

# Resource caps mirroring opentine's own MCP surface (see
# opentine/mcp_server.py: MAX_MCP_RUN_BYTES, MAX_MCP_SCAN_RUNS,
# MAX_MCP_LIST_RUNS, MAX_MCP_SUMMARY_BYTES, MAX_MCP_RENDER_STEPS,
# MAX_MCP_TEXT_CHARS). They are re-declared rather than imported because
# opentine 0.2.0 is a supported floor and its module layout may differ.
_MAX_LIST = 200
_MAX_STEPS = 25
#: Refuse a single artifact larger than this *before* loading it.
_MAX_RUN_BYTES = 256 * 1024 * 1024
#: Stop globbing a runs dir after this many artifacts (prefix lookup + list).
_MAX_SCAN_RUNS = 5_000
#: Aggregate bytes ``list_runs`` will parse in one call.
_MAX_SUMMARY_BYTES = 64 * 1024 * 1024
#: Hard ceiling on rendered steps/hops, whatever the caller asks for.
_MAX_RENDER_STEPS = 1_000
#: Ceiling on the JSON/text size of a single tool response.
_MAX_TEXT_CHARS = 256_000
#: The mermaid string ships *inside* a response, so it gets a fraction of the
#: response cap -- at the full cap it alone would blow the payload budget.
_MAX_MERMAID_CHARS = _MAX_TEXT_CHARS // 2
#: Degradation limits, applied only once a response is over the cap.
_MAX_FIELD_CHARS = 2_000
_MAX_MAP_ENTRIES = 50
_MAX_KEY_CHARS = 120

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _confined(root: Path, candidate: Path) -> Path | None:
    """Resolve ``candidate`` and accept it only if it stays under ``root``.

    ``resolve(strict=True)`` follows symlinks *before* the containment test,
    so neither ``../../etc/passwd`` nor a symlink planted inside the runs dir
    can smuggle a file out of the sandbox. The ``.tine`` suffix requirement
    keeps ``Run.load`` off arbitrary files that happen to sit in the dir.
    """
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError, RuntimeError):
        # OSError: missing/loop; ValueError: outside root or embedded null.
        return None
    if not resolved.is_file() or resolved.suffix != ".tine":
        return None
    return resolved


def _normalize_ref(ref: str, runs_dir: Path) -> Path:
    """Resolve a path | file name | run-id prefix to a confined artifact path.

    Relative refs resolve against ``runs_dir``, never the process cwd, and an
    absolute ref is accepted only when it lands inside ``runs_dir`` -- a
    model-supplied ref must not be able to name a file the operator did not
    put in the runs directory.

    Prefix resolution matches file stems only (graphforge saves artifacts as
    ``{run_id}.tine``), so it never loads artifacts just to resolve a name.
    """
    root = Path(runs_dir).expanduser().resolve()
    text = str(ref)
    direct = Path(text).expanduser()
    if not direct.is_absolute():
        direct = root / direct
    for candidate in (direct, root / f"{text}.tine"):
        found = _confined(root, candidate)
        if found is not None:
            return found
    scanned = list(islice(root.glob("*.tine"), _MAX_SCAN_RUNS + 1))
    if len(scanned) > _MAX_SCAN_RUNS:
        raise ValueError(
            f"more than {_MAX_SCAN_RUNS} artifacts under {root}; use an exact run id"
        )
    matches = [
        found
        for path in sorted(scanned)
        if path.stem.startswith(text) and (found := _confined(root, path)) is not None
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(f"ambiguous run ref {ref!r}: {[p.name for p in matches]}")
    raise FileNotFoundError(f"no run artifact matches {ref!r} under {root}")


def _load_run(path: Path) -> Run:
    """Load an artifact after a size check.

    ``Run.load`` reads and parses the whole file into memory, so the cap has
    to be applied to the size on disk *before* the load rather than after.
    """
    size = path.stat().st_size
    if size > _MAX_RUN_BYTES:
        raise ValueError(
            f"run artifact {path.name} is {size} bytes; this MCP read path caps "
            f"artifacts at {_MAX_RUN_BYTES} bytes"
        )
    return Run.load(path)


def _payload_chars(payload: Any) -> int:
    try:
        return len(json.dumps(payload, default=str))
    except (TypeError, ValueError):  # pragma: no cover - safe_json runs first
        return len(repr(payload))


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _top_entries(mapping: dict[str, Any]) -> dict[str, Any]:
    """Largest ``_MAX_MAP_ENTRIES`` of a cost map, plus a count of the rest.

    ``by_model``/``by_kind`` keys come from recorded steps, i.e. from an
    ingested trace, so both their number and their length are attacker-set.
    """
    items = sorted(mapping.items(), key=lambda kv: (-_as_float(kv[1]), str(kv[0])))
    kept: dict[str, Any] = {
        str(key)[:_MAX_KEY_CHARS]: value for key, value in items[:_MAX_MAP_ENTRIES]
    }
    if len(items) > _MAX_MAP_ENTRIES:
        kept["__truncated__"] = len(items) - _MAX_MAP_ENTRIES
    return kept


def _degrade_metadata(payload: dict[str, Any]) -> None:
    """Shrink the run-controlled metadata fields, in place.

    ``_run_metadata`` embeds artifact content that no step-trimming can
    reach: a goal, tags, policies and per-model/per-kind cost maps. A run with
    a one-megabyte goal and three steps would otherwise sail past the cap
    while the response claimed to be truncated.
    """
    payload["truncated"] = True
    for key in ("goal", "kind", "status", "created_at", "error"):
        value = payload.get(key)
        if isinstance(value, str) and len(value) > _MAX_FIELD_CHARS:
            payload[key] = value[:_MAX_FIELD_CHARS] + "..."
    tags = payload.get("tags")
    if isinstance(tags, list):
        payload["tags"] = [str(tag)[:_MAX_KEY_CHARS] for tag in tags[:_MAX_MAP_ENTRIES]]
    cost = payload.get("cost")
    if isinstance(cost, dict):
        for key in ("by_model", "by_kind"):
            mapping = cost.get(key)
            if isinstance(mapping, dict):
                cost[key] = _top_entries(mapping)
    policies = payload.get("policies")
    if policies is not None and _payload_chars(policies) > _MAX_FIELD_CHARS:
        payload["policies"] = {"truncated": True}


def _minimal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Identity-only response, guaranteed small.

    The backstop for "the oversize is in a field nobody thought to shrink":
    better to return the run id and say so than to return megabytes flagged
    ``truncated``. Every kept value is clipped, since a foreign artifact's
    run id is itself untrusted.
    """
    minimal: dict[str, Any] = {}
    for key in ("run_id", "path", "status", "step_count", "step_count_total", "left", "right"):
        value = payload.get(key)
        if isinstance(value, str):
            value = value[:_MAX_KEY_CHARS]
        if value is not None:
            minimal[key] = value
    minimal["truncated"] = True
    minimal["error"] = f"response exceeded {_MAX_TEXT_CHARS} chars and was reduced"
    return minimal


def _bounded_payload(payload: dict[str, Any], *, trim: tuple[str, ...]) -> dict[str, Any]:
    """Guarantee a response fits the output cap.

    An MCP response is model context: a 4000-step run would otherwise return
    megabytes of JSON. Named lists are halved first (cheapest, most useful),
    then run-controlled metadata is degraded, and a payload that still does
    not fit is replaced outright -- a ``truncated`` flag on an oversized
    response is a lie, and this is the read path a model drives.
    """
    if _payload_chars(payload) <= _MAX_TEXT_CHARS:
        return payload
    # Stamp the flag before measuring further: it is part of the response.
    payload["truncated"] = True
    for key in trim:
        value = payload.get(key)
        if isinstance(value, list):
            while value and _payload_chars(payload) > _MAX_TEXT_CHARS:
                value = value[: len(value) // 2]
                payload[key] = value
        elif isinstance(value, str):
            payload[key] = value[:_MAX_FIELD_CHARS] + "..."
        elif value is not None:
            payload[key] = {"truncated": True}
        if _payload_chars(payload) <= _MAX_TEXT_CHARS:
            return payload
    _degrade_metadata(payload)
    if _payload_chars(payload) <= _MAX_TEXT_CHARS:
        return payload
    return _minimal_payload(payload)


def _bounded_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cap a listing's own size, not just the bytes parsed to build it."""
    if _payload_chars(entries) <= _MAX_TEXT_CHARS:
        return entries
    marker = {"path": None, "error": f"truncated: response cap {_MAX_TEXT_CHARS} chars"}
    budget = _MAX_TEXT_CHARS - _payload_chars(marker) - 2
    for entry in entries:
        _degrade_metadata(entry)
    if any(_payload_chars(entry) > budget for entry in entries):
        entries = [_minimal_payload(entry) for entry in entries]
    while entries and _payload_chars(entries) > budget:
        entries = entries[: len(entries) // 2]
    entries.append(marker)
    return entries


def _bounded_text(lines: list[str]) -> str:
    rendered = "\n".join(lines)
    marker = "\n... (truncated)"
    if len(rendered) <= _MAX_MERMAID_CHARS:
        return rendered
    return rendered[: _MAX_MERMAID_CHARS - len(marker)] + marker


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
    """List graph run artifacts (newest first), bounded by count and bytes."""
    base = Path(runs_dir).expanduser().resolve()
    if not base.is_dir():
        return []
    scanned = list(islice(base.glob("*.tine"), _MAX_SCAN_RUNS + 1))
    truncated = len(scanned) > _MAX_SCAN_RUNS
    # Confine here too: a symlink dropped into the runs dir is a read of
    # whatever it points at, and .glob() happily returns it.
    paths = [found for path in scanned[:_MAX_SCAN_RUNS] if (found := _confined(base, path))]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    truncated |= len(paths) > _MAX_LIST
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for path in paths[:_MAX_LIST]:
        size = path.stat().st_size
        if total_bytes + size > _MAX_SUMMARY_BYTES:
            truncated = True
            break
        total_bytes += size
        try:
            run = _load_run(path)
        except Exception as exc:  # noqa: BLE001 - one bad artifact must not hide the rest
            entries.append({"path": str(path), "error": str(exc)})
            continue
        meta = _run_metadata(run)
        meta["path"] = str(path)
        entries.append(meta)
    if truncated:
        entries.append({"path": None, "error": f"truncated: limits {_MAX_LIST} runs"})
    # The byte budget above bounds what was *parsed*; this bounds what is
    # returned. 200 runs each carrying a large goal blows the response cap
    # without ever approaching the parse cap.
    return _bounded_entries(entries)


def show_run(
    ref: str, runs_dir: str = str(DEFAULT_RUNS_DIR), *, steps: int = _MAX_STEPS
) -> dict[str, Any]:
    """Show a run's metadata, graph topology, and the path of nodes executed."""
    path = _normalize_ref(ref, Path(runs_dir).expanduser())
    run = _load_run(path)
    manifest = run.manifest if isinstance(run.manifest, dict) else {}
    result = _run_metadata(run)
    result["path"] = str(path)
    result["graph"] = safe_json(manifest.get("graph", {}))
    # A caller-supplied step count is model input; clamp it to the render cap.
    limit = max(0, min(int(steps), _MAX_RENDER_STEPS))
    rendered = []
    for step_id in run.graph.order[:limit]:
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
    return _bounded_payload(result, trim=("steps", "graph"))


def show_graph(ref: str, runs_dir: str = str(DEFAULT_RUNS_DIR)) -> dict[str, Any]:
    """Topology + execution trail (edges actually taken) + mermaid rendering."""
    path = _normalize_ref(ref, Path(runs_dir).expanduser())
    run = _load_run(path)
    manifest = run.manifest if isinstance(run.manifest, dict) else {}
    topology = manifest.get("graph", {})
    trail: list[dict[str, Any]] = []
    for step_id in run.graph.order:
        if len(trail) >= _MAX_RENDER_STEPS:
            break
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

    # Dedupe edges by (source, target, label): a cyclic graph takes the same
    # edge once per iteration, and seven identical "propose -- static -->
    # evaluate" lines say nothing more than one does. dict preserves the
    # order the edges were first taken in, which is the readable order.
    edge_keys: dict[tuple[str, str, str], None] = {}
    for hop in trail:
        edge = hop.get("edge") or {}
        sources = edge.get("sources") or ([edge["source"]] if edge.get("source") else [])
        if not isinstance(sources, list):
            sources = [sources]
        label = _mermaid_label(edge.get("label") or edge.get("kind", ""))
        for source in sources:
            edge_keys.setdefault((str(source), hop["node"], label), None)

    nodes = sorted({hop["node"] for hop in trail} | {key[0] for key in edge_keys})
    ids = _mermaid_ids(nodes)
    lines = ["flowchart TD"]
    lines.extend(f'    {ids[node]}["{_mermaid_label(node)}"]' for node in nodes)
    for source, target, label in edge_keys:
        # Piped link text (``-->|"label"|``), never ``-- label -->``: mermaid
        # has no link terminator inside |"..."|, whereas an edge label of
        # "a --- b" in the bare form closes the link early and declares a
        # node "b" that never ran.
        arrow = f'-->|"{label}"|' if label else "-->"
        lines.append(f"    {ids[source]} {arrow} {ids[target]}")

    result = {
        "run_id": run.run_id,
        "topology": safe_json(topology),
        "trail": trail,
        "mermaid_trail": _bounded_text(lines),
    }
    return _bounded_payload(result, trim=("trail", "topology", "mermaid_trail"))


def diff_runs(left: str, right: str, runs_dir: str = str(DEFAULT_RUNS_DIR)) -> dict[str, Any]:
    """Semantic diff between two run artifacts (opentine lineage alignment)."""
    base = Path(runs_dir).expanduser()
    run_a = _load_run(_normalize_ref(left, base))
    run_b = _load_run(_normalize_ref(right, base))
    diff = run_a.diff(run_b)
    result = {
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
    return _bounded_payload(result, trim=("changed", "only_left", "only_right"))


def graph_diff_runs(
    left: str, right: str, runs_dir: str = str(DEFAULT_RUNS_DIR)
) -> dict[str, Any]:
    """Graph-layer diff: node path, first divergence, edges, visits, score drift.

    Complements :func:`diff_runs`: opentine aligns runs by lineage position
    over steps, which reports two behaviourally identical graph runs as
    entirely different. This one compares the graph.
    """
    base = Path(runs_dir).expanduser()
    run_a = _load_run(_normalize_ref(left, base))
    run_b = _load_run(_normalize_ref(right, base))
    result = safe_json(graph_diff(run_a, run_b))
    return _bounded_payload(result, trim=("left_path", "right_path", "edges_only_left"))


def fork_run(
    ref: str,
    from_step: str,
    branch: str = "operator-fork",
    runs_dir: str = str(DEFAULT_RUNS_DIR),
) -> dict[str, Any]:
    """Fork a run artifact at a step (full or prefix step id); saves a new .tine."""
    base = Path(runs_dir).expanduser().resolve()
    path = _normalize_ref(ref, base)
    run = _load_run(path)
    matches = [sid for sid in run.graph.order if sid.startswith(from_step)]
    if len(matches) != 1:
        raise ValueError(
            f"step {from_step!r} is {'ambiguous' if matches else 'not found'} in {run.run_id}"
        )
    # Unique run id: opentine's default fork id is deterministic from
    # (parent, fork point), so two forks at the same step would overwrite
    # each other's artifact.
    new_run_id = f"{run.run_id[:12]}-{slug(branch)}-{uuid4().hex[:8]}"
    forked = run.fork(from_step_id=matches[0], new_run_id=new_run_id, branch=branch)
    # Write into the runs dir itself (never path.parent): the source may sit
    # in a subdirectory, and list_runs only globs the top level.
    target = base / f"{forked.run_id}.tine"
    saved = forked.save(target)
    return {"run_id": forked.run_id, "path": str(saved), "fork_point": matches[0]}


def repo_summary(repo_path: str = ".") -> dict[str, Any]:
    """Refs, object count, shallow boundary and fsck verdict for a v3 repo."""
    return safe_json(repo_backend.repo_status(repo_path))


def repo_candidates(
    repo_path: str = ".", min_score: float | None = None, limit: int = 20
) -> list[dict[str, Any]]:
    """Evaluated candidate runs in a v3 repository (needs scored attestations)."""
    results = repo_backend.candidates(
        repo_path, min_score=min_score, limit=max(1, min(int(limit), _MAX_LIST))
    )
    return [
        safe_json(asdict(result) if is_dataclass(result) else result) for result in results
    ]


def _server_class() -> Any:
    """FastMCP moved to ``mcp.server.MCPServer`` in mcp 2.0; support both."""
    try:
        from mcp.server import MCPServer as server_class  # mcp >= 2.0
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as server_class  # mcp 1.x
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "MCP support needs the mcp package: "
                'pip install "opentine-graph-engineering[mcp]"'
            ) from exc
    return server_class


def _register_repo_tools(server: Any, repo_path: str) -> bool:
    """Register graphforge's v3 repository tools; False when unavailable."""
    if not repo_backend.repo_available():
        _LOG.info("repo_path=%s ignored: v3 repositories need opentine>=0.3.0", repo_path)
        return False
    try:
        # Probe once at build time so a bad --repo is a startup log line
        # rather than a tool that fails on every call. The handle is not
        # kept: each tool re-opens, which keeps the server usable after a
        # repository is repacked or replaced underneath it.
        repo_backend.open_repo(repo_path)
    except FileNotFoundError as exc:
        _LOG.warning("no .tine repository at %s (%s); repository tools disabled", repo_path, exc)
        return False

    @server.tool()
    def repo_status() -> dict[str, Any]:
        """Refs, object count, shallow boundary and fsck verdict of the repository."""
        return repo_summary(repo_path)

    @server.tool()
    def list_repo_candidates(
        min_score: float | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """List evaluated candidate runs in the repository, best score first."""
        return repo_candidates(repo_path, min_score=min_score, limit=limit)

    return True


def _register_opentine_repository_tools(server: Any, repo_path: str) -> bool:
    """Compose opentine's own v3 repository tools onto our server.

    Reimplementing search/inspect/context_slice/semantic_diff/fork/resume/
    evaluate/attest here would mean four near-duplicates with weaker limits;
    composing keeps one implementation and one security review.
    """
    try:
        from opentine.mcp_repository import register_repository_tools
    except ImportError:  # opentine 0.2.0 has no v3 repository at all
        _LOG.info(
            "opentine.mcp_repository is unavailable (needs opentine>=0.3.0); "
            "serving graphforge tools only"
        )
        return False
    try:
        # allow_promotion stays False, always: a promotion ref is a release
        # gate, and run content a model reads here is untrusted input, so
        # "promote this run" is a one-step prompt-injection payload.
        register_repository_tools(server, str(repo_path), allow_promotion=False)
    except FileNotFoundError as exc:
        # opentine's own create_server swallows this silently; an operator who
        # passed --repo deserves to know why the v3 tools are missing.
        _LOG.warning(
            "no .tine repository at %s (%s); opentine repository tools not registered",
            repo_path,
            exc,
        )
        return False
    return True


def build_server(runs_dir: str = str(DEFAULT_RUNS_DIR), repo_path: str | None = None) -> Any:
    """Register the plain functions as MCP tools (mcp 1.x and 2.x).

    ``repo_path`` additionally exposes the v3 repository: graphforge's own
    status/candidate tools plus opentine's repository tool set, composed
    rather than reimplemented.
    """
    server = _server_class()("graphforge")

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

    # Explicit tool name: the local function cannot be called graph_diff_runs
    # without shadowing (and then recursing into) the module-level one.
    @server.tool(name="graph_diff_runs")
    def graph_diff_runs_tool(left: str, right: str) -> dict[str, Any]:
        """Diff two runs at the graph layer: path, divergence, edges, visits, scores."""
        return graph_diff_runs(left, right, runs_dir)

    @server.tool()
    def fork_graph_run(run: str, from_step: str, branch: str = "mcp-fork") -> dict[str, Any]:
        """Fork a run at a step into a new artifact."""
        return fork_run(run, from_step, branch, runs_dir)

    if repo_path is not None:
        _register_repo_tools(server, repo_path)
        _register_opentine_repository_tools(server, repo_path)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="graphforge MCP server (stdio)")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument(
        "--repo",
        default=None,
        help="also expose opentine's v3 repository tools from this repository path",
    )
    args = parser.parse_args()
    server = build_server(runs_dir=args.runs_dir, repo_path=args.repo)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
