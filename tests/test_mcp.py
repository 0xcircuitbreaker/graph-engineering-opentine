from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
from opentine import Run, RunStatus, StepKind

from graphforge import GraphEngine, mcp_server
from graphforge.examples import fanout_graph, refinement_graph
from graphforge.mcp_server import (
    build_server,
    diff_runs,
    fork_run,
    graph_diff_runs,
    list_runs,
    show_graph,
    show_run,
)
from graphforge.repo_backend import repo_available


def _has_repository_mcp() -> bool:
    """opentine 0.2.0 ships no v3 repository MCP module at all."""
    try:
        import opentine.mcp_repository  # noqa: F401
    except ImportError:
        return False
    return True


requires_repo = pytest.mark.skipif(
    not repo_available(), reason="v3 repositories need opentine>=0.3.0"
)
requires_repo_mcp = pytest.mark.skipif(
    not _has_repository_mcp(), reason="opentine.mcp_repository needs opentine>=0.3.0"
)

#: A mermaid identifier position must never contain caller-controlled syntax,
#: and label text must only ever appear inside a quoted position. These
#: patterns assert that *structure* -- deliberately not "whatever the
#: sanitizer happens to emit", which is how a label able to close a link with
#: ``---`` passed review once already. ``[^"|]`` is the load-bearing part:
#: nothing inside a label can end the quote or the pipe pair.
_ID = r"[A-Za-z_][A-Za-z0-9_]*"
_NODE_LINE = re.compile(rf'^    {_ID}\["[^"|]*"\]$')
_EDGE_LINE = re.compile(rf'^    {_ID}(?: -->\|"[^"|]*"\|| -->) {_ID}$')


@pytest.fixture()
def runs(tmp_path):
    refine = GraphEngine(refinement_graph(10.0), runs_dir=tmp_path).run(
        goal="ten", initial_state={"x": 0.0}
    )
    fanout = GraphEngine(fanout_graph(), runs_dir=tmp_path).run(
        goal="q", initial_state={"question": "q"}
    )
    return tmp_path, refine, fanout


def test_list_runs(runs):
    tmp_path, refine, fanout = runs
    entries = list_runs(str(tmp_path))
    assert len(entries) == 2
    ids = {e["run_id"] for e in entries}
    assert {refine.run_id, fanout.run_id} <= ids
    assert all(e["kind"] == "graphforge" for e in entries)


def test_show_run_by_prefix(runs):
    tmp_path, refine, _ = runs
    shown = show_run(refine.run_id[:10], str(tmp_path))
    assert shown["run_id"] == refine.run_id
    assert shown["graph"]["entry"] == "propose"
    assert any(s["node"] == "evaluate" for s in shown["steps"])


def test_show_graph_trail(runs):
    tmp_path, refine, _ = runs
    result = show_graph(refine.run_id[:10], str(tmp_path))
    assert result["topology"]["name"].startswith("refine-to")
    assert any(hop["node"] == "propose" for hop in result["trail"])
    assert "flowchart TD" in result["mermaid_trail"]
    assert "propose" in result["mermaid_trail"]


def test_show_graph_renders_join_fan_in(runs):
    tmp_path, _, fanout = runs
    mermaid = show_graph(fanout.run_id[:10], str(tmp_path))["mermaid_trail"]
    # piped link text, not "-- join -->": see _mermaid_label
    assert 'research_a -->|"join"| merge' in mermaid
    assert 'research_b -->|"join"| merge' in mermaid


def test_diff_runs_shape(runs):
    tmp_path, refine, fanout = runs
    diff = diff_runs(refine.run_id[:10], fanout.run_id[:10], str(tmp_path))
    assert diff["left"] == refine.run_id
    assert diff["right"] == fanout.run_id
    # regression pin on opentine FieldDelta/StepChange attribute names
    for change in diff["changed"]:
        assert set(change) == {"step_a", "step_b", "fields"}
        for field in change["fields"]:
            assert set(field) == {"name", "before", "after", "changed_keys"}


def _first_node_step(artifact) -> str:
    run = Run.load(artifact)
    return next(sid for sid in run.graph.order if run.graph.steps[sid].inputs.get("node"))


def test_fork_run_produces_new_artifact(runs):
    tmp_path, refine, _ = runs
    node_step = _first_node_step(refine.artifact)
    result = fork_run(refine.run_id[:10], node_step[:12], "test-fork", str(tmp_path))
    assert result["fork_point"] == node_step
    forked = Run.load(result["path"])
    assert forked.metadata.get("forked_from") == refine.run_id
    assert node_step in forked.graph.steps


@pytest.mark.parametrize(
    "branch", ["../../../evil", "/etc/evil", "a/../../b", "..", "....//", "  "]
)
def test_fork_branch_cannot_escape_the_runs_dir(runs, branch):
    """The branch name is an MCP tool argument folded into a file name."""
    tmp_path, refine, _ = runs
    node_step = _first_node_step(refine.artifact)
    result = fork_run(refine.run_id[:10], node_step[:12], branch, str(tmp_path))
    written = Path(result["path"]).resolve()
    # The property is "one file name, inside the runs dir": interior dots are
    # harmless once no separator survives, which is what slug() guarantees.
    assert written.parent == Path(tmp_path).resolve()
    assert "/" not in written.name and not written.name.startswith(".")
    assert Run.load(written).run_id == result["run_id"]


def test_ambiguous_and_missing_refs(runs):
    tmp_path, _, _ = runs
    with pytest.raises(FileNotFoundError):
        show_run("does-not-exist", str(tmp_path))


# --- confinement -------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref",
    [
        "../../etc/passwd",
        "../outside",
        "../outside.tine",
        "..",
        "/etc/passwd",
        "~/.bashrc",
        "sub/../../outside.tine",
    ],
)
def test_traversal_refs_are_rejected(runs, ref):
    tmp_path, _, _ = runs
    with pytest.raises(FileNotFoundError):
        show_run(ref, str(tmp_path))


def test_absolute_ref_outside_runs_dir_is_rejected(tmp_path):
    """An artifact that is real, loadable, and outside the sandbox stays unreachable."""
    outside = tmp_path / "outside"
    inside = tmp_path / "runs"
    outside.mkdir()
    inside.mkdir()
    secret = GraphEngine(refinement_graph(2.0), runs_dir=outside).run(
        goal="secret", initial_state={"x": 0.0}
    )
    with pytest.raises(FileNotFoundError):
        show_run(str(secret.artifact), str(inside))
    with pytest.raises(FileNotFoundError):
        show_graph(secret.run_id[:10], str(inside))
    with pytest.raises(FileNotFoundError):
        fork_run(str(secret.artifact), "x", "b", str(inside))


def test_symlink_escape_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    inside = tmp_path / "runs"
    outside.mkdir()
    inside.mkdir()
    secret = GraphEngine(refinement_graph(2.0), runs_dir=outside).run(
        goal="secret", initial_state={"x": 0.0}
    )
    link = inside / "innocent.tine"
    link.symlink_to(secret.artifact)
    with pytest.raises(FileNotFoundError):
        show_run("innocent", str(inside))
    with pytest.raises(FileNotFoundError):
        show_run("innocent.tine", str(inside))
    # the prefix scan must not launder it either
    with pytest.raises(FileNotFoundError):
        show_run("inno", str(inside))
    assert list_runs(str(inside)) == []


def test_absolute_ref_inside_runs_dir_still_works(runs):
    """Confinement must not break the legitimate path/filename UX."""
    tmp_path, refine, _ = runs
    assert show_run(str(refine.artifact), str(tmp_path))["run_id"] == refine.run_id
    assert show_run(refine.artifact.name, str(tmp_path))["run_id"] == refine.run_id
    assert show_run(refine.artifact.stem, str(tmp_path))["run_id"] == refine.run_id


def test_non_tine_file_in_runs_dir_is_not_loadable(runs):
    tmp_path, _, _ = runs
    (tmp_path / "notes.txt").write_text("not an artifact")
    with pytest.raises(FileNotFoundError):
        show_run("notes.txt", str(tmp_path))


# --- resource limits ---------------------------------------------------------------


def test_oversized_artifact_is_refused_before_load(runs, monkeypatch):
    tmp_path, refine, _ = runs
    monkeypatch.setattr("graphforge.mcp_server._MAX_RUN_BYTES", 16)
    with pytest.raises(ValueError, match="caps artifacts"):
        show_run(refine.run_id[:10], str(tmp_path))
    with pytest.raises(ValueError, match="caps artifacts"):
        show_graph(refine.run_id[:10], str(tmp_path))
    with pytest.raises(ValueError, match="caps artifacts"):
        graph_diff_runs(refine.run_id[:10], refine.run_id[:10], str(tmp_path))
    # listing degrades to per-artifact errors instead of failing wholesale
    entries = list_runs(str(tmp_path))
    assert entries and all("error" in entry for entry in entries)


def test_show_run_clamps_a_hostile_step_count(runs):
    tmp_path, refine, _ = runs
    shown = show_run(refine.run_id[:10], str(tmp_path), steps=10**9)
    assert len(shown["steps"]) <= 1000
    assert shown["step_count_total"] == shown["step_count"]


def test_responses_are_bounded(runs, monkeypatch):
    """The ordinary path: trimming the step list is enough, so detail survives."""
    tmp_path, refine, _ = runs
    monkeypatch.setattr("graphforge.mcp_server._MAX_TEXT_CHARS", 2_000)
    shown = show_run(refine.run_id[:10], str(tmp_path))
    assert _fits_cap(shown)
    assert shown["truncated"] is True
    assert 0 <= len(shown["steps"]) < shown["step_count_total"]
    graph_view = show_graph(refine.run_id[:10], str(tmp_path))
    assert _fits_cap(graph_view)
    assert graph_view["truncated"] is True


def test_backstop_returns_identity_rather_than_an_oversized_payload(runs, monkeypatch):
    """A cap too small for any real detail still yields a compliant response."""
    tmp_path, refine, _ = runs
    monkeypatch.setattr("graphforge.mcp_server._MAX_TEXT_CHARS", 400)
    shown = show_run(refine.run_id[:10], str(tmp_path))
    assert _fits_cap(shown)
    assert shown["truncated"] is True
    assert shown["run_id"] == refine.run_id
    assert "steps" not in shown


def _fits_cap(payload) -> bool:
    """The cap is on what is actually returned, so measure exactly that."""
    return len(json.dumps(payload, default=str)) <= mcp_server._MAX_TEXT_CHARS


def test_cap_holds_when_the_oversize_is_outside_the_trimmed_keys(tmp_path):
    """A run with a huge goal and huge tags, but only one step and no topology.

    Trimming "steps"/"graph" cannot reclaim a single byte here; before the fix
    this returned ~850 KB with truncated=True, i.e. the flag lied.
    """
    run = Run(
        manifest={"kind": "graphforge", "goal": "G" * 800_000},
        status=RunStatus.completed,
        user_prompt="p",
        tags=["t" * 50_000 for _ in range(20)],
    )
    run.add_step(StepKind.think, inputs={"node": "a"}, outputs={})
    run.save(tmp_path / "fat.tine")

    shown = show_run("fat", str(tmp_path))
    assert _fits_cap(shown)
    assert shown["truncated"] is True
    # degraded, not discarded: the run is still identifiable and readable
    assert shown["run_id"] == run.run_id
    assert len(shown["goal"]) < 3_000
    assert len(shown["tags"]) <= 50


def test_cap_holds_for_a_flooded_cost_breakdown(tmp_path):
    """by_model keys come from step.model_info, i.e. from an ingested trace."""
    run = Run(manifest={"kind": "graphforge"}, status=RunStatus.completed, user_prompt="p")
    for index in range(200):
        run.add_step(StepKind.model, inputs={}, outputs={}, model_info=f"{index}-{'m' * 2000}")
    run.save(tmp_path / "models.tine")

    shown = show_run("models", str(tmp_path))
    assert _fits_cap(shown)
    assert shown["truncated"] is True
    by_model = shown["cost"]["by_model"]
    assert by_model["__truncated__"] == 200 - 50
    assert all(len(key) <= 120 for key in by_model)


def test_show_graph_cap_holds_when_the_mermaid_dominates(tmp_path):
    """The mermaid string ships inside the response and must fit inside it too."""
    run = Run(manifest={"kind": "graphforge"}, status=RunStatus.completed, user_prompt="p")
    for index in range(1000):
        run.add_step(
            StepKind.think,
            inputs={"node": f"node-{index}-{'x' * 200}", "edge": {"source": "start"}},
            outputs={},
        )
    run.save(tmp_path / "wide.tine")

    view = show_graph("wide", str(tmp_path))
    assert _fits_cap(view)
    assert view["truncated"] is True


def test_list_runs_output_is_capped(tmp_path):
    for index in range(5):
        run = Run(
            manifest={"kind": "graphforge", "goal": f"{index}" + "G" * 200_000},
            status=RunStatus.completed,
            user_prompt="p",
        )
        run.add_step(StepKind.think, inputs={"node": "a"}, outputs={})
        run.save(tmp_path / f"big{index}.tine")

    entries = list_runs(str(tmp_path))
    assert _fits_cap(entries)
    assert entries[-1]["error"].startswith("truncated: response cap")


# --- mermaid safety ----------------------------------------------------------------


def test_mermaid_edges_are_deduped(runs):
    """A cycle takes the same edge every iteration; the diagram shows it once."""
    tmp_path, refine, _ = runs
    result = show_graph(refine.run_id[:10], str(tmp_path))
    hops = [hop for hop in result["trail"] if hop["node"] == "evaluate"]
    assert len(hops) > 1, "fixture must actually loop for this to prove anything"
    lines = result["mermaid_trail"].splitlines()
    assert lines == list(dict.fromkeys(lines))
    assert lines.count('    propose -->|"static"| evaluate') == 1


def test_mermaid_slugifies_hostile_node_ids(tmp_path):
    hostile = 'evil"] --> owned["pwned'
    run = Run(manifest={"kind": "graphforge"}, status=RunStatus.completed, user_prompt="p")
    run.add_step(StepKind.think, inputs={"node": 42, "edge": {}}, outputs={})
    run.add_step(
        StepKind.think,
        inputs={"node": hostile, "edge": {"source": 42, "kind": 'x --> y["z'}},
        outputs={},
    )
    run.save(tmp_path / "hostile.tine")

    mermaid = show_graph("hostile", str(tmp_path))["mermaid_trail"]
    lines = mermaid.splitlines()
    assert lines[0] == "flowchart TD"
    for line in lines[1:]:
        assert _NODE_LINE.match(line) or _EDGE_LINE.match(line), line
    # ids are slugged (a leading digit is not a legal mermaid identifier)...
    assert '    n_42["42"]' in mermaid
    # ...and the original text survives only inside the quoted label
    assert "#quot;" in mermaid


def test_mermaid_edge_label_cannot_terminate_a_link(tmp_path):
    """`` --- `` is a complete link terminator in mermaid's flowchart grammar.

    In the old ``{src} -- {label} --> {dst}`` form, an edge label of
    ``ok --- injected`` rendered as ``start -- ok --- injected --> end``,
    which declares a node ``injected`` that never ran -- straight into the
    diagram an operator reads to answer "what did this run do?".
    """
    run = Run(manifest={"kind": "graphforge"}, status=RunStatus.completed, user_prompt="p")
    run.add_step(StepKind.think, inputs={"node": "start", "edge": {}}, outputs={})
    run.add_step(
        StepKind.think,
        inputs={
            "node": "finish",
            "edge": {"source": "start", "kind": 'ok --- injected["x"] |y| -->'},
        },
        outputs={},
    )
    run.save(tmp_path / "linkinject.tine")

    lines = show_graph("linkinject", str(tmp_path))["mermaid_trail"].splitlines()
    assert lines[0] == "flowchart TD"
    for line in lines[1:]:
        assert _NODE_LINE.match(line) or _EDGE_LINE.match(line), line
    declared = {line.strip().split("[", 1)[0] for line in lines[1:] if _NODE_LINE.match(line)}
    assert declared == {"start", "finish"}
    # every identifier mentioned anywhere was declared: no smuggled node
    referenced = {
        part
        for line in lines[1:]
        if _EDGE_LINE.match(line)
        for part in (line.strip().split(" ")[0], line.strip().split(" ")[-1])
    }
    assert referenced <= declared
    assert "injected" not in referenced


def test_mermaid_reserved_ids_are_prefixed(tmp_path):
    """Lowercase ``end`` as a node id aborts the whole mermaid flowchart."""
    run = Run(manifest={"kind": "graphforge"}, status=RunStatus.completed, user_prompt="p")
    run.add_step(StepKind.think, inputs={"node": "end", "edge": {}}, outputs={})
    run.save(tmp_path / "reserved.tine")
    mermaid = show_graph("reserved", str(tmp_path))["mermaid_trail"]
    assert '    n_end["end"]' in mermaid


def test_mermaid_slug_collisions_stay_distinct(tmp_path):
    run = Run(manifest={"kind": "graphforge"}, status=RunStatus.completed, user_prompt="p")
    for node in ("a b", "a-b", "a.b"):
        run.add_step(StepKind.think, inputs={"node": node, "edge": {}}, outputs={})
    run.save(tmp_path / "collide.tine")
    lines = show_graph("collide", str(tmp_path))["mermaid_trail"].splitlines()[1:]
    idents = [line.strip().split("[", 1)[0] for line in lines]
    assert len(set(idents)) == 3


# --- graph-layer diff --------------------------------------------------------------


def test_graph_diff_runs_round_trip(runs):
    tmp_path, refine, fanout = runs
    same = graph_diff_runs(refine.run_id[:10], refine.run_id[:10], str(tmp_path))
    assert same["identical_path"] is True
    assert same["first_divergence"] is None
    assert same["edges_only_left"] == [] and same["edges_only_right"] == []

    other = graph_diff_runs(refine.run_id[:10], fanout.run_id[:10], str(tmp_path))
    assert other["left_run"] == refine.run_id
    assert other["right_run"] == fanout.run_id
    assert other["identical_path"] is False
    assert other["first_divergence"]["index"] == 0
    assert other["visit_delta"]["propose"]["right"] == 0
    assert other["visit_delta"]["propose"]["left"] >= 2
    assert any(edge[1] == "merge" for edge in other["edges_only_right"])


def test_graph_diff_refs_are_confined(runs):
    tmp_path, refine, _ = runs
    with pytest.raises(FileNotFoundError):
        graph_diff_runs(refine.run_id[:10], "../../etc/passwd", str(tmp_path))


# --- server composition ------------------------------------------------------------


def _tool_names(server) -> set[str]:
    return {tool.name for tool in asyncio.run(server.list_tools())}


def test_build_server_registers_graph_tools(tmp_path):
    names = _tool_names(build_server(str(tmp_path)))
    assert names == {
        "list_graph_runs",
        "show_graph_run",
        "show_graph_topology",
        "diff_graph_runs",
        "graph_diff_runs",
        "fork_graph_run",
    }


def test_build_server_without_repo_ignores_repo_tools(tmp_path, caplog):
    """A --repo pointing at a non-repository logs and degrades, never raises.

    INFO covers both reasons the repository tools can be missing: opentine
    0.2.0 (no v3 at all, INFO) and a path that is not a repository (WARNING).
    """
    with caplog.at_level("INFO", logger="graphforge.mcp_server"):
        names = _tool_names(build_server(str(tmp_path), repo_path=str(tmp_path)))
    assert "repo_status" not in names
    assert "search_runs" not in names
    assert caplog.records, "a skipped repository must be explained, never silent"


@requires_repo
@requires_repo_mcp
def test_build_server_composes_opentine_repository_tools(tmp_path):
    from graphforge.repo_backend import open_repo

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    open_repo(repo_dir, create=True)
    names = _tool_names(build_server(str(tmp_path), repo_path=str(repo_dir)))
    # graphforge's graph layer ...
    assert {"show_graph_topology", "graph_diff_runs"} <= names
    assert {"repo_status", "list_repo_candidates"} <= names
    # ... composed with opentine's own v3 repository tools rather than reimplemented
    assert {
        "search_runs",
        "inspect_object",
        "context_slice",
        "semantic_diff",
        "fork_run_v3",
        "resume_run_v3",
        "evaluate_run",
        "attest_run",
    } <= names
    # promotion is a release gate: never exposed to a model
    assert "promote_run" not in names


@requires_repo
@requires_repo_mcp
def test_composed_repository_tool_is_callable(tmp_path):
    """Listing a composed tool is not proof it runs; call one end to end."""
    from graphforge.repo_backend import open_repo

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    open_repo(repo_dir, create=True)
    server = build_server(str(tmp_path), repo_path=str(repo_dir))
    result = asyncio.run(server.call_tool("search_runs", {"query": ""}))
    assert not getattr(result, "is_error", False)


@requires_repo
def test_repo_tools_report_status_and_candidates(tmp_path):
    from graphforge.mcp_server import repo_candidates, repo_summary
    from graphforge.repo_backend import evaluate, open_repo, save_to_repo

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = open_repo(repo_dir, create=True)
    recorded = GraphEngine(refinement_graph(4.0), runs_dir=tmp_path).run(
        goal="four", initial_state={"x": 0.0}
    )
    from opentine import Run

    stored = save_to_repo(Run.load(recorded.artifact), repo, ref="experiments/mcp")
    evaluate(repo, stored["run_id"], {"quality": 0.9}, signer="test")

    status = repo_summary(str(repo_dir))
    assert status["ok"] is True
    assert "experiments/mcp" in status["refs"]

    found = repo_candidates(str(repo_dir), min_score=0.5)
    assert any(entry["run_id"] == stored["run_id"] for entry in found)
    assert repo_candidates(str(repo_dir), min_score=0.99) == []


def test_opentine_surface_regression_pins(tmp_path):
    """Pin the exact opentine attributes graphforge relies on (0.2.x and 0.3.x)."""
    run = Run(manifest={"kind": "pin"}, status=RunStatus.running, user_prompt="p", policies={})
    step = run.add_step(
        StepKind.think,
        inputs={"node": "a"},
        outputs={"observation": "o"},
        parent_id=None,
        duration=0.0,
        cost=0.0,
    )
    assert step.id
    child = run.add_step(StepKind.done, inputs={}, parent_ids=[step.id])
    assert step.id in child.parent_ids
    forked = run.fork(from_step_id=step.id, branch="b")
    assert forked.run_id != run.run_id
    diff = run.diff(forked)
    assert hasattr(diff, "common_ancestor")
    assert hasattr(diff, "only_a") and hasattr(diff, "only_b") and hasattr(diff, "changed")
    path = run.save(tmp_path / "pin.tine")
    integrity = Run.verify_integrity(path)
    assert integrity.ok is True
    assert integrity.expected
    breakdown = run.cost_breakdown()
    for attr in ("total_cost", "total_tokens", "by_model", "by_kind"):
        assert hasattr(breakdown, attr)
