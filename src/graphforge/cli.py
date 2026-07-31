"""graphforge CLI: run the demo graphs, then inspect/verify/fork the artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opentine import Run

from . import mcp_server as mcp_mod
from .engine import GraphEngine
from .examples import fanout_graph, refinement_graph
from .models import StaticModelAdapter, build_json_model_node
from .policy import GraphPolicy
from .recorder import DEFAULT_RUNS_DIR, verify_fork_identity
from .spec import END, GraphSpec

try:
    import typer

    app = typer.Typer(help="Graph engineering with opentine provenance", no_args_is_help=True)
    _HAS_TYPER = True
except ImportError:  # pragma: no cover - CLI extra not installed
    typer = None  # type: ignore[assignment]

    class _NoOpApp:
        def command(self, *args: Any, **kwargs: Any):
            def wrap(fn: Any) -> Any:
                return fn

            return wrap

        def __call__(self) -> None:
            raise RuntimeError(
                'CLI needs typer: pip install "opentine-graph-engineering[cli]"'
            )

    app = _NoOpApp()  # type: ignore[assignment]
    _HAS_TYPER = False


def _emit(payload: dict[str, Any], out: Path | None) -> None:
    text = json.dumps(payload, indent=2, default=repr)
    if out:
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


def _summary(result: Any) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "status": result.status,
        "stop_reason": result.stop_reason,
        "final_state": result.final_state,
        "best": (
            {
                "node": result.best.node_id,
                "score": result.best.score,
                "step_id": result.best.step_id,
            }
            if result.best
            else None
        ),
        "path": [f"{hop['node']}#{hop['visit']}" for hop in result.path],
        "artifact": str(result.artifact),
    }


def _policy(
    max_nodes: int | None,
    max_cost: float | None,
    max_duration: float | None,
    max_visits: int | None,
    min_score: float | None,
) -> GraphPolicy:
    return GraphPolicy(
        max_node_executions=max_nodes,
        max_total_cost=max_cost,
        max_duration_seconds=max_duration,
        max_visits_per_node=max_visits if max_visits is not None else 25,
        min_score=min_score,
    )


if _HAS_TYPER:

    @app.command("run-demo-refine")
    def run_demo_refine(
        target: float = typer.Option(42.0, help="value the cycle converges to"),
        start: float = typer.Option(0.0, help="starting value"),
        tolerance: float = typer.Option(0.5),
        max_nodes: int = typer.Option(60, help="run-wide node execution cap"),
        max_visits: int = typer.Option(20, help="cycle guard: max visits per node"),
        runs_dir: str = typer.Option(str(DEFAULT_RUNS_DIR)),
        out: Path | None = typer.Option(None, help="also write the JSON summary here"),
    ) -> None:
        """Run the cyclic propose/evaluate refinement demo."""
        engine = GraphEngine(
            refinement_graph(target, tolerance=tolerance),
            policy=_policy(max_nodes, None, None, max_visits, None),
            runs_dir=runs_dir,
        )
        result = engine.run(goal=f"refine x to {target:g}", initial_state={"x": start})
        _emit(_summary(result), out)

    @app.command("run-demo-fanout")
    def run_demo_fanout(
        question: str = typer.Option("graphs vs loops", help="question to fan out over"),
        runs_dir: str = typer.Option(str(DEFAULT_RUNS_DIR)),
        out: Path | None = typer.Option(None),
    ) -> None:
        """Run the fan-out/join research demo (multi-parent lineage)."""
        engine = GraphEngine(fanout_graph(), runs_dir=runs_dir)
        result = engine.run(goal=question, initial_state={"question": question})
        _emit(_summary(result), out)

    @app.command("run-model-json-demo")
    def run_model_json_demo(
        goal: str = typer.Argument("demo goal"),
        static_response: str = typer.Option(
            '```json\n{"observation": "ok", "update": {"draft": "v1"}, '
            '"score": 0.97, "stop": true}\n```',
            help="scripted model response (shows fence-safe JSON parsing)",
        ),
        runs_dir: str = typer.Option(str(DEFAULT_RUNS_DIR)),
        out: Path | None = typer.Option(None),
    ) -> None:
        """Run a one-node model-backed graph off a static adapter (offline)."""
        node = build_json_model_node(
            adapter=StaticModelAdapter(text=static_response),
            prompt_fn=lambda ctx: f"[{ctx.node_id}] {ctx.goal}: {ctx.state}",
        )
        graph = (
            GraphSpec(name="model-json-demo")
            .add_node("generate", node, kind="model")
            .add_edge("generate", END)
            .set_entry("generate")
        )
        engine = GraphEngine(graph, runs_dir=runs_dir)
        result = engine.run(goal=goal, initial_state={})
        _emit(_summary(result), out)

    @app.command()
    def show(
        artifact: str = typer.Argument(..., help="path, file name, or run-id prefix"),
        runs_dir: str = typer.Option(str(DEFAULT_RUNS_DIR)),
        steps: int = typer.Option(25),
    ) -> None:
        """Show a recorded graph run."""
        _emit(mcp_mod.show_run(artifact, runs_dir, steps=steps), None)

    @app.command()
    def mermaid(
        artifact: str = typer.Argument(...),
        runs_dir: str = typer.Option(str(DEFAULT_RUNS_DIR)),
    ) -> None:
        """Print a mermaid flowchart of the edges the run actually took."""
        print(mcp_mod.show_graph(artifact, runs_dir)["mermaid_trail"])

    @app.command()
    def verify(
        artifact: Path = typer.Argument(..., exists=True, dir_okay=False),
        hmac_key: str | None = typer.Option(
            None, help="HMAC key to check the signature against"
        ),
        public_key: Path | None = typer.Option(
            None, help="Ed25519 public key file to check the signature against"
        ),
    ) -> None:
        """Check artifact integrity and, when a key is given, its signature.

        Integrity is a checksum: it proves the file was not corrupted, NOT
        that nobody rewrote it. Only a signature is tamper evidence, and even
        then tags sit outside what the signature covers.
        """
        integrity = Run.verify_integrity(artifact)
        # Report whether a signature EXISTS separately from whether it was
        # checked, so "signed but unverified" is greppable and never reads
        # the same as "not signed".
        signature_present = False
        try:
            body = json.loads(artifact.read_text(encoding="utf-8"))
            signature_present = bool(
                (body.get("metadata") or {}).get("integrity", {}).get("signature")
            )
        except (OSError, ValueError, AttributeError):
            signature_present = False
        # The signature covers the run body plus an allowlist of metadata keys
        # whose membership is version-dependent. Report the installed list
        # rather than a count, and say plainly whether tags are in it — they
        # are not, in any version shipped so far.
        try:
            from opentine.signing import _SIGNED_METADATA_KEYS as signed_keys
        except ImportError:  # pragma: no cover - defensive
            signed_keys = ()
        payload: dict[str, Any] = {
            "integrity_ok": integrity.ok,
            "digest": integrity.expected,
            "signature_present": signature_present,
            "signature_checked": False,
            "signed_metadata_keys": sorted(str(k) for k in signed_keys),
            "tags_covered_by_signature": "tags" in set(signed_keys),
        }
        # A fork's id can prove the basis it was derived from (opentine 0.4.0+).
        try:
            fork_verdict = verify_fork_identity(Run.load(artifact))
        except Exception:  # noqa: BLE001 - never let provenance reporting fail
            fork_verdict = None
        if fork_verdict is not None:
            payload["fork_id_verified"] = fork_verdict
        if not integrity.ok:
            payload["integrity_reason"] = getattr(integrity, "reason", None)

        key_material: dict[str, Any] = {}
        # Fail closed on an empty key: an unset environment variable is the
        # normal way CI produces one, and treating it as "no key given" would
        # silently downgrade to a checksum and exit 0 on a tampered artifact.
        if hmac_key is not None:
            if not hmac_key.strip():
                raise typer.BadParameter(
                    "--hmac-key was empty (unset environment variable?); "
                    "pass a real key or omit the flag"
                )
            key_material["hmac_key"] = hmac_key.encode("utf-8")
        if public_key is not None:
            material = public_key.read_bytes()
            if not material.strip():
                raise typer.BadParameter(f"--public-key file {public_key} is empty")
            key_material["public_key"] = material
        signature_ok = True
        if key_material:
            signature = Run.verify_signature(artifact, **key_material)
            state = getattr(signature, "state", "")
            payload.update(
                {
                    "signature_checked": True,
                    "signature_ok": signature.ok,
                    # Never render verified-tofu as plain "verified": it means
                    # the embedded key was trusted on first use, not checked.
                    "signature_state": state,
                    "signer": getattr(signature, "signer", None),
                    "key_id": getattr(signature, "key_id", None),
                    "signature_reason": getattr(signature, "reason", None),
                }
            )
            signature_ok = signature.ok and state == "verified"
        _emit(payload, None)
        if not integrity.ok or not signature_ok:
            raise typer.Exit(code=1)

    @app.command()
    def compare(
        left: str = typer.Argument(...),
        right: str = typer.Argument(...),
        runs_dir: str = typer.Option(str(DEFAULT_RUNS_DIR)),
    ) -> None:
        """Diff two recorded runs (lineage-aligned)."""
        _emit(mcp_mod.diff_runs(left, right, runs_dir), None)

    @app.command()
    def fork(
        artifact: str = typer.Argument(...),
        from_step: str = typer.Option(..., help="step id (or unique prefix) to fork at"),
        branch: str = typer.Option("cli-fork"),
        runs_dir: str = typer.Option(str(DEFAULT_RUNS_DIR)),
    ) -> None:
        """Fork a run at a step into a new artifact."""
        _emit(mcp_mod.fork_run(artifact, from_step, branch, runs_dir), None)

    @app.command("ingest-otel")
    def ingest_otel(
        spans_file: Path = typer.Argument(..., exists=True, dir_okay=False),
        goal: str = typer.Option("otel ingest", help="goal recorded on the run"),
        runs_dir: str = typer.Option(str(DEFAULT_RUNS_DIR)),
    ) -> None:
        """Record a JSON file of OTel GenAI spans as a graph run."""
        from .ingest import ingest_otel_spans

        payload = json.loads(spans_file.read_text(encoding="utf-8"))
        spans = payload.get("spans", payload) if isinstance(payload, dict) else payload
        artifact = ingest_otel_spans(goal, spans, runs_dir=runs_dir)
        _emit({"artifact": str(artifact)}, None)

    @app.command("ingest-jsonl")
    def ingest_jsonl_cmd(
        events_file: Path = typer.Argument(..., exists=True, dir_okay=False),
        goal: str = typer.Option("jsonl ingest", help="goal recorded on the run"),
        runs_dir: str = typer.Option(str(DEFAULT_RUNS_DIR)),
    ) -> None:
        """Replay a JSONL event stream into a recorded graph run."""
        from .ingest import ingest_jsonl

        artifact = ingest_jsonl(goal, events_file, runs_dir=runs_dir)
        _emit({"artifact": str(artifact)}, None)

    @app.command("graph-diff")
    def graph_diff_cmd(
        left: str = typer.Argument(...),
        right: str = typer.Argument(...),
        runs_dir: str = typer.Option(str(DEFAULT_RUNS_DIR)),
    ) -> None:
        """Diff two runs at the GRAPH layer: path, edges, visits, score drift.

        Unlike a step diff, this answers "did these two runs behave the
        same?" — opentine's own diffs cannot, because v3 event identity
        includes a timestamp.
        """
        # Via the MCP function, not Run.load directly: that path is confined
        # to runs_dir and size-capped before loading.
        _emit(mcp_mod.graph_diff_runs(left, right, runs_dir), None)

    # ---------------------------------------------------- v3 repository ----

    def _require_repo() -> Any:
        from . import repo_backend

        if not repo_backend.repo_available():
            print("v3 repositories need opentine>=0.3.0 (the installed opentine has no Repo)")
            raise typer.Exit(code=2)
        return repo_backend

    @app.command("repo-init")
    def repo_init(path: str = typer.Argument(".", help="where to create .tine/")) -> None:
        """Create a v3 .tine repository."""
        backend = _require_repo()
        repo = backend.open_repo(path, create=True)
        _emit({"worktree": str(repo.worktree), "repo": str(repo.path)}, None)

    @app.command("repo-save")
    def repo_save(
        artifact: str = typer.Argument(..., help="a .tine file, name, or run-id prefix"),
        repo_path: str = typer.Option(".", "--repo", help="repository worktree"),
        ref: str | None = typer.Option(
            None, help="ref to advance, e.g. experiments/my-run (graphforge writes experiments/*)"
        ),
        runs_dir: str = typer.Option(str(DEFAULT_RUNS_DIR)),
    ) -> None:
        """Store a recorded run in a v3 repository as content-addressed objects."""
        backend = _require_repo()
        run = Run.load(mcp_mod._normalize_ref(artifact, Path(runs_dir).expanduser()))
        result = backend.save_to_repo(run, repo_path, ref=ref)
        result["event_map"] = len(result["event_map"])  # count, not the whole map
        _emit(result, None)

    @app.command("repo-status")
    def repo_status_cmd(
        repo_path: str = typer.Option(".", "--repo"),
        deep: bool = typer.Option(True, help="run a deep fsck"),
    ) -> None:
        """Refs, object count, shallow boundary, and an fsck verdict."""
        backend = _require_repo()
        _emit(backend.repo_status(repo_path, deep=deep), None)

    @app.command("repo-evaluate")
    def repo_evaluate(
        run_oid: str = typer.Argument(..., help="run:sha256:... or a ref name"),
        score: float = typer.Option(..., help="the number you actually gate on"),
        metric: str = typer.Option("gate", help="metric name"),
        signer: str = typer.Option(..., help="who is asserting this (a label, not an identity)"),
        repo_path: str = typer.Option(".", "--repo"),
    ) -> None:
        """Attach an evaluation attestation so the run becomes searchable.

        Emit ONE metric per attestation: opentine scores a claim by the mean
        of its scores dict, so mixing a pass rate with a cost is meaningless.
        """
        backend = _require_repo()
        try:
            _emit(
                {
                    "attestation": backend.evaluate(
                        repo_path, run_oid, {metric: score}, signer=signer
                    )
                },
                None,
            )
        except ValueError as exc:
            print(str(exc))
            raise typer.Exit(code=1) from exc

    @app.command("repo-candidates")
    def repo_candidates(
        min_score: float | None = typer.Option(None, help="only runs scored at or above this"),
        model: str | None = typer.Option(None, help="only runs that used this model"),
        repo_path: str = typer.Option(".", "--repo"),
        limit: int = typer.Option(50),
    ) -> None:
        """List evaluated runs, best first (needs repo-evaluate to have scored them)."""
        backend = _require_repo()
        found = backend.candidates(
            repo_path, min_score=min_score, model=model, limit=limit
        )
        _emit(
            {
                "candidates": [
                    {
                        "run_id": c.run_id,
                        "status": c.status,
                        "score": c.score,
                        "cost": c.cost,
                        "models": list(c.models),
                    }
                    for c in found
                ]
            },
            None,
        )

    @app.command("repo-promote")
    def repo_promote(
        run_oid: str = typer.Argument(...),
        name: str = typer.Argument(..., help="release gate name, e.g. prod"),
        signer: str | None = typer.Option(
            None, help="record an approval attestation as this label"
        ),
        force: bool = typer.Option(False, help="advance a gate that already points elsewhere"),
        repo_path: str = typer.Option(".", "--repo"),
    ) -> None:
        """Move a release gate to a run, under compare-and-swap."""
        backend = _require_repo()
        try:
            _emit(backend.promote(repo_path, run_oid, name, force=force, signer=signer), None)
        except ValueError as exc:
            print(str(exc))
            raise typer.Exit(code=1) from exc

    @app.command("mcp-server")
    def mcp_server_cmd(
        runs_dir: str = typer.Option(str(DEFAULT_RUNS_DIR)),
        repo_path: str | None = typer.Option(
            None, "--repo", help="also expose opentine's v3 repository tools"
        ),
    ) -> None:
        """Start the graphforge MCP server over stdio."""
        server = mcp_mod.build_server(runs_dir=runs_dir, repo_path=repo_path)
        server.run(transport="stdio")


if __name__ == "__main__":
    app()
