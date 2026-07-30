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
from .recorder import DEFAULT_RUNS_DIR
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
    ) -> None:
        """Verify artifact integrity (digest check)."""
        result = Run.verify_integrity(artifact)
        payload: dict[str, Any] = {"ok": result.ok, "digest": result.expected}
        if not result.ok:
            payload["reason"] = getattr(result, "reason", None)
        _emit(payload, None)
        if not result.ok:
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

    @app.command("mcp-server")
    def mcp_server_cmd(
        runs_dir: str = typer.Option(str(DEFAULT_RUNS_DIR)),
    ) -> None:
        """Start the graphforge MCP server over stdio."""
        server = mcp_mod.build_server(runs_dir=runs_dir)
        server.run(transport="stdio")


if __name__ == "__main__":
    app()
