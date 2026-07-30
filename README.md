# graph-engineering-opentine

A public, reusable setup for **Graph Engineering** backed by [`opentine`](https://github.com/0xcircuitbreaker/opentine): every node execution in a graph run becomes a recorded step in a content-addressed `.tine` artifact — with the **edge taken recorded first-class**, **multi-parent lineage for joins**, and **lossless state snapshots that make any step a fork/resume point**.

Sibling of [`loop-engineering-opentine`](https://github.com/0xcircuitbreaker/loop-engineering-opentine): loops defer architecture, graphs declare it. This repo is the declared-architecture half.

## What this repo gives you

- `GraphSpec` — framework-agnostic graph definition: nodes, static edges, routers (conditional edges as functions of state), join nodes with reducers, topology digest, mermaid export.
- `GraphEngine` — executes a `GraphSpec` with fan-out, join fan-in, cycles (bounded by policy), and records everything into one opentine run.
- `GraphEngine.resume` — **time travel for any graph**: fork a recorded artifact at any step, mutate state, and continue the graph from there.
- `GraphRecorder` — the single opentine integration point (one file to touch on opentine upgrades).
- `GraphRunCapture` — the universal push-based adapter: record **any** external graph framework (LangGraph, Burr, CrewAI Flows, LlamaIndex Workflows, pydantic-graph, homegrown engines) via `node_start` / `node_end` / `model_call` / `tool_call` / `edge` / `end` events. Never-throw discipline: recorder failures never break the host.
- Shipped adapters — all zero-change attach points, all import-safe without their framework: **LangGraph** (checkpointer wrapper + callback handler), **Burr** (lifecycle hook), **CrewAI Flows** (event-bus listener with per-flow isolation), **LlamaIndex Workflows** (checkpoint conversion), **pydantic-graph V2** (`graph.iter()` driver with `fork_stack` lineage).
- Zero-glue ingestion for everything else: `ingest_otel_spans` (OTel GenAI semantic conventions) and `ingest_jsonl` (plain JSON-lines event streams).
- `GraphPolicy` — **run-wide** budgets: node executions, total cost, wall clock, cycle guard (`max_visits_per_node`), success threshold.
- CLI + MCP server for running demos and listing/showing/diffing/forking artifacts.

## Why opentine under a graph framework

Observability platforms (LangSmith, Langfuse, AgentOps, OTel GenAI) record *traces*; none record the **edge actually taken**, none give **content-addressed fork lineage across runs**, and none produce **verifiable artifacts** (`tine verify`, signatures). LangGraph's checkpointing gives time travel inside one thread on one machine; a `.tine` artifact is portable provenance you can diff, fork, attest, and ship.

## Install

```bash
cd graph-engineering-opentine
pip install -e ".[cli,mcp]"
```

Requires Python >= 3.11 and `opentine>=0.2.0,<0.4` (tested against 0.2.0 and 0.3.0).

## Quick start

```bash
# cyclic refinement demo (router + cycle guard)
graphforge run-demo-refine --target 42 --start 0

# fan-out/join demo (multi-parent lineage)
graphforge run-demo-fanout --question "graphs vs loops"

# model-backed node off a static adapter (offline, fence-safe JSON parsing)
graphforge run-model-json-demo "draft a status update"

# inspect artifacts
graphforge show <run-id-prefix>
graphforge mermaid <run-id-prefix>       # edges the run actually took
graphforge verify ~/.local/share/graphforge/<run-id>.tine
graphforge compare <left> <right>
graphforge fork <run> --from-step <step-prefix>
```

## Define and run a graph

```python
from graphforge import END, GraphEngine, GraphNodeResult, GraphPolicy, GraphSpec

def draft(ctx):
    return GraphNodeResult(observation="drafted", update={"draft": f"v{ctx.visit}"})

def review(ctx):
    ok = len(ctx.state["draft"]) > 1
    return GraphNodeResult(observation="reviewed", score=1.0 if ok else 0.3,
                           update={"approved": ok})

def route(state):
    return END if state.get("approved") else "draft"

graph = (
    GraphSpec(name="draft-review")
    .add_node("draft", draft)
    .add_node("review", review)
    .add_edge("draft", "review")
    .add_router("review", route)          # conditional edge, function of state
    .set_entry("draft")
)

engine = GraphEngine(graph, policy=GraphPolicy(max_node_executions=20, max_visits_per_node=5))
result = engine.run(goal="ship a draft", initial_state={"draft": ""})
print(result.status, result.final_state, result.artifact)
```

Every node execution is a step in the artifact carrying: the node id and visit count, the **edge that fired it** (`static` / `router` / `goto` / `join` / `entry`), state digests before/after, a key-level delta, a lossless `state_after` snapshot, score, cost/usage, and DAG parent links (multi-parent at joins).

## Fork any step, continue the graph

```python
# fork at a recorded step, push the state somewhere else, re-run the rest
resumed = engine.resume(result.artifact, some_step_id,
                        state_update={"draft": "adversarial input"})
# artifacts stay linked: resumed run has metadata.forked_from + fork_point,
# and topology drift between record-time and resume-time graphs is tagged.
```

Resume verifies the snapshot against its recorded state digest. One caveat: opentine redacts credential-named keys (tokens, secrets, passwords) at save time, so those values do not round-trip — resume fails loudly and tells you to re-supply them via `state_update` (the fork is then tagged `graph:snapshot-redacted`).

## Record ANY graph framework (universal capture)

```python
from graphforge import GraphRunCapture

cap = GraphRunCapture("my burr app", framework="burr")
cap.node_start("plan", state={"q": "..."})
cap.model_call(model="claude-sonnet-5", prompt="...", response="...",
               cost=0.002, usage={"input": 900, "output": 200})
cap.node_end("plan", update={"plan": ["a", "b"]}, kind="model")
cap.edge("plan", "execute", label="default")
cap.node_start("execute")
cap.node_end("execute", update={"done": True})
cap.end(summary="finished")               # -> ~/.local/share/graphforge/<run>.tine
```

`GraphRunCapture` never raises into the host (`strict=True` re-raises for tests), truncates all payloads, sanitizes metrics for strict opentine validation, and keeps DAG-aware parent links (explicit `parents=[...]` for fan-in).

### LangGraph (shipped adapter)

```python
from graphforge import GraphRunCapture
from graphforge.adapters import RecordingCallbackHandler, RecordingCheckpointer
from langgraph.checkpoint.memory import InMemorySaver

cap = GraphRunCapture("langgraph run", framework="langgraph")
saver = RecordingCheckpointer(cap, inner=InMemorySaver())   # one line at compile
handler = RecordingCallbackHandler(cap, price_table={"gpt-4o-mini": (0.15, 0.60)})

graph = builder.compile(checkpointer=saver)
graph.invoke(inputs, config={"configurable": {"thread_id": "t1"},
                             "callbacks": [handler]})
cap.end()
```

The checkpointer wrapper records every superstep checkpoint (the forkable-point table: `checkpoint_id`, parent checkpoint, `source`, full channel values). The callback handler records node executions keyed by `langgraph_node` metadata, the trigger that fired each node, and model/tool calls with token usage.

### Burr (shipped adapter)

```python
from graphforge import GraphRunCapture
from graphforge.adapters import TineBurrHook

cap = GraphRunCapture("my burr app", framework="burr")
app = (ApplicationBuilder()
       .with_actions(...).with_transitions(...)
       .with_hooks(TineBurrHook(cap))     # one line, zero action changes
       .build())
app.run(halt_after=["done"])
cap.end()
```

Records the static topology (actions + transitions) and fork lineage from `post_application_create`, then one node per executed action with pre/post state snapshots and the edge from the previous action.

### CrewAI Flows (shipped adapter)

```python
from graphforge.adapters import TineFlowListener

listener = TineFlowListener(runs_dir="~/.local/share/graphforge")
# with crewai installed the listener self-registers on the event bus;
# every flow gets its own capture keyed by state id (bus is a singleton),
# @router return labels become edge labels, LLM calls carry usage/cost.
# finished flows land in listener.artifacts.
```

### LlamaIndex Workflows (shipped adapter)

```python
from graphforge import GraphRunCapture
from graphforge.adapters import record_checkpoints

wc = WorkflowCheckpointer(workflow=wf)
handler = wc.run(topic="...")
await handler
# WorkflowCheckpointer stores checkpoints in memory only — persist them:
record_checkpoints(GraphRunCapture("wf run", framework="llamaindex"), wc)
```

Edges reconstruct as (producer step, event type, consumer step); every checkpoint's full `ctx_state` becomes a forkable marker.

### pydantic-graph V2 (shipped adapter)

```python
from graphforge import GraphRunCapture
from graphforge.adapters import record_graph_run

cap = GraphRunCapture("pg run", framework="pydantic-graph")
output, artifact = await record_graph_run(cap, graph, state=MyState())
```

Drives `graph.iter()` — nodes execute exactly as they would; every scheduled `GraphTask` is recorded with its `fork_stack` (pydantic-graph's exact parallel-branch ancestry) and per-event state snapshots.

## Zero-glue ingestion (no adapter needed)

Any setup that can produce OTel GenAI spans or append JSON lines gets recording with no Python integration:

```python
from graphforge import ingest_otel_spans, ingest_jsonl

# OTel GenAI semantic conventions: invoke_agent -> nodes, chat -> model
# calls, execute_tool -> tool calls, attributed via the span parent tree
ingest_otel_spans("traced run", spans)

# lowest common denominator: JSONL events
# {"type": "node_start"|"node_end"|"model_call"|"tool_call"|"edge"|
#  "checkpoint"|"error"|"end", ...}
ingest_jsonl("logged run", "events.jsonl")
```

```bash
graphforge ingest-otel spans.json
graphforge ingest-jsonl events.jsonl
```

## Policy

```python
GraphPolicy(
    max_node_executions=200,    # run-wide execution cap
    max_total_cost=0.50,        # run-wide USD ceiling
    max_duration_seconds=120,   # wall clock, checked before AND after each node
    max_visits_per_node=25,     # cycle guard
    min_score=0.95,             # success threshold: stop as soon as reached
)
```

All budgets are **run-wide** and every enforcement records an honest, tagged `gate:<reason>` step in the artifact — a budget kill is never disguised as success.

## MCP integration

```bash
graphforge-mcp --runs-dir ~/.local/share/graphforge
# or: graphforge mcp-server
```

Tools: `list_graph_runs`, `show_graph_run`, `show_graph_topology` (topology + trail of edges actually taken + mermaid), `diff_graph_runs`, `fork_graph_run`. All backed by plain functions that work without the `mcp` package installed.

## Repository structure

- `src/graphforge/spec.py` — graph definitions + topology digest + mermaid.
- `src/graphforge/engine.py` — frontier execution, fan-out/join, cycles, resume.
- `src/graphforge/recorder.py` — opentine integration (the only writer).
- `src/graphforge/capture.py` — universal push-based framework capture.
- `src/graphforge/adapters/` — LangGraph, Burr, CrewAI Flows, LlamaIndex Workflows, pydantic-graph.
- `src/graphforge/ingest.py` — OTel GenAI span + JSONL event ingestion.
- `src/graphforge/models.py` — model adapters + fence-safe JSON node builder.
- `src/graphforge/policy.py` — run-wide budgets.
- `src/graphforge/examples.py` — deterministic offline demo graphs.
- `src/graphforge/cli.py`, `src/graphforge/mcp_server.py` — operator surfaces.
- `tests/` — fully offline suite incl. opentine-surface regression pins.

## Contributing

```bash
pytest -q
ruff check src tests
```

All generated `.tine` artifacts land under `~/.local/share/graphforge` by default.
