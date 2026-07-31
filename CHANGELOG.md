# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 — first public release

Supports `opentine>=0.2.0,<0.5`. CI exercises 0.2.0, 0.3.0 and 0.4.0 on
Python 3.11 and 3.13. Features needing a newer opentine are selected by
capability, never by version string, and degrade rather than fail.

(The project iterated through unpublished internal versions before this one.
Nothing was released to PyPI until now, so this is 0.1.0 and the earlier
numbering exists only in git history.)

### Graph engine

- `GraphSpec` / `GraphEngine` — declared graphs with static edges, routers
  (conditional edges as functions of state), join nodes with reducers, and
  cycles bounded by a policy cycle guard.
- `GraphEngine.resume` — fork a recorded artifact at any step, mutate state,
  and continue the graph. The snapshot is verified against its recorded
  digest first.
- `GraphPolicy` — run-wide budgets (node executions, cost, wall clock, cycle
  guard, success threshold), also declared as an opentine `Budget`.

### Recording any framework

- `GraphRunCapture` — a never-throw push API that records any external graph
  framework. Recorder failures cannot break the host.
- Shipped adapters: LangGraph (checkpointer wrapper + callback handler), Burr,
  CrewAI Flows, LlamaIndex Workflows, pydantic-graph V2.
- Zero-glue ingestion: raw OTLP/JSON GenAI exports, opentine `TraceEvent`
  streams, framework callback records, and JSON-lines event streams.

### Provenance

- Every node execution records the **edge that fired it**, state digests
  before and after, a key-level delta, a lossless state snapshot, and DAG
  parent links — multi-parent at joins.
- **v3 repositories** (opentine 0.3.0+): content-addressed storage with
  compare-and-swap refs, evaluation attestations, release-gate promotion,
  search and `fsck`. graphforge writes only `experiments/*`,
  `heads/graphforge/*` and `tags/*`.
- **Priced billing** — model calls priced against opentine's signed rate-card
  catalog, carrying catalog id and an explicit complete/partial status. An
  unpriceable call keeps the caller's own figure rather than being zeroed.
- **Provider usage normalization** onto opentine's mutually exclusive
  dimensions, verified case by case against opentine's own normalizers for
  OpenAI (nested and flat Responses shapes), Anthropic (including the 5m/1h
  cache split), Google and Ollama.
- **Signing**, and a `verify` command that reports integrity and signature
  separately, plus `fork_id_verified` and `tags_covered_by_signature` — tags
  are outside the signature in every opentine shipped so far, which is why
  every safety signal is also recorded as a step.
- **Per-act fork identity** (opentine 0.4.0) via `verify_fork_identity`,
  wrapping a check opentine still keeps private.
- `graph_diff` — a graph-layer comparison (node path, first divergence, edges
  taken, visit deltas, score drift) that opentine's step-level diffs cannot
  express, because a v3 event's identity includes a timestamp.

### Surfaces

- `graphforge` CLI and an MCP server. MCP refs are confined to the runs
  directory, artifacts are size-checked before loading, responses are capped,
  and mermaid identifiers are slugified. When a repository is available,
  opentine's own repository tools are composed onto the same server, without
  promotion.
