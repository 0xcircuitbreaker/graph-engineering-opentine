# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.3.0 — 2026-07-31

Supports `opentine>=0.2.0,<0.5`. CI exercises both ends of that range;
features needing a newer opentine degrade rather than fail, and are selected
by capability probe, never by version string.

### Added

- **v3 repository backend** (`graphforge.repo_backend`) — store graph runs as
  content-addressed objects with compare-and-swap refs, evaluation
  attestations, release-gate promotion, search, and `fsck`, instead of one
  `.tine` file per run. Fan-in lineage survives the round trip.
- **Graph-layer diff** (`graphforge.graph_diff`) — compare two runs by node
  path, first divergence, edges taken, visit counts and score drift. opentine's
  own diffs cannot express this: a v3 event's identity includes a timestamp, so
  two behaviourally identical runs share zero events.
- **Signed-catalog pricing** (`graphforge.pricing`) — price a model call
  against opentine's signed rate cards and record the catalog id, rate-card id
  and an explicit complete/partial status, instead of an unattributed float.
- **Provider usage normalization** — map provider-native token counts onto
  opentine's mutually exclusive dimensions, correcting the inclusive counts
  OpenAI, Anthropic, Google and Ollama each report differently.
- **Artifact signing** — `GraphRecorder.save(sign_key=...)`, and a `verify`
  command that reports integrity and signature separately.
- **Per-act fork identity** (opentine 0.4.0) — forks defer to opentine, which
  mints a unique id and records a verifiable basis. On older opentine,
  graphforge supplies its own unique id, because the native id collides.
- **Framework adapters** — LangGraph, Burr, CrewAI Flows, LlamaIndex
  Workflows, pydantic-graph.
- **Zero-glue ingestion** — OTLP/JSON GenAI exports, opentine `TraceEvent`
  streams, framework callback records, and JSON-lines event streams.

### Fixed

- An unpaired UTF-16 surrogate anywhere in graph state made opentine refuse
  the artifact and lose the entire recorded trace. Any JavaScript or Java
  producer emits one for a truncated emoji.
- Resume from a repository-loaded run silently failed for any state holding a
  float with an integral value: RFC 8785 canonicalizes `21.0` to `21`, so the
  recorded state digest stopped matching.
- A join fed twice from the same node recorded duplicate parent ids, which a
  v3 repository rejects permanently — the run saved as a file and could then
  never be stored.
- A chained join could fire with partial arrivals and execute twice.
- Cost was pooled into one anonymous bucket; `cost_breakdown().by_model` now
  attributes it. Provider-native usage names were passed through verbatim, so
  a run recorded from an OpenAI-shaped payload reported zero tokens.
- Router and join-reducer exceptions, non-finite scores, and un-deepcopyable
  state each destroyed the whole trace; they now record an error step and save.
- The MCP read path could read artifacts outside the runs directory, resolved
  a run-id prefix by fully parsing every artifact, and emitted mermaid that an
  ingested node id or edge label could break out of.

### Security

- MCP refs are confined to the runs directory (traversal, absolute paths and
  symlink escapes all rejected), artifacts are size-checked before loading,
  and responses are capped.
- A fork branch name supplied over MCP can no longer escape the runs
  directory through the artifact filename.

## 0.2.0 — 2026-07-30

### Added

- Universal framework adapters and zero-glue ingestion.

## 0.1.0 — 2026-07-29

### Added

- Initial graph engineering toolkit: `GraphSpec`, `GraphEngine`,
  `GraphRecorder`, `GraphRunCapture`, `GraphPolicy`, CLI and MCP server.
