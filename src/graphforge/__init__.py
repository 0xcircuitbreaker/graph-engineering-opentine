"""graphforge — graph engineering with opentine-native provenance.

Every node execution in a graph run becomes a recorded step in a
content-addressed ``.tine`` artifact: edges taken are first-class, joins get
multi-parent lineage, state snapshots make any step a fork/resume point, and
artifacts verify, diff, and replay through standard opentine tooling.
"""

from ._version import __version__
from .capture import GraphRunCapture
from .diffing import graph_diff
from .engine import (
    BestSnapshot,
    GraphEngine,
    GraphExecutionError,
    GraphExecutionResult,
    GraphNodeContext,
    GraphNodeResult,
)
from .ingest import (
    FRAMEWORK_DIALECTS,
    ingest_framework,
    ingest_jsonl,
    ingest_otel_spans,
    ingest_trace_events,
    normalize_otlp,
)
from .models import (
    GraphModelAdapter,
    GraphModelOutput,
    OpenAICompatAdapter,
    StaticModelAdapter,
    build_json_model_node,
    extract_json,
)
from .policy import GraphPolicy, GraphPolicyViolation
from .recorder import DEFAULT_RUNS_DIR, GraphRecorder, verify_fork_identity
from .repo_backend import (
    RepoUnavailableError,
    candidates,
    evaluate,
    load_from_repo,
    open_repo,
    promote,
    repo_available,
    repo_status,
    save_to_repo,
)
from .spec import END, EdgeSpec, GraphSpec, GraphValidationError, NodeSpec

__all__ = [
    "DEFAULT_RUNS_DIR",
    "END",
    "FRAMEWORK_DIALECTS",
    "BestSnapshot",
    "EdgeSpec",
    "GraphEngine",
    "GraphExecutionError",
    "GraphExecutionResult",
    "GraphModelAdapter",
    "GraphModelOutput",
    "GraphNodeContext",
    "GraphNodeResult",
    "GraphPolicy",
    "GraphPolicyViolation",
    "GraphRecorder",
    "GraphRunCapture",
    "GraphSpec",
    "GraphValidationError",
    "NodeSpec",
    "OpenAICompatAdapter",
    "RepoUnavailableError",
    "StaticModelAdapter",
    "build_json_model_node",
    "candidates",
    "evaluate",
    "extract_json",
    "graph_diff",
    "ingest_framework",
    "ingest_jsonl",
    "ingest_otel_spans",
    "ingest_trace_events",
    "load_from_repo",
    "normalize_otlp",
    "open_repo",
    "promote",
    "repo_available",
    "repo_status",
    "save_to_repo",
    "verify_fork_identity",
    "__version__",
]
