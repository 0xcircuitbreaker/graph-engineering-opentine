"""graphforge — graph engineering with opentine-native provenance.

Every node execution in a graph run becomes a recorded step in a
content-addressed ``.tine`` artifact: edges taken are first-class, joins get
multi-parent lineage, state snapshots make any step a fork/resume point, and
artifacts verify, diff, and replay through standard opentine tooling.
"""

from ._version import __version__
from .capture import GraphRunCapture
from .engine import (
    BestSnapshot,
    GraphEngine,
    GraphExecutionError,
    GraphExecutionResult,
    GraphNodeContext,
    GraphNodeResult,
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
from .recorder import DEFAULT_RUNS_DIR, GraphRecorder
from .spec import END, EdgeSpec, GraphSpec, GraphValidationError, NodeSpec

__all__ = [
    "DEFAULT_RUNS_DIR",
    "END",
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
    "StaticModelAdapter",
    "build_json_model_node",
    "extract_json",
    "__version__",
]
