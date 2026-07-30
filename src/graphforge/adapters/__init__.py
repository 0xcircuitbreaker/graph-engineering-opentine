"""Framework adapters that feed a GraphRunCapture from external graph engines.

Shipped adapters (all import-safe without their framework installed):

- ``langgraph``      — RecordingCheckpointer (checkpointer wrapper) +
  RecordingCallbackHandler (node/model/tool capture via callbacks)
- ``burr``           — TineBurrHook (lifecycle hooks; one ``with_hooks`` call)
- ``crewai``         — TineFlowListener (event-bus listener, per-flow captures)
- ``llamaindex``     — record_checkpoints (WorkflowCheckpointer conversion)
- ``pydantic_graph`` — record_graph_run (graph.iter() driver with fork_stack
  lineage)

Everything else: use :class:`graphforge.GraphRunCapture` directly, or the
zero-glue paths in :mod:`graphforge.ingest` (OTel GenAI spans, JSONL events).
"""

from .burr import TineBurrHook
from .crewai import TineFlowListener
from .langgraph import RecordingCallbackHandler, RecordingCheckpointer
from .llamaindex import record_checkpoints
from .pydantic_graph import record_graph_run

__all__ = [
    "RecordingCallbackHandler",
    "RecordingCheckpointer",
    "TineBurrHook",
    "TineFlowListener",
    "record_checkpoints",
    "record_graph_run",
]
