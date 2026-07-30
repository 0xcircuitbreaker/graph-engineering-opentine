"""Framework adapters that feed a GraphRunCapture from external graph engines.

Shipped: LangGraph (checkpointer wrapper + callback handler). The universal
:class:`graphforge.GraphRunCapture` surface covers everything else — Burr
lifecycle hooks, CrewAI event listeners, LlamaIndex WorkflowCheckpointer,
pydantic-graph iteration — with a few lines of glue (see README adapters
section for sketches).
"""

from .langgraph import RecordingCallbackHandler, RecordingCheckpointer

__all__ = ["RecordingCallbackHandler", "RecordingCheckpointer"]
