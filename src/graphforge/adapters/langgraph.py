"""LangGraph adapter: record LangGraph runs into opentine with zero graph changes.

Two attach points, combinable:

- :class:`RecordingCheckpointer` wraps any ``BaseCheckpointSaver`` — one line
  at ``compile(checkpointer=...)`` — and records every superstep checkpoint
  (the forkable-point table: thread_id / checkpoint_ns / checkpoint_id).
- :class:`RecordingCallbackHandler` goes into ``config={"callbacks": [...]}``
  at invoke time and records node executions (``langgraph_node`` metadata),
  the edge/trigger that fired each node, and model/tool calls with token
  usage.

Both are import-safe without langgraph/langchain installed (the classes
degrade to duck-typed bases), so this module never forces the dependency.
"""

from __future__ import annotations

import logging
from typing import Any

from ..capture import GraphRunCapture
from ..state import safe_json

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised only with langgraph installed
    from langgraph.checkpoint.base import BaseCheckpointSaver as _SaverBase

    HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover
    _SaverBase = object
    HAS_LANGGRAPH = False

try:  # pragma: no cover - exercised only with langchain installed
    from langchain_core.callbacks import BaseCallbackHandler as _HandlerBase

    HAS_LANGCHAIN = True
except ImportError:  # pragma: no cover
    _HandlerBase = object
    HAS_LANGCHAIN = False


class RecordingCheckpointer(_SaverBase):  # type: ignore[misc,valid-type]
    """Delegating checkpointer wrapper that records every checkpoint."""

    def __init__(self, capture: GraphRunCapture, inner: Any = None) -> None:
        if HAS_LANGGRAPH and isinstance(_SaverBase, type) and _SaverBase is not object:
            try:
                super().__init__()
            except TypeError:  # pragma: no cover - saver bases with kwargs
                pass
        self.capture = capture
        self.inner = inner

    # -- recording ---------------------------------------------------------

    def _record(self, config: Any, checkpoint: Any, metadata: Any) -> None:
        try:
            conf = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
            meta = dict(metadata) if isinstance(metadata, dict) else {}
            values = checkpoint.get("channel_values") if isinstance(checkpoint, dict) else None
            self.capture.checkpoint(
                str(checkpoint.get("id", "")) if isinstance(checkpoint, dict) else "",
                state=values if isinstance(values, dict) else None,
                parent_checkpoint_id=conf.get("checkpoint_id"),
                source=str(meta.get("source", "")),
                step=meta.get("step"),
            )
        except Exception:  # noqa: BLE001 - recording must never break the graph
            logger.debug("graphforge langgraph checkpoint record failed", exc_info=True)

    # -- BaseCheckpointSaver surface (sync) --------------------------------

    def put(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        result = (
            self.inner.put(config, checkpoint, metadata, new_versions) if self.inner else config
        )
        self._record(config, checkpoint, metadata)
        return result

    def put_writes(self, config: Any, writes: Any, task_id: Any, task_path: str = "") -> Any:
        if self.inner:
            return self.inner.put_writes(config, writes, task_id, task_path)
        return None

    def get_tuple(self, config: Any) -> Any:
        return self.inner.get_tuple(config) if self.inner else None

    def list(self, config: Any, **kwargs: Any) -> Any:
        return self.inner.list(config, **kwargs) if self.inner else iter(())

    def delete_thread(self, thread_id: Any) -> Any:
        if self.inner and hasattr(self.inner, "delete_thread"):
            return self.inner.delete_thread(thread_id)
        return None

    def get_next_version(self, current: Any, channel: Any = None) -> Any:
        if self.inner and hasattr(self.inner, "get_next_version"):
            return self.inner.get_next_version(current, channel)
        return (current or 0) + 1 if isinstance(current, int | type(None)) else current

    # -- BaseCheckpointSaver surface (async) -------------------------------

    async def aput(self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any) -> Any:
        if self.inner and hasattr(self.inner, "aput"):
            result = await self.inner.aput(config, checkpoint, metadata, new_versions)
        else:
            result = self.put(config, checkpoint, metadata, new_versions)
            return result
        self._record(config, checkpoint, metadata)
        return result

    async def aput_writes(self, config: Any, writes: Any, task_id: Any, task_path: str = "") -> Any:
        if self.inner and hasattr(self.inner, "aput_writes"):
            return await self.inner.aput_writes(config, writes, task_id, task_path)
        return self.put_writes(config, writes, task_id, task_path)

    async def aget_tuple(self, config: Any) -> Any:
        if self.inner and hasattr(self.inner, "aget_tuple"):
            return await self.inner.aget_tuple(config)
        return self.get_tuple(config)

    async def alist(self, config: Any, **kwargs: Any) -> Any:
        if self.inner and hasattr(self.inner, "alist"):
            async for item in self.inner.alist(config, **kwargs):
                yield item
            return
        for item in self.list(config, **kwargs):
            yield item

    async def adelete_thread(self, thread_id: Any) -> Any:
        if self.inner and hasattr(self.inner, "adelete_thread"):
            return await self.inner.adelete_thread(thread_id)
        return self.delete_thread(thread_id)


class RecordingCallbackHandler(_HandlerBase):  # type: ignore[misc,valid-type]
    """LangChain callback handler recording LangGraph node + model executions.

    ``price_table`` maps model name -> (usd_per_million_input_tokens,
    usd_per_million_output_tokens) for cost computation; unknown models
    record usage with zero cost.
    """

    raise_error = False
    run_inline = True

    def __init__(
        self,
        capture: GraphRunCapture,
        *,
        price_table: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.capture = capture
        self.price_table = price_table or {}
        self._runs: dict[str, dict[str, Any]] = {}

    # -- helpers ------------------------------------------------------------

    def _node_entry_for(self, parent_run_id: Any) -> tuple[str, str] | None:
        """Walk the parent chain to the enclosing node: (node name, its run key)."""
        cursor = str(parent_run_id) if parent_run_id else None
        for _ in range(32):
            if cursor is None:
                return None
            entry = self._runs.get(cursor)
            if entry is None:
                return None
            if entry.get("node"):
                return str(entry["node"]), cursor
            cursor = entry.get("parent")
        return None

    def _cost(self, model: str, usage: dict[str, int]) -> float:
        rates = self.price_table.get(model)
        if not rates:
            return 0.0
        return (
            usage.get("input", 0) * rates[0] + usage.get("output", 0) * rates[1]
        ) / 1_000_000

    # -- chain (node) lifecycle ---------------------------------------------

    def on_chain_start(
        self,
        serialized: Any,
        inputs: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: Any = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        try:
            meta = metadata if isinstance(metadata, dict) else {}
            node = meta.get("langgraph_node")
            if node is not None:
                # langgraph_node metadata is inherited by every Runnable
                # invoked INSIDE the node; only the outermost run is the
                # node execution.
                enclosing = self._node_entry_for(parent_run_id)
                if enclosing is not None and enclosing[0] == str(node):
                    node = None
            self._runs[str(run_id)] = {
                "node": node,
                "parent": str(parent_run_id) if parent_run_id else None,
            }
            if not node:
                return
            triggers = meta.get("langgraph_triggers")
            edge = None
            if triggers:
                trigger = triggers[0] if isinstance(triggers, list | tuple) else triggers
                edge = {"kind": "trigger", "source": str(trigger), "label": ""}
            state = inputs if isinstance(inputs, dict) else None
            self.capture.node_start(
                str(node),
                state=safe_json(state) if state is not None else None,
                edge=edge,
                key=str(run_id),
            )
        except Exception:  # noqa: BLE001
            logger.debug("graphforge langgraph on_chain_start failed", exc_info=True)

    def on_chain_end(self, outputs: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        try:
            entry = self._runs.pop(str(run_id), None)
            if not entry or not entry.get("node"):
                return
            update = outputs if isinstance(outputs, dict) else None
            self.capture.node_end(
                str(entry["node"]),
                update=safe_json(update) if update is not None else None,
                key=str(run_id),
            )
        except Exception:  # noqa: BLE001
            logger.debug("graphforge langgraph on_chain_end failed", exc_info=True)

    def on_chain_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        try:
            entry = self._runs.pop(str(run_id), None)
            if not entry or not entry.get("node"):
                return
            self.capture.node_end(
                str(entry["node"]), error=str(error), key=str(run_id)
            )
        except Exception:  # noqa: BLE001
            logger.debug("graphforge langgraph on_chain_error failed", exc_info=True)

    # -- model lifecycle ------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: Any,
        messages: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        self._model_start(serialized, messages, run_id, parent_run_id, metadata, kwargs)

    def on_llm_start(
        self,
        serialized: Any,
        prompts: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        self._model_start(serialized, prompts, run_id, parent_run_id, metadata, kwargs)

    def _model_start(
        self,
        serialized: Any,
        prompt: Any,
        run_id: Any,
        parent_run_id: Any,
        metadata: Any,
        kwargs: dict[str, Any],
    ) -> None:
        try:
            meta = metadata if isinstance(metadata, dict) else {}
            invocation = kwargs.get("invocation_params") or {}
            model = meta.get("ls_model_name") or invocation.get("model_name") or invocation.get(
                "model"
            )
            if not model and isinstance(serialized, dict):
                model = serialized.get("name", "")
            self._runs[str(run_id)] = {
                "node": None,
                "parent": str(parent_run_id) if parent_run_id else None,
                "model": str(model or ""),
                "prompt": safe_json(prompt),
            }
        except Exception:  # noqa: BLE001
            logger.debug("graphforge langgraph model_start failed", exc_info=True)

    def on_llm_end(self, response: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        try:
            entry = self._runs.pop(str(run_id), None) or {}
            usage = _extract_usage(response)
            model = entry.get("model", "")
            enclosing = self._node_entry_for(entry.get("parent"))
            node, node_key = enclosing if enclosing else (None, None)
            self.capture.model_call(
                model=model,
                prompt=entry.get("prompt"),
                response=_extract_text(response),
                usage=usage,
                cost=self._cost(model, usage),
                node_id=node,
                key=node_key,
            )
        except Exception:  # noqa: BLE001
            logger.debug("graphforge langgraph on_llm_end failed", exc_info=True)

    def on_llm_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        self._runs.pop(str(run_id), None)

    # -- tool lifecycle -------------------------------------------------------

    def on_tool_start(
        self,
        serialized: Any,
        input_str: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        try:
            name = serialized.get("name", "tool") if isinstance(serialized, dict) else "tool"
            self._runs[str(run_id)] = {
                "node": None,
                "parent": str(parent_run_id) if parent_run_id else None,
                "tool": str(name),
                "args": safe_json(input_str),
            }
        except Exception:  # noqa: BLE001
            logger.debug("graphforge langgraph on_tool_start failed", exc_info=True)

    def on_tool_end(self, output: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        try:
            entry = self._runs.pop(str(run_id), None) or {}
            if "tool" not in entry:
                return
            enclosing = self._node_entry_for(entry.get("parent"))
            node, node_key = enclosing if enclosing else (None, None)
            self.capture.tool_call(
                entry["tool"],
                args=entry.get("args"),
                result=safe_json(output),
                node_id=node,
                key=node_key,
            )
        except Exception:  # noqa: BLE001
            logger.debug("graphforge langgraph on_tool_end failed", exc_info=True)

    def on_tool_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        try:
            entry = self._runs.pop(str(run_id), None) or {}
            if "tool" not in entry:
                return
            self.capture.tool_call(entry["tool"], args=entry.get("args"), error=str(error))
        except Exception:  # noqa: BLE001
            logger.debug("graphforge langgraph on_tool_error failed", exc_info=True)


def _extract_usage(response: Any) -> dict[str, int]:
    """Pull token usage out of an LLMResult across provider shapes.

    Providers report usage once per request; with n>1 candidate generations
    the same usage_metadata appears on every candidate, so only the first
    occurrence is taken (summing would multiply the real count).
    """
    usage: dict[str, int] = {}
    try:
        generations = getattr(response, "generations", None) or []
        for group in generations:
            for gen in group:
                message = getattr(gen, "message", None)
                meta = getattr(message, "usage_metadata", None)
                if isinstance(meta, dict) and meta:
                    usage["input"] = int(meta.get("input_tokens", 0) or 0)
                    usage["output"] = int(meta.get("output_tokens", 0) or 0)
                    break
            if usage:
                break
        if not usage:
            llm_output = getattr(response, "llm_output", None) or {}
            token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            if isinstance(token_usage, dict) and token_usage:
                usage["input"] = int(token_usage.get("prompt_tokens", 0) or 0)
                usage["output"] = int(token_usage.get("completion_tokens", 0) or 0)
    except Exception:  # noqa: BLE001
        logger.debug("graphforge langgraph usage extraction failed", exc_info=True)
    return usage


def _extract_text(response: Any) -> str:
    try:
        generations = getattr(response, "generations", None) or []
        parts: list[str] = []
        for group in generations:
            for gen in group:
                text = getattr(gen, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    except Exception:  # noqa: BLE001
        return ""
