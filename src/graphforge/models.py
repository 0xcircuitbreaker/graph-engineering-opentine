"""Model adapter layer: run graph nodes off any chat-completions endpoint.

Deterministic adapters keep the demos and test suite fully offline; the
OpenAI-compatible adapter covers real providers with stdlib urllib (no SDK
dependency). ``extract_json`` handles the fenced-JSON forms real models emit
(including standard ```` ```json ```` fences).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .engine import GraphNodeContext, GraphNodeResult
from .state import sane_score

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*[ \t]*\r?\n(.*?)```", re.DOTALL)


class GraphModelOutput:
    def __init__(
        self,
        text: str,
        *,
        cost: float = 0.0,
        duration: float = 0.0,
        usage: dict[str, int] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.text = text
        self.cost = cost
        self.duration = duration
        self.usage = usage or {}
        self.meta = meta or {}


class GraphModelAdapter(Protocol):
    def complete(self, prompt: str) -> GraphModelOutput: ...


@dataclass
class StaticModelAdapter:
    """Deterministic adapter for tests/demos.

    ``script`` responses are consumed in order (useful for cyclic graphs);
    once exhausted — or when only ``text`` is set — every call returns
    ``text`` (or the last scripted response).
    """

    text: str = ""
    script: list[str] = field(default_factory=list)
    cost: float = 0.0
    duration: float = 0.0
    _cursor: int = field(default=0, repr=False)

    def complete(self, prompt: str) -> GraphModelOutput:
        if self.script:
            index = min(self._cursor, len(self.script) - 1)
            self._cursor += 1
            text = self.script[index]
        else:
            text = self.text
        return GraphModelOutput(text, cost=self.cost, duration=self.duration)

    def reset(self) -> None:
        """Rewind the script cursor (call between engine runs that share the adapter)."""
        self._cursor = 0


#: Ceiling on a single model response body. Far above any real completion,
#: low enough that a misbehaving endpoint cannot exhaust memory.
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _check_endpoint(endpoint: str) -> None:
    """Only send the API key over HTTPS, or plain HTTP to this machine.

    ``urllib`` also opens ``file://`` and ``ftp://`` URLs, and a plain-HTTP
    remote endpoint would put the bearer token on the wire unencrypted.
    """
    parsed = urllib.parse.urlsplit(str(endpoint))
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if scheme == "https" and host:
        return
    if scheme == "http" and host in _LOOPBACK_HOSTS:
        return
    raise ValueError(
        f"refusing model endpoint {endpoint!r}: use https://, or http:// only for "
        "localhost, 127.0.0.1 or ::1"
    )

@dataclass
class OpenAICompatAdapter:
    """Chat-completions adapter for any OpenAI-protocol endpoint.

    The API key is resolved at call time (constructor never raises), from
    the ``api_key`` field or the OPENAI_API_KEY environment variable.
    """

    model: str = "gpt-4o-mini"
    api_key: str | None = None
    endpoint: str = "https://api.openai.com/v1/chat/completions"
    timeout: float = 30.0
    temperature: float = 0.2
    max_tokens: int | None = None
    cost_per_million_input: float | None = None
    cost_per_million_output: float | None = None

    def complete(self, prompt: str) -> GraphModelOutput:
        _check_endpoint(self.endpoint)
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OpenAICompatAdapter needs an API key (api_key field or OPENAI_API_KEY env)"
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise RuntimeError(
                f"model response exceeds {_MAX_RESPONSE_BYTES} bytes; refusing to parse it"
            )
        body = json.loads(raw.decode("utf-8"))
        duration = time.monotonic() - started
        choices = body.get("choices") or []
        first = choices[0] if choices and isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        text = message.get("content") or ""
        usage_raw = body.get("usage", {}) or {}
        prompt_tokens = int(usage_raw.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage_raw.get("completion_tokens", 0) or 0)
        cost = 0.0
        if self.cost_per_million_input is not None and self.cost_per_million_output is not None:
            cost = (
                prompt_tokens * self.cost_per_million_input
                + completion_tokens * self.cost_per_million_output
            ) / 1_000_000
        return GraphModelOutput(
            text,
            cost=cost,
            duration=duration,
            usage={"input": prompt_tokens, "output": completion_tokens},
            meta={"model": body.get("model", self.model)},
        )


def _first_balanced_object(text: str) -> str | None:
    """Return the first brace-balanced ``{...}`` substring (string-aware)."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output.

    Tries fenced blocks (```json / ``` / any language tag), the raw text,
    the first brace-balanced object, and finally the first-``{`` to
    last-``}`` substring. Raises ValueError when nothing parses to a dict.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("no JSON object found in empty text")
    candidates = [match.strip() for match in _FENCE_RE.findall(text) if match.strip()]
    candidates.append(text.strip())
    balanced = _first_balanced_object(text)
    if balanced:
        candidates.append(balanced)
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("no JSON object found in text")


def _clean_score(value: Any) -> float | None:
    return sane_score(value)


def build_json_model_node(
    *,
    adapter: GraphModelAdapter,
    prompt_fn: Callable[[GraphNodeContext], str],
    fallback: Callable[[GraphNodeContext, Exception], GraphNodeResult] | None = None,
) -> Callable[[GraphNodeContext], GraphNodeResult]:
    """Build a node function from a model adapter and a prompt template.

    The model is asked for a JSON object shaped::

        {"observation": str, "update": {..state keys..}, "score": float,
         "stop": bool, "goto": str | [str], "metadata": {...}}

    Parse or transport failures go to ``fallback`` (default: stop the run
    with the error recorded in outputs).
    """

    def default_fallback(ctx: GraphNodeContext, error: Exception) -> GraphNodeResult:
        return GraphNodeResult(
            observation=f"model step failed: {error}",
            stop=True,
            outputs={"error": str(error)},
        )

    handler = fallback or default_fallback

    def node_fn(ctx: GraphNodeContext) -> GraphNodeResult:
        prompt = prompt_fn(ctx)
        try:
            output = adapter.complete(prompt)
            data = extract_json(output.text)
        except Exception as exc:  # noqa: BLE001 - surfaced through the fallback
            return handler(ctx, exc)
        update = data.get("update")
        goto = data.get("goto")
        if isinstance(goto, list):
            goto = [str(g) for g in goto]
        elif goto is not None:
            goto = str(goto)
        return GraphNodeResult(
            observation=str(data.get("observation", "")),
            update=dict(update) if isinstance(update, dict) else {},
            score=_clean_score(data.get("score")),
            stop=bool(data.get("stop", False)),
            goto=goto,
            outputs={
                "model": output.meta.get("model", ""),
                "metadata": data.get("metadata", {}),
            },
            cost=output.cost,
            duration=output.duration,
            usage=output.usage,
        )

    return node_fn
