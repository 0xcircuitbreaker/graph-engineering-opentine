"""State serialization, hashing, and diffing utilities.

Everything persisted into a .tine artifact flows through these helpers so
artifacts stay bounded and deterministic regardless of what a framework
emits. Metric sanitizers keep opentine >=0.3.0's strict Step validation
happy (finite, non-negative, integer token dimensions) while remaining
harmless on 0.2.0.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

MAX_CHARS = 4000
MAX_ITEMS = 100

# UTF-16 surrogate code points have no UTF-8 spelling, so neither .tine files
# nor v3 objects can carry one; opentine refuses the whole artifact at write.
# Any JavaScript/Java producer emits them for a truncated emoji, and graph
# state comes from arbitrary frameworks — repair them here rather than lose
# the entire recorded trace to one bad character.
_SURROGATES = re.compile("[\ud800-\udfff]")
SURROGATE_REPLACEMENT = "�"

# opentine's canonical JSON (and RFC 8785) rejects integers beyond +/-(2^53-1).
MAX_SAFE_INT = 2**53 - 1

# opentine's kernel rejects canonical JSON nested deeper than 512 levels;
# stay far under it (and under Python's recursion limit) so a pathological
# state degrades to a repr string instead of failing the save.
MAX_DEPTH = 100


def clean_text(text: str) -> str:
    """Replace UTF-16 surrogates, which no opentine format can encode.

    Returns ``text`` unchanged when it is already encodable (the common
    case costs one regex search).
    """
    if _SURROGATES.search(text) is None:
        return text
    return _SURROGATES.sub(SURROGATE_REPLACEMENT, text)


_UNSAFE_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def slug(text: str, *, fallback: str = "fork", max_chars: int = 32) -> str:
    """Filesystem- and identifier-safe form of caller-supplied text.

    Run ids become file names (``{run_id}.tine``), so any caller string
    folded into one — a fork branch name, which is an MCP tool argument —
    must not be able to carry path separators or ``..`` out of the runs
    directory.
    """
    cleaned = _UNSAFE_SLUG.sub("-", clean_text(str(text)))
    # Strip leading dots so no prefix of ".." can survive as a path segment.
    cleaned = cleaned.strip("-.")[:max_chars].strip("-.")
    return cleaned or fallback


def to_jsonable(
    value: Any,
    *,
    max_chars: int | None = None,
    max_items: int | None = None,
    canonical_numbers: bool = False,
    _depth: int = 0,
) -> Any:
    """Convert an arbitrary value into JSON-serializable form.

    Optional truncation caps string length and collection size; without the
    caps the conversion is lossless for JSON-shaped data (used for state
    snapshots that must survive a resume). Non-finite floats, integers
    outside the canonical-JSON safe range, and nesting beyond ``MAX_DEPTH``
    are encoded as strings so opentine never rejects the payload at save
    time.
    """
    if _depth >= MAX_DEPTH:
        text = repr(value)
        return text[:200] + "...[depth-capped]" if len(text) > 200 else text + " [depth-capped]"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > MAX_SAFE_INT:
            return str(value)
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return repr(value)
        if canonical_numbers:
            # RFC 8785 (what opentine's v3 repository stores) has no float/int
            # distinction: 21.0 canonicalizes to 21 and -0.0 to 0. Fold the
            # same way so a digest matches whether the run round-tripped
            # through a .tine file or a v3 repository.
            if value == 0.0:
                return 0
            if value.is_integer() and abs(value) <= MAX_SAFE_INT:
                return int(value)
        return value
    if isinstance(value, str):
        value = clean_text(value)
        if max_chars is not None and len(value) > max_chars:
            return value[:max_chars] + f"...[truncated {len(value) - max_chars} chars]"
        return value
    if isinstance(value, dict):
        items = list(value.items())
        truncated = False
        if max_items is not None and len(items) > max_items:
            items = items[:max_items]
            truncated = True
        out = {
            clean_text(str(k)): to_jsonable(
                v,
                max_chars=max_chars,
                max_items=max_items,
                canonical_numbers=canonical_numbers,
                _depth=_depth + 1,
            )
            for k, v in items
        }
        if truncated:
            out["__truncated__"] = f"{len(value) - max_items} items dropped"
        return out
    if isinstance(value, list | tuple | set | frozenset):
        items = list(value)
        note = None
        if max_items is not None and len(items) > max_items:
            note = f"...[truncated {len(items) - max_items} items]"
            items = items[:max_items]
        out_list = [
            to_jsonable(
                v,
                max_chars=max_chars,
                max_items=max_items,
                canonical_numbers=canonical_numbers,
                _depth=_depth + 1,
            )
            for v in items
        ]
        if note is not None:
            out_list.append(note)
        return out_list
    text = clean_text(repr(value))
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def safe_json(value: Any, *, max_chars: int = MAX_CHARS, max_items: int = MAX_ITEMS) -> Any:
    """Bounded JSON-safe conversion for observations and framework payloads."""
    return to_jsonable(value, max_chars=max_chars, max_items=max_items)


def state_digest(state: Any) -> str:
    """Short deterministic digest of a state object (16 hex chars)."""
    canonical = json.dumps(
        to_jsonable(state, canonical_numbers=True), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def state_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    """Shallow key-level delta between two state dicts.

    Values are deliberately omitted (the full after-state snapshot lives on
    the step already); this is the quick "what moved" summary for diffing.
    """
    before = before or {}
    after = after or {}
    added = sorted(clean_text(str(k)) for k in after if k not in before)
    removed = sorted(clean_text(str(k)) for k in before if k not in after)
    changed = sorted(
        clean_text(str(k))
        for k in after
        if k in before and state_digest(after[k]) != state_digest(before[k])
    )
    return {"added": added, "removed": removed, "changed": changed}


def sane_metric(value: Any) -> float:
    """Clamp a cost/duration metric to a finite non-negative float.

    ``float()`` on a huge int raises ``OverflowError`` (not ValueError), and
    an untrusted trace can carry one, so the conversion is guarded.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return 0.0
    if not math.isfinite(number) or number < 0:
        return 0.0
    return number


#: opentine's canonical token dimensions. Anything else is a custom
#: dimension, which opentine accepts as a finite number rather than an int.
TOKEN_DIMENSIONS = frozenset(
    {"input", "output", "cache_read", "cache_write_5m", "cache_write_1h", "reasoning", "total"}
)

#: Input spellings that already contain the cache sub-buckets.
_INCLUSIVE_INPUT = frozenset(
    {"prompt_tokens", "prompttokens", "prompt_token_count", "prompttokencount"}
)

#: OpenAI-family cache spellings. Their presence means the input count is
#: inclusive whichever input spelling was used — the Responses API reports
#: ``input_tokens`` with ``cached_tokens``, and opentine subtracts there too.
#: Anthropic's ``cache_*_input_tokens`` deliberately stay out of this set.
_OPENAI_CACHE = frozenset(
    {
        "cached_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "prompt_cache_hit_tokens",
        "num_cached_tokens",
    }
)

#: Output spellings that already contain reasoning/thinking tokens. Google's
#: ``candidates_token_count`` is NOT one: it excludes thoughts.
_INCLUSIVE_OUTPUT = frozenset(
    {"completion_tokens", "completiontokens", "output_tokens", "outputtokens"}
)

#: Nested containers real SDKs put the sub-buckets in. They are flattened
#: into the scalar namespace before mapping; without this the cached and
#: reasoning counts are simply dropped.
_DETAIL_CONTAINERS = (
    "prompt_tokens_details",
    "input_tokens_details",
    "completion_tokens_details",
    "output_tokens_details",
    "cache_creation",
)

#: The OpenAI-family containers that break down an *inclusive* input count.
#: A cache bucket sourced from one of these means the sibling input total
#: already contains it — which is how opentine decides, and it is why
#: provenance matters rather than the key name alone: Anthropic reports a
#: top-level ``cache_creation_input_tokens`` whose input total EXCLUDES it, so
#: keying on the name would wrongly subtract there.
_OPENAI_INPUT_DETAILS = ("prompt_tokens_details", "input_tokens_details")

USAGE_ALIASES = {
    # OpenAI / OpenAI-compatible
    "prompt_tokens": "input",
    "prompttokens": "input",
    "input_tokens": "input",
    "inputtokens": "input",
    "completion_tokens": "output",
    "completiontokens": "output",
    "output_tokens": "output",
    "outputtokens": "output",
    "cached_tokens": "cache_read",
    "cachedtokens": "cache_read",
    "prompt_cache_hit_tokens": "cache_read",
    "num_cached_tokens": "cache_read",
    "cache_read_tokens": "cache_read",
    "cache_write_tokens": "cache_write_5m",
    "reasoning_tokens": "reasoning",
    "reasoningtokens": "reasoning",
    "total_tokens": "total",
    "totaltokens": "total",
    # Anthropic
    "cache_read_input_tokens": "cache_read",
    "cachereadinputtokens": "cache_read",
    "cache_creation_input_tokens": "cache_write_5m",
    "cachecreationinputtokens": "cache_write_5m",
    "ephemeral_5m_input_tokens": "cache_write_5m",
    "ephemeral_1h_input_tokens": "cache_write_1h",
    "cache_write_1h_tokens": "cache_write_1h",
    "thinking_tokens": "reasoning",
    # Google (the lookup lowercases, so camelCase is keyed lowercased)
    "prompt_token_count": "input",
    "prompttokencount": "input",
    "candidates_token_count": "output",
    "candidatestokencount": "output",
    "cached_content_token_count": "cache_read",
    "cachedcontenttokencount": "cache_read",
    "thoughts_token_count": "reasoning",
    "thoughtstokencount": "reasoning",
    "total_token_count": "total",
    "totaltokencount": "total",
    # Ollama (both already exclusive)
    "prompt_eval_count": "input",
    "eval_count": "output",
}


def _flatten_usage(usage: dict[Any, Any]) -> tuple[dict[str, Any], bool, set[str]]:
    """Merge known nested detail containers into the scalar namespace.

    Returns the flat mapping and whether an explicit 5m/1h cache-write split
    was found — that split takes precedence over the flat
    ``cache_creation_input_tokens`` total, which would otherwise be counted
    as both buckets.

    The split counts only when a bucket is actually non-zero. Anthropic sends
    the container with both buckets at 0 alongside a populated flat total, and
    treating its mere presence as authoritative discarded the whole
    cache-write count — an undercount of the entire cached prefix. This
    matches opentine's own ``anthropic_usage``, which falls back to the flat
    total in exactly that case.
    """
    flat: dict[str, Any] = {}
    split = False
    from_input_details: set[str] = set()
    for key, value in usage.items():
        name = str(key)
        if name.lower() in _DETAIL_CONTAINERS and isinstance(value, dict):
            inclusive_container = name.lower() in _OPENAI_INPUT_DETAILS
            for inner_key, inner in value.items():
                inner_name = str(inner_key)
                if inner_name.lower() in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
                    split = split or bool(inner)
                if inclusive_container:
                    from_input_details.add(inner_name.lower())
                flat.setdefault(inner_name, inner)
            continue
        flat[name] = value
    return flat, split, from_input_details


def sane_usage(usage: Any) -> dict[str, int | float] | None:
    """Normalize a usage mapping onto opentine's exclusive dimensions.

    Provider-native names (``prompt_tokens``, ``completion_tokens``, …) are
    mapped onto opentine's canonical token dimensions so token totals and
    cost breakdowns are non-zero, and inclusive provider counts have their
    sub-buckets subtracted so cached and reasoning tokens are not billed
    twice. Canonical token dimensions are coerced to bounded ints (0.2.0
    types usage as ``dict[str, int]`` and 0.3.0 rejects non-int token dims);
    custom dimensions keep their finite float value, which opentine permits
    and truncating would destroy.

    Dimension NAMES are cleaned too: usage is the one mapping written into a
    step whose keys would otherwise skip :func:`to_jsonable`, so a surrogate
    in a dimension name would make opentine refuse the whole artifact.
    """
    if not isinstance(usage, dict):
        return None
    flat, explicit_cache_split, from_input_details = _flatten_usage(usage)

    out: dict[str, int | float] = {}
    inclusive_input = False
    inclusive_output = False
    for raw_key, value in flat.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        try:
            number = float(value)
        except (OverflowError, ValueError):  # pragma: no cover - defensive
            continue
        if not math.isfinite(number) or number < 0 or number > MAX_SAFE_INT:
            continue
        key = clean_text(str(raw_key)).strip()
        if not key:
            continue  # opentine rejects an empty dimension name
        lowered = key.lower()
        name = USAGE_ALIASES.get(lowered, key)
        if name in TOKEN_DIMENSIONS:
            if lowered in _INCLUSIVE_INPUT:
                inclusive_input = True
            if lowered in _OPENAI_CACHE:
                # OpenAI reports an inclusive input whichever spelling it used.
                inclusive_input = True
            if lowered in from_input_details and name in (
                "cache_read",
                "cache_write_5m",
                "cache_write_1h",
            ):
                # A cache bucket broken out of an OpenAI input-details
                # container is part of the sibling input total, whatever it is
                # spelled. Without this, `input_tokens` alongside a nested
                # `cache_creation_input_tokens` over-reported input by the
                # whole cache-write count.
                inclusive_input = True
            if lowered in _INCLUSIVE_OUTPUT:
                inclusive_output = True
            if (
                explicit_cache_split
                and lowered in ("cache_creation_input_tokens", "cachecreationinputtokens")
            ):
                # The nested 5m/1h split is authoritative; the flat total
                # would otherwise land in cache_write_5m on top of it.
                continue
            merged = int(number)
            # Two spellings can land on one dimension; keep the larger rather
            # than letting declaration order decide.
            out[name] = max(out.get(name, 0), merged) if name in out else merged
        elif name not in out:
            out[name] = number

    if inclusive_input and "input" in out:
        nested = sum(
            int(out.get(dim, 0)) for dim in ("cache_read", "cache_write_5m", "cache_write_1h")
        )
        out["input"] = max(int(out["input"]) - nested, 0)
    if inclusive_output and "output" in out:
        out["output"] = max(int(out["output"]) - int(out.get("reasoning", 0)), 0)
    # Drop zero-valued canonical dimensions, matching opentine's own compact
    # serialization: a zeroed nested detail container would otherwise emit a
    # bucket the provider never reported.
    out = {k: v for k, v in out.items() if k not in TOKEN_DIMENSIONS or v}
    return out or None


def sane_score(value: Any) -> float | None:
    """Normalize a score to a finite float (None for bools and non-finite)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number
