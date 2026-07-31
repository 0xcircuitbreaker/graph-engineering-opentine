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

#: Spellings whose INPUT count already contains the cache sub-buckets.
#: OpenAI reports ``prompt_tokens`` inclusive of ``cached_tokens``; Anthropic
#: reports ``input_tokens`` exclusive of them. opentine's dimensions are
#: exclusive, so an inclusive spelling must have its sub-buckets subtracted
#: or every cached token is counted twice — and priced twice.
_INCLUSIVE_INPUT = frozenset({"prompt_tokens", "prompttokens"})

#: Spellings whose OUTPUT count already contains reasoning/thinking tokens.
#: Both OpenAI (``completion_tokens``) and Anthropic (``output_tokens``) do.
_INCLUSIVE_OUTPUT = frozenset(
    {"completion_tokens", "completiontokens", "output_tokens", "outputtokens"}
)

#: Provider- and framework-native spellings mapped onto those dimensions.
#: Without this, a run recorded from an OpenAI/LangChain-shaped payload
#: reports total_tokens == 0 and an all-zero cost breakdown, because opentine
#: only counts dimensions it recognizes.
USAGE_ALIASES = {
    "prompt_tokens": "input",
    "prompttokens": "input",
    "input_tokens": "input",
    "inputtokens": "input",
    "completion_tokens": "output",
    "completiontokens": "output",
    "output_tokens": "output",
    "outputtokens": "output",
    "cache_read_input_tokens": "cache_read",
    "cachereadinputtokens": "cache_read",
    "cache_read_tokens": "cache_read",
    "cached_tokens": "cache_read",
    "cache_creation_input_tokens": "cache_write_5m",
    "cachecreationinputtokens": "cache_write_5m",
    "cache_write_tokens": "cache_write_5m",
    "reasoning_tokens": "reasoning",
    "reasoningtokens": "reasoning",
    "total_tokens": "total",
    "totaltokens": "total",
}


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
    out: dict[str, int | float] = {}
    inclusive_input = False
    inclusive_output = False
    for raw_key, value in usage.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        number = float(value)
        if not math.isfinite(number) or number < 0:
            continue
        key = clean_text(str(raw_key)).strip()
        if not key:
            continue  # opentine rejects an empty dimension name outright
        lowered = key.lower()
        name = USAGE_ALIASES.get(lowered, key)
        if number > MAX_SAFE_INT:
            # Out of canonical-JSON range. Clamping would turn a plainly
            # bogus count into a plausible one inside the digested step body;
            # opentine's own importer discards such a dimension, so do that.
            continue
        if name in TOKEN_DIMENSIONS:
            if lowered in _INCLUSIVE_INPUT:
                inclusive_input = True
            if lowered in _INCLUSIVE_OUTPUT:
                inclusive_output = True
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
    return out or None


def sane_score(value: Any) -> float | None:
    """Normalize a score to a finite float (None for bools and non-finite)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number
