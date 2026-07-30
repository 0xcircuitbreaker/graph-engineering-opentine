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
from typing import Any

MAX_CHARS = 4000
MAX_ITEMS = 100

# opentine's canonical JSON (and RFC 8785) rejects integers beyond +/-(2^53-1).
MAX_SAFE_INT = 2**53 - 1

# opentine's kernel rejects canonical JSON nested deeper than 512 levels;
# stay far under it (and under Python's recursion limit) so a pathological
# state degrades to a repr string instead of failing the save.
MAX_DEPTH = 100


def to_jsonable(
    value: Any,
    *,
    max_chars: int | None = None,
    max_items: int | None = None,
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
        return value
    if isinstance(value, str):
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
            str(k): to_jsonable(v, max_chars=max_chars, max_items=max_items, _depth=_depth + 1)
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
            to_jsonable(v, max_chars=max_chars, max_items=max_items, _depth=_depth + 1)
            for v in items
        ]
        if note is not None:
            out_list.append(note)
        return out_list
    text = repr(value)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text


def safe_json(value: Any, *, max_chars: int = MAX_CHARS, max_items: int = MAX_ITEMS) -> Any:
    """Bounded JSON-safe conversion for observations and framework payloads."""
    return to_jsonable(value, max_chars=max_chars, max_items=max_items)


def state_digest(state: Any) -> str:
    """Short deterministic digest of a state object (16 hex chars)."""
    canonical = json.dumps(to_jsonable(state), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def state_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    """Shallow key-level delta between two state dicts.

    Values are deliberately omitted (the full after-state snapshot lives on
    the step already); this is the quick "what moved" summary for diffing.
    """
    before = before or {}
    after = after or {}
    added = sorted(k for k in after if k not in before)
    removed = sorted(k for k in before if k not in after)
    changed = sorted(
        k for k in after if k in before and state_digest(after[k]) != state_digest(before[k])
    )
    return {"added": added, "removed": removed, "changed": changed}


def sane_metric(value: Any) -> float:
    """Clamp a cost/duration metric to a finite non-negative float."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return 0.0
    return number


def sane_usage(usage: Any) -> dict[str, int] | None:
    """Normalize a usage mapping to non-negative integer dimensions.

    opentine 0.2.0 types usage as dict[str, int] and 0.3.0 validates token
    dimensions as bool-free integers within the canonical-JSON safe range,
    so bounded integers are the safe intersection.
    """
    if not isinstance(usage, dict):
        return None
    out: dict[str, int] = {}
    for key, value in usage.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float) and math.isfinite(float(value)) and value >= 0:
            out[str(key)] = min(int(value), MAX_SAFE_INT)
    return out or None


def sane_score(value: Any) -> float | None:
    """Normalize a score to a finite float (None for bools and non-finite)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number
