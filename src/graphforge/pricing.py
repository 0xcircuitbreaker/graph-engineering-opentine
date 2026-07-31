"""Priced model calls: turn raw token counts into opentine billing records.

graphforge's own cost fields are plain floats a caller supplied. opentine
0.3.0 can price a call against a signed rate-card catalog and return a
``BillingResult`` carrying the catalog id, catalog hash, rate-card id, and
an explicit ``status`` (``complete`` / ``partial`` / ``unknown`` /
``unmetered``). Recording that instead of a bare float is what lets a reader
tell a real charge from a guess.

Every import is lazy and guarded: opentine 0.2.0 has no billing module, and
``add_step(billing=...)`` raises ``TypeError`` there. When pricing is
unavailable this module returns ``None`` and callers fall back to the plain
cost float, so behaviour on 0.2.0 is exactly what it was.
"""

from __future__ import annotations

from typing import Any

from .state import sane_usage

#: opentine's canonical token dimensions, in Usage's constructor order.
_USAGE_FIELDS = (
    "input",
    "output",
    "cache_read",
    "cache_write_5m",
    "cache_write_1h",
    "reasoning",
    "total",
)


def billing_available() -> bool:
    """Whether the installed opentine can price a model call."""
    try:
        from opentine.billing import bill  # noqa: F401
    except ImportError:
        return False
    return True


def to_usage(raw: Any) -> Any | None:
    """Build an opentine ``Usage`` from any provider-native usage mapping.

    Names are normalized by :func:`graphforge.state.sane_usage` first, so
    ``prompt_tokens`` / ``completion_tokens`` / ``cache_read_input_tokens``
    all land on the right dimension. Unrecognized dimensions go to ``extra``,
    which opentine preserves.
    """
    try:
        from opentine import Usage
    except ImportError:  # pragma: no cover - opentine 0.2.0
        return None
    normalized = sane_usage(raw)
    if not normalized:
        return None
    known: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in normalized.items():
        if key in _USAGE_FIELDS:
            known[key] = int(value)
        else:
            extra[key] = value
    if extra:
        known["extra"] = extra
    try:
        return Usage(**known)
    except (ValueError, TypeError):
        # Usage validates strictly; an unusable payload must not break the
        # recording, only the pricing.
        return None


def price_call(
    provider: str,
    model: str,
    usage: Any,
    *,
    effective_at: Any = None,
    service_tier: str | None = None,
    unmetered: bool = False,
) -> Any | None:
    """Price one model call, or ``None`` when pricing is unavailable.

    Never raises: an unknown model, a missing catalog, or opentine 0.2.0 all
    return ``None`` so the caller keeps its own cost figure.
    """
    try:
        from opentine.billing import CatalogError, bill
    except ImportError:  # pragma: no cover - opentine 0.2.0
        return None
    usage_obj = usage if _is_usage(usage) else to_usage(usage)
    if usage_obj is None:
        return None
    try:
        return bill(
            str(provider or ""),
            str(model or ""),
            usage_obj,
            effective_at=effective_at,
            service_tier=service_tier,
            unmetered=unmetered,
        )
    except (CatalogError, ValueError, TypeError, KeyError):
        return None


def _is_usage(value: Any) -> bool:
    try:
        from opentine import Usage
    except ImportError:  # pragma: no cover - opentine 0.2.0
        return False
    return isinstance(value, Usage)


def billing_fields(result: Any) -> dict[str, Any]:
    """The ``cost`` / ``billing`` kwargs for a priced step.

    ``billing`` is the JSON-safe ``BillingResult.to_dict()`` (its ``Decimal``
    amounts become strings); passing the dataclass or a raw ``Decimal`` is
    rejected downstream.

    Returns ``{}`` for an ``unknown`` result. That case is not merely
    uninformative: opentine's ``total_cost`` prefers a step's billing
    subtotal over its ``cost``, so attaching an unknown record — whose
    subtotal is 0 — would silently zero a cost the caller actually knew.
    The caller's own figure is kept instead, and :func:`pricing_note`
    records that pricing was attempted and failed.
    """
    if result is None:
        return {}
    try:
        from opentine.billing import known_cost
    except ImportError:  # pragma: no cover - opentine 0.2.0
        return {}
    if getattr(result, "status", "") == "unknown":
        return {}
    try:
        return {"billing": result.to_dict(), "cost": known_cost(result)}
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return {}


def pricing_note(result: Any) -> dict[str, Any] | None:
    """A short record of an unsuccessful pricing attempt, for step outputs."""
    if result is None or getattr(result, "status", "") != "unknown":
        return None
    return {
        "status": "unknown",
        "warnings": list(getattr(result, "warnings", ()) or ()),
    }
