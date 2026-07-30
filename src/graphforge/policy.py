"""Run-wide execution budgets for graph runs.

Every limit here is enforced against the WHOLE run (all branches of the
frontier combined), not per-branch — artifacts embed the policy under
``run.policies["graph"]`` so they self-describe the budgets that governed
them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GraphPolicy:
    """Budgets governing a graph run.

    - ``max_node_executions``: total node executions across the run.
    - ``max_total_cost``: run-wide spend ceiling in USD (sum of step costs).
    - ``max_duration_seconds``: wall-clock ceiling, checked before and after
      every node so a slow node is caught immediately.
    - ``max_visits_per_node``: cycle guard — max executions of any single
      node (graphs may loop; runaway cycles may not).
    - ``min_score``: success threshold — the run stops as soon as any node
      reports a score at or above it.
    """

    max_node_executions: int | None = None
    max_total_cost: float | None = None
    max_duration_seconds: float | None = None
    max_visits_per_node: int | None = 25
    min_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> GraphPolicy:
        if not data:
            return cls()
        known = {f: data.get(f) for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


class GraphPolicyViolation(RuntimeError):
    """Raised when a policy is structurally invalid (not on budget breach)."""
