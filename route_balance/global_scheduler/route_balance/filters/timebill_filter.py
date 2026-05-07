"""TimeBill-style pointwise budget filter ().

Paper story (for §L3 filter ablation vs RouteBalanceCDF): a deterministic, point-
estimate budget filter. Reject an instance if the point estimate of the
request's cost — computed from the instance's predicted TTFT+TPOT and the
request's predicted output length — exceeds the request's cost budget.

This is the **pointwise budget** counterpart to RouteBalanceCDF's **probabilistic
CDF** budget check. Same budget SLO, same workload; the comparison
measures e2e goodput + false rejection rate at a fixed SLO target.

Accept iff
    cost(predicted_ttft + N * predicted_tpot)  <=  budget

where `cost` is a per-instance cost-per-token rate (loaded from
instance metadata when present) times the predicted wall time.

If cost metadata is missing, falls back to using the product
(ttft + N * tpot) as a time-budget proxy against `slo.ttft_ms +
N * slo.tpot_ms` (i.e. degenerates gracefully to a time-based pointwise
check comparable to SLOs-Serve cumulative-deadline but stricter on the
point estimate).
"""
from typing import Dict, List, Optional

from .base import FilterResult, InstanceState, SLOConstraints, SLOFilter


class TimeBillFilter(SLOFilter):
    """Pointwise deterministic budget filter."""

    def __init__(
        self,
        cost_per_sec_by_instance_type: Optional[Dict[str, float]] = None,
        cost_per_sec_default: float = 1.0,
    ):
        """
        Args:
            cost_per_sec_by_instance_type: {instance_type: $/sec} — if provided,
                multiplied by predicted wall time to compute cost. Typical
                values come from instance_metadata in scheduler_config.json.
            cost_per_sec_default: Fallback rate when instance_type is not in
                the map. Also used when no map is provided; then the filter
                degenerates to a time-budget proxy.
        """
        self._rate_map = cost_per_sec_by_instance_type or {}
        self._rate_default = float(cost_per_sec_default)

    def filter(
        self,
        slo: SLOConstraints,
        instances: List[InstanceState],
        predicted_output_tokens: int = 100,
    ) -> List[FilterResult]:
        # Determine budget. Prefer slo.budget_tokens as "cost budget"; callers
        # can translate a $ budget into this field. Fall back to time budget.
        time_budget_ms = slo.ttft_ms + predicted_output_tokens * slo.tpot_ms
        has_cost_budget = bool(self._rate_map) or slo.budget_tokens > 0

        results: List[FilterResult] = []
        for inst in instances:
            predicted_wall_ms = (
                inst.predicted_ttft_ms
                + predicted_output_tokens * inst.predicted_tpot_ms
            )

            if has_cost_budget:
                rate = self._rate_map.get(
                    getattr(inst, "gpu_type", "") or "", self._rate_default
                )
                predicted_cost = (predicted_wall_ms / 1000.0) * rate
                budget = float(slo.budget_tokens)  # reinterpret as $ budget
                accept = predicted_cost <= budget
                reason = (
                    "pointwise_budget_met"
                    if accept
                    else f"reject:cost={predicted_cost:.4f}>budget={budget:.4f}"
                )
            else:
                # Time-budget proxy (stricter pointwise variant of
                # SLOs-Serve cumulative-deadline).
                accept = predicted_wall_ms <= time_budget_ms
                reason = (
                    "pointwise_time_budget_met"
                    if accept
                    else f"reject:t={predicted_wall_ms:.0f}ms>budget="
                    f"{time_budget_ms:.0f}ms"
                )

            results.append(
                FilterResult(
                    instance_id=inst.instance_id,
                    accepted=accept,
                    reason=reason,
                    predicted_ttft_ms=inst.predicted_ttft_ms,
                    predicted_tpot_ms=inst.predicted_tpot_ms,
                )
            )
        return results
