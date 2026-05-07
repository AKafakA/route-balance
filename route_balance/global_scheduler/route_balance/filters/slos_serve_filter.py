"""
SLOs-Serve style filter: point prediction binary accept/reject.

Based on: "SLOs-Serve: Multi-SLO Serving" (arXiv 2504.08784)

Simplified from the full DP admission control: we use point predictions
from XGBoost (instead of roofline model) and binary per-instance filtering
(instead of global optimal subset selection).

Accept condition:
    predicted_ttft <= ttft_slo AND predicted_tpot <= tpot_slo

If ALL instances rejected → request goes to least-bad instance (shortest queue).
This is the simplest baseline: no confidence, no distribution, no relaxation.
"""

from typing import List

from .base import SLOFilter, SLOConstraints, InstanceState, FilterResult


class SLOsServeFilter(SLOFilter):
    """Binary point-prediction SLO filter (SLOs-Serve simplified)."""

    def filter(
        self,
        slo: SLOConstraints,
        instances: List[InstanceState],
        predicted_output_tokens: int = 100,
    ) -> List[FilterResult]:
        results = []
        for inst in instances:
            ttft_ok = inst.predicted_ttft_ms <= slo.ttft_ms
            tpot_ok = inst.predicted_tpot_ms <= slo.tpot_ms

            if ttft_ok and tpot_ok:
                results.append(FilterResult(
                    instance_id=inst.instance_id,
                    accepted=True,
                    reason="point_prediction_within_slo",
                    predicted_ttft_ms=inst.predicted_ttft_ms,
                    predicted_tpot_ms=inst.predicted_tpot_ms,
                ))
            else:
                reasons = []
                if not ttft_ok:
                    reasons.append(f"ttft={inst.predicted_ttft_ms:.0f}>{slo.ttft_ms:.0f}")
                if not tpot_ok:
                    reasons.append(f"tpot={inst.predicted_tpot_ms:.0f}>{slo.tpot_ms:.0f}")
                results.append(FilterResult(
                    instance_id=inst.instance_id,
                    accepted=False,
                    reason="reject:" + ",".join(reasons),
                    predicted_ttft_ms=inst.predicted_ttft_ms,
                    predicted_tpot_ms=inst.predicted_tpot_ms,
                ))
        return results
