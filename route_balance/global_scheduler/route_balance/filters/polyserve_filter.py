"""
PolyServe style filter: cumulative deadline-based SLO check.

Based on: "PolyServe: Multi-SLO Serving at Scale" (arXiv 2507.17769)

Deadline-based SLO: the i-th output token must arrive before
    deadline = TTFT_SLO + i * TPOT_SLO

Unlike strict per-token TPOT checking, this allows TTFT/TPOT tradeoff:
if TTFT is fast, TPOT can be slightly slower and vice versa, as long as
the cumulative deadline holds.

Accept condition:
    predicted_ttft + predicted_output_tokens * predicted_tpot <= deadline
    where deadline = ttft_slo + predicted_output_tokens * tpot_slo

This is more permissive than SLOs-Serve's strict AND condition.
"""

from typing import List

from .base import SLOFilter, SLOConstraints, InstanceState, FilterResult


class PolyServeFilter(SLOFilter):
    """Cumulative deadline SLO filter (PolyServe simplified)."""

    def filter(
        self,
        slo: SLOConstraints,
        instances: List[InstanceState],
        predicted_output_tokens: int = 100,
    ) -> List[FilterResult]:
        # Cumulative deadline: last token must arrive by TTFT_SLO + N * TPOT_SLO
        deadline_ms = slo.ttft_ms + predicted_output_tokens * slo.tpot_ms

        results = []
        for inst in instances:
            # Predicted total completion time
            predicted_e2e_ms = (
                inst.predicted_ttft_ms
                + predicted_output_tokens * inst.predicted_tpot_ms
            )

            if predicted_e2e_ms <= deadline_ms:
                results.append(FilterResult(
                    instance_id=inst.instance_id,
                    accepted=True,
                    reason="cumulative_deadline_met",
                    predicted_ttft_ms=inst.predicted_ttft_ms,
                    predicted_tpot_ms=inst.predicted_tpot_ms,
                ))
            else:
                results.append(FilterResult(
                    instance_id=inst.instance_id,
                    accepted=False,
                    reason=f"reject:e2e={predicted_e2e_ms:.0f}ms>deadline={deadline_ms:.0f}ms",
                    predicted_ttft_ms=inst.predicted_ttft_ms,
                    predicted_tpot_ms=inst.predicted_tpot_ms,
                ))
        return results
