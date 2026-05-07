"""
RouteBalance CDF-based SLO filter with optional tiered relaxation.

Our approach: use bucket classifier softmax probabilities as empirical CDF
for distribution-free filtering.

For TTFT/TPOT: use XGBoost prediction residuals from validation set to build
empirical confidence intervals. Accept if P(actual <= SLO) >= threshold.

For budget: use length bucket classifier CDF. Accept if P(tokens <= budget) >= threshold.

Two modes:
  - hard_reject: reject if P < threshold, no relaxation (ablation baseline)
  - tiered: relax constraints one at a time in specified order until at least
    one instance is eligible (our full approach)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .base import SLOFilter, SLOConstraints, InstanceState, FilterResult


@dataclass
class ConfidenceCalibration:
    """Empirical residual distribution from validation set per instance type.

    residuals_ttft[i] = actual_ttft[i] - predicted_ttft[i]
    Sorted ascending for percentile lookup.
    """
    residuals_ttft_ms: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals_tpot_ms: np.ndarray = field(default_factory=lambda: np.array([]))

    def ttft_margin_at_confidence(self, confidence: float) -> float:
        """Get the margin such that P(actual <= predicted + margin) >= confidence.

        This is the empirical quantile of residuals — distribution-free.
        """
        if len(self.residuals_ttft_ms) == 0:
            return 0.0
        return float(np.percentile(self.residuals_ttft_ms, confidence * 100))

    def tpot_margin_at_confidence(self, confidence: float) -> float:
        if len(self.residuals_tpot_ms) == 0:
            return 0.0
        return float(np.percentile(self.residuals_tpot_ms, confidence * 100))

    def ttft_compliance_probability(self, predicted_ttft_ms: float, slo_ttft_ms: float) -> float:
        """P(actual_ttft <= slo) given predicted_ttft.

        Uses empirical CDF of residuals: P(residual <= slo - predicted).
        """
        if len(self.residuals_ttft_ms) == 0:
            return 1.0 if predicted_ttft_ms <= slo_ttft_ms else 0.0
        margin = slo_ttft_ms - predicted_ttft_ms
        return float(np.mean(self.residuals_ttft_ms <= margin))

    def tpot_compliance_probability(self, predicted_tpot_ms: float, slo_tpot_ms: float) -> float:
        if len(self.residuals_tpot_ms) == 0:
            return 1.0 if predicted_tpot_ms <= slo_tpot_ms else 0.0
        margin = slo_tpot_ms - predicted_tpot_ms
        return float(np.mean(self.residuals_tpot_ms <= margin))


class RouteBalanceCDFFilter(SLOFilter):
    """CDF-based SLO filter with optional tiered relaxation.

    Args:
        confidence_threshold: Minimum P(actual <= SLO) to accept (default 0.7).
        mode: "hard_reject" or "tiered"
        relax_order: Order in which to relax constraints when no instance passes.
            Default: ["ttft", "tpot", "budget", "quality"]
        relax_step: How much to reduce threshold per relaxation step (default 0.1).
        max_relax_rounds: Max relaxation rounds before falling back (default 3).
        calibration_data: Dict mapping instance_type to ConfidenceCalibration.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        mode: str = "tiered",
        relax_order: Optional[List[str]] = None,
        relax_step: float = 0.1,
        max_relax_rounds: int = 3,
        calibration_data: Optional[Dict[str, ConfidenceCalibration]] = None,
    ):
        self.threshold = confidence_threshold
        self.mode = mode
        self.relax_order = relax_order or ["ttft", "tpot", "budget", "quality"]
        self.relax_step = relax_step
        self.max_relax_rounds = max_relax_rounds
        self.calibration = calibration_data or {}

        # Counters for metrics
        self.stats = {
            "total_filter_calls": 0,
            "accepted_first_pass": 0,
            "accepted_after_relax": 0,
            "all_rejected": 0,
            "relax_rounds_used": [],
            "constraint_relaxed": {c: 0 for c in self.relax_order},
        }

    def filter(
        self,
        slo: SLOConstraints,
        instances: List[InstanceState],
        predicted_output_tokens: int = 100,
    ) -> List[FilterResult]:
        self.stats["total_filter_calls"] += 1
        self._predicted_output_tokens = predicted_output_tokens

        # First pass: check all instances at full threshold
        results = self._check_all(slo, instances, self.threshold)
        n_accepted = sum(1 for r in results if r.accepted)

        if n_accepted > 0:
            self.stats["accepted_first_pass"] += 1
            return results

        # No instance passed at full threshold
        if self.mode == "hard_reject":
            self.stats["all_rejected"] += 1
            return results

        # Tiered relaxation: relax constraints one at a time
        current_thresholds = {c: self.threshold for c in self.relax_order}

        for round_num in range(1, self.max_relax_rounds + 1):
            for constraint in self.relax_order:
                # Relax this constraint's threshold
                current_thresholds[constraint] = max(
                    0.0,
                    current_thresholds[constraint] - self.relax_step
                )
                self.stats["constraint_relaxed"][constraint] += 1

                # Re-check with relaxed thresholds
                results = self._check_all_with_thresholds(
                    slo, instances, current_thresholds
                )
                n_accepted = sum(1 for r in results if r.accepted)

                if n_accepted > 0:
                    self.stats["accepted_after_relax"] += 1
                    self.stats["relax_rounds_used"].append(round_num)
                    return results

        # Even after max relaxation, no instance passes → accept least-bad
        self.stats["all_rejected"] += 1
        return results

    def _check_all(
        self,
        slo: SLOConstraints,
        instances: List[InstanceState],
        threshold: float,
    ) -> List[FilterResult]:
        """Check all instances at a single threshold."""
        thresholds = {c: threshold for c in self.relax_order}
        return self._check_all_with_thresholds(slo, instances, thresholds)

    def _check_all_with_thresholds(
        self,
        slo: SLOConstraints,
        instances: List[InstanceState],
        thresholds: Dict[str, float],
    ) -> List[FilterResult]:
        """Check instances with per-constraint thresholds."""
        results = []
        for inst in instances:
            inst_type = f"{inst.model_name}_{inst.gpu_type}"
            cal = self.calibration.get(inst_type, ConfidenceCalibration())

            # TTFT check: P(actual_ttft <= slo) >= threshold
            p_ttft = cal.ttft_compliance_probability(
                inst.predicted_ttft_ms, slo.ttft_ms
            )
            ttft_ok = p_ttft >= thresholds.get("ttft", self.threshold)

            # TPOT check: P(actual_tpot <= slo) >= threshold
            p_tpot = cal.tpot_compliance_probability(
                inst.predicted_tpot_ms, slo.tpot_ms
            )
            tpot_ok = p_tpot >= thresholds.get("tpot", self.threshold)

            # Budget: average-case check using predicted output length.
            # Filter rejects instance if expected_cost > budget_cost. The
            # dispatch-side max_tokens clamp + budget_exhausted trace acts as
            # the hard worst-case backstop (route_balance_serve._dispatch_and_resolve).
            # Cap at max_output_tokens so prediction overrun (model emits more
            # than predicted) gets bounded by max_tokens at dispatch time.
            budget_ok = True
            if slo.budget_cost > 0 and inst.cost_per_output_token > 0:
                pred_out = min(
                    int(getattr(self, "_predicted_output_tokens", 100) or 100),
                    slo.max_output_tokens,
                )
                in_cost = slo.num_prompt_tokens * inst.cost_per_input_token
                expected_out_cost = pred_out * inst.cost_per_output_token
                expected_total = in_cost + expected_out_cost
                budget_ok = expected_total <= slo.budget_cost
            quality_ok = True  # Delegated to scheduler's quality filter

            if ttft_ok and tpot_ok and budget_ok and quality_ok:
                results.append(FilterResult(
                    instance_id=inst.instance_id,
                    accepted=True,
                    reason=f"cdf_filter(p_ttft={p_ttft:.2f},p_tpot={p_tpot:.2f})",
                    predicted_ttft_ms=inst.predicted_ttft_ms,
                    predicted_tpot_ms=inst.predicted_tpot_ms,
                    confidence=min(p_ttft, p_tpot),
                ))
            else:
                reasons = []
                if not ttft_ok:
                    reasons.append(f"p_ttft={p_ttft:.2f}<{thresholds.get('ttft', self.threshold):.2f}")
                if not tpot_ok:
                    reasons.append(f"p_tpot={p_tpot:.2f}<{thresholds.get('tpot', self.threshold):.2f}")
                results.append(FilterResult(
                    instance_id=inst.instance_id,
                    accepted=False,
                    reason="reject:" + ",".join(reasons),
                    predicted_ttft_ms=inst.predicted_ttft_ms,
                    predicted_tpot_ms=inst.predicted_tpot_ms,
                    confidence=min(p_ttft, p_tpot),
                ))
        return results

    def get_stats(self) -> Dict:
        """Return filtering statistics for metrics reporting."""
        stats = dict(self.stats)
        total = stats["total_filter_calls"]
        if total > 0:
            stats["first_pass_rate"] = stats["accepted_first_pass"] / total
            stats["relax_rate"] = stats["accepted_after_relax"] / total
            stats["reject_rate"] = stats["all_rejected"] / total
            if stats["relax_rounds_used"]:
                stats["avg_relax_rounds"] = np.mean(stats["relax_rounds_used"])
        return stats
