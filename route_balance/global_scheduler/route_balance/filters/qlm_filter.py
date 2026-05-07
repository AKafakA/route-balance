"""
QLM style filter: Normal confidence bounds for SLO checking.

Based on: "QLM: Queue Management for LLM Serving" (arXiv 2407.00047)

Uses CLT to model prediction uncertainty as Normal distribution.
Accept condition uses conservative upper bound:
    predicted_ttft + z * std_ttft <= ttft_slo
    predicted_tpot + z * std_tpot <= tpot_slo

Where z is the confidence level (e.g., z=1.645 for 95% one-sided).
Higher z = more conservative = fewer false accepts but more false rejects.

The std is computed from XGBoost prediction residuals on the validation set,
stored per instance type during model calibration.
"""

from typing import Dict, List, Optional

from .base import SLOFilter, SLOConstraints, InstanceState, FilterResult


# Standard Normal z-values for common confidence levels
Z_VALUES = {
    0.80: 0.842,
    0.85: 1.036,
    0.90: 1.282,
    0.95: 1.645,
    0.975: 1.960,
    0.99: 2.326,
}


class QLMFilter(SLOFilter):
    """Normal confidence bound SLO filter (QLM simplified).

    Args:
        confidence: One-sided confidence level (default 0.95).
            Higher = more conservative. Must be in Z_VALUES.
        calibration_data: Dict mapping instance_type to
            {"ttft_std_ms": float, "tpot_std_ms": float}
            from validation set residual analysis.
    """

    def __init__(
        self,
        confidence: float = 0.95,
        calibration_data: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        if confidence not in Z_VALUES:
            raise ValueError(f"confidence must be one of {list(Z_VALUES.keys())}")
        self.z = Z_VALUES[confidence]
        self.confidence = confidence
        self.calibration = calibration_data or {}

    def filter(
        self,
        slo: SLOConstraints,
        instances: List[InstanceState],
        predicted_output_tokens: int = 100,
    ) -> List[FilterResult]:
        results = []
        for inst in instances:
            # Get std from calibration data or instance state
            inst_type = f"{inst.model_name}_{inst.gpu_type}"
            cal = self.calibration.get(inst_type, {})
            ttft_std = cal.get("ttft_std_ms", inst.ttft_std_ms)
            tpot_std = cal.get("tpot_std_ms", inst.tpot_std_ms)

            # Conservative upper bound: prediction + z * std
            ttft_upper = inst.predicted_ttft_ms + self.z * ttft_std
            tpot_upper = inst.predicted_tpot_ms + self.z * tpot_std

            ttft_ok = ttft_upper <= slo.ttft_ms
            tpot_ok = tpot_upper <= slo.tpot_ms

            if ttft_ok and tpot_ok:
                # Compute confidence that SLO will be met
                # P(pred + noise <= SLO) = Phi((SLO - pred) / std)
                conf = min(
                    _normal_cdf_approx((slo.ttft_ms - inst.predicted_ttft_ms) / max(ttft_std, 1e-6)),
                    _normal_cdf_approx((slo.tpot_ms - inst.predicted_tpot_ms) / max(tpot_std, 1e-6)),
                )
                results.append(FilterResult(
                    instance_id=inst.instance_id,
                    accepted=True,
                    reason=f"confidence_bound_met(z={self.z:.2f})",
                    predicted_ttft_ms=inst.predicted_ttft_ms,
                    predicted_tpot_ms=inst.predicted_tpot_ms,
                    confidence=conf,
                ))
            else:
                reasons = []
                if not ttft_ok:
                    reasons.append(f"ttft_upper={ttft_upper:.0f}>{slo.ttft_ms:.0f}")
                if not tpot_ok:
                    reasons.append(f"tpot_upper={tpot_upper:.0f}>{slo.tpot_ms:.0f}")
                results.append(FilterResult(
                    instance_id=inst.instance_id,
                    accepted=False,
                    reason="reject:" + ",".join(reasons),
                    predicted_ttft_ms=inst.predicted_ttft_ms,
                    predicted_tpot_ms=inst.predicted_tpot_ms,
                    confidence=0.0,
                ))
        return results


def _normal_cdf_approx(z: float) -> float:
    """Approximate standard Normal CDF using Abramowitz and Stegun."""
    import math
    if z < -6:
        return 0.0
    if z > 6:
        return 1.0
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    p = 0.2316419
    t = 1.0 / (1.0 + p * abs(z))
    zp = math.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    cdf = 1.0 - zp * (b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5)
    if z < 0:
        cdf = 1.0 - cdf
    return cdf
