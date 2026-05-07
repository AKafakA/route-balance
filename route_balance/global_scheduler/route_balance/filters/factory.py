"""
Factory for creating SLO filters from scheduler config.

Usage in route_balance_serve.py:
    from route_balance.global_scheduler.route_balance.filters.factory import create_filter
    slo_filter = create_filter(slo_defaults.get("filter", {}))
"""

from typing import Dict, Optional

from .base import SLOFilter
from .slos_serve_filter import SLOsServeFilter
from .polyserve_filter import PolyServeFilter
from .qlm_filter import QLMFilter
from .route_balance_cdf_filter import RouteBalanceCDFFilter
from .timebill_filter import TimeBillFilter


def create_filter(filter_config: Dict) -> SLOFilter:
    """Create an SLO filter from scheduler config.

    Args:
        filter_config: Dict with at minimum {"type": "..."}.
            Supported types:
                "route_balance_tiered" (default) — CDF filter with tiered relaxation
                "route_balance_hard_reject" — CDF filter without relaxation
                "slos_serve" — point prediction binary accept/reject
                "polyserve" — cumulative deadline check
                "qlm" — Normal confidence bounds

    Returns:
        SLOFilter instance.
    """
    filter_type = filter_config.get("type", "route_balance_tiered")

    if filter_type == "route_balance_tiered":
        return RouteBalanceCDFFilter(
            confidence_threshold=filter_config.get("confidence_threshold", 0.7),
            mode="tiered",
            relax_order=filter_config.get("relax_order", ["ttft", "tpot", "budget", "quality"]),
            relax_step=filter_config.get("relax_step", 0.1),
            max_relax_rounds=filter_config.get("max_relax_rounds", 3),
        )
    elif filter_type == "route_balance_hard_reject":
        return RouteBalanceCDFFilter(
            confidence_threshold=filter_config.get("confidence_threshold", 0.7),
            mode="hard_reject",
        )
    elif filter_type == "slos_serve":
        return SLOsServeFilter()
    elif filter_type == "polyserve":
        return PolyServeFilter()
    elif filter_type == "qlm":
        return QLMFilter(
            confidence=filter_config.get("confidence", 0.95),
            calibration_data=filter_config.get("calibration_data"),
        )
    elif filter_type == "timebill":
        return TimeBillFilter(
            cost_per_sec_by_instance_type=filter_config.get(
                "cost_per_sec_by_instance_type"
            ),
            cost_per_sec_default=filter_config.get("cost_per_sec_default", 1.0),
        )
    elif filter_type == "none":
        # No filtering — accept all instances (for baseline comparison)
        return _NoFilter()
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


class _NoFilter(SLOFilter):
    """No-op filter that accepts all instances."""

    requires_stats = False

    def filter(self, slo, instances, predicted_output_tokens=100):
        from .base import FilterResult
        return [
            FilterResult(
                instance_id=inst.instance_id,
                accepted=True,
                reason="no_filter",
                predicted_ttft_ms=inst.predicted_ttft_ms,
                predicted_tpot_ms=inst.predicted_tpot_ms,
            )
            for inst in instances
        ]
