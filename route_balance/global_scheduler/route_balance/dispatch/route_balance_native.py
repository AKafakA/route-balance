"""RouteBalance multi-objective dispatch SCORE (ablation-only).

IMPORTANT — naming clarification (2026-04-15)
----------------------------------------------
NOT "RouteBalance the paper system". RouteBalance the system is the integrated batch
scheduler (LPT + balance + SLO filter), invoked via `--scheduling route_balance`.
This class exposes only the per-instance multi-objective score for use as
a pipeline-dispatcher ablation endpoint.

Registered factory types:
    route_balance_score    — primary name
    route_balance_native   — deprecated alias (warns on use)
"""
import logging
from typing import Any, Dict, List, Optional

from .base import DispatchBase, DispatchRequest


logger = logging.getLogger(__name__)


class RouteBalanceScoreDispatch(DispatchBase):
    """Multi-objective dispatch signal (kv_util + queue) — ablation only."""

    def __init__(self, scoring_weights: Optional[dict] = None):
        self._weights = scoring_weights or {
            "w_latency": 0.5,
            "w_balance": 0.5,
        }

    async def choose_instance(
        self,
        candidates: List[Any],
        req: DispatchRequest,
        *,
        stats_map: Optional[Dict[str, dict]] = None,
    ) -> Any:
        if not candidates:
            raise ValueError("RouteBalanceScoreDispatch: empty candidate list")

        best = candidates[0]
        best_score = float("inf")
        for inst in candidates:
            s = {}
            if stats_map is not None:
                s = stats_map.get(getattr(inst, "_instance_id", None), {}) or {}
            kv_util = float(s.get("kv_cache_utilization", 0.0) or 0.0)
            queue = int(s.get("num_running", 0) or 0) + int(
                s.get("num_waiting", 0) or 0
            )
            max_seqs = max(1, int(s.get("max_num_seqs", 256) or 256))
            load = queue / max_seqs
            score = (
                self._weights.get("w_latency", 0.5) * load
                + self._weights.get("w_balance", 0.5) * kv_util
            )
            if score < best_score:
                best_score = score
                best = inst
        return best


class RouteBalanceNativeDispatch(RouteBalanceScoreDispatch):
    """DEPRECATED alias of RouteBalanceScoreDispatch."""

    def __init__(self, *args, **kwargs):
        logger.warning(
            "RouteBalanceNativeDispatch is deprecated; use RouteBalanceScoreDispatch / "
            "dispatch.type='route_balance_score'. Alias kept for back-compat only."
        )
        super().__init__(*args, **kwargs)
