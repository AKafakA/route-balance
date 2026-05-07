"""Dispatcher factory — create a DispatchBase from config.

Registered dispatchers (selectable via /v1/config dispatch.type):
    round_robin           Rotate across candidates (per-model counter).
    random                Uniform random over candidates.
    shortest_queue        min(num_running + num_waiting).
    num_waiting           Tie-break by num_waiting only.
    route_balance_native  Multi-objective score (latency + balance).
    route_balance_score   Convex-combination scorer (alias for back-compat).
    llumnix_minus         Llumnix dispatch (Sun et al., OSDI'24).
    predicted_latency           Power-of-two-choices with predicted latency.
"""
from typing import Any, Dict

from .base import DispatchBase
from .route_balance_native import RouteBalanceNativeDispatch, RouteBalanceScoreDispatch
from .num_waiting import NumWaitingDispatch
from .random_dispatch import RandomDispatch
from .round_robin import RoundRobinDispatch
from .shortest_queue import ShortestQueueDispatch


def create_dispatcher(
    dispatch_config: Dict[str, Any],
    *,
    scoring_weights=None,
    latency_predictor=None,
) -> DispatchBase:
    """Create a Dispatcher from config.

    Config shape:
        {"type": "round_robin"}
        {"type": "shortest_queue"}
        {"type": "route_balance_native"}
        {"type": "llumnix_minus"}                # once A1.1 lands
        {"type": "predicted_latency", "po2": false}
    """
    dtype = dispatch_config.get("type", "route_balance_native")

    if dtype == "round_robin":
        return RoundRobinDispatch()
    if dtype == "random":
        return RandomDispatch(seed=dispatch_config.get("seed"))
    if dtype == "shortest_queue":
        return ShortestQueueDispatch()
    if dtype == "num_waiting":
        return NumWaitingDispatch()
    if dtype == "route_balance_score":
        return RouteBalanceScoreDispatch(scoring_weights=scoring_weights)
    if dtype == "route_balance_native":
        return RouteBalanceNativeDispatch(scoring_weights=scoring_weights)

    if dtype == "llumnix_minus":
        from .llumnix_minus import LlumnixMinusDispatch
        return LlumnixMinusDispatch(**dispatch_config.get("kwargs", {}))
    if dtype == "predicted_latency":
        from .predicted_latency import PredictedLatencyDispatch
        kwargs = dict(dispatch_config.get("kwargs", {}))
        kwargs.setdefault("po2", dispatch_config.get("po2", False))
        return PredictedLatencyDispatch(
            latency_predictor=latency_predictor,
            **kwargs,
        )

    raise ValueError(f"Unknown dispatcher type: {dtype!r}")
