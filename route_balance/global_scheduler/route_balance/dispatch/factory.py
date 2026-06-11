"""Dispatcher factory — create a DispatchBase from config.

Registered dispatchers:
    round_robin       — rotate across candidates (per-model counter)
    random            — uniform over candidates
    shortest_queue    — min(num_running + num_waiting)
    route_balance_native       — multi-objective latency + balance
    llumnix_minus     — L2 #D2  (STUB, A1.1 task #24)
    block_style       — L2 #D3  (STUB, A1.2 task #25)
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
        {"type": "block_style", "po2": false}    # once A1.2 lands
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
        # Deprecated alias
        return RouteBalanceNativeDispatch(scoring_weights=scoring_weights)

    if dtype == "llumnix_minus":
        from .llumnix_minus import LlumnixMinusDispatch  # A1.1
        return LlumnixMinusDispatch(**dispatch_config.get("kwargs", {}))
    if dtype == "block_style":
        from .block_style import BlockStyleDispatch  # A1.2
        kwargs = dict(dispatch_config.get("kwargs", {}))
        kwargs.setdefault("po2", dispatch_config.get("po2", False))
        return BlockStyleDispatch(
            latency_predictor=latency_predictor,
            **kwargs,
        )

    raise ValueError(f"Unknown dispatcher type: {dtype!r}")
