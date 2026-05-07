"""Router factory — instantiate a RouterBase from config.

Registered routers:
    route_balance_native    — L1 #R1, existing behavior (argmax multi-objective)
    null_fixed     — L1 #R8, always same model; RouteBalance-as-baseline
    random         — trivial baseline
    dual_pool      — L1 #R2, arXiv 2604.08075 (EMA bytes/token)         [stub]
    best_route     — L1 #R3, HF artifact                                [stub]
    routellm       — L1 #R4, ICML'24                                    [stub]
    vllm_sr        — L1 #R5, external HTTP to Envoy+extproc             [stub]
    avengers_pro   — L1 #R6, DAI'25 2508.12631                          [stub]
    omnirouter     — L1 #R7, arXiv 2502.20576                           [stub]

Stubs raise NotImplementedError("see task #Nxx"). They let the config layer
accept these names right now; the behavior lands as part of A3.2 ... A3.7.
"""
from typing import Any, Dict

from .base import RouterBase
from .route_balance_native import RouteBalanceNativeRouter, RouteBalanceScoreRouter
from .null_fixed import NullFixedRouter
from .random_router import RandomRouter


def create_router(
    router_config: Dict[str, Any],
    *,
    model_estimator=None,
    scoring_weights=None,
) -> RouterBase:
    """Create a Router from config dict.

    Config shape:
        {"type": "route_balance_native", "model_estimator_ref": "default", ...}
        {"type": "null_fixed", "model_name": "Qwen2.5-72B-Instruct"}
        {"type": "random", "seed": 0}

    `model_estimator` and `scoring_weights` are runtime objects supplied by
    route_balance_serve during init; some routers (route_balance_native) need them, others
    ignore.
    """
    rtype = router_config.get("type", "route_balance_native")

    if rtype == "route_balance_score":
        return RouteBalanceScoreRouter(
            model_estimator=model_estimator,
            scoring_weights=scoring_weights,
        )
    if rtype == "route_balance_native":
        # Deprecated alias; kept for back-compat with earlier dry-run configs.
        return RouteBalanceNativeRouter(
            model_estimator=model_estimator,
            scoring_weights=scoring_weights,
        )
    if rtype == "qlm":
        from .qlm_router import QLMRouter  # A3.8
        return QLMRouter(**router_config.get("kwargs", {}))
    if rtype == "null_fixed":
        return NullFixedRouter(
            model_name=router_config.get("model_name"),
            pool_preference=router_config.get("pool_preference", "largest"),
        )
    if rtype == "random":
        return RandomRouter(seed=router_config.get("seed"))

    # Stubs: load lazily to avoid hard dependencies at import time.
    if rtype == "dual_pool":
        from .dual_pool import DualPoolRouter  # A3.2
        return DualPoolRouter(**router_config.get("kwargs", {}))
    if rtype == "best_route":
        # Legacy quality-argmax via ModelEstimator. Per Apr 15 PM decision this
        # is equivalent to RouteBalance-full with w_quality=1 / other weights=0 and
        # lives as the `quality_greedy_ablation` row in the ablation table. The
        # router remains registered for back-compat with Phase-1 dry-run configs.
        from .best_route import BestRouteRouter  # A3.3
        kwargs = dict(router_config.get("kwargs", {}))
        return BestRouteRouter(
            model_estimator=model_estimator, **kwargs
        )
    if rtype == "best_route_wrapper":
        # Trained DeBERTa-v3-small router per Ding 2025 (stripped-BoN) — §5.15.
        # Main-table row 5. Requires checkpoint at models/route_balance/best_route_wrapper_qwen/
        # (task #62 training pipeline).
        from .best_route_wrapper import BestRouteWrapperRouter
        return BestRouteWrapperRouter(**router_config.get("kwargs", {}))
    if rtype == "best_route_4way":
        # 4-class extension: routes among {3B, 7B, 14B, 72B} by argmax.
        # Checkpoint: models/route_balance/best_route_4way_qwen/ (May 2 build, val_acc 0.39).
        from .best_route_4way import BestRoute4WayRouter
        return BestRoute4WayRouter(**router_config.get("kwargs", {}))
    if rtype == "passthrough":
        # No-router baseline: returns "__ALL__" sentinel so _select_via_pipeline
        # exposes all instances to the dispatcher. Used for RR/SQ/Random
        # dispatcher-only baselines.
        from .passthrough import PassthroughRouter
        return PassthroughRouter()
    if rtype == "routellm":
        from .routellm import RouteLLMRouter  # A3.4
        return RouteLLMRouter(**router_config.get("kwargs", {}))
    if rtype == "vllm_sr":
        from .vllm_sr import VLLMSemanticRouter  # A3.5
        return VLLMSemanticRouter(**router_config.get("kwargs", {}))
    if rtype == "avengers_pro":
        from .avengers_pro import AvengersProRouter  # A3.6
        return AvengersProRouter(**router_config.get("kwargs", {}))
    if rtype == "omnirouter":
        from .omnirouter import OmniRouterRouter  # A3.7
        kwargs = dict(router_config.get("kwargs", {}))
        return OmniRouterRouter(
            model_estimator=model_estimator, **kwargs
        )

    raise ValueError(f"Unknown router type: {rtype!r}")
