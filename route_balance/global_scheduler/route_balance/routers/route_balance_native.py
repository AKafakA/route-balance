"""RouteBalance multi-objective SCORE as a router signal (ablation-only).

IMPORTANT — naming clarification (2026-04-15)
----------------------------------------------
The class here is NOT "RouteBalance the paper system". RouteBalance the system is the
integrated batch scheduler (LPT sort + cluster memory balance + SLO filter
+ RouteBalanceCDF + model estimator), invoked via `--scheduling route_balance` — it does
NOT decompose into a (router, dispatcher, filter) tuple.

This class exposes only RouteBalance's multi-objective MODEL-SELECTION SCORE for
use inside the pluggable pipeline as a baseline-ablation endpoint, e.g.
"route_balance_score router + round_robin dispatcher + none filter" — an ablation
that isolates the model-selection signal from the batch/balance logic.

Registered factory types:
    route_balance_score    — primary name (use this going forward)
    route_balance_native   — deprecated alias kept for backward-compat with earlier
                    configs / dry-run matrix. Logs a warning on use.
"""
import logging
from typing import List, Optional

from .base import RouterBase, RouterDecision, RouterRequest


logger = logging.getLogger(__name__)


class RouteBalanceScoreRouter(RouterBase):
    """Router using RouteBalance's multi-objective score — ablation endpoint only."""

    def __init__(
        self,
        model_estimator=None,
        scoring_weights: Optional[dict] = None,
    ):
        self._model_estimator = model_estimator
        self._weights = scoring_weights or {
            "w_quality": 0.6,
            "w_cost": 0.3,
            "w_length": 0.1,
        }

    async def choose_model(
        self,
        req: RouterRequest,
        model_pool: List[str],
    ) -> RouterDecision:
        if not model_pool:
            raise ValueError("model_pool is empty")

        if self._model_estimator is None:
            return RouterDecision(
                model_name=_largest_model(model_pool),
                score=0.0,
                reason="route_balance_score:no_estimator:fallback_largest",
            )

        estimates = self._model_estimator.estimate(req.prompt, req.budget_tokens)

        best_name = None
        best_score = float("-inf")
        for name in model_pool:
            est = estimates.get(name)
            if est is None:
                continue
            quality = getattr(est, "score", 0.0) or 0.0
            length = getattr(est, "length_expected", 0.0) or 0.0
            cost_proxy = _size_proxy(name) / max(
                _size_proxy(n) for n in model_pool
            )
            score = (
                self._weights.get("w_quality", 0.6) * quality
                - self._weights.get("w_cost", 0.3) * cost_proxy
                - self._weights.get("w_length", 0.1) * (length / 1024.0)
            )
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            best_name = _largest_model(model_pool)

        return RouterDecision(
            model_name=best_name,
            score=best_score,
            reason="route_balance_score:multi_obj",
        )


# Back-compat alias — old name used in early dry-run matrix runs.
class RouteBalanceNativeRouter(RouteBalanceScoreRouter):
    """DEPRECATED alias of RouteBalanceScoreRouter. Use route_balance_score in new configs."""

    def __init__(self, *args, **kwargs):
        logger.warning(
            "RouteBalanceNativeRouter is deprecated; use RouteBalanceScoreRouter / "
            "router.type='route_balance_score'. This alias remains for backward-"
            "compat only. The class does NOT represent 'RouteBalance the paper "
            "system' — that is invoked via --scheduling route_balance."
        )
        super().__init__(*args, **kwargs)


def _size_proxy(model_name: str) -> float:
    import re
    m = re.search(r"(\d+(?:\.\d+)?)[Bb]", model_name)
    return float(m.group(1)) if m else 1.0


def _largest_model(pool: List[str]) -> str:
    return max(pool, key=_size_proxy)
