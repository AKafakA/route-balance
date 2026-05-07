"""best-route router — .

Interpretation note (flagged in BASELINE_IMPL_TRACKER.md for user review):
    "best-route" in this project's prior docs (route_balance_paper/codex/codex-route_balance-plan.md,
    route_balance_paper/claude/ROUTE_BALANCE_PROJECT_STATUS.md) refers to the *internal* quality-
    estimator baseline — a trained per-model quality classifier used standalone
    (without multi-objective scoring). There is no external best-route HF
    artifact to "download". This implementation therefore realizes the
    quality-only routing policy:

        model_id = argmax_m  Q̂(prompt, m)

    where Q̂ comes from the loaded ModelEstimator (same estimator RouteBalance's native
    router consumes for its multi-objective score). This isolates the quality
    term from cost/latency — a meaningful ablation and a faithful realization
    of the original codex-route_balance-plan intent for "quality-only baseline".

    If the user intended a different external baseline (e.g. RouteLLM-BERT,
    a specific HF artifact, or a non-trivial paper named "best-route"), this
    stub is trivial to repoint — just swap the quality source. Logged as a
    decision in the tracker.

Use case vs RouteBalance native:
    route_balance_native:   argmax over α·Q − β·Cost − γ·Length
    best_route:    argmax over Q alone (= quality-greedy at model granularity)
    quality_greedy (legacy): argmax over model SIZE, not quality prediction
"""
from typing import List, Optional

from .base import RouterBase, RouterDecision, RouterRequest


class BestRouteRouter(RouterBase):
    """Quality-only router using the fused ModelEstimator's per-model score."""

    def __init__(
        self,
        model_estimator=None,
        tie_break: str = "smallest",
    ):
        """
        Args:
            model_estimator: ModelEstimator instance exposing
                estimate(prompt, budget) -> {model_name: Estimate(..., score)}.
                May be None at construction; route_balance_serve backfills it post-load
                (same pattern as RouteBalanceNativeRouter).
            tie_break: When multiple models tie on quality, pick "smallest"
                (cheapest) or "largest". Default "smallest" — faithful to
                the "get same quality cheaper" intent of quality-based routers.
        """
        self._model_estimator = model_estimator
        self._tie_break = tie_break

    async def choose_model(
        self,
        req: RouterRequest,
        model_pool: List[str],
    ) -> RouterDecision:
        if not model_pool:
            raise ValueError("model_pool is empty")

        if self._model_estimator is None:
            # Cold-start: no estimator → fall back to largest model (quality
            # proxy by size), matching the legacy quality_greedy semantics.
            return RouterDecision(
                model_name=_largest(model_pool),
                score=0.0,
                reason="best_route:no_estimator:fallback_largest",
            )

        try:
            estimates = self._model_estimator.estimate(
                req.prompt, req.budget_tokens
            )
        except Exception as e:
            return RouterDecision(
                model_name=_largest(model_pool),
                score=0.0,
                reason=f"best_route:estimator_error:{type(e).__name__}",
            )

        # Gather (model, score) pairs restricted to the runtime pool.
        scored = []
        for name in model_pool:
            est = estimates.get(name)
            if est is None:
                continue
            q = getattr(est, "score", 0.0) or 0.0
            scored.append((name, float(q)))
        if not scored:
            return RouterDecision(
                model_name=_largest(model_pool),
                score=0.0,
                reason="best_route:no_overlap:fallback_largest",
            )

        # Argmax on quality; tie-break by size.
        max_q = max(s for _, s in scored)
        ties = [n for n, s in scored if abs(s - max_q) < 1e-9]
        if len(ties) == 1:
            chosen = ties[0]
        else:
            chosen = _smallest(ties) if self._tie_break == "smallest" else _largest(ties)

        return RouterDecision(
            model_name=chosen,
            score=max_q,
            reason=f"best_route:argmax_quality={max_q:.3f}",
        )


def _size_proxy(model_name: str) -> float:
    import re
    m = re.search(r"(\d+(?:\.\d+)?)[Bb]", model_name)
    return float(m.group(1)) if m else 1.0


def _largest(pool: List[str]) -> str:
    return max(pool, key=_size_proxy)


def _smallest(pool: List[str]) -> str:
    return min(pool, key=_size_proxy)
