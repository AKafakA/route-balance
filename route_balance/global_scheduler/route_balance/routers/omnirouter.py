"""OmniRouter (arXiv 2502.20576) — .

Paper mechanism
---------------
OmniRouter frames routing as a constrained optimization: minimize total
cost subject to a quality floor. Per the paper, it uses a hybrid retrieval-
augmented predictor to estimate both (a) per-model capability for the
prompt and (b) per-model cost, then solves a Lagrangian dual decomposition
with adaptive multipliers.

Per-request degenerate form (single request):
    minimize_{m ∈ pool}  cost(m)   s.t.   quality(m) ≥ Q_min
    → argmin_{m : q̂(m) ≥ Q_min} cost(m)

Upstream repos
--------------
The paper links `github.com/dongyuanjushi/OmniRouter` (v1 ref, now 404) and
`github.com/agiresearch/ECCOS` (v2 ref, also 404 at check-time). Neither
public link resolves, so this adapter implements the paper's algorithm
directly using RouteBalance's existing per-model predictors:
    - Quality q̂(m): from the fused ModelEstimator (same signal best_route
      and route_balance_native use).
    - Cost ĉ(m): product of predicted wall time and per-instance cost
      rate (from instance_metadata), or a model-size proxy if cost
      metadata is absent.

If either public repo becomes available later, swap
`OmniRouterRouter._solve` for the upstream solver — the interface is
unchanged.

Implementation decisions (flagged in BASELINE_IMPL_TRACKER.md):
- Constrained form: argmin cost s.t. q̂ ≥ q_min. When no model meets the
  quality floor, relax to the model with highest quality (soft fallback).
- Batch-level Lagrangian decomposition (proper OmniRouter) would need a
  batch queue and adaptive multipliers. The per-request degenerate solver
  is a faithful single-request projection and matches how the paper's
  abstract describes the routing decision for online (latency-sensitive)
  deployment.
"""
from typing import List, Optional

from .base import RouterBase, RouterDecision, RouterRequest


def _size_proxy(model_name: str) -> float:
    import re
    m = re.search(r"(\d+(?:\.\d+)?)[Bb]", model_name)
    return float(m.group(1)) if m else 1.0


class OmniRouterRouter(RouterBase):
    """Constrained-optimization router (min cost s.t. quality floor)."""

    def __init__(
        self,
        *,
        model_estimator=None,
        quality_floor: float = 0.5,
        cost_per_b_proxy: float = 1.0,
        cost_rates: Optional[dict] = None,
        relax_if_infeasible: bool = True,
    ):
        """
        Args:
            model_estimator: RouteBalance fused ModelEstimator (same one route_balance_native
                consumes). Backfilled by route_balance_serve post-load.
            quality_floor: Minimum acceptable quality q_min ∈ [0, 1].
            cost_per_b_proxy: Cost per billion parameters when no cost
                metadata is supplied (keeps the signal monotone in model
                size).
            cost_rates: Optional {model_name: $/sec-like rate}. When
                provided, cost(m) = rate(m) × expected_wall_time_proxy.
            relax_if_infeasible: If no model meets q_min, pick the
                highest-quality model instead of refusing (standard paper
                behavior for online decisions).
        """
        self._me = model_estimator
        self._q_min = float(quality_floor)
        self._cost_per_b = float(cost_per_b_proxy)
        self._rates = dict(cost_rates or {})
        self._relax = bool(relax_if_infeasible)

    def _cost(self, model_name: str, est) -> float:
        # If explicit per-model rate: rate × expected wall time proxy
        # (prompt-length-dependent from length_expected). Otherwise use
        # size proxy: large models cost more.
        if model_name in self._rates:
            length = float(getattr(est, "length_expected", 100) or 100)
            return self._rates[model_name] * (length / 100.0)
        return self._cost_per_b * _size_proxy(model_name)

    def _quality(self, est) -> float:
        if est is None:
            return 0.0
        return float(getattr(est, "score", 0.0) or 0.0)

    async def choose_model(
        self,
        req: RouterRequest,
        model_pool: List[str],
    ) -> RouterDecision:
        if not model_pool:
            raise ValueError("model_pool is empty")

        # Cold-start without estimator: degenerate to largest (no signal to
        # choose on).
        if self._me is None:
            return RouterDecision(
                model_name=max(model_pool, key=_size_proxy),
                score=0.0,
                reason="omnirouter:no_estimator:fallback_largest",
            )

        try:
            estimates = self._me.estimate(req.prompt, req.budget_tokens)
        except Exception as e:
            return RouterDecision(
                model_name=max(model_pool, key=_size_proxy),
                score=0.0,
                reason=f"omnirouter:estimator_err:{type(e).__name__}",
            )

        candidates = []
        for name in model_pool:
            est = estimates.get(name)
            q = self._quality(est)
            c = self._cost(name, est)
            candidates.append((name, q, c))

        # Primal: argmin cost s.t. q ≥ q_min.
        feasible = [(n, q, c) for n, q, c in candidates if q >= self._q_min]
        if feasible:
            best = min(feasible, key=lambda t: t[2])
            return RouterDecision(
                model_name=best[0],
                score=best[1],
                reason=(
                    f"omnirouter:feasible:q={best[1]:.3f}"
                    f":cost={best[2]:.3f}:q_min={self._q_min}"
                ),
            )

        if not self._relax:
            return RouterDecision(
                model_name=max(candidates, key=lambda t: t[1])[0],
                score=0.0,
                reason="omnirouter:infeasible:strict",
            )

        # Infeasible → relax to max-quality.
        best = max(candidates, key=lambda t: t[1])
        return RouterDecision(
            model_name=best[0],
            score=best[1],
            reason=(
                f"omnirouter:infeasible:relax_to_max_q"
                f":q={best[1]:.3f}:cost={best[2]:.3f}"
            ),
        )
