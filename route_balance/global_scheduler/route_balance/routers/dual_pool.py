"""Dual-Pool router (arXiv 2604.08075) — A3.2 / #27.

Paper sketch
------------
Dual-Pool partitions serving capacity into a *short* pool and a *long* pool
based on EMA bytes-per-token of prompt classes. Within a pool, dispatch is
round-robin. In our two-layer architecture (L1 router × L2 dispatcher) the
Dual-Pool contribution is the pool-assignment rule — the RR part lives in
the L2 dispatcher.

Mapping to heterogeneous RouteBalance clusters
--------------------------------------
The original paper assumes a homogeneous fleet where pools differ only in
expected output length. In a heterogeneous RouteBalance pool (Qwen-3B / 7B / 72B),
we interpret the two pools as *model tiers*:
    short pool → smallest model(s)
    long  pool → largest model(s)

This preserves the paper's intent: route cheap-looking requests to cheap
capacity, expensive-looking requests to expensive capacity.

Classification signal
---------------------
The router maintains an EMA of observed output-bytes-per-input-token per
"prompt class". Two class-id options are supported:
    bucket   (default) — prompt_tokens binned into configurable buckets
    dataset  — the optional `extra["dataset"]` field from the client
For a new class, EMA is seeded with `cold_start_ratio` (default 1.0). The
EMA is compared against `threshold_ratio` — above goes to the long pool,
below to the short pool.

EMA can be updated post-hoc via `observe(class_id, observed_ratio)`. When no
feedback is wired, the router degrades to a rule-based short/long split
purely from prompt length (cold-start behavior) — which is still a
reasonable Dual-Pool variant used as a baseline in related work.

Ref: arXiv 2604.08075; also see roadmap TASK #27 and
`route_balance_paper/claude/PROPOSAL_DUAL_POOL.md` (if present).
"""
from typing import Dict, List, Optional, Sequence

from .base import RouterBase, RouterDecision, RouterRequest


_DEFAULT_BUCKETS = (128, 512, 2048)  # prompt_tokens edges


def _size_proxy(model_name: str) -> float:
    import re
    m = re.search(r"(\d+(?:\.\d+)?)[Bb]", model_name)
    return float(m.group(1)) if m else 1.0


class DualPoolRouter(RouterBase):
    """Dual-Pool router — EMA-based short/long pool assignment."""

    def __init__(
        self,
        *,
        alpha: float = 0.3,
        threshold_ratio: float = 1.0,
        cold_start_ratio: float = 0.5,
        class_by: str = "bucket",
        buckets: Sequence[int] = _DEFAULT_BUCKETS,
        short_models: Optional[List[str]] = None,
        long_models: Optional[List[str]] = None,
    ):
        """
        Args:
            alpha: EMA smoothing factor in (0,1]. Higher = faster adaptation.
            threshold_ratio: Above this EMA value a class is considered
                "long" and routed to the long pool.
            cold_start_ratio: Seed EMA value for a class not seen before.
            class_by: "bucket" (prompt length bucket, default) or "dataset"
                (read from `req.extra['dataset']`).
            buckets: Edges of the prompt-length buckets; only used when
                class_by='bucket'.
            short_models / long_models: Explicit per-pool model lists. When
                None, pools are derived from the runtime pool at each call:
                short pool = {smallest model}, long pool = {largest model}.
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0,1], got {alpha}")
        self._alpha = float(alpha)
        self._threshold = float(threshold_ratio)
        self._cold_start = float(cold_start_ratio)
        self._class_by = class_by
        self._buckets = tuple(sorted(set(buckets)))
        self._short_models = (
            list(short_models) if short_models is not None else None
        )
        self._long_models = (
            list(long_models) if long_models is not None else None
        )
        self._ema: Dict[str, float] = {}

    # --- EMA machinery -------------------------------------------------

    def _class_id(self, req: RouterRequest) -> str:
        if self._class_by == "dataset":
            return str(req.extra.get("dataset", "default"))
        # bucket
        n = int(req.num_prompt_tokens or 0)
        for edge in self._buckets:
            if n < edge:
                return f"b<{edge}"
        return f"b>={self._buckets[-1]}"

    def observe(self, class_id: str, observed_ratio: float) -> None:
        """Feedback hook: update EMA for a class given an observed ratio.

        Call this after the request completes with
        observed_ratio = output_tokens / max(input_tokens, 1).
        """
        prev = self._ema.get(class_id, self._cold_start)
        self._ema[class_id] = (
            self._alpha * float(observed_ratio) + (1.0 - self._alpha) * prev
        )

    def _ema_for(self, class_id: str) -> float:
        return self._ema.get(class_id, self._cold_start)

    # --- model selection ----------------------------------------------

    def _resolve_pools(self, model_pool: List[str]) -> (List[str], List[str]):
        if self._short_models and self._long_models:
            short = [m for m in self._short_models if m in model_pool]
            longp = [m for m in self._long_models if m in model_pool]
            if not short or not longp:
                # Fall through to runtime derivation if explicit pools are
                # no longer in the runtime pool (e.g. instance churn).
                pass
            else:
                return short, longp
        # Runtime derivation: smallest → short, largest → long. With more
        # than two tiers the middle tiers go to short (conservative).
        sorted_pool = sorted(model_pool, key=_size_proxy)
        if len(sorted_pool) == 1:
            return sorted_pool, sorted_pool
        return sorted_pool[:-1], [sorted_pool[-1]]

    async def choose_model(
        self,
        req: RouterRequest,
        model_pool: List[str],
    ) -> RouterDecision:
        if not model_pool:
            raise ValueError("model_pool is empty")

        short, longp = self._resolve_pools(model_pool)
        class_id = self._class_id(req)
        ratio = self._ema_for(class_id)

        # Also use prompt length as a tiebreaker cold-start signal: when EMA
        # is at the cold-start value (never observed), also consider a prompt
        # length heuristic — promotes long prompts to the long pool.
        bucket_prefers_long = False
        if self._class_by == "bucket" and class_id == self._class_id(req):
            # Promote to long for the top-most bucket.
            if class_id.startswith("b>="):
                bucket_prefers_long = True

        use_long = ratio >= self._threshold or bucket_prefers_long
        chosen_pool = longp if use_long else short
        if not chosen_pool:
            chosen_pool = short or longp or model_pool

        # Pick the first model in the chosen pool (deterministic). Callers
        # that want within-pool variety should stack with a non-RR
        # dispatcher or use EMA-class tagging.
        chosen = chosen_pool[0]
        return RouterDecision(
            model_name=chosen,
            score=ratio,
            reason=(
                f"dual_pool:class={class_id}"
                f":ratio={ratio:.2f}"
                f":pool={'long' if use_long else 'short'}"
            ),
        )
