"""Passthrough router — skips L1 model selection.

Returns a sentinel `model_name = "__ALL__"` that signals the caller
(`_select_via_pipeline`) to bypass the per-model candidate filter and
expose all instances to the dispatcher. Used to evaluate dispatcher-only
baselines (RR / SQ / Random over all 13 instances) where there is no
quality-aware model selection.
"""
from __future__ import annotations

from typing import List

from .base import RouterBase, RouterDecision, RouterRequest


PASSTHROUGH_SENTINEL = "__ALL__"


class PassthroughRouter(RouterBase):
    async def choose_model(
        self,
        req: RouterRequest,
        model_pool: List[str],
    ) -> RouterDecision:
        if not model_pool:
            raise ValueError("model_pool is empty")
        return RouterDecision(
            model_name=PASSTHROUGH_SENTINEL,
            score=1.0 / max(len(model_pool), 1),
            reason="passthrough:no_routing",
        )
