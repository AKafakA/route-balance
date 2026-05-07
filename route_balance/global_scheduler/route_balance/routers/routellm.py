"""RouteLLM router adapter (ICML'24, arXiv 2406.18665) — .

Wraps the upstream RouteLLM library as a pluggable RouterBase. Upstream's
API is:
    router.calculate_strong_win_rate(prompt: str) -> float in [0, 1]
    router.route(prompt, threshold, routed_pair) -> model_id

RouteLLM was trained for the GPT-4/Mixtral "strong vs weak" pair. To run
natively against our Qwen-2.5-3B/7B/72B pool, we:
    1. Map pool → (strong, weak) per config (e.g. 72B strong, 3B weak).
    2. Use a published checkpoint by default (uncalibrated for Qwen —
       documented in the adapter; scores will need recalibration on our
       val set via scripts/recalibrate_routellm_qwen.py).
    3. Respect a per-router `threshold` param; the routed pair comes from
       the runtime pool at request time.

For richer pools (3-way), we expose `tiers` so a 3-tier heterogeneous setup
can use 2 thresholds (low/high); scored as a piecewise-linear win-rate.

Recalibration: RouteLLM's BERT classifier head takes strong-win pair labels.
Training data = our `data/route_balance/scored/` with labels derived from deepeval
score diffs (`score_strong - score_weak > 0 → 1`). Scaffolding only for now
— retraining job lives under `scripts/recalibrate_routellm_qwen.py`
(created by A3.4 as a follow-up).
"""
from pathlib import Path
from typing import List, Optional

from .base import RouterBase, RouterDecision, RouterRequest


class RouteLLMRouter(RouterBase):
    """Adapter over upstream RouteLLM routers."""

    def __init__(
        self,
        *,
        router_type: str = "mf",
        checkpoint_path: str = "routellm/mf_gpt4_augmented",
        threshold: float = 0.11593,
        strong_model: Optional[str] = None,
        weak_model: Optional[str] = None,
        tiers: Optional[List[dict]] = None,
        routellm_src: Optional[str] = None,
    ):
        """
        Args:
            router_type: One of "bert", "causal_llm", "mf" (matrix-factor),
                "sw_ranking". Default "mf" — paper-recommended (Ong ICLR'25).
            checkpoint_path: HF repo id or local path to the router ckpt.
                Default `routellm/mf_gpt4_augmented` = paper's published
                matrix-factorization checkpoint (trained on GPT-4 vs Mixtral
                preference data; domain-shift footnote when applied to Qwen).
            threshold: Decision threshold on strong-win-rate. Default 0.11593
                = paper's 50%-strong calibration point against Chatbot Arena.
            strong_model / weak_model: Override pool mapping. When unset,
                chosen at request time as the largest / smallest in the
                runtime pool.
            tiers: Optional list of {"threshold": float, "model": str} tiers
                for 3+ tier pools. Ignored when strong/weak are set.
            routellm_src: Path to a local RouteLLM clone (added to
                ``sys.path`` when the routellm package is not otherwise
                importable). Falls back to ``ROUTE_BALANCE_ROUTELLM_SRC``
                env var or ``$HOME/RouteLLM``.
        """
        self._type = router_type
        self._ckpt = checkpoint_path
        self._threshold = float(threshold)
        self._strong = strong_model
        self._weak = weak_model
        self._tiers = list(tiers) if tiers else None

        # Ensure routellm is importable. If the package is not installed,
        # fall back to a local clone path supplied via argument, env var,
        # or `$HOME/RouteLLM`.
        try:
            import routellm  # noqa: F401
        except ImportError:
            import os
            import sys
            src = (
                routellm_src
                or os.environ.get("ROUTE_BALANCE_ROUTELLM_SRC")
                or str(Path.home() / "RouteLLM")
            )
            if Path(src).exists() and src not in sys.path:
                sys.path.insert(0, src)

        try:
            from routellm.routers.routers import (
                BERTRouter,
                CausalLLMRouter,
                MatrixFactorizationRouter,
                SWRankingRouter,
            )
        except ImportError as e:
            raise RuntimeError(
                "RouteLLM not importable. Either `pip install routellm` or "
                "clone https://github.com/lm-sys/RouteLLM to "
                "~/Code/llm/RouteLLM. Original error: " + str(e)
            ) from e

        if router_type == "bert":
            self._router = BERTRouter(checkpoint_path=checkpoint_path)
        elif router_type == "causal_llm":
            self._router = CausalLLMRouter(checkpoint_path=checkpoint_path)
        elif router_type == "mf":
            self._router = MatrixFactorizationRouter(
                checkpoint_path=checkpoint_path,
                strong_model=strong_model or "gpt-4-1106-preview",
                weak_model=weak_model or "mixtral-8x7b-instruct-v0.1",
            )
        elif router_type == "sw_ranking":
            # SWRanking needs arena datasets; skip for our adapter (heavy).
            raise NotImplementedError(
                "sw_ranking variant not wired — requires arena battle "
                "datasets. Use router_type='bert' or 'mf'."
            )
        else:
            raise ValueError(f"unknown router_type={router_type!r}")

    def _pool_mapping(self, pool: List[str]) -> (str, str):
        if self._strong and self._weak:
            return self._strong, self._weak
        import re

        def size(n):
            m = re.search(r"(\d+(?:\.\d+)?)[Bb]", n)
            return float(m.group(1)) if m else 1.0

        sorted_pool = sorted(pool, key=size)
        return sorted_pool[-1], sorted_pool[0]

    async def choose_model(
        self,
        req: RouterRequest,
        model_pool: List[str],
    ) -> RouterDecision:
        if not model_pool:
            raise ValueError("model_pool is empty")

        strong, weak = self._pool_mapping(model_pool)
        try:
            win = float(self._router.calculate_strong_win_rate(req.prompt))
        except Exception as e:
            return RouterDecision(
                model_name=weak,
                score=0.0,
                reason=f"routellm:error:{type(e).__name__}:fallback_weak",
            )

        # 3+ tier piecewise mapping, if configured.
        if self._tiers:
            chosen = weak
            for tier in self._tiers:
                if win >= float(tier["threshold"]):
                    if tier["model"] in model_pool:
                        chosen = tier["model"]
            return RouterDecision(
                model_name=chosen,
                score=win,
                reason=f"routellm:{self._type}:win={win:.3f}:tiered",
            )

        # 2-tier default.
        chosen = strong if win >= self._threshold else weak
        if chosen not in model_pool:
            chosen = strong if strong in model_pool else model_pool[0]
        return RouterDecision(
            model_name=chosen,
            score=win,
            reason=f"routellm:{self._type}:win={win:.3f}:th={self._threshold}",
        )
