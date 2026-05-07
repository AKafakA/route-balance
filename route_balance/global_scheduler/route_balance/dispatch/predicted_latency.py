"""RouteBalance-style dispatch — predicted-latency minimization ().

Signal (per RouteBalance paper main.tex §4, socc_po2_ablation.tex §6.8):
    latency_i = predicted_ttft_i + N * predicted_tpot_i
    pick argmin(latency_i)

Optional Po2: sample 2 candidates uniformly at random and pick min between
them — O(1) message overhead at the cost of slightly worse tail.

Predicted TTFT/TPOT come from each instance's per-node XGBoost sidecar
(same /predict_latency endpoint RouteBalance already uses). The per-instance
prediction already incorporates current queue state, so this realizes
RouteBalance's "simulation-based dispatching" in the RouteBalance deployment.

Falls back to shortest-queue (num_running + num_waiting) when the sidecar
is unavailable — keeps dispatch total-available rather than failing over
to a less predictable default.
"""
import asyncio
import random as _random
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from .base import DispatchBase, DispatchRequest


class PredictedLatencyDispatch(DispatchBase):
    """Predicted-latency dispatch with optional Po2."""

    def __init__(
        self,
        po2: bool = False,
        latency_predictor: Optional[Callable] = None,
        predict_timeout_s: float = 3.0,
        **kwargs,
    ):
        """
        Args:
            po2: If True, sample 2 candidates and pick min between them.
            latency_predictor: Optional async callable
                `(instance, num_prompt_tokens, num_predicted_output_tokens) ->
                 {"ttft": float, "tpot": float, "e2e_latency": float} | None`.
                Defaults to calling the instance's sidecar /predict_latency
                endpoint the same way route_balance_serve._call_sidecar_predict_latency
                does.
            predict_timeout_s: Per-call timeout for the sidecar predict.
        """
        self._po2 = bool(po2)
        self._predict = latency_predictor
        self._timeout = float(predict_timeout_s)
        self._rng = _random.Random()

    async def _sidecar_predict(
        self,
        inst: Any,
        num_prompt_tokens: int,
        num_predicted_output_tokens: int,
    ) -> Optional[dict]:
        if self._predict is not None:
            return await self._predict(
                inst, num_prompt_tokens, num_predicted_output_tokens
            )
        urls = getattr(inst, "_predictor_urls", None) or []
        if not urls:
            return None
        base = urls[0].rsplit("/", 1)[0]
        url = f"{base}/predict_latency"
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            ) as session:
                async with session.post(
                    url,
                    json={
                        "num_prompt_tokens": num_prompt_tokens,
                        "num_predicted_output_tokens": num_predicted_output_tokens,
                    },
                    ssl=False,
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception:
            return None
        return None

    async def choose_instance(
        self,
        candidates: List[Any],
        req: DispatchRequest,
        *,
        stats_map: Optional[Dict[str, dict]] = None,
    ) -> Any:
        if not candidates:
            raise ValueError("PredictedLatencyDispatch: empty candidate list")

        if self._po2 and len(candidates) > 2:
            sampled = self._rng.sample(candidates, 2)
        else:
            sampled = list(candidates)

        n = req.predicted_output_tokens or req.max_output_tokens
        preds = await asyncio.gather(
            *(
                self._sidecar_predict(inst, req.num_prompt_tokens, n)
                for inst in sampled
            ),
            return_exceptions=True,
        )

        best = None
        best_latency = float("inf")
        for inst, pred in zip(sampled, preds):
            if isinstance(pred, Exception) or not pred:
                # Fallback signal: shortest queue from stats_map.
                s = {}
                if stats_map is not None:
                    s = stats_map.get(
                        getattr(inst, "_instance_id", None), {}
                    ) or {}
                queue = int(s.get("num_running", 0) or 0) + int(
                    s.get("num_waiting", 0) or 0
                )
                # Large constant + queue keeps fallback candidates worse than
                # any actual prediction but preserves relative ordering.
                latency = 1e6 + queue
            else:
                ttft = float(pred.get("ttft", 0.0))
                tpot = float(pred.get("tpot", 0.0))
                latency = ttft + n * tpot
            if latency < best_latency:
                best_latency = latency
                best = inst

        # Should not happen when candidates is non-empty.
        return best if best is not None else sampled[0]
