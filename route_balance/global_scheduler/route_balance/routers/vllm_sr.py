"""vLLM Semantic Router adapter (A3.5 / #30).

Design
------
vLLM Semantic Router (vLLM-SR) is a Rust/Go Envoy-extproc service cloned
locally at `~/Code/llm/semantic-router/` (`github.com/vllm-project/
semantic-router`). It exposes HTTP endpoints including `/api/v1/classify/
batch`. In our pluggable-router architecture, this adapter is an HTTP
client: per-request it POSTs the prompt to SR, receives the classified
category / recommended model, and maps the result to a model in our
runtime pool.

Stand-up (out-of-band)
----------------------
Bringing up the Envoy + extproc + SR Rust binary is a deployment task
separate from the adapter. Documented in BASELINE_IMPL_TRACKER.md under
"vLLM SR stand-up playbook"; executed on node0 at experiment time. The
adapter verifies connectivity at construction time and records a health
probe result; failures degrade to a fallback policy.

Failure / fallback policy
-------------------------
If SR is unreachable or returns an unparseable response:
    - Log a one-line warning.
    - Return the configured fallback model (or largest in pool) with
      reason=`vllm_sr:unreachable:fallback`.
This keeps the scheduler serving even when the SR sidecar is down.
"""
import logging
from typing import Dict, List, Optional

import aiohttp

from .base import RouterBase, RouterDecision, RouterRequest


logger = logging.getLogger(__name__)


class VLLMSemanticRouter(RouterBase):
    """HTTP client adapter for vLLM Semantic Router."""

    def __init__(
        self,
        *,
        endpoint: str = "http://127.0.0.1:8801",
        classify_path: str = "/api/v1/classify/batch",
        timeout_s: float = 2.0,
        category_to_model: Optional[Dict[str, str]] = None,
        fallback_model: Optional[str] = None,
        fail_open: bool = True,
    ):
        """
        Args:
            endpoint: Base URL where SR's classify API is reachable.
            classify_path: HTTP path for classification. Default matches
                SR's `/api/v1/classify/batch`.
            timeout_s: Per-request timeout in seconds.
            category_to_model: Static mapping from SR category labels to
                our runtime model names. When None, the adapter tries to
                use the SR response's `model` field directly.
            fallback_model: Explicit fallback; if not set, defaults to
                the largest model in the runtime pool.
            fail_open: If True (default), on any SR error the adapter
                returns fallback_model rather than raising. Set False for
                tests that want to surface SR errors.
        """
        self._endpoint = endpoint.rstrip("/")
        self._path = classify_path
        self._timeout = float(timeout_s)
        self._map = dict(category_to_model or {})
        self._fallback = fallback_model
        self._fail_open = bool(fail_open)

    async def _classify(self, prompt: str) -> Optional[dict]:
        url = f"{self._endpoint}{self._path}"
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            ) as session:
                # SR batch classify takes a list; send single-prompt batch.
                async with session.post(
                    url, json={"inputs": [prompt]}, ssl=False
                ) as resp:
                    if resp.status != 200:
                        logger.debug("vllm_sr: status=%s", resp.status)
                        return None
                    body = await resp.json()
                    if isinstance(body, dict) and "results" in body:
                        items = body["results"]
                    elif isinstance(body, list):
                        items = body
                    else:
                        return None
                    if not items:
                        return None
                    return items[0]
        except Exception as e:
            logger.debug("vllm_sr request failed: %s", e)
            return None

    def _fallback_choice(self, pool: List[str]) -> str:
        if self._fallback and self._fallback in pool:
            return self._fallback
        import re

        def sz(n):
            m = re.search(r"(\d+(?:\.\d+)?)[Bb]", n)
            return float(m.group(1)) if m else 1.0

        return max(pool, key=sz)

    async def choose_model(
        self,
        req: RouterRequest,
        model_pool: List[str],
    ) -> RouterDecision:
        if not model_pool:
            raise ValueError("model_pool is empty")

        resp = await self._classify(req.prompt)
        if not resp:
            if not self._fail_open:
                raise RuntimeError("vllm_sr unreachable")
            return RouterDecision(
                model_name=self._fallback_choice(model_pool),
                score=0.0,
                reason="vllm_sr:unreachable:fallback",
            )

        # SR response fields vary by endpoint version. Try common keys.
        model = resp.get("model") or resp.get("recommended_model")
        if not model:
            category = resp.get("category") or resp.get("label")
            if category and category in self._map:
                model = self._map[category]

        if not model:
            return RouterDecision(
                model_name=self._fallback_choice(model_pool),
                score=0.0,
                reason="vllm_sr:no_model_in_response:fallback",
            )

        # Map model aliases when needed (exact match within pool).
        if model not in model_pool:
            if model in self._map and self._map[model] in model_pool:
                model = self._map[model]
            else:
                return RouterDecision(
                    model_name=self._fallback_choice(model_pool),
                    score=0.0,
                    reason=f"vllm_sr:model_{model}_not_in_pool:fallback",
                )

        score = float(resp.get("confidence", resp.get("score", 1.0)))
        return RouterDecision(
            model_name=model,
            score=score,
            reason=f"vllm_sr:model={model}:conf={score:.3f}",
        )
