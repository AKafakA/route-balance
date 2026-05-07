"""Null/fixed router — always returns the same model.

Use-cases:
- RouteBalance-as-baseline: no model selection (always 72B), dispatch by predicted
  latency. Matches the operator-documented framing that "RouteBalance is simple —
  uses predicted latency for dispatching and ignores the model selection
  part".
- Debugging other components (dispatch, filter) with a trivial router.
"""
from typing import List, Optional

from .base import RouterBase, RouterDecision, RouterRequest


class NullFixedRouter(RouterBase):
    """Always returns the configured fixed model name."""

    def __init__(self, model_name: Optional[str] = None, pool_preference: str = "largest"):
        """
        Args:
            model_name: If set, always return this model. Must exist in the
                pool at request time or router raises.
            pool_preference: If model_name is not set, pick deterministically
                from the pool. One of "largest" (default), "smallest",
                "alphabetical_first".
        """
        self._fixed = model_name
        self._pref = pool_preference

    async def choose_model(
        self,
        req: RouterRequest,
        model_pool: List[str],
    ) -> RouterDecision:
        if not model_pool:
            raise ValueError("model_pool is empty")

        if self._fixed is not None:
            if self._fixed not in model_pool:
                raise ValueError(
                    f"NullFixedRouter configured for {self._fixed!r} but "
                    f"pool is {model_pool!r}"
                )
            return RouterDecision(
                model_name=self._fixed, score=1.0, reason="null_fixed:explicit"
            )

        if self._pref == "smallest":
            name = min(model_pool, key=_size_proxy)
        elif self._pref == "alphabetical_first":
            name = sorted(model_pool)[0]
        else:
            name = max(model_pool, key=_size_proxy)
        return RouterDecision(
            model_name=name, score=1.0, reason=f"null_fixed:{self._pref}"
        )


def _size_proxy(model_name: str) -> float:
    import re
    m = re.search(r"(\d+(?:\.\d+)?)[Bb]", model_name)
    return float(m.group(1)) if m else 1.0
