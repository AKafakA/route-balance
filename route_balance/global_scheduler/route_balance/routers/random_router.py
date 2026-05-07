"""Random router — uniform over pool. Useful as a trivial baseline."""
import random as _random
from typing import List, Optional

from .base import RouterBase, RouterDecision, RouterRequest


class RandomRouter(RouterBase):
    def __init__(self, seed: Optional[int] = None):
        self._rng = _random.Random(seed) if seed is not None else _random

    async def choose_model(
        self,
        req: RouterRequest,
        model_pool: List[str],
    ) -> RouterDecision:
        if not model_pool:
            raise ValueError("model_pool is empty")
        return RouterDecision(
            model_name=self._rng.choice(model_pool),
            score=1.0 / len(model_pool),
            reason="random",
        )
