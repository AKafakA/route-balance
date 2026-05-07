"""Random dispatch — uniform over candidates."""
import random as _random
from typing import Any, Dict, List, Optional

from .base import DispatchBase, DispatchRequest


class RandomDispatch(DispatchBase):
    def __init__(self, seed: Optional[int] = None):
        self._rng = _random.Random(seed) if seed is not None else _random

    async def choose_instance(
        self,
        candidates: List[Any],
        req: DispatchRequest,
        *,
        stats_map: Optional[Dict[str, dict]] = None,
    ) -> Any:
        if not candidates:
            raise ValueError("RandomDispatch: empty candidate list")
        return self._rng.choice(candidates)
