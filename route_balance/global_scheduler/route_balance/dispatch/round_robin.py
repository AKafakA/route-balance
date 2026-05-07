"""Round-robin dispatch — rotate through candidates by a per-model counter."""
from typing import Any, Dict, List, Optional

from .base import DispatchBase, DispatchDecision, DispatchRequest


class RoundRobinDispatch(DispatchBase):
    requires_stats = False

    def __init__(self):
        # counter[model_name] = next index
        self._counters: Dict[str, int] = {}

    async def choose_instance(
        self,
        candidates: List[Any],
        req: DispatchRequest,
        *,
        stats_map: Optional[Dict[str, dict]] = None,
    ) -> Any:
        if not candidates:
            raise ValueError("RoundRobinDispatch: empty candidate list")
        # All candidates share the same _model_name by construction.
        model_name = getattr(candidates[0], "_model_name", "_default")
        idx = self._counters.get(model_name, 0) % len(candidates)
        self._counters[model_name] = idx + 1
        return candidates[idx]
