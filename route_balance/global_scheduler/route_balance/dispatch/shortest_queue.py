"""Shortest-queue dispatch — pick instance with fewest running+waiting reqs."""
from typing import Any, Dict, List, Optional

from .base import DispatchBase, DispatchRequest


class ShortestQueueDispatch(DispatchBase):
    async def choose_instance(
        self,
        candidates: List[Any],
        req: DispatchRequest,
        *,
        stats_map: Optional[Dict[str, dict]] = None,
    ) -> Any:
        if not candidates:
            raise ValueError("ShortestQueueDispatch: empty candidate list")

        best = candidates[0]
        best_q = float("inf")
        for inst in candidates:
            s = None
            if stats_map is not None:
                s = stats_map.get(getattr(inst, "_instance_id", None))
            if s is None:
                # Fallback: use per-instance counter (less accurate but no RPC).
                q = getattr(inst, "total_request", 0)
            else:
                q = s.get("num_running", 0) + s.get("num_waiting", 0)
            if q < best_q:
                best_q = q
                best = inst
        return best
