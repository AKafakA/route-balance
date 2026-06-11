"""num_waiting dispatcher — pick instance with smallest num_waiting backlog.

Distinct from shortest_queue (running + waiting): an instance at full
concurrent batch depth with zero queue appears fully loaded under
shortest_queue but "empty" under num_waiting. Paper ablation for L2.

Fallbacks:
    no stats → per-instance total_request counter
    stats lack num_waiting → (1 - kv_cache_utilization) * max_num_seqs proxy

Task: A1.3 (#35).
"""
from typing import Any, Dict, List, Optional

from .base import DispatchBase, DispatchRequest


class NumWaitingDispatch(DispatchBase):
    async def choose_instance(
        self,
        candidates: List[Any],
        req: DispatchRequest,
        *,
        stats_map: Optional[Dict[str, dict]] = None,
    ) -> Any:
        if not candidates:
            raise ValueError("NumWaitingDispatch: empty candidate list")

        best = candidates[0]
        best_waiting = float("inf")
        for inst in candidates:
            s = None
            if stats_map is not None:
                s = stats_map.get(getattr(inst, "_instance_id", None))
            if s is None:
                waiting = getattr(inst, "total_request", 0)
            elif "num_waiting" in s:
                waiting = int(s["num_waiting"] or 0)
            else:
                util = float(s.get("kv_cache_utilization", 0.0) or 0.0)
                max_seqs = int(s.get("max_num_seqs", 256) or 256)
                # Proxy: implied backlog if queue exceeds capacity.
                waiting = max(0, int(util * max_seqs) - max_seqs // 2)

            if waiting < best_waiting:
                best_waiting = waiting
                best = inst
        return best
