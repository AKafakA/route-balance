"""Llumnix-- dispatch — scheduling-only Llumnix (no KV migration).

Signal (per RouteBalance paper §6.7, socc_llumnix_comparison.tex L32;
original: Sun et al., "Llumnix", OSDI 2024):

    load = -(kv_free_blocks / max(num_running + num_waiting, 1))
    pick argmin(load)     # equivalently argmax(free_per_active)

i.e. choose the instance with the most free KV cache blocks per active
request (running + waiting). This is a memory-per-request pressure metric.

Where kv_free_blocks comes from /instance_stats — the same field RouteBalance's
own scheduler consumes. If not reported by an instance, falls back to
(1 - kv_cache_utilization) * max_num_seqs as a proxy.

Task: A1.1 . Ports the metric from
route_balance/global_scheduler/api_server.py:187-209 into RouteBalance's dispatcher plugin.
"""
from typing import Any, Dict, List, Optional

from .base import DispatchBase, DispatchRequest


class LlumnixMinusDispatch(DispatchBase):
    """Scheduling-only Llumnix-- (min-load within model pool)."""

    def __init__(self, **kwargs):
        # Accept-and-ignore unknown kwargs for forward-compat with the
        # factory. Llumnix-- is parameter-free.
        pass

    async def choose_instance(
        self,
        candidates: List[Any],
        req: DispatchRequest,
        *,
        stats_map: Optional[Dict[str, dict]] = None,
    ) -> Any:
        if not candidates:
            raise ValueError("LlumnixMinusDispatch: empty candidate list")

        best = candidates[0]
        best_score = float("-inf")  # maximize free_per_active
        for inst in candidates:
            s = {}
            if stats_map is not None:
                s = stats_map.get(getattr(inst, "_instance_id", None), {}) or {}

            free_blocks = s.get("kv_free_blocks")
            if free_blocks is None:
                # Proxy: (1 - util) * max_num_seqs. Keeps same direction
                # (higher = more free capacity) even if magnitudes differ.
                util = float(s.get("kv_cache_utilization", 0.0) or 0.0)
                max_seqs = int(s.get("max_num_seqs", 256) or 256)
                free_blocks = max(0.0, (1.0 - util) * float(max_seqs))
            else:
                free_blocks = float(free_blocks)

            active = int(s.get("num_running", 0) or 0) + int(
                s.get("num_waiting", 0) or 0
            )
            score = free_blocks / max(active, 1)
            if score > best_score:
                best_score = score
                best = inst
        return best
