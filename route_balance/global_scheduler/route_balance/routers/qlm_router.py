"""QLM full-router (arXiv 2407.00047) — .

QLM (Queue Management for LLM Serving) is typically cited for its SLO
check (our `qlm_filter.py` already implements that). The paper's core
contribution, however, is a **queue-aware request-to-model assignment**
policy that does:

    1. Maintain a virtual queue state per (model, gpu_type) group.
    2. On request arrival, estimate projected completion time per group
       using virtual queue + predicted per-request cost.
    3. Deadline-aware assignment: argmin over groups subject to
       projected_completion ≤ request.deadline.
    4. If infeasible everywhere: reject-before-enqueue (we degrade by
       returning best-effort = group with smallest constraint violation,
       since our scheduler never rejects — a softer policy documented
       in the adapter).
    5. Update the virtual queue after assignment.

This router operates at the L1 model-selection granularity. Within a
chosen model's instance pool, any L2 dispatcher (round-robin, llumnix--,
block-style) is acceptable — QLM's queue model is per group, not per
instance.

Cold-start behavior
-------------------
When no per-request latency predictor is available (route_balance_score-less
pipeline), we fall back to:
    projected_wait ≈ virtual_queue_size * mean_service_time
where mean_service_time is a config-level estimate per model (default
derived from model size).

If the call provides `predicted_ttft_ms` / `predicted_tpot_ms` via
`RouterRequest.extra`, we use those directly.

Decisions
---------
- We keep the QLMFilter as a separate filter — both QLM-as-router and
  QLM-as-filter can coexist (the filter stays active under any router).
- We use per-model virtual queues (not per-instance) to match the paper's
  granularity; instance-level refinement is the dispatcher's job.
- Deadline = request.ttft_slo + predicted_output_tokens * request.tpot_slo.

Ref: arXiv 2407.00047 Algorithm 1.
"""
from collections import defaultdict
import logging
import time
from typing import Dict, List, Optional

from .base import RouterBase, RouterDecision, RouterRequest


logger = logging.getLogger(__name__)


class QLMRouter(RouterBase):
    """Queue-aware deadline-based model router."""

    def __init__(
        self,
        *,
        mean_service_ms_by_model: Optional[Dict[str, float]] = None,
        default_mean_service_ms: float = 1000.0,
        predicted_output_tokens_default: int = 100,
        rejection_policy: str = "best_effort",
        virtual_queue_decay_s: float = 30.0,
    ):
        """
        Args:
            mean_service_ms_by_model: {model_name: avg serving time in ms}.
                When missing, falls back to `default_mean_service_ms` scaled
                by relative model size (bigger model → longer service).
            default_mean_service_ms: Cold-start fallback service time.
            predicted_output_tokens_default: N used in deadline computation
                when the request doesn't carry a prediction.
            rejection_policy:
                - "best_effort": pick the group with smallest constraint
                  violation (never rejects — matches our always-serve
                  invariant).
                - "strict": raise RuntimeError (not used in production;
                  kept for testing).
            virtual_queue_decay_s: Virtual-queue entries older than this
                are decayed — prevents the queue from growing unboundedly
                when feedback is missing. A completed request removes its
                contribution; if feedback never arrives, decay cleans up.
        """
        self._svc_by_model = dict(mean_service_ms_by_model or {})
        self._svc_default = float(default_mean_service_ms)
        self._pred_N_default = int(predicted_output_tokens_default)
        self._policy = rejection_policy
        self._decay_s = float(virtual_queue_decay_s)

        # Virtual queue: model_name → list[(arrival_ts, est_service_ms)]
        self._vq: Dict[str, list] = defaultdict(list)

    # --- virtual queue maintenance -----------------------------------

    def _decay(self, now: float) -> None:
        cutoff = now - self._decay_s
        for m, entries in list(self._vq.items()):
            self._vq[m] = [(ts, s) for ts, s in entries if ts >= cutoff]

    def _projected_wait_ms(self, model_name: str, now: float) -> float:
        # Sum of remaining service time estimates for entries still in VQ.
        total = 0.0
        for ts, svc in self._vq.get(model_name, []):
            elapsed = max(0.0, (now - ts) * 1000.0)
            remaining = max(0.0, svc - elapsed)
            total += remaining
        return total

    def _service_ms(self, model_name: str, pool: List[str]) -> float:
        if model_name in self._svc_by_model:
            return self._svc_by_model[model_name]
        # Size-scaled fallback: normalize against largest-in-pool.
        import re

        def sz(n):
            m = re.search(r"(\d+(?:\.\d+)?)[Bb]", n)
            return float(m.group(1)) if m else 1.0

        max_sz = max(sz(n) for n in pool) if pool else 1.0
        return self._svc_default * (sz(model_name) / max_sz)

    # --- Router API ---------------------------------------------------

    async def choose_model(
        self,
        req: RouterRequest,
        model_pool: List[str],
    ) -> RouterDecision:
        if not model_pool:
            raise ValueError("model_pool is empty")

        now = time.time()
        self._decay(now)

        # Deadline for completing the request.
        N = int(
            req.extra.get("predicted_output_tokens", self._pred_N_default)
        )
        deadline_ms = req.ttft_slo_ms + N * req.tpot_slo_ms

        # Per-model service cost estimate.
        pred_ttft = float(req.extra.get("predicted_ttft_ms", 0.0))
        pred_tpot = float(req.extra.get("predicted_tpot_ms", 0.0))
        use_predicted = pred_ttft > 0.0 or pred_tpot > 0.0

        # Score each candidate: projected_completion - deadline.
        # Negative = feasible (has slack); positive = infeasible (overshoot).
        best_feasible = None
        best_feasible_slack = float("-inf")  # maximize slack among feasible
        smallest_violation = None
        smallest_violation_excess = float("inf")  # minimize excess if all infeasible

        for model_name in model_pool:
            wait = self._projected_wait_ms(model_name, now)
            if use_predicted:
                svc = pred_ttft + N * pred_tpot
            else:
                svc = self._service_ms(model_name, model_pool)
            projected_completion = wait + svc
            slack = deadline_ms - projected_completion
            if slack >= 0:
                if slack > best_feasible_slack:
                    best_feasible_slack = slack
                    best_feasible = (model_name, projected_completion, svc)
            else:
                excess = -slack
                if excess < smallest_violation_excess:
                    smallest_violation_excess = excess
                    smallest_violation = (model_name, projected_completion, svc)

        # Assign
        if best_feasible is not None:
            chosen, proj, svc = best_feasible
            reason = (
                f"qlm:feasible:wait+svc={proj:.0f}ms"
                f":deadline={deadline_ms:.0f}ms"
                f":slack={best_feasible_slack:.0f}ms"
            )
            score = best_feasible_slack
        elif self._policy == "strict":
            raise RuntimeError(
                f"QLMRouter: no feasible model; strict policy rejects."
            )
        else:
            chosen, proj, svc = smallest_violation
            reason = (
                f"qlm:infeasible:best_effort:excess="
                f"{smallest_violation_excess:.0f}ms"
                f":wait+svc={proj:.0f}ms>deadline={deadline_ms:.0f}ms"
            )
            score = -smallest_violation_excess

        # Update virtual queue with the assigned request.
        self._vq[chosen].append((now, svc))

        return RouterDecision(model_name=chosen, score=score, reason=reason)

    # --- Optional feedback hook ---------------------------------------

    def on_complete(self, model_name: str, arrival_ts: float) -> None:
        """Remove the completed request from virtual queue.

        Called by instrumentation once a response is received; if not
        wired, the decay mechanism cleans up stale entries.
        """
        entries = self._vq.get(model_name, [])
        # Remove the first entry whose arrival_ts matches (closest).
        for i, (ts, _) in enumerate(entries):
            if abs(ts - arrival_ts) < 1e-3:
                entries.pop(i)
                break
