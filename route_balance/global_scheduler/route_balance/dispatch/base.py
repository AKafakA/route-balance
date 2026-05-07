"""Dispatcher base class — L2 instance selection.

Contract
--------
Given a chosen model_id (from L1 Router) and all instances serving that
model, pick one instance. Dispatchers may be round-robin, min-load, or
latency-predictive.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DispatchRequest:
    """Per-request context for the dispatcher."""
    prompt: str
    num_prompt_tokens: int = 0
    max_output_tokens: int = 256
    predicted_output_tokens: Optional[int] = None
    ttft_slo_ms: float = 10000.0
    tpot_slo_ms: float = 200.0
    request_id: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class DispatchDecision:
    """Dispatcher output."""
    instance_id: str
    score: float = 1.0
    reason: str = ""


class DispatchBase(ABC):
    """L2 Dispatcher — selects instance within chosen-model pool."""

    # Subclasses set True if their `choose_instance` reads `stats_map`
    # (e.g. shortest-queue uses queue depth). When False, the caller can
    # skip `_fetch_all_instance_stats` for a cheap path. Default True for
    # safety — override to False where genuinely unused.
    requires_stats: bool = True

    @abstractmethod
    async def choose_instance(
        self,
        candidates: List[Any],
        req: DispatchRequest,
        *,
        stats_map: Optional[Dict[str, dict]] = None,
    ) -> Any:
        """Return one instance from `candidates`.

        Args:
            candidates: Instance objects whose _model_name matches the
                router-chosen model. Guaranteed non-empty by the caller.
            req: Per-request context.
            stats_map: Optional pre-fetched {instance_id: /instance_stats}
                (avoids double round-trips when the caller already has them).

        Returns:
            One Instance from `candidates`.
        """
        raise NotImplementedError
