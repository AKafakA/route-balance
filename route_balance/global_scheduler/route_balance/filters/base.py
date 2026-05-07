"""
Base interface for SLO filters.

All filters take a request and a list of candidate instances, and return
the subset of instances that can serve the request within SLO constraints.

Filters are pluggable — the scheduler config selects which filter to use.
This enables A/B comparison of different SLO enforcement strategies
with the same scheduling algorithm (e.g., round-robin for ablation).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SLOConstraints:
    """Per-request SLO constraints."""
    ttft_ms: float = 10000.0      # Time to first token SLO (ms)
    tpot_ms: float = 200.0        # Time per output token SLO (ms)
    budget_tokens: int = 256      # Max output token budget
    quality_min: float = 0.0      # Minimum quality score
    # Monetary budget (USD). When > 0, filters reject instances whose
    # worst-case cost (input + max_output_tokens × cost_per_output_token)
    # exceeds budget_cost.
    budget_cost: float = 0.0
    num_prompt_tokens: int = 0    # Used to compute worst-case input cost.
    max_output_tokens: int = 256  # Upper bound on output tokens (max_tokens
                                  # sent to vLLM). Used for worst-case budget check.


@dataclass
class InstanceState:
    """Instance state snapshot for filtering decisions."""
    instance_id: str
    model_name: str
    gpu_type: str
    # Predicted latencies (from XGBoost sidecar)
    predicted_ttft_ms: float = 0.0
    predicted_tpot_ms: float = 0.0
    predicted_e2e_ms: float = 0.0
    # Queue state
    num_running: int = 0
    num_waiting: int = 0
    kv_cache_utilization: float = 0.0
    # Confidence bounds (for QLM-style filtering)
    ttft_std_ms: float = 0.0      # Standard deviation of TTFT prediction
    tpot_std_ms: float = 0.0      # Standard deviation of TPOT prediction
    # Pricing (USD per token). Populated from instance_meta for budget filter.
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0


@dataclass
class FilterResult:
    """Result of filtering decision for one (request, instance) pair."""
    instance_id: str
    accepted: bool
    reason: str = ""
    # For metrics
    predicted_ttft_ms: float = 0.0
    predicted_tpot_ms: float = 0.0
    confidence: float = 1.0       # Confidence that SLO will be met


class SLOFilter(ABC):
    """Base class for SLO filters."""

    @abstractmethod
    def filter(
        self,
        slo: SLOConstraints,
        instances: List[InstanceState],
        predicted_output_tokens: int = 100,
    ) -> List[FilterResult]:
        """Filter instances that can meet the request's SLO.

        Args:
            slo: Per-request SLO constraints.
            instances: List of candidate instance states.
            predicted_output_tokens: Predicted output token count for the request.

        Returns:
            List of FilterResult for each instance (accepted/rejected with reason).
        """
        pass

    def get_eligible(
        self,
        slo: SLOConstraints,
        instances: List[InstanceState],
        predicted_output_tokens: int = 100,
    ) -> List[InstanceState]:
        """Convenience: return only accepted instances."""
        results = self.filter(slo, instances, predicted_output_tokens)
        accepted_ids = {r.instance_id for r in results if r.accepted}
        return [inst for inst in instances if inst.instance_id in accepted_ids]
