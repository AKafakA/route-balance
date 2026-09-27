"""Core value types shared across the scheduler."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    """One (model, GPU) serving tier with its prices (dollars per token)."""
    tier_id: str
    model: str
    gpu: str
    cost_in: float
    cost_out: float
    max_num_seqs: int      # vLLM's --max-num-seqs for this tier (the engine's concurrent-sequence limit)


@dataclass(frozen=True)
class Instance:
    """One vLLM instance: a tier deployed on a node."""
    instance_id: str
    tier: Tier
    node: str
    address: str
    port: int

    @property
    def model(self) -> str:
        return self.tier.model

    @property
    def base_url(self) -> str:
        return f"http://{self.address}:{self.port}"
