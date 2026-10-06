"""Core value types shared across the scheduler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Tier:
    """One (model, GPU) serving tier with its prices (dollars per token)."""
    tier_id: str
    model: str
    gpu: str
    cost_in: float
    cost_out: float
    max_num_seqs: int      # vLLM's --max-num-seqs for this tier (the engine's concurrent-sequence limit)
    stop_token_ids: tuple[int, ...]   # every request to this tier also stops at these token ids


@dataclass(frozen=True)
class Instance:
    """One vLLM instance: a tier deployed on a node."""
    instance_id: str
    tier: Tier
    node: str
    address: str                   # the node's name for deploys and ssh (hosts.json address)
    port: int
    service_address: Optional[str] = None   # where the engine is served to the scheduler and the drivers, when not
    # `address` (the experiment network); every HTTP request to the engine goes there

    @property
    def model(self) -> str:
        return self.tier.model

    @property
    def base_url(self) -> str:
        host = self.address if self.service_address is None else self.service_address
        return f"http://{host}:{self.port}"
