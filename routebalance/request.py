"""A request as the scheduler receives it, the instance chosen for it, and the two ways a request is refused."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from routebalance.telemetry import DispatchToken, InstanceStats
from routebalance.types import Instance


class Busy(RuntimeError):
    """A configuration is being applied: the request or the new configuration is refused (503 or 409)."""


class ServeError(RuntimeError):
    """The engine failed the request (502)."""


class Completion(BaseModel):
    """One request as bench/client.py sends it; budget_cost is null except in budget cells."""
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    request_id: str = Field(min_length=1)
    prompt: str
    prompt_tokens: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0)
    repetition_penalty: float = Field(gt=0)
    frequency_penalty: float
    budget_cost: Optional[float] = Field(gt=0)   # null = no budget (never 0)


@dataclass
class Assignment:
    instance: Instance
    token: DispatchToken
    state: InstanceStats        # the chosen instance's state the decision read, before this request
    evidence: dict              # what the decision log keeps about the choice
    timings_ms: dict            # the scheduler's stages before the backend call
