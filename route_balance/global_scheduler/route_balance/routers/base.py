"""Router base class — L1 model selection.

Contract
--------
A router receives a request and the set of currently-available model names
(unique across all instances) and returns one model name.

Routers must NOT pick an instance — that is the Dispatcher's job. This is
the classical two-layer split: router chooses the *model*, dispatcher chooses
the *replica*.

Routers may be learned (RouteBalance native, best-route, RouteLLM, Avengers-Pro,
OmniRouter), rule-based (Dual-Pool EMA, random, null_fixed), or external
blackbox endpoints (vLLM Semantic Router via HTTP).

Routers may be asynchronous — e.g. external HTTP routers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RouterRequest:
    """Minimal request context handed to a router."""
    prompt: str
    num_prompt_tokens: int = 0
    max_output_tokens: int = 256
    budget_tokens: int = 256
    ttft_slo_ms: float = 10000.0
    tpot_slo_ms: float = 200.0
    quality_min: float = 0.0
    request_id: str = ""
    # Optional metadata from client (e.g. dataset tag, prompt class).
    extra: dict = field(default_factory=dict)


@dataclass
class RouterDecision:
    """Router output.

    `model_name` is the chosen model id (matching an instance._model_name).
    `score` is an optional confidence/score for logging.
    `reason` is a short tag for telemetry.
    """
    model_name: str
    score: float = 1.0
    reason: str = ""


class RouterBase(ABC):
    """L1 Router — selects a model for an incoming request."""

    @abstractmethod
    async def choose_model(
        self,
        req: RouterRequest,
        model_pool: List[str],
    ) -> RouterDecision:
        """Return the chosen model name from `model_pool`.

        Implementations MUST return a name that is in `model_pool`, otherwise
        the dispatcher will have no candidate instances. A router that wants
        to express abstention should return the most permissive model and set
        `reason='abstain'`.
        """
        raise NotImplementedError
