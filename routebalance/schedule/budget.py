"""Per-request cost budgets (appendix experiment).

A request may carry budget_cost, in dollars; 0 means no budget. Two rules use it:

  clamp (every system, after the instance is chosen; v3 formula): on the chosen tier,
      remaining  = budget_cost - prompt_tokens * cost_in
      affordable = 1 if remaining <= 0 (the budget cannot pay for the prompt: "exhausted"),
                   else max(1, floor(remaining / cost_out))
      max_tokens = min(requested max_tokens, affordable)
  admission (RouteBalance only, when configured; v3's budget filter, routebalance/schedule/assign.py
      Admission): a candidate is admitted when prompt_tokens * cost_in + min(expected length, max_tokens) *
      cost_out <= budget_cost; when none is, the cheapest candidates by worst-case cost (max_tokens) remain.

The reported cost of a finished request includes its prompt: prompt_tokens * cost_in + output_tokens *
cost_out. The scheduling objective's cost term counts the prompt the same way, with the expected length in
place of the output tokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from routebalance.types import Tier



@dataclass(frozen=True)
class Clamp:
    max_tokens: int
    affordable_tokens: int
    exhausted: bool          # the budget does not even cover the prompt


def _check_budget(budget_cost: float) -> None:
    if not (math.isfinite(budget_cost) and budget_cost > 0):
        raise ValueError(f"budget_cost must be a positive number of dollars, got {budget_cost!r}")


def affordable_tokens(tier: Tier, budget_cost: float, prompt_tokens: int) -> tuple[int, bool]:
    _check_budget(budget_cost)
    remaining = budget_cost - prompt_tokens * tier.cost_in
    if remaining <= 0:
        return 1, True
    return max(1, int(remaining / tier.cost_out)), False


def clamp(tier: Tier, budget_cost: float, prompt_tokens: int, requested_max_tokens: int) -> Clamp:
    affordable, exhausted = affordable_tokens(tier, budget_cost, prompt_tokens)
    return Clamp(min(requested_max_tokens, affordable), affordable, exhausted)


def request_cost(tier: Tier, prompt_tokens: int, output_tokens: int) -> float:
    return prompt_tokens * tier.cost_in + output_tokens * tier.cost_out
