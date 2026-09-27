import math

import pytest

from routebalance.schedule.budget import affordable_tokens, clamp, request_cost
from routebalance.types import Tier

T = Tier("t", "M", "a30", 1e-7, 2e-7, 256)


def test_clamp_is_the_v3_formula():
    c = clamp(T, budget_cost=1e-4, prompt_tokens=200, requested_max_tokens=1024)
    assert c.affordable_tokens == int((1e-4 - 200 * 1e-7) / 2e-7) and not c.exhausted
    assert c.max_tokens == min(1024, c.affordable_tokens)
    assert clamp(T, 1.0, 200, 64).max_tokens == 64                      # a large budget never raises max_tokens


def test_a_budget_below_the_prompt_cost_leaves_one_token():
    c = clamp(T, budget_cost=1e-5, prompt_tokens=200, requested_max_tokens=1024)
    assert (c.max_tokens, c.affordable_tokens, c.exhausted) == (1, 1, True)


def test_affordable_is_at_least_one_token():
    assert affordable_tokens(T, budget_cost=200 * 1e-7 + 1e-8, prompt_tokens=200) == (1, False)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_no_budget_is_not_a_budget(bad):
    with pytest.raises(ValueError):
        clamp(T, bad, 10, 10)


def test_reported_cost_includes_the_prompt():
    assert request_cost(T, 100, 50) == 100 * 1e-7 + 50 * 2e-7
