"""A malformed stream or an early loop stall is never hidden."""
import asyncio
import time

import pytest

from routebalance.backend import _choices, _completion_tokens
from routebalance.decision_log import LagProbe


@pytest.mark.parametrize("event", [{}, {"choices": None}, {"choices": "x"}])
def test_an_event_without_choices_is_malformed(event):
    with pytest.raises(ValueError, match="malformed choices"):
        _choices(event)


@pytest.mark.parametrize("n", [-1, True, 1.0])
def test_completion_counts_are_non_negative_integers(n):
    with pytest.raises(ValueError, match="malformed usage"):
        _completion_tokens({"usage": {"completion_tokens": n}})
    assert _completion_tokens({"usage": {"completion_tokens": 0}}) == 0 and _completion_tokens({}) is None


def test_the_loop_lag_summary_keeps_an_early_stall():
    probe = LagProbe()
    probe._samples = [100.0] + [0.1] * 50_000
    assert probe.summary()["max_ms"] == 100.0 and probe.summary()["samples"] == 50_001


def _probe_around(before_ms: float, after_ms: float, yield_between: bool) -> dict:
    """The loop blocked before_ms while the probe sleeps, then the reset (as Service.apply: collect, freeze, reset, no
    await between), then blocked after_ms, with or without a turn of the loop between; the summary 100 ms later."""
    async def go() -> dict:
        probe = LagProbe()
        task = asyncio.create_task(probe.run())
        await asyncio.sleep(0.05)
        time.sleep(before_ms / 1000)
        probe.reset()
        if yield_between:
            await asyncio.sleep(0.03)
        time.sleep(after_ms / 1000)
        await asyncio.sleep(0.1)
        task.cancel()
        return probe.summary()
    return asyncio.run(go())


def test_a_stall_before_the_reset_is_not_the_next_configurations():
    s = _probe_around(200, 0, yield_between=False)
    assert s["samples"] > 0 and s["max_ms"] < 100


def test_a_stall_right_after_the_reset_is_kept_by_the_sample_begun_before_it():
    s = _probe_around(200, 200, yield_between=False)
    assert 150 <= s["max_ms"] < 350          # the part after the reset only


def test_a_later_stall_after_the_reset_is_kept():
    assert _probe_around(0, 200, yield_between=True)["max_ms"] >= 150
