"""A malformed stream or an early loop stall is never hidden (review of 09-27)."""
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


def _probe_around(block_before_reset: bool) -> dict:
    """The loop blocked 200 ms while the probe sleeps, the reset in the same step (as Service.apply: collect, freeze,
    reset) or 30 ms after it; the summary 100 ms later."""
    async def go() -> dict:
        probe = LagProbe()
        task = asyncio.create_task(probe.run())
        await asyncio.sleep(0.05)
        if block_before_reset:
            time.sleep(0.2)
            probe.reset()
        else:
            probe.reset()
            await asyncio.sleep(0.03)
            time.sleep(0.2)
        await asyncio.sleep(0.1)
        task.cancel()
        return probe.summary()
    return asyncio.run(go())


def test_a_stall_before_the_reset_is_not_the_next_configurations():
    s = _probe_around(block_before_reset=True)
    assert s["samples"] > 0 and s["max_ms"] < 100


def test_a_stall_after_the_reset_is_kept():
    assert _probe_around(block_before_reset=False)["max_ms"] >= 150
