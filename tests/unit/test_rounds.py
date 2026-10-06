import asyncio
import math
import time

import pytest

from routebalance.config import Rounds
from routebalance.schedule.rounds import RoundCollector, adaptive_limits, busy_ratio
from routebalance.telemetry import InstanceStats, Load


def load(stats_waiting=0, engine_waiting=0):
    return Load(InstanceStats(1, stats_waiting, 0.1, 0, 256, 0), age_ms=10.0, in_flight=1 + stats_waiting,
                engine_waiting=engine_waiting)


@pytest.mark.parametrize("ratio,limits", [(0.0, (4, 50.0)), (0.29, (4, 50.0)), (0.3, (16, 100.0)),
                                          (0.69, (16, 100.0)), (0.7, (32, 200.0)), (1.0, (32, 200.0))])
def test_adaptive_limits_follow_the_v3_thresholds(ratio, limits):
    assert adaptive_limits(ratio) == limits


def test_nan_ratio_raises():
    with pytest.raises(ValueError):
        adaptive_limits(math.nan)


def test_busy_means_the_engine_itself_has_waiting_requests():
    # dead-reckoned waiting (dispatched, not yet visible as running) does not make an instance busy
    loads = {"a": load(stats_waiting=3, engine_waiting=0), "b": load(engine_waiting=2), "c": load(), "d": load()}
    assert busy_ratio(loads) == 0.25


def test_adaptive_collector_starts_idle_and_adapts():
    c = RoundCollector(Rounds(adaptive=True))
    assert (c.max_batch, c.timeout_ms) == (4, 50.0)
    c.adapt({"a": load(engine_waiting=1), "b": load(engine_waiting=1), "c": load()})
    assert (c.max_batch, c.timeout_ms) == (16, 100.0)


@pytest.mark.parametrize("waiting,taken,limit", [(2, 2, 4), (4, 4, 4), (9, 9, 9), (40, 32, 32)])
def test_an_adaptive_round_takes_the_schedulers_backlog_up_to_the_largest_size(waiting, taken, limit):
    """at the idle limits (4 / 50 ms) a round takes every request already waiting, up to 32, at once."""
    async def scenario():
        c = RoundCollector(Rounds(adaptive=True))
        for i in range(waiting):
            c.submit(i)
        t0 = time.monotonic()
        batch = await c.next_round()
        return batch, time.monotonic() - t0, c.limits, c.backlog, c.pending()
    batch, elapsed, limits, backlog, left = asyncio.run(scenario())
    assert batch == list(range(taken)) and limits == (limit, 50.0) and backlog == waiting and left == waiting - taken
    if waiting >= limit:
        assert elapsed < 0.05                  # a round that the backlog fills closes without waiting


def test_a_round_opened_by_an_arrival_keeps_the_ratios_limits():
    async def scenario():
        c = RoundCollector(Rounds(adaptive=True))
        loop = asyncio.get_running_loop()
        loop.call_later(0.01, c.submit, "a")
        batch = await c.next_round()
        return batch, c.limits, c.backlog
    assert asyncio.run(scenario()) == (["a"], (4, 50.0), 0)


def test_a_fixed_round_ignores_the_backlog():
    async def scenario():
        c = RoundCollector(Rounds(adaptive=False, max_batch=4, timeout_ms=1000.0))
        for i in range(9):
            c.submit(i)
        return await c.next_round(), c.limits, c.backlog
    assert asyncio.run(scenario()) == ([0, 1, 2, 3], (4, 1000.0), 9)


def test_fixed_collector_never_adapts():
    c = RoundCollector(Rounds(adaptive=False, max_batch=1, timeout_ms=5.0))
    c.adapt({"a": load(engine_waiting=1)})
    assert (c.max_batch, c.timeout_ms) == (1, 5.0)


def test_a_full_round_closes_without_waiting_and_the_rest_waits_for_the_next():
    async def scenario():
        c = RoundCollector(Rounds(adaptive=False, max_batch=3, timeout_ms=10_000.0))
        for i in range(5):
            c.submit(i)
        t0 = time.monotonic()
        first = await c.next_round()
        return first, time.monotonic() - t0, c.pending()
    first, elapsed, left = asyncio.run(scenario())
    assert first == [0, 1, 2] and elapsed < 1.0 and left == 2


def test_a_round_closes_at_its_timeout_and_takes_arrivals_until_then():
    async def scenario():
        c = RoundCollector(Rounds(adaptive=False, max_batch=10, timeout_ms=100.0))
        c.submit("a")

        async def late():
            await asyncio.sleep(0.02)
            c.submit("b")
            await asyncio.sleep(0.3)
            c.submit("c")
        task = asyncio.create_task(late())
        t0 = time.monotonic()
        got = await c.next_round()
        elapsed = time.monotonic() - t0
        await task
        return got, elapsed, c.pending()
    got, elapsed, left = asyncio.run(scenario())
    assert got == ["a", "b"] and 0.09 <= elapsed < 0.3 and left == 1


def test_a_cancelled_collection_keeps_its_requests_for_the_next_round():
    async def scenario():
        c = RoundCollector(Rounds(adaptive=False, max_batch=4, timeout_ms=1000.0))
        c.submit("a")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(c.next_round(), 0.01)     # cancelled while waiting for the second request
        assert c.pending() == 1
        c.submit("b")
        c.max_batch = 2
        return await c.next_round()
    assert asyncio.run(scenario()) == ["a", "b"]


def test_a_cancel_is_never_swallowed_when_a_request_lands_in_the_same_tick():
    async def scenario():
        swallowed = kept = 0
        for _ in range(200):
            c = RoundCollector(Rounds(adaptive=False, max_batch=4, timeout_ms=1000.0))
            c.submit("a")
            task = asyncio.create_task(c.next_round())
            await asyncio.sleep(0)                             # task takes "a", waits for the second request
            await asyncio.sleep(0)
            c.submit("b")                                      # wakes the waiting get ...
            task.cancel()                                      # ... in the same tick as the cancel
            try:
                await task
                swallowed += 1
            except asyncio.CancelledError:
                pass
            kept += c.pending() == 2
        return swallowed, kept
    assert asyncio.run(scenario()) == (0, 200)                 # cancelled every time, and nothing lost


def test_drain_returns_carried_then_queued_requests():
    async def scenario():
        c = RoundCollector(Rounds(adaptive=False, max_batch=4, timeout_ms=1000.0))
        c.submit("a")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(c.next_round(), 0.01)
        c.submit("b")
        return c.drain(), c.pending()
    assert asyncio.run(scenario()) == (["a", "b"], 0)
