"""Scheduling rounds (v3 BatchQueue).

A round opens with the first waiting request and closes when it holds max_batch requests or timeout_ms
after it opened, whichever comes first. Requests that arrive while a round is being scheduled wait for the
next one.

Adaptive limits follow the busy-instance ratio, the share of instances whose engine has requests waiting
(the engine's own gauge: work it has received but cannot run yet). The ratio is read with each round's
loads and sets the limits of the rounds after it:
    ratio < 0.3: 4 requests / 50 ms      ratio < 0.7: 16 / 100 ms      otherwise: 32 / 200 ms
A run starts at the idle limits (v3 used a configured 8 / 100 ms until its first adaptation). With
adaptive off, the configured max_batch and timeout_ms apply to every round (the round-size ablations).

The scheduler's own backlog: an adaptive round may also take every request already waiting in the
scheduler when it opens, up to the largest adaptive size (32); its time window stays the ratio's. The engines'
ratio sees only work the engines hold, so when requests queue at the scheduler instead (at 42 req/s every round
was full at 4 while the engines had none waiting, and requests waited 2.2 s for a round), the round size now
follows that queue. With no backlog at the open, a round's limits are the ratio's, as before.
"""
from __future__ import annotations

import asyncio
import math
from typing import Generic, Mapping, TypeVar

from routebalance.config import Rounds
from routebalance.telemetry import Load

T = TypeVar("T")

ADAPTIVE_LIMITS = ((0.3, 4, 50.0), (0.7, 16, 100.0), (math.inf, 32, 200.0))   # (ratio below, size, timeout ms)
BACKLOG_CAP = ADAPTIVE_LIMITS[-1][1]      # a round takes the scheduler's backlog up to the largest size


def busy_ratio(loads: Mapping[str, Load]) -> float:
    if not loads:
        raise ValueError("no instances to measure")
    return sum(1 for load in loads.values() if load.engine_waiting > 0) / len(loads)


def adaptive_limits(ratio: float) -> tuple[int, float]:
    for below, size, timeout_ms in ADAPTIVE_LIMITS:
        if ratio < below:
            return size, timeout_ms
    raise ValueError(f"busy ratio {ratio!r} is not a number")


class RoundCollector(Generic[T]):
    def __init__(self, cfg: Rounds):
        self._adaptive = cfg.adaptive
        if cfg.adaptive:
            self.max_batch, self.timeout_ms = adaptive_limits(0.0)
        else:
            self.max_batch, self.timeout_ms = cfg.max_batch, cfg.timeout_ms
        self._queue: asyncio.Queue[T] = asyncio.Queue()
        self._carry: list[T] = []    # taken from the queue by a cancelled collection; they open the next round
        self.limits = (self.max_batch, self.timeout_ms)   # the last round's own limits, as it collected
        self.backlog = 0                                  # the requests already waiting when that round opened

    def submit(self, item: T) -> None:
        self._queue.put_nowait(item)

    def pending(self) -> int:
        return len(self._carry) + self._queue.qsize()

    def drain(self) -> list[T]:
        """Remove and return every request not yet in a round (carried first, then queued, in order)."""
        items, self._carry = self._carry, []
        while not self._queue.empty():
            items.append(self._queue.get_nowait())
        return items

    async def next_round(self) -> list[T]:
        """Wait for the first request, then collect until the round is full or its timeout expires.
        If the collection is cancelled, the requests it took are kept, in order, for the next round."""
        loop = asyncio.get_running_loop()
        batch, self._carry = self._carry, []
        self.backlog = len(batch) + self._queue.qsize()
        size = max(self.max_batch, min(self.backlog, BACKLOG_CAP)) if self._adaptive else self.max_batch
        self.limits = (size, self.timeout_ms)
        try:
            if not batch:
                batch.append(await self._queue.get())
            deadline = loop.time() + self.timeout_ms / 1000.0
            while len(batch) < size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                if not await self._take(batch, remaining):
                    break
        except asyncio.CancelledError:
            self._carry = batch
            raise
        return batch

    async def _take(self, batch: list[T], timeout_s: float) -> bool:
        """Append the next request if one arrives within timeout_s. Not asyncio.wait_for: on 3.10 it can
        return a result and swallow a cancellation that lands in the same tick (bpo-42130)."""
        getter = asyncio.ensure_future(self._queue.get())
        try:
            done, _ = await asyncio.wait({getter}, timeout=timeout_s)
        except asyncio.CancelledError:
            if getter.done() and not getter.cancelled():
                batch.append(getter.result())      # it took a request: keep it with the carry
            else:
                getter.cancel()                    # a cancelled Queue.get leaves its item in the queue
            raise
        if getter in done:
            batch.append(getter.result())
            return True
        getter.cancel()
        return False

    def adapt(self, loads: Mapping[str, Load]) -> None:
        if self._adaptive:
            self.max_batch, self.timeout_ms = adaptive_limits(busy_ratio(loads))
