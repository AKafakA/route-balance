"""Arrival schedules for the open-loop benchmark client.

The generator:
  1. seed the legacy MT19937 stream with the run seed;
  2. draw one gap per request, in request order: gap_i ~ Gamma(shape=b, scale=1/(rate*b));
  3. take the running sum;
  4. rescale every time by (n/rate) / (last time), so the last arrival lands at n/rate (to within one
     float rounding of the product).
b = 1 is a Poisson process; the squared coefficient of variation of the gaps is 1/b (b = 0.25 gives CV^2 = 4).
The first arrival comes after t = 0, never at it.

A private RandomState(seed) draws the same stream as seeding numpy's global generator, and no other library
call can draw from it between seeding and use.

Time-varying load is a sequence of constant-rate segments, a step trace such as 6 -> 30 -> 6 req/s.
Segment k holds n_k requests at rate r_k and spans (T_k, T_k + n_k/r_k], where T_k is the end of the
previous segment. Its gaps continue the same stream and are rescaled so that its last arrival is at its end
(to within one rounding). Every segment therefore carries exactly its request count and ends on time, and a
one-segment trace is the generator above.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
from pydantic import Field

from bench.schema import Strict

# the digest hashes the sleep intervals a pacing loop would request with the clock frozen at this value:
# interval = (clock + t) - clock. That rounds t to the clock's float spacing (~1.2e-10 s), and a non-positive
# interval never reaches the sleep, so it is not digested.
FROZEN_CLOCK_S = 1_000_000.0


class Segment(Strict):
    n_requests: int = Field(ge=1)
    rate: float = Field(gt=0)            # requests per second


class Trace(Strict):
    burstiness: float = Field(gt=0)      # gamma shape b; 1 = Poisson
    segments: list[Segment] = Field(min_length=1)

    @property
    def n_requests(self) -> int:
        return sum(s.n_requests for s in self.segments)


def arrival_times(trace: Trace, seed: int) -> list[float]:
    """Arrival time of every request, in seconds after the start of the run, in send order."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError(f"seed must be an int in [0, 2**32), got {seed!r}")
    stream = np.random.RandomState(seed)
    times: list[float] = []
    start = 0.0
    for seg in trace.segments:
        times += [start + t for t in _segment(stream, seg.n_requests, seg.rate, trace.burstiness)]
        start += seg.n_requests / seg.rate
    return times


def _segment(stream: np.random.RandomState, n: int, rate: float, burstiness: float) -> list[float]:
    scale = 1.0 / (rate * burstiness)
    running = 0.0
    cumulative = []
    for _ in range(n):                    # one draw per request; the order is the stream
        running += stream.gamma(shape=burstiness, scale=scale)
        cumulative.append(running)
    if not cumulative[-1] > 0:
        raise ValueError("all gamma draws were zero; the segment cannot be rescaled")
    factor = (n / rate) / cumulative[-1]
    return [float(t * factor) for t in cumulative]


def schedule_digest(times: Sequence[float]) -> str:
    """The arrival schedule's digest (16 hex digits)."""
    h = hashlib.sha256()
    for t in times:
        interval = (FROZEN_CLOCK_S + t) - FROZEN_CLOCK_S
        if interval > 0:
            h.update(f"{interval:.9f}|".encode())
    return h.hexdigest()[:16]
