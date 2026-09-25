"""Arrival schedules for the open-loop benchmark client.

The v3 generator, reproduced exactly (`get_request` in the v3 benchmark client):
  1. seed the legacy MT19937 stream with the run seed;
  2. draw one gap per request, in request order: gap_i ~ Gamma(shape=b, scale=1/(rate*b));
  3. take the running sum;
  4. rescale every time by (n/rate) / (last time), so the last arrival lands at n/rate (to within one
     float rounding of the product, as in v3).
b = 1 is a Poisson process; the squared coefficient of variation of the gaps is 1/b (b = 0.25 gives CV^2 = 4).
The first arrival comes after t = 0, never at it.

v3 seeded the global stream (np.random.seed). A private RandomState(seed) is the same stream, and no other
library call can draw from it between seeding and use (in v3 any such call would have shifted every schedule).

Time-varying load is a sequence of constant-rate segments, a step trace such as 6 -> 30 -> 6 req/s.
Segment k holds n_k requests at rate r_k and spans (T_k, T_k + n_k/r_k], where T_k is the end of the
previous segment. Its gaps continue the same stream and are rescaled so that its last arrival is at its end
(to within one rounding). Every segment therefore carries exactly its request count and ends on time, and a
one-segment trace is the v3 generator.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
from pydantic import Field

from bench.schema import Strict

# v3 digested the sleep intervals its pacing loop would request with the clock frozen at this value
# (verify_pairing.py): interval = (clock + t) - clock. That rounds t to the clock's float spacing (~1.2e-10 s),
# and a non-positive interval never reached the sleep, so it was not digested. Digests of raw times never
# match the v3 records.
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


def constant_trace(n_requests: int, rate: float, burstiness: float) -> Trace:
    return Trace(burstiness=burstiness, segments=[Segment(n_requests=n_requests, rate=rate)])


def step_trace(steps: Sequence[tuple[int, float]], burstiness: float) -> Trace:
    """Piecewise-constant load: steps = [(n_requests, rate), ...], e.g. [(360, 6.0), (1800, 30.0), (360, 6.0)]
    is 60 s at 6 req/s, 60 s at 30 req/s, 60 s at 6 req/s."""
    return Trace(burstiness=burstiness, segments=[Segment(n_requests=n, rate=r) for n, r in steps])


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
    for _ in range(n):                    # one draw per request, as v3 did; the order is the stream
        running += stream.gamma(shape=burstiness, scale=scale)
        cumulative.append(running)
    if not cumulative[-1] > 0:
        raise ValueError("all gamma draws were zero; the segment cannot be rescaled")
    factor = (n / rate) / cumulative[-1]
    return [float(t * factor) for t in cumulative]


def schedule_digest(times: Sequence[float]) -> str:
    """The v3 arrival-schedule digest (16 hex digits), comparable with the banked ARRIVAL_DIGESTS records."""
    h = hashlib.sha256()
    for t in times:
        interval = (FROZEN_CLOCK_S + t) - FROZEN_CLOCK_S
        if interval > 0:
            h.update(f"{interval:.9f}|".encode())
    return h.hexdigest()[:16]
