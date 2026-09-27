"""Trace builders for test fixtures (bench/arrivals.py Trace)."""
from __future__ import annotations

from typing import Sequence

from bench.arrivals import Segment, Trace


def constant_trace(n_requests: int, rate: float, burstiness: float) -> Trace:
    return Trace(burstiness=burstiness, segments=[Segment(n_requests=n_requests, rate=rate)])


def step_trace(steps: Sequence[tuple[int, float]], burstiness: float) -> Trace:
    """Piecewise-constant load: steps = [(n_requests, rate), ...], e.g. [(360, 6.0), (1800, 30.0), (360, 6.0)]
    is 60 s at 6 req/s, 60 s at 30 req/s, 60 s at 6 req/s."""
    return Trace(burstiness=burstiness, segments=[Segment(n_requests=n, rate=r) for n, r in steps])
