"""The arrival generator must be the v3 generator bit for bit, and step traces must be exact per segment."""
import hashlib
import math

import numpy as np
import pytest
from pydantic import ValidationError

from bench.arrivals import FROZEN_CLOCK_S, arrival_times, schedule_digest
from tests.traces import constant_trace, step_trace

_ONE_ROUNDING = 4e-16   # the rescale t * ((n/rate) / t_last) can miss n/rate by one rounding, as in v3


def _v3_delays(n, rate, burstiness, seed):
    """The v3 client's delay computation (get_request with a fixed rate), transcribed: global legacy seeding,
    one scalar draw per request, in-place running sum, rescale to n/rate."""
    np.random.seed(seed)
    delay_ts = []
    for _ in range(n):
        theta = 1.0 / (rate * burstiness)
        delay_ts.append(np.random.gamma(shape=burstiness, scale=theta))
    for i in range(1, len(delay_ts)):
        delay_ts[i] += delay_ts[i - 1]
    if delay_ts[-1] != 0:
        normalize_factor = (n / rate) / delay_ts[-1]
        delay_ts = [delay * normalize_factor for delay in delay_ts]
    return delay_ts


@pytest.mark.parametrize("n,rate,b,seed", [(3534, 30.0, 1.0, 1), (3534, 12.0, 0.25, 6), (500, 6.0, 1.0, 3),
                                           (7, 24.0, 0.25, 0)])
def test_bitwise_equal_to_the_v3_generator(n, rate, b, seed):
    assert arrival_times(constant_trace(n, rate, b), seed) == _v3_delays(n, rate, b, seed)


def test_schedule_shape():
    t = arrival_times(constant_trace(1000, 30.0, 1.0), 1)
    assert len(t) == 1000 and t[0] > 0 and math.isclose(t[-1], 1000 / 30.0, rel_tol=_ONE_ROUNDING)
    assert all(a <= b for a, b in zip(t, t[1:], strict=False))
    assert all(isinstance(x, float) for x in t)


def test_deterministic_and_seed_sensitive():
    tr = constant_trace(200, 12.0, 0.25)
    assert arrival_times(tr, 4) == arrival_times(tr, 4)
    assert arrival_times(tr, 4) != arrival_times(tr, 5)


@pytest.mark.parametrize("b,cv2", [(1.0, 1.0), (0.25, 4.0)])
def test_gap_cv2_is_one_over_b(b, cv2):
    t = np.array(arrival_times(constant_trace(40000, 30.0, b), 7))
    gaps = np.diff(np.concatenate([[0.0], t]))
    assert abs(gaps.var() / gaps.mean() ** 2 - cv2) / cv2 < 0.1


def test_step_trace_segments_are_exact():
    tr = step_trace([(360, 6.0), (1800, 30.0), (360, 6.0)], burstiness=1.0)
    t = arrival_times(tr, 2)
    assert tr.n_requests == len(t) == 2520
    ends = [60.0, 120.0, 180.0]
    for last, end in zip((t[359], t[2159], t[-1]), ends, strict=True):
        assert math.isclose(last, end, rel_tol=_ONE_ROUNDING)
    slack = 1 + _ONE_ROUNDING
    assert all(0 < x <= ends[0] * slack for x in t[:360])
    assert all(ends[0] < x <= ends[1] * slack for x in t[360:2160])
    assert all(ends[1] < x <= ends[2] * slack for x in t[2160:])
    assert np.isclose(np.mean(np.diff(t[360:2160])), 1 / 30, rtol=0.1)


def test_one_segment_step_trace_is_the_constant_trace():
    assert arrival_times(step_trace([(3534, 24.0)], 0.25), 3) == arrival_times(constant_trace(3534, 24.0, 0.25), 3)


def test_digest_uses_the_frozen_clock_intervals():
    times = [0.1234567891, 1.5, 117.8]
    expected = hashlib.sha256()
    for t in times:
        expected.update(f"{(FROZEN_CLOCK_S + t) - FROZEN_CLOCK_S:.9f}|".encode())
    assert schedule_digest(times) == expected.hexdigest()[:16]
    assert len(schedule_digest(times)) == 16


def test_digest_skips_intervals_that_round_to_zero():
    # v3's pacing loop never slept for a non-positive interval, so such a time was never digested
    assert (FROZEN_CLOCK_S + 1e-12) - FROZEN_CLOCK_S == 0.0
    assert schedule_digest([1e-12, 0.5]) == schedule_digest([0.5])


@pytest.mark.parametrize("make", [
    lambda: constant_trace(0, 30.0, 1.0),
    lambda: constant_trace(10, 0.0, 1.0),
    lambda: constant_trace(10, -1.0, 1.0),
    lambda: constant_trace(10, float("inf"), 1.0),
    lambda: constant_trace(10, 30.0, 0.0),
    lambda: constant_trace(10, 30.0, float("nan")),
    lambda: step_trace([], 1.0),
])
def test_invalid_traces_are_rejected(make):
    with pytest.raises(ValidationError):
        make()


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.0, "1"])
def test_invalid_seeds_are_rejected(seed):
    with pytest.raises(ValueError):
        arrival_times(constant_trace(10, 30.0, 1.0), seed)
