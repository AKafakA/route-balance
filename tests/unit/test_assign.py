from dataclasses import dataclass

import pytest

from routebalance.config import Weights
from routebalance.schedule.assign import Admission, RoundRequest, _decode_tokens, lpt_key, schedule_round
from routebalance.telemetry import InstanceStats, Load
from routebalance.types import Instance, Tier

SMALL = Tier("small_a30", "S", "a30", 1e-7, 1e-7, 4)
LARGE = Tier("large_a100", "L", "a100", 4e-7, 4e-7, 4)
FLEET = [Instance("s0", SMALL, "n1", "h1", 8000), Instance("s1", SMALL, "n2", "h2", 8000),
         Instance("l0", LARGE, "n3", "h3", 8000)]
W = Weights(w_quality=0.5, w_latency=0.3, w_cost=0.2)


@dataclass(frozen=True)
class Est:
    quality: float
    length: float
    bucket_probs: tuple = (1.0,)


class QueueLatency:
    """TPOT 10 ms (S) / 40 ms (L) per token, times (1 + running + waiting); records every call."""
    def __init__(self):
        self.calls = []

    def latencies(self, instances, states, lengths, num_prompt_tokens):
        self.calls.append([(s.num_running, s.num_waiting, s.pending_decode_tokens) for s in states])
        return [(0.01 if i.model == "S" else 0.04) * lengths[i.model] * (1 + s.num_running + s.num_waiting)
                for i, s in zip(instances, states, strict=True)]


def idle_loads():
    return {i.instance_id: Load(InstanceStats(0, 0, 0.0, 0, 4, 0), 5.0, 0, 0) for i in FLEET}


def req(rid, s_len=100.0, l_len=80.0, qs=0.5, ql=0.9, budget=0.0, s_probs=(1.0,), l_probs=(1.0,)):
    return RoundRequest(rid, 50, {"S": Est(qs, s_len, s_probs), "L": Est(ql, l_len, l_probs)}, budget)


def run_round(requests, loads=None, latency=None, **kw):
    args = dict(weights=W, normalization="max_divide", lpt="max", pending_decode="own_length",
                last_assigned={}, clock=lambda: 1.0) | kw
    return schedule_round(requests, FLEET, loads or idle_loads(), latency or QueueLatency(), **args)


def test_lpt_takes_the_longest_largest_expectation_first_and_is_stable():
    rs = [req("a", s_len=10, l_len=20), req("b", s_len=300, l_len=5), req("c", s_len=20, l_len=10),
          req("d", s_len=5, l_len=300)]
    assert lpt_key(rs[1], "max") == 300
    assert [d.request.request_id for d in run_round(rs)] == ["b", "d", "a", "c"]
    assert [d.request.request_id for d in run_round(rs, lpt="off")] == ["a", "b", "c", "d"]


def test_later_requests_see_earlier_assignments_in_the_round():
    w = Weights(w_quality=0.0, w_latency=1.0, w_cost=0.0)
    lat = QueueLatency()
    ds = run_round([req("a"), req("b"), req("c")], latency=lat, weights=w)
    assert [d.instance.instance_id for d in ds] == ["s0", "s1", "s0"]
    assert lat.calls[1][0] == (0, 1, 100)                   # s0 carries a's waiting request and decode tokens


def test_pending_decode_records_own_length_or_the_lpt_key():
    w = Weights(w_quality=1.0, w_latency=0.0, w_cost=0.0)   # quality: the large model wins
    assert run_round([req("a")], weights=w)[0].decode_tokens == 80
    assert run_round([req("a")], weights=w, pending_decode="lpt_key")[0].decode_tokens == 100


def test_fixed_ref_scores_against_idle_instances():
    loads = idle_loads()
    loads["s0"] = Load(InstanceStats(3, 1, 0.5, 0, 4, 400), 5.0, 4, 1)
    lat = QueueLatency()
    ds = run_round([req("a")], loads=loads, latency=lat, normalization="fixed_ref")
    assert lat.calls[1] == [(0, 0, 0)] * 3                  # the second call is the unloaded estimate
    s0 = next(s for s in ds[0].scored if s.candidate.instance_id == "s0")
    assert s0.latency_term == pytest.approx(1 / 5)          # best unloaded (idle S) over s0's (1 + 4) x


def test_winner_is_stamped_least_recently_used():
    last = {}
    run_round([req("a")], last_assigned=last, clock=lambda: 42.0)
    assert list(last.values()) == [42.0]


def test_a_missing_model_estimate_raises():
    bad = RoundRequest("x", 10, {"S": Est(0.5, 10.0)}, 0.0)
    with pytest.raises(ValueError, match="no estimate"):
        run_round([bad])


def test_admission_drops_candidates_unlikely_to_fit_and_keeps_all_if_none_fit():
    w = Weights(w_quality=1.0, w_latency=0.0, w_cost=0.0)
    # budget 2e-5: S affords (2e-5 - 50e-7) / 1e-7 = 150 tokens, L affords (2e-5 - 200e-7)/4e-7 -> exhausted, 1 token
    fits_s = req("a", budget=2e-5, s_probs=(0.2, 0.7, 0.1), l_probs=(0.1, 0.9))
    d = run_round([fits_s], weights=w, admission=Admission(confidence=0.5))[0]
    assert d.instance.model == "S" and d.admitted_all and {s.candidate.model for s in d.scored} == {"S"}
    fits_none = req("b", budget=2e-5, s_probs=(0.1, 0.1, 0.1, 0.7), l_probs=(0.1, 0.9))   # 150 tokens: P=0.3
    d = run_round([fits_none], weights=w, admission=Admission(confidence=0.5))[0]
    assert not d.admitted_all and d.instance.model == "L" and len(d.scored) == 3


def test_unknown_modes_raise():
    with pytest.raises(ValueError):
        run_round([req("a")], lpt="median")
    with pytest.raises(ValueError):
        run_round([req("a")], pending_decode="max")


def test_admission_never_admits_a_tier_whose_prompt_alone_exceeds_the_budget():
    w = Weights(w_quality=1.0, w_latency=0.0, w_cost=0.0)
    # budget 1.9e-5: S affords 140 tokens but fits with P = 0.3; L's prompt alone costs 2e-5 (exhausted)
    r = req("a", budget=1.9e-5, s_probs=(0.1, 0.1, 0.1, 0.7), l_probs=(0.8, 0.2))
    d = run_round([r], weights=w, admission=Admission(confidence=0.5))[0]
    assert not d.admitted_all and len(d.scored) == 3          # nothing admitted: all kept, the clamp bounds cost


def test_a_negative_budget_is_rejected():
    with pytest.raises(ValueError, match="budget_cost"):
        req("a", budget=-1e-5)


# keys (min / mean / max): a 10/15/20, b 40/42.5/45, c 5/52.5/100, d 60/60/60: each statistic orders differently
@pytest.mark.parametrize("lpt,order,key_a", [("min", ["d", "b", "a", "c"], 10.0), ("mean", ["d", "c", "b", "a"], 15.0),
                                             ("max", ["c", "d", "b", "a"], 20.0)])
def test_lpt_key_statistics_order_the_round_and_feed_lpt_key_pending(lpt, order, key_a):
    rs = [req("a", s_len=10, l_len=20), req("b", s_len=40, l_len=45), req("c", s_len=5, l_len=100),
          req("d", s_len=60, l_len=60)]
    ds = run_round(rs, lpt=lpt, pending_decode="lpt_key")
    assert [d.request.request_id for d in ds] == order
    a = next(d for d in ds if d.request.request_id == "a")
    assert a.lpt_key == key_a and a.decode_tokens == int(key_a)


def test_an_expectation_below_one_token_is_recorded_as_one_token():
    ds = run_round([req("a", s_len=0.4, l_len=50.0)], lpt="min", pending_decode="lpt_key")
    assert ds[0].lpt_key == 0.4 and ds[0].decode_tokens == 1
    assert _decode_tokens(0.0) == 1                              # a zero key is not an error either
    with pytest.raises(ValueError):
        _decode_tokens(float("nan"))


def test_lpt_off_keeps_arrival_order_and_records_the_max_as_v3():
    rs = [req("a", s_len=10, l_len=20), req("c", s_len=5, l_len=100)]
    ds = run_round(rs, lpt="off", pending_decode="lpt_key")
    assert [d.request.request_id for d in ds] == ["a", "c"] and [d.lpt_key for d in ds] == [20.0, 100.0]
    assert [d.decode_tokens for d in ds] == [20, 100]
