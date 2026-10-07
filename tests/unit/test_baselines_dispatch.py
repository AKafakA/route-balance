"""Dispatchers on handcrafted loads: v3 tie-breaking by fleet order, RNG streams, LOR semantics."""
import random

import pytest
from pydantic import ValidationError

from routebalance.baselines.dispatch import (LeastOutstanding, LowestLatency, PowerOfTwo, RandomDispatch, RoundRobin,
                                             build_dispatcher)
from routebalance.config import ConfigError, Dispatch
from tests.fleet_s2 import FLEET, load, of_model

SEVEN = of_model("7B")
IDLE = {i.instance_id: load() for i in FLEET}


def loads_for(cands, **by_index):
    """loads for `cands`; by_index maps 'i<k>' -> Load for the k-th candidate, others idle."""
    out = {c.instance_id: load() for c in cands}
    for key, value in by_index.items():
        out[cands[int(key[1:])].instance_id] = value
    return out


@pytest.mark.parametrize("d", [RoundRobin(), LeastOutstanding(), PowerOfTwo(1), RandomDispatch(1),
                               LowestLatency(10, 1)])
def test_empty_candidates_raise(d):
    with pytest.raises(ValueError):
        d.choose([], {})


# ---------------------------------------------------------------- round robin

def test_round_robin_rotates_in_fleet_order_per_model():
    rr = RoundRobin()
    three = of_model("3B")
    picks = [rr.choose(SEVEN, IDLE).instance_id for _ in range(6)]
    assert picks == [i.instance_id for i in SEVEN] + [SEVEN[0].instance_id]
    assert rr.choose(three, IDLE) is three[0]            # another model has its own rotation
    assert rr.choose(SEVEN, IDLE) is SEVEN[1]


def test_round_robin_under_passthrough_is_one_rotation_over_the_fleet():
    rr = RoundRobin()
    assert [rr.choose(FLEET, IDLE) for _ in range(len(FLEET) + 1)] == FLEET + [FLEET[0]]


def test_round_robin_ignores_load():
    rr = RoundRobin()
    assert rr.choose(SEVEN, loads_for(SEVEN, i0=load(running=50))) is SEVEN[0]


# ---------------------------------------------------------------- least outstanding

class Ticks:
    """A clock that advances one unit per reading."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


def test_idle_candidates_rotate_by_least_recent_assignment():
    d = LeastOutstanding(clock=Ticks())
    assert [d.choose(SEVEN, IDLE) for _ in range(12)] == SEVEN + SEVEN + SEVEN[:2]


def test_idle_candidates_rotate_even_on_a_frozen_clock():
    d = LeastOutstanding(clock=lambda: 5.0)       # every stamp equal: never-assigned (0.0) still goes first
    assert [d.choose(SEVEN, IDLE) for _ in range(5)] == SEVEN


def test_a_smaller_key_beats_recency():
    busy = load(running=1, in_flight=1)
    d = LeastOutstanding(clock=Ticks())
    d.choose(SEVEN, IDLE)                       # SEVEN[0] is now the most recently assigned
    loads = {c.instance_id: busy for c in SEVEN} | {SEVEN[0].instance_id: load()}
    assert d.choose(SEVEN, loads) is SEVEN[0]


def test_a_new_config_forgets_assignment_times():
    cfg = Dispatch(type="least_outstanding", seed=1, count="engine")
    d = build_dispatcher(cfg)
    assert [d.choose(SEVEN, IDLE) for _ in range(3)] == SEVEN[:3]
    assert build_dispatcher(cfg).choose(SEVEN, IDLE) is SEVEN[0]


def test_least_outstanding_reads_the_views_queue():
    """running + waiting as the view reports them: a baseline's view reads the engines' own counts, so the
    scheduler's in_flight count is not its load and is ignored (v3's shortest queue)."""
    loads = loads_for(SEVEN, i0=load(running=3), i1=load(running=1, in_flight=40), i2=load(waiting=1),
                      i3=load(running=1, waiting=1), i4=load(running=5))
    busy = {c.instance_id: load(running=9) for c in SEVEN[5:]}
    assert LeastOutstanding().choose(SEVEN, loads | busy) is SEVEN[1]   # tie between i1 and i2, neither assigned yet


def test_under_dead_reckoning_the_queue_is_the_in_flight_count():
    """On a dead-reckoned view (RouteBalance's own, and the latency sweep's) running + waiting is the in-flight
    count whatever the gauges say, so one queue rule serves every view."""
    import asyncio
    from routebalance.telemetry import TelemetryView
    gauges = {"a": (0, 0), "b": (7, 3)}
    ids = [i.instance_id for i in FLEET[:2]]

    async def fetch(inst):
        r, w = gauges["a" if inst.instance_id == ids[0] else "b"]
        lab = '{engine="0",model_name="M"}'
        return (f"vllm:num_requests_running{lab} {r}\nvllm:num_requests_waiting{lab} {w}\n"
                f"vllm:kv_cache_usage_perc{lab} 0.1\nvllm:num_preemptions_total{lab} 0\n")
    view = TelemetryView(FLEET[:2], fetch, poll_ms=50.0, dead_reckoning=True, clock=lambda: 1.0)
    rng = random.Random(3)
    for _ in range(40):
        view.record_dispatch(rng.choice(ids), 5)
        gauges["a"], gauges["b"] = (rng.randint(0, 9), rng.randint(0, 9)), (rng.randint(0, 9), rng.randint(0, 9))
        for ld in asyncio.run(view.fetch(ids)).values():
            assert ld.stats.num_running + ld.stats.num_waiting == ld.in_flight


def test_least_outstanding_needs_every_candidate_load():
    with pytest.raises(KeyError):
        LeastOutstanding().choose(SEVEN, {SEVEN[0].instance_id: load()})


# ---------------------------------------------------------------- power of two

def test_po2_draws_like_random_sample_and_keeps_the_shorter():
    po2, ref = PowerOfTwo(7), random.Random(7)
    queues = {c.instance_id: load(running=k) for k, c in enumerate(SEVEN)}
    for _ in range(50):
        pair = ref.sample(SEVEN, 2)
        expect = pair[0] if queues[pair[0].instance_id].stats.num_running <= queues[pair[1].instance_id].stats.num_running \
            else pair[1]
        assert po2.choose(SEVEN, queues) is expect


def test_po2_tie_goes_to_the_first_drawn():
    po2, ref = PowerOfTwo(3), random.Random(3)
    for _ in range(20):
        assert po2.choose(SEVEN, IDLE) is ref.sample(SEVEN, 2)[0]


def test_po2_single_candidate_consumes_no_randomness():
    po2 = PowerOfTwo(5)
    before = po2._rng.getstate()
    one = of_model("72B")[:1]
    assert po2.choose(one, IDLE) is one[0]
    assert po2._rng.getstate() == before


def test_po2_same_seed_same_sequence():
    a, b = PowerOfTwo(11), PowerOfTwo(11)
    assert [a.choose(FLEET, IDLE) for _ in range(30)] == [b.choose(FLEET, IDLE) for _ in range(30)]


# ---------------------------------------------------------------- random

def test_random_is_the_seeded_choice_stream():
    d, ref = RandomDispatch(123), random.Random(123)
    assert [d.choose(FLEET, IDLE) for _ in range(40)] == [ref.choice(FLEET) for _ in range(40)]


# ---------------------------------------------------------------- lowest latency (LiteLLM)

def observed(window=10, seed=1, samples=()):
    d = LowestLatency(window, seed)
    for iid, e2e, n in samples:
        d.observe_completion(iid, e2e, n)
    return d


def test_lowest_latency_takes_the_lowest_mean_seconds_per_token():
    d = observed(samples=[(c.instance_id, e2e, n) for c, (e2e, n) in
                          zip(SEVEN, [(2.0, 100), (3.0, 200), (1.0, 20), (0.9, 100), (5.0, 100)], strict=True)])
    assert d.choose(SEVEN, IDLE) is SEVEN[3]                    # 0.009 s/token
    assert d.choose(SEVEN, loads_for(SEVEN, i3=load(running=99, in_flight=99))) is SEVEN[3]   # load is not read


def test_lowest_latency_mean_over_a_sliding_window():
    iid = SEVEN[0].instance_id
    d = observed(window=3, samples=[(iid, x, 1) for x in (9.0, 1.0, 2.0, 3.0)])
    assert d.mean(iid) == pytest.approx(2.0)                    # 9.0 slid out
    d.observe_completion(iid, 0.5, 0)                           # no tokens: the raw seconds
    assert d.mean(iid) == pytest.approx((2.0 + 3.0 + 0.5) / 3)


def test_lowest_latency_timeouts_add_a_1000_second_sample():
    iid = SEVEN[0].instance_id
    d = observed(samples=[(iid, 1.0, 100)])
    d.observe_timeout(iid)
    assert d.mean(iid) == pytest.approx((0.01 + 1000.0) / 2)


def test_lowest_latency_unseen_instances_count_as_zero_and_are_drawn_at_random():
    seen = SEVEN[0].instance_id
    picks = set()
    for seed in range(40):
        d = observed(seed=seed, samples=[(seen, 0.001, 1)])
        pick = d.choose(SEVEN, IDLE)
        assert pick is not SEVEN[0]
        picks.add(pick.instance_id)
    assert len(picks) == 4                                        # uniform over the four unseen instances


def test_lowest_latency_stores_litellms_zero_sample_for_unseen_candidates():
    """LiteLLM keeps the 0 it writes for an unseen deployment once the group's latency map exists: A completes
    at 0.03 s/token, a dispatch runs while B is unseen, B completes at 0.05, so B averages [0, 0.05] and wins."""
    a, b = SEVEN[:2]
    d = LowestLatency(10, 1)
    assert d.choose([a, b], IDLE) in (a, b)                 # no samples anywhere yet: no zero is stored
    assert d._samples == {}
    d.observe_completion(a.instance_id, 3.0, 100)
    assert d.choose([a, b], IDLE) is b                      # B is unseen: its stored 0 sample makes it the lowest
    d.observe_completion(b.instance_id, 5.0, 100)
    assert list(d._samples[b.instance_id]) == [0.0, 0.05] and d.mean(b.instance_id) == pytest.approx(0.025)
    assert d.choose([a, b], IDLE) is b


def test_lowest_latency_tie_draws_follow_the_seed():
    ref, d = random.Random(5), LowestLatency(10, 5)
    assert [d.choose(SEVEN, IDLE) for _ in range(20)] == [ref.choice(SEVEN) for _ in range(20)]


def test_lowest_latency_unique_minimum_consumes_no_randomness():
    d = observed(samples=[(c.instance_id, 1.0 + k, 1) for k, c in enumerate(SEVEN)])
    before = d._rng.getstate()
    assert d.choose(SEVEN, IDLE) is SEVEN[0] and d._rng.getstate() == before


@pytest.mark.parametrize("e2e,n", [(1.0, -1), (-1.0, 10), (float("nan"), 10), (float("inf"), 10)])
def test_lowest_latency_rejects_meaningless_observations(e2e, n):
    with pytest.raises(ValueError):
        LowestLatency(10, 1).observe_completion("x", e2e, n)


def test_lowest_latency_window_must_be_positive():
    with pytest.raises(ConfigError):
        LowestLatency(0, 1)


def test_other_dispatchers_ignore_completions_and_timeouts():
    rr = RoundRobin()
    rr.observe_completion(SEVEN[0].instance_id, 1.0, 10)
    rr.observe_timeout(SEVEN[0].instance_id)
    assert rr.choose(SEVEN, IDLE) is SEVEN[0]


# ---------------------------------------------------------------- construction

@pytest.mark.parametrize("dtype,cls", [("round_robin", RoundRobin), ("least_outstanding", LeastOutstanding),
                                       ("po2", PowerOfTwo), ("random", RandomDispatch)])
def test_build_dispatcher_types(dtype, cls):
    count = "engine" if dtype in ("least_outstanding", "po2") else None
    assert type(build_dispatcher(Dispatch(type=dtype, seed=1, count=count))) is cls


def test_shortest_queue_is_not_a_dispatcher():
    with pytest.raises(ValidationError, match="least_outstanding"):
        Dispatch(type="shortest_queue", seed=1)                  # merged into least_outstanding


def test_build_lowest_latency_takes_the_configured_window():
    d = build_dispatcher(Dispatch(type="lowest_latency", seed=1, latency_window=10))
    assert type(d) is LowestLatency and d._window == 10


def test_each_applied_config_reseeds():
    cfg = Dispatch(type="po2", seed=4, count="engine")
    first = build_dispatcher(cfg)
    seq = [first.choose(FLEET, IDLE) for _ in range(10)]
    again = build_dispatcher(cfg)
    assert [again.choose(FLEET, IDLE) for _ in range(10)] == seq


def test_build_dispatcher_rejects_an_unvalidated_type():
    cfg = Dispatch.model_construct(type="llumnix", seed=1)   # bypasses validation, as a corrupted object would
    with pytest.raises(ConfigError):
        build_dispatcher(cfg)
