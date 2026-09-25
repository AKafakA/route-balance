"""Dispatchers vs the v3 dispatch modules on one seeded synthetic trace of 20,000 dispatch events.

Each event is a routed dispatch (one tier's instances, tiers drawn with BEST-Route's t=0 mix) or a
passthrough one (all 13), with running/waiting/in-flight drawn in [0, 8] so ties are frequent. v3 sees
the same numbers as a complete stats_map (v3's fail-loud default). The instance-id sequences must match.
v3's shortest_queue has no rebuild counterpart (merged into least_outstanding, D40), and least_outstanding
must match its own definition (the shortest running + waiting queue, ties to the least recently assigned).
"""
import asyncio
import random
from dataclasses import dataclass

import dataclasses

import numpy as np
import pytest

from routebalance.baselines.dispatch import LeastOutstanding, PowerOfTwo, RandomDispatch, RoundRobin
from tests.fleet_s2 import FLEET, load, of_model

N_EVENTS = 20_000
TIER_MIX = {"3B": 0.503, "7B": 0.219, "14B": 0.155, "72B": 0.123}


@dataclass
class V3Instance:
    _instance_id: str
    _model_name: str
    total_request: int = 0


@dataclass(frozen=True)
class Event:
    candidates: tuple          # rebuild Instances, fleet order
    running: tuple
    waiting: tuple
    in_flight: tuple


def make_trace(n=N_EVENTS, seed=20260924):
    rng = np.random.default_rng(seed)
    sizes, probs = list(TIER_MIX), np.array(list(TIER_MIX.values()))
    out = []
    for _ in range(n):
        cands = tuple(FLEET) if rng.random() < 0.3 else tuple(of_model(sizes[rng.choice(len(sizes), p=probs / probs.sum())]))
        draw = rng.integers(0, 9, size=(3, len(cands)))
        out.append(Event(cands, *(tuple(int(x) for x in row) for row in draw)))
    return out


TRACE = make_trace()


def rebuild_ids(dispatcher, trace=TRACE):
    ids = []
    for ev in trace:
        loads = {c.instance_id: load(r, w, f)
                 for c, r, w, f in zip(ev.candidates, ev.running, ev.waiting, ev.in_flight, strict=True)}
        ids.append(dispatcher.choose(ev.candidates, loads).instance_id)
    return ids


def v3_ids(dispatcher, v3_base, trace=TRACE):
    v3 = {i.instance_id: V3Instance(i.instance_id, i.model) for i in FLEET}

    async def drive():
        ids = []
        for ev in trace:
            cands = [v3[c.instance_id] for c in ev.candidates]
            stats = {c.instance_id: {"num_running": r, "num_waiting": w}
                     for c, r, w in zip(ev.candidates, ev.running, ev.waiting, strict=True)}
            inst = await dispatcher.choose_instance(cands, v3_base.DispatchRequest(prompt=""), stats_map=stats)
            inst.total_request += 1          # as the v3 Instance does when it is queried
            ids.append(inst._instance_id)
        return ids
    return asyncio.run(drive())


@pytest.fixture(scope="module")
def v3_dispatch(v3):
    return v3.v3_package("dispatch")


def test_trace_has_ties_and_both_candidate_shapes(v3):
    passthrough = sum(len(ev.candidates) == len(FLEET) for ev in TRACE)
    queues = [[r + w for r, w in zip(ev.running, ev.waiting, strict=True)] for ev in TRACE]
    tied = sum(q.count(min(q)) > 1 for q in queues)          # the first-candidate rule decides these
    assert 0.25 * N_EVENTS < passthrough < 0.35 * N_EVENTS and tied > 0.1 * N_EVENTS


@pytest.mark.parametrize("name,make_v3,make_rebuild", [
    ("round_robin", lambda m: m("round_robin").RoundRobinDispatch(), RoundRobin),
    ("po2 v3 default seed", lambda m: m("po2_queue").Po2QueueDispatch(), lambda: PowerOfTwo(1)),
    ("po2 seed 0", lambda m: m("po2_queue").Po2QueueDispatch(seed=0), lambda: PowerOfTwo(0)),
    ("po2 seed 7", lambda m: m("po2_queue").Po2QueueDispatch(seed=7), lambda: PowerOfTwo(7)),
    ("random seed 123", lambda m: m("random_dispatch").RandomDispatch(seed=123), lambda: RandomDispatch(123)),
])
def test_same_instance_sequence_as_v3(v3_dispatch, name, make_v3, make_rebuild):
    expect, got = v3_ids(make_v3(v3_dispatch), v3_dispatch("base")), rebuild_ids(make_rebuild())
    diff = [k for k, (a, b) in enumerate(zip(expect, got, strict=True)) if a != b]
    assert len(got) == N_EVENTS and not diff, f"{name}: {len(diff)} of {N_EVENTS} differ, first at {diff[:5]}"


class Ticks:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


def least_recent_reference(key_of):
    """Lowest key, then least recently chosen (never = 0), then fleet order, written out longhand."""
    stamp, ids = {}, []
    for n, ev in enumerate(TRACE, start=1):
        keys = key_of(ev)
        k = min(range(len(ev.candidates)), key=lambda j: (keys[j], stamp.get(ev.candidates[j].instance_id, 0), j))
        stamp[ev.candidates[k].instance_id] = n
        ids.append(ev.candidates[k].instance_id)
    return ids


def test_least_outstanding_is_its_definition(v3):
    """The shortest running + waiting queue the view reports (a baseline's view: the engines' own counts), ties
    to the least recently assigned."""
    queue = lambda ev: [r + w for r, w in zip(ev.running, ev.waiting, strict=True)]   # noqa: E731
    assert rebuild_ids(LeastOutstanding(clock=Ticks())) == least_recent_reference(queue)


def test_po2_single_candidate_consumes_no_randomness(v3_dispatch):
    one = of_model("72B")[:1]
    v3_d, ours = v3_dispatch("po2_queue").Po2QueueDispatch(seed=9), PowerOfTwo(9)
    v3_inst = [V3Instance(one[0].instance_id, one[0].model)]
    before = ours._rng.getstate()
    asyncio.run(v3_d.choose_instance(v3_inst, v3_dispatch("base").DispatchRequest(prompt=""),
                                     stats_map={one[0].instance_id: {"num_running": 0, "num_waiting": 0}}))
    ours.choose(one, {one[0].instance_id: load()})
    assert ours._rng.getstate() == before == v3_d._rng.getstate()
    assert random.Random(9).getstate() == before


def test_round_robin_warm_up_offset_matches_the_banked_passthrough_cell(v3):
    """v3 measured runs start after 6 dispatches (5 warm-up + the benchmark's test request). The banked
    pt_rr cell matches order[(i + 6) % 13] except where concurrent arrivals were reordered."""
    rr, idle = RoundRobin(), {i.instance_id: load() for i in FLEET}
    for _ in range(6):
        rr.choose(FLEET, idle)
    ours = [rr.choose(FLEET, idle).instance_id for _ in range(3534)]
    banked = [r["instance_id"] for r in v3.banked("pt")]
    assert sum(a == b for a, b in zip(ours, banked, strict=True)) == 3486


def test_least_outstanding_is_v3_shortest_queue_when_in_flight_is_the_queue(v3_dispatch):
    """D40: least_outstanding is v3's shortest queue on running + waiting (the engines' own counts, as the
    baselines read them): it must pick v3's shortest-queue instance wherever the minimum is unique; ties differ
    by design (least recent vs v3's first candidate)."""
    trace = [dataclasses.replace(ev, in_flight=tuple(r + w for r, w in zip(ev.running, ev.waiting, strict=True)))
             for ev in TRACE]
    expect = v3_ids(v3_dispatch("shortest_queue").ShortestQueueDispatch(), v3_dispatch("base"), trace)
    got = rebuild_ids(LeastOutstanding(clock=Ticks()), trace)
    unique = [k for k, ev in enumerate(trace) if ev.in_flight.count(min(ev.in_flight)) == 1]
    assert len(unique) > N_EVENTS // 2 and all(got[k] == expect[k] for k in unique)
