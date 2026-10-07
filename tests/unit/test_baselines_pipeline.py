"""Pipeline: router -> candidates in fleet order -> shared telemetry (fail-loud) -> dispatcher."""
import asyncio

import pytest

from routebalance.baselines.dispatch import LeastOutstanding, PowerOfTwo, RoundRobin
from routebalance.baselines.pipeline import fleet_models, select_instance, unwrap_chat_prompt
from routebalance.baselines.routers import ALL_INSTANCES, PassthroughRouter, Router, RouterDecision, RouterRequest
from routebalance.telemetry import InstanceStateUnavailable, TelemetryView
from tests.fleet_s2 import FLEET, POOL, load, of_model
from tests.unit.test_telemetry import metrics

ALL64 = {m: 64 for m in POOL}     # every fleet model's recorded decode length

QWEN = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n  What is 2+2?\n<|im_end|>\n"
        "<|im_start|>assistant\n")
GEMMA = "<bos><start_of_turn>user\nName a prime.  <end_of_turn>\n<start_of_turn>model\n"


def run(coro):
    return asyncio.run(coro)


class FixedRouter(Router):
    def __init__(self, model):
        self.model, self.seen = model, []

    async def choose_model(self, request, pool):
        self.seen.append((request, list(pool)))
        return RouterDecision(self.model, 0.5, "fixed")


class FakeView:
    def __init__(self, loads=None, error=None):
        self.asked, self.recorded, self._loads, self._error = [], [], loads, error

    def loads(self, ids):
        ids = list(ids)
        self.asked.append(ids)
        if self._error:
            raise self._error
        return {i: (self._loads[i] if self._loads else load()) for i in ids}

    async def fetch(self, ids):
        self.fetched = list(ids)
        return self.loads(ids)

    def record_dispatch(self, instance_id, predicted_decode_tokens):
        self.recorded.append((instance_id, predicted_decode_tokens))
        return ("token", instance_id)


def select(router, dispatcher, view, prompt="hi", probe=False):
    return run(select_instance(router, dispatcher, FLEET, view, RouterRequest(prompt, "r1"), decode_tokens=ALL64,
                               probe=probe))


# ---------------------------------------------------------------- the unwrap

@pytest.mark.parametrize("raw,user", [
    (QWEN, "What is 2+2?"),
    (GEMMA, "Name a prime."),
    ("  plain prompt \n", "plain prompt"),
    ("<|im_start|>user no newline, no match", "<|im_start|>user no newline, no match"),
    ("", ""),
])
def test_unwrap_chat_prompt(raw, user):
    assert unwrap_chat_prompt(raw) == user


def test_unwrap_is_a_no_op_on_a_stripped_raw_prompt():
    assert unwrap_chat_prompt("What is 2+2?") == "What is 2+2?"


def test_router_sees_the_unwrapped_prompt_and_the_fleet_pool():
    router = FixedRouter("Qwen/Qwen2.5-7B")
    select(router, RoundRobin(), FakeView(), prompt=QWEN)
    request, pool = router.seen[0]
    assert request == RouterRequest("What is 2+2?", "r1") and pool == POOL == fleet_models(FLEET)


# ---------------------------------------------------------------- candidates and loads

def test_routed_candidates_are_the_models_instances_in_fleet_order():
    view = FakeView()
    sel = select(FixedRouter("Qwen/Qwen2.5-7B"), RoundRobin(), view)
    assert view.asked == [[i.instance_id for i in of_model("7B")]]
    assert view.recorded == [(of_model("7B")[0].instance_id, 64)] and sel.token == ("token", sel.instance.instance_id)
    assert sel.instance is of_model("7B")[0] and sel.model == "Qwen/Qwen2.5-7B" and sel.reason == "fixed"


def test_passthrough_candidates_are_every_instance():
    view = FakeView()
    sel = select(PassthroughRouter(), RoundRobin(), view)
    assert view.asked == [[i.instance_id for i in FLEET]]
    assert sel.instance is FLEET[0] and sel.model == FLEET[0].model and sel.reason == "passthrough:no_routing"


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="no instance serves"):
        select(FixedRouter("Qwen/Qwen2.5-1B"), RoundRobin(), FakeView())


def test_telemetry_failure_propagates_for_every_dispatcher():
    for d in (RoundRobin(), LeastOutstanding()):
        view = FakeView(error=InstanceStateUnavailable("stale"))
        with pytest.raises(InstanceStateUnavailable):
            select(FixedRouter("Qwen/Qwen2.5-3B"), d, view)
        assert view.recorded == []


def test_dispatcher_gets_the_views_loads():
    seven = of_model("7B")
    loads = {i.instance_id: load(running=5, in_flight=5) for i in FLEET} | {
        seven[3].instance_id: load(running=1, in_flight=1)}
    sel = select(FixedRouter("Qwen/Qwen2.5-7B"), LeastOutstanding(), FakeView(loads))
    assert sel.instance is seven[3]


def test_a_probe_fetches_the_routers_candidates_only():
    """v3's per-request arm fetches the chosen model's instances, then dispatches on that fetch."""
    seven = of_model("7B")
    loads = {i.instance_id: load(running=5) for i in FLEET} | {seven[2].instance_id: load(running=1)}
    probed, cached = FakeView(loads), FakeView(loads)
    assert select(FixedRouter("Qwen/Qwen2.5-7B"), LeastOutstanding(), probed, probe=True).instance is seven[2]
    assert probed.fetched == [i.instance_id for i in seven]
    assert select(FixedRouter("Qwen/Qwen2.5-7B"), LeastOutstanding(), cached).instance is seven[2]
    assert not hasattr(cached, "fetched")                     # the cache read never fetches


def test_a_power_of_two_probe_fetches_its_drawn_pair_with_the_cache_forms_draws():
    """po2-probe fetches only the two instances it compares, drawn as the cache form draws them."""
    loads = {i.instance_id: load(running=n) for n, i in enumerate(FLEET)}
    cached, probed = FakeView(loads), FakeView(loads)
    a, b = PowerOfTwo(seed=1), PowerOfTwo(seed=1)
    pairs = set()
    for _ in range(20):
        via_cache = select(PassthroughRouter(), a, cached)
        via_probe = select(PassthroughRouter(), b, probed, probe=True)
        assert via_probe.instance is via_cache.instance
        assert len(probed.fetched) == 2 and via_probe.instance.instance_id in probed.fetched
        assert probed.asked[-1] == probed.fetched                    # dispatched on the fetch, nothing else
        pairs.add(tuple(sorted(probed.fetched)))
    assert len(pairs) > 1                                            # the pair is drawn anew for each request
    assert all(len(ids) == len(FLEET) for ids in cached.asked)       # the cache form reads every candidate


def test_overhead_is_split_and_adds_up():
    o = select(PassthroughRouter(), RoundRobin(), FakeView()).overhead
    assert min(o.router_ms, o.telemetry_ms, o.dispatch_ms) >= 0
    assert o.total_ms == pytest.approx(o.router_ms + o.telemetry_ms + o.dispatch_ms, abs=1e-6)


@pytest.mark.parametrize("bad", [{m: 0 for m in POOL}, {m: 64 for m in POOL[1:]}, {m: 64.0 for m in POOL}])
def test_a_bad_prediction_is_refused_before_the_dispatcher_moves(bad):
    rr, view = RoundRobin(), FakeView()
    with pytest.raises(ValueError, match="decode_tokens"):
        run(select_instance(PassthroughRouter(), rr, FLEET, view, RouterRequest("x", "r"), decode_tokens=bad,
                              probe=False))
    assert view.asked == [] and view.recorded == [] and select(PassthroughRouter(), rr, view).instance is FLEET[0]


def test_the_chosen_models_decode_tokens_and_load_are_recorded():
    per_model = {m: 10 + k for k, m in enumerate(POOL)}
    view = FakeView({i.instance_id: load(in_flight=3) for i in FLEET})
    sel = run(select_instance(PassthroughRouter(), RoundRobin(), FLEET, view, RouterRequest("x", "r"),
                              decode_tokens=per_model, probe=False))
    assert view.recorded == [(sel.instance.instance_id, per_model[sel.model])] and sel.load.in_flight == 3


def test_empty_fleet_raises():
    with pytest.raises(ValueError):
        run(select_instance(PassthroughRouter(), RoundRobin(), [], FakeView(), RouterRequest("x", "r"),
                            decode_tokens=ALL64, probe=False))


# ---------------------------------------------------------------- the real shared view

def real_view(gauges=None):
    """A baseline's view: the engines' own counts (dead reckoning is RouteBalance's own)."""
    gauges = gauges if gauges is not None else {}

    async def fetch(inst):
        r, w = gauges.get(inst.instance_id, (0, 0))   # default-ok: an engine not listed is idle
        return metrics(running=r, waiting=w, kv=0.0)
    view = TelemetryView(FLEET, fetch, poll_ms=50.0, dead_reckoning=False, clock=lambda: 100.0)
    run(view.refresh())
    return view


def test_each_selection_is_recorded_before_the_next_one_reads_the_view():
    """Concurrent selections on unchanged engine gauges spread by the least-recently-assigned tie-break alone,
    and every dispatch is recorded (in flight, with its decode tokens) before the next selection reads."""
    ticks = iter(range(1, 100))
    view, router, lor = real_view(), FixedRouter("Qwen/Qwen2.5-7B"), LeastOutstanding(clock=lambda: next(ticks))

    async def burst():
        return await asyncio.gather(*(select_instance(router, lor, FLEET, view, RouterRequest("hi", str(k)),
                                                      decode_tokens=ALL64, probe=False) for k in range(7)))
    sels = run(burst())
    seven = of_model("7B")
    assert [s.instance for s in sels] == seven + seven[:2]
    assert [view.loads([i.instance_id])[i.instance_id].in_flight for i in seven] == [2, 2, 1, 1, 1]
    assert [(s.token.instance_id, s.token.predicted_decode_tokens) for s in sels] == [(s.instance.instance_id, 64)
                                                                                      for s in sels]


def test_on_frozen_gauges_a_burst_goes_to_the_engines_they_show_empty():
    """the engines' own gauges, frozen between scrapes with one engine busy: least outstanding sends the whole
    burst to the engines the gauge shows empty, as a balancer reading scraped engine metrics does."""
    three = of_model("3B")
    gauges = {three[0].instance_id: (5, 0)}          # a stale gauge: engine 0 busy, 1 and 2 idle
    router = FixedRouter("Qwen/Qwen2.5-3B")

    async def burst(view, lor):
        return await asyncio.gather(*(select_instance(router, lor, FLEET, view, RouterRequest("hi", str(k)),
                                                      decode_tokens=ALL64, probe=False) for k in range(6)))
    view = real_view(gauges)
    run(burst(view, LeastOutstanding()))
    assert [view.loads([i.instance_id])[i.instance_id].in_flight for i in three] == [0, 3, 3]


def test_the_engines_own_counts_steer_least_outstanding():
    three = of_model("3B")
    gauges = {three[0].instance_id: (3, 1), three[1].instance_id: (0, 0), three[2].instance_id: (1, 0)}
    view, router, lor = real_view(gauges), FixedRouter("Qwen/Qwen2.5-3B"), LeastOutstanding()
    assert select(router, lor, view).instance is three[1]           # the engine that reports an empty queue
    assert view.loads([three[1].instance_id])[three[1].instance_id].stats.num_running == 0   # not dead-reckoned


def test_failed_selection_leaves_nothing_recorded():
    view = real_view()
    with pytest.raises(ValueError):
        select(FixedRouter("Qwen/Qwen2.5-1B"), LeastOutstanding(), view)
    assert all(ld.in_flight == 0 for ld in view.loads([i.instance_id for i in FLEET]).values())


def test_sentinel_constant():
    assert ALL_INSTANCES == "__ALL__"
