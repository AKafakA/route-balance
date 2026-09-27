"""Event-loop lag: a 1 ms heartbeat stays under 5 ms while 64 concurrent BEST-Route choose_model calls run,
and the concurrent decisions equal the banked ones. A control run with the classifier called inline (as v3
did) must show the lag, which proves the probe can see a blocked loop.

The 5 ms bar applies at the 99.9th percentile of heartbeats: isolated 5-13 ms beats (a few in ~5,000) also
appear with a single router thread, with a 1 ms interpreter switch interval and on the idle loop of a busy
host, so they are host scheduling noise, not routing work on the loop (the idle baseline is printed). Any
single beat above MAX_SPIKE_MS fails too: one forward pass on the loop costs far more than that here.
"""
import asyncio
import statistics
import time

import pytest

from routebalance.estimate.prompt import unwrap_chat_prompt
from routebalance.baselines.routers import BestRouteRouter, RouterRequest, load_best_route, router_executor
from tests.fleet_s2 import POOL

N_CONCURRENT = 64
TICK_S = 0.001
MAX_LAG_MS = 5.0
MAX_SPIKE_MS = 25.0


async def heartbeat(stop: asyncio.Event, lags: list):
    while not stop.is_set():
        t = time.perf_counter()
        await asyncio.sleep(TICK_S)
        lags.append((time.perf_counter() - t - TICK_S) * 1000.0)


async def under_heartbeat(work):
    stop, lags = asyncio.Event(), []
    beat = asyncio.create_task(heartbeat(stop, lags))
    await asyncio.sleep(0.05)
    t = time.perf_counter()
    result = await work()
    elapsed = time.perf_counter() - t
    stop.set()
    await beat
    return result, lags, elapsed


async def idle_heartbeat(seconds=1.0):
    """The same heartbeat on an idle loop: the host's own noise floor, printed next to each result."""
    _, lags, _ = await under_heartbeat(lambda: asyncio.sleep(seconds))
    return lags


def p999(lags):
    return statistics.quantiles(lags, n=1000)[998]


def summary(lags):
    q = statistics.quantiles(lags, n=100)
    return f"n={len(lags)} p50={q[49]:.2f} ms p99={q[98]:.2f} ms p99.9={p999(lags):.2f} ms max={max(lags):.2f} ms"


@pytest.fixture(scope="module")
def classifier(v3):
    return load_best_route(v3.artifacts / "best_route_4way_qwen", 512)


@pytest.fixture(scope="module")
def sample(v3):
    rows = v3.banked("br4_t00")[:N_CONCURRENT]
    return [unwrap_chat_prompt(r["prompt"]) for r in rows], [r["model"] for r in rows]


def test_off_loop_routing_keeps_the_loop_responsive(classifier, sample):
    classify, labels = classifier
    router = BestRouteRouter(classify, labels, threshold=0.0, fallback="Qwen/Qwen2.5-3B", executor=router_executor())
    prompts, banked = sample

    async def burst():
        return await asyncio.gather(*(router.choose_model(RouterRequest(p, str(k)), POOL) for k, p in enumerate(prompts)))
    print(f"idle loop: {summary(asyncio.run(idle_heartbeat()))}")
    decisions, lags, elapsed = asyncio.run(under_heartbeat(burst))
    print(f"off-loop, {N_CONCURRENT} concurrent in {elapsed:.2f} s: {summary(lags)}")
    assert [d.model for d in decisions] == banked
    assert p999(lags) < MAX_LAG_MS and max(lags) < MAX_SPIKE_MS


def test_the_probe_sees_an_inline_classifier(classifier, sample):
    classify, _ = classifier
    prompts, _ = sample

    async def inline():
        for p in prompts[:4]:
            classify(p)                   # v3's execution: the forward pass on the event loop
            await asyncio.sleep(0)
    _, lags, _ = asyncio.run(under_heartbeat(inline))
    print(f"inline control: {summary(lags)}")
    assert max(lags) > MAX_SPIKE_MS
