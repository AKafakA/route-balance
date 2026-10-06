"""Contract with vLLM 0.18.1's Prometheus /metrics, over real HTTP against the fake server."""
import asyncio
import socket

import aiohttp
import pytest
from aiohttp import web

from routebalance.telemetry import InstanceStateUnavailable, TelemetryView, fetch_metrics
from routebalance.types import Instance, Tier
from tests.contract.fake_vllm import FakeState, make_app

TIER = Tier("t", "M", "a30", 1e-7, 1e-7, 256, ())


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _serve(state: FakeState):
    port = _free_port()
    runner = web.AppRunner(make_app(state))
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner, Instance("fake_0", TIER, "n0", "127.0.0.1", port)


def _view(inst, session):
    # raw gauges: these tests check the exposition format; dead reckoning is unit-tested
    return TelemetryView([inst], lambda i: fetch_metrics(session, i, 3.0), poll_ms=50.0, dead_reckoning=False)


def _read(state: FakeState, mutate=None):
    async def scenario():
        runner, inst = await _serve(state)
        try:
            async with aiohttp.ClientSession() as session:
                view = _view(inst, session)
                await view.refresh()
                first = view.loads([inst.instance_id])[inst.instance_id]
                if mutate is None:
                    return first
                mutate(state)
                await view.refresh()
                return view.loads([inst.instance_id])[inst.instance_id]
        finally:
            await runner.cleanup()
    return asyncio.run(scenario())


def test_gauges_are_read_through_the_real_exposition_format():
    s = _read(FakeState(running=7, waiting=3, kv_usage=0.42, preemptions=5)).stats
    assert (s.num_running, s.num_waiting, s.kv_cache_utilization, s.num_preemptions) == (7, 3, 0.42, 5)
    assert s.max_num_seqs == 256 and s.pending_decode_tokens == 0


@pytest.mark.parametrize("state,match", [
    (FakeState(status=404), "HTTP 404"),
    (FakeState(status=500), "HTTP 500"),
    (FakeState(omit=("vllm:kv_cache_usage_perc",)), "lacks"),
    (FakeState(duplicate=("vllm:num_requests_running",)), "more than once"),
])
def test_an_instance_that_never_answered_well_is_unavailable(state, match):
    """No good snapshot at all: there is no load to read (the only case that fails)."""
    with pytest.raises(InstanceStateUnavailable, match=match):
        _read(state)


@pytest.mark.parametrize("raw", [b"\xff\xfe", b"vllm:num_requests_running{a=\"b\"} NaN\n", b"no metrics here\n"])
def test_a_malformed_body_after_a_good_snapshot_keeps_it_and_records_the_error(raw):
    """Counted, never refused: the last good snapshot stays in use."""
    def mutate(state):
        state.raw_body = raw
    load = _read(FakeState(running=1), mutate)
    assert load.stats.num_running == 1 and load.error


def test_unreachable_instance_is_unavailable():
    async def scenario():
        inst = Instance("gone", TIER, "n0", "127.0.0.1", _free_port())
        async with aiohttp.ClientSession() as session:
            view = _view(inst, session)
            await view.refresh()
            view.loads(["gone"])

    with pytest.raises(InstanceStateUnavailable, match="no usable telemetry"):
        asyncio.run(scenario())
