"""W1 calibration, the warm-up and the drain, against fake vLLM engines over real HTTP: every request goes to
the engines directly, with the serving payload."""
import asyncio
import json

import aiohttp
import pytest
from aiohttp import web

from bench.records import Sampling
from routebalance.backend import BackendClient
from routebalance.config import ConfigError
from routebalance.types import Instance, Tier
from tests.contract.fake_vllm import FakeState, make_app
from tests.contract.test_server import free_port
from tools import calibrate_driver, direct
from tools.calibrate_driver import calibrate, records

SMALL = Tier("small_a30", "S", "a30", 1e-7, 1e-7, 8, ())
LARGE = Tier("large_a100", "L", "a100", 4e-7, 4e-7, 8, ())
SAMPLING = {"temperature": 0.0, "repetition_penalty": 1.0, "frequency_penalty": 1.2, "max_tokens": 64}


def plan(row_params):
    return {"bench": {"sampling": SAMPLING, "max_connections": 64, "request_timeout_s": 10.0, "max_send_lag_ms": 50.0},
            "warmup": {"requests_per_instance": 2, "max_tokens": 4, "prompt": "ready?"}, "drain": {"timeout_s": 5.0},
            "rows": [], "fleets": {}, "datasets": {}}, {"id": "W", "params": row_params}


class Engines:
    def __init__(self, tiers, delay_s=0.0):
        self.states = [FakeState(chat_delay_s=delay_s, live_running=True) for _ in tiers]
        self.tiers, self.runners = tiers, []

    async def __aenter__(self):
        self.fleet = []
        for k, (tier, st) in enumerate(zip(self.tiers, self.states, strict=True)):
            port = free_port()
            r = web.AppRunner(make_app(st))
            await r.setup()
            await web.TCPSite(r, "127.0.0.1", port).start()
            self.runners.append(r)
            self.fleet.append(Instance(f"{tier.model}_{k}", tier, f"n{k}", "127.0.0.1", port))
        return self

    async def __aexit__(self, *exc):
        for r in self.runners:
            await r.cleanup()


TRAIN = [direct.Prompt(f"t{i}", f"train {i}?") for i in range(200)]


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_calibration_measures_every_instance_at_every_level(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate_driver, "POLL_S", 0.005)

    async def scenario():
        async with Engines([SMALL, SMALL, LARGE], delay_s=0.06) as e:   # wide enough windows under a loaded CPU
            p, row = plan({"levels": [1, 2, 3, 4, 5, 6], "requests_per_level": 3})
            manifest = {}
            return e, await calibrate(p, row, e.fleet, TRAIN, tmp_path, manifest), manifest
    e, lines, manifest = asyncio.run(scenario())
    measured = [x for x in lines if not x["filler"]]
    assert len(measured) == 6 * 3 * 3 and all(x["success"] and x["b"] == x["level"] and x["held"] for x in measured)
    assert {k: tuple(v) for k, v in manifest["slices"]["1"].items()} == {"S_0": (0, 3), "S_1": (3, 6), "L_2": (0, 3)}
    assert {k: tuple(v) for k, v in manifest["slices"]["2"].items()} == {"S_0": (6, 9), "S_1": (9, 12), "L_2": (6, 9)}
    sent = [r["messages"][0]["content"] for r in e.states[0].chat_requests if ".m" in r["request_id"]]
    assert len(sent) == len(set(sent)) == 6 * 3                  # no measured prompt twice on one engine
    per = manifest["per_level"]["6"]["instances"]
    assert set(per) == {"S_0", "S_1", "L_2"} and all(v["held"] and v["max_waiting"] == 0 for v in per.values())
    recs = records(lines)
    assert len(recs) == len(measured) and all(r["tpot_s"] > 0 and r["output_tokens"] == 2 for r in recs)
    assert _lines(tmp_path / "requests.jsonl") == lines


def test_a_fast_tier_holds_although_it_finishes_long_before_a_slow_one(tmp_path, monkeypatch):
    """Found in the 09-25 rehearsal: the 14B finished level 1 in ~260 s of the level's 672 s, and reads taken
    after its closed loop ended (running 0) made its median 0. Only an instance's own window counts."""
    monkeypatch.setattr(calibrate_driver, "POLL_S", 0.005)

    async def scenario():
        async with Engines([SMALL, LARGE], delay_s=0.02) as e:
            e.states[1].chat_delay_s = 0.2                    # the large engine is 10x slower per token
            p, row = plan({"levels": [1, 2, 3, 4, 5, 6], "requests_per_level": 3})
            manifest = {}
            return await calibrate(p, row, e.fleet, TRAIN, tmp_path, manifest), manifest
    lines, manifest = asyncio.run(scenario())
    assert all(x["held"] for x in lines if not x["filler"])
    fast, slow = (manifest["per_level"]["1"]["instances"][k]["running"] for k in ("S_0", "L_1"))
    assert 0 < len(fast) < len(slow) / 2                     # the fast instance's reads stop with its own loop


def test_a_level_with_a_preemption_or_a_short_batch_does_not_hold(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate_driver, "POLL_S", 0.005)

    async def scenario():
        async with Engines([SMALL, LARGE], delay_s=0.03) as e:
            p, row = plan({"levels": [1, 2, 3, 4, 5, 6], "requests_per_level": 3})
            original = direct.closed_loop

            async def pressured(backend, inst, *a, **kw):
                if inst.instance_id == "L_1" and a[1] == 6:     # the large engine preempts at level 6
                    e.states[1].preemptions += 1
                if inst.instance_id == "S_0" and a[1] == 5:     # the small engine runs one request short at 5
                    return await original(backend, inst, a[0], 4, *a[2:], **kw)
                e.states[0].waiting = 1 if a[1] == 4 else 0     # a waiting request alone is normal
                try:
                    return await original(backend, inst, *a, **kw)
                finally:
                    e.states[0].waiting = 0
            monkeypatch.setattr(direct, "closed_loop", pressured)
            return await calibrate(p, row, e.fleet, TRAIN, tmp_path, {})
    lines = asyncio.run(scenario())
    held = {(x["instance_id"], x["level"]): x["held"] for x in lines}
    assert not held[("L_1", 6)] and not held[("S_0", 5)]
    assert held[("S_0", 4)] and held[("S_0", 6)] and held[("L_1", 5)]
    assert not any(r["b"] == 6 and r["tier"] == "large_a100" for r in records(lines))


def test_an_unreachable_engine_keeps_the_finished_levels(tmp_path):
    async def scenario():
        async with Engines([SMALL, LARGE], delay_s=0.002) as e:
            p, row = plan({"levels": [1, 2, 3, 4, 5, 6], "requests_per_level": 2})
            original = direct.drain
            calls = []

            async def drain_then_kill(sess, fleet, **kw):
                calls.append(1)
                if len(calls) == 4:                          # after two levels, the large engine goes away
                    await e.runners[1].cleanup()
                return await original(sess, fleet, **kw)
            direct.drain = drain_then_kill
            try:
                await calibrate(p, row, e.fleet, TRAIN, tmp_path, {})
            finally:
                direct.drain = original
    with pytest.raises(aiohttp.ClientError):                   # the drain cannot read the lost engine
        asyncio.run(scenario())
    assert {x["level"] for x in _lines(tmp_path / "requests.jsonl")} == {1, 2}


def test_levels_above_the_engine_batch_are_refused(tmp_path):
    async def scenario():
        async with Engines([SMALL]) as e:
            p, row = plan({"levels": [1, 2, 4, 8, 16, 32], "requests_per_level": 2})
            await calibrate(p, row, e.fleet, TRAIN, tmp_path, {})
    with pytest.raises(direct.ConfigError, match="max_num_seqs"):
        asyncio.run(scenario())


def test_drain_waits_for_idle_engines_and_fails_loudly_at_its_bound():
    async def scenario():
        async with Engines([SMALL, LARGE]) as e:
            e.states[1].running, e.states[1].waiting = 2, 1
            async with direct.session(4) as sess:
                async def idle_later():
                    await asyncio.sleep(0.8)
                    e.states[1].running = e.states[1].waiting = 0
                task = asyncio.create_task(idle_later())
                waited = await direct.drain(sess, e.fleet, timeout_s=5.0)
                await task
                e.states[0].waiting = 1
                with pytest.raises(direct.NotIdle, match="S_0"):
                    await direct.drain(sess, e.fleet, timeout_s=0.6)
                return waited
    assert asyncio.run(scenario()) >= 0.5


def test_a_failed_warm_up_stops_the_run(tmp_path):
    async def scenario():
        async with Engines([SMALL]) as e:
            e.states[0].chat_mode = "http500"
            p, row = plan({"levels": [1, 2, 3, 4, 5, 6], "requests_per_level": 1})
            await calibrate(p, row, e.fleet, TRAIN, tmp_path, {})
    with pytest.raises(direct.NotIdle, match="warm-up"):
        asyncio.run(scenario())


# ---- the warm-up empties every engine's prefix cache and proves it

def _with_engines(tiers, setup, n=2):
    async def scenario():
        async with Engines(tiers) as e:
            setup(e)
            async with direct.session(4) as sess:
                backend = BackendClient(sess, 10.0)
                lines = await direct.warm_up(sess, backend, e.fleet, requests_per_instance=n, max_tokens=4,
                                             prompt="ready?", sampling=Sampling.model_validate(SAMPLING), run_id="r1")
            return e, lines
    return asyncio.run(scenario())


def test_the_warm_up_empties_every_engine_prefix_cache_and_proves_it():
    def setup(e):
        for st in e.states:
            st.cached.update({"ready?", "a prompt of the previous run"})
    e, lines = _with_engines([SMALL, LARGE], setup, n=3)
    assert [st.resets for st in e.states] == [1, 1]
    assert all("a prompt of the previous run" not in st.cached for st in e.states)   # the whole cache went
    for inst in e.fleet:
        mine = [(x["cache_step"], x["cache_queried"], x["cache_hit"]) for x in lines if x["instance_id"] == inst.instance_id]
        assert mine == [("prime", 5, 4), ("after_reset", 5, 0), ("control", 5, 4), ("warmup", 5, 4)]
    assert all(x["purpose"] == "warmup" and x["success"] for x in lines)


def test_an_engine_without_the_developer_mode_stops_the_run():
    with pytest.raises(direct.CacheNotReset, match="VLLM_SERVER_DEV_MODE=1"):
        _with_engines([SMALL, LARGE], lambda e: setattr(e.states[1], "dev_mode", False))


def test_a_reset_that_empties_nothing_is_caught():
    with pytest.raises(direct.CacheNotReset, match="the cache was not emptied"):
        _with_engines([SMALL], lambda e: setattr(e.states[0], "reset_ignored", True))


def test_an_engine_that_does_not_report_its_prefix_cache_stops_the_run():
    with pytest.raises(direct.CacheNotReset, match="does not report its prefix cache"):
        _with_engines([SMALL], lambda e: setattr(e.states[0], "omit", ("vllm:prefix_cache_hits_total",)))


def test_the_warm_up_needs_the_reset_check_and_its_control():
    with pytest.raises(ConfigError, match="at least 2 requests"):
        _with_engines([SMALL], lambda e: None, n=1)


def test_prefix_cache_counters_sum_the_engine_series_and_refuse_malformed_ones():
    text = ('vllm:prefix_cache_queries_total{engine="0",model_name="M"} 10.0\n'
            'vllm:prefix_cache_queries_total{engine="1",model_name="M"} 6.0\n'
            'vllm:prefix_cache_queries_created{engine="0",model_name="M"} 1.7e9\n'
            'vllm:prefix_cache_hits_total{engine="0",model_name="M"} 3.0\n'
            'vllm:prefix_cache_hits_total{engine="1",model_name="M"} 1.0\n')
    assert direct.prefix_cache_counters("x", text) == (16, 4)
    with pytest.raises(direct.CacheNotReset, match="malformed"):
        direct.prefix_cache_counters("x", text.replace(" 1.0\n", " 99.0\n"))    # more hits than queries
