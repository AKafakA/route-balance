"""W0 references, W1 calibration and the drain, against fake vLLM engines over real HTTP: every request goes to
the engines directly, with the serving payload."""
import asyncio
import json

import aiohttp
import pytest
from aiohttp import web

from routebalance.backend import BackendUnreachable
from routebalance.types import Instance, Tier
from tests.contract.fake_vllm import FakeState, make_app
from tests.contract.test_server import free_port
from tools import calibrate_driver, direct
from tools.calibrate_driver import calibrate, records
from tools.reference_generation import divergence, generate, one_per_model

SMALL = Tier("small_a30", "S", "a30", 1e-7, 1e-7, 8)
LARGE = Tier("large_a100", "L", "a100", 4e-7, 4e-7, 8)
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


PROMPTS = [direct.Prompt(f"p{i}", f"question {i}?") for i in range(12)]
TRAIN = [direct.Prompt(f"t{i}", f"train {i}?") for i in range(200)]
W0 = {"concurrency_per_instance": 3, "divergence_prompts": 4, "divergence_concurrency_per_instance": 6}


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_references_go_once_per_prompt_and_model_to_one_instance_each(tmp_path):
    async def scenario():
        async with Engines([SMALL, SMALL, LARGE], delay_s=0.002) as e:
            p, row = plan(W0)
            manifest = {}
            texts = await generate(p, row, e.fleet, PROMPTS, tmp_path, manifest)
            return e, texts, manifest
    e, texts, manifest = asyncio.run(scenario())
    assert [i.instance_id for i in one_per_model(e.fleet)] == ["S_0", "L_2"] == manifest["instances"]
    ref = [t for t in texts if t["pass"] == "reference"]
    assert len(ref) == 24
    assert {(t["model"], t["prompt_id"]) for t in ref} == {(m, p.prompt_id) for m in "SL" for p in PROMPTS}
    assert all(t["success"] and t["text"] == "Hello world" for t in texts)
    assert len([t for t in texts if t["pass"] == "divergence"]) == 8
    assert _lines(tmp_path / "texts.jsonl") == texts                  # written as the run goes
    assert e.states[1].chat_requests == []                       # the second small instance was never used
    served = [r for r in e.states[0].chat_requests if r["request_id"].startswith("W.")]
    assert served and all(r["messages"] == [{"role": "user", "content": r["messages"][0]["content"]}] for r in served)
    assert all((r["temperature"], r["frequency_penalty"], r["max_completion_tokens"]) == (0.0, 1.2, 64) for r in served)
    assert any(".divergence." in r["request_id"] and ".f" in r["request_id"] for r in served)   # the load is held
    assert divergence(texts) == {m: {"sampled": 4, "both_ok": 4, "identical": 4, "different": 0, "different_rate": 0.0}
                                 for m in "LS"}


def test_an_engine_lost_mid_pass_keeps_every_finished_text(tmp_path):
    many = [direct.Prompt(f"p{i}", f"question {i}?") for i in range(40)]

    async def scenario():
        async with Engines([SMALL, LARGE]) as e:
            e.states[1].chat_delay_s = 0.05                  # the large engine is slow, then goes away
            p, row = plan(W0)

            async def kill_later():
                await asyncio.sleep(0.5)
                await e.runners[1].cleanup()
            killer = asyncio.create_task(kill_later())
            try:
                await generate(p, row, e.fleet, many, tmp_path, {})
            finally:
                await killer
    with pytest.raises(BackendUnreachable):                     # new requests to the lost engine are refused
        asyncio.run(scenario())
    kept = _lines(tmp_path / "texts.jsonl")
    assert {t["prompt_id"] for t in kept if t["model"] == "S"} == {p.prompt_id for p in many}   # the healthy engine
    assert 0 < sum(t["model"] == "L" for t in kept) < len(many)          # and what the lost one finished


def test_the_divergence_load_must_differ_from_the_reference_load(tmp_path):
    p, row = plan({**W0, "divergence_concurrency_per_instance": 3})
    with pytest.raises(direct.ConfigError, match="another load"):
        asyncio.run(generate(p, row, [], PROMPTS, tmp_path, {}))


def test_calibration_measures_every_instance_at_every_level(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate_driver, "POLL_S", 0.005)

    async def scenario():
        async with Engines([SMALL, SMALL, LARGE], delay_s=0.03) as e:
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
            p, row = plan({"concurrency_per_instance": 1, "divergence_prompts": 1,
                           "divergence_concurrency_per_instance": 2})
            await generate(p, row, e.fleet, PROMPTS, tmp_path, {})
    with pytest.raises(direct.NotIdle, match="warm-up"):
        asyncio.run(scenario())
