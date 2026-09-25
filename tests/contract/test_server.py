"""The scheduler server over real HTTP: bench-client contract, both modes, config swaps, failures.

Engines are fake vLLM servers (fake_vllm.py); model components are small stand-ins injected through the
Service factory (the real ones load artifacts; their equivalence is tested elsewhere).
"""
import asyncio
import copy
import json
import socket
from dataclasses import dataclass

import aiohttp
import pytest
from aiohttp import web

from bench.records import Reply
from routebalance.baselines.dispatch import build_dispatcher
from routebalance.baselines.routers import PassthroughRouter
from routebalance.config import parse_scheduler
from routebalance.estimate.latency import ConstantTPOT, LatencyModel
from routebalance.server import Assignment, Completion, Components, Service, build_app, idle_state
from routebalance.types import Instance, Tier
from tests.contract.fake_vllm import FakeState, make_app

SMALL = Tier("small_a30", "S", "a30", 1e-7, 1e-7, 8)
LARGE = Tier("large_a100", "L", "a100", 4e-7, 4e-7, 8)

RB = {
    "mode": "routebalance", "weights": {"w_quality": 0.5, "w_latency": 0.3, "w_cost": 0.2},
    "rounds": {"adaptive": False, "max_batch": 4, "timeout_ms": 20.0}, "lpt": "max", "pending_decode": "own_length",
    "dead_reckoning": True, "telemetry_read": "per_round",
    "latency": {"source": "constant", "envelope": "own_length", "normalization": "max_divide",
                "table": "calib/t.json", "table_md5": "0" * 32},
    "estimator": {"index_dir": "knn/x", "index_md5": "0" * 32, "encoder_revision": "0" * 40, "k": 50,
                  "score_type": "deepeval", "prior": "knn"},
    "budget": {"admission_filter": False, "confidence": 0.5},
    "telemetry": {"poll_ms": 20.0, "max_age_ms": 2000.0, "timeout_s": 2.0},
    "pipeline": None,
}


def pipeline_cfg(dispatch="round_robin"):
    d = copy.deepcopy(RB)
    d.update(mode="pipeline", weights=None, pending_decode=None, telemetry_read="poll", dead_reckoning=False,
             pipeline={"router": {"type": "passthrough", "kwargs": {}}, "dispatch": {"type": dispatch, "seed": 1}})
    return d


@dataclass(frozen=True)
class Est:
    quality: float
    length: float
    bucket_probs: tuple = (1.0,)
    p_under_budget: float = 1.0


class StubEstimator:
    def estimate_batch(self, prompts, budget_tokens):
        return [{"S": Est(0.4, 20.0), "L": Est(0.9, 30.0)} for _ in prompts]


def factory(cfg, fleet, *, artifact_root, avengers_repo, router_lane):
    c = Components()
    if cfg.mode == "routebalance" or cfg.pending_decode == "own_length":
        c.estimator = StubEstimator()
    if cfg.mode == "routebalance":
        c.latency = LatencyModel(ConstantTPOT({"small_a30": 10.0, "large_a100": 40.0}), "own_length")
    if cfg.pipeline is not None:
        c.router, c.dispatcher = PassthroughRouter(), build_dispatcher(cfg.pipeline.dispatch)
    return c


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Harness:
    def __init__(self, tmp_path, states, timeout_s=5.0, max_connections=64):
        self.tmp_path, self.states, self.runners, self.timeout_s = tmp_path, states, [], timeout_s
        self.max_connections = max_connections

    async def __aenter__(self):
        self.fleet = []
        for k, (tier, st) in enumerate(self.states):
            port = free_port()
            r = web.AppRunner(make_app(st))
            await r.setup()
            await web.TCPSite(r, "127.0.0.1", port).start()
            self.runners.append(r)
            self.fleet.append(Instance(f"{tier.model}_{k}", tier, f"n{k}", "127.0.0.1", port))
        self.service = Service(self.fleet, artifact_root=self.tmp_path, avengers_repo=None,
                               request_timeout_s=self.timeout_s, max_connections=self.max_connections, torch_threads=1,
                               log_dir=self.tmp_path / "logs", factory=factory)
        return self

    async def start(self, cfg: dict):
        await self.service.start(parse_scheduler(cfg))
        self.port = free_port()
        self.app = web.AppRunner(build_app(self.service))
        await self.app.setup()
        await web.TCPSite(self.app, "127.0.0.1", self.port).start()
        self.http = aiohttp.ClientSession()

    async def post(self, path, body):
        async with self.http.post(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body)) as r:
            return r.status, await r.json()

    async def get(self, path):
        async with self.http.get(f"http://127.0.0.1:{self.port}{path}") as r:
            return r.status, await r.json()

    async def __aexit__(self, *exc):
        await self.http.close()
        await self.app.cleanup()
        await self.service.stop()
        for r in self.runners:
            await r.cleanup()


def body(i, **kw):
    return {"request_id": f"cell.run.{i}", "prompt": "What is 2+2?", "prompt_tokens": 5, "max_tokens": 64,
            "temperature": 0.0, "repetition_penalty": 1.0, "frequency_penalty": 1.2} | kw


def fleet_states():
    return [(SMALL, FakeState()), (SMALL, FakeState()), (LARGE, FakeState())]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_pipeline_replies_follow_the_bench_contract_and_logs_are_written(tmp_path):
    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            await h.start(pipeline_cfg())
            out = [await h.post("/v1/completions", body(i)) for i in range(3)]
            status, ident = await h.get("/v1/config")
            await h.service.decisions.flush()
            return out, ident, h.fleet, h.service.view.loads([i.instance_id for i in h.fleet])
    out, ident, fleet, loads = asyncio.run(scenario())
    assert all(s == 200 and r["success"] for s, r in out)
    replies = [Reply.model_validate({k: v for k, v in r.items() if k != "success"}) for _, r in out]
    assert [r.instance_id for r in replies] == [i.instance_id for i in fleet]          # round robin, fleet order
    assert all(r.generated_text == "Hello world" and r.prompt_tokens == 5 and r.output_tokens == 2 for r in replies)
    assert ident["fleet_digest"] and len(ident["config_digest"]) > 8
    assert all(load.in_flight == 0 for load in loads.values())                         # every dispatch closed
    decisions = read_jsonl(tmp_path / "logs/decisions.jsonl")
    assert len(decisions) == 3 and all(d["decode_tokens"] == 64 and d["success"] for d in decisions)   # max_tokens
    assert all(d["config_digest"] == ident["config_digest"] for d in decisions)
    assert not list((tmp_path / "logs").glob("latency*"))          # only the D38 sweep writes training rows


def test_routebalance_serves_concurrent_requests_in_rounds(tmp_path):
    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            await h.start(RB)
            out = await asyncio.gather(*(h.post("/v1/completions", body(i)) for i in range(10)))
            stats = h.service.stats()
            await h.service.decisions.flush()
            return out, stats, h.service.view.loads([i.instance_id for i in h.fleet])
    out, stats, loads = asyncio.run(scenario())
    assert all(s == 200 for s, _ in out) and stats["rounds"] >= 3        # max_batch 4: at least three rounds
    assert all(load.in_flight == 0 and load.stats.pending_decode_tokens == 0 for load in loads.values())
    decisions = read_jsonl(tmp_path / "logs/decisions.jsonl")
    assert all(len(d["candidates"]) == 3 and d["decode_tokens"] in (20, 30) for d in decisions)   # own_length
    assert all(d["lpt_key"] == 30.0 and d["feature_tokens"] == 30 for d in decisions)          # lpt max; heads
    assert stats["booster_calls"] == 0 and stats["loop_lag"]["samples"] > 0


def test_a_config_swap_is_refused_while_serving_and_applied_when_idle(tmp_path):
    slow = [(SMALL, FakeState(chat_delay_s=0.3)), (SMALL, FakeState(chat_delay_s=0.3)), (LARGE, FakeState())]

    async def scenario():
        async with Harness(tmp_path, slow) as h:
            await h.start(pipeline_cfg())
            _, before = await h.get("/v1/config")
            pending = asyncio.create_task(h.post("/v1/completions", body(0)))
            await asyncio.sleep(0.1)
            refused = await h.post("/v1/config", pipeline_cfg("least_outstanding"))
            served = await pending
            applied = await h.post("/v1/config", pipeline_cfg("least_outstanding"))
            return before, refused, served, applied
    before, refused, served, applied = asyncio.run(scenario())
    assert refused[0] == 409 and served[0] == 200
    assert applied[0] == 200 and applied[1]["config_digest"] != before["config_digest"]


def test_an_invalid_config_is_rejected_and_the_old_one_stays_live(tmp_path):
    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            await h.start(pipeline_cfg())
            _, before = await h.get("/v1/config")
            bad = pipeline_cfg()
            bad["telemetry"]["poll_ms"] = -1
            rejected = await h.post("/v1/config", bad)
            _, after = await h.get("/v1/config")
            return before, rejected, after
    before, rejected, after = asyncio.run(scenario())
    assert rejected[0] == 400 and after["config_digest"] == before["config_digest"]


def test_an_engine_failure_is_a_failed_request_and_closes_the_dispatch(tmp_path):
    states = [(SMALL, FakeState(chat_mode="engine_error")), (SMALL, FakeState()), (LARGE, FakeState())]

    async def scenario():
        async with Harness(tmp_path, states) as h:
            await h.start(pipeline_cfg())
            first = await h.post("/v1/completions", body(0))
            return first, h.service.view.loads([h.fleet[0].instance_id])
    (status, reply), loads = asyncio.run(scenario())
    assert status == 502 and reply["success"] is False and "engine died" in reply["error"]
    assert next(iter(loads.values())).in_flight == 0


def test_a_budget_clamps_the_tokens_sent_to_the_engine(tmp_path):
    states = fleet_states()

    async def scenario():
        async with Harness(tmp_path, states) as h:
            await h.start(pipeline_cfg())
            return await h.post("/v1/completions", body(0, budget_cost=5e-7 + 10 * 1e-7))  # prompt 5e-7 + 10 tokens
    status, _ = asyncio.run(scenario())
    assert status == 200 and states[0][1].chat_requests[-1]["max_completion_tokens"] == 10


@pytest.mark.parametrize("change", [{"extra": 1}, {"prompt_tokens": 0}, {"max_tokens": "64"}, {"budget_cost": 0.0}])
def test_malformed_requests_are_rejected(tmp_path, change):
    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            await h.start(pipeline_cfg())
            return await h.post("/v1/completions", body(0, **change))
    status, reply = asyncio.run(scenario())
    assert status == 400 and reply["success"] is False


def test_the_latency_sweep_logs_the_served_feature_vector(tmp_path):
    sweep = pipeline_cfg("random")
    sweep["pending_decode"] = "own_length"
    sweep["dead_reckoning"] = True          # its features are RouteBalance's view, as the heads are served

    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            await h.start(sweep)
            out = [await h.post("/v1/completions", body(i)) for i in range(6)]
            await h.service.latency_log.flush()
            await h.service.decisions.flush()
            return out, h.service.digest
    out, digest = asyncio.run(scenario())
    assert all(s == 200 for s, _ in out)
    rows = read_jsonl(tmp_path / f"logs/latency.{digest[:16]}.jsonl")
    decisions = read_jsonl(tmp_path / "logs/decisions.jsonl")
    assert len(rows) == 6 and all(set(r) == {"tier", "features", "tpot_s", "output_tokens", "success"} for r in rows)
    # the heads are fed the largest expectation over the models (S 20, L 30), whichever tier served
    assert all(r["features"]["num_predicted_output_tokens"] == 30.0 for r in rows)
    assert {d["decode_tokens"] for d in decisions} <= {20, 30}                        # pending decode: own length
    assert all(r["features"]["num_prompt_tokens"] == 5.0 and r["tpot_s"] is not None for r in rows)


def test_an_unreachable_engine_is_a_logged_failed_request(tmp_path):
    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            await h.start(pipeline_cfg())
            await h.runners[0].cleanup()               # the engine exits after its last good scrape
            first = await h.post("/v1/completions", body(0))
            await h.service.decisions.flush()
            return first, h.service.view.loads([h.fleet[0].instance_id])
    (status, reply), loads = asyncio.run(scenario())
    assert status == 502 and reply["success"] is False and next(iter(loads.values())).in_flight == 0
    (line,) = read_jsonl(tmp_path / "logs/decisions.jsonl")
    assert line["success"] is False and "Unreachable" in line["error"] and line["instance_id"].endswith("_0")


def test_a_timeout_teaches_lowest_latency_its_1000_s_sample(tmp_path):
    states = [(t, FakeState(chat_delay_s=1.0)) for t in (SMALL, SMALL, LARGE)]     # every engine too slow
    cfg = pipeline_cfg("lowest_latency")
    cfg["pipeline"]["dispatch"]["latency_window"] = 10

    async def scenario():
        async with Harness(tmp_path, states, timeout_s=0.3) as h:
            await h.start(cfg)
            first = await h.post("/v1/completions", body(0))
            await h.service.decisions.flush()
            return first, h.service._comps.dispatcher
    (status, reply), dispatcher = asyncio.run(scenario())
    (line,) = read_jsonl(tmp_path / "logs/decisions.jsonl")
    assert status == 502 and "Timeout" in reply["error"] and line["timed_out"] is True
    assert dispatcher.mean(line["instance_id"]) == 1000.0


def test_a_failed_round_fails_its_requests_loudly_and_logs_them(tmp_path):
    class Broken(StubEstimator):
        def estimate_batch(self, prompts, budget_tokens):
            raise RuntimeError("index not loaded")

    def broken(cfg, fleet, **kw):
        c = factory(cfg, fleet, **kw)
        c.estimator = Broken()
        return c

    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            h.service._factory = broken
            await h.start(RB)
            out = await asyncio.gather(*(h.post("/v1/completions", body(i)) for i in range(3)))
            await h.service.decisions.flush()
            return out
    out = asyncio.run(scenario())
    assert all(s == 500 and "index not loaded" in r["error"] for s, r in out)
    assert len(read_jsonl(tmp_path / "logs/decisions.jsonl")) == 3


def test_duplicate_request_ids_in_one_round_do_not_hang(tmp_path):
    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            await h.start(RB)
            return await asyncio.wait_for(asyncio.gather(*(h.post("/v1/completions", body(0)) for _ in range(3))), 5.0)
    assert all(s == 200 for s, _ in asyncio.run(scenario()))


def test_a_handler_cancelled_after_its_round_closes_the_dispatch(tmp_path):
    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            await h.start(RB)
            svc = h.service
            fut = asyncio.get_running_loop().create_future()
            svc._rb.submit = lambda c: fut                          # the round's future, resolved by hand below
            task = asyncio.create_task(svc.complete(Completion.model_validate(body(0))))
            await asyncio.sleep(0)                                  # the handler now waits on the future
            token = svc.view.record_dispatch(h.fleet[0].instance_id, 10)
            fut.set_result(Assignment(h.fleet[0], token, idle_state(SMALL), {"decode_tokens": 10,
                                                                              "feature_tokens": 10}, {}))
            task.cancel()                                           # before the handler resumes
            with pytest.raises(asyncio.CancelledError):
                await task
            return svc.view.loads([h.fleet[0].instance_id]), svc.busy
    loads, busy = asyncio.run(scenario())
    assert next(iter(loads.values())).in_flight == 0 and not busy


def test_a_non_finite_estimate_fails_the_round_before_any_dispatch(tmp_path):
    class NaNLength(StubEstimator):
        def estimate_batch(self, prompts, budget_tokens):
            return [{"S": Est(0.4, 20.0), "L": Est(0.9, float("inf"))} for _ in prompts]

    def nan_factory(cfg, fleet, **kw):
        c = factory(cfg, fleet, **kw)
        c.estimator = NaNLength()
        return c

    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            h.service._factory = nan_factory
            await h.start(RB)
            out = await asyncio.wait_for(asyncio.gather(*(h.post("/v1/completions", body(i)) for i in range(3))), 5.0)
            return out, h.service.view.loads([i.instance_id for i in h.fleet]), h.service.busy
    out, loads, busy = asyncio.run(scenario())
    assert all(s == 500 for s, _ in out) and not busy
    assert all(load.in_flight == 0 for load in loads.values())


def test_routebalance_fetches_load_per_round_and_pipelines_poll(tmp_path):
    """D53: RouteBalance scrapes every engine once per round, with no background pollers; a pipeline system reads the
    pollers' cache."""
    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            await h.start(RB)
            before = [st.scrapes for _, st in h.states]
            rb_view_state = h.service.view._state
            out = await asyncio.gather(*(h.post("/v1/completions", body(i)) for i in range(8)))
            await asyncio.sleep(0.3)                           # no pollers: nothing scrapes between rounds
            after = [st.scrapes for _, st in h.states]
            rounds, rb_stats = h.service._rb.rounds, h.service.stats()
            await h.post("/v1/config", pipeline_cfg())
            pipe_state, pipe_stats = h.service.view._state, h.service.stats()
            return out, before, after, rounds, rb_view_state, rb_stats, pipe_state, pipe_stats
    out, before, after, rounds, rb_state, rb_stats, pipe_state, pipe_stats = asyncio.run(scenario())
    assert all(s == 200 for s, _ in out) and rb_state != "running"
    assert all(a - b == rounds for a, b in zip(after, before, strict=True))    # exactly one scrape per engine per round
    assert rb_stats["telemetry_read"] == "per_round" and rb_stats["telemetry_age_ms"] is None
    assert pipe_state == "running" and pipe_stats["telemetry_read"] == "poll" and len(pipe_stats["telemetry_age_ms"]) == 3


def test_a_switch_from_the_cache_to_per_round_stops_the_pollers(tmp_path):
    """One scheduler serves poll cells, then per-round cells: the old pollers must not keep scraping."""
    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            await h.start(pipeline_cfg())
            polling = (await h.get("/v1/manifest"))[1]["loaded"]["telemetry_pollers"]
            await h.post("/v1/config", RB)
            before = [st.scrapes for _, st in h.states]
            out = await asyncio.gather(*(h.post("/v1/completions", body(i)) for i in range(8)))
            await asyncio.sleep(0.3)                           # a leftover poller would scrape every 20 ms here
            after = [st.scrapes for _, st in h.states]
            manifest = (await h.get("/v1/manifest"))[1]
            return polling, out, before, after, h.service._rb.rounds, manifest
    polling, out, before, after, rounds, manifest = asyncio.run(scenario())
    assert polling is True and all(s == 200 for s, _ in out)
    assert all(a - b == rounds for a, b in zip(after, before, strict=True))
    assert manifest["loaded"]["telemetry_pollers"] is False and manifest["system"]["telemetry_read"] == "per_round"


def test_a_failed_scrape_fails_the_round_even_with_a_young_snapshot(tmp_path):
    states = fleet_states()

    async def scenario():
        async with Harness(tmp_path, states) as h:
            await h.start(RB)                                   # every engine answered at apply
            states[2][1].metrics_drop = True                    # now one engine's /metrics drops the connection
            out = await asyncio.gather(*(h.post("/v1/completions", body(i)) for i in range(3)))
            young = h.service.view.loads([h.fleet[2].instance_id])   # its snapshot is still young and usable ...
            loads = h.service.view.loads([i.instance_id for i in h.fleet])
            return out, young, loads, h.service.busy
    out, young, loads, busy = asyncio.run(scenario())
    assert young and all(s == 500 and "load fetch failed" in r["error"] for s, r in out)   # ... but a round never uses it
    assert all(load.in_flight == 0 for load in loads.values()) and not busy                # no dispatch was opened


def test_an_rb_config_is_refused_while_an_engine_cannot_be_fetched(tmp_path):
    """apply() fetches every engine once before the swap: a failure leaves the old configuration live."""
    states = fleet_states()

    async def scenario():
        async with Harness(tmp_path, states) as h:
            await h.start(pipeline_cfg())
            old = h.service.digest
            states[1][1].status = 500
            status, reply = await h.post("/v1/config", RB)
            return status, reply, old, h.service.digest, h.service.cfg.mode
    status, reply, old, now, mode = asyncio.run(scenario())
    assert status != 200 and "load fetch failed" in json.dumps(reply) and now == old and mode == "pipeline"


def test_routebalance_on_the_pollers_cache_never_fetches_per_round(tmp_path):
    """D54's control: the same round loop, its loads read from the background pollers' cache."""
    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            await h.start(RB | {"telemetry_read": "poll"})
            out = await asyncio.gather(*(h.post("/v1/completions", body(i)) for i in range(8)))
            return out, h.service.view._state, h.service.stats(), h.service._rb.rounds
    out, state, stats, rounds = asyncio.run(scenario())
    assert all(s == 200 for s, _ in out) and rounds >= 1 and state == "running"
    assert stats["telemetry_read"] == "poll" and len(stats["telemetry_age_ms"]) == 3


def test_the_probe_arm_fetches_the_candidates_per_request_and_reads_engine_counts(tmp_path):
    """D55, v3's per-request baseline: every request scrapes its candidates once, with no pollers, and least
    outstanding reads the engines' own queues (dead reckoning off), so it follows the gauges, not the in-flight count."""
    states = fleet_states()
    probe = pipeline_cfg("least_outstanding") | {"telemetry_read": "per_request", "dead_reckoning": False}

    async def scenario():
        async with Harness(tmp_path, states) as h:
            await h.start(probe)
            for k, (_, st) in enumerate(states):
                st.running = [5, 0, 7][k]                       # the engines report their own queues
            before = [st.scrapes for _, st in states]
            out = [await h.post("/v1/completions", body(i)) for i in range(4)]   # one at a time: no overlap
            after = [st.scrapes for _, st in states]
            return out, before, after, h.service.view._state, h.service.stats(), h.fleet
    out, before, after, state, stats, fleet = asyncio.run(scenario())
    assert all(s == 200 for s, _ in out) and state != "running"
    assert [a - b for a, b in zip(after, before, strict=True)] == [4, 4, 4]    # passthrough: every instance per request
    assert {r["instance_id"] for _, r in out} == {fleet[1].instance_id}        # the engine that reports an empty queue
    assert stats["telemetry_read"] == "per_request" and stats["telemetry_age_ms"] is None


def test_a_request_that_waits_for_an_engine_connection_is_counted(tmp_path):
    """The bound is set far above any in-flight count; if it is ever reached, stats show it and the arm gate
    invalidates the cell (owner 09-25)."""
    states = [(SMALL, FakeState(chat_delay_s=0.05)), (SMALL, FakeState(chat_delay_s=0.05)),
              (LARGE, FakeState(chat_delay_s=0.05))]

    async def scenario():
        async with Harness(tmp_path, states, max_connections=1) as h:
            await h.start(pipeline_cfg())
            before = h.service.stats()["connection_waits"]
            out = await asyncio.gather(*(h.post("/v1/completions", body(i)) for i in range(3)))
            return out, before, h.service.stats()["connection_waits"]
    out, before, after = asyncio.run(scenario())
    assert all(s == 200 for s, _ in out) and before["engine"] == 0
    assert after["engine"] - before["engine"] >= 1       # the other two streams waited for the one connection
    assert after["telemetry"] >= 1                       # three pollers share the telemetry pool's one connection


def test_a_pipeline_router_is_built_on_the_one_at_a_time_lane(tmp_path):
    """The routers' model calls run on the lane (one request at a time), never on RouteBalance's shared pool."""
    seen = {}

    def recording(cfg, fleet, *, artifact_root, avengers_repo, router_lane):
        seen["lane"] = router_lane
        return factory(cfg, fleet, artifact_root=artifact_root, avengers_repo=avengers_repo, router_lane=router_lane)

    async def scenario():
        async with Harness(tmp_path, fleet_states()) as h:
            h.service._factory = recording
            await h.start(pipeline_cfg())
            manifest = (await h.get("/v1/manifest"))[1]
            return h.service, manifest
    service, manifest = asyncio.run(scenario())
    assert seen["lane"] is service.router_lane and seen["lane"] is not service.executor
    assert service.router_lane._max_workers == 1 and manifest["launch"]["router_lane_threads"] == 1
