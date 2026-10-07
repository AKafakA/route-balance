"""The scheduler's service: one fleet, the live configuration and everything it loaded, and the path of every request.

What every system shares, by construction: one HTTP session to the engines (vLLM bench's stock connector), one
telemetry view (dead reckoning, read per round, per request or from background pollers: telemetry.py), one
backend client, one thread pool for RouteBalance's model calls (the estimator, the scheduling round) beside the
pipeline routers' own lane (baselines/routers.py), and one dispatch path: record the dispatch in the view, clamp
max_tokens to the request's budget, stream from the engine reporting progress to the view, close the dispatch.
RouteBalance schedules requests in rounds (round_loop.py); a pipeline system (a router, then a dispatcher) places
each request as it arrives (baselines/pipeline.py).

Evidence under the log directory: decisions.jsonl, one line per request of every configuration (what was chosen,
the state it was chosen on, the stage timings, the outcome, the config digest; RouteBalance lines also carry the
round, the request's position in it and the limits it was collected under, every scored candidate and the
estimator's expected length per model); and, for the latency sweep only (a pipeline configuration with
pending_decode own_length), latency.<config digest>.jsonl in the format tools/tpot_sweep.py reads. Its features
are exactly the learned heads' serving inputs: the state the decision read and, as the latency model feeds every
head, the largest expected length over the models.
"""
from __future__ import annotations

import asyncio
import functools
import gc
import logging
import os
import sys
import time
from concurrent.futures import Executor
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import aiohttp
import uvicorn

from routebalance.backend import BackendClient, GenerationRequest, GenerationResult, stock_connector
from routebalance.baselines.pipeline import fleet_models, select_instance
from routebalance.baselines.routers import RouterRequest, router_executor, router_lane
from routebalance.components import P_UNDER_BUDGET_TOKENS, Components, load_components, longest
from routebalance.config import ConfigError, SchedulerConfig, config_digest, fleet_digest
from routebalance.decision_log import FLUSH_S, JsonlLog, LagProbe
from routebalance.estimate.features import feature_values
from routebalance.execution import (MANIFEST_FORMAT, PACKAGE_DIR, THREAD_ENV, http_settings, library_versions,
                                    native_threadpools, open_files_limit, py_tree_digest, session_settings,
                                    tcp_listen_counters, type_name)
from routebalance.request import Assignment, Busy, Completion, ServeError
from routebalance.round_loop import RouteBalance
from routebalance.schedule import budget
from routebalance.telemetry import TelemetryView, fetch_metrics
from routebalance.types import Instance

log = logging.getLogger(__name__)


class Service:
    def __init__(self, fleet: list[Instance], *, artifact_root: Path, avengers_repo: Optional[Path],
                 request_timeout_s: float, max_connections: int, torch_threads: Optional[int], log_dir: Path,
                 factory: Callable[..., Components] = load_components, best_route_repo: Optional[Path] = None):
        if not fleet:
            raise ConfigError("empty fleet")
        self.fleet = fleet
        self.fleet_digest = fleet_digest(fleet)
        self._models = fleet_models(fleet)
        self._artifacts, self._avengers, self._best_route = artifact_root, avengers_repo, best_route_repo
        self._timeout_s, self._max_conn, self._torch_threads = request_timeout_s, max_connections, torch_threads
        # the code this process runs, hashed at start (a later edit of the files does not change what runs)
        self._code = self._code_digests()
        self._versions = library_versions()
        self._factory = factory
        self.executor = router_executor()
        self.router_lane: Optional[Executor] = None   # built per configuration (apply): its router's concurrency
        self._log_dir = log_dir
        self.decisions = JsonlLog(log_dir / "decisions.jsonl", FLUSH_S)
        self.latency_log: Optional[JsonlLog] = None      # the latency sweep's configurations only
        self._latency_logs: dict[str, JsonlLog] = {}
        self.lag = LagProbe()
        self.cfg: Optional[SchedulerConfig] = None
        self.digest: Optional[str] = None
        self.view: Optional[TelemetryView] = None
        self._comps = Components()
        self._rb: Optional[RouteBalance] = None
        self._tasks: list[asyncio.Task] = []
        self._mode_task: Optional[asyncio.Task] = None
        self._active = 0
        self._swapping = False
        self.failure: Optional[str] = None
        # no refusal: a failure to write the evidence (decision, latency or lag logs) is
        # recorded here and makes the run invalid (gates/cell.py isolation_problems); requests keep being served
        self.evidence_failure: Optional[str] = None
        self.connection_waits = {"engine": 0, "telemetry": 0}   # waits for a pooled connection (a requeued one counts again)
        self.http: Optional[uvicorn.Server] = None                # the server this service is served by (the manifest)

    async def start(self, cfg: SchedulerConfig) -> None:
        # as vLLM bench: the engine streams and the telemetry scrapes each get vLLM bench's stock
        # connector (no connection limit, keep-alive); a wait for a connection is still counted (none can happen)
        self._session = aiohttp.ClientSession(connector=stock_connector(),
                                              trace_configs=[self._wait_counter("engine")])
        self._telemetry_session = aiohttp.ClientSession(connector=stock_connector(),
                                                        trace_configs=[self._wait_counter("telemetry")])
        self.backend = BackendClient(self._session, self._timeout_s)
        for coro in (self.decisions.run(), self.lag.run()):
            self._tasks.append(self._watch(asyncio.create_task(coro), evidence=True))
        await self.apply(cfg)

    def _wait_counter(self, session: str) -> aiohttp.TraceConfig:
        trace = aiohttp.TraceConfig()

        async def queued(_session, _ctx, _params):
            self.connection_waits[session] += 1
        trace.on_connection_queued_start.append(queued)
        return trace

    def _watch(self, task: asyncio.Task, on_fail: Optional[Callable[[BaseException], None]] = None, *,
               evidence: bool = False) -> asyncio.Task:
        def done(t: asyncio.Task):
            if not t.cancelled() and t.exception() is not None:
                what = f"{t.get_coro().__qualname__}: {t.exception()!r}"
                if evidence:                     # a log writer: counted, never a reason to refuse a request
                    if self.evidence_failure is None:          # the first failure is the evidence
                        self.evidence_failure = what
                    log.error("an evidence writer failed; requests are still served, the run is invalid: %s", what)
                    return
                self.failure = what
                log.error("background task failed, refusing requests: %s", self.failure)
                if on_fail is not None:
                    on_fail(RuntimeError(f"the scheduler has failed: {self.failure}"))
        task.add_done_callback(done)
        return task

    @property
    def busy(self) -> bool:
        return self._active > 0

    async def apply(self, cfg: SchedulerConfig) -> None:
        """Build everything the new configuration needs, then swap it in whole, only while idle."""
        if self.busy or self._swapping:
            raise Busy(f"{self._active} request(s) in service; apply configs between cells")
        self._swapping = True
        try:
            gc.unfreeze()                     # the previous configuration's objects can be collected again
            loop = asyncio.get_running_loop()
            threads = None if cfg.pipeline is None else cfg.pipeline.router.concurrency
            lane = None if threads is None else router_lane(threads)
            try:                              # until the swap, a failure leaves the old config live: drop the new lane
                comps = await loop.run_in_executor(self.executor, functools.partial(
                    self._factory, cfg, self.fleet, artifact_root=self._artifacts, avengers_repo=self._avengers,
                    router_lane=lane, best_route_repo=self._best_route))
                t = cfg.telemetry
                view = TelemetryView(self.fleet, lambda i: fetch_metrics(self._telemetry_session, i, t.timeout_s),
                                     poll_ms=t.poll_ms, dead_reckoning=cfg.dead_reckoning)
                # every instance answers once, or this raises and the old config stays live
                if cfg.telemetry_read == "poll":  # the pollers' cache
                    await view.start()
                else:                             # fetched per round or per request: no pollers
                    await view.fetch_round()
            except BaseException:
                if lane is not None:
                    lane.shutdown(wait=False, cancel_futures=True)
                raise
            if self._mode_task is not None:
                self._mode_task.cancel()
                await asyncio.gather(self._mode_task, return_exceptions=True)
                self._mode_task = None
            if self.view is not None:
                await self.view.stop()
            if self.router_lane is not None:   # idle (apply refuses while busy): the old configuration's lane
                self.router_lane.shutdown(wait=False, cancel_futures=True)
            self.router_lane = lane
            self.view, self._comps, self.cfg = view, comps, cfg
            self.digest = config_digest(cfg)
            # the models and indexes just loaded, and the server's state, are long-lived; a full collection over
            # them paused the event loop 170-470 ms once per run. Collected once here (between cells, idle) and
            # frozen, a collection during a run scans only the requests' own objects
            gc.collect()
            gc.freeze()
            self.lag.reset()                  # the probe's evidence is per configuration (per cell)
            self.latency_log = None
            if cfg.mode == "pipeline" and cfg.pending_decode == "own_length":
                if self.digest not in self._latency_logs:
                    lg = JsonlLog(self._log_dir / f"latency.{self.digest[:16]}.jsonl", FLUSH_S)
                    self._latency_logs[self.digest] = lg
                    self._tasks.append(self._watch(asyncio.create_task(lg.run()), evidence=True))
                self.latency_log = self._latency_logs[self.digest]
            self._rb = None
            if cfg.mode == "routebalance":
                self._rb = RouteBalance(cfg, self.fleet, view, comps, self.executor)
                self._mode_task = self._watch(asyncio.create_task(self._rb.run()), on_fail=self._rb.fail_pending)
        finally:
            self._swapping = False

    async def complete(self, c: Completion) -> dict:
        arrived = time.perf_counter()           # the request's arrival, for its first-token time (as v2's client_ttft)
        if self.failure is not None:
            raise RuntimeError(f"the scheduler has failed: {self.failure}")
        if self._swapping or self.cfg is None:
            raise Busy("a configuration is being applied")
        self._active += 1
        try:
            try:
                if self.cfg.budget.admission_filter and c.budget_cost is None:
                    raise ServeError(f"{c.request_id}: the budget filter is on and the request has no budget_cost")
                a = await (self._await_round(c) if self._rb is not None else self._select(c))
            except BaseException as e:        # no dispatch happened: log the request anyway
                self._log(c, None, None, c.max_tokens, None, f"{type(e).__name__}: {e}")
                raise
            return await self._serve(c, a, arrived)
        finally:
            self._active -= 1

    async def _await_round(self, c: Completion) -> Assignment:
        fut = self._rb.submit(c)
        try:
            return await fut
        except asyncio.CancelledError:        # cancelled after the round placed it: close that dispatch here
            if fut.done() and not fut.cancelled() and fut.exception() is None:
                self.view.record_failure(fut.result().token)
            raise

    async def _select(self, c: Completion) -> Assignment:
        if self.cfg.pending_decode == "own_length":     # the latency sweep: record the estimator's length per model
            (est,) = await asyncio.get_running_loop().run_in_executor(
                self.executor, self._comps.estimator.estimate_batch, [c.prompt], P_UNDER_BUDGET_TOKENS)
            decode = {m: max(1, int(est[m].length)) for m in self._models}
            feature_tokens = longest(est)
            lengths = {"lengths": {m: e.length for m, e in est.items()}}
        else:
            decode = {m: c.max_tokens for m in self._models}     # v3: baselines record max_tokens
            feature_tokens = c.max_tokens
            lengths = {}
        sel = await select_instance(self._comps.router, self._comps.dispatcher, self.fleet, self.view,
                                    RouterRequest(c.prompt, c.request_id, c.prompt_tokens), decode_tokens=decode,
                                    probe=self.cfg.telemetry_read == "per_request")
        o = sel.overhead
        return Assignment(sel.instance, sel.token, sel.load.stats,
                          {"reason": sel.reason, "router_score": sel.score, "decode_tokens": decode[sel.model],
                           "feature_tokens": feature_tokens, "in_flight": sel.load.in_flight,
                           "telemetry_age_ms": round(sel.load.age_ms, 1), "telemetry_error": sel.load.error,
                           **lengths},
                          {"router_ms": o.router_ms, "telemetry_ms": o.telemetry_ms, "dispatch_ms": o.dispatch_ms})

    async def _serve(self, c: Completion, a: Assignment, arrived: float) -> dict:
        inst, token = a.instance, a.token
        max_tokens, clamp = c.max_tokens, None
        if c.budget_cost is not None:
            clamp = budget.clamp(inst.tier, c.budget_cost, c.prompt_tokens, c.max_tokens)
            max_tokens = clamp.max_tokens
        req = GenerationRequest(c.request_id, c.prompt, max_tokens, c.temperature, c.repetition_penalty,
                                c.frequency_penalty)
        sent = time.perf_counter()
        try:
            res = await self.backend.generate(inst, req, on_progress=functools.partial(self.view.record_progress, token))
        except BaseException as e:     # unreachable instance, cancellation: the dispatch is over either way
            self.view.record_failure(token)
            self._log(c, a, None, max_tokens, clamp, f"{type(e).__name__}: {e}")
            raise
        (self.view.record_completion if res.success else self.view.record_failure)(token)
        d = self._comps.dispatcher
        if d is not None:
            if res.success:
                d.observe_completion(inst.instance_id, res.latency_s, res.output_tokens)
            elif res.timed_out:
                d.observe_timeout(inst.instance_id)
        self._log(c, a, res, max_tokens, clamp, res.error)
        if not res.success:
            raise ServeError(res.error)
        return {"success": True, "request_id": c.request_id, "model": inst.model, "instance_id": inst.instance_id,
                "generated_text": res.text, "prompt_tokens": res.prompt_tokens, "output_tokens": res.output_tokens,
                "ttft_s": res.ttft_s, "server_latency_s": res.latency_s, "timings_ms": a.timings_ms,
                "first_token_s": sent - arrived + res.ttft_s}

    def _log(self, c: Completion, a: Optional[Assignment], res: Optional[GenerationResult], max_tokens: int,
             clamp, error: str) -> None:
        """One decision line per request, whatever happened; a latency line for the sweep configurations."""
        ok = res is not None and res.success
        line = {"request_id": c.request_id, "config_digest": self.digest, "success": ok, "error": error,
                "max_tokens": max_tokens, "budget": None if clamp is None else asdict(clamp),
                "request": {"prompt_tokens": c.prompt_tokens, "max_tokens": c.max_tokens,
                            "budget_cost": c.budget_cost}}      # as sent: the budget filter's inputs (gates/arm.py)
        if a is not None:
            line |= {"instance_id": a.instance.instance_id, "tier": a.instance.tier.tier_id, "state": asdict(a.state),
                     **a.evidence, "timings_ms": a.timings_ms}
        if res is not None:
            line |= {"timed_out": res.timed_out, "prompt_tokens": res.prompt_tokens,
                     "output_tokens": res.output_tokens, "ttft_s": res.ttft_s, "latency_s": res.latency_s}
        self.decisions.write(line)
        if self.latency_log is not None and a is not None:
            n = 0 if res is None else res.output_tokens
            tpot = (res.latency_s - res.ttft_s) / (n - 1) if ok and n >= 2 else None   # per token: merged chunks
            self.latency_log.write({"tier": a.instance.tier.tier_id,
                                    "features": feature_values(a.state, c.prompt_tokens, a.evidence["feature_tokens"]),
                                    "tpot_s": tpot, "output_tokens": n, "success": ok})

    def telemetry_read(self) -> Optional[str]:
        return None if self.cfg is None else self.cfg.telemetry_read

    def stats(self) -> dict:
        ages = {}
        if self.view is not None and self.telemetry_read() == "poll":
            try:
                loads = self.view.loads([x.instance_id for x in self.fleet])
                ages = {i: round(load.age_ms, 1) for i, load in loads.items()}
            except Exception as e:     # stale telemetry is itself the evidence
                ages = {"error": str(e)}
        lat = self._comps.latency
        return {"loop_lag": self.lag.summary(), "telemetry_read": self.telemetry_read(),
                "telemetry_age_ms": ages if self.telemetry_read() == "poll" else None, "active": self._active,
                "rounds": None if self._rb is None else self._rb.rounds,
                "booster_calls": None if lat is None else lat.predict_calls, "failure": self.failure,
                "evidence_failure": self.evidence_failure,
                "connection_waits": dict(self.connection_waits), "tcp_listen": tcp_listen_counters(),
                "logged": {"decisions": self.decisions.written, "decisions_buffered": self.decisions.buffered,
                           "latency": None if self.latency_log is None else self.latency_log.written}}

    def _code_digests(self) -> dict[str, Optional[str]]:
        return {"routebalance": py_tree_digest(PACKAGE_DIR),
                "avengers_pro": None if self._avengers is None else py_tree_digest(self._avengers)}

    async def manifest(self) -> dict:
        """Everything this process executes for the live configuration, read from the running objects."""
        if self._swapping or self.cfg is None:
            raise Busy("a configuration is being applied")
        cfg, view, comps, rb = self.cfg, self.view, self._comps, self._rb       # one consistent snapshot
        lane = self.router_lane                                                  # with it
        pollers = view.polling
        loop = asyncio.get_running_loop()
        torch = sys.modules.get("torch")      # imported by main() before serving; measured on a pool thread
        threads = None if torch is None else await loop.run_in_executor(self.executor, torch.get_num_threads)
        pools = await loop.run_in_executor(None, native_threadpools)
        # the code on disk now: a module imported after start (the xgboost source, the AvengersPro checkout)
        # runs the files as they are when it is imported, so a change since start must show
        code_now = await loop.run_in_executor(None, self._code_digests)
        pipe, lat = cfg.pipeline, comps.latency

        def dump(section):
            return None if section is None else section.model_dump(mode="json")
        return {
            "format": MANIFEST_FORMAT,
            "config_digest": config_digest(cfg),
            "fleet_digest": self.fleet_digest,
            "fleet": [{"instance_id": i.instance_id, "tier": i.tier.tier_id, "model": i.model, "gpu": i.tier.gpu,
                       "cost_in": i.tier.cost_in, "cost_out": i.tier.cost_out, "max_num_seqs": i.tier.max_num_seqs,
                       "node": i.node} for i in self.fleet],
            "system": {"mode": cfg.mode, "telemetry_read": cfg.telemetry_read, "weights": dump(cfg.weights),
                       "rounds": dump(cfg.rounds), "lpt": cfg.lpt, "normalize_over": cfg.normalize_over,
                       "pending_decode": cfg.pending_decode, "latency": dump(cfg.latency),
                       "estimator": dump(cfg.estimator), "router": None if pipe is None else dump(pipe.router),
                       "dispatch": None if pipe is None else dump(pipe.dispatch)},
            "budget": dump(cfg.budget),
            "telemetry": {"poll_ms": view.poll_ms, "timeout_s": cfg.telemetry.timeout_s,
                          "dead_reckoning": view.dead_reckoning},
            "loaded": {"estimator": type_name(comps.estimator), "latency_source": None if lat is None else lat.source.kind,
                       "latency_envelope": None if lat is None else lat.envelope, "router": type_name(comps.router),
                       "dispatcher": type_name(comps.dispatcher), "round_loop": rb is not None,
                       "telemetry_pollers": pollers,
                       "router_lane_threads": None if lane is None else lane._max_workers},
            "artifacts": dict(comps.artifacts),
            "code": {**self._code, **{f"{k}_now": v for k, v in code_now.items()}},
            "threadpools": pools,
            "versions": dict(self._versions),
            "launch": {"torch_threads": self._torch_threads, "torch_threads_effective": threads,
                       "max_connections": self._max_conn, "request_timeout_s": self._timeout_s,
                       "router_threads": self.executor._max_workers,
                       "engine_session": session_settings(self._session),
                       "telemetry_session": session_settings(self._telemetry_session),
                       "event_loop": f"{type(loop).__module__}.{type(loop).__qualname__}",
                       "http_server": http_settings(self.http), "open_files": open_files_limit(),
                       "cpus": {"count": os.cpu_count(), "affinity": len(os.sched_getaffinity(0))},
                       "env": {k: os.environ.get(k) for k in THREAD_ENV}},   # env-ok: recorded, not a setting
        }

    async def stop(self) -> None:
        for t in [*self._tasks, *([self._mode_task] if self._mode_task else [])]:
            t.cancel()
        await asyncio.gather(*self._tasks, *([self._mode_task] if self._mode_task else []), return_exceptions=True)
        for lg in (self.decisions, *self._latency_logs.values()):
            try:
                await lg.flush()
            except OSError as e:          # the evidence is incomplete (recorded); the shutdown still completes
                if self.evidence_failure is None:
                    self.evidence_failure = f"final flush of {lg.path.name}: {e!r}"
                log.error("the final flush of %s failed: %r", lg.path, e)
        if self.view is not None:
            await self.view.stop()
        await self._session.close()
        await self._telemetry_session.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.router_lane is not None:
            self.router_lane.shutdown(wait=False, cancel_futures=True)
