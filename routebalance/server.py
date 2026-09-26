"""The scheduler: one HTTP server for RouteBalance and every baseline.

  POST /v1/completions   one request; the contract is documented in bench/client.py
  GET  /v1/config        the live configuration with its config_digest and the fleet_digest
  POST /v1/config        replace the configuration whole: parsed strictly, built off the event loop, and
                         swapped in only while no request is being served (409 otherwise)
  GET  /v1/stats         parity evidence: event-loop lag, telemetry age, rounds, booster calls, logged lines
  GET  /v1/manifest      the execution manifest: everything this process executes for the live configuration
                         (code, library versions, launch parameters, pools, telemetry, loaded components and
                         artifact md5s); gates/parity.py diffs it across the systems of a comparison group
  GET  /health

What every system shares, by construction: one bounded HTTP session to the engines, one telemetry view (dead
reckoning, read per round, per request or from background pollers: telemetry.py), one backend client, one
thread pool for every model call (router models, the estimator, the scheduling round), and one dispatch path:
record the dispatch in the view, clamp max_tokens to the request's budget, stream from the engine reporting
progress to the view, close the dispatch. RouteBalance schedules requests in rounds (schedule/); a pipeline
system (a router, then a dispatcher) places each request as it arrives (baselines/pipeline.py).

Evidence under --log-dir: decisions.jsonl, one line per request of every configuration (what was chosen,
the state it was chosen on, the stage timings, the outcome, the config digest; RouteBalance lines also carry
the round, the request's position in it and the limits it was collected under, every scored candidate and
the estimator's expected length per model); and, for the D38 latency
sweep only (a pipeline configuration with pending_decode own_length), latency.<config digest>.jsonl in the
format tools/tpot_sweep.py reads. Its features are exactly the learned heads' serving inputs: the state the
decision read and, as the latency model feeds every head, the largest expected length over the models.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import importlib.metadata
import logging
import math
import os
import sys
import time
from concurrent.futures import Executor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import aiohttp
from aiohttp import web
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from routebalance.backend import BackendClient, BackendUnreachable, GenerationRequest, GenerationResult
from routebalance.baselines.dispatch import Dispatcher, build_dispatcher
from routebalance.baselines.pipeline import fleet_models, select_instance
from routebalance.baselines.routers import (BEST_ROUTE_MODULES, DecoupledInputs, Estimate as DecoupledEstimate, Router,
                                            RouterRequest, build_router, router_executor, router_lane)
from routebalance.config import (ConfigError, SchedulerConfig, config_digest, fleet_digest, load_fleet, load_scheduler,
                                 loads_strict, parse_scheduler)
from routebalance.decision_log import JsonlLog, LagProbe
from routebalance.estimate.artifacts import md5_tree
from routebalance.estimate.features import feature_values
from routebalance.schedule import budget
from routebalance.schedule.assign import Admission, Decision, RoundRequest, schedule_round
from routebalance.schedule.rounds import RoundCollector
from routebalance.telemetry import DispatchToken, InstanceStats, TelemetryView, fetch_metrics
from routebalance.types import Instance, Tier

log = logging.getLogger("routebalance.server")

P_UNDER_BUDGET_TOKENS = 256   # the estimator also reports P(length <= this); nothing here reads it (v3 passed 256)
MANIFEST_FORMAT = "rb-manifest/1"
PACKAGE_DIR = Path(__file__).resolve().parent
# Environment variables that size native thread pools (OpenMP, BLAS, the tokenizers' Rayon pool): recorded,
# never read as settings.
THREAD_ENV = ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES", "KMP_AFFINITY", "MKL_NUM_THREADS",
              "MKL_DYNAMIC", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
              "RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM", "CUDA_VISIBLE_DEVICES")
# The router option that names the artifact directory a router loads by path (its files are md5-recorded).
ROUTER_ARTIFACT_OPTION = {"best_route": "checkpoint", "best_route_upstream": "checkpoint", "avengers_pro": "artifact_dir"}


class Busy(RuntimeError):
    pass


# ---------------------------------------------------------------- the request

class Completion(BaseModel):
    """One request as bench/client.py sends it; budget_cost is present only in budget cells."""
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    request_id: str = Field(min_length=1)
    prompt: str
    prompt_tokens: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0)
    repetition_penalty: float = Field(gt=0)
    frequency_penalty: float
    budget_cost: Optional[float] = Field(default=None, gt=0)   # default-ok: absent = no budget (never 0)


@dataclass
class Assignment:
    instance: Instance
    token: DispatchToken
    state: InstanceStats        # the chosen instance's state the decision read, before this request
    evidence: dict              # what the decision log keeps about the choice
    timings_ms: dict            # the scheduler's stages before the backend call


# ---------------------------------------------------------------- what a configuration loads

@dataclass
class Components:
    estimator: object = None    # KnnEstimator-like: estimate_batch(prompts, budget_tokens) -> [{model: Estimate}]
    latency: object = None      # LatencyModel-like: latencies(instances, states, lengths, num_prompt_tokens)
    router: Optional[Router] = None
    dispatcher: Optional[Dispatcher] = None
    artifacts: dict[str, str] = field(default_factory=dict)   # {path under the artifact root: md5} of what was loaded


def needs(cfg: SchedulerConfig) -> tuple[bool, bool]:
    """(estimator, latency model) the configuration uses."""
    decoupled = cfg.pipeline is not None and cfg.pipeline.router.type == "rb_decoupled"
    rb = cfg.mode == "routebalance"
    return any((rb, decoupled, cfg.pending_decode == "own_length")), any((rb, decoupled))


def _tiers(fleet: list[Instance]) -> list[Tier]:
    return list({i.tier.tier_id: i.tier for i in fleet}.values())


def longest(estimates) -> int:
    """The output length the latency model feeds every learned head: the largest expectation over the
    models (v3's convention, estimate/latency.py)."""
    return max(1, int(max(e.length for e in estimates.values())))


def idle_state(tier: Tier) -> InstanceStats:
    return InstanceStats(0, 0, 0.0, 0, tier.max_num_seqs, 0)


def load_components(cfg: SchedulerConfig, fleet: list[Instance], *, artifact_root: Path,
                    avengers_repo: Optional[Path], router_lane: Executor,
                    best_route_repo: Optional[Path] = None) -> Components:
    """Loads models and tables (seconds): call it off the event loop. A pipeline router makes its model calls on
    router_lane (one request at a time, as shipped routers do); RouteBalance's own calls use the shared pool."""
    from routebalance.estimate.knn import INDEX_FILE, load_knn
    from routebalance.estimate.latency import load_latency_model
    need_est, need_lat = needs(cfg)
    c = Components()
    if need_est:
        c.estimator = load_knn(cfg.estimator, artifact_root, fleet_models(fleet))
        c.artifacts[f"{cfg.estimator.index_dir}/{INDEX_FILE}"] = cfg.estimator.index_md5   # verified at load
        c.artifacts |= model_files("estimator", c.estimator.index.encoder_name, cfg.estimator.encoder_revision)
    if need_lat:
        c.latency = load_latency_model(cfg.latency, artifact_root, [t.tier_id for t in _tiers(fleet)])
        c.artifacts |= latency_artifacts(cfg, c.latency)
    if cfg.pipeline is not None:
        decoupled = None
        if cfg.pipeline.router.type == "rb_decoupled":
            if cfg.latency.source == "xgboost":
                raise ConfigError("rb_decoupled scores idle instances; the xgboost source needs the request's "
                                  "prompt length, which the router interface does not carry")
            decoupled = _decoupled_inputs(cfg, fleet, c.estimator, c.latency)
        priced: dict[str, tuple[float, float]] = {}   # (input, output) per token, by model: one pair per model
        for i in fleet:
            pair = (i.tier.cost_in, i.tier.cost_out)
            if priced.setdefault(i.model, pair) != pair:   # default-ok: the first instance of the model sets it
                raise ConfigError(f"model {i.model} has two prices in the fleet ({priced[i.model]} and {pair})")
        prices = {m: (cin + cout) / 2 for m, (cin, cout) in priced.items()}   # the routers' cheapest-first order
        c.router = build_router(cfg.pipeline.router, artifact_root=artifact_root, avengers_repo=avengers_repo,
                                executor=router_lane, decoupled=decoupled, best_route_repo=best_route_repo,
                                prices=prices)
        c.dispatcher = build_dispatcher(cfg.pipeline.dispatch)
        options = cfg.pipeline.router.options()
        option = ROUTER_ARTIFACT_OPTION.get(cfg.pipeline.router.type)
        if option is not None:          # loaded by path, not pinned by md5: hash every file of the directory
            c.artifacts |= md5_tree(artifact_root, str(options[option]))
        if cfg.pipeline.router.type == "avengers_pro":      # its embedder is loaded by name at a pinned revision
            c.artifacts |= model_files("avengers_pro", str(options["embedder"]), str(options["embedder_revision"]))
        if cfg.pipeline.router.type == "best_route_upstream":   # upstream's modules and its ranker's backbone
            c.artifacts |= {f"best_route_upstream@{rel}": hashlib.md5((best_route_repo / rel).read_bytes()).hexdigest()
                            for rel in BEST_ROUTE_MODULES}
            c.artifacts |= model_files("best_route_upstream", str(options["base_model"]), str(options["base_revision"]))
    return c


def model_files(role: str, name: str, revision: Optional[str]) -> dict[str, str]:
    """{"<role>@<model>/<file>": md5} for every file of a model that a component loaded by name
    (sentence-transformers): the local directory `name`, or else the Hugging Face cache snapshot that `name`
    resolves to at `revision` (None: the cached main), found without network access. The role keeps two
    components that load the same model apart (a parity diff compares a file with the same role only)."""
    local = Path(name)
    if local.is_dir():
        return {f"{role}@{k}": m for k, m in md5_tree(local.parent, local.name).items()}
    from huggingface_hub import snapshot_download     # installed with sentence-transformers
    # the cache SentenceTransformer itself uses when given no cache_folder (None: the default Hub cache)
    cache = os.environ.get("SENTENCE_TRANSFORMERS_HOME")   # env-ok: the loader's own cache, read to find its files
    snap = Path(snapshot_download(name, revision=revision, local_files_only=True, cache_dir=cache))
    return {f"{role}@hf:{name}/{k.split('/', 1)[1]}": m for k, m in md5_tree(snap.parent, snap.name).items()}


def latency_artifacts(cfg: SchedulerConfig, latency) -> dict[str, str]:
    """The files the latency model read, each verified against its pinned md5 at load (the xgboost source
    lists its md5s.json and every file that manifest pinned and it read)."""
    t = cfg.latency
    if t.source != "xgboost":
        return {t.table: t.table_md5}
    return {f"{t.table}/{n}": m for n, m in latency.source.files.items()}


def py_tree_digest(directory: Path) -> str:
    """sha256 of the sha256sum listing of every .py file under `directory`, by path relative to it, in byte
    order: `(cd DIR && find . -name '*.py' -printf '%P\\n' | LC_ALL=C sort | xargs sha256sum) | sha256sum`."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ConfigError(f"{directory} is not a directory")
    files = sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*.py") if p.is_file())
    listing = "".join(f"{hashlib.sha256((directory / f).read_bytes()).hexdigest()}  {f}\n" for f in files)
    return hashlib.sha256(listing.encode()).hexdigest()


def library_versions() -> dict[str, str]:
    """The interpreter and every installed distribution (not a chosen list: a dependency of a dependency, such
    as aiohttp's connection library, runs too). Two installs of one name are joined, so they show."""
    found: dict[str, list[str]] = {}
    for dist in importlib.metadata.distributions():
        name = str(dist.metadata["Name"]).lower().replace("_", "-")
        found.setdefault(name, []).append(dist.version)   # default-ok: grouping
    return {"python": sys.version, **{n: "|".join(sorted(v)) for n, v in sorted(found.items())}}


def native_threadpools() -> dict[str, int]:
    """{"<api>:<library dir>/<library file>": threads} for every native thread pool loaded in the process
    (threadpoolctl), as a thread that never set its own count sees it: the launch environment's value."""
    from threadpoolctl import threadpool_info      # installed with scikit-learn
    out: dict[str, int] = {}
    for pool in threadpool_info():
        lib = Path(pool["filepath"])
        key = base = f"{pool['internal_api']}:{lib.parent.name}/{lib.name}"
        n = 2
        while key in out:
            key, n = f"{base}#{n}", n + 1
        out[key] = pool["num_threads"]
    return out


def _session_settings(session: aiohttp.ClientSession) -> dict:
    conn = session.connector
    return {"limit": conn.limit, "limit_per_host": conn.limit_per_host, "force_close": conn.force_close}


def _name(obj) -> Optional[str]:
    return None if obj is None else type(obj).__name__


def _decoupled_inputs(cfg: SchedulerConfig, fleet: list[Instance], estimator, latency) -> DecoupledInputs:
    first: dict[str, Instance] = {}          # each tier's first instance in fleet order
    for inst in fleet:
        if inst.tier.tier_id not in first:
            first[inst.tier.tier_id] = inst

    def estimate(prompts):
        return [{m: DecoupledEstimate(e.quality, e.length) for m, e in d.items()}
                for d in estimator.estimate_batch(list(prompts), P_UNDER_BUDGET_TOKENS)]

    def unloaded(tier: Tier, tokens: float) -> float:
        inst = first[tier.tier_id]
        return latency.latencies([inst], [idle_state(tier)], {tier.model: tokens}, 0)[0]
    return DecoupledInputs(_tiers(fleet), cfg.weights, cfg.latency.normalization, estimate, unloaded)


# ---------------------------------------------------------------- RouteBalance: rounds

@dataclass
class _Round:
    """One scheduling round as its dispatches see it: its number, size and limits, its stage times, and the
    waiting futures by request object. schedule_ms of a request runs from the load read to its own decision."""
    number: int
    size: int
    limits: tuple
    closed: float
    estimated: float
    fetched: float
    waiting: dict
    dispatched: int = 0   # default-ok: counts the round's dispatches as they happen


class RouteBalance:
    def __init__(self, cfg: SchedulerConfig, fleet: list[Instance], view: TelemetryView, comps: Components,
                 executor: Executor):
        self._cfg, self._fleet, self._view, self._c, self._executor = cfg, fleet, view, comps, executor
        self.collector: RoundCollector = RoundCollector(cfg.rounds)
        self.last_assigned: dict[str, float] = {}      # for this configuration's lifetime (the LRU tie-break)
        self.rounds = 0
        self._dispatch_error: Optional[BaseException] = None

    def submit(self, c: Completion) -> asyncio.Future:
        fut = asyncio.get_running_loop().create_future()
        self.collector.submit((c, fut, time.perf_counter()))
        return fut

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = await self.collector.next_round()
            closed = time.perf_counter()
            limits = (self.collector.max_batch, self.collector.timeout_ms)   # this round's; adapt() sets the next
            try:
                ests = await loop.run_in_executor(self._executor, self._c.estimator.estimate_batch,
                                                  [c.prompt for c, _, _ in batch], P_UNDER_BUDGET_TOKENS)
                estimated = time.perf_counter()
                if self._cfg.telemetry_read == "per_round":   # the per-batch load fetch (D53); raises if any scrape failed
                    loads = await self._view.fetch_round()
                else:                                          # the D54 control: the pollers' cache
                    loads = self._view.loads([i.instance_id for i in self._fleet])
                fetched = time.perf_counter()
                self.collector.adapt(loads)
                requests = [RoundRequest(c.request_id, c.prompt_tokens, e,
                                         0.0 if c.budget_cost is None else c.budget_cost, c.max_tokens)
                            for (c, _, _), e in zip(batch, ests, strict=True)]
                cfg = self._cfg
                admission = Admission() if cfg.budget.admission_filter else None
                self.rounds += 1                  # numbered before any dispatch; a round that fails leaves a gap
                rnd = _Round(self.rounds, len(batch), limits, closed, estimated, fetched,
                             {id(r): (fut, arrived) for r, (_, fut, arrived) in zip(requests, batch, strict=True)})

                def decided(d: Decision, rnd: _Round = rnd) -> None:
                    """On the pool's thread, right after each decision: dispatch it on the loop now (D78, v3's
                    per-pick dispatch), while the rest of the round is still being scored."""
                    n_feature = longest(d.request.estimates)   # may raise: this request is then not dispatched
                    loop.call_soon_threadsafe(self._dispatch, rnd, d, n_feature, time.perf_counter())

                await loop.run_in_executor(self._executor, functools.partial(
                    schedule_round, requests, self._fleet, loads, self._c.latency, weights=cfg.weights,
                    normalization=cfg.latency.normalization, lpt=cfg.lpt, pending_decode=cfg.pending_decode,
                    last_assigned=self.last_assigned, clock=time.monotonic, admission=admission,
                    on_decision=decided))
            except Exception as e:     # the round fails loudly: every request not yet dispatched gets the error
                for _, fut, _ in batch:
                    if not fut.done():
                        fut.set_exception(e)
            # the dispatch callbacks were queued before the round's result, so they have all run by now
            if self._dispatch_error is not None:
                raise self._dispatch_error

    def _dispatch(self, rnd: "_Round", d: Decision, n_feature: int, decided: float) -> None:
        """On the loop: record the dispatch in the view and hand the request its instance. A failure here fails
        the request and then the round loop (run() re-raises it), as the dispatch loop did before D78."""
        fut, arrived = rnd.waiting[id(d.request)]          # by object: duplicate request ids cannot collide
        try:
            pos = rnd.dispatched
            rnd.dispatched += 1
            token = self._view.record_dispatch(d.instance.instance_id, d.decode_tokens)
            best = next(s for s in d.scored if s.candidate.instance_id == d.instance.instance_id)
            evidence = {"round": rnd.number, "round_size": rnd.size, "round_pos": pos,
                        "round_max_batch": rnd.limits[0], "round_timeout_ms": rnd.limits[1],
                        "decode_tokens": d.decode_tokens, "lpt_key": d.lpt_key, "feature_tokens": n_feature,
                        "lengths": {m: e.length for m, e in d.request.estimates.items()},
                        "admitted_all": d.admitted_all, "score": best.score,
                        "candidates": [{"inst": s.candidate.instance_id, "score": s.score,
                                        "q": s.quality_term, "c": s.cost_term, "l": s.latency_term,
                                        "lat_s": s.candidate.latency_s} for s in d.scored]}
            ms = 1000.0
            timings = {"round_wait_ms": (rnd.closed - arrived) * ms, "estimator_ms": (rnd.estimated - rnd.closed) * ms,
                       "stats_fetch_ms": (rnd.fetched - rnd.estimated) * ms, "schedule_ms": (decided - rnd.fetched) * ms}
            if not fut.done():
                fut.set_result(Assignment(d.instance, token, d.state, evidence, timings))
            else:              # the client went away while the round was scheduled
                self._view.record_failure(token)
        except Exception as e:
            self._dispatch_error = e
            if not fut.done():
                fut.set_exception(e)

    def fail_pending(self, exc: BaseException) -> None:
        """The round loop is gone: every request still waiting for a round gets the error."""
        for _, fut, _ in self.collector.drain():
            if not fut.done():
                fut.set_exception(exc)


# ---------------------------------------------------------------- the service

class Service:
    def __init__(self, fleet: list[Instance], *, artifact_root: Path, avengers_repo: Optional[Path],
                 request_timeout_s: float, max_connections: int, torch_threads: int, log_dir: Path,
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
        self.router_lane = router_lane()
        self._log_dir = log_dir
        self.decisions = JsonlLog(log_dir / "decisions.jsonl")
        self.latency_log: Optional[JsonlLog] = None      # the D38 sweep configurations only
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
        self.connection_waits = {"engine": 0, "telemetry": 0}   # waits for a pooled connection (a requeued one counts again)

    async def start(self, cfg: SchedulerConfig) -> None:
        # streams get the whole bound; telemetry scrapes have their own keep-alive pool, as large, so that v3's
        # per-request probes (D55: every request fetches its candidates at once) never queue on the pool; the
        # pollers and the per-round fetch keep at most one scrape per instance in flight either way. The bound is
        # set far above any in-flight count; a request that ever waits for a connection is counted, and the arm
        # gate invalidates that cell (rerun it with a larger max_connections), so the bound never shapes a result
        self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=self._max_conn, force_close=True),
                                              trace_configs=[self._wait_counter("engine")])
        self._telemetry_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=self._max_conn),
                                                        trace_configs=[self._wait_counter("telemetry")])
        self.backend = BackendClient(self._session, self._timeout_s)
        for coro in (self.decisions.run(), self.lag.run()):
            self._tasks.append(self._watch(asyncio.create_task(coro)))
        await self.apply(cfg)

    def _wait_counter(self, session: str) -> aiohttp.TraceConfig:
        trace = aiohttp.TraceConfig()

        async def queued(_session, _ctx, _params):
            self.connection_waits[session] += 1
        trace.on_connection_queued_start.append(queued)
        return trace

    def _watch(self, task: asyncio.Task, on_fail: Optional[Callable[[BaseException], None]] = None) -> asyncio.Task:
        def done(t: asyncio.Task):
            if not t.cancelled() and t.exception() is not None:
                self.failure = f"{t.get_coro().__qualname__}: {t.exception()!r}"
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
            loop = asyncio.get_running_loop()
            comps = await loop.run_in_executor(self.executor, functools.partial(
                self._factory, cfg, self.fleet, artifact_root=self._artifacts, avengers_repo=self._avengers,
                router_lane=self.router_lane, best_route_repo=self._best_route))
            t = cfg.telemetry
            view = TelemetryView(self.fleet, lambda i: fetch_metrics(self._telemetry_session, i, t.timeout_s),
                                 poll_ms=t.poll_ms, max_age_ms=t.max_age_ms, dead_reckoning=cfg.dead_reckoning)
            # every instance answers once, or this raises and the old config stays live
            if cfg.telemetry_read == "poll":  # the pollers' cache (D19)
                await view.start()
            else:                             # fetched per round (D53) or per request (D55): no pollers
                await view.fetch_round()
            if self._mode_task is not None:
                self._mode_task.cancel()
                await asyncio.gather(self._mode_task, return_exceptions=True)
                self._mode_task = None
            if self.view is not None:
                await self.view.stop()
            self.view, self._comps, self.cfg = view, comps, cfg
            self.digest = config_digest(cfg)
            self.lag.reset()                  # the probe's evidence is per configuration (per cell)
            self.latency_log = None
            if cfg.mode == "pipeline" and cfg.pending_decode == "own_length":
                if self.digest not in self._latency_logs:
                    lg = JsonlLog(self._log_dir / f"latency.{self.digest[:16]}.jsonl")
                    self._latency_logs[self.digest] = lg
                    self._tasks.append(self._watch(asyncio.create_task(lg.run())))
                self.latency_log = self._latency_logs[self.digest]
            self._rb = None
            if cfg.mode == "routebalance":
                self._rb = RouteBalance(cfg, self.fleet, view, comps, self.executor)
                self._mode_task = self._watch(asyncio.create_task(self._rb.run()), on_fail=self._rb.fail_pending)
        finally:
            self._swapping = False

    async def complete(self, c: Completion) -> dict:
        if self.failure is not None:
            raise RuntimeError(f"the scheduler has failed: {self.failure}")
        if self._swapping or self.cfg is None:
            raise Busy("a configuration is being applied")
        self._active += 1
        try:
            try:
                a = await (self._await_round(c) if self._rb is not None else self._select(c))
            except BaseException as e:        # no dispatch happened: log the request anyway
                self._log(c, None, None, c.max_tokens, None, f"{type(e).__name__}: {e}")
                raise
            return await self._serve(c, a)
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
        if self.cfg.pending_decode == "own_length":     # the D38 sweep: record the estimator's length per model
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
                                    RouterRequest(c.prompt, c.request_id), decode_tokens=decode,
                                    probe=self.cfg.telemetry_read == "per_request")
        o = sel.overhead
        return Assignment(sel.instance, sel.token, sel.load.stats,
                          {"reason": sel.reason, "router_score": sel.score, "decode_tokens": decode[sel.model],
                           "feature_tokens": feature_tokens, **lengths},
                          {"router_ms": o.router_ms, "telemetry_ms": o.telemetry_ms, "dispatch_ms": o.dispatch_ms})

    async def _serve(self, c: Completion, a: Assignment) -> dict:
        inst, token = a.instance, a.token
        max_tokens, clamp = c.max_tokens, None
        if c.budget_cost is not None:
            clamp = budget.clamp(inst.tier, c.budget_cost, c.prompt_tokens, c.max_tokens)
            max_tokens = clamp.max_tokens
        req = GenerationRequest(c.request_id, c.prompt, max_tokens, c.temperature, c.repetition_penalty,
                                c.frequency_penalty)
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
                "ttft_s": res.ttft_s, "server_latency_s": res.latency_s, "timings_ms": a.timings_ms}

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
                "connection_waits": dict(self.connection_waits),
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
                       "rounds": dump(cfg.rounds), "lpt": cfg.lpt,
                       "pending_decode": cfg.pending_decode, "latency": dump(cfg.latency),
                       "estimator": dump(cfg.estimator), "router": None if pipe is None else dump(pipe.router),
                       "dispatch": None if pipe is None else dump(pipe.dispatch)},
            "budget": dump(cfg.budget),
            "telemetry": {"poll_ms": view.poll_ms, "max_age_ms": view.max_age_ms, "timeout_s": cfg.telemetry.timeout_s,
                          "dead_reckoning": view.dead_reckoning},
            "loaded": {"estimator": _name(comps.estimator), "latency_source": None if lat is None else lat.source.kind,
                       "latency_envelope": None if lat is None else lat.envelope, "router": _name(comps.router),
                       "dispatcher": _name(comps.dispatcher), "round_loop": rb is not None,
                       "telemetry_pollers": pollers},
            "artifacts": dict(comps.artifacts),
            "code": {**self._code, **{f"{k}_now": v for k, v in code_now.items()}},
            "threadpools": pools,
            "versions": dict(self._versions),
            "launch": {"torch_threads": self._torch_threads, "torch_threads_effective": threads,
                       "max_connections": self._max_conn, "request_timeout_s": self._timeout_s,
                       "router_threads": self.executor._max_workers,
                       "router_lane_threads": self.router_lane._max_workers,
                       "engine_session": _session_settings(self._session),
                       "telemetry_session": _session_settings(self._telemetry_session),
                       "event_loop": f"{type(loop).__module__}.{type(loop).__qualname__}",
                       "cpus": {"count": os.cpu_count(), "affinity": len(os.sched_getaffinity(0))},
                       "env": {k: os.environ.get(k) for k in THREAD_ENV}},   # env-ok: recorded, not a setting
        }

    async def stop(self) -> None:
        for t in [*self._tasks, *([self._mode_task] if self._mode_task else [])]:
            t.cancel()
        await asyncio.gather(*self._tasks, *([self._mode_task] if self._mode_task else []), return_exceptions=True)
        await self.decisions.flush()
        for lg in self._latency_logs.values():
            await lg.flush()
        if self.view is not None:
            await self.view.stop()
        await self._session.close()
        await self._telemetry_session.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.router_lane.shutdown(wait=False, cancel_futures=True)


class ServeError(RuntimeError):
    pass


# ---------------------------------------------------------------- HTTP

def build_app(service: Service) -> web.Application:
    async def completions(request: web.Request) -> web.Response:
        try:
            c = Completion.model_validate(loads_strict(await request.text()))
        except (ValidationError, ConfigError, ValueError) as e:
            return web.json_response({"success": False, "error": f"bad request: {e}"}, status=400)
        try:
            return web.json_response(await service.complete(c))
        except Busy as e:
            return web.json_response({"success": False, "error": str(e)}, status=503)
        except (ServeError, BackendUnreachable) as e:
            return web.json_response({"success": False, "error": str(e)}, status=502)
        except Exception as e:         # the scheduler failed this request: loud, never a fabricated reply
            log.exception("request %s failed", c.request_id)
            return web.json_response({"success": False, "error": f"{type(e).__name__}: {e}"}, status=500)

    async def get_config(request: web.Request) -> web.Response:
        cfg = service.cfg
        return web.json_response({"config": None if cfg is None else cfg.model_dump(mode="json"),
                                  "config_digest": None if cfg is None else config_digest(cfg),
                                  "fleet_digest": service.fleet_digest})

    async def post_config(request: web.Request) -> web.Response:
        try:
            cfg = parse_scheduler(loads_strict(await request.text()))
        except (ConfigError, ValueError) as e:
            return web.json_response({"error": str(e)}, status=400)
        try:
            await service.apply(cfg)
        except Busy as e:
            return web.json_response({"error": str(e)}, status=409)
        except Exception as e:         # nothing was swapped: the old configuration is still live
            return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=400)
        return await get_config(request)

    async def stats(request: web.Request) -> web.Response:
        return web.json_response(service.stats())

    async def manifest(request: web.Request) -> web.Response:
        try:
            return web.json_response(await service.manifest())
        except Busy as e:
            return web.json_response({"error": str(e)}, status=503)

    async def health(request: web.Request) -> web.Response:
        ok = service.failure is None and service.cfg is not None
        return web.Response(text="ok" if ok else "failed", status=200 if ok else 503)

    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/v1/config", get_config)
    app.router.add_post("/v1/config", post_config)
    app.router.add_get("/v1/stats", stats)
    app.router.add_get("/v1/manifest", manifest)
    app.router.add_get("/health", health)
    return app


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fleet", type=Path, required=True)
    p.add_argument("--hosts", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True, help="the scheduler configuration served at start")
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--avengers-repo", type=Path, default=None, help="upstream AvengersPro checkout (its router)")
    p.add_argument("--best-route-repo", type=Path, default=None, help="upstream best-route-llm checkout (its ranker)")
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--request-timeout-s", type=float, required=True)
    p.add_argument("--max-connections", type=int, required=True, help="engine connection bound, above peak in flight")
    p.add_argument("--torch-threads", type=int, required=True, help="intra-op threads for every model call")
    p.add_argument("--log-dir", type=Path, required=True)
    a = p.parse_args(argv)
    if not (a.request_timeout_s > 0 and math.isfinite(a.request_timeout_s)) or a.max_connections < 1 or a.torch_threads < 1:
        p.error("timeouts, connection bound and thread count must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    import torch                      # one explicit thread count for every system's models
    torch.set_num_threads(a.torch_threads)
    import uvloop
    uvloop.install()
    fleet, cfg = load_fleet(a.fleet, a.hosts), load_scheduler(a.config)

    async def serve():
        service = Service(fleet, artifact_root=a.artifacts, avengers_repo=a.avengers_repo,
                          request_timeout_s=a.request_timeout_s, max_connections=a.max_connections,
                          torch_threads=a.torch_threads, log_dir=a.log_dir, best_route_repo=a.best_route_repo)
        await service.start(cfg)
        runner = web.AppRunner(build_app(service), access_log=None)
        await runner.setup()
        await web.TCPSite(runner, a.host, a.port).start()
        log.info("serving %s on %s:%d (config %s, fleet %s)", cfg.mode, a.host, a.port, config_digest(cfg),
                 service.fleet_digest)
        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()
            await service.stop()
    asyncio.run(serve())


if __name__ == "__main__":
    main()
