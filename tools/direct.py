"""Requests sent directly to the engines, never through the scheduler.

Used for what must not enter the scheduler's decision log or counters: waiting for idle engines (drain),
warming the engines up before a cell, and the closed-loop runs of W1 (calibration).
Every request goes through the scheduler's own backend client (routebalance.backend), so it is sent exactly as
a served request is: the raw prompt as the single user message, templated once by vLLM, with the scope's
sampling. Engine load is read from the same Prometheus /metrics the telemetry view scrapes.

The warm-up also empties every engine's prefix cache (v2 gave every main-table cell a freshly deployed
cluster; here the engines stay up, so each run resets them instead). Per engine, one at a time and all engines at once:
the warm-up prompt once (so it is cached), vLLM's POST /reset_prefix_cache (served only with VLLM_SERVER_DEV_MODE=1),
then the warm-up requests. The first after the reset must find no cached token and the second, the same prompt, must
find some: the reset emptied the cache, and the counters (vllm:prefix_cache_queries_total and _hits_total, read before
and after each request, the engine idle) do show it. Anything else raises, so a cell never runs on an unproven cache.
Every cell therefore starts from the same cache: empty but for the warm-up prompt's own blocks.

closed_loop keeps `concurrency` requests in flight on one instance until every measured request has finished.
When fewer measured requests are left than the concurrency, filler requests (discarded, never measured) keep the
batch at its level, so every measured request decodes with the level's number of requests on the instance; the
fillers still in flight are cancelled (the client disconnect makes vLLM abort them) and the caller drains. Each
filler prompt is sent at most once (a repeat would be a prefix-cache hit); a run that uses them all up before its
last measured request finishes reports it, and its level did not hold.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import aiohttp

from bench.client import check_fd_limit
from bench.dataset import file_sha256, read_rows
from bench.records import Sampling
from routebalance.backend import BackendClient, GenerationRequest, GenerationResult
from routebalance.config import ConfigError, fleet_digest, load_fleet
from routebalance.telemetry import fetch_metrics, parse_metrics
from routebalance.types import Instance

POLL_S = 0.5                       # how often a drain re-reads the engines
SCRAPE_TIMEOUT_S = 5.0             # one /metrics read (a failure bound, as the drain timeout)
COUNTED_S = 5.0                    # how long a warm-up request may take to appear in the prefix-cache counters (a bound)
COUNTER_POLL_S = 0.05
PREFIX_COUNTERS = ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total")


def lease_fleet(plan: dict, name: str, fleet_path: Path, hosts_path: Path) -> list[Instance]:
    """The fleet file a plan's row names, refused unless it is byte for byte the rendered one and resolves to the
    rendered fleet digest under these hosts."""
    entry = plan["fleets"][name]
    sha = file_sha256(fleet_path)
    if entry["fleet_sha256"] is None or sha != entry["fleet_sha256"]:
        raise ConfigError(f"{fleet_path} (sha256 {sha}) is not the {name} fleet the plan was rendered with "
                          f"({entry['fleet_sha256']})")
    fleet = load_fleet(fleet_path, hosts_path)
    if fleet_digest(fleet) != entry["fleet_digest"]:
        raise ConfigError(f"the {name} fleet under {hosts_path} has digest {fleet_digest(fleet)}, the plan "
                          f"{entry['fleet_digest']}")
    return fleet


def lease_prompts(plan: dict, name: str, data_root: Path) -> list["Prompt"]:
    """The plan's dataset, in file order, refused unless its bytes and row count are the rendered ones."""
    entry = plan["datasets"][name]
    path = Path(data_root) / entry["path"]
    sha = file_sha256(path)
    if sha != entry["sha256"]:
        raise ConfigError(f"{path} has sha256 {sha}, the plan expects {entry['sha256']}")
    rows = read_rows(path)
    if len(rows) != entry["rows"]:
        raise ConfigError(f"{path} has {len(rows)} rows, the plan expects {entry['rows']}")
    return [Prompt(pid, text) for pid, text in rows]


class NotIdle(RuntimeError):
    pass


class CacheNotReset(NotIdle):
    """An engine's prefix cache could not be reset, or the reset could not be proven."""


def prefix_cache_counters(instance_id: str, text: str) -> tuple[int, int]:
    """(prompt tokens queried, of them found cached) from one engine's /metrics, summed over its engine series."""
    got = {name: [] for name in PREFIX_COUNTERS}
    for line in text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in got:
            got[name].append(float(line.rsplit(" ", 1)[1]))
    missing = [n for n, v in got.items() if not v]
    if missing:
        raise CacheNotReset(f"{instance_id}: /metrics has no {missing}: the engine does not report its prefix cache")
    q, h = (sum(got[n]) for n in PREFIX_COUNTERS)
    if q != int(q) or h != int(h) or not 0 <= h <= q:
        raise CacheNotReset(f"{instance_id}: malformed prefix-cache counters (queried {q}, hit {h})")
    return int(q), int(h)


async def cache_counters(sess: aiohttp.ClientSession, inst: Instance) -> tuple[int, int]:
    return prefix_cache_counters(inst.instance_id, await fetch_metrics(sess, inst, SCRAPE_TIMEOUT_S))


async def reset_prefix_cache(sess: aiohttp.ClientSession, inst: Instance) -> None:
    """vLLM's POST /reset_prefix_cache on one idle engine; anything but HTTP 200 raises."""
    url = f"{inst.base_url}/reset_prefix_cache"
    async with sess.post(url, timeout=aiohttp.ClientTimeout(total=SCRAPE_TIMEOUT_S)) as resp:
        status, body = resp.status, (await resp.text())[:200]
    if status == 404:
        raise CacheNotReset(f"{inst.instance_id}: {url} is not served: start the engine with VLLM_SERVER_DEV_MODE=1")
    if status != 200:
        raise CacheNotReset(f"{inst.instance_id}: HTTP {status} from {url}: {body}")


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    text: str


def session(connections: int) -> aiohttp.ClientSession:
    """One pooled session sized to the requests that will be in flight at once, so none waits for a connection."""
    if connections < 1:
        raise ConfigError("a session needs at least one connection")
    check_fd_limit(connections)
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=connections, limit_per_host=connections))


async def engine_queues(sess: aiohttp.ClientSession, instances: Sequence[Instance], timeout_s: float) -> dict[str, int]:
    """running + waiting per instance, from each engine's /metrics (a malformed or failed scrape raises)."""
    texts = await asyncio.gather(*(fetch_metrics(sess, i, timeout_s) for i in instances))
    out = {}
    for inst, text in zip(instances, texts, strict=True):
        snap = parse_metrics(inst.instance_id, text)
        out[inst.instance_id] = snap.num_running + snap.num_waiting
    return out


async def drain(sess: aiohttp.ClientSession, instances: Sequence[Instance], *, timeout_s: float,
                clock: Callable[[], float] = time.monotonic) -> float:
    """Wait until every engine reports 0 running and 0 waiting; the seconds waited. Raises NotIdle at timeout_s."""
    start = clock()
    while True:
        busy = {i: n for i, n in (await engine_queues(sess, instances, SCRAPE_TIMEOUT_S)).items() if n}
        if not busy:
            return clock() - start
        if clock() - start >= timeout_s:
            raise NotIdle(f"engines still busy after {timeout_s} s (running + waiting): {busy}")
        await asyncio.sleep(POLL_S)


def request(request_id: str, prompt: Prompt, sampling: Sampling, max_tokens: Optional[int] = None) -> GenerationRequest:
    """A generation request with the scope's sampling; max_tokens overrides the cap (the warm-up only)."""
    return GenerationRequest(request_id, prompt.text, sampling.max_tokens if max_tokens is None else max_tokens,
                             sampling.temperature, sampling.repetition_penalty, sampling.frequency_penalty)


def outcome(res: GenerationResult, **fields) -> dict:
    """One request's evidence line: the backend result without its inter-token gaps, plus the caller's fields."""
    d = asdict(res)
    del d["itl_s"]
    return {**fields, **d}


async def warm_up(sess: aiohttp.ClientSession, backend: BackendClient, instances: Sequence[Instance], *,
                  requests_per_instance: int, max_tokens: int, prompt: str, sampling: Sampling,
                  run_id: str) -> list[dict]:
    """Empty every engine's prefix cache and prove it (see the module docstring), then the warm-up requests: a few short
    requests to every engine, one at a time per engine, all engines at once. Any failure raises: an engine that cannot
    answer a warm-up cannot serve a cell. Each line carries its request's prefix-cache counts (cache_queried,
    cache_hit) and its step (prime, first after the reset, control, warm-up)."""
    if requests_per_instance < 2:
        raise ConfigError("the warm-up needs at least 2 requests per instance: the reset's check and its control")

    async def counted(inst: Instance, rid: str, step: str) -> dict:
        q0, h0 = await cache_counters(sess, inst)
        res = await backend.generate(inst, request(rid, Prompt("warmup", prompt), sampling, max_tokens))
        if not res.success:
            raise NotIdle(f"warm-up request {rid} failed on {inst.instance_id}: {res.error}")
        deadline = time.monotonic() + COUNTED_S
        while True:
            q, h = await cache_counters(sess, inst)
            if q - q0 >= res.prompt_tokens or time.monotonic() > deadline:
                break
            await asyncio.sleep(COUNTER_POLL_S)
        if q - q0 != res.prompt_tokens:
            raise CacheNotReset(f"{inst.instance_id}: warm-up request {rid} ({res.prompt_tokens} prompt tokens) moved the "
                                f"prefix-cache queries by {q - q0} within {COUNTED_S} s (the engine must be idle)")
        return outcome(res, purpose="warmup", cache_step=step, cache_queried=q - q0, cache_hit=h - h0)

    async def one(inst: Instance) -> list[dict]:
        base = f"warmup.{run_id}.{inst.instance_id}"
        lines = [await counted(inst, f"{base}.prime", "prime")]
        await reset_prefix_cache(sess, inst)
        for k in range(requests_per_instance):
            line = await counted(inst, f"{base}.{k}", ("after_reset", "control")[k] if k < 2 else "warmup")
            if k == 0 and line["cache_hit"] != 0:
                raise CacheNotReset(f"{inst.instance_id}: the first request after the reset found {line['cache_hit']} "
                                    f"of its {line['cache_queried']} prompt tokens cached: the cache was not emptied")
            if k == 1 and line["cache_hit"] == 0:
                raise CacheNotReset(f"{inst.instance_id}: the same prompt again found no cached token: the counters do "
                                    "not show the prefix cache, so the reset is not proven")
            lines.append(line)
        return lines
    return [line for per in await asyncio.gather(*(one(i) for i in instances)) for line in per]


@dataclass
class LoopResult:
    measured: list                  # one GenerationResult per measured prompt, in order
    fillers: list                   # the fillers that finished
    fillers_ran_out: bool           # every filler was used before the last measured request finished


async def closed_loop(backend: BackendClient, inst: Instance, measured: Sequence[Prompt], concurrency: int,
                      sampling: Sampling, *, request_prefix: str, fillers: Sequence[Prompt]) -> LoopResult:
    """See the module docstring."""
    if concurrency < 1:
        raise ConfigError("concurrency must be at least 1")
    queue = list(enumerate(measured))[::-1]           # popped from the end: measured prompts in order
    spare = list(enumerate(fillers))[::-1]            # each filler at most once
    results: list[Optional[GenerationResult]] = [None] * len(measured)
    finished_fillers: list[GenerationResult] = []
    left = len(measured)
    ran_out = False
    done = asyncio.Event()
    if left == 0:
        done.set()

    async def worker() -> None:
        nonlocal left, ran_out
        while True:
            if queue:
                i, p = queue.pop()
                results[i] = await backend.generate(inst, request(f"{request_prefix}.m{i}", p, sampling))
                left -= 1
                if left == 0:
                    done.set()
            elif left > 0 and spare:
                j, p = spare.pop()
                finished_fillers.append(await backend.generate(inst, request(f"{request_prefix}.f{j}", p, sampling)))
            else:
                if left > 0 and fillers:
                    ran_out = True
                return

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    waiter = asyncio.create_task(done.wait())
    waiting = {waiter, *tasks}
    try:
        while not done.is_set():          # a worker that raises (an unreachable engine) ends the run at once
            finished, waiting = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
            for t in finished:
                if t is not waiter and t.exception() is not None:
                    raise t.exception()
    finally:
        for t in (*tasks, waiter):
            t.cancel()
        await asyncio.gather(*tasks, waiter, return_exceptions=True)
    return LoopResult(list(results), finished_fillers, ran_out)
