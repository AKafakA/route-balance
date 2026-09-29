"""Preflight gate, run before every cell: the fleet and the scheduler are exactly what the cell expects, and idle.

Usage: python -m gates.preflight --scheduler URL --fleet FLEET.json --hosts HOSTS.json
                                 --expect-config-digest D --expect-fleet-digest F
(exit 0 on pass, 1 on fail; one reason per line)

The fleet files
  load (routebalance.config.load_fleet) into the fleet whose digest is F: the engines probed below are the
  deployment the cell expects.
Every engine of the fleet
  GET /health        200
  GET /version       vLLM 0.18.1 (D24: official vLLM, no fork, on every node)
  GET /v1/models     exactly one model, the tier's
  GET /metrics       running + waiting == 0: the engine is drained. The D30 precondition: only the scheduler
                     sends to an engine, so load starts from zero.
The scheduler
  GET /health        200 "ok"
  GET /v1/config     config_digest == D and fleet_digest == F
  GET /v1/manifest   the same two digests, the same fleet in the same order, and code on disk unchanged since
                     the scheduler started
  GET /v1/stats      no failure; active == 0; no decision line buffered (the previous cell's log is on disk);
                     reading the pollers' cache (telemetry_read "poll", as the manifest says): a snapshot of every
                     instance (its age is counted, never a failure: D133); per round or per request: none
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp

from gates.cell import GateError, parse_manifest, parse_stats, run
from routebalance.config import ConfigError, fleet_digest, load_fleet, loads_strict
from routebalance.telemetry import InstanceStateUnavailable, parse_metrics
from routebalance.types import Instance

VLLM_VERSION = "0.18.1"            # D24
PROBE_TIMEOUT_S = 10.0             # per HTTP request of this gate (not an experiment parameter)
ENGINE_PATHS = ("/health", "/version", "/v1/models", "/metrics")
SCHEDULER_PATHS = ("/health", "/v1/config", "/v1/manifest", "/v1/stats")


@dataclass(frozen=True)
class Reply:
    status: Optional[int]          # None: no HTTP reply at all
    body: str
    error: Optional[str]           # the transport error, when status is None


async def fetch(session: aiohttp.ClientSession, url: str) -> Reply:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=PROBE_TIMEOUT_S)) as resp:
            raw = await resp.read()
            return Reply(resp.status, raw.decode("utf-8", errors="replace"), None)
    except Exception as e:         # every failure to probe is a reason to refuse the cell
        return Reply(None, "", f"{type(e).__name__}: {e}")


def _json(what: str, reply: Reply, problems: list[str]) -> Optional[dict]:
    """The reply's JSON object when it is a 200 with a JSON object body; otherwise a reason is recorded and None
    returned (so None always comes with a reason)."""
    if reply.status is None:
        problems.append(f"{what}: no reply ({reply.error})")
        return None
    if reply.status != 200:
        problems.append(f"{what}: HTTP {reply.status}: {reply.body[:200]}")
        return None
    try:
        data = loads_strict(reply.body)
    except ConfigError as e:
        problems.append(f"{what}: {e}")
        return None
    if not isinstance(data, dict):
        problems.append(f"{what}: the reply is not a JSON object: {reply.body[:200]}")
        return None
    return data


def engine_problems(inst: Instance, replies: dict[str, Reply]) -> list[str]:
    where = f"engine {inst.instance_id}"
    problems: list[str] = []
    health = replies["/health"]
    if health.status != 200:
        problems.append(f"{where}: /health is {health.status if health.status is not None else health.error}")
    version = _json(f"{where}: /version", replies["/version"], problems)
    if version is not None:
        got = version["version"] if isinstance(version, dict) and "version" in version else None
        if got != VLLM_VERSION:
            problems.append(f"{where}: runs vLLM {got!r}, the campaign runs {VLLM_VERSION} (D24)")
    models = _json(f"{where}: /v1/models", replies["/v1/models"], problems)
    if models is not None:
        data = models["data"] if isinstance(models, dict) and "data" in models else None
        if not (isinstance(data, list) and all(isinstance(m, dict) and "id" in m for m in data)):
            problems.append(f"{where}: /v1/models has no list of models: {str(models)[:200]}")
        else:
            ids = [m["id"] for m in data]
            if ids != [inst.model]:
                roots = [m["root"] if "root" in m else None for m in data]
                problems.append(f"{where}: serves {ids} (roots {roots}), the tier {inst.tier.tier_id} is {inst.model!r}")
    metrics = replies["/metrics"]
    if metrics.status != 200:
        problems.append(f"{where}: /metrics is {metrics.status if metrics.status is not None else metrics.error}")
    else:
        try:
            snap = parse_metrics(inst.instance_id, metrics.body)
        except InstanceStateUnavailable as e:
            problems.append(f"{where}: {e}")
        else:
            if snap.num_running + snap.num_waiting != 0:
                problems.append(f"{where}: not drained ({snap.num_running} running, {snap.num_waiting} waiting); "
                                "only the scheduler may send to an engine (D30)")
    return problems


def scheduler_problems(replies: dict[str, Reply], fleet: list[Instance], expect_config: str,
                       expect_fleet: str) -> list[str]:
    problems: list[str] = []
    health = replies["/health"]
    if health.status != 200 or health.body != "ok":
        problems.append(f"scheduler: /health is {health.status if health.status is not None else health.error} "
                        f"{health.body[:200]!r}")
    cfg = _json("scheduler: /v1/config", replies["/v1/config"], problems)
    if cfg is not None:
        for key, want in (("config_digest", expect_config), ("fleet_digest", expect_fleet)):
            got = cfg[key] if isinstance(cfg, dict) and key in cfg else None
            if got != want:
                problems.append(f"scheduler: /v1/config {key} is {got}, the cell expects {want}")
    manifest = None
    raw = _json("scheduler: /v1/manifest", replies["/v1/manifest"], problems)
    if raw is not None:
        try:
            manifest = parse_manifest(raw, "scheduler: /v1/manifest")
        except GateError as e:
            problems.append(str(e))
    if manifest is not None:
        if (manifest.config_digest, manifest.fleet_digest) != (expect_config, expect_fleet):
            problems.append(f"scheduler: the manifest is for config {manifest.config_digest} on fleet "
                            f"{manifest.fleet_digest}, the cell expects {expect_config} on {expect_fleet}")
        for change in manifest.code.changed():
            problems.append(f"scheduler: its code changed on disk after it started ({change}); restart it")
        if [f.instance_id for f in manifest.fleet] != [i.instance_id for i in fleet]:
            problems.append(f"scheduler: executes fleet {[f.instance_id for f in manifest.fleet]}, "
                            f"the fleet files list {[i.instance_id for i in fleet]}")
    raw = _json("scheduler: /v1/stats", replies["/v1/stats"], problems)
    if raw is not None:
        try:
            stats = parse_stats(raw, "scheduler: /v1/stats")
        except GateError as e:
            problems.append(str(e))
            return problems
        if stats.failure is not None:
            problems.append(f"scheduler: has failed: {stats.failure}")
        if stats.active != 0:
            problems.append(f"scheduler: {stats.active} requests are in service")
        if stats.logged.decisions_buffered != 0:
            problems.append(f"scheduler: {stats.logged.decisions_buffered} decision lines are not yet on disk")
        ages = stats.telemetry_age_ms
        if manifest is None:
            problems.append("scheduler: telemetry cannot be checked without the manifest (its read mode)")
        elif stats.telemetry_read != manifest.system.telemetry_read:
            problems.append(f"scheduler: /v1/stats reads telemetry '{stats.telemetry_read}', "
                            f"the manifest's {manifest.system.mode} system '{manifest.system.telemetry_read}'")
        elif stats.telemetry_read != "poll":
            pass       # fetched per round (D53) or per request (D55), a miss failing the round or the request;
                       # the engines themselves were checked directly above
        elif set(ages) != {i.instance_id for i in fleet}:
            problems.append(f"scheduler: the pollers hold no snapshot for every instance: {ages}")
        # D133 (owner): a snapshot's age is counted (every decision logs it), never gated
    return problems


async def preflight(scheduler_url: str, fleet: list[Instance], expect_config: str, expect_fleet: str) -> list[str]:
    problems: list[str] = []
    local = fleet_digest(fleet)
    if local != expect_fleet:
        problems.append(f"the fleet files describe fleet {local}, the cell expects {expect_fleet}")
    base = scheduler_url.rstrip("/")
    connector = aiohttp.TCPConnector(limit=4 * (len(fleet) + 1), force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def probe(root: str, paths: tuple[str, ...]) -> dict[str, Reply]:
            replies = await asyncio.gather(*(fetch(session, root + path) for path in paths))
            return dict(zip(paths, replies, strict=True))
        engines = await asyncio.gather(*(probe(inst.base_url, ENGINE_PATHS) for inst in fleet))
        sched = await probe(base, SCHEDULER_PATHS)
    for inst, replies in zip(fleet, engines, strict=True):
        problems += engine_problems(inst, replies)
    problems += scheduler_problems(sched, fleet, expect_config, expect_fleet)
    return problems


def _check(a: argparse.Namespace) -> tuple[list[str], str]:
    fleet = load_fleet(a.fleet, a.hosts)
    problems = asyncio.run(preflight(a.scheduler, fleet, a.expect_config_digest, a.expect_fleet_digest))
    return problems, (f"{len(fleet)} engines drained on vLLM {VLLM_VERSION}, each serving its tier's model; "
                      f"scheduler idle on config {a.expect_config_digest[:12]} and fleet {a.expect_fleet_digest[:12]}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Check the engines and the scheduler before a cell.")
    ap.add_argument("--scheduler", required=True, help="the scheduler's base URL")
    ap.add_argument("--fleet", type=Path, required=True)
    ap.add_argument("--hosts", type=Path, required=True)
    ap.add_argument("--expect-config-digest", required=True)
    ap.add_argument("--expect-fleet-digest", required=True)
    return run("preflight", lambda: _check(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
