"""The scheduler as the runner sees it: its HTTP API, the run's window cut out of its logs, its code and launch
values, and its restart as a fresh process on its node."""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import aiohttp

from bench.schema import read_json
from routebalance.config import ConfigError, config_digest, fleet_digest, load_fleet, load_scheduler, loads_strict
from routebalance.types import Instance
from runner.store import write_json
from tools import direct

REPO = Path(__file__).resolve().parents[1]
SETTLE_S = 3.0          # how long the logs may take to reach the disk (the server flushes every 1 s)
POLL_S = 0.1
HEALTHY_S = 600         # how long a restarted scheduler may take to load its configuration (scripts/node_ctl.sh)


async def get(url: str, path: str) -> dict:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
        async with s.get(f"{url.rstrip('/')}{path}") as r:
            text = await r.text()
            if r.status != 200:
                raise ConfigError(f"GET {path}: HTTP {r.status}: {text[:500]}")
    data = loads_strict(text)
    if not isinstance(data, dict):
        raise ConfigError(f"GET {path} did not return an object")
    return data


async def post(url: str, path: str, body: bytes) -> tuple[int, str]:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        async with s.post(f"{url.rstrip('/')}{path}", data=body, headers={"Content-Type": "application/json"}) as r:
            return r.status, await r.text()


def _logged(stats: dict) -> dict:
    logged = stats["logged"]
    for key in ("decisions", "decisions_buffered"):
        if type(logged[key]) is not int:
            raise ConfigError(f"/v1/stats logged.{key} is {logged[key]!r}, not a count")
    return logged


# ---------------------------------------------------------------- before the run

async def drain(url: str, instances: list[Instance], bound_s: float) -> float:
    """Wait for the scheduler to hold no request and every engine to be idle, within the plan's drain bound."""
    start = time.monotonic()
    while (await get(url, "/v1/stats"))["active"] != 0:
        if time.monotonic() - start > bound_s:
            raise direct.NotIdle(f"the scheduler still serves requests after {bound_s} s")
        await asyncio.sleep(POLL_S)
    async with direct.session(2 * len(instances)) as sess:
        return await direct.drain(sess, instances, timeout_s=max(bound_s - (time.monotonic() - start), 0.0))


async def require_fresh(url: str) -> None:
    """A run starts only on a fresh scheduler process, one that has logged no decision; checked before the run so
    that a scheduler not restarted costs no run (the verdict checks it again)."""
    logged = (await get(url, "/v1/stats"))["logged"]
    seen = logged["decisions"] + logged["decisions_buffered"]     # flushed or not yet
    if seen != 0:
        raise ConfigError(f"the scheduler has logged {seen} decisions: not a fresh process (the rows runner "
                          "restarts it before every cell with --fresh-scheduler)")


async def post_config(url: str, attempt: Path, config_digest: str, fleet_digest: str) -> None:
    """POST the cell's configuration (every cell: it also resets the dispatchers' state) and check what it runs."""
    status, text = await post(url, "/v1/config", (attempt / "config.json").read_bytes())
    (attempt / "post_config.json").write_text(text)
    if status != 200:
        raise ConfigError(f"POST /v1/config: HTTP {status}: {text[:500]}")
    reply = loads_strict(text)
    got = (reply["config_digest"], reply["fleet_digest"])
    if got != (config_digest, fleet_digest):
        raise ConfigError(f"the scheduler now runs config {got[0]} on fleet {got[1]}; the cell needs "
                          f"{config_digest} on {fleet_digest}")


# ---------------------------------------------------------------- the run's window of the logs

def _complete_lines(path: Path, start: int) -> tuple[int, int]:
    """(complete lines after byte `start`, byte offset after the last of them)."""
    if not path.exists():
        return 0, start
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read()
    end = data.rfind(b"\n") + 1
    return data[:end].count(b"\n"), start + end


def _latency_log(log_dir: Path, digest: str) -> Path:
    return Path(log_dir) / f"latency.{digest[:16]}.jsonl"


async def _settled(url: str, check: Optional[Callable[[dict], bool]] = None,
                   why: Optional[Callable[[dict], str]] = None) -> dict:
    """/v1/stats once nothing is buffered and nothing is in service (and, with `check`, the files agree)."""
    deadline = time.monotonic() + SETTLE_S
    while True:
        stats = await get(url, "/v1/stats")
        logged = _logged(stats)
        if logged["decisions_buffered"] == 0 and stats["active"] == 0 and (check is None or check(stats)):
            return stats
        if time.monotonic() > deadline:
            raise ConfigError(f"the scheduler's logs did not settle within {SETTLE_S} s: active {stats['active']}, "
                              f"buffered {logged['decisions_buffered']}" + ("" if why is None else f"; {why(stats)}"))
        await asyncio.sleep(POLL_S)


async def snapshot_before(url: str, attempt: Path, log_dir: Path, digest: str, is_sweep: bool) -> None:
    """stats_before.json, manifest_before.json and window.json: where the run's window opens in each log."""
    stats = await _settled(url)
    manifest = await get(url, "/v1/manifest")
    if manifest["config_digest"] != digest:
        raise ConfigError(f"the manifest reports config {manifest['config_digest']}, the cell needs {digest}")
    n_all, start = _complete_lines(Path(log_dir) / "decisions.jsonl", 0)
    if n_all != stats["logged"]["decisions"]:     # the scheduler starts on an empty log directory (node_ctl.sh)
        raise ConfigError(f"{log_dir}/decisions.jsonl holds {n_all} lines, the scheduler has logged "
                          f"{stats['logged']['decisions']}: --scheduler-log-dir is not the running scheduler's")
    window = {"decisions": {"path": str(Path(log_dir) / "decisions.jsonl"), "start": start,
                            "logged_start": stats["logged"]["decisions"]}}
    if is_sweep:
        if type(stats["logged"]["latency"]) is not int:
            raise ConfigError("the sweep configuration is live but /v1/stats reports no latency log")
        lat = _latency_log(log_dir, digest)
        window["latency"] = {"path": str(lat), "start": _complete_lines(lat, 0)[1],
                             "logged_start": stats["logged"]["latency"]}
    for name, data in (("stats_before.json", stats), ("manifest_before.json", manifest), ("window.json", window)):
        write_json(attempt / name, data)


async def snapshot_after(url: str, attempt: Path, digest: str) -> None:
    """stats_after.json and manifest_after.json once the logs hold every line of the run; the run's lines cut out
    of the decision log (and the latency sweep's), refused if a line of another configuration is among them."""
    window = read_json(attempt / "window.json")
    dec = window["decisions"]
    path = Path(dec["path"])

    def decisions_on_disk(stats) -> bool:
        return _complete_lines(path, dec["start"])[0] == stats["logged"]["decisions"] - dec["logged_start"]

    def decisions_why(stats) -> str:
        return (f"{_complete_lines(path, dec['start'])[0]} lines on disk since the window opened, "
                f"{stats['logged']['decisions'] - dec['logged_start']} logged")
    stats = await _settled(url, decisions_on_disk, decisions_why)
    n, dec["end"] = _complete_lines(path, dec["start"])            # the window closes here, once
    if n != stats["logged"]["decisions"] - dec["logged_start"]:
        raise ConfigError("decision lines arrived after the logs settled: foreign traffic")
    lines = _slice(path, dec["start"], dec["end"])
    wrong = sorted({ln["config_digest"] for ln in lines} - {digest})
    if wrong:
        raise ConfigError(f"the cell's window holds decision lines of other configurations {wrong}: foreign traffic")
    _copy_bytes(attempt / "decisions.jsonl", path, dec["start"], dec["end"])
    if "latency" in window:
        lat = window["latency"]
        lpath = Path(lat["path"])
        assigned = sum(1 for ln in lines if "instance_id" in ln)

        def latency_on_disk(s) -> bool:
            on_disk = _complete_lines(lpath, lat["start"])[0]
            return on_disk == s["logged"]["latency"] - lat["logged_start"] == assigned

        def latency_why(s) -> str:
            return (f"latency lines on disk {_complete_lines(lpath, lat['start'])[0]}, logged "
                    f"{s['logged']['latency'] - lat['logged_start']}, assigned requests {assigned}")
        stats = await _settled(url, latency_on_disk, latency_why)
        n, lat["end"] = _complete_lines(lpath, lat["start"])
        if n != assigned:
            raise ConfigError("latency lines arrived after the logs settled: foreign traffic")
        _copy_bytes(attempt / "latency.jsonl", lpath, lat["start"], lat["end"])
    manifest = await get(url, "/v1/manifest")
    for name, data in (("stats_after.json", stats), ("manifest_after.json", manifest)):
        write_json(attempt / name, data)
    (attempt / "window.json").write_text(json.dumps(window, indent=1, sort_keys=True) + "\n")


def _slice(path: Path, start: int, end: int) -> list[dict]:
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    return [loads_strict(line) for line in data.decode("utf-8").split("\n") if line]


def _copy_bytes(dest: Path, src: Path, start: int, end: int) -> None:
    with open(src, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    if dest.exists():
        raise ConfigError(f"{dest} exists")
    dest.write_bytes(data)


# ---------------------------------------------------------------- the scheduler's code and launch values

def code_changes(manifest: dict) -> list[str]:
    """The code digests that changed on disk since the scheduler started: it must be restarted."""
    code = manifest["code"]
    return sorted(k for k in code if f"{k}_now" in code and code[k] != code[f"{k}_now"])


def launch_problems(plan: dict, manifest: dict) -> list[str]:
    """The scheduler's launch values against the plan's."""
    server, launch = plan["server"], manifest["launch"]
    if server is None:
        return ["the plan's server launch values are not set"]
    return [f"{k}: the scheduler runs {launch[k]!r}, the plan says {v!r}" for k, v in server.items() if launch[k] != v]


def current(url: str, restart_cmd: Optional[str], say: Callable[[str], None]) -> bool:
    """The scheduler runs the code on disk; with `restart_cmd` a stale one is restarted (the command must return
    once it is launched) and checked again."""
    def changed() -> Optional[list[str]]:
        try:
            return code_changes(asyncio.run(get(url, "/v1/manifest")))
        except Exception as e:      # any failure to read it: the code is not shown current
            say(f"the scheduler's manifest could not be read: {type(e).__name__}: {e}")
            return None
    stale = changed()
    if stale == []:
        return True
    if stale:
        say(f"the scheduler's code changed on disk since it started: {stale}")
    if not restart_cmd:
        say("the scheduler runs stale code; restart it (or pass --restart)")
        return False
    say(f"restarting the scheduler: {restart_cmd}")
    if subprocess.run(["bash", "-c", restart_cmd]).returncode != 0:
        say("the restart command failed")
        return False
    for _ in range(120):
        if _healthy(url):
            break
        time.sleep(2)
    stale = changed()
    if stale:
        say(f"the scheduler's code changed on disk since it started: {stale}")
    return stale == []


def _healthy(url: str) -> bool:
    async def probe() -> bool:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
            async with s.get(f"{url.rstrip('/')}/health") as r:
                return r.status == 200
    try:
        return asyncio.run(probe())
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        return False


# ---------------------------------------------------------------- a fresh scheduler process

def _ctl(*args) -> str:
    """scripts/node_ctl.sh on this node (streamed after node_layout.sh, as deploy.sh runs it on a node): its output.
    A failure raises with what it printed."""
    script = (REPO / "scripts" / "node_layout.sh").read_text() + (REPO / "scripts" / "node_ctl.sh").read_text()
    r = subprocess.run(["bash", "-s", "--", *map(str, args)], input=script, capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        raise ConfigError(f"node_ctl.sh {args[0]} failed (exit {r.returncode}): {(r.stdout + r.stderr).strip()[-2000:]}")
    return r.stdout


def restart(run_dir: Path, log_dir: Path, say: Callable[[str], None]) -> None:
    """Stop the scheduler on this node and start it again as a fresh process from RUN_DIR, the run directory
    deploy.sh scheduler made here (fleet.json, hosts.json, scheduler.json, launch.json, scheduler.cmd): the same
    launch values, the same artifact root, this tree, and LOG_DIR, which must be new. Returns once it is healthy
    and serves the digests of RUN_DIR's configuration and fleet."""
    run_dir, log_dir = Path(run_dir), Path(log_dir)
    for f in ("fleet.json", "hosts.json", "scheduler.json", "launch.json", "scheduler.cmd"):
        if not (run_dir / f).is_file():
            raise ConfigError(f"{run_dir} has no {f} (not a deploy.sh scheduler run directory)")
    if log_dir.exists():
        raise ConfigError(f"log dir {log_dir} exists; every restart gets a new one")
    launch = read_json(run_dir / "launch.json")
    threads = "none" if launch.get("torch_threads") is None else launch["torch_threads"]
    art = re.search(r"--artifacts ([^ \n]*)", (run_dir / "scheduler.cmd").read_text())
    if art is None or not art.group(1):
        raise ConfigError(f"{run_dir}/scheduler.cmd records no --artifacts")
    want = (f"{config_digest(load_scheduler(run_dir / 'scheduler.json'))} "
            f"{fleet_digest(load_fleet(run_dir / 'fleet.json', run_dir / 'hosts.json'))}")
    _ctl("sched-stop")
    say(_ctl("sched-start", run_dir, REPO, art.group(1), log_dir, launch["host"], launch["port"], threads,
             launch["max_connections"], launch["request_timeout_s"]).strip())
    got = [line.split(" ", 1)[1] for line in _ctl("sched-wait", launch["port"], HEALTHY_S).splitlines()
           if line.startswith("DIGESTS ")]
    if got != [want]:
        raise ConfigError(f"the restarted scheduler serves digests {got}, expected {want!r}")
    say(f"scheduler restarted: log dir {log_dir}, digests {want}")
