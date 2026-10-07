"""One cell, end to end and fail-loud: the steps of runner/__init__.py in order.

Exit: 0 the cell is valid and passed its gates; 1 it ran and is invalid, a gate failed, or its group failed parity
(published, kept); 2 it could not run (no record: the attempt stays under _attempts); 3 the second location refused
or failed.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TextIO

from bench.records import Sampling
from bench.schema import read_json
from routebalance.backend import BackendClient
from routebalance.config import loads_strict
from runner import mirror, render as rd, scheduler as sched, store
from tools import direct, engines

VALID, INVALID, NOT_RUN, MIRROR_FAILED = 0, 1, 2, 3
REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Setup:
    """Where cells run (python -m runner cell and rows take these)."""
    render: Path                # a render: scope.json, plan.json, configs/
    scheduler: str              # the scheduler's URL
    scheduler_log_dir: Path     # the running scheduler's --log-dir (decisions.jsonl, latency.<digest>.jsonl)
    fleet: Path
    hosts: Path
    engines_record: Path        # the --record file of the deploy.sh engines run that launched the fleet
    data_root: Path
    cells_root: Path            # the first location (this node)
    mirror: str                 # the second location (rsync syntax: a local path or host:path), which must exist


def ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Log:
    """The runner's lines for one cell, to the terminal and, once the attempt exists, to its driver.log."""

    def __init__(self, cell: str):
        self.cell = cell
        self.file: Optional[TextIO] = None

    def open(self, path: Path) -> None:
        self.file = open(path, "a", buffering=1)

    def raw(self, line: str) -> None:
        print(line, flush=True)
        if self.file is not None:
            self.file.write(line + "\n")

    def __call__(self, msg: str) -> None:
        self.raw(f"{ts()} [runner] {self.cell}: {msg}")

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


class NotRun(Exception):
    """A step before the record exists failed: the cell did not run."""


def node_env() -> dict[str, str]:
    """The clean environment the benchmark client starts in, as the scheduler and the engines do
    (scripts/node_ctl.sh): the node layout's PATH and Hub home, offline."""
    out = subprocess.run(["bash", "-c", 'source scripts/node_layout.sh && printf "%s\\n%s\\n" "$RB_PATH" "$RB_HF_HOME"'],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout.split("\n")
    return {"HOME": str(Path.home()), "PATH": out[0], "LANG": "C.UTF-8", "HF_HOME": out[1], "HF_HUB_OFFLINE": "1"}


def _status(returncode: int) -> int:
    """A process's exit status as a shell reports it: 128 + the signal for a process a signal ended."""
    return 128 - returncode if returncode < 0 else returncode


def _relay(cmd: list[str], log: Log, env: Optional[dict] = None) -> int:
    """Run a command from the repository root, its output line by line into the log; 127 if it cannot start."""
    try:
        with subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True) as p:
            for line in p.stdout:
                log.raw(line.rstrip("\n"))
    except OSError as e:
        log(f"{cmd[0]} could not start: {e}")
        return 127
    return _status(p.returncode)


def _to_file(cmd: list[str], path: Path) -> int:
    """Run a command from the repository root, its output into a file: its exit status, 1 if the file cannot be
    written and 127 if the command cannot start (as a shell reports them)."""
    try:
        f = open(path, "w")
    except OSError:
        return 1
    with f:
        try:
            return _status(subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode)
        except OSError:
            return 127


def _step(log: Log, what: str, fn: Callable[[], object]):
    """A step the cell cannot run without: its failure stops the cell (exit 2)."""
    try:
        return fn()
    except Exception as e:          # every failure is reported with its cause, never retried or skipped
        log(f"{what}: {type(e).__name__}: {e}")
        raise NotRun(what) from e


def _stop(log: Log, why: str):
    log(why)
    raise NotRun(why)


def run(s: Setup, cell: str) -> int:
    log = Log(cell)
    try:
        return _run(s, cell, log)
    except NotRun as e:
        log(f"stopped: {e} (the cell did not run; the attempt, if any, stays under _attempts)")
        return NOT_RUN
    finally:
        log.close()


def _run(s: Setup, cell: str, log: Log) -> int:
    py = sys.executable
    # ---- the attempt directory, with the cell's configuration and group
    scope, plan, exp, instances, attempt, fleet_digest, config = _step(log, "prepare refused", lambda: _prepare(s, cell))
    try:
        log.open(attempt.parent / "driver.log")      # beside the attempt: it stays under _attempts
    except OSError as e:                             # the run goes on, its lines on the terminal only
        log(f"the runner's log could not be opened: {e}")
    url, digest = s.scheduler, exp.config.config_digest
    log(f"config {exp.config_id} ({digest}), group {exp.group}, attempt {attempt}")

    # ---- 1. drain; a fresh scheduler; POST the configuration; preflight
    waited = _step(log, "drain", lambda: asyncio.run(sched.drain(url, instances, plan["drain"]["timeout_s"])))
    log(f"engines idle after {waited:.1f} s")
    _step(log, "the scheduler is not a fresh process (python -m runner rows --fresh-scheduler restarts it)",
          lambda: asyncio.run(sched.require_fresh(url)))
    _step(log, "POST /v1/config", lambda: asyncio.run(sched.post_config(url, attempt, digest, fleet_digest)))
    rc = _to_file([py, "-m", "gates.preflight", "--scheduler", url, "--fleet", str(s.fleet), "--hosts", str(s.hosts),
                   "--expect-config-digest", digest, "--expect-fleet-digest", fleet_digest],
                  attempt / "gates" / "preflight.log")
    if rc != 0:
        for line in (attempt / "gates" / "preflight.log").read_text().splitlines():
            log.raw(line)
        _stop(log, "preflight failed")

    # ---- 2. the engines as they report themselves, with their serving counters, BEFORE the warm-up, so that the
    #         traffic check covers the reset, its proof and the whole run; then the warm-up straight to the engines
    #         (the proven prefix-cache reset), then drain
    _step(log, "the engines could not be read", lambda: asyncio.run(
        _engines(instances, s.engines_record, attempt / "engines_before.json")))
    _step(log, "warm-up", lambda: asyncio.run(_warm_up(scope, plan, instances, attempt)))

    # ---- 3. stats and manifest before (settled logs); the scheduler's code and launch values
    _step(log, "snapshot before", lambda: asyncio.run(sched.snapshot_before(
        url, attempt, s.scheduler_log_dir, digest, rd.is_sweep(loads_strict(config)))))
    def scheduler_problems() -> list[str]:
        manifest = read_json(attempt / "manifest_before.json")
        return ([f"code changed since start: {k}" for k in sched.code_changes(manifest)]
                + sched.launch_problems(plan, manifest))
    wrong = _step(log, "restart the scheduler (see above)", scheduler_problems)
    for w in wrong:
        log(w)
    if wrong:
        _stop(log, "restart the scheduler (see above)")

    # ---- 4. the benchmark client (writes <cell>.cell.json into the attempt directory): offline, from the pinned
    #         snapshots, in a clean environment like the scheduler and the engines
    bench = _relay([py, "-m", "bench.runner", "--scope", str(Path(s.render) / "scope.json"), "--cell", cell,
                    "--scheduler-url", url, "--data-root", str(s.data_root), "--out-dir", str(attempt)], log,
                   env=node_env())
    if not (attempt / f"{cell}.cell.json").is_file():
        _stop(log, f"the benchmark client wrote no record (exit {bench})")

    # ---- 5. stats and manifest after, and the cell's slice of the logs; the engines again (a restart shows)
    try:
        asyncio.run(sched.snapshot_after(url, attempt, digest))
    except Exception as e:
        log(f"the logs could not be sliced ({type(e).__name__}: {e}); the cell is kept but cannot pass its gates")
    try:
        asyncio.run(_engines(instances, s.engines_record, attempt / "engines_after.json"))
    except Exception as e:
        log(f"the engines could not be read after the cell ({type(e).__name__}: {e}); the cell is kept and invalid")

    # ---- 6. the run gates, the verdict, publish
    gates = {}
    for g in store.GATES:
        extra = ["--config", str(attempt / "config.json")] if g == "arm" else []
        gates[g] = _to_file([py, "-m", f"gates.{g}", "--cell", str(attempt), *extra], attempt / "gates" / f"{g}.log")
        log(f"gate {g}: exit {gates[g]}")
    try:
        v = store.verdict(scope, attempt, gates)
        log(("VALID" if v["valid"] else "INVALID") + f" {v['cell_id']}: bench {v['bench_problems']}; gates {gates}")
        valid = v["valid"]
    except Exception as e:
        log(f"the verdict could not be written: {type(e).__name__}: {e}")
        valid = False
    published = _step(log, "publish", lambda: store.publish(attempt, s.cells_root))
    log(f"published at {published}")

    # ---- 7. the second location: copy, then prove it identical (checksums, nothing extra on the mirror); a mirror
    #         that already holds another run of this cell (a retired one) is never written over
    if not mirror.mirror_cell(s.cells_root, s.mirror, cell, log):
        log("the second location refused or failed (see above)")
        return MIRROR_FAILED
    log(f"mirrored to {s.mirror}/{cell}")

    # ---- 8. parity once the cell completes its group (every run kept under _groups/<group>/), mirrored too
    try:
        prc = store.parity(scope, plan, s.cells_root, exp.group)
        log("parity: not ready" if prc is None else f"parity exit {prc}")
    except Exception as e:
        log(f"the parity gate could not run: {type(e).__name__}: {e}")
        prc = 1
    if not mirror.mirror_bookkeeping(s.cells_root, s.mirror, log, ["_groups"]):
        log("the mirror of _groups failed")
        return MIRROR_FAILED
    if prc not in (None, 0):
        log(f"group {exp.group} failed its parity gate (or the gate could not run)")
        valid = False
    if not valid:
        log("INVALID (kept; see gates/verdict.json)")
        return INVALID
    log("valid")
    return VALID


def _prepare(s: Setup, cell: str):
    scope, plan = rd.load(s.render)
    exp = rd.expected(scope, cell)
    instances = rd.fleet_of(plan, exp.group, s.fleet, s.hosts)
    fleet_digest = plan["fleets"][plan["groups"][exp.group]["fleet"]]["fleet_digest"]
    attempt = store.prepare(scope, exp, s.render, s.cells_root)
    (attempt / "gates").mkdir()
    return scope, plan, exp, instances, attempt, fleet_digest, (attempt / "config.json").read_text()


async def _engines(instances, record: Path, out: Path) -> None:
    """Every engine of the fleet as it reports itself, with its deploy-record launch (tools/engines.py)."""
    store.write_json(out, await engines.engines(instances, record, direct.SCRAPE_TIMEOUT_S))


async def _warm_up(scope, plan: dict, instances, attempt: Path) -> None:
    """The warm-up straight to the engines (never through the scheduler, whose log is the cell's evidence): it
    resets every prefix cache and proves it empty; then a drain."""
    w, sampling = plan["warmup"], Sampling.model_validate(scope.bench.sampling.model_dump())
    async with direct.session(len(instances)) as sess:
        backend = BackendClient(sess, plan["drain"]["timeout_s"])   # the kept pre-run bound
        lines = await direct.warm_up(sess, backend, instances, requests_per_instance=w["requests_per_instance"],
                                     max_tokens=w["max_tokens"], prompt=w["prompt"], sampling=sampling,
                                     run_id=attempt.parent.name)
        waited = await direct.drain(sess, instances, timeout_s=plan["drain"]["timeout_s"])
    store.write_json(attempt / "warmup.json", {"requests": lines, "drain_s": waited})
