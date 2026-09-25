"""The steps of scripts/run_cell.sh and scripts/run_scope.sh, one subcommand each, every one fail-loud.

A cell runs in an attempt directory, <cells root>/_attempts/<cell_id>.<stamp>/<cell_id>/, and is published to
<cells root>/<cell_id>/ (a rename, never over an existing directory) once its record exists, valid or not; an
attempt that produced no record stays under _attempts as evidence and the cell stays missing, so it can simply
be run again. A published cell is never overwritten: to run it again, retire it first (moved under _retired).
The cell directory holds (the layout the gates read):
  <cell_id>.cell.json                 the bench record (bench.runner)
  config.json                         the scheduler document POSTed for the cell, byte for byte the rendered one
  post_config.json                    the scheduler's reply to that POST
  group.json                          {group, reference, intended_differences} for the parity gate
  warmup.json                         the direct warm-up requests
  stats_before.json, stats_after.json, manifest_before.json, manifest_after.json
  decisions.jsonl                     the decision-log lines of this cell's window, all with the cell's digest
  latency.jsonl                       the latency-sweep lines of the window (the D38 sweep configuration only)
  window.json                         the byte offsets of the two logs that bound the window
  engines_before.json, engines_after.json   each engine as it reports itself (start time, KV cache config,
                                      model, context) with its deploy.sh launch command (D48); a change
                                      between the two, a restart, makes the cell invalid
  gates/<gate>.log, gates/verdict.json
Exit codes of every subcommand: 0 done, 1 refused or failed (the message says why).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import aiohttp

from analysis.manifest import cell_problems
from bench.records import CELL_SUFFIX, cell_dir, cell_path, read_cell, records_in
from bench.records import Sampling
from bench.schema import parse, read_json
from bench.scope import ExpectedCell, Scope, expected_cells, group_record
from routebalance.backend import BackendClient
from routebalance.config import ConfigError, loads_strict
from gates.cell import GateError, engine_changes, open_cell
from tools import direct, engines
from tools.campaign import load_plan

SETTLE_S = 3.0          # how long the logs may take to reach the disk (the server flushes every 1 s)
POLL_S = 0.1
GATES = ("arm", "zero_call")


# ---------------------------------------------------------------- the render and the cell

def load(render: Path) -> tuple[Scope, dict]:
    return parse(Scope, read_json(Path(render) / "scope.json"), f"rendered scope {render}"), load_plan(render)


def expected(scope: Scope, cell: str) -> ExpectedCell:
    cells = {c.cell_id: c for c in expected_cells(scope)}
    if cell not in cells:
        raise ConfigError(f"cell {cell} is not in the rendered scope")
    return cells[cell]


def fleet_of(plan: dict, group: str, fleet: Path, hosts: Path):
    return direct.lease_fleet(plan, plan["groups"][group]["fleet"], fleet, hosts)


def sweep(config: dict) -> bool:
    """The D38 latency sweep: the configuration for which the server writes latency.<digest16>.jsonl."""
    return config["mode"] == "pipeline" and config["pending_decode"] == "own_length"


def _write_json(path: Path, data) -> None:
    if path.exists():
        raise ConfigError(f"{path} exists")
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


# ---------------------------------------------------------------- scheduler HTTP

async def _get(url: str, path: str) -> dict:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
        async with s.get(f"{url.rstrip('/')}{path}") as r:
            text = await r.text()
            if r.status != 200:
                raise ConfigError(f"GET {path}: HTTP {r.status}: {text[:500]}")
    data = loads_strict(text)
    if not isinstance(data, dict):
        raise ConfigError(f"GET {path} did not return an object")
    return data


async def _post(url: str, path: str, body: bytes) -> tuple[int, str]:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        async with s.post(f"{url.rstrip('/')}{path}", data=body, headers={"Content-Type": "application/json"}) as r:
            return r.status, await r.text()


def _logged(stats: dict) -> dict:
    logged = stats["logged"]
    for key in ("decisions", "decisions_buffered"):
        if type(logged[key]) is not int:
            raise ConfigError(f"/v1/stats logged.{key} is {logged[key]!r}, not a count")
    return logged


# ---------------------------------------------------------------- steps

def prepare(render: Path, cell: str, root: Path, fleet: Path, hosts: Path) -> dict:
    """Create the attempt directory with config.json and group.json; the values run_cell.sh needs."""
    scope, plan = load(render)
    exp = expected(scope, cell)
    published = cell_dir(root, cell)
    if published.exists():
        raise ConfigError(f"{published} exists: a cell is never overwritten (retire it first to run it again)")
    if unpublished(root, cell):
        raise ConfigError(f"{unpublished(root, cell)} already hold a record of {cell}: publish or retire it first")
    instances = fleet_of(plan, exp.group, fleet, hosts)
    config_bytes = (Path(render) / "configs" / f"{exp.config_id}.json").read_bytes()
    config = loads_strict(config_bytes.decode("utf-8"))
    attempt = Path(root) / "_attempts" / f"{cell}.{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}" / cell
    attempt.mkdir(parents=True)
    (attempt / "config.json").write_bytes(config_bytes)
    _write_json(attempt / "group.json", group_record(scope, exp))
    entry = plan["fleets"][plan["groups"][exp.group]["fleet"]]
    return {"ATTEMPT": str(attempt), "CONFIG_ID": exp.config_id, "CONFIG_DIGEST": exp.config.config_digest,
            "FLEET_DIGEST": entry["fleet_digest"], "GROUP": exp.group, "SWEEP": "1" if sweep(config) else "0",
            "INSTANCES": str(len(instances))}


async def drain(render: Path, cell: str, fleet: Path, hosts: Path, url: str) -> float:
    """Wait for the scheduler to hold no request and every engine to be idle (bounded by the plan's drain)."""
    scope, plan = load(render)
    exp = expected(scope, cell)
    instances = fleet_of(plan, exp.group, fleet, hosts)
    bound = plan["drain"]["timeout_s"]
    start = time.monotonic()
    while (await _get(url, "/v1/stats"))["active"] != 0:
        if time.monotonic() - start > bound:
            raise direct.NotIdle(f"the scheduler still serves requests after {bound} s")
        await asyncio.sleep(POLL_S)
    async with direct.session(2 * len(instances)) as sess:
        return await direct.drain(sess, instances, timeout_s=max(bound - (time.monotonic() - start), 0.0))


async def post_config(url: str, attempt: Path, config_digest: str, fleet_digest: str) -> None:
    status, text = await _post(url, "/v1/config", (attempt / "config.json").read_bytes())
    (attempt / "post_config.json").write_text(text)
    if status != 200:
        raise ConfigError(f"POST /v1/config: HTTP {status}: {text[:500]}")
    reply = loads_strict(text)
    got = (reply["config_digest"], reply["fleet_digest"])
    if got != (config_digest, fleet_digest):
        raise ConfigError(f"the scheduler now runs config {got[0]} on fleet {got[1]}; the cell needs "
                          f"{config_digest} on {fleet_digest}")


async def engines_snapshot(render: Path, cell: str, fleet: Path, hosts: Path, record: Path, out: Path) -> None:
    """Every engine of the cell's fleet as it reports itself, with its deploy-record launch (tools/engines.py)."""
    scope, plan = load(render)
    exp = expected(scope, cell)
    instances = fleet_of(plan, exp.group, fleet, hosts)
    _write_json(out, await engines.engines(instances, record, direct.SCRAPE_TIMEOUT_S))


def engine_problems(d: Path) -> list[str]:
    """D48: every engine served the whole cell as one process with one launch."""
    try:
        cell = open_cell(d)
        changed = engine_changes(cell.engines("before"), cell.engines("after"))
    except GateError as e:
        return [f"the engine record is missing or malformed: {str(e)[:300]}"]
    return [f"an engine restarted or changed during the cell: {changed}"] if changed else []


async def warm_up(render: Path, cell: str, fleet: Path, hosts: Path, attempt: Path) -> None:
    """The warm-up, straight to the engines (never through the scheduler, whose log is the cell's evidence),
    then a drain."""
    scope, plan = load(render)
    exp = expected(scope, cell)
    instances = fleet_of(plan, exp.group, fleet, hosts)
    w, sampling = plan["warmup"], Sampling.model_validate(scope.bench.sampling.model_dump())
    async with direct.session(len(instances)) as sess:
        backend = BackendClient(sess, scope.bench.request_timeout_s)
        lines = await direct.warm_up(backend, instances, requests_per_instance=w["requests_per_instance"],
                                     max_tokens=w["max_tokens"], prompt=w["prompt"], sampling=sampling,
                                     run_id=attempt.parent.name)
        waited = await direct.drain(sess, instances, timeout_s=plan["drain"]["timeout_s"])
    _write_json(attempt / "warmup.json", {"requests": lines, "drain_s": waited})


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


async def _settled(url: str, want: Optional[dict] = None) -> dict:
    """/v1/stats once nothing is buffered and nothing is in service (and, with `want`, the file checks pass)."""
    deadline = time.monotonic() + SETTLE_S
    while True:
        stats = await _get(url, "/v1/stats")
        logged = _logged(stats)
        ok = bool(logged["decisions_buffered"] == 0 and stats["active"] == 0 and (want is None or want["check"](stats)))
        if ok:
            return stats
        if time.monotonic() > deadline:
            raise ConfigError(f"the scheduler's logs did not settle within {SETTLE_S} s: active {stats['active']}, "
                              f"buffered {logged['decisions_buffered']}"
                              + ("" if want is None else f"; {want['why'](stats)}"))
        await asyncio.sleep(POLL_S)


async def snapshot_before(url: str, attempt: Path, log_dir: Path, digest: str, is_sweep: bool) -> None:
    stats = await _settled(url)
    manifest = await _get(url, "/v1/manifest")
    if manifest["config_digest"] != digest:
        raise ConfigError(f"the manifest reports config {manifest['config_digest']}, the cell needs {digest}")
    window = {"decisions": {"path": str(Path(log_dir) / "decisions.jsonl"),
                            "start": _complete_lines(Path(log_dir) / "decisions.jsonl", 0)[1],
                            "logged_start": stats["logged"]["decisions"]}}
    if is_sweep:
        if type(stats["logged"]["latency"]) is not int:
            raise ConfigError("the sweep configuration is live but /v1/stats reports no latency log")
        lat = _latency_log(log_dir, digest)
        window["latency"] = {"path": str(lat), "start": _complete_lines(lat, 0)[1],
                             "logged_start": stats["logged"]["latency"]}
    for name, data in (("stats_before.json", stats), ("manifest_before.json", manifest), ("window.json", window)):
        _write_json(attempt / name, data)


async def snapshot_after(url: str, attempt: Path, digest: str) -> None:
    window = read_json(attempt / "window.json")
    dec = window["decisions"]
    path = Path(dec["path"])

    def decisions_ok(stats) -> bool:
        return _complete_lines(path, dec["start"])[0] == stats["logged"]["decisions"] - dec["logged_start"]

    def decisions_why(stats) -> str:
        return (f"{_complete_lines(path, dec['start'])[0]} lines on disk since the window opened, "
                f"{stats['logged']['decisions'] - dec['logged_start']} logged")
    stats = await _settled(url, {"check": decisions_ok, "why": decisions_why})
    n, dec["end"] = _complete_lines(path, dec["start"])            # the window closes here, once
    if n != stats["logged"]["decisions"] - dec["logged_start"]:
        raise ConfigError("decision lines arrived after the logs settled: foreign traffic")
    lines = _slice(path, dec["start"], dec["end"])
    wrong = sorted({ln["config_digest"] for ln in lines} - {digest})
    if wrong:
        raise ConfigError(f"the cell's window holds decision lines of other configurations {wrong}: foreign traffic")
    _write_lines(attempt / "decisions.jsonl", path, dec["start"], dec["end"])
    if "latency" in window:
        lat = window["latency"]
        lpath = Path(lat["path"])
        assigned = sum(1 for ln in lines if "instance_id" in ln)

        def latency_ok(s) -> bool:
            on_disk = _complete_lines(lpath, lat["start"])[0]
            return on_disk == s["logged"]["latency"] - lat["logged_start"] == assigned

        def latency_why(s) -> str:
            return (f"latency lines on disk {_complete_lines(lpath, lat['start'])[0]}, logged "
                    f"{s['logged']['latency'] - lat['logged_start']}, assigned requests {assigned}")
        stats = await _settled(url, {"check": latency_ok, "why": latency_why})
        n, lat["end"] = _complete_lines(lpath, lat["start"])
        if n != assigned:
            raise ConfigError("latency lines arrived after the logs settled: foreign traffic")
        _write_lines(attempt / "latency.jsonl", lpath, lat["start"], lat["end"])
    manifest = await _get(url, "/v1/manifest")
    for name, data in (("stats_after.json", stats), ("manifest_after.json", manifest)):
        _write_json(attempt / name, data)
    (attempt / "window.json").write_text(json.dumps(window, indent=1, sort_keys=True) + "\n")


def _slice(path: Path, start: int, end: int) -> list[dict]:
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    return [loads_strict(line) for line in data.decode("utf-8").split("\n") if line]


def _write_lines(dest: Path, src: Path, start: int, end: int) -> None:
    with open(src, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    if dest.exists():
        raise ConfigError(f"{dest} exists")
    dest.write_bytes(data)


def check_code(manifest: dict) -> list[str]:
    """Code digests that changed on disk since the scheduler started: it must be restarted."""
    code = manifest["code"]
    return sorted(k for k in code if f"{k}_now" in code and code[k] != code[f"{k}_now"])


def check_launch(plan: dict, manifest: dict) -> list[str]:
    """The scheduler's launch values against the scope's (D44)."""
    server, launch = plan["server"], manifest["launch"]
    if server is None:
        return ["the scope's server launch values are still OWNER-MUST-SET"]
    return [f"{k}: the scheduler runs {launch[k]!r}, the scope says {v!r}" for k, v in server.items() if launch[k] != v]


def verdict(render: Path, attempt: Path, gates: dict[str, int]) -> dict:
    scope, _ = load(render)
    exp = expected(scope, attempt.name)
    problems = record_problems(scope, exp, attempt) + engine_problems(attempt)
    out = {"cell_id": attempt.name, "bench_problems": problems, "gates": gates,
           "valid": not problems and all(rc == 0 for rc in gates.values()) and set(gates) == set(GATES)}
    _write_json(attempt / "gates" / "verdict.json", out)
    return out


def publish(attempt: Path, root: Path) -> Path:
    target = cell_dir(root, attempt.name)
    if target.exists():
        raise ConfigError(f"{target} exists: a cell is never overwritten")
    if not records_in(attempt):
        raise ConfigError(f"{attempt} has no cell record: it stays an attempt")
    os.rename(attempt, target)
    return target


def unpublished(root: Path, cell: str) -> list[Path]:
    """Attempts of a cell that wrote a record but were never published (the driver stopped in between)."""
    base = Path(root) / "_attempts"
    if not base.is_dir():
        return []
    return sorted(d / cell for d in base.iterdir() if d.name.startswith(f"{cell}.") and records_in(d / cell))


def status(scope: Scope, exp: ExpectedCell, root: Path) -> str:
    """new | done | invalid | not-gated (published without a verdict) | unpublished (an attempt holds a record)."""
    d = cell_dir(root, exp.cell_id)
    if not d.exists():
        return "unpublished" if unpublished(root, exp.cell_id) else "new"
    v = d / "gates" / "verdict.json"
    if not v.exists():
        return "not-gated"
    if not read_json(v)["valid"]:
        return "invalid"
    return "invalid" if record_problems(scope, exp, d) else "done"


def record_problems(scope: Scope, exp: ExpectedCell, d: Path) -> list[str]:
    """What makes the directory's record invalid (bench and analysis checks); empty for a valid cell."""
    names = records_in(d)
    if names != [f"{exp.cell_id}{CELL_SUFFIX}"]:
        return [f"the cell directory holds {names}, not exactly its own record"]
    try:
        return cell_problems(read_cell(cell_path(d, exp.cell_id)), exp, scope.bench)
    except ConfigError as e:
        return [f"unreadable record: {str(e)[:500]}"]


def group_ready(render: Path, root: Path, group: str) -> list[str]:
    """The directories of a group's done cells once every cell the scope does not account for is done and none of
    its blocks is still pending in this render (an accounted cell that ran and is done is compared too)."""
    scope, plan = load(render)
    cells = [c for c in expected_cells(scope) if c.group == group]
    states = {c.cell_id: status(scope, c, root) for c in cells}
    if group in plan["pending_groups"] or any(states[c.cell_id] != "done" for c in cells
                                              if c.cell_id not in scope.accounted):
        return []
    return [str(cell_dir(root, c.cell_id)) for c in cells if states[c.cell_id] == "done"]


def _runs(dirs: list[str]) -> list[list[str]]:
    """[directory, sha256 of its record] of each cell: a parity record covers these exact runs (a retired and
    rerun cell has the same directory and another record)."""
    out = []
    for d in dirs:
        (name,) = records_in(Path(d))
        out.append([d, hashlib.sha256((Path(d) / name).read_bytes()).hexdigest()])
    return out


GATES_DIR = Path(__file__).resolve().parents[1] / "gates"


def gate_code() -> str:
    """sha256 over the gates' source (names and bytes): a parity record made by other gate code is not reused."""
    h = hashlib.sha256()
    for f in sorted(GATES_DIR.glob("*.py")):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def _parity_records(root: Path, group: str) -> list[Path]:
    d = Path(root) / "_groups" / group
    return sorted(d.glob("parity.*.json")) if d.is_dir() else []


def parity(render: Path, root: Path, group: str) -> Optional[int]:
    """Run the parity gate over a ready group, unless its latest record already covers exactly these runs (each
    cell's directory and record sha256) under the same gate code (a fixed gate runs again); each run is kept
    (_groups/<group>/parity.<stamp>.{log,json}), none is overwritten. The gate's exit code, or None while the
    group is not ready."""
    dirs = group_ready(render, root, group)
    if not dirs:
        return None
    runs, code = _runs(dirs), gate_code()
    records = _parity_records(root, group)
    if records:
        last = read_json(records[-1])
        if last["cells"] == runs and last.get("gate_code") == code:   # a record without one predates the key
            return last["rc"]
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    out = Path(root) / "_groups" / group
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"parity.{stamp}.log"
    with open(log, "x") as f:                            # "x": never an existing record
        rc = subprocess.run([sys.executable, "-m", "gates.parity", "--cells", *dirs], stdout=f,
                            stderr=subprocess.STDOUT).returncode
    _write_json(out / f"parity.{stamp}.json", {"group": group, "rc": rc, "cells": runs, "gate_code": code,
                                                "log": log.name})
    return rc


def _requested(scope: Scope, plan: dict, rows: list[str]) -> list[ExpectedCell]:
    """The rendered cells of the requested rows, in the plan's order."""
    known = {r["id"]: r for r in plan["rows"]}
    unknown = [r for r in rows if r not in known or known[r]["kind"] != "cells"]
    if unknown:
        raise ConfigError(f"not cell rows of the plan: {unknown}")
    wanted = {c for r in plan["rows"] if r["id"] in rows for c in r["cells"]}
    return [c for c in expected_cells(scope) if c.cell_id in wanted]


def cells_to_run(render: Path, rows: list[str], root: Path, fleet: Path, hosts: Path) -> tuple[list[str], list[str]]:
    """(cells to run in the plan's order, a report of the rows' other cells)."""
    scope, plan = load(render)
    cells = _requested(scope, plan, rows)
    for name in {plan["groups"][c.group]["fleet"] for c in cells}:
        direct.lease_fleet(plan, name, fleet, hosts)         # every requested cell runs on the fleet given
    run, report = [], []
    for c in cells:
        s = status(scope, c, root)
        if s == "new" and c.cell_id in scope.accounted:
            report.append(f"accounted  {c.cell_id}: {scope.accounted[c.cell_id]} (not run)")
        elif s == "new":
            run.append(c.cell_id)
        elif s == "unpublished":
            report.append(f"{s:10s} {c.cell_id}: publish or retire {unpublished(root, c.cell_id)}")
        else:
            report.append(f"{s:10s} {c.cell_id}")
    known = {r["id"]: r for r in plan["rows"]}
    for r in rows:
        for block, reason in known[r]["pending"].items():
            report.append(f"PENDING    {r}/{block}: {reason}")
        for block, why in known[r]["disabled_blocks"].items():
            report.append(f"DISABLED   {r}/{block}: {why}")
    return run, report


def rows_status(render: Path, rows: list[str], root: Path) -> tuple[list[str], list[str]]:
    """(problems, the published cell ids) of the requested rows once run: every rendered cell the scope does not
    account for must be done, and every group of theirs that is ready must have passed parity over exactly its
    current cells. Pending blocks are listed by the plan report."""
    scope, plan = load(render)
    cells = _requested(scope, plan, rows)
    problems = [f"{status(scope, c, root)}: {c.cell_id}" for c in cells
                if c.cell_id not in scope.accounted and status(scope, c, root) != "done"]
    for group in sorted({c.group for c in cells}):
        dirs = group_ready(render, root, group)
        if not dirs:
            continue
        records = _parity_records(root, group)
        last = read_json(records[-1]) if records else None
        if last is None or last["cells"] != _runs(dirs) or last.get("gate_code") != gate_code():
            problems.append(f"group {group} is complete but parity has not run over its current cells "
                            "with the current gate code")
        elif last["rc"] != 0:
            problems.append(f"group {group} failed its parity gate (see {records[-1].parent / last['log']})")
    published = [c.cell_id for c in cells if cell_dir(root, c.cell_id).is_dir()]
    return problems, published


def row_groups(render: Path, rows: list[str]) -> list[str]:
    scope, plan = load(render)
    return sorted({c.group for c in _requested(scope, plan, rows)})


def retire(root: Path, cell: str) -> Path:
    src = cell_dir(root, cell)
    if not src.is_dir():
        raise ConfigError(f"{src} is not a cell directory")
    dest = Path(root) / "_retired" / f"{cell}.{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
    dest.parent.mkdir(exist_ok=True)
    os.rename(src, dest)
    return dest


# ---------------------------------------------------------------- command line

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name, *args):
        p = sub.add_parser(name)
        for a in args:
            p.add_argument(f"--{a}", required=True, type=str)
        return p
    cmd("prepare", "render", "cell", "cells-root", "fleet", "hosts")
    cmd("drain", "render", "cell", "fleet", "hosts", "scheduler")
    cmd("post-config", "scheduler", "attempt", "config-digest", "fleet-digest")
    cmd("warmup", "render", "cell", "fleet", "hosts", "attempt")
    cmd("engines", "render", "cell", "fleet", "hosts", "engines-record", "out")
    cmd("before", "scheduler", "attempt", "log-dir", "config-digest", "sweep")
    cmd("after", "scheduler", "attempt", "config-digest")
    cmd("check-scheduler", "render", "attempt")
    cmd("verdict", "render", "attempt").add_argument("--gate", action="append", default=[], help="NAME=RC")
    cmd("publish", "attempt", "cells-root")
    cmd("parity", "render", "cells-root", "group")
    cmd("groups", "render", "rows")
    cmd("plan", "render", "rows", "cells-root", "fleet", "hosts")
    cmd("status", "render", "rows", "cells-root")
    cmd("published", "render", "rows", "cells-root")
    cmd("code", "scheduler")
    cmd("retire", "cells-root", "cell")
    a = ap.parse_args(argv)
    p = {k: Path(v) for k, v in vars(a).items()
         if k in ("render", "cells_root", "fleet", "hosts", "attempt", "log_dir", "engines_record", "out")}
    if a.cmd == "prepare":
        for k, v in prepare(p["render"], a.cell, p["cells_root"], p["fleet"], p["hosts"]).items():
            print(f"{k}={shlex.quote(v)}")
    elif a.cmd == "drain":
        waited = asyncio.run(drain(p["render"], a.cell, p["fleet"], p["hosts"], a.scheduler))
        print(f"engines idle after {waited:.1f} s")
    elif a.cmd == "post-config":
        asyncio.run(post_config(a.scheduler, p["attempt"], a.config_digest, a.fleet_digest))
    elif a.cmd == "engines":
        asyncio.run(engines_snapshot(p["render"], a.cell, p["fleet"], p["hosts"], p["engines_record"], p["out"]))
    elif a.cmd == "warmup":
        asyncio.run(warm_up(p["render"], a.cell, p["fleet"], p["hosts"], p["attempt"]))
    elif a.cmd == "before":
        asyncio.run(snapshot_before(a.scheduler, p["attempt"], p["log_dir"], a.config_digest, a.sweep == "1"))
    elif a.cmd == "after":
        asyncio.run(snapshot_after(a.scheduler, p["attempt"], a.config_digest))
    elif a.cmd == "check-scheduler":
        _, plan = load(p["render"])
        manifest = read_json(p["attempt"] / "manifest_before.json")
        wrong = [f"code changed since start: {k}" for k in check_code(manifest)] + check_launch(plan, manifest)
        if wrong:
            print("\n".join(wrong), file=sys.stderr)
            return 1
    elif a.cmd == "verdict":
        gates = {}
        for g in a.gate:
            name, _, rc = g.partition("=")
            gates[name] = int(rc)
        v = verdict(p["render"], p["attempt"], gates)
        print(("VALID" if v["valid"] else "INVALID") + f" {v['cell_id']}: bench {v['bench_problems']}; gates {gates}")
        return 0 if v["valid"] else 1
    elif a.cmd == "publish":
        print(publish(p["attempt"], p["cells_root"]))
    elif a.cmd == "parity":
        rc = parity(p["render"], p["cells_root"], a.group)
        print("not ready" if rc is None else f"parity exit {rc}")
        return 0 if rc in (None, 0) else 1
    elif a.cmd == "groups":
        print("\n".join(row_groups(p["render"], [r for r in a.rows.split(",") if r])))
    elif a.cmd == "plan":
        run, report = cells_to_run(p["render"], [r for r in a.rows.split(",") if r], p["cells_root"], p["fleet"],
                                   p["hosts"])
        print("\n".join(report), file=sys.stderr)
        print("\n".join(run))
    elif a.cmd in ("status", "published"):
        problems, published = rows_status(p["render"], [r for r in a.rows.split(",") if r], p["cells_root"])
        if a.cmd == "published":                          # the published cells, for the mirror check
            if published:
                print("\n".join(published))
            return 0
        if problems:
            print("\n".join(problems), file=sys.stderr)
        return 0 if not problems else 1
    elif a.cmd == "code":
        changed = check_code(asyncio.run(_get(a.scheduler, "/v1/manifest")))
        if changed:
            print(f"the scheduler's code changed on disk since it started: {changed}", file=sys.stderr)
            return 1
    elif a.cmd == "retire":
        print(f"{retire(p['cells_root'], a.cell)}; move the second location's copy of {a.cell} aside too before the "
              "cell runs again (run_scope.sh refuses a mirror that holds files the new run does not)")
    return 0



if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (ConfigError, direct.NotIdle, OSError, KeyError) as e:
        print(f"cell_driver: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
