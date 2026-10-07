"""The runs on disk under the cells root: attempts, published cells, their verdicts, and the groups' parity records.

  <root>/_attempts/<cell>.<stamp>/<cell>/   a run in progress, or one that wrote no record (kept as evidence)
  <root>/_attempts/<cell>.<stamp>/driver.log  the runner's log of that attempt
  <root>/<cell>/                            a published run, never overwritten
  <root>/_retired/<cell>.<stamp>/           a published run moved aside so that the cell can run again
  <root>/_groups/<group>/parity.<stamp>.{log,json}   every parity gate run over a group's cells
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from analysis.manifest import cell_problems
from bench.dataset import file_sha256
from bench.records import CELL_SUFFIX, cell_dir, cell_path, read_cell, records_in
from bench.schema import read_json
from bench.scope import ExpectedCell, Scope, expected_cells, group_record
from gates.cell import GateError, engine_changes, fleet_uncovered, isolation_problems, open_cell
from routebalance.config import ConfigError, loads_strict
from runner import render as rd

GATES = ("arm", "zero_call")                 # the run gates every cell passes (gates/arm.py, gates/zero_call.py)
GATES_DIR = Path(__file__).resolve().parents[1] / "gates"


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())


def write_json(path: Path, data) -> None:
    """A new JSON file; an existing one is never written over."""
    if path.exists():
        raise ConfigError(f"{path} exists")
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


# ---------------------------------------------------------------- one cell

def prepare(scope: Scope, exp: ExpectedCell, render: Path, root: Path) -> Path:
    """A new attempt directory with the cell's configuration (byte for byte the rendered one) and its group."""
    published = cell_dir(root, exp.cell_id)
    if published.exists():
        raise ConfigError(f"{published} exists: a cell is never overwritten (retire it first to run it again)")
    if unpublished(root, exp.cell_id):
        raise ConfigError(f"{unpublished(root, exp.cell_id)} already hold a record of {exp.cell_id}: publish or "
                          "retire it first")
    config_bytes = (Path(render) / "configs" / f"{exp.config_id}.json").read_bytes()
    loads_strict(config_bytes.decode("utf-8"))                            # a malformed configuration is refused here
    attempt = Path(root) / "_attempts" / f"{exp.cell_id}.{stamp()}" / exp.cell_id
    attempt.mkdir(parents=True)
    (attempt / "config.json").write_bytes(config_bytes)
    write_json(attempt / "group.json", group_record(scope, exp))
    return attempt


def unpublished(root: Path, cell: str) -> list[Path]:
    """Attempts of a cell that wrote a record but were never published (the runner stopped in between)."""
    base = Path(root) / "_attempts"
    if not base.is_dir():
        return []
    return sorted(d / cell for d in base.iterdir() if d.name.startswith(f"{cell}.") and records_in(d / cell))


def record_problems(scope: Scope, exp: ExpectedCell, d: Path) -> list[str]:
    """What makes the directory's record invalid (the benchmark and analysis checks); empty for a valid cell."""
    names = records_in(d)
    if names != [f"{exp.cell_id}{CELL_SUFFIX}"]:
        return [f"the cell directory holds {names}, not exactly its own record"]
    try:
        return cell_problems(read_cell(cell_path(d, exp.cell_id)), exp, scope.bench)
    except ConfigError as e:
        return [f"unreadable record: {str(e)[:500]}"]


def engine_problems(d: Path) -> list[str]:
    """Every engine of the cell's fleet served the whole cell as one process with one launch, from a proven empty
    prefix cache, and served nothing but the warm-up and the cell's own requests."""
    try:
        cell = open_cell(d)
        before = cell.engines("before")
        uncovered = fleet_uncovered(before, cell.manifest("before"))
        changed = engine_changes(before, cell.engines("after"))
    except GateError as e:
        return [f"the engine record is missing or malformed: {str(e)[:300]}"]
    return ([f"the engine record does not cover the cell's fleet: {gap}" for gap in uncovered]
            + ([f"an engine restarted or changed during the cell: {changed}"] if changed else [])
            + isolation_problems(d))


def verdict(scope: Scope, attempt: Path, gates: dict[str, int]) -> dict:
    """gates/verdict.json: valid when the record and the engines show no problem and both run gates passed."""
    exp = rd.expected(scope, attempt.name)
    problems = record_problems(scope, exp, attempt) + engine_problems(attempt)
    out = {"cell_id": attempt.name, "bench_problems": problems, "gates": gates,
           "valid": not problems and all(rc == 0 for rc in gates.values()) and set(gates) == set(GATES)}
    write_json(attempt / "gates" / "verdict.json", out)
    return out


def publish(attempt: Path, root: Path) -> Path:
    target = cell_dir(root, attempt.name)
    if target.exists():
        raise ConfigError(f"{target} exists: a cell is never overwritten")
    if not records_in(attempt):
        raise ConfigError(f"{attempt} has no cell record: it stays an attempt")
    os.rename(attempt, target)
    return target


def retire(root: Path, cell: str) -> Path:
    src = cell_dir(root, cell)
    if not src.is_dir():
        raise ConfigError(f"{src} is not a cell directory")
    dest = Path(root) / "_retired" / f"{cell}.{stamp()}"
    dest.parent.mkdir(exist_ok=True)
    os.rename(src, dest)
    return dest


def _verdict_valid(v: Path) -> bool:
    valid = read_json(v)["valid"]
    if not isinstance(valid, bool):
        raise ConfigError(f"{v}: 'valid' must be a boolean, got {valid!r}")
    return valid


def status(scope: Scope, exp: ExpectedCell, root: Path) -> str:
    """new | done | invalid | not-gated (published without a verdict) | unpublished (an attempt holds a record).
    A valid verdict is not trusted on its own: the record and the isolation evidence are checked again."""
    d = cell_dir(root, exp.cell_id)
    if not d.exists():
        return "unpublished" if unpublished(root, exp.cell_id) else "new"
    v = d / "gates" / "verdict.json"
    if not v.exists():
        return "not-gated"
    if not _verdict_valid(v):
        return "invalid"
    return "invalid" if record_problems(scope, exp, d) or isolation_problems(d) else "done"


def gated_valid(root: Path, exp: ExpectedCell) -> bool:
    """Published with a valid verdict (the verdict only: status() then checks the record and the evidence)."""
    v = cell_dir(root, exp.cell_id) / "gates" / "verdict.json"
    return v.exists() and _verdict_valid(v)


# ---------------------------------------------------------------- a group's parity

def gate_code() -> str:
    """sha256 over the gates' source (names and bytes): a parity record made by other gate code is not reused."""
    h = hashlib.sha256()
    for f in sorted(GATES_DIR.glob("*.py")):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def group_ready(scope: Scope, plan: dict, root: Path, group: str) -> list[str]:
    """The directories of a group's done cells, once every cell the scope does not account for is done and the
    group is not pending in this render (an accounted cell that ran and is done is compared too); else []."""
    cells = [c for c in expected_cells(scope) if c.group == group]
    if not cells:
        raise ConfigError(f"group {group!r} has no cells in the render")
    required = [c for c in cells if c.cell_id not in scope.accounted]
    if group in plan["pending_groups"] or not all(gated_valid(root, c) for c in required):
        return []                         # before reading any record: this runs after every cell
    states = {c.cell_id: status(scope, c, root) for c in cells}
    if any(states[c.cell_id] != "done" for c in required):
        return []
    return [str(cell_dir(root, c.cell_id)) for c in cells if states[c.cell_id] == "done"]


def _runs(dirs: list[str]) -> list[list[str]]:
    """[directory, sha256 of its record] of each cell: a parity record covers these exact runs (a retired and rerun
    cell has the same directory and another record)."""
    out = []
    for d in dirs:
        (name,) = records_in(Path(d))
        out.append([d, file_sha256(Path(d) / name)])
    return out


def parity_records(root: Path, group: str) -> list[Path]:
    d = Path(root) / "_groups" / group
    return sorted(d.glob("parity.*.json")) if d.is_dir() else []


def parity(scope: Scope, plan: dict, root: Path, group: str) -> Optional[int]:
    """The parity gate over a ready group, unless its latest record already covers exactly these runs under the
    same gate code (a fixed gate runs again). Every run of the gate is kept, none overwritten. The gate's exit code,
    or None while the group is not ready."""
    dirs = group_ready(scope, plan, root, group)
    if not dirs:
        return None
    runs, code = _runs(dirs), gate_code()
    records = parity_records(root, group)
    if records:
        last = read_json(records[-1])
        if last["cells"] == runs and last["gate_code"] == code:
            return last["rc"]
    when = stamp()
    out = Path(root) / "_groups" / group
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"parity.{when}.log"
    with open(log, "x") as f:                            # "x": never an existing record
        rc = subprocess.run([sys.executable, "-m", "gates.parity", "--cells", *dirs], stdout=f,
                            stderr=subprocess.STDOUT).returncode
    write_json(out / f"parity.{when}.json", {"group": group, "rc": rc, "cells": runs, "gate_code": code,
                                              "log": log.name})
    return rc


# ---------------------------------------------------------------- rows

def cells_to_run(scope: Scope, plan: dict, rows: list[str], root: Path) -> tuple[list[str], list[str]]:
    """(the cells to run, in the plan's order; a report of the rows' other cells). A row with a pending block is
    refused."""
    cells = rd.requested(scope, plan, rows)
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
    pending = [f"{r}/{block}: {reason}" for r in rows for block, reason in known[r]["pending"].items()]
    if pending:
        raise ConfigError("requested rows have blocks pending in this render (render them first):\n  "
                          + "\n  ".join(pending))
    for r in rows:
        for block, why in known[r]["disabled_blocks"].items():
            report.append(f"DISABLED   {r}/{block}: {why}")
    return run, report


def rows_status(scope: Scope, plan: dict, rows: list[str], root: Path) -> tuple[list[str], list[str]]:
    """(problems, the published cells) of the requested rows: every rendered cell the scope does not account for
    must be done, and every group of theirs that is ready must have passed parity over exactly its current cells."""
    cells = rd.requested(scope, plan, rows)
    problems = [f"{status(scope, c, root)}: {c.cell_id}" for c in cells
                if c.cell_id not in scope.accounted and status(scope, c, root) != "done"]
    for group in sorted({c.group for c in cells}):
        dirs = group_ready(scope, plan, root, group)
        if not dirs:
            continue
        records = parity_records(root, group)
        last = read_json(records[-1]) if records else None
        if last is None or last["cells"] != _runs(dirs) or last["gate_code"] != gate_code():
            problems.append(f"group {group} is complete but parity has not run over its current cells "
                            "with the current gate code")
        elif last["rc"] != 0:
            problems.append(f"group {group} failed its parity gate (see {records[-1].parent / last['log']})")
    published = [c.cell_id for c in cells if cell_dir(root, c.cell_id).is_dir()]
    return problems, published
