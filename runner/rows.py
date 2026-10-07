"""The cells of some rows of a render, in the plan's order, resuming: a published cell never runs again (a valid
one is done; an invalid or ungated one is reported and left for the owner to account for or retire).

Every requested cell must belong to the fleet given (a topology or model-family change is a redeploy, then a run of
its own rows). With --fresh-scheduler RUN_DIR every cell runs on a fresh scheduler process, restarted from RUN_DIR
(the deploy.sh scheduler run directory on this node) with a new log directory, SCHED_LOG_DIR/<cell>.<stamp>. Before
every cell the scheduler must run the code on disk (its manifest reports both); with --restart CMD a stale one is
restarted by CMD and checked again, without it the rows stop.

A cell that could not run (exit 2) or whose mirror failed (exit 3) stops the rows; an invalid cell (exit 1) counts
toward --max-invalid, and the rows stop once more than that many were seen. Before the first cell and after the last,
the bookkeeping directories are copied to the second location and every published cell of the rows is copied and
proven identical there (a copy that failed earlier is completed; a mirror copy of another run of a cell is never
written over: the rows stop). The rows succeed only when every rendered cell of theirs is done (published, valid,
gated) and every group of theirs whose parity ran has passed it; a row with a pending block is refused before any
run. The manifest of the whole render is printed at the end.
"""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from runner import cell as run_cell, mirror, render as rd, scheduler as sched, store
from runner.cell import INVALID, MIRROR_FAILED, NOT_RUN, VALID, Setup


def say(msg: str) -> None:
    print(f"{run_cell.ts()} [rows] {msg}", flush=True)


def _mirror_all(s: Setup, rows: list[str]) -> bool:
    """The bookkeeping directories, then every published cell of the rows: never over another run, then identical."""
    if not mirror.mirror_bookkeeping(s.cells_root, s.mirror, say):
        return False
    try:
        scope, plan = rd.load(s.render)
        _, published = store.rows_status(scope, plan, rows, s.cells_root)
    except Exception as e:
        say(f"the rows' published cells could not be listed: {type(e).__name__}: {e}")
        return False
    return all(mirror.mirror_cell(s.cells_root, s.mirror, c, say) for c in published)


def run(s: Setup, rows: list[str], max_invalid: int, restart_cmd: Optional[str], fresh_run_dir: Optional[Path]) -> int:
    os.makedirs(s.cells_root, exist_ok=True)
    if not _mirror_all(s, rows):
        say(f"the second location at {s.mirror} does not match {s.cells_root}")
        return MIRROR_FAILED
    try:
        scope, plan = rd.load(s.render)
        requested = rd.requested(scope, plan, rows)
        for group in {c.group for c in requested}:               # every requested cell runs on the fleet given
            rd.fleet_of(plan, group, s.fleet, s.hosts)
        cells, report = store.cells_to_run(scope, plan, rows, s.cells_root)
    except Exception as e:
        say(f"the plan could not be read: {type(e).__name__}: {e}")
        return NOT_RUN
    for line in report:
        print(line, file=sys.stderr, flush=True)
    say(f"{len(cells)} cells to run in rows {','.join(rows)}")
    invalid = 0
    for cell in cells:
        log_dir = s.scheduler_log_dir
        if fresh_run_dir is not None:
            log_dir = Path(s.scheduler_log_dir) / f"{cell}.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
            try:
                sched.restart(fresh_run_dir, log_dir, say)
            except Exception as e:
                say(f"stopping: the scheduler could not be restarted for {cell}: {type(e).__name__}: {e}")
                return NOT_RUN
        if not sched.current(s.scheduler, restart_cmd, say):
            return NOT_RUN
        rc = run_cell.run(dataclasses.replace(s, scheduler_log_dir=log_dir), cell)
        if rc == INVALID:
            invalid += 1
            say(f"{cell} INVALID ({invalid} so far, at most {max_invalid})")
            if invalid > max_invalid:
                say(f"stopping: more than {max_invalid} invalid cells")
                return INVALID
        elif rc != VALID:
            say(f"stopping: {cell} exited {rc}")
            return rc
    for group in rd.row_groups(scope, plan, rows):     # a group completed by a cell that stopped early gets its parity
        try:
            rc = store.parity(scope, plan, s.cells_root, group)
            say(f"group {group}: " + ("parity not ready" if rc is None else f"parity exit {rc}"))
        except Exception as e:
            say(f"group {group}: the parity gate could not run: {type(e).__name__}: {e}")
    if not _mirror_all(s, rows):
        say(f"the second location at {s.mirror} does not match {s.cells_root}")
        return MIRROR_FAILED
    # the whole render, for the log only: the rows' done-ness below rechecks each run's record and isolation
    subprocess.run([sys.executable, "-m", "analysis.manifest", "--scope", str(Path(s.render) / "scope.json"),
                    "--cells", str(s.cells_root)], cwd=run_cell.REPO)
    try:
        problems, _ = store.rows_status(scope, plan, rows, s.cells_root)
    except Exception as e:
        problems = [f"the rows' status could not be read: {type(e).__name__}: {e}"]
    for p in problems:
        print(p, file=sys.stderr, flush=True)
    say(f"rows {','.join(rows)}: {invalid} cells invalid in this run")
    if problems:
        say(f"rows {','.join(rows)} are not done (listed above)")
        return INVALID
    say(f"rows {','.join(rows)} done")
    return VALID
