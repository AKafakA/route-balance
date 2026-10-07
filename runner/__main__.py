"""python -m runner: the rows of a render, one cell, a group's parity, the rows' status, retiring a cell.

  python -m runner rows   --render DIR --rows R1,R2 --max-invalid N [--restart CMD] [--fresh-scheduler RUN_DIR]
                          --scheduler URL --scheduler-log-dir DIR --fleet F --hosts H --engines-record FILE
                          --data-root DIR --cells-root DIR --mirror DEST
  python -m runner cell   --render DIR --cell ID --scheduler URL --scheduler-log-dir DIR --fleet F --hosts H
                          --engines-record FILE --data-root DIR --cells-root DIR --mirror DEST
  python -m runner parity --render DIR --cells-root DIR --group G
  python -m runner status --render DIR --rows R1,R2 --cells-root DIR
  python -m runner retire --cells-root DIR --cell ID

Run it with the scheduler environment's python. Paths are relative to the repository root, where it runs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from routebalance.config import ConfigError
from runner import cell, render as rd, rows, store
from tools import direct

SETUP = ("scheduler", "scheduler-log-dir", "fleet", "hosts", "engines-record", "data-root", "cells-root", "mirror")


def _given(value: str) -> str:
    """A required value is never empty (an empty --cells-root would be the repository itself)."""
    if not value:
        raise argparse.ArgumentTypeError("must not be empty")
    return value


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m runner", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name: str, *flags: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name)
        for f in flags:
            p.add_argument(f"--{f}", required=True, type=_given)
        return p
    r = cmd("rows", "render", "rows", "max-invalid", *SETUP)
    r.add_argument("--restart", help="a command that restarts a scheduler running stale code")
    r.add_argument("--fresh-scheduler", metavar="RUN_DIR", help="restart the scheduler from RUN_DIR before every cell")
    cmd("cell", "render", "cell", *SETUP)
    cmd("parity", "render", "cells-root", "group")
    cmd("status", "render", "rows", "cells-root")
    cmd("retire", "cells-root", "cell")
    return ap


def _rows(ap: argparse.ArgumentParser, value: str) -> list[str]:
    names = [r for r in value.split(",") if r]
    if not names:
        ap.error(f"--rows {value!r} names no row")
    return names


def _setup(a: argparse.Namespace) -> cell.Setup:
    return cell.Setup(render=Path(a.render), scheduler=a.scheduler, scheduler_log_dir=Path(a.scheduler_log_dir),
                      fleet=Path(a.fleet), hosts=Path(a.hosts), engines_record=Path(a.engines_record),
                      data_root=Path(a.data_root), cells_root=Path(a.cells_root), mirror=a.mirror)


def main(argv: list[str]) -> int:
    ap = _parser()
    a = ap.parse_args(argv)
    if a.cmd == "rows":
        if not a.max_invalid.isdigit():
            ap.error(f"--max-invalid {a.max_invalid!r} is not a count")
        if a.fresh_scheduler and not Path(a.scheduler_log_dir).is_absolute():
            ap.error("--scheduler-log-dir must be absolute with --fresh-scheduler")
        return rows.run(_setup(a), _rows(ap, a.rows), int(a.max_invalid), a.restart if a.restart else None,
                        Path(a.fresh_scheduler) if a.fresh_scheduler else None)
    if a.cmd == "cell":
        return cell.run(_setup(a), a.cell)
    if a.cmd == "parity":
        scope, plan = rd.load(Path(a.render))
        rc = store.parity(scope, plan, Path(a.cells_root), a.group)
        print("not ready" if rc is None else f"parity exit {rc}")
        return 0 if rc in (None, 0) else 1
    if a.cmd == "status":
        scope, plan = rd.load(Path(a.render))
        problems, _ = store.rows_status(scope, plan, _rows(ap, a.rows), Path(a.cells_root))
        if problems:
            print("\n".join(problems), file=sys.stderr)
        return 0 if not problems else 1
    if a.cmd == "retire":
        print(f"{store.retire(Path(a.cells_root), a.cell)}; move the second location's copy of {a.cell} aside too "
              "before the cell runs again (the rows runner refuses a mirror that holds files the new run does not)")
    return 0


if __name__ == "__main__":
    os.chdir(cell.REPO)                       # every relative path is the repository root's
    try:
        sys.exit(main(sys.argv[1:]))
    except (ConfigError, direct.NotIdle, OSError, KeyError) as e:
        print(f"runner: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
