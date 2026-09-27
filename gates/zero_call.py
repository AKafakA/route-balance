"""Zero-call gate: a cell whose latency source is constant or profiled_curve never called a booster.

Usage: python -m gates.zero_call --cell DIR   (exit 0 on pass, 1 on fail; one reason per line)

The source is read from what the scheduler executed, not only from the config:
  - manifest_before.json and manifest_after.json must be identical, so nothing changed during the cell;
  - the loaded latency model's source is the configured one, or none is loaded (a baseline that never
    scores latency);
  - with a constant or profiled_curve source, /v1/stats booster_calls is 0 in stats_before.json and in
    stats_after.json; with no latency model loaded it is null in both;
  - with the xgboost source the gate does not apply and says so.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gates.cell import Cell, changed_keys, open_cell, run

BOOSTER_FREE = ("constant", "profiled_curve")


def zero_call_problems(cell: Cell) -> tuple[list[str], str]:
    before, after = cell.manifest("before"), cell.manifest("after")
    problems = []
    if before != after:
        problems.append(f"the execution manifest changed during the cell: {changed_keys(before, after)}")
    source = before.system.latency["source"]
    loaded = before.loaded.latency_source
    if loaded is not None and loaded != source:
        problems.append(f"the configured latency source is {source!r} but the scheduler loaded {loaded!r}")
    if source == "xgboost":
        return problems, f"{cell.cell_id}: not applicable ({source!r} source)"
    if source not in BOOSTER_FREE:
        return problems + [f"unknown latency source {source!r}"], f"{cell.cell_id}: unknown latency source"

    for when in ("before", "after"):
        calls = cell.stats(when).booster_calls
        if loaded is None and calls is not None:
            problems.append(f"stats_{when}.json reports {calls} booster calls, but no latency model is loaded")
        elif loaded is not None and calls != 0:
            problems.append(f"{source} source: stats_{when}.json reports {calls!r} booster calls, not 0")
    what = "no latency model loaded" if loaded is None else "0 booster calls before and after"
    return problems, f"{cell.cell_id}: {source} source, {what}"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Check that a constant or profiled-curve cell never called a booster.")
    ap.add_argument("--cell", type=Path, required=True, help="the cell's evidence directory")
    a = ap.parse_args(argv)
    return run("zero_call", lambda: zero_call_problems(open_cell(a.cell)))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
