"""Expected-cells manifest: the cells the scope calls for, and what a cells root holds of them.

Each cell lives in its own directory, <root>/<cell_id>/, which holds its one record <cell_id>.cell.json
(bench/records.py). The expected set is enumerated from the scope (bench/scope.py), never by listing files.
Each expected cell is
  ok         <root>/<cell_id>/<cell_id>.cell.json reads, matches its expectation and has no execution problem
  missing    there is no such record (a directory without one is an aborted attempt, reported as such)
  invalid    the directory holds another record than exactly that one, the file is unreadable, a request
             failed, it does not match its expectation (id, labels, seed, trace, prompt order, dataset,
             tokenizer, bench parameters, the scheduler configuration digest, the expected arrival and
             prompt-order digests), a request started later than max_send_lag_ms after its scheduled arrival,
             or an execution check failed (analysis/validity.py)
  accounted  missing or invalid, with its reason recorded in the scope's "accounted" table
A record in a directory the scope does not list, or directly in the root, is unexpected. No other file is a
cell (logs, decision logs and sidecars never end in .cell.json). Every readable, not accounted cell of one
comparison group must have run on one fleet. A scope with pending blocks is never complete. Quality needs the
scored corpus and is checked by analysis/metrics.py; the per-cell gates' verdicts by scripts/run_cell.sh.

Usage: python -m analysis.manifest --scope <rendered scope.json> --cells <cells root>   (exit 0 only when complete)
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from analysis.validity import failed_requests, integrity_problems
from bench.records import CELL_SUFFIX, CellRecord, cell_dir, cell_path, read_cell, records_in
from bench.scope import BenchParams, ExpectedCell, Scope, expected_cells, load_scope
from routebalance.config import ConfigError

_REASON_CHARS = 500


@dataclass(frozen=True)
class Coverage:
    expected: int
    ok: tuple[str, ...]
    missing: tuple[str, ...]
    invalid: dict[str, tuple[str, ...]]
    accounted: dict[str, str]
    unexpected: tuple[str, ...]
    unused_accounting: tuple[str, ...]     # accounted cells that are in fact ok
    fleet_problems: dict[str, str]         # group -> why its cells cannot be compared
    pending: dict[str, str]                # blocks not rendered into this scope yet (bench/scope.py)
    aborted: tuple[str, ...]               # missing cells whose directory exists without a record

    @property
    def complete(self) -> bool:
        return not (self.missing or self.invalid or self.unexpected or self.fleet_problems or self.pending)


def expectation_problems(cell: CellRecord, exp: ExpectedCell, bench: BenchParams) -> list[str]:
    checks = [
        ("cell id", cell.cell_id, exp.cell_id),
        ("labels", cell.labels, exp.labels),
        ("seed", cell.seed, exp.seed),
        ("trace", cell.trace, exp.trace),
        ("prompt order", cell.prompt_order, exp.prompt_order),
        ("dataset sha256", cell.dataset_sha256, exp.dataset.sha256),
        ("tokenizer", cell.tokenizer_id, exp.dataset.tokenizer_id),
        ("sampling", cell.sampling, bench.sampling),
        ("max_connections", cell.max_connections, bench.max_connections),
        ("request timeout", cell.request_timeout_s, bench.request_timeout_s),
        ("scheduler configuration digest", cell.identity_start.config_digest, exp.config.config_digest),
        ("arrival digest", cell.arrival_digest, exp.arrival_digest()),
        ("prompt-order digest", cell.prompt_order_digest, exp.order_digest()),
    ]
    problems = [f"{what} is {got!r}, the scope expects {want!r}" for what, got, want in checks if got != want]
    lag_ms = max((r.sent_s - r.arrival_s for r in cell.requests), default=0.0) * 1000  # default-ok: no requests
    if lag_ms > bench.max_send_lag_ms:
        problems.append(f"a request started {lag_ms:.1f} ms after its arrival (bound {bench.max_send_lag_ms} ms): "
                        "not the scheduled open-loop load")
    return problems


def cell_problems(cell: CellRecord, exp: ExpectedCell, bench: BenchParams) -> list[str]:
    """Everything that makes a readable cell invalid; empty for a valid cell."""
    failed = failed_requests(cell)
    return ([f"{failed}/{len(cell.requests)} requests failed"] if failed else []) \
        + expectation_problems(cell, exp, bench) + integrity_problems(cell)


def check(scope: Scope, cells_root: Path) -> Coverage:
    cells = expected_cells(scope)
    ok, missing, invalid, fleet_of, aborted = [], [], {}, {}, []
    for exp in cells:
        directory = cell_dir(cells_root, exp.cell_id)
        path = cell_path(directory, exp.cell_id)
        found = records_in(directory)
        if not found:
            missing.append(exp.cell_id)
            if directory.is_dir():
                aborted.append(exp.cell_id)
            continue
        if found != [path.name]:
            invalid[exp.cell_id] = (f"the cell directory must hold exactly {path.name}, it holds {found}",)
            continue
        try:
            rec = read_cell(path)
        except ConfigError as e:
            invalid[exp.cell_id] = (f"unreadable: {str(e)[:_REASON_CHARS]}",)
            continue
        problems = cell_problems(rec, exp, scope.bench)
        fleet_of[exp.cell_id] = (exp.group, rec.identity_start.fleet_digest)
        if problems:
            invalid[exp.cell_id] = tuple(problems)
        else:
            ok.append(exp.cell_id)
    accounted = {c: scope.accounted[c] for c in [*missing, *invalid] if c in scope.accounted}
    expected_ids = {c.cell_id for c in cells}
    root = Path(cells_root)
    flat = {p.name for p in root.iterdir() if p.name.endswith(CELL_SUFFIX)}          # never where a cell belongs
    stray = {p.name for p in root.iterdir() if p.is_dir() and records_in(p) and p.name not in expected_ids}
    fleets: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))  # default-ok: grouping
    for c, (group, digest) in fleet_of.items():
        if c not in accounted:                   # an accounted cell is missing or invalid, and never compared
            fleets[group][digest].append(c)
    fleet_problems = {g: f"cells ran on {len(by)} different fleets: " +
                      "; ".join(f"{d[:12]}: {sorted(ids)}" for d, ids in sorted(by.items()))
                      for g, by in sorted(fleets.items()) if len(by) > 1}
    return Coverage(
        expected=len(cells), ok=tuple(ok),
        missing=tuple(c for c in missing if c not in accounted),
        invalid={c: p for c, p in invalid.items() if c not in accounted},
        accounted=accounted,
        unexpected=tuple(sorted(flat | {f"{d}/" for d in stray})),
        unused_accounting=tuple(sorted(c for c in scope.accounted if c in ok)),
        fleet_problems=fleet_problems, pending=dict(sorted(scope.pending.items())),
        aborted=tuple(c for c in aborted if c not in accounted))


def report(cov: Coverage) -> str:
    lines = [f"expected {cov.expected}: ok {len(cov.ok)}, missing {len(cov.missing)}, invalid {len(cov.invalid)}, "
             f"accounted {len(cov.accounted)}, unexpected {len(cov.unexpected)}, pending blocks {len(cov.pending)}"]
    lines += [f"  MISSING    {c}" + (" (a directory without a record: an aborted attempt)" if c in cov.aborted else "")
              for c in cov.missing]
    lines += [f"  INVALID    {c}: {'; '.join(p)}" for c, p in sorted(cov.invalid.items())]
    lines += [f"  ACCOUNTED  {c}: {r}" for c, r in sorted(cov.accounted.items())]
    lines += [f"  UNEXPECTED {c}" for c in cov.unexpected]
    lines += [f"  FLEET      group {g}: {p}" for g, p in sorted(cov.fleet_problems.items())]
    lines += [f"  PENDING    block {b}: {r}" for b, r in cov.pending.items()]
    lines += [f"  note: accounted cell {c} is ok; its accounting entry is stale" for c in cov.unused_accounting]
    lines.append("COMPLETE" if cov.complete else "NOT COMPLETE")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Check a cells root against the scope's expected cells.")
    ap.add_argument("--scope", type=Path, required=True)
    ap.add_argument("--cells", type=Path, required=True)
    a = ap.parse_args(argv)
    if not a.cells.is_dir():
        raise ConfigError(f"{a.cells} is not a directory")
    cov = check(load_scope(a.scope), a.cells)
    print(report(cov))
    return 0 if cov.complete else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
