"""The campaign's plan: its rows of runs in the plan's order, read from a render (configs/render36/plan.json).

A render is a directory of three parts, read by the drivers and the analysis:
  scope.json          the scope in bench/scope.py's schema: every cell with its block, group, trace and seed
  configs/<id>.json   each configuration's complete scheduler document, which scripts/run_cell.sh POSTs
  plan.json           the plan, below

plan.json ("format": "rb-plan/1")
  rows            [...] in the plan's order. Every row has id, kind, prio, enabled (true), why (null), and about
                  and plan, free text that no code acts on. By kind:
                    calibration  params {fleet, dataset, levels, requests_per_level} and blocked (null: it can run),
                                 for tools/calibrate_driver.py
                    cells        cells (the ids of the cells it runs), groups, pending and disabled_blocks (both
                                 empty), for tools/cell_driver.py and scripts/run_scope.sh --rows
  groups          {"<group>": {fleet, reference, intended_differences}}: the fleet each comparison group runs on
  fleets          {"<name>": {family, about, fleet_sha256, fleet_digest}}: the fleet file a group's cells must use
  server          {torch_threads, max_connections, request_timeout_s}: the scheduler's launch values
  warmup          {requests_per_instance, max_tokens, prompt}: sent to the engines directly before a cell
  drain           {timeout_s}: the bound on waiting for idle engines
  scored_corpora  {"<family>": {path, sha256}}: the (prompt, model) score lookup the analysis reads
  datasets        {"<name>": {path, sha256, rows, tokenizer_id, budgets}}: the prompt sets, pinned
  bench           the benchmark client's sampling, connection limit and request timeout
  pending_groups  groups whose parity cannot be decided yet (empty)
"""
from __future__ import annotations

from pathlib import Path

from bench.schema import read_json
from routebalance.config import ConfigError

PLAN_FORMAT = "rb-plan/1"


def load_plan(render_dir: Path) -> dict:
    plan = read_json(Path(render_dir) / "plan.json")
    if not isinstance(plan, dict) or "format" not in plan or plan["format"] != PLAN_FORMAT:
        raise ConfigError(f"{render_dir}/plan.json is not a {PLAN_FORMAT} plan")
    return plan


def plan_row(plan: dict, row_id: str, kind: str) -> dict:
    """A direct row's entry, refused unless it can run."""
    rows = [r for r in plan["rows"] if r["id"] == row_id]
    if len(rows) != 1:
        raise ConfigError(f"the plan has no row {row_id!r}")
    row = rows[0]
    if row["kind"] != kind:
        raise ConfigError(f"row {row_id} is a {row['kind']} row, not {kind}")
    if row["blocked"] is not None:
        raise ConfigError(f"row {row_id} cannot run: {row['blocked']}")
    return row
