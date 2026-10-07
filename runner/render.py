"""A render (configs/render36/): its scope and plan, a cell's entry in the scope, and the fleet its group runs on."""
from __future__ import annotations

from pathlib import Path

from bench.schema import parse, read_json
from bench.scope import ExpectedCell, Scope, expected_cells
from routebalance.config import ConfigError
from routebalance.types import Instance
from tools import direct
from tools.campaign import load_plan


def load(render: Path) -> tuple[Scope, dict]:
    """The render's scope (bench/scope.py) and plan (tools/campaign.py)."""
    return parse(Scope, read_json(Path(render) / "scope.json"), f"rendered scope {render}"), load_plan(render)


def expected(scope: Scope, cell: str) -> ExpectedCell:
    cells = {c.cell_id: c for c in expected_cells(scope)}
    if cell not in cells:
        raise ConfigError(f"cell {cell} is not in the rendered scope")
    return cells[cell]


def fleet_of(plan: dict, group: str, fleet: Path, hosts: Path) -> list[Instance]:
    """The instances of the fleet the group runs on, refused unless the fleet file and the hosts give the rendered
    fleet (tools/direct.py lease_fleet)."""
    return direct.lease_fleet(plan, plan["groups"][group]["fleet"], fleet, hosts)


def is_sweep(config: dict) -> bool:
    """The latency sweep: the configuration for which the scheduler writes latency.<digest16>.jsonl."""
    return config["mode"] == "pipeline" and config["pending_decode"] == "own_length"


def requested(scope: Scope, plan: dict, rows: list[str]) -> list[ExpectedCell]:
    """The rendered cells of the requested rows, in the plan's order."""
    known = {r["id"]: r for r in plan["rows"]}
    unknown = [r for r in rows if r not in known or known[r]["kind"] != "cells"]
    if unknown:
        raise ConfigError(f"not cell rows of the plan: {unknown}")
    wanted = {c for r in plan["rows"] if r["id"] in rows for c in r["cells"]}
    return [c for c in expected_cells(scope) if c.cell_id in wanted]


def row_groups(scope: Scope, plan: dict, rows: list[str]) -> list[str]:
    return sorted({c.group for c in requested(scope, plan, rows)})
