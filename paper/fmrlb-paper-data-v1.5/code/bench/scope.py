"""The experiment scope: which cells exist and how each one is run (a render's scope.json, configs/render36/).

{
  "bench":    {"sampling": {...Sampling}, "max_connections": int, "request_timeout_s": float,
               "max_send_lag_ms": float (optional; no longer a bound)},
  "datasets": {"<name>": {"path": "<relative to the data root>", "sha256": "<hex>", "rows": int,
                          "tokenizer_id": "<HF id of the reference tokenizer>",
                          "budgets": null | {"path": "<relative to the data root>", "sha256": "<hex>"}}},
  "traces":   {"<trace_id>": {"burstiness": float, "segments": [{"n_requests": int, "rate": float}, ...]}},
  "configs":  {"<config_id>": {"system": str, "knob": str, "dispatch": str,
                               "config_digest": "<fmrlb.config.config_digest of its scheduler config>"}},
  "blocks":   [{"name": str, "group": str, "dataset": "<name>", "configs": [...], "traces": [...],
                "seeds": [int, ...], "prompt_order": "fixed" | "per_seed"}],
  "groups":   {"<group>": {"reference": null | "<config_id>",
                           "intended_differences": {"<config_id>": ["<manifest path>", ...]}}},
  "accounted": {"<cell_id>": "<why this cell is missing or invalid>"},
  "pending":  {"<block name>": "<why its cells are not in this scope yet>"}
}
A cell is one (config, trace, seed); its id is "<config_id>__<trace_id>__s<seed>" and it belongs to exactly
one block. The expected cells are enumerated from this file; nothing is counted by listing a directory.
A block whose configurations need an artifact or a decision that does not exist yet is named under "pending"
instead of listed, and a scope with anything pending is never complete (analysis/manifest.py).

A group is the unit of comparison: frontiers and equal-quality deltas compare cells of one group only, and
the manifest requires every cell of a group to have run on one fleet. The blocks of a group share their
dataset and prompt order (e.g. the seed-1 main grid and its seed-2/3 repeats); a different model family,
topology or order is a different group. Every group a block names is declared under "groups" with the
intended differences of each of its configurations: the execution-manifest paths (GET /v1/manifest; "a.*" is
the subtree under a) in which that configuration's cells may differ. In a comparison group (reference null)
every configuration lists the same system-defining paths; in an ablation group each one lists what it changes
from the group's reference configuration, and the reference itself lists nothing. Everything else the
manifest records must be identical (the parity gate reads each cell's group.json, scripts/run_cell.sh).
max_send_lag_ms is no longer a bound (a late send is counted, never a gate); the report gives each
cell's latest start after its scheduled arrival. Scopes made before it may still carry a value, which nothing reads.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

from pydantic import AfterValidator, Field

from bench.arrivals import Trace, arrival_times, schedule_digest
from bench.dataset import Order, order_digest, prompt_order
from bench.records import Sampling
from bench.schema import Strict, parse, read_json
from fmrlb.config import ConfigError

_ID = re.compile(r"[A-Za-z0-9.-]+(_[A-Za-z0-9.-]+)*")   # no "__": it separates the parts of a cell id
_SHA256 = r"^[0-9a-f]{64}$"


class BenchParams(Strict):
    sampling: Sampling
    max_connections: int = Field(ge=1)
    request_timeout_s: float = Field(gt=0)
    max_send_lag_ms: Optional[float] = Field(default=None, gt=0)   # no longer a bound (scopes before it carry one)


def _relative_path(v: str) -> str:
    if not v or v != v.strip() or v.startswith("/") or ".." in v.split("/"):
        raise ValueError(f"{v!r} must be a relative path inside the data root")
    return v


DataPath = Annotated[str, AfterValidator(_relative_path)]          # relative to the data root given at run time


class BudgetTable(Strict):
    """Per-request cost budgets (bench/dataset.py attach_budgets): one line per prompt of the dataset."""
    path: DataPath
    sha256: str = Field(pattern=_SHA256)


class DatasetEntry(Strict):
    path: DataPath
    sha256: str = Field(pattern=_SHA256)
    rows: int = Field(ge=1)
    tokenizer_id: str = Field(min_length=1)
    budgets: Optional[BudgetTable]       # the budget sweep's datasets only (W7b); null for every other


class ConfigEntry(Strict):
    system: str = Field(min_length=1)
    knob: str = Field(min_length=1)
    dispatch: str = Field(min_length=1)
    config_digest: str = Field(pattern=_SHA256)


class ScopeBlock(Strict):
    name: str = Field(min_length=1)
    group: str = Field(min_length=1)
    dataset: str
    configs: list[str] = Field(min_length=1)
    traces: list[str] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    prompt_order: Order


class GroupSpec(Strict):
    reference: Optional[str]
    intended_differences: dict[str, list[str]]      # config id -> manifest paths


def group_record(scope: "Scope", cell: "ExpectedCell") -> dict:
    """What a cell's directory records about its group (group.json)."""
    spec = scope.groups[cell.group]
    return {"group": cell.group, "reference": spec.reference,
            "intended_differences": list(spec.intended_differences[cell.config_id])}


class Scope(Strict):
    bench: BenchParams
    datasets: dict[str, DatasetEntry]
    traces: dict[str, Trace]
    configs: dict[str, ConfigEntry]
    blocks: list[ScopeBlock] = Field(min_length=1)
    groups: dict[str, GroupSpec]
    accounted: dict[str, str]
    pending: dict[str, str]


@dataclass(frozen=True)
class ExpectedCell:
    cell_id: str
    block: str
    group: str
    config_id: str
    config: ConfigEntry
    trace_id: str
    trace: Trace
    seed: int
    prompt_order: Order
    dataset: DatasetEntry

    @property
    def labels(self) -> dict[str, str]:
        return {"block": self.block, "group": self.group, "config_id": self.config_id, "trace_id": self.trace_id,
                "system": self.config.system, "knob": self.config.knob, "dispatch": self.config.dispatch}

    def arrival_digest(self) -> str:
        return schedule_digest(arrival_times(self.trace, self.seed))

    def order_digest(self) -> str:
        return order_digest(prompt_order(self.dataset.rows, self.prompt_order, self.seed)[:self.trace.n_requests])


def cell_id(config_id: str, trace_id: str, seed: int) -> str:
    return f"{config_id}__{trace_id}__s{seed}"


def load_scope(path: Path) -> Scope:
    return parse(Scope, read_json(path), f"scope {path}")


def expected_cells(scope: Scope) -> list[ExpectedCell]:
    """Every cell the scope calls for, in block order. Raises on any inconsistency in the scope."""
    for kind, ids in (("config", scope.configs), ("trace", scope.traces), ("dataset", scope.datasets)):
        bad = sorted(i for i in ids if not _ID.fullmatch(i))
        if bad:
            raise ConfigError(f"{kind} ids must match {_ID.pattern}: {bad}")
    group_basis: dict[str, tuple[str, str]] = {}
    cells: dict[str, ExpectedCell] = {}
    for block in scope.blocks:
        if block.group not in scope.groups:
            raise ConfigError(f"block {block.name}: group {block.group!r} is not declared under groups")
        spec = scope.groups[block.group]
        basis = group_basis.setdefault(block.group, (block.dataset, block.prompt_order))  # default-ok: first block
        if basis != (block.dataset, block.prompt_order):
            raise ConfigError(f"group {block.group}: block {block.name} uses dataset/prompt order "
                              f"{(block.dataset, block.prompt_order)}, the group uses {basis}")
        dataset = _lookup(scope.datasets, block.dataset, "dataset", block.name)
        for config_id in block.configs:
            config = _lookup(scope.configs, config_id, "config", block.name)
            if config_id not in spec.intended_differences:
                raise ConfigError(f"block {block.name}: group {block.group} declares no intended differences for "
                                  f"{config_id}")
            if spec.reference is not None and (config_id == spec.reference) == bool(spec.intended_differences[config_id]):
                raise ConfigError(f"group {block.group}: the reference {spec.reference} differs from itself in "
                                  f"nothing, and every other configuration names what it changes ({config_id})")
            for trace_id in block.traces:
                trace = _lookup(scope.traces, trace_id, "trace", block.name)
                if trace.n_requests > dataset.rows:
                    raise ConfigError(f"block {block.name}: trace {trace_id} needs {trace.n_requests} prompts, "
                                      f"dataset {block.dataset} has {dataset.rows}")
                for seed in block.seeds:
                    if not (1 if block.prompt_order == "per_seed" else 0) <= seed < 2**32:
                        raise ConfigError(f"block {block.name}: seed {seed} is not valid for {block.prompt_order}")
                    cid = cell_id(config_id, trace_id, seed)
                    if cid in cells:
                        raise ConfigError(f"cell {cid} is listed twice (blocks {cells[cid].block}, {block.name})")
                    cells[cid] = ExpectedCell(cid, block.name, block.group, config_id, config, trace_id, trace,
                                              seed, block.prompt_order, dataset)
    stray = sorted(set(scope.accounted) - set(cells))
    if stray:
        raise ConfigError(f"accounted entries for cells the scope does not contain: {stray}")
    blank = sorted(c for c, reason in scope.accounted.items() if not reason.strip())
    if blank:
        raise ConfigError(f"accounted entries need a reason: {blank}")
    blank = sorted(b for b, reason in scope.pending.items() if not reason.strip())
    if blank:
        raise ConfigError(f"pending entries need a reason: {blank}")
    return list(cells.values())


def _lookup(table: dict, key: str, kind: str, block: str):
    if key not in table:
        raise ConfigError(f"block {block}: unknown {kind} {key!r}")
    return table[key]
