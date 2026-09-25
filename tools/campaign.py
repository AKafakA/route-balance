"""The lease campaign: every configuration (configs/presets.json) and every lease row (configs/scope.json),
rendered into the files the drivers run from.

    python -m tools.campaign dry-run [--presets P] [--scope S] [--json OUT]
    python -m tools.campaign render --out DIR [--pins PINS.json] [--decisions DECISIONS.json]
                                    [--fleet NAME=FLEET.json ...] [--hosts HOSTS.json] [--previous DIR]

A render writes a new directory (never an existing one):
  scope.json          the scope in bench/scope.py's schema; bench.runner and analysis.manifest read it
  configs/<id>.json   each configuration's complete scheduler document; scripts/run_cell.sh POSTs it
  plan.json           the rows in lease order with their cells, the direct rows' parameters (W0 references,
                      W1 calibration), groups and fleets, server launch values, warm-up and drain bounds
  render.json         the inputs (sha256 of every file) and everything left out
The render is staged over the lease: pins (artifacts built or measured on the lease, such as the W1 TPOT
table) and decisions (owner choices made after a gate) arrive as it goes. A block whose configurations need a
pin, a decision or a fleet file that is not given yet, or an owner value still OWNER-MUST-SET, is left out and
listed under scope.json "pending" with the reason, so the manifest cannot call the campaign complete. Each
later render must keep every earlier configuration and cell (--previous checks it). The dry run needs none of
the lease inputs: it lists every row, block and cell, estimated hours, and every owner value, pin and decision
still missing.

presets.json
  "lists"      {"<name>": [str, ...]}              named template axes
  "parts"      {"<name>": <any JSON>}              whole values
  "templates"  [{"id", "axes": {"<var>": [str, ...] | "$list:<name>"}, "system", "knob", "dispatch",
                 "config": {<every key of a scheduler configuration>}}]
  A template expands over the product of its axes and must have a "family" axis (the model family, checked
  against the fleet of every group that runs it). "{var}" in any string is replaced by the axis value. A string
  value that is exactly one of these is replaced:
    "$part:<name>"                  the named part, rendered the same way
    "$pin:<name>:<field>"           a field ("path" or "md5") of a lease pin: pins.json {"<name>": {...}}
    "$without:<weight>:<part>"      the weights part with <weight> set to 0 and the other two rescaled to sum
                                    to 1 (the per-term drops; the rescaling leaves the argmax unchanged)
    "OWNER-MUST-SET"                an owner value not set yet
  Nothing is merged: every configuration states every key, inline or as a whole part, and is checked by
  routebalance.config.parse_scheduler and, for a pipeline, by the router's own option schema.

scope.json (the lease rows around bench/scope.py's blocks)
  bench, datasets, traces, accounted      as bench/scope.py (the render writes the configs table)
  server {torch_threads, max_connections, request_timeout_s}     the scheduler launch values (D44)
  warmup {requests_per_instance, max_tokens, prompt}             sent to the engines directly before a cell
  drain {timeout_s}                                              the bound on waiting for idle engines
  references {"<family>": {path, sha256}}   the corpus with each prompt's reference answer (the judge's input)
  fleets {"<name>": {family, about}};  groups {"<group>": {fleet}}
  owner {"<name>": [str, ...]}              owner-chosen axes, e.g. the two G1 presets
  decisions {"<name>": {after, about}}      owner choices made during the lease, after the named row
  rows [...]                                in the plan's order; kind references | calibration | cells | pack
A cells row holds blocks: bench/scope.py's block (name, group, dataset, traces, seeds, prompt_order) whose
configs are patterns over the block's axes ({"<var>": [str, ...] | "$list:<name>" | "$owner:<name>" |
"$decision:<name>"}). "alias" prefixes every configuration id (the same configuration run again in another
group, e.g. the topology row, gets its own cell ids); "fill" drops cells an earlier block already lists (without
it a repeat is an error); "planned" gives the dry run the size of an axis that is not decided yet.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Optional, Sequence, Union

from pydantic import Field

from analysis.rules import PLACEHOLDER
from bench.arrivals import Trace
from bench.dataset import Order
from bench.scope import BenchParams, DatasetEntry, Scope, cell_id, expected_cells
from bench.schema import Strict, parse, read_json
from routebalance.baselines.routers import validate_options
from routebalance.config import ConfigError, config_digest, fleet_digest, load_fleet, parse_scheduler

REPO = Path(__file__).resolve().parents[1]
PRESETS = REPO / "configs" / "presets.json"
SCOPE = REPO / "configs" / "scope.json"
PLAN_FORMAT = "rb-plan/1"

# Planning estimates for the dry run only (plan section 5): measured v3 minutes per cell of 3534 requests at
# each rate, and the driver's own time per cell (drain, configuration swap, warm-up, gates, mirror).
V3_MINUTES = {6.0: 10.3, 12.0: 5.5, 18.0: 4.1, 24.0: 3.3, 30.0: 2.9}
V3_REQUESTS = 3534
DRIVER_MINUTES = 1.0

Owner = Literal["OWNER-MUST-SET"]
_VAR = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_NAME = re.compile(r"[A-Za-z0-9._-]+")
_SCHEDULER_KEYS = frozenset({"mode", "weights", "rounds", "lpt", "pending_decode", "dead_reckoning", "telemetry_read",
                             "latency", "estimator", "budget", "telemetry", "pipeline"})
_WEIGHT_KEYS = ("w_quality", "w_latency", "w_cost")


def placeholders(data: Any, path: str = "") -> list[str]:
    """Where OWNER-MUST-SET values sit in a JSON value."""
    if data == PLACEHOLDER:
        return [path if path else "<root>"]
    if isinstance(data, dict):
        return [p for k, v in data.items() for p in placeholders(v, f"{path}.{k}" if path else k)]
    if isinstance(data, list):
        return [p for i, v in enumerate(data) for p in placeholders(v, f"{path}[{i}]")]
    return []


def substitute(text: str, variables: Mapping[str, str], where: str) -> str:
    def replace(m: re.Match) -> str:
        if m.group(1) not in variables:
            raise ConfigError(f"{where}: unknown variable {{{m.group(1)}}} in {text!r}")
        return variables[m.group(1)]
    return _VAR.sub(replace, text)


def _distinct(values: list, where: str) -> list:
    if not values or len(set(values)) != len(values):
        raise ConfigError(f"{where}: an axis needs distinct values, got {values}")
    return values


# ---------------------------------------------------------------- presets

class Template(Strict):
    id: str
    axes: dict[str, Union[list[str], str]]
    system: str
    knob: str
    dispatch: str
    ablates: list[str]              # manifest paths an ablation arm changes from its reference ([] otherwise)
    config: dict[str, Any]


class PresetsFile(Strict):
    notes: dict[str, str]
    lists: dict[str, list[str]]
    parts: dict[str, Any]
    templates: list[Template] = Field(min_length=1)


@dataclass(frozen=True)
class Preset:
    preset_id: str
    family: str
    system: str
    knob: str
    dispatch: str
    ablates: tuple[str, ...]
    config: Any                     # the template's config; references are resolved at render time
    variables: tuple[tuple[str, str], ...]


def _list_axis(axis: Union[list[str], str], lists: Mapping[str, list[str]], where: str) -> list[str]:
    if isinstance(axis, list):
        return _distinct(list(axis), where)
    if axis.startswith("$list:") and axis[6:] in lists:
        return _distinct(list(lists[axis[6:]]), where)
    raise ConfigError(f"{where}: a template axis is a list or \"$list:<known list>\", got {axis!r}")


def load_presets(path: Path = PRESETS) -> PresetsFile:
    return parse(PresetsFile, read_json(path), f"presets {path}")


def expand(presets: PresetsFile) -> dict[str, Preset]:
    """Every configuration the templates define, by id, in template order."""
    out: dict[str, Preset] = {}
    for n, t in enumerate(presets.templates):
        where = f"template {n} ({t.id})"
        if "family" not in t.axes:
            raise ConfigError(f"{where}: every template has a family axis")
        if set(t.config) != _SCHEDULER_KEYS:
            missing, unknown = sorted(_SCHEDULER_KEYS - set(t.config)), sorted(set(t.config) - _SCHEDULER_KEYS)
            raise ConfigError(f"{where}: a configuration states every key; missing {missing}, unknown {unknown}")
        names = list(t.axes)
        values = [_list_axis(t.axes[v], presets.lists, f"{where} axis {v}") for v in names]
        for combo in itertools.product(*values):
            variables = dict(zip(names, combo, strict=True))
            pid = substitute(t.id, variables, where)
            if not _NAME.fullmatch(pid) or "__" in pid:
                raise ConfigError(f"{where}: configuration id {pid!r} must match {_NAME.pattern} without '__'")
            if pid in out:
                raise ConfigError(f"{where}: configuration {pid!r} is defined twice")
            out[pid] = Preset(pid, variables["family"], substitute(t.system, variables, where),
                              substitute(t.knob, variables, where), substitute(t.dispatch, variables, where),
                              tuple(t.ablates), t.config, tuple(variables.items()))
    return out


class _Missing:
    def __repr__(self) -> str:
        return "<pin not given>"


MISSING = _Missing()


@dataclass
class Needs:
    owner: set[str] = field(default_factory=set)       # where an OWNER-MUST-SET value was reached
    pins: set[str] = field(default_factory=set)        # pins referenced and not given
    decisions: set[str] = field(default_factory=set)   # decisions referenced and not made
    other: list[str] = field(default_factory=list)     # anything else missing (a fleet file)

    def empty(self) -> bool:
        return not (self.owner or self.pins or self.decisions or self.other)

    def update(self, other: "Needs") -> None:
        self.owner |= other.owner
        self.pins |= other.pins
        self.decisions |= other.decisions
        self.other += [o for o in other.other if o not in self.other]

    def reason(self) -> str:
        parts = []
        if self.owner:
            parts.append(f"{PLACEHOLDER} at {sorted(self.owner)}")
        if self.pins:
            parts.append(f"pins not given yet: {sorted(self.pins)}")
        if self.decisions:
            parts.append(f"decisions not made yet: {sorted(self.decisions)}")
        return "; ".join(parts + self.other)


class Resolver:
    """Resolves a configuration's references: parts, pins, weight drops and owner placeholders. An owner
    placeholder is reported where it is written: presets.json parts.<name>... or the template's config."""

    def __init__(self, presets: PresetsFile, pins: Mapping[str, Mapping[str, str]]):
        self._parts = presets.parts
        self._pins = pins

    def resolve(self, value: Any, where: str, needs: Needs, variables: Mapping[str, str],
                stack: tuple[str, ...] = ()) -> Any:
        if isinstance(value, dict):
            return {k: self.resolve(v, f"{where}.{k}", needs, variables, stack) for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve(v, f"{where}[{i}]", needs, variables, stack) for i, v in enumerate(value)]
        if not isinstance(value, str):
            return value
        s = substitute(value, variables, where)
        if s == PLACEHOLDER:
            needs.owner.add(f"presets {where}")
            return PLACEHOLDER
        if s.startswith("$part:"):
            return self._part(s[6:], where, needs, variables, stack)
        if s.startswith("$pin:"):
            name, sep, field_name = s[5:].rpartition(":")
            if not sep or not _NAME.fullmatch(name) or field_name not in ("path", "md5"):
                raise ConfigError(f"{where}: a pin reference is \"$pin:<name>:path|md5\", got {s!r}")
            if name not in self._pins:
                needs.pins.add(name)
                return MISSING
            if field_name not in self._pins[name]:
                raise ConfigError(f"{where}: pin {name!r} has no {field_name!r}")
            return self._pins[name][field_name]
        if s.startswith("$without:"):
            key, sep, part = s[9:].partition(":")
            if not sep:
                raise ConfigError(f"{where}: a weight drop is \"$without:<weight>:<part>\", got {s!r}")
            return _without(self._part(part, where, needs, variables, stack), key, where)
        if s.startswith("$"):
            raise ConfigError(f"{where}: unknown reference {s!r}")
        return s

    def _part(self, name: str, where: str, needs: Needs, variables: Mapping[str, str], stack: tuple[str, ...]):
        if name in stack:
            raise ConfigError(f"{where}: parts refer to each other in a cycle: {[*stack, name]}")
        if name not in self._parts:
            raise ConfigError(f"{where}: unknown part {name!r}")
        return self.resolve(self._parts[name], f"parts.{name}", needs, variables, (*stack, name))


def _without(weights: Any, key: str, where: str) -> Any:
    if not isinstance(weights, dict) or set(weights) != set(_WEIGHT_KEYS) or key not in _WEIGHT_KEYS:
        raise ConfigError(f"{where}: $without drops one of {_WEIGHT_KEYS} from a weights part")
    if any(bool(v == PLACEHOLDER or v is MISSING) for v in weights.values()):
        return weights                                # unresolved; the needs already say so
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in weights.values()):
        raise ConfigError(f"{where}: weights must be numbers, got {weights}")
    rest = math.fsum(v for k, v in weights.items() if k != key)
    if not rest > 0:
        raise ConfigError(f"{where}: dropping {key} leaves no weight")
    return {k: 0.0 if k == key else v / rest for k, v in weights.items()}


@dataclass(frozen=True)
class Rendered:
    preset: Preset
    document: dict
    digest: str


def render_config(preset: Preset, resolver: Resolver) -> tuple[Optional[Rendered], Needs]:
    """(the configuration, no needs) when it renders; (None, what it still needs) otherwise. A configuration
    that resolves completely but is invalid raises: that is a defect of the presets."""
    needs = Needs()
    doc = resolver.resolve(preset.config, preset.preset_id, needs, dict(preset.variables))
    if not needs.empty():
        return None, needs
    cfg = parse_scheduler(doc)
    if cfg.pipeline is not None:
        validate_options(cfg.pipeline.router)
    return Rendered(preset, doc, config_digest(cfg)), needs


def referenced_pins(presets: PresetsFile, expanded: Mapping[str, Preset]) -> set[str]:
    """Every pin name some configuration references: the only names a pins file may hold."""
    names: set[str] = set()
    resolver = Resolver(presets, {})
    for p in expanded.values():
        needs = Needs()
        resolver.resolve(p.config, p.preset_id, needs, dict(p.variables))
        names |= needs.pins
    return names


# ---------------------------------------------------------------- the lease scope

class BlockSource(Strict):
    name: str = Field(min_length=1)
    group: str
    dataset: str
    axes: dict[str, Union[list[str], str]]
    configs: list[str] = Field(min_length=1)
    traces: list[str] = Field(min_length=1)
    seeds: Union[list[int], Owner]
    prompt_order: Order
    fill: bool
    alias: Optional[str]
    planned: dict[str, Annotated[int, Field(ge=1)]]
    enabled: bool                   # false: the owner removed the block (never run, never counted)
    why: Optional[str]
    hold: Optional[str]             # a block that must run but cannot yet (no implementation, an open owner
                                    # decision): always pending with this reason, so the campaign is never complete


class _Row(Strict):
    id: str = Field(min_length=1)
    prio: Literal["must", "must-if-g1-passes", "should", "fill", "good-to-have"]
    enabled: bool
    why: Optional[str]              # required when disabled: what is missing
    about: str
    plan: str                       # the plan's cell count and hours for this row (section 5)


class ReferencesRow(_Row):
    kind: Literal["references"]
    fleet: str
    dataset: str
    references: str
    concurrency_per_instance: Union[int, Owner]
    divergence_prompts: Union[int, Owner]
    divergence_concurrency_per_instance: Union[int, Owner]


class CalibrationRow(_Row):
    kind: Literal["calibration"]
    fleet: str
    dataset: str
    levels: list[Union[int, Owner]]
    requests_per_level: Union[int, Owner]


class CellsRow(_Row):
    kind: Literal["cells"]
    blocks: list[BlockSource] = Field(min_length=1)


class PackRow(_Row):
    kind: Literal["pack"]


Row = Annotated[Union[ReferencesRow, CalibrationRow, CellsRow, PackRow], Field(discriminator="kind")]
_ROW_META = {"id", "kind", "prio", "enabled", "why", "about", "plan"}


class FleetEntry(Strict):
    family: str
    about: str


class GroupEntry(Strict):
    """What the group's cells may differ in (the parity gate reads it from each cell's group.json):
    a comparison group lists the system-defining manifest paths, the same for every cell (reference null);
    an ablation group names its reference configuration (a pattern over scalar decisions, e.g.
    "abl.qwen.rb.{headline}"), and each other cell may differ from it only in its template's "ablates"."""
    fleet: str
    reference: Optional[str]
    intended_differences: Optional[list[str]]


class DecisionEntry(Strict):
    after: str
    about: str


class ReferencesEntry(Strict):
    path: str
    sha256: Union[str, Owner]


class ScopeFile(Strict):
    notes: dict[str, str]
    server: dict[str, Any]
    bench: dict[str, Any]
    warmup: dict[str, Any]
    drain: dict[str, Any]
    datasets: dict[str, dict[str, Any]]
    references: dict[str, ReferencesEntry]
    traces: dict[str, dict[str, Any]]
    fleets: dict[str, FleetEntry]
    groups: dict[str, GroupEntry]
    owner: dict[str, Union[list[str], Owner]]
    decisions: dict[str, DecisionEntry]
    rows: list[Row] = Field(min_length=1)
    accounted: dict[str, str]


class Server(Strict):
    torch_threads: int = Field(ge=1)
    max_connections: int = Field(ge=1)
    request_timeout_s: float = Field(gt=0)


class Warmup(Strict):
    requests_per_instance: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    prompt: str = Field(min_length=1)


class Drain(Strict):
    timeout_s: float = Field(gt=0)


def load_scope_file(path: Path = SCOPE) -> ScopeFile:
    scope = parse(ScopeFile, read_json(path), f"lease scope {path}")
    ids = [r.id for r in scope.rows]
    if len(set(ids)) != len(ids):
        raise ConfigError(f"{path}: row ids repeat: {ids}")
    for r in scope.rows:
        things = [(r.id, r)] + ([(f"{r.id}/{b.name}", b) for b in r.blocks] if isinstance(r, CellsRow) else [])
        for name, thing in things:
            if not thing.enabled and not (thing.why and thing.why.strip()):
                raise ConfigError(f"{path}: {name} is disabled without a reason")
        if isinstance(r, (ReferencesRow, CalibrationRow)) and r.fleet not in scope.fleets:
            raise ConfigError(f"{path}: row {r.id}: unknown fleet {r.fleet!r}")
    for g, entry in scope.groups.items():
        if entry.fleet not in scope.fleets:
            raise ConfigError(f"{path}: group {g}: unknown fleet {entry.fleet!r}")
        if (entry.reference is None) == (entry.intended_differences is None):
            raise ConfigError(f"{path}: group {g} has either a reference (an ablation group) or its "
                              "intended_differences (a comparison group)")
    for name, d in scope.decisions.items():
        if d.after not in ids:
            raise ConfigError(f"{path}: decision {name} comes after an unknown row {d.after!r}")
    return scope


def _section(model, data: Any, where: str):
    unset = placeholders(data, where)
    if unset:
        raise ConfigError(f"owner values not set yet: {unset}")
    return parse(model, data, where)


def trace_minutes(trace: Trace) -> float:
    """Planning estimate of one cell's wall time (V3_MINUTES)."""
    arrivals = math.fsum(s.n_requests / s.rate for s in trace.segments) / 60.0
    nearest = min(V3_MINUTES, key=lambda r: abs(r - trace.segments[-1].rate))
    return arrivals + (V3_MINUTES[nearest] - V3_REQUESTS / nearest / 60.0) + DRIVER_MINUTES


# ---------------------------------------------------------------- blocks

@dataclass(frozen=True)
class ConfigRef:
    config_id: str                 # as the scope lists it (alias + preset id)
    preset_id: str


@dataclass
class BlockPlan:
    row: str
    block: BlockSource
    configs: list[ConfigRef]       # the decided configurations, in order
    undecided: int                 # configurations still behind an undecided axis (planned sizes)
    needs: Needs
    cells: list[tuple[ConfigRef, str, int]] = field(default_factory=list)
    reason: Optional[str] = None   # why the block is not rendered (None: rendered)


def _block_axes(block: BlockSource, scope: ScopeFile, lists: Mapping[str, list[str]],
                decisions: Mapping[str, list[str]], needs: Needs, where: str) -> dict[str, tuple[Optional[list[str]], int]]:
    """var -> (values, or None while undecided; the axis size, or the planned size while undecided)."""
    out: dict[str, tuple[Optional[list[str]], int]] = {}
    for var, axis in block.axes.items():
        at = f"{where} axis {var}"
        values: Optional[list[str]]
        size: Optional[int] = None
        if isinstance(axis, list) or axis.startswith("$list:"):
            values = _list_axis(axis, lists, at)
        elif axis.startswith("$owner:"):
            name = axis[7:]
            if name not in scope.owner:
                raise ConfigError(f"{at}: unknown owner list {name!r}")
            raw = scope.owner[name]
            if raw == PLACEHOLDER:
                needs.owner.add(f"owner.{name}")
                values = None
            else:
                unset = [f"owner.{name}[{i}]" for i, v in enumerate(raw) if v == PLACEHOLDER]
                needs.owner.update(unset)
                values, size = (None, len(raw)) if unset else (_distinct(list(raw), at), None)
        elif axis.startswith("$decision:"):
            name = axis[10:]
            if name not in scope.decisions:
                raise ConfigError(f"{at}: undeclared decision {name!r}")
            if name in decisions:
                values = _distinct(list(decisions[name]), at)
            else:
                needs.decisions.add(name)
                values = None
        else:
            raise ConfigError(f"{at}: an axis is a list, \"$list:\", \"$owner:\" or \"$decision:\", got {axis!r}")
        if values is not None:
            size = len(values)
        elif size is None:
            if var not in block.planned:
                raise ConfigError(f"{at}: an axis decided later needs a planned size (block \"planned\")")
            size = block.planned[var]
        out[var] = (values, size)
    return out


def _configs(block: BlockSource, axes: Mapping[str, tuple[Optional[list[str]], int]],
             where: str) -> tuple[list[ConfigRef], int]:
    refs: list[ConfigRef] = []
    undecided = 0
    prefix = "" if block.alias is None else f"{block.alias}."
    for pattern in block.configs:
        substitute(pattern, {v: "" for v in axes}, where)          # every variable must be one of the axes
        used = [v for v in axes if "{" + v + "}" in pattern]
        if any(axes[v][0] is None for v in used):
            undecided += math.prod(axes[v][1] for v in used)
            continue
        for combo in itertools.product(*(axes[v][0] for v in used)):
            pid = substitute(pattern, dict(zip(used, combo, strict=True)), where)
            refs.append(ConfigRef(prefix + pid, pid))
    return refs, undecided


def plan_blocks(scope: ScopeFile, presets: Mapping[str, Preset], lists: Mapping[str, list[str]],
                decisions: Mapping[str, list[str]]) -> list[BlockPlan]:
    """Every enabled block of every enabled cells row, in row order, with its decided configurations."""
    out = []
    for row in scope.rows:
        if not (isinstance(row, CellsRow) and row.enabled):
            continue
        for block in row.blocks:
            if not block.enabled:
                continue
            where = f"{row.id}/{block.name}"
            if block.group not in scope.groups:
                raise ConfigError(f"{where}: unknown group {block.group!r}")
            if block.dataset not in scope.datasets:
                raise ConfigError(f"{where}: unknown dataset {block.dataset!r}")
            unknown = [t for t in block.traces if t not in scope.traces]
            if unknown:
                raise ConfigError(f"{where}: unknown traces {unknown}")
            needs = Needs()
            refs, undecided = _configs(block, _block_axes(block, scope, lists, decisions, needs, where), where)
            if block.hold is not None:             # its configurations may not exist yet: never rendered
                needs.other.append(f"held: {block.hold}")
                out.append(BlockPlan(row.id, block, refs, undecided, needs))
                continue
            missing = sorted({r.preset_id for r in refs if r.preset_id not in presets})
            if missing:
                raise ConfigError(f"{where}: configurations no template defines: {missing}")
            family = scope.fleets[scope.groups[block.group].fleet].family
            wrong = sorted({r.preset_id for r in refs if presets[r.preset_id].family != family})
            if wrong:
                raise ConfigError(f"{where}: group {block.group} runs a {family} fleet; these configurations are for "
                                  f"another family: {wrong}")
            _reference(scope, block.group, decisions, needs)
            if block.seeds == PLACEHOLDER:
                needs.owner.add(f"{where}.seeds")
            for tid in block.traces:
                needs.owner.update(placeholders(scope.traces[tid], f"traces.{tid}"))
            needs.owner.update(placeholders(scope.datasets[block.dataset], f"datasets.{block.dataset}"))
            out.append(BlockPlan(row.id, block, refs, undecided, needs))
    return out


def _reference(scope: ScopeFile, group: str, decisions: Mapping[str, list[str]], needs: Needs) -> Optional[str]:
    """An ablation group's reference configuration id; None for a comparison group or while undecided."""
    pattern = scope.groups[group].reference
    if pattern is None:
        return None
    names = _VAR.findall(pattern)
    for name in names:
        if name not in scope.decisions:
            raise ConfigError(f"group {group}: its reference {pattern!r} names an undeclared decision {name!r}")
        if name in decisions and len(decisions[name]) != 1:
            raise ConfigError(f"group {group}: decision {name} must be one value to name the reference")
    missing = [n for n in names if n not in decisions]
    if missing:
        needs.decisions.update(missing)
        return None
    return substitute(pattern, {n: decisions[n][0] for n in names}, f"group {group}")


def _list_cells(bp: BlockPlan, listed: set[str]) -> None:
    """The block's cells, config x trace x seed, minus those an earlier block lists (fill)."""
    for ref in bp.configs:
        for tid in bp.block.traces:
            for seed in bp.block.seeds:
                cid = cell_id(ref.config_id, tid, seed)
                if cid in listed:
                    if not bp.block.fill:
                        raise ConfigError(f"{bp.row}/{bp.block.name}: cell {cid} is listed by an earlier block "
                                          "(mark the block fill to reuse it)")
                    continue
                listed.add(cid)
                bp.cells.append((ref, tid, seed))


# ---------------------------------------------------------------- the render

@dataclass
class RenderResult:
    scope: Optional[dict]                         # None while no cell block renders (before W1)
    plan: dict
    configs: dict[str, dict]                      # config id -> scheduler document
    pending: dict[str, str]                       # "<row>/<block>" -> why it is not rendered
    blocked_rows: dict[str, str]                  # direct rows that cannot run yet


def _decisions(scope: ScopeFile, raw: Mapping[str, Any]) -> dict[str, list[str]]:
    out = {}
    for name, value in raw.items():
        if name not in scope.decisions:
            raise ConfigError(f"decision {name!r} is not declared in the scope")
        ok = bool(isinstance(value, str) or (isinstance(value, list) and value and all(isinstance(v, str) for v in value)))
        if not ok or placeholders(value):
            raise ConfigError(f"decision {name!r} must be a string or a non-empty list of strings, got {value!r}")
        out[name] = [value] if isinstance(value, str) else list(value)
    return out


def _pins(raw: Mapping[str, Any], referenced: set[str]) -> dict[str, dict[str, str]]:
    out = {}
    for name, pin in raw.items():
        if name not in referenced:
            raise ConfigError(f"pin {name!r} is referenced by no configuration (a typo?)")
        if not isinstance(pin, dict) or not pin or not set(pin) <= {"path", "md5"} \
                or not all(isinstance(v, str) and v for v in pin.values()):
            raise ConfigError(f"pin {name!r} must be {{\"path\": str[, \"md5\": str]}}")
        if "md5" in pin and not re.fullmatch(r"[0-9a-f]{32}", pin["md5"]):
            raise ConfigError(f"pin {name!r}: md5 {pin['md5']!r} is not 32 hex digits")
        out[name] = dict(pin)
    return out


def render(presets_file: PresetsFile, scope: ScopeFile, *, pins: Mapping[str, Any], decisions: Mapping[str, Any],
           fleets: Mapping[str, tuple[str, str]]) -> RenderResult:
    """fleets: fleet name -> (sha256 of the fleet file, its fleet_digest under the lease hosts)."""
    expanded = expand(presets_file)
    resolver = Resolver(presets_file, _pins(pins, referenced_pins(presets_file, expanded)))
    bench = _section(BenchParams, scope.bench, "bench")
    datasets = {n: parse(DatasetEntry, d, f"datasets.{n}") for n, d in scope.datasets.items() if not placeholders(d)}
    made = _decisions(scope, decisions)
    plans = plan_blocks(scope, expanded, presets_file.lists, made)
    rendered: dict[str, Rendered] = {}
    config_needs: dict[str, Needs] = {}
    listed: set[str] = set()
    s8_configs: dict[str, dict] = {}
    s8_blocks, pending = [], {}
    config_of: dict[str, str] = {}                # config id -> preset id
    # every cell needs the scheduler's launch values, the warm-up and the drain bound
    run_values = placeholders(scope.model_dump(mode="json", include={"server", "warmup", "drain"}))
    for n, bp in enumerate(plans):
        where = f"{bp.row}/{bp.block.name}"
        needs = Needs()
        needs.update(bp.needs)
        needs.owner.update(run_values)
        if bp.block.fill:
            # a fill block renders after every earlier block of its group that could list one of its cells, so
            # the cells it skips are final and no later stage moves a cell from one block to another
            for earlier in plans[:n]:
                if earlier.reason is not None and earlier.block.group == bp.block.group \
                        and _could_overlap(earlier.block, bp.block):
                    needs.other.append(f"waits for {earlier.row}/{earlier.block.name}, whose cells it would skip")
        for ref in ([] if bp.block.hold is not None else bp.configs):
            if ref.preset_id not in config_needs:
                got, need = render_config(expanded[ref.preset_id], resolver)
                config_needs[ref.preset_id] = need
                if got is not None:
                    rendered[ref.preset_id] = got
            needs.update(config_needs[ref.preset_id])
        fleet = scope.groups[bp.block.group].fleet
        if fleet not in fleets:
            needs.other.append(f"fleet {fleet!r} not given (--fleet {fleet}=FILE)")
        if bp.undecided:
            needs.other.append(f"{bp.undecided} configurations behind undecided axes")
        if not needs.empty():
            bp.reason = needs.reason()
            pending[where] = bp.reason
            continue
        _list_cells(bp, listed)
        for ref in bp.configs:
            r = rendered[ref.preset_id]
            entry = {"system": r.preset.system, "knob": r.preset.knob, "dispatch": r.preset.dispatch,
                     "config_digest": r.digest}
            if s8_configs.setdefault(ref.config_id, entry) != entry:            # default-ok: first listing
                raise ConfigError(f"{where}: configuration id {ref.config_id} names two configurations")
            config_of[ref.config_id] = ref.preset_id
            for tid in bp.block.traces:
                seeds = [s for c, t, s in bp.cells if c == ref and t == tid]
                if seeds:
                    s8_blocks.append({"name": bp.block.name, "group": bp.block.group, "dataset": bp.block.dataset,
                                      "configs": [ref.config_id], "traces": [tid], "seeds": seeds,
                                      "prompt_order": bp.block.prompt_order})
    scope_doc = None              # before W1 no cell can render (every configuration reads the W1 table)
    if s8_blocks:
        used_traces = sorted({b["traces"][0] for b in s8_blocks})
        scope_doc = {"bench": bench.model_dump(mode="json"),
                     "datasets": {n: datasets[n].model_dump(mode="json")
                                  for n in sorted({b["dataset"] for b in s8_blocks})},
                     "traces": {t: parse(Trace, scope.traces[t], f"trace {t}").model_dump(mode="json")
                                for t in used_traces},
                     "configs": s8_configs, "blocks": s8_blocks,
                     "groups": _groups(scope, s8_blocks, {c: expanded[p] for c, p in config_of.items()}, made),
                     "accounted": {c: r for c, r in scope.accounted.items() if c in listed},
                     "pending": pending}
        expected_cells(parse(Scope, scope_doc, "rendered scope"))   # the rendered scope is consistent
    stray = sorted(set(scope.accounted) - listed)
    if stray and not pending:
        raise ConfigError(f"accounted entries for cells the scope does not contain: {stray}")
    configs = {cid: rendered[pid].document for cid, pid in config_of.items()}
    plan, blocked = _plan(scope, plans, fleets, datasets)
    plan["bench"] = bench.model_dump(mode="json")
    plan["pending_groups"] = {}
    for bp in plans:
        if bp.reason is not None:
            plan["pending_groups"].setdefault(bp.block.group, []).append(f"{bp.row}/{bp.block.name}")  # default-ok: grouping
    return RenderResult(scope_doc, plan, configs, pending, blocked)


def _could_overlap(a: BlockSource, b: BlockSource) -> bool:
    """Whether two blocks can list a common cell: a shared trace and a shared seed (unknown seeds may)."""
    if not set(a.traces) & set(b.traces):
        return False
    return bool(a.seeds == PLACEHOLDER or b.seeds == PLACEHOLDER or set(a.seeds) & set(b.seeds))


def _groups(scope: ScopeFile, blocks: list[dict], presets_of: Mapping[str, Preset],
            decisions: Mapping[str, list[str]]) -> dict[str, dict]:
    """The rendered groups table (bench/scope.py): each configuration's intended differences."""
    out: dict[str, dict] = {}
    for b in blocks:
        g, cid = b["group"], b["configs"][0]
        entry = scope.groups[g]
        if g not in out:
            out[g] = {"reference": _reference(scope, g, decisions, Needs()), "intended_differences": {}}
        ref = out[g]["reference"]
        if entry.reference is None:
            paths = list(entry.intended_differences)
        elif cid == ref:
            paths = []
        else:
            paths = list(presets_of[cid].ablates)
            if not paths:
                raise ConfigError(f"group {g}: {cid} is an arm of the ablation group but its template declares no "
                                  "ablated paths")
        out[g]["intended_differences"][cid] = paths
    return out


def _plan(scope: ScopeFile, plans: list[BlockPlan], fleets: Mapping[str, tuple[str, str]],
          datasets: Mapping[str, DatasetEntry]) -> tuple[dict, dict[str, str]]:
    rows, blocked = [], {}
    for row in scope.rows:
        entry: dict[str, Any] = {"id": row.id, "kind": row.kind, "prio": row.prio, "enabled": row.enabled,
                                 "why": row.why, "about": row.about, "plan": row.plan}
        if isinstance(row, CellsRow):
            bps = [bp for bp in plans if bp.row == row.id]
            entry["cells"] = [cell_id(ref.config_id, tid, seed) for bp in bps for ref, tid, seed in bp.cells]
            entry["groups"] = sorted({bp.block.group for bp in bps if bp.reason is None})
            entry["pending"] = {bp.block.name: bp.reason for bp in bps if bp.reason is not None}
            entry["disabled_blocks"] = {b.name: b.why for b in row.blocks if not b.enabled}
        elif isinstance(row, (ReferencesRow, CalibrationRow)):
            if row.dataset not in scope.datasets:
                raise ConfigError(f"row {row.id}: unknown dataset {row.dataset!r}")
            if isinstance(row, ReferencesRow) and row.references not in scope.references:
                raise ConfigError(f"row {row.id}: unknown references {row.references!r}")
            params = row.model_dump(mode="json", exclude=_ROW_META)
            unset = placeholders(params, row.id) + placeholders(scope.datasets[row.dataset], f"datasets.{row.dataset}") \
                + placeholders(scope.model_dump(mode="json", include={"warmup", "drain"}))
            if isinstance(row, ReferencesRow):
                unset += placeholders(scope.references[row.references].model_dump(mode="json"),
                                      f"references.{row.references}")
            entry["params"] = params
            if not row.enabled:
                entry["blocked"] = f"disabled: {row.why}"
            elif unset:
                entry["blocked"] = blocked[row.id] = f"{PLACEHOLDER} at {unset}"
            elif row.fleet not in fleets:
                entry["blocked"] = blocked[row.id] = f"fleet {row.fleet!r} not given"
            else:
                entry["blocked"] = None                               # runnable
        rows.append(entry)

    def section(model, data, where):
        return None if placeholders(data) else _section(model, data, where).model_dump(mode="json")
    plan = {"format": PLAN_FORMAT, "rows": rows,
            "groups": {g: e.model_dump(mode="json") for g, e in scope.groups.items()},
            "fleets": {name: {"family": f.family, "about": f.about,
                              "fleet_sha256": fleets[name][0] if name in fleets else None,
                              "fleet_digest": fleets[name][1] if name in fleets else None}
                       for name, f in scope.fleets.items()},
            "server": section(Server, scope.server, "server"), "warmup": section(Warmup, scope.warmup, "warmup"),
            "drain": section(Drain, scope.drain, "drain"),
            "references": {k: v.model_dump(mode="json") for k, v in scope.references.items()},
            "datasets": {n: d.model_dump(mode="json") for n, d in datasets.items()}}
    return plan, blocked


def write_render(result: RenderResult, out: Path, inputs: dict) -> None:
    out = Path(out)
    if out.exists():
        raise ConfigError(f"{out} exists; every render goes to a new directory")
    (out / "configs").mkdir(parents=True)
    for cid, doc in result.configs.items():
        (out / "configs" / f"{cid}.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    if result.scope is not None:
        (out / "scope.json").write_text(json.dumps(result.scope, indent=1) + "\n")
    (out / "plan.json").write_text(json.dumps(result.plan, indent=1) + "\n")
    (out / "render.json").write_text(json.dumps({**inputs, "pending": result.pending,
                                                 "blocked_rows": result.blocked_rows}, indent=1) + "\n")


def check_previous(result: RenderResult, previous: Path) -> None:
    """An earlier stage's configurations and cells survive unchanged into this one."""
    prev = Path(previous)
    for f in sorted((prev / "configs").glob("*.json")):
        cid = f.name[:-len(".json")]
        if cid not in result.configs:
            raise ConfigError(f"configuration {cid} of {prev} is no longer rendered")
        if read_json(f) != result.configs[cid]:
            raise ConfigError(f"configuration {cid} changed since {prev}: its cells would no longer match")
    def cells(doc) -> dict[str, dict]:
        return {} if doc is None else {c.cell_id: c.labels for c in expected_cells(parse(Scope, doc, "scope"))}
    before = cells(read_json(prev / "scope.json") if (prev / "scope.json").exists() else None)
    now = cells(result.scope)
    lost = sorted(set(before) - set(now))
    if lost:
        raise ConfigError(f"{len(lost)} cells of {prev} are no longer in the scope: {lost[:5]}")
    moved = sorted(c for c in before if before[c] != now[c])
    if moved:
        raise ConfigError(f"{len(moved)} cells of {prev} changed block, group or labels (a fill block now lists "
                          f"them from an earlier block?): {moved[:5]}")


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


# ---------------------------------------------------------------- the dry run

def dry_run(presets_file: PresetsFile, scope: ScopeFile) -> dict:
    """Every row, block and cell the tracked files call for, estimated hours, and what is still missing. No
    lease input is needed: pins and decisions are reported, never required."""
    expanded = expand(presets_file)
    resolver = Resolver(presets_file, {})
    plans = plan_blocks(scope, expanded, presets_file.lists, {})
    owner = set(placeholders(scope.model_dump(mode="json", include={"server", "bench", "warmup", "drain",
                                                                   "datasets", "references"})))
    for p in expanded.values():                     # every value any configuration reads, decided or not
        needs = Needs()
        resolver.resolve(p.config, p.preset_id, needs, dict(p.variables))
        owner |= needs.owner
    pins = referenced_pins(presets_file, expanded)
    decisions: set[str] = set()
    listed: set[str] = set()
    rows_out = []
    for row in scope.rows:
        entry: dict[str, Any] = {"id": row.id, "kind": row.kind, "prio": row.prio, "enabled": row.enabled,
                                 "why": row.why, "plan": row.plan}
        if isinstance(row, (ReferencesRow, CalibrationRow)):
            entry["owner_unset"] = placeholders(row.model_dump(mode="json", exclude=_ROW_META), row.id)
            if row.enabled:
                owner.update(entry["owner_unset"])
        if isinstance(row, CellsRow):
            blocks, cells_total, hours_total, complete = [], 0, 0.0, row.enabled
            for block in row.blocks:
                bp = next((b for b in plans if b.block is block), None)         # default-ok: disabled blocks
                if bp is None:
                    blocks.append({"name": block.name, "enabled": False, "why": block.why})
                    continue
                needs = Needs()
                needs.update(bp.needs)
                for ref in ([] if block.hold is not None else bp.configs):
                    needs.update(render_config(expanded[ref.preset_id], resolver)[1])
                seeds_known = block.seeds != PLACEHOLDER
                if seeds_known:
                    _list_cells(bp, listed)
                minutes, known = 0.0, seeds_known
                for tid in block.traces:
                    t = scope.traces[tid]
                    if placeholders(t):
                        known = False
                        continue
                    per = trace_minutes(parse(Trace, t, f"trace {tid}"))
                    n_undecided = bp.undecided * (len(block.seeds) if seeds_known else 0)
                    minutes += per * (sum(1 for _, t2, _ in bp.cells if t2 == tid) + n_undecided)
                known = known and bp.undecided == 0
                n = len(bp.cells) + bp.undecided * len(block.traces) * (len(block.seeds) if seeds_known else 0)
                owner |= needs.owner
                pins |= needs.pins
                decisions |= needs.decisions
                blocks.append({"name": block.name, "enabled": True, "group": block.group, "cells": n,
                               "cell_ids": [cell_id(r.config_id, t2, s) for r, t2, s in bp.cells],
                               "undecided_configs": bp.undecided, "hours": round(minutes / 60.0, 2),
                               "estimate_complete": known,
                               "hold": block.hold, "needs": {"owner": sorted(needs.owner), "pins": sorted(needs.pins),
                                         "decisions": sorted(needs.decisions)}})
                cells_total += n
                hours_total += minutes / 60.0
                complete = complete and known
            entry.update(blocks=blocks, cells=cells_total, hours=round(hours_total, 2), estimate_complete=complete)
        rows_out.append(entry)
    return {"configurations": len(expanded), "rows": rows_out, "owner_unset": sorted(owner),
            "pins_needed": sorted(pins), "decisions_needed": sorted(decisions),
            "estimate": {"v3_minutes_per_3534": {str(k): v for k, v in V3_MINUTES.items()},
                         "driver_minutes_per_cell": DRIVER_MINUTES}}


def dry_run_text(summary: dict) -> str:
    lines = [f"{summary['configurations']} configurations defined in the presets"]
    for r in summary["rows"]:
        head = f"{r['id']:<6} {r['kind']:<12} {r['prio']:<18}"
        if not r["enabled"]:
            lines.append(f"{head} DISABLED: {r['why']}")
            continue
        if r["kind"] != "cells":
            unset = r.get("owner_unset")
            lines.append(f"{head} plan: {r['plan']}" + (f"   unset: {unset}" if unset else ""))
            continue
        est = "" if r["estimate_complete"] else " (partial: undecided axes, seeds or traces)"
        lines.append(f"{head} {r['cells']:>4} cells {r['hours']:6.2f} h{est}   plan: {r['plan']}")
        for b in r["blocks"]:
            if not b["enabled"]:
                lines.append(f"       block {b['name']}: DISABLED: {b['why']}")
                continue
            more = f" ({b['undecided_configs']} configurations undecided)" if b["undecided_configs"] else ""
            more += f"   HELD: {b['hold']}" if b["hold"] else ""
            needs = "; ".join(f"{k} {', '.join(v)}" for k, v in b["needs"].items() if v)
            lines.append(f"       block {b['name']:<20} {b['cells']:>4} cells {b['hours']:6.2f} h{more}"
                         + (f"   needs {needs}" if needs else ""))
    lines.append(f"owner values still {PLACEHOLDER} ({len(summary['owner_unset'])}):")
    lines += [f"  {p}" for p in summary["owner_unset"]]
    lines.append(f"pins the configurations reference ({len(summary['pins_needed'])}): "
                 f"{', '.join(summary['pins_needed'])}")
    lines.append(f"decisions made during the lease: {', '.join(summary['decisions_needed'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------- command line

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fleet_args(values: Sequence[str], hosts: Optional[Path]) -> dict[str, tuple[str, str]]:
    out = {}
    for v in values:
        name, sep, path = v.partition("=")
        if not sep or not name or not path:
            raise ConfigError(f"--fleet takes NAME=FILE, got {v!r}")
        if hosts is None:
            raise ConfigError("--fleet needs --hosts: the fleet digest includes the lease addresses")
        out[name] = (_sha256(Path(path)), fleet_digest(load_fleet(Path(path), hosts)))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run")
    r = sub.add_parser("render")
    for p in (d, r):
        p.add_argument("--presets", type=Path, default=PRESETS)
        p.add_argument("--scope", type=Path, default=SCOPE)
    d.add_argument("--json", type=Path, help="also write the summary as JSON")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--pins", type=Path)
    r.add_argument("--decisions", type=Path)
    r.add_argument("--fleet", action="append", default=[], help="NAME=FLEET.json for every fleet to render")
    r.add_argument("--hosts", type=Path)
    r.add_argument("--previous", type=Path, help="the previous render: none of its configurations or cells may change")
    r.add_argument("--require-complete", action="store_true", help="exit 1 if anything is pending or blocked")
    a = ap.parse_args(argv)
    presets, scope = load_presets(a.presets), load_scope_file(a.scope)
    if a.cmd == "dry-run":
        summary = dry_run(presets, scope)
        print(dry_run_text(summary))
        if a.json is not None:
            a.json.write_text(json.dumps(summary, indent=1) + "\n")
        return 0
    pins = {} if a.pins is None else read_json(a.pins)
    decisions = {} if a.decisions is None else read_json(a.decisions)
    if not isinstance(pins, dict) or not isinstance(decisions, dict):
        raise ConfigError("the pins and decisions files are JSON objects")
    fleets = _fleet_args(a.fleet, a.hosts)
    result = render(presets, scope, pins=pins, decisions=decisions, fleets=fleets)
    if a.previous is not None:
        check_previous(result, a.previous)
    inputs = {"presets_sha256": _sha256(a.presets), "scope_sha256": _sha256(a.scope), "pins": pins,
              "decisions": decisions, "hosts_sha256": None if a.hosts is None else _sha256(a.hosts),
              "fleets": {n: {"sha256": s, "fleet_digest": fd} for n, (s, fd) in fleets.items()},
              "previous": None if a.previous is None else str(a.previous)}
    write_render(result, a.out, inputs)
    n = 0 if result.scope is None else sum(len(b["seeds"]) for b in result.scope["blocks"])
    print(f"{a.out}: {len(result.configs)} configurations, {n} cells"
          + ("" if result.scope is not None else " (no scope.json yet: only the direct rows can run)"))
    for where, reason in result.pending.items():
        print(f"  PENDING {where}: {reason}")
    for row, reason in result.blocked_rows.items():
        print(f"  BLOCKED {row}: {reason}")
    return 1 if a.require_complete and (result.pending or result.blocked_rows) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
