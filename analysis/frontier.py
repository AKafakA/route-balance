"""Frontier points per system per trace.

Everything is compared within one comparison group (bench/scope.py: one dataset, prompt order and fleet) and
one trace; cells of different groups are never pooled. A configuration's point is the mean over its seeds of
each per-seed cell metric (quality, mean and p99 end-to-end latency, decode cost and total cost, the SLO
miss shares at the
tight and loose levels (1 - attainment, analysis/slo.py), throughput);
`seeds` says which seeds. Only cells that are ok in the manifest and valid in their summary enter.
A configuration is on its system's frontier for a metric when no other configuration of the same system,
group and trace has at least its quality and at most its metric value, with one of the two strict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from analysis.manifest import Coverage
from analysis.metrics import CellSummary
from bench.scope import ExpectedCell
from routebalance.config import ConfigError

METRICS = ("e2e_mean_ms", "e2e_p99_ms", "cost_decode_per_1k", "cost_total_per_1k", "slo_miss_tight",
           "slo_miss_loose")   # lower is better; total cost (prompt + output) beside the decode cost
_AVERAGED = ("quality", *METRICS, "throughput_rps")
Key = tuple[str, str, str]


@dataclass(frozen=True)
class CellPoint:
    """One valid cell reduced to the numbers the comparisons use."""
    group: str
    config_id: str
    system: str
    knob: str
    dispatch: str
    trace_id: str
    seed: int
    quality: float
    e2e_mean_ms: float
    e2e_p99_ms: float
    cost_decode_per_1k: float
    cost_total_per_1k: float
    slo_miss_tight: float
    slo_miss_loose: float
    throughput_rps: float


@dataclass(frozen=True)
class ConfigPoint:
    group: str
    config_id: str
    system: str
    knob: str
    dispatch: str
    trace_id: str
    seeds: tuple[int, ...]
    quality: float
    e2e_mean_ms: float
    e2e_p99_ms: float
    cost_decode_per_1k: float
    cost_total_per_1k: float
    slo_miss_tight: float
    slo_miss_loose: float
    throughput_rps: float


def cell_point(exp: ExpectedCell, s: CellSummary, coverage: Coverage) -> CellPoint:
    if s.cell_id != exp.cell_id:
        raise ConfigError(f"summary {s.cell_id} paired with expected cell {exp.cell_id}")
    if exp.cell_id not in coverage.ok:
        raise ConfigError(f"cell {exp.cell_id} is not ok in the manifest and cannot enter a comparison")
    if exp.group in coverage.fleet_problems:
        raise ConfigError(f"group {exp.group}: {coverage.fleet_problems[exp.group]}; its cells cannot be compared")
    if not s.valid:
        raise ConfigError(f"cell {s.cell_id} is invalid and cannot enter a comparison: {list(s.problems)}")
    if s.slo_attainment_tight is None or s.slo_attainment_loose is None:
        raise ConfigError(f"cell {s.cell_id} was summarised without its SLO")
    if s.cost_total_per_1k is None:
        raise ConfigError(f"cell {s.cell_id} was summarised without input prices")
    return CellPoint(group=exp.group, config_id=exp.config_id, system=exp.config.system, knob=exp.config.knob,
                     dispatch=exp.config.dispatch, trace_id=exp.trace_id, seed=exp.seed, quality=s.quality,
                     e2e_mean_ms=s.e2e_ms.mean, e2e_p99_ms=s.e2e_ms.p99, cost_decode_per_1k=s.cost_decode_per_1k,
                     cost_total_per_1k=s.cost_total_per_1k,
                     slo_miss_tight=1.0 - s.slo_attainment_tight, slo_miss_loose=1.0 - s.slo_attainment_loose,
                     throughput_rps=s.throughput_rps)


def by_config(points: Iterable[CellPoint]) -> dict[Key, dict[int, CellPoint]]:
    """{(group, trace_id, config_id): {seed: point}}; a repeated cell or a relabelled config is an error."""
    table: dict[Key, dict[int, CellPoint]] = {}
    for p in points:
        per_seed = table.setdefault((p.group, p.trace_id, p.config_id), {})  # default-ok: grouping into a new dict
        if p.seed in per_seed:
            raise ConfigError(f"two cells for {p.config_id} at {p.group}/{p.trace_id}, seed {p.seed}")
        first = next(iter(per_seed.values()), None)
        if first is not None and (first.system, first.knob, first.dispatch) != (p.system, p.knob, p.dispatch):
            raise ConfigError(f"config {p.config_id} carries two different labels")
        per_seed[p.seed] = p
    return table


def config_points(points: Iterable[CellPoint]) -> list[ConfigPoint]:
    out = []
    for (group, trace_id, config_id), per_seed in sorted(by_config(points).items()):
        seeds = tuple(sorted(per_seed))
        first = per_seed[seeds[0]]
        means = {f: float(np.mean([getattr(per_seed[s], f) for s in seeds])) for f in _AVERAGED}
        out.append(ConfigPoint(group=group, config_id=config_id, system=first.system, knob=first.knob,
                               dispatch=first.dispatch, trace_id=trace_id, seeds=seeds, **means))
    return out


def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """a = (quality, metric) has at least b's quality and at most b's metric, and is not b."""
    return a[0] >= b[0] and a[1] <= b[1] and a != b


def pareto(pairs: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of the pairs (quality, metric) that no other pair dominates (quality up, metric down),
    sorted by quality, then metric."""
    efficient = [i for i, p in enumerate(pairs) if not any(_dominates(other, p) for other in pairs)]
    return sorted(efficient, key=lambda i: pairs[i])


def frontiers(points: Sequence[ConfigPoint], metric: str) -> dict[Key, list[ConfigPoint]]:
    """{(group, trace_id, system): the system's frontier configurations for `metric`, by increasing quality}."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    groups: dict[Key, list[ConfigPoint]] = {}
    for p in points:
        groups.setdefault((p.group, p.trace_id, p.system), []).append(p)  # default-ok: grouping into a new list
    return {key: [grp[i] for i in pareto([(p.quality, getattr(p, metric)) for p in grp])]
            for key, grp in sorted(groups.items())}

