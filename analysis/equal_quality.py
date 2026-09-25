"""Equal-quality comparison: every baseline configuration against the treatment at the same quality.

For a baseline configuration B at one trace of one comparison group (bench/scope.py; the treatment presets
come from the same group and trace, never from another fleet, dataset or order), over B's seeds S (paired: a
seed fixes the arrivals, and with per_seed order also the prompt order, for every system):
  1. B's quality q_B and metric value m_B are its means over S.
  2. The treatment presets used are those with a cell for every seed in S (the rest are listed as excluded);
     each preset's point is its mean over S.
  3. For each metric (mean E2E, p99 E2E, decode cost, the tight and loose SLO miss shares), the treatment's
     frontier is its Pareto-efficient presets (quality up, metric down), by increasing quality. The treatment's
     value at q_B is the linear
     interpolation between the two frontier presets around q_B. Past either end, the end preset is used if its
     quality is within the tolerance of q_B; otherwise there is no equal-quality point (never extrapolated).
  4. delta = treatment - baseline (negative: the treatment is faster or cheaper); relative = treatment/baseline - 1.
  5. CI: percentile bootstrap over seeds. A resample draws |S| seeds with replacement, the same draw for B and
     every preset, and repeats 1-4. There is a CI only with >= 2 seeds and when every resample has an
     equal-quality point (the count of those that do not is reported).
The presets whose own quality is within the tolerance of q_B are listed too (`matched_presets`).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from analysis.frontier import METRICS, CellPoint, by_config, pareto
from analysis.rules import EqualQuality
from routebalance.config import ConfigError


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    baseline: float
    treatment: Optional[float]           # None: no treatment point at the baseline's quality
    delta: Optional[float]
    relative: Optional[float]
    ci: Optional[tuple[float, float]]
    undefined_resamples: int


@dataclass(frozen=True)
class EqualQualityRow:
    group: str
    trace_id: str
    config_id: str
    system: str
    seeds: tuple[int, ...]
    quality: float
    matched_presets: tuple[str, ...]
    excluded_presets: tuple[str, ...]
    deltas: tuple[MetricDelta, ...]


def value_at(quality: float, frontier: Sequence[tuple[float, float]], tolerance: float) -> Optional[float]:
    """The frontier's metric value at `quality` (frontier: (quality, metric) by increasing quality)."""
    if not frontier:
        return None
    (q_lo, m_lo), (q_hi, m_hi) = frontier[0], frontier[-1]
    if quality <= q_lo:
        return m_lo if q_lo - quality <= tolerance else None
    if quality >= q_hi:
        return m_hi if quality - q_hi <= tolerance else None
    i = bisect.bisect_right([q for q, _ in frontier], quality)       # q_lo < quality < q_hi, so 1 <= i < len
    (q0, m0), (q1, m1) = frontier[i - 1], frontier[i]
    return m0 + (m1 - m0) * (quality - q0) / (q1 - q0)


def _frontier(qualities: Sequence[float], values: Sequence[float]) -> list[tuple[float, float]]:
    pairs = list(zip(qualities, values, strict=True))
    return [pairs[i] for i in pareto(pairs)]


class _Draws:
    """Per-seed values of the configurations in one comparison group, their means over the seeds, and their
    means over each bootstrap resample of the seeds."""

    def __init__(self, table: dict, where: tuple[str, str], seeds: tuple[int, ...], rules: EqualQuality):
        self._table, self._where, self._seeds = table, where, seeds
        k = len(seeds)
        rng = np.random.default_rng(rules.bootstrap_seed)
        self.idx = rng.integers(0, k, size=(rules.bootstrap_resamples, k)) if k >= 2 else None

    def values(self, config_id: str, field: str) -> np.ndarray:
        per_seed = self._table[(*self._where, config_id)]
        return np.array([getattr(per_seed[s], field) for s in self._seeds], dtype=float)

    def mean(self, config_id: str, field: str) -> float:
        return float(np.mean(self.values(config_id, field)))

    def resampled(self, config_id: str, field: str) -> np.ndarray:
        return self.values(config_id, field)[self.idx].mean(axis=1)


class _Treatment:
    """The treatment's frontiers for one (group, trace, seeds, presets): at the means, and per resample."""

    def __init__(self, draws: _Draws, presets: tuple[str, ...]):
        self.presets = presets
        self.quality = [draws.mean(p, "quality") for p in presets]
        self.point = {m: _frontier(self.quality, [draws.mean(p, m) for p in presets]) for m in METRICS}
        self.boot: dict[str, list[list[tuple[float, float]]]] = {}
        if draws.idx is not None:
            q = np.array([draws.resampled(p, "quality") for p in presets])              # presets x resamples
            for m in METRICS:
                v = np.array([draws.resampled(p, m) for p in presets])
                self.boot[m] = [_frontier(q[:, b], v[:, b]) for b in range(q.shape[1])]


def equal_quality(points: Iterable[CellPoint], treatment: str, rules: EqualQuality) -> list[EqualQualityRow]:
    """One row per baseline configuration, group and trace; the treatment is the system named `treatment`."""
    table = by_config(points)
    system_of = {key: next(iter(per_seed.values())).system for key, per_seed in table.items()}
    sides: dict[tuple, tuple[_Draws, _Treatment]] = {}
    rows = []
    for (group, trace_id, config_id), per_seed in sorted(table.items()):
        if system_of[(group, trace_id, config_id)] == treatment:
            continue
        where = (group, trace_id)
        seeds = tuple(sorted(per_seed))
        presets = sorted(c for (g, t, c), s in system_of.items() if (g, t) == where and s == treatment)
        eligible = tuple(c for c in presets if set(seeds) <= set(table[(*where, c)]))
        if not eligible:
            raise ConfigError(f"no {treatment} preset at {group}/{trace_id} has cells for every seed of "
                              f"{config_id} {list(seeds)}")
        key = (where, seeds, eligible)
        if key not in sides:
            draws = _Draws(table, where, seeds, rules)
            sides[key] = (draws, _Treatment(draws, eligible))
        draws, side = sides[key]
        rows.append(_row(where, config_id, system_of[(*where, config_id)], seeds, draws, side,
                         [c for c in presets if c not in eligible], rules))
    return rows


def _row(where: tuple[str, str], config_id: str, system: str, seeds: tuple[int, ...], draws: _Draws,
         side: _Treatment, excluded: list[str], rules: EqualQuality) -> EqualQualityRow:
    tol = rules.tolerance
    q_b = draws.mean(config_id, "quality")
    q_boot = draws.resampled(config_id, "quality") if draws.idx is not None else None
    deltas = []
    for m in METRICS:
        base = draws.mean(config_id, m)
        at = value_at(q_b, side.point[m], tol)
        ci, undefined = None, 0
        if q_boot is not None:
            m_boot = draws.resampled(config_id, m)
            boot = [value_at(q_boot[b], side.boot[m][b], tol) for b in range(len(q_boot))]
            undefined = sum(v is None for v in boot)
            if not undefined:
                d = np.array(boot) - m_boot
                lo, hi = np.percentile(d, [50 * (1 - rules.confidence), 50 * (1 + rules.confidence)])
                ci = (float(lo), float(hi))
        deltas.append(MetricDelta(metric=m, baseline=base, treatment=at,
                                  delta=None if at is None else at - base,
                                  relative=None if at is None or base == 0 else at / base - 1,
                                  ci=ci, undefined_resamples=undefined))
    matched = tuple(p for p, q in zip(side.presets, side.quality, strict=True) if abs(q - q_b) <= tol)
    return EqualQualityRow(group=where[0], trace_id=where[1], config_id=config_id, system=system, seeds=seeds,
                           quality=q_b, matched_presets=matched, excluded_presets=tuple(excluded),
                           deltas=tuple(deltas))
