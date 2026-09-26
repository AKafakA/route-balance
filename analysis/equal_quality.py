"""Equal-quality comparison: every baseline configuration against the treatment at the same quality (D80, D81).

For a baseline configuration B at one trace of one comparison group (bench/scope.py; the treatment presets
come from the same group and trace, never from another fleet, dataset or order), over B's seeds S (paired: a
seed fixes the arrivals, and with per_seed order also the prompt order, for every system):
  1. B's quality q_B and metric values are its means over S.
  2. The treatment presets used are those with a cell for every seed in S (the rest are listed as excluded);
     each preset's point is its mean over S.
  3. One treatment configuration for every metric (D80): the treatment's frontier is its presets that are
     Pareto-efficient in quality (up) and mean E2E latency (down), by increasing quality. At q_B the treatment's
     configuration lies between the two frontier presets around q_B, at weight w = (q_B - q_lo) / (q_hi - q_lo),
     and every metric (mean and p99 E2E, decode cost, the tight and loose SLO miss shares) is read there:
     m_lo + w (m_hi - m_lo). Past either end the end preset is used if its quality is within the tolerance of
     q_B; otherwise there is no equal-quality point (never extrapolated).
  4. delta = treatment - baseline (negative: the treatment is faster or cheaper); relative = treatment/baseline - 1.
  5. The range over the seeds (D81, the main text): steps 3-4 on each seed of S alone, B's cell at that seed
     against the frontier of the presets' cells at that seed. `per_seed` holds each seed's delta and relative
     change; `range` and `range_relative` are their minimum and maximum, when S has >= 2 seeds and every seed
     has the value. `outside_range` marks a main value outside its range (the frontier of the means is not the
     mean of the seeds' frontiers).
  6. The confidence interval (D80, the appendix): a percentile bootstrap over requests within the paired seeds.
     A resample draws, in each seed, as many requests as the seed's cells have, with replacement, by send
     position, the same draw for B and every preset: the same position is the same prompt and arrival in every
     system's cell (checked). Each cell's metrics are recomputed on the drawn requests
     (analysis/metrics.py RequestValues), then steps 1-4; `ci` and `ci_relative` are the percentile intervals
     of the resampled deltas and of the resampled relative changes. A resample in which a compared cell drew
     no scored request (only corpus gaps) has no quality for that cell and no equal-quality point. There is a
     CI only when the cells' requests are given and every resample has an equal-quality point (the count of
     those that do not is reported); no ci_relative when a resampled baseline is 0.
The presets whose own quality is within the tolerance of q_B are listed too (`matched_presets`), and so are the
two frontier presets and the weight of step 3 (`frontier_presets`, `weight`).
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from analysis.frontier import METRICS, CellPoint, by_config, pareto
from analysis.metrics import RequestValues
from analysis.rules import EqualQuality
from routebalance.config import ConfigError

FRONTIER_METRIC = "e2e_mean_ms"
FIELDS = ("quality", *METRICS)
SampleKey = tuple[str, str, str, int]        # (group, trace_id, config_id, seed)
_EPS = 1e-12


@dataclass(frozen=True)
class SeedValue:
    seed: int
    delta: Optional[float]                   # None: no equal-quality point on this seed
    relative: Optional[float]


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    baseline: float
    treatment: Optional[float]           # None: no treatment point at the baseline's quality
    delta: Optional[float]
    relative: Optional[float]
    per_seed: tuple[SeedValue, ...]
    range: Optional[tuple[float, float]]
    range_relative: Optional[tuple[float, float]]
    outside_range: bool
    ci: Optional[tuple[float, float]]
    ci_relative: Optional[tuple[float, float]]
    undefined_resamples: int


@dataclass(frozen=True)
class EqualQualityRow:
    group: str
    trace_id: str
    config_id: str
    system: str
    seeds: tuple[int, ...]
    quality: float
    frontier_presets: Optional[tuple[str, str]]     # step 3's two presets (the same one twice at an end)
    weight: Optional[float]
    matched_presets: tuple[str, ...]
    excluded_presets: tuple[str, ...]
    deltas: tuple[MetricDelta, ...]


def position_at(quality: float, qualities: Sequence[float],
                tolerance: float) -> Optional[tuple[int, int, float]]:
    """(i, j, w): `quality` lies between the i-th and j-th of `qualities` (increasing) at weight w."""
    if math.isnan(quality) or any(math.isnan(q) for q in qualities):
        raise ConfigError("an equal-quality reading was given a quality that is not a number")
    if not qualities:
        return None
    last = len(qualities) - 1
    if quality <= qualities[0]:
        return (0, 0, 0.0) if qualities[0] - quality <= tolerance else None
    if quality >= qualities[last]:
        return (last, last, 0.0) if quality - qualities[last] <= tolerance else None
    i = bisect.bisect_right(list(qualities), quality)         # q[0] < quality < q[last], so 1 <= i <= last
    return i - 1, i, (quality - qualities[i - 1]) / (qualities[i] - qualities[i - 1])


def value_at(quality: float, frontier: Sequence[tuple[float, float]], tolerance: float) -> Optional[float]:
    """The frontier's metric value at `quality` (frontier: (quality, metric) by increasing quality)."""
    pos = position_at(quality, [q for q, _ in frontier], tolerance)
    if pos is None:
        return None
    i, j, w = pos
    return frontier[i][1] + (frontier[j][1] - frontier[i][1]) * w


@dataclass(frozen=True)
class _Frontier:
    """The presets on the quality-latency frontier (indices, by increasing quality) and their qualities."""
    order: tuple[int, ...]
    quality: tuple[float, ...]

    @classmethod
    def of(cls, quality: Sequence[float], latency: Sequence[float]) -> "_Frontier":
        order = pareto([(float(q), float(m)) for q, m in zip(quality, latency, strict=True)])
        return cls(tuple(order), tuple(float(quality[i]) for i in order))

    def at(self, quality: float, tolerance: float) -> Optional[tuple[int, int, float]]:
        pos = position_at(quality, self.quality, tolerance)
        return None if pos is None else (self.order[pos[0]], self.order[pos[1]], pos[2])


def _read(pos: Optional[tuple[int, int, float]], values: np.ndarray) -> Optional[float]:
    if pos is None:
        return None
    i, j, w = pos
    return float(values[i] + (values[j] - values[i]) * w)


def _change(at: Optional[float], base: float) -> tuple[Optional[float], Optional[float]]:
    if at is None:
        return None, None
    return at - base, (None if base == 0 else at / base - 1)


class _Side:
    """The treatment presets at one (group, trace, seeds): their metric values at the means over the seeds, at
    each seed, and at each request resample, with the frontier of each."""

    def __init__(self, table: dict, where: tuple[str, str], seeds: tuple[int, ...], presets: tuple[str, ...],
                 draws: Optional[dict[int, np.ndarray]], samples: Optional[Mapping[SampleKey, RequestValues]]):
        self.presets = presets
        cells = [table[(*where, p)] for p in presets]
        self.seed = {s: {f: np.array([getattr(c[s], f) for c in cells]) for f in FIELDS} for s in seeds}
        self.mean = {f: np.mean([self.seed[s][f] for s in seeds], axis=0) for f in FIELDS}
        self.front = _Frontier.of(self.mean["quality"], self.mean[FRONTIER_METRIC])
        self.front_seed = {s: _Frontier.of(v["quality"], v[FRONTIER_METRIC]) for s, v in self.seed.items()}
        self.boot: Optional[dict[str, np.ndarray]] = None             # field -> presets x resamples
        self.front_boot: list[Optional[_Frontier]] = []          # None: a preset drew no scored request
        if draws is not None and samples is not None:
            per = [_resampled(where, p, seeds, draws, samples) for p in presets]
            self.boot = {f: np.array([r[f] for r in per]) for f in FIELDS}
            self.front_boot = [None if np.isnan(self.boot["quality"][:, b]).any() else
                               _Frontier.of(self.boot["quality"][:, b], self.boot[FRONTIER_METRIC][:, b])
                               for b in range(self.boot["quality"].shape[1])]


def _resampled(where: tuple[str, str], config_id: str, seeds: tuple[int, ...], draws: Mapping[int, np.ndarray],
               samples: Mapping[SampleKey, RequestValues]) -> dict[str, np.ndarray]:
    """Each field of the configuration, the mean over the seeds of its cells' values on each request resample."""
    per_seed = [samples[(*where, config_id, s)].at(draws[s]) for s in seeds]
    return {f: np.mean([v[f] for v in per_seed], axis=0) for f in FIELDS}


def _draws(where: tuple[str, str], seeds: tuple[int, ...], samples: Mapping[SampleKey, RequestValues],
           reference: str, rules: EqualQuality) -> dict[int, np.ndarray]:
    """seed -> resamples x requests: the send positions each resample draws, for every configuration at
    (group, trace) over these seeds; the sizes are the reference configuration's cells'."""
    rng = np.random.default_rng(rules.bootstrap_seed)
    return {s: rng.integers(0, len(samples[(*where, reference, s)].index),
                            size=(rules.bootstrap_resamples, len(samples[(*where, reference, s)].index)))
            for s in seeds}


def _check_paired(where: tuple[str, str], seeds: tuple[int, ...], configs: Sequence[str],
                  samples: Mapping[SampleKey, RequestValues]) -> None:
    """Every configuration's cell at a seed holds the same requests (send positions and prompts)."""
    for s in seeds:
        missing = [c for c in configs if (*where, c, s) not in samples]
        if missing:
            raise ConfigError(f"{where[0]}/{where[1]} seed {s}: no per-request values for {missing}")
        first = samples[(*where, configs[0], s)]
        for c in configs[1:]:
            other = samples[(*where, c, s)]
            if other.index != first.index or other.prompt_ids != first.prompt_ids:
                raise ConfigError(f"{where[0]}/{where[1]} seed {s}: the cells of {configs[0]} and {c} do not hold "
                                  "the same requests, so the request bootstrap cannot pair them")


def equal_quality(points: Iterable[CellPoint], treatment: str, rules: EqualQuality,
                  samples: Optional[Mapping[SampleKey, RequestValues]] = None) -> list[EqualQualityRow]:
    """One row per baseline configuration, group and trace; the treatment is the system named `treatment`.
    `samples` holds every compared cell's per-request values, for the request bootstrap (none: no CI)."""
    table = by_config(points)
    system_of = {key: next(iter(per_seed.values())).system for key, per_seed in table.items()}
    sides: dict[tuple, _Side] = {}
    drawn: dict[tuple, tuple[str, dict[int, np.ndarray]]] = {}   # (reference, draws)
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
        draws = None
        if samples is not None:
            reference = drawn[(where, seeds)][0] if (where, seeds) in drawn else eligible[0]
            _check_paired(where, seeds, (reference, *eligible, config_id), samples)
            if (where, seeds) not in drawn:
                drawn[(where, seeds)] = (reference, _draws(where, seeds, samples, reference, rules))
            draws = drawn[(where, seeds)][1]
        key = (where, seeds, eligible)
        if key not in sides:
            sides[key] = _Side(table, where, seeds, eligible, draws, samples)
        base_boot = None if draws is None else _resampled(where, config_id, seeds, draws, samples)
        rows.append(_row(where, config_id, system_of[(*where, config_id)], seeds, per_seed, sides[key], base_boot,
                         [c for c in presets if c not in eligible], rules))
    return rows


def _row(where: tuple[str, str], config_id: str, system: str, seeds: tuple[int, ...],
         per_seed: Mapping[int, CellPoint], side: _Side, base_boot: Optional[dict[str, np.ndarray]],
         excluded: list[str], rules: EqualQuality) -> EqualQualityRow:
    tol = rules.tolerance
    q_b = float(np.mean([per_seed[s].quality for s in seeds]))
    pos = side.front.at(q_b, tol)
    pos_seed = {s: side.front_seed[s].at(per_seed[s].quality, tol) for s in seeds}
    pos_boot = None
    if base_boot is not None and side.boot is not None:
        pos_boot = [None if f is None or math.isnan(base_boot["quality"][b]) else f.at(float(base_boot["quality"][b]), tol)
                    for b, f in enumerate(side.front_boot)]
    deltas = []
    for m in METRICS:
        base = float(np.mean([getattr(per_seed[s], m) for s in seeds]))
        at = _read(pos, side.mean[m])
        delta, relative = _change(at, base)
        seed_values = tuple(SeedValue(s, *_change(_read(pos_seed[s], side.seed[s][m]), getattr(per_seed[s], m)))
                            for s in seeds)
        rng_d = _span([v.delta for v in seed_values]) if len(seeds) >= 2 else None
        rng_r = _span([v.relative for v in seed_values]) if len(seeds) >= 2 else None
        ci, ci_rel, undefined = None, None, 0
        if pos_boot is not None:
            m_boot = base_boot[m]
            boot = [_read(p, side.boot[m][:, b]) for b, p in enumerate(pos_boot)]
            undefined = sum(v is None for v in boot)
            if not undefined:
                ci = _interval(np.array(boot) - m_boot, rules.confidence)
                if np.all(m_boot != 0):
                    ci_rel = _interval(np.array(boot) / m_boot - 1, rules.confidence)
        deltas.append(MetricDelta(metric=m, baseline=base, treatment=at, delta=delta, relative=relative,
                                  per_seed=seed_values, range=rng_d, range_relative=rng_r,
                                  outside_range=any((_outside(delta, rng_d), _outside(relative, rng_r))),
                                  ci=ci, ci_relative=ci_rel, undefined_resamples=undefined))
    matched = tuple(p for p, q in zip(side.presets, side.mean["quality"], strict=True) if abs(q - q_b) <= tol)
    return EqualQualityRow(group=where[0], trace_id=where[1], config_id=config_id, system=system, seeds=seeds,
                           quality=q_b, frontier_presets=None if pos is None else
                           (side.presets[pos[0]], side.presets[pos[1]]),
                           weight=None if pos is None else pos[2], matched_presets=matched,
                           excluded_presets=tuple(excluded), deltas=tuple(deltas))


def _span(values: Sequence[Optional[float]]) -> Optional[tuple[float, float]]:
    if any(v is None for v in values):
        return None
    return float(min(values)), float(max(values))


def _outside(value: Optional[float], span: Optional[tuple[float, float]]) -> bool:
    return value is not None and span is not None and not span[0] - _EPS <= value <= span[1] + _EPS


def _interval(x: np.ndarray, confidence: float) -> tuple[float, float]:
    lo, hi = np.percentile(x, [50 * (1 - confidence), 50 * (1 + confidence)])
    return float(lo), float(hi)
