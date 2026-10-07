"""The SLO (after SOLA): per-request attainment against each tier's single-request latency, and goodput.

A request meets its SLO at a level ("tight" or "loose") when
  TTFT <= k * its tier's single-request TTFT   and   TPOT <= k * its tier's single-request TPOT,
with k the level's value for the served model (configs/prereg_rules.json "slo") and the single-request values
measured on the lease stack in W1 at b = 1 (tools/slo_base.py). On the record's clocks:
  TTFT  first_token_s, the scheduler's clock from the request's arrival to its first token (its queue and decision,
        RouteBalance's round wait, then the engine's TTFT), as v2's client_ttft; the reply's delivery after the last
        token is not counted
  TPOT  (server_latency - ttft) / (output_tokens - 1), the engine's time per output token; a one-token reply
        has no TPOT and is judged on its TTFT alone (vLLM's benchmark defines TPOT from two tokens on)
Attainment: the share of the cell's requests that meet the SLO; a failed request never does. The comparisons
use its complement, the miss share (lower is better, as every other metric of analysis/frontier.py).
Goodput at an attainment level (90%, 99%): over one configuration's stationary traces in increasing rate, the
largest rate before its attainment first falls below the level, linearly interpolated between that rate and
the one before; None when the lowest rate already falls below it, and the highest rate, marked censored, when
no rate does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from bench.schema import Strict, parse, read_json
from routebalance.config import ConfigError
from routebalance.types import Instance

if TYPE_CHECKING:
    from analysis.frontier import ConfigPoint

LEVELS = ("tight", "loose")


class SloBase(Strict):
    """tools/slo_base.py's output: each tier's single-request TTFT and TPOT in seconds."""
    unit: str
    ttft_s: dict[str, float]
    tpot_s: dict[str, float]
    provenance: dict

    @classmethod
    def load(cls, path: Path) -> "SloBase":
        base = parse(cls, read_json(path), f"SLO bases {path}")
        if base.unit != "s" or set(base.ttft_s) != set(base.tpot_s):
            raise ConfigError(f"{path}: SLO bases need unit 's' and the same tiers for TTFT and TPOT")
        if not all(v > 0 and math.isfinite(v) for d in (base.ttft_s, base.tpot_s) for v in d.values()):
            raise ConfigError(f"{path}: every SLO base is a positive number of seconds")
        return base


@dataclass(frozen=True)
class Limits:
    ttft_s: float
    tpot_s: float


@dataclass(frozen=True)
class CellSlo:
    """The TTFT and TPOT limits of every instance of the cell's fleet, at each level."""
    limits: Mapping[str, Mapping[str, Limits]]          # level -> instance_id -> limits

    @classmethod
    def of(cls, k: Mapping[str, Mapping[str, float]], base: SloBase, fleet: Sequence[Instance]) -> "CellSlo":
        limits: dict[str, dict[str, Limits]] = {level: {} for level in LEVELS}
        for inst in fleet:
            tier, model = inst.tier.tier_id, inst.tier.model
            if tier not in base.ttft_s:
                raise ConfigError(f"tier {tier} has no single-request latency in the SLO bases")
            for level in LEVELS:
                if model not in k[level]:
                    raise ConfigError(f"model {model} has no {level} SLO multiplier in the rules")
                f = k[level][model]
                limits[level][inst.instance_id] = Limits(f * base.ttft_s[tier], f * base.tpot_s[tier])
        return cls(limits)

    def instances(self) -> frozenset[str]:
        return frozenset(i for per in self.limits.values() for i in per)

    def met(self, level: str, instance_id: str, first_token_s: float, server_latency_s: float,
            ttft_s: float, output_tokens: int) -> bool:
        """first_token_s: the scheduler's clock from the request's arrival to its first token (v2's client_ttft); no
        longer reconstructed as E2E minus the decode, which also counted the reply's delivery after the last token."""
        lim = self.limits[level].get(instance_id)
        if lim is None:
            raise ConfigError(f"instance {instance_id} is not in the cell's fleet")
        decode_s = server_latency_s - ttft_s
        if first_token_s > lim.ttft_s:
            return False
        if output_tokens < 2:
            return True                                  # a one-token reply has no TPOT
        return decode_s / (output_tokens - 1) <= lim.tpot_s


@dataclass(frozen=True)
class Goodput:
    rate: Optional[float]           # req/s; None: the lowest rate already falls below the level
    censored: bool                  # attainment held at the highest rate: the goodput is at least `rate`


def goodput(attainment_by_rate: Sequence[tuple[float, float]], level: float) -> Goodput:
    """See the module docstring. attainment_by_rate: (rate, attainment) of one configuration's stationary
    traces, one per rate."""
    pts = sorted(attainment_by_rate)
    rates = [r for r, _ in pts]
    if not pts or len(set(rates)) != len(rates):
        raise ConfigError(f"goodput needs one attainment per rate, got {pts}")
    for i, (rate, att) in enumerate(pts):
        if att < level:
            if i == 0:
                return Goodput(None, False)
            (r0, a0) = pts[i - 1]
            return Goodput(r0 + (rate - r0) * (a0 - level) / (a0 - att), False)
    return Goodput(pts[-1][0], True)


def goodputs(points: Sequence["ConfigPoint"], rates: Mapping[str, float], attainment_level: float,
             slo_level: str) -> dict[tuple[str, str], Goodput]:
    """{(group, config_id): its goodput at `attainment_level` of the `slo_level` SLO} over the configurations'
    stationary traces (analysis/frontier.py ConfigPoints; `rates` maps each stationary trace id to its rate, and
    points on other traces are left out)."""
    if slo_level not in LEVELS:
        raise ConfigError(f"unknown SLO level {slo_level!r}")
    curves: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for p in points:
        if p.trace_id in rates:
            curves.setdefault((p.group, p.config_id), []).append(   # default-ok: grouping into a new list
                (rates[p.trace_id], 1.0 - getattr(p, f"slo_miss_{slo_level}")))
    return {key: goodput(curve, attainment_level) for key, curve in sorted(curves.items())}
