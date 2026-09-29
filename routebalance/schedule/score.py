"""RouteBalance's objective for one request over its candidate instances.

For request r and candidate instance i (serving model m on GPU g), all three terms in [0, 1], higher better:

    score_i = w_q * Q_i + w_c * C_i + w_l * L_i

    Q_i = (q_i - min_j q_j) / (max_j q_j - min_j q_j + eps)    predicted quality, min-max over the candidates
    C_i = 1 - c_i / (max_j c_j + eps)                          predicted cost: prompt tokens x input price
                                                               + expected length x output price (D117)
    L_i = 1 - l_i / (max_j l_j + eps)                          "max_divide" (the v3 normalization), or
    L_i = min_j l0_j / l_i                                     "fixed_ref": against the request's best UNLOADED latency

l_i is the latency model's estimate on the instance as it is now; l0_i is the same estimate on an idle
instance. Under max_divide the slowest candidate scores 0 however loaded it is, so load only moves traffic
within the faster tiers; fixed_ref keeps every tier's load visible.

The sum is taken in v3's order (quality, cost, latency; v3's fourth term had weight 0 in every cell), so
the scores are bitwise v3's: float addition is not associative, and exact ties decide instances.
The winner is the highest score; exact ties go to the shorter queue (running + waiting, including this
round's earlier assignments), then to the least recently assigned instance, then to fleet order (v3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

EPS = 1e-8   # v3

Normalization = Literal["fixed_ref", "max_divide"]


@dataclass(frozen=True)
class Candidate:
    instance_id: str
    model: str
    quality: float                        # predicted quality of this request's answer on the model
    cost: float                           # predicted cost, dollars: prompt at the input price + expected output (D117)
    latency_s: float                      # load-aware latency estimate
    unloaded_latency_s: Optional[float]   # the same estimate on an idle instance; fixed_ref only
    queue_depth: int                      # running + waiting (tie-break)
    last_assigned: float                  # when this instance last won a request; 0.0 if never (tie-break)


@dataclass(frozen=True)
class Scored:
    candidate: Candidate
    score: float
    quality_term: float
    cost_term: float
    latency_term: float


def _check(c: Candidate, normalization: Normalization) -> None:
    ok = (math.isfinite(c.quality) and math.isfinite(c.cost) and c.cost >= 0
          and math.isfinite(c.latency_s) and c.latency_s > 0)
    if normalization == "fixed_ref":
        u = c.unloaded_latency_s
        ok = ok and u is not None and math.isfinite(u) and u > 0
    if not ok:
        raise ValueError(f"{c.instance_id}: quality and cost must be finite, cost >= 0, latencies > 0; got {c}")


def score(cands: Sequence[Candidate], *, w_quality: float, w_cost: float, w_latency: float,
          normalization: Normalization) -> list[Scored]:
    if not cands:
        raise ValueError("no candidates to score")
    if normalization not in ("fixed_ref", "max_divide"):
        raise ValueError(f"unknown normalization {normalization!r}")
    for c in cands:
        _check(c, normalization)
    q_min = min(c.quality for c in cands)
    q_max = max(c.quality for c in cands)
    c_max = max(c.cost for c in cands)
    l_max = max(c.latency_s for c in cands)
    ref = min(c.unloaded_latency_s for c in cands) if normalization == "fixed_ref" else None
    out = []
    for c in cands:
        q = (c.quality - q_min) / (q_max - q_min + EPS)
        cost = 1.0 - c.cost / (c_max + EPS)
        lat = 1.0 - c.latency_s / (l_max + EPS) if ref is None else ref / c.latency_s
        out.append(Scored(c, w_quality * q + w_cost * cost + w_latency * lat, q, cost, lat))
    return out


def pick(scored: Sequence[Scored]) -> Scored:
    """Highest score; exact ties to the shorter queue, then the least recently assigned, then fleet order."""
    return min(scored, key=lambda s: (-s.score, s.candidate.queue_depth, s.candidate.last_assigned))
