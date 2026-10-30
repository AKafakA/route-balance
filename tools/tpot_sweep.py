"""Records of the TPOT latency sweep the scheduler itself logs (official vLLM), parsed without numpy.

One JSON object per request, written by the server:
  {"tier": <tier_id>, "features": {<exactly the learned heads' features, as xgboost_tpot.feature_values built
   them from the dispatch-time InstanceStats and the request>}, "tpot_s": <realized decode TPOT, s/token>,
   "output_tokens": <int>, "success": <bool>}
Every record must have exactly this shape, or reading fails. Failed requests and
missing (null) or non-positive targets are dropped, and counted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from fmrlb.config import loads_strict

KEYS = frozenset({"tier", "features", "tpot_s", "output_tokens", "success"})


@dataclass
class TierRows:
    features: list[list[float]]
    tpot_s: list[float]
    output_tokens: list[int]


@dataclass(frozen=True)
class Sweep:
    tiers: dict[str, TierRows]
    failed: int            # success == false
    no_target: int         # tpot_s null or <= 0


def _number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def read_sweep(paths: Sequence[Path], feature_names: Sequence[str]) -> Sweep:
    """Rows per tier in file order; each row's features in `feature_names` order."""
    tiers: dict[str, TierRows] = {}
    failed = no_target = 0
    for path in paths:
        with open(path) as f:
            for n, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                where = f"{path}:{n}"
                rec = loads_strict(line)
                if not isinstance(rec, dict) or set(rec) != KEYS:
                    raise ValueError(f"{where}: a sweep record has exactly the keys {sorted(KEYS)}")
                feats, tier, y, out = rec["features"], rec["tier"], rec["tpot_s"], rec["output_tokens"]
                if not isinstance(feats, dict) or set(feats) != set(feature_names):
                    raise ValueError(f"{where}: features must be exactly {list(feature_names)}")
                if not all(_number(v) for v in feats.values()):
                    raise ValueError(f"{where}: every feature must be a finite number")
                if not (isinstance(tier, str) and tier) or not isinstance(rec["success"], bool):
                    raise ValueError(f"{where}: tier must be a non-empty string and success a boolean")
                if type(out) is not int or out < 0 or not (y is None or _number(y)):
                    raise ValueError(f"{where}: output_tokens must be a non-negative integer and tpot_s a number or null")
                if not rec["success"]:
                    failed += 1
                    continue
                if y is None or y <= 0:
                    no_target += 1
                    continue
                if tier not in tiers:
                    tiers[tier] = TierRows([], [], [])
                rows = tiers[tier]
                rows.features.append([float(feats[k]) for k in feature_names])
                rows.tpot_s.append(float(y))
                rows.output_tokens.append(out)
    return Sweep(tiers, failed, no_target)
