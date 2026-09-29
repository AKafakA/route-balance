"""The latency value that enters the objective: a TPOT source combined with an envelope.

TPOT sources, per tier (the fleet's tier_id; an unknown tier raises, nothing is matched by model name):
  constant        one ms/token value per tier
  profiled_curve  ms/token as a function of the running batch size b: linear between calibrated points,
                  clamped to the first and last point
  xgboost         per-tier boosters on the instance state and the request (xgboost_tpot.py), retrained on
                  what the scheduler knows at serving time (InstanceState below)

Envelopes, in seconds. L is the candidate model's expected output length in tokens (the estimator's).
  own_length   tpot * L                                                  (v3 static arm: queue-blind)
  queue_aware  own = max(1, int(L)); if running < max_num_seqs and waiting == 0: own * tpot,
               else (pending_decode / max(running, 1) + own) * tpot      (v3 learned arm, verbatim)
  batch_aware  b = running + waiting; if b < max_num_seqs: TPOT(b + 1) * L,
               else TPOT(max_num_seqs) * (L + pending_decode / max(running, 1))
The supported (source, envelope) pairs are routebalance.config.LATENCY_PAIRS; batch_aware reads a constant
source as a flat curve (TPOT(b) = the constant), the control that isolates the curve.

Constant and curve tables are JSON documents written by tools/calibrate_tpot_curve.py:
  {"unit": "ms_per_token", "constant": {tier: ms}, "curve": {tier: [[b, ms], ...]}, "provenance": {...}}
Values are ms per output token and at least 1 ms (v3 floored every TPOT at 1 ms, so a smaller value
would have been served differently); curve points have integer b >= 1, strictly increasing, and a curve never
decreases in b (D47: a physical TPOT(b) does not fall with the batch; a measured dip is sampling noise, and it
would make a fixed_ref latency term exceed 1). tools/calibrate_tpot_curve.py writes its curves as a running
max over b.

The constant and curve sources never import xgboost and never call a booster: their predict_calls is
always 0 (the zero-call gate).

Prefill (D118, optional: Latency.prefill): every envelope's value plus the prompt's prefill on the tier,
  intercept + per_token * P, with P the request's prompt tokens,
from a table written by tools/fit_prefill.py: {"unit": "ms", "prefill": {tier: {"intercept_ms", "per_token_ms",
"n", "r2"}}, "provenance": {...}}, both values positive. The line is fitted with the engines' prefix cache on, so it
carries the calibration's average caching; no cached share is subtracted per request. The same addend enters a
candidate's loaded and idle latency, so a fixed_ref term stays within (0, 1].
"""
from __future__ import annotations

import bisect
import math
from functools import partial
from pathlib import Path
from typing import Callable, Collection, Mapping, Optional, Protocol, Sequence

from routebalance.config import LATENCY_PAIRS, Latency as LatencyConfig, loads_strict
from routebalance.estimate.artifacts import ArtifactError, read_verified
from routebalance.types import Instance

MIN_TPOT_MS = 1.0
_TABLE_KEYS = {"unit", "constant", "curve", "provenance"}
_PREFILL_KEYS = {"unit", "prefill", "provenance"}
_PREFILL_LINE = {"intercept_ms", "per_token_ms", "n", "r2"}


class InstanceState(Protocol):
    """What the scheduler knows about an instance under official vLLM, and all a latency model may read.

    num_running, num_waiting   vllm:num_requests_running / vllm:num_requests_waiting (Prometheus /metrics)
    kv_cache_utilization       the KV-cache usage gauge, a fraction in [0, 1]
    max_num_seqs               the instance's configured batch capacity (deployment config)
    pending_decode_tokens      computed by the scheduler: remaining predicted decode tokens of its in-flight
                               requests on the instance, from their streamed progress
    The envelopes read the first two, max_num_seqs and pending_decode_tokens; the learned heads may also
    read kv_cache_utilization.
    """
    num_running: int
    num_waiting: int
    kv_cache_utilization: float
    max_num_seqs: int
    pending_decode_tokens: int


class LatencyError(ValueError):
    pass


# ---------------------------------------------------------------- TPOT sources


def _ms(v: object, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < MIN_TPOT_MS:
        raise LatencyError(f"{where}: {v!r} is not a TPOT of at least {MIN_TPOT_MS} ms/token")
    return float(v)


class ConstantTPOT:
    kind = "constant"

    def __init__(self, ms_per_token: Mapping[str, object]):
        if not ms_per_token:
            raise LatencyError("the constant TPOT table is empty")
        self._s = {tier: _ms(v, f"constant[{tier!r}]") / 1000.0 for tier, v in ms_per_token.items()}

    @property
    def tiers(self) -> frozenset[str]:
        return frozenset(self._s)

    @property
    def predict_calls(self) -> int:
        return 0

    def tpot_s(self, tier_id: str) -> float:
        if tier_id not in self._s:
            raise LatencyError(f"no constant TPOT for tier {tier_id!r}; the table has {sorted(self._s)}")
        return self._s[tier_id]

    def curve(self, tier_id: str) -> Callable[[int], float]:
        """TPOT (s) as a function of the batch size: flat."""
        tpot = self.tpot_s(tier_id)
        return lambda b: tpot

    def tpots(self, tier_ids: Sequence[str], states: Sequence[object], num_prompt_tokens: int,
              output_tokens: int) -> list[float]:
        del states, num_prompt_tokens, output_tokens   # a constant ignores the instance and the request
        return [self.tpot_s(t) for t in tier_ids]


class ProfiledCurveTPOT:
    kind = "profiled_curve"

    def __init__(self, curves: Mapping[str, Sequence[Sequence[object]]]):
        if not curves:
            raise LatencyError("the TPOT curve table is empty")
        self._b: dict[str, list[int]] = {}
        self._ms: dict[str, list[float]] = {}
        for tier, points in curves.items():
            if not points:
                raise LatencyError(f"curve[{tier!r}] has no points")
            bs, ms = [], []
            for point in points:
                if not (isinstance(point, (list, tuple)) and len(point) == 2 and type(point[0]) is int and point[0] >= 1):
                    raise LatencyError(f"curve[{tier!r}]: {point!r} is not a [b >= 1, ms] point")
                if bs and point[0] <= bs[-1]:
                    raise LatencyError(f"curve[{tier!r}]: batch sizes must strictly increase, got {point[0]} "
                                       f"after {bs[-1]}")
                bs.append(point[0])
                ms.append(_ms(point[1], f"curve[{tier!r}] at b={point[0]}"))
            self._b[tier], self._ms[tier] = bs, ms
        for tier, ms in self._ms.items():                # every point is a TPOT first, then the shape (D47)
            bs = self._b[tier]
            drop = next((i for i in range(1, len(ms)) if ms[i] < ms[i - 1]), None)
            if drop is not None:
                raise LatencyError(f"curve[{tier!r}] decreases from {ms[drop - 1]} ms at b={bs[drop - 1]} to "
                                   f"{ms[drop]} ms at b={bs[drop]}: a TPOT curve never decreases in b (D47)")

    @property
    def tiers(self) -> frozenset[str]:
        return frozenset(self._b)

    @property
    def predict_calls(self) -> int:
        return 0

    def tpot_s(self, tier_id: str, b: int) -> float:
        if tier_id not in self._b:
            raise LatencyError(f"no TPOT curve for tier {tier_id!r}; the table has {sorted(self._b)}")
        bs, ms = self._b[tier_id], self._ms[tier_id]
        j = bisect.bisect_right(bs, b)
        if j == 0:
            return ms[0] / 1000.0
        if j == len(bs):
            return ms[-1] / 1000.0
        b0, b1, m0, m1 = bs[j - 1], bs[j], ms[j - 1], ms[j]
        return (m0 + (m1 - m0) * (b - b0) / (b1 - b0)) / 1000.0

    def curve(self, tier_id: str) -> Callable[[int], float]:
        """TPOT (s) as a function of the batch size."""
        self.tpot_s(tier_id, 1)   # an unknown tier raises here, not at the first call
        return partial(self.tpot_s, tier_id)


def load_table(path: Path, md5: str) -> dict:
    """A calibration table document, verified against its pinned md5."""
    try:
        doc = loads_strict(read_verified(path, md5).decode("utf-8"))
    except UnicodeDecodeError as e:
        raise ArtifactError(f"{path}: not UTF-8: {e}") from e
    if not isinstance(doc, dict) or set(doc) - _TABLE_KEYS or "unit" not in doc or "provenance" not in doc:
        raise LatencyError(f"{path}: a TPOT table has keys {sorted(_TABLE_KEYS)} with unit and provenance required")
    if doc["unit"] != "ms_per_token" or not isinstance(doc["provenance"], dict):
        raise LatencyError(f"{path}: unit must be 'ms_per_token' and provenance an object")
    return doc


# ---------------------------------------------------------------- prefill


def _positive_ms(v: object, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise LatencyError(f"{where}: {v!r} is not a positive time in ms")
    return float(v)


class Prefill:
    """The prompt's prefill time (s) on a tier: intercept + per_token * prompt tokens."""

    def __init__(self, lines: Mapping[str, Mapping[str, object]]):
        if not lines:
            raise LatencyError("the prefill table is empty")
        self._line: dict[str, tuple[float, float]] = {}
        for tier, line in lines.items():
            if not isinstance(line, Mapping) or set(line) != _PREFILL_LINE:
                raise LatencyError(f"prefill[{tier!r}]: a line has exactly the keys {sorted(_PREFILL_LINE)}")
            self._line[tier] = (_positive_ms(line["intercept_ms"], f"prefill[{tier!r}].intercept_ms") / 1000.0,
                                _positive_ms(line["per_token_ms"], f"prefill[{tier!r}].per_token_ms") / 1000.0)

    @property
    def tiers(self) -> frozenset[str]:
        return frozenset(self._line)

    def seconds(self, tier_id: str, prompt_tokens: int) -> float:
        if tier_id not in self._line:
            raise LatencyError(f"no prefill line for tier {tier_id!r}; the table has {sorted(self._line)}")
        if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or prompt_tokens < 1:
            raise LatencyError(f"a prefill needs the prompt's token count (>= 1), got {prompt_tokens!r}")
        intercept, per_token = self._line[tier_id]
        return intercept + per_token * prompt_tokens


def load_prefill(path: Path, md5: str) -> Prefill:
    """A prefill table (tools/fit_prefill.py), verified against its pinned md5."""
    try:
        doc = loads_strict(read_verified(path, md5).decode("utf-8"))
    except UnicodeDecodeError as e:
        raise ArtifactError(f"{path}: not UTF-8: {e}") from e
    if not isinstance(doc, dict) or set(doc) != _PREFILL_KEYS:
        raise LatencyError(f"{path}: a prefill table has exactly the keys {sorted(_PREFILL_KEYS)}")
    if doc["unit"] != "ms" or not isinstance(doc["provenance"], dict) or not isinstance(doc["prefill"], dict):
        raise LatencyError(f"{path}: unit must be 'ms', prefill and provenance objects")
    return Prefill(doc["prefill"])


# ---------------------------------------------------------------- envelopes


def _check_slots(state: InstanceState) -> None:
    if state.max_num_seqs < 1:   # v3 read a missing or zero max_num_seqs as 256; it is always configured
        raise LatencyError(f"max_num_seqs={state.max_num_seqs} is not a batch capacity")


def own_length(tpot_s: float, length: float) -> float:
    return tpot_s * length


def queue_aware(tpot_s: float, state: InstanceState, length: float) -> float:
    _check_slots(state)
    own = max(1, int(length))
    if state.num_running < state.max_num_seqs and state.num_waiting == 0:
        return own * tpot_s
    return (float(state.pending_decode_tokens) / max(state.num_running, 1) + own) * tpot_s


def batch_aware(tpot_at: Callable[[int], float], state: InstanceState, length: float) -> float:
    _check_slots(state)
    b = state.num_running + state.num_waiting
    if b < state.max_num_seqs:
        return tpot_at(b + 1) * length
    return tpot_at(state.max_num_seqs) * (length + state.pending_decode_tokens / max(state.num_running, 1))


# ---------------------------------------------------------------- the model the scheduler calls


class LatencyModel:
    def __init__(self, source, envelope: str, prefill: Optional[Prefill]):
        if (source.kind, envelope) not in LATENCY_PAIRS:
            raise LatencyError(f"TPOT source {source.kind!r} has no envelope {envelope!r}")
        self.source, self.envelope, self.prefill = source, envelope, prefill

    @property
    def predict_calls(self) -> int:
        """Booster calls so far; always 0 for the constant and curve sources."""
        return self.source.predict_calls

    def latencies(self, instances: Sequence[Instance], states: Sequence[InstanceState],
                  lengths: Mapping[str, float], num_prompt_tokens: int) -> list[float]:
        """Scored latency (s) of one request on each candidate. `lengths` maps every estimator model to its
        expected output length; `states` are the candidates' instance states, in the same order."""
        if len(states) != len(instances):
            raise LatencyError(f"{len(instances)} instances but {len(states)} states")
        own = []
        for inst in instances:
            if inst.model not in lengths:
                raise LatencyError(f"no length estimate for {inst.model!r} (instance {inst.instance_id})")
            own.append(lengths[inst.model])
        tiers = [inst.tier.tier_id for inst in instances]
        if self.envelope == "batch_aware":
            out = [batch_aware(self.source.curve(t), s, L) for t, s, L in zip(tiers, states, own, strict=True)]
        else:
            # v3 fed the booster one output length for every tier: the largest expectation over all models
            tpots = self.source.tpots(tiers, states, num_prompt_tokens, max(1, int(max(lengths.values()))))
            if self.envelope == "own_length":
                out = [own_length(t, L) for t, L in zip(tpots, own, strict=True)]
            else:
                out = [queue_aware(t, s, L) for t, s, L in zip(tpots, states, own, strict=True)]
        if self.prefill is None:
            return out
        return [x + self.prefill.seconds(t, num_prompt_tokens) for x, t in zip(out, tiers, strict=True)]


def load_latency_model(cfg: LatencyConfig, artifact_root: Path, tier_ids: Collection[str]) -> LatencyModel:
    """The configured latency model; its source must cover every fleet tier."""
    path = Path(artifact_root) / cfg.table
    if cfg.source == "xgboost":
        from routebalance.estimate.xgboost_tpot import XGBoostTPOT   # only the learned arm imports xgboost
        source = XGBoostTPOT.from_manifest(path, cfg.table_md5, tier_ids)
    else:
        doc = load_table(path, cfg.table_md5)
        section = "constant" if cfg.source == "constant" else "curve"
        if not isinstance(doc.get(section), dict):
            raise LatencyError(f"{path}: the {cfg.source} source needs a {section!r} object")
        source = ConstantTPOT(doc[section]) if cfg.source == "constant" else ProfiledCurveTPOT(doc[section])
    absent = sorted(set(tier_ids) - source.tiers)
    if absent:
        raise LatencyError(f"{path}: no TPOT for fleet tiers {absent}")
    prefill = None
    if cfg.prefill is not None:
        prefill_path = Path(artifact_root) / cfg.prefill.table
        prefill = load_prefill(prefill_path, cfg.prefill.table_md5)
        absent = sorted(set(tier_ids) - prefill.tiers)
        if absent:
            raise LatencyError(f"{prefill_path}: no prefill line for fleet tiers {absent}")
    return LatencyModel(source, cfg.envelope, prefill)
