"""Preregistered analysis rules (configs/prereg_rules.json), fixed before the lease and read by the analysis.

  judge_key       the quality metric of record: the key of the DeepEval score in the scored corpora
  quality_floor   goodput is compared only between configurations whose mean quality is at least this
  equal_quality   tolerance: a treatment preset is "at equal quality" with a baseline when their mean
                  qualities differ by at most this (absolute score units); interpolation along the
                  treatment's frontier may also reach this far past its end points
                  bootstrap_resamples, confidence, bootstrap_seed: the percentile bootstrap over seeds
  goodput_slo     per-request deadline: a successful request meets its SLO when
                  e2e_ms <= base_ms + per_output_token_ms * output_tokens
  g1              the thesis gate (D16, D36), decided from the G1 cells (configs/scope.json) as soon as they
                  finish. RB-new (the profiled-curve arm) is compared with RB-decoupled on the same preset:
                    tier_mix_shift_min            RB-new's tier mix moves with load: the largest change of any
                                                  tier's request share from lambda 12 to lambda 30 is at least this
                                                  (share units, 0..1)
                    mean_latency_gain_l30_min     at lambda 30, 1 - mean E2E(RB-new) / mean E2E(RB-decoupled) is at
                                                  least this
                    p99_ratio_l30_max             at lambda 30, p99 E2E(RB-new) / p99 E2E(RB-decoupled) is at most
                                                  this ("p99 no worse")
                    quality_tolerance             at lambda 30, quality(RB-decoupled) - quality(RB-new) is at most
                                                  this (absolute score units)
                    l12_mean_latency_ratio_max    the lambda-12 guard: mean E2E(RB-new) / mean E2E(RB-decoupled) at
                                                  lambda 12 is at most this
                    presets_required              "all": G1 passes when both G1 presets pass; "any": when one does

Every value is required. A value still set to the placeholder "OWNER-MUST-SET" fails loading and is named,
so an unfinished rules file can never be used by accident.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from bench.schema import Strict, parse, read_json
from routebalance.config import ConfigError

PLACEHOLDER = "OWNER-MUST-SET"


class GoodputSlo(Strict):
    kind: Literal["e2e_affine"]
    base_ms: float = Field(gt=0)
    per_output_token_ms: float = Field(ge=0)

    def met(self, e2e_s: float, output_tokens: int) -> bool:
        return e2e_s * 1000.0 <= self.base_ms + self.per_output_token_ms * output_tokens


class EqualQuality(Strict):
    tolerance: float = Field(ge=0, le=1)
    bootstrap_resamples: int = Field(ge=100)
    confidence: float = Field(gt=0, lt=1)
    bootstrap_seed: int = Field(ge=0)


class G1Rule(Strict):
    tier_mix_shift_min: float = Field(ge=0, le=1)
    mean_latency_gain_l30_min: float = Field(lt=1)
    p99_ratio_l30_max: float = Field(gt=0)
    quality_tolerance: float = Field(ge=0, le=1)
    l12_mean_latency_ratio_max: float = Field(gt=0)
    presets_required: Literal["all", "any"]


class Rules(Strict):
    judge_key: str = Field(min_length=1)
    quality_floor: float = Field(ge=0, le=1)
    equal_quality: EqualQuality
    goodput_slo: GoodputSlo
    g1: G1Rule


def _placeholders(data: Any, path: str = "") -> list[str]:
    if data == PLACEHOLDER:
        return [path if path else "<root>"]
    if isinstance(data, dict):
        return [p for k, v in data.items() for p in _placeholders(v, f"{path}.{k}" if path else k)]
    if isinstance(data, list):
        return [p for i, v in enumerate(data) for p in _placeholders(v, f"{path}[{i}]")]
    return []


def parse_rules(data: Any) -> Rules:
    unset = _placeholders(data)
    if unset:
        raise ConfigError(f"preregistered values not set yet (the owner must fill them): {unset}")
    return parse(Rules, data, "rules")


def load_rules(path: Path) -> Rules:
    return parse_rules(read_json(path))
