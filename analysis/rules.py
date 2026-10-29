"""Preregistered analysis rules (configs/prereg_rules.json), fixed before the runs and read by the analysis.

  judge_key       the quality metric of record: the key of the DeepEval score in the scored corpora
  equal_quality   tolerance: a treatment preset is "at equal quality" with a baseline when their mean
                  qualities differ by at most this (absolute score units); interpolation along the
                  treatment's frontier may also reach this far past its end points
                  bootstrap_resamples, confidence, bootstrap_seed: the percentile bootstrap over requests
                  within the paired seeds (analysis/equal_quality.py)
  slo             after SOLA (analysis/slo.py): the multiplier k of each level, by served model, on the
                  served tier's single-request TTFT and TPOT (tight, loose); attainment_levels, the attainment
                  shares at which goodput is read (the largest rate that keeps them)
  g1              the preregistered gate on the scope's G1 configurations (configs/render36/scope.json), decided
                  when they finish. RB-new (RouteBalance with the profiled latency curve) is compared with
                  RB-decoupled on the same preset:
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

from pydantic import Field, model_validator

from bench.schema import Strict, parse, read_json
from routebalance.config import ConfigError

PLACEHOLDER = "OWNER-MUST-SET"


class SloRule(Strict):
    tight: dict[str, float] = Field(min_length=1)       # k by served model
    loose: dict[str, float] = Field(min_length=1)
    attainment_levels: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> "SloRule":
        if set(self.tight) != set(self.loose):
            raise ValueError("tight and loose name the same models")
        if not all(0 < self.tight[m] <= self.loose[m] for m in self.tight):
            raise ValueError("every k is positive and a model's loose k is at least its tight k")
        if not all(0 < a <= 1 for a in self.attainment_levels) or \
                len(set(self.attainment_levels)) != len(self.attainment_levels):
            raise ValueError("attainment levels are distinct shares in (0, 1]")
        return self

    def k(self) -> dict[str, dict[str, float]]:
        return {"tight": dict(self.tight), "loose": dict(self.loose)}


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
    equal_quality: EqualQuality
    slo: SloRule
    g1: G1Rule


def placeholders(data: Any, path: str = "") -> list[str]:
    """Where OWNER-MUST-SET values sit in a JSON value."""
    if data == PLACEHOLDER:
        return [path if path else "<root>"]
    if isinstance(data, dict):
        return [p for k, v in data.items() for p in placeholders(v, f"{path}.{k}" if path else k)]
    if isinstance(data, list):
        return [p for i, v in enumerate(data) for p in placeholders(v, f"{path}[{i}]")]
    return []


def parse_rules(data: Any) -> Rules:
    unset = placeholders(data)
    if unset:
        raise ConfigError(f"preregistered values not set yet (they must be filled in): {unset}")
    return parse(Rules, data, "rules")


def load_rules(path: Path) -> Rules:
    return parse_rules(read_json(path))
