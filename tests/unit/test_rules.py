"""The rules file fails loudly until the owner has filled every preregistered value."""
import copy
import json
from pathlib import Path

import pytest

from analysis.rules import PLACEHOLDER, load_rules, parse_rules
from routebalance.config import ConfigError

TRACKED = Path(__file__).resolve().parents[2] / "configs" / "prereg_rules.json"
FILLED = {
    "judge_key": "deepeval-judge",
    "quality_floor": 0.35,
    "equal_quality": {"tolerance": 0.005, "bootstrap_resamples": 1000, "confidence": 0.95, "bootstrap_seed": 0},
    "goodput_slo": {"kind": "e2e_affine", "base_ms": 2000.0, "per_output_token_ms": 50.0},
    "g1": {"tier_mix_shift_min": 0.05, "mean_latency_gain_l30_min": 0.1, "p99_ratio_l30_max": 1.0,
           "quality_tolerance": 0.005, "l12_mean_latency_ratio_max": 1.05, "presets_required": "all"},
}


def test_the_tracked_file_names_every_unset_value():
    with pytest.raises(ConfigError) as e:
        load_rules(TRACKED)
    for name in ("quality_floor", "equal_quality.tolerance", "equal_quality.bootstrap_resamples",
                 "equal_quality.confidence", "equal_quality.bootstrap_seed", "goodput_slo.base_ms",
                 "goodput_slo.per_output_token_ms", "g1.tier_mix_shift_min", "g1.mean_latency_gain_l30_min",
                 "g1.p99_ratio_l30_max", "g1.quality_tolerance", "g1.l12_mean_latency_ratio_max",
                 "g1.presets_required"):
        assert name in str(e.value)


def test_the_tracked_file_has_the_schema_of_a_filled_file():
    tracked = json.loads(TRACKED.read_text())
    assert tracked.keys() == FILLED.keys() and tracked["judge_key"] == "deepeval-llama3.1-8b-it_reference"
    for section in ("equal_quality", "goodput_slo", "g1"):
        assert tracked[section].keys() == FILLED[section].keys()


def test_a_filled_file_loads(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(FILLED))
    rules = load_rules(path)
    assert rules.equal_quality.tolerance == 0.005 and rules.goodput_slo.base_ms == 2000.0
    assert rules.g1.presets_required == "all" and rules.g1.p99_ratio_l30_max == 1.0


@pytest.mark.parametrize("mutate", [
    lambda d: d["equal_quality"].update(tolerance=PLACEHOLDER),
    lambda d: d.update(quality_floor=1.5),
    lambda d: d["equal_quality"].update(confidence=1.0),
    lambda d: d["equal_quality"].update(bootstrap_resamples=10),
    lambda d: d["goodput_slo"].update(kind="ttft"),
    lambda d: d["goodput_slo"].update(base_ms="2000"),
    lambda d: d["goodput_slo"].pop("per_output_token_ms"),
    lambda d: d.update(own_best_rule=True),
    lambda d: d.pop("g1"),
    lambda d: d["g1"].update(presets_required="most"),
    lambda d: d["g1"].update(tier_mix_shift_min=1.5),
    lambda d: d["g1"].update(p99_ratio_l30_max=0.0),
    lambda d: d["g1"].update(l12_guard=1.0),
])
def test_invalid_rules_fail(mutate):
    d = copy.deepcopy(FILLED)
    mutate(d)
    with pytest.raises(ConfigError):
        parse_rules(d)


def test_the_slo_deadline_grows_with_output_length():
    slo = parse_rules(FILLED).goodput_slo
    assert slo.met(2.0, 0) and not slo.met(2.001, 0)
    assert slo.met(2.5, 10) and not slo.met(2.501, 10)
