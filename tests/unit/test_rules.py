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
    "equal_quality": {"tolerance": 0.005, "bootstrap_resamples": 1000, "confidence": 0.95, "bootstrap_seed": 0},
    "slo": {"tight": {"small": 10.0, "large": 5.0}, "loose": {"small": 15.0, "large": 10.0},
            "attainment_levels": [0.9, 0.99]},
    "g1": {"tier_mix_shift_min": 0.05, "mean_latency_gain_l30_min": 0.1, "p99_ratio_l30_max": 1.0,
           "quality_tolerance": 0.005, "l12_mean_latency_ratio_max": 1.05, "presets_required": "all"},
}


def test_the_tracked_file_holds_the_owners_values():
    """Set 09-25 (owner): equal quality within 0.005, 1000 paired resamples at 95%; the day-1 check's thresholds."""
    rules = load_rules(TRACKED)
    assert rules.equal_quality.model_dump() == {"tolerance": 0.005, "bootstrap_resamples": 1000, "confidence": 0.95,
                                                "bootstrap_seed": 0}
    assert rules.g1.model_dump() == {"tier_mix_shift_min": 0.05, "mean_latency_gain_l30_min": 0.1,
                                     "p99_ratio_l30_max": 1.0, "quality_tolerance": 0.005,
                                     "l12_mean_latency_ratio_max": 1.05, "presets_required": "all"}


def test_an_unset_value_is_named(tmp_path):
    d = copy.deepcopy(FILLED)
    d["g1"]["p99_ratio_l30_max"] = PLACEHOLDER
    d["equal_quality"]["tolerance"] = PLACEHOLDER
    with pytest.raises(ConfigError) as e:
        parse_rules(d)
    assert "g1.p99_ratio_l30_max" in str(e.value) and "equal_quality.tolerance" in str(e.value)


def test_the_tracked_file_has_the_schema_of_a_filled_file():
    tracked = json.loads(TRACKED.read_text())
    assert tracked.keys() == FILLED.keys() and tracked["judge_key"] == "deepeval-llama3.1-8b-it_reference"
    for section in ("equal_quality", "slo", "g1"):
        assert tracked[section].keys() == FILLED[section].keys()


def test_a_filled_file_loads(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(FILLED))
    rules = load_rules(path)
    assert rules.equal_quality.tolerance == 0.005 and rules.slo.k()["loose"]["large"] == 10.0
    assert rules.g1.presets_required == "all" and rules.g1.p99_ratio_l30_max == 1.0


@pytest.mark.parametrize("mutate", [
    lambda d: d["equal_quality"].update(tolerance=PLACEHOLDER),
    lambda d: d["equal_quality"].update(confidence=1.0),
    lambda d: d["equal_quality"].update(bootstrap_resamples=10),
    lambda d: d["slo"]["loose"].pop("large"),
    lambda d: d["slo"]["loose"].update(small=5.0),
    lambda d: d["slo"]["tight"].update(small=0.0),
    lambda d: d["slo"].update(attainment_levels=[0.9, 1.5]),
    lambda d: d["slo"].update(attainment_levels=[]),
    lambda d: d["slo"]["tight"].update(small="10"),
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


def test_the_tracked_slo_is_sola_by_model_size():
    """D57: tight / loose k = 10 / 15 for models up to 14B, 5 / 10 for 27B and above; goodput at 90% and 99%."""
    slo = json.loads(TRACKED.read_text())["slo"]
    fleets = [json.loads((TRACKED.parent / f).read_text()) for f in ("fleet_qwen.json", "fleet_gemma.json")]
    models = {t["model"] for f in fleets for t in f["tiers"].values()}
    assert set(slo["tight"]) == set(slo["loose"]) == models
    big = {"Qwen/Qwen2.5-72B", "google/gemma-3-27b-it"}
    assert all((slo["tight"][m], slo["loose"][m]) == ((5.0, 10.0) if m in big else (10.0, 15.0)) for m in models)
    assert slo["attainment_levels"] == [0.9, 0.99]
