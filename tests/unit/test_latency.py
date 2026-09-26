"""Latency sources, envelopes and the model that combines them (pure Python: no numpy, no xgboost)."""
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from routebalance.config import parse_scheduler
from routebalance.estimate.artifacts import ArtifactError
from routebalance.estimate.latency import (ConstantTPOT, LatencyError, LatencyModel, ProfiledCurveTPOT, batch_aware,
                                           load_latency_model, own_length, queue_aware)
from routebalance.types import Instance, Tier

A30_3B = Tier("qwen2.5-3b_a30", "Qwen/Qwen2.5-3B", "a30", 6e-08, 6e-08, 256)
V100_14B = Tier("qwen2.5-14b_v100", "Qwen/Qwen2.5-14B", "v100", 2e-07, 2e-07, 256)
INSTANCES = [Instance("i3", A30_3B, "n1", "h1", 8000), Instance("i14", V100_14B, "n2", "h2", 8000)]
LENGTHS = {"Qwen/Qwen2.5-3B": 120.7, "Qwen/Qwen2.5-14B": 80.2, "Qwen/Qwen2.5-72B": 300.9}
CURVES = {"qwen2.5-3b_a30": [[1, 10.0], [3, 12.0], [9, 18.0]], "qwen2.5-14b_v100": [[1, 16.0]]}


@dataclass
class State:
    """latency.InstanceState. No envelope reads kv_cache_utilization; it is here to match the protocol."""
    num_running: int
    num_waiting: int
    max_num_seqs: int
    pending_decode_tokens: int
    kv_cache_utilization: float = 0.25


IDLE = State(0, 0, 256, 0)


# ---------------------------------------------------------------- sources

def test_constant_is_ms_over_1000():
    src = ConstantTPOT({"qwen2.5-3b_a30": 10.7})
    assert src.tpot_s("qwen2.5-3b_a30") == 10.7 / 1000.0     # the v3 conversion, bit for bit
    assert src.predict_calls == 0


@pytest.mark.parametrize("table", [{}, {"t": 0.5}, {"t": True}, {"t": "10"}, {"t": float("nan")}, {"t": -3.0}])
def test_constant_rejects_bad_tables(table):
    with pytest.raises(LatencyError):
        ConstantTPOT(table)


def test_constant_unknown_tier_raises_never_matches_by_substring():
    src = ConstantTPOT({"3b": 10.7})
    with pytest.raises(LatencyError, match="no constant TPOT"):
        src.tpot_s("qwen2.5-3b_a30")


def test_curve_interpolates_and_clamps():
    src = ProfiledCurveTPOT(CURVES)
    t = "qwen2.5-3b_a30"
    assert src.tpot_s(t, 1) == 10.0 / 1000.0 and src.tpot_s(t, 3) == 12.0 / 1000.0
    assert src.tpot_s(t, 2) == pytest.approx(0.011) and src.tpot_s(t, 6) == pytest.approx(0.015)
    assert src.tpot_s(t, 0) == 0.010 and src.tpot_s(t, 1000) == 0.018        # clamped at both ends
    assert src.tpot_s("qwen2.5-14b_v100", 57) == 0.016                        # one point: flat
    assert src.predict_calls == 0


@pytest.mark.parametrize("curve", [[], [[0, 10.0]], [[1, 10.0], [1, 11.0]], [[2, 10.0], [1, 11.0]], [[1.5, 10.0]],
                                   [[True, 10.0]], [[1, 0.2]], [[1, 10.0, 5]], [5]])
def test_curve_rejects_bad_points(curve):
    with pytest.raises(LatencyError):
        ProfiledCurveTPOT({"t": curve})


def test_a_decreasing_curve_is_rejected_and_a_flat_one_accepted():
    """D47: a TPOT curve never decreases in b (a dip would make a fixed_ref latency term exceed 1)."""
    with pytest.raises(LatencyError, match=r"decreases from 10.0 ms at b=1 to 9.5 ms at b=4"):
        ProfiledCurveTPOT({"a": [[1, 9.0], [2, 9.0]], "t": [[1, 10.0], [4, 9.5], [8, 12.0]]})
    assert ProfiledCurveTPOT({"t": [[1, 10.0], [4, 10.0], [8, 12.0]]}).tpot_s("t", 6) == pytest.approx(0.011)
    # every point is checked as a TPOT before any curve's shape: a sub-ms point anywhere is reported as such
    with pytest.raises(LatencyError, match="not a TPOT"):
        ProfiledCurveTPOT({"t": [[1, 10.0], [4, 9.5]], "u": [[1, 0.2]]})


def test_curve_unknown_tier_raises():
    with pytest.raises(LatencyError, match="no TPOT curve"):
        ProfiledCurveTPOT(CURVES).tpot_s("qwen2.5-7b_a30", 1)


# ---------------------------------------------------------------- envelopes

def test_own_length_is_queue_blind():
    assert own_length(0.0165, 92.4) == 0.0165 * 92.4


def test_queue_aware_open_slot_uses_the_integer_own_length():
    assert queue_aware(0.02, State(3, 0, 256, 900), 92.9) == 92 * 0.02
    assert queue_aware(0.02, State(3, 0, 256, 900), 0.4) == 1 * 0.02                  # at least one token


@pytest.mark.parametrize("state", [State(3, 1, 256, 900), State(256, 0, 256, 900)])
def test_queue_aware_queued_adds_the_decode_backlog_per_running_request(state):
    assert queue_aware(0.02, state, 92.9) == (900.0 / max(state.num_running, 1) + 92) * 0.02


def test_queue_aware_with_nothing_running_divides_by_one():
    assert queue_aware(0.02, State(0, 2, 256, 500), 10.0) == (500.0 + 10) * 0.02


def test_batch_aware_below_capacity_reads_the_curve_at_the_new_batch_size():
    seen = []

    def tpot_at(b):
        seen.append(b)
        return 0.001 * b

    assert batch_aware(tpot_at, State(3, 2, 8, 700), 50.5) == 0.001 * 6 * 50.5
    assert seen == [6]


def test_batch_aware_full_instance_waits_for_the_backlog():
    assert batch_aware(lambda b: 0.001 * b, State(6, 2, 8, 700), 50.5) == 0.001 * 8 * (50.5 + 700 / 6)


@pytest.mark.parametrize("envelope", [queue_aware, lambda t, s, L: batch_aware(lambda b: t, s, L)])
def test_envelopes_reject_a_zero_capacity(envelope):
    # v3 read max_num_seqs=0 as 256; the fork always reports its configured capacity
    with pytest.raises(LatencyError, match="max_num_seqs"):
        envelope(0.02, State(0, 0, 0, 0), 10.0)


# ---------------------------------------------------------------- the model

class RecordingSource:
    kind = "constant"
    predict_calls = 0

    def __init__(self):
        self.calls = []

    def tpots(self, tier_ids, states, num_prompt_tokens, output_tokens):
        self.calls.append((list(tier_ids), num_prompt_tokens, output_tokens))
        return [0.01 for _ in tier_ids]


def test_model_feeds_the_largest_expected_length_as_the_shared_request_feature():
    src = RecordingSource()
    model = LatencyModel(src, "queue_aware")
    out = model.latencies(INSTANCES, [IDLE, IDLE], LENGTHS, 88)
    assert src.calls == [(["qwen2.5-3b_a30", "qwen2.5-14b_v100"], 88, 300)]    # max over every model, as v3
    assert out == [120 * 0.01, 80 * 0.01]


def test_model_own_length_uses_each_candidates_float_length():
    model = LatencyModel(ConstantTPOT({"qwen2.5-3b_a30": 10.7, "qwen2.5-14b_v100": 16.5}), "own_length")
    assert model.latencies(INSTANCES, [IDLE, IDLE], LENGTHS, 88) == [10.7 / 1000.0 * 120.7, 16.5 / 1000.0 * 80.2]


def test_model_batch_aware_reads_each_tiers_curve():
    model = LatencyModel(ProfiledCurveTPOT(CURVES), "batch_aware")
    out = model.latencies(INSTANCES, [State(2, 0, 256, 0), State(0, 0, 256, 0)], LENGTHS, 88)
    assert out == [pytest.approx(0.012 * 120.7), 0.016 * 80.2]


def test_model_batch_aware_on_a_constant_is_a_flat_curve():
    model = LatencyModel(ConstantTPOT({"qwen2.5-3b_a30": 10.7, "qwen2.5-14b_v100": 16.5}), "batch_aware")
    idle, full = State(2, 0, 256, 0), State(8, 3, 8, 800)
    assert model.latencies(INSTANCES, [idle, full], LENGTHS, 88) == [10.7 / 1000.0 * 120.7,
                                                                      16.5 / 1000.0 * (80.2 + 800 / 8)]


def test_model_raises_on_a_model_without_a_length_estimate():
    model = LatencyModel(ConstantTPOT({"qwen2.5-3b_a30": 10.7, "qwen2.5-14b_v100": 16.5}), "own_length")
    with pytest.raises(LatencyError, match="no length estimate"):
        model.latencies(INSTANCES, [IDLE, IDLE], {"Qwen/Qwen2.5-3B": 10.0}, 88)


def test_model_raises_when_states_do_not_line_up():
    model = LatencyModel(ConstantTPOT({"qwen2.5-3b_a30": 10.7, "qwen2.5-14b_v100": 16.5}), "own_length")
    with pytest.raises(LatencyError, match="states"):
        model.latencies(INSTANCES, [IDLE], LENGTHS, 88)


@pytest.mark.parametrize("source,envelope", [(ProfiledCurveTPOT({"t": [[1, 10.0]]}), "queue_aware"),
                                             (ProfiledCurveTPOT({"t": [[1, 10.0]]}), "own_length")])
def test_model_rejects_pairs_that_are_not_an_arm(source, envelope):
    with pytest.raises(LatencyError):
        LatencyModel(source, envelope)


# ---------------------------------------------------------------- loading from config

def _table(tmp_path, doc):
    raw = json.dumps(doc).encode()
    (tmp_path / "tpot.json").write_bytes(raw)
    return hashlib.md5(raw).hexdigest()


def _cfg(source, envelope, md5):
    doc = {
        "mode": "routebalance", "weights": {"w_quality": 0.5, "w_latency": 0.3, "w_cost": 0.2},
        "rounds": {"adaptive": True}, "lpt": "max", "pending_decode": "own_length", "dead_reckoning": True,
        "telemetry_read": "per_round",
        "latency": {"source": source, "envelope": envelope, "normalization": "fixed_ref", "table": "tpot.json",
                    "table_md5": md5},
        "estimator": {"index_dir": "knn", "index_md5": "0" * 32, "encoder_revision": "a" * 40, "k": 50,
                      "score_type": "deepeval", "prior": "knn"},
        "budget": {"admission_filter": False},
        "telemetry": {"poll_ms": 50.0, "max_age_ms": 500.0, "timeout_s": 3.0}, "pipeline": None,
    }
    return parse_scheduler(doc).latency


TABLE = {"unit": "ms_per_token", "constant": {"qwen2.5-3b_a30": 10.7, "qwen2.5-14b_v100": 16.5},
         "curve": CURVES, "provenance": {"method": "test"}}
TIERS = {"qwen2.5-3b_a30", "qwen2.5-14b_v100"}


@pytest.mark.parametrize("source,envelope", [("constant", "own_length"), ("constant", "queue_aware"),
                                             ("profiled_curve", "batch_aware"), ("constant", "batch_aware")])
def test_load_from_config(tmp_path, source, envelope):
    model = load_latency_model(_cfg(source, envelope, _table(tmp_path, TABLE)), tmp_path, TIERS)
    assert model.envelope == envelope and len(model.latencies(INSTANCES, [IDLE, IDLE], LENGTHS, 88)) == 2


def test_load_rejects_an_unpinned_table(tmp_path):
    _table(tmp_path, TABLE)
    with pytest.raises(ArtifactError, match="does not match"):
        load_latency_model(_cfg("constant", "own_length", "0" * 32), tmp_path, TIERS)


def test_load_rejects_a_table_that_misses_a_fleet_tier(tmp_path):
    md5 = _table(tmp_path, TABLE)
    with pytest.raises(LatencyError, match="no TPOT for fleet tiers"):
        load_latency_model(_cfg("constant", "own_length", md5), tmp_path, TIERS | {"qwen2.5-72b_a100"})


@pytest.mark.parametrize("mutate", [lambda d: d.pop("curve"), lambda d: d.update(unit="s_per_token"),
                                    lambda d: d.pop("provenance"), lambda d: d.update(extra=1)])
def test_load_rejects_malformed_tables(tmp_path, mutate):
    doc = json.loads(json.dumps(TABLE))
    mutate(doc)
    with pytest.raises(LatencyError):
        load_latency_model(_cfg("profiled_curve", "batch_aware", _table(tmp_path, doc)), tmp_path, TIERS)


# ---------------------------------------------------------------- zero-call gate

ZERO_CALL = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("xgboost", "numpy"):
            raise ImportError("blocked: " + name)
sys.meta_path.insert(0, Block())
from pathlib import Path
from test_latency import INSTANCES, LENGTHS, State, TIERS, _cfg
from routebalance.estimate.latency import load_latency_model
root, md5 = Path(sys.argv[1]), sys.argv[2]
for source, envelope in (("constant", "own_length"), ("constant", "queue_aware"), ("profiled_curve", "batch_aware"),
                         ("constant", "batch_aware")):
    model = load_latency_model(_cfg(source, envelope, md5), root, TIERS)
    for r in range(1000):
        model.latencies(INSTANCES, [State(r % 7, r % 3, 8, 13 * r), State(r % 5, 0, 8, r)], LENGTHS, r)
    assert model.predict_calls == 0, model.predict_calls
assert "xgboost" not in sys.modules
print("zero-call ok")
"""


def test_constant_and_curve_never_import_or_call_a_booster(tmp_path):
    md5 = _table(tmp_path, TABLE)
    repo = Path(__file__).resolve().parents[2]
    run = subprocess.run([sys.executable, "-c", ZERO_CALL, str(tmp_path), md5], capture_output=True, text=True,
                         cwd=Path(__file__).parent, env={"PYTHONPATH": f"{repo}:{Path(__file__).parent}"})
    assert run.returncode == 0 and "zero-call ok" in run.stdout, run.stderr
