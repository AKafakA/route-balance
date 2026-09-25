"""T4: learned TPOT.

v3 side: tools/fork_tpot_features.py rebuilds the vector the v3 heads were served with, bit for bit
against v3's build_feature_vector, on a synthetic grid of fork /instance_stats bodies (idle, all zeros,
full, queued, p95 = 0) and 5,000 recorded states restricted to the fork's fields, each crossed with prompt
tokens {0, 1, 88, 4000} and output tokens {1, 97, 1023}. The retrain tool scores the v3 heads with it.

Serving side: the retrained heads load from their manifest, read only serving-time features, and are
called as v3 called its heads (per tier, that tier's candidates stacked in fleet order, one call, the
1 ms floor).
"""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import xgboost as xgb

from routebalance.estimate.artifacts import ArtifactError
from routebalance.estimate.xgboost_tpot import FEATURES, TPOT_FLOOR_S, XGBoostTPOT, feature_vector
from tests.equivalence.oracle_files import (SERVED_HEADS, SERVED_HEADS_MD5, V3_TIERS, V3_TPOT_MD5, tpot_dir,
                                            v3_module)
from tests.equivalence.states import fork_body, instance_state, recorded, synthetic
from tools.fork_tpot_features import V3_FEATURES, v3_serving_vector
from tools.retrain_xgb_served import V3_SWEEP_FEATURES

PROMPTS = (0, 1, 88, 4000)
OUTPUTS = (1, 97, 1023)


@pytest.fixture(scope="module")
def v3(oracle):
    return v3_module(oracle, "xgboost")


@pytest.fixture(scope="module")
def served_heads(artifacts):
    d = artifacts / SERVED_HEADS
    return XGBoostTPOT.from_manifest(d, SERVED_HEADS_MD5, V3_TIERS)


def test_fork_serving_vector_is_bitwise_v3(oracle, v3):
    assert tuple(v3.ALL_FEATURES) == V3_FEATURES
    rec = recorded(oracle, 5000)
    n = 0
    for body in synthetic() + [fork_body(s) for s in rec]:
        for p in PROMPTS:
            for o in OUTPUTS:
                ours, theirs = v3_serving_vector(body, p, o), v3.build_feature_vector(body, p, o)
                assert ours.dtype == np.float32 and ours.tobytes() == theirs.tobytes()
                n += 1
    trained = sum(not np.array_equal(v3.build_feature_vector(s, 88, 97), v3_serving_vector(fork_body(s), 88, 97))
                  for s in rec)
    print(f"T4: {n} fork serving vectors bitwise equal to v3; {trained}/{len(rec)} recorded states differ from their "
          "training-time vector (the fields the fork never served)")
    assert trained > 0.99 * len(rec)


def test_served_heads_read_only_serving_time_features(artifacts, served_heads):
    assert served_heads.feature_names == V3_SWEEP_FEATURES and set(V3_SWEEP_FEATURES) <= set(FEATURES)
    report = json.loads((artifacts / SERVED_HEADS / "metrics.json").read_text())
    assert report["v3_heads"] == V3_TPOT_MD5          # compared with exactly the v3 heads


def test_served_heads_are_called_as_v3_called_its_heads(oracle, served_heads):
    states = [instance_state(s) for s in synthetic() + recorded(oracle, 500)]
    fleet = [V3_TIERS[i % 4] for i in range(13)]                 # tiers interleaved, as instances in a fleet
    for start in range(0, len(states) - 13, 13):
        group = states[start:start + 13]
        for p in PROMPTS:
            for o in OUTPUTS:
                want = {}
                for tier in dict.fromkeys(fleet):                # v3 _batched_tpot_per_request
                    members = [i for i, t in enumerate(fleet) if t == tier]
                    x = np.stack([feature_vector(V3_SWEEP_FEATURES, group[i], p, o) for i in members])
                    preds = served_heads.booster(tier).inplace_predict(x)
                    dm = served_heads.booster(tier).predict(xgb.DMatrix(x, feature_names=list(V3_SWEEP_FEATURES)))
                    assert np.max(np.abs(preds - dm)) <= 1e-7
                    want.update({i: max(0.001, float(v)) for i, v in zip(members, preds, strict=True)})
                assert served_heads.tpots(fleet, group, p, o) == [want[i] for i in range(len(fleet))]
    assert served_heads.predict_calls > 0


def test_features_are_the_documented_values():
    s = SimpleNamespace(num_running=5, num_waiting=2, kv_cache_utilization=0.5, max_num_seqs=8, pending_decode_tokens=77)
    v = dict(zip(FEATURES, feature_vector(FEATURES, s, 100, 20).tolist(), strict=True))
    assert v == {"num_running": 5.0, "num_waiting": 2.0, "pending_decode_tokens": 77.0, "kv_cache_utilization": 0.5,
                 "num_prompt_tokens": 100.0, "num_predicted_output_tokens": 20.0, "req_decode_ctx_avg": 109.5,
                 "req_decode_ctx_max": 119.0}


def test_booster_floor_applies_to_every_prediction():
    class Negative:
        def inplace_predict(self, x):
            return np.full(len(x), -0.5, dtype=np.float32)

    src = XGBoostTPOT({"t": Negative()}, FEATURES)
    s = SimpleNamespace(num_running=0, num_waiting=0, kv_cache_utilization=0.0, max_num_seqs=8, pending_decode_tokens=0)
    assert src.tpots(["t", "t"], [s, s], 10, 10) == [TPOT_FLOOR_S, TPOT_FLOOR_S] and src.predict_calls == 1


def test_load_refuses_bad_dirs(oracle, artifacts, tmp_path):
    d = artifacts / SERVED_HEADS
    manifest = json.loads((d / "md5s.json").read_text())
    with pytest.raises(ArtifactError, match="no TPOT head"):
        XGBoostTPOT.load(d, manifest, V3_TIERS + ("qwen2.5-7b_v100",))
    with pytest.raises(ArtifactError, match="does not match"):
        XGBoostTPOT.load(d, dict(manifest, **{"qwen2.5-3b_a30.xgb": "0" * 32}), V3_TIERS)
    with pytest.raises(ArtifactError, match="does not match"):
        XGBoostTPOT.from_manifest(d, "0" * 32, V3_TIERS)
    for extra in ("decode_ctx_p95", "ema_decode_iter_ms"):             # fork-only fields cannot be served
        names = json.dumps(list(FEATURES) + [extra]).encode()
        (tmp_path / "feature_names.json").write_bytes(names)
        with pytest.raises(ArtifactError, match="cannot be computed at serving time"):
            XGBoostTPOT.load(tmp_path, {"feature_names.json": hashlib.md5(names).hexdigest()}, ())
    with pytest.raises(ArtifactError, match="no pinned md5"):
        XGBoostTPOT.load(tpot_dir(oracle), {"xgboost_metadata.json": "0" * 32}, V3_TIERS)
