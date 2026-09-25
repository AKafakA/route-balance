"""tools/retrain_xgb_served.py on a synthetic scheduler sweep: heads on every serving feature, loadable by the
scheduler, and refusals for a feature that never varies and for v3-only options."""
import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from routebalance.estimate.xgboost_tpot import FEATURES, XGBoostTPOT, feature_values
from tools.retrain_xgb_served import main

TIERS = ("qwen2.5-3b_a30", "qwen2.5-14b_v100")


def _sweep(path, n=600, constant_prompt=False):
    rng = random.Random(0)
    lines = []
    for i in range(n):
        tier = TIERS[i % 2]
        s = SimpleNamespace(num_running=rng.randint(0, 40), num_waiting=rng.randint(0, 5), max_num_seqs=64,
                            kv_cache_utilization=rng.random(), pending_decode_tokens=rng.randint(0, 9000))
        p, o = (100 if constant_prompt else rng.randint(1, 2000)), rng.randint(1, 900)
        tpot = (0.010 if tier == TIERS[0] else 0.016) * (1 + s.num_running / 40) + rng.gauss(0, 1e-4)
        lines.append(json.dumps({"tier": tier, "features": feature_values(s, p, o), "tpot_s": tpot,
                                 "output_tokens": o, "success": i % 50 != 7}))
    path.write_text("\n".join(lines) + "\n")
    return path


def test_scheduler_sweep_trains_loadable_heads_on_every_serving_feature(tmp_path):
    out = tmp_path / "heads"
    assert main(["--sweep", str(_sweep(tmp_path / "sweep.jsonl")), "--threads", "2", "--out", str(out)]) == 0
    manifest = (out / "md5s.json").read_bytes()
    heads = XGBoostTPOT.from_manifest(out, hashlib.md5(manifest).hexdigest(), TIERS)
    assert heads.feature_names == FEATURES
    pinned = json.loads(manifest)                    # the execution manifest records every file that was read
    assert heads.files == {"md5s.json": hashlib.md5(manifest).hexdigest(),
                           "feature_names.json": pinned["feature_names.json"],
                           **{f"{t}.xgb": pinned[f"{t}.xgb"] for t in sorted(TIERS)}}
    report = json.loads((out / "metrics.json").read_text())
    assert report["input"] == "scheduler sweep" and report["dropped_failed"] == 12
    for tier in TIERS:
        val = report["tiers"][tier]["val"]
        assert set(val) == {"served"} and val["served"]["mae"] < 1e-3          # learned the num_running slope
    busy = SimpleNamespace(num_running=40, num_waiting=0, max_num_seqs=64, kv_cache_utilization=0.5,
                           pending_decode_tokens=4000)
    idle = SimpleNamespace(num_running=0, num_waiting=0, max_num_seqs=64, kv_cache_utilization=0.5,
                           pending_decode_tokens=0)
    fast, slow = heads.tpots([TIERS[0], TIERS[0]], [idle, busy], 100, 100)
    assert slow > fast and np.isfinite([fast, slow]).all()


def test_a_feature_constant_in_training_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="constant in the training rows"):
        main(["--sweep", str(_sweep(tmp_path / "sweep.jsonl", constant_prompt=True)), "--threads", "2",
              "--out", str(tmp_path / "heads")])


def test_a_sweep_with_no_usable_record_is_refused(tmp_path):
    path = tmp_path / "sweep.jsonl"
    lines = [json.loads(line) for line in _sweep(path).read_text().splitlines() if line]
    path.write_text("".join(json.dumps(dict(r, tpot_s=None)) + "\n" for r in lines))
    with pytest.raises(SystemExit, match="no usable training records"):
        main(["--sweep", str(path), "--threads", "2", "--out", str(tmp_path / "heads")])
    assert not (tmp_path / "heads").exists()


def test_a_tier_below_the_v3_minimum_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="fewer than 50"):
        main(["--sweep", str(_sweep(tmp_path / "sweep.jsonl", n=80)), "--threads", "2", "--out", str(tmp_path / "h")])


def test_v3_options_do_not_go_with_the_scheduler_sweep(tmp_path):
    with pytest.raises(SystemExit, match="go with --v3-sweep"):
        main(["--sweep", str(_sweep(tmp_path / "sweep.jsonl")), "--v3-dir", str(tmp_path), "--threads", "2",
              "--out", str(tmp_path / "heads")])
