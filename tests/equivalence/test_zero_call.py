"""T6: the constant and curve arms never import xgboost or call a booster, even where xgboost is installed;
the learned arm calls one booster per tier per request (v3's counter semantics)."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from routebalance.estimate.xgboost_tpot import XGBoostTPOT
from routebalance.types import Instance, Tier
from tests.equivalence.oracle_files import SERVED_HEADS, SERVED_HEADS_MD5, TPOT_TABLE, TPOT_TABLE_MD5, V3_TIERS

STATIC_ARMS = """
import sys
from pathlib import Path
from types import SimpleNamespace
from routebalance.estimate.latency import ConstantTPOT, LatencyModel, ProfiledCurveTPOT, load_table
from routebalance.types import Instance, Tier
doc = load_table(Path(sys.argv[1]), sys.argv[2])
fleet = [Instance(f"i{n}", Tier(t, "m-" + t, "a30", 1e-7, 1e-7, 8, ()), "n", "h", 8000 + n)
         for n, t in enumerate(sorted(doc["constant"]))]
lengths = {i.model: 50.0 + n for n, i in enumerate(fleet)}
for model in (LatencyModel(ConstantTPOT(doc["constant"]), "own_length"),
              LatencyModel(ConstantTPOT(doc["constant"]), "queue_aware"),
              LatencyModel(ProfiledCurveTPOT(doc["curve"]), "batch_aware"),
              LatencyModel(ConstantTPOT(doc["constant"]), "batch_aware")):
    for r in range(1000):
        states = [SimpleNamespace(num_running=r % 9, num_waiting=r % 2, max_num_seqs=8, pending_decode_tokens=7 * r)
                  for _ in fleet]
        model.latencies(fleet, states, lengths, r)
    assert model.predict_calls == 0
import importlib.util
assert importlib.util.find_spec("xgboost") is not None, "xgboost must be installed for this check to mean anything"
assert "xgboost" not in sys.modules
print("zero-call ok")
"""


def test_static_arms_never_touch_xgboost(artifacts):
    table = artifacts / TPOT_TABLE
    repo = Path(__file__).resolve().parents[2]
    run = subprocess.run([sys.executable, "-c", STATIC_ARMS, str(table), TPOT_TABLE_MD5], capture_output=True,
                         text=True, cwd=repo)
    assert run.returncode == 0 and "zero-call ok" in run.stdout, run.stderr


def test_learned_arm_counts_one_call_per_tier_per_request(artifacts):
    d = artifacts / SERVED_HEADS
    src = XGBoostTPOT.from_manifest(d, SERVED_HEADS_MD5, V3_TIERS)
    fleet = [Instance(f"i{n}", Tier(V3_TIERS[n % 4], "m", "a30", 1e-7, 1e-7, 8, ()), "n", "h", 8000 + n) for n in range(13)]
    state = SimpleNamespace(num_running=3, num_waiting=1, kv_cache_utilization=0.3, max_num_seqs=8,
                            pending_decode_tokens=400)
    for r in range(250):
        src.tpots([i.tier.tier_id for i in fleet], [state] * len(fleet), r, 97)
    assert src.predict_calls == 250 * 4
