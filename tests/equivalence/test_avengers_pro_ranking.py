"""The rebuild's AvengersPro cluster ranking (tools/router_training.py, D72) is upstream AvengersPro's own: on random
clusters with coarse scores (so ties occur), upstream's CostEfficiencyAnalyzer.compute_balance_score over the same
per-model means, sorted as upstream sorts them (balance_cluster_router.py), gives the same ranking."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.router_training import balance_ranking

MODELS = ["a", "b", "c", "d"]


def _upstream(root: Path):
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    mod = importlib.import_module("balance_cluster_router")
    assert Path(mod.__file__).resolve().parent == root
    return mod


@pytest.mark.parametrize("pw", [0.0, 0.25, 0.39, 0.53, 0.70, 0.80, 1.0])
def test_the_cluster_ranking_is_upstreams(v3, pw):
    up = _upstream(v3.upstream)
    analyzer = up.CostEfficiencyAnalyzer()
    rng = np.random.default_rng(0)
    for trial in range(300):
        n = int(rng.integers(1, 400))                       # long enough for summation order to matter
        # values like 0.1 and 0.2 are not exact in binary, so the order of summation shows (Codex review)
        quality = rng.choice([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0], size=(n, len(MODELS)))
        if trial % 7 == 0:                                   # equal means reached through different sums
            quality = np.array([[0.2, 0.0, 0.2, 0.2]] * (n // 2 + 1) + [[0.2, 0.4, 0.2, 0.2]] * (n // 2 + 1))
            n = len(quality)
            cost = rng.uniform(1e-6, 1e-4, size=(n, len(MODELS)))
        if trial % 7 != 0:
            cost = rng.uniform(1e-6, 1e-4, size=(n, len(MODELS)))
        if trial % 10 == 0:
            cost[:, 1] = cost[:, 0]                          # equal costs too
        # upstream's own reduction: np.mean over each model's list of values (analyze_cluster_cost_efficiency)
        metrics = {m: up.CostEfficiencyMetrics(model_name=m, accuracy=float(np.mean(quality[:, j].tolist())),
                                               avg_cost=float(np.mean(cost[:, j].tolist())), cost_efficiency=0.0,
                                               total_queries=n, total_cost=float(cost[:, j].sum()))
                   for j, m in enumerate(MODELS)}
        top_cost = max(x.avg_cost for x in metrics.values())
        top_acc, low_acc = max(x.accuracy for x in metrics.values()), min(x.accuracy for x in metrics.values())
        scores = {m: analyzer.compute_balance_score(x, pw, 1.0 - pw, top_cost, top_acc, low_acc)
                  for m, x in metrics.items()}
        expected = [m for m, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        assert balance_ranking(quality, cost, MODELS, pw) == expected, f"trial {trial}"
