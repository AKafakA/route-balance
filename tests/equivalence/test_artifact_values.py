"""Invalid artifact values and predictions raise instead of becoming estimates (review of 09-27). Needs numpy, faiss
and xgboost, so it runs with the equivalence tests."""
import hashlib
import pickle

import numpy as np
import pytest

from routebalance.estimate import knn
from routebalance.estimate.artifacts import ArtifactError
from routebalance.estimate.xgboost_tpot import FEATURES, XGBoostTPOT

def _index(tmp_path, k=50, lengths=(10.0, 20.0), labels=(0.2, 0.9)):
    state = {"embedding_model_name": "m", "k": k, "embeddings": np.eye(2, dtype=np.float32), "model_names": ["A"],
             "output_lengths": {"A": np.array(lengths, dtype=np.float32)},
             "per_judge_scores": {"A": {knn.DEEPEVAL_KEY: np.array(labels, dtype=np.float32)}}}
    d = tmp_path / "idx"
    d.mkdir(exist_ok=True)
    raw = pickle.dumps(state)
    (d / knn.INDEX_FILE).write_bytes(raw)
    return d, hashlib.md5(raw).hexdigest()


def test_the_knn_index_refuses_a_fractional_k_and_values_out_of_range(tmp_path):
    assert knn.load_index(*_index(tmp_path)).k == 50
    for bad, match in (({"k": 50.9}, "not a positive integer"), ({"k": True}, "not a positive integer"),
                       ({"lengths": (-1.0, 20.0)}, "lengths"), ({"labels": (0.2, 1.5)}, "labels"),
                       ({"labels": (0.2, float("nan"))}, "labels")):
        with pytest.raises(ArtifactError, match=match):
            knn.load_index(*_index(tmp_path, **bad))


class _NanBooster:
    def inplace_predict(self, x):
        return np.full(len(x), np.nan)


def test_a_head_predicting_nan_fails_instead_of_becoming_the_floor(monkeypatch):
    heads = XGBoostTPOT({"t": _NanBooster()}, [FEATURES[0]])
    monkeypatch.setattr(heads, "features", lambda state, p, o: np.zeros(1))
    with pytest.raises(ArtifactError, match="predicted nan"):
        heads.tpots(["t"], [None], 10, 20)
