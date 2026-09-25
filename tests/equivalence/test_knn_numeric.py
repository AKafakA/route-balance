"""The KNN estimator on tiny hand-built indexes: closed-form values, load-time refusals, prior mode, threads."""
import hashlib
import pickle
import threading

import numpy as np
import pytest

from routebalance.config import Estimator
from routebalance.estimate.artifacts import ArtifactError
from routebalance.estimate.knn import DEEPEVAL_KEY, KnnEstimator, load_index, load_knn

EMB = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
LENGTHS = {"m-b": [10.0, 70.0, 200.0, 5.0], "m-a": [1000.0, 64.0, 63.0, 2000.0]}
LABELS = {"m-b": [0.9, 0.5, 0.0, 0.1], "m-a": [0.2, 0.4, 0.6, 0.8]}


class StubEncoder:
    def __init__(self, table):
        self.table, self.calls = table, 0

    def encode(self, sentences, *, normalize_embeddings, batch_size):
        assert normalize_embeddings and batch_size >= max(len(sentences), 32)
        self.calls += 1
        return np.stack([self.table[s] for s in sentences])


def _state(k=3, models=("m-b", "m-a"), labels=LABELS):
    return {"embedding_model_name": "stub", "k": k, "embeddings": EMB, "model_names": list(models),
            "output_lengths": {m: np.array(LENGTHS[m], dtype=np.float32) for m in models},
            "per_judge_scores": {m: {DEEPEVAL_KEY: np.array(labels[m], dtype=np.float32)} for m in labels}}


def _write(tmp_path, state):
    raw = pickle.dumps(state)
    (tmp_path / "knn_estimator.pkl").write_bytes(raw)
    return tmp_path, hashlib.md5(raw).hexdigest()


def _estimator(tmp_path, prior="knn", state=None, k=3):
    index = load_index(*_write(tmp_path, state if state is not None else _state()))
    enc = StubEncoder({"x-axis": np.array([1.0, 0.0], dtype=np.float32), "diag": np.array([0.8, 0.6], dtype=np.float32)})
    return KnnEstimator(index, enc, k=k, prior=prior), enc


def test_closed_form_values(tmp_path):
    est, _ = _estimator(tmp_path)
    got = est.estimate_batch(["<|im_start|>user\nx-axis<|im_end|>\n"], budget_tokens=64)[0]
    assert list(got) == ["m-b", "m-a"]                                   # pickle order, not sorted
    d = np.array([0.0, 0.4, 1.0])                                        # neighbours 0, 1, 2 of (1, 0)
    w = 1.0 / (d + 1e-6)
    w /= w.sum()
    b = got["m-b"]
    assert b.quality == pytest.approx(float(w @ [0.9, 0.5, 0.0]), rel=1e-6)
    assert b.length == pytest.approx(float(w @ [10.0, 70.0, 200.0]), rel=1e-6)
    assert b.p_under_budget == pytest.approx(float(w[0]), rel=1e-6)     # only length 10 is <= 64
    assert b.bucket_probs[0] == pytest.approx(w[0], rel=1e-6) and b.bucket_probs[1] == pytest.approx(w[1], rel=1e-6)
    assert b.bucket_probs[3] == pytest.approx(w[2], rel=1e-6) and len(b.bucket_probs) == 16
    a = got["m-a"]
    assert a.bucket_probs[15] == pytest.approx(w[0], rel=1e-6)          # 1000 tokens: the open-ended last bucket
    assert a.p_under_budget == pytest.approx(float(w[1] + w[2]), rel=1e-6)   # 64 and 63 tokens


def test_search_returns_rank_order_and_float64_distances(tmp_path):
    est, _ = _estimator(tmp_path)
    idx, dist = est.search(["diag"])
    assert idx.dtype == np.int64 and dist.dtype == np.float64 and idx.tolist() == [[1, 0, 2]]
    assert np.all(np.diff(dist[0]) >= 0)


def test_prior_mode_is_prompt_blind_but_keeps_neighbour_lengths(tmp_path):
    knn, _ = _estimator(tmp_path)
    blind, _ = _estimator(tmp_path, prior="per_model_mean")
    for prompt in ("x-axis", "diag"):
        k_est, b_est = knn.estimate_batch([prompt], 256)[0], blind.estimate_batch([prompt], 256)[0]
        for m in ("m-b", "m-a"):
            assert b_est[m].quality == float(np.mean(np.array(LABELS[m], dtype=np.float32), dtype=np.float64))
            assert (b_est[m].length, b_est[m].bucket_probs) == (k_est[m].length, k_est[m].bucket_probs)


def test_empty_batch_does_not_touch_the_encoder(tmp_path):
    est, enc = _estimator(tmp_path)
    assert est.estimate_batch([], 256) == [] and enc.calls == 0


def test_threads_get_the_serial_answers(tmp_path):
    est, _ = _estimator(tmp_path)
    prompts = ["x-axis", "diag"] * 8
    serial = est.estimate_batch(prompts, 256)
    results = [None] * 8

    def run(i):
        results[i] = est.estimate_batch(prompts, 256)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r == serial for r in results) and est.batch_calls == 9


def test_configured_k_must_equal_the_baked_k(tmp_path):
    with pytest.raises(ArtifactError, match="built with k=3"):
        _estimator(tmp_path, k=2)


def test_k_larger_than_the_corpus_uses_every_prompt_as_v3(tmp_path):
    est, _ = _estimator(tmp_path, state=_state(k=5), k=5)
    idx, _ = est.search(["x-axis"])
    assert est.k == 4 and sorted(idx[0].tolist()) == [0, 1, 2, 3]
    w = 1.0 / (np.array([0.0, 0.4, 1.0, 2.0]) + 1e-6)
    w /= w.sum()
    assert est.estimate_batch(["x-axis"], 64)[0]["m-b"].length == pytest.approx(float(w @ [10.0, 70.0, 200.0, 5.0]),
                                                                                rel=1e-6)


def test_a_model_without_deepeval_labels_is_refused(tmp_path):
    with pytest.raises(ArtifactError, match="labels"):
        load_index(*_write(tmp_path, _state(labels={"m-b": LABELS["m-b"]})))


@pytest.mark.parametrize("mutate", [lambda s: s.pop("k"), lambda s: s.update(model_names=[]),
                                    lambda s: s.update(model_names=["m-b", "m-b"]),
                                    lambda s: s.update(embeddings=EMB.astype(np.float64)),
                                    lambda s: s["output_lengths"].update({"m-a": np.zeros(3, dtype=np.float32)})])
def test_malformed_indexes_are_refused(tmp_path, mutate):
    state = _state()
    mutate(state)
    with pytest.raises(ArtifactError):
        load_index(*_write(tmp_path, state))


class _Tripwire:
    def __reduce__(self):
        return (_tripped, ())


TRIPPED = []


def _tripped():
    TRIPPED.append(True)
    return {}


def test_an_unpinned_pickle_is_never_unpickled(tmp_path):
    (tmp_path / "knn_estimator.pkl").write_bytes(pickle.dumps(_Tripwire()))
    with pytest.raises(ArtifactError, match="does not match"):
        load_index(tmp_path, "0" * 32)
    assert TRIPPED == []


def test_zero_labels_are_reported_at_load(tmp_path, caplog):
    caplog.set_level("INFO")
    _estimator(tmp_path)
    assert "'m-b': 1" in caplog.text and "'m-a': 0" in caplog.text


def test_a_fleet_model_missing_from_the_index_is_refused_before_the_encoder_loads(tmp_path):
    (tmp_path / "knn").mkdir()
    _, md5 = _write(tmp_path / "knn", _state())
    cfg = Estimator(index_dir="knn", index_md5=md5, encoder_revision="a" * 40, k=3, score_type="deepeval", prior="knn")
    with pytest.raises(ArtifactError, match="no labels for fleet models"):
        load_knn(cfg, tmp_path, {"m-a", "m-c"})
