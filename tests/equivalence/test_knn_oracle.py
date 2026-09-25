"""T1: the KNN estimator is bitwise equal to v3 DefaultModelEstimator.estimate_batch for the same query embeddings.

Both sides get the same stub encoder (text -> fixed vector), so this isolates retrieval and aggregation.
Queries: 2,000 corpus rows perturbed with N(0, 0.05^2) noise and renormalised; every duplicated corpus
embedding exactly (the pairs carry different labels, so ties are exercised); and the 3,534 evaluation
prompts as the real encoder embeds them. Chunkings 1, 4, 16 and 32 are run on both sides because faiss
switches kernels at 20 queries.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from routebalance.estimate.knn import DEEPEVAL_KEY, KnnEstimator, load_index
from routebalance.estimate.prompt import unwrap_chat_prompt
from tests.equivalence.oracle_files import KNN_F50, KNN_K50, TRAIN_SCORED, V3_SOURCES, knn_dir, md5_of, v3_module
from tests.equivalence.encoder import eval_prompts, real_encoder

CHUNKS = (1, 4, 16, 32)


class TableEncoder:
    def __init__(self, table):
        self.table = table

    def encode(self, sentences, *, normalize_embeddings, batch_size, **_):
        assert normalize_embeddings
        return np.stack([self.table[s] for s in sentences])


@pytest.fixture(scope="module")
def v3_me(oracle):
    return v3_module(oracle, "model_estimator")


def _queries(oracle, index_dir, md5):
    index = load_index(index_dir, md5)
    emb = index.embeddings
    rng = np.random.default_rng(0)
    rows = rng.choice(len(emb), 2000, replace=False)
    noisy = emb[rows] + rng.normal(0.0, 0.05, (2000, emb.shape[1])).astype(np.float32)
    noisy = (noisy / np.linalg.norm(noisy, axis=1, keepdims=True)).astype(np.float32)
    table = {f"perturbed {i}": v for i, v in enumerate(noisy)}
    seen, dups = {}, []
    for i, row in enumerate(emb):
        key = row.tobytes()
        if key in seen:
            dups += [seen[key], i]
        seen[key] = i
    table.update({f"duplicate {i}": emb[i] for i in dups})
    prompts = eval_prompts(oracle)
    real = real_encoder(oracle).encode([unwrap_chat_prompt(p) for p in prompts], normalize_embeddings=True,
                                       batch_size=32)
    table.update({unwrap_chat_prompt(p): v for p, v in zip(prompts, real, strict=True)})
    return index, table, len(dups)


@pytest.mark.parametrize("key,md5", [("k50", KNN_K50), ("f50", KNN_F50)])
def test_bitwise_equal_to_v3(oracle, v3_me, key, md5):
    index, table, n_dups = _queries(oracle, knn_dir(oracle, md5), md5)
    enc = TableEncoder(table)
    ours = KnnEstimator(index, enc, k=50, prior="knn")
    v3 = v3_me.DefaultModelEstimator({"score_type": "deepeval", "quality_config": {
        "model_dir": str(knn_dir(oracle, md5)), "k": 50, "index_type": "faiss_flat"}})
    v3._knn._encoder = enc
    assert md5_of(Path(sys.modules[type(v3._knn).__module__].__file__)) == V3_SOURCES["knn"][1]   # v3's own KNN code
    texts = list(table)
    checked = 0
    for chunk in CHUNKS:
        for budget in ((256,) if chunk != 32 else (256, 64, 1024)):
            for start in range(0, len(texts), chunk):
                batch = texts[start:start + chunk]
                for mine, theirs in zip(ours.estimate_batch(batch, budget), v3.estimate_batch(batch, budget),
                                        strict=True):
                    assert list(mine) == list(theirs) == list(v3._knn.model_names)
                    for m, e in mine.items():
                        t = theirs[m]
                        assert (e.quality, e.length, e.p_under_budget) == (t.score, t.length_expected, t.p_under_budget)
                        assert np.array_equal(np.array(e.bucket_probs, dtype=np.float32), t.length_bucket_probs)
                        checked += 1
    q = np.stack([table[t] for t in texts[:32]])
    sims, idx = v3._knn._faiss_index.search(q, 50)
    my_idx, my_dist = ours.search(texts[:32])
    assert np.array_equal(my_idx, idx) and np.array_equal(my_dist, (1.0 - sims).astype(np.float64))
    print(f"T1 {key}: {len(texts)} queries ({n_dups} duplicate rows), {checked} (prompt, model, chunking, "
          "budget) estimates bitwise equal")


def test_prior_mode_keeps_the_v3_lengths(oracle):
    index, table, _ = _queries(oracle, knn_dir(oracle, KNN_K50), KNN_K50)
    enc = TableEncoder(table)
    knn = KnnEstimator(index, enc, k=50, prior="knn")
    blind = KnnEstimator(index, enc, k=50, prior="per_model_mean")
    texts = list(table)[:512]
    for a, b in zip(knn.estimate_batch(texts, 256), blind.estimate_batch(texts, 256), strict=True):
        for m in index.models:
            assert (a[m].length, a[m].bucket_probs, a[m].p_under_budget) == (b[m].length, b[m].bucket_probs,
                                                                              b[m].p_under_budget)
            assert b[m].quality == float(np.mean(index.labels[m], dtype=np.float64))


def test_missing_labels_are_served_as_zero_like_v3(oracle):
    """The v3 index stores the source file's missing DeepEval labels as 0.0 (7/11/9/13 for 14B/3B/72B/7B)."""
    import json

    index = load_index(knn_dir(oracle, KNN_K50), KNN_K50)
    missing = {m: [] for m in index.models}
    with open(oracle.data_file(TRAIN_SCORED)) as f:
        for row, line in enumerate(f):
            models = json.loads(line)["models"]
            for m in index.models:
                if models.get(m, {}).get("llm_judge_scores", {}).get(DEEPEVAL_KEY) is None:
                    missing[m].append(row)
    assert {m.split("-")[-1]: len(r) for m, r in missing.items()} == {"14B": 7, "3B": 11, "72B": 9, "7B": 13}
    assert all(float(index.labels[m][r]) == 0.0 for m, rows in missing.items() for r in rows)
