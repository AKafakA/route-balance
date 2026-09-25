"""T2: the estimator embeds the evaluation prompts as v3 did (same encoder, same batch composition), within 1e-6.

Also measures how far a prompt embedded alone moves from the same prompt embedded in a batch, which sizes
the tolerance the banked-cell replay (T3) needs: v3 embedded each prompt among different batch-mates.
"""
import numpy as np

from routebalance.estimate.prompt import unwrap_chat_prompt
from tests.equivalence.oracle_files import KNN_K50, knn_dir, v3_module
from tests.equivalence.encoder import eval_prompts, real_encoder


def _v3_encoder(oracle):
    knn = v3_module(oracle, "knn")
    v3 = knn.KNNEstimator.load(str(knn_dir(oracle, KNN_K50)), device="cpu", index_type="faiss_flat")
    return v3._get_encoder()


def test_embeddings_match_v3(oracle):
    texts = [unwrap_chat_prompt(p) for p in eval_prompts(oracle)]
    ours, theirs = real_encoder(oracle), _v3_encoder(oracle)
    worst, bitwise = 0.0, 0
    for start in range(0, len(texts), 32):
        batch = texts[start:start + 32]
        a = ours.encode(batch, normalize_embeddings=True, batch_size=max(len(batch), 32))
        b = theirs.encode(batch, normalize_embeddings=True, batch_size=max(len(batch), 32))
        assert np.isfinite(a).all() and np.isfinite(b).all()
        worst = max(worst, float(np.max(np.abs(a - b))))
        bitwise += int(np.sum(np.all(a == b, axis=1)))
    print(f"T2: {bitwise}/{len(texts)} embeddings bitwise equal, max |diff| {worst:.3g}")
    assert worst <= 1e-6


def test_batch_composition_noise_is_small(oracle):
    texts = [unwrap_chat_prompt(p) for p in eval_prompts(oracle)][:320]
    enc = real_encoder(oracle)
    batched = enc.encode(texts, normalize_embeddings=True, batch_size=32)
    alone = np.stack([enc.encode([t], normalize_embeddings=True, batch_size=32)[0] for t in texts])
    noise = float(np.max(np.abs(batched - alone)))
    print(f"T2: single vs batched embedding, max |diff| {noise:.3g} over {len(texts)} prompts")
    assert noise <= 1e-5
