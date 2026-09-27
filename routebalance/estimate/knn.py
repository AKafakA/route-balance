"""Per-model quality and output length from the prompt's nearest neighbours.

A port of the v3 serving path (DefaultModelEstimator.estimate_batch over a KNNEstimator index):
unwrap the chat template, embed with the index's sentence encoder (normalised), take the k corpus
prompts with the largest inner product (faiss IndexFlatIP, so cosine similarity s), give neighbour i the
weight w_i = 1 / (d_i + 1e-6) with d = 1 - s, normalised to sum to 1, and then for every model:
  quality         sum_i w_i q_i over the neighbours' DeepEval labels q; with prior="per_model_mean"
                  it is instead the model's mean label over the whole index (the prompt-blind ablation)
  length          sum_i w_i len_i over the neighbours' output lengths, in tokens
  bucket_probs    the weights histogrammed by length into 64-token buckets, the 16th open-ended
  p_under_budget  the share of the weight on neighbours with len_i <= budget_tokens
Each float32/float64 step follows v3 operation by operation, so for the same query embeddings the
results are bitwise equal to v3 (tests/equivalence).

The index artifact is the v3 pickle <index_dir>/knn_estimator.pkl. Its md5 is checked before it is
unpickled, its baked k must equal the configured k, and every model must carry DeepEval labels. The v3
index builder stored a missing label as 0.0 and the pickle cannot tell those apart from real zeros, so
they are served as 0.0, as in v3; the number of 0.0 labels per model is logged at load.
"""
from __future__ import annotations

import logging
import math
import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Literal, Protocol, Sequence

import faiss
import numpy as np

from routebalance.config import Estimator as EstimatorConfig
from routebalance.estimate.artifacts import ArtifactError, read_verified
from routebalance.estimate.prompt import unwrap_chat_prompt

log = logging.getLogger(__name__)

INDEX_FILE = "knn_estimator.pkl"
DEEPEVAL_KEY = "deepeval-llama3.1-8b-it_reference"   # judge key of the DeepEval labels in the index
BUCKET_TOKENS = 64                                    # v3 served only these bucket settings
NUM_BUCKETS = 16
_DISTANCE_EPS = 1e-6
_MIN_ENCODE_BATCH = 32                                # v3 encode batch_size = max(n, 32)

Prior = Literal["knn", "per_model_mean"]


class Encoder(Protocol):
    def encode(self, sentences: list[str], *, normalize_embeddings: bool, batch_size: int) -> np.ndarray: ...


@dataclass(frozen=True)
class Estimate:
    quality: float                    # predicted quality of this model's answer, in [0, 1]
    length: float                     # expected output length, tokens
    bucket_probs: tuple[float, ...]   # P(length in [64b, 64b + 64)), the last bucket open-ended
    p_under_budget: float             # P(length <= budget_tokens)


@dataclass(frozen=True)
class KnnIndex:
    encoder_name: str
    k: int                            # the k the index was built for
    embeddings: np.ndarray            # (N, d) float32, rows L2-normalised
    models: tuple[str, ...]           # in pickle order, which is the order of every result
    lengths: dict[str, np.ndarray]    # model -> (N,) float32 output tokens
    labels: dict[str, np.ndarray]     # model -> (N,) float32 DeepEval scores


def load_index(index_dir: Path, md5: str) -> KnnIndex:
    state = pickle.loads(read_verified(Path(index_dir) / INDEX_FILE, md5))   # pickle runs code: md5 first
    if not isinstance(state, dict):
        raise ArtifactError(f"{index_dir}: the index pickle is not a dict")
    missing = [key for key in ("embedding_model_name", "k", "embeddings", "model_names", "output_lengths",
                               "per_judge_scores") if key not in state]
    if missing:
        raise ArtifactError(f"{index_dir}: the index lacks {missing}")
    emb = state["embeddings"]
    if not (isinstance(emb, np.ndarray) and emb.dtype == np.float32 and emb.ndim == 2 and len(emb) > 0):
        raise ArtifactError(f"{index_dir}: embeddings must be a non-empty (N, d) float32 array")
    norms = np.linalg.norm(emb.astype(np.float64), axis=1)
    if not np.all(np.abs(norms - 1.0) <= 5e-7):     # unit rows (float32 rounding; the pinned indexes <= 1.5e-7), so
        #                                               1 - similarity + _DISTANCE_EPS stays positive
        raise ArtifactError(f"{index_dir}: embedding rows must be L2-normalised (norms {norms.min():.4f}-{norms.max():.4f})")
    models = tuple(state["model_names"])
    if not models or len(set(models)) != len(models):
        raise ArtifactError(f"{index_dir}: model names {models!r} are empty or repeated")
    lengths, labels = {}, {}
    for m in models:
        judges = state["per_judge_scores"].get(m)
        if m not in state["output_lengths"] or judges is None or DEEPEVAL_KEY not in judges:
            raise ArtifactError(f"{index_dir}: model {m!r} has no output lengths or no {DEEPEVAL_KEY!r} labels")
        lengths[m] = _column(state["output_lengths"][m], len(emb), f"{index_dir}: lengths of {m!r}", 0.0, math.inf)
        labels[m] = _column(judges[DEEPEVAL_KEY], len(emb), f"{index_dir}: labels of {m!r}", 0.0, 1.0)
    k = state["k"]
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ArtifactError(f"{index_dir}: the stored k {k!r} is not a positive integer")
    return KnnIndex(str(state["embedding_model_name"]), int(k), emb, models, lengths, labels)


def _column(a: object, n: int, what: str, low: float, high: float) -> np.ndarray:
    """A float32 column of length n, every value finite and in [low, high]."""
    if not (isinstance(a, np.ndarray) and a.dtype == np.float32 and a.shape == (n,)):
        raise ArtifactError(f"{what} must be a float32 array of length {n}")
    if not (np.isfinite(a).all() and (a >= low).all() and (a <= high).all()):
        raise ArtifactError(f"{what} must be finite and within [{low}, {high}]")
    return a


class KnnEstimator:
    """estimate_batch is safe to call from worker threads; calls run one at a time."""

    def __init__(self, index: KnnIndex, encoder: Encoder, *, k: int, prior: Prior):
        if index.k != k:
            raise ArtifactError(f"configured k={k} but the index was built with k={index.k}; they describe "
                                "different experiments")
        self.index, self.prior = index, prior
        self.k = min(k, len(index.embeddings))    # v3: a corpus smaller than k serves all its prompts
        self._encoder = encoder
        self._faiss = faiss.IndexFlatIP(index.embeddings.shape[1])
        self._faiss.add(index.embeddings.astype(np.float32))
        self._mean_label = {m: float(np.mean(index.labels[m], dtype=np.float64)) for m in index.models}
        self._lock = threading.Lock()
        self.batch_calls = 0
        zeros = {m: int(np.count_nonzero(index.labels[m] == 0.0)) for m in index.models}
        log.info("KNN index: %d prompts, k=%d, prior=%s; labels equal to 0.0 per model (v3 stored missing labels "
                 "as 0.0, so these include them): %s", len(index.embeddings), k, prior, zeros)

    @property
    def models(self) -> tuple[str, ...]:
        return self.index.models

    def search(self, prompts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Neighbour indices (n, k) int64 and distances 1 - similarity (n, k) float64, in rank order."""
        texts = [unwrap_chat_prompt(p) for p in prompts]
        with self._lock:     # the encoder is not guaranteed thread-safe
            self.batch_calls += 1
            emb = self._encoder.encode(texts, normalize_embeddings=True, batch_size=max(len(texts), _MIN_ENCODE_BATCH))
            queries = np.ascontiguousarray(np.asarray(emb, dtype=np.float32)).reshape(len(texts), -1)
            sims, idx = self._faiss.search(queries, self.k)
        return idx.astype(np.int64), (1.0 - sims).astype(np.float64)   # float32 subtraction, then widened, as v3

    def estimate_batch(self, prompts: Sequence[str], budget_tokens: int) -> list[dict[str, Estimate]]:
        """One {model: Estimate} per prompt, models in index order."""
        if not prompts:
            return []
        idx, dist = self.search(prompts)
        return [self._aggregate(idx[i], dist[i], budget_tokens) for i in range(len(prompts))]

    def _aggregate(self, idx: np.ndarray, dist: np.ndarray, budget_tokens: int) -> dict[str, Estimate]:
        if not np.all(dist + _DISTANCE_EPS > 0):
            raise ArtifactError(f"a neighbour's distance {float(dist.min())!r} is below -{_DISTANCE_EPS}: rows not unit")
        w = 1.0 / (dist + _DISTANCE_EPS)
        w /= w.sum()
        w32 = w.astype(np.float32)   # v3 kept a float32 copy for the histogram and the budget share
        out = {}
        for m in self.index.models:
            lengths = self.index.lengths[m][idx]
            if self.prior == "knn":
                quality = float(np.average(self.index.labels[m][idx], weights=w))
            else:
                quality = self._mean_label[m]
            out[m] = Estimate(quality=quality, length=float(np.average(lengths, weights=w)),
                              bucket_probs=_bucket_probs(lengths, w32),
                              p_under_budget=float(np.sum(w32[lengths <= budget_tokens]) / max(w32.sum(), 1e-9)))
        return out


def _bucket_probs(lengths: np.ndarray, w32: np.ndarray) -> tuple[float, ...]:
    probs = np.zeros(NUM_BUCKETS, dtype=np.float32)
    for length, w in zip(lengths, w32, strict=True):    # float32 accumulation in rank order, as v3
        probs[min(int(length / BUCKET_TOKENS), NUM_BUCKETS - 1)] += w
    total = float(probs.sum())
    if total > 0:
        probs = probs / total
    return tuple(float(p) for p in probs)


def load_knn(cfg: EstimatorConfig, artifact_root: Path, fleet_models: Collection[str]) -> KnnEstimator:
    """The configured estimator; every model the fleet serves must be in the index (v3 silently scored a
    missing one with quality 0.5 and the maximum length)."""
    index = load_index(Path(artifact_root) / cfg.index_dir, cfg.index_md5)
    absent = sorted(set(fleet_models) - set(index.models))
    if absent:
        raise ArtifactError(f"the KNN index has no labels for fleet models {absent}; it covers {list(index.models)}")
    from sentence_transformers import SentenceTransformer   # torch loads only where an encoder is really needed
    encoder = SentenceTransformer(index.encoder_name, device="cpu", revision=cfg.encoder_revision)
    if encoder.get_sentence_embedding_dimension() != index.embeddings.shape[1]:
        raise ArtifactError(f"encoder {index.encoder_name} does not produce {index.embeddings.shape[1]}-d embeddings")
    est = KnnEstimator(index, encoder, k=cfg.k, prior=cfg.prior)
    est.search(["construction probe"])   # the first encode pays one-time setup here, not in a measured cell
    return est
