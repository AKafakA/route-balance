#!/usr/bin/env python3
"""The k-NN quality and output-length index the scheduler's estimator loads (routebalance/estimate/knn.py), built from
the training rows: the builder the paper's runs used, unchanged but for this module's layout.

    python -m tools.knn_index --input ROWS.jsonl --k 50 --output-dir DIR --device cpu

ROWS is `router_training filter`'s output (one JSON object per line: the prompt, and for every model its output
length and its judged scores). Every prompt is embedded with --embedding-model (default
sentence-transformers/all-MiniLM-L6-v2, normalised); DIR/knn_estimator.pkl holds the embeddings, k, the model names
and every model's output lengths and judged scores per row, and DIR/knn_metadata.json their counts. With --test-input,
the index is evaluated on a held-out file (DIR/evaluation_results.json); without it, on 10 training rows.

The pickle's md5 differs from machine to machine (the embeddings' last bits); a configuration names an index by its
md5, so a rebuilt index is pinned again (docs/reproduce/02-run-experiments.md, step 2).
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# The DeepEval judge key: the quality metric the index stores and the scheduler reads.
DEEPEVAL_JUDGE_KEY = "deepeval-llama3.1-8b-it_reference"


class KNNEstimator:
    """KNN-based length and quality estimator.

    Stores training embeddings and per-model labels (output_length, quality scores).
    At inference: embed query prompt, find k nearest neighbors, return
    distance-weighted aggregation of neighbor values.

    Each judge's scores are kept in their own array (per_judge_scores) and averaged
    over the neighbours separately; the scheduler reads the DeepEval judge's.
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        k: int = 10,
        device: str = "cpu",
        index_type: str = "numpy",  # "numpy" | "faiss_flat" | "faiss_ivfpq"
    ):
        self.embedding_model_name = embedding_model_name
        self.k = k
        self.device = device
        self.index_type = index_type

        # Loaded at build/load time
        self.embeddings: Optional[np.ndarray] = None  # (N, dim)
        self.model_names: List[str] = []
        # Per-model arrays: model_name -> (N,) array
        self.output_lengths: Dict[str, np.ndarray] = {}
        self.similarity_scores: Dict[str, np.ndarray] = {}
        self.llm_judge_scores: Dict[str, np.ndarray] = {}  # legacy: one judge's scores (build_index picks it)
        self.reference_similarity_scores: Dict[str, np.ndarray] = {}
        # Per-judge score arrays: {model_name: {judge_key: np.ndarray}}
        self.per_judge_scores: Dict[str, Dict[str, np.ndarray]] = {}

        self._encoder = None
        self._faiss_index = None  # FAISS index (built lazily)

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._encoder = SentenceTransformer(
                self.embedding_model_name, device=self.device
            )
            logger.info(f"Embedding model loaded: dim={self._encoder.get_sentence_embedding_dimension()}")
        return self._encoder

    def build_index(self, training_data: List[Dict]):
        """Build KNN index from processed training data.

        Args:
            training_data: List of dicts with {prompt, input_len, models: {model: {output_length, ...}}}
                Supports both old schema (quality_score, ttft) and new schema
                (similarity_score, llm_judge_scores dict).
        """
        n = len(training_data)
        logger.info(f"Building KNN index from {n} training examples...")

        # Extract prompts and encode
        prompts = [d["prompt"] for d in training_data]
        encoder = self._get_encoder()
        self.embeddings = encoder.encode(
            prompts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )
        logger.info(f"Embeddings shape: {self.embeddings.shape}")

        # Collect model names from first example
        self.model_names = sorted(training_data[0]["models"].keys())

        # Detect schema
        first_model_data = next(iter(training_data[0]["models"].values()))
        has_new_schema = "similarity_score" in first_model_data or "llm_judge_scores" in first_model_data
        if has_new_schema:
            logger.info("Detected new schema (similarity_score + llm_judge_scores)")
        else:
            logger.info("Detected old schema (quality_score)")

        # Collect all judge keys from first entry
        first_model_data = next(iter(training_data[0]["models"].values()))
        all_judge_keys = sorted(first_model_data.get("llm_judge_scores", {}).keys())
        logger.info(f"Judge keys found: {all_judge_keys}")

        # Build per-model label arrays
        for model in self.model_names:
            lengths = []
            sim_scores = []
            ref_scores = []
            per_judge = {jk: [] for jk in all_judge_keys}

            for d in training_data:
                m_data = d["models"].get(model, {})
                lengths.append(m_data.get("output_length", 0))
                sim_scores.append(m_data.get("similarity_score", 0.0))
                ref_scores.append(float(m_data.get("reference_score", 0.0) or 0.0))
                # Store each judge score separately
                js = m_data.get("llm_judge_scores", {})
                for jk in all_judge_keys:
                    per_judge[jk].append(float(js.get(jk, 0.0) or 0.0))

            self.output_lengths[model] = np.array(lengths, dtype=np.float32)
            self.similarity_scores[model] = np.array(sim_scores, dtype=np.float32)
            self.reference_similarity_scores[model] = np.array(ref_scores, dtype=np.float32)
            # Per-judge arrays
            self.per_judge_scores[model] = {
                jk: np.array(vals, dtype=np.float32) for jk, vals in per_judge.items()
            }
            # Legacy: llm_judge_scores = the first Qwen judge whose key is not a blind one (backward compat)
            qwen_key = next((k for k in all_judge_keys if "Qwen" in k and "blind" not in k.lower()), None)
            if qwen_key and qwen_key in self.per_judge_scores[model]:
                self.llm_judge_scores[model] = self.per_judge_scores[model][qwen_key]
            else:
                # Fallback: first non-protectai, non-deepeval judge
                fallback = next((k for k in all_judge_keys if "protectai" not in k and "deepeval" not in k), None)
                self.llm_judge_scores[model] = self.per_judge_scores[model].get(fallback, np.zeros(len(lengths), dtype=np.float32))

        logger.info(
            f"Index built: {n} examples, {len(self.model_names)} models, "
            f"embedding_dim={self.embeddings.shape[1]}"
        )

    def predict_length(
        self, prompt: str, model_name: str, weighted: bool = True
    ) -> Dict[str, float]:
        """Predict output length for a prompt on a given model.

        Returns:
            Dict with mean, p50, p90, p95 predictions
        """
        indices, distances = self._find_neighbors(prompt)
        weights = self._distance_weights(distances, weighted=weighted)

        values = self.output_lengths[model_name][indices]
        weighted_mean = float(np.average(values, weights=weights))

        return {
            "mean": weighted_mean,
            "p50": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "raw_values": values.tolist(),
        }

    def predict_quality(
        self, prompt: str, model_name: str, weighted: bool = True
    ) -> float:
        """Predict quality score for a prompt on a given model.

        Uses reference_score (unified quality: exact-match for factual datasets,
        embedding similarity for others, refusal score for harmful prompts).
        Falls back to llm_judge_scores if reference_score not available.
        """
        indices, distances = self._find_neighbors(prompt)
        weights = self._distance_weights(distances, weighted=weighted)
        # Prefer reference_score (ground-truth-grounded)
        if model_name in self.reference_similarity_scores:
            values = self.reference_similarity_scores[model_name][indices]
            score = float(np.average(values, weights=weights))
            if score > 0:
                return score
        # Fallback to judge scores
        if model_name in self.llm_judge_scores:
            values = self.llm_judge_scores[model_name][indices]
            return float(np.average(values, weights=weights))
        return 0.0

    def predict_reference_similarity(
        self, prompt: str, model_name: str, weighted: bool = True
    ) -> float:
        if model_name not in self.reference_similarity_scores:
            return 0.0
        indices, distances = self._find_neighbors(prompt)
        weights = self._distance_weights(distances, weighted=weighted)
        values = self.reference_similarity_scores[model_name][indices]
        return float(np.average(values, weights=weights))

    def predict_similarity(
        self, prompt: str, model_name: str, weighted: bool = True
    ) -> float:
        if model_name not in self.similarity_scores:
            return 0.0
        indices, distances = self._find_neighbors(prompt)
        weights = self._distance_weights(distances, weighted=weighted)
        values = self.similarity_scores[model_name][indices]
        return float(np.average(values, weights=weights))

    def predict_judge(
        self, prompt: str, model_name: str, weighted: bool = True
    ) -> float:
        if model_name not in self.llm_judge_scores:
            return 0.0
        indices, distances = self._find_neighbors(prompt)
        weights = self._distance_weights(distances, weighted=weighted)
        values = self.llm_judge_scores[model_name][indices]
        return float(np.average(values, weights=weights))

    def predict_by_judge_key(
        self, prompt: str, model_name: str, judge_key: str, weighted: bool = True
    ) -> float:
        """Predict score for a specific judge key (e.g., deepeval, prometheus)."""
        pj = self.per_judge_scores.get(model_name, {})
        if judge_key not in pj:
            return 0.0
        indices, distances = self._find_neighbors(prompt)
        weights = self._distance_weights(distances, weighted=weighted)
        values = pj[judge_key][indices]
        return float(np.average(values, weights=weights))

    def predict_bucket(
        self, prompt: str, model_name: str,
        bucket_size: int = 64, num_buckets: int = 8,
        weighted: bool = True,
    ) -> np.ndarray:
        """Predict bucket distribution over output length for a (prompt, model).

        Returns array of shape (num_buckets,) summing to 1. Bucket i covers
        token range [i*bucket_size, (i+1)*bucket_size); last bucket is open-ended.
        """
        if model_name not in self.output_lengths:
            return np.ones(num_buckets) / num_buckets
        indices, distances = self._find_neighbors(prompt)
        weights = self._distance_weights(distances, weighted=weighted)
        lengths = self.output_lengths[model_name][indices]
        bucket_idx = np.minimum(lengths // bucket_size, num_buckets - 1).astype(int)
        probs = np.zeros(num_buckets, dtype=np.float64)
        for bi, w in zip(bucket_idx, weights):
            probs[bi] += w
        s = probs.sum()
        if s > 0:
            probs /= s
        else:
            probs[:] = 1.0 / num_buckets
        return probs

    def predict_all_models(
        self, prompt: str
    ) -> Dict[str, Dict[str, float]]:
        """Predict length and quality for all models at once.

        Returns:
            {model_name: {length_mean, length_p95, reference_score, judge_score,
                          deepeval_score?}}

        `deepeval_score` is present only for models that carry a DeepEval array.
        Callers that need DeepEval must read `deepeval_score` and treat its absence
        as an error, never default it.
        """
        indices, distances = self._find_neighbors(prompt)
        weights = self._distance_weights(distances)

        results = {}
        for model in self.model_names:
            lengths = self.output_lengths[model][indices]
            ref_scores = self.reference_similarity_scores.get(model, np.zeros(len(indices)))[indices]
            judge_scores = self.llm_judge_scores.get(model, np.zeros(len(indices)))[indices]
            results[model] = {
                "length_mean": float(np.average(lengths, weights=weights)),
                "length_p95": float(np.percentile(lengths, 95)),
                "reference_score": float(np.average(ref_scores, weights=weights)),
                "judge_score": float(np.average(judge_scores, weights=weights)),
            }
            de = self.per_judge_scores.get(model, {}).get(DEEPEVAL_JUDGE_KEY)
            if de is not None:
                results[model]["deepeval_score"] = float(
                    np.average(de[indices], weights=weights)
                )
        return results

    def _build_faiss_index(self):
        """Build FAISS index from embeddings (called lazily on first search)."""
        if self._faiss_index is not None or self.embeddings is None:
            return
        if self.index_type == "numpy":
            return

        try:
            import faiss
        except ImportError:
            logger.warning("faiss not available, falling back to numpy search")
            self.index_type = "numpy"
            return

        dim = self.embeddings.shape[1]
        n = self.embeddings.shape[0]

        if self.index_type == "faiss_flat":
            # Exact search using inner product (cosine sim for normalized vectors)
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(self.embeddings.astype(np.float32))
            logger.info(f"FAISS Flat index built: {n} vectors, dim={dim}")

        elif self.index_type == "faiss_ivfpq":
            # Approximate search: IVF + Product Quantization
            nlist = min(int(np.sqrt(n)), 256)  # number of Voronoi cells
            m_pq = min(8, dim // 4)  # PQ sub-vectors (must divide dim)
            # Ensure m_pq divides dim
            while dim % m_pq != 0 and m_pq > 1:
                m_pq -= 1
            nbits = 8  # bits per sub-vector

            quantizer = faiss.IndexFlatIP(dim)
            self._faiss_index = faiss.IndexIVFPQ(
                quantizer, dim, nlist, m_pq, nbits,
                faiss.METRIC_INNER_PRODUCT
            )
            # Train on embeddings
            self._faiss_index.train(self.embeddings.astype(np.float32))
            self._faiss_index.add(self.embeddings.astype(np.float32))
            # Search more cells for better recall
            self._faiss_index.nprobe = min(16, nlist)
            logger.info(
                f"FAISS IVF-PQ index built: {n} vectors, dim={dim}, "
                f"nlist={nlist}, m_pq={m_pq}, nprobe={self._faiss_index.nprobe}"
            )
        else:
            logger.warning(f"Unknown index_type '{self.index_type}', using numpy")
            self.index_type = "numpy"

    def _find_neighbors(self, prompt: str) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors for a prompt.

        Uses FAISS index if available, falls back to numpy brute-force.

        Returns:
            (indices, distances) arrays of shape (k,)
            distances are in [0, 2] range (1 - cosine_sim)
        """
        encoder = self._get_encoder()
        query = encoder.encode([prompt], normalize_embeddings=True)

        k = min(self.k, len(self.embeddings))

        # Build FAISS index lazily on first call
        if self._faiss_index is None and self.index_type != "numpy":
            self._build_faiss_index()

        if self._faiss_index is not None:
            # FAISS search (returns inner product similarities for IP index)
            query_f32 = query.astype(np.float32)
            similarities_k, indices_k = self._faiss_index.search(query_f32, k)
            top_indices = indices_k.squeeze()
            top_sims = similarities_k.squeeze()
            # Convert similarity to distance
            distances = 1 - top_sims
            return top_indices.astype(np.int64), distances.astype(np.float64)

        # Numpy fallback: brute-force cosine similarity
        similarities = self.embeddings @ query.T  # (N, 1)
        similarities = similarities.squeeze()  # (N,)

        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        # Convert similarity to distance (1 - cosine_sim)
        distances = 1 - similarities[top_indices]

        return top_indices, distances

    def _distance_weights(self, distances: np.ndarray, weighted: bool = True) -> np.ndarray:
        """Convert distances to weights. If weighted=False, return uniform 1/k."""
        if not weighted:
            return np.ones_like(distances) / len(distances)
        eps = 1e-6
        weights = 1.0 / (distances + eps)
        weights /= weights.sum()
        return weights

    def save(self, output_dir: str):
        """Save model to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        state = {
            "embedding_model_name": self.embedding_model_name,
            "k": self.k,
            "embeddings": self.embeddings,
            "model_names": self.model_names,
            "output_lengths": self.output_lengths,
            "similarity_scores": self.similarity_scores,
            "llm_judge_scores": self.llm_judge_scores,
            "reference_similarity_scores": self.reference_similarity_scores,
            "per_judge_scores": self.per_judge_scores,
        }
        with open(output_path / "knn_estimator.pkl", "wb") as f:
            pickle.dump(state, f)

        # Also save metadata as JSON for inspection
        metadata = {
            "embedding_model_name": self.embedding_model_name,
            "k": self.k,
            "num_examples": len(self.embeddings) if self.embeddings is not None else 0,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "model_names": self.model_names,
        }
        with open(output_path / "knn_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"KNN estimator saved to {output_path}")

    @classmethod
    def load(
        cls, model_dir: str, device: str = "cpu",
        index_type: str = "numpy",
    ) -> "KNNEstimator":
        """Load model from disk.

        Args:
            model_dir: Directory containing knn_estimator.pkl
            device: Device for sentence-transformer ("cpu" or "cuda")
            index_type: Search index type:
                - "numpy": brute-force numpy dot product (default)
                - "faiss_flat": exact FAISS inner product search
                - "faiss_ivfpq": approximate FAISS IVF-PQ search
        """
        model_path = Path(model_dir) / "knn_estimator.pkl"
        with open(model_path, "rb") as f:
            state = pickle.load(f)

        estimator = cls(
            embedding_model_name=state["embedding_model_name"],
            k=state["k"],
            device=device,
            index_type=index_type,
        )
        estimator.embeddings = state["embeddings"]
        estimator.model_names = state["model_names"]
        estimator.output_lengths = state["output_lengths"]
        estimator.similarity_scores = state.get("similarity_scores", {})
        estimator.reference_similarity_scores = state.get("reference_similarity_scores", {})
        estimator.llm_judge_scores = state.get("llm_judge_scores", {})
        estimator.per_judge_scores = state.get("per_judge_scores", {})

        logger.info(
            f"KNN estimator loaded: {len(estimator.embeddings)} examples, "
            f"{len(estimator.model_names)} models, index_type={index_type}"
        )
        return estimator


def evaluate(estimator: KNNEstimator, test_data: list) -> dict:
    """Evaluate estimator on test data.

    Returns dict with per-model metrics for both length and quality.
    """
    from collections import defaultdict

    length_errors = defaultdict(list)  # model -> list of absolute errors
    quality_errors = defaultdict(list)
    quality_rankings = []  # list of (true_best, predicted_best)

    for req in test_data:
        prompt = req["prompt"]

        # Get predictions for all models at once
        predictions = estimator.predict_all_models(prompt)

        # Per-model evaluation
        true_qualities = {}
        pred_qualities = {}
        for model_name, true_vals in req["models"].items():
            if model_name not in predictions:
                continue
            pred = predictions[model_name]

            # Length
            true_len = true_vals["output_length"]
            pred_len = pred["length_mean"]
            length_errors[model_name].append(abs(true_len - pred_len))

            # Quality: the DeepEval score only, never another judge's score or a
            # reference score in its place.
            judge_scores = true_vals.get("llm_judge_scores", {})
            true_q = judge_scores.get("deepeval-llama3.1-8b-it_reference")
            if true_q is None:
                raise RuntimeError(
                    f"tools.knn_index: no deepeval score for model "
                    f"{model_name!r}. Refusing to substitute reference_score or "
                    "another judge — that silently changes the metric."
                )
            true_q = float(true_q)
            # The DeepEval prediction, read explicitly and never defaulted.
            if "deepeval_score" not in pred:
                raise RuntimeError(
                    f"tools.knn_index: predict_all_models returned no "
                    f"'deepeval_score' for {model_name!r} (keys: "
                    f"{sorted(pred)}). The index carries no DeepEval array for "
                    "this model — rebuild it rather than scoring against a "
                    "default."
                )
            pred_q = float(pred["deepeval_score"])
            quality_errors[model_name].append(abs(true_q - pred_q))

            true_qualities[model_name] = true_q
            pred_qualities[model_name] = pred_q

        # Best model selection accuracy
        if true_qualities and pred_qualities:
            true_best = max(true_qualities, key=true_qualities.get)
            pred_best = max(pred_qualities, key=pred_qualities.get)
            quality_rankings.append((true_best, pred_best))

    # Compute aggregate metrics
    results = {"length": {}, "quality": {}}

    for model in sorted(length_errors.keys()):
        errs = length_errors[model]
        results["length"][model] = {
            "mae": float(np.mean(errs)),
            "median_ae": float(np.median(errs)),
            "acc_50": float(np.mean([1 if e <= 50 else 0 for e in errs])),
            "acc_100": float(np.mean([1 if e <= 100 else 0 for e in errs])),
            "n": len(errs),
        }

    for model in sorted(quality_errors.keys()):
        errs = quality_errors[model]
        results["quality"][model] = {
            "mae": float(np.mean(errs)),
            "n": len(errs),
        }

    # Best model selection accuracy
    if quality_rankings:
        correct = sum(1 for t, p in quality_rankings if t == p)
        results["quality"]["best_model_accuracy"] = correct / len(quality_rankings)
        results["quality"]["n_rankings"] = len(quality_rankings)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build the k-NN quality and length index"
    )
    parser.add_argument("--input", required=True, help="Training data JSON")
    parser.add_argument("--test-input", default=None, help="Test data JSON (for evaluation)")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model for embeddings",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    parser.add_argument(
        "--output-dir",
        default="knn_index",
        help="Output directory for trained model",
    )
    parser.add_argument("--device", default="cpu", help="Device for embeddings")
    args = parser.parse_args()

    # Load training data (supports both JSON and JSONL)
    with open(args.input) as f:
        if args.input.endswith(".jsonl"):
            train_data = [json.loads(line) for line in f]
        else:
            raw = json.load(f)
            train_data = raw["requests"] if "requests" in raw else raw
    logger.info(f"Training data: {len(train_data)} requests")

    # Build KNN index
    t0 = time.time()
    estimator = KNNEstimator(
        embedding_model_name=args.embedding_model,
        k=args.k,
        device=args.device,
    )
    estimator.build_index(train_data)
    build_time = time.time() - t0
    logger.info(f"Index built in {build_time:.1f}s")

    # Save model
    estimator.save(args.output_dir)

    # Evaluate on test data if provided
    if args.test_input:
        with open(args.test_input) as f:
            if args.test_input.endswith(".jsonl"):
                test_data = [json.loads(line) for line in f]
            else:
                raw = json.load(f)
                test_data = raw["requests"] if "requests" in raw else raw
        logger.info(f"Test data: {len(test_data)} requests")

        t0 = time.time()
        results = evaluate(estimator, test_data)
        eval_time = time.time() - t0

        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)

        print("\nLength Prediction (MAE / Acc@50 / Acc@100):")
        for model, metrics in sorted(results["length"].items()):
            print(
                f"  {model}: MAE={metrics['mae']:.1f}, "
                f"Acc@50={metrics['acc_50']:.3f}, "
                f"Acc@100={metrics['acc_100']:.3f} "
                f"(n={metrics['n']})"
            )

        print("\nQuality Prediction (MAE):")
        for model, metrics in sorted(results["quality"].items()):
            if isinstance(metrics, dict) and "mae" in metrics:
                print(f"  {model}: MAE={metrics['mae']:.4f} (n={metrics['n']})")

        if "best_model_accuracy" in results["quality"]:
            print(
                f"\nBest Model Selection Accuracy: "
                f"{results['quality']['best_model_accuracy']:.3f} "
                f"(n={results['quality']['n_rankings']})"
            )

        print(f"\nEvaluation time: {eval_time:.1f}s")
        print(f"Inference speed: {eval_time/len(test_data)*1000:.1f}ms per query")

        # Save evaluation results
        eval_path = Path(args.output_dir) / "evaluation_results.json"
        with open(eval_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {eval_path}")
    else:
        # Self-evaluate on training data (just to verify it works)
        logger.info("No test data provided, running quick self-evaluation on 10 training samples...")
        results = evaluate(estimator, train_data[:10])
        print("\nSelf-evaluation (training data, 10 samples):")
        for model, metrics in sorted(results["length"].items()):
            print(f"  {model}: MAE={metrics['mae']:.1f}")


if __name__ == "__main__":
    main()
