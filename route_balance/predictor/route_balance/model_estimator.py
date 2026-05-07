"""
Model Estimator for ROUTE_BALANCE scheduler.

Provides a general interface to predict per-model (score, length, budget compliance)
given a prompt. Runs at the scheduler side, once per prompt before scheduling.

Separates prompt-dependent predictions (quality, length) from instance-state-dependent
predictions (TTFT, TPOT) which remain per-instance.

Default implementation combines:
- ModernBERT bucket classifiers for output length distribution (per target model)
- KNN estimator for quality scores (similarity, judge, or combined)
"""
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelEstimate:
    """Per-model prediction result for a single prompt."""
    model_name: str
    length_expected: float                          # E[output_tokens]
    length_bucket_probs: Optional[np.ndarray]       # bucket distribution (num_buckets,)
    p_under_budget: float                           # P(tokens <= budget)
    score: float                                    # quality score [0,1]
    score_type: str                                 # "judge" | "similarity" | "combined"


class ModelEstimator(ABC):
    """Abstract interface: given a prompt, return per-model estimates.

    Implementations can use KNN, ModernBERT, MLP, or any combination.
    """

    @abstractmethod
    def estimate(
        self, prompt: str, budget_tokens: int = 256
    ) -> Dict[str, ModelEstimate]:
        """Full estimation: returns {model_name: ModelEstimate} for all known models."""
        ...

    @abstractmethod
    def estimate_length(self, prompt: str) -> Dict[str, float]:
        """Lightweight length-only estimation for LPT sort.

        Returns {model_name: expected_length}.
        """
        ...

    @property
    @abstractmethod
    def model_names(self) -> List[str]:
        """List of model names this estimator knows about."""
        ...


class DefaultModelEstimator(ModelEstimator):
    """Default implementation combining ModernBERT (length) + KNN (quality).

    Supports two modes for bucket classifiers:
    - **Fused** (preferred): single multi-head model, one forward pass for all models.
      Config: bucket_config.fused_model_dir pointing to a fused_model.pt checkpoint.
    - **Per-model** (legacy): separate ModernBERT per target model.
      Config: bucket_config.model_dir + model_map.

    KNN estimator is loaded once and provides all models in a single lookup.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        """
        Args:
            config: Dict with keys:
                - score_type: "judge" | "similarity" | "combined" (default "judge")
                - bucket_config: {fused_model_dir, model_dir, model_map, bucket_size, max_buckets, max_length}
                - quality_config: {type, model_dir, embedding_model, k}
            device: Device for ModernBERT inference (default "cpu")
        """
        self._score_type = config.get("score_type", "judge")
        self._device = device

        # Bucket classifier config
        bucket_cfg = config.get("bucket_config", {})
        self._bucket_size = bucket_cfg.get("bucket_size", 64)
        self._max_buckets = bucket_cfg.get("max_buckets", 16)
        self._max_input_length = bucket_cfg.get("max_length", 1024)

        # Fused multi-head models (preferred: 1 forward pass → all models)
        # Each is a (MultiHeadEncoder, tokenizer) tuple
        self._fused_bucket = None       # fused bucket classifier
        self._fused_bucket_tok = None
        self._fused_length = None       # fused length regressor
        self._fused_length_tok = None
        self._fused_length_log_transform = False

        # Per-model bucket classifiers (legacy fallback): {model_name: (model, tokenizer)}
        self._bucket_models: Dict[str, tuple] = {}

        # KNN estimator (shared across all models — already "fused" by design:
        # one FAISS lookup returns quality/similarity/reference scores for all model sizes)
        self._knn = None

        self._load_knn(config.get("quality_config", {}))
        self._load_bucket_classifiers(bucket_cfg)
        self._load_length_regressor(config.get("length_config", {}))

    def _load_knn(self, quality_config: dict):
        """Load KNN estimator."""
        knn_dir = quality_config.get("model_dir", "")
        if not knn_dir or not Path(knn_dir).exists():
            logger.warning(f"KNN model dir not found: {knn_dir}")
            return

        try:
            from route_balance.predictor.route_balance.estimators.knn_estimator import KNNEstimator
            knn_device = quality_config.get("device", "cpu")
            knn_index_type = quality_config.get("index_type", "faiss_flat")
            self._knn = KNNEstimator.load(
                knn_dir, device=knn_device, index_type=knn_index_type
            )
            logger.info(
                f"ModelEstimator: KNN loaded ({len(self._knn.model_names)} models, "
                f"{len(self._knn.embeddings)} examples)"
            )
        except Exception as e:
            logger.error(f"Failed to load KNN estimator: {e}")

    def _load_bucket_classifiers(self, bucket_config: dict):
        """Load bucket classifiers — fused (preferred) or per-model (legacy).

        Fused: single multi-head model → one forward pass for all models.
        Per-model: separate AutoModelForSequenceClassification per model.
        """
        # Try fused model first
        fused_dir = bucket_config.get("fused_model_dir", "")
        if fused_dir and Path(fused_dir).exists() and (Path(fused_dir) / "fused_model.pt").exists():
            try:
                from route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor import (
                    load_fused_model,
                )
                self._fused_bucket, self._fused_bucket_tok = load_fused_model(
                    fused_dir, device=self._device,
                )
                logger.info(
                    f"ModelEstimator: Fused bucket classifier loaded "
                    f"({len(self._fused_bucket.model_names)} models in 1 model)"
                )
                return
            except Exception as e:
                # Fused model was explicitly configured — don't silently fall back.
                # Raise so the operator knows the estimator is misconfigured.
                raise RuntimeError(
                    f"Failed to load fused bucket model from {fused_dir}: {e}. "
                    f"Fix the model/tokenizer or remove fused_model_dir from config."
                ) from e

        # Legacy: per-model classifiers
        bucket_dir = bucket_config.get("model_dir", "")
        model_map = bucket_config.get("model_map", {})

        if not bucket_dir or not model_map:
            logger.info("ModelEstimator: No bucket classifiers configured")
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch  # noqa: F401 — needed at runtime
        except ImportError:
            logger.warning("transformers not available, bucket classifiers disabled")
            return

        for model_name, subdir in model_map.items():
            model_path = Path(bucket_dir) / subdir
            if not model_path.exists():
                logger.warning(f"Bucket classifier not found: {model_path}")
                continue

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path), trust_remote_code=True
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(model_path), trust_remote_code=True
                )
                model.eval()
                model = model.to(self._device)
                self._bucket_models[model_name] = (model, tokenizer)
                logger.info(f"Bucket classifier loaded for {model_name}")
            except Exception as e:
                logger.error(f"Failed to load bucket classifier for {model_name}: {e}")

        logger.info(
            f"ModelEstimator: {len(self._bucket_models)}/{len(model_map)} "
            f"bucket classifiers loaded"
        )

    def _load_length_regressor(self, length_config: dict):
        """Load fused length regression model (optional, for LPT sort)."""
        fused_dir = length_config.get("fused_model_dir", "")
        if not fused_dir or not Path(fused_dir).exists():
            return
        if not (Path(fused_dir) / "fused_model.pt").exists():
            return

        try:
            from route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor import (
                load_fused_model,
            )
            import json
            self._fused_length, self._fused_length_tok = load_fused_model(
                fused_dir, device=self._device,
            )
            # Check if log-transform was used
            config_path = Path(fused_dir) / "fused_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                self._fused_length_log_transform = cfg.get("log_transform", False)

            logger.info(
                f"ModelEstimator: Fused length regressor loaded "
                f"({len(self._fused_length.model_names)} models, "
                f"log_transform={self._fused_length_log_transform})"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load fused length model from {fused_dir}: {e}. "
                f"Fix the model/tokenizer or remove length_config from config."
            ) from e

    @property
    def model_names(self) -> List[str]:
        """List of model names from KNN (canonical source)."""
        if self._knn is not None:
            return self._knn.model_names
        if self._fused_bucket is not None:
            return self._fused_bucket.model_names
        return list(self._bucket_models.keys())

    @property
    def _has_fused_bucket(self) -> bool:
        """Whether fused multi-head bucket model is loaded."""
        return self._fused_bucket is not None

    @property
    def _has_fused_length(self) -> bool:
        """Whether fused multi-head length regressor is loaded."""
        return self._fused_length is not None

    def _predict_buckets_fused(self, prompt: str) -> Dict[str, np.ndarray]:
        """Predict bucket distributions for ALL models in one forward pass.

        Calls the model via __call__ (not .predict()) so torch.compile-wrapped
        modules route through the compiled forward. Constructs the per-model
        dict locally (mirrors MultiHeadEncoder.predict logic).
        """
        import torch

        inputs = self._fused_bucket_tok(
            prompt,
            truncation=True,
            max_length=self._max_input_length,
            return_tensors="pt",
        )
        device = next(self._fused_bucket.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Go through __call__ so torch.compile wrapper is active. Output:
        # {"loss": None, "logits": (batch, num_models, num_labels)}
        with torch.no_grad():
            out = self._fused_bucket(**inputs)
        all_logits = out["logits"]
        # Access model_names via _orig_mod if the module is compile-wrapped
        model_names = getattr(self._fused_bucket, "model_names", None) or \
                      getattr(self._fused_bucket, "_orig_mod", self._fused_bucket).model_names
        result = {}
        for i, name in enumerate(model_names):
            probs = torch.softmax(all_logits[:, i, :].squeeze(0), dim=-1).cpu().numpy()
            result[name] = probs
        return result

    def _predict_buckets_fused_batch(self, prompts: List[str]) -> Dict[str, List[np.ndarray]]:
        """Batch predict bucket distributions for ALL models in one forward pass.

        See _predict_buckets_fused for why we call the model via __call__.
        """
        import torch

        inputs = self._fused_bucket_tok(
            prompts,
            truncation=True,
            max_length=self._max_input_length,
            padding=True,
            return_tensors="pt",
        )
        device = next(self._fused_bucket.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._fused_bucket(**inputs)
        all_logits = out["logits"]
        model_names = getattr(self._fused_bucket, "model_names", None) or \
                      getattr(self._fused_bucket, "_orig_mod", self._fused_bucket).model_names
        result = {}
        for i, name in enumerate(model_names):
            probs = torch.softmax(all_logits[:, i, :], dim=-1).cpu().numpy()
            result[name] = [probs[j] for j in range(len(prompts))]
        return result

    def _predict_buckets(self, prompt: str, model_name: str) -> np.ndarray:
        """Predict bucket distribution using per-model ModernBERT (legacy)."""
        import torch

        model, tokenizer = self._bucket_models[model_name]
        inputs = tokenizer(
            prompt,
            truncation=True,
            max_length=self._max_input_length,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits.squeeze()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        return probs

    def _expected_length(self, bucket_probs: np.ndarray) -> float:
        """Compute E[output_tokens] from bucket probabilities."""
        midpoints = np.array([
            (i * self._bucket_size + self._bucket_size / 2)
            for i in range(len(bucket_probs))
        ])
        return float(np.sum(bucket_probs * midpoints))

    def _budget_compliance(
        self, bucket_probs: np.ndarray, budget_tokens: int
    ) -> float:
        """Compute P(output_tokens <= budget) from bucket distribution."""
        budget_bucket = min(
            budget_tokens // self._bucket_size, len(bucket_probs) - 1
        )
        return float(np.sum(bucket_probs[:budget_bucket + 1]))

    def _get_scores_all_models(
        self, indices: np.ndarray, weights: np.ndarray
    ) -> Dict[str, float]:
        """Get quality scores for all models using pre-computed KNN neighbors.

        Avoids redundant embedding calls by reusing indices/weights from
        a single _find_neighbors() call.
        """
        scores = {}
        for model_name in self._knn.model_names:
            if self._score_type == "similarity":
                if model_name in self._knn.similarity_scores:
                    values = self._knn.similarity_scores[model_name][indices]
                    scores[model_name] = float(np.average(values, weights=weights))
                else:
                    scores[model_name] = 0.0
            elif self._score_type == "judge":
                if model_name in self._knn.llm_judge_scores:
                    values = self._knn.llm_judge_scores[model_name][indices]
                    scores[model_name] = float(np.average(values, weights=weights))
                else:
                    scores[model_name] = 0.0
            elif self._score_type == "reference":
                if model_name in self._knn.reference_similarity_scores:
                    values = self._knn.reference_similarity_scores[model_name][indices]
                    scores[model_name] = float(np.average(values, weights=weights))
                else:
                    scores[model_name] = 0.0
            elif self._score_type == "deepeval":
                # Use per-judge array for deepeval
                pj = self._knn.per_judge_scores.get(model_name, {})
                deepeval_key = "deepeval-llama3.1-8b-it_reference"
                if deepeval_key in pj:
                    values = pj[deepeval_key][indices]
                    scores[model_name] = float(np.average(values, weights=weights))
                else:
                    scores[model_name] = 0.0
            else:  # "combined" or unknown — use reference_score
                if model_name in self._knn.reference_similarity_scores:
                    values = self._knn.reference_similarity_scores[model_name][indices]
                    scores[model_name] = float(np.average(values, weights=weights))
                elif model_name in self._knn.llm_judge_scores:
                    values = self._knn.llm_judge_scores[model_name][indices]
                    scores[model_name] = float(np.average(values, weights=weights))
                else:
                    scores[model_name] = 0.0
        return scores

    def estimate(
        self, prompt: str, budget_tokens: int = 256
    ) -> Dict[str, ModelEstimate]:
        """Full estimation for all models.

        Returns {model_name: ModelEstimate} with length, quality, and budget compliance.
        Optimized: single KNN embedding lookup, reused for all models and score types.
        """
        t0 = time.monotonic()
        results = {}

        # KNN: ONE embedding lookup → reuse indices for all models + score types
        knn_results = {}
        scores_map = {}
        if self._knn is not None:
            indices, distances = self._knn._find_neighbors(prompt)
            weights = self._knn._distance_weights(distances)

            # Get lengths for all models
            for model in self._knn.model_names:
                lengths = self._knn.output_lengths[model][indices]
                knn_results[model] = {
                    "length_mean": float(np.average(lengths, weights=weights)),
                }

            # Get scores for all models (reuses same indices)
            scores_map = self._get_scores_all_models(indices, weights)

        # Bucket inference: fused (1 forward pass) or per-model (N forward passes)
        bucket_results = {}  # {model_name: (probs, length, p_budget)}
        if self._has_fused_bucket:
            # Single forward pass → all models
            all_probs = self._predict_buckets_fused(prompt)
            for name, probs in all_probs.items():
                length = self._expected_length(probs)
                p_budget = self._budget_compliance(probs, budget_tokens)
                bucket_results[name] = (probs, length, p_budget)
        else:
            # Legacy: parallel per-model inference
            models_with_buckets = [
                m for m in self.model_names if m in self._bucket_models
            ]
            if models_with_buckets:
                from concurrent.futures import ThreadPoolExecutor

                def _infer_bucket(model_name):
                    probs = self._predict_buckets(prompt, model_name)
                    length = self._expected_length(probs)
                    p_budget = self._budget_compliance(probs, budget_tokens)
                    return model_name, (probs, length, p_budget)

                with ThreadPoolExecutor(max_workers=len(models_with_buckets)) as pool:
                    futures = [pool.submit(_infer_bucket, m) for m in models_with_buckets]
                    for fut in futures:
                        name, result = fut.result()
                        bucket_results[name] = result

        for model_name in self.model_names:
            knn = knn_results.get(model_name, {})

            # Quality score (pre-computed, no extra embedding call)
            score = scores_map.get(model_name, 0.5)

            # Bucket distribution (ModernBERT) — pre-computed in parallel below
            bucket_result = bucket_results.get(model_name)
            if bucket_result is not None:
                bucket_probs, length_expected, p_under_budget = bucket_result
            else:
                # Fallback to KNN length
                length_expected = knn.get("length_mean", 128.0)
                bucket_probs = None
                p_under_budget = (
                    1.0 if length_expected <= budget_tokens else 0.0
                )

            results[model_name] = ModelEstimate(
                model_name=model_name,
                length_expected=length_expected,
                length_bucket_probs=bucket_probs,
                p_under_budget=p_under_budget,
                score=score,
                score_type=self._score_type,
            )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            f"ModelEstimator.estimate: {len(results)} models in {elapsed_ms:.1f}ms"
        )
        return results

    def _predict_buckets_batch(
        self, prompts: List[str], model_name: str
    ) -> List[np.ndarray]:
        """Batch predict bucket distributions for multiple prompts."""
        import torch

        model, tokenizer = self._bucket_models[model_name]
        inputs = tokenizer(
            prompts,
            truncation=True,
            max_length=self._max_input_length,
            padding=True,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits  # (batch, num_classes)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        return [probs[i] for i in range(len(prompts))]

    def estimate_batch(
        self, prompts: List[str], budget_tokens: int = 256
    ) -> List[Dict[str, ModelEstimate]]:
        """Batch estimation for multiple prompts.

        Much faster than calling estimate() N times because:
        - Single batched sentence-transformer embedding
        - Single batched ModernBERT forward pass per model
        - Single FAISS/numpy search for all queries

        Returns list of {model_name: ModelEstimate} dicts, one per prompt.
        """
        if not prompts:
            return []

        t0 = time.monotonic()
        n = len(prompts)

        # 1. Batch KNN: encode all prompts at once
        all_knn_results = [{} for _ in range(n)]
        all_scores_maps = [{} for _ in range(n)]

        if self._knn is not None:
            encoder = self._knn._get_encoder()
            query_embeddings = encoder.encode(
                prompts, normalize_embeddings=True, batch_size=max(n, 32)
            )

            if self._knn._faiss_index is None and self._knn.index_type != "numpy":
                self._knn._build_faiss_index()

            for i, query in enumerate(query_embeddings):
                query = query.reshape(1, -1)
                k = min(self._knn.k, len(self._knn.embeddings))

                if self._knn._faiss_index is not None:
                    q32 = query.astype(np.float32)
                    sims_k, idx_k = self._knn._faiss_index.search(q32, k)
                    indices = idx_k.squeeze().astype(np.int64)
                    distances = (1 - sims_k.squeeze()).astype(np.float64)
                else:
                    sims = self._knn.embeddings @ query.T
                    sims = sims.squeeze()
                    top_idx = np.argpartition(sims, -k)[-k:]
                    indices = top_idx[np.argsort(-sims[top_idx])]
                    distances = 1 - sims[indices]

                weights = self._knn._distance_weights(distances)

                knn_res = {}
                scores_res = {}
                for model in self._knn.model_names:
                    lengths = self._knn.output_lengths[model][indices]
                    knn_res[model] = {
                        "length_mean": float(np.average(lengths, weights=weights)),
                        # Save neighbor lengths + weights so the assembly step can
                        # derive a bucket probability distribution from them
                        # (instead of running a separate RoBERTa forward pass).
                        # This makes the entire estimator a single KNN call: one
                        # MiniLM encode + faiss search → quality, length_mean,
                        # and bucket distribution all from the same K neighbors.
                        "_neighbor_lengths": np.asarray(lengths, dtype=np.float32),
                        "_neighbor_weights": np.asarray(weights, dtype=np.float32),
                    }
                    if self._score_type == "similarity" and model in self._knn.similarity_scores:
                        vals = self._knn.similarity_scores[model][indices]
                    elif self._score_type == "judge" and model in self._knn.llm_judge_scores:
                        vals = self._knn.llm_judge_scores[model][indices]
                    elif model in self._knn.reference_similarity_scores:
                        vals = self._knn.reference_similarity_scores[model][indices]
                    else:
                        scores_res[model] = 0.5
                        continue
                    scores_res[model] = float(np.average(vals, weights=weights))

                all_knn_results[i] = knn_res
                all_scores_maps[i] = scores_res

        # 2. Batch bucket inference: fused (1 pass) or per-model (N passes)
        all_bucket_probs = {}  # {model_name: [probs_per_prompt]}
        if self._has_fused_bucket:
            # Single batched forward pass → all models × all prompts
            all_bucket_probs = self._predict_buckets_fused_batch(prompts)
        else:
            # Legacy: parallel per-model batched inference
            from concurrent.futures import ThreadPoolExecutor
            models_with_buckets = [
                m for m in self.model_names if m in self._bucket_models
            ]
            if models_with_buckets:
                def _batch_infer(model_name):
                    return model_name, self._predict_buckets_batch(prompts, model_name)

                with ThreadPoolExecutor(max_workers=len(models_with_buckets)) as pool:
                    futures = [pool.submit(_batch_infer, m) for m in models_with_buckets]
                    for fut in futures:
                        mname, probs_list = fut.result()
                        all_bucket_probs[mname] = probs_list

        # 3. Assemble per-prompt results
        all_results = []
        for i in range(n):
            results = {}
            knn = all_knn_results[i]
            scores = all_scores_maps[i]

            for model_name in self.model_names:
                score = scores.get(model_name, 0.5)
                knn_m = knn.get(model_name, {})

                if model_name in all_bucket_probs:
                    bucket_probs = all_bucket_probs[model_name][i]
                    length_expected = self._expected_length(bucket_probs)
                    p_under_budget = self._budget_compliance(bucket_probs, budget_tokens)
                elif "_neighbor_lengths" in knn_m:
                    # Neighbor-derived bucket distribution: histogram K neighbor
                    # response lengths into bucket_size-token buckets, weighted by
                    # KNN distance. Gives the same downstream signals (length_expected,
                    # bucket probs for budget control, p_under_budget) as a learned
                    # length-bucket classifier — but free, since the neighbors are
                    # already fetched for the quality lookup. Trades the RoBERTa
                    # fused-batch length forward (~80ms/batch) for a small numpy
                    # histogram (~µs).
                    nl = knn_m["_neighbor_lengths"]
                    nw = knn_m["_neighbor_weights"]
                    bucket_probs = np.zeros(self._max_buckets, dtype=np.float32)
                    for _len, _w in zip(nl, nw):
                        bidx = min(int(_len / self._bucket_size), self._max_buckets - 1)
                        bucket_probs[bidx] += _w
                    _s = float(bucket_probs.sum())
                    if _s > 0:
                        bucket_probs = bucket_probs / _s
                    length_expected = float(knn_m.get("length_mean", 128.0))
                    p_under_budget = float(np.sum(nw[nl <= budget_tokens]) / max(nw.sum(), 1e-9))
                else:
                    length_expected = knn_m.get("length_mean", 128.0)
                    bucket_probs = None
                    p_under_budget = (
                        1.0 if length_expected <= budget_tokens else 0.0
                    )

                results[model_name] = ModelEstimate(
                    model_name=model_name,
                    length_expected=length_expected,
                    length_bucket_probs=bucket_probs,
                    p_under_budget=p_under_budget,
                    score=score,
                    score_type=self._score_type,
                )
            all_results.append(results)

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            f"ModelEstimator.estimate_batch: {n} prompts x "
            f"{len(self.model_names)} models in {elapsed_ms:.1f}ms "
            f"({elapsed_ms/n:.1f}ms/prompt)"
        )
        return all_results

    def estimate_length(self, prompt: str) -> Dict[str, float]:
        """Fast path for LPT sort.

        Priority: fused length regressor > KNN > fused bucket > per-model bucket.
        """
        import torch

        # Best: fused length regressor (1 forward pass → all models)
        if self._has_fused_length:
            inputs = self._fused_length_tok(
                prompt, truncation=True, max_length=self._max_input_length,
                return_tensors="pt",
            )
            device = next(self._fused_length.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Use __call__ so torch.compile wrapper engages (see _predict_buckets_fused)
            with torch.no_grad():
                out = self._fused_length(**inputs)
            all_logits = out["logits"]
            model_names = getattr(self._fused_length, "model_names", None) or \
                          getattr(self._fused_length, "_orig_mod", self._fused_length).model_names
            result = {}
            for i, name in enumerate(model_names):
                val = all_logits[:, i, :].squeeze().item()
                if self._fused_length_log_transform:
                    val = float(np.expm1(val))
                result[name] = val
            return result

        # Good: KNN (1 FAISS lookup → all models)
        if self._knn is not None:
            knn_results = self._knn.predict_all_models(prompt)
            return {m: r["length_mean"] for m, r in knn_results.items()}

        # OK: fused bucket classifier (1 forward pass → expected lengths)
        if self._has_fused_bucket:
            all_probs = self._predict_buckets_fused(prompt)
            return {name: self._expected_length(probs) for name, probs in all_probs.items()}

        # Legacy: per-model bucket classifiers
        result = {}
        for model_name in self._bucket_models:
            probs = self._predict_buckets(prompt, model_name)
            result[model_name] = self._expected_length(probs)
        return result


class PFSModelEstimator(ModelEstimator):
    """PFS-style (Past-Future Scheduler) model estimator baseline.

    Training-free: uses sliding window of recent completions to predict
    output length via empirical distribution. Quality scores default to 0.5
    (PFS doesn't predict quality).

    Config:
        - model_names: list of model names
        - window_size: sliding window size (default 1000)
        - use_input_bins: condition on input length bins (default True)
        - train_data_path: optional path to bootstrap from training data
    """

    def __init__(self, config: dict, device: str = "cpu"):
        from route_balance.predictor.route_balance.estimators.pfs_estimator import PFSEstimator

        model_names = config.get("model_names", [])
        window_size = config.get("window_size", 1000)
        use_input_bins = config.get("use_input_bins", True)
        bucket_size = config.get("bucket_size", 64)
        max_buckets = config.get("max_buckets", 16)

        self._pfs = PFSEstimator(
            model_names=model_names,
            window_size=window_size,
            use_input_bins=use_input_bins,
            bucket_size=bucket_size,
            max_buckets=max_buckets,
        )
        self._bucket_size = bucket_size
        self._max_buckets = max_buckets

        # Bootstrap from training data if provided
        train_path = config.get("train_data_path", "")
        if train_path:
            import json
            from pathlib import Path
            p = Path(train_path)
            if p.exists():
                with open(p) as f:
                    if str(p).endswith(".jsonl"):
                        data = [json.loads(line) for line in f]
                    else:
                        raw = json.load(f)
                        data = raw.get("requests", raw)
                self._pfs.bootstrap(data)

    @property
    def model_names(self) -> List[str]:
        return self._pfs.model_names

    def record_completion(self, model_name: str, output_length: float, input_len: int = 0):
        """Call this after each request completes to update the sliding window."""
        self._pfs.record_completion(model_name, output_length, input_len)

    def estimate(
        self, prompt: str, budget_tokens: int = 256
    ) -> Dict[str, ModelEstimate]:
        t0 = time.monotonic()
        input_len = len(prompt.split())  # approximate token count
        results = {}

        for model_name in self._pfs.model_names:
            bucket_probs = self._pfs.predict_bucket_distribution(model_name, input_len)
            midpoints = np.array([
                (i * self._bucket_size + self._bucket_size / 2)
                for i in range(len(bucket_probs))
            ])
            length_expected = float(np.sum(bucket_probs * midpoints))

            budget_bucket = min(budget_tokens // self._bucket_size, len(bucket_probs) - 1)
            p_under_budget = float(np.sum(bucket_probs[:budget_bucket + 1]))

            results[model_name] = ModelEstimate(
                model_name=model_name,
                length_expected=length_expected,
                length_bucket_probs=bucket_probs,
                p_under_budget=p_under_budget,
                score=0.5,  # PFS doesn't predict quality
                score_type="none",
            )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(f"PFSModelEstimator.estimate: {len(results)} models in {elapsed_ms:.1f}ms")
        return results

    def estimate_length(self, prompt: str) -> Dict[str, float]:
        input_len = len(prompt.split())
        return {
            m: self._pfs.predict_length(m, input_len)
            for m in self._pfs.model_names
        }


class KNNModelEstimator(ModelEstimator):
    """KNN-only model estimator — single embedding call for everything.

    Derives bucket distributions from k neighbors' output lengths (histogram),
    quality scores from weighted average, and budget compliance from the
    empirical CDF. No ModernBERT needed — pure KNN with FAISS index.

    Config:
        - quality_config: {model_dir, embedding_model, k, device, index_type}
        - score_type: "judge" | "similarity" | "reference" | "combined"
        - bucket_size: token bucket width (default 64)
        - max_buckets: number of buckets (default 16)
    """

    def __init__(self, config: dict, device: str = "cpu"):
        from route_balance.predictor.route_balance.estimators.knn_estimator import KNNEstimator

        quality_cfg = config.get("quality_config", {})
        knn_dir = quality_cfg.get("model_dir", "")
        knn_device = quality_cfg.get("device", device)
        knn_index_type = quality_cfg.get("index_type", "faiss_flat")
        k = quality_cfg.get("k", 10)

        if not knn_dir:
            raise ValueError("KNNModelEstimator requires quality_config.model_dir")
        self._knn = KNNEstimator.load(
            knn_dir, device=knn_device, index_type=knn_index_type
        )
        self._knn.k = k
        self._score_type = config.get("score_type", "judge")
        self._bucket_size = config.get("bucket_size", 64)
        self._max_buckets = config.get("max_buckets", 16)

        logger.info(
            f"KNNModelEstimator: loaded ({len(self._knn.model_names)} models, "
            f"{len(self._knn.embeddings)} examples, k={k}, "
            f"index={knn_index_type}, score={self._score_type})"
        )

    @property
    def model_names(self) -> List[str]:
        return self._knn.model_names

    def _neighbor_bucket_probs(
        self, lengths: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Build a bucket distribution from k neighbors' output lengths.

        Weighted histogram: each neighbor contributes its weight to its bucket.
        """
        probs = np.zeros(self._max_buckets, dtype=np.float64)
        for length, w in zip(lengths, weights):
            bucket = min(int(length) // self._bucket_size, self._max_buckets - 1)
            probs[bucket] += w
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs[0] = 1.0
        return probs

    def _get_score(
        self, model_name: str, indices: np.ndarray, weights: np.ndarray
    ) -> float:
        """Get quality score for a model from k neighbors."""
        if self._score_type == "similarity":
            arr = self._knn.similarity_scores.get(model_name)
        elif self._score_type == "judge":
            arr = self._knn.llm_judge_scores.get(model_name)
        elif self._score_type == "reference":
            arr = self._knn.reference_similarity_scores.get(model_name)
        elif self._score_type == "deepeval":
            pj = self._knn.per_judge_scores.get(model_name, {})
            arr = pj.get("deepeval-llama3.1-8b-it_reference")
        else:  # "combined" — use reference_score
            arr = self._knn.reference_similarity_scores.get(model_name)
            if arr is None:
                arr = self._knn.llm_judge_scores.get(model_name)
        if arr is None:
            return 0.0
        return float(np.average(arr[indices], weights=weights))

    def estimate(
        self, prompt: str, budget_tokens: int = 256
    ) -> Dict[str, ModelEstimate]:
        t0 = time.monotonic()

        # Single embedding + neighbor lookup
        indices, distances = self._knn._find_neighbors(prompt)
        weights = self._knn._distance_weights(distances)

        results = {}
        for model_name in self._knn.model_names:
            lengths = self._knn.output_lengths[model_name][indices]

            # Bucket distribution from neighbors
            bucket_probs = self._neighbor_bucket_probs(lengths, weights)

            # Expected length from bucket midpoints
            midpoints = np.array([
                i * self._bucket_size + self._bucket_size / 2
                for i in range(self._max_buckets)
            ])
            length_expected = float(np.sum(bucket_probs * midpoints))

            # Budget compliance from CDF
            budget_bucket = min(
                budget_tokens // self._bucket_size, self._max_buckets - 1
            )
            p_under_budget = float(np.sum(bucket_probs[:budget_bucket + 1]))

            # Quality
            score = self._get_score(model_name, indices, weights)

            results[model_name] = ModelEstimate(
                model_name=model_name,
                length_expected=length_expected,
                length_bucket_probs=bucket_probs,
                p_under_budget=p_under_budget,
                score=score,
                score_type=self._score_type,
            )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            f"KNNModelEstimator.estimate: {len(results)} models in {elapsed_ms:.1f}ms"
        )
        return results

    def estimate_length(self, prompt: str) -> Dict[str, float]:
        indices, distances = self._knn._find_neighbors(prompt)
        weights = self._knn._distance_weights(distances)
        return {
            m: float(np.average(self._knn.output_lengths[m][indices], weights=weights))
            for m in self._knn.model_names
        }
