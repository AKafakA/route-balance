"""KNN adapter for evaluation.

Supports configurable k, weighted/unweighted aggregation, and bucket-distribution
prediction so the eval harness can run a (k × weighted) sweep + bucket comparison
against RoBERTa.
"""

import logging
from typing import Dict
from .base import BaseAdapter

logger = logging.getLogger(__name__)


class KNNAdapter(BaseAdapter):
    def __init__(self, model_dir: str, train_data: list, target: str, device: str,
                 index_type: str = "faiss_flat", k: int = 10, weighted: bool = True,
                 bucket_size: int = 64, num_buckets: int = 8):
        try:
            from route_balance.predictor.route_balance.estimators.knn_estimator import KNNEstimator
        except ImportError:
            from offline_training.estimators.knn_estimator import KNNEstimator
        from pathlib import Path

        self.target = target
        self.weighted = weighted
        self.bucket_size = bucket_size
        self.num_buckets = num_buckets

        # For targets requiring per_judge_scores (deepeval, prometheus), always
        # build from training data since old pickles may not have per_judge_scores.
        needs_per_judge = target in ("deepeval", "prometheus")
        pkl_path = Path(model_dir) / "knn_estimator.pkl"

        if pkl_path.exists() and not needs_per_judge:
            self.est = KNNEstimator.load(model_dir, device=device, index_type=index_type)
            logger.info(f"KNN loaded from {model_dir}, index_type={index_type}")
        elif train_data:
            self.est = KNNEstimator(k=k, device=device, index_type=index_type)
            self.est.build_index(train_data)
            logger.info(f"KNN built from {len(train_data)} examples, k={k}, weighted={weighted}, index_type={index_type}")
        else:
            self.est = KNNEstimator.load(model_dir, device=device, index_type=index_type)
            logger.warning(f"KNN loaded from pickle (no train_data for rebuild), per_judge may be missing")

        # Force-override k post-load (load may have set k=10 from pickle)
        self.est.k = int(k)
        logger.info(f"KNNAdapter ready: target={target}, k={self.est.k}, weighted={weighted}")

    def predict(self, prompt: str, target_models: list) -> Dict[str, float]:
        out = {}
        w = self.weighted
        for m in target_models:
            if m not in self.est.model_names:
                continue
            if self.target == "length":
                pred = self.est.predict_length(prompt, m, weighted=w)
                out[m] = pred["mean"]
            elif self.target == "similarity":
                out[m] = self.est.predict_similarity(prompt, m, weighted=w)
            elif self.target == "judge":
                out[m] = self.est.predict_judge(prompt, m, weighted=w)
            elif self.target == "reference_score":
                out[m] = self.est.predict_reference_similarity(prompt, m, weighted=w)
            elif self.target == "deepeval":
                out[m] = self.est.predict_by_judge_key(prompt, m, "deepeval-llama3.1-8b-it_reference", weighted=w)
            elif self.target == "prometheus":
                out[m] = self.est.predict_by_judge_key(prompt, m, "prometheus-7b-v2_reference", weighted=w)
            elif self.target in ("bucket", "length_bucket"):
                out[m] = self.est.predict_bucket(prompt, m, self.bucket_size, self.num_buckets, weighted=w)
        return out
