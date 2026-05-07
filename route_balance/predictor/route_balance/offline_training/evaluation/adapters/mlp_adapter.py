"""MLP adapter for evaluation."""

from typing import Dict
from .base import BaseAdapter


class MLPAdapter(BaseAdapter):
    def __init__(self, model_dir: str, target: str, device: str):
        from route_balance.predictor.route_balance.estimators.mlp_estimator import MLPEstimator
        self.est = MLPEstimator.load(model_dir)
        self.target = target

    def predict(self, prompt: str, target_models: list) -> Dict[str, float]:
        result = self.est.predict_all_models(prompt)
        out = {}
        for m in target_models:
            if m in result:
                if self.target == "length":
                    out[m] = result[m].get("length_mean", 0)
                elif self.target == "similarity":
                    out[m] = result[m].get("similarity_score", 0)
                elif self.target == "judge":
                    out[m] = result[m].get("judge_score", result[m].get("quality_score", 0))
        return out
