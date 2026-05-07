"""LoRA-encoder (SR-style) adapter for evaluation.

Loads models saved by train_lora_bert_predictor.py:
    lora_encoder_model.pt + tokenizer files

Architecture: frozen encoder + LoRA adapters + per-model linear heads.
One model per target (like SR uses one LoRA per task).
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch

from .base import BaseAdapter

logger = logging.getLogger(__name__)


class LoraEncoderAdapter(BaseAdapter):
    """Adapter for LoRA-encoder models (SR-style per-target).

    Supports both regression and classification (bucket) targets.
    """

    def __init__(self, model_dir: str, target: str, device: str):
        from pathlib import Path

        self.device = device
        self.target = target
        self._is_bucket = target == "length_bucket"

        from route_balance.predictor.route_balance.offline_training.train_lora_bert_predictor import (
            load_lora_encoder,
        )

        self.model, self.tokenizer = load_lora_encoder(model_dir, device=device)

        # Read config for max_length
        config_path = Path(model_dir) / "lora_encoder_model.pt"
        checkpoint = torch.load(config_path, map_location="cpu", weights_only=False)
        self._max_length = checkpoint.get("max_length", 512)
        self._num_labels = checkpoint.get("num_labels", 1)
        self._problem_type = checkpoint.get("problem_type", "regression")
        self._bucket_size = checkpoint.get("bucket_size", 64)
        self._log_transform = checkpoint.get("log_transform", False)

        logger.info(
            f"Loaded LoRA encoder from {model_dir}: "
            f"models={self.model.model_names}, "
            f"num_labels={self._num_labels}, "
            f"problem_type={self._problem_type}, "
            f"max_length={self._max_length}"
        )

    def predict(self, prompt: str, target_models: list) -> Dict[str, float]:
        inputs = self.tokenizer(
            prompt, truncation=True, max_length=self._max_length, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        preds = self.model.predict(**inputs)
        result = {}
        for m, val in preds.items():
            val_sq = val.squeeze().cpu()
            if self._problem_type == "regression":
                v = float(val_sq)
                if self._log_transform:
                    v = float(np.expm1(v))
            elif val_sq.dim() > 0 and val_sq.shape[0] > 1:
                if self._is_bucket:
                    # Expected length from bucket probs
                    probs = torch.softmax(val_sq, dim=-1).numpy()
                    midpoints = np.array([
                        (i * self._bucket_size + self._bucket_size / 2)
                        for i in range(len(probs))
                    ])
                    v = float(np.sum(probs * midpoints))
                else:
                    # Classification: argmax / (num_classes - 1) -> [0, 1]
                    num_classes = val_sq.shape[0]
                    v = float(torch.argmax(val_sq)) / max(num_classes - 1, 1)
            else:
                v = float(val_sq)
            result[m] = v

        return {m: result.get(m, 0.0) for m in target_models}

    def predict_probs(self, prompt: str, target_models: list) -> Optional[Dict[str, np.ndarray]]:
        if not self._is_bucket:
            return None

        inputs = self.tokenizer(
            prompt, truncation=True, max_length=self._max_length, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        preds = self.model.predict(**inputs)
        result = {}
        for m, val in preds.items():
            logits = val.squeeze().cpu().numpy()
            probs = np.exp(logits - np.max(logits))
            probs = probs / probs.sum()
            result[m] = probs

        return {m: result.get(m, np.zeros(self._num_labels)) for m in target_models}

    @property
    def supports_probs(self) -> bool:
        return self._is_bucket
