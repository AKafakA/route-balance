"""Encoder (ModernBERT/RoBERTa/DeBERTaV3) adapter for evaluation.

Handles both regression models (length, similarity, judge) and
bucket classification models (length_bucket).
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch

from .base import BaseAdapter

logger = logging.getLogger(__name__)


class EncoderAdapter(BaseAdapter):
    """Adapter for regression encoder models.

    Supports both:
    - Standard HF checkpoints (config.json + model.safetensors)
    - Fused multi-head models (fused_model.pt + fused_config.json)
    """

    def __init__(self, model_dir: str, target: str, device: str):
        import json
        from pathlib import Path

        self.device = device
        self.target = target
        self._fused = False
        self._fused_log_transform = False
        self._max_length = 1024  # default, overridden per model

        fused_path = Path(model_dir) / "fused_model.pt"
        if fused_path.exists():
            # Fused multi-head model
            try:
                from route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor import load_fused_model
            except ImportError:
                from offline_training.train_fused_bert_predictor import load_fused_model
            self.model, self.tokenizer = load_fused_model(model_dir, device=device)
            self._fused = True
            # Respect model's max position embeddings (RoBERTa=514, ModernBERT=8192)
            config_path = Path(model_dir) / "fused_config.json"
            if config_path.exists():
                cfg = json.loads(config_path.read_text())
                max_len = cfg.get("max_length", 1024)
                self._max_length = max_len
            # Check log-transform
            config_path = Path(model_dir) / "fused_config.json"
            if config_path.exists():
                cfg = json.loads(config_path.read_text())
                self._fused_log_transform = cfg.get("log_transform", False)
            logger.info(f"Loaded FUSED encoder from {model_dir}, models={self.model.model_names}, log_transform={self._fused_log_transform}")
        else:
            # Standard HF checkpoint
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir, trust_remote_code=True
            )
            self.model.eval()
            self.model.to(device)
            logger.info(f"Loaded HF encoder from {model_dir}, num_labels={self.model.config.num_labels}")

    def predict(self, prompt: str, target_models: list) -> Dict[str, float]:
        if self._fused:
            inputs = self.tokenizer(prompt, truncation=True, max_length=self._max_length, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            preds = self.model.predict(**inputs)  # {model_name: tensor}
            result = {}
            for m, val in preds.items():
                val_sq = val.squeeze().cpu()
                if val_sq.dim() == 0:
                    # Single regression output
                    v = float(val_sq)
                    if self._fused_log_transform:
                        v = float(np.expm1(v))
                elif val_sq.shape[0] > 1:
                    # Multi-class classifier (e.g., judge 10-class, bucket 16-class)
                    # Convert argmax class → score: class_idx / (num_classes - 1)
                    num_classes = val_sq.shape[0]
                    v = float(torch.argmax(val_sq)) / max(num_classes - 1, 1)
                else:
                    v = float(val_sq)
                result[m] = v
            return {m: result.get(m, 0.0) for m in target_models}
        else:
            inputs = self.tokenizer(prompt, truncation=True, max_length=self._max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits.squeeze().cpu()
            if logits.dim() == 0:
                return {m: float(logits) for m in target_models}
            if logits.shape[0] > 1:
                num_classes = logits.shape[0]
                v = float(torch.argmax(logits)) / max(num_classes - 1, 1)
                return {m: v for m in target_models}
            return {m: float(logits) for m in target_models}


class BucketEncoderAdapter(BaseAdapter):
    """Adapter for bucket classification encoder models.

    Supports both standard HF and fused multi-head bucket classifiers.
    Returns both point predictions (E[length]) and probability distributions.
    """

    def __init__(self, model_dir: str, target: str, device: str, bucket_size: int = 64):
        import json
        from pathlib import Path

        self.device = device
        self.bucket_size = bucket_size
        self._fused = False
        self._max_length = 1024

        fused_path = Path(model_dir) / "fused_model.pt"
        if fused_path.exists():
            try:
                from route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor import load_fused_model
            except ImportError:
                from offline_training.train_fused_bert_predictor import load_fused_model
            self.model, self.tokenizer = load_fused_model(model_dir, device=device)
            self._fused = True
            config_path = Path(model_dir) / "fused_config.json"
            cfg = json.loads(config_path.read_text()) if config_path.exists() else {}
            self.num_buckets = cfg.get("num_labels", 16)
            self._max_length = cfg.get("max_length", 1024)
            logger.info(f"Loaded FUSED bucket model: {self.num_buckets} buckets, max_length={self._max_length}, models={self.model.model_names}")
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir, trust_remote_code=True
            )
            self.model.eval()
            self.model.to(device)
            self.num_buckets = self.model.config.num_labels
            logger.info(f"Loaded HF bucket model: {self.num_buckets} buckets")

        self.midpoints = np.array([(i * bucket_size + bucket_size / 2) for i in range(self.num_buckets)])

    def predict(self, prompt: str, target_models: list) -> Dict[str, float]:
        """Point prediction: E[length] from probability distribution."""
        probs_dict = self.predict_probs(prompt, target_models)
        return {m: float(np.sum(probs * self.midpoints)) for m, probs in probs_dict.items()}

    def predict_probs(self, prompt: str, target_models: list) -> Optional[Dict[str, np.ndarray]]:
        """Probability distribution over buckets."""
        if self._fused:
            inputs = self.tokenizer(prompt, truncation=True, max_length=self._max_length, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            preds = self.model.predict(**inputs)  # {model_name: logits_tensor}
            result = {}
            for m, logits in preds.items():
                logits_np = logits.squeeze().cpu().numpy()
                probs = np.exp(logits_np - np.max(logits_np))
                probs = probs / probs.sum()
                result[m] = probs
            return {m: result.get(m, np.zeros(self.num_buckets)) for m in target_models}
        else:
            inputs = self.tokenizer(prompt, truncation=True, max_length=self._max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits.squeeze().cpu().numpy()
            probs = np.exp(logits - np.max(logits))
            probs = probs / probs.sum()
            return {m: probs for m in target_models}

    @property
    def supports_probs(self) -> bool:
        return True
