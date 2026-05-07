"""Multi-target model adapter for evaluation.

Loads models saved by train_fused_multitarget.py or train_lora_multitarget.py:
    - fused_multitarget_model.pt (full fine-tuned)
    - lora_multitarget_model.pt (LoRA)

Architecture: single encoder + per-(model, target) heads.
One forward pass → predictions for all models × all targets.

The adapter evaluates one target at a time (matching run_evaluation.py's
per-target evaluation loop), but uses the multi-target model's shared encoder.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .base import BaseAdapter

logger = logging.getLogger(__name__)


class MultiTargetAdapter(BaseAdapter):
    """Adapter for multi-target models (fused or LoRA).

    Wraps multi-target models to evaluate one target at a time,
    compatible with run_evaluation.py's per-target loop.
    """

    def __init__(self, model_dir: str, target: str, device: str):
        self.device = device
        self.target = target
        self._is_bucket = target == "length_bucket"

        model_path = Path(model_dir)
        self._max_length = 512

        # Detect model type: fused or LoRA
        if (model_path / "fused_multitarget_model.pt").exists():
            self._load_fused(model_dir, device)
        elif (model_path / "lora_multitarget_model.pt").exists():
            self._load_lora(model_dir, device)
        else:
            raise FileNotFoundError(
                f"No multi-target model found in {model_dir}. "
                f"Expected fused_multitarget_model.pt or lora_multitarget_model.pt"
            )

        # Find target spec for the requested target
        self._target_spec = None
        for spec in self._target_specs:
            if spec.name == target:
                self._target_spec = spec
                break

        if self._target_spec is None:
            available = [s.name for s in self._target_specs]
            raise ValueError(
                f"Target '{target}' not found in model. Available: {available}"
            )

        self._bucket_size = self._target_spec.bucket_size
        self._is_bucket = "classification" in self._target_spec.problem_type

        logger.info(
            f"Loaded multi-target model from {model_dir}: "
            f"evaluating target={target}, "
            f"all_targets={[s.name for s in self._target_specs]}, "
            f"models={self._model_names}"
        )

    def _load_fused(self, model_dir: str, device: str):
        from route_balance.predictor.route_balance.offline_training.train_fused_multitarget import (
            load_multitarget_model,
        )
        self.model, self.tokenizer, self._target_specs = load_multitarget_model(
            model_dir, device=device
        )
        self._model_names = self.model.model_names

        # Load config for max_length
        config_path = Path(model_dir) / "fused_multitarget_config.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            self._max_length = cfg.get("max_length", 512)

    def _load_lora(self, model_dir: str, device: str):
        from route_balance.predictor.route_balance.offline_training.train_fused_multitarget import TargetSpec
        from route_balance.predictor.route_balance.offline_training.train_lora_multitarget import (
            LoRAMultiTargetEncoder,
        )
        from transformers import AutoTokenizer

        model_path = Path(model_dir)
        checkpoint = torch.load(
            model_path / "lora_multitarget_model.pt",
            map_location=device, weights_only=False,
        )

        self._target_specs = [
            TargetSpec(name=t[0], problem_type=t[1], num_labels=t[2],
                       bucket_size=t[3] if len(t) > 3 else 64)
            for t in checkpoint["target_specs"]
        ]
        self._model_names = checkpoint["model_names"]

        self.model = LoRAMultiTargetEncoder(
            encoder_name=checkpoint["encoder_name"],
            model_names=self._model_names,
            target_specs=self._target_specs,
            lora_r=checkpoint.get("lora_r", 16),
            lora_alpha=checkpoint.get("lora_alpha", 32),
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.model = self.model.to(device)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
        except (ValueError, OSError):
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint["encoder_name"], trust_remote_code=True
            )

    def _get_target_logits(self, inputs: dict) -> torch.Tensor:
        """Run model and extract logits for self.target.

        Both fused predict() and LoRA forward() return per-target tensors
        with shape (batch, num_models, num_labels). This method handles
        both return formats and returns the tensor for the requested target.
        """
        with torch.no_grad():
            if hasattr(self.model, 'predict'):
                all_preds = self.model.predict(**inputs)
            else:
                result = self.model(**inputs)
                all_preds = result["logits"] if isinstance(result, dict) and "logits" in result else result
        # all_preds = {target_name: tensor(batch, num_models, num_labels)}
        return all_preds.get(self.target, torch.zeros(1))

    def predict(self, prompt: str, target_models: list) -> Dict[str, float]:
        inputs = self.tokenizer(
            prompt, truncation=True, max_length=self._max_length, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # tensor shape: (1, num_models, num_labels)
        logits = self._get_target_logits(inputs).squeeze(0).cpu()  # (num_models, num_labels)

        result = {}
        for i, m in enumerate(self._model_names):
            if i >= logits.shape[0]:
                break
            val = logits[i]  # (num_labels,)
            if self._target_spec.problem_type == "regression":
                v = float(val.squeeze())
            elif val.dim() > 0 and val.shape[0] > 1:
                if self._is_bucket:
                    probs = torch.softmax(val, dim=-1).numpy()
                    midpoints = np.array([
                        (j * self._bucket_size + self._bucket_size / 2)
                        for j in range(len(probs))
                    ])
                    v = float(np.sum(probs * midpoints))
                else:
                    num_classes = val.shape[0]
                    v = float(torch.argmax(val)) / max(num_classes - 1, 1)
            else:
                v = float(val.squeeze())
            result[m] = v

        return {m: result.get(m, 0.0) for m in target_models}

    def predict_probs(self, prompt: str, target_models: list) -> Optional[Dict[str, np.ndarray]]:
        if not self._is_bucket:
            return None

        inputs = self.tokenizer(
            prompt, truncation=True, max_length=self._max_length, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # tensor shape: (1, num_models, num_labels)
        logits_all = self._get_target_logits(inputs).squeeze(0).cpu()  # (num_models, num_labels)

        result = {}
        for i, m in enumerate(self._model_names):
            if i >= logits_all.shape[0]:
                break
            logits = logits_all[i].numpy()
            probs = np.exp(logits - np.max(logits))
            probs = probs / probs.sum()
            result[m] = probs

        num_labels = self._target_spec.num_labels
        return {m: result.get(m, np.zeros(num_labels)) for m in target_models}

    @property
    def supports_probs(self) -> bool:
        return self._is_bucket
