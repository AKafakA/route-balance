"""Qwen-0.5B LoRA adapter for evaluation.

Auto-detects fused vs per-model mode from training_results.json.
Fused: one prompt → all models in one generation (fast).
Per-model: one prompt per model (slower, 4× more generations).
"""

import logging
import re
from typing import Dict

import torch

from .base import BaseAdapter

logger = logging.getLogger(__name__)


def _model_short_name(model_name: str) -> str:
    """Qwen/Qwen2.5-7B → 7B"""
    name = model_name.split("/")[-1]
    parts = name.split("-")
    for p in parts:
        if p.endswith("B") and p[:-1].replace(".", "").isdigit():
            return p
    return name


class LLMAdapter(BaseAdapter):
    """Adapter for LoRA fine-tuned LLM predictors (Qwen-0.5B)."""

    def __init__(self, model_dir: str, target: str, device: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import json
        from pathlib import Path

        adapter_config = json.loads((Path(model_dir) / "adapter_config.json").read_text())
        base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B")

        # Detect fused vs per-model
        self._fused = False
        results_path = Path(model_dir) / "training_results.json"
        if results_path.exists():
            tr = json.loads(results_path.read_text())
            self._fused = tr.get("fused", tr.get("mode", "") == "fused")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, model_dir)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.target = target
        logger.info(f"Loaded LLM adapter from {model_dir}, base={base_model_name}, fused={self._fused}")

    def predict(self, prompt: str, target_models: list) -> Dict[str, float]:
        if self._fused:
            return self._predict_fused(prompt, target_models)
        else:
            return self._predict_per_model(prompt, target_models)

    def _predict_fused(self, prompt: str, target_models: list) -> Dict[str, float]:
        """Fused: one generation → parse all models from structured output.

        Input:  "Predict length: {prompt}\nAnswer: "
        Output: "14B:256 3B:312 72B:189 7B:445"
        """
        input_text = f"Predict {self.target}: {prompt}\nAnswer: "
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True,
                                max_length=974).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=50, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Parse "KEY:VALUE" pairs
        short_to_full = {_model_short_name(m): m for m in target_models}
        results = {}
        for match in re.finditer(r'(\w+):([\d.]+)', generated):
            key, val = match.group(1), match.group(2)
            full_name = short_to_full.get(key)
            if full_name:
                try:
                    results[full_name] = float(val)
                except ValueError:
                    pass

        for m in target_models:
            if m not in results:
                results[m] = 0.0
        return results

    def _predict_per_model(self, prompt: str, target_models: list) -> Dict[str, float]:
        """Per-model: one generation per model."""
        results = {}
        for model_name in target_models:
            model_short = _model_short_name(model_name)
            input_text = f"Predict {self.target} for model {model_short}: {prompt}\nAnswer: "
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True,
                                    max_length=974).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=10, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            try:
                results[model_name] = float(generated.split()[0])
            except (ValueError, IndexError):
                results[model_name] = 0.0
        return results
