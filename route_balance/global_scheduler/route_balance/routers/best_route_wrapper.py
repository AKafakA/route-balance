"""BEST-Route-wrapper router — Ding 2025 (arXiv 2506.22716).

Stripped-Best-of-N variant adapted for the Qwen pool (design §5.15):
    Per-prompt DeBERTa-v3-small classifier, trained on armoRM pair-wise
    preference labels over {Qwen/Qwen2.5-3B, Qwen/Qwen2.5-7B} responses.
    Predicts strong-win-probability; routes to strong iff prob >= threshold.

This router is structurally similar to RouteLLM-mf on a binary pool, but uses
an in-domain classifier trained directly on Qwen pair-wise preferences rather
than the upstream Chatbot Arena MF embedding.

Training pipeline: route_balance_paper/smoke_test_apr_13/scripts/train_best_route_wrapper.py
(task #62). Produces `models/route_balance/best_route_wrapper_qwen/` with:
    - model.safetensors       DeBERTa-v3-small fine-tuned head
    - tokenizer/              slow tokenizer
    - config.json / training_args.json

If the checkpoint directory does not exist at init, raise a clear error with
the training command — the sweep script's /v1/config POST returns 400 and the
row is skipped with a visible reason.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .base import RouterBase, RouterDecision, RouterRequest


class BestRouteWrapperRouter(RouterBase):
    def __init__(
        self,
        *,
        checkpoint_path: str = "models/route_balance/best_route_wrapper_qwen",
        threshold: float = 0.6,
        strong_model: str = "Qwen/Qwen2.5-7B",
        weak_model: str = "Qwen/Qwen2.5-3B",
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        self._threshold = float(threshold)
        self._strong = strong_model
        self._weak = weak_model
        self._max_length = int(max_length)

        ckpt = Path(checkpoint_path)
        if not ckpt.exists() or not (ckpt / "config.json").exists():
            raise RuntimeError(
                f"BEST-Route-wrapper checkpoint not found at {ckpt}. "
                "Training pipeline lives at "
                "route_balance_paper/smoke_test_apr_13/scripts/train_best_route_wrapper.py "
                "(task #62). Run on a spare A30 with armoRM + Qwen-3B/7B "
                "responses; ~3-4h wall-clock."
            )

        # Lazy import — DeBERTa pulls transformers + torch on the hot path.
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._device = device or "cpu"
        self._tok = AutoTokenizer.from_pretrained(str(ckpt), use_fast=False)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(str(ckpt))
            .to(self._device)
            .eval()
        )
        self._torch = torch

    async def choose_model(
        self,
        req: RouterRequest,
        model_pool: List[str],
    ) -> RouterDecision:
        if not model_pool:
            raise ValueError("model_pool is empty")

        strong = self._strong if self._strong in model_pool else None
        weak = self._weak if self._weak in model_pool else None
        if strong is None or weak is None:
            # Pool doesn't include both labelled models; fall back to smallest.
            return RouterDecision(
                model_name=model_pool[0],
                score=0.0,
                reason=f"best_route_wrapper:pool_mismatch:pool={model_pool}",
            )

        enc = self._tok(
            req.prompt,
            max_length=self._max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        ).to(self._device)
        with self._torch.no_grad():
            logits = self._model(**enc).logits.squeeze(0)
            prob_strong = float(self._torch.softmax(logits, dim=-1)[1])

        chosen = strong if prob_strong >= self._threshold else weak
        return RouterDecision(
            model_name=chosen,
            score=prob_strong,
            reason=f"best_route_wrapper:prob_strong={prob_strong:.3f}:th={self._threshold}",
        )
