"""BEST-Route 4-way wrapper — Ding 2025 (arXiv 2506.22716) extended to N>2.

DeBERTa-v3-small classifier with num_labels=4, fine-tuned to predict the
best Qwen size per prompt across {3B, 7B, 14B, 72B}. Routes by argmax.

Checkpoint: models/route_balance/best_route_4way_qwen/ — May 2 build, val_acc=0.392
on the held-out test set, label_to_model in training_results.json.

Optional `confidence_threshold`: if max-prob < threshold, fall back to
`fallback_model` (default: smallest in pool). Setting threshold=0 disables
fallback (pure argmax).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .base import RouterBase, RouterDecision, RouterRequest


class BestRoute4WayRouter(RouterBase):
    def __init__(
        self,
        *,
        checkpoint_path: str = "models/route_balance/best_route_4way_qwen",
        confidence_threshold: float = 0.0,
        fallback_model: Optional[str] = None,
        label_to_model: Optional[Dict[str, str]] = None,
        max_length: int = 512,
        device: Optional[str] = None,
        batch_scoring: bool = False,
        batch_window_ms: float = 10.0,
        batch_max: int = 64,
    ):
        self._threshold = float(confidence_threshold)
        self._fallback = fallback_model
        self._max_length = int(max_length)
        # Micro-batched scoring (reviewer experiment): concurrent choose_model
        # calls are collected for batch_window_ms (or batch_max) and scored in
        # ONE padded DeBERTa forward inside a thread executor, so the CPU
        # forward neither serializes per request nor blocks the event loop.
        self._batch_scoring = bool(batch_scoring)
        self._batch_window_s = float(batch_window_ms) / 1000.0
        self._batch_max = int(batch_max)
        self._pending: list = []  # list[(prompt, asyncio.Future)]
        self._drain_task = None

        ckpt = Path(checkpoint_path)
        if not ckpt.exists() or not (ckpt / "config.json").exists():
            raise RuntimeError(
                f"BEST-Route-4way checkpoint not found at {ckpt}. "
                "Training pipeline lives at "
                "route_balance_paper/smoke_test_apr_13/scripts/train_best_route_wrapper.py "
                "(task #62) with --num-labels 4."
            )

        # Resolve label_to_model: explicit kwarg > training_results.json
        if label_to_model is None:
            tr_path = ckpt / "training_results.json"
            if tr_path.exists():
                tr = json.loads(tr_path.read_text())
                label_to_model = tr.get("label_to_model")
        if not label_to_model:
            raise RuntimeError(
                f"BEST-Route-4way: no label_to_model mapping found at "
                f"{ckpt}/training_results.json and none provided in kwargs."
            )
        # Normalize keys to int → model_name
        self._label_to_model: Dict[int, str] = {
            int(k): v for k, v in label_to_model.items()
        }

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._device = device or "cpu"
        self._tok = AutoTokenizer.from_pretrained(str(ckpt), use_fast=False)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(str(ckpt))
            .to(self._device)
            .eval()
        )
        self._torch = torch

    def _score_batch_sync(self, prompts):
        enc = self._tok(
            prompts,
            max_length=self._max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self._device)
        with self._torch.no_grad():
            logits = self._model(**enc).logits
            probs = self._torch.softmax(logits, dim=-1)
            top = self._torch.argmax(probs, dim=-1)
        return [(int(top[i].item()), float(probs[i, int(top[i].item())].item()))
                for i in range(len(prompts))]

    async def _drain_loop(self):
        import asyncio
        while True:
            await asyncio.sleep(self._batch_window_s)
            if not self._pending:
                continue
            batch, self._pending = self._pending[: self._batch_max], self._pending[self._batch_max:]
            prompts = [p for p, _ in batch]
            loop = asyncio.get_event_loop()
            try:
                results = await loop.run_in_executor(None, self._score_batch_sync, prompts)
                for (_, fut), r in zip(batch, results):
                    if not fut.done():
                        fut.set_result(r)
            except Exception as e:
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)

    async def _score_one(self, prompt):
        import asyncio
        if not self._batch_scoring:
            loop = asyncio.get_event_loop()
            return (await loop.run_in_executor(None, self._score_batch_sync, [prompt]))[0]
        if self._drain_task is None:
            self._drain_task = asyncio.get_event_loop().create_task(self._drain_loop())
        fut = asyncio.get_event_loop().create_future()
        self._pending.append((prompt, fut))
        return await fut

    async def choose_model(
        self,
        req: RouterRequest,
        model_pool: List[str],
    ) -> RouterDecision:
        if not model_pool:
            raise ValueError("model_pool is empty")

        top_idx, top_prob = await self._score_one(req.prompt)
        chosen = self._label_to_model.get(top_idx)
        if chosen is None or chosen not in model_pool:
            chosen = self._fallback or model_pool[0]
            return RouterDecision(
                model_name=chosen,
                score=top_prob,
                reason=f"best_route_4way:pool_miss:label={top_idx}:fallback",
            )
        if self._threshold > 0 and top_prob < self._threshold:
            chosen = self._fallback or model_pool[0]
            return RouterDecision(
                model_name=chosen,
                score=top_prob,
                reason=f"best_route_4way:low_conf:p={top_prob:.3f}<{self._threshold}",
            )
        return RouterDecision(
            model_name=chosen,
            score=top_prob,
            reason=f"best_route_4way:argmax_label={top_idx}:p={top_prob:.3f}",
        )

    async def _legacy_choose_model(
        self,
        req: RouterRequest,
        model_pool: List[str],
    ) -> RouterDecision:
        if not model_pool:
            raise ValueError("model_pool is empty")

        enc = self._tok(
            req.prompt,
            max_length=self._max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        ).to(self._device)
        with self._torch.no_grad():
            logits = self._model(**enc).logits.squeeze(0)
            probs = self._torch.softmax(logits, dim=-1)
            top_idx = int(self._torch.argmax(probs).item())
            top_prob = float(probs[top_idx].item())

        chosen = self._label_to_model.get(top_idx)
        if chosen is None or chosen not in model_pool:
            # Label predicted a model not in pool — fall back to smallest in pool
            chosen = self._fallback or model_pool[0]
            return RouterDecision(
                model_name=chosen,
                score=top_prob,
                reason=f"best_route_4way:pool_miss:label={top_idx}:fallback",
            )

        # Confidence floor
        if self._threshold > 0 and top_prob < self._threshold:
            chosen = self._fallback or model_pool[0]
            return RouterDecision(
                model_name=chosen,
                score=top_prob,
                reason=f"best_route_4way:low_conf:p={top_prob:.3f}<{self._threshold}",
            )

        return RouterDecision(
            model_name=chosen,
            score=top_prob,
            reason=f"best_route_4way:argmax_label={top_idx}:p={top_prob:.3f}",
        )
