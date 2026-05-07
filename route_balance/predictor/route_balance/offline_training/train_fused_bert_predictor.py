#!/usr/bin/env python3
"""
True fused multi-head encoder predictor for ROUTE_BALANCE.

One shared encoder (ModernBERT/RoBERTa/DeBERTa) with multiple output heads,
one per target LLM model size. A single forward pass produces predictions
for ALL model sizes simultaneously.

Architecture:
    Input prompt → Shared Encoder → [CLS] embedding
                                      ├── Head_3B  → prediction
                                      ├── Head_7B  → prediction
                                      ├── Head_14B → prediction
                                      └── Head_72B → prediction

Benefits over per-model training:
    - Shared encoder learns cross-model representations
    - 1 forward pass instead of N (4x faster inference for 4 models)
    - Single checkpoint (~575MB) instead of N×575MB

Usage:
    python -m route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor \
        --input data/route_balance/training_data/train_fixed.jsonl \
        --test-input data/route_balance/training_data/test_fixed.jsonl \
        --encoder-name answerdotai/ModernBERT-base \
        --target length_bucket --epochs 5 --lr 1e-5 --device cuda \
        --output-dir models/route_balance/length_bucket/deploy
"""

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.simplefilter(action="ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiHeadEncoder(nn.Module):
    """Shared encoder with per-model-size output heads.

    One forward pass → predictions for all target LLM model sizes.
    """

    def __init__(
        self,
        encoder_name: str,
        model_names: List[str],
        num_labels: int = 1,
        problem_type: str = "regression",
        attn_impl: str = "eager",
    ):
        """
        attn_impl: "eager" (default, training-compatible) or "sdpa" (inference-fast).
          Switching attention implementation does NOT change model weights — the checkpoint
          state_dict is interchangeable. For inference, prefer "sdpa" + torch.compile
          (see load_fused_model below) which measured 1.17-1.63× speedup on CPU per
          2026-04-14 bench on d7525-10s10317.
        """
        super().__init__()
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(encoder_name, trust_remote_code=True)
        self.hidden_size = config.hidden_size

        try:
            self.encoder = AutoModel.from_pretrained(
                encoder_name, attn_implementation=attn_impl, trust_remote_code=True,
            )
        except (ValueError, TypeError):
            self.encoder = AutoModel.from_pretrained(
                encoder_name, trust_remote_code=True,
            )

        self.model_names = model_names
        self.num_labels = num_labels
        self.problem_type = problem_type

        # One head per target model size
        self.heads = nn.ModuleDict({
            self._key(name): nn.Linear(self.hidden_size, num_labels)
            for name in model_names
        })

        # Dropout before heads (same as HF SequenceClassification)
        drop_rate = getattr(config, "classifier_dropout", None) or getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(drop_rate)

    @staticmethod
    def _key(model_name: str) -> str:
        """Convert model name to valid module key."""
        return model_name.replace("/", "_").replace("-", "_").replace(".", "_")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: (batch, num_models) — float for regression, long for classification.
                    NaN entries are masked (missing labels for some models).

        Returns:
            dict with 'loss' (scalar) and 'logits' (batch, num_models, num_labels)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Pool: [CLS] token (first token)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])

        batch_size = pooled.size(0)
        num_models = len(self.model_names)
        all_logits = torch.zeros(
            batch_size, num_models, self.num_labels, device=pooled.device
        )

        total_loss = torch.tensor(0.0, device=pooled.device)
        n_valid_heads = 0

        for i, name in enumerate(self.model_names):
            key = self._key(name)
            logits = self.heads[key](pooled)  # (batch, num_labels)
            all_logits[:, i, :] = logits

            if labels is not None:
                model_labels = labels[:, i]
                # Mask out NaN entries (missing labels)
                mask = ~torch.isnan(model_labels)
                if mask.any():
                    if self.problem_type == "single_label_classification":
                        loss = F.cross_entropy(logits[mask], model_labels[mask].long())
                    else:
                        loss = F.mse_loss(logits[mask].squeeze(-1), model_labels[mask])
                    total_loss = total_loss + loss
                    n_valid_heads += 1

        # Average loss across heads (so magnitude doesn't scale with num_models)
        if n_valid_heads > 0:
            total_loss = total_loss / n_valid_heads

        return {
            "loss": total_loss if labels is not None else None,
            "logits": all_logits,  # (batch, num_models, num_labels)
        }

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Inference: returns {model_name: logits} for all models.

        Uses self(...) rather than self.forward(...) so the call goes through
        __call__, which propagates to torch.compile-wrapped modules properly
        (a direct .forward() call bypasses the compiled graph).
        """
        with torch.no_grad():
            out = self(input_ids, attention_mask)
        result = {}
        for i, name in enumerate(self.model_names):
            result[name] = out["logits"][:, i, :]  # (batch, num_labels)
        return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _extract_label(m_data: dict, target: str, bucket_size: int = 64,
                   is_harmful: bool = False) -> Optional[float]:
    """Extract a single label for a model from its data dict."""
    if target == "length":
        return float(m_data.get("output_length", 0))
    elif target == "length_bucket":
        length = float(m_data.get("output_length", 0))
        bucket = int(length) // bucket_size
        max_bucket = 1024 // bucket_size - 1
        return float(min(bucket, max_bucket))
    elif target == "similarity":
        sim = m_data.get("similarity_score")
        if sim is not None:
            return float(sim)
        if "quality_score" in m_data:
            return float(m_data["quality_score"])
        return 0.0
    elif target == "judge_class":
        return float(_extract_judge_class(m_data, is_harmful))
    elif target == "reference_score":
        ref = m_data.get("reference_score")
        return float(ref) if ref is not None else None
    elif target == "deepeval":
        scores = m_data.get("llm_judge_scores", {})
        val = scores.get("deepeval-llama3.1-8b-it_reference")
        return float(val) if val is not None else None
    elif target == "quality":
        return _extract_quality(m_data)
    return None


def _extract_quality(m_data: dict) -> float:
    if "quality_score" in m_data:
        return float(m_data["quality_score"])
    sim = m_data.get("similarity_score")
    judge_scores = m_data.get("llm_judge_scores", {})
    valid_judges = [v for v in judge_scores.values() if v is not None]
    judge_mean = sum(valid_judges) / len(valid_judges) if valid_judges else None
    if sim is not None and judge_mean is not None:
        return 0.5 * float(sim) + 0.5 * float(judge_mean)
    elif sim is not None:
        return float(sim)
    elif judge_mean is not None:
        return float(judge_mean)
    return 0.0


def _get_judge_score(m_data: dict, is_harmful: bool = False) -> float:
    judge_scores = m_data.get("llm_judge_scores", {})
    if is_harmful:
        protectai = judge_scores.get("protectai_distilroberta-base-rejection-v1")
        return float(protectai) if protectai is not None else 0.0
    else:
        qwen_scores = [v for k, v in judge_scores.items()
                       if v is not None and "protectai" not in k]
        return sum(qwen_scores) / len(qwen_scores) if qwen_scores else 0.0


def _extract_judge_class(m_data: dict, is_harmful: bool = False) -> int:
    score = _get_judge_score(m_data, is_harmful)
    if score <= 1.0:
        return max(0, min(9, int(score * 10)))
    else:
        return max(0, min(9, int(score) - 1))


class FusedDataset(Dataset):
    """Dataset that returns labels for ALL model sizes per sample.

    labels tensor shape: (num_models,) with NaN for missing entries.
    """

    def __init__(
        self,
        data: list,
        model_names: List[str],
        target: str,
        tokenizer,
        max_length: int = 1024,
        bucket_size: int = 64,
        log_transform: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_names = model_names
        self.target = target
        self.log_transform = log_transform

        self.texts = []
        self.labels = []  # list of (num_models,) arrays
        self.raw_lengths = []  # list of (num_models,) — real output_length for bucket MAE

        is_classification = target in ("length_bucket", "judge_class")

        for req in data:
            is_harmful = req.get("is_harmful", False)
            prompt = req["prompt"]
            row_labels = []
            row_raw_lengths = []
            has_any = False

            for model_name in model_names:
                m_data = req["models"].get(model_name, {})
                if not m_data:
                    row_labels.append(float("nan"))
                    row_raw_lengths.append(float("nan"))
                    continue

                label = _extract_label(m_data, target, bucket_size, is_harmful)
                if label is None:
                    row_labels.append(float("nan"))
                    row_raw_lengths.append(float("nan"))
                    continue

                if log_transform and not is_classification:
                    label = float(np.log1p(label))

                row_labels.append(label)
                row_raw_lengths.append(float(m_data.get("output_length", 0)))
                has_any = True

            if has_any:
                self.texts.append(prompt)
                self.labels.append(row_labels)
                self.raw_lengths.append(row_raw_lengths)

        logger.info(
            f"FusedDataset: {len(self.texts)} samples, "
            f"{len(model_names)} models, target={target}"
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_fused_metrics(model_names: List[str], target: str,
                       bucket_size: int = 64, log_transform: bool = False,
                       raw_lengths: Optional[np.ndarray] = None):
    """Create metrics fn that evaluates each head separately and returns averages.

    Args:
        raw_lengths: (N, num_models) array of real output_length values for
            computing mae_tokens_actual (bucket target only). If None, only
            mae_tokens_bucket (vs bucket midpoint) is reported.
    """

    is_classification = target in ("length_bucket", "judge_class")

    def compute_metrics(eval_pred):
        # logits: (N, num_models, num_labels), labels: (N, num_models)
        logits, labels = eval_pred
        num_models = len(model_names)
        results = {}
        per_model = {}

        for i, name in enumerate(model_names):
            model_labels = labels[:, i]
            mask = ~np.isnan(model_labels)
            if not mask.any():
                continue

            m_logits = logits[:, i, :]  # (N, num_labels)
            m_labels = model_labels[mask]
            m_logits_masked = m_logits[mask]

            short_name = name.split("/")[-1] if "/" in name else name

            if is_classification:
                preds = np.argmax(m_logits_masked, axis=-1)
                acc = float(np.mean(preds == m_labels.astype(int)))
                adj = float(np.mean(np.abs(preds - m_labels.astype(int)) <= 1))
                # Class-level MAE (always valid)
                mae_class = float(np.mean(np.abs(preds - m_labels.astype(int))))

                metrics = {
                    "accuracy": acc, "adjacent_accuracy": adj,
                    "mae_class": mae_class, "n": int(mask.sum()),
                }

                # MAE in token space — only meaningful for length_bucket
                if target == "length_bucket":
                    num_buckets = m_logits_masked.shape[-1]
                    exp_logits = np.exp(m_logits_masked - np.max(m_logits_masked, axis=-1, keepdims=True))
                    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                    midpoints = np.array([(j * bucket_size + bucket_size / 2) for j in range(num_buckets)])
                    expected_tokens = np.sum(probs * midpoints, axis=-1)
                    # MAE vs bucket midpoint (quantized truth)
                    bucket_midpoints = (m_labels * bucket_size) + bucket_size / 2
                    metrics["mae_tokens_bucket"] = float(np.mean(np.abs(bucket_midpoints - expected_tokens)))
                    # MAE vs real output_length (exact truth)
                    if raw_lengths is not None:
                        raw_col = raw_lengths[:, i]
                        raw_masked = raw_col[mask]
                        valid = ~np.isnan(raw_masked)
                        if valid.any():
                            metrics["mae_tokens_actual"] = float(np.mean(np.abs(raw_masked[valid] - expected_tokens[valid])))
                    # Backward compat: mae_tokens = actual if available, else bucket
                    metrics["mae_tokens"] = metrics.get("mae_tokens_actual", metrics["mae_tokens_bucket"])

                per_model[short_name] = metrics
            else:
                preds = m_logits_masked.squeeze(-1)
                if log_transform:
                    preds = np.expm1(preds)
                    m_labels = np.expm1(m_labels)
                errors = np.abs(preds - m_labels)
                non_zero = m_labels > 0
                mae = float(np.mean(errors))
                from scipy.stats import spearmanr
                rho = float(spearmanr(m_labels, preds).correlation) if len(m_labels) > 2 else 0.0

                per_model[short_name] = {
                    "mae": mae,
                    "median_ae": float(np.median(errors)),
                    "acc_50": float(np.mean(errors <= 50)) if target in ("length",) else None,
                    "mape": float(np.mean(errors[non_zero] / m_labels[non_zero]) * 100) if non_zero.any() else 0.0,
                    "spearman_r": rho if not np.isnan(rho) else 0.0,
                    "n": int(mask.sum()),
                }

        # Aggregate across models
        if is_classification:
            results["accuracy"] = np.mean([m["accuracy"] for m in per_model.values()])
            results["adjacent_accuracy"] = np.mean([m["adjacent_accuracy"] for m in per_model.values()])
            results["mae_class"] = np.mean([m["mae_class"] for m in per_model.values()])
            if target == "length_bucket":
                results["mae_tokens"] = np.mean([m["mae_tokens"] for m in per_model.values()])
                if any("mae_tokens_bucket" in m for m in per_model.values()):
                    results["mae_tokens_bucket"] = np.mean([m.get("mae_tokens_bucket", m["mae_tokens"]) for m in per_model.values()])
                if any("mae_tokens_actual" in m for m in per_model.values()):
                    results["mae_tokens_actual"] = np.mean([m["mae_tokens_actual"] for m in per_model.values() if "mae_tokens_actual" in m])
        else:
            results["mae"] = np.mean([m["mae"] for m in per_model.values()])
            maes = [m.get("mape", 0) for m in per_model.values()]
            results["mape"] = np.mean(maes) if maes else 0.0
            rhos = [m.get("spearman_r", 0) for m in per_model.values()]
            results["spearman_r"] = np.mean(rhos) if rhos else 0.0

        return results

    return compute_metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fused_model(
    train_data: list,
    test_data: list,
    model_names: List[str],
    target: str,
    encoder_name: str = "answerdotai/ModernBERT-base",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-5,
    max_length: int = 1024,
    bucket_size: int = 64,
    log_transform: bool = False,
    device: str = "cuda",
    precision: str = "fp16",
    output_dir: str = "models/route_balance/fused",
    seed: int = 42,
    scheduler: str = "polynomial",
    early_stopping_patience: int = 0,
    save_total_limit: int = 2,
    resume_from_checkpoint: str = None,
    weight_decay: float = 0.0,
    warmup_ratio: float = 0.03,
) -> Dict:
    """Train a fused multi-head model.

    Returns {model_name: metrics} dict.
    """
    from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback

    logger.info(f"Training fused {target} predictor")
    logger.info(f"  Encoder: {encoder_name}")
    logger.info(f"  Models: {model_names}")
    logger.info(f"  Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Classification setup
    is_classification = target in ("length_bucket", "judge_class")
    if target == "length_bucket":
        num_labels = 1024 // bucket_size  # 16 buckets
        problem_type = "single_label_classification"
    elif target == "judge_class":
        num_labels = 10
        problem_type = "single_label_classification"
    else:
        num_labels = 1
        problem_type = "regression"

    logger.info(f"  Problem: {problem_type}, num_labels: {num_labels}")

    # Datasets
    train_dataset = FusedDataset(
        train_data, model_names, target, tokenizer, max_length,
        bucket_size, log_transform,
    )
    val_dataset = FusedDataset(
        test_data, model_names, target, tokenizer, max_length,
        bucket_size, log_transform,
    )

    # Model
    model = MultiHeadEncoder(
        encoder_name=encoder_name,
        model_names=model_names,
        num_labels=num_labels,
        problem_type=problem_type,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_head_params = sum(p.numel() for name, p in model.named_parameters() if "heads" in name)
    logger.info(f"  Parameters: {n_params:,} total, {n_head_params:,} in heads")

    # Training args
    metric_name = "accuracy" if is_classification else "mae"
    greater_is_better = is_classification

    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=seed,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=scheduler,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=greater_is_better,
        logging_steps=100,
        report_to="none",
        fp16=(device == "cuda" and precision == "fp16"),
        bf16=(device == "cuda" and precision == "bf16"),
        dataloader_num_workers=0,
    )

    # Custom collator for dynamic padding
    from transformers import DataCollatorWithPadding

    class FusedCollator(DataCollatorWithPadding):
        """Pads input_ids/attention_mask and stacks labels."""

        def __call__(self, features):
            labels = [f.pop("labels") for f in features]
            batch = super().__call__(features)
            batch["labels"] = torch.stack(labels)
            return batch

    collator = FusedCollator(tokenizer=tokenizer)

    # Custom Trainer that handles our model's dict output
    class FusedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(labels=labels, **inputs)
            loss = outputs["loss"]
            # Trainer expects outputs with .logits attribute
            class _Out:
                def __init__(self, logits, loss):
                    self.logits = logits
                    self.loss = loss
            out = _Out(outputs["logits"].detach(), loss)
            return (loss, out) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs = self._prepare_inputs(inputs)
            labels = inputs.pop("labels")
            with torch.no_grad():
                outputs = model(labels=labels, **inputs)
            loss = outputs["loss"]
            logits = outputs["logits"]  # (batch, num_models, num_labels)
            return (loss, logits, labels)

    # Pass raw output_length for bucket MAE vs actual (not just vs bucket midpoint)
    test_raw_lengths = None
    if target == "length_bucket" and hasattr(val_dataset, 'raw_lengths') and val_dataset.raw_lengths:
        test_raw_lengths = np.array(val_dataset.raw_lengths)

    metrics_fn = make_fused_metrics(
        model_names, target, bucket_size, log_transform,
        raw_lengths=test_raw_lengths,
    )

    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = FusedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=metrics_fn,
        callbacks=callbacks,
    )

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    train_time = time.time() - t0
    logger.info(f"  Training time: {train_time:.1f}s")

    # Final evaluation
    eval_results = trainer.evaluate()
    logger.info(f"  Final eval: {eval_results}")

    # Save model
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save the full model (encoder + heads)
    torch.save({
        "encoder_name": encoder_name,
        "model_names": model_names,
        "num_labels": num_labels,
        "problem_type": problem_type,
        "state_dict": model.state_dict(),
    }, out_path / "fused_model.pt")

    # Also save the underlying encoder + per-model HF-compatible checkpoints
    # so the existing ModelEstimator can still load them individually if needed
    tokenizer.save_pretrained(str(out_path))

    # Save config for loading
    config = {
        "encoder_name": encoder_name,
        "model_names": model_names,
        "num_labels": num_labels,
        "problem_type": problem_type,
        "target": target,
        "bucket_size": bucket_size,
        "max_length": max_length,
        "log_transform": log_transform,
        "train_time_s": train_time,
        "train_samples": len(train_dataset),
        "test_samples": len(val_dataset),
    }
    with open(out_path / "fused_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Per-model evaluation
    all_results = {}
    model.eval()
    with torch.no_grad():
        # Run full eval to get per-model metrics
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size * 2, shuffle=False,
            collate_fn=collator,
        )
        all_logits = []
        all_labels = []
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            all_logits.append(outputs["logits"].cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        logits_cat = np.concatenate(all_logits, axis=0)
        labels_cat = np.concatenate(all_labels, axis=0)

        for i, name in enumerate(model_names):
            m_labels = labels_cat[:, i]
            mask = ~np.isnan(m_labels)
            if not mask.any():
                continue

            m_logits = logits_cat[:, i, :][mask]
            m_labels_valid = m_labels[mask]

            if is_classification:
                preds = np.argmax(m_logits, axis=-1)
                acc = float(np.mean(preds == m_labels_valid.astype(int)))
                adj = float(np.mean(np.abs(preds - m_labels_valid.astype(int)) <= 1))
                mae_class = float(np.mean(np.abs(preds - m_labels_valid.astype(int))))

                result = {
                    "accuracy": acc,
                    "adjacent_accuracy": adj,
                    "mae_class": mae_class,
                    "n": int(mask.sum()),
                    "train_time_s": train_time,
                }

                # MAE in token space — only meaningful for length_bucket
                if target == "length_bucket":
                    num_buckets = m_logits.shape[-1]
                    exp_logits = np.exp(m_logits - np.max(m_logits, axis=-1, keepdims=True))
                    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                    midpoints = np.array([(j * bucket_size + bucket_size / 2) for j in range(num_buckets)])
                    expected = np.sum(probs * midpoints, axis=-1)
                    bucket_mid = (m_labels_valid * bucket_size) + bucket_size / 2
                    result["mae_tokens_bucket"] = float(np.mean(np.abs(bucket_mid - expected)))
                    # MAE vs real output_length if available
                    if hasattr(val_dataset, 'raw_lengths') and val_dataset.raw_lengths:
                        raw_arr = np.array(val_dataset.raw_lengths)[:, i]
                        raw_masked = raw_arr[mask]
                        valid = ~np.isnan(raw_masked)
                        if valid.any():
                            result["mae_tokens_actual"] = float(np.mean(np.abs(raw_masked[valid] - expected[valid])))
                    result["mae_tokens"] = result.get("mae_tokens_actual", result["mae_tokens_bucket"])

                all_results[name] = result
            else:
                preds = m_logits.squeeze(-1)
                if log_transform:
                    preds = np.expm1(preds)
                    m_labels_valid = np.expm1(m_labels_valid)
                errors = np.abs(preds - m_labels_valid)
                non_zero = m_labels_valid > 0
                from scipy.stats import spearmanr
                rho = float(spearmanr(m_labels_valid, preds).correlation) if len(m_labels_valid) > 2 else 0.0

                all_results[name] = {
                    "mae": float(np.mean(errors)),
                    "median_ae": float(np.median(errors)),
                    "mape": float(np.mean(errors[non_zero] / m_labels_valid[non_zero]) * 100) if non_zero.any() else 0.0,
                    "spearman_r": rho if not np.isnan(rho) else 0.0,
                    "n": int(mask.sum()),
                    "train_time_s": train_time,
                }

            logger.info(f"  {name}: {all_results[name]}")

    # Save results
    results_out = {
        "encoder": encoder_name,
        "fused": True,
        "target": target,
        "results": all_results,
    }
    with open(out_path / "training_results.json", "w") as f:
        json.dump(results_out, f, indent=2)

    # Save learning curve
    log_history = trainer.state.log_history
    with open(out_path / "log_history.json", "w") as f:
        json.dump(log_history, f, indent=2)

    # Checkpoints managed by HF Trainer (save_total_limit).
    # Best model saved at top level, last checkpoint kept for resume.
    ckpt_dirs = sorted(out_path.glob("checkpoint-*"), key=lambda d: d.name)
    if ckpt_dirs:
        logger.info(f"  Checkpoints kept: {[d.name for d in ckpt_dirs]}")

    logger.info(f"  Saved fused model to {out_path}")
    return all_results


# ---------------------------------------------------------------------------
# Loading (for inference)
# ---------------------------------------------------------------------------

def load_fused_model(
    model_dir: str,
    device: str = "cpu",
    attn_impl: str = "sdpa",
    compile_mode: Optional[str] = "reduce-overhead",
) -> Tuple[MultiHeadEncoder, "AutoTokenizer"]:
    """Load a trained fused model for inference.

    Args:
        model_dir: directory containing fused_model.pt + config.
        device: torch device ("cpu" or "cuda").
        attn_impl: attention implementation. Default "sdpa" (fast inference path).
                   Pass "eager" to disable (matches pre-2026-04-14 behavior).
        compile_mode: torch.compile mode ("reduce-overhead", "default", "max-autotune").
                      Pass None to disable compilation.

    Returns (model, tokenizer).

    The attn_impl + compile defaults give 1.17-1.63× speedup on CPU vs the prior
    "eager + no compile" path (measured 2026-04-14 on d7525, 5 reps × 60 iters,
    paired t-test p≈0 across batch sizes 1-16). State_dict is identical for any
    attn_impl — only the attention kernel changes.
    """
    from transformers import AutoTokenizer

    model_path = Path(model_dir)
    checkpoint = torch.load(model_path / "fused_model.pt", map_location=device, weights_only=False)

    model = MultiHeadEncoder(
        encoder_name=checkpoint["encoder_name"],
        model_names=checkpoint["model_names"],
        num_labels=checkpoint["num_labels"],
        problem_type=checkpoint["problem_type"],
        attn_impl=attn_impl,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model = model.to(device)

    if compile_mode is not None:
        try:
            model = torch.compile(model, mode=compile_mode, fullgraph=False)
            logger.info("Wrapped fused model with torch.compile(mode=%s)", compile_mode)
        except Exception as e:
            logger.warning("torch.compile failed (%s); falling back to eager", e)

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    except (ValueError, OSError) as e:
        # Saved tokenizer_config may reference a class unavailable in this
        # transformers version (e.g. "TokenizersBackend" from 5.x on 4.50).
        # Fall back to loading from the original encoder name.
        logger.warning(
            "Failed to load tokenizer from %s (%s), "
            "falling back to encoder name: %s",
            model_path, e, checkpoint["encoder_name"],
        )
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint["encoder_name"], trust_remote_code=True
        )

    return model, tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train fused multi-head encoder predictor for ROUTE_BALANCE"
    )
    parser.add_argument("--input", required=True, help="Training data JSON/JSONL")
    parser.add_argument("--test-input", default=None, help="Test data JSON/JSONL")
    parser.add_argument("--encoder-name", default="answerdotai/ModernBERT-base",
                        help="Base encoder model")
    parser.add_argument("--target",
                        choices=["length", "length_bucket", "similarity",
                                 "judge_class", "reference_score", "deepeval", "quality"],
                        default="length_bucket",
                        help="Prediction target")
    parser.add_argument("--target-models", nargs="+", default=None,
                        help="Target LLM models (default: all in data)")
    parser.add_argument("--bucket-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--log-transform", action="store_true")
    parser.add_argument("--scheduler", default="polynomial",
                        choices=["polynomial", "cosine", "linear", "constant_with_warmup"])
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--precision", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="models/route_balance/fused")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-total-limit", type=int, default=2,
                        help="Max checkpoints to keep (best + last). HF Trainer manages these.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Resume training from a checkpoint directory")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="AdamW weight decay (HF TrainingArguments default 0.0)")
    parser.add_argument("--warmup-ratio", type=float, default=0.03,
                        help="LR warmup ratio over total training steps")
    args = parser.parse_args()

    # Load data
    def load_data(path):
        with open(path) as f:
            if path.endswith(".jsonl"):
                return [json.loads(line) for line in f]
            else:
                raw = json.load(f)
                return raw["requests"] if "requests" in raw else raw

    train_data = load_data(args.input)
    logger.info(f"Training data: {len(train_data)} requests")

    test_data = None
    if args.test_input:
        test_data = load_data(args.test_input)
        logger.info(f"Test data: {len(test_data)} requests")
    else:
        test_data = train_data[-500:]
        train_data = train_data[:-500]
        logger.info(f"Split: {len(train_data)} train, {len(test_data)} test")

    # Determine target models
    model_names = args.target_models or sorted(train_data[0]["models"].keys())
    logger.info(f"Target models: {model_names}")

    results = train_fused_model(
        train_data=train_data,
        test_data=test_data,
        model_names=model_names,
        target=args.target,
        encoder_name=args.encoder_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        bucket_size=args.bucket_size,
        log_transform=args.log_transform,
        device=args.device,
        precision=args.precision,
        output_dir=args.output_dir,
        seed=args.seed,
        scheduler=args.scheduler,
        early_stopping_patience=args.early_stopping_patience,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
    )

    logger.info("=== Final Results ===")
    for name, metrics in results.items():
        logger.info(f"  {name}: {metrics}")


if __name__ == "__main__":
    main()
