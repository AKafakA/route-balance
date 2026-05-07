#!/usr/bin/env python3
"""
LLM-based fused predictor for ROUTE_BALANCE (ablation baseline).

Finetunes Qwen2.5-0.5B with LoRA to predict values for ALL model sizes
in a single forward pass using structured output.

Input:  "Predict {target}: {prompt}"
Output: "3B:256 7B:312 14B:189 72B:445"

One LoRA adapter handles all model sizes — truly fused like the multi-head
ModernBERT approach but using a generative LLM.

Usage:
    python -m route_balance.predictor.route_balance.offline_training.train_llm_predictor \
        --input data/route_balance/training_data/train_fixed.jsonl \
        --test-input data/route_balance/training_data/test_fixed.jsonl \
        --output-dir models/route_balance/baselines/qwen05b_lora/length \
        --target length --device cuda
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label extraction helpers
# ---------------------------------------------------------------------------

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


def _extract_value(m_data: dict, target: str, is_harmful: bool = False) -> Optional[float]:
    """Extract prediction target value from model data."""
    if target == "length" or target == "length_bucket":
        return float(m_data.get("output_length", 0))
    elif target == "similarity":
        sim = m_data.get("similarity_score")
        return float(sim) if sim is not None else float(m_data.get("quality_score", 0.0))
    elif target == "judge":
        return _get_judge_score(m_data, is_harmful)
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


BUCKET_SIZE = 64
NUM_BUCKETS = 16
MAX_LENGTH_BUCKET = BUCKET_SIZE * NUM_BUCKETS  # 1024


def _length_to_bucket_probs_str(length: float) -> str:
    """Convert output length to one-hot bucket probability string.

    E.g., length=287 → "0-64:0.00 64-128:0.00 ... 256-320:1.00 320-384:0.00 ..."
    """
    bucket_idx = min(int(length) // BUCKET_SIZE, NUM_BUCKETS - 1)
    parts = []
    for i in range(NUM_BUCKETS):
        lo = i * BUCKET_SIZE
        hi = (i + 1) * BUCKET_SIZE
        prob = "1.00" if i == bucket_idx else "0.00"
        parts.append(f"{lo}-{hi}:{prob}")
    return " ".join(parts)


def _parse_bucket_probs(text: str) -> Optional[List[float]]:
    """Parse bucket probability string back to list of floats.

    Input: "0-64:0.05 64-128:0.15 128-192:0.40 ..."
    Returns: [0.05, 0.15, 0.40, ...] (length NUM_BUCKETS)
    """
    probs = []
    for match in re.finditer(r'\d+-\d+:([\d.]+)', text):
        try:
            probs.append(float(match.group(1)))
        except ValueError:
            probs.append(0.0)
    return probs if len(probs) == NUM_BUCKETS else None


def _format_value(value: float, target: str) -> str:
    """Format a value for text output."""
    if target == "length":
        return str(int(value))
    else:
        return f"{value:.3f}"


def _model_short_name(model_name: str) -> str:
    """Qwen/Qwen2.5-7B → 7B"""
    name = model_name.split("/")[-1]
    # Extract size part: Qwen2.5-7B → 7B
    parts = name.split("-")
    for p in parts:
        if p.endswith("B") and p[:-1].replace(".", "").isdigit():
            return p
    return name


# ---------------------------------------------------------------------------
# Datasets — fused (1 example → all models) and per-model (N examples)
# ---------------------------------------------------------------------------

class PerModelPredictionDataset(Dataset):
    """Per-model dataset: N examples per prompt, one per model size.

    Input:  "Predict {target} for model {model_short}: {prompt}"
    Output: "{value}"
    """

    def __init__(self, data: list, tokenizer, model_names: List[str],
                 target: str = "length", max_length: int = 512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for req in data:
            prompt = req["prompt"]
            is_harmful = req.get("is_harmful", False)
            for model_name in model_names:
                m_data = req["models"].get(model_name, {})
                if not m_data:
                    continue
                val = _extract_value(m_data, target, is_harmful)
                if val is None:
                    continue

                short = _model_short_name(model_name)
                if target == "length_bucket":
                    input_text = f"Predict output length bucket probabilities for model {short}: {prompt}"
                    target_text = _length_to_bucket_probs_str(val)
                else:
                    input_text = f"Predict {target} for model {short}: {prompt}"
                    target_text = _format_value(val, target)
                self.examples.append((input_text, target_text, val, model_name))

        logger.info(
            f"PerModelPredictionDataset: {len(self.examples)} examples, "
            f"{len(model_names)} models, target={target}"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_text, target_text, _, _ = self.examples[idx]
        full_text = f"{input_text}\nAnswer: {target_text}"

        encoding = self.tokenizer(
            full_text, max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        answer_tokens = self.tokenizer.encode("\nAnswer: ", add_special_tokens=False)
        for i in range(len(input_ids) - len(answer_tokens)):
            if input_ids[i: i + len(answer_tokens)].tolist() == answer_tokens:
                labels[:i + len(answer_tokens)] = -100
                break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class FusedPredictionDataset(Dataset):
    """Fused dataset: 1 example per prompt → all model predictions.

    Input:  "Predict {target}: {prompt}"
    Output: "3B:256 7B:312 14B:189 72B:445"
    """

    def __init__(self, data: list, tokenizer, model_names: List[str],
                 target: str = "length", max_length: int = 512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_names = model_names
        self.model_short = [_model_short_name(m) for m in model_names]

        skipped = 0
        for req in data:
            prompt = req["prompt"]
            is_harmful = req.get("is_harmful", False)

            values = {}
            for model_name in model_names:
                m_data = req["models"].get(model_name, {})
                if not m_data:
                    continue
                val = _extract_value(m_data, target, is_harmful)
                if val is not None:
                    values[model_name] = val

            if not values:
                skipped += 1
                continue

            if target == "length_bucket":
                # Bucket probability format:
                # "3B: 0-64:0.00 64-128:0.00 ... 256-320:1.00 ...
                #  7B: 0-64:0.00 ..."
                parts = []
                for mn, short in zip(model_names, self.model_short):
                    if mn in values:
                        bucket_str = _length_to_bucket_probs_str(values[mn])
                        parts.append(f"{short}: {bucket_str}")
                input_text = f"Predict output length bucket probabilities: {prompt}"
                target_text = "\n".join(parts)
            else:
                # Structured output: "3B:256 7B:312 14B:189 72B:445"
                parts = []
                for mn, short in zip(model_names, self.model_short):
                    if mn in values:
                        parts.append(f"{short}:{_format_value(values[mn], target)}")
                input_text = f"Predict {target}: {prompt}"
                target_text = " ".join(parts)

            self.examples.append((input_text, target_text, values))

        logger.info(
            f"FusedPredictionDataset: {len(self.examples)} examples, "
            f"{len(model_names)} models, target={target}, skipped={skipped}"
        )
        if self.examples:
            vals_flat = [v for ex in self.examples for v in ex[2].values()]
            logger.info(
                f"  Label stats: mean={np.mean(vals_flat):.3f}, "
                f"min={np.min(vals_flat):.3f}, max={np.max(vals_flat):.3f}"
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_text, target_text, _ = self.examples[idx]
        full_text = f"{input_text}\nAnswer: {target_text}"

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Labels: mask input portion, only predict the answer
        labels = input_ids.clone()
        answer_tokens = self.tokenizer.encode("\nAnswer: ", add_special_tokens=False)
        for i in range(len(input_ids) - len(answer_tokens)):
            if input_ids[i: i + len(answer_tokens)].tolist() == answer_tokens:
                labels[:i + len(answer_tokens)] = -100
                break

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def parse_fused_output(text: str, model_names: List[str], target: str = "length") -> Dict:
    """Parse structured output into dict.

    For regression: "3B:256 7B:312" → {model_name: float}
    For length_bucket: "3B: 0-64:0.05 ... \n7B: ..." → {model_name: [probs]}
    """
    result = {}
    short_to_full = {_model_short_name(m): m for m in model_names}

    if target == "length_bucket":
        # Parse per-model bucket probabilities
        # Format: "3B: 0-64:0.05 64-128:0.15 ...\n7B: 0-64:0.03 ..."
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Find model prefix (e.g., "3B:")
            for short, full in short_to_full.items():
                if line.startswith(f"{short}:"):
                    bucket_text = line[len(short) + 1:].strip()
                    probs = _parse_bucket_probs(bucket_text)
                    if probs:
                        result[full] = probs
                    break
    else:
        # Parse "KEY:VALUE" pairs
        for match in re.finditer(r'(\w+):([\d.]+)', text):
            key, val = match.group(1), match.group(2)
            full_name = short_to_full.get(key)
            if full_name:
                try:
                    result[full_name] = float(val)
                except ValueError:
                    pass

    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_fused(model, tokenizer, test_data: list, model_names: List[str],
                   target: str, device: str = "cuda", max_length: int = 1024) -> Dict:
    """Evaluate fused LoRA model on test data."""
    model.eval()
    all_preds = {m: [] for m in model_names}
    all_labels = {m: [] for m in model_names}

    for req in test_data:
        prompt = req["prompt"]
        is_harmful = req.get("is_harmful", False)

        # Ground truth
        gt = {}
        for mn in model_names:
            m_data = req["models"].get(mn, {})
            if m_data:
                val = _extract_value(m_data, target, is_harmful)
                if val is not None:
                    gt[mn] = val

        if not gt:
            continue

        # Generate prediction
        if target == "length_bucket":
            input_text = f"Predict output length bucket probabilities: {prompt}\nAnswer: "
            # 16 buckets × 4 models × ~15 chars each ≈ 250 tokens
            gen_max_tokens = 300
        else:
            input_text = f"Predict {target}: {prompt}\nAnswer: "
            gen_max_tokens = 50

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True,
                           max_length=max_length - gen_max_tokens).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=gen_max_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
        preds = parse_fused_output(generated, model_names, target=target)

        for mn in model_names:
            if mn in gt and mn in preds:
                all_labels[mn].append(gt[mn])
                all_preds[mn].append(preds[mn])

    # Compute per-model metrics
    results = {}
    for mn in model_names:
        if not all_labels[mn]:
            continue
        short = _model_short_name(mn)

        if target == "length_bucket":
            # Bucket classification metrics
            n_correct = 0
            n_adjacent = 0
            n_total = 0
            n_parsed = 0
            for label_len, pred_probs in zip(all_labels[mn], all_preds[mn]):
                if not isinstance(pred_probs, list):
                    continue
                n_parsed += 1
                actual_bucket = min(int(label_len) // BUCKET_SIZE, NUM_BUCKETS - 1)
                pred_bucket = int(np.argmax(pred_probs))
                n_total += 1
                if pred_bucket == actual_bucket:
                    n_correct += 1
                if abs(pred_bucket - actual_bucket) <= 1:
                    n_adjacent += 1

            results[mn] = {
                "accuracy": n_correct / n_total if n_total > 0 else 0.0,
                "adjacent_accuracy": n_adjacent / n_total if n_total > 0 else 0.0,
                "n": len(all_labels[mn]),
                "n_parsed": n_parsed,
            }
        elif target == "length":
            labels = np.array(all_labels[mn])
            preds = np.array(all_preds[mn])
            errors = np.abs(preds - labels)
            non_zero = labels > 0
            results[mn] = {
                "mae": float(np.mean(errors)),
                "median_ae": float(np.median(errors)),
                "mape": float(np.mean(errors[non_zero] / labels[non_zero]) * 100) if non_zero.any() else 0.0,
                "n": len(labels),
                "n_parsed": len(preds),
            }
        else:
            labels = np.array(all_labels[mn])
            preds = np.array(all_preds[mn])
            errors = np.abs(preds - labels)
            from scipy.stats import spearmanr
            rho = float(spearmanr(labels, preds).correlation) if len(labels) > 2 else 0.0
            results[mn] = {
                "mae": float(np.mean(errors)),
                "spearman_r": rho if not np.isnan(rho) else 0.0,
                "n": len(labels),
                "n_parsed": len(preds),
            }
        logger.info(f"  {short}: {results[mn]}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train fused LLM-based predictor for ROUTE_BALANCE (Qwen-0.5B LoRA)"
    )
    parser.add_argument("--input", required=True, help="Training data JSON/JSONL")
    parser.add_argument("--test-input", default=None, help="Test data JSON/JSONL")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B",
                        help="Base model to finetune")
    parser.add_argument("--target",
                        choices=["length", "length_bucket", "similarity", "judge", "reference_score", "deepeval", "quality"],
                        default="length")
    parser.add_argument("--target-models", nargs="+", default=None,
                        help="Target LLM models (default: all in data)")
    parser.add_argument("--mode", choices=["fused", "per_model"], default="fused",
                        help="fused: 1 call → all models. per_model: model name in prompt.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--output-dir", default="models/route_balance/baselines/qwen05b_lora")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType

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

    # Determine model names
    model_names = args.target_models or sorted(train_data[0]["models"].keys())
    logger.info(f"Target models: {model_names}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create dataset (fused or per-model)
    DatasetClass = FusedPredictionDataset if args.mode == "fused" else PerModelPredictionDataset
    logger.info(f"Mode: {args.mode}")

    train_dataset = DatasetClass(
        train_data, tokenizer, model_names, args.target, args.max_length,
    )
    logger.info(f"Training examples: {len(train_dataset)}")

    eval_dataset = None
    if test_data:
        eval_dataset = DatasetClass(
            test_data, tokenizer, model_names, args.target, args.max_length,
        )
        logger.info(f"Eval examples: {len(eval_dataset)}")

    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=8,  # effective batch = batch_size * 8 = 32
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    logger.info(f"Training completed in {train_time:.1f}s")

    # Save adapter
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluate if test data available
    results = {}
    if test_data:
        logger.info("=== Evaluation ===")
        results = evaluate_fused(
            model, tokenizer, test_data, model_names,
            args.target, args.device, args.max_length,
        )

    # Save metadata + results
    out = {
        "base_model": args.base_model,
        "target": args.target,
        "mode": args.mode,
        "fused": args.mode == "fused",
        "model_names": model_names,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "train_time_s": train_time,
        "num_train_examples": len(train_dataset),
        "results": results,
    }
    with open(Path(args.output_dir) / "training_results.json", "w") as f:
        json.dump(out, f, indent=2)

    logger.info(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
