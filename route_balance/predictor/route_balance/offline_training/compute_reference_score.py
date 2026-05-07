#!/usr/bin/env python3
"""
Compute reference_score for training data.

Recovered from Claude session d3311e06 (March 27, 2026).
This was originally run interactively on GPU VM.

reference_score is a unified quality metric per (prompt, model):
  - GSM8K: exact-match accuracy (0 or 1) — extract final number, compare to reference
  - SQuAD: token-level F1 score (0.0-1.0) — compare response tokens to answer spans
  - beaver_tails: ProtectAI refusal classifier score — high = correctly refused
  - Others (code_ultra, reward_bench, mix_instruct, LMSYS): embedding cosine similarity

Prerequisites:
  - Input data must have `reference_similarity` field (from add_reference_similarity.py)
  - Input data must have `dataset` field
  - Needs access to original raw data for request_id → source mapping
  - Needs HuggingFace datasets: openai/gsm8k, rajpurkar/squad

Usage:
    python3 -m route_balance.predictor.route_balance.offline_training.compute_reference_score \
        --input data/route_balance/training_data_with_ref/train_fixed.jsonl \
        --raw-data data/route_balance/training_data/route_balance_v3_all_training_fixed.json \
        --output data/route_balance/training_data_with_ref/train_fixed.jsonl

    python3 -m route_balance.predictor.route_balance.offline_training.compute_reference_score \
        --input data/route_balance/training_data_with_ref/test_fixed.jsonl \
        --raw-data data/route_balance/training_data/route_balance_v3_all_training_fixed.json \
        --output data/route_balance/training_data_with_ref/test_fixed.jsonl
"""

import argparse
import json
import logging
import re
from collections import Counter

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_final_number(text: str):
    """Extract the last number from a model response (GSM8K answer extraction).

    Tries #### pattern first (GSM8K standard format), falls back to last number.
    """
    if not text:
        return None
    # Look for #### pattern first (GSM8K standard)
    m = re.search(r'####\s*([0-9,.\-]+)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    # Otherwise find the last number in the text
    numbers = re.findall(r'[-]?[0-9][0-9,]*\.?[0-9]*', text)
    return numbers[-1].replace(',', '').strip() if numbers else None


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth (SQuAD-style)."""
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(gt_tokens) if gt_tokens else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser(
        description="Compute reference_score for training data"
    )
    parser.add_argument("--input", required=True, help="Input JSONL with reference_similarity")
    parser.add_argument("--raw-data", required=True,
                        help="Original raw JSON with request_id → source mapping")
    parser.add_argument("--output", required=True, help="Output JSONL with reference_score added")
    args = parser.parse_args()

    # Load input data
    with open(args.input) as f:
        data = [json.loads(l) for l in f]
    logger.info(f"Loaded {len(data)} entries from {args.input}")

    # Load raw data for source mapping
    with open(args.raw_data) as f:
        raw = json.load(f)
    rid_to_source = {r["request_id"]: r["source"] for r in raw["requests"]}
    logger.info(f"Source mapping: {len(rid_to_source)} entries")

    # Load HF datasets
    from datasets import load_dataset
    gsm_ds = load_dataset("openai/gsm8k", "main", split="train")
    squad_ds = load_dataset("rajpurkar/squad", split="train")
    logger.info(f"GSM8K: {len(gsm_ds)}, SQuAD: {len(squad_ds)}")

    # Process each entry
    stats = Counter()
    per_model_gsm_correct = Counter()
    per_model_gsm_total = Counter()
    per_model_squad_f1 = {}

    for entry in data:
        ds = entry.get("dataset", "")
        source = rid_to_source.get(entry["request_id"], "")

        if ds == "gsm8k" and source.startswith("gsm8k/"):
            idx = int(source.split("/")[1])
            answer_text = gsm_ds[idx]["answer"]
            ref_match = re.search(r"####\s*(.+)", answer_text)
            ref_number = ref_match.group(1).strip().replace(',', '') if ref_match else None

            for model_name, model_data in entry.get("models", {}).items():
                predicted = extract_final_number(model_data.get("response", ""))
                is_correct = (predicted == ref_number) if (predicted and ref_number) else False
                model_data["reference_score"] = 1.0 if is_correct else 0.0
                per_model_gsm_correct[model_name] += int(is_correct)
                per_model_gsm_total[model_name] += 1
            stats["gsm8k"] += 1

        elif ds == "squad" and source.startswith("squad/"):
            idx = int(source.split("/")[1])
            answers = squad_ds[idx]["answers"]["text"]

            for model_name, model_data in entry.get("models", {}).items():
                response = model_data.get("response", "").strip()
                best_f1 = max(compute_f1(response, ans) for ans in answers) if answers else 0.0
                model_data["reference_score"] = round(best_f1, 4)
                if model_name not in per_model_squad_f1:
                    per_model_squad_f1[model_name] = []
                per_model_squad_f1[model_name].append(best_f1)
            stats["squad"] += 1

        elif ds == "beaver_tails":
            for model_name, model_data in entry.get("models", {}).items():
                js = model_data.get("llm_judge_scores", {})
                safety = js.get("protectai_distilroberta-base-rejection-v1", 0.0) or 0.0
                model_data["reference_score"] = float(safety)
            stats["beaver_tails"] += 1

        else:
            # Others: use embedding similarity as reference_score
            for model_name, model_data in entry.get("models", {}).items():
                model_data["reference_score"] = model_data.get("reference_similarity")
            stats[ds or "unknown"] += 1

    # Save output
    with open(args.output, "w") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(data)} entries to {args.output}")

    # Report stats
    logger.info("=== reference_score coverage ===")
    for ds, cnt in sorted(stats.items()):
        logger.info(f"  {ds}: {cnt}")

    if per_model_gsm_correct:
        logger.info("=== GSM8K exact-match accuracy ===")
        for model in sorted(per_model_gsm_total.keys()):
            acc = per_model_gsm_correct[model] / per_model_gsm_total[model] * 100
            logger.info(f"  {model.split('/')[-1]}: {per_model_gsm_correct[model]}/{per_model_gsm_total[model]} = {acc:.1f}%")

    if per_model_squad_f1:
        logger.info("=== SQuAD token F1 ===")
        for model in sorted(per_model_squad_f1.keys()):
            vals = per_model_squad_f1[model]
            logger.info(f"  {model.split('/')[-1]}: mean={np.mean(vals):.3f}, median={np.median(vals):.3f}")


if __name__ == "__main__":
    main()
