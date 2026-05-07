#!/usr/bin/env python3
"""
Evaluate bucket-based budget filtering for ROUTE_BALANCE scheduling.

Budget modes derived from per-prompt actual response lengths:
  - restrictive: budget = min(actual_lengths) → 1 model guaranteed
  - average: budget = median(actual_lengths) → ~2 models pass
  - open: budget = second_largest(actual_lengths) → only longest filtered

For each (mode, threshold) pair, measures:
  - Per-model: false accept rate, false reject rate, actual compliance
  - Per-prompt: how many models passed vs should have passed

Usage:
    python -m route_balance.predictor.route_balance.offline_training.evaluate_bucket_filtering \
        --test-input data/route_balance/training_data/test_fixed.jsonl \
        --model-dir models/route_balance/study/modernbert_fused_length_bucket/ \
        --device cuda
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_bucket_model(model_dir: str, device: str = "cuda"):
    """Load a trained bucket classifier and return (model, tokenizer, num_buckets, bucket_size)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, trust_remote_code=True
    )
    model.eval()
    model.to(device)

    num_buckets = model.config.num_labels
    bucket_size = 1024 // num_buckets

    logger.info(f"Loaded bucket model: {num_buckets} buckets, bucket_size={bucket_size}")
    return model, tokenizer, num_buckets, bucket_size


def predict_bucket_probs(model, tokenizer, prompts: list, device: str, batch_size: int = 32):
    """Run bucket classifier on prompts, return softmax probabilities (N, num_buckets)."""
    import torch

    all_probs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch, truncation=True, max_length=1024,
            padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


def compute_budgets_per_prompt(test_data: list, target_models: list):
    """Compute per-prompt budgets for each mode.

    Returns:
        dict of mode -> np.ndarray of shape (N,) with budget per prompt
        actual_lengths: dict of model -> np.ndarray of shape (N,)
    """
    n = len(test_data)
    actual_lengths = {m: np.zeros(n) for m in target_models}

    for i, req in enumerate(test_data):
        for m in target_models:
            m_data = req["models"].get(m, {})
            actual_lengths[m][i] = m_data.get("output_length", 0)

    # Stack all models: (N, num_models)
    all_lens = np.stack([actual_lengths[m] for m in target_models], axis=1)

    # Sort per prompt
    sorted_lens = np.sort(all_lens, axis=1)
    num_models = len(target_models)

    budgets = {
        "restrictive": sorted_lens[:, 0],                          # min → 1 model guaranteed
        "average": np.median(all_lens, axis=1),                    # median → ~2 models
        "open": sorted_lens[:, -2] if num_models > 1 else sorted_lens[:, 0],  # 2nd largest → filter longest only
    }

    return budgets, actual_lengths


def evaluate_filtering(
    probs_per_model: dict,
    actual_lengths: dict,
    budgets_per_mode: dict,
    thresholds: list,
    bucket_size: int,
    num_buckets: int,
):
    """Evaluate filtering across mode × threshold grid.

    Args:
        probs_per_model: {model_name: (N, num_buckets) array}
        actual_lengths: {model_name: (N,) array}
        budgets_per_mode: {mode: (N,) array}
        thresholds: list of threshold values
    """
    results = {}
    target_models = list(probs_per_model.keys())
    n = len(next(iter(actual_lengths.values())))

    for mode, budgets in budgets_per_mode.items():
        for threshold in thresholds:
            # Per-prompt, per-model: does it pass the filter?
            total_tp = 0
            total_fa = 0
            total_fr = 0
            total_tn = 0
            models_passed_per_prompt = []
            models_should_pass_per_prompt = []

            per_model_stats = {}

            for m in target_models:
                probs = probs_per_model[m]
                actuals = actual_lengths[m]

                tp = fa = fr = tn = 0
                for i in range(n):
                    budget = budgets[i]
                    if budget <= 0:
                        continue

                    max_bucket = min(int(budget) // bucket_size, num_buckets - 1)
                    cdf = probs[i, :max_bucket + 1].sum()
                    passed = cdf >= threshold
                    within = actuals[i] <= budget

                    if passed and within:
                        tp += 1
                    elif passed and not within:
                        fa += 1
                    elif not passed and within:
                        fr += 1
                    else:
                        tn += 1

                total_tp += tp
                total_fa += fa
                total_fr += fr
                total_tn += tn

                n_passed = tp + fa
                n_within = tp + fr
                per_model_stats[m] = {
                    "tp": tp, "fa": fa, "fr": fr, "tn": tn,
                    "false_accept_rate": fa / max(n_passed, 1),
                    "false_reject_rate": fr / max(n_within, 1),
                    "compliance": tp / max(n_passed, 1),
                }

            # Per-prompt aggregation
            for i in range(n):
                budget = budgets[i]
                if budget <= 0:
                    continue
                max_bucket = min(int(budget) // bucket_size, num_buckets - 1)

                passed_count = 0
                should_count = 0
                for m in target_models:
                    cdf = probs_per_model[m][i, :max_bucket + 1].sum()
                    if cdf >= threshold:
                        passed_count += 1
                    if actual_lengths[m][i] <= budget:
                        should_count += 1
                models_passed_per_prompt.append(passed_count)
                models_should_pass_per_prompt.append(should_count)

            passed_arr = np.array(models_passed_per_prompt)
            should_arr = np.array(models_should_pass_per_prompt)

            total_n_passed = total_tp + total_fa
            total_n_within = total_tp + total_fr

            results[(mode, threshold)] = {
                "mode": mode,
                "threshold": threshold,
                "total_decisions": total_tp + total_fa + total_fr + total_tn,
                "total_compliance": total_tp / max(total_n_passed, 1),
                "total_false_accept": total_fa / max(total_n_passed, 1),
                "total_false_reject": total_fr / max(total_n_within, 1),
                "avg_models_passed": float(passed_arr.mean()),
                "avg_models_should_pass": float(should_arr.mean()),
                "avg_budget": float(budgets[budgets > 0].mean()),
                "per_model": per_model_stats,
            }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate bucket-based budget filtering"
    )
    parser.add_argument("--test-input", required=True, help="Test data JSONL")
    parser.add_argument("--model-dir", required=True,
                        help="Trained bucket classifier directory")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.5, 0.7, 0.9],
                        help="Confidence thresholds")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    # Load test data
    with open(args.test_input) as f:
        if args.test_input.endswith(".jsonl"):
            test_data = [json.loads(line) for line in f]
        else:
            raw = json.load(f)
            test_data = raw["requests"] if "requests" in raw else raw
    logger.info(f"Test data: {len(test_data)} requests")

    target_models = sorted(test_data[0]["models"].keys())
    logger.info(f"Target models: {target_models}")

    # Compute per-prompt budgets
    budgets_per_mode, actual_lengths = compute_budgets_per_prompt(test_data, target_models)
    for mode, budgets in budgets_per_mode.items():
        logger.info(f"  {mode}: mean_budget={budgets.mean():.0f}, median={np.median(budgets):.0f}")

    # Load bucket model and predict for each target model
    model_dir = Path(args.model_dir)
    probs_per_model = {}

    # Check if model_dir has per-model subdirs or is a fused model
    prompts = [req["prompt"] for req in test_data]

    for tm in target_models:
        tm_key = tm.replace("/", "_")
        subdir = model_dir / f"{tm_key}_length_bucket"
        if not subdir.exists():
            # Fused model — same model for all targets
            subdir = model_dir

        if tm not in probs_per_model:
            logger.info(f"Predicting bucket probs for {tm} from {subdir}")
            model, tokenizer, num_buckets, bucket_size = load_bucket_model(str(subdir), args.device)
            probs_per_model[tm] = predict_bucket_probs(model, tokenizer, prompts, args.device)

            # Free GPU memory
            del model
            import torch
            torch.cuda.empty_cache()

    # Evaluate
    results = evaluate_filtering(
        probs_per_model, actual_lengths, budgets_per_mode,
        args.thresholds, bucket_size, num_buckets,
    )

    # Print summary
    print(f"\n{'=' * 90}")
    print(f"  BUCKET FILTERING EVALUATION")
    print(f"  {len(test_data)} prompts, {len(target_models)} models")
    print(f"{'=' * 90}")

    header = (f"{'Mode':<13} {'Thresh':>6} {'AvgBudget':>10} "
              f"{'Compliance':>11} {'FalseAccept':>12} {'FalseReject':>12} "
              f"{'AvgPassed':>10} {'AvgShould':>10}")
    print(header)
    print("-" * 90)

    for mode in ["restrictive", "average", "open"]:
        for threshold in args.thresholds:
            r = results[(mode, threshold)]
            print(
                f"  {mode:<11} {threshold:>6.1f} {r['avg_budget']:>10.0f} "
                f"{r['total_compliance']:>10.1%} {r['total_false_accept']:>11.1%} "
                f"{r['total_false_reject']:>11.1%} "
                f"{r['avg_models_passed']:>10.1f} {r['avg_models_should_pass']:>10.1f}"
            )
        print()

    # Per-model breakdown for average mode, threshold=0.7
    key = ("average", 0.7)
    if key in results:
        print(f"\nPer-model detail (mode=average, threshold=0.7):")
        print(f"{'Model':<25} {'Compliance':>11} {'FalseAccept':>12} {'FalseReject':>12}")
        print("-" * 65)
        for m, stats in results[key]["per_model"].items():
            m_short = m.split("/")[-1]
            print(
                f"  {m_short:<23} {stats['compliance']:>10.1%} "
                f"{stats['false_accept_rate']:>11.1%} {stats['false_reject_rate']:>11.1%}"
            )

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for (mode, thresh), v in results.items():
            k = f"{mode}_t{thresh}"
            serializable[k] = {kk: vv for kk, vv in v.items() if kk != "per_model"}
            serializable[k]["per_model"] = {
                m.split("/")[-1]: s for m, s in v["per_model"].items()
            }
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
