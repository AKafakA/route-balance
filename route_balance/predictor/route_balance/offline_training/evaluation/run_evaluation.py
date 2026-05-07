#!/usr/bin/env python3
"""
Unified evaluation runner for ALL RouteBalance predictors.

Loads adapters, runs inference on test set, computes comparable metrics
across all models for each target (length, similarity, judge).
Bucket models also get classification + filtering metrics.

Usage:
    python -m route_balance.predictor.route_balance.offline_training.evaluation.run_evaluation \
        --test-input data/route_balance/training_data/test_fixed.jsonl \
        --train-input data/route_balance/training_data/train_fixed.jsonl \
        --config evaluation_config.json \
        --device cuda
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .ground_truth import get_ground_truth
from .metrics import regression_metrics, bucket_classification_metrics, bucket_filtering_metrics
from .adapters.base import BaseAdapter
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_adapter(pred_type: str, pred_path: str, target: str, device: str,
                 train_data: list = None, **kwargs) -> BaseAdapter:
    """Factory: create adapter by type."""
    if pred_type == "knn":
        from .adapters.knn_adapter import KNNAdapter
        return KNNAdapter(pred_path, train_data, target, device)
    elif pred_type == "mlp":
        from .adapters.mlp_adapter import MLPAdapter
        return MLPAdapter(pred_path, target, device)
    elif pred_type in ("encoder", "modernbert", "roberta", "debertav3"):
        from .adapters.encoder_adapter import EncoderAdapter
        return EncoderAdapter(pred_path, target, device)
    elif pred_type in ("bucket", "bucket_encoder"):
        from .adapters.encoder_adapter import BucketEncoderAdapter
        return BucketEncoderAdapter(pred_path, target, device,
                                     bucket_size=kwargs.get("bucket_size", 64))
    elif pred_type in ("llm", "qwen05b"):
        from .adapters.llm_adapter import LLMAdapter
        return LLMAdapter(pred_path, target, device)
    elif pred_type in ("lora_encoder", "lora_sr"):
        from .adapters.lora_encoder_adapter import LoraEncoderAdapter
        return LoraEncoderAdapter(pred_path, target, device)
    elif pred_type in ("multitarget", "multi_target", "fused_multitarget", "lora_multitarget"):
        from .adapters.multitarget_adapter import MultiTargetAdapter
        return MultiTargetAdapter(pred_path, target, device)
    else:
        raise ValueError(f"Unknown predictor type: {pred_type}")


def evaluate_single(
    adapter: BaseAdapter, test_data: list, target_models: list,
    target: str, name: str,
) -> Dict:
    """Evaluate one adapter on all target models.

    Returns dict with per-model metrics and aggregate.
    """
    per_model = {}
    probs_per_model = {}  # for bucket filtering

    # Pre-compute predictions for all prompts × all models at once.
    # For fused adapters (LoRA, fused encoder), this avoids redundant generation
    # by calling predict(prompt, all_models) once per prompt instead of N times.
    t0 = time.time()
    all_actuals = {tm: [] for tm in target_models}  # tm -> [actual]
    all_preds = {tm: [] for tm in target_models}    # tm -> [pred]
    all_probs_map = {tm: [] for tm in target_models}  # tm -> [probs]

    # Per-prompt aligned data for cross-model metrics (Bug 2 fix)
    # Each entry: {model: (actual, pred)} — only prompts where ALL models have data
    prompt_aligned = []

    n_total = len(test_data)
    n_processed = 0
    log_interval = max(1, n_total // 20)  # Log every 5% progress

    for req in test_data:
        # Get ground truth for all models first
        gt = {}
        for tm in target_models:
            actual = get_ground_truth(req, tm, target)
            if actual is not None:
                gt[tm] = actual
        if not gt:
            continue

        # Single predict call with all models that have ground truth
        models_with_gt = list(gt.keys())
        preds = adapter.predict(req["prompt"], models_with_gt)

        n_processed += 1
        if n_processed % log_interval == 0:
            elapsed = time.time() - t0
            rate = n_processed / elapsed if elapsed > 0 else 0
            eta = (n_total - n_processed) / rate if rate > 0 else 0
            logger.info(
                f"  [{name}] {n_processed}/{n_total} ({100*n_processed/n_total:.0f}%) "
                f"| {elapsed:.0f}s elapsed | {rate:.1f} prompts/s | ETA {eta:.0f}s"
            )
            import sys; sys.stdout.flush(); sys.stderr.flush()

        probs = None
        if adapter.supports_probs:
            probs = adapter.predict_probs(req["prompt"], models_with_gt)

        # Collect per-model data
        prompt_entry = {}
        for tm in models_with_gt:
            if tm in preds:
                all_actuals[tm].append(gt[tm])
                all_preds[tm].append(preds[tm])
                if probs and tm in probs:
                    all_probs_map[tm].append(probs[tm])
                prompt_entry[tm] = (gt[tm], preds[tm])

        # Only include in aligned data if ALL target models have data for this prompt
        if len(prompt_entry) == len(target_models):
            prompt_aligned.append(prompt_entry)

    total_elapsed = time.time() - t0

    # Now compute per-model metrics from collected predictions
    for tm in target_models:
        actual_vals = all_actuals[tm]
        pred_vals = all_preds[tm]
        all_probs = all_probs_map[tm]
        elapsed = total_elapsed / max(len(target_models), 1)  # approximate per-model time
        if not actual_vals:
            continue

        actual_arr = np.array(actual_vals)
        pred_arr = np.array(pred_vals)

        # Regression metrics (all adapters)
        metrics = regression_metrics(actual_arr, pred_arr)
        metrics["inference_time_s"] = elapsed
        metrics["samples_per_sec"] = len(actual_vals) / elapsed if elapsed > 0 else 0

        # Bucket classification metrics (bucket adapters only)
        if all_probs:
            probs_arr = np.stack(all_probs)
            bucket_metrics = bucket_classification_metrics(actual_arr, probs_arr)
            metrics["bucket"] = bucket_metrics
            probs_per_model[tm] = probs_arr

        per_model[tm] = metrics
        ms = tm.split("/")[-1]
        logger.info(
            f"  {ms}: MAE={metrics['mae']:.2f} MAPE={metrics['mape']:.1f}% "
            f"ρ={metrics['spearman_r']:.3f} ({elapsed:.1f}s)"
        )
        if "bucket" in metrics:
            b = metrics["bucket"]
            logger.info(
                f"    Bucket: acc={b['accuracy']:.3f} adj={b['adjacent_accuracy']:.3f} "
                f"top3={b['top3_accuracy']:.3f} E[len]_MAE={b['expected_length_mae']:.1f}"
            )

    # Aggregate
    if per_model:
        agg = {
            k: float(np.mean([m[k] for m in per_model.values() if k in m]))
            for k in ["mae", "mape", "acc_50", "acc_100", "spearman_r"]
        }
    else:
        agg = {}

    # Cross-model metrics: per-prompt ranking quality (using aligned data)
    cross_model = _compute_cross_model_metrics(
        prompt_aligned, target_models, target
    )
    if cross_model:
        agg.update(cross_model)
        logger.info(
            f"  Cross-model: best_model_acc={cross_model.get('best_model_accuracy', 0):.3f} "
            f"rank_ρ={cross_model.get('mean_rank_correlation', 0):.3f}"
        )

    # Bucket filtering (if bucket adapter)
    filtering = None
    if probs_per_model:
        actual_per_model = {}
        for tm in probs_per_model:
            actuals = []
            for req in test_data:
                a = get_ground_truth(req, tm, target)
                if a is not None:
                    actuals.append(a)
            actual_per_model[tm] = np.array(actuals)

        filtering = bucket_filtering_metrics(actual_per_model, probs_per_model)

    return {
        "name": name,
        "target": target,
        "per_model": per_model,
        "aggregate": agg,
        "filtering": filtering,
    }


def _compute_cross_model_metrics(
    prompt_aligned: List[Dict],
    target_models: List[str],
    target: str,
) -> Dict:
    """Compute cross-model ranking metrics per prompt.

    Args:
        prompt_aligned: List of dicts, each {model: (actual, pred)} for prompts
            where ALL target models have both actual and predicted values.
        target_models: List of model names.
        target: Target name (for direction interpretation).

    For each prompt compute:
    - best_model_accuracy: does argmax(predicted) == argmax(actual)?
    - top2_accuracy: is the predicted best model in the actual top-2?
    - mean_rank_correlation: Spearman ρ of model ranking per prompt
    """
    if len(target_models) < 2 or not prompt_aligned:
        return {}

    best_match = 0
    top2_match = 0
    rank_corrs = []
    n_valid = 0

    for entry in prompt_aligned:
        actual_scores = []
        pred_scores = []
        for tm in target_models:
            if tm in entry:
                actual_scores.append(entry[tm][0])
                pred_scores.append(entry[tm][1])

        if len(actual_scores) < 2:
            continue

        n_valid += 1
        actual_arr = np.array(actual_scores)
        pred_arr = np.array(pred_scores)

        # Best model accuracy (higher = better for quality targets)
        # Handle ties: all models with the max actual score count as "best"
        pred_best = np.argmax(pred_arr)
        actual_max = np.max(actual_arr)
        actual_best_set = set(np.where(actual_arr == actual_max)[0])
        if pred_best in actual_best_set:
            best_match += 1

        # Top-2 accuracy: is predicted best among models with top-2 score values?
        # Handle ties: include ALL models with the 1st or 2nd highest actual score
        unique_scores = sorted(set(actual_arr), reverse=True)
        top2_threshold = unique_scores[min(1, len(unique_scores) - 1)]
        actual_top2 = set(np.where(actual_arr >= top2_threshold)[0])
        if pred_best in actual_top2:
            top2_match += 1

        # Per-prompt rank correlation
        if np.std(actual_arr) > 0 and np.std(pred_arr) > 0:
            rho, _ = stats.spearmanr(actual_arr, pred_arr)
            if not np.isnan(rho):
                rank_corrs.append(rho)

    if n_valid == 0:
        return {}

    return {
        "best_model_accuracy": best_match / n_valid,
        "top2_model_accuracy": top2_match / n_valid,
        "mean_rank_correlation": float(np.mean(rank_corrs)) if rank_corrs else 0.0,
        "n_cross_model_prompts": n_valid,
    }


def print_comparison(results: list, target_models: list, target: str):
    """Print comparison table across all evaluated predictors."""
    print(f"\n{'=' * 90}")
    print(f"  COMPARISON: {target.upper()} prediction")
    print(f"{'=' * 90}")

    # Per-model table
    for tm in target_models:
        ms = tm.split("/")[-1]
        print(f"\n  --- {ms} ---")
        header = f"{'Predictor':<30} {'MAE':>8} {'MAPE':>8} {'Acc@50':>8} {'ρ':>8}"
        print(f"  {header}")
        print(f"  {'-' * 66}")

        for r in results:
            if tm in r["per_model"]:
                m = r["per_model"][tm]
                print(
                    f"  {r['name']:<30} {m['mae']:>8.2f} {m['mape']:>7.1f}% "
                    f"{m.get('acc_50', 0):>8.3f} {m['spearman_r']:>8.3f}"
                )
                if "bucket" in m:
                    b = m["bucket"]
                    print(
                        f"  {'  (bucket)':<30} acc={b['accuracy']:.3f} "
                        f"adj={b['adjacent_accuracy']:.3f} "
                        f"E[len]_MAE={b['expected_length_mae']:.1f}"
                    )

    # Aggregate table
    has_cross = any(r["aggregate"].get("best_model_accuracy") is not None for r in results)
    print(f"\n  --- Aggregate ---")
    header = f"{'Predictor':<30} {'MAE':>8} {'MAPE':>8} {'ρ':>8}"
    if has_cross:
        header += f" {'BestAcc':>8} {'Top2Acc':>9} {'RankCorr':>9}"
    print(f"  {header}")
    print(f"  {'-' * (56 + (28 if has_cross else 0))}")
    for r in results:
        if r["aggregate"]:
            a = r["aggregate"]
            line = f"  {r['name']:<30} {a['mae']:>8.2f} {a['mape']:>7.1f}% {a['spearman_r']:>8.3f}"
            if has_cross:
                line += f" {a.get('best_model_accuracy', 0):>8.3f}"
                line += f" {a.get('top2_model_accuracy', 0):>9.3f}"
                line += f" {a.get('mean_rank_correlation', 0):>9.3f}"
            print(line)

    # Filtering results
    filtering_results = [r for r in results if r.get("filtering")]
    if filtering_results:
        print(f"\n  --- Bucket Filtering ---")
        header = f"{'Predictor':<20} {'Mode':<10} {'Thresh':>7} {'Compliance':>11} {'FalseAccept':>12} {'FalseReject':>12}"
        print(f"  {header}")
        print(f"  {'-' * 75}")
        for r in filtering_results:
            for key, filt in sorted(r["filtering"].items()):
                print(
                    f"  {r['name']:<20} {filt['mode']:<10} {filt['threshold']:>7.1f} "
                    f"{filt['compliance']:>10.1%} {filt['false_accept']:>11.1%} "
                    f"{filt['false_reject']:>11.1%}"
                )


def main():
    parser = argparse.ArgumentParser(description="Unified ROUTE_BALANCE predictor evaluation")
    parser.add_argument("--test-input", required=True, help="Test data JSONL")
    parser.add_argument("--train-input", default=None, help="Training data (for KNN)")
    parser.add_argument(
        "--predictors", nargs="+", required=True,
        help="type:path pairs (e.g., knn:models/route_balance/knn encoder:models/route_balance/modernbert)"
    )
    parser.add_argument("--target", choices=["length", "length_bucket", "similarity", "judge", "reference_score", "deepeval", "prometheus"], required=True)
    parser.add_argument("--target-models", nargs="+", default=None)
    parser.add_argument("--bucket-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--csv", default=None, help="Also save results as CSV")
    args = parser.parse_args()

    # Load data
    with open(args.test_input) as f:
        if args.test_input.endswith(".jsonl"):
            test_data = [json.loads(line) for line in f]
        else:
            raw = json.load(f)
            test_data = raw["requests"] if "requests" in raw else raw

    if args.max_samples > 0:
        test_data = test_data[:args.max_samples]

    train_data = None
    if args.train_input:
        with open(args.train_input) as f:
            if args.train_input.endswith(".jsonl"):
                train_data = [json.loads(line) for line in f]
            else:
                raw = json.load(f)
                train_data = raw["requests"] if "requests" in raw else raw

    target_models = args.target_models or sorted(test_data[0]["models"].keys())
    logger.info(f"Test: {len(test_data)}, models: {[m.split('/')[-1] for m in target_models]}, target: {args.target}")

    # Evaluate each predictor
    all_results = []
    for spec in args.predictors:
        parts = spec.split(":", 1)
        if len(parts) != 2:
            logger.warning(f"Invalid spec: {spec}")
            continue
        pred_type, pred_path = parts
        name = f"{pred_type}:{Path(pred_path).name}"

        logger.info(f"\nEvaluating: {name}")
        try:
            adapter = load_adapter(
                pred_type, pred_path, args.target, args.device,
                train_data=train_data, bucket_size=args.bucket_size,
            )
            result = evaluate_single(adapter, test_data, target_models, args.target, name)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Failed {name}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        print_comparison(all_results, target_models, args.target)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy to float for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=convert)
        logger.info(f"Results saved to {args.output}")

    if args.csv and all_results:
        _save_csv(all_results, target_models, args.csv)
        logger.info(f"CSV saved to {args.csv}")


def _save_csv(results: list, target_models: list, csv_path: str):
    """Save results as CSV for easy import into paper tables."""
    import csv

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["predictor", "target", "model", "mae", "mape", "spearman_r",
                  "acc_50", "acc_100", "n"]
        # Add bucket columns if any result has them
        has_bucket = any("bucket" in m for r in results for m in r.get("per_model", {}).values())
        if has_bucket:
            header += ["bucket_acc", "bucket_adj_acc", "bucket_top3", "bucket_elen_mae"]
        # Add cross-model columns
        header += ["best_model_acc", "top2_model_acc", "rank_corr"]
        writer.writerow(header)

        # Per-model rows
        for r in results:
            for tm in target_models:
                m = r.get("per_model", {}).get(tm, {})
                if not m:
                    continue
                row = [
                    r["name"], r["target"], tm.split("/")[-1],
                    f"{m['mae']:.4f}", f"{m['mape']:.1f}",
                    f"{m['spearman_r']:.4f}",
                    f"{m.get('acc_50', 0):.3f}", f"{m.get('acc_100', 0):.3f}",
                    m.get("n", 0),
                ]
                if has_bucket:
                    b = m.get("bucket", {})
                    row += [
                        f"{b.get('accuracy', 0):.3f}" if b else "",
                        f"{b.get('adjacent_accuracy', 0):.3f}" if b else "",
                        f"{b.get('top3_accuracy', 0):.3f}" if b else "",
                        f"{b.get('expected_length_mae', 0):.1f}" if b else "",
                    ]
                # Cross-model from aggregate (same for all models)
                agg = r.get("aggregate", {})
                row += [
                    f"{agg.get('best_model_accuracy', ''):.3f}" if agg.get('best_model_accuracy') is not None else "",
                    f"{agg.get('top2_model_accuracy', ''):.3f}" if agg.get('top2_model_accuracy') is not None else "",
                    f"{agg.get('mean_rank_correlation', ''):.3f}" if agg.get('mean_rank_correlation') is not None else "",
                ]
                writer.writerow(row)

        # Aggregate row
        for r in results:
            agg = r.get("aggregate", {})
            if agg:
                row = [
                    r["name"], r["target"], "AGGREGATE",
                    f"{agg['mae']:.4f}", f"{agg['mape']:.1f}",
                    f"{agg['spearman_r']:.4f}",
                    f"{agg.get('acc_50', 0):.3f}", f"{agg.get('acc_100', 0):.3f}",
                    "",
                ]
                if has_bucket:
                    row += ["", "", "", ""]
                row += [
                    f"{agg.get('best_model_accuracy', ''):.3f}" if agg.get('best_model_accuracy') is not None else "",
                    f"{agg.get('top2_model_accuracy', ''):.3f}" if agg.get('top2_model_accuracy') is not None else "",
                    f"{agg.get('mean_rank_correlation', ''):.3f}" if agg.get('mean_rank_correlation') is not None else "",
                ]
                writer.writerow(row)


if __name__ == "__main__":
    main()
