#!/usr/bin/env python3
"""
Unified evaluation of ALL latency predictors.

Compares XGBoost, LSTM v1, LSTM v2, Roofline, and Static baselines
on the same test set. Supports per-target evaluation (E2E, TTFT, TPOT).

Usage:
    # Evaluate all predictors on test data
    python -m route_balance.predictor.route_balance.offline_training.evaluation.eval_latency_predictors \
        --test-data data/route_balance/latency_data/all/latency_test_tagged.jsonl \
        --xgboost-dir models/route_balance/xgboost/ \
        --output eval_results/latency_comparison.json

    # Evaluate specific predictors
    python -m route_balance.predictor.route_balance.offline_training.evaluation.eval_latency_predictors \
        --test-data data/route_balance/latency_data/all/latency_test_tagged.jsonl \
        --predictors xgboost roofline static \
        --target actual_e2e_latency \
        --output eval_results/latency_e2e.json

    # TTFT-only or TPOT-only evaluation
    python -m route_balance.predictor.route_balance.offline_training.evaluation.eval_latency_predictors \
        --test-data data/route_balance/latency_data/all/latency_test_tagged.jsonl \
        --xgboost-dir models/route_balance/xgboost/ \
        --target actual_ttft \
        --output eval_results/latency_ttft.json
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Baseline predictors (no model needed)
# -------------------------------------------------------------------

class StaticTPOTPredictor:
    """Static TPOT baseline: predict E2E = TTFT_static + TPOT_static × output_tokens.

    Uses fixed TPOT per instance type (from profiling or literature).
    """

    # Approximate static TPOT values (ms/token) from vLLM profiling
    # These are rough estimates — real values depend on load
    STATIC_TPOT_MS = {
        "qwen2.5-3b_p100": 15.0,
        "qwen2.5-3b_a30": 8.0,
        "qwen2.5-7b_a30": 12.0,
        "qwen2.5-14b_v100": 25.0,
        "qwen2.5-72b_a100": 35.0,
    }
    STATIC_TTFT_MS = {
        "qwen2.5-3b_p100": 50.0,
        "qwen2.5-3b_a30": 30.0,
        "qwen2.5-7b_a30": 40.0,
        "qwen2.5-14b_v100": 80.0,
        "qwen2.5-72b_a100": 120.0,
    }

    def predict(self, record: Dict, inst_type: str, target: str) -> float:
        tpot = self.STATIC_TPOT_MS.get(inst_type, 20.0)
        ttft = self.STATIC_TTFT_MS.get(inst_type, 50.0)
        output_tokens = record.get("actual_output_tokens") or record.get("num_predicted_output_tokens", 100)

        if target in ("actual_ttft", "ttft"):
            return ttft / 1000.0  # ms → s
        elif target in ("actual_tpot", "tpot"):
            return tpot / 1000.0
        else:  # e2e
            return (ttft + tpot * output_tokens) / 1000.0


class RooflinePredictor:
    """Roofline baseline: predict based on profiled throughput.

    E2E ≈ prompt_tokens / prefill_rate + output_tokens / decode_rate
    Uses EMA rates from the schedule_state if available.
    """

    def predict(self, record: Dict, inst_type: str, target: str) -> float:
        ss = record.get("schedule_state", {})
        prompt_tokens = record.get("num_prompt_tokens", 100)
        output_tokens = record.get("actual_output_tokens") or record.get("num_predicted_output_tokens", 100)

        prefill_rate = ss.get("ema_prefill_tok_per_s", 1000.0) or 1000.0
        decode_rate = ss.get("ema_decode_tok_per_s", 50.0) or 50.0

        ttft_est = prompt_tokens / prefill_rate
        tpot_est = 1.0 / decode_rate if decode_rate > 0 else 0.02
        e2e_est = ttft_est + output_tokens * tpot_est

        if target in ("actual_ttft", "ttft"):
            return ttft_est
        elif target in ("actual_tpot", "tpot"):
            return tpot_est
        else:
            return e2e_est


class MedianPredictor:
    """Median baseline: predict the median of training set per instance type."""

    def __init__(self, train_records: Dict[str, List[Dict]], target: str):
        self.medians = {}
        for inst_type, records in train_records.items():
            vals = [self._get_target(r, target) for r in records]
            vals = [v for v in vals if v is not None and v > 0]
            self.medians[inst_type] = float(np.median(vals)) if vals else 1.0

    def _get_target(self, record: Dict, target: str) -> Optional[float]:
        val = record.get(target) or record.get("actual_e2e_latency")
        return float(val) if val and float(val) > 0 else None

    def predict(self, record: Dict, inst_type: str, target: str) -> float:
        return self.medians.get(inst_type, 1.0)


# -------------------------------------------------------------------
# Evaluation logic
# -------------------------------------------------------------------

def compute_metrics(actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    errors = np.abs(actuals - predictions)
    rel_errors = errors / np.maximum(actuals, 1e-6)

    return {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mape": float(np.mean(rel_errors) * 100),
        "median_ae": float(np.median(errors)),
        "p50_error": float(np.percentile(errors, 50)),
        "p90_error": float(np.percentile(errors, 90)),
        "p95_error": float(np.percentile(errors, 95)),
        "p99_error": float(np.percentile(errors, 99)),
        "max_error": float(np.max(errors)),
        "spearman_r": float(_spearman(actuals, predictions)),
        "n": int(len(actuals)),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation."""
    from scipy import stats
    r, _ = stats.spearmanr(x, y)
    return r if not np.isnan(r) else 0.0


def evaluate_predictor(
    name: str,
    predict_fn: Callable,
    records_by_type: Dict[str, List[Dict]],
    target: str,
) -> Dict:
    """Evaluate a predictor on all instance types.

    Args:
        name: Predictor name
        predict_fn: Callable(record, inst_type, target) → float
        records_by_type: {inst_type: [records]}
        target: Target variable name

    Returns:
        {per_instance_type: metrics, aggregate: metrics}
    """
    per_type = {}
    all_actuals = []
    all_preds = []

    for inst_type, records in sorted(records_by_type.items()):
        actuals = []
        preds = []

        for rec in records:
            actual = rec.get(target) or rec.get("actual_e2e_latency")
            if actual is None or float(actual) <= 0:
                continue

            try:
                pred = predict_fn(rec, inst_type, target)
                if pred is not None and pred >= 0:
                    actuals.append(float(actual))
                    preds.append(float(pred))
            except Exception:
                continue

        if actuals:
            actuals_arr = np.array(actuals)
            preds_arr = np.array(preds)
            metrics = compute_metrics(actuals_arr, preds_arr)
            per_type[inst_type] = metrics
            all_actuals.extend(actuals)
            all_preds.extend(preds)

            logger.info(
                f"  {inst_type}: MAE={metrics['mae']:.4f}s "
                f"MAPE={metrics['mape']:.1f}% ρ={metrics['spearman_r']:.3f} "
                f"(n={metrics['n']})"
            )

    # Aggregate
    aggregate = {}
    if all_actuals:
        aggregate = compute_metrics(np.array(all_actuals), np.array(all_preds))

    return {
        "name": name,
        "target": target,
        "per_instance_type": per_type,
        "aggregate": aggregate,
    }


def load_test_data(path: str) -> Dict[str, List[Dict]]:
    """Load test data grouped by instance type."""
    records_by_type = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            inst_type = rec.get("instance_type", "unknown")
            records_by_type[inst_type].append(rec)

    for inst_type, records in records_by_type.items():
        logger.info(f"  Test: {inst_type}: {len(records)} records")

    return dict(records_by_type)


def print_comparison(results: List[Dict], target: str):
    """Print comparison table."""
    print(f"\n{'='*90}")
    print(f"  LATENCY PREDICTOR COMPARISON (target: {target})")
    print(f"{'='*90}")

    # Aggregate table
    print(f"\n  {'Predictor':<25} {'MAE':>10} {'MAPE':>8} {'P50':>10} {'P95':>10} {'ρ':>8} {'N':>8}")
    print(f"  {'-'*80}")
    for r in results:
        a = r.get("aggregate", {})
        if a:
            print(
                f"  {r['name']:<25} {a['mae']:>9.4f}s {a['mape']:>7.1f}% "
                f"{a['p50_error']:>9.4f}s {a['p95_error']:>9.4f}s "
                f"{a['spearman_r']:>7.3f} {a['n']:>7d}"
            )

    # Per-instance type
    all_types = set()
    for r in results:
        all_types.update(r.get("per_instance_type", {}).keys())

    for inst_type in sorted(all_types):
        print(f"\n  --- {inst_type} ---")
        print(f"  {'Predictor':<25} {'MAE':>10} {'MAPE':>8} {'P95':>10} {'ρ':>8}")
        print(f"  {'-'*60}")
        for r in results:
            m = r.get("per_instance_type", {}).get(inst_type)
            if m:
                print(
                    f"  {r['name']:<25} {m['mae']:>9.4f}s {m['mape']:>7.1f}% "
                    f"{m['p95_error']:>9.4f}s {m['spearman_r']:>7.3f}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Unified latency predictor evaluation"
    )
    parser.add_argument("--test-data", required=True, help="Test JSONL file")
    parser.add_argument("--train-data", default=None, help="Train JSONL (for median baseline)")
    parser.add_argument(
        "--target", default="actual_e2e_latency",
        choices=["actual_e2e_latency", "e2el", "actual_ttft", "ttft", "actual_tpot", "tpot"],
    )
    parser.add_argument("--xgboost-dir", default=None, help="XGBoost model directory")
    parser.add_argument("--lstm-v1-dir", default=None, help="LSTM v1 model directory")
    parser.add_argument("--lstm-v2-dir", default=None, help="LSTM v2 model directory")
    parser.add_argument(
        "--predictors", nargs="+",
        default=["xgboost", "roofline", "static", "median"],
        help="Predictors to evaluate",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_by_type = load_test_data(args.test_data)

    if args.max_samples > 0:
        for inst_type in test_by_type:
            test_by_type[inst_type] = test_by_type[inst_type][:args.max_samples]

    total = sum(len(v) for v in test_by_type.values())
    logger.info(f"Total: {total} records across {len(test_by_type)} instance types")

    all_results = []

    # --- XGBoost ---
    if "xgboost" in args.predictors and args.xgboost_dir:
        logger.info(f"\nEvaluating: XGBoost ({args.xgboost_dir})")
        try:
            from route_balance.predictor.route_balance.estimators.xgboost_predictor import (
                XGBoostLatencyPredictor, build_feature_vector,
            )
            xgb_predictor = XGBoostLatencyPredictor.load(args.xgboost_dir)

            def xgb_predict(rec, inst_type, target):
                ss = rec.get("schedule_state", {})
                if not ss:
                    return None
                prompt = int(rec.get("num_prompt_tokens", 0))
                output = int(rec.get("actual_output_tokens") or rec.get("num_predicted_output_tokens", 0))
                result = xgb_predictor.predict(inst_type, ss, prompt, output)
                # XGBoost always returns {"e2e_latency": val} regardless of target
                # The separate TTFT/TPOT models predict that metric under the same key
                return result.get("e2e_latency", 0)

            result = evaluate_predictor("XGBoost", xgb_predict, test_by_type, args.target)
            all_results.append(result)
        except Exception as e:
            logger.error(f"XGBoost evaluation failed: {e}")
            import traceback; traceback.print_exc()

    # --- LSTM v2 ---
    if "lstm_v2" in args.predictors and args.lstm_v2_dir:
        logger.info(f"\nEvaluating: LSTM v2 ({args.lstm_v2_dir})")
        try:
            from route_balance.predictor.route_balance.estimators.lstm_v2_predictor import LSTMv2LatencyPredictor
            lstm_predictor = LSTMv2LatencyPredictor.load(args.lstm_v2_dir)

            def lstm_v2_predict(rec, inst_type, target):
                ss = rec.get("schedule_state", {})
                prompt = rec.get("num_prompt_tokens", 0)
                output = rec.get("num_predicted_output_tokens", 0)
                result = lstm_predictor.predict(inst_type, ss, int(prompt), int(output))
                return result.get("e2e_latency", 0)

            result = evaluate_predictor("LSTM-v2 (queue)", lstm_v2_predict, test_by_type, args.target)
            all_results.append(result)
        except Exception as e:
            logger.error(f"LSTM v2 evaluation failed: {e}")
            import traceback; traceback.print_exc()

    # --- Roofline ---
    if "roofline" in args.predictors:
        logger.info("\nEvaluating: Roofline")
        roofline = RooflinePredictor()
        result = evaluate_predictor("Roofline", roofline.predict, test_by_type, args.target)
        all_results.append(result)

    # --- Static TPOT ---
    if "static" in args.predictors:
        logger.info("\nEvaluating: Static TPOT")
        static = StaticTPOTPredictor()
        result = evaluate_predictor("Static TPOT", static.predict, test_by_type, args.target)
        all_results.append(result)

    # --- Median ---
    if "median" in args.predictors:
        logger.info("\nEvaluating: Median baseline")
        if args.train_data:
            train_by_type = load_test_data(args.train_data)
        else:
            train_by_type = test_by_type  # Use test as fallback
        median_pred = MedianPredictor(train_by_type, args.target)
        result = evaluate_predictor("Median", median_pred.predict, test_by_type, args.target)
        all_results.append(result)

    # Print comparison
    if all_results:
        print_comparison(all_results, args.target)

    # Save results
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)

        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=convert)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
