#!/usr/bin/env python3
"""
PFS-style (Past-Future Scheduler) baselines for ROUTE_BALANCE.

Training-free output length prediction baselines inspired by Gong et al.
(ASPLOS 2025). Uses historical distributions only — no prompt content.

Baselines:
1. Global PFS: P(l) from all training data (simulates stable workload)
2. Input-length-binned PFS: P(l | input_len_bin) — adapts to prompt length
3. Static majority: always predict most common bucket
4. Global mean: always predict mean length

Usage:
    python -m route_balance.predictor.route_balance.offline_training.eval_pfs_baseline \
        --train data/route_balance/training_data/train_fixed.jsonl \
        --test data/route_balance/training_data/test_fixed.jsonl \
        --output models/route_balance/results/pfs_baselines.json
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


BUCKET_SIZE = 64
MAX_BUCKETS = 16
MODELS = ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-72B"]

# Input length bins for binned PFS (token counts)
INPUT_LENGTH_BINS = [0, 64, 128, 256, 512, 1024, float("inf")]


def length_to_bucket(length: float) -> int:
    return min(int(length) // BUCKET_SIZE, MAX_BUCKETS - 1)


def input_length_bin(input_len: int) -> int:
    """Map input token count to bin index."""
    for i in range(len(INPUT_LENGTH_BINS) - 1):
        if input_len < INPUT_LENGTH_BINS[i + 1]:
            return i
    return len(INPUT_LENGTH_BINS) - 2


def build_historical_distributions(train_data: list) -> dict:
    """Build all historical distributions from training data.

    Returns dict with:
        global_lengths: {model: [lengths]}
        global_buckets: {model: [buckets]}
        binned_lengths: {model: {input_bin: [lengths]}}
        binned_buckets: {model: {input_bin: [buckets]}}
    """
    global_lengths = defaultdict(list)
    global_buckets = defaultdict(list)
    binned_lengths = defaultdict(lambda: defaultdict(list))
    binned_buckets = defaultdict(lambda: defaultdict(list))

    for req in train_data:
        input_len = req.get("input_len", len(req.get("prompt", "").split()))
        ibin = input_length_bin(input_len)

        for model in MODELS:
            m_data = req["models"].get(model, {})
            if not m_data:
                continue
            length = float(m_data.get("output_length", 0))
            bucket = length_to_bucket(length)

            global_lengths[model].append(length)
            global_buckets[model].append(bucket)
            binned_lengths[model][ibin].append(length)
            binned_buckets[model][ibin].append(bucket)

    return {
        "global_lengths": global_lengths,
        "global_buckets": global_buckets,
        "binned_lengths": binned_lengths,
        "binned_buckets": binned_buckets,
    }


def evaluate_baseline(test_data: list, distributions: dict, method: str) -> dict:
    """Evaluate a PFS-style baseline on test data.

    Methods:
        global_pfs: sample from global P(l) — predict median of distribution
        global_pfs_bucket: predict majority bucket from global distribution
        binned_pfs: sample from P(l | input_len_bin) — predict median per bin
        binned_pfs_bucket: predict majority bucket per input_len bin
        static_majority: always predict global majority bucket
        global_mean: always predict global mean length → bucket
    """
    results = {}

    for model in MODELS:
        gl = np.array(distributions["global_lengths"][model])
        gb = np.array(distributions["global_buckets"][model])
        bl = distributions["binned_lengths"][model]
        bb = distributions["binned_buckets"][model]

        # Precompute predictions per method
        if method == "static_majority":
            counts = np.bincount(gb.astype(int), minlength=MAX_BUCKETS)
            pred_bucket_global = int(np.argmax(counts))
        elif method == "global_mean":
            pred_len_global = float(np.mean(gl))
        elif method == "global_pfs":
            pred_len_global = float(np.median(gl))
        elif method == "global_pfs_bucket":
            counts = np.bincount(gb.astype(int), minlength=MAX_BUCKETS)
            pred_bucket_global = int(np.argmax(counts))

        # Per-bin precomputation for binned methods
        bin_medians = {}
        bin_majority_buckets = {}
        bin_bucket_distributions = {}
        for ibin in range(len(INPUT_LENGTH_BINS) - 1):
            if ibin in bl[model] and bl[model][ibin]:
                bin_medians[ibin] = float(np.median(bl[model][ibin]))
            else:
                bin_medians[ibin] = float(np.median(gl))  # fallback

            if ibin in bb[model] and bb[model][ibin]:
                bc = np.bincount(
                    np.array(bb[model][ibin], dtype=int), minlength=MAX_BUCKETS
                )
                bin_majority_buckets[ibin] = int(np.argmax(bc))
                bin_bucket_distributions[ibin] = bc / bc.sum()
            else:
                bc = np.bincount(gb.astype(int), minlength=MAX_BUCKETS)
                bin_majority_buckets[ibin] = int(np.argmax(bc))
                bin_bucket_distributions[ibin] = bc / bc.sum()

        # Evaluate on test
        correct = 0
        adjacent = 0
        total = 0
        mae_tokens_list = []
        mae_length_list = []
        actuals = []
        predictions = []

        for req in test_data:
            m_data = req["models"].get(model, {})
            if not m_data:
                continue

            actual_len = float(m_data.get("output_length", 0))
            actual_bucket = length_to_bucket(actual_len)
            input_len = req.get("input_len", len(req.get("prompt", "").split()))
            ibin = input_length_bin(input_len)

            # Predict
            if method == "static_majority":
                pred_bucket = pred_bucket_global
                pred_len = pred_bucket * BUCKET_SIZE + BUCKET_SIZE / 2
            elif method == "global_mean":
                pred_len = pred_len_global
                pred_bucket = length_to_bucket(pred_len)
            elif method == "global_pfs":
                pred_len = pred_len_global
                pred_bucket = length_to_bucket(pred_len)
            elif method == "global_pfs_bucket":
                pred_bucket = pred_bucket_global
                pred_len = pred_bucket * BUCKET_SIZE + BUCKET_SIZE / 2
            elif method == "binned_pfs":
                pred_len = bin_medians[ibin]
                pred_bucket = length_to_bucket(pred_len)
            elif method == "binned_pfs_bucket":
                pred_bucket = bin_majority_buckets[ibin]
                pred_len = pred_bucket * BUCKET_SIZE + BUCKET_SIZE / 2

            correct += pred_bucket == actual_bucket
            adjacent += abs(pred_bucket - actual_bucket) <= 1

            actual_tok = actual_bucket * BUCKET_SIZE + BUCKET_SIZE / 2
            pred_tok = pred_bucket * BUCKET_SIZE + BUCKET_SIZE / 2
            mae_tokens_list.append(abs(actual_tok - pred_tok))
            mae_length_list.append(abs(actual_len - pred_len))
            actuals.append(actual_len)
            predictions.append(pred_len)
            total += 1

        # Spearman (without scipy — use rank correlation)
        def spearman(x, y):
            n = len(x)
            if n < 3:
                return 0.0
            rx = np.argsort(np.argsort(x)).astype(float)
            ry = np.argsort(np.argsort(y)).astype(float)
            d = rx - ry
            return float(1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1)))

        short = model.split("/")[-1]
        results[model] = {
            "accuracy": correct / total,
            "adjacent_accuracy": adjacent / total,
            "mae_tokens": float(np.mean(mae_tokens_list)),
            "mae_length": float(np.mean(mae_length_list)),
            "median_ae_length": float(np.median(mae_length_list)),
            "spearman_r": spearman(actuals, predictions),
            "n": total,
        }
        logger.info(f"  {short} ({method}): acc={correct/total:.3f}, "
                     f"adj={adjacent/total:.3f}, mae_tok={np.mean(mae_tokens_list):.1f}, "
                     f"mae_len={np.mean(mae_length_list):.1f}")

    return results


def evaluate_online_pfs(data: list, window_size: int = 1000,
                        use_input_bins: bool = False) -> dict:
    """Simulate PFS online: sliding window, sequential processing.

    Each request is predicted using only previously completed requests
    in the sliding window. Simulates real serving conditions.

    Args:
        data: all requests (processed sequentially, as if arriving one by one)
        window_size: sliding window of recent completions
        use_input_bins: if True, use input-length-conditioned PFS
    """
    results = {}

    for model in MODELS:
        # Sliding window of completed request lengths
        window = []  # (output_length, input_len_bin)

        correct = 0
        adjacent = 0
        total = 0
        mae_tokens_list = []
        mae_length_list = []
        skipped_cold_start = 0

        for req in data:
            m_data = req["models"].get(model, {})
            if not m_data:
                continue

            actual_len = float(m_data.get("output_length", 0))
            actual_bucket = length_to_bucket(actual_len)
            input_len = req.get("input_len", len(req.get("prompt", "").split()))
            ibin = input_length_bin(input_len)

            # Predict from sliding window
            if len(window) < 10:
                # Cold start: not enough history, skip evaluation
                skipped_cold_start += 1
            else:
                if use_input_bins:
                    # Filter window to same input-length bin
                    bin_lengths = [l for l, b in window if b == ibin]
                    if len(bin_lengths) < 3:
                        # Fallback to global window
                        bin_lengths = [l for l, _ in window]
                    pred_len = float(np.median(bin_lengths))
                else:
                    all_lengths = [l for l, _ in window]
                    pred_len = float(np.median(all_lengths))

                pred_bucket = length_to_bucket(pred_len)

                correct += pred_bucket == actual_bucket
                adjacent += abs(pred_bucket - actual_bucket) <= 1
                actual_tok = actual_bucket * BUCKET_SIZE + BUCKET_SIZE / 2
                pred_tok = pred_bucket * BUCKET_SIZE + BUCKET_SIZE / 2
                mae_tokens_list.append(abs(actual_tok - pred_tok))
                mae_length_list.append(abs(actual_len - pred_len))
                total += 1

            # "Complete" this request: add to sliding window
            window.append((actual_len, ibin))
            if len(window) > window_size:
                window.pop(0)

        short = model.split("/")[-1]
        if total > 0:
            results[model] = {
                "accuracy": correct / total,
                "adjacent_accuracy": adjacent / total,
                "mae_tokens": float(np.mean(mae_tokens_list)),
                "mae_length": float(np.mean(mae_length_list)),
                "median_ae_length": float(np.median(mae_length_list)),
                "n": total,
                "skipped_cold_start": skipped_cold_start,
            }
            logger.info(f"  {short}: acc={correct/total:.3f}, adj={adjacent/total:.3f}, "
                         f"mae_tok={np.mean(mae_tokens_list):.1f}, cold_start_skip={skipped_cold_start}")
        else:
            results[model] = {"accuracy": 0, "n": 0}

    return results


def main():
    parser = argparse.ArgumentParser(description="PFS-style baselines for ROUTE_BALANCE")
    parser.add_argument("--train", required=True, help="Training data JSONL")
    parser.add_argument("--test", required=True, help="Test data JSONL")
    parser.add_argument("--output", default="models/route_balance/results/pfs_baselines.json")
    args = parser.parse_args()

    # Load data
    def load(path):
        with open(path) as f:
            return [json.loads(line) for line in f]

    train_data = load(args.train)
    test_data = load(args.test)
    logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Build distributions
    distributions = build_historical_distributions(train_data)

    # Log distribution stats
    for model in MODELS:
        short = model.split("/")[-1]
        gl = distributions["global_lengths"][model]
        logger.info(f"{short}: mean={np.mean(gl):.1f}, median={np.median(gl):.1f}, "
                     f"std={np.std(gl):.1f}, n={len(gl)}")
        for ibin in range(len(INPUT_LENGTH_BINS) - 1):
            bl = distributions["binned_lengths"][model].get(ibin, [])
            if bl:
                lo, hi = INPUT_LENGTH_BINS[ibin], INPUT_LENGTH_BINS[ibin + 1]
                hi_str = str(int(hi)) if hi != float("inf") else "+"
                logger.info(f"  bin [{int(lo)}-{hi_str}): n={len(bl)}, "
                             f"mean={np.mean(bl):.1f}, median={np.median(bl):.1f}")

    # Run all baselines
    methods = [
        "static_majority",
        "global_mean",
        "global_pfs",          # PFS: global median
        "global_pfs_bucket",   # PFS: global majority bucket (same as static)
        "binned_pfs",          # PFS + input-length binning: per-bin median
        "binned_pfs_bucket",   # PFS + input-length binning: per-bin majority bucket
    ]

    all_results = {}
    for method in methods:
        logger.info(f"\n=== {method} ===")
        all_results[method] = evaluate_baseline(test_data, distributions, method)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {args.output}")

    # Online PFS simulation (sequential, sliding window)
    # Use combined train+test as the full request stream
    all_data = train_data + test_data  # train serves as warm-up history

    for name, use_bins, win in [
        ("online_pfs_w1000", False, 1000),
        ("online_pfs_w100", False, 100),
        ("online_pfs_binned_w1000", True, 1000),
        ("online_pfs_binned_w100", True, 100),
    ]:
        logger.info(f"\n=== {name} (window={win}) ===")
        all_results[name] = evaluate_online_pfs(
            all_data, window_size=win, use_input_bins=use_bins,
        )

    # Re-save with online results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 100)
    print(f"{'Method':<28} {'3B':>16} {'7B':>16} {'14B':>16} {'72B':>16}")
    print(f"{'':28} {'Acc / MAE':>16} {'Acc / MAE':>16} {'Acc / MAE':>16} {'Acc / MAE':>16}")
    print("-" * 100)
    for method in list(all_results.keys()):
        row = f"{method:<28}"
        for model in MODELS:
            r = all_results[method].get(model, {})
            if r and r.get("n", 0) > 0:
                row += f" {r['accuracy']*100:5.1f}%/{r['mae_tokens']:5.0f} "
            else:
                row += f"{'—':>16} "
        print(row)
    print("=" * 100)


if __name__ == "__main__":
    main()
