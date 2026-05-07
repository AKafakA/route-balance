"""Metrics computation for ROUTE_BALANCE predictor evaluation."""

from typing import Dict, List

import numpy as np
from scipy import stats


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics: MAE, MAPE, Spearman rho, Acc@thresholds."""
    errors = np.abs(actual - predicted)
    non_zero = actual > 0

    rho = float(stats.spearmanr(actual, predicted).correlation) if len(actual) > 2 else 0.0

    return {
        "mae": float(np.mean(errors)),
        "median_ae": float(np.median(errors)),
        "mape": float(np.mean(errors[non_zero] / actual[non_zero]) * 100) if non_zero.any() else 0.0,
        "acc_50": float(np.mean(errors <= 50)),
        "acc_100": float(np.mean(errors <= 100)),
        "spearman_r": rho if not np.isnan(rho) else 0.0,
        "n": len(actual),
    }


def bucket_classification_metrics(
    actual_lengths: np.ndarray, predicted_probs: np.ndarray, bucket_size: int = 64
) -> Dict[str, float]:
    """Compute bucket classification metrics from probability distributions.

    Args:
        actual_lengths: (N,) actual output lengths in tokens
        predicted_probs: (N, num_buckets) softmax probabilities
        bucket_size: tokens per bucket
    """
    num_buckets = predicted_probs.shape[1]
    actual_buckets = np.clip(actual_lengths.astype(int) // bucket_size, 0, num_buckets - 1)
    predicted_buckets = np.argmax(predicted_probs, axis=-1)

    accuracy = float(np.mean(predicted_buckets == actual_buckets))
    adjacent = float(np.mean(np.abs(predicted_buckets - actual_buckets) <= 1))

    # Top-3 accuracy
    top3 = np.argsort(predicted_probs, axis=-1)[:, -3:]
    top3_hit = float(np.mean([actual_buckets[i] in top3[i] for i in range(len(actual_buckets))]))

    # Expected length from distribution
    midpoints = np.array([(i * bucket_size + bucket_size / 2) for i in range(num_buckets)])
    expected_lengths = np.sum(predicted_probs * midpoints, axis=-1)
    reg_metrics = regression_metrics(actual_lengths, expected_lengths)

    # P(actual bucket)
    prob_at_actual = np.array([predicted_probs[i, actual_buckets[i]] for i in range(len(actual_buckets))])
    nll = -float(np.mean(np.log(np.clip(prob_at_actual, 1e-10, 1.0))))

    return {
        "accuracy": accuracy,
        "adjacent_accuracy": adjacent,
        "top3_accuracy": top3_hit,
        "mean_prob_actual": float(np.mean(prob_at_actual)),
        "nll": nll,
        # Also include regression metrics from E[length]
        "expected_length_mae": reg_metrics["mae"],
        "expected_length_mape": reg_metrics["mape"],
        "expected_length_spearman_r": reg_metrics["spearman_r"],
        "n": len(actual_lengths),
    }


def bucket_filtering_metrics(
    actual_lengths_per_model: Dict[str, np.ndarray],
    predicted_probs_per_model: Dict[str, np.ndarray],
    bucket_size: int = 64,
    thresholds: List[float] = None,
) -> Dict[str, Dict]:
    """Compute budget filtering metrics with strict/balanced/relaxed modes.

    Budget modes derived from per-prompt actual lengths:
      strict: budget = min(actual) across models → 1 model guaranteed
      balanced: budget = median(actual) across models → ~2 models pass
      relaxed: budget = 2nd largest(actual) → only longest filtered

    Args:
        actual_lengths_per_model: {model: (N,) array}
        predicted_probs_per_model: {model: (N, num_buckets) array}
    """
    if thresholds is None:
        thresholds = [0.5, 0.7, 0.9]

    models = sorted(actual_lengths_per_model.keys())
    n = len(next(iter(actual_lengths_per_model.values())))
    num_buckets = next(iter(predicted_probs_per_model.values())).shape[1]

    # Stack all model lengths: (N, num_models)
    all_lens = np.stack([actual_lengths_per_model[m] for m in models], axis=1)
    sorted_lens = np.sort(all_lens, axis=1)

    budget_modes = {
        "strict": sorted_lens[:, 0],
        "balanced": np.median(all_lens, axis=1),
        "relaxed": sorted_lens[:, -2] if len(models) > 1 else sorted_lens[:, 0],
    }

    results = {}
    for mode, budgets in budget_modes.items():
        for threshold in thresholds:
            total_tp = total_fa = total_fr = total_tn = 0
            passed_counts = []
            should_counts = []

            for i in range(n):
                budget = budgets[i]
                if budget <= 0:
                    continue
                max_bucket = min(int(budget) // bucket_size, num_buckets - 1)

                passed = 0
                should = 0
                for m in models:
                    cdf = predicted_probs_per_model[m][i, :max_bucket + 1].sum()
                    within = actual_lengths_per_model[m][i] <= budget

                    if cdf >= threshold and within:
                        total_tp += 1
                    elif cdf >= threshold and not within:
                        total_fa += 1
                    elif cdf < threshold and within:
                        total_fr += 1
                    else:
                        total_tn += 1

                    if cdf >= threshold:
                        passed += 1
                    if within:
                        should += 1

                passed_counts.append(passed)
                should_counts.append(should)

            n_passed = total_tp + total_fa
            n_within = total_tp + total_fr

            results[f"{mode}_t{threshold}"] = {
                "mode": mode,
                "threshold": threshold,
                "avg_budget": float(budgets[budgets > 0].mean()),
                "compliance": total_tp / max(n_passed, 1),
                "false_accept": total_fa / max(n_passed, 1),
                "false_reject": total_fr / max(n_within, 1),
                "avg_models_passed": float(np.mean(passed_counts)),
                "avg_models_should_pass": float(np.mean(should_counts)),
            }

    return results
