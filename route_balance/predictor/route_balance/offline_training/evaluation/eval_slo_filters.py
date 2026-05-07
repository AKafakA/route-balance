#!/usr/bin/env python3
"""
Evaluate SLO filters using XGBoost calibration residuals.

Compares SLOs-Serve (point prediction), QLM (Normal confidence bounds),
and RouteBalance CDF (empirical residuals) at multiple SLO thresholds.

Usage:
    python -m route_balance.predictor.route_balance.offline_training.evaluation.eval_slo_filters \
        --test-data data/route_balance/latency_data/enriched/latency_test_tagged_enriched.jsonl \
        --xgboost-dir models/route_balance/xgboost_enriched/ \
        --output eval_results/20260410/slo_filter_results.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SLO filters")
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--xgboost-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-output", default=None,
                        help="Also save calibration stats")
    args = parser.parse_args()

    from route_balance.predictor.route_balance.estimators.xgboost_predictor import (
        XGBoostLatencyPredictor, build_feature_vector,
    )

    logger.info("Loading XGBoost models from %s", args.xgboost_dir)
    xgb = XGBoostLatencyPredictor.load(args.xgboost_dir)

    logger.info("Loading test data from %s", args.test_data)
    # Load directly from file (not directory)
    test_by_type = defaultdict(list)
    with open(args.test_data) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            inst_type = rec.get("instance_type", "unknown")
            test_by_type[inst_type].append(rec)
    test_by_type = dict(test_by_type)
    for inst_type, recs in test_by_type.items():
        logger.info("  %s: %d records", inst_type, len(recs))

    # Step 1: Generate calibration residuals
    logger.info("=== Generating XGBoost calibration residuals ===")
    calibration = {}

    for inst_type, records in sorted(test_by_type.items()):
        actuals = []
        preds = []
        for rec in records:
            actual = rec.get("actual_e2e_latency", 0)
            if actual <= 0:
                continue
            ss = rec.get("schedule_state", {})
            if not ss:
                continue
            prompt = int(rec.get("num_prompt_tokens", 0))
            output = int(rec.get("actual_output_tokens") or rec.get("num_predicted_output_tokens", 0))
            if output <= 0:
                continue
            try:
                result = xgb.predict(inst_type, ss, prompt, output)
                pred_e2e = result.get("e2e_latency", 0)
                if pred_e2e > 0:
                    actuals.append(actual)
                    preds.append(pred_e2e)
            except Exception:
                continue

        if actuals:
            residuals = np.array(actuals) - np.array(preds)
            calibration[inst_type] = {
                "residuals": residuals,
                "mean": float(residuals.mean()),
                "std": float(residuals.std()),
                "n": len(residuals),
                "mae": float(np.abs(residuals).mean()),
            }
            logger.info("  %s: n=%d, residual mean=%.4fs, std=%.4fs, mae=%.4fs",
                        inst_type, len(residuals), residuals.mean(), residuals.std(),
                        np.abs(residuals).mean())

    # Save calibration stats
    if args.calibration_output:
        Path(args.calibration_output).parent.mkdir(parents=True, exist_ok=True)
        cal_stats = {k: {kk: vv for kk, vv in v.items() if kk != "residuals"}
                     for k, v in calibration.items()}
        with open(args.calibration_output, "w") as f:
            json.dump(cal_stats, f, indent=2)
        logger.info("Calibration saved to %s", args.calibration_output)

    # Step 2: Evaluate SLO filters
    logger.info("\n=== Evaluating SLO filters ===")

    SLO_CONFIGS = {
        "e2e_slo_s": [5.0, 10.0, 20.0, 30.0],
        "ttft_slo_s": [0.5, 1.0, 2.0, 5.0],
    }

    filter_results = {}

    for slo_type, thresholds in SLO_CONFIGS.items():
        for thresh in thresholds:
            key = f"{slo_type}={thresh}"
            filter_results[key] = {}

            for inst_type, cal in calibration.items():
                residuals = cal["residuals"]
                std = cal["std"]
                n = len(residuals)
                if n == 0:
                    continue

                test_recs = test_by_type[inst_type]
                results_per_filter = {
                    "slos_serve": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                    "qlm_95": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                    "route_balance_cdf_90": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                }

                for rec in test_recs:
                    if "e2e" in slo_type:
                        actual = rec.get("actual_e2e_latency", 0)
                    else:
                        actual = rec.get("actual_ttft", 0)
                    if actual <= 0:
                        continue

                    ss = rec.get("schedule_state", {})
                    prompt = int(rec.get("num_prompt_tokens", 0))
                    output = int(rec.get("actual_output_tokens") or
                                 rec.get("num_predicted_output_tokens", 0))
                    if output <= 0:
                        continue

                    try:
                        result = xgb.predict(inst_type, ss, prompt, output)
                        if "e2e" in slo_type:
                            predicted = result.get("e2e_latency", 0)
                        else:
                            predicted = result.get("ttft", 0)
                    except Exception:
                        continue

                    meets_slo = actual <= thresh

                    # SLOs-Serve: accept if predicted <= SLO
                    slos_accept = predicted <= thresh
                    # QLM: accept if predicted + 1.96*std <= SLO
                    qlm_accept = (predicted + 1.96 * std) <= thresh
                    # RouteBalanceCDF: accept if P(residual <= SLO - predicted) >= 0.90
                    route_balance_accept = float(np.mean(residuals <= (thresh - predicted))) >= 0.90

                    for fname, accepted in [("slos_serve", slos_accept),
                                            ("qlm_95", qlm_accept),
                                            ("route_balance_cdf_90", route_balance_accept)]:
                        if accepted and meets_slo:
                            results_per_filter[fname]["tp"] += 1
                        elif accepted and not meets_slo:
                            results_per_filter[fname]["fp"] += 1
                        elif not accepted and meets_slo:
                            results_per_filter[fname]["fn"] += 1
                        else:
                            results_per_filter[fname]["tn"] += 1

                for fname, counts in results_per_filter.items():
                    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
                    precision = tp / max(tp + fp, 1)
                    recall = tp / max(tp + fn, 1)
                    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)

                    if inst_type not in filter_results[key]:
                        filter_results[key][inst_type] = {}
                    filter_results[key][inst_type][fname] = {
                        "precision": round(precision, 4),
                        "recall": round(recall, 4),
                        "f1": round(f1, 4),
                        "accuracy": round(accuracy, 4),
                        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    }

            # Print summary
            logger.info("\n  %s:", key)
            for inst_type in sorted(filter_results[key]):
                for fname, m in filter_results[key][inst_type].items():
                    logger.info("    %s/%s: F1=%.3f P=%.3f R=%.3f Acc=%.3f",
                                inst_type, fname, m["f1"], m["precision"],
                                m["recall"], m["accuracy"])

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(filter_results, f, indent=2)
    logger.info("\nResults saved to %s", args.output)


if __name__ == "__main__":
    main()
