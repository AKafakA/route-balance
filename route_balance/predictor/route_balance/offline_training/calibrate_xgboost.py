#!/usr/bin/env python3
"""
Calibrate XGBoost latency predictors for SLO confidence estimation.

Computes prediction residuals on validation set per instance type,
then stores empirical distributions for:
  - QLM filter: Normal(mean, std) of residuals
  - RouteBalance CDF filter: sorted residuals for percentile lookup

Usage:
    python -m route_balance.predictor.route_balance.offline_training.calibrate_xgboost \
        --test-data data/route_balance/latency_data/all/latency_test_tagged.jsonl \
        --xgboost-dir models/route_balance/latency/deploy \
        --output models/route_balance/latency/calibration.json
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate XGBoost for SLO confidence estimation"
    )
    parser.add_argument("--test-data", required=True, help="Validation latency JSONL")
    parser.add_argument("--xgboost-dir", required=True, help="XGBoost model directory")
    parser.add_argument("--output", required=True, help="Output calibration JSON")
    args = parser.parse_args()

    from route_balance.predictor.route_balance.estimators.xgboost_predictor import (
        XGBoostLatencyPredictor, build_feature_vector,
    )

    # Load XGBoost models
    predictor = XGBoostLatencyPredictor.load(args.xgboost_dir)
    logger.info(f"Loaded XGBoost models: {list(predictor.models.keys())}")

    # Load validation data
    records_by_type = defaultdict(list)
    with open(args.test_data) as f:
        for line in f:
            rec = json.loads(line)
            inst_type = rec.get("instance_type", "unknown")
            records_by_type[inst_type].append(rec)

    logger.info(f"Validation data: {sum(len(v) for v in records_by_type.values())} records, "
                f"{len(records_by_type)} instance types")

    # Compute residuals per instance type
    calibration = {}
    for inst_type, records in sorted(records_by_type.items()):
        if inst_type not in predictor.models:
            logger.warning(f"No model for {inst_type}, skipping")
            continue

        ttft_residuals = []
        tpot_residuals = []
        e2e_residuals = []

        for rec in records:
            schedule_state = rec.get("schedule_state", {})
            if not schedule_state:
                continue

            prompt_tokens = rec.get("num_prompt_tokens", 0)
            output_tokens = rec.get("num_predicted_output_tokens", 0)
            actual_ttft = rec.get("actual_ttft")
            actual_tpot = rec.get("actual_tpot")
            actual_e2e = rec.get("actual_e2e_latency")

            if not all([actual_ttft, actual_tpot, actual_e2e]):
                continue

            fv = build_feature_vector(schedule_state, int(prompt_tokens), int(output_tokens))
            pred = predictor.predict(inst_type, fv)

            pred_ttft = pred.get("ttft", pred.get("e2e_latency", 0))
            pred_tpot = pred.get("tpot", 0)
            pred_e2e = pred.get("e2e_latency", 0)

            # Residual = actual - predicted (positive means under-predicted)
            if pred_ttft > 0:
                ttft_residuals.append((actual_ttft - pred_ttft) * 1000)  # s → ms
            if pred_tpot > 0:
                tpot_residuals.append((actual_tpot - pred_tpot) * 1000)
            if pred_e2e > 0:
                e2e_residuals.append((actual_e2e - pred_e2e) * 1000)

        if not ttft_residuals:
            logger.warning(f"No valid predictions for {inst_type}")
            continue

        ttft_arr = np.array(ttft_residuals)
        tpot_arr = np.array(tpot_residuals)
        e2e_arr = np.array(e2e_residuals)

        cal = {
            "instance_type": inst_type,
            "n_samples": len(ttft_residuals),
            # For QLM: Normal parameters
            "ttft_residual_mean_ms": float(np.mean(ttft_arr)),
            "ttft_residual_std_ms": float(np.std(ttft_arr)),
            "tpot_residual_mean_ms": float(np.mean(tpot_arr)),
            "tpot_residual_std_ms": float(np.std(tpot_arr)),
            "e2e_residual_mean_ms": float(np.mean(e2e_arr)),
            "e2e_residual_std_ms": float(np.std(e2e_arr)),
            # For RouteBalance CDF: percentiles
            "ttft_residual_percentiles_ms": {
                str(p): float(np.percentile(ttft_arr, p))
                for p in [50, 75, 80, 85, 90, 95, 99]
            },
            "tpot_residual_percentiles_ms": {
                str(p): float(np.percentile(tpot_arr, p))
                for p in [50, 75, 80, 85, 90, 95, 99]
            },
            # For RouteBalance CDF: full sorted residuals (for exact percentile lookup)
            "ttft_residuals_sorted_ms": sorted(ttft_arr.tolist()),
            "tpot_residuals_sorted_ms": sorted(tpot_arr.tolist()),
        }

        calibration[inst_type] = cal
        logger.info(
            f"  {inst_type}: n={len(ttft_residuals)}, "
            f"TTFT residual mean={np.mean(ttft_arr):.1f}ms std={np.std(ttft_arr):.1f}ms, "
            f"TPOT residual mean={np.mean(tpot_arr):.1f}ms std={np.std(tpot_arr):.1f}ms"
        )

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(calibration, f, indent=2)
    logger.info(f"Calibration saved to {args.output}")


if __name__ == "__main__":
    main()
