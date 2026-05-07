#!/usr/bin/env python3
"""
Calibrate roofline (analytical) latency predictor for ROUTE_BALANCE.

Reads latency benchmark data and computes optimal rates per instance type
using the RooflineLatencyPredictor.calibrate() method.

Usage:
    python -m route_balance.predictor.route_balance.offline_training.train_roofline \
        --data-dir data/route_balance/latency_data/all/ \
        --output-dir models/route_balance/roofline/
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from route_balance.predictor.route_balance.estimators.roofline_predictor import RooflineLatencyPredictor
from route_balance.predictor.route_balance.offline_training.train_xgboost import load_latency_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


TARGET_ALIASES = {
    "e2el": ["e2el", "actual_e2e_latency"],
    "ttft": ["ttft", "actual_ttft"],
    "tpot": ["tpot", "actual_tpot"],
}


def evaluate_roofline(predictor: RooflineLatencyPredictor, records: list,
                      instance_type: str, target: str = "e2el") -> dict:
    """Evaluate calibrated roofline model on records."""
    errors = []
    for rec in records:
        target_val = None
        for field in TARGET_ALIASES.get(target, [target]):
            target_val = rec.get(field)
            if target_val is not None:
                break
        if target_val is None or target_val <= 0:
            continue

        schedule_state = rec.get("schedule_state", {})
        if not schedule_state:
            schedule_state = {k: rec.get(k, 0) for k in [
                "pending_prefill_tokens", "pending_decode_tokens",
                "ema_prefill_tok_per_s", "ema_decode_tok_per_s",
            ]}

        num_prompt = rec.get("num_prompt_tokens") or rec.get("input_len", 0)
        num_output = rec.get("output_len", 0) or rec.get("num_predicted_output_tokens", 0)

        pred = predictor.predict(instance_type, schedule_state, int(num_prompt), int(num_output))
        errors.append(abs(target_val - pred["e2e_latency"]))

    if not errors:
        return {"mae": float("inf"), "n": 0}

    errors = np.array(errors)
    return {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "p50_error": float(np.percentile(errors, 50)),
        "p95_error": float(np.percentile(errors, 95)),
        "n": len(errors),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate roofline latency predictor for ROUTE_BALANCE"
    )
    parser.add_argument("--data-dir", required=True, help="Latency data directory")
    parser.add_argument("--instance-types", nargs="+", default=None)
    parser.add_argument("--target", default="e2el")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--output-dir", default="models/route_balance/roofline")
    args = parser.parse_args()

    records_by_type = load_latency_data(args.data_dir, args.instance_types)
    if not records_by_type:
        logger.error("No latency data found!")
        return

    predictor = RooflineLatencyPredictor(rates={})
    all_metrics = {}

    for inst_type, records in sorted(records_by_type.items()):
        logger.info(f"Calibrating roofline for {inst_type} ({len(records)} records)")

        if len(records) < 50:
            logger.warning(f"Too few records for {inst_type}, skipping")
            continue

        # Split for evaluation
        n_val = int(len(records) * args.val_split)
        indices = np.random.RandomState(42).permutation(len(records))
        train_records = [records[i] for i in indices[: len(records) - n_val]]
        val_records = [records[i] for i in indices[len(records) - n_val :]]

        predictor.calibrate(inst_type, train_records, target=args.target)
        metrics = evaluate_roofline(predictor, val_records, inst_type, args.target)
        all_metrics[inst_type] = metrics

        rates = predictor.rates.get(inst_type, (0, 0, 0))
        logger.info(
            f"  {inst_type}: prefill={rates[0]:.0f} tok/s, decode={rates[1]:.0f} tok/s, "
            f"MAE={metrics['mae']:.4f}s, n_val={metrics['n']}"
        )

    predictor.save(args.output_dir)

    output_path = Path(args.output_dir)
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("ROOFLINE LATENCY PREDICTOR RESULTS")
    print("=" * 70)
    for inst_type, m in sorted(all_metrics.items()):
        print(f"  {inst_type}: MAE={m['mae']:.4f}s, RMSE={m.get('rmse', 0):.4f}s, n={m['n']}")


if __name__ == "__main__":
    main()
