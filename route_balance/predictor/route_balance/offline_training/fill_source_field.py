#!/usr/bin/env python3
"""
Fill missing 'source' field in training/test JSONL files.

The source field maps each entry to its original HuggingFace dataset row ID
(e.g., "gsm8k/4222", "squad/77478", "lmsys/abc123").

During initial data collection, only harmful datasets (beaver_tails, reward_bench
safety subsets) had source populated. This script backfills the source for all
entries by matching request_id against the raw data which has full source coverage.

Usage:
    python3 -m route_balance.predictor.route_balance.offline_training.fill_source_field \
        --raw-data data/route_balance/training_data/route_balance_v3_all_training_fixed.json \
        --input data/route_balance/training_data_with_ref/train_fixed.jsonl \
        --output data/route_balance/training_data_with_ref/train_fixed.jsonl
"""

import argparse
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fill missing source field")
    parser.add_argument("--raw-data", required=True,
                        help="Raw JSON with full source mapping")
    parser.add_argument("--input", required=True, help="Input JSONL to fix")
    parser.add_argument("--output", required=True, help="Output JSONL (can be same as input)")
    args = parser.parse_args()

    # Build request_id → source mapping
    with open(args.raw_data) as f:
        raw = json.load(f)
    rid_to_source = {r["request_id"]: r["source"] for r in raw["requests"]}
    logger.info(f"Source mapping: {len(rid_to_source)} entries")

    # Load and fix
    with open(args.input) as f:
        data = [json.loads(l) for l in f]

    filled = 0
    already = 0
    missing = 0
    for entry in data:
        if entry.get("source"):
            already += 1
        else:
            source = rid_to_source.get(entry["request_id"])
            if source:
                entry["source"] = source
                filled += 1
            else:
                missing += 1

    # Save
    with open(args.output, "w") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Done: {filled} filled, {already} already had source, {missing} no mapping found")


if __name__ == "__main__":
    main()
