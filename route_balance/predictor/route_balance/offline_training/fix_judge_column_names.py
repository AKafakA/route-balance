#!/usr/bin/env python3
"""
Fix judge score column names for consistency.

Renames judge score keys in llm_judge_scores dict:
  Old: "Qwen_Qwen2.5-7B-Instruct" → New: "qwen2.5-7b-it_blind"

Also merges prometheus_score/prometheus_raw fields into llm_judge_scores:
  prometheus_score → llm_judge_scores["qwen2.5-7b-it_reference"]

Usage:
    # Fix column names in a JSONL file
    python -m route_balance.predictor.route_balance.offline_training.fix_judge_column_names \
        --input data/test_with_prometheus.jsonl \
        --output data/test_fixed_names.jsonl

    # Also rename blind judge key
    python -m route_balance.predictor.route_balance.offline_training.fix_judge_column_names \
        --input data/test_with_prometheus.jsonl \
        --output data/test_fixed_names.jsonl \
        --rename-blind
"""

import argparse
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Mapping of old keys to new keys
BLIND_JUDGE_RENAMES = {
    "Qwen_Qwen2.5-7B-Instruct": "qwen2.5-7b-it_blind",
}


def fix_entry(entry: dict, rename_blind: bool = False) -> dict:
    """Fix judge score column names in a single entry."""
    for model_name, m_data in entry.get("models", {}).items():
        judge_scores = m_data.get("llm_judge_scores", {})

        # Rename blind judge keys
        if rename_blind:
            for old_key, new_key in BLIND_JUDGE_RENAMES.items():
                if old_key in judge_scores:
                    judge_scores[new_key] = judge_scores.pop(old_key)

        # Merge prometheus_score into llm_judge_scores
        if "prometheus_score" in m_data:
            judge_scores["qwen2.5-7b-it_reference"] = m_data.pop("prometheus_score")
        if "prometheus_raw" in m_data:
            m_data.pop("prometheus_raw")  # Remove raw, only keep normalized

        m_data["llm_judge_scores"] = judge_scores

    return entry


def main():
    parser = argparse.ArgumentParser(description="Fix judge score column names")
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--rename-blind", action="store_true",
                        help="Also rename blind judge keys")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print first entry only, don't write")
    args = parser.parse_args()

    with open(args.input) as f:
        data = [json.loads(line) for line in f]

    logger.info(f"Loaded {len(data)} entries from {args.input}")

    # Check current keys
    first_model = list(data[0]["models"].values())[0]
    old_keys = list(first_model.get("llm_judge_scores", {}).keys())
    has_prometheus = "prometheus_score" in first_model
    logger.info(f"Current judge keys: {old_keys}")
    logger.info(f"Has prometheus_score field: {has_prometheus}")

    for entry in data:
        fix_entry(entry, rename_blind=args.rename_blind)

    # Show result
    first_model_fixed = list(data[0]["models"].values())[0]
    new_keys = list(first_model_fixed.get("llm_judge_scores", {}).keys())
    logger.info(f"New judge keys: {new_keys}")

    if args.dry_run:
        print(json.dumps(data[0], indent=2, ensure_ascii=False)[:2000])
        return

    with open(args.output, "w") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(data)} entries to {args.output}")


if __name__ == "__main__":
    main()
