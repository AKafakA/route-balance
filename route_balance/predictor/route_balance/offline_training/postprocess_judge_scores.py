#!/usr/bin/env python3
"""
Post-process judge scores: rename blind judge key + merge into raw data + output final files.

- Rename "Qwen_Qwen2.5-7B-Instruct" → "qwen2.5-7b-it_blind" in llm_judge_scores
- Merge scored train/test back into raw all-data JSON
- Output final train.jsonl, test.jsonl, all.json for HF upload

Usage:
    python -m route_balance.predictor.route_balance.offline_training.postprocess_judge_scores \
        --train-scored data/scored/train_scored.jsonl \
        --test-scored data/scored/test_scored.jsonl \
        --raw-data data/route_balance/training_data/route_balance_v3_all_training_fixed.json \
        --output-dir data/final
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BLIND_RENAMES = {
    "Qwen_Qwen2.5-7B-Instruct": "qwen2.5-7b-it_blind",
}


def rename_blind_keys(entry: dict) -> dict:
    """Rename blind judge keys in llm_judge_scores for all models."""
    for model_name, m_data in entry.get("models", {}).items():
        js = m_data.get("llm_judge_scores", {})
        for old_key, new_key in BLIND_RENAMES.items():
            if old_key in js:
                js[new_key] = js.pop(old_key)
    return entry


def main():
    parser = argparse.ArgumentParser(description="Post-process judge scores")
    parser.add_argument("--train-scored", required=True)
    parser.add_argument("--test-scored", required=True)
    parser.add_argument("--raw-data", required=True, help="Raw all-data JSON for merging")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scored data
    for split, path in [("train", args.train_scored), ("test", args.test_scored)]:
        logger.info(f"Processing {split}: {path}")
        with open(path) as f:
            data = [json.loads(line) for line in f]

        # Rename blind judge keys
        renamed = 0
        for entry in data:
            for model_name, m_data in entry.get("models", {}).items():
                js = m_data.get("llm_judge_scores", {})
                for old_key in BLIND_RENAMES:
                    if old_key in js:
                        renamed += 1
            rename_blind_keys(entry)
        logger.info(f"  Renamed {renamed} blind judge keys")

        # Check new judge coverage
        judge_keys = set()
        for entry in data:
            for m_data in entry["models"].values():
                judge_keys.update(m_data.get("llm_judge_scores", {}).keys())
        logger.info(f"  Judge keys: {sorted(judge_keys)}")

        # Save
        out_path = output_dir / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"  Saved to {out_path}")

    # Merge into raw all-data JSON
    logger.info(f"Merging into raw data: {args.raw_data}")
    with open(args.raw_data) as f:
        raw = json.load(f)

    # Build lookup from scored data
    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    scored_lookup = {}
    for path in [train_path, test_path]:
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                scored_lookup[entry["request_id"]] = entry

    # Merge new fields into raw
    merged = 0
    for req in raw.get("requests", []):
        rid = req.get("request_id")
        if rid in scored_lookup:
            scored = scored_lookup[rid]
            # Copy over new fields
            if "reference_text" in scored:
                req["reference_text"] = scored["reference_text"]
            for model_name in req.get("models", {}):
                if model_name in scored.get("models", {}):
                    scored_m = scored["models"][model_name]
                    # Merge llm_judge_scores
                    req["models"][model_name]["llm_judge_scores"] = scored_m.get(
                        "llm_judge_scores", req["models"][model_name].get("llm_judge_scores", {})
                    )
                    # Copy reference_score/similarity if updated
                    for field in ["reference_score", "reference_similarity"]:
                        if field in scored_m:
                            req["models"][model_name][field] = scored_m[field]
            merged += 1

    logger.info(f"  Merged {merged}/{len(raw.get('requests', []))} entries")

    all_path = output_dir / "route_balance_v3_all_training_final.json"
    with open(all_path, "w") as f:
        json.dump(raw, f, ensure_ascii=False)
    logger.info(f"  Saved to {all_path}")

    logger.info("Post-processing complete")


if __name__ == "__main__":
    main()
