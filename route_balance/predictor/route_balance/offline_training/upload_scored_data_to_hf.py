#!/usr/bin/env python3
"""
Upload scored training data to HuggingFace.

Uploads train.jsonl and test.jsonl with updated fields:
  - reference_text (dataset ground truth)
  - llm_judge_scores.deepeval-llama3.1-8b-it_reference (G-Eval 0-10)
  - llm_judge_scores.qwen2.5-7b-it_blind (renamed from Qwen_Qwen2.5-7B-Instruct)

Usage:
    python -m route_balance.predictor.route_balance.offline_training.upload_scored_data_to_hf \
        --train data/final/train.jsonl \
        --test data/final/test.jsonl \
        --repo asdwb/route_balance_model_estimator \
        --token $HF_TOKEN
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Upload scored data to HF")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--repo", default="asdwb/route_balance_model_estimator")
    parser.add_argument("--token", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Verify data
    for split, path in [("train", args.train), ("test", args.test)]:
        with open(path) as f:
            data = [json.loads(line) for line in f]
        logger.info(f"{split}: {len(data)} entries")

        # Check fields
        first_model = list(data[0]["models"].values())[0]
        js_keys = sorted(first_model.get("llm_judge_scores", {}).keys())
        has_ref = "reference_text" in data[0]
        logger.info(f"  judge keys: {js_keys}")
        logger.info(f"  has reference_text: {has_ref}")

    if args.dry_run:
        logger.info("Dry run — not uploading")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)

    for split, path in [("train", args.train), ("test", args.test)]:
        logger.info(f"Uploading {split}: {path}")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"{split}.jsonl",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=f"Update {split}: add deepeval-llama3.1-8b-it_reference scores, reference_text, rename blind judge",
        )

    logger.info(f"Upload complete: {args.repo}")


if __name__ == "__main__":
    main()
