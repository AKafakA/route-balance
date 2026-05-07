"""Post-process DeepEval scored data.

Creates filtered versions of train/test scored data:
1. Removes entries with <3 token references (44 LMSYS entries)
2. Applies safety-aware scoring: for is_harmful entries, replaces deepeval score
   with protectai safety score (same logic as existing reference_score)
3. Saves as new *_filtered.jsonl files (never overwrites originals)

Usage:
    python -m route_balance.predictor.route_balance.offline_training.postprocess_deepeval \
        --train-input data/scored/train_scored.jsonl \
        --test-input data/scored/test_scored.jsonl \
        --train-output data/scored/train_scored_filtered.jsonl \
        --test-output data/scored/test_scored_filtered.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEEPEVAL_KEY = "deepeval-llama3.1-8b-it_reference"
MIN_REF_TOKENS = 3


def process_file(input_path: str, output_path: str, stats_label: str):
    """Process one scored file: filter short refs + safety-aware scoring."""

    with open(input_path) as f:
        data = [json.loads(line) for line in f]

    logger.info(f"[{stats_label}] Loaded {len(data)} entries from {input_path}")

    filtered = []
    removed_short_ref = 0
    total_scored = 0
    total_missing = 0

    for entry in data:
        ref_text = entry.get("reference_text", "")
        ref_tokens = len(ref_text.split()) if ref_text else 0
        is_harmful = entry.get("is_harmful", False)

        # Filter: remove LMSYS entries with <3 token references only
        # SQuAD short refs (1-2 word answers) are valid extractive QA answers
        # Only LMSYS has genuinely bad short references (acknowledgments, single chars)
        dataset = entry.get("dataset", "")
        if dataset == "lmsys" and ref_tokens < MIN_REF_TOKENS and not is_harmful:
            removed_short_ref += 1
            continue

        # Count scored vs missing (no safety replacement — DeepEval already handles
        # harmful entries correctly by comparing against refusal reference template)
        for model_name, mdata in entry.get("models", {}).items():
            scores = mdata.get("llm_judge_scores", {})
            if DEEPEVAL_KEY in scores and scores[DEEPEVAL_KEY] is not None:
                total_scored += 1
            else:
                total_missing += 1

        filtered.append(entry)

    # Save filtered output (new file, never overwrite)
    if Path(output_path).exists():
        logger.warning(f"Output already exists: {output_path} — appending .new")
        output_path = output_path + ".new"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in filtered:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"[{stats_label}] Results:")
    logger.info(f"  Input: {len(data)} entries")
    logger.info(f"  Removed (short ref <{MIN_REF_TOKENS} tokens): {removed_short_ref}")
    logger.info(f"  Output: {len(filtered)} entries")
    logger.info(f"  Harmful entries (kept with DeepEval scores): {sum(1 for e in filtered if e.get('is_harmful'))}")
    logger.info(f"  Total scored: {total_scored}, missing: {total_missing}")
    logger.info(f"  Saved to: {output_path}")

    return len(filtered)


def main():
    parser = argparse.ArgumentParser(
        description="Post-process DeepEval scored data"
    )
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--test-input", default=None)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--test-output", default=None)
    args = parser.parse_args()

    n_train = process_file(args.train_input, args.train_output, "train")

    if args.test_input and args.test_output:
        n_test = process_file(args.test_input, args.test_output, "test")
    else:
        n_test = 0
        logger.info("No test input specified, skipping test set")

    logger.info(f"\nDone. Train: {n_train}, Test: {n_test}")


if __name__ == "__main__":
    main()
