#!/usr/bin/env python3
"""
Add reference_text field to training/test data for reference-grounded judging.

Loads actual reference answers from HuggingFace datasets and stores them
as a top-level `reference_text` field (one per prompt, shared across models).

For Prometheus/reference-grounded LLM judges, the reference text is the
ground truth answer — NOT another model's response.

Reference sources:
  gsm8k: Full step-by-step solution (answer field, including ####)
  squad: First answer span
  beaver_tails: Standard refusal template (harmful prompts)
  code_ultra_feedback: Highest-rated response
  reward_bench: Human-chosen response
  mix_instruct: Dataset output field
  lmsys: Original assistant response from conversation

Usage:
    python -m route_balance.predictor.route_balance.offline_training.add_reference_text \
        --input data/route_balance/training_data_with_ref/test_fixed.jsonl \
        --raw-data data/route_balance/training_data/route_balance_v3_all_training_fixed.json \
        --output data/route_balance/training_data_with_ref/test_with_reftext.jsonl \
        --device cpu
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Standard refusal for harmful prompts (beaver_tails, reward_bench safety subsets)
REFUSAL_TEMPLATE = (
    "I'm sorry, but I can't help with that request. "
    "It involves potentially harmful or dangerous content. "
    "If you have other questions I can help with, please feel free to ask."
)


def load_references_from_hf(sources: List[str]) -> Dict[str, str]:
    """Load full reference texts from HuggingFace datasets.

    Unlike add_reference_similarity.py which extracts just numbers/scores,
    this loads the FULL reference text for use by LLM judges.
    """
    from datasets import load_dataset

    by_dataset = defaultdict(list)
    for s in sources:
        parts = s.split("/", 1)
        if len(parts) == 2:
            by_dataset[parts[0]].append((s, parts[1]))

    references = {}

    # GSM8K: FULL step-by-step answer (not just final number)
    if "gsm8k" in by_dataset:
        logger.info(f"Loading GSM8K references ({len(by_dataset['gsm8k'])} entries)...")
        ds = load_dataset("openai/gsm8k", "main", split="train")
        for source_id, idx_str in by_dataset["gsm8k"]:
            idx = int(idx_str)
            if idx < len(ds):
                references[source_id] = ds[idx].get("answer", "")
        logger.info(f"  Loaded {sum(1 for s in references if s.startswith('gsm8k'))} GSM8K")

    # SQuAD: answer spans
    if "squad" in by_dataset:
        logger.info(f"Loading SQuAD references ({len(by_dataset['squad'])} entries)...")
        ds = load_dataset("rajpurkar/squad", split="train")
        for source_id, idx_str in by_dataset["squad"]:
            idx = int(idx_str)
            if idx < len(ds):
                answers = ds[idx].get("answers", {})
                texts = answers.get("text", [])
                if texts:
                    references[source_id] = texts[0]
        logger.info(f"  Loaded {sum(1 for s in references if s.startswith('squad'))} SQuAD")

    # LMSYS: original assistant response
    if "lmsys" in by_dataset:
        logger.info(f"Loading LMSYS references ({len(by_dataset['lmsys'])} entries)...")
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            all_files = list_repo_files("lmsys/lmsys-chat-1m", repo_type="dataset")
            parquet_files = [f for f in all_files if f.endswith(".parquet")]

            import pandas as pd
            conv_ids_needed = {idx_str for _, idx_str in by_dataset["lmsys"]}

            for pf in parquet_files:
                local = hf_hub_download(
                    repo_id="lmsys/lmsys-chat-1m", filename=pf, repo_type="dataset"
                )
                df = pd.read_parquet(local)
                for _, row in df.iterrows():
                    cid = str(row.get("conversation_id", ""))
                    if cid in conv_ids_needed:
                        convs = row.get("conversation", [])
                        for msg in convs:
                            if isinstance(msg, dict) and msg.get("role") == "assistant":
                                references[f"lmsys/{cid}"] = msg.get("content", "")
                                break
                        conv_ids_needed.discard(cid)
                if not conv_ids_needed:
                    break
            logger.info(f"  Loaded {sum(1 for s in references if s.startswith('lmsys/'))} LMSYS")
        except Exception as e:
            logger.warning(f"  Could not load LMSYS: {e}")

    # reward_bench: chosen response
    rb_subsets = [
        "xstest-should-respond", "xstest-should-refuse", "donotanswer",
        "refusals-offensive", "refusals-dangerous",
        "hep-python", "hep-go", "hep-cpp", "hep-java", "hep-js", "hep-rust"
    ]
    if any(ds_name in by_dataset for ds_name in rb_subsets):
        logger.info("Loading reward_bench references...")
        try:
            ds = load_dataset("allenai/reward-bench", split="filtered")
            rb_index = {}
            for row in ds:
                key = f"{row['subset']}/{row['id']}"
                rb_index[key] = row.get("chosen", "")
            for ds_name in by_dataset:
                if ds_name.startswith(("xstest", "donotanswer", "refusals", "hep-")):
                    for source_id, idx_str in by_dataset[ds_name]:
                        ref = rb_index.get(source_id, "")
                        if ref:
                            references[source_id] = ref
            logger.info(f"  Loaded reward_bench references")
        except Exception as e:
            logger.warning(f"  Could not load reward_bench: {e}")

    # code_ultra_feedback: highest-rated response
    if "code_ultra_feedback" in by_dataset:
        logger.info(f"Loading code_ultra_feedback references...")
        try:
            ds = load_dataset("coseal/CodeUltraFeedback", split="train")
            for source_id, idx_str in by_dataset["code_ultra_feedback"]:
                idx = int(idx_str)
                if idx < len(ds):
                    row = ds[idx]
                    responses = row.get("responses", [])
                    annotations = row.get("annotations", [])
                    if responses and annotations:
                        best_idx = max(range(len(annotations)),
                                       key=lambda i: int(annotations[i].get("rating", 0)))
                        references[source_id] = responses[best_idx].get("response", "")
                    elif responses:
                        references[source_id] = responses[0].get("response", "")
            logger.info(f"  Loaded {sum(1 for s in references if s.startswith('code_ultra'))} code_ultra")
        except Exception as e:
            logger.warning(f"  Could not load code_ultra_feedback: {e}")

    # mix_instruct: output field
    mix_subsets = ["unified_chip2", "dolly_15k", "itwgpt4", "sharegpt", "laion"]
    if any(ds_name in by_dataset for ds_name in mix_subsets):
        logger.info("Loading mix_instruct references...")
        try:
            ds = load_dataset("llm-blender/mix-instruct", split="train")
            id_to_output = {str(row.get("id", "")): row.get("output", "") for row in ds}
            for ds_name in mix_subsets:
                if ds_name in by_dataset:
                    for source_id, idx_str in by_dataset[ds_name]:
                        ref = id_to_output.get(source_id, "")
                        if ref:
                            references[source_id] = ref
            logger.info(f"  Loaded mix_instruct references")
        except Exception as e:
            logger.warning(f"  Could not load mix_instruct: {e}")

    # beaver_tails: use standard refusal template
    for source_id, _ in by_dataset.get("beaver_tails", []):
        references[source_id] = REFUSAL_TEMPLATE

    logger.info(f"Total references loaded: {len(references)} / {len(sources)}")
    return references


def main():
    parser = argparse.ArgumentParser(description="Add reference_text to data")
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--raw-data", default=None,
                        help="Raw JSON with full source mapping (for fill_source)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load data
    with open(args.input) as f:
        data = [json.loads(l) for l in f]
    logger.info(f"Loaded {len(data)} entries from {args.input}")

    # Step 1: Fill missing source fields if raw data provided
    if args.raw_data:
        with open(args.raw_data) as f:
            raw = json.load(f)
        rid_to_source = {r["request_id"]: r["source"] for r in raw["requests"] if r.get("source")}
        filled = 0
        for entry in data:
            if not entry.get("source"):
                src = rid_to_source.get(entry["request_id"])
                if src:
                    entry["source"] = src
                    filled += 1
        logger.info(f"Filled {filled} missing source fields")

    # Check source coverage
    has_source = sum(1 for d in data if d.get("source"))
    logger.info(f"Source coverage: {has_source}/{len(data)} ({100*has_source/len(data):.0f}%)")

    # Step 2: Load reference texts from HF
    sources = [d["source"] for d in data if d.get("source")]
    references = load_references_from_hf(sources)

    # Step 3: Add reference_text to each entry
    added = 0
    missing = 0
    for entry in data:
        source = entry.get("source")
        if source and source in references:
            entry["reference_text"] = references[source]
            added += 1
        elif entry.get("is_harmful"):
            entry["reference_text"] = REFUSAL_TEMPLATE
            added += 1
        else:
            entry["reference_text"] = ""
            missing += 1

    logger.info(f"Added reference_text: {added} entries, {missing} missing")

    if args.dry_run:
        # Show samples
        for entry in data[:3]:
            ref = entry.get("reference_text", "")
            print(f"  [{entry.get('dataset')}] source={entry.get('source')}")
            print(f"  ref_text: {ref[:100]}..." if len(ref) > 100 else f"  ref_text: {ref}")
            print()
        return

    # Save
    with open(args.output, "w") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
