#!/usr/bin/env python3
"""
Upload clean latency prediction data to HuggingFace (v2).

Replaces the old corrupted (concurrency-capped) dataset with the clean
sweep 2 data (March 18-19). Uploads 4 files:

  - train.jsonl / test.jsonl: Flat schema (schedule_state fields at top level,
    no per-request lists). For XGBoost / simple models.
  - train_queue_details.jsonl / test_queue_details.jsonl: Full schema with
    per-request running_requests[] and waiting_requests[] including enriched
    actual_output_tokens per queue entry. For LSTM models.

Usage:
    # Dry run (print stats only)
    python -m route_balance.predictor.route_balance.offline_training.upload_latency_v2 \
        --train-data data/route_balance/latency_data/enriched/latency_train_tagged_enriched.jsonl \
        --test-data data/route_balance/latency_data/enriched/latency_test_tagged_enriched.jsonl \
        --dry-run

    # Upload
    python -m route_balance.predictor.route_balance.offline_training.upload_latency_v2 \
        --train-data data/route_balance/latency_data/enriched/latency_train_tagged_enriched.jsonl \
        --test-data data/route_balance/latency_data/enriched/latency_test_tagged_enriched.jsonl \
        --token $HF_TOKEN
"""

import argparse
import json
import logging
import tempfile
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Schedule state fields to flatten into top-level
SCHEDULE_STATE_FIELDS = [
    "num_running", "num_waiting", "num_active_decode_seqs",
    "decode_ctx_p50", "decode_ctx_p95", "decode_ctx_max",
    "pending_prefill_tokens", "pending_decode_tokens",
    "kv_cache_utilization", "kv_free_blocks",
    "token_budget_per_iter", "prefill_chunk_size", "max_num_seqs",
    "num_preempted",
    "ema_decode_tok_per_s", "ema_prefill_tok_per_s",
    "ema_decode_iter_ms", "kv_evictions_per_s",
]


def flatten_record(rec: dict) -> dict:
    """Flatten schedule_state into top-level fields, drop per-request lists."""
    ss = rec.get("schedule_state", {})
    flat = {}

    # Top-level fields
    for k in ["request_id", "instance_id", "instance_type",
              "num_prompt_tokens", "num_predicted_output_tokens",
              "actual_output_tokens",
              "actual_e2e_latency", "actual_ttft", "actual_tpot",
              "prediction_timestamp", "completion_timestamp",
              "prediction_latency_ms", "probe_latency_ms"]:
        if k in rec:
            flat[k] = rec[k]

    # Flatten schedule_state
    for k in SCHEDULE_STATE_FIELDS:
        flat[k] = ss.get(k, 0)

    # Add derived counts
    flat["running_requests_count"] = len(ss.get("running_requests", []))
    flat["waiting_requests_count"] = len(ss.get("waiting_requests", []))

    return flat


def prepare_files(input_path: str, flat_path: str, queue_path: str) -> dict:
    """Read input, write flat + queue_details files. Return stats."""
    total = 0
    types = Counter()

    with open(input_path) as fin, \
         open(flat_path, "w") as f_flat, \
         open(queue_path, "w") as f_queue:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            types[rec.get("instance_type", "unknown")] += 1

            # Flat version (no per-request lists)
            flat = flatten_record(rec)
            f_flat.write(json.dumps(flat) + "\n")

            # Queue details version (full, with enriched actual_output_tokens)
            f_queue.write(json.dumps(rec) + "\n")

    return {"total": total, "types": dict(types.most_common())}


def main():
    parser = argparse.ArgumentParser(description="Upload clean latency data to HF (v2)")
    parser.add_argument("--train-data", required=True, help="Enriched train JSONL")
    parser.add_argument("--test-data", required=True, help="Enriched test JSONL")
    parser.add_argument("--repo", default="asdwb/route_balance_latency_prediction")
    parser.add_argument("--token", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None,
                        help="Save prepared files locally (skip upload)")
    args = parser.parse_args()

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = Path(tempfile.mkdtemp())

    # Prepare files
    for split, path in [("train", args.train_data), ("test", args.test_data)]:
        flat_path = str(out / f"{split}.jsonl")
        queue_path = str(out / f"{split}_queue_details.jsonl")

        logger.info(f"Preparing {split}: {path}")
        stats = prepare_files(path, flat_path, queue_path)
        logger.info(f"  {stats['total']} records")
        for t, c in stats["types"].items():
            logger.info(f"    {t}: {c}")

        flat_size = Path(flat_path).stat().st_size / 1e6
        queue_size = Path(queue_path).stat().st_size / 1e6
        logger.info(f"  {split}.jsonl: {flat_size:.0f}MB")
        logger.info(f"  {split}_queue_details.jsonl: {queue_size:.0f}MB")

    if args.dry_run:
        logger.info("Dry run — files at {}".format(out))
        return

    if args.output_dir:
        logger.info(f"Files saved to {out}")
        return

    if not args.token:
        logger.error("--token required for upload")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)

    # Upload all 4 data files
    files_to_upload = [
        ("train.jsonl", "Upload clean train data (flat schema)"),
        ("test.jsonl", "Upload clean test data (flat schema)"),
        ("train_queue_details.jsonl", "Upload train data with per-request queue details"),
        ("test_queue_details.jsonl", "Upload test data with per-request queue details"),
    ]

    for fname, msg in files_to_upload:
        fpath = str(out / fname)
        size_mb = Path(fpath).stat().st_size / 1e6
        logger.info(f"Uploading {fname} ({size_mb:.0f}MB)")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=f"{msg} — sweep 2 (March 18-19), enriched with actual_output_tokens",
        )

    logger.info(f"Upload complete: {args.repo}")


if __name__ == "__main__":
    main()
