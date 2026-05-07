#!/usr/bin/env python3
"""
Enrich latency data with actual_output_tokens per queue entry.

Cross-references queue entry request_ids (stripping 'chatcmpl-' prefix) with
top-level records to recover actual output token counts. This enables LSTM v2
to use actual output length as a per-queue-entry feature.

Records where ANY queue entry has an unresolvable request_id are dropped
to ensure clean training data.

Usage:
    python -m route_balance.predictor.route_balance.offline_training.enrich_latency_queue \
        --input data/route_balance/latency_data/all/latency_train_tagged.jsonl \
               data/route_balance/latency_data/all/latency_test_tagged.jsonl \
        --output-dir data/route_balance/latency_data/enriched/
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_output_token_lookup(input_files: list[str]) -> dict[str, int]:
    """Build request_id -> actual_output_tokens lookup from all input files.

    Uses pre-tagged actual_output_tokens if present, otherwise derives from
    round((e2e - ttft) / tpot) + 1.
    """
    lookup = {}
    for fpath in input_files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rid = rec.get("request_id", "")
                if not rid:
                    continue

                actual = rec.get("actual_output_tokens", 0)
                if not actual or actual <= 0:
                    ttft = rec.get("actual_ttft", 0)
                    tpot = rec.get("actual_tpot", 0)
                    e2e = rec.get("actual_e2e_latency", 0)
                    if tpot and tpot > 0 and e2e and ttft:
                        actual = round((e2e - ttft) / tpot) + 1

                if actual and actual > 0:
                    lookup[rid] = int(actual)

    return lookup


def enrich_file(
    input_path: str,
    output_path: str,
    lookup: dict[str, int],
) -> dict:
    """Enrich a single JSONL file with per-queue-entry actual_output_tokens.

    Returns stats dict with counts.
    """
    total = 0
    kept = 0
    dropped = 0
    queue_entries_enriched = 0
    queue_entries_total = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1

            ss = rec.get("schedule_state", {})
            drop = False

            for list_key in ["running_requests", "waiting_requests"]:
                requests = ss.get(list_key, [])
                for qr in requests:
                    queue_entries_total += 1
                    qr_id = qr.get("request_id", "")
                    stripped = qr_id.replace("chatcmpl-", "")

                    if stripped in lookup:
                        qr["actual_output_tokens"] = lookup[stripped]
                        queue_entries_enriched += 1
                    else:
                        drop = True
                        break
                if drop:
                    break

            if drop:
                dropped += 1
                continue

            # Also ensure top-level actual_output_tokens is set
            rid = rec.get("request_id", "")
            if rid in lookup and not rec.get("actual_output_tokens"):
                rec["actual_output_tokens"] = lookup[rid]

            fout.write(json.dumps(rec) + "\n")
            kept += 1

    return {
        "total": total,
        "kept": kept,
        "dropped": dropped,
        "queue_entries_total": queue_entries_total,
        "queue_entries_enriched": queue_entries_enriched,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Enrich latency data with per-queue-entry actual_output_tokens"
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Input JSONL files (train + test)"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for enriched files"
    )
    args = parser.parse_args()

    # Step 1: Build lookup from ALL files
    logger.info(f"Building output token lookup from {len(args.input)} files...")
    lookup = build_output_token_lookup(args.input)
    logger.info(f"Lookup: {len(lookup)} request_ids with actual_output_tokens")

    # Step 2: Enrich each file
    for fpath in args.input:
        fname = Path(fpath).name
        # Add _enriched suffix before extension
        stem = fname.replace(".jsonl", "")
        out_fname = f"{stem}_enriched.jsonl"
        out_path = str(Path(args.output_dir) / out_fname)

        logger.info(f"\nEnriching {fname} -> {out_fname}")
        stats = enrich_file(fpath, out_path, lookup)

        logger.info(
            f"  Records: {stats['kept']}/{stats['total']} kept, "
            f"{stats['dropped']} dropped ({stats['dropped']/max(stats['total'],1)*100:.3f}%)"
        )
        logger.info(
            f"  Queue entries: {stats['queue_entries_enriched']}/{stats['queue_entries_total']} "
            f"enriched ({stats['queue_entries_enriched']/max(stats['queue_entries_total'],1)*100:.2f}%)"
        )

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
