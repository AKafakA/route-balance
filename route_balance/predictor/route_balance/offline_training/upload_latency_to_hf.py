#!/usr/bin/env python3
"""
Upload latency prediction data to HuggingFace.

Replaces the old, buggy route_balance_latency_prediction dataset with correct data.
The old dataset had:
  - Concurrency-capped records (QPS lower than actual)
  - Split into separate files per instance type
  - Missing per-request queue shapes

The new data has:
  - 404,929 records (359,929 train + 45,000 test)
  - Full schedule_state with running_requests[] and waiting_requests[]
  - Correct actual_output_tokens (recovered via (e2e-ttft)/tpot + 1)
  - 5 instance types across 4 GPU types

Usage:
    python -m route_balance.predictor.route_balance.offline_training.upload_latency_to_hf \
        --train-data data/route_balance/latency_data/all/latency_train_tagged.jsonl \
        --test-data data/route_balance/latency_data/all/latency_test_tagged.jsonl \
        --repo asdwb/route_balance_latency_prediction \
        --token $HF_TOKEN

    # Dry run (just print stats)
    python -m route_balance.predictor.route_balance.offline_training.upload_latency_to_hf \
        --train-data data/route_balance/latency_data/all/latency_train_tagged.jsonl \
        --test-data data/route_balance/latency_data/all/latency_test_tagged.jsonl \
        --dry-run
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


DATACARD = """---
license: apache-2.0
task_categories:
  - tabular-regression
language:
  - en
tags:
  - llm-serving
  - latency-prediction
  - heterogeneous-serving
  - vllm
size_categories:
  - 100K<n<1M
---

# LLM Latency Prediction Dataset

Per-request latency records from heterogeneous LLM serving on CloudLab,
with full queue state snapshots for latency prediction research.

## Overview

| Property | Value |
|----------|-------|
| Total records | 404,929 |
| Train / Test | 359,929 / 45,000 |
| Instance types | 5 (3B/P100, 3B/A30, 7B/A30, 14B/V100, 72B/A100) |
| Features | 17 instance-level + 5 per-request queue shape |
| Targets | E2E latency, TTFT, TPOT |
| Collection | vLLM serving, mixed-dataset workload, QPS 8-24 |

## Schema

Each record is one completed request with prediction-time state:

```json
{
  "request_id": "chatcmpl-xxx",
  "instance_type": "qwen2.5-7b_a30",
  "instance_id": "instance-2",
  "num_prompt_tokens": 585,
  "num_predicted_output_tokens": 256,
  "actual_output_tokens": 142,
  "actual_e2e_latency": 5.23,
  "actual_ttft": 0.31,
  "actual_tpot": 0.035,
  "prediction_timestamp": 1711234567.89,
  "completion_timestamp": 1711234573.12,
  "schedule_state": {
    "num_running": 9,
    "num_waiting": 2,
    "ema_decode_tok_per_s": 48.5,
    "ema_prefill_tok_per_s": 2500.0,
    "kv_cache_utilization": 0.67,
    "running_requests": [
      {
        "num_prompt_tokens": 200,
        "num_computed_tokens": 200,
        "total_num_tokens": 342,
        "num_output_tokens": 142
      }
    ],
    "waiting_requests": [...]
  }
}
```

### Per-request queue shape features

The `running_requests` and `waiting_requests` arrays provide **per-request
state snapshots** at prediction time. Each entry has:

| Field | Description |
|-------|-------------|
| `num_prompt_tokens` | Request's prompt length |
| `num_computed_tokens` | Tokens already processed (prefill progress) |
| `total_num_tokens` | Total sequence length at snapshot |
| `num_output_tokens` | Output tokens generated so far |

These enable LSTM-style models that encode queue shape as a sequence,
rather than just using aggregate statistics.

### Instance-level features (in schedule_state)

| Feature | Description |
|---------|-------------|
| `ema_decode_tok_per_s` | Exponential moving average decode throughput |
| `ema_prefill_tok_per_s` | EMA prefill throughput |
| `ema_decode_iter_ms` | EMA decode iteration time |
| `decode_ctx_p50/p95/max` | Running requests' context length percentiles |
| `num_running` | Number of running requests |
| `num_waiting` | Number of waiting requests |
| `pending_prefill_tokens` | Total pending prefill tokens in queue |
| `pending_decode_tokens` | Total pending decode tokens |
| `kv_cache_utilization` | KV cache utilization fraction |
| `kv_free_blocks` | Free KV cache blocks |
| `token_budget_per_iter` | vLLM token budget per scheduler iteration |

## Instance Types

| Instance Type | Model | GPU | TP |
|--------------|-------|-----|-----|
| qwen2.5-3b_p100 | Qwen2.5-3B | P100 16GB | 1 |
| qwen2.5-3b_a30 | Qwen2.5-3B | A30 24GB | 1 |
| qwen2.5-7b_a30 | Qwen2.5-7B | A30 24GB | 1 |
| qwen2.5-14b_v100 | Qwen2.5-14B | V100×4 32GB | 4 |
| qwen2.5-72b_a100 | Qwen2.5-72B | A100×2 80GB | 2 |

## Usage

```python
import json

with open("train.jsonl") as f:
    records = [json.loads(line) for line in f]

for rec in records:
    # Per-request prediction features
    prompt_tokens = rec["num_prompt_tokens"]
    state = rec["schedule_state"]

    # Aggregate features (for XGBoost)
    num_running = state["num_running"]
    kv_util = state["kv_cache_utilization"]

    # Per-request queue shapes (for LSTM)
    running = state.get("running_requests", [])
    waiting = state.get("waiting_requests", [])

    # Targets
    e2e = rec["actual_e2e_latency"]
    ttft = rec["actual_ttft"]
    tpot = rec["actual_tpot"]
```

## License

Apache 2.0
"""


def main():
    parser = argparse.ArgumentParser(description="Upload latency data to HF")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--repo", default="asdwb/route_balance_latency_prediction")
    parser.add_argument("--token", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Print stats
    for split, path in [("train", args.train_data), ("test", args.test_data)]:
        with open(path) as f:
            records = [json.loads(line) for line in f]
        types = Counter(r.get("instance_type") for r in records)
        has_queue = sum(1 for r in records if r.get("schedule_state", {}).get("running_requests"))
        logger.info(f"{split}: {len(records)} records, {has_queue} with queue data")
        for t, c in types.most_common():
            logger.info(f"  {t}: {c}")

    if args.dry_run:
        print(DATACARD)
        return

    if not args.token:
        logger.error("--token required for upload")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)

    # Upload README
    readme_path = "/tmp/README_latency.md"
    with open(readme_path, "w") as f:
        f.write(DATACARD)

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Update datacard: per-request queue shapes, correct data",
    )

    # Upload data files
    for split, path in [("train", args.train_data), ("test", args.test_data)]:
        logger.info(f"Uploading {split}: {path}")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=f"{split}.jsonl",
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=f"Upload {split} data ({Path(path).stat().st_size / 1e6:.0f}MB)",
        )

    logger.info(f"Upload complete: {args.repo}")


if __name__ == "__main__":
    main()
