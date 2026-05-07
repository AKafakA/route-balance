#!/usr/bin/env python3
"""
Update HuggingFace datacard for route_balance_model_estimator dataset.

Usage:
    python3 -m route_balance.predictor.route_balance.offline_training.update_hf_datacard \
        --token <HF_TOKEN> --repo anon/route_balance_model_estimator
"""

import argparse

DATACARD = """---
dataset_info:
  features:
  - name: request_id
    dtype: string
  - name: prompt
    dtype: string
  - name: input_len
    dtype: int64
  - name: dataset
    dtype: string
  - name: source
    dtype: string
  - name: is_harmful
    dtype: bool
  - name: models
    dtype: string
  splits:
  - name: train
    num_examples: 14963
  - name: test
    num_examples: 3642
---

# ROUTE_BALANCE Model Estimator Dataset

Multi-model quality and length prediction dataset for heterogeneous LLM serving.

## Overview

18,605 prompts from 7 public datasets, each broadcast to 4 Qwen2.5 model sizes (3B, 7B, 14B, 72B). Each entry contains the prompt, all model responses, and per-model quality/length annotations.

## Schema

### Top-level fields
| Field | Type | Description |
|-------|------|-------------|
| `request_id` | string | Unique request identifier |
| `prompt` | string | Input prompt (chat-formatted) |
| `input_len` | int | Prompt length in tokens |
| `dataset` | string | Source dataset name (gsm8k, squad, beaver_tails, mix_instruct, code_ultra_feedback, lmsys, reward_bench) |
| `source` | string | Original dataset row ID (e.g., "gsm8k/4222", "squad/77478", "lmsys/abc123", "beaver_tails/6234") |
| `is_harmful` | bool | Whether the prompt is harmful (from beaver_tails or reward_bench safety subsets) |

### Per-model fields (under `models.<model_name>`)
| Field | Type | Description |
|-------|------|-------------|
| `output_length` | int | Number of generated tokens |
| `compression_ratio` | float | Output/input length ratio |
| `is_truncated` | bool | Whether generation hit max_tokens |
| `response` | string | Full generated text |
| `similarity_score` | float [0,1] | Cosine similarity to 72B response (sentence-transformers/all-MiniLM-L6-v2) |
| `llm_judge_scores` | dict | Per-judge quality scores. Keys: `protectai_distilroberta-base-rejection-v1` (safety), Qwen judge (quality) |
| `reference_similarity` | float [0,1] | Cosine similarity to dataset reference response (sentence-transformers) |
| `reference_score` | float [0,1] | **Unified quality score** — dataset-appropriate metric (see below) |

### reference_score methodology

| Dataset | Source | Metric | Range | Description |
|---------|--------|--------|-------|-------------|
| gsm8k | `openai/gsm8k` train split | Exact-match | {0, 1} | Extract final number from response (regex), compare to reference answer after `####`. Standard GSM8K evaluation metric. |
| squad | `rajpurkar/squad` train split | Token F1 | [0, 1] | Token-level F1 between response and answer spans. Standard SQuAD evaluation metric. |
| beaver_tails | N/A (harmful prompts) | Refusal score | [0, 1] | ProtectAI `distilroberta-base-rejection-v1` classifier. High = correctly refused harmful request. |
| code_ultra_feedback | `coseal/CodeUltraFeedback` | Embedding similarity | [0, 1] | Cosine similarity to highest-rated response in dataset |
| reward_bench | `allenai/reward-bench` | Embedding similarity | [0, 1] | Cosine similarity to human-preferred (`chosen`) response |
| mix_instruct | `llm-blender/mix-instruct` | Embedding similarity | [0, 1] | Cosine similarity to dataset `output` field |
| lmsys | `lmsys/lmsys-chat-1m` | Embedding similarity | [0, 1] | Cosine similarity to original assistant response |

**Known limitation (GSM8K):** ~2.3% of entries scored as correct have models that solved the math problem correctly but then hallucinated unrelated continuation text. The correct answer appears in the math solution; the regex may also match a coincidental number from the hallucinated tail. This is consistent with standard GSM8K evaluation methodology used in published benchmarks.

### Safety-aware scoring

For harmful prompts (`is_harmful=True`), quality signals are inverted:
- `llm_judge_scores.protectai_*`: High score = model correctly **refused** the harmful request
- `reference_score`: Uses ProtectAI refusal classifier (not embedding similarity)
- A model that complies with a harmful request gets a LOW score (bad behavior)
- A model that refuses gets a HIGH score (correct behavior)

## Data sources

| Dataset | # Prompts | Type | HuggingFace Source |
|---------|-----------|------|-------------------|
| gsm8k | 2,363 (train) / 592 (test) | Math word problems | `openai/gsm8k` (main split) |
| squad | 2,348 / 588 | Reading comprehension QA | `rajpurkar/squad` (train split) |
| beaver_tails | 2,318 / 580 | Harmful prompts (safety) | `PKU-Alignment/BeaverTails` |
| mix_instruct | 2,300 / 575 | Mixed instructions | `llm-blender/mix-instruct` (train split) |
| code_ultra_feedback | 2,243 / 561 | Code generation | `coseal/CodeUltraFeedback` (train split) |
| lmsys | 2,055 / 514 | Real user conversations | `lmsys/lmsys-chat-1m` (English only) |
| reward_bench | 1,336 / 232 | Safety + code evaluation | `allenai/reward-bench` (filtered split) |

## Models

All responses generated using vLLM with `temperature=0.0` (greedy decoding):
- `Qwen/Qwen2.5-72B` (4×A100, tensor parallel=4)
- `Qwen/Qwen2.5-14B` (4×V100, tensor parallel=4)
- `Qwen/Qwen2.5-7B` (1×A30)
- `Qwen/Qwen2.5-3B` (1×A30 or 1×P100)

## Usage

```python
from datasets import load_dataset
ds = load_dataset("anon/route_balance_model_estimator")

# Access a training example
example = ds["train"][0]
print(example["prompt"][:100])
print(example["dataset"], example["source"])

# Per-model quality scores
import json
models = json.loads(example["models"])
for model_name, data in models.items():
    print(f"{model_name}: length={data['output_length']}, ref_score={data.get('reference_score')}")
```
"""


def main():
    parser = argparse.ArgumentParser(description="Update HF datacard")
    parser.add_argument("--token", required=True, help="HuggingFace token")
    parser.add_argument("--repo", default="anon/route_balance_model_estimator")
    parser.add_argument("--dry-run", action="store_true", help="Print but don't upload")
    args = parser.parse_args()

    if args.dry_run:
        print(DATACARD)
        return

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)

    # Write README
    with open("/tmp/README_datacard.md", "w") as f:
        f.write(DATACARD)

    api.upload_file(
        path_or_fileobj="/tmp/README_datacard.md",
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="Update datacard: add reference_score methodology, source field, data sources table",
    )
    print(f"Datacard uploaded to {args.repo}")


if __name__ == "__main__":
    main()
