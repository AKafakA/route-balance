#!/usr/bin/env python3
"""
Score model responses using DeepEval G-Eval framework.

Uses LocalModel pointing at vLLM server with Llama-3.1-8B-Instruct.
Reference-grounded evaluation via expected_output parameter.
Scores 0-10 (normalized to 0.0-1.0 by DeepEval).

Usage:
    python -m route_balance.predictor.route_balance.offline_training.score_with_deepeval \
        --input data/test_with_reftext.jsonl \
        --output data/scored/test_scored.jsonl \
        --judge-key deepeval-llama3.1-8b-it_reference
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Score with DeepEval G-Eval (reference-grounded)"
    )
    parser.add_argument("--input", required=True, help="Input JSONL with reference_text")
    parser.add_argument("--output", required=True, help="Output JSONL with scores added")
    parser.add_argument("--judge-key", default="deepeval-llama3.1-8b-it_reference",
                        help="Key name in llm_judge_scores dict")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-concurrent", type=int, default=4,
                        help="Max concurrent async requests to vLLM")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Number of test cases per chunk (saves after each)")
    parser.add_argument("--start-chunk", type=int, default=0,
                        help="Resume from this chunk index (0-based). Loads existing scores from output file.")
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.input}")
    with open(args.input) as f:
        data = [json.loads(line) for line in f]
    if args.max_samples > 0:
        data = data[:args.max_samples]

    n_with_ref = sum(1 for d in data if d.get("reference_text"))
    logger.info(f"Loaded {len(data)} entries, {n_with_ref} with reference_text")

    # Setup DeepEval
    import os
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "dummy")
    os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "YES"
    os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS"] = "600"
    os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS"] = "300"
    os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = "3"

    from deepeval.models import LocalModel
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    model = LocalModel(
        model=args.model,
        base_url=args.base_url,
        api_key="dummy",
        timeout=120,
    )

    metric = GEval(
        name="Quality",
        criteria="Rate how well the actual output answers the question compared to the expected output. Consider correctness, completeness, and helpfulness.",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model,
    )

    # Build all test cases
    all_models = sorted(data[0]["models"].keys())
    total_pairs = len(data) * len(all_models)
    logger.info(f"Scoring {len(data)} entries × {len(all_models)} models = {total_pairs} pairs")
    logger.info(f"Judge key: llm_judge_scores.{args.judge_key}")

    test_cases = []
    test_case_meta = []  # (entry_idx, model_name)

    for i, entry in enumerate(data):
        ref_text = entry.get("reference_text", "")
        if not ref_text:
            continue
        for model_name in all_models:
            m_data = entry["models"].get(model_name, {})
            response = m_data.get("response", "")
            if not response:
                continue
            tc = LLMTestCase(
                input=entry["prompt"][:2048],
                actual_output=response[:2048],
                expected_output=ref_text[:2048],
            )
            test_cases.append(tc)
            test_case_meta.append((i, model_name))

    logger.info(f"Built {len(test_cases)} test cases")

    # Process in chunks of 1000 test cases with incremental saving
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, ErrorConfig, DisplayConfig

    CHUNK_SIZE = args.chunk_size
    t0 = time.time()
    scored = 0
    failed = 0
    failed_indices = []
    n_chunks = (len(test_cases) + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Resume support: load existing scores from output file if resuming
    if args.start_chunk > 0 and Path(args.output).exists():
        logger.info(f"Resuming from chunk {args.start_chunk}. Loading existing scores from {args.output}")
        with open(args.output) as f:
            existing_data = [json.loads(line) for line in f]
        # Merge existing scores into data (same order, same entries)
        for i, entry in enumerate(existing_data):
            if i >= len(data):
                break
            for model_name, mdata in entry.get("models", {}).items():
                existing_scores = mdata.get("llm_judge_scores", {})
                if args.judge_key in existing_scores:
                    data[i]["models"][model_name].setdefault("llm_judge_scores", {})
                    data[i]["models"][model_name]["llm_judge_scores"][args.judge_key] = existing_scores[args.judge_key]
                    scored += 1
        logger.info(f"Loaded {scored} existing scores, resuming from chunk {args.start_chunk}/{n_chunks}")

    for chunk_idx in range(n_chunks):
        if chunk_idx < args.start_chunk:
            continue
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, len(test_cases))
        chunk_cases = test_cases[chunk_start:chunk_end]
        chunk_meta = test_case_meta[chunk_start:chunk_end]

        logger.info(f"Chunk {chunk_idx+1}/{n_chunks}: {chunk_start}-{chunk_end} ({len(chunk_cases)} cases)")

        try:
            results = evaluate(
                test_cases=chunk_cases,
                metrics=[metric],
                async_config=AsyncConfig(run_async=True, max_concurrent=args.max_concurrent),
                error_config=ErrorConfig(ignore_errors=True),
                display_config=DisplayConfig(print_results=False, show_indicator=False),
            )

            # Extract scores from this chunk
            chunk_scored = 0
            chunk_failed = 0
            score_map = {}
            for tr in results.test_results:
                idx = int(tr.name.split("_")[-1])
                if tr.metrics_data and tr.metrics_data[0].score is not None:
                    score_map[idx] = tr.metrics_data[0].score

            for local_idx, (entry_idx, model_name) in enumerate(chunk_meta):
                score = score_map.get(local_idx)
                if score is not None:
                    data[entry_idx]["models"][model_name].setdefault("llm_judge_scores", {})
                    data[entry_idx]["models"][model_name]["llm_judge_scores"][args.judge_key] = score
                    chunk_scored += 1
                    scored += 1
                else:
                    chunk_failed += 1
                    failed += 1
                    failed_indices.append((entry_idx, model_name))

        except Exception as e:
            chunk_failed = len(chunk_cases)
            failed += chunk_failed
            logger.error(f"Chunk {chunk_idx+1} FAILED: {e}")
            for entry_idx, model_name in chunk_meta:
                failed_indices.append((entry_idx, model_name))

        elapsed = time.time() - t0
        rate = scored / elapsed if elapsed > 0 else 0
        remaining = len(test_cases) - chunk_end
        eta = remaining / rate if rate > 0 else 0
        logger.info(
            f"Chunk {chunk_idx+1}/{n_chunks} done: "
            f"+{chunk_scored} scored, +{chunk_failed} failed | "
            f"Total: {scored}/{len(test_cases)} ({100*scored/len(test_cases):.1f}%) | "
            f"{rate:.1f} pairs/s | ETA {eta:.0f}s"
        )
        sys.stdout.flush()

        # Incremental save after each chunk
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        # Save full output (overwrites with latest progress)
        with open(args.output, "w") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Save per-chunk checkpoint
        chunk_dir = Path(args.output).parent / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.jsonl"
        with open(chunk_path, "w") as f:
            for local_idx, (entry_idx, model_name) in enumerate(chunk_meta):
                score = score_map.get(local_idx)
                f.write(json.dumps({"entry_idx": entry_idx, "model_name": model_name, "score": score}, ensure_ascii=False) + "\n")
        logger.info(f"Saved chunk {chunk_idx+1} to {chunk_path} + checkpoint to {args.output}")

    total_elapsed = time.time() - t0
    logger.info(f"Scoring complete: {scored} scored, {failed} failed in {total_elapsed:.0f}s")
    logger.info(f"Rate: {scored/total_elapsed:.1f} pairs/s")

    if failed_indices:
        logger.warning(f"Failed indices: {len(failed_indices)} total")
        fail_path = Path(args.output).with_suffix(".failures.json")
        with open(fail_path, "w") as f:
            json.dump(failed_indices, f)

    # Score summary per model
    logger.info("\n=== Per-Model Summary ===")
    for model_name in all_models:
        scores = []
        for entry in data:
            s = entry["models"].get(model_name, {}).get("llm_judge_scores", {}).get(args.judge_key)
            if s is not None:
                scores.append(s)
        if scores:
            import numpy as np
            arr = np.array(scores)
            logger.info(
                f"  {model_name.split('/')[-1]}: mean={arr.mean():.3f}, "
                f"std={arr.std():.3f}, n={len(scores)}"
            )

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
