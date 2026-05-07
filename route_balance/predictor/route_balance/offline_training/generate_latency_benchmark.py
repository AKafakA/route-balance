"""Generate requests and send them through the ROUTE_BALANCE scheduler to collect
latency training data for the XGBoost latency predictor.

Instance state is captured per-request by the sidecar predictor running on
each vLLM node (route_balance_predictor_api_server). route_balance_serve must be started with
--enable-predictor-feedback for data collection to work.

Supports two length-sampling modes:
  1. Synthetic: lognormal/uniform/fixed distributions (default)
  2. Real-data: sample prompts from preprocessed training data

Usage:
    # Synthetic lengths
    python -m route_balance.predictor.route_balance.offline_training.generate_latency_benchmark \
        --host 127.0.0.1 --port 8200 \
        --num-prompts 20000 --request-rate 18 \
        --output latency_data/qps_18.jsonl

    # Real-data prompts from preprocessed training data
    python -m route_balance.predictor.route_balance.offline_training.generate_latency_benchmark \
        --host 127.0.0.1 --port 8200 \
        --num-prompts 20000 --request-rate 18 \
        --real-data data/route_balance/training_data/route_balance_v3_all_training.json \
        --output latency_data/qps_18.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Real-data length sampling
# ---------------------------------------------------------------------------


def sample_real_requests(
    data_path: str,
    num: int,
    rng: np.random.Generator,
    max_tokens: int = 1024,
    target_model: str | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Sample real prompts and input_lens from training data.

    max_tokens is the vLLM generation cap (sampling_params.max_tokens).
    predicted_output_lens are the actual/expected output lengths from the
    training data, passed to vLLM as predicted_decode_tokens for accurate
    pending_decode_tokens tracking in /instance_stats and /schedule_trace.

    target_model: HF model name (e.g. "Qwen/Qwen2.5-72B"). When set, the
    per-model oracle output length is read from req["models"][target_model]
    ["output_length"] — this is the actual length the model produced when the
    dataset was scored. Required for per-(model, GPU) instance-level sweeps
    where each mini-cluster only routes to one model.

    Without target_model, falls back to top-level fields (legacy datasets
    that pre-resolved a single model). If neither path yields a positive
    value the request is DROPPED rather than padded to max_tokens — silent
    fallback to 1024 for every request masquerades as "uniform predicted
    decode" and corrupts predictor training data (the all-1024 bug).

    Returns:
        prompts (list[str]),
        input_lens (numpy array),
        max_tokens_arr (numpy array, all max_tokens — the generation cap),
        predicted_output_lens (numpy array — per-model oracle output lengths),
        source_request_ids (list[str])
    """
    with open(data_path) as f:
        if data_path.endswith(".jsonl"):
            requests = [json.loads(line) for line in f]
        else:
            data = json.load(f)
            requests = data["requests"]
    n_total = len(requests)

    # Filter to records that have a usable oracle output length for the
    # target model. Drop the rest — never silently pad to max_tokens.
    filtered: list[dict] = []
    n_dropped_missing_model = 0
    n_dropped_no_length = 0
    for req in requests:
        if target_model is not None:
            mdata = (req.get("models") or {}).get(target_model)
            if not isinstance(mdata, dict):
                n_dropped_missing_model += 1
                continue
            pred = mdata.get("output_length") or mdata.get("actual_output_tokens")
        else:
            pred = (
                req.get("actual_output_tokens")
                or req.get("output_length")
                or req.get("output_len")
            )
        if not pred or pred <= 0:
            n_dropped_no_length += 1
            continue
        filtered.append((req, int(pred)))
    if not filtered:
        raise ValueError(
            f"No usable records: {n_total} loaded, "
            f"{n_dropped_missing_model} missing target_model={target_model!r}, "
            f"{n_dropped_no_length} with non-positive length. Refusing to "
            f"silently fall back to max_tokens={max_tokens}."
        )
    if n_dropped_missing_model or n_dropped_no_length:
        logger.warning(
            "Dropped %d records (%d missing models[%s], %d non-positive length). "
            "Sampling from %d remaining.",
            n_dropped_missing_model + n_dropped_no_length,
            n_dropped_missing_model,
            target_model,
            n_dropped_no_length,
            len(filtered),
        )

    input_lens_all = np.array([req["input_len"] for req, _ in filtered])
    prompts_all = [req["prompt"] for req, _ in filtered]
    request_ids_all = [req.get("request_id", f"idx_{i}") for i, (req, _) in enumerate(filtered)]
    predicted_lens_all = np.array([p for _, p in filtered])

    indices = rng.choice(len(filtered), size=num, replace=True)
    sampled_prompts = [prompts_all[i] for i in indices]
    source_ids = [request_ids_all[i] for i in indices]
    max_tokens_arr = np.full(num, max_tokens, dtype=int)
    return sampled_prompts, input_lens_all[indices], max_tokens_arr, predicted_lens_all[indices], source_ids


def sample_real_lengths(
    data_path: str,
    num: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy: sample only lengths (for synthetic prompt mode)."""
    _, input_lens, output_lens, _, _ = sample_real_requests(data_path, num, rng)
    return input_lens, output_lens


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

BASE_SENTENCE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
)


def sample_lengths(
    num: int,
    dist: str,
    mean: float,
    std: float,
    lo: int,
    hi: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an int array of *num* lengths from the chosen distribution,
    clamped to [lo, hi]."""
    if dist == "fixed":
        lengths = np.full(num, int(mean))
    elif dist == "uniform":
        lengths = rng.integers(lo, hi, endpoint=True, size=num)
    elif dist == "lognormal":
        # For log-normal: if the user specifies mean=256, std=2.0 we
        # interpret mean as the *desired median* of the distribution and
        # std as the sigma of the underlying normal.
        mu = np.log(mean)
        sigma = std
        lengths = np.round(rng.lognormal(mu, sigma, size=num)).astype(int)
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    return np.clip(lengths, lo, hi).astype(int)


def build_dummy_prompt(target_tokens: int, tokenizer: Any) -> str:
    """Create a prompt string that tokenizes to exactly *target_tokens* tokens.

    Strategy: repeat the base sentence enough times to overshoot, tokenize,
    truncate to the exact count, then decode back to text.
    """
    if target_tokens <= 0:
        return ""

    # Estimate ~4 chars per token; overshoot by 2x for safety.
    repeats = max(1, (target_tokens * 8) // len(BASE_SENTENCE) + 2)
    long_text = BASE_SENTENCE * repeats

    token_ids = tokenizer.encode(long_text, add_special_tokens=False)
    if len(token_ids) < target_tokens:
        # Extremely unlikely with 2x overshoot, but handle it.
        extra_repeats = (target_tokens // len(token_ids) + 2)
        long_text = long_text * extra_repeats
        token_ids = tokenizer.encode(long_text, add_special_tokens=False)

    trimmed_ids = token_ids[:target_tokens]
    return tokenizer.decode(trimmed_ids)


# ---------------------------------------------------------------------------
# Request / response data
# ---------------------------------------------------------------------------

@dataclass
class LatencyRecord:
    request_id: str = ""
    source_request_id: str = ""  # original request_id from training data (for traceability)
    input_len: int = 0
    output_len: int = 0          # actual completion tokens
    max_tokens: int = 0          # requested max_tokens
    num_predicted_output_tokens: int = 0  # oracle prediction sent to vLLM (anti-1024-bug audit field)
    ttft: float = 0.0            # time to first token (server-reported)
    tpot: float = 0.0            # time per output token (from ITL mean)
    e2el: float = 0.0            # client-side end-to-end latency
    server_latency: float = 0.0  # server-reported E2E
    scheduling_overhead: float = 0.0
    model: str = ""
    host: str = ""
    instance_id: str = ""
    success: bool = False
    error: str = ""
    timestamp: float = 0.0       # request start time (epoch)
    request_rate: float = 0.0    # QPS level for this run
    # Apr 26: capture extra RouteBalance response fields for parity with benchmark_serving.py.
    # Discarded by old generate_latency_benchmark; preserved here for downstream use.
    scheduling_overhead_breakdown: dict = field(default_factory=dict)
    predicted_quality: float = 0.0
    predicted_length: float = 0.0
    predicted_best_model: str = ""
    predicted_best_hit: bool = False
    all_model_scores: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Async request sender
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt: str,
    input_len: int,
    max_tokens: int,
    request_id: str,
    model_name: str = "route_balance",
    num_predicted_output_tokens: int = 0,
) -> LatencyRecord:
    """Send a single completion request and return a LatencyRecord."""

    payload = {
        "request_id": request_id,
        "model": model_name,
        "prompt": prompt,
        "prompt_len": input_len,
        "max_tokens": max_tokens,
        "num_predicted_output_tokens": num_predicted_output_tokens or max_tokens,
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        # Apr 26: matches route_balance/exp/route_balance/test_broadcasting.sh:72
        # FREQUENCY_PENALTY=1.2, tuned via sweep_broadcasting.sh to prevent
        # degenerate repetition. The model_estimator training data was
        # generated with this setting; mismatched bench/inference would produce
        # different output_len distributions and KV pressure patterns.
        "frequency_penalty": 1.2,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}

    record = LatencyRecord(
        request_id=request_id,
        input_len=input_len,
        max_tokens=max_tokens,
        num_predicted_output_tokens=num_predicted_output_tokens or max_tokens,
    )

    st = time.perf_counter()
    record.timestamp = time.time()
    try:
        async with session.post(api_url, json=payload, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                record.e2el = time.perf_counter() - st

                # Detect response format: ROUTE_BALANCE coordinator vs raw vLLM (OpenAI-compatible)
                if "choices" in data:
                    # Raw vLLM OpenAI-compatible format
                    choices = data.get("choices", [])
                    usage = data.get("usage", {})
                    if choices:
                        record.success = True
                        record.output_len = usage.get("completion_tokens", 0)
                        record.model = data.get("model", "")
                        record.server_latency = record.e2el  # no separate server_latency
                        record.scheduling_overhead = 0.0
                        # vLLM doesn't return TTFT/ITL in non-streaming mode
                        # TPOT estimate: (e2el - estimated_prefill) / output_tokens
                        if record.output_len > 0:
                            record.tpot = record.e2el / record.output_len
                    else:
                        record.error = "Empty choices in vLLM response"

                elif data.get("success", False):
                    # ROUTE_BALANCE coordinator format
                    record.success = True
                    record.output_len = data.get("output_tokens", 0)
                    record.ttft = data.get("ttft", 0.0)
                    record.model = data.get("model", "")
                    record.instance_id = data.get("instance_id", "")
                    record.host = data.get("host", "")
                    record.server_latency = data.get("server_latency", 0.0)
                    record.scheduling_overhead = record.e2el - record.server_latency
                    # Compute TPOT from ITL if available
                    itl = data.get("itl", [])
                    if itl and len(itl) > 0:
                        record.tpot = sum(itl) / len(itl)  # already in seconds
                    # Apr 26: capture extra RouteBalance fields for parity with benchmark_serving.py
                    record.scheduling_overhead_breakdown = data.get(
                        "scheduling_overhead_breakdown", {}
                    )
                    record.predicted_quality = data.get("predicted_quality", 0.0)
                    record.predicted_length = data.get("predicted_length", 0.0)
                    record.predicted_best_model = data.get("predicted_best_model", "")
                    record.predicted_best_hit = data.get("predicted_best_hit", False)
                    record.all_model_scores = data.get("all_model_scores", {})

                else:
                    record.error = data.get("error", "Server returned success=False")
                    record.model = data.get("model", "")
                    record.instance_id = data.get("instance_id", "")
                    record.host = data.get("host", "")
            else:
                record.e2el = time.perf_counter() - st
                try:
                    err_data = await resp.json()
                    record.error = err_data.get("error", f"HTTP {resp.status}")
                except Exception:
                    record.error = f"HTTP {resp.status}: {resp.reason or 'Unknown'}"
    except Exception as exc:
        record.e2el = time.perf_counter() - st
        record.error = f"{type(exc).__name__}: {exc}"

    return record


async def run_benchmark(
    api_url: str,
    prompts: list[str],
    input_lens: np.ndarray,
    output_lens: np.ndarray,
    request_rate: float,
    model_name: str = "route_balance",
    source_request_ids: list[str] | None = None,
    predicted_output_lens: np.ndarray | None = None,
    num_warmups: int = 0,
) -> list[LatencyRecord]:
    """Send all requests with Poisson rate limiting. No concurrency cap.

    If num_warmups > 0, runs that many warmup requests BEFORE the timed bench
    using prompts[0:num_warmups]. Warmup results are discarded. Matches
    benchmark_serving.py warmup behavior (Apr 26).
    """

    num_prompts = len(prompts)
    records: list[LatencyRecord] = []
    lock = asyncio.Lock()

    pbar = tqdm(total=num_prompts, desc="Sending requests")

    async def _task(idx: int) -> None:
        rid = str(uuid.uuid4())
        pred_out = int(predicted_output_lens[idx]) if predicted_output_lens is not None else 0
        rec = await send_request(
            session, api_url, prompts[idx],
            int(input_lens[idx]), int(output_lens[idx]), rid,
            model_name=model_name,
            num_predicted_output_tokens=pred_out,
        )
        if source_request_ids:
            rec.source_request_id = source_request_ids[idx]
        async with lock:
            records.append(rec)
        pbar.update(1)

    # Apr 26: bumped 600s → 1800s per user direction. Aligned with benchmark_serving.py
    # for train/inference consistency. Long timeout = saturation truths reflected as
    # extra-long latencies (data) instead of as failures.
    timeout = aiohttp.ClientTimeout(total=1800)  # 30 min per request
    # Apr 27: connector limit was the dominant bottleneck under load. aiohttp's
    # default TCPConnector caps at 100 concurrent connections. With ~10s mean
    # latency, that capped end-to-end throughput at ~9.7 req/s regardless of
    # offered QPS — masquerading as a cluster-side cap. Bumping to 2048 lets
    # the bench actually inject the full offered rate.
    connector = aiohttp.TCPConnector(
        limit=2048, limit_per_host=2048,
        force_close=False, enable_cleanup_closed=True,
    )
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Apr 26: warmup phase per benchmark_serving.py parity. Sends `num_warmups`
        # requests through route_balance_serve before the timed bench, discards the responses.
        # Fills KV caches across the cluster; ensures first timed request doesn't
        # see cold-start latency.
        if num_warmups > 0:
            warmup_n = min(num_warmups, num_prompts)
            print(f"[warmup] sending {warmup_n} warmup requests (results discarded)...")
            warmup_tasks = []
            for i in range(warmup_n):
                wrid = "warmup-" + str(uuid.uuid4())
                pred_out = int(predicted_output_lens[i]) if predicted_output_lens is not None else 0
                warmup_tasks.append(asyncio.create_task(send_request(
                    session, api_url, prompts[i],
                    int(input_lens[i]), int(output_lens[i]), wrid,
                    model_name=model_name,
                    num_predicted_output_tokens=pred_out,
                )))
            await asyncio.gather(*warmup_tasks)
            print(f"[warmup] done.")

        tasks: list[asyncio.Task] = []
        start = time.perf_counter()

        for i in range(num_prompts):
            # Rate limiting via Poisson inter-arrival times
            if request_rate < float("inf") and i > 0:
                interval = 1.0 / request_rate
                elapsed = time.perf_counter() - start
                expected = i * interval
                sleep_time = expected - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            task = asyncio.create_task(_task(i))
            tasks.append(task)

        await asyncio.gather(*tasks)

    pbar.close()
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary(records: list[LatencyRecord]) -> None:
    successes = [r for r in records if r.success]
    failures = [r for r in records if not r.success]

    print(f"\n{'=' * 60}")
    print(f"  Latency Benchmark Summary")
    print(f"{'=' * 60}")
    print(f"  Total requests:    {len(records)}")
    print(f"  Successful:        {len(successes)}")
    print(f"  Failed:            {len(failures)}")

    if successes:
        e2els = np.array([r.e2el for r in successes])
        ttfts = np.array([r.ttft for r in successes])
        in_lens = np.array([r.input_len for r in successes])
        out_lens = np.array([r.output_len for r in successes])

        print(f"\n  E2E Latency (s):")
        print(f"    mean={e2els.mean():.3f}  p50={np.median(e2els):.3f}  "
              f"p95={np.percentile(e2els, 95):.3f}  p99={np.percentile(e2els, 99):.3f}")
        print(f"  TTFT (s):")
        print(f"    mean={ttfts.mean():.4f}  p50={np.median(ttfts):.4f}  "
              f"p95={np.percentile(ttfts, 95):.4f}")
        print(f"  Input length:")
        print(f"    mean={in_lens.mean():.0f}  min={in_lens.min()}  max={in_lens.max()}")
        print(f"  Output length:")
        print(f"    mean={out_lens.mean():.0f}  min={out_lens.min()}  max={out_lens.max()}")

        # Per-model breakdown
        models = set(r.model for r in successes)
        if len(models) > 1:
            print(f"\n  Per-model breakdown:")
            for m in sorted(models):
                model_recs = [r for r in successes if r.model == m]
                me = np.array([r.e2el for r in model_recs])
                print(f"    {m}: n={len(model_recs)}, "
                      f"e2el mean={me.mean():.3f}s p95={np.percentile(me, 95):.3f}s")

    if failures:
        error_counts: dict[str, int] = {}
        for r in failures:
            key = str(r.error)[:80] if r.error else "unknown"
            error_counts[key] = error_counts.get(key, 0) + 1
        print(f"\n  Error breakdown (top 5):")
        for err, cnt in sorted(error_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"    [{cnt}x] {err}")

    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic latency benchmark data for ROUTE_BALANCE XGBoost predictor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)

    # Workload
    parser.add_argument("--num-prompts", type=int, default=20000)
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Requests per second (inf = no rate limit)")

    # Input length distribution
    parser.add_argument("--input-len-dist", type=str, default="lognormal",
                        choices=["lognormal", "uniform", "fixed"])
    parser.add_argument("--input-len-mean", type=float, default=256)
    parser.add_argument("--input-len-std", type=float, default=2.0)
    parser.add_argument("--input-len-min", type=int, default=32)
    parser.add_argument("--input-len-max", type=int, default=512)

    # Output length distribution
    parser.add_argument("--output-len-dist", type=str, default="lognormal",
                        choices=["lognormal", "uniform", "fixed"])
    parser.add_argument("--output-len-mean", type=float, default=128)
    parser.add_argument("--output-len-std", type=float, default=1.5)
    parser.add_argument("--output-len-min", type=int, default=16)
    parser.add_argument("--output-len-max", type=int, default=512)

    # Real-data mode (overrides synthetic distributions)
    parser.add_argument("--real-data", type=str, default=None,
                        help="Path to preprocessed training JSON; sample (input_len, output_len) "
                             "from actual data distribution instead of synthetic")

    # Model name for the /v1/completions payload (use actual model name for direct vLLM)
    parser.add_argument("--model", type=str, default="route_balance",
                        help="Model name in API payload. Use 'route_balance' for coordinator, "
                             "or actual model name (e.g. 'Qwen/Qwen2.5-7B') for direct vLLM")

    # Target model for per-(model, GPU) sweeps: dataset records have per-model
    # output_length nested under models[<hf_model_name>]['output_length'].
    # When set, predicted output lengths come from that nested field — required
    # for instance-type-level latency sweeps so num_predicted_output_tokens
    # reflects the model's actual output behavior, not a max_tokens fallback.
    parser.add_argument("--target-model", type=str, default=None,
                        help="HF model name to extract per-model output_length "
                             "from dataset (e.g. 'Qwen/Qwen2.5-72B'). Required "
                             "when the bench targets a single (model, GPU) "
                             "mini-cluster and the dataset carries per-model "
                             "scoring nested under models[<name>].")

    # Tokenizer
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-3B")

    # Max tokens cap (should match broadcasting config)
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max output tokens per request (should match broadcasting cap)")

    # Output
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output JSONL file")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    # Apr 26: warmup phase per benchmark_serving.py parity. Default 0 = opt-in.
    parser.add_argument("--num-warmups", type=int, default=0,
                        help="Number of warmup requests (sent before timed bench, "
                             "results discarded). Matches benchmark_serving.py behavior. "
                             "Default 0 = no warmup.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    rng = np.random.default_rng(args.seed)

    # ---- 1. Sample requests ----
    real_prompts: list[str] | None = None
    source_request_ids: list[str] | None = None
    predicted_output_lens: np.ndarray | None = None
    if args.real_data:
        logger.info("Sampling %d real requests from: %s",
                     args.num_prompts, args.real_data)
        real_prompts, input_lens, output_lens, predicted_output_lens, source_request_ids = sample_real_requests(
            args.real_data, args.num_prompts, rng, max_tokens=args.max_tokens,
            target_model=args.target_model,
        )
    else:
        logger.info("Sampling %d (input_len, output_len) from synthetic distributions ...",
                     args.num_prompts)
        input_lens = sample_lengths(
            args.num_prompts,
            args.input_len_dist,
            args.input_len_mean,
            args.input_len_std,
            args.input_len_min,
            args.input_len_max,
            rng,
        )
        output_lens = sample_lengths(
            args.num_prompts,
            args.output_len_dist,
            args.output_len_mean,
            args.output_len_std,
            args.output_len_min,
            args.output_len_max,
            rng,
        )

    logger.info(
        "Input lengths: mean=%.0f, min=%d, max=%d | Output lengths: mean=%.0f, min=%d, max=%d",
        input_lens.mean(), input_lens.min(), input_lens.max(),
        output_lens.mean(), output_lens.min(), output_lens.max(),
    )

    # ---- 2. Build prompts ----
    if real_prompts is not None:
        logger.info("Using %d real prompts from training data.", len(real_prompts))
        prompts = real_prompts
    else:
        logger.info("Loading tokenizer %s ...", args.tokenizer)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

        logger.info("Generating %d dummy prompts ...", args.num_prompts)
        prompt_cache: dict[int, str] = {}
        prompts: list[str] = []
        for ilen in input_lens:
            ilen = int(ilen)
            if ilen not in prompt_cache:
                prompt_cache[ilen] = build_dummy_prompt(ilen, tokenizer)
            prompts.append(prompt_cache[ilen])
        logger.info("Cached %d unique prompt lengths.", len(prompt_cache))

    # ---- 3. Send requests ----
    api_url = f"http://{args.host}:{args.port}/v1/completions"
    logger.info(
        "Sending %d requests to %s (rate=%.1f rps) ...",
        args.num_prompts, api_url, args.request_rate,
    )

    # ---- 4. Run benchmark ----
    records = asyncio.run(run_benchmark(
        api_url, prompts, input_lens, output_lens,
        args.request_rate,
        model_name=args.model,
        source_request_ids=source_request_ids,
        predicted_output_lens=predicted_output_lens if real_prompts else None,
        num_warmups=args.num_warmups,
    ))

    # Tag each record with request_rate
    for rec in records:
        rec.request_rate = args.request_rate

    # ---- 5. Save results ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec)) + "\n")

    logger.info("Saved %d latency records to %s", len(records), out_path)

    # ---- 6. Print summary ----
    print_summary(records)


if __name__ == "__main__":
    main()
