"""Benchmark model estimator overhead on CPU and GPU.

Measures per-component latency for the fused encoder (RoBERTa/ModernBERT)
used in RouteBalance's scheduling pipeline. Reports tokenization, forward pass,
KNN lookup, and total estimate() time.

Usage:
    # CPU only (default, co-located with scheduler)
    python route_balance/exp/route_balance/benchmark_estimator_overhead.py \
        --config route_balance/config/route_balance/scheduler_config_p100_c3.json

    # GPU (dedicated scheduler node)
    python route_balance/exp/route_balance/benchmark_estimator_overhead.py \
        --config route_balance/config/route_balance/scheduler_config_p100_c3.json \
        --device cuda:0

    # Compare CPU vs GPU
    python route_balance/exp/route_balance/benchmark_estimator_overhead.py \
        --config route_balance/config/route_balance/scheduler_config_p100_c3.json \
        --compare
"""

import argparse
import json
import time
import sys

import numpy as np
import torch


def profile_estimator(config_path: str, device: str, n_warmup: int = 5, n_iter: int = 50):
    """Profile model estimator on given device."""
    from route_balance.predictor.route_balance.model_estimator import DefaultModelEstimator

    config = json.load(open(config_path))
    me_config = config["model_estimator"]

    # Load estimator
    t0 = time.monotonic()
    estimator = DefaultModelEstimator(me_config, device=device)
    load_time = time.monotonic() - t0

    prompt = (
        "Explain the theory of relativity in simple terms that a high school "
        "student could understand. Include examples from everyday life."
    )

    # Warmup
    for _ in range(n_warmup):
        estimator.estimate(prompt, budget_tokens=256)

    # Profile full estimate()
    times = []
    for _ in range(n_iter):
        t0 = time.monotonic()
        estimator.estimate(prompt, budget_tokens=256)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.monotonic() - t0) * 1000)

    # Profile components if fused bucket exists
    component_times = {}
    if estimator._fused_bucket is not None:
        tok = estimator._fused_bucket_tok

        # Tokenization
        t_tok = []
        for _ in range(n_iter):
            t0 = time.monotonic()
            inputs = tok(prompt, truncation=True, max_length=512, return_tensors="pt")
            t_tok.append((time.monotonic() - t0) * 1000)
        component_times["tokenization"] = t_tok

        # Transfer to device
        if device.startswith("cuda"):
            inputs_cpu = tok(prompt, truncation=True, max_length=512, return_tensors="pt")
            t_transfer = []
            for _ in range(n_iter):
                t0 = time.monotonic()
                _ = {k: v.to(device) for k, v in inputs_cpu.items()}
                t_transfer.append((time.monotonic() - t0) * 1000)
            component_times["cpu_to_gpu_transfer"] = t_transfer

        # Forward pass
        dev = next(estimator._fused_bucket.parameters()).device
        inputs_dev = {k: v.to(dev) for k, v in inputs.items()}
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_fwd = []
        for _ in range(n_iter):
            t0 = time.monotonic()
            with torch.no_grad():
                estimator._fused_bucket.predict(**inputs_dev)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t_fwd.append((time.monotonic() - t0) * 1000)
        component_times["encoder_forward"] = t_fwd

    # Profile batch estimate
    prompts_batch = [prompt] * 8
    batch_times = []
    if hasattr(estimator, 'estimate_batch'):
        for _ in range(n_warmup):
            estimator.estimate_batch(prompts_batch, budget_tokens=256)
        for _ in range(n_iter):
            t0 = time.monotonic()
            estimator.estimate_batch(prompts_batch, budget_tokens=256)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            batch_times.append((time.monotonic() - t0) * 1000)

    return {
        "device": device,
        "load_time_s": round(load_time, 2),
        "model_names": estimator.model_names,
        "has_fused_bucket": estimator._has_fused_bucket,
        "estimate_ms": {
            "p50": round(np.percentile(times, 50), 2),
            "p95": round(np.percentile(times, 95), 2),
            "p99": round(np.percentile(times, 99), 2),
            "mean": round(np.mean(times), 2),
        },
        "components": {
            k: {
                "p50": round(np.percentile(v, 50), 2),
                "p95": round(np.percentile(v, 95), 2),
                "mean": round(np.mean(v), 2),
            }
            for k, v in component_times.items()
        },
        "batch_8_ms": {
            "p50": round(np.percentile(batch_times, 50), 2),
            "p95": round(np.percentile(batch_times, 95), 2),
            "mean": round(np.mean(batch_times), 2),
            "per_prompt": round(np.mean(batch_times) / 8, 2),
        } if batch_times else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark model estimator overhead")
    parser.add_argument("--config", required=True, help="Scheduler config JSON")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda:0")
    parser.add_argument("--compare", action="store_true", help="Run both CPU and GPU")
    parser.add_argument("--n-iter", type=int, default=50, help="Number of iterations")
    parser.add_argument("--output", default=None, help="Save results to JSON")
    args = parser.parse_args()

    results = {}

    if args.compare:
        print("=== CPU Benchmark ===")
        results["cpu"] = profile_estimator(args.config, "cpu", n_iter=args.n_iter)
        print_results(results["cpu"])

        if torch.cuda.is_available():
            print("\n=== GPU Benchmark ===")
            results["gpu"] = profile_estimator(args.config, "cuda:0", n_iter=args.n_iter)
            print_results(results["gpu"])

            print("\n=== Speedup (CPU → GPU) ===")
            cpu_ms = results["cpu"]["estimate_ms"]["p50"]
            gpu_ms = results["gpu"]["estimate_ms"]["p50"]
            print("  estimate P50: %.2fms → %.2fms (%.1fx)" % (cpu_ms, gpu_ms, cpu_ms / gpu_ms))
            if results["cpu"]["batch_8_ms"] and results["gpu"]["batch_8_ms"]:
                cpu_b = results["cpu"]["batch_8_ms"]["per_prompt"]
                gpu_b = results["gpu"]["batch_8_ms"]["per_prompt"]
                print("  batch=8 per-prompt: %.2fms → %.2fms (%.1fx)" % (cpu_b, gpu_b, cpu_b / gpu_b))
        else:
            print("\nGPU not available, skipping GPU benchmark")
    else:
        results[args.device] = profile_estimator(args.config, args.device, n_iter=args.n_iter)
        print_results(results[args.device])

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print("\nSaved to %s" % args.output)


def print_results(r):
    print("  Device: %s" % r["device"])
    print("  Load time: %.2fs" % r["load_time_s"])
    print("  Models: %s" % [m.split("/")[-1] for m in r["model_names"]])
    print("  Fused bucket: %s" % r["has_fused_bucket"])
    print("  estimate() P50=%.2fms P95=%.2fms mean=%.2fms" % (
        r["estimate_ms"]["p50"], r["estimate_ms"]["p95"], r["estimate_ms"]["mean"]))
    for comp, v in r.get("components", {}).items():
        print("    %s: P50=%.2fms P95=%.2fms" % (comp, v["p50"], v["p95"]))
    if r.get("batch_8_ms"):
        b = r["batch_8_ms"]
        print("  batch=8: P50=%.2fms (%.2fms/prompt)" % (b["p50"], b["per_prompt"]))


if __name__ == "__main__":
    main()
