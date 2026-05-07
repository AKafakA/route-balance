#!/usr/bin/env python3
"""
Analyze and plot ROUTE_BALANCE smoke test results.

Reads JSON result files from benchmark_serving.py, computes aggregate metrics,
prints comparison tables, and generates plots.

Usage:
    python3 analyze_smoketest.py --result-dir experiment_output/smoketest_results/
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_results(result_dir: str) -> Dict[str, dict]:
    """Load all JSON result files from directory."""
    results = {}
    for f in sorted(Path(result_dir).glob("*.json")):
        if f.name.endswith("_console.log"):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
            results[f.stem] = data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load {f}: {e}")
    return results


def extract_metrics(data: dict) -> dict:
    """Extract key metrics from a benchmark result."""
    m = {}
    # Standard vLLM benchmark fields
    m["num_requests"] = data.get("total_input_tokens", 0) // max(data.get("mean_input_tokens", 1), 1)
    m["completed"] = data.get("completed", data.get("num_requests", 0))
    m["duration"] = data.get("duration", 0)
    m["request_throughput"] = data.get("request_throughput", 0)
    m["output_throughput"] = data.get("output_throughput", 0)

    # Latency metrics
    m["mean_ttft"] = data.get("mean_ttft_ms", 0)
    m["median_ttft"] = data.get("median_ttft_ms", 0)
    m["p95_ttft"] = data.get("p95_ttft_ms", data.get("std_ttft_ms", 0))
    m["p99_ttft"] = data.get("p99_ttft_ms", 0)
    m["mean_e2e"] = data.get("mean_e2el_ms", 0)
    m["p95_e2e"] = data.get("p95_e2el_ms", 0)
    m["mean_tpot"] = data.get("mean_tpot_ms", 0)
    m["p95_tpot"] = data.get("p95_tpot_ms", 0)
    m["mean_itl"] = data.get("mean_itl_ms", 0)

    # Per-request details (if saved with --save-detailed)
    details = data.get("response_details", [])
    if details:
        m["num_requests"] = len(details)
        models = [d.get("model", "unknown") for d in details if d.get("success", True)]
        m["model_distribution"] = {}
        for model in models:
            m["model_distribution"][model] = m["model_distribution"].get(model, 0) + 1

        hosts = [d.get("host", "unknown") for d in details if d.get("success", True)]
        m["host_distribution"] = {}
        for host in hosts:
            m["host_distribution"][host] = m["host_distribution"].get(host, 0) + 1

        # Compute success rate
        successes = sum(1 for d in details if d.get("success", True))
        m["success_rate"] = successes / len(details) if details else 0

        # Compute from raw data if available
        ttfts = [d["ttft"] for d in details if d.get("ttft") and d.get("success", True)]
        e2es = [d["e2el"] for d in details if d.get("e2el") and d.get("success", True)]
        if ttfts:
            m["mean_ttft"] = np.mean(ttfts) * 1000
            m["median_ttft"] = np.median(ttfts) * 1000
            m["p95_ttft"] = np.percentile(ttfts, 95) * 1000
            m["p99_ttft"] = np.percentile(ttfts, 99) * 1000
        if e2es:
            m["mean_e2e"] = np.mean(e2es) * 1000
            m["p95_e2e"] = np.percentile(e2es, 95) * 1000

    # Metadata
    metadata = data.get("metadata", {})
    m["scheduler"] = metadata.get("scheduler", data.get("scheduling", "unknown"))
    m["qps"] = metadata.get("qps", data.get("request_rate", "?"))

    # Goodput metrics (from per-request details)
    if details:
        ttft_slo_ms = 10000  # default SLO
        tpot_slo_ms = 200
        budget_tokens = 256

        # Latency-SLO goodput: TTFT < SLO
        latency_good = sum(
            1 for d in details
            if d.get("success", True) and d.get("ttft", 99) * 1000 < ttft_slo_ms
        )
        m["latency_slo_goodput"] = latency_good / len(details) if details else 0

        # Budget goodput: output_tokens <= budget
        budget_good = sum(
            1 for d in details
            if d.get("success", True)
            and d.get("output_len", 0) <= budget_tokens
        )
        m["budget_goodput"] = budget_good / len(details) if details else 0

        # Overall goodput: both latency AND budget
        overall_good = sum(
            1 for d in details
            if d.get("success", True)
            and d.get("ttft", 99) * 1000 < ttft_slo_ms
            and d.get("output_len", 0) <= budget_tokens
        )
        m["overall_goodput"] = overall_good / len(details) if details else 0

        # Quality metrics (if available from ROUTE_BALANCE scheduler)
        qualities = [d.get("predicted_quality", 0) for d in details
                     if d.get("predicted_quality", 0) > 0]
        if qualities:
            m["avg_predicted_quality"] = float(np.mean(qualities))

        best_hits = [d.get("predicted_best_hit", False) for d in details
                     if "predicted_best_hit" in d]
        if best_hits:
            m["predicted_best_model_hit_rate"] = sum(best_hits) / len(best_hits)

        # Overhead breakdown (if available from ROUTE_BALANCE scheduler)
        overheads = [
            d.get("scheduling_overhead_breakdown", {})
            for d in details if d.get("scheduling_overhead_breakdown")
        ]
        if overheads:
            m["mean_batch_wait_ms"] = np.mean([o["batch_wait_ms"] for o in overheads])
            m["mean_estimator_ms"] = np.mean([o["estimator_ms"] for o in overheads])
            m["mean_scoring_ms"] = np.mean([o["scoring_ms"] for o in overheads])
            m["mean_total_sched_ms"] = np.mean([o["total_scheduling_ms"] for o in overheads])
            m["mean_batch_size"] = np.mean([o["batch_size"] for o in overheads])

    return m


def print_comparison_table(all_metrics: Dict[str, dict]):
    """Print a comparison table."""
    print("\n" + "=" * 110)
    print("ROUTE_BALANCE SMOKE TEST RESULTS")
    print("=" * 110)

    header = (
        f"{'Run':<25} {'N':>4} {'OK%':>5} {'Thpt':>6} "
        f"{'TTFT_avg':>9} {'TTFT_p95':>9} {'E2E_avg':>9} {'E2E_p95':>9} "
        f"{'LatGP':>6} {'BgtGP':>6} "
        f"{'Models':>25}"
    )
    print(header)
    print("-" * 130)

    for name, m in sorted(all_metrics.items()):
        if m["num_requests"] == 0:
            continue
        models_str = ""
        if m.get("model_distribution"):
            models_str = ", ".join(
                f"{k.split('/')[-1]}:{v}"
                for k, v in sorted(m["model_distribution"].items())
            )[:25]

        success_pct = m.get("success_rate", 1.0) * 100
        lat_gp = m.get("latency_slo_goodput", 0) * 100
        bgt_gp = m.get("budget_goodput", 0) * 100

        print(
            f"  {name:<23} {m['num_requests']:>4} {success_pct:>4.0f}% "
            f"{m['request_throughput']:>6.2f} "
            f"{m['mean_ttft']:>8.1f}ms {m['p95_ttft']:>8.1f}ms "
            f"{m['mean_e2e']:>8.1f}ms {m['p95_e2e']:>8.1f}ms "
            f"{lat_gp:>5.0f}% {bgt_gp:>5.0f}% "
            f"{models_str:>25}"
        )

    # Print overhead breakdown for ROUTE_BALANCE runs
    route_balance_runs = {n: m for n, m in all_metrics.items()
                 if m.get("mean_total_sched_ms", 0) > 0}
    if route_balance_runs:
        print("\n--- ROUTE_BALANCE Scheduling Overhead Breakdown ---")
        print(f"{'Run':<25} {'BatchWait':>10} {'Estimator':>10} {'Scoring':>10} {'Total':>10} {'BatchSz':>8}")
        print("-" * 75)
        for name, m in sorted(route_balance_runs.items()):
            print(
                f"  {name:<23} "
                f"{m.get('mean_batch_wait_ms',0):>9.1f}ms "
                f"{m.get('mean_estimator_ms',0):>9.1f}ms "
                f"{m.get('mean_scoring_ms',0):>9.1f}ms "
                f"{m.get('mean_total_sched_ms',0):>9.1f}ms "
                f"{m.get('mean_batch_size',0):>7.1f}"
            )

    print("=" * 110)


def generate_plots(all_metrics: Dict[str, dict], output_dir: str):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Group by QPS
    by_qps = {}
    for name, m in all_metrics.items():
        qps = str(m.get("qps", "?"))
        if qps not in by_qps:
            by_qps[qps] = {}
        by_qps[qps][m["scheduler"]] = m

    # Plot 1: TTFT comparison bar chart
    fig, axes = plt.subplots(1, len(by_qps), figsize=(6 * len(by_qps), 5))
    if len(by_qps) == 1:
        axes = [axes]

    for ax, (qps, schedulers) in zip(axes, sorted(by_qps.items())):
        names = sorted(schedulers.keys())
        avg_ttfts = [schedulers[n]["mean_ttft"] for n in names]
        p95_ttfts = [schedulers[n]["p95_ttft"] for n in names]

        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width / 2, avg_ttfts, width, label="Mean TTFT", alpha=0.8)
        ax.bar(x + width / 2, p95_ttfts, width, label="P95 TTFT", alpha=0.8)
        ax.set_xlabel("Scheduler")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title(f"TTFT @ QPS={qps}")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ttft_comparison.png"), dpi=150)
    print(f"Saved: {output_dir}/ttft_comparison.png")
    plt.close()

    # Plot 2: E2E latency comparison
    fig, axes = plt.subplots(1, len(by_qps), figsize=(6 * len(by_qps), 5))
    if len(by_qps) == 1:
        axes = [axes]

    for ax, (qps, schedulers) in zip(axes, sorted(by_qps.items())):
        names = sorted(schedulers.keys())
        avg_e2e = [schedulers[n]["mean_e2e"] for n in names]
        p95_e2e = [schedulers[n]["p95_e2e"] for n in names]

        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width / 2, avg_e2e, width, label="Mean E2E", alpha=0.8)
        ax.bar(x + width / 2, p95_e2e, width, label="P95 E2E", alpha=0.8)
        ax.set_xlabel("Scheduler")
        ax.set_ylabel("E2E Latency (ms)")
        ax.set_title(f"E2E Latency @ QPS={qps}")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "e2e_comparison.png"), dpi=150)
    print(f"Saved: {output_dir}/e2e_comparison.png")
    plt.close()

    # Plot 3: Model distribution (pie charts for route_balance vs random)
    route_balance_metrics = {n: m for n, m in all_metrics.items() if m["scheduler"] == "route_balance"}
    random_metrics = {n: m for n, m in all_metrics.items() if m["scheduler"] == "random"}

    if route_balance_metrics and random_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (label, metrics) in zip(axes, [
            ("ROUTE_BALANCE", next(iter(route_balance_metrics.values()))),
            ("Random", next(iter(random_metrics.values()))),
        ]):
            dist = metrics.get("model_distribution", {})
            if dist:
                labels = [k.split("/")[-1] for k in dist.keys()]
                sizes = list(dist.values())
                ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90)
                ax.set_title(f"Model Distribution ({label})")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_distribution.png"), dpi=150)
        print(f"Saved: {output_dir}/model_distribution.png")
        plt.close()

    # Plot 4: Throughput comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    for qps, schedulers in sorted(by_qps.items()):
        names = sorted(schedulers.keys())
        thpts = [schedulers[n]["request_throughput"] for n in names]
        ax.bar(
            [f"{n}\nQPS={qps}" for n in names],
            thpts,
            alpha=0.8,
            label=f"QPS={qps}",
        )
    ax.set_ylabel("Request Throughput (req/s)")
    ax.set_title("Throughput Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_comparison.png"), dpi=150)
    print(f"Saved: {output_dir}/throughput_comparison.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze ROUTE_BALANCE smoke test results")
    parser.add_argument("--result-dir", required=True, help="Directory with JSON results")
    parser.add_argument("--plot-dir", default=None, help="Directory for plots (default: result-dir/plots)")
    args = parser.parse_args()

    results = load_results(args.result_dir)
    if not results:
        print(f"No results found in {args.result_dir}")
        sys.exit(1)

    print(f"Loaded {len(results)} result files")

    # Extract metrics, inferring scheduler/qps from filename
    all_metrics = {}
    for name, data in results.items():
        m = extract_metrics(data)
        # Infer from filename: e.g. "random_qps2" -> scheduler=random, qps=2
        if m["scheduler"] == "unknown" and "_qps" in name:
            parts = name.rsplit("_qps", 1)
            m["scheduler"] = parts[0]
            m["qps"] = parts[1] if len(parts) > 1 else m["qps"]
        all_metrics[name] = m

    # Print comparison table
    print_comparison_table(all_metrics)

    # Generate plots
    plot_dir = args.plot_dir or os.path.join(args.result_dir, "plots")
    generate_plots(all_metrics, plot_dir)

    # Save summary
    summary_path = os.path.join(args.result_dir, "summary.json")
    with open(summary_path, "w") as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(
            {k: {kk: convert(vv) for kk, vv in v.items()} for k, v in all_metrics.items()},
            f, indent=2, default=str,
        )
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
