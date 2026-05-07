#!/usr/bin/env python3
"""
Aggregate per-node monitor CSVs into cluster-wide metrics.

Reads all <node_id>.csv files from a monitor output directory and produces:
1. A merged timeline CSV with all nodes aligned by timestamp
2. Cluster-wide summary statistics (mean, std, max per metric)
3. Per-interval cluster aggregates (mean/max across nodes at each timestep)

Usage:
    python3 aggregate_monitor.py --input-dir experiment_output/monitor \
        --output-dir experiment_output/monitor_aggregated
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_node_csv(path: str) -> list:
    """Load a per-node monitor CSV into a list of dicts."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            parsed = {}
            for k, v in row.items():
                if v == "":
                    parsed[k] = None
                    continue
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def aggregate_timeseries(all_nodes: dict) -> list:
    """Produce per-second cluster-wide aggregates.

    For each elapsed_s bucket, compute mean/max/min across all nodes
    for key metrics.
    """
    # Bucket all rows by elapsed_s (rounded to nearest interval)
    time_buckets = defaultdict(list)  # elapsed_s -> [(node_id, row)]
    for node_id, rows in all_nodes.items():
        for row in rows:
            t = row.get("elapsed_s")
            if t is None:
                continue
            # Round to nearest 5s bucket
            bucket = round(t / 5) * 5
            time_buckets[bucket].append((node_id, row))

    # Aggregate per bucket
    agg_metrics = [
        "gpu0_util_pct", "gpu0_mem_used_mb", "gpu0_mem_bw_pct",
        "kv_cache_util", "num_running", "num_waiting",
        "load_1m", "sys_mem_used_pct",
    ]
    sched_metrics = ["sched_cpu_pct", "sched_rss_mb"]

    agg_rows = []
    for t in sorted(time_buckets.keys()):
        entries = time_buckets[t]
        row = {"elapsed_s": t, "num_nodes": len(entries)}

        for metric in agg_metrics:
            vals = []
            for nid, r in entries:
                v = r.get(metric)
                if v is not None:
                    vals.append(float(v))

            if vals:
                row[f"{metric}_mean"] = round(statistics.mean(vals), 2)
                row[f"{metric}_max"] = round(max(vals), 2)
                if len(vals) > 1:
                    row[f"{metric}_std"] = round(statistics.stdev(vals), 2)
                else:
                    row[f"{metric}_std"] = 0.0
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_max"] = None
                row[f"{metric}_std"] = None

        # Scheduler metrics (only from scheduler node, no aggregation)
        for metric in sched_metrics:
            for nid, r in entries:
                v = r.get(metric)
                if v is not None:
                    row[metric] = round(float(v), 2)
                    break

        agg_rows.append(row)

    return agg_rows


def compute_summary(all_nodes: dict) -> dict:
    """Compute overall summary statistics across all nodes and time."""
    summary = {}

    metrics = [
        "gpu0_util_pct", "gpu0_mem_used_mb", "gpu0_mem_bw_pct",
        "kv_cache_util", "num_running", "num_waiting",
        "load_1m", "sys_mem_used_pct",
        "sched_cpu_pct", "sched_rss_mb",
    ]

    for metric in metrics:
        all_vals = []
        per_node = {}
        for node_id, rows in all_nodes.items():
            vals = [float(r[metric]) for r in rows
                    if r.get(metric) is not None]
            if vals:
                per_node[node_id] = {
                    "mean": round(statistics.mean(vals), 2),
                    "max": round(max(vals), 2),
                    "min": round(min(vals), 2),
                }
                all_vals.extend(vals)

        if all_vals:
            summary[metric] = {
                "cluster_mean": round(statistics.mean(all_vals), 2),
                "cluster_max": round(max(all_vals), 2),
                "cluster_min": round(min(all_vals), 2),
                "cluster_std": round(statistics.stdev(all_vals), 2) if len(all_vals) > 1 else 0.0,
                "per_node": per_node,
            }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-node monitor CSVs")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing per-node CSV files")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <input-dir>/aggregated)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all node CSVs
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        sys.exit(1)

    all_nodes = {}
    for fpath in csv_files:
        if fpath.parent.name == "aggregated":
            continue  # Skip our own output
        node_id = fpath.stem
        rows = load_node_csv(str(fpath))
        all_nodes[node_id] = rows
        print(f"  Loaded {node_id}: {len(rows)} samples")

    print(f"\n{len(all_nodes)} nodes loaded from {input_dir}")

    # 1. Cluster-wide timeseries
    agg_ts = aggregate_timeseries(all_nodes)
    ts_path = output_dir / "cluster_timeseries.csv"
    if agg_ts:
        fieldnames = list(agg_ts[0].keys())
        with open(ts_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(agg_ts)
        print(f"\nCluster timeseries: {ts_path} ({len(agg_ts)} intervals)")

    # 2. Summary statistics
    summary = compute_summary(all_nodes)
    summary_path = output_dir / "cluster_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Cluster summary: {summary_path}")

    # 3. Print key metrics
    print(f"\n{'='*60}")
    print("CLUSTER RESOURCE SUMMARY")
    print(f"{'='*60}")
    fmt = "  {:<25s} mean={:<8s} max={:<8s} std={:<8s}"
    for metric in ["gpu0_util_pct", "gpu0_mem_used_mb", "kv_cache_util",
                    "num_running", "load_1m", "sys_mem_used_pct",
                    "sched_cpu_pct", "sched_rss_mb"]:
        s = summary.get(metric, {})
        if s:
            print(fmt.format(
                metric,
                f"{s['cluster_mean']:.1f}",
                f"{s['cluster_max']:.1f}",
                f"{s['cluster_std']:.1f}",
            ))
            # Per-node breakdown
            for nid, ns in s.get("per_node", {}).items():
                print(f"    {nid}: mean={ns['mean']:.1f} max={ns['max']:.1f}")

    print(f"\nOutputs in: {output_dir}/")


if __name__ == "__main__":
    main()
