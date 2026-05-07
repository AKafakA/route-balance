#!/usr/bin/env python3
"""
Per-node resource monitor for RouteBalance experiments.

Run on EACH node independently. Collects:
1. Local GPU metrics (via pynvml — ~100μs, no process spawn)
2. Local vLLM /metrics (KV cache, queue depth — zero GPU overhead)
3. System CPU load + memory
4. Optional: scheduler/predictor process CPU + RSS (if --scheduler-pid or --predictor-pid)

Each node produces its own CSV. Aggregate during analysis.

Usage (on vLLM node):
    python3 monitor.py --node-id node0 --vllm-port 8000 --interval 5

Usage (on scheduler node):
    python3 monitor.py --node-id scheduler --vllm-port 8000 \
        --scheduler-pid <PID> --predictor-pid <PID> \
        --interval 5
"""

import argparse
import csv
import os
import time
from datetime import datetime


def parse_vllm_metrics(text: str) -> dict:
    """Parse Prometheus-format /metrics response from vLLM."""
    metrics = {}
    for line in text.strip().split("\n"):
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0]
            try:
                val = float(parts[1])
            except ValueError:
                continue
            if "gpu_cache_usage_perc" in key:
                metrics["kv_cache_util"] = val
            elif "num_requests_running" in key:
                metrics["num_running"] = val
            elif "num_requests_waiting" in key:
                metrics["num_waiting"] = val
            elif "num_preemptions_total" in key:
                metrics["preemptions"] = val
    return metrics


def get_gpu_stats_nvml() -> list:
    """Get GPU stats via pynvml (no process spawn, ~100μs overhead)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = 0
            gpus.append({
                "mem_used_mb": mem.used / 1024 / 1024,
                "mem_total_mb": mem.total / 1024 / 1024,
                "gpu_util_pct": util.gpu,
                "mem_bw_util_pct": util.memory,
                "temp_c": temp,
            })
        return gpus
    except Exception:
        return _get_gpu_stats_nvidia_smi_fallback()


def _get_gpu_stats_nvidia_smi_fallback() -> list:
    """Fallback: nvidia-smi subprocess."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "mem_used_mb": float(parts[1]),
                    "mem_total_mb": float(parts[2]),
                    "gpu_util_pct": float(parts[3]),
                    "temp_c": float(parts[4]),
                })
        return gpus
    except Exception:
        return []


def get_system_stats() -> dict:
    """System-wide CPU load and memory from /proc (zero overhead)."""
    stats = {}
    try:
        with open("/proc/loadavg") as f:
            load = f.read().split()
            stats["load_1m"] = float(load[0])
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
            total = meminfo.get("MemTotal", 1)
            avail = meminfo.get("MemAvailable", 0)
            stats["sys_mem_used_pct"] = round(100.0 * (1 - avail / total), 1)
            stats["sys_mem_used_gb"] = round((total - avail) / 1024 / 1024, 2)
    except Exception:
        pass
    return stats


def get_process_stats(pid: int) -> dict:
    """CPU ticks and RSS memory for a specific process from /proc."""
    stats = {}
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
            stats["cpu_ticks"] = int(parts[13]) + int(parts[14])
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    stats["rss_mb"] = round(int(line.split()[1]) / 1024, 1)
                elif line.startswith("Threads:"):
                    stats["threads"] = int(line.split()[1])
    except (FileNotFoundError, IndexError, ValueError):
        pass
    return stats


def fetch_url(url: str, timeout: float = 2) -> str:
    """Simple HTTP GET."""
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode()
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Per-node RouteBalance monitor")
    parser.add_argument("--node-id", required=True,
                        help="Node identifier (e.g., node0, node1, scheduler)")
    parser.add_argument("--vllm-port", type=int, default=8000,
                        help="Local vLLM port for /metrics")
    parser.add_argument("--scheduler-pid", type=int, default=0,
                        help="Scheduler process PID (if this is the scheduler node)")
    parser.add_argument("--predictor-pid", type=int, default=0,
                        help="Predictor sidecar PID (if running on this node)")
    parser.add_argument("--interval", type=float, default=5,
                        help="Polling interval in seconds")
    parser.add_argument("--output-dir", default="experiment_output/monitor",
                        help="Output directory for CSV files")
    parser.add_argument("--duration", type=int, default=0,
                        help="Max duration in seconds (0=unlimited)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.node_id}.csv")

    # Build CSV header based on what this node collects
    fields = ["timestamp", "elapsed_s", "load_1m", "sys_mem_used_pct", "sys_mem_used_gb"]

    # Local vLLM metrics
    fields.extend(["kv_cache_util", "num_running", "num_waiting", "preemptions"])

    # Local GPU metrics (per GPU)
    gpus_initial = get_gpu_stats_nvml()
    num_gpus = len(gpus_initial) or 1
    for g in range(num_gpus):
        fields.extend([f"gpu{g}_mem_used_mb", f"gpu{g}_mem_total_mb",
                        f"gpu{g}_util_pct", f"gpu{g}_mem_bw_pct", f"gpu{g}_temp"])

    # Optional process tracking
    if args.scheduler_pid:
        fields.extend(["sched_cpu_pct", "sched_rss_mb", "sched_threads"])
    if args.predictor_pid:
        fields.extend(["pred_cpu_pct", "pred_rss_mb"])

    hz = os.sysconf("SC_CLK_TCK")
    prev_sched_ticks = 0
    prev_pred_ticks = 0
    prev_time = time.time()

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

        start_time = time.time()
        print(f"[{args.node_id}] Monitor started: vllm=:{args.vllm_port}, "
              f"gpus={num_gpus}, interval={args.interval}s")
        if args.scheduler_pid:
            print(f"[{args.node_id}] Tracking scheduler PID {args.scheduler_pid}")
        if args.predictor_pid:
            print(f"[{args.node_id}] Tracking predictor PID {args.predictor_pid}")
        print(f"[{args.node_id}] Output: {output_path}")

        while True:
            now = time.time()
            elapsed = now - start_time
            dt = max(now - prev_time, 0.001)
            prev_time = now

            if args.duration and elapsed > args.duration:
                break

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "elapsed_s": round(elapsed, 1),
            }

            # System stats
            row.update(get_system_stats())

            # Local vLLM /metrics (no GPU driver overhead)
            text = fetch_url(f"http://localhost:{args.vllm_port}/metrics")
            if text:
                m = parse_vllm_metrics(text)
                row["kv_cache_util"] = round(m.get("kv_cache_util", 0), 3)
                row["num_running"] = int(m.get("num_running", 0))
                row["num_waiting"] = int(m.get("num_waiting", 0))
                row["preemptions"] = int(m.get("preemptions", 0))

            # Local GPU stats (NVML — ~100μs per GPU)
            gpus = get_gpu_stats_nvml()
            for g, gpu in enumerate(gpus):
                row[f"gpu{g}_mem_used_mb"] = round(gpu["mem_used_mb"], 0)
                row[f"gpu{g}_mem_total_mb"] = round(gpu["mem_total_mb"], 0)
                row[f"gpu{g}_util_pct"] = gpu["gpu_util_pct"]
                row[f"gpu{g}_mem_bw_pct"] = gpu.get("mem_bw_util_pct", 0)
                row[f"gpu{g}_temp"] = gpu.get("temp_c", 0)

            # Scheduler process (if this is the scheduler node)
            if args.scheduler_pid:
                ps = get_process_stats(args.scheduler_pid)
                ticks = ps.get("cpu_ticks", 0)
                cpu_pct = (ticks - prev_sched_ticks) / (dt * hz) * 100
                prev_sched_ticks = ticks
                row["sched_cpu_pct"] = round(min(cpu_pct, 100 * os.cpu_count()), 1)
                row["sched_rss_mb"] = ps.get("rss_mb", 0)
                row["sched_threads"] = ps.get("threads", 0)

            # Predictor sidecar (if running on this node)
            if args.predictor_pid:
                ps = get_process_stats(args.predictor_pid)
                ticks = ps.get("cpu_ticks", 0)
                cpu_pct = (ticks - prev_pred_ticks) / (dt * hz) * 100
                prev_pred_ticks = ticks
                row["pred_cpu_pct"] = round(min(cpu_pct, 100 * os.cpu_count()), 1)
                row["pred_rss_mb"] = ps.get("rss_mb", 0)

            writer.writerow(row)
            csvfile.flush()
            time.sleep(args.interval)

    print(f"[{args.node_id}] Monitor stopped after {elapsed:.0f}s → {output_path}")


if __name__ == "__main__":
    main()
