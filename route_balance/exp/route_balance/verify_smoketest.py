#!/usr/bin/env python3
"""Verify comprehensive smoke test results for completeness and sanity."""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_result(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def check_basic(name: str, data: dict) -> list:
    """Check basic metrics exist and are reasonable."""
    issues = []
    # Check success rate
    details = data.get("response_details", [])
    if not details:
        issues.append(f"no response_details")
        return issues
    success = sum(1 for d in details if not d.get("error"))
    total = len(details)
    rate = success / max(total, 1)
    if rate < 0.8:
        issues.append(f"low success rate: {success}/{total} ({rate:.0%})")

    # Check latency metrics exist
    for key in ["mean_ttft_ms", "mean_e2el_ms"]:
        val = data.get(key, 0)
        if val <= 0:
            issues.append(f"{key} = {val}")

    return issues


def check_model_distribution(name: str, data: dict) -> dict:
    """Return model distribution from response details."""
    details = data.get("response_details", [])
    models = defaultdict(int)
    for d in details:
        m = d.get("model", "unknown")
        models[m] += 1
    return dict(models)


def check_scheduling_detail(name: str, data: dict) -> list:
    """Check ROUTE_BALANCE-specific fields in response details."""
    issues = []
    details = data.get("response_details", [])
    if not details:
        return issues

    sample = details[0]
    expected_fields = ["predicted_quality", "predicted_best_model", "model",
                       "scheduling_overhead_breakdown"]
    for field in expected_fields:
        if field not in sample:
            issues.append(f"missing field: {field}")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Error: {result_dir} not found")
        sys.exit(1)

    files = sorted(result_dir.glob("*.json"))
    if not files:
        print(f"No result files in {result_dir}")
        sys.exit(1)

    print(f"Verifying {len(files)} result files in {result_dir}/\n")

    phases = defaultdict(list)
    total_pass = 0
    total_fail = 0

    for fpath in files:
        name = fpath.stem
        phase = name.split("_")[0]  # p1, p2, etc.

        try:
            data = load_result(str(fpath))
        except (json.JSONDecodeError, Exception) as e:
            print(f"  FAIL {name}: invalid JSON ({e})")
            phases[phase].append((name, "FAIL", f"invalid JSON: {e}"))
            total_fail += 1
            continue

        issues = check_basic(name, data)

        # Phase-specific checks
        if phase == "p2":
            dist = check_model_distribution(name, data)
            if dist:
                print(f"  {name}: models={dist}")

        if "route_balance" in name or phase in ("p2", "p3", "p4", "p5", "p6", "p7"):
            route_balance_issues = check_scheduling_detail(name, data)
            issues.extend(route_balance_issues)

        if issues:
            status = "FAIL"
            total_fail += 1
            print(f"  FAIL {name}: {'; '.join(issues)}")
        else:
            status = "PASS"
            total_pass += 1
            # Compact output for passing tests
            throughput = data.get("request_throughput", 0)
            ttft = data.get("mean_ttft_ms", 0)
            print(f"  PASS {name} (throughput={throughput:.1f} req/s, ttft={ttft:.0f}ms)")

        phases[phase].append((name, status, "; ".join(issues) if issues else "ok"))

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {total_pass} PASS, {total_fail} FAIL, {len(files)} total")
    print(f"{'='*60}")
    for phase in sorted(phases.keys()):
        results = phases[phase]
        p = sum(1 for _, s, _ in results if s == "PASS")
        f = sum(1 for _, s, _ in results if s == "FAIL")
        print(f"  {phase}: {p} pass, {f} fail ({len(results)} runs)")

    # Phase 2 weight sweep: check model distribution shifts
    if "p2" in phases:
        print(f"\n--- Weight Sweep Model Distribution ---")
        for fpath in files:
            if fpath.stem.startswith("p2_"):
                data = load_result(str(fpath))
                dist = check_model_distribution(fpath.stem, data)
                print(f"  {fpath.stem}: {dist}")

    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
