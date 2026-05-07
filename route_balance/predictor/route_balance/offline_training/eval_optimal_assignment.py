#!/usr/bin/env python3
"""
Brute-force optimal assignment evaluator for RouteBalance.

For small batches (|B| <= 8 with few instances), enumerate ALL possible
assignments, simulate queue cascading effects, and find the true optimal.
Compare against RouteBalance's greedy to measure competitive ratio.

Only feasible for:
  - 2 instances: batch <= 20 (2^20 = 1M)
  - 5 instances: batch <= 8  (5^8 = 390K)
  - 18 instances: batch <= 4  (18^4 = 105K)

Usage:
    python -m route_balance.predictor.route_balance.offline_training.eval_optimal_assignment \
        --trace experiment_output/smoketest_full/route_balance_fused_qps5_trace.json
"""

import argparse
import itertools
import json
import logging
import time
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def simulate_assignment(
    requests: List[dict],
    instances: List[dict],
    assignment: Tuple[int, ...],
    weights: dict,
) -> Tuple[float, dict]:
    """Simulate a batch assignment with queue cascading.

    Args:
        requests: list of {prompt_tokens, predicted_output_tokens, quality: {inst_id: score}, cost: {inst_id: cost}}
        instances: list of {id, base_ttft, base_tpot, initial_queue_depth, prefill_rate, capacity}
        assignment: tuple of instance indices, one per request
        weights: {w_lat, w_cost, w_qual, w_bal}

    Returns:
        (total_objective, details_dict)
    """
    n_instances = len(instances)

    # Copy initial states
    queue_tokens = [inst.get("initial_pending_tokens", 0) for inst in instances]
    queue_count = [inst.get("initial_queue_depth", 0) for inst in instances]

    total_lat = 0.0
    total_cost = 0.0
    total_qual = 0.0
    per_request = []

    for r_idx, inst_idx in enumerate(assignment):
        req = requests[r_idx]
        inst = instances[inst_idx]
        inst_id = inst["id"]

        # Estimate latency with current queue state
        prompt_tokens = req["prompt_tokens"]
        output_tokens = req["predicted_output_tokens"]
        prefill_rate = inst.get("prefill_rate", 2000)  # tokens/sec
        tpot = inst.get("base_tpot", 0.03)  # sec/token

        # Queue wait: pending tokens / prefill_rate (simplified)
        queue_wait = queue_tokens[inst_idx] / max(prefill_rate, 1)
        prefill_time = prompt_tokens / max(prefill_rate, 1)
        decode_time = output_tokens * tpot
        e2e = queue_wait + prefill_time + decode_time

        # Quality and cost
        quality = req.get("quality", {}).get(inst_id, 0.5)
        cost_per_token = inst.get("cost_per_token", 0.01)
        cost = output_tokens * cost_per_token

        total_lat += e2e
        total_cost += cost
        total_qual += quality

        per_request.append({
            "request_idx": r_idx,
            "instance_idx": inst_idx,
            "instance_id": inst_id,
            "e2e": e2e,
            "quality": quality,
            "cost": cost,
        })

        # Update queue state (cascading effect)
        queue_tokens[inst_idx] += prompt_tokens + output_tokens
        queue_count[inst_idx] += 1

    n = len(requests)
    w = weights

    # Compute balance penalty
    utils = [queue_count[i] / max(instances[i].get("capacity", 8), 1) for i in range(n_instances)]
    mean_util = np.mean(utils)
    balance_penalty = sum(max(0, u - mean_util) for u in utils) / max(mean_util, 1e-8)

    # Objective (lower is better)
    objective = (
        w.get("w_lat", 0.3) * total_lat / n
        + w.get("w_cost", 0.2) * total_cost / n
        - w.get("w_qual", 0.3) * total_qual / n
        + w.get("w_bal", 0.2) * balance_penalty
    )

    details = {
        "objective": objective,
        "mean_e2e": total_lat / n,
        "mean_cost": total_cost / n,
        "mean_quality": total_qual / n,
        "balance_penalty": balance_penalty,
        "assignment": list(assignment),
        "per_request": per_request,
    }
    return objective, details


def find_optimal(
    requests: List[dict],
    instances: List[dict],
    weights: dict,
    max_assignments: int = 1_000_000,
) -> Tuple[dict, dict]:
    """Brute-force find optimal assignment.

    Returns (optimal_details, stats)
    """
    n_req = len(requests)
    n_inst = len(instances)
    total_assignments = n_inst ** n_req

    if total_assignments > max_assignments:
        logger.warning(
            f"Too many assignments: {n_inst}^{n_req} = {total_assignments:,}. "
            f"Max allowed: {max_assignments:,}. Skipping."
        )
        return None, {"skipped": True, "total_assignments": total_assignments}

    logger.info(f"Enumerating {total_assignments:,} assignments ({n_req} requests × {n_inst} instances)")

    best_obj = float("inf")
    best_details = None
    worst_obj = float("-inf")
    all_objectives = []

    t0 = time.time()
    for i, assignment in enumerate(itertools.product(range(n_inst), repeat=n_req)):
        obj, details = simulate_assignment(requests, instances, assignment, weights)
        all_objectives.append(obj)

        if obj < best_obj:
            best_obj = obj
            best_details = details

        if obj > worst_obj:
            worst_obj = obj

        if (i + 1) % 100000 == 0:
            elapsed = time.time() - t0
            logger.info(f"  {i+1:,}/{total_assignments:,} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    all_objectives = np.array(all_objectives)

    stats = {
        "total_assignments": total_assignments,
        "elapsed_s": elapsed,
        "best_objective": best_obj,
        "worst_objective": worst_obj,
        "mean_objective": float(np.mean(all_objectives)),
        "std_objective": float(np.std(all_objectives)),
        "p5_objective": float(np.percentile(all_objectives, 5)),
        "p95_objective": float(np.percentile(all_objectives, 95)),
    }

    logger.info(f"Optimal: {best_obj:.4f}, Worst: {worst_obj:.4f}, "
                f"Mean: {np.mean(all_objectives):.4f} ({elapsed:.1f}s)")

    return best_details, stats


def evaluate_greedy_vs_optimal(
    requests: List[dict],
    instances: List[dict],
    greedy_assignment: List[int],
    weights: dict,
) -> dict:
    """Compare greedy assignment against brute-force optimal."""
    # Evaluate greedy
    greedy_obj, greedy_details = simulate_assignment(
        requests, instances, tuple(greedy_assignment), weights
    )

    # Find optimal
    optimal_details, enum_stats = find_optimal(requests, instances, weights)

    if optimal_details is None:
        return {"skipped": True, **enum_stats}

    competitive_ratio = greedy_obj / optimal_details["objective"] if optimal_details["objective"] != 0 else 1.0

    result = {
        "greedy_objective": greedy_obj,
        "optimal_objective": optimal_details["objective"],
        "competitive_ratio": competitive_ratio,
        "greedy_assignment": greedy_assignment,
        "optimal_assignment": optimal_details["assignment"],
        "same_assignment": greedy_assignment == optimal_details["assignment"],
        "greedy_e2e": greedy_details["mean_e2e"],
        "optimal_e2e": optimal_details["mean_e2e"],
        "greedy_quality": greedy_details["mean_quality"],
        "optimal_quality": optimal_details["mean_quality"],
        **enum_stats,
    }

    logger.info(f"Greedy: {greedy_obj:.4f}, Optimal: {optimal_details['objective']:.4f}, "
                f"Ratio: {competitive_ratio:.4f}, Same: {result['same_assignment']}")

    return result


def compute_pareto_frontier(points: List[Tuple[float, float]], minimize_both=True) -> List[int]:
    """Find indices of Pareto-optimal points in 2D.

    Args:
        points: list of (x, y) tuples
        minimize_both: if True, both objectives are minimized.
                       For (latency, -quality), set True.
    Returns:
        list of indices into points that are on the Pareto frontier
    """
    n = len(points)
    is_dominated = [False] * n
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # j dominates i if j is <= on all objectives and < on at least one
            if minimize_both:
                if points[j][0] <= points[i][0] and points[j][1] <= points[i][1]:
                    if points[j][0] < points[i][0] or points[j][1] < points[i][1]:
                        is_dominated[i] = True
                        break
    return [i for i in range(n) if not is_dominated[i]]


def pareto_analysis(
    requests: List[dict],
    instances: List[dict],
    weights: dict,
    greedy_assignment: List[int],
    max_assignments: int = 1_000_000,
    sample_size: int = 10_000,
) -> dict:
    """Compute Pareto frontier and greedy position.

    For each assignment, compute 3 raw metrics (no normalization):
    - mean_e2e (lower is better)
    - mean_quality (higher is better → negate for minimization)
    - mean_cost (lower is better)

    Returns dict with frontier points, greedy position, and distance to frontier.
    """
    n_req = len(requests)
    n_inst = len(instances)
    total = n_inst ** n_req

    # Collect metrics for all (or sampled) assignments
    all_metrics = []  # (mean_e2e, mean_quality, mean_cost, assignment)

    if total <= max_assignments:
        # Enumerate all
        generator = itertools.product(range(n_inst), repeat=n_req)
        n_eval = total
    else:
        # Random sample
        def random_assignments(n, k, count):
            for _ in range(count):
                yield tuple(np.random.randint(0, k, size=n))
        generator = random_assignments(n_req, n_inst, sample_size)
        n_eval = sample_size
        logger.info(f"Sampling {sample_size} of {total:,} assignments")

    t0 = time.time()
    for idx, assignment in enumerate(generator):
        _, details = simulate_assignment(requests, instances, assignment, weights)
        all_metrics.append((
            details["mean_e2e"],
            details["mean_quality"],
            details["mean_cost"],
            list(assignment),
        ))
        if (idx + 1) % 100000 == 0:
            logger.info(f"  {idx+1:,}/{n_eval:,} ({time.time()-t0:.1f}s)")

    # Greedy metrics
    _, greedy_details = simulate_assignment(
        requests, instances, tuple(greedy_assignment), weights
    )
    greedy_point = (
        greedy_details["mean_e2e"],
        greedy_details["mean_quality"],
        greedy_details["mean_cost"],
    )

    # 2D Pareto frontiers (3 projections)
    # Projection 1: latency vs quality (minimize latency, maximize quality)
    points_lq = [(m[0], -m[1]) for m in all_metrics]  # negate quality for minimization
    frontier_lq = compute_pareto_frontier(points_lq)

    # Projection 2: cost vs quality
    points_cq = [(m[2], -m[1]) for m in all_metrics]
    frontier_cq = compute_pareto_frontier(points_cq)

    # Projection 3: latency vs cost
    points_lc = [(m[0], m[2]) for m in all_metrics]
    frontier_lc = compute_pareto_frontier(points_lc)

    # Distance from greedy to Pareto frontier (latency-quality projection)
    frontier_lq_points = [(all_metrics[i][0], all_metrics[i][1]) for i in frontier_lq]
    greedy_lq = (greedy_point[0], greedy_point[1])

    # Normalized distance: how far is greedy from nearest frontier point?
    if frontier_lq_points:
        lat_range = max(m[0] for m in all_metrics) - min(m[0] for m in all_metrics)
        qual_range = max(m[1] for m in all_metrics) - min(m[1] for m in all_metrics)
        min_dist = min(
            np.sqrt(((greedy_lq[0] - fp[0]) / max(lat_range, 1e-8)) ** 2 +
                     ((greedy_lq[1] - fp[1]) / max(qual_range, 1e-8)) ** 2)
            for fp in frontier_lq_points
        )
    else:
        min_dist = 0.0

    # Is greedy on the frontier?
    greedy_on_frontier = any(
        abs(greedy_lq[0] - fp[0]) < 1e-6 and abs(greedy_lq[1] - fp[1]) < 1e-6
        for fp in frontier_lq_points
    )

    result = {
        "n_assignments_evaluated": len(all_metrics),
        "n_total_assignments": total,
        "elapsed_s": time.time() - t0,
        "greedy": {
            "e2e": greedy_point[0],
            "quality": greedy_point[1],
            "cost": greedy_point[2],
        },
        "greedy_on_pareto_frontier": greedy_on_frontier,
        "distance_to_frontier": min_dist,
        "frontier_latency_quality": [
            {"e2e": all_metrics[i][0], "quality": all_metrics[i][1]}
            for i in frontier_lq
        ],
        "frontier_cost_quality": [
            {"cost": all_metrics[i][2], "quality": all_metrics[i][1]}
            for i in frontier_cq
        ],
        "frontier_latency_cost": [
            {"e2e": all_metrics[i][0], "cost": all_metrics[i][2]}
            for i in frontier_lc
        ],
        # Save all points for plotting (subsample if too many)
        "all_points": [
            {"e2e": m[0], "quality": m[1], "cost": m[2]}
            for m in all_metrics[::max(1, len(all_metrics) // 5000)]  # max 5K for plotting
        ],
    }

    logger.info(
        f"Pareto: {len(frontier_lq)} frontier points (lat-qual), "
        f"greedy on frontier: {greedy_on_frontier}, "
        f"distance: {min_dist:.4f}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Brute-force optimal assignment evaluator")
    parser.add_argument("--trace", required=True, help="Saved batch trace JSON")
    parser.add_argument("--max-batch", type=int, default=8, help="Max batch size to enumerate")
    parser.add_argument("--max-assignments", type=int, default=1_000_000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.trace) as f:
        trace = json.load(f)

    # Process each batch in the trace
    results = []
    for batch in trace.get("batches", []):
        requests = batch["requests"]
        instances = batch["instances"]
        greedy = batch["greedy_assignment"]
        weights = batch.get("weights", {"w_lat": 0.3, "w_cost": 0.2, "w_qual": 0.3, "w_bal": 0.2})

        if len(requests) > args.max_batch:
            logger.info(f"Batch size {len(requests)} > {args.max_batch}, skipping")
            continue

        result = evaluate_greedy_vs_optimal(requests, instances, greedy, weights)
        results.append(result)

    # Summary
    if results:
        valid = [r for r in results if not r.get("skipped")]
        if valid:
            ratios = [r["competitive_ratio"] for r in valid]
            same = sum(1 for r in valid if r["same_assignment"])
            print(f"\n=== Competitive Ratio Summary ({len(valid)} batches) ===")
            print(f"Mean: {np.mean(ratios):.4f}")
            print(f"Median: {np.median(ratios):.4f}")
            print(f"Min (worst): {np.max(ratios):.4f}")
            print(f"Optimal match: {same}/{len(valid)} ({100*same/len(valid):.0f}%)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
