#!/usr/bin/env python3
"""Generate budget-annotated test datasets for T3 budget control experiments.

For each prompt sampled at fraction f ∈ {10, 25, 50, 100}%:
  budget_cost = uniform(min_cost, max_cost)
  where:
    min_cost = (input_tokens × 3B_in_price + predicted_out_tokens × 3B_out_price) / 1e6
    max_cost = (input_tokens × 72B_in_price + predicted_out_tokens × 72B_out_price) / 1e6
Non-sampled prompts get no `budget_cost` field (route_balance treats as unconstrained).

Output: data/route_balance/best-route-v3-test-3534-eval-budget-{f}.jsonl
        (one file per fraction; same prompt order as input; budget_cost set on a
        random subset of size f%, others unchanged).

Pricing per `route_balance/config/route_balance/model_deployment.json`:
  Qwen2.5-3B  : $0.06 / 1M (in & out)
  Qwen2.5-72B : $0.38 / 1M in, $0.40 / 1M out
"""
from __future__ import annotations

import json
import random
import argparse
from pathlib import Path


PRICE_3B_IN  = 0.06 / 1_000_000
PRICE_3B_OUT = 0.06 / 1_000_000
PRICE_72B_IN  = 0.38 / 1_000_000
PRICE_72B_OUT = 0.40 / 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",
        default="data/route_balance/best-route-v3-test-3534-eval.jsonl",
        help="Source bench dataset (one JSON per line)")
    ap.add_argument("--out-dir", default="data/route_balance")
    ap.add_argument("--fractions", default="10,25,50,100",
        help="Comma-separated percentages of prompts to budget-annotate")
    ap.add_argument("--predicted-output-default", type=int, default=128,
        help="Fallback output-length estimate when prompt has no length hint")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.input)
    rows = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    n = len(rows)
    print(f"Loaded {n} prompts from {src}")

    fractions = [int(x) for x in args.fractions.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in fractions:
        n_sample = (n * f) // 100
        # Use a per-fraction RNG so different fractions are nested subsets where possible.
        local_rng = random.Random(args.seed + f)
        idxs = set(local_rng.sample(range(n), n_sample))
        out_path = out_dir / f"best-route-v3-test-3534-eval-budget-{f}.jsonl"
        n_budgeted = 0
        with open(out_path, "w") as out:
            for i, row in enumerate(rows):
                row_out = dict(row)  # shallow copy, preserves all fields
                if i in idxs:
                    # Estimate input tokens from prompt text (rough char/4 heuristic).
                    prompt_text = row.get("prompt", "")
                    if isinstance(prompt_text, list):
                        prompt_text = " ".join(prompt_text)
                    in_toks = max(1, len(prompt_text) // 4)
                    out_toks_est = args.predicted_output_default
                    min_cost = in_toks * PRICE_3B_IN + out_toks_est * PRICE_3B_OUT
                    max_cost = in_toks * PRICE_72B_IN + out_toks_est * PRICE_72B_OUT
                    if max_cost <= min_cost:
                        max_cost = min_cost * 2
                    budget = local_rng.uniform(min_cost, max_cost)
                    row_out["budget_cost"] = round(budget, 8)
                    row_out["budget_pct"] = f
                    n_budgeted += 1
                out.write(json.dumps(row_out) + "\n")
        print(f"  f={f}%: wrote {n_budgeted} budgeted of {n} → {out_path}")

    print("done")


if __name__ == "__main__":
    main()
