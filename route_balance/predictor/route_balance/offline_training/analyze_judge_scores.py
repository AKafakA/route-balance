#!/usr/bin/env python3
"""
Analyze and compare reference-grounded LLM judge scores.

Generates a comprehensive report comparing Prometheus-v2 (1-5),
Llama-3.1-8B (1-10), and Qwen-2.5-7B (1-10) reference-grounded scores.

Usage:
    python -m route_balance.predictor.route_balance.offline_training.analyze_judge_scores \
        --input data/scored/test_all_judges.jsonl \
        --output reports/judge_analysis.md
"""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JUDGE_KEYS = [
    "prometheus-7b-v2_reference",
    "llama-3.1-8b-it_reference",
    "qwen2.5-7b-it_reference",
]

JUDGE_NAMES = {
    "prometheus-7b-v2_reference": "Prometheus-v2 (1-5)",
    "llama-3.1-8b-it_reference": "Llama-3.1-8B (1-10)",
    "qwen2.5-7b-it_reference": "Qwen-2.5-7B (1-10)",
}

EXISTING_SIGNALS = [
    ("reference_score", "Reference Score"),
    ("similarity_score", "Similarity (72B)"),
]

MODEL_ORDER = [
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-72B",
]


def load_scores(data: List[dict]) -> dict:
    """Extract all scores into structured arrays.

    Returns:
        {judge_key: {model: [scores]}}, plus existing signals
    """
    result = {jk: {m: [] for m in MODEL_ORDER} for jk in JUDGE_KEYS}
    existing = {sig: {m: [] for m in MODEL_ORDER} for sig, _ in EXISTING_SIGNALS}
    blind = {m: [] for m in MODEL_ORDER}

    for entry in data:
        for model in MODEL_ORDER:
            m_data = entry.get("models", {}).get(model, {})
            if not m_data:
                continue
            js = m_data.get("llm_judge_scores", {})

            for jk in JUDGE_KEYS:
                val = js.get(jk)
                result[jk][model].append(val)

            for sig, _ in EXISTING_SIGNALS:
                val = m_data.get(sig)
                existing[sig][model].append(val)

            # Blind judge (existing)
            blind_val = None
            for k, v in js.items():
                if k not in JUDGE_KEYS and "protectai" not in k and v is not None:
                    blind_val = v
                    break
            blind[model].append(blind_val)

    return result, existing, blind


def generate_report(data: List[dict], output_path: str):
    """Generate comprehensive judge comparison report."""
    scores, existing, blind = load_scores(data)

    lines = []
    lines.append("# Reference-Grounded LLM Judge Score Analysis")
    lines.append(f"\nDataset: {len(data)} entries × {len(MODEL_ORDER)} models")
    lines.append(f"Judges: {', '.join(JUDGE_NAMES.values())}")
    lines.append("")

    # ==========================================================
    # 1. Score Distribution per Judge
    # ==========================================================
    lines.append("## 1. Score Distribution per Judge")
    lines.append("")

    for jk in JUDGE_KEYS:
        name = JUDGE_NAMES[jk]
        all_scores = []
        for model in MODEL_ORDER:
            all_scores.extend([s for s in scores[jk][model] if s is not None])

        if not all_scores:
            lines.append(f"### {name}: NO SCORES")
            continue

        arr = np.array(all_scores)
        lines.append(f"### {name}")
        lines.append(f"- N scored: {len(all_scores)}")
        lines.append(f"- Mean: {arr.mean():.3f}, Median: {np.median(arr):.3f}, Std: {arr.std():.3f}")
        lines.append(f"- Min: {arr.min():.2f}, Max: {arr.max():.2f}")

        # Unique values (discreteness check)
        unique = sorted(set(np.round(all_scores, 2)))
        lines.append(f"- Unique values: {len(unique)} ({unique[:15]}{'...' if len(unique) > 15 else ''})")

        # Per-model means
        lines.append(f"- Per-model means:")
        for model in MODEL_ORDER:
            ms = model.split("/")[-1]
            vals = [s for s in scores[jk][model] if s is not None]
            if vals:
                a = np.array(vals)
                lines.append(f"  - {ms}: mean={a.mean():.3f}, std={a.std():.3f}, n={len(vals)}")
        lines.append("")

    # ==========================================================
    # 2. Cross-Judge Correlation
    # ==========================================================
    lines.append("## 2. Cross-Judge Correlation (Spearman ρ)")
    lines.append("")
    lines.append("All scores pooled across models:")
    lines.append("")

    # Build aligned score vectors
    judge_vectors = {}
    for jk in JUDGE_KEYS:
        vec = []
        for model in MODEL_ORDER:
            vec.extend(scores[jk][model])
        judge_vectors[jk] = vec

    # Pairwise correlation
    header = f"| {'':>30} |"
    for jk in JUDGE_KEYS:
        header += f" {JUDGE_NAMES[jk]:>22} |"
    lines.append(header)
    lines.append("|" + "-" * 31 + "|" + (("-" * 23 + "|") * len(JUDGE_KEYS)))

    for jk1 in JUDGE_KEYS:
        row = f"| {JUDGE_NAMES[jk1]:>30} |"
        for jk2 in JUDGE_KEYS:
            v1 = judge_vectors[jk1]
            v2 = judge_vectors[jk2]
            # Align where both have values
            pairs = [(a, b) for a, b in zip(v1, v2) if a is not None and b is not None]
            if len(pairs) > 2:
                a, b = zip(*pairs)
                rho, _ = stats.spearmanr(a, b)
                row += f" {rho:>22.3f} |"
            else:
                row += f" {'N/A':>22} |"
        lines.append(row)
    lines.append("")

    # Per-model correlation
    lines.append("Per-model cross-judge correlation:")
    lines.append("")
    for model in MODEL_ORDER:
        ms = model.split("/")[-1]
        lines.append(f"**{ms}:**")
        for i, jk1 in enumerate(JUDGE_KEYS):
            for jk2 in JUDGE_KEYS[i + 1:]:
                v1 = scores[jk1][model]
                v2 = scores[jk2][model]
                pairs = [(a, b) for a, b in zip(v1, v2) if a is not None and b is not None]
                if len(pairs) > 2:
                    a, b = zip(*pairs)
                    rho, _ = stats.spearmanr(a, b)
                    lines.append(f"  {JUDGE_NAMES[jk1]} vs {JUDGE_NAMES[jk2]}: ρ={rho:.3f} (n={len(pairs)})")
        lines.append("")

    # ==========================================================
    # 3. Model Ranking Agreement
    # ==========================================================
    lines.append("## 3. Model Ranking Agreement per Prompt")
    lines.append("")
    lines.append("For each prompt, which model does each judge rank highest?")
    lines.append("")

    for jk in JUDGE_KEYS:
        name = JUDGE_NAMES[jk]
        best_counts = Counter()
        n_valid = 0
        n_ties = 0

        for i in range(len(data)):
            model_scores = {}
            for model in MODEL_ORDER:
                s = scores[jk][model][i] if i < len(scores[jk][model]) else None
                if s is not None:
                    model_scores[model] = s
            if len(model_scores) < 2:
                continue
            n_valid += 1
            max_score = max(model_scores.values())
            best_models = [m for m, s in model_scores.items() if s == max_score]
            if len(best_models) > 1:
                n_ties += 1
            best_counts[best_models[0].split("/")[-1]] += 1

        lines.append(f"**{name}:**")
        lines.append(f"  Best model distribution: {dict(best_counts)}")
        lines.append(f"  Tie rate: {n_ties}/{n_valid} ({100*n_ties/n_valid:.1f}%)")
        lines.append("")

    # Cross-judge best-model agreement
    lines.append("**Cross-judge agreement on best model:**")
    agree_all3 = 0
    agree_any2 = 0
    n_compare = 0

    for i in range(len(data)):
        bests = []
        for jk in JUDGE_KEYS:
            model_scores = {}
            for model in MODEL_ORDER:
                s = scores[jk][model][i] if i < len(scores[jk][model]) else None
                if s is not None:
                    model_scores[model] = s
            if model_scores:
                best = max(model_scores, key=model_scores.get)
                bests.append(best)
        if len(bests) == 3:
            n_compare += 1
            if bests[0] == bests[1] == bests[2]:
                agree_all3 += 1
            if bests[0] == bests[1] or bests[1] == bests[2] or bests[0] == bests[2]:
                agree_any2 += 1

    if n_compare > 0:
        lines.append(f"  All 3 agree: {agree_all3}/{n_compare} ({100*agree_all3/n_compare:.1f}%)")
        lines.append(f"  At least 2 agree: {agree_any2}/{n_compare} ({100*agree_any2/n_compare:.1f}%)")
    lines.append("")

    # ==========================================================
    # 4. Scheduling Granularity (Ties Analysis)
    # ==========================================================
    lines.append("## 4. Scheduling Granularity — Tie Analysis")
    lines.append("")
    lines.append("For scheduling, we need the quality score to discriminate between models.")
    lines.append("Ties = same score for 2+ models on same prompt → quality term useless for that prompt.")
    lines.append("")

    for jk in JUDGE_KEYS:
        name = JUDGE_NAMES[jk]
        n_prompts = 0
        n_any_tie = 0
        n_all_tie = 0
        unique_scores_per_prompt = []

        for i in range(len(data)):
            model_scores = []
            for model in MODEL_ORDER:
                s = scores[jk][model][i] if i < len(scores[jk][model]) else None
                if s is not None:
                    model_scores.append(s)
            if len(model_scores) < 2:
                continue
            n_prompts += 1
            unique = len(set(np.round(model_scores, 4)))
            unique_scores_per_prompt.append(unique)
            if unique < len(model_scores):
                n_any_tie += 1
            if unique == 1:
                n_all_tie += 1

        lines.append(f"**{name}:**")
        lines.append(f"  Prompts with ANY tie: {n_any_tie}/{n_prompts} ({100*n_any_tie/n_prompts:.1f}%)")
        lines.append(f"  Prompts with ALL tied: {n_all_tie}/{n_prompts} ({100*n_all_tie/n_prompts:.1f}%)")
        if unique_scores_per_prompt:
            lines.append(f"  Mean unique scores per prompt: {np.mean(unique_scores_per_prompt):.2f}/4")
        lines.append("")

    # ==========================================================
    # 5. Safety Alignment
    # ==========================================================
    lines.append("## 5. Safety Alignment (Harmful Prompts)")
    lines.append("")

    harmful_entries = [d for d in data if d.get("is_harmful")]
    safe_entries = [d for d in data if not d.get("is_harmful")]
    lines.append(f"Harmful prompts: {len(harmful_entries)}, Safe prompts: {len(safe_entries)}")
    lines.append("")

    for jk in JUDGE_KEYS:
        name = JUDGE_NAMES[jk]
        lines.append(f"**{name} — Harmful prompts (higher=correctly refused):**")
        for model in MODEL_ORDER:
            ms = model.split("/")[-1]
            vals = []
            for entry in harmful_entries:
                s = entry.get("models", {}).get(model, {}).get("llm_judge_scores", {}).get(jk)
                if s is not None:
                    vals.append(s)
            if vals:
                a = np.array(vals)
                lines.append(f"  {ms}: mean={a.mean():.3f}, refusal_rate(>=0.8)={np.mean(a >= 0.8):.1%}")
        lines.append("")

    # ==========================================================
    # 6. Correlation with Existing Signals
    # ==========================================================
    lines.append("## 6. Correlation with Existing Signals")
    lines.append("")

    for sig, sig_name in EXISTING_SIGNALS + [("blind", "Qwen Blind Judge")]:
        lines.append(f"**{sig_name}:**")
        for jk in JUDGE_KEYS:
            jname = JUDGE_NAMES[jk]
            all_pairs = []
            for model in MODEL_ORDER:
                jvals = scores[jk][model]
                if sig == "blind":
                    svals = blind[model]
                else:
                    svals = existing[sig][model]
                for j, s in zip(jvals, svals):
                    if j is not None and s is not None:
                        all_pairs.append((j, s))
            if len(all_pairs) > 2:
                a, b = zip(*all_pairs)
                rho, _ = stats.spearmanr(a, b)
                lines.append(f"  vs {jname}: ρ={rho:.3f} (n={len(all_pairs)})")
        lines.append("")

    # ==========================================================
    # 7. Recommendation
    # ==========================================================
    lines.append("## 7. Summary for Decision")
    lines.append("")
    lines.append("| Criterion | Prometheus-v2 (1-5) | Llama-3.1-8B (1-10) | Qwen-2.5-7B (1-10) |")
    lines.append("|-----------|--------------------|--------------------|---------------------|")
    lines.append("| Scale | 1-5 (normalized /5) | 1-10 (normalized /10) | 1-10 (normalized /10) |")
    lines.append("| Model family | Mistral (unbiased) | Meta (unbiased) | Qwen (BIASED) |")
    lines.append("| Purpose-built | Yes (fine-tuned for eval) | No (general chat) | No (general chat) |")
    lines.append("| Tie rate | See §4 | See §4 | See §4 |")
    lines.append("| 72B ranking | See §3 | See §3 | See §3 (expect bias) |")
    lines.append("")
    lines.append("Key questions for decision:")
    lines.append("1. Does 5-class (Prometheus) have too many ties for scheduling?")
    lines.append("2. Does Llama 10-class provide enough discrimination?")
    lines.append("3. How much does Qwen bias affect model ranking?")
    lines.append("4. Do all 3 judges agree on which model is best per prompt?")
    lines.append("5. Should we train on one judge or multiple?")

    # Write report
    report = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {output_path}")
    print(report)


def main():
    parser = argparse.ArgumentParser(description="Analyze judge scores")
    parser.add_argument("--input", required=True, help="Scored JSONL file")
    parser.add_argument("--output", default="reports/judge_analysis.md")
    args = parser.parse_args()

    with open(args.input) as f:
        data = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(data)} entries")

    generate_report(data, args.output)


if __name__ == "__main__":
    main()
