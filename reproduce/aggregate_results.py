#!/usr/bin/env python3
"""Aggregate ALL 215 cells across overnight grid + T2/T3/T4/T5 tracks.

Outputs:
  _aggregate/AGGREGATE_FULL.csv       — flat row per cell, all tracks
  _aggregate/AGGREGATE_FULL.jsonl     — same with model_distribution
  _aggregate/per_track_summary.json   — concise per-track summary
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(
    "/home/anon/Code/llm/RouteBalance/route_balance_paper/smoke_test_apr_13/results/main_table_full_3534/route_balance_final_e2e"
)
SCORED = Path("/home/anon/Code/llm/RouteBalance/data/route_balance/scored/test_scored_filtered.jsonl")
OUT_DIR = ROOT / "_aggregate"
OUT_DIR.mkdir(exist_ok=True)

PRICING_PER_1M = {
    "Qwen/Qwen2.5-3B": 0.06,
    "Qwen/Qwen2.5-7B": 0.07,
    "Qwen/Qwen2.5-14B": 0.15,
    "Qwen/Qwen2.5-72B": 0.40,
}
DEEPEVAL_KEY = "deepeval-llama3.1-8b-it_reference"
_USER_BLOCK_RE = re.compile(r"<\|im_start\|>user\n(.*?)(?:\n?<\|im_end\|>|\Z)", re.DOTALL)
DISP_MAP = {"ro": "round_robin", "sh": "shortest_queue", "ra": "random"}


def unwrap_chat(p: str) -> str:
    if "<|im_start|>user" not in p:
        return p.strip()
    m = _USER_BLOCK_RE.search(p)
    return (m.group(1) if m else p).strip()


def build_deepeval_lookup() -> dict:
    table = {}
    with open(SCORED) as f:
        for line in f:
            rec = json.loads(line)
            prompt = (rec.get("prompt") or "").strip()
            if not prompt:
                continue
            per_model = {}
            for model_name, model_rec in (rec.get("models") or {}).items():
                judge = (model_rec or {}).get("llm_judge_scores") or {}
                if DEEPEVAL_KEY in judge:
                    per_model[f"Qwen/{model_name}"] = float(judge[DEEPEVAL_KEY])
                    per_model[model_name] = float(judge[DEEPEVAL_KEY])
            if per_model:
                table[prompt] = per_model
    return table


def parse_cell(stem: str, subdir: str) -> dict:
    """Return metadata dict for cell stem; track-specific."""
    if subdir == "route_balance_weight_sweep":
        m = re.match(r"wq([\d.]+)_wl([\d.]+)_wc([\d.]+)_lambda(\d+)_n(\d+)", stem)
        if not m: return {}
        wq, wl, wc, lam, n = m.groups()
        # Strategy label
        wq_f, wl_f, wc_f = float(wq), float(wl), float(wc)
        if abs(wq_f - 0.33) < 0.05 and abs(wl_f - 0.33) < 0.05:
            strat = "balance"
        elif wq_f >= 0.5: strat = "quality_pri"
        elif wl_f >= 0.5: strat = "latency_pri"
        elif wc_f >= 0.5: strat = "cost_pri"
        else: strat = "mixed"
        return {"track": "route_balance_weight_sweep", "system": "route_balance", "router": "route_balance_native",
                "dispatcher": "route_balance_lpt", "wq": wq_f, "wl": wl_f, "wc": wc_f,
                "strategy": strat, "lambda": int(lam)}
    if subdir == "dispatcher_only_baselines":
        m = re.match(r"passthru_(\w\w)_l(\d+)_n(\d+)", stem)
        if not m: return {}
        disp, lam, _ = m.groups()
        return {"track": "dispatcher_only_baselines", "system": "dispatcher_only",
                "router": "passthrough", "dispatcher": DISP_MAP.get(disp, disp),
                "lambda": int(lam), "strategy": "passthrough"}
    if subdir == "best_route_4way_threshold":
        m = re.match(r"br4(?:_argmax)?_t?([\d.]+)?_(\w\w)_l(\d+)_n(\d+)", stem)
        if not m:
            m = re.match(r"br4_argmax_(\w\w)_l(\d+)_n(\d+)", stem)
            if m:
                disp, lam, _ = m.groups()
                return {"track": "best_route_4way_threshold", "system": "best_route_4way",
                        "router": "best_route_4way", "dispatcher": DISP_MAP.get(disp, disp),
                        "threshold": 0.0, "lambda": int(lam), "strategy": "argmax"}
            return {}
        thr, disp, lam, _ = m.groups()
        return {"track": "best_route_4way_threshold", "system": "best_route_4way",
                "router": "best_route_4way", "dispatcher": DISP_MAP.get(disp, disp),
                "threshold": float(thr) if thr else 0.0, "lambda": int(lam),
                "strategy": f"thr{thr}"}
    if subdir == "avengers_pro_4m_pw_sweep":
        m = re.match(r"avg_pw([\d.]+)_(\w\w)_l(\d+)_n(\d+)", stem)
        if not m: return {}
        pw, disp, lam, _ = m.groups()
        return {"track": "avengers_pro_4m_pw_sweep", "system": "avengers_pro",
                "router": "avengers_pro", "dispatcher": DISP_MAP.get(disp, disp),
                "pw": float(pw), "lambda": int(lam), "strategy": f"pw{pw}"}
    if subdir == "budget_calibrated_demo":
        # Parse: {router}_{filter|argmax|nofilter}_calibrated[_X_Y][_25_25_25_25]_l{λ}_n{n}
        m = re.match(r"(route_balance|br4)_(filter|nofilter|argmax)_calibrated(?:_(\d+_\d+(?:_\d+_\d+)?))?_l(\d+)_n(\d+)", stem)
        if not m: return {}
        sys_kind, mode, ratio, lam, _ = m.groups()
        ratio = ratio or "default_25_25_25_25"
        return {"track": "budget_calibrated_demo",
                "system": "route_balance" if sys_kind == "route_balance" else "best_route_4way",
                "router": "route_balance_native" if sys_kind == "route_balance" else "best_route_4way",
                "dispatcher": "route_balance_lpt" if sys_kind == "route_balance" else "round_robin",
                "filter_mode": mode, "ratio": ratio, "lambda": int(lam),
                "strategy": f"budget_{mode}_{ratio}"}
    if subdir == "ablation_lpt":
        m = re.match(r"route_balance_uniform_lpt_off_l(\d+)_n(\d+)", stem)
        if not m: return {}
        lam, _ = m.groups()
        return {"track": "ablation_lpt", "system": "route_balance", "router": "route_balance_native",
                "dispatcher": "fifo_no_lpt", "ablation": "lpt_off",
                "lambda": int(lam), "strategy": "lpt_off"}
    if subdir == "ablation_batch":
        # route_balance_uniform_adaptive_off_l{λ} OR route_balance_uniform_bs{N}_l{λ}
        m = re.match(r"route_balance_uniform_adaptive_off_l(\d+)_n(\d+)", stem)
        if m:
            lam, _ = m.groups()
            return {"track": "ablation_batch", "system": "route_balance", "router": "route_balance_native",
                    "dispatcher": "route_balance_lpt", "ablation": "adaptive_off",
                    "lambda": int(lam), "strategy": "adaptive_off"}
        m = re.match(r"route_balance_uniform_bs(\d+)_l(\d+)_n(\d+)", stem)
        if m:
            bs, lam, _ = m.groups()
            return {"track": "ablation_batch", "system": "route_balance", "router": "route_balance_native",
                    "dispatcher": "route_balance_lpt", "ablation": f"bs{bs}",
                    "batch_size": int(bs), "lambda": int(lam), "strategy": f"bs{bs}"}
        return {}
    if subdir == "slo_ablation":
        m = re.match(r"route_balance_uniform_filter_(on|off)_l(\d+)_n(\d+)", stem)
        if not m: return {}
        mode, lam, _ = m.groups()
        return {"track": "slo_ablation", "system": "route_balance", "router": "route_balance_native",
                "dispatcher": "route_balance_lpt", "filter_mode": mode,
                "lambda": int(lam), "strategy": f"filter_{mode}"}
    return {}


def aggregate_cell(path: Path, meta: dict, deepeval_lookup: dict) -> dict:
    with open(path) as f:
        d = json.load(f)
    rd = d.get("response_details", []) or []
    completed = int(d.get("completed", 0) or 0)
    failed = int(d.get("failed", 0) or 0)

    model_counter = Counter(r.get("model") for r in rd if not r.get("error"))
    total_resps = sum(model_counter.values())
    cost_usd = 0.0
    actual_cost_total = 0.0
    n_budget_exhausted = 0
    for r in rd:
        if r.get("error"): continue
        m = r.get("model")
        out_tok = int(r.get("output_len", 0) or 0)
        in_tok = int(r.get("input_len", 0) or 0)
        price = PRICING_PER_1M.get(m, 0.0)
        cost_usd += (in_tok + out_tok) * price / 1_000_000
        if r.get("budget_exhausted"):
            n_budget_exhausted += 1
        ac = r.get("actual_cost")
        if ac:
            actual_cost_total += float(ac)

    q_scores, matched = [], 0
    for r in rd:
        if r.get("error"): continue
        prompt = unwrap_chat(r.get("prompt") or "")
        per_model = deepeval_lookup.get(prompt) or {}
        if not per_model: continue
        m = r.get("model") or ""
        score = per_model.get(m) or per_model.get(m.replace("Qwen/", ""))
        if score is not None:
            q_scores.append(float(score))
            matched += 1
    deepeval_mean = float(np.mean(q_scores)) if q_scores else None
    quality_coverage = matched / max(total_resps, 1)

    pred_q = [r.get("predicted_quality") for r in rd
              if not r.get("error") and r.get("predicted_quality") is not None]
    pred_q_mean = float(np.mean(pred_q)) if pred_q else None

    out = {
        **meta, "cell": path.stem, "completed": completed, "failed": failed,
        "request_throughput": float(d.get("request_throughput", 0) or 0),
        "output_throughput": float(d.get("output_throughput", 0) or 0),
        "mean_e2el_ms": float(d.get("mean_e2el_ms", 0) or 0),
        "p50_e2el_ms": float(d.get("p50_e2el_ms", 0) or d.get("median_e2el_ms", 0) or 0),
        "p99_e2el_ms": float(d.get("p99_e2el_ms", 0) or 0),
        "mean_ttft_ms": float(d.get("mean_ttft_ms", 0) or 0),
        "p99_ttft_ms": float(d.get("p99_ttft_ms", 0) or 0),
        "mean_tpot_ms": float(d.get("mean_tpot_ms", 0) or 0),
        "p99_tpot_ms": float(d.get("p99_tpot_ms", 0) or 0),
        "mean_scheduling_overhead_ms": float(d.get("mean_scheduling_overhead_ms", 0) or 0),
        "request_goodput": float(d.get("request_goodput", 0) or 0),
        "duration_s": float(d.get("duration", 0) or 0),
        "n_distinct_models": len(model_counter),
        "model_distribution": dict(model_counter),
        "cost_usd_total": cost_usd,
        "cost_per_req_usd": cost_usd / max(total_resps, 1),
        "deepeval_quality_mean": deepeval_mean,
        "quality_coverage": quality_coverage,
        "predicted_quality_mean": pred_q_mean,
        "n_budget_exhausted": n_budget_exhausted,
        "actual_cost_total": actual_cost_total,
    }
    return out


def main():
    print("Building deepeval lookup...")
    de = build_deepeval_lookup()
    print(f"  {len(de)} prompts in lookup")

    rows, errors = [], []
    subdirs = [
        "route_balance_weight_sweep", "dispatcher_only_baselines",
        "best_route_4way_threshold", "avengers_pro_4m_pw_sweep",
        "budget_calibrated_demo", "ablation_lpt", "ablation_batch", "slo_ablation",
    ]
    for sub in subdirs:
        p = ROOT / sub
        if not p.is_dir():
            print(f"  skip missing: {sub}")
            continue
        n_in = 0
        for cell_path in sorted(p.glob("*.json")):
            meta = parse_cell(cell_path.stem, sub)
            if not meta:
                errors.append(f"{sub}/{cell_path.stem}: parse failed")
                continue
            try:
                row = aggregate_cell(cell_path, meta, de)
                rows.append(row)
                n_in += 1
            except Exception as e:
                errors.append(f"{sub}/{cell_path.stem}: {e}")
        print(f"  {sub}: {n_in} cells")

    print(f"\nTotal cells aggregated: {len(rows)}")
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:10]: print(f"  {e}")

    # CSV (flat, omit model_distribution)
    if rows:
        all_keys = set()
        for r in rows: all_keys.update(r.keys())
        all_keys.discard("model_distribution")
        cols = sorted(all_keys)
        out_csv = OUT_DIR / "AGGREGATE_FULL.csv"
        with open(out_csv, "w") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"Wrote {out_csv}")

        # JSONL (preserves model_distribution)
        out_jsonl = OUT_DIR / "AGGREGATE_FULL.jsonl"
        with open(out_jsonl, "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        print(f"Wrote {out_jsonl}")

        # Per-track summary
        per_track = {}
        for r in rows:
            t = r.get("track", "?")
            d = per_track.setdefault(t, {"n_cells": 0, "errors_total": 0,
                                         "completed_total": 0,
                                         "deepeval_mean_avg": [],
                                         "mean_e2el_ms_avg": [],
                                         "cost_per_req_avg": []})
            d["n_cells"] += 1
            d["errors_total"] += r.get("failed", 0)
            d["completed_total"] += r.get("completed", 0)
            if r.get("deepeval_quality_mean") is not None:
                d["deepeval_mean_avg"].append(r["deepeval_quality_mean"])
            if r.get("mean_e2el_ms"):
                d["mean_e2el_ms_avg"].append(r["mean_e2el_ms"])
            if r.get("cost_per_req_usd"):
                d["cost_per_req_avg"].append(r["cost_per_req_usd"])
        for t, d in per_track.items():
            d["deepeval_mean_avg"] = float(np.mean(d["deepeval_mean_avg"])) if d["deepeval_mean_avg"] else None
            d["mean_e2el_ms_avg"] = float(np.mean(d["mean_e2el_ms_avg"])) if d["mean_e2el_ms_avg"] else None
            d["cost_per_req_avg"] = float(np.mean(d["cost_per_req_avg"])) if d["cost_per_req_avg"] else None
        out_summary = OUT_DIR / "per_track_summary.json"
        with open(out_summary, "w") as f:
            json.dump(per_track, f, indent=2)
        print(f"Wrote {out_summary}")

        print("\n=== PER-TRACK SUMMARY ===")
        for t, d in sorted(per_track.items()):
            print(f"  {t:35s}: cells={d['n_cells']:3d}  err_total={d['errors_total']}  "
                  f"q_avg={d['deepeval_mean_avg']:.4f}  "
                  f"e2e_avg={d['mean_e2el_ms_avg']:.0f}ms  "
                  f"cost_avg=${d['cost_per_req_avg']*1000:.4f}/1k req")


if __name__ == "__main__":
    main()
