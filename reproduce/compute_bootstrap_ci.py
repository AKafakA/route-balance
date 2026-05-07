#!/usr/bin/env python3
"""Compute bootstrap 95% CIs for the headline λ=12 row + p99 tail analysis + model distribution.

Outputs:
  _aggregate/ci_lambda12.json — bootstrap CIs for E2E mean and quality on the 7 headline systems
  _aggregate/p99_lambda_axis.json — p99 e2e + p99 ttft per system per λ for the tail story
  _aggregate/model_distribution.tex — LaTeX table: model-mix percentages per RouteBalance strategy
  _aggregate/fig_model_dist.{pdf,png} — stacked bar of routing decisions per strategy at λ=12
"""
import json, csv
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/home/anon/Code/llm/RouteBalance/route_balance_paper/smoke_test_apr_13/results/main_table_full_3534/route_balance_final_e2e")
OUT = ROOT / "_aggregate"

PRICING = {"Qwen/Qwen2.5-3B": 0.06, "Qwen/Qwen2.5-7B": 0.07,
           "Qwen/Qwen2.5-14B": 0.15, "Qwen/Qwen2.5-72B": 0.40}
SCORED_PATH = Path("/home/anon/Code/llm/RouteBalance/data/route_balance/scored/test_scored_filtered.jsonl")
DEEPEVAL_KEY = "deepeval-llama3.1-8b-it_reference"

import re
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)(?:\n?<\|im_end\|>|\Z)", re.DOTALL)
def unwrap(p):
    m = USER_RE.search(p) if "<|im_start|>user" in p else None
    return (m.group(1) if m else p).strip()

def load_deepeval():
    table = {}
    with open(SCORED_PATH) as f:
        for line in f:
            r = json.loads(line)
            prompt = (r.get("prompt") or "").strip()
            if not prompt: continue
            per_model = {}
            for mn, mr in (r.get("models") or {}).items():
                judge = (mr or {}).get("llm_judge_scores") or {}
                if DEEPEVAL_KEY in judge:
                    per_model[f"Qwen/{mn}"] = float(judge[DEEPEVAL_KEY])
                    per_model[mn] = float(judge[DEEPEVAL_KEY])
            if per_model: table[prompt] = per_model
    return table

def cell_per_request(path, deepeval):
    """Return per-request (e2e_ms, quality, model) for one cell."""
    with open(path) as f:
        d = json.load(f)
    out = []
    for r in d.get("response_details", []) or []:
        if r.get("error"): continue
        e2e = float(r.get("e2el", 0) or 0) * 1000.0  # → ms
        if e2e <= 0: continue
        prompt = unwrap(r.get("prompt") or "")
        per_m = deepeval.get(prompt) or {}
        m = r.get("model") or ""
        q = per_m.get(m) or per_m.get(m.replace("Qwen/", ""))
        out.append((e2e, q, m))
    return out

def bootstrap_ci(values, n=1000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.array([v for v in values if v is not None])
    if len(arr) < 2: return (None, None, None)
    means = []
    for _ in range(n):
        means.append(rng.choice(arr, size=len(arr), replace=True).mean())
    lo, hi = np.quantile(means, [alpha/2, 1-alpha/2])
    return float(arr.mean()), float(lo), float(hi)

def headline_systems_lambda12():
    """Return {label: cell_path} for the 7 headline systems at λ=12."""
    return {
        "RouteBalance-Balance": ROOT / "route_balance_weight_sweep" / "wq0.33_wl0.33_wc0.33_lambda12_n3534.json",
        "RouteBalance-Quality": ROOT / "route_balance_weight_sweep" / "wq0.8_wl0.1_wc0.1_lambda12_n3534.json",
        "RouteBalance-Latency": ROOT / "route_balance_weight_sweep" / "wq0.1_wl0.8_wc0.1_lambda12_n3534.json",
        "RouteBalance-Cost":    ROOT / "route_balance_weight_sweep" / "wq0.1_wl0.1_wc0.8_lambda12_n3534.json",
        "AvengersPro pw=0.25": ROOT / "avengers_pro_4m_pw_sweep" / "avg_pw0.25_ro_l12_n3534.json",
        "AvengersPro pw=0.8":  ROOT / "avengers_pro_4m_pw_sweep" / "avg_pw0.8_ro_l12_n3534.json",
        "BEST-Route t=0.5":    ROOT / "best_route_4way_threshold" / "br4_t0.5_ro_l12_n3534.json",
        "Dispatcher RR":       ROOT / "dispatcher_only_baselines" / "passthru_ro_l12_n3534.json",
    }

def main():
    print("Loading deepeval lookup...")
    de = load_deepeval()

    # 1. Bootstrap CIs at λ=12
    print("\n=== Bootstrap CIs at λ=12 (B=1000) ===")
    ci_results = {}
    for label, path in headline_systems_lambda12().items():
        if not path.exists():
            print(f"  {label}: MISSING {path.name}"); continue
        per = cell_per_request(path, de)
        e2es = [x[0] for x in per]
        qs = [x[1] for x in per if x[1] is not None]
        e2e_mean, e2e_lo, e2e_hi = bootstrap_ci(e2es)
        q_mean, q_lo, q_hi = bootstrap_ci(qs)
        ci_results[label] = {
            "n_req": len(per), "n_q": len(qs),
            "e2e_mean_ms": e2e_mean, "e2e_ci95_lo": e2e_lo, "e2e_ci95_hi": e2e_hi,
            "e2e_ci_halfwidth_ms": (e2e_hi - e2e_lo)/2 if e2e_mean else None,
            "q_mean": q_mean, "q_ci95_lo": q_lo, "q_ci95_hi": q_hi,
            "q_ci_halfwidth": (q_hi - q_lo)/2 if q_mean else None,
        }
        print(f"  {label:25s} n={len(per):4d}  e2e={e2e_mean:.0f}±{(e2e_hi-e2e_lo)/2:.0f}ms  q={q_mean:.4f}±{(q_hi-q_lo)/2:.4f}")

    json.dump(ci_results, open(OUT/"ci_lambda12.json","w"), indent=2)

    # 2. p99 E2E per system per λ
    print("\n=== p99 E2E across λ ===")
    p99_data = defaultdict(dict)  # system → λ → p99
    rows = list(csv.DictReader(open(OUT/"AGGREGATE_FULL.csv")))
    def f(r,k,d=0):
        try: v=r.get(k); return float(v) if v not in ('','None',None) else d
        except: return d
    series_def = [
        ("RouteBalance-Balance", lambda r: r["track"]=="route_balance_weight_sweep" and abs(f(r,"wq")-0.33)<0.05 and abs(f(r,"wl")-0.33)<0.05),
        ("RouteBalance-Quality", lambda r: r["track"]=="route_balance_weight_sweep" and abs(f(r,"wq")-0.8)<0.05),
        ("AvengersPro pw=0.8",  lambda r: r["track"]=="avengers_pro_4m_pw_sweep" and abs(f(r,"pw")-0.8)<0.01 and r["dispatcher"]=="round_robin"),
        ("AvengersPro pw=0.39", lambda r: r["track"]=="avengers_pro_4m_pw_sweep" and abs(f(r,"pw")-0.39)<0.01 and r["dispatcher"]=="round_robin"),
        ("BEST-Route t=0.5",    lambda r: r["track"]=="best_route_4way_threshold" and r.get("strategy")=="thr0.5" and r["dispatcher"]=="round_robin"),
        ("Dispatcher RR",       lambda r: r["track"]=="dispatcher_only_baselines" and r["dispatcher"]=="round_robin"),
    ]
    for sys, fn in series_def:
        for r in rows:
            if not fn(r): continue
            lam = int(f(r,"lambda"))
            p99_data[sys][lam] = {
                "p99_e2el_ms": f(r,"p99_e2el_ms"),
                "p99_ttft_ms": f(r,"p99_ttft_ms"),
                "p99_tpot_ms": f(r,"p99_tpot_ms"),
                "mean_e2el_ms": f(r,"mean_e2el_ms"),
            }
    json.dump(p99_data, open(OUT/"p99_lambda_axis.json","w"), indent=2)
    for sys, by_lam in p99_data.items():
        print(f"  {sys}:")
        for lam in sorted(by_lam):
            d = by_lam[lam]
            print(f"    λ={lam:3d}: mean_e2e={d['mean_e2el_ms']:.0f}ms p99_e2e={d['p99_e2el_ms']:.0f}ms p99_ttft={d['p99_ttft_ms']:.0f}ms")

    # 3. Model distribution at λ=12 per RouteBalance strategy + AvengersPro pw=0.25 / 0.8 + BEST-Route t=0.5
    print("\n=== Routing distribution at λ=12 ===")
    dist_data = {}
    for label, path in headline_systems_lambda12().items():
        if not path.exists(): continue
        with open(path) as f_:
            d = json.load(f_)
        ctr = Counter(r.get("model") for r in d.get("response_details", []) if not r.get("error"))
        total = sum(ctr.values())
        pct = {m: 100*ctr.get(m, 0)/total for m in ["Qwen/Qwen2.5-3B","Qwen/Qwen2.5-7B","Qwen/Qwen2.5-14B","Qwen/Qwen2.5-72B"]}
        dist_data[label] = pct
        print(f"  {label:25s}  3B={pct['Qwen/Qwen2.5-3B']:5.1f}%  7B={pct['Qwen/Qwen2.5-7B']:5.1f}%  14B={pct['Qwen/Qwen2.5-14B']:5.1f}%  72B={pct['Qwen/Qwen2.5-72B']:5.1f}%")
    json.dump(dist_data, open(OUT/"model_dist_lambda12.json","w"), indent=2)

    # Figure: stacked bar
    sys_names = list(dist_data.keys())
    models = ["Qwen/Qwen2.5-3B","Qwen/Qwen2.5-7B","Qwen/Qwen2.5-14B","Qwen/Qwen2.5-72B"]
    colors = ["#56B4E9","#009E73","#E69F00","#D55E00"]  # cb-safe gradient
    pcts = np.array([[dist_data[s][m] for m in models] for s in sys_names])
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    left = np.zeros(len(sys_names))
    for i, m in enumerate(models):
        ax.barh(sys_names, pcts[:,i], left=left, color=colors[i], edgecolor="white", linewidth=0.5,
                label=m.replace("Qwen/Qwen2.5-",""))
        # In-bar labels for >5% slices
        for j, v in enumerate(pcts[:,i]):
            if v >= 5:
                ax.text(left[j]+v/2, j, f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, color="white" if i>0 else "black")
        left += pcts[:,i]
    ax.set_xlabel("Routed share of requests (%)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.legend(title="Model", loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=4, fontsize=8, frameon=False)
    ax.set_title("Routing distribution at λ=12 — RouteBalance strategies span the model mix; baselines have fixed mixes", pad=8)
    fig.tight_layout()
    fig.savefig(OUT/"fig_model_dist.pdf", bbox_inches="tight")
    fig.savefig(OUT/"fig_model_dist.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT/'fig_model_dist.pdf'}")

    # LaTeX table for distribution
    out = []
    out.append(r"\begin{table}[t]\centering\small")
    out.append(r"\caption{Routing distribution at $\lambda=12$. RouteBalance strategy weights produce qualitatively different routing mixes from a single deployed predictor stack; baselines have fixed mixes regardless of objective. AvengersPro $pw{=}0.25$ concentrates on the smallest model; $pw{=}0.8$ on the largest. BEST-Route splits across two tiers based on confidence.}")
    out.append(r"\label{tab:routing}")
    out.append(r"\begin{tabular}{lrrrr}")
    out.append(r"\toprule")
    out.append(r"System & 3B (\%) & 7B (\%) & 14B (\%) & 72B (\%) \\")
    out.append(r"\midrule")
    for s in sys_names:
        d = dist_data[s]
        out.append(f"{s} & {d['Qwen/Qwen2.5-3B']:.1f} & {d['Qwen/Qwen2.5-7B']:.1f} & {d['Qwen/Qwen2.5-14B']:.1f} & {d['Qwen/Qwen2.5-72B']:.1f} \\\\")
    out.append(r"\bottomrule\end{tabular}\end{table}")
    open(OUT/"model_distribution.tex","w").write("\n".join(out)+"\n")
    print(f"Wrote {OUT/'model_distribution.tex'}")

if __name__ == "__main__":
    main()
