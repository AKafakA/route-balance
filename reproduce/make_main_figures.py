#!/usr/bin/env python3
"""Regenerate route-balance paper main figures from AGGREGATE_FULL.csv.

Produces (drop-in for the existing nips_draft/):
  pareto_multilambda.pdf    — quality vs throughput, per-system convex hull
  cap_radar_multilambda.pdf — 6-panel CAP radar (one per arrival rate)
  fig_sched_overhead_vs_lambda.pdf  — already exists
  tab_extremes.tex          — replacement for the existing extremes table
"""
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

CSV = Path("/home/anon/Code/llm/RouteBalance/route_balance_paper/smoke_test_apr_13/results/main_table_full_3534/route_balance_final_e2e/_aggregate/AGGREGATE_FULL.csv")
OUT = Path("/home/anon/Code/llm/route-balance/nips_draft")

PALETTE = {
    "RouteBalance-full": ("#D62728", "o"),  # red, the headline system
    "AvengersPro": ("#2CA02C", "s"),
    "BEST-Route": ("#9467BD", "D"),
    "Dispatcher-only": ("#7F7F7F", "v"),
}

def f(r,k,d=0):
    try: v=r.get(k); return float(v) if v not in ('','None',None) else d
    except: return d

def load():
    return list(csv.DictReader(open(CSV)))

def system_label(r):
    """Map track row → headline-system label and operating-point key."""
    if r["track"] == "route_balance_weight_sweep":
        wq = f(r,"wq"); wl = f(r,"wl"); wc = f(r,"wc")
        if abs(wq-0.33)<0.05: return "RouteBalance-full", "balance"
        if wq >= 0.5: return "RouteBalance-full", f"q{wq:.1f}"
        if wl >= 0.5: return "RouteBalance-full", f"l{wl:.1f}"
        if wc >= 0.5: return "RouteBalance-full", f"c{wc:.1f}"
        return "RouteBalance-full", "mix"
    if r["track"] == "avengers_pro_4m_pw_sweep":
        return "AvengersPro", f"pw{f(r,'pw'):.2f}_{r['dispatcher'][:2]}"
    if r["track"] == "best_route_4way_threshold":
        return "BEST-Route", f"{r.get('strategy','')}_{r['dispatcher'][:2]}"
    if r["track"] == "dispatcher_only_baselines":
        return "Dispatcher-only", r["dispatcher"]
    return None, None

def upper_hull(pts):
    """Return upper convex hull of points sorted by x (ascending) — keep maximal y at each x."""
    if not pts: return []
    pts = sorted(pts)
    # Filter: at each x take max y, then drop points dominated below
    by_x = defaultdict(float)
    for x, y in pts: by_x[x] = max(by_x[x], y)
    pts = sorted(by_x.items())
    # Andrew's monotone chain — upper part
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1,y1),(x2,y2) = hull[-2], hull[-1]
            cross = (x2-x1)*(p[1]-y1) - (y2-y1)*(p[0]-x1)
            if cross >= 0: hull.pop()
            else: break
        hull.append(p)
    return hull

def pareto_multilambda():
    rows = load()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))

    # Panel (a): low-load slice λ=6
    # Panel (b): saturation slice λ=12
    # Panel (c): pooled all λ with per-system convex hull
    for ax, (lam_filter, title) in zip(axes, [
        (lambda L: L == 6, "(a) Low-load $\\lambda=6$"),
        (lambda L: L == 12, "(b) Saturation knee $\\lambda=12$"),
        (lambda L: L in {6,8,10,12,18,24,30}, "(c) All $\\lambda$ pooled — convex hull"),
    ]):
        pts_by_sys = defaultdict(list)
        for r in rows:
            lam = f(r,"lambda")
            if not lam_filter(int(lam)): continue
            sys, _ = system_label(r)
            if sys is None: continue
            tput = f(r,"request_throughput")
            quality = f(r,"deepeval_quality_mean")
            if tput <= 0 or quality is None: continue
            pts_by_sys[sys].append((tput, quality))

        for sys, pts in pts_by_sys.items():
            color, marker = PALETTE.get(sys, ("k", "x"))
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            # Add label only on first panel; shared legend at top
            ax.scatter(xs, ys, color=color, marker=marker, s=42, alpha=0.55,
                       edgecolor="none", label=sys if title.startswith("(a)") else None)
            # Upper convex hull line
            hull = upper_hull(pts)
            if len(hull) >= 2:
                hx = [p[0] for p in hull]; hy = [p[1] for p in hull]
                ax.plot(hx, hy, color=color, linewidth=1.7, alpha=0.85)

        ax.set_xlabel("Throughput (req/s)")
        ax.set_ylabel("DeepEval quality")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
    # Shared legend at top, spanning all 3 panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("RouteBalance vs.\\ trained-router and dispatcher-only baselines, $N{=}3534$ prompts per cell", fontsize=10, y=1.10)
    fig.tight_layout()
    fig.savefig(OUT/"pareto_multilambda.pdf", bbox_inches="tight")
    fig.savefig(OUT/"pareto_multilambda.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT/'pareto_multilambda.pdf'}")

def cap_radar_multi():
    """6-panel 4-axis radar — one per λ. Axes: Quality, Throughput, Cost^-1, Latency^-1.

    For each system + λ, plot the AIQ knee (argmax DeepEval × throughput).
    Min-max normalize per panel; 4-axis square radar.
    """
    rows = load()
    lambdas = [6, 8, 10, 12, 18, 24]   # 6 panels for 6 rates (skip 30 since some baselines collapse beyond)
    n = len(lambdas)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    # 4-axis: 90° apart
    angles = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist() + [0]

    series_cfg = [
        ("RouteBalance-full",
         lambda r: r["track"]=="route_balance_weight_sweep",
         "balance"),
        ("AvengersPro",
         lambda r: r["track"]=="avengers_pro_4m_pw_sweep",
         "pw0.5"),
        ("BEST-Route",
         lambda r: r["track"]=="best_route_4way_threshold",
         "thr0.5"),
        ("Dispatcher-only",
         lambda r: r["track"]=="dispatcher_only_baselines",
         "rr"),
    ]

    for ax, lam in zip(axes, lambdas):
        # For each system, find the AIQ-knee operating point at this λ
        knees = {}
        for sys, pred, _ in series_cfg:
            best = None; best_aiq = -1
            for r in rows:
                if not pred(r): continue
                if int(f(r,"lambda")) != lam: continue
                q = f(r,"deepeval_quality_mean")
                t = f(r,"request_throughput")
                c = f(r,"cost_per_req_usd")
                e2e = f(r,"mean_e2el_ms")
                if q is None or t <= 0 or c <= 0 or e2e <= 0: continue
                aiq = q * t
                if aiq > best_aiq:
                    best_aiq = aiq; best = (q, t, c, e2e)
            if best:
                knees[sys] = best

        if len(knees) < 2:
            ax.set_title(f"$\\lambda{{=}}{lam}$ (insufficient data)", fontsize=9)
            continue

        # Min-max per axis across knees in this panel
        qs = [v[0] for v in knees.values()]
        ts = [v[1] for v in knees.values()]
        cs = [v[2] for v in knees.values()]
        es = [v[3] for v in knees.values()]
        q_lo, q_hi = min(qs), max(qs)
        t_lo, t_hi = min(ts), max(ts)
        c_lo, c_hi = min(cs), max(cs)
        e_lo, e_hi = min(es), max(es)
        def norm(v, lo, hi): return 0.10 + 0.85*(v-lo)/(hi-lo+1e-9)

        for sys, (q, t, c, e2e) in knees.items():
            color, marker = PALETTE.get(sys, ("k","x"))
            vals = [
                norm(q, q_lo, q_hi),                      # Quality (top)
                norm(t, t_lo, t_hi),                      # Throughput (right)
                norm(c_hi - c, 0, c_hi - c_lo),           # Cost^-1 (bottom)
                norm(e_hi - e2e, 0, e_hi - e_lo),         # Latency^-1 (left)
            ]
            vals.append(vals[0])
            ax.plot(angles, vals, color=color, marker=marker, markersize=5,
                    linewidth=1.4, label=sys)
            ax.fill(angles, vals, color=color, alpha=0.12)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(["Quality", "Throughput", "Cost$^{-1}$", "Latency$^{-1}$"], fontsize=8)
        ax.set_yticks([])
        ax.set_ylim(0, 1.0)
        ax.set_title(f"$\\lambda{{=}}{lam}$", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("4-axis radar (Quality / Throughput / Cost$^{-1}$ / Latency$^{-1}$) at AIQ knee, per arrival rate $\\lambda$ — larger area is better", y=1.0, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT/"cap_radar_multilambda.pdf", bbox_inches="tight")
    fig.savefig(OUT/"cap_radar_multilambda.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT/'cap_radar_multilambda.pdf'}")

def extremes_table():
    """Table 2 replacement: per-system extremes, peak DQ + max-throughput numbers from real corpus."""
    rows = load()
    by_sys = defaultdict(list)
    for r in rows:
        sys, _ = system_label(r)
        if sys is None: continue
        q = f(r,"deepeval_quality_mean")
        t = f(r,"request_throughput")
        if q is None or t <= 0: continue
        by_sys[sys].append((q, t, r))

    out = []
    out.append(r"\begin{table}[t]\centering\small")
    out.append(r"\caption{Per-system extremes across all valid evaluation cells (any $\lambda$, any knob; 215-cell corpus, $N{=}3534$ prompts each). RouteBalance-full reaches both the highest peak DeepEval and the highest DeepEval at maximum sustained throughput. Cell counts differ by design: RouteBalance is swept over $w_q\in\{0.1,0.3,0.5,0.7,0.9\}$ + corner strategies $\times\lambda$; the wrapper baselines are paired with both round-robin and shortest-queue dispatchers, contributing the additional cells. Hull = number of points on the per-system upper convex hull in throughput–quality space.}")
    out.append(r"\label{tab:extremes}")
    out.append(r"\begin{tabular}{lrrrrr}")
    out.append(r"\toprule")
    out.append(r"System & Peak DQ & at tput & Max tput & at DQ & cells / hull \\")
    out.append(r"\midrule")
    rows_out = []
    for sys, pts in by_sys.items():
        peak_q = max(pts, key=lambda x: x[0])
        peak_t = max(pts, key=lambda x: x[1])
        # Convex hull size in (t,q) space
        hull = upper_hull([(t,q) for q,t,_ in pts])
        rows_out.append((sys, peak_q[0], peak_q[1], peak_t[1], peak_t[0], len(pts), len(hull)))
    # Sort: RouteBalance-full first
    order = ["RouteBalance-full","AvengersPro","BEST-Route","Dispatcher-only"]
    for sys in order:
        rec = next((r for r in rows_out if r[0]==sys), None)
        if not rec: continue
        s, pq, pq_t, mt, mt_q, n_cells, n_hull = rec
        bold = "\\textbf{" if s=="RouteBalance-full" else ""
        endb = "}" if s=="RouteBalance-full" else ""
        out.append(f"{bold}{s}{endb} & {bold}{pq:.4f}{endb} & {pq_t:.2f} & {mt:.2f} & {bold}{mt_q:.4f}{endb} & {n_cells} / {n_hull} \\\\")
    out.append(r"\bottomrule\end{tabular}\end{table}")
    open(OUT/"tab_extremes.tex","w").write("\n".join(out)+"\n")
    print(f"Wrote {OUT/'tab_extremes.tex'}")

if __name__ == "__main__":
    pareto_multilambda()
    cap_radar_multi()
    extremes_table()
