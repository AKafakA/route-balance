# Block_exp — Data Index

Last updated: 2026-04-23 (post-cleanup reorg)

Concise map of where each dataset lives, which script uses it, and which paper figure it feeds. After the 2026-04-23 cleanup, the repo has two data buckets: **`experiment_results_a30/` + `experiment_results_a100/`** (active plot_data — used by current plotters and AE scripts) and **`archival/`** (legacy pre-phase data kept for provenance but not used by any current figure).

---

## 1. Top-level data layout

```
Block_exp/
├── experiment_results_a30/        # PLOT DATA — A30 campaign (canonical, used by regen_figures.py)
│   ├── phase11_main/              # Fanout + 5 baselines × QPS 20-36  → Fig 6, 7
│   ├── phase12_po2/               # Po2 est/oracle × QPS 20-36         → Fig 6, 7, 10, 12
│   ├── phase12_cputracking_progress/   # cputracking progress logs
│   ├── phase12_pretracking_baseline/   # pre-cputracking baseline
│   ├── phase2_n_ablation/         # N-ablation early scans
│   ├── phase3_1_burstiness_po2/   # burst α ∈ {0.25,0.5,1.0,2.0}       → Fig 11
│   ├── phase3_2_error_heatmap_po2/     # base 16-cell grid             → Fig 12
│   ├── phase3_2_error_heatmap_rerun/   # 5 suspect-cell reruns (prio 1)→ Fig 12
│   ├── phase3_2_error_heatmap_redo2/   # (100,0) triple-check (prio 2) → Fig 12
│   ├── phase4_1_float/            # float-QPS capacity scans           → macros.tex
│   ├── phase4_2_burstgpt_refine/  # BurstGPT oracle capacity           → Table 3
│   ├── phase4_2_generality/       # bs=24, cs=2048 sweeps              → Table 3
│   ├── phase4_2_qwen/             # Qwen2-7B capacity sweep            → Table 3
│   ├── phase7_cpu_tracker_po2/    # Po2 CPU tracking (v1)
│   ├── phase7_cpu_tracker_po2_v2/ # Po2 CPU tracking (v2, authoritative)→ Fig 13
│   └── phase7_po4po8/             # Po4, Po8 capacity search           → Fig 9
│
├── experiment_results_a100/       # PLOT DATA — A100 Llumnix head-to-head
│   └── phase57_block_estimated/   # A100 A100x4 × 2 nodes              → Fig 8
│
├── revision/                 # Auxiliary data + older plot scripts
│   ├── cpu_tracker/               # Fanout CPU baseline (pre-Po2)       → Fig 13
│   ├── llumnix_compare/           # A100 Llumnix raw + plot_llumnix_aggregate.py → Fig 8
│   ├── plot_cpu_overhead.py       # (legacy) standalone CPU plotter
│   └── cpu_overhead.png, cpu_overhead.pdf  # legacy outputs
│
├── archival/                      # LEGACY — not used by current plotters
│   └── a30_legacy/
│       ├── main_fanout_estimated/ # pre-phase11 Fanout est sweep
│       ├── main_fanout_oracle/    # pre-phase11 Fanout oracle sweep
│       ├── main_po2_estimated/    # pre-phase12 Po2 est sweep
│       ├── main_po2_oracle/       # pre-phase12 Po2 oracle sweep
│       ├── po2_estimated/         # pre-phase12 Po2 snapshot
│       ├── po2_len_oracle/        # pre-phase12 Po2 oracle snapshot
│       ├── burstiness_po2/        # pre-phase3_1 burstiness sweep
│       ├── error_heatmap_po2/     # pre-phase3_2 error heatmap base
│       ├── generality/            # pre-phase4_2 generality sweep
│       └── llumnix_verify/        # Llumnix verification runs
│
├── Block-Pow2/                    # Paper source (main.tex, socc_*.tex, figures, bib)
│   ├── main.pdf                   # submission PDF (v25, SYSTOR 2026 review-ready)
│   ├── PAPER_META.md              # paper provenance + version log
│   └── tmp/
│       ├── regen_figures.py       # master figure regenerator (reads A30 + SOCC/cpu_tracker)
│       ├── aggregate_data.py      # NPZ → macros.tex
│       ├── macros.tex             # LaTeX inline numbers
│       ├── summary.md             # top-line numbers (human-readable)
│       └── aggregated.csv         # long-form scheduler × qps × metric
│
├── Block_paper/                   # Companion SoCC original draft (Fanout-default)
├── block/                         # Scheduler source + artifact-evaluation scripts
│   └── exp/end_to_end_exp_scripts/artifact_evaluation/   # 15+ AE scripts (00-15)
├── vidur/                         # Vidur simulator source (used by Predictor sidecar)
├── scripts/                       # Helper scripts (sync, compare, etc.)
├── docs/                          # REPRODUCE_FROM_SCRATCH.md, EXPERIMENT_GUIDE.md, etc.
├── output/                        # Review outputs (paper-reviewing/2026-04-22-191200/)
└── astrolabe_systor2026_backup_2026-04-23.zip   # Pre-cleanup backup (237 MB)
```

---

## 2. Figure → data-source map

Canonical figure regeneration command:
```bash
python3 Block-Pow2/tmp/regen_figures.py [main|memory|oracle|n|cpu|heat|burst|all]
```

| Figure | Paper section | Data source | regen_figures.py function |
|---|---|---|---|
| Fig 6 (request metrics, 8-panel) | §6.3 | `experiment_results_a30/phase11_main/` + `phase12_po2/` | `fig_request_metrics()` |
| Fig 7 (GPU memory, 3-row) | §6.3 GPU memory dynamics | same as Fig 6 | `fig_gpu_memory()` |
| Fig 8 (A100 Llumnix head-to-head) | §6.4 | `revision/llumnix_compare/` | `plot_llumnix_aggregate.py --default po2` |
| Fig 9 (N-tuning bars) | §6.5 | `phase12_po2/` + `phase7_po4po8/` + `phase11_main/` @ QPS=32 | `fig_n_ablation()` |
| Fig 10 (oracle upper bounds) | §6.6 | `phase11_main/` + `phase12_po2/` (est vs oracle × Po2 vs Fanout) | `fig_oracle_comparison()` |
| Fig 11 (burstiness) | §6.7 | `phase3_1_burstiness_po2/` + `phase12_po2/` + `phase11_main/` (α=1.0 Poisson baseline) | `fig_burstiness()` |
| Fig 12 (prediction-error heatmap) | §6.8 | `phase3_2_error_heatmap_po2/` + `_rerun/` + `_redo2/` (priority merge); Llumnix overlay from `phase11_main/` | `fig_heatmap()` |
| Fig 13 (CPU + memory per predictor) | §6.9 | `phase7_cpu_tracker_po2_v2/` (Po2) + `revision/cpu_tracker/` (Fanout) | `fig_cpu_overhead()` |

**Pre-existing figures** (not regenerated): Fig 1 (preemption), Fig 2 (chunked prefill), Fig 3 (workflow), Fig 4 (Astrolabe architecture), Fig 5 (latency prediction metrics) — these are illustrations/schematic; PNGs live in `Block-Pow2/figure/` and are edited out-of-band.

---

## 3. Table → data-source map

| Table | Paper section | Source |
|---|---|---|
| Table 1 — Response Length Prediction | §6.2 | Offline RoBERTa eval on ShareGPT 10k test split (not in experiment_results_a30/; reported from training run) |
| Table 2 — Per-scheduler TTFT/E2E snapshot | §6.3 | `phase11_main/` + `phase12_po2/` @ QPS ∈ {20,24,28,32,36} |
| Table 3 — Astrolabe capacity across configurations | §6.10 | `phase4_2_generality/` (bs=24, cs=2048) + `phase4_2_qwen/` (Qwen2-7B) + `phase4_2_burstgpt_refine/` (BurstGPT oracle) |

---

## 4. LaTeX inline-number macros

Defined in `Block-Pow2/tmp/macros.tex` (auto-generated by `aggregate_data.py`). Regenerate:
```bash
python3 Block-Pow2/tmp/aggregate_data.py
```

Key macros used in body: `\capPoTwoEst{}` (31.6), `\capFanoutEst{}` (31.7), `\capLlumnix{}` (31.5), `\capPoFourEst{}` (31.9), `\capQwenPoTwoOracle{}` (78.5), `\cpuPoTwoMeanAtTwenty{}` (20.3), `\cpuFanoutMeanAtTwenty{}` (56.3), `\cpuPoTwoVsFanoutRatio{}` (2.8), `\sloSec{}` (10). Full list in `PAPER_META.md`.

---

## 5. Cleanup history (2026-04-23)

The following was removed or relocated in the 2026-04-23 cleanup, after the v25 SYSTOR submission was committed:

**Deleted** (recoverable from `astrolabe_systor2026_backup_2026-04-23.zip`):
- `experiment_results_a30_final_sync/` (658 MB) — full mirror of `experiment_results_a30/`
- `revision/burstiness/` (13 MB) — superseded by `fig_burstiness()`
- `revision/po2/` (30 MB) — superseded by `fig_n_ablation()`
- `revision/prediction_error/` (56 MB) — superseded by `fig_heatmap()`
- `revision/a100_llama70b_results/` (292 KB) — vestigial

**Relocated** to `archival/a30_legacy/`:
- Pre-phase subdirs of `experiment_results_a30/`: `main_{fanout,po2}_{estimated,oracle}/`, `po2_estimated/`, `po2_len_oracle/`, `burstiness_po2/`, `error_heatmap_po2/`, `generality/`, `llumnix_verify/` (131 MB total, superseded by corresponding `phaseN_*` dirs)

**Total freed from active tree**: ~889 MB (757 MB deleted + 131 MB moved to archival). The backup zip preserves the pre-cleanup state.

---

## 6. Rebuild recipes

```bash
# Regen all figures from NPZ data
python3 Block-Pow2/tmp/regen_figures.py all

# Regen specific figure
python3 Block-Pow2/tmp/regen_figures.py heat     # Fig 12 prediction-error heatmap
python3 Block-Pow2/tmp/regen_figures.py burst    # Fig 11 burstiness
python3 Block-Pow2/tmp/regen_figures.py n        # Fig 9 N-tuning
python3 Block-Pow2/tmp/regen_figures.py oracle   # Fig 10 oracle upper bounds
python3 Block-Pow2/tmp/regen_figures.py cpu      # Fig 13 CPU overhead
python3 Block-Pow2/tmp/regen_figures.py main     # Fig 6 request metrics
python3 Block-Pow2/tmp/regen_figures.py memory   # Fig 7 GPU memory dynamics

# Regen Fig 8 (A100 Llumnix)
python3 revision/llumnix_compare/plot_llumnix_aggregate.py --default po2

# Regen inline-number macros
python3 Block-Pow2/tmp/aggregate_data.py

# Rebuild paper PDF
cd Block-Pow2 && latexmk -pdf main.tex
```
