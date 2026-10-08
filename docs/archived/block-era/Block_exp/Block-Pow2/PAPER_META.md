# Astrolabe (Block-Pow2) — Paper Provenance & Meta Document

Living document tracking how the paper's data, figures, tables, and inline numbers are produced, and the current open issues. **Update this whenever a figure, table, or macro changes.**

> **See also:** [`../DATA_INDEX.md`](../DATA_INDEX.md) — top-level repo data layout after the 2026-04-23 cleanup: active `experiment_results_{a30,a100}/` + `SoCC_revision/` plot_data, vs legacy `archival/a30_legacy/`. The per-figure data-source map in §2 below is a subset of DATA_INDEX.md §2.

Last updated: 2026-04-23
PDF: `Block-Pow2/main.pdf` (Po2 default, `sigplan,nonacm,review,anonymous` class for SYSTOR 2026 submission) — **15 pp total, 12 pp body** (meets SYSTOR 12-pp body limit; submission-ready double-blind)
Companion PDF: `Block_paper/main.pdf` (Fanout default, original SoCC draft) — 18 pp total, ~13 pp body

## Review feedback cycle (Apr 22-23 2026)

A multi-round framing and polish cycle converted v1 (pre-review snapshot)
through v25 (submission-ready). Frozen PDF snapshots at each major step
are in `output/paper-reviewing/2026-04-22-191200/`:

| Version | Commit | Key change |
|---|---|---|
| v1 | `8f8f9fe` | Pre-review-feedback snapshot |
| v2 | `d996cc6` | Framing improvements (herding signposting, §6.2↔§6.8 cross-ref, Llumnix- self-justification) |
| v22 | `f7614fd` | Figures regenerated with Astrolabe naming; Fig 12 Llumnix colorbar overlay; word-level trims to fit 12-page body |
| v25 | `87e6752` | Figures 0.91× linewidth (Pareto); `review,anonymous` flags added |

Astrolabe-Po2 → Astrolabe global rename in v4-v5. Full Fig naming
convention in §2 below.

## Submission-ready state (SYSTOR 2026)

- `\documentclass[sigplan,nonacm,review,anonymous]{acmart}` — double-blind per SYSTOR CFP
- `\settopmatter{printacmref=false, printfolios=true}` — page numbers on, ACM ref suppressed
- Body ends on page 12 ("...paves predictive scheduling as a new direction for LLM serving optimization."); references pp. 13-15
- Anonymous repo: `https://anonymous.4open.science/r/Astrolabe-BA60/`
- Author block in `.tex` (`University of Cambridge`, lines 52/57) hidden by `anonymous` flag in rendered PDF

**Camera-ready reversal** (after acceptance):
```diff
- \documentclass[sigplan,nonacm,review,anonymous]{acmart}
+ \documentclass[sigplan,nonacm]{acmart}
- \settopmatter{printacmref=false, printfolios=true}
+ \settopmatter{printacmref=false, printfolios=false}
```
Plus add `\acmDOI{...}`, `\acmISBN{979-8-4007-1181-7}`, `\setcopyright{...}` per eRights form.

---

## 1. Data pipeline

```
experiment_results_a30/                    (315 NPZs, campaign output)
  phase11_main/      Fanout + 5 baselines × QPS 20-36
  phase12_po2/       Po2 est/oracle × QPS 20-36
  phase7_po4po8/     Po4, Po8 capacity search
  phase7_cpu_tracker_po2_v2/   Po2 CPU + memory tracking
  phase3_1_burstiness_po2/     Po2 vs Llumnix × {α=0.25,0.5,1.0,2.0}
  phase3_2_error_heatmap_po2/          base 16-cell grid
  phase3_2_error_heatmap_rerun/        5 "suspect" cells (priority override)
  phase3_2_error_heatmap_redo2/        (100,0) triple-check (highest priority)
  phase4_qwen/       Qwen2-7B capacity sweep
  phase4_burstgpt/   BurstGPT oracle capacity
  phase4_bs24/, phase4_chunk2048/

SoCC_revision/cpu_tracker/   A30 Fanout CPU + memory (pre-Po2 baseline)

         │ aggregated by
         ▼
Block-Pow2/tmp/aggregate_data.py
         │ outputs
         ├─▶ tmp/aggregated.csv  (long-form: scheduler, N, est, qps, metric, value)
         ├─▶ tmp/macros.tex      (LaTeX macros used in main.tex — see §3 below)
         └─▶ tmp/summary.md      (human-readable top-line numbers)

Block-Pow2/tmp/regen_figures.py
         │ reads experiment_results_a30/ + SoCC_revision/cpu_tracker
         │ writes figures
         ├─▶ figure/exp_plots/cluster_metrics/qps.png     (Fig 6)
         ├─▶ figure/exp_plots/cluster_metrics/linear.png  (Fig 7)
         ├─▶ figure/SoCC_revision/po2_comparison.png           (Fig 9)
         ├─▶ figure/SoCC_revision/prediction_error_heatmap.png (Fig 11)
         ├─▶ figure/SoCC_revision/cpu_overhead.png             (Fig 12)
         └─▶ figure/SoCC_revision/burstiness_lines.png         (Fig 10)
```

**Rebuild commands:**
```bash
# Full regen from NPZ → macros + all figures
python3 Block-Pow2/tmp/aggregate_data.py
python3 Block-Pow2/tmp/regen_figures.py        # all 6

# Regen single figure
python3 Block-Pow2/tmp/regen_figures.py main      # Fig 6 (qps.png)
python3 Block-Pow2/tmp/regen_figures.py memory    # Fig 7 (linear.png)
python3 Block-Pow2/tmp/regen_figures.py n         # Fig 9 (po2_comparison)
python3 Block-Pow2/tmp/regen_figures.py heat      # Fig 11 (heatmap)
python3 Block-Pow2/tmp/regen_figures.py cpu       # Fig 12 (cpu_overhead)
python3 Block-Pow2/tmp/regen_figures.py burst     # Fig 10 (burstiness)

# Rebuild PDF
cd Block-Pow2 && pdflatex -interaction=nonstopmode main.tex
```

---

## 2. Figure → source → claim map

| # | File | Section | Function in `regen_figures.py` | Data source | Key claim |
|---|------|---------|--------------------------------|-------------|-----------|
| 1 | `figure/introduction/preemption.png` | §2 (Background) | pre-existing | — | vLLM preemption illustration |
| 2 | `figure/introduction/chunked_prefill.png` | §2 | pre-existing | — | CP vs prioritized-prefill |
| 3 | `figure/introduction/inference_workflow.png` | §3 (Related Work / Motivation) | pre-existing | — | Serving stack overview |
| 4 | `figure/block_diargram.png` | §4 (Design) | pre-existing | — | Astrolabe architecture |
| 5 | `figure/exp_plots/cluster_metrics/profiling.png` | §6.2 (Prediction Accuracy) | pre-existing | sampled prediction RPC trace | Latency prediction error <15%, top-rank hit 40–80% |
| **6** | `figure/exp_plots/cluster_metrics/qps.png` | §6.3 | `fig_request_metrics()` | Phase 1.1 + Phase 1.2 NPZs × QPS 20–36. Line plots = Po2-est + 5 non-predictive baselines (6 series). Capacity bar = same 6 series. | **Po2-est dominates all non-predictive baselines; 31.6 QPS vs 31.5 for Llumnix-.** Fanout/Po2-oracle reference variants moved to Fig 10 for clarity. |
| **7** | `figure/exp_plots/cluster_metrics/linear.png` | §6.3 (GPU memory subpar.) | `fig_gpu_memory()` | Same 6-scheduler roster as Fig 6. `avg_gpu_blocks` / `var_gpu_blocks` / `num_preempted` from Phase 1.1 + 1.2 | **Po2-est balances memory; baselines show high variance and preemptions** |
| **8** | `figure/SoCC_revision/llumnix_comparison_v2_po2.png` | §6.4 (Llumnix) | external (A100 cluster run) | A100 sweep/benchmark logs + recovered Llumnix dense table (not in A30 repo); Llumnix 0.1.1 with rayrpc | **Po2 beats full Llumnix migration 2.57× throughput / 123× per-token at QPS=36** |
| **9** | `figure/SoCC_revision/po2_comparison.png` | §6.5 (N-tunable) | `fig_n_ablation()` | Phase 1.2 Po2-est, phase7_po4po8 Po4/Po8, Phase 1.1 Fanout-est @ QPS=32 (all est) | **Po4 peaks (31.9 QPS), Po8/Fanout match (31.7); Po2 within 0.3 QPS at 6× lower cost** |
| **10** | `figure/SoCC_revision/oracle_comparison.png` | **§6.6 (Oracle)** — NEW | `fig_oracle_comparison()` | Phase 1.2 Po2-est + Po2-oracle, Phase 1.1 Fanout-est + Fanout-oracle @ QPS=32 | **Oracle lengths add ~0.8 QPS for Po2 (31.6→32.4), ~0.9 QPS for Fanout (31.7→32.6) — ~3% cap. Oracle gap independent of N.** |
| **11** | `figure/SoCC_revision/burstiness_lines.png` | §6.7 | `fig_burstiness()` | phase3_1_burstiness_po2 × {0.25,0.5,1.0,2.0} | **Po2 6–9% lower mean e2e than Llumnix- under α=0.25 bursts** |
| **12** | `figure/SoCC_revision/prediction_error_heatmap.png` | §6.8 | `fig_heatmap()` | phase3_2_error_heatmap_po2 + _rerun + _redo2 (priority merge). **0%-noise baseline cell = Po2-est with σ=0 (NOT Po2-oracle — that's a separate variable, length source)** | **Worst-case +6.3% mean degradation at 100% length / 25% latency noise; 100%/100% is +5.1%; P99 stays below Llumnix-** |
| **13** | `figure/SoCC_revision/cpu_overhead.png` | §6.9 | `fig_cpu_overhead()` | phase7_cpu_tracker_po2_v2 (Po2) + SoCC_revision/cpu_tracker (Fanout) | **Po2 uses 2.8× less CPU per predictor than Fanout; memory invariant in N** |

**Heatmap merge rule** (`fig_heatmap()`): for each (len_err, lat_err) cell, use the value from the highest-priority phase that contains it: `po2 (prio 0) < rerun (prio 1) < redo2 (prio 2)`. The noise-free (0,0) baseline comes from Phase 1.2 QPS=32 Po2-est. Source NPZs are never modified.

---

## 3. Table → source → claim map

| # | Location | Data source | Claim |
|---|----------|-------------|-------|
| **1** | §6.2 Table 1: Response Length Prediction | RoBERTa regressor eval (ShareGPT 10k test split, offline) | MAE 78.8, MAPE 24.4%, Acc-100 = 77.2% — comparable to 7B prompt-based Sequence-Scheduling |
| **2** | §6.9 Table 2: Generality Study (Astrolabe-Po2 serving capacity) | phase4_{bs24, chunk2048, qwen, burstgpt} NPZs + phase11 Llumnix baseline | **Po2 matches or beats Llumnix- across bs∈{24}, cs∈{2048}, Qwen2-7B, and BurstGPT (oracle)** — robustness to backend, model, dataset changes |

Table 2 formatting: `\resizebox{\linewidth}{!}` is required — without it the Gain row (`7.1%/9.4% | 1.0%/1.9% | 6.0%/12.6% | / /4.7%`) overflows the column width.

---

## 4. LaTeX macros (inline numbers in body text)

Generated by `tmp/aggregate_data.py` into `tmp/macros.tex`; loaded via `\input{tmp/macros}` at the top of `main.tex`.

| Macro | Value | Source | Where used |
|-------|-------|--------|------------|
| `\sloSec` | 10 | fixed design decision | All SLO references |
| `\capPoTwoEst` | 31.6 | phase12_po2 + SLO interpolation | Abstract, §6.3, §6.6 |
| `\capPoTwoOracle` | 32.4 | phase12_po2 len_est=false | §6.3 |
| `\capFanoutEst` | 31.7 | phase11_main N=12 len_est=true | §6.3, §6.6 |
| `\capFanoutOracle` | 32.6 | phase11_main N=12 len_est=false | §6.3 |
| `\capLlumnix` | 31.5 | phase11_main min_lunmnix_load | §6.3 |
| `\capPoFourEst` | 31.9 | phase7_po4po8/po4 float search | §6.6 (peak) |
| `\capPoEightEst` | 31.7 | phase7_po4po8/po8 float search | §6.6 |
| `\capQwenPoTwoEst` | 73.9 | phase4_qwen/po2_est | §6.9 Table 2 |
| `\capQwenPoTwoOracle` | 78.5 | phase4_qwen/po2_oracle | §6.9 Table 2 |
| `\capQwenLlumnix` | 69.7 | phase4_qwen/llumnix | §6.9 Table 2 |
| `\capBurstGPTPoTwoOracle` | 64.5 | phase4_burstgpt/po2_oracle | §6.9 Table 2 |
| `\cpuPoTwoMeanAtTwenty` | 20.3 | phase7_cpu_tracker_po2_v2 QPS=20 | §6.8 CPU Overhead |
| `\cpuFanoutMeanAtTwenty` | 56.3 | SoCC_revision/cpu_tracker QPS=20 N=12 | §6.8 CPU Overhead |
| `\cpuPoTwoVsFanoutRatio` | 2.8 | cpuFanoutMean/cpuPoTwoMean | §6.3, §6.8 |
| `\poFourEstTail` | 39081 | phase7_po4po8/po4 e2e P99 ms | §6.6 |
| `\poEightEstTail` | 38095 | phase7_po4po8/po8 e2e P99 ms | §6.6 |
| `\poFourEstTPut`, `\poEightEstTPut` | 16210, 16185 | throughput tok/s | §6.6 |

**Invariant:** Never edit `tmp/macros.tex` by hand. Re-run `tmp/aggregate_data.py` after any campaign update.

---

## 5. Inline numbers NOT driven by macros (hardcoded in tex)

These are frozen at the time of the rewrite; update by hand if the underlying source changes:

| Number | Where | Source | Reason not a macro |
|--------|-------|--------|--------------------|
| 2.57× / 36× / 123× | §6.4 Llumnix comparison | A100 cluster QPS=36 (external, not in A30 repo) | One-off headline; no re-run planned |
| 2.58× / 40× / 135× (CP variant) | §6.4 | A100 cluster, Astrolabe-Po2 with CP | Same |
| 18,662 vs 7,273 tok/s | §6.4 | A100 cluster QPS=36 | Same |
| 77.15% / 69.93% (Acc-100 / Acc-50) | §6.2 Table 1 | RoBERTa eval script output | Table generated once |
| MAPE 16.6% (70B prediction accuracy) | §6.2.2 | A100 70B TP=4 sampling run | External one-off |
| 6–9% burstiness lead | §6.7 | phase3_1 α=0.25 Po2 vs Llumnix | Inline claim derived from figure |
| 7.1%/9.4%, 1.0%/1.9%, 6.0%/12.6%, /,/4.7% | Table 2 Gain row | Pair-ratios over Llumnix baseline | Computed from Po2 + Llumnix macros but embedded explicitly |

---

## 6. Section-by-section key conclusions

| § | Title | Primary claim |
|---|-------|---------------|
| Abstract | — | Astrolabe-Po2 matches Fanout at 6× less probe cost, 2.8× less CPU; beats Llumnix 2.57×/123× under migration-hostile load; 6–9% faster under bursts |
| §1 Intro | — | 3 costs of full-fanout prediction-guided dispatch: O(N) control-plane, O(N) CPU, and scheduler herding — all addressed by randomized Po-N |
| §2 Background | — | vLLM, paged attention, continuous batching, chunked prefill primer |
| §3 Related Work & Motivation | — | Length prediction + simulation-based metrics combine into one-shot dispatch; contrasts with migration-based (Llumnix) |
| §4 Design | — | Predictor sidecars + stateless global scheduler + Po-N probe; simulation is per-instance local (Vidur-adapted) |
| §5 Implementation | — | 4,000 LoC, FastAPI+REST, vLLM 0.7.2 + FlashInfer; baselines implemented in same framework |
| §6.1 Setup | — | 12×A30 Llama-2-7B, 2×4-A100 Llama-2-7B (TP=1) / Llama-2-70B (TP=4), ShareGPT primary |
| §6.2 Prediction Accuracy | — | RoBERTa-base 125M MAE=78.8/MAPE=24.4%; simulator e2e error 10–15%, 70B generalizes at MAPE=16.6%; relative ranking is what matters |
| §6.3 Request Latency Performance (+ memory) | Fig 6, 7 | **Po2-est beats non-predictive baselines (31.6 QPS vs Llumnix- 31.5); memory variance and preemptions lower across cluster** |
| §6.4 End-to-end Comparison with Llumnix | Fig 8 | **Po2 2.57× throughput, 36× e2e, 123× per-token lower than Llumnix at QPS=36 on A100** |
| §6.5 N-tunable Dispatch Family | Fig 9 | **Po4 peaks (31.9 QPS); Po2 is 1% behind at half the probe cost; randomization wins at bursts where Fanout herds** |
| §6.6 Oracle-Length Upper Bounds (NEW) | Fig 10 | **Oracle lengths recover ~0.8 QPS for Po2 (31.6→32.4) and ~0.9 for Fanout (31.7→32.6); oracle-minus-est gap independent of N; heavier length predictor is Pareto-dominated by Po2 CPU savings** |
| §6.7 Robustness to Bursty Workloads | Fig 11 | **Po2 beats Llumnix- 6–9% under α=0.25; advantage comes from randomized probe, not CP** |
| §6.8 Sensitivity to Prediction Errors | Fig 12 | **+6.3% worst-case mean degradation at 100% length / 25% latency noise; 100%/100% is +5.1%; relative ranking absorbs most noise. Note: heatmap's (0,0) cell is Po2-est with σ=0, NOT Po2-oracle (which is a separate variable, length source).** |
| §6.9 CPU Overhead: Po2 vs Fanout | Fig 13 | **Po2 uses 20.3% of one core vs Fanout 56.3% at QPS=20 — 2.8× reduction holds across QPS; memory flat in N** |
| §6.10 Generality Study | Table 2 | **Po2 matches or beats Llumnix- on bs=24, cs=2048, Qwen2-7B, BurstGPT** |
| §7 Conclusion | — | Po2 + simulation + length prediction makes Astrolabe practical for commodity network-constrained clusters |

Note: §6.6 Oracle-Length Upper Bounds added 2026-04-21 (separates the oracle-vs-est axis from the N axis). Old §6.4 GPU Memory merged into §6.3 as a trailing subparagraph.

---

## 7. Open issues and discussion

### 7.1 Page limit — RESOLVED (Apr-22)
- **Final body: ~12 pp (meets SYSTOR 12-pp limit).** Final PDF 16 pp total (body + 4 pp references).
- **Applied trims (cumulative):** caption shortening; 4-obs bulleted lists → compact paragraphs; §6.3 Variants/Capacity/Latency/Overhead sub-headers folded into one paragraph; §6.4 GPU memory merged into §6.3 as trailing subparagraph; conclusion condensed to 2 sentences; §6.10 Generality Study collapsed to one paragraph (no per-variation number repetition); §6.1 Experimental Setup Testbed + Data/Config paragraphs merged; §2 Background prose trimmed; §5 Implementation cross-refs §4.1 instead of restating API contract; Algorithm 1 formatted as compact algpseudocode block (no inline step labels); paragraph-level word trims on 8 runover-prone paragraphs; Fig 4 (Astrolabe Architecture) shrunk to 0.95\linewidth; Fig 7 declaration moved after the GPU-memory subparagraph (separates from Fig 6); all socc_*.tex figures at 0.95\linewidth; \raggedbottom + `[!tb]` placement; \settopmatter{printacmref=false}; \resizebox on Table 2.
- **Biggest single win:** replacing long Gemini bibtex entry (hundreds of authors rendered verbatim) with the short `team2023gemini` entry (`... and others`) — saved ~3 pages of References. Same treatment applied to GPT-4 (`achiam2023gpt`). Total PDF dropped 19 pp → 16 pp from these two bib entries alone.
- **What was NOT touched** (user preferences preserved): no template modifications (no \titlespacing, no margin override, no font-size change); §4.1/§4.2/§4.3 subsection structure kept (merging would trivialize the paper); Vidur paragraph in §5 kept verbatim (key implementation basis); all figures preserved, no content reduction.

### 7.2 Review-surface weaknesses (from internal NeurIPS/generic review pass, `output/paper-reviewing/`)
- **No variance / seed reporting.** "Po2 matches Fanout within noise" is a point estimate; no σ, CI, or multi-seed.  → Action: if time permits, run 3 seeds at QPS=30/32/34 and add ±σ to capacity bars.
- **Scheduler herding not directly measured.** Argued via α=0.25 latency gap, not via per-instance dispatch distribution. → Action: add a small supplementary plot of KL(empirical || uniform) under Fanout vs Po2 in bursty arrivals.
- **A100 comparison restricted to Llama-2-7B.** §6.2.2 shows 70B prediction accuracy but the Llumnix head-to-head is on 7B. → Action: acknowledge as future work OR schedule a 70B Llumnix run if A100 time available.
- **Figure 7 caption/content mismatch (fixed 2026-04-21).** Was showing TTFT/Throughput under a GPU-memory caption; fixed by regenerating `linear.png` with proper `avg_gpu_blocks`/`var_gpu_blocks`/`num_preempted` data.
- **ML novelty narrow for NeurIPS main track.** If retargeting NeurIPS, recommend ML Systems track or adding a small theoretical section (Chernoff bound on probe-set collision under correlated arrivals).

### 7.3 Open reproducibility questions
- **Po4 / Po8 rerun.** Only single-run numbers from phase7_po4po8. If capacity hierarchy (Po4=31.9 > Po8=31.7 ≈ Fanout=31.7) flips on a re-run, the "Po4 is the sweet spot" narrative needs adjustment.
- **BurstGPT has no length predictor.** Only oracle variants evaluated because the dataset provides only length traces without prompt text. Ideally, train a BurstGPT-prompt-synthesized predictor, but that's effectively building a new dataset.
- **CPU tracker memory measurements depend on predictor fork behavior.** `phase7_cpu_tracker_po2_v2` and `SoCC_revision/cpu_tracker` may use slightly different sampling cadences; `regen_figures.fig_cpu_overhead()` collapses to mean-per-replica which is robust.

### 7.4 Missing related work (from internal review)
- Nexus (SOSP '19), Clockwork (OSDI '20), AlpaServe (OSDI '23) — classical DNN serving precedents using Po-N / simulation
- Consistent Hashing with Bounded Loads (Mirrokni 2018) — theoretical backbone
- Join-Idle-Queue (Lu 2011) — pre-Po-N classical LB
- FastServe, Preble, LoongServe — contemporary LLM serving with different scheduling axes

Where to add: §3 Related Work. Budget: ~1 paragraph if page limit allows.

### 7.5 Figure / table style decisions
- **Figure 7 legend:** currently omits Fanout-est, Fanout-oracle, Po4, Po8 (kept to 7 schedulers for readability). If a reviewer asks for full scheduler set, add in `regen_figures.fig_gpu_memory()`.
- **Figure 11 heatmap vmax:** currently auto-computed per-panel (`max(6.0, nanmax(abs(grid)))`). If the P99 panel's range looks too compressed vs the mean panel, consider a shared color scale.
- **Figure 12 memory panel:** y-axis in MB. If reviewers object that "1200 MB" is too abstract, consider adding a secondary y-axis in % of node RAM (128 GB).
- **Table 2 `/` notation:** "/" means "not applicable because BurstGPT has no prompt text." Caption used to explain this; trimmed for space. If a reviewer flags confusion, restore 1-line caption addendum.

### 7.6 Branch / commit state (Apr-22 final)
- Branch: `paper-block-pow2-rewrite` (local, not pushed to remote)
- Document class: `\documentclass[sigplan,nonacm]{acmart}` (SYSTOR requirement — single-conf 2-col with author-box header, no ACM copyright footer, no line numbers).
- Target venue: SYSTOR 2026 (or equivalent sigplan-style conference)
- Final PDF: **16 pp total, body ~12 pp, references ~4 pp**.
- Key commits (most recent first):
  - short GPT-4 bibtex (`achiam2023gpt`) — final page-count reduction
  - `b6f8da3` short Gemini bibtex + Qwen3/DeepSeek cites + intro update — 19→16 pp (biggest single win)
  - `023e3d3` word-level trims across 7 runover-prone paragraphs
  - `5a21b0f` Algorithm default n=2; split GetStatus/PredictMetric; §6.10 trim; figs to 0.95
  - `2c8edbf` redistributed Figs 5/6 across p8/p9; ShareGPT citation widow fix; "becomes"→"is"
  - `b2a96f8` §2 Background + §5 Implementation compression (cross-ref §4.1)
  - `aaef1ad` Fig 4 shrink to 0.82 (later restored to 0.95); §6.1 Testbed/Data compression
  - `0fa9a11` §6.3 fold + §6.4 merge into §6.3 + §7 conclusion to 2 sentences
  - `d6a5a02` figure regen + caption trim + Table 2 overflow fix
  - `f7e1dba` added §6.6 Oracle-Length Upper Bounds (Fig 10); drop oracle variants from Fig 6 line plots
  - `346e652` Fig 6 line plots shrunk to 6 schedulers; capacity bars still include all 9

### 7.7 Runover / layout knobs landed (all CFP-safe)
- `\raggedbottom` — columns end naturally at last paragraph; eliminates mid-column inter-paragraph inflation.
- `[!tb]` on every figure/figure* float — override default placement restrictions.
- `\settopmatter{printacmref=false}` — drop ACM Reference Format block (officially supported for `nonacm`).
- `\resizebox{\linewidth}{!}{...}` on Table 2 (Generality Study) — prevents column overflow.
- `\setlength{\tabcolsep}{4pt}` on Table 2 only (local to that environment, not document-wide).
- Figures 4 / 6 / 7 / 8 / 9 / 10 / 11 / 12 / 13 at 0.95\linewidth (uniform).
- Figure 4 (Astrolabe Architecture) re-anchored: declared before §4 heading to allow earlier float placement.
- Figure 7 (GPU memory) re-anchored: declared after the "GPU memory dynamics" subparagraph in §6.3, so it doesn't bunch with Figure 6 at top of the same page.
- **NOT applied (flagged as CFP-risky):** `\titlespacing*` (section heading spacing override), `\setlength{\parskip}{0pt}`, `\setlength{\textfloatsep}` — these would likely trigger desk-rejection under SYSTOR's "use the template as provided" rule.

---

## 8. Files to touch when modifying the paper

| Change | Files to edit |
|--------|---------------|
| Add a new figure | `tmp/regen_figures.py` (add function), `main.tex` or one of `socc_*.tex` (add `\begin{figure}` block) |
| Update a numerical claim | Run `python3 tmp/aggregate_data.py` → regenerates `tmp/macros.tex` → pdflatex picks up new value |
| Add a new scheduler to Fig 6 | `tmp/regen_figures.py` COLORS / MARKERS / `order` list in `fig_request_metrics()` |
| Merge experiment phases | `tmp/aggregate_data.py` (add source phase dir to `PHASES` loop) |
| Fix page-overflow | Caption trimming first, then paragraph folding, last resort: drop a section/figure |
| Retarget to a new venue | `main.tex` template (ACM `acmart` vs NeurIPS `neurips_2026`); `reference.bib` stays the same |

---

## 9. External artifacts (not in this repo)

- **A100 cluster data** (§6.4 Llumnix comparison, §6.2.2 70B prediction accuracy): the Llumnix comparison was generated from A100 sweep/benchmark logs under `experiment_results_a100/`, summarized into `figure/SoCC_revision/llumnix_comparison_v2_po2.png`; those source logs/raw per-request files are not currently synced to this repo.
- **A100 Llumnix dense table recovery:** the uploaded `astrolabe_systor2026_backup` archive is missing the Llumnix sweep logs. The Llumnix curve in Fig. 8 is preserved in the Claude chat trace printed on 2026-04-17 17:29 from the real CloudLab d8545 run (full dense QPS 16--36, 10k requests per point). A pixel-level check confirms these values match the blue Llumnix curve in `llumnix_comparison_v2_po2.png`.

| QPS | tok/s | mean E2E (ms) | token latency (ms) |
|---:|---:|---:|---:|
| 16 | 8874 | 5925 | 19.4 |
| 19 | 10461 | 6341 | 20.9 |
| 20 | 9226 | 106263 | 1150.6 |
| 24 | 9849 | 125937 | 1367.5 |
| 28 | 8398 | 212944 | 2384.7 |
| 32 | 8209 | 233263 | 2593.7 |
| 36 | 7273 | 300151 | 3204.4 |

At QPS=36, the recovered denominator is 7,273 tok/s, 300.151 s mean E2E, and 3.204 s mean token latency, matching the §6.4 prose ratios.
- **Block_paper (Fanout-default version):** parallel directory `Block_paper/` with its own `main.tex`, `figure/`, and `reference.bib`. Shares most structure but uses Fanout as default and pre-dates the Po2 rewrite.
- **RoBERTa length model checkpoints:** stored on the GPU VM (`wd312@gxp-l40s-2`); not in this repo (too large).

---

## 10. Apr-22 final sprint summary

Between Apr-21 and Apr-22 the paper moved from a two-column ACM sigconf SoCC submission (20 pp with body spilling over the 12-pp limit) to a sigplan SYSTOR-style submission (**16 pp total, body ~12 pp, meets 12-pp limit**).

**Things that worked (roughly by impact):**
1. **Short bibtex entries for author-heavy model papers** (Gemini, GPT-4) — single biggest win, ~3 pp saved.
2. **Paragraph-level word trims on runover-prone paragraphs** — 7-8 paragraphs tightened by 1-3 words each to kill widow lines. Accumulated to ~0.5 pp.
3. **Section content merges / cross-references** — §6.4 GPU Memory merged into §6.3; §5 Implementation cross-refs §4.1 instead of restating API contract; §6.2.1 + §6.2.2 merged into §6.2 `\paragraph` blocks; §6.10 Generality Study collapsed to one paragraph with inline 1)/2)/3).
4. **Figure float redistribution** — moving Fig 6 / Fig 7 declarations to different source locations so they float to different pages.
5. **Uniform 0.95\linewidth on all SoCC-revision figures** — small height reduction lets columns pack more text.

**Things that did NOT move the page count meaningfully** (don't repeat on future papers):
- `\raggedbottom` alone (only redistributes whitespace; doesn't add content capacity)
- `[!tb]` placement hints alone (doesn't shrink floats)
- Switching sigconf → sigplan (actually expanded by 2 pp; had to compress *around* the class change)
- Shortening §6.10 when there was still overflow elsewhere — doesn't cross a page boundary

**Things deliberately not attempted** (desk-reject risk under SYSTOR CFP):
- Custom `\titlespacing` via titlesec
- `\setlength{\parskip}{0pt}`, `\setlength{\textfloatsep}{6pt}`, `\setlength{\abovecaptionskip}{3pt}`
- Any margin / font-size / line-spacing override

**Final submission-ready artifacts:**
- `Block-Pow2/main.pdf` — 16 pp, sigplan/nonacm, SYSTOR-ready.
- Source: `Block-Pow2/main.tex`, `socc_*.tex`, `reference.bib`, `tmp/{macros,aggregate_data,regen_figures}.*`.
- Branch: `paper-block-pow2-rewrite` (local; not pushed).
- Companion: `Block_paper/main.pdf` — Fanout-default draft (18 pp) kept for reviewer comparison.

---

## 11. SYSTOR '26 camera-ready (2026-07-17 →)

Paper #11 ACCEPTED at SYSTOR '26 (A/B/D Accept, C Weak Reject). Camera-ready work moved to
`~/Code/submission/Block-systor/paper/` (own git repo; baseline = accepted Overleaf export `8283a14`).
Key changes vs v25: system renamed Astrolabe→Block, title "Block: Balancing Load in LLM Serving with
Randomized Predictive Scheduling", all reviewer cosmetic fixes, de-anonymized ACM topmatter with ORCIDs,
official template \documentclass[sigconf,10pt], figures regenerated with Block labels (pixel-verified
identical data; Fig 8 legend-strip surgery). Deadline Aug 3 (shepherd) / Aug 10. Pending: eRights
DOI/copyright, chair answers (page limit under sigconf, template link). Public artifact pushed:
github.com/AKafakA/Block branch `systor-ae` (+ README on main). This dir (Block-Pow2) is FROZEN as the
submission-era record.
