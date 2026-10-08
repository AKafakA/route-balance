# Astrolabe Paper Review + Revision Handoff

**Date of review**: 2026-04-22
**Date of final revision**: 2026-04-23
**Submission target**: SYSTOR 2026 (full paper track, 12-pp body limit)

---

## Contents

| File | Purpose |
|---|---|
| `review.md` | Original EuroSys-style review of v1 (pre-revision) — 6/10 Weak Accept |
| `metadata.json` | Review metadata (paper title, scores, reviewer confidence) |
| `missing_references.json` | Search results for potentially missing citations (FastServe, SSJF, Preble, DualMap, CascadeInfer) |
| `astrolabe_v1.pdf` … `astrolabe_v25_review.pdf` | Frozen PDF snapshots at each revision step |
| `README.md` | This file |

## PDF version map

| Version | File | Key change | Git commit |
|---|---|---|---|
| v1 | `astrolabe_v1.pdf` | Pre-review-feedback baseline | `8f8f9fe` |
| v2 | `astrolabe_v2.pdf` | Framing improvements (herding signposting, §6.2↔§6.8 cross-ref, Llumnix- self-justification). Overweight (conclusion on p13). | `d996cc6` |
| v3 | `astrolabe_v3.pdf` | v2 + all figures 0.95→0.90. Still overweight. | — |
| v4 | `astrolabe_v4.pdf` | v3 + Llumnix- concise + remove "fresh vLLM deploy" + `Astrolabe-Po2`→`Astrolabe` global rename | — |
| v5 | `astrolabe_v5.pdf` | v4 + all 7 figures regenerated with new labels (per-figure DISPLAY dicts in `regen_figures.py`); Fig 12 Llumnix colorbar overlay | — |
| v6 | `astrolabe_v6.pdf` | v5 + Background section paragraph titles removed; Llumnix- formula uses `\allowbreak` for column break | — |
| v7–v14 | (intermediate) | Word-level trims (1,800→1.8k, 10,000→10k, 1,151 ms→1.15 s, etc.); caption compression; conclusion shortening | — |
| v15–v18 | (intermediate) | §6.8 opening condensation + clamp-rationale; §6.9 hardware dup removed; Fig 10 caption shortened | — |
| v19 | `astrolabe_v19.pdf` | 3-item review-feedback pass: abstract 123× → orders-of-magnitude; §4.1 "first systems" weakened; §6.8 Llumnix-crossover acknowledged | — |
| v22 | `astrolabe_v22.pdf` | Figures regenerated + Fig 4 PNG updated + paper polish; 0.90 figure widths; body fits 12 pp | `f7614fd` |
| v23 | `astrolabe_v23.pdf` | Test: 0.95 figure widths → overflows | — |
| v24 | `astrolabe_v24.pdf` | Binary-search sweet spot at **0.91 figure widths** (0.92/0.93/0.95 all overflow) | — |
| **v25** | **`astrolabe_v25_review.pdf`** | **Final: 0.91 widths + `review,anonymous` flags → submission-ready double-blind** | **`87e6752`** |

**Canonical submission PDF**: `astrolabe_v25_review.pdf`

## Review-driven changes (v1 → v25)

### Framing improvements (addressed original review's 3 major concerns)
1. **Herding evidence**: §6.7 renamed "Scheduler Herding Under Bursty Workloads"; §1 contribution 1 adds forward-ref to §6.7. Converts the burstiness sweep from "nice ablation" into "the herding measurement."
2. **Simulator confidence**: §6.2 now cross-references §6.8's ±100% noise envelope when reporting the 16.6% A100 simulator error. Frames the result as "bounded by our own robustness analysis" rather than a standalone concern.
3. **Llumnix- baseline fairness**: §5 baseline bullet self-justifies at point of introduction ("infeasible on 25 Gbps A30 Ethernet, single-GPU-per-host, cross-host KV transfers saturate bandwidth"). §6.4 runs full Llumnix on 100 Gbps A100 for the fair head-to-head.

### Figure updates
- Global rename: `Astrolabe-Po2` → `Astrolabe` in all figure labels, prose, and Table 3.
- Fig 9 (N-tuning): `Po2 (Astrolabe)` / `Po4` / `Po8` / `Fanout`
- Fig 10 (oracle): `Astrolabe` / `Astrolabe-oracle` / `Astrolabe-fanout` / `Astrolabe-fanout-oracle`
- Fig 12 (error heatmap): added Llumnix- degradation reference line on both colorbars (+4.0% mean, +4.1% P99)
- Fig 4 PNG regenerated with "Astrolabe" branding (was "Block System")
- Fig 13 caption compressed to 1 line: "Predictor's CPU and memory across QPS"

### Paper polish
- Abstract: `123×` per-token latency → `orders-of-magnitude lower` (specific number retained in §6.4 prose)
- §4.1: "among the first systems…" → "Astrolabe differs from prior length-estimation schedulers (e.g., TetriServe) by coupling each prediction with per-instance simulation…"
- §6.8: direct Table 1 / Fig 5 refs instead of §6.2; clamp rationale added; Llumnix colorbar crossover acknowledged
- §6.4: A10 vs A100 clarified ("both Ampere, so A10 is a lower-tier variant")
- Algorithm 1: `Fallback(r)` → `reject r and throw exception` (removes undefined primitive)
- Background (§2): paragraph titles removed (LLM Inference, Continuous Batching, Chunked Prefill) for cleaner flow
- Word-level compressions for 12-pp fit: `1,800`→`1.8k`, `10,000`→`10k`, `1,151 ms`→`1.15 s`, `3,204 ms`→`3.2 s`, etc.
- Figure widths tuned to 0.91× linewidth (Pareto-optimal; 0.92+ overflows)

## Camera-ready reversal checklist

After SYSTOR acceptance, undo these submission-only changes:

1. `\documentclass[sigplan,nonacm,review,anonymous]{acmart}` → `\documentclass[sigplan,nonacm]{acmart}`
2. `\settopmatter{printacmref=false, printfolios=true}` → `\settopmatter{printacmref=false, printfolios=false}` (remove page numbers)
3. Add per eRights form: `\acmDOI{...}`, `\acmISBN{979-8-4007-1181-7}`, `\setcopyright{acmcopyright|acmlicensed|rightsretained}`
4. De-anonymize the 4open.science repo link or replace with the permanent repo URL
5. Verify author block in `.tex` lines 52/57 is correct (currently "University of Cambridge" — confirm affiliations, ORCID IDs)

## Reproducibility pointers

- **Data source**: `experiment_results_a30/` (327 NPZs, 17 phase subdirs) and `experiment_results_a100/` (A100 Llumnix head-to-head data)
- **Aggregation**: `Block-Pow2/tmp/aggregate_data.py` → `tmp/aggregated.csv` + `tmp/macros.tex` + `tmp/summary.md`
- **Figure regeneration**: `python3 Block-Pow2/tmp/regen_figures.py [all|main|memory|oracle|n|cpu|heat|burst]`
- **A100 Llumnix figure**: `python3 SoCC_revision/llumnix_compare/plot_llumnix_aggregate.py --default po2`
- **PDF build**: `cd Block-Pow2 && latexmk -pdf main.tex`

See `Block-Pow2/PAPER_META.md` for the full provenance map (figures → scripts → NPZ paths → LaTeX macros).

## Independent-review note

The v1 → v25 cycle was run with Claude Opus 4.7 acting as first-pass reviewer *and* editor. My (the agent's) review is therefore **not independent** — I helped shape the final text. For a true second opinion, hand `astrolabe_v25_review.pdf` to a fresh agent (e.g., Codex) with the original `review.md` for context.
