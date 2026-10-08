# Baseline Comparison Design — Evaluation Methodology, Baselines, Figures (2026-04-15, revision 2)

**Audience:** codex, for comprehensive review.
**Status:** supersedes the Apr 15 (revision 1) design. See `CODEX_REVIEW_PACKAGE.md` in the same directory for the change-log since revision 1.

**Context:** 4× A30 CloudLab smoke-test window (2026-04-13 → 2026-04-17) is Phase 1; 18-node heterogeneous run (2026-04-26 → 2026-05-06) is Phase 2. This document is the methodology source-of-truth for both phases.

---

## 0. TL;DR

Cara is evaluated against external LLM routers (RouteLLM-mf, Avengers-Pro, vllm-SR, and BEST-Route) on a unified benchmark via a **knob-sweep cost–quality–latency Pareto analysis** with goodput and TTFT/TPOT as hard constraints. Quality is measured via pre-computed deepeval scores from `test_scored_filtered.jsonl` (the HF dataset `asdwb/cara_model_estimator`). Memory-balance is reported as a scheduler-specific contribution in a separate appendix study. Related-work citations draw on **RouterBench** (Hu 2024), **RouterEval** (Huang 2025), and **MoE-CAP** (Jiang 2025); the radar visualisation re-uses the Apache-2.0 `normalize` / `normalize_cost` helpers from the [MoE-CAP-Dashboard](https://huggingface.co/spaces/auto-cap/MoE-CAP-Dashboard).

---

## 1. The Novel-Position Problem

Cara is an integrated online batch scheduler. It combines:

- LPT-style request ordering,
- bucket-based probabilistic feasibility filtering (`CaraCDFFilter` today covers TTFT/TPOT; budget + quality feasibility remain in the legacy inline path — a unified four-constraint bucket filter is planned, see §5.11),
- multi-objective scoring over **latency / quality / cost / cluster memory balance** (four scorer terms, see §5.12), and
- score-based joint model / instance assignment,

— **online, per-request, on a heterogeneous cluster**.

No single prior system occupies this point:

| System | Quality routing | Load dispatch | SLO filter | Cost-aware | Online per-req | Heterogeneous |
|---|---|---|---|---|---|---|
| RouteLLM (Ong ICLR'25), Avengers-Pro (Zhang DAI'25), BEST-Route (Ding 2025), Dual-Pool, Hybrid-LLM (Ding ICLR'24) | ✅ | ❌ | ❌ | partial | ✅ | partial |
| vLLM Semantic Router (SR) | ✅ | ✅ (Envoy weighted-RR) | ❌ | ❌ | ✅ | ✅ |
| QLM, SLOs-Serve, PolyServe | ❌ | partial | ✅ | ❌ | ✅ | ❌ |
| BOute (Jiang MLSys'26, arXiv 2602.10729) | ✅ (threshold-set) | — | ❌ | ✅ | ❌ **offline MOBO** | ✅ |
| Helix, SkyServe, ThunderServe, SpotServe | partial | partial | ❌ | ✅ | ❌ offline | ✅ |
| **Cara** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

The lack of a drop-in prior competitor is both the gap we fill and the evaluation challenge addressed below.

---

## 2. Evaluation Framework — Inherited From Three Precedents

We combine established methodology from three published frameworks rather than inventing our own.

| Source | What we adopt | Where |
|---|---|---|
| **RouterBench** (Hu NeurIPS-D&B'24, arXiv 2403.12031) | Knob-sweep per router → cost-quality operating points → **convex-hull Pareto** → **AIQ** scalar (Average Improvement in Quality). Their `score = λ · P − cost` is structurally Cara's weighted scorer — same knob family. | §4.2, §4.3 |
| **MoE-CAP** (Jiang NeurIPS-D&B'25, arXiv 2412.07067) | 3-axis **CAP radar** diagram (Cost / Accuracy / Performance). Per-axis native-unit scales, axes inverted for lower-is-better metrics, polygon area as visual heuristic. Normalization helper code open-sourced Apache 2.0. | §4.3 |
| **RouterEval** (Huang EMNLP-findings'25, aclanthology 2025.findings-emnlp.208) | Model-level scaling-up phenomenon; comprehensive router benchmark across 12 evals. Complementary to RouterBench (quality-only lens). | §2, §5.17 |
| **LLMRouter library** (ulab-uiuc) | 16-router unified plugin interface (`MetaRouter.route_single/route_batch`). Extensibility reference; no overhead measurement. | §5.17 |
| **BentoML LLM Performance Explorer** | Interactive Pareto dashboard pattern — our static-figure analog. | §5.17 |
| **BOute** (Jiang MLSys'26) | Decompose-per-axis evaluation strategy when no drop-in competitor exists. | §2.b |

### 2.a What RouterBench unifies

Given router `R_θ: prompt → LLM_i` with parameter θ (threshold, willingness-to-pay, scoring-weight, etc.):

- Sweep θ over ≥5 operating points.
- Each point → measured (cost, quality) tuple → scatter on cost–quality plane.
- Compute non-decreasing convex hull per router.
- Report **AIQ** = `(1 / (c_max − c_min)) · ∫ R̃_θ(c) dc` over a common cost range.

Cara's scorer with `w_quality / w_cost` sweep is structurally identical to RouterBench's `λ · P − cost` — we are **one router family on the same operating curve**, not a qualitatively different thing. This framing makes the comparison crisp.

### 2.b BOute's decompose-per-axis strategy

For the integrated-system claim, borrow BOute's §7.2-7.3 pattern:

1. Decompose per axis — show Cara-full's routing-only component matches or beats RouteLLM on routing; feasibility-filter-only component matches QLM / SLOs-Serve on filtering; joint multi-objective scoring beats any naive composition.
2. Reference-point floors — `3B-only` and `7B-only` as trivial lower/upper bounds in the **appendix only** (not main table).
3. Composed-baseline steelmen — where no single prior system has all our axes, compose one from existing techniques (e.g. `RouteLLM-mf + shortest_queue + cara_tiered`).

---

## 3. Evaluation Structure

### 3.1 Component axes — for supporting studies, NOT for defining Cara

| Axis | Mechanism | Comparable peers in the literature |
|---|---|---|
| **Routing signal** (model selection, quality-driven) | `cara_score` internal; wrapper signals for baselines | RouteLLM-mf, SR semantic classifier, Avengers-Pro, BEST-Route, Hybrid-LLM, Dual-Pool, `null_fixed` |
| **Assignment heuristic** (model / instance selection, load-aware) | score-based joint assignment in Cara; harness variants for supporting studies | `round_robin`, `random`, `shortest_queue`, `num_waiting`, `llumnix_minus`, `block_style` |
| **Filter** (latency-deadline / budget feasibility gate) | CaraCDF (`cara_tiered` in current configs) | QLM (Normal confidence), SLOs-Serve (point AND), PolyServe (cumulative deadline), TimeBill (pointwise budget), `none` |

These are pluggable slots in `block/global_scheduler/cara/{routers,dispatch,filters}/`, selectable via `/v1/config`. **This decomposition exists for ablation and supporting analysis. `Cara-full` in the main paper is the integrated batch scheduler, not merely one tuple row in this matrix.**

### 3.2 Main table — integrated system vs practical wrapper baselines

| # | Row | Represents | Knob swept |
|---|---|---|---|
| 1 | **Cara-full** | integrated online batch scheduler: LPT ordering + bucket feasibility filter + four-term scoring (latency / quality / cost / balance) + score-based joint (model, instance) assignment | `w_quality ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` with `w_cost = 1 − w_quality − w_balance` |
| 2 | **vllm_sr-wrapper** | external peer scheduler (vllm-semantic-router); Envoy + extproc container stack | — single operating point (no threshold knob) |
| 3 | **RouteLLM-wrapper (mf)** | paper-recommended `mf` variant + pretrained HF checkpoint `routellm/mf_gpt4_augmented` + cost threshold sweep; **paired with both RR and shortest_queue dispatch** for fair evaluation | `α ∈ {0.05, 0.11593, 0.2, 0.3, 0.5}` — 0.11593 is RouteLLM's published 50%-strong calibration point |
| 4 | **Avengers-Pro-wrapper** | upstream `SimpleClusterRouter`, retrained K-means + rankings on our Qwen deepeval pool; **paired with both RR and shortest_queue dispatch** | `performance_weight ∈ {0.25, 0.39, 0.53, 0.7, 0.8}` — paper's published α values |
| 5 | **BEST-Route-wrapper** | custom-trained DeBERTa-v3-small multi-head router on Qwen pair-wise preference labels (armoRM scoring) with threshold knob; N=1 (no Best-of-N — see §5.15) | `t ∈ {0.3, 0.5, 0.6, 0.7, 0.8}` — match-probability threshold |

**Notes:**

- Wrapper rows 3-5 each contribute **TWO curves** (RR dispatch, shortest_queue dispatch). `llumnix_minus` dispatch is optional, appendix-only if run.
- **SR is frozen at one config** (v3 with 2 backend_refs per model, Envoy weighted-RR); it has no external threshold exposed.
- Do **not** include `3B-only` or `7B-only` in the main table — those appear in the appendix as reference-point floors (§3.7).
- Do **not** define `Cara-full` as a `router + dispatcher + filter` tuple. The pluggable-pipeline infrastructure is supporting-study scaffolding (§3.4-3.5), not the paper system.
- `best_route-wrapper` (quality-argmax) from revision 1 of this doc is **dropped from the main table** — renamed to `quality_greedy_ablation` and moved to §3.3 as an ingredient-study row. It was never BEST-Route; calling it so was a misattribution in revision 1. BEST-Route (row 5 above) is the real paper (Ding 2025, arXiv 2506.22716) with its own training artifact.

### 3.3 Main ablation table — integrated Cara ingredients

| # | Row | What it isolates |
|---|---|---|
| 1 | `Cara-full` | full integrated system |
| 2 | `Cara-no-LPT` (FIFO) | value of LPT-style batch ordering |
| 3 | `Cara-no-filter` | value of feasibility filtering |
| 4 | `Cara-point-filter` | value of bucket-based filtering vs simpler point filtering — planned, contingent on the unified four-constraint bucket filter landing (§5.11) |
| 5 | `Cara-no-quality-term` | value of quality-aware routing (`w_quality = 0`) |
| 6 | `Cara-no-latency-term` | value of latency-awareness (`w_latency = 0`) |
| 7 | `Cara-no-cost-term` | value of cost-awareness (`w_cost = 0`) |
| 7b | `Cara-no-balance-term` | value of cluster-memory-balance term (`w_balance = 0`) |
| 8a | `Cara-simple-shortestQ` | LPT + shortest_queue across all instances (no joint scoring); isolates the value of score-based assignment |
| 8b | `Cara-simple-llumnix` | LPT + `llumnix_minus` across all instances; same comparator against a richer load signal |
| 9 | `quality_greedy_ablation` | argmax over per-model predicted quality, no cost/latency/balance terms; reference for "routing signal alone without multi-objective scoring" |

This is the main ablation table. It tests the claimed ingredients of the integrated scheduler directly rather than enumerating a combinatorial cross-product.

### 3.4 Supporting study 1 — routing-signal swap inside a fixed scheduler shell

Hold the non-routing scheduler shell fixed (LPT + cara_tiered filter + scoring assignment), swap only the routing / quality signal:

- `cara_score` (internal signal reference)
- `best_route_signal` (internal quality-argmax)
- `vllm_sr` (HTTP adapter)
- `routellm_mf` (pretrained HF checkpoint)
- `avengers_pro` (trained K-means + rankings)
- `null_fixed` (largest-only baseline)

Answers: **how much of the gain comes from the routing signal itself, versus the integrated scheduler around it?**

### 3.5 Supporting study 2 — filter comparison inside a fixed Cara shell

Hold ordering and assignment fixed, compare filters: `none`, `cara_tiered`, `qlm`, `slos_serve`, `polyserve`, `timebill`.

Supporting evidence for the filter ingredient; does not redefine `Cara-full`.

### 3.6 Standalone SR sanity-check table

SR-as-standalone-system (Envoy end-to-end, outside our harness) gets a small separate table with non-SLO metrics only (latency distribution, throughput, response quality). **Goodput is blank with footnote** because SR is SLO-blind. Not a headline row.

### 3.7 Appendix — reference points + overhead decomposition

- `vllm_native_3b` / `vllm_native_7b` — raw vLLM bypassing both schedulers. Pure model latency floor.
- `3B-only` / `7B-only` via our harness — homogeneous-model reference-point floors.
- SR Prometheus metrics + Jaeger spans for SR-internal stage breakdown (§5.1).
- Monitor.py per-node CSVs (see §4.4) for hardware-side overhead attribution.

---

## 4. Metrics, Figures, and Normalization

### 4.1 Axis roles

| Role | Metric(s) | Treatment |
|---|---|---|
| **Optimization axes** (continuous, regression, the scorer targets) | **Cost** ($/request), **Quality** (deepeval mean), **Mean E2E latency** (ms) | Pareto + radar + scalar |
| **Constraint axes** (pass/fail, the filter enforces) | TTFT SLO, TPOT SLO, Budget → aggregated into **Goodput @ SLO** at two tiers (balanced + strict) | Bar chart; constraint filter for viability |
| **Scheduler-specific contribution** | Cluster-memory balance (CV of KV utilization across instances over request index) | Appendix sensitivity (§4.5) |

Rationale (user-driven): goodput is a constraint-compliance rate (pass/fail per request), not something the scorer optimizes. Mean E2E latency is a regression target in the scorer (`w_lat × e2e_norm`). Keeping them on separate roles matches the code.

### 4.2 Cost model — per-token vendor pricing (MaaS framing)

```
cost_per_request[model] = prompt_tokens × input_price[model]
                        + output_tokens × output_price[model]
```

Reference rates (Alibaba Cloud Qwen-2.5 published API pricing, rounded for reproducibility):

| Model | input ($/1M tok) | output ($/1M tok) |
|---|---|---|
| Qwen-2.5-3B | 0.10 | 0.20 |
| Qwen-2.5-7B | 0.20 | 0.40 |

Rates publish at roughly a 2× step per model-size doubling. Values documented in `block/config/cara/scheduler_config_smoketest.json`; re-derive from vendor pages for Apr 26 rerun.

We intentionally **do not use** MoE-CAP's hardware-TCO formula (`C_token = (C_hw + C_energy · kWh) / (T_token · R)`) because that captures the operator-side TCO view, whereas our paper's framing is **MaaS platform / router** where the cost is the user-facing API billing. MoE-CAP's formula is cited as alternative methodology but not used in our main numbers.

### 4.3 Figures

**Figure 1 (main — headline).** 2D Pareto: **Cost vs Quality**.
- Each router = one curve; knob-sweep points + convex hull.
- Marker size ∝ goodput@balanced (larger = more SLO-compliant → more viable).
- Points with goodput < 50% are faded or marked "✗" — honestly signal "this operating point is broken."
- Native units on both axes; no normalization.
- RouterBench's AIQ scalar reported in caption per router.

**Figure 2 (main — partner).** 2D Pareto: **Cost vs Mean E2E Latency**.
- Same conventions as Figure 1.
- Paired with Figure 1 so the reader sees the quality and latency dimensions jointly.

**Figure 3 (main — radar).** CAP-style polar radar, one per router.
- Three axes: Cost ($/request), Quality (deepeval), Mean E2E latency (ms).
- Per-axis native-unit scaling. Axes inverted for lower-is-better (cost, latency) so **outer = better universally**.
- Polygon area is a visual heuristic for overall trade-off.
- Normalization helpers directly adapted from the [MoE-CAP-Dashboard](https://huggingface.co/spaces/auto-cap/MoE-CAP-Dashboard) (Apache 2.0, attribution maintained):

  ```python
  # Higher-is-better axis (quality):
  def normalize(val, vmin, vmax, baseline=20):
      if vmax == vmin: return baseline + 40
      return baseline + (val - vmin) / (vmax - vmin) * (100 - baseline)

  # Lower-is-better axis (cost, latency):
  def normalize_cost(val, max_tick, baseline=20):
      if max_tick == 0: return baseline + 40
      return baseline + (max_tick - min(val, max_tick)) / max_tick * (100 - baseline)
  ```
  Baseline=20 prevents polygon collapse to origin; `vmin, vmax` are computed across the compared systems only.

**Figure 4 (main — constraint viz).** Goodput bar chart.
- Grouped bars per (router, knob-value): goodput@balanced (solid) + goodput@strict (patterned).
- Answers: "which operating points are production-viable?"
- Separate figure from the optimization-axis plots because goodput is a constraint, not an optimization target.

**Figure 5 (main — raw work done).** Throughput bar chart.
- Grouped bars: request_throughput (req/s) + output_throughput (tokens/s).
- Complements goodput (raw vs SLO-filtered work done).

**Figure 6 (§5.X — balance time-series).** Memory-utilization variance over request-index.
- X: request index (or wall time).
- Y: CV (coefficient of variation) of KV-cache utilization across instances.
- One line per router at a fixed operating point.
- Source: `monitor.py` per-node CSV (see §4.4).

**Figure 7 (§5.X — w_balance sensitivity).** Line plot.
- X: `w_balance ∈ {0, 0.1, 0.2, 0.3, 0.5}` (Cara-full only).
- Y: dual axes — balance CV on one, cost/quality/latency on another.
- Shows the trade-off curve for this single scorer term.

**Figure 8 (§5.X — routing overhead).** Stacked bar per router.
- Stack components: `router_ms + filter_ms + dispatch_ms` (pluggable pipeline) OR `estimator_ms + sidecar_ms + scoring_ms + batch_wait_ms` (Cara-full).
- Source: `scheduling_overhead_breakdown` per-request + `monitor.py` process-CPU.

### 4.4 Observability: monitor.py

Pre-existing resource monitor at `block/exp/cara/monitor.py`. Collects per-node every 5 s:
- GPU: utilization, memory, power, temperature (pynvml, ~100 µs / sample).
- vLLM `/metrics`: KV cache util, queue depth, num_running / num_waiting (Prometheus format).
- System: CPU load, memory.
- Optional process-level: scheduler / predictor CPU + RSS (via `--scheduler-pid` / `--predictor-pid`).

**Status: planned for the upcoming knob-sweep re-run on the clean bench, not yet deployed during prior sweeps.** The Apr 15 morning ablation + main-table runs were executed without monitor.py — that was the data gap that motivated this addition. Sweep wrapper (task #61, ~30 min) launches `monitor.py` on every node before each bench and stops on completion; aggregated post-hoc via `aggregate_monitor.py`.

**Critical:** monitor.py polls backend vLLM `/metrics` directly, so KV-cache and queue-depth time-series are captured **independent of who made the routing decision**. This works uniformly for Cara, wrappers, and SR.

### 4.6 Predeclared reporting — arrival-rate sweep + sweep-scope layering

**Primary load axis: request arrival rate λ** (Poisson arrivals).
Fixed workload (clean 500-prompt bench from §5.14); fixed SLO tiers (balanced + strict, §5.6); only λ varies.

**Pilot + anchor-rate rule** (predeclared before data is seen):
- Day 1 of Phase 2 runs a coarse pilot across λ ∈ {4, 8, 16, 24, 32} to find the knee.
- Main sweep grid frozen from pilot; typical expected: λ ∈ {4, 8, 12, 16, 20, 24, 32}.
- Three **anchor rates** selected by rule:
  - **low** = first clearly-underloaded point (all routers ≥ 95% of their peak balanced-goodput).
  - **knee** = highest λ at which Cara-full balanced-goodput stays ≥ 80% of its peak.
  - **stress** = first λ where at least one strong baseline drops below 50% of its peak balanced-goodput.

**Sweep-scope layering** (avoids knob × rate combinatorial explosion):

| Figure / table | Scope | Runs |
|---|---|---|
| **F1/F2/F3 λ-axis curves** (goodput / throughput / latency vs λ) | 5 main-table rows at **single representative knob each**, full λ sweep | ~5 × 7 = **35 runs** shared across F1/F2/F3 |
| **F4 2D Pareto at knee rate**, **F5 2D Pareto at stress rate** (cost-quality + cost-latency) | 5 rows × **full knob sweep** × 2 anchor rates | ~5 × 5 × 2 = **50 runs** (reusing default-knob points from above → net ~40 additional) |
| **F3r CAP radar at knee** | 5 rows × each row's **best Pareto-optimal knob** at knee | reuses F4 data — 0 additional |
| **Ablations** (Cara-full ingredients, rows 2-9) | Cara-full at default knob, **3 anchor rates** (low / knee / stress) | ~10 × 3 = **30 runs** |
| **Supporting studies** (routing-signal swap, filter swap) | Knee rate only, single knob each, appendix | ~12 rows × 1 = **12 runs** |
| **F6 memory-variance time-series** | Cara-full vs Cara-no-balance at knee rate | uses ablation-run data — 0 additional |
| **F7 `w_balance` sensitivity** | Cara-full × `w_balance ∈ {0, 0.1, 0.2, 0.3, 0.5}` at knee rate | ~5 runs (appendix) |
| **F8 routing overhead** | All 5 rows at knee rate default knob | uses F1-F3 data — 0 additional |
| **Table 1** full dump | All runs above | 0 additional |

**Total Phase 2 bench volume**: ~127 runs × 1-2 min each ≈ **2.5-4 h cluster time**.

**Default knob per router** (predeclared, used for λ-axis curves F1-F3, F8):
- Cara-full: `w_quality = 0.5` (middle of 5-value grid)
- RouteLLM-mf: `α = 0.11593` (paper's published 50%-strong-calls calibration)
- Avengers-Pro: `performance_weight = 0.7` (paper's common operating point)
- BEST-Route: `t` picked per training calibration (target ~50% weak-model routing; placeholder 0.6)
- vllm-SR: no knob (single config per §3.2)

Knob sweep per router retains 5 values per §3.2; only the **default representative** is used for λ-axis curves. Knob-sweep data lives in F4/F5 Pareto panels.

### 4.5 Balance analysis — main + appendix scope

**Main paper (§5.X):**
- One comparison: Cara-full (`w_balance=0.2` default) vs Cara-no-balance (`w_balance=0`).
- One figure (F6: memory-variance time-series overlaying the two).
- One claim: "The `w_balance` term reduces cluster-memory-CV by Nx without harming goodput or quality."

**Appendix sensitivity:**
- Full `w_balance ∈ {0, 0.1, 0.2, 0.3, 0.5}` sweep (Figure 7).
- Demonstrates diminishing returns and identifies operating-point sweet-spot.

Balance is a Cara-specific contribution; the main-table Pareto analysis excludes it (because external baselines have no analogue). Heterogeneity effect is more visible on the Apr 26 18-node cluster; Phase 1 result validates the methodology.

---

## 5. Resolved and Open Issues

### 5.1 SR overhead measurement (pending task #49)

Earlier workload-aggregate subtraction claim ("SR overhead ~1.5 ms TTFT") was dropped — that is confounded by batching variance and not per-request. Replacement: fetch SR Jaeger spans per request_id from `http://128.105.146.39:16686/api/traces?...`, extract `total_span_duration − backend_call_span_duration` as the SR-side overhead. Symmetric with Cara's `scheduling_overhead_breakdown`.

### 5.2 Wrapper rows labelled honestly (resolved)

SR / RouteLLM-mf / Avengers-Pro / BEST-Route do not accept per-request SLO inputs natively. Their evaluation on the same harness is fair as long as we (a) label them as **wrapper / signal baselines** not integrated schedulers, (b) pair them with multiple dispatcher options (§3.2), and (c) report goodput as an indication of SLO-compliance rather than a metric they claim to optimize.

### 5.3 Filter ≠ router (resolved)

Filter and router are orthogonal axes:
- Router signal = per-model quality scalar.
- Filter signal = per-model **distribution** over output-length buckets (CDF).

RouteLLM / BEST-Route / Avengers-Pro produce scalar quality signals — not CDFs. Coercing them into filters would shoehorn. CaraCDF's peer on the filter axis is other purpose-built SLO filters (QLM, SLOs-Serve, PolyServe, TimeBill) — which is what §3.5 evaluates.

### 5.4 SR apples-to-apples dispatch (partial)

SR's Envoy does **weighted round-robin** — not load-aware. Cara's dispatcher is. This advantages Cara on the dispatcher axis by design. Paper footnote: "SR's Envoy data-plane dispatch is stateless by design; Cara's is load-aware."

### 5.5 Quality evaluation — RouterBench-style pre-computed lookup (planned methodology, code-pending)

**Methodology (locked):** for every (router, prompt) decision, look up the pre-computed deepeval score for `(prompt, chosen_model)` from the scored HF dataset (`asdwb/cara_model_estimator`). No live judge call; no response-generation variance contamination; every router evaluated on identical quality data. This is how RouterBench does it; isolates routing decisions from generation stochasticity.

**Code status (NOT YET LANDED):** the current `cara_paper/smoke_test_apr_13/scripts/aggregate_e2e_results.py` still emits `predicted_quality_mean` (Cara's own RoBERTa estimate) as a placeholder, with a note that "true deepeval re-scoring is deferred to #54". For revision-2 numbers, **#54 must be re-scoped** as "swap in the pre-computed lookup join" (10-line patch — read scored JSONL into a `dict[prompt → {model: deepeval}]`, replace `predicted_quality` lookup with `scored[prompt][chosen_model]`). No live judge required. **This is a follow-up code task gated on the re-run sweep with the clean bench.**

Live per-response deepeval re-scoring remains available for robustness checks in the appendix but is not the primary metric.

**Action (task #54 re-scoped):** patch `aggregate_e2e_results.py` to perform RouterBench-style lookup against `data/cara/scored/test_scored_filtered.jsonl`. Drop the `predicted_quality_mean` placeholder; replace with `deepeval_quality_mean` from the lookup. ETA ~30 min once we kick off the re-run.

### 5.6 SLO tiers (resolved)

Two main SLO tiers in the paper:

| Tier | TTFT | TPOT | E2EL |
|---|---|---|---|
| balanced | ≤ 1000 ms | ≤ 50 ms | ≤ 5000 ms |
| strict | ≤ 500 ms | ≤ 30 ms | ≤ 3000 ms |

`lenient` (ttft ≤ 5000, tpot ≤ 200, e2el ≤ 10000) retained for sanity / appendix only.

### 5.7 Workload — single bench set per paper (resolved)

Canonical paper bench: **500 clean prompts resampled from `test_scored_filtered.jsonl`** (see §5.14). Covers 7 source datasets uniformly (gsm8k, squad, beaver_tails, code_ultra_feedback, lmsys, reward_bench, mix_instruct). Used by both Phase 1 (preflight, §7.A) and Phase 2 (paper-facing, §7.B). "Hard-skewed subset" deferred to appendix or Phase 2 supplement if space allows.

### 5.8 Arrival-rate sweep — primary Phase 2 load axis (promoted from supplementary)

**Phase 2 primary load axis = request arrival rate λ** (Poisson). Sweep-scope layering, anchor-rate rule, and predeclared figure set in §4.6.

Phase 1 (Apr 15-17 4-node) runs at a **single fixed λ = 10 req/s** for **preflight only** — metric + monitor + script validation on the clean bench. Paper numbers come exclusively from Phase 2 on the 18-node cluster.

### 5.9 Cost calibration — locked (resolved)

Per-token vendor pricing per §4.2. Reference rates cited.

### 5.10 (reserved)

### 5.11 CaraCDF filter scope (open, honest)

`CaraCDFFilter` today checks TTFT/TPOT only; budget + quality feasibility run in the legacy inline path at `cara_serve.py:906-960`. Paper describes "bucket-based probabilistic feasibility filter" as the *target* design; the `cara_tiered` filter exposed via `/v1/config` is a strict subset. `Cara-point-filter` ablation row 4 (§3.3) is therefore marked **planned / contingent** on either (a) finishing the unified four-constraint bucket filter, or (b) restricting the filter ablation to TTFT/TPOT variants only and scoping explicitly.

### 5.12 `w_balance` scorer term (resolved)

Four-term scorer confirmed: `w_latency, w_cost, w_quality, w_balance` at `cara_serve.py:780, :1117`. Ablation row 7b (`w_balance = 0`) added. Appendix sensitivity sweep retained (§4.5). `/v1/config` verified to accept per-term zeros (task #53, complete).

### 5.13 Apr 26 Phase 2 — cluster size, not just a rerun

Moved to §7.B below.

### 5.14 (NEW) Bench data integrity — resampled clean 500 (resolved)

**Problem.** `data/cara/best-route-v3-test-500.jsonl` overlapped 289/500 prompts with the predictor's training split (`train_scored_filtered.jsonl`), 61/500 with its clean held-out test split, and 150/500 in neither (no deepeval labels at all). Using it for evaluation mixes in-distribution prompts with out-of-distribution prompts and ungraded prompts — indefensible for a paper.

**Verification.** Predictor training files (`training_data_with_ref/train_fixed.jsonl`) ↔ scored training file (`train_scored_filtered.jsonl`, 14,919 prompts with 3B+7B deepeval) ↔ scored held-out test file (`test_scored_filtered.jsonl`, 3,634 prompts with 3B+7B deepeval). `test ∩ train = 0` — the predictor's splits are clean; only the bench file was bad.

**Fix (task #56).** Regenerate `data/cara/best-route-v3-test-500-clean.jsonl` by stratified-sampling 500 prompts from `test_scored_filtered.jsonl` (3,634 clean prompts). Zero overlap with predictor training, 100% deepeval coverage for 3B and 7B. Used as the canonical bench set for all Phase 1 and Phase 2 runs.

**All Apr 15 morning sweep results (ablation + main_cara + main_pipeline) used the contaminated file** and are superseded — the sweep is re-run on the clean bench before paper numbers are quoted.

### 5.15 (NEW) BEST-Route wrapper — training plan (in progress)

BEST-Route (Ding 2025, arXiv 2506.22716) requires training on target-pool preference data; no pretrained weights for Qwen exist on GitHub (neither `microsoft/best-route-llm` nor `BEST-Route2025/BEST-Route` ships a checkpoint).

**Adapted implementation plan (excluding Best-of-N):**

1. Deploy armoRM (`RLHFlow/ArmoRM-Llama3-8B-v0.1`, 8B, 16 GB fp16) on one spare A30 via vLLM.
2. Generate one response per (prompt, model) for ~3000 train prompts × Qwen-3B/7B via node0's vLLM = 6000 responses (≈20 min).
3. armoRM-score all responses (≈1-2 h).
4. Build pair-wise training labels: `y = 1 if armoRM(3B-resp) ≥ armoRM(7B-resp) else 0`.
5. Fine-tune DeBERTa-v3-small with BCE loss, LR=1e-5, batch=32, 5 epochs (≈30 min on A30).
6. Integrate checkpoint into `block/global_scheduler/cara/routers/best_route.py` (replace current argmax-quality implementation — that was a misattribution in revision 1).

Total ≈ 3-4 h wall-clock. Fits before Apr 17 cluster release.

Threshold knob calibration: sweep `t ∈ {0.3, 0.5, 0.6, 0.7, 0.8}` on a held-out val fold; use the point closest to 50% weak-model routing as the published default, per RouteLLM's calibration convention.

Because we skip Best-of-N, BEST-Route on our 2-model pool degenerates to a binary-threshold router structurally similar to RouteLLM — but with armoRM-derived match-probability labels instead of human-preference labels. Paper will state this explicitly.

### 5.18 (NEW) Train/val hygiene for retrained baselines (rule)

Every retrained baseline artifact (Avengers-Pro K-means + rankings, BEST-Route DeBERTa router, any RouteLLM re-calibration if we choose to do one) uses:

- **Training split**: prompts from `data/cara/train_scored_filtered.jsonl` (predictor-train; 14,919 prompts with deepeval).
- **Validation split**: held-out slice of `data/cara/scored/test_scored_filtered.jsonl` (3,634 clean prompts), **EXCLUDING the 500 prompts** sampled for `best-route-v3-test-500-clean.jsonl`.

The bench file itself (`best-route-v3-test-500-clean.jsonl`) is **never** touched during training / hyper-param selection / threshold calibration — it is strictly for inference-time evaluation. Enforced mechanically via the resample script's deterministic seed (`seed=42`): exclude-from-val filter applies the same prompts.

This rule resolves codex's Apr 15 PM concern about retrained baselines potentially leaking into the evaluation bench.

### 5.16 (NEW) RouteLLM switch from BERT to MF (resolved)

Revision 1 used RouteLLM's `bert` variant (re-trained locally with val_acc=0.389, near-random). RouteLLM paper (Ong ICLR'25) recommends `mf` (matrix factorization); pretrained checkpoint exists at `routellm/mf_gpt4_augmented` on HF. Domain-shift caveat (trained on GPT-4 vs Mixtral preference data, applied to Qwen-3B vs Qwen-7B) disclosed in one-line paper footnote. Published calibration threshold for 50%-strong routing: `α = 0.11593` (from `routellm.calibrate_threshold` against the Chatbot Arena 55k battles dataset).

### 5.17 (NEW) Related-work positioning

| Paper / Tool | Position vs Cara | Paper section |
|---|---|---|
| RouterBench (Hu 2024) | Benchmark methodology we inherit (AIQ, convex hull) | §5 |
| RouterEval (Huang 2025) | Scaling-law analysis complementary to our Pareto | §5 |
| MoE-CAP (Jiang 2025) | CAP radar methodology we adapt | §5 |
| LLMRouter lib (ulab-uiuc) | Extensibility precedent; 16-router plugin API | §5 |
| BentoML Performance Explorer | Interactive Pareto dashboard; our static analog | §5 |
| BOute (Jiang 2026) | Offline joint-optimization (routing + deployment); complementary to our online scheduler | §2, §5 |
| RouteLLM (Ong ICLR'25) | External-router baseline | §5 / §6 |
| Avengers-Pro (Zhang DAI'25) | External-router baseline | §5 / §6 |
| BEST-Route (Ding 2025) | External-router baseline | §5 / §6 |
| Hybrid-LLM (Ding ICLR'24) | Earlier cascade from same authors as BEST-Route; referenced in related work, not benchmarked | §6 |
| vllm-SR | External peer scheduler | §5 / §6 |

LLMRouter adapter is deferred to post-Apr 17 — 2 h build, would add 2-3 more router baselines but each still requires Qwen-specific training; diminishing returns for Phase 1.

---

## 6. Execution — Task Status

**Done (verified this session):**
- Pluggable router / dispatch / filter architecture (tasks #23, #24, #25, #26, #27, #28, #30, #31, #32, #33, #34, #35, #36, #37).
- 9 routers, 6 dispatchers, 7 filters implemented.
- Cara-full LPT toggle (#51), assignment_strategy toggle (#52), per-term zero-weight verify (#53).
- Main + ablation sweep script (#55).
- vLLM-SR stack deployed on node2 via `/mydata` NVMe; 2 backend_refs per model.
- monitor.py pre-existing at `block/exp/cara/monitor.py`.

**All blockers DONE (verified Apr 16 03:30 UTC):**
- **#56** resample clean 500-prompt bench from `test_scored_filtered` — ✅ landed 2026-04-15 19:30 UTC.
- **#54** deepeval post-hoc lookup (RouterBench-style, §5.5) — ✅ landed, cov=1.0 on clean bench.
- **#60** `sweep_with_monitor.sh` — ✅ launched monitor.py on all cluster nodes, CSVs collected.
- **#61** p95 → p99 key-mismatch fix in sweep scripts — ✅ landed.
- **#62** BEST-Route training pipeline — ✅ complete. DeBERTa-v3-small, 5 epochs, **val_acc=0.58**, threshold=0.591. Checkpoint at `models/cara/best_route_wrapper_qwen/`.
- **#63** sweep scripts updated for 5-row × knob × λ matrix per §4.6 — ✅ landed.
- **#64** RouteLLM BERT→MF swap (pretrained + α=0.11593) — ✅ landed. **Note: domain shift detected** — pretrained `mf_gpt4_augmented` routes all-to-3B on Qwen pair. Needs footnote + possible threshold recalibration on Qwen data.
- Avengers-Pro α-sweep — ✅ 5 artifacts produced (pw=0.25/0.39/0.53/0.7/0.8).
- **#49** Jaeger per-request SR overhead (§5.1) — appendix only, still pending.

**Re-run after the above:**
- Main table (Cara-full × 5 knobs + 4 wrappers × 2 dispatchers × 5 knobs ≈ 45 bench runs × 1-2 min = ~60-90 min cluster time).
- Ablation table (10 rows × ~1 min = 10 min).
- Supporting studies 1 and 2 (~30 min combined).

**Superseded / retired:**
- ~~9-row pluggable sweep × 3 SLO × 2 workload × 5 QPS = 270 runs~~ — pluggable pipeline is supporting-study scaffold, not main table.
- ~~Workload-aggregate SR overhead subtraction~~ — replaced by Jaeger per-request (task #49).

---

## 7. Phase 1 and Phase 2

### 7.A Phase 1 — 4× A30 preflight (Apr 15-17)

**Purpose**: metric + monitor + script validation. **NOT paper numbers.** Outputs are debugging artifacts to catch script bugs (field names, monitor CSV format, aggregator behavior) before committing 18-node cluster time.

- Cluster: 4× A30 d7525 (2× Qwen-3B + 2× Qwen-7B, homogeneous GPU).
- Bench: clean 500-prompt resample (task #56, landed).
- Load: single fixed λ = 10 req/s.
- Subset of main-table rows (not the full 5-router × 5-knob matrix): at least Cara-full at default knob + 1 wrapper + SR, so every code path gets exercised.
- Ablation toggles exercised at least once each (verify /v1/config acceptance, verify scoring_overhead fields populate correctly).
- Monitor.py deployed via `scripts/sweep_with_monitor.sh` (task #61) on all 4 nodes.
- Aggregator emits `deepeval_quality_mean` via lookup (task #54 re-scoped patch).
- p95-export key-mismatch fix landed (emit_row uses p99 keys that the bench actually produces).

Pass criteria:
- [ ] Every detail JSON has populated `overhead` breakdown
- [ ] Every summary row has non-zero latency quantiles (no p95=0 artifacts)
- [ ] `deepeval_quality_mean` populated (not `predicted_quality_mean`)
- [ ] Per-node monitor CSV lands + aggregate_monitor.py parses
- [ ] Row naming in emitted JSON matches design §3.2 (Cara-full, vllm_sr-wrapper, RouteLLM-wrapper-mf, Avengers-Pro-wrapper, BEST-Route-wrapper; **no** `best_route-wrapper` rows)

### 7.B Phase 2 — 18-node heterogeneous (Apr 26 - May 6, primary paper numbers)

- Cluster: 8× A30 + 3× V100 + 5× P100 + 2× A100 (genuinely heterogeneous).
- Workload: clean 500-prompt bench (same as Phase 1).
- Primary load axis: **arrival-rate sweep λ**, pilot-justified grid (§4.6).

**Day-by-day plan** (within the Apr 26 - May 6 window):

- **Day 1 (~24 h)**: cluster bring-up, 4-node-preflight parity check, coarse λ pilot (one router at λ ∈ {4, 8, 16, 24, 32}) to identify knee/stress points. Lock the final grid.
- **Day 2-4 (~60 h)**: main-table runs — 5 rows × 7 λ at default knob (F1-F3 λ-axis data) + 5 rows × 5 knobs at 2 anchor rates (F4-F5 Pareto data) = ~75 runs with knob reuse.
- **Day 5 (~16 h)**: ablations — 10 rows × 3 anchor rates = ~30 runs. w_balance sensitivity (~5 runs).
- **Day 6 (~12 h)**: supporting studies — routing-signal swap + filter swap at knee = ~12 runs.
- **Day 7-8 (~16 h)**: aggregation, figure generation, robustness re-runs if needed.
- **Day 9-10 (~24 h)**: buffer for surprises, slow-runs, cluster-reservation overlap.

**Primary additions over Phase 1**:
- Balance CV time-series (§4.5) genuinely meaningful (heterogeneous GPUs have different KV capacities).
- Full λ sweep enables knee/stress Pareto and λ-axis curves (§4.6).
- CAP radar at per-GPU-type cost: A100 / V100 / P100 have >10× cost ratio — widens the cost axis meaningfully.
- Methodology identical to Phase 1; Apr 15 design is the frozen spec, Apr 26 is execution.

### 7.C Go / no-go checklist — what must be true before Apr 26 cluster window opens

**Already landed** (done earlier in Apr 15 session — verified via tracker; NOT blockers):
- Cara-full LPT toggle (#51), assignment_strategy toggle (#52), per-term zero-weight /v1/config acceptance (#53).
- Clean bench resample (#56): `data/cara/best-route-v3-test-500-clean.jsonl` (500 prompts, zero predictor-train leak, 100% 3B+7B deepeval coverage).

**Hard blockers** — must be checked before Apr 26 cluster window opens:

- [ ] `aggregate_e2e_results.py` emits `deepeval_quality_mean` via pre-computed lookup against `test_scored_filtered.jsonl` (re-scoped #54). **Smoke test: 1 row on clean bench → field present, value in [0,1].**
- [ ] `scripts/sweep_with_monitor.sh` (#60) launches monitor.py on all cluster nodes before each bench, stops on completion; post-hoc aggregation via `aggregate_monitor.py` produces `kv_util_cv` time-series CSV.
- [ ] p95 / p99 key-name fix (#61): emit_row no longer outputs `ttft_p95_ms: 0`; uses the p99 fields the bench actually emits.
- [ ] `run_main_table.sh` + `run_cara_full_ablation.sh` updated for 5-row matrix + arrival-rate outer loop + 5-knob inner loop (#63, §4.6 layering). Supersedes the earlier #55 sweep script which used the legacy 4-row matrix + contaminated bench.
- [ ] RouteLLM-mf swap landed: pretrained `routellm/mf_gpt4_augmented` + threshold α=0.11593 + domain-shift footnote + 1-prompt smoke returns a valid model name.
- [ ] Avengers-Pro α-sweep: 5 trained artifacts (one per α value) or a single artifact with runtime α switch — validated via /v1/config.
- [ ] Train/val hygiene (§5.18) enforced: no retrained baseline has seen any of the 500 clean-bench prompts during training or threshold calibration.
- [ ] BEST-Route: either training completes + DeBERTa checkpoint integrated into `routers/best_route.py` (full pipeline §5.15), **OR** soft deadline hit (Apr 24 morning, 48 h buffer) → cite-only fallback in related-work paragraph + row 5 removed from main table with explicit note. No delay of the Apr 26 main sweep.
- [ ] 4-node preflight (§7.A) pass criteria all green.

Soft blockers (nice to have, not blockers):

- [ ] Cara-point-filter (ablation row 4) — planned-contingent per §5.11. If not landed by Apr 25, move row 4 to supplement instead of main ablation.
- [ ] LLMRouter library adapter (§5.17) — deferred per decision #10.
- [ ] Jaeger per-request SR overhead (#49) — appendix only; main paper uses `scheduling_overhead_breakdown` for Cara-full and process-level CPU/RSS from monitor.py for SR.

---

## 8. What Codex is Asked to Review

See the companion `CODEX_REVIEW_PACKAGE.md` for the explicit ask list + changelog-since-revision-1.

In short:

1. Is the citation set (RouterBench / RouterEval / MoE-CAP / LLMRouter / BentoML / Hybrid-LLM / BEST-Route / RouteLLM / Avengers-Pro / BOute) sufficient for related work?
2. Is the fair-comparison methodology (knob sweep + Pareto on 3 continuous axes + goodput as constraint + radar + hypervolume optional) defensible against reviewers concerned with cherry-picking or strawman baselines?
3. Is the data-integrity narrative (bench resample from `test_scored_filtered`, predictor splits clean) clearly explained and sufficient?
4. Is the BEST-Route stripped-down plan (no Best-of-N, binary pair-wise router on Qwen preference labels, DeBERTa-v3-small) an acceptable approximation of the paper's router OR should we cite BEST-Route only in related work?
5. Is the wrapper-dispatcher pairing (RR + shortest_queue + optional `llumnix_minus` per wrapper) sufficient steelmanning?
6. Is the balance-scope split (minimal main + sensitivity appendix) reasonable, or should balance be fully elevated to the main story?
