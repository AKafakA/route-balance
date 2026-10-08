# Paper Updates TODO — Updated April 15, 2026 PM (revision 2)

**⚠ Revision 2 supersedes the Apr 15 afternoon revision for baselines, figures, data, and methodology.**
**Source of truth**: `cara_paper/codex/apr15_baseline_comparison_design.md` (revision 2) + `cara_paper/codex/CODEX_REVIEW_PACKAGE.md`.

**Key revision-2 deltas since the Apr 15 AM afternoon revision**:
- **Bench resampled** to `best-route-v3-test-500-clean.jsonl` from the HF clean held-out split (the legacy `best-route-v3-test-500.jsonl` overlapped 289/500 prompts with predictor-train).
- **Baselines finalized**: Cara-full + vllm-SR + RouteLLM-**mf** (pretrained, threshold 0.11593) + Avengers-Pro (α sweep) + BEST-Route (custom-trained Qwen+armoRM DeBERTa-v3-small). Wrappers paired with RR + shortest-queue dispatch.
- **`best_route` argmax-quality** renamed to `quality_greedy_ablation` (was a misattribution; real BEST-Route is Ding 2025 arXiv 2506.22716, now trained properly — see design §5.15).
- **Quality metric**: pre-computed deepeval lookup (RouterBench-style), not live judge calls.
- **Figures**: 2D Pareto (cost-quality, cost-latency) + CAP radar (MoE-CAP-Dashboard Apache-2.0 normalize helpers) + goodput bar + throughput bar + memory-variance time-series + `w_balance` sensitivity + overhead stack + Table 1.
- **Cost model**: per-token vendor pricing (MaaS) — not MoE-CAP hardware TCO.
- **Methodology cited**: RouterBench (Hu 2024) for knob-sweep AIQ Pareto; MoE-CAP (Jiang 2025) for radar; RouterEval (Huang 2025) for scaling context; LLMRouter library for extensibility; BentoML Explorer for interactive analog.

**April 15 morning additions** (retained): §6 Related Work gained a subsection on BOute + Dual-Pool + Avengers-Pro + OmniRouter + DRL-routing with verified quotes.

**April 15 afternoon correction** (retained): Cara-full is an **integrated online batch scheduler** — LPT ordering + bucket-based probabilistic feasibility filtering + multi-objective per-request scoring over quality / latency / cost / **balance** + scoring-based joint `(model, instance)` assignment. Invoked via `--scheduling cara`. The pluggable Router × Dispatcher × Filter infrastructure is **engineering scaffolding** for wrapper baselines and supporting studies — not the paper's Cara.

## Already Done (March 26-31 sessions)
1. Algorithm 1: LPT sort key = `max_m E[len_m(r)]` with max-across-models loop
2. Section 4: batch benefits (normalization, amortization, thundering herd, adaptive sizing)
3. Scoring function: 4-term (latency + cost - quality + balance) with per-batch normalization
4. Constraint handling: STRICT/RELAXED/TIERED modes with progressive relaxation

## Paper Contributions (agreed March 31)

**C1: Quality-Aware Scheduling for Heterogeneous LLM Serving** — Main problem. First system addressing full heterogeneity stack (model + GPU + backend). Key insight: existing systems decouple model routing and load balancing — RouteBalance fuses both into a single decision plane.

**C2: Global Batch Orchestration** — System design enabling C1. LPT greedy assignment, two-phase normalization, prediction amortization (4.9×), thundering herd avoidance, adaptive batch sizing, filter-then-rank.

**C3: Fused Multi-Head Predictor** — Single RoBERTa encoder, per-model per-target heads. Wins all 5 targets. Bucket classifier enables C4.

**C4: CDF-Based Probabilistic SLO Enforcement + Tiered Relaxation** — Distribution-free filtering (no Normal assumption). Every request served via progressive relaxation. Ablation vs SLOs-Serve/PolyServe/QLM.

**C5: Multi-Signal Quality Estimation** — 4 quality signals compared. Safety-aware scoring. Reference-grounded (Prometheus) judging.

**C6: Datasets + Evaluation** — 18.6K quality + 405K latency records. Comprehensive ablation.

### Critical Framing Issue: Cross-Model Quality Ranking
- **Within-model ρ** (0.4-0.7): "for model X, rank prompts by quality" — this is what we report but **NOT what routing needs**
- **Cross-model accuracy** per prompt: "for this prompt, which model is best?" — KNN smoke test showed 26% (near random for 4 models), rank correlation ≈ 0
- This is **not our problem alone** — ALL quality-aware routers (RouteLLM, best_route) face this. The subarea doesn't report it honestly.
- **Our defense**: RouteBalance is a **scheduling** paper, not a routing paper. Quality is ONE of 4 objectives. Even if quality prediction is weak, the system wins on latency/cost/balance. Quality is additive benefit, not the main claim.
- **Paper framing**: Don't overclaim quality routing. Report cross-model metrics honestly. Show value comes from the **combination** of all 4 objectives. Position quality as "imperfect but beneficial at the margin."
- **TODO**: Run cross-model accuracy for RoBERTa encoder on full test set (may be better than KNN). Compute cost-quality Pareto to show even coarse routing saves cost.

---

## Needs Updating

### §3 Scheduling Framework

#### 3.1 Prediction Architecture
- **Add**: Fused multi-head encoder design — single RoBERTa encoder, per-model per-target heads
- **Add**: Architecture comparison results (RoBERTa > ModernBERT > KNN >> LoRA on all 5 targets)
- **Add**: Quality signal hierarchy: reference_score (ρ=0.592) > similarity (0.419) > judge (0.399)
- **Update**: Estimator uses RoBERTa (not ModernBERT) as primary encoder — selected via 5-target ablation

#### 3.2 Budget Filtering
- **Already in paper**: P(tokens ≤ B) ≥ θ via bucket classifier CDF
- **Add**: Filtering results table (compliance/false_accept/false_reject at various θ)
- **Add**: Same CDF framework extends to quality: P(quality ≥ min) ≥ θ using judge classifier

#### 3.3 SLO Enforcement (NEW — major addition)
- **Add**: Comparison of SLO filtering strategies:
  - SLOs-Serve: binary admit/reject from point prediction
  - PolyServe: cumulative deadline (TTFT + i×TPOT)
  - QLM: Normal confidence bounds (CLT assumption)
  - Ours: CDF-based filtering + tiered relaxation
- **Add**: Our tiered relaxation as key differentiator — every request served, constraints relaxed in order
- **Add**: Ablation results showing CDF filtering > point threshold > confidence bounds

### §4 Quality Signals (NEW section or subsection)

#### 4.1 Quality Signal Design
- **Add**: 4 quality signals comparison:
  - `similarity_score`: embedding cosine to 72B (ceiling effect — 72B always 1.0)
  - `llm_judge_scores.qwen2.5-7b-it_blind`: blind LLM judge (noisy, ρ=0.399)
  - `reference_score`: dataset-specific ground truth (GSM8K exact-match, SQuAD F1, etc.)
  - `llm_judge_scores.prometheus-7b-v2_reference`: reference-grounded LLM judge (pending)
- **Add**: Reference-grounded judging adds nuance beyond embedding similarity (e.g., catches quality gaps that cosine misses)

#### 4.2 Safety-Aware Scoring
- **Already partial**: protectai for harmful prompts
- **Add**: How safety signals are inverted — model that refuses harmful request scores HIGH

### §5 Evaluation

#### 5.1 Predictor Comparison — FINAL NUMBERS — verified April 11
- **Table**: Architecture comparison across 6 targets (3,634 prompts × 4 Qwen models):
  | Target | Best Model | MAE | BestAcc | Top2Acc | RankCorr |
  |--------|-----------|-----|---------|---------|----------|
  | length | RoBERTa fused | 102.4 | 42.8% | 68.3% | 0.288 |
  | length_bucket | RoBERTa fused bucket | 102.3 | 44.3% | 70.0% | 0.292 |
  | similarity | KNN | 0.054 | 100.0% | 100.0% | 0.658 (trivial — 72B always best) |
  | judge | LoRA-encoder ModernBERT | 0.243 | 62.1% | 87.1% | 0.148 |
  | reference_score | RoBERTa fused | 0.158 | 44.3% | 71.2% | 0.203 |
  | deepeval | RoBERTa fused | 0.226 | 55.7% | 87.9% | 0.180 |
- Multi-target RoBERTa B: bucket MAE=105.1, deepeval MAE=0.233 (slight quality drop for 2x overhead reduction)
- **Recommended quality signal: deepeval** (RankCorr=0.180, BestAcc=55.7%, Top2Acc=87.9%)
- BestAcc/Top2Acc now correctly handle ties
- **Add Table**: Per-model breakdown from quality_*.json
- **Add**: Bucket classification metrics from quality_length_bucket.json

#### 5.2 SLO Filtering Ablation — FINAL NUMBERS — verified April 11
- At E2E SLO=5s: SLOs-Serve F1=0.976-0.998, CaraCDF F1=0.837-0.997, QLM F1=0.583-0.997
- SLOs-Serve: highest F1, but highest FP (accepts violators)
- QLM: most conservative, low recall on P100 at tight SLOs (F1=0.583 at 5s)
- CaraCDF: balanced — high precision + good recall (F1=0.837 at 5s P100)
- **Add Figure**: SLO attainment vs QPS (needs CloudLab online data)
- **Key finding**: All filters > 0.89 F1 at moderate SLOs. Differentiation at tight SLOs only.

#### 5.3 Latency Predictor Ablation — FINAL NUMBERS — verified April 11
- **Table** (45K test records × 5 instance types):
  | Model | E2E MAE | E2E MAPE | rho |
  |-------|---------|----------|-----|
  | XGBoost (oracle output) | 0.289s | 3.7% | 0.999 |
  | Roofline | 0.531s | 10.7% | 0.987 |
  | XGBoost TTFT | 0.015s | 16.2% | 0.946 |
  | XGBoost TPOT | 0.002s | 42.0% | 0.978 |
- LSTM v2 ≈ blind baseline (per-request queue features don't help in low-contention data)
- **Position**: LSTM needs high-contention data from CloudLab overloaded QPS levels
- Output length is dominant factor — E2E blind (1024) = 245-379% MAPE vs actual = 1-6%

#### 5.4 End-to-End (from CloudLab — PENDING)
- Best baseline vs full Cara: goodput, SLO attainment, quality, cost

### §6 Related Work

#### SLO-Oriented Serving (NEW paragraph)
- **Add**: Dedicated comparison of QLM, SLOs-Serve, PolyServe, PARD, Mooncake
- **Add**: Gap analysis — none use distribution-free prediction intervals
- **Add**: Our contribution: CDF-based filtering is distribution-free and composable

#### Cache-Aware LLM Scheduling (NEW paragraph)
A growing body of work addresses KV cache-aware request routing:
- **SGLang RadixAttention** (Zheng et al., SOSP 2024) — prefix tree for single-instance cache-aware scheduling
- **Preble** (Srivatsa et al., ICLR 2025) — distributed prefix-aware scheduling with exploit/explore E2 algorithm, 1.5-14.5x latency improvement
- **DualMap** (arXiv 2026) — power-of-two-choices with dual hash for cache affinity + load balancing, SLO-aware
- **DLPM** (Cao et al., arXiv 2025) — first locality-aware fair scheduling, balances prefix locality with client fairness
- **BatchLLM** (arXiv 2024) — global prefix tree construction for offline batch inference
- **GORGO** (arXiv 2026) — cross-region cache-aware routing without centralized coordination
- **LMetric** (Zhang et al., arXiv 2026) — simple `P_tokens × batch_size` heuristic, zero hyperparameters
- **Online Scheduling w/ KV Cache** (Jaillet et al., arXiv 2025) — theoretical treatment, proves NP-hardness
- **Key gap**: All focus on single-model homogeneous clusters and latency-only optimization. None considers quality-aware routing or heterogeneous multi-model scheduling.
- **Position Cara**: Our learned latency predictor implicitly captures prefix caching effects via `kv_cache_utilization` and `pending_prefill_tokens` features — vLLM's `enable_prefix_caching=True` is active on all instances. Complementary to explicit prefix-aware approaches.

#### Future Work — KV Cache Awareness
- **Add**: Explicit prefix cache features (`prefix_cache_hit_rate`, `P_token` / prefix match length) as XGBoost input would combine the benefits of LMetric/Preble-style cache awareness with our learned multi-objective approach. Natural extension — just additional features in the latency predictor, no architectural changes needed.

#### Quality-Aware Routing & Semantic Routing (NEW — from April 2 study)
- **Add**: vLLM Semantic Router comparison (arXiv 2603.04444, 2603.21354)
  - SR routes to MODELS (which model?), Cara fuses model selection + load balancing (which instance?)
  - SR: N LoRA adapters × N separate forward passes vs Cara: 1 fused multi-head × 1 forward pass
  - SR: no batch scheduling, no SLO filtering, no budget compliance, no load awareness
  - SR: Rust+Candle for sub-ms heuristic routing; Cara: Python+ONNX for ~15ms CPU inference
  - Position: complementary layers — SR picks model family, Cara picks (model, instance) pair jointly
- **Add**: RouterArena (arXiv 2510.00202) as evaluation methodology reference
  - StandardizedLLM router eval: quality, cost, latency overhead, robustness
  - CPU overhead is standard for evaluation (not dedicated GPU)
- **Add**: Prometheus reference-grounded judging (cite prometheus-eval)
- **Add**: Distinction from LLM routing papers (RouteLLM, best_route) — we combine quality with latency/cost
- **Add**: DeepEval G-Eval for reference-grounded quality scoring (if results are useful)

#### Heterogeneous LLM Serving & Deployment Optimization (NEW — added April 15, 2026)

Concurrent work on heterogeneous LLM serving operates at different layers of the stack. We position Cara explicitly against the most recent and directly comparable papers, verified from the papers themselves:

**BOute** (Jiang, Fu, Yoneki — arXiv 2602.10729, Feb 2026; MLSys 2026) — "Cost-Efficient LLM Serving with Heterogeneous LLMs and GPUs via Multi-Objective Bayesian Optimization".

Positioning (verified with direct quotes from the paper after full read on April 15, 2026):

- **Layer**: BOute is an **offline deployment-plan optimizer**, not a runtime scheduler. From §4.3 (problem formulation): `min_{τ, A, P} (L(τ, A, P), −Q(τ))` subject to resource + budget constraints, where τ = routing thresholds, A = GPU allocation matrix, P = parallelism strategy (TP × PP). The **decision variables are a fixed (τ, A, P) tuple**, not per-request routing decisions.

- **Runtime router (§4.1)**: BOute inherits the threshold-based router from RouteLLM [Ong et al., 2024] / HybridLLM [Ding et al., 2024]. Direct quote: *"Our router. We adopt a threshold-based model routing mechanism. Specifically, for each incoming query, the router computes a routing score that estimates the query's difficulty or the likelihood that a smaller model can produce a satisfactory response. This score is then compared against configurable thresholds τ."* BOute's contribution is finding τ\*, not the inference-time routing mechanism itself.

- **Two phases (§5)**: (§5.1) Offline Preparation — builds a performance database mapping `(deployment, system load) → p95 latency` using the ETH-EASL Scratchpad simulator. (§5.2) "Online" Optimization — the **BO search loop** (qNEHVI acquisition + GP surrogates) that picks (τ\*, A\*, P\*) from the database. "Online" here refers to the BO iteration phase, not request-time adaptation.

- **Reported scheduling time (§7.3, Table 2)**: 12–30+ minutes for 7–24 GPUs. This is the BO optimization time, confirming the "scheduling" in their paper is **offline plan search**, not per-request dispatch.

- **Simulator role**: The Scratchpad simulator is BOute's **BO surrogate** — it evaluates candidate plans without live GPU runs to keep BO tractable. They validate selected plans on live GPUs (H100 / RTX PRO 6000 / RTX 5090 / RTX 4090) afterward.

- **Workload (§7)**: Fixed traces — GSM8K + MTBench with golden-MMLU annotations. No dynamic workload / live arrivals assumption.

**Clean differentiators Cara retains:**

1. **Stack layer**: BOute outputs a deployment plan; Cara dispatches requests within a pre-deployed cluster. The two systems are complementary — BOute's output is a natural input to Cara.
2. **Fusion of model selection + load balancing into one learned score** (THIS IS THE CONTRIBUTION, not the pluggable interface): BOute's runtime path is a two-layer pipeline (threshold router → pooled serving → per-model instance fan-out); model selection and load decisions are **decoupled**. Cara's novelty is fusing these decisions into **one learned multi-objective score** that jointly optimizes quality, predicted latency, cost, and cluster memory balance (C1 / C2 / C5 below). This is what makes the Cara full-system row in the evaluation different from any (router × dispatcher) baseline tuple — including ones built with the same RouteLLM thresholds BOute uses.
3. **Load / queue-state awareness integrated into model selection**: Cara's score consumes `num_running`, `num_waiting`, `kv_cache_utilization`, `kv_free_blocks`, and predicted TTFT/TPOT **at the moment of model selection**. BOute's model selection is a fixed threshold comparison independent of current cluster state — all load awareness happens offline during (A, P) plan optimization.
4. **Probabilistic SLO filter**: CaraCDF uses the bucket classifier's softmax as empirical CDF for distribution-free budget + deadline filtering. BOute's threshold mechanism is deterministic; it does not model prediction uncertainty.
5. **Workload regime**: Cara targets dynamic MaaS traffic where arrival distribution is not known at deploy-time. BOute targets fixed workload traces.

**What is NOT a Cara contribution (to avoid over-claiming):**
- Pluggable router interface / two-layer scheduler structure — both are standard in prior serving work. Cara exposes them as an **engineering affordance for baseline comparison**, not as novelty. The baselines matrix (§5.Y) is instrumentation; the paper's claimed contributions are the fused scoring + batch orchestration + CDF filter.

**Draft Related Work paragraph (save-ready; grounded in quoted facts):**

> "BOute [Jiang et al., MLSys 2026] solves a different problem at a different layer. It performs **offline** multi-objective Bayesian optimization over (routing thresholds τ, GPU allocation A, parallelism strategy P) to produce a cost-efficient deployment plan for a known workload trace; its inference-time routing is the standard two-layer pipeline (threshold router + per-model instance fan-out), inheriting the threshold mechanism from RouteLLM [Ong et al., ICML 2024]. BOute's simulator serves as the BO surrogate, mapping (deployment, load) pairs to workload-level p95 latency — structurally analogous to Cara's per-request latency predictor but applied inside the offline optimization loop rather than inside the request path. Cara operates one layer below: given a **pre-deployed** heterogeneous cluster, it handles **dynamic MaaS traffic online** by fusing model selection and instance-level load balancing into a **single learned multi-objective score** over quality, predicted latency, cost, and cluster memory balance — making one joint decision per request rather than two sequential ones. A probabilistic CDF filter enforces SLO and budget constraints on top of this decision plane. The two systems are complementary — BOute's output is a natural input to Cara — and we use RouteLLM (the runtime routing mechanism BOute inherits) together with a battery of seven other routers as L1 baselines in our evaluation, all instantiated within the standard two-layer pipeline so the fused-score contribution is measured apples-to-apples against decoupled alternatives."

**Dual-Pool** (arXiv 2604.08075, Apr 2026) — token-budget EMA bytes-per-token router for homogeneous fleets. Partitions instance capacity into short/long pools based on EMA of output-bytes-per-token per prompt class. Adapted to heterogeneous Cara clusters by mapping pool-ID to model-ID (short → smallest model, long → largest). Positioned in the motivation/intro as evidence that **budget-aware routing matters at scale** (paper reports 80–95% of real traffic is "short" prompts and that naïve uniform allocation wastes 4–8× capacity on long prompts in a $15.4M deployment study). Used as an **optional L1 baseline** in our evaluation; reimplementation flagged as paper-faithful at the router level but without the joint adaptation loop.

**Avengers-Pro** (arXiv 2508.12631, DAI 2025 Best Paper) — K-means cluster-based router with per-cluster performance-efficiency rankings. Requires re-training on the target model pool (originally trained on GPT-5 / Gemini / Claude); our wrapper imports `SimpleClusterRouter` from their open-source repo and retrains clusters + rankings on Qwen-2.5 val labels (scripts/retrain_avengers_pro_qwen.py).

**OmniRouter / ECCOS** (arXiv 2502.20576) — constrained-optimization router with Lagrangian dual decomposition and adaptive multipliers. Both linked public repos (github.com/dongyuanjushi/OmniRouter and github.com/agiresearch/ECCOS) return 404 as of April 15, 2026 — we implement a **per-request degenerate form** (argmin cost s.t. quality ≥ q_min) from the paper, flagged as optional baseline due to missing code. Batch-level Lagrangian + RAG-augmented predictor (the non-trivial parts of the paper) are acknowledged limitations of our reimplementation.

**QoS-aware DRL routing** (arXiv 2508.00234, IEEE TMC) — DRL + heterogeneous graph attention network for edge LLM routing. No open-source code; full DRL reproduction (simulator, reward shaping, exploration schedule) is out of scope. Cited as related but not reimplemented.

### §7 Dataset
- **Update**: HF dataset with prometheus scores and renamed columns
- **Update**: reference_score methodology table
- **Update**: `llm_judge_scores` naming convention (`{model}_{mode}`)

## Novelty Framing — Preemptive Rebuttal for "Incremental Combination"

Block was rejected for this concern. Must address proactively in the paper.

**What is genuinely NEW (not combination):**
1. **Batch orchestration algorithm** (§3.3) — fused routing + scheduling at batch granularity with LPT + two-phase normalization across heterogeneous instances. No prior work does joint model-selection and instance-assignment in one batched pass.
2. **CDF-based probabilistic SLO filtering** (§3.4) — using bucket classifier softmax as empirical CDF for distribution-free filtering. Nobody in serving literature does `P(signal ≤ threshold) ≥ θ` via learned classifiers. QLM assumes Normal, SLOs-Serve uses points, PolyServe uses deadlines.
3. **Tiered constraint relaxation** — every request served (no hard rejection). Progressive relaxation in operator-specified order. New policy that changes system semantics vs SLOs-Serve (reject) and PARD (drop).
4. **Fused multi-head encoder** — one forward pass → all models × all targets, providing both probability distributions (for CDF filtering) and point estimates (for scoring).

**What IS standard engineering (not claimed as contribution):**
- XGBoost latency prediction, KNN quality lookup, sentence-transformer embeddings, LLM judges for quality scoring, vLLM prefix caching

**Rebuttal framing:**
> "The batch orchestration algorithm and CDF-based probabilistic filtering are new contributions, not combinations. No prior system performs joint model-selection and instance-assignment at batch granularity with two-phase normalization, and no prior SLO enforcement uses learned classifier distributions for distribution-free filtering. Standard predictors (XGBoost, BERT) are used within this new framework — that is engineering, not our claimed contribution."

**Where to address in paper:**
- §1 Introduction: clearly state what's new vs what's engineering
- §3 System Design: emphasize the algorithmic novelty of batch orchestration + CDF filtering
- §6 Related Work: explicit comparison showing no prior work does these specific things
- §5 Evaluation: ablation isolating contribution of batch (vs per-request) and CDF (vs point threshold)

**TODO: Full review after CloudLab evaluation and deployment model decisions finalized.**

---

### §5.X Overhead Analysis (UPDATED — April 14 measurements on 4× A30 CloudLab)

**Status:** Apr 2 numbers above (55 ms encoder, ~60 ms total) were CPU-bench, single-threaded, projected ONNX. Apr 14 on real cluster + real scheduler measured two optimizations actually landed; ONNX projected numbers did NOT materialize and are removed.

**Per-request scheduling overhead on 4× A30 cluster (4 QPS, Qwen-2.5-7B workload, `scheduling_overhead_breakdown` from 50 reqs × `filter=cara_tiered`):**

| Stage | total_sched_ms p50 | estimator_ms p50 | batch_wait_ms p50 | Notes |
|-------|:-:|:-:|:-:|------|
| **Baseline** (eager + `num_running>0` busy signal) | 251.4 | 96.6 | 111.9 | pre-Apr-14 state |
| + busy-signal fix (Task #20) | 180.2 | 99.7 | 50.9 | `num_waiting` / capacity-pressure signal |
| + SDPA + torch.compile (Task #22) | **161.5** | **87.1** | 50.6 | default in `load_fused_model` |
| **Net reduction** | **−36%** | −10% | −55% | — |

**Two independent optimizations, orthogonal, both land clean.**

**Optimization 1 — busy-signal fix** (`cara_serve.py:184-218`, `adapt_params`):
- Prior definition: `busy = (num_running > 0)` → on ANY live cluster, busy_ratio ≈ 1.0 → adaptive picked **overload preset** (`max_batch=32, timeout=200ms`) even at trivial load
- New definition: continuous capacity-pressure ratio = `max(seq_pressure, kv_pressure)` averaged across instances, where `seq_pressure = max(num_waiting, num_running − max_num_seqs/2) / max_num_seqs` and `kv_pressure = kv_cache_utilization`
- At 4 QPS on 4 instances, new ratio ≈ 0 → picks **idle preset** (`max_batch=4, timeout=50ms`)
- Measured: `batch_wait_ms` p50 111.9 → 50.9 (−55%); `total_scheduling_ms` p50 251.4 → 180.2 (−28%); E2E unchanged

**Optimization 2 — SDPA + torch.compile** (`train_fused_bert_predictor.py:load_fused_model`, now default):
- Prior: `attn_implementation="eager"` (training-compat), no torch.compile
- New defaults at inference load: `attn_impl="sdpa"` + `torch.compile(mode="reduce-overhead")` (state_dict weights identical — only attention kernel + compiled graph)
- Also patched `model_estimator.py:_predict_buckets_fused(_batch)` and `estimate_length` to call model via `__call__` instead of `.predict()` — torch.compile's OptimizedModule wraps `__call__`; `.predict()` bypassed the wrapper
- Controlled bench on idle node1 (5 reps × 60 iters, `torch.set_num_threads(16)`, paired t p≈0): **1.17–1.63× estimator speedup** at batch {1,4,8,16}
- In-scheduler estimator_ms p50 96.6 → 87.1 (~10% reduction, less than bench predicted due to shape-varying recompile + async overhead)

**What did NOT work (tested + documented as negative results):**

| Attempt | Outcome | Reason |
|---------|:-:|------|
| ONNX Runtime (CPU) | 0.66-0.70× (slower than PyTorch 2.9) | ModernBERT rotary+alternating-attention pattern doesn't match `onnxruntime.transformers.optimizer(model_type="bert")`; Attention op not fused; MLAS vs MKL-DNN kernel gap |
| ONNX INT8 dynamic quant | Shape-inference error on fused per-model heads | Quant tool infers `(768,)` while heads have `(16,)` output; fixable but not trivial |
| SDPA alone (no compile) | ~1.0× (no change) | HF ModernBERT falls back to eager for non-standard mask patterns |
| torch.compile alone (eager attn) | 0.97-1.02× | Compile needs SDPA to expose fusable patterns |

**Paper framing:**
- **Remove** the projected ONNX column from the overhead table (was speculative, measured to be slower)
- **Add** the busy-fix + SDPA-compile lineage table above as the actual overhead story
- **Add** a sentence in Related Work / discussion: "ONNX Runtime did not improve inference on our ModernBERT fused predictor (PyTorch 2.9 CPU with SDPA + torch.compile is already near-optimal). ONNX remains viable for GPU deployment or vanilla-encoder variants."

**Dual-runtime infrastructure retained** (ONNX export + parity validation scripts in workspace) for future: GPU deployment, vanilla BERT/RoBERTa encoder swap, Rust-based serving path.

**Reproducibility artifacts** (all under `cara_paper/smoke_test_apr_13/scripts/`):
- `bench_variants_v2.py` — variance-controlled bench with paired t-test
- `export_fused_to_onnx.py` — ONNX export + parity + CPU bench (negative result)
- `bench_inference_variants.py` — first-attempt variant matrix (variance-limited on busy node)

### §5.Y Baselines (REWRITTEN April 15, 2026 afternoon — Cara-full = integrated batch scheduler)

**Correction notice.** The prior version of this section described a "two-layer pluggable matrix" as the evaluation structure. That was a misframe. The pluggable Router × Dispatcher × Filter infrastructure built in Phase A (#23–#33) is **engineering scaffolding for running wrapper/ingredient baselines**, not the thing the paper claims as Cara.

**Cara-full (the paper system).** Cara is an **integrated online batch scheduler**. On each scheduling tick:

1. **LPT sort** the pending-request queue by predicted output length (longest first).
2. Dispatch in sorted order **one request at a time**.
3. For each dispatched request:
   - **Bucket-based probabilistic feasibility** — in the target design, a TTFT / TPOT / budget / quality check driven by the length-bucket CDF from the fused estimator. **Current code status:** `CaraCDFFilter` at `block/global_scheduler/cara/filters/cara_cdf_filter.py` covers only the TTFT / TPOT portion; budget and quality feasibility are today enforced inline in the legacy path at `block/global_scheduler/cara/cara_serve.py:906`. A unified four-constraint bucket filter is planned — see codex design doc §5.11 for the honest scope and its implication for the `Cara-point-filter` ablation.
   - **Multi-objective per-request scoring** over **four** terms: quality + latency + cost + cluster-memory **balance** (`w_quality`, `w_latency`, `w_cost`, `w_balance`; all four weighted and summed in `cara_serve.py:780` and `:1117`) → pick the best `(model, instance)` pair.
4. **No batch-level joint optimization** (Hungarian-style matching is skipped because it can't tractably enumerate all combinations).

Invoked via `--scheduling cara`. Design notes in `cara_paper/codex/apr15_baseline_comparison_design.md` (codex-reviewed).

---

**Main evaluation table — Cara-full vs wrapper / composed baselines.**

| # | Row | Role | Knob |
|---|---|---|---|
| 1 | **Cara-full** | integrated system (LPT + bucket filter + four-term scoring + joint `(model, instance)` assignment) | `w_quality` sweep |
| 2 | `vllm_sr-wrapper` | external peer scheduler (Envoy + extproc) | — single point |
| 3 | `RouteLLM-wrapper (mf)` | paper-recommended `mf` variant + pretrained HF checkpoint + published α=0.11593 calibration; paired with RR + shortest_queue | α sweep |
| 4 | `Avengers-Pro-wrapper` | upstream `SimpleClusterRouter`, Qwen-trained K-means + rankings; paired with RR + shortest_queue | `performance_weight` sweep |
| 5 | `BEST-Route-wrapper` | DeBERTa-v3-small multi-head router, Qwen-armoRM pair-wise labels (stripped Best-of-N); paired with RR + shortest_queue | threshold `t` sweep |

Retired: `best_route-wrapper` (argmax quality — renamed to `quality_greedy_ablation` and moved to ablation §3.3) and "strongest composed baseline" (replaced by wrapper + load-aware-dispatch pairings above).

Not in the main table: `vllm_native_3b/7b` (pure model-latency floor — appendix only); `3B-only` or `7B-only` rows (uninformative).

---

**Main ablation table — Cara-full ingredients (primary ablation).**

| # | Row | Ablates | Isolates |
|---|---|---|---|
| 1 | `Cara-full` | — | reference |
| 2 | `Cara-no-LPT` (FIFO) | LPT sort | value of LPT ordering |
| 3 | `Cara-no-filter` | bucket filter | value of feasibility filtering |
| 4 | `Cara-point-filter` | bucket-CDF → point estimate | value of bucket-based vs point filtering |
| 5 | `Cara-no-quality-term` | `w_quality = 0` | value of quality-aware scoring |
| 6 | `Cara-no-latency-term` | `w_latency = 0` | value of latency-aware scoring |
| 7 | `Cara-no-cost-term` | `w_cost = 0` | value of cost-aware scoring |
| 7b | `Cara-no-balance-term` | `w_balance = 0` | value of cluster-memory-balance term in the score |
| 8a | `Cara-simple-shortestQ` | drop joint scoring → LPT order + `shortest-queue` | value of scoring-based joint `(model, instance)` assignment vs simple load-aware dispatch |
| 8b | `Cara-simple-llumnix` | drop joint scoring → LPT order + `llumnix_minus` | same comparator against a richer load signal |

Row 4 (`Cara-point-filter`) depends on finishing the unified bucket-feasibility filter (planned — see above). If we cannot land that in time, either restrict the filter ablation to TTFT/TPOT variants or move this row to supplement.

This is the **primary ablation**. It tests Cara-full's claimed ingredients directly rather than enumerating a combinatorial cross-product.

---

**Supporting study 1 — routing-signal swap inside a fixed scheduler shell.**

Hold the non-routing scheduler shell fixed; swap only the routing / quality signal. Exercised via `--scheduling pipeline` + `/v1/config`:
- `best_route`
- `vllm_sr`
- `routellm`
- optionally `avengers_pro` (if stable)
- `cara_score` as an internal signal reference

Answers: *how much of Cara-full's gain comes from its routing signal versus the integrated scheduler around it?*

**Supporting study 2 — filter swap inside a fixed Cara shell.**

Hold LPT + assignment fixed; swap only the filter:
- `none`, `cara_tiered` (CaraCDF), `qlm`, `slos_serve`, `polyserve`, `timebill`

Supporting evidence for the filter axis; does NOT redefine Cara-full as "a router/dispatcher/filter tuple".

**Sanity-check table — standalone SR (Envoy end-to-end).**

SR running end-to-end on its own stack with 2 backend_refs per model. Metrics: latency distribution, throughput, response quality. Goodput reported blank with footnote — SR is SLO-blind.

**Appendix — vllm-native floor + SR internal overhead via Jaeger.**

---

**Quality metric (primary).** deepeval. Other judges only as dataset notes or appendix.

**Implementation status (April 15 afternoon, 2026):**
- Pluggable Router × Dispatcher × Filter infra — ✅ in place (engineering scaffolding for supporting studies and wrapper baselines).
- Cara-full (`--scheduling cara`) — ✅ code path exists (`_schedule_cara_batch`); needs new toggles for the ingredient ablations above. Tasks #51 (LPT on/off), #52 (assignment strategy `{scoring, shortest_queue, llumnix_minus}` operating over all instances), #53 (verify per-term zero-weights `w_quality / w_latency / w_cost / w_balance` round-trip through `/v1/config`), #54 (deepeval post-hoc), #55 (main + ablation sweep script).
- Wrapper router baselines — status varies:
  - `best_route`, `null_fixed`, `random`, `dual_pool`, `omnirouter`, `qlm`, `cara_score` — working in pluggable pipeline.
  - `routellm` — recalibrated on `data/cara/scored/` (val_acc≈0.389 on a 3-class strong-tie-weak task, near random; usable as a signal baseline but document the low classifier quality).
  - `avengers_pro` — retrained on runtime model names (`Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-7B`); wrapper now loads the upstream `SimpleClusterRouter` with local-embedder injection. Routes consistently to the larger model on our test prompts — document this bias in the paper text.
  - `vllm_sr` — peer Envoy+extproc stack operational on d7525-10s10315 (`--minimal`, 2 backend_refs per model matching cluster). Used both (a) as an HTTP-proxy signal inside the pluggable pipeline and (b) as a standalone peer scheduler in the sanity-check table.
- `block/benchmark/cara/benchmark_serving.py` — ✅ supports `--backend cara`, `--backend openai-chat`, goodput, RSO, detailed save.
- E2E bench sweep — ✅ 30-prompt smoke at 4 QPS run against the pluggable path. ❌ main-table numbers against `--scheduling cara` not yet collected; gated on #51–#55.

**Known corrections / unresolved items** — tracked as the live queue #51–#55 in the tracker, with the fairness audit and open items in `cara_paper/codex/apr15_baseline_comparison_design.md` §5 (§§5.6 SLO tiers, 5.7 workload mix, 5.8 QPS sweep, 5.9 cost calibration, 5.11 CaraCDF scope, 5.12 `w_balance`, 5.13 Apr 26 cluster rerun).

---

## Wait For Final Results — Updated April 7
- CSD3 100-epoch RoBERTa — ✅ DONE (marginal over 5ep, overfits on classification)
- DeepEval scoring — ✅ DONE (59,812 train + 14,564 test scored)
- DeepEval as quality signal — ✅ RoBERTa fused MAE=0.226, BestAcc=55.7%, Top2Acc=87.9% (recommended signal)
- LSTM v2 — ✅ DONE (5 instance types trained)
- MLP baseline — ✅ DONE (all quality targets)
- LoRA-encoder RoBERTa (SR-style) — ✅ DONE (6 targets)
- LoRA-encoder ModernBERT (SR-style) — ✅ DONE (6 targets)
- LoRA multi-target (RoBERTa + ModernBERT) — ✅ DONE
- **STILL WAITING**:
  - CSD3 Qwen LoRA (job 27090405): fused judge/ref_score + per-model deepeval/ref_score
  - ~~Comprehensive evaluation: all models × all targets × filtered test set (3,634)~~ — ✅ DONE, VERIFIED April 11, Phase 1 CLOSED
  - Prometheus-7b-v2 scoring (deferred)
  - CloudLab 18-node e2e evaluation (April 26)

## New Architecture Comparison (for §4/§5)
- **Tier 2 (fused-model, per-target)**: RoBERTa FT, ModernBERT FT, LoRA-encoder RoBERTa, LoRA-encoder ModernBERT, KNN, MLP, Qwen LoRA, PFS
- **Tier 3 (fused-model, fused-target)**: Full FT multi-B/A (RoBERTa, ModernBERT), LoRA multi-B (RoBERTa, ModernBERT)
- SR comparison: our fused multi-head (1 forward pass) vs SR's N LoRA passes
- All models on personal VM, ready for eval on filtered test set

---

## Baseline Cost-Handling Comparison (added 2026-05-03; for §3.5 / §5 / Related Work)

A side-by-side audit of how each main-table baseline incorporates cost into
its routing decision. Built from direct code inspection (file:line cited);
candidate for inclusion as **Table X** in the paper to motivate Cara's
fine-grained per-(request, instance) cost model.

### Vendor pricing — May 2026 industry rates (Table A)

Sourced from current public pricing pages of the cheapest-available host per
Qwen 2.5 size. **72B has asymmetric input/output pricing** ($0.38 vs $0.40).

| Model | Input ($/1M tokens) | Output ($/1M tokens) | Provider | Source |
|---|---|---|---|---|
| Qwen 2.5 3B  | $0.06 | $0.06 | Together AI | https://www.together.ai/pricing |
| Qwen 2.5 7B  | $0.07 | $0.07 | Novita AI    | https://novita.ai/pricing |
| Qwen 2.5 14B | $0.15 | $0.15 | Novita AI    | https://novita.ai/models/llm |
| Qwen 2.5 72B | **$0.38** | **$0.40** | Novita AI    | https://novita.ai/models/model-detail/qwen-qwen-2.5-72b-instruct |

These rates are wired into:
- `block/config/cara/model_deployment_apr26_phase2.json` `pricing:` block per model
  (with `provider` + `source` fields for paper citation)
- `/mydata/vllm_sr/config.yaml` `providers.models[*].pricing` per model
- `home/wd312/avengers_4model_recluster_v3.py` `--cost-per-{input,output}-token-1m` defaults
- `home/wd312/best-route-llm/hybrid_llm/pair_ranker/data.py:16-17` `model2prompt_cost` + `model2output_cost` dicts

### Cost-handling table (updated 2026-05-03 PM with Novita/Together pricing)

| Baseline | Trained? | Cost at training | Cost at inference | Input/output split (CODE) | Input/output split (CONFIG) | Granularity | Cost knob | Source-of-truth |
|---|---|---|---|---|---|---|---|---|
| **AvengersPro v3 (post 2026-05-03 fix)** | Yes (KMeans + rankings) | `(cost_in × mean_input_len + cost_out × mean_output_len)` per (cluster, model); split-aware | None — uses pre-baked `cluster_rankings.json` | **Yes** (script v3) | $0.06/$0.07/$0.15/(0.38 in, 0.40 out) per 1M | Cluster-level (32 clusters) | `pw ∈ {0.25, 0.39, 0.53, 0.7, 0.8}` | `avengers_4model_recluster_v3.py:30-46, 76-95`; `AvengersPro/balance_cluster_router.py:192-198` |
| **vLLM-SR (AutoMix)** | No (semantic classifier + escalation looper) | n/a | `params.Pricing.PromptPer1M` constant — sorts/scores models by prompt rate; AutoMix POMDP value = (1−t)·quality + t·(1−norm_cost) | **No** (prompt rate only; completion_per_1m is telemetry-only) | Same prices, same asymmetry — but only the prompt rate is consumed by routing logic | Model-level (rate constant, no per-request scaling) | `cost_quality_tradeoff` (default 0.3) + `escalation_order ∈ {automix, cost, size}` | `looper/confidence.go:182-260, 190, 215, 254`; `sessiontelemetry/telemetry.go:191-192` |
| **BEST-Route 4-way faithful (post 2026-05-03 fix; retrain in flight)** | Yes (DeBERTa-v3-small NClassReranker, prob_nlabels loss) | `prompt_cost × prompt_tokens + output_cost × output_tokens` per (prompt, model); split-aware | None — cost-quality tradeoff in trained softmax | **Yes (training only)** | $0.06/$0.07/$0.15/(0.38, 0.40) wired in `data.py:16-17` | Per-(prompt, model) at train; per-prompt at inference | threshold `t ∈ {0.3, 0.5, 0.6, 0.7, 0.8}` | `hybrid_llm/pair_ranker/data.py:16-17, 191`; `ranker.py:98, 164, 202` |
| **Cara (post 2026-05-03 fix)** | n/a (sidecar XGBoost + KNN/RoBERTa) | n/a | `prompt_tokens × cost_per_input_token + expected_output_len × cost_per_output_token` per (request, instance), runtime-computed | **Yes** (full split, runtime-asymmetric) | Same prices, asymmetry honored end-to-end | **Per-request, per-instance — finest granularity** | `w_cost ∈ scoring_weights` + per-instance `cost_per_{in,out}_token` injected from `model_deployment.pricing` | `cara_serve.py:838-845, 1321-1324`; `cara_serve.py:2259-2280`; `model_deployment_apr26_phase2.json:pricing:` |

**Where input/output split is actually exercised in routing:**
- AvengersPro: split costs flow into `mean_cost` per cluster → into `balance_score`
  → into ranked-model selection at training. **Split applied.**
- BEST-Route: split costs flow into per-record `candidate["cost"]` → into
  `_calculate_loss(...,scores,costs)` → DeBERTa learns cost-quality tradeoff.
  **Split applied at training (encoded in trained softmax).**
- Cara: per-request, runtime: `cost = prompt_tok × cost_in + output_tok × cost_out`.
  **Split applied at every routing decision.**
- vLLM-SR: code reads only `Pricing.PromptPer1M` for routing; `CompletionPer1M`
  is loaded but ignored by router. **Split NOT applied to routing** — would
  need an upstream patch to fix. Acceptable for paper if framed as "vLLM-SR
  routes by prompt-rate only, by upstream design."

**Asymmetric-pricing sensitivity (with $0.38 input vs $0.40 output for 72B):**
- Cara: routes 72B more cautiously when output-heavy requests arrive (cost_out
  is 5% higher than cost_in). Real per-request asymmetry.
- BEST-Route: training data records have explicit per-record split costs;
  trained model softmax reflects asymmetry on average across train set.
- AvengersPro: per-cluster aggregate; asymmetry visible in cluster-level
  cost statistics (mean_inlen × $0.38 + mean_outlen × $0.40 for 72B).
- vLLM-SR: doesn't see the asymmetry — routes 72B as if priced at $0.38 flat.

**Implications for the paper:**
- Insert vendor pricing table (Table A above) in §5 with provider citations.
- Insert cost-handling comparison table in §3.5 (Baseline Comparison) or
  §6.4 (Related Work, quality-cost routing).
- Granularity column supports claim "Cara is the most fine-grained cost-aware
  system in the comparison."
- Asymmetric pricing is REAL in our data (72B is the only Qwen with split
  prices); systems that only consume prompt rate (vLLM-SR) cannot fully
  reflect this — note as upstream limitation.
