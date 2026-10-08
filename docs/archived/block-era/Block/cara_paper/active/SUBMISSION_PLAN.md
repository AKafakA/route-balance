# CARA Paper Submission Plan

**Document Purpose**: Strategic analysis and submission planning for the CARA paper from a distributed systems research perspective.

**Created**: January 25, 2026
**Status**: Planning Phase — **revision 2 methodology locked Apr 15 PM** (see pointer below)

---

## ⚠ Evaluation methodology — revision 2 (Apr 15 PM)

The main-table + ablation + figures + data plan is **frozen** per `cara_paper/codex/apr15_baseline_comparison_design.md` (revision 2). Change-log in `cara_paper/codex/CODEX_REVIEW_PACKAGE.md`.

Summary for this document's §5 / §6 sections:

- **Main table** (5 rows): Cara-full, vllm-SR, RouteLLM-mf (pretrained, α sweep), Avengers-Pro (α sweep), BEST-Route (Qwen-armoRM trained, t sweep). Wrappers paired with RR + shortest-queue dispatch.
- **Ablation** (10 rows): Cara-full, no-LPT, no-filter, point-filter (planned), no-{quality,latency,cost,balance}-term, Cara-simple-shortestQ, Cara-simple-llumnix, quality_greedy_ablation.
- **Figures**: 2D cost-quality Pareto (headline) + 2D cost-latency Pareto + CAP radar per router + goodput bar + throughput bar + memory-variance time-series + w_balance sensitivity (appendix) + routing-overhead stack + Table 1.
- **Data**: `best-route-v3-test-500-clean.jsonl` resampled from `test_scored_filtered.jsonl` (100% deepeval coverage, zero predictor-train leakage). Supersedes the contaminated legacy bench.
- **Quality**: pre-computed deepeval lookup (RouterBench-style), not live judge.
- **Cost**: per-token vendor pricing (MaaS framing).
- **Citations**: RouterBench (Hu 2024), MoE-CAP (Jiang 2025), RouterEval (Huang 2025) + LLMRouter lib + BentoML Explorer + Hybrid-LLM + BEST-Route + RouteLLM + Avengers-Pro + BOute + vllm-SR.

All earlier sections of this submission plan that conflict with the above are **superseded by revision 2**.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Paper Positioning](#paper-positioning)
3. [Novelty Analysis](#novelty-analysis)
4. [Strengths and Weaknesses](#strengths-and-weaknesses)
5. [Venue Analysis](#venue-analysis)
6. [Reviewer Concerns and Responses](#reviewer-concerns-and-responses)
7. [Evaluation Plan](#evaluation-plan)
8. [Submission Timeline](#submission-timeline)
9. [Action Items](#action-items)

---

## Executive Summary

### Paper Title
**Cara: Co-optimizing Super Heterogeneous LLM Serving through Model Routing and Predictive Batch Scheduling**

### Core Contribution
CARA unifies **model routing** (which model?) and **load balancing** (which instance?) into a single **batch-level multi-objective optimization**, enabling efficient LLM serving on heterogeneous GPU clusters.

### Assessment
- **Overall Grade**: B+ / A- (Solid work, publishable at good venue)
- **Best Fit Venues**: SoCC, ATC, EuroSys
- **Strongest Angle**: Heterogeneous hardware support (real gap in literature)

---

## Paper Positioning

### The Problem
Modern LLM serving clusters are heterogeneous:
- Multiple GPU generations (P100 → A100)
- Multiple model sizes (3B → 72B)
- Multiple software versions (vLLM v0.4 vs v0.6)

Current systems handle routing and scheduling separately, leading to suboptimal decisions.

### The Insight
Routing and load balancing must be **coupled** to exploit heterogeneity:
- Same model on different hardware has different performance
- Same request on different models has different quality
- Per-request decisions miss batch-level optimization opportunities

### The Solution
CARA: A unified batch scheduler that:
1. Pulls fresh state from all instances before each batch
2. Sorts requests by predicted tokens (LPT heuristic)
3. Assigns each request considering latency + quality + cost jointly
4. Supports Request-Specific Objectives (RSO) for per-request constraints

### Recommended Framing
> "Modern LLM clusters are heterogeneous: different GPU generations, model sizes, and software versions coexist. We show that naive routing + load balancing leaves 30% performance on the table compared to joint optimization that exploits this heterogeneity."

---

## Novelty Analysis

### What Exists in Prior Work

| Component | Prior Art | Limitation |
|-----------|-----------|------------|
| Model routing | RouteLLM, FrugalGPT, Hybrid LLM | No load awareness |
| Load balancing | JSQ, Po2, Least-loaded | No quality awareness |
| Batch scheduling | Borg, Kubernetes | Not LLM-specific |
| LLM local scheduling | vLLM, Orca, Sarathi | Single instance only |
| Multi-objective | Common in cloud | Not for LLM serving |

### CARA's Novel Contributions

| Contribution | Novelty Level | Notes |
|--------------|---------------|-------|
| Fusing routing + balancing | **Medium-High** | Surprisingly unexplored |
| Batch-level (not per-request) | **Medium** | Enables global optimization |
| RSO (per-request objectives) | **Medium** | Nice abstraction |
| Heterogeneous hardware | **High** | Real gap in literature |
| LPT for LLM | **Low** | Classic technique, new application |
| XGBoost with rich features | **Medium** | Context distribution, not just counts |

### Closest Related Work

| System | Focus | Gap CARA Fills |
|--------|-------|----------------|
| QLM | Queue-based scheduling | Single model, no quality routing |
| DistServe | P/D disaggregation | Within-model only |
| RouteLLM | Quality-aware routing | No load awareness |
| Helix/HexGen | Heterogeneous placement | Offline, not runtime |
| Clipper | Model serving | No LLM-specific optimizations |

---

## Strengths and Weaknesses

### Strengths

1. **Unified Control Plane** ✓
   - Combining routing + scheduling is the right architectural insight
   - Current systems make these decisions separately

2. **Practical Heterogeneity Handling** ✓
   - Testbed (P100 → A100, 3B → 72B) reflects real deployments
   - Most papers assume homogeneous clusters

3. **LPT Heuristic is Well-Grounded** ✓
   - Graham (1969) gives 4/3-approximation for makespan
   - Theoretical backing, not just "we use greedy"

4. **Clean State Management** ✓
   - Active pull + within-batch updates is simple and correct
   - Avoids distributed state synchronization complexity

5. **Rich Instance State API** ✓
   - Context length distribution (p50/p95/max) not just queue lengths
   - Captures O(n²) attention cost

### Weaknesses

1. **Scalability Question** ⚠️
   - 18 nodes is small
   - Active pull: 1000 HTTP requests per batch at scale?
   - Need scalability analysis or hierarchical design

2. **LPT Approximation Caveat** ⚠️
   - 4/3-approximation is for **identical machines**
   - Heterogeneous case has no proven bound
   - Either prove new bound or be careful about claims

3. **Quality Estimator Under-specified** ⚠️
   - Latency predictor is well-defined (XGBoost + features)
   - Quality estimator is vague: "predict Q_{r,i}"
   - This is actually harder than latency prediction

4. **Missing Prefix Caching** ⚠️
   - No consideration for shared prefixes
   - Modern systems (SGLang, vLLM) optimize heavily for this
   - May need to acknowledge limitation

5. **Single Point of Failure** ⚠️
   - Centralized scheduler has no failover
   - No mention of recovery mechanisms

6. **Weak Baselines** ⚠️
   - Random and round-robin are too weak
   - Need: JSQ, Po2, RouteLLM, least-loaded

---

## Venue Analysis

### Venue Comparison

| Venue | Fit | Novelty Bar | Acceptance | Notes |
|-------|-----|-------------|------------|-------|
| **SoCC** | ★★★★★ | Medium | ~25% | Best fit (cloud serving) |
| **ATC** | ★★★★☆ | Medium | ~20% | Practical systems valued |
| **EuroSys** | ★★★★☆ | Medium-High | ~18% | Wants ablations |
| **NSDI** | ★★★☆☆ | High | ~15% | Networking focus |
| **OSDI/SOSP** | ★★☆☆☆ | Very High | ~15% | Needs breakthrough |

### Venue-Specific Requirements

| Venue | Must Have | Nice to Have |
|-------|-----------|--------------|
| **SoCC** | Cost analysis, cloud framing | Multi-tenant, spot instances |
| **ATC** | Real deployment, practical lessons | Production traces |
| **EuroSys** | Thorough ablations, clean design | Formal analysis |
| **NSDI** | Network overhead analysis | Geo-distributed |

### Recommendation

```
┌────────────────────────────────────────────────────────┐
│  Recommended Strategy                                   │
├────────────────────────────────────────────────────────┤
│                                                         │
│  1. Primary: SoCC 2026                                 │
│     - Best fit for heterogeneous cloud serving         │
│     - Practical contribution valued                    │
│     - Lower novelty pressure                           │
│                                                         │
│  2. Backup: ATC 2026                                   │
│     - Broader systems audience                         │
│     - Quick turnaround if SoCC rejects                 │
│                                                         │
│  3. Stretch: EuroSys 2027                              │
│     - Higher prestige                                  │
│     - Requires theoretical grounding                   │
│                                                         │
└────────────────────────────────────────────────────────┘
```

---

## Reviewer Concerns and Responses

### Anticipated Questions

| Question | Response |
|----------|----------|
| "What's the overhead of pulling `/instance_stats`?" | Measure RTT distribution, show amortized over batch |
| "How does prediction error affect routing?" | Sensitivity analysis with varying MAE |
| "Why XGBoost over neural network?" | Lower latency, interpretable, sufficient accuracy |
| "How do you handle bursty arrivals?" | Adaptive batching adjusts to load |
| "What if quality and latency are anti-correlated?" | Pareto frontier analysis, RSO lets user choose |
| "Does LPT approximation hold for heterogeneous?" | Acknowledge limitation, show empirical improvement |
| "What about prefix caching?" | Acknowledge as orthogonal optimization |
| "Scalability beyond 18 nodes?" | Discuss hierarchical extension |

### Potential Weaknesses to Address

1. **"This is just combining existing techniques"**
   - Counter: The combination is non-trivial; no existing system does this fusion
   - Evidence: Show fusion > sum of parts empirically

2. **"Quality estimator is hand-wavy"**
   - Option A: Fully specify (more work)
   - Option B: Scope out ("we focus on latency; quality is orthogonal")

3. **"Small-scale evaluation"**
   - Counter: Real hardware heterogeneity is the point
   - Mitigation: Add simulation for larger scale

---

## Evaluation Plan

### Required Experiments

#### 1. Baseline Comparisons (revision 2)
```
- Cara-full (`--scheduling cara`: LPT + bucket feasibility filter + four-term multi-objective scoring + joint assignment)
- vllm_sr-wrapper (external peer scheduler via Envoy + extproc, single point)
- RouteLLM-wrapper (mf, pretrained + published α=0.11593; paired with RR + shortest_queue)
- Avengers-Pro-wrapper (Qwen-trained K-means + rankings; `performance_weight` sweep; paired with RR + shortest_queue)
- BEST-Route-wrapper (DeBERTa-v3-small on Qwen + armoRM pair-wise labels, stripped Best-of-N; t sweep; paired with RR + shortest_queue)
```
Retired: `best_route-wrapper` argmax-quality (renamed to `quality_greedy_ablation`, moved to ablations) and "strongest composed baseline" (replaced by wrapper + load-aware-dispatch pairings above).
> Superseded: ~~Random / RR / Least-loaded / JSQ / Po2 / RouteLLM-only / Decoupled-best-router-best-LB~~. Random / RR / Shortest-Queue / JSQ / Po2 still appear as dispatcher variants inside Supporting Study 1 (routing-signal swap inside a fixed pluggable shell), not as top-level rows. See `cara_paper/codex/apr15_baseline_comparison_design.md`.

#### 2. Ablation Study — per corrected Apr 15 design
```
| Configuration                  | Purpose                                           |
|--------------------------------|---------------------------------------------------|
| Cara-full                      | Complete system (reference)                       |
| Cara-no-LPT (FIFO)              | LPT ordering contribution                         |
| Cara-no-filter                  | Feasibility filter contribution                   |
| Cara-point-filter               | Bucket-CDF vs point-estimate filter (planned*)    |
| Cara-no-quality-term            | w_quality = 0 in scorer                           |
| Cara-no-latency-term            | w_latency = 0 in scorer                           |
| Cara-no-cost-term               | w_cost = 0 in scorer                              |
| Cara-no-balance-term            | w_balance = 0 (cluster-memory-balance term)       |
| Cara-simple-shortestQ           | LPT + shortest-queue across all instances         |
| Cara-simple-llumnix             | LPT + `llumnix_minus` across all instances        |
| Cara-homogeneous                | Cara on a single-model cluster (8× A30)           |
```
*`Cara-point-filter` requires the unified bucket-feasibility filter to land first (see design §5.11). If it doesn't land in time, scope the filter ablation to TTFT/TPOT only.

Retired: ~~"without batch scheduling" as its own ablation~~ — Cara dispatches LPT-sorted requests one at a time; there is no batch-level joint optimization to ablate. Its signal is captured by `Cara-simple-shortestQ` / `Cara-simple-llumnix` (scoring-based joint assignment vs simple load-aware).

#### 3. Sensitivity Analysis
```
- Prediction error (MAE: 100ms, 200ms, 500ms) — predictor-noise injection on the quality/latency heads
- QPS / load level (low, medium, high, saturating, overloaded)
- Heterogeneity level (2 HW types, 4 HW types)
- SLO tier (balanced vs strict, per design §5.6)
- Workload profile (balanced mix vs hard-skewed, per design §5.7)
```
Retired: ~~Batch size (4, 16, 64, 256)~~ — Cara dispatches LPT-sorted requests one at a time, no batch-size axis.

#### 4. Overhead Breakdown — Cara-full per-request fields (emitted by `_schedule_cara_batch`)
```
- batch_wait_ms          time spent waiting in the LPT-sorted queue before dispatch
- estimator_ms           fused model-estimator forward pass
- stats_fetch_ms         cross-instance /instance_stats pull
- sidecar_ms             XGBoost / latency-predictor sidecar call
- scoring_ms             multi-objective joint (model, instance) scoring
- total_scheduling_ms    sum of above + dispatch plumbing
```
Pluggable-pipeline path (used for supporting studies only) emits a different set (`router_ms`, `estimator_ms`, `stats_fetch_ms`, `filter_ms`, `dispatcher_ms`, `total_sched_ms`); do not mix the two in the same figure.

#### 5. Scalability (Simulation)
```
- 50, 100, 200, 500 instances
- Hierarchical scheduling design
```

### Metrics to Report

| Category | Metrics |
|----------|---------|
| Latency | P50, P95, P99, mean E2E latency |
| Throughput | Requests/sec, tokens/sec |
| Quality | Average quality score, quality distribution |
| SLO | Goodput (% meeting SLO) |
| Fairness | Jain's fairness index across instances |
| Overhead | Scheduler CPU, memory, network |
| Cost | $/request (if cost model available) |

---

## Submission Timeline

### Deadlines

| Venue | Abstract | Paper | Notification |
|-------|----------|-------|--------------|
| SoCC 2026 | ~June 2026 | ~June 2026 | ~Sept 2026 |
| ATC 2026 | ~Jan 2026 | ~Jan 2026 | ~April 2026 |
| EuroSys 2026 | ~Oct 2025 | ~Oct 2025 | ~Jan 2026 |
| EuroSys 2027 | ~Oct 2026 | ~Oct 2026 | ~Jan 2027 |

### Suggested Plan

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Implementation (Now - Feb 2026)                       │
├─────────────────────────────────────────────────────────────────┤
│  - Complete XGBoost latency predictor                           │
│  - Implement LPT batch scheduler in cara_serve.py               │
│  - Implement adaptive batching                                  │
│  - Implement tiered constraint handling                         │
│  - Basic end-to-end integration                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Evaluation (Feb - April 2026)                         │
├─────────────────────────────────────────────────────────────────┤
│  - Implement all baselines                                      │
│  - Run comprehensive experiments                                │
│  - Ablation studies                                             │
│  - Sensitivity analysis                                         │
│  - Overhead measurements                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: Writing (April - June 2026)                           │
├─────────────────────────────────────────────────────────────────┤
│  - Complete paper draft                                         │
│  - Internal review                                              │
│  - Revision based on feedback                                   │
│  - Submit to SoCC 2026                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Backup: ATC 2027 or EuroSys 2027                               │
├─────────────────────────────────────────────────────────────────┤
│  - If SoCC rejects, strengthen based on reviews                 │
│  - Add requested experiments                                    │
│  - Resubmit to ATC or EuroSys                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Action Items

### Immediate (Next 2 Weeks)

- [ ] Complete `/instance_stats` client in Block scheduler
- [ ] Implement `ServiceRateTracker` for EMA computation
- [ ] Implement LPT-based batch scheduler skeleton
- [ ] Collect training data for XGBoost

### Short-term (Next Month)

- [ ] Train XGBoost latency predictor
- [ ] Implement adaptive batching
- [ ] Implement tiered constraint handling
- [ ] Basic end-to-end integration test

### Medium-term (Next 2 Months)

- [ ] Implement baseline schedulers (JSQ, Po2, etc.)
- [ ] Design quality estimator (or scope out)
- [ ] Run initial comparison experiments
- [ ] Identify gaps in evaluation

### Pre-submission (1 Month Before)

- [ ] Complete all experiments
- [ ] Write full paper draft
- [ ] Internal review and revision
- [ ] Prepare supplementary materials

---

## Key Technical Decisions

### Finalized Design Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Predictor model | XGBoost | Non-linear effects, interpretable |
| State query | Per-batch active pull | Simple, accurate |
| Scheduling algorithm | LPT + greedy | O(N×M), proven heuristic |
| State updates | Within-batch only | No staleness complexity |
| Constraint handling | Tiered relaxation | Configurable strictness |
| Batch sizing | Load-aware adaptive | Amortize overhead |

### Open Design Questions

| Question | Options | Decision Needed By |
|----------|---------|-------------------|
| Quality estimator | Full implementation vs scope out | Feb 2026 |
| Prefix caching | Support vs acknowledge limitation | Feb 2026 |
| Scalability | Hierarchical design vs simulation | March 2026 |
| Failure handling | Full recovery vs basic | April 2026 |

---

## Success Criteria

### For SoCC Acceptance

1. **Clear contribution**: Fused routing + scheduling for heterogeneous LLM serving
2. **Solid evaluation**: Beat strong baselines by meaningful margin (>15%)
3. **Practical relevance**: Real deployment on heterogeneous hardware
4. **Clean design**: Well-structured system with clear abstractions

### Stretch Goals

1. Theoretical analysis (approximation bounds)
2. Large-scale simulation (100+ instances)
3. Production deployment experience
4. Open-source release with adoption

---

## References for Positioning

### Must-Cite Papers

- Clipper (NSDI '17) - Model serving baseline
- Orca (OSDI '22) - LLM scheduling
- vLLM (SOSP '23) - PagedAttention
- RouteLLM (arXiv '24) - Quality-aware routing
- DistServe (OSDI '24) - P/D disaggregation

### Positioning Against

- QLM: We add quality-aware routing
- RouteLLM: We add load-aware scheduling
- Helix: We do runtime adaptation, not offline placement

---

**Document End**

*Last Updated: January 25, 2026*
