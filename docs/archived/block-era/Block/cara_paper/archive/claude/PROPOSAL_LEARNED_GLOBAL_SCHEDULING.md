# Research Proposal: Learning to Schedule LLM Inference Batches via Global Ranking

**Status**: Early-stage research direction. Linked to Task #22.
**Priority**: Future work — not for current RouteBalance submission.
**Potential venues**: MLSys, EuroSys, NeurIPS (ML-for-Systems), NSDI

---

## 1. Problem Statement

Modern LLM serving systems decouple prediction and scheduling: separate models predict latency (XGBoost), quality (RoBERTa), and output length (bucket classifier), and a hand-designed scoring function combines them into assignment decisions. This modular design is practical but leaves performance on the table:

1. **No cross-signal interaction**: The latency predictor doesn't know about quality, and vice versa. A "hard math prompt on a busy A100" has complex interactions between prompt difficulty, instance load, and expected quality that separate models can't capture.
2. **Hand-tuned scoring**: The multi-objective scoring function (`w_lat * lat + w_cost * cost - w_qual * qual + w_bal * bal`) requires weight tuning per workload. Weights that work for chatbot traffic fail for code generation.
3. **No cross-request reasoning**: Per-request scoring treats each prompt independently. Batch-level patterns ("this batch has 5 hard prompts and 3 easy ones — distribute the hard ones first") require the LPT heuristic, but optimal batch assignment may be more nuanced.
4. **Sequential suboptimality**: Greedy assignment (LPT) is an approximation. Each assignment affects subsequent options, but the scoring function doesn't anticipate future decisions.

**Research question**: Can we replace hand-designed batch scheduling with an end-to-end learned policy that directly maps (batch of prompts, cluster state) → optimal assignment, trained from execution data?

---

## 2. Proposed Approach

### 2.1 Architecture: Cross-Attention Ranking Network

```
Batch of N prompts                    M instance states
    │                                      │
    ▼                                      ▼
[Prompt Encoder]                   [Instance Encoder]
(RoBERTa-base, 125M)              (MLP, ~1M params)
    │                                      │
    ▼                                      ▼
N × d embeddings                   M × d embeddings
    │                                      │
    └──────── Cross-Attention Layers ──────┘
              (prompts attend to instances,
               instances attend to prompts)
                      │
                      ▼
              N × M score matrix
                      │
                      ▼
              Assignment layer
              (argmax / Sinkhorn / greedy)
```

**Prompt Encoder**: Shared RoBERTa-base encodes each prompt into a d-dimensional embedding. Captures prompt difficulty, topic, expected output length — all from text alone.

**Instance Encoder**: Small MLP encodes instance state features from the `/instance_stats` API: queue depth, KV cache utilization, EMA decode/prefill throughput, pending tokens, running request shapes. Same features as the current XGBoost predictor (~25 features).

**Cross-Attention**: The core innovation. Prompt embeddings attend to instance embeddings and vice versa. This captures interactions like:
- "This math prompt needs high compute → attend to A100 instances"
- "This instance has 8 running requests → long-prompt attention weight decreases"
- "Multiple similar prompts in batch → distribute across instances for cache diversity"

**Output**: N × M score matrix where score(i,j) represents the desirability of assigning prompt i to instance j. Higher = better assignment.

### 2.2 Training Formulations

We propose a progressive training strategy, starting simple and adding complexity:

#### Stage 1: Homogeneous, Single-Metric (Stepping Stone)
- **Setting**: Same model, same GPU type, optimize for throughput/E2E latency
- **Labels**: Output lengths from offline LLM runs → SJF ordering is provably optimal
- **Loss**: ListMLE (listwise ranking loss) on the optimal assignment order
- **Baseline**: LTR for local scheduling (Fu et al., NeurIPS 2024) — we extend from single-instance to multi-instance global scheduling
- **Validation**: Compare against round-robin, random, shortest-queue, power-of-two-choices

This stage validates that the cross-attention architecture can learn scheduling decisions. Labels are trivially available from offline runs.

#### Stage 2: Heterogeneous, Single-Metric
- **Setting**: Multiple model sizes × GPU types, optimize for throughput
- **Labels**: Per-(model, GPU) output lengths + per-instance latency from execution logs
- **Loss**: ListMLE on hindsight-optimal assignment (computed from actual TTFT/TPOT after completion)
- **Challenge**: Counterfactual estimation — we don't know what would have happened with alternative assignments
- **Solution**: Use XGBoost predictor to estimate counterfactual latencies, then compute hindsight-optimal assignment as ground truth ranking

#### Stage 3: Heterogeneous, Multi-Objective
- **Setting**: Full RouteBalance problem — latency + quality + cost + balance
- **Labels**: Multi-objective Pareto ranking from execution data
- **Loss Options**:
  - ListMLE on Pareto-dominance ordering
  - Decision-focused learning (SPO loss) — train to minimize scheduling regret, not prediction error
  - RL (REINFORCE) with reward = SLO attainment + quality - cost
- **Challenge**: No single "optimal" ordering when objectives conflict
- **Solution**: Scalarize with operator-specified weights (same as current Cara), then rank by scalarized score

### 2.3 Assignment Layer Options

Given the N × M score matrix, how to produce the final assignment:

| Method | Differentiable? | Optimal? | Complexity | Notes |
|--------|----------------|----------|------------|-------|
| **Per-row argmax** | No (straight-through) | No (ignores conflicts) | O(NM) | Simplest, same as current Cara |
| **Greedy sequential** | No | Approx | O(NM log N) | Current LPT approach |
| **Hungarian algorithm** | No | Yes (for linear objective) | O(N³) | Optimal bipartite matching |
| **Sinkhorn layer** | Yes | Approx | O(NM × iterations) | Differentiable relaxation of matching (Mena et al., ICLR 2018) |
| **Gumbel-Sinkhorn** | Yes (with sampling) | Stochastic | O(NM × iterations) | Enables exploration during training |

**Recommendation**: Start with per-row argmax (simplest), validate the ranking quality, then explore Sinkhorn for end-to-end differentiable training.

---

## 3. Training Data

We have rich training data already collected:

| Dataset | Size | Contents |
|---------|------|----------|
| **Latency records** | 405K | (prompt, instance_state, actual_TTFT, actual_TPOT, actual_E2E) per request |
| **Quality annotations** | 18.6K × 4 models | (prompt, model, quality_score) via LLM judges |
| **Batch logs** | From CloudLab | (batch composition, assignment, actual outcomes) per scheduling decision |

**Label generation for Stage 1** (homogeneous):
1. Take batch of N prompts from execution logs
2. Actual output lengths are known post-completion
3. SJF ordering = sort by output length ascending
4. This ordering is the ground truth ranking for ListMLE loss

**Label generation for Stage 2** (heterogeneous):
1. Take batch of N prompts with M instances
2. For the actual assignment, we know (actual_TTFT, actual_TPOT) from logs
3. For counterfactual assignments, use trained XGBoost to predict (est_TTFT, est_TPOT)
4. Compute optimal assignment by Hungarian algorithm on the estimated cost matrix
5. This assignment ordering is the ground truth for training

**Label generation for Stage 3** (multi-objective):
1. Same as Stage 2 but cost matrix = scalarized multi-objective score
2. Or use actual post-completion quality scores + latency for the realized assignment
3. Train with decision-focused loss (SPO) or RL reward

---

## 4. Experimental Design

### 4.1 Clean 2×2 Comparison

| | Heuristic Scoring | Learned Ranking (LTR) |
|--|-------------------|----------------------|
| **Per-request dispatching** | Baseline (Block-style) | Learned scoring replaces heuristic |
| **Batch scheduling** | Current Cara | Full learned scheduler |

This isolates two orthogonal contributions:
- Row comparison: batch vs per-request (scheduling structure)
- Column comparison: heuristic vs learned (scoring quality)

### 4.2 Ablation Studies

1. **Architecture ablation**:
   - MLP on concatenated features (no cross-attention)
   - Self-attention only (no cross-attention between prompts and instances)
   - Full cross-attention
   - → Measures value of prompt-instance interaction modeling

2. **Training formulation ablation**:
   - Pointwise MSE (predict latency, then rank)
   - Pairwise margin ranking loss (PARS-style)
   - Listwise ListMLE (LTR-style)
   - Decision-focused SPO loss
   - → Measures which training objective produces best scheduling decisions

3. **Assignment layer ablation**:
   - Per-row argmax
   - Greedy sequential (LPT-style)
   - Hungarian optimal
   - Sinkhorn differentiable
   - → Measures whether optimal assignment matters vs simple greedy

4. **Scale ablation**:
   - 2 instances → 18 instances
   - Batch size 4 → 32
   - → Measures scalability of learned scheduling

### 4.3 Metrics

- **Goodput** (requests/sec meeting SLO)
- **SLO attainment rate** (% requests meeting TTFT + TPOT targets)
- **Mean/P50/P99 E2E latency**
- **Quality-weighted throughput** (goodput × average quality score)
- **Scheduling overhead** (time to compute assignment, must be < 20ms)
- **Kendall's Tau** (ranking correlation with hindsight-optimal assignment)

### 4.4 Baselines

| Baseline | Type | Source |
|----------|------|--------|
| Round-robin | Heuristic | Standard |
| Shortest-queue | Heuristic | Standard |
| Power-of-two-choices | Heuristic | Block |
| LPT + scoring function | Heuristic | Cara/RouteBalance |
| LMetric (P_tokens × BS) | Heuristic | Zhang et al. 2026 |
| LTR local (SJF ranking) | Learned (local) | Fu et al. NeurIPS 2024 |
| DeepRM | Learned (RL) | Mao et al. 2016 |
| Decima-style GNN+RL | Learned (RL+GNN) | Mao et al. SIGCOMM 2019 |

---

## 5. Key Technical Challenges

### 5.1 Counterfactual Estimation
We only observe outcomes for the assignment that was actually made. For training, we need outcomes for alternative assignments. Solutions:
- **XGBoost simulator**: Use trained latency/quality predictors as approximate world model
- **Off-policy correction**: Importance sampling to reweight observed data
- **Online A/B**: Run learned scheduler alongside heuristic, collect comparative data
- **Limitation**: Simulator accuracy bounds the oracle quality. Must validate simulator MAPE before using as label generator.

### 5.2 Variable Input Sizes
Batch size N varies (4-32), instance count M varies (deployment changes). Architecture must handle variable set sizes:
- **Set transformer / attention**: Natural for variable-size inputs
- **Padding**: Pad to max size with mask — simple but wasteful
- **Graph neural network**: Represent as bipartite graph (Decima-style)

### 5.3 Scheduling Overhead
The learned scheduler must be fast enough to not become a bottleneck:
- **Target**: < 20ms per batch (current Cara: ~18ms including sidecar calls)
- **RoBERTa forward pass**: ~55ms for batch of 8 (already batched in current Cara)
- **Cross-attention + assignment**: Must add < 5ms overhead
- **Solution**: Cross-attention is lightweight (N×M is at most 32×18=576, tiny compared to language model attention). The overhead is dominated by prompt encoding, which is already required.

### 5.4 Cold Start
Learned scheduler needs data from the heuristic scheduler to train:
- **Phase 1**: Run Cara heuristic, collect (batch, assignment, outcome) logs
- **Phase 2**: Train learned scheduler offline on collected data
- **Phase 3**: Deploy learned scheduler, continue collecting data for fine-tuning
- **Fallback**: If learned scheduler degrades, revert to heuristic (always available)

---

## 6. Connection to Classical Problems

This work sits at the intersection of several well-studied areas:

### 6.1 Weighted Bipartite Matching
The assignment problem (N prompts → M instances) is a weighted bipartite matching where edge weights are the scheduling scores. The Hungarian algorithm finds the optimal solution in O(N³). Our learned approach aims to approximate this optimum efficiently while handling the dynamic, online nature of serving.

### 6.2 Learning to Rank (LTR)
The ListMLE loss (Xia et al., 2008) provides a principled way to train a model to produce rankings that correlate with ground truth. Applied to job scheduling by Fu et al. (NeurIPS 2024) for local SJF ordering. We extend to global multi-instance ranking.

### 6.3 Differentiable Combinatorial Optimization
Sinkhorn networks (Mena et al., ICLR 2018) and Gumbel-Sinkhorn provide differentiable relaxations of permutation matrices, enabling end-to-end gradient-based training through the assignment layer. SuperGlue (Sarlin et al., CVPR 2020) demonstrated this at scale for feature matching.

### 6.4 Decision-Focused Learning
SPO+ (Elmachtoub & Grigas, 2022) and subsequent work show that training predictors with a decision-aware loss (measuring scheduling regret, not prediction error) improves downstream decision quality. This directly applies to Cara: we should train quality/latency predictors to minimize scheduling suboptimality, not MAE/MAPE.

### 6.5 Learned Job Scheduling
DeepRM (Mao et al., 2016) and Decima (Mao et al., SIGCOMM 2019) showed that RL can learn scheduling policies for data processing clusters. Gavel (Narayanan et al., OSDI 2020) introduced heterogeneity-aware scheduling with the "effective throughput" abstraction. Our work extends these ideas to LLM inference serving with its unique characteristics (prefix caching, auto-regressive generation, quality-latency tradeoffs).

---

## 7. Relationship to Current Work (RouteBalance)

| Aspect | RouteBalance (current) | Learned Scheduler (this proposal) |
|--------|----------------------|----------------------------------|
| **Scoring** | Hand-designed multi-objective | Learned from execution data |
| **Assignment** | Greedy LPT | Learned ranking + Sinkhorn/greedy |
| **Predictions** | Separate models (XGBoost, RoBERTa, KNN) | Fused cross-attention model |
| **Training** | Each predictor trained independently | End-to-end for scheduling quality |
| **Adaptability** | Weight tuning per workload | Learns workload-specific patterns |
| **Interpretability** | Weights and scores are inspectable | Black-box (but ranking is interpretable) |
| **Overhead** | ~18ms per batch | Target ~20ms (dominated by prompt encoding) |

**Key insight**: RouteBalance provides the infrastructure, data, and baselines. The learned scheduler builds on top — it uses RouteBalance's execution logs for training and RouteBalance's heuristic as the primary baseline.

---

## 8. Scope and Boundaries

### In Scope
- **Global scheduler only**: Decides which instance gets which request
- **Same API boundary**: Reads `/instance_stats`, sends requests to instances
- **Offline training**: Train from execution logs, deploy as drop-in replacement for scoring function
- **Works with any serving backend**: vLLM, TGI, Ollama — instances are black boxes

### Simulation Platform: vllm-emulator (Internal)
We have an internal **vLLM emulator** (`Code/llm/vllm-emulator/`) that enables GPU-free scheduling evaluation:
- **vLLM v1 plugin architecture** — registers as a platform (like CUDA/ROCm), runs real vLLM scheduler code
- **Profile-based cost oracles** — interpolates prefill/decode latency from GPU profiles (similar to Revati)
- **Network cost emulation** — NVLink, PCIe, InfiniBand with bandwidth saturation
- **PD disaggregation support** — separate prefill/decode timing + KV transfer costs
- **Online/offline modes** — real-time blocking or virtual time (Revati-style)
- **Configurable GPU memory** — emulate any GPU type via profile pack

For Phase 3 (large-scale simulation), we can:
1. Profile each GPU type (A30, V100, P100, A100) on real hardware once
2. Create profile packs for each (prompt_tokens, batch_size) → latency mapping
3. Run N emulated instances on a single CPU machine, each with different profile pack
4. Evaluate scheduling policies at 100+ instance scale without CloudLab

This eliminates the Revati/Vidur dependency and gives us full control over the simulation platform.

### Out of Scope (Future Work)
- **Local scheduling**: How instances internally order prefill/decode (addressed by Fu et al.)
- **Co-design with vLLM**: Joint optimization of global routing + local scheduling
- **Online RL**: Real-time policy updates during serving (start with offline training)
- **KV cache-aware routing**: Explicit prefix matching (addressed by Preble, DualMap, LMetric)

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Learned scheduler doesn't beat heuristic | Medium | High | Start with homogeneous (where SJF is optimal), validate incrementally |
| Counterfactual labels are too noisy | Medium | Medium | Validate XGBoost simulator accuracy first; use conservative confidence thresholds |
| Overhead exceeds budget (>20ms) | Low | High | Cross-attention on 32×18 matrix is tiny; profile early |
| Insufficient training data | Low | Medium | 405K latency records + CloudLab batch logs; can generate more |
| Generalization across workloads | Medium | Medium | Train on mixed workloads; evaluate transfer |

---

## 10. Estimated Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Stage 1: Homogeneous LTR | 2-3 weeks | Global LTR beating heuristics on homogeneous cluster |
| Stage 2: Heterogeneous extension | 2-3 weeks | Cross-attention model on heterogeneous cluster |
| Stage 3: Multi-objective | 3-4 weeks | Full learned scheduler matching/beating Cara |
| Evaluation + paper writing | 3-4 weeks | Submission-ready paper |
| **Total** | **~3 months** | |

---

## 11. Key References

1. Fu et al., "Efficient LLM Scheduling by Learning to Rank," NeurIPS 2024 — LTR for local SJF scheduling
2. Tao et al., "PARS: Prompt-Aware Ranking Scheduler," arXiv 2025 — Pairwise ranking for reasoning LLMs
3. Mao et al., "Decima: Learning Scheduling Algorithms for Data Processing Clusters," SIGCOMM 2019 — GNN+RL for cluster scheduling
4. Mao et al., "Resource Management with Deep Reinforcement Learning," HotNets 2016 — RL for job scheduling
5. Narayanan et al., "Gavel: Heterogeneity-Aware Cluster Scheduling," OSDI 2020 — Effective throughput for mixed GPUs
6. Mena et al., "Learning Latent Permutations with Gumbel-Sinkhorn Networks," ICLR 2018 — Differentiable assignment
7. Sarlin et al., "SuperGlue: Learning Feature Matching with Graph Neural Networks," CVPR 2020 — Sinkhorn matching at scale
8. Elmachtoub & Grigas, "Smart Predict then Optimize," Management Science 2022 — Decision-focused learning
9. Zhang et al., "LMetric: Simple is Better," arXiv 2026 — Heuristic cache-aware scheduling
10. Srivatsa et al., "Preble: Efficient Distributed Prompt Scheduling," ICLR 2025 — Distributed prefix-aware scheduling
11. Georgiev & Lio, "Neural Bipartite Matching," ICML Workshop 2020 — GNN for matching
12. Alomrani et al., "Deep Policies for Online Bipartite Matching," 2022 — RL for online assignment
13. Learned Best-Effort LLM Serving, arXiv 2024 — RL router across models
14. Jaillet et al., "Online Scheduling for LLM Inference with KV Cache Constraints," arXiv 2025 — Theoretical treatment
