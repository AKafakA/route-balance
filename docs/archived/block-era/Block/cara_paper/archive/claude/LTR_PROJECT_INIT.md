# Learned Global LLM Scheduling (LTR) — Project Initialization

**Created**: April 1, 2026
**Context**: Brainstormed during Cara/RouteBalance development session
**Full proposal**: `cara_paper/claude/PROPOSAL_LEARNED_GLOBAL_SCHEDULING.md`
**Task**: #22 in current session

---

## 1. What This Project Is

Replace hand-designed LLM batch scheduling (LPT + multi-objective scoring) with a learned ranking policy. Uses Learning-to-Rank (LTR) with cross-attention architecture to rank (prompt, instance) pairs across a heterogeneous cluster.

## 2. Why It Matters

- RouteBalance (Cara) uses handcrafted scoring: `w_lat * lat + w_cost * cost - w_qual * qual + w_bal * bal`
- Requires weight tuning per workload
- Can't capture complex prompt-instance interactions
- Greedy LPT is an approximation of optimal assignment
- A learned scheduler could outperform heuristics and adapt to workload shifts

## 3. Relationship to Existing Projects

| Project | Role | Status |
|---------|------|--------|
| **Block** | Phase 1 infrastructure — homogeneous 12×A30 cluster, Po2 baseline | SoCC under review |
| **RouteBalance/Cara** | Phase 2 infrastructure — heterogeneous 18-node cluster, batch scheduling baseline | EuroSys target |
| **vllm-emulator** | Phase 3 simulation — GPU-free evaluation at 100+ node scale | Needs revival (`~/Code/llm/vllm-emulator/`) |

The LTR paper builds on ALL three: uses Block/Cara's clusters for real evaluation, heuristic schedulers as baselines, and vllm-emulator for large-scale simulation.

## 4. Three-Phase Plan

### Phase 1: Homogeneous Single-Metric
- **Cluster**: Block's 12×A30
- **Setting**: Same model, same GPU, optimize E2E latency
- **Labels**: Output lengths from offline runs → SJF is optimal
- **Loss**: ListMLE (listwise ranking)
- **Baselines**: Round-robin, shortest-queue, Po2 (Block), LTR-local (Fu et al. NeurIPS'24)
- **Contribution**: First global LTR for LLM serving (extending Fu et al. from local to global)

### Phase 2: Heterogeneous Multi-Objective
- **Cluster**: Cara's 18-node mixed (A30/V100/P100/A100 × Qwen 3B/7B/14B/72B)
- **Setting**: Multi-model, multi-GPU, optimize latency + quality + cost
- **Labels**: Counterfactual estimation via XGBoost predictor
- **Architecture**: Cross-attention (prompts attend to instances)
- **Baselines**: All Phase 1 + Cara heuristic + SLOs-Serve/PolyServe/QLM
- **Contribution**: Cross-attention captures prompt-instance interactions

### Phase 3: Large-Scale Simulation
- **Platform**: vllm-emulator (internal, `~/Code/llm/vllm-emulator/`)
- **Scale**: 100-1000 emulated instances
- **Validation**: Compare emulated results with Phase 1/2 real results
- **Contribution**: Scalability analysis beyond real cluster sizes

## 5. Key Technical Decisions

### Architecture
```
Prompts → RoBERTa encoder → N embeddings ──┐
                                            ├── Cross-Attention → N×M scores → Assignment
Instance stats → MLP encoder → M embeddings ┘
```

### Training Labels
| Phase | Label source | Optimality |
|-------|-------------|-----------|
| 1 (homogeneous) | Output lengths → SJF ordering | Provably optimal |
| 2 (heterogeneous) | XGBoost counterfactual estimation | Approximate |
| 3 (simulation) | Emulator ground truth | Exact (within simulator) |

### Assignment Layer Options
- Per-row argmax (simplest, start here)
- Greedy sequential (LPT-style)
- Hungarian algorithm (optimal for linear objective)
- Sinkhorn differentiable (end-to-end training)

### Loss Functions to Compare
- ListMLE (listwise, used by Fu et al.)
- Pairwise margin ranking (used by PARS)
- Decision-focused SPO loss (Elmachtoub & Grigas)
- REINFORCE with SLO attainment reward

## 6. Clean 2×2 Experiment Design

| | Heuristic Scoring | Learned (LTR) |
|--|-------------------|---------------|
| **Per-request** (Block) | Baseline | Learned scoring |
| **Batched** (Cara) | Cara heuristic | Full learned scheduler |

Isolates: batch vs per-request (scheduling structure) AND heuristic vs learned (scoring quality).

## 7. Data Available

| Dataset | Size | Use |
|---------|------|-----|
| Latency records | 405K | Training: (prompt, instance_state, actual_TTFT/TPOT) |
| Quality annotations | 18.6K × 4 models | Quality prediction features |
| Batch logs | From CloudLab | (batch composition, assignment, outcomes) |
| vllm-emulator profiles | TBD | GPU-specific latency profiles |

## 8. Key References

1. Fu et al., "Efficient LLM Scheduling by Learning to Rank," **NeurIPS 2024** — LTR for local SJF
2. Tao et al., "PARS: Prompt-Aware Ranking Scheduler," arXiv 2025 — Pairwise ranking
3. Mao et al., "Decima," **SIGCOMM 2019** — GNN+RL for cluster scheduling
4. Mao et al., "DeepRM," HotNets 2016 — RL for job scheduling
5. Narayanan et al., "Gavel," **OSDI 2020** — Heterogeneity-aware scheduling
6. Mena et al., "Gumbel-Sinkhorn Networks," **ICLR 2018** — Differentiable assignment
7. Sarlin et al., "SuperGlue," **CVPR 2020** — Sinkhorn matching at scale
8. Elmachtoub & Grigas, "SPO," Management Science 2022 — Decision-focused learning
9. Georgiev & Lio, "Neural Bipartite Matching," ICML Workshop 2020
10. Zhang et al., "LMetric," arXiv 2026 — Heuristic cache-aware scheduling
11. Srivatsa et al., "Preble," **ICLR 2025** — Distributed prefix-aware scheduling

## 9. Potential Venues

- **MLSys** (primary) — ML for systems, perfect fit
- **EuroSys** — if strong systems evaluation
- **NeurIPS** (ML-for-Systems track) — if strong ML contribution
- **NSDI** — if strong networking/distributed angle

## 10. Also Serves As

- **Backup novelty for Block** — if SoCC rejects for "limited novelty," add global LTR to Block
- **Backup novelty for Cara** — if reviewers say "incremental combination," add learned scheduling comparison

## 11. Risks

| Risk | Mitigation |
|------|-----------|
| Learned doesn't beat heuristic | Start with homogeneous where SJF is optimal |
| Counterfactual labels noisy | Validate XGBoost simulator accuracy first |
| Overhead >20ms | Cross-attention on 32×18 is tiny |
| vllm-emulator not ready | Phase 1-2 don't need it (real clusters) |

## 12. Next Steps (for new session)

1. Review full proposal: `PROPOSAL_LEARNED_GLOBAL_SCHEDULING.md`
2. Revival vllm-emulator: `~/Code/llm/vllm-emulator/` (separate session)
3. Implement Phase 1 ListMLE training on Block's homogeneous data
4. Quick win: just replace Cara's `score()` function with a small MLP trained on execution logs
