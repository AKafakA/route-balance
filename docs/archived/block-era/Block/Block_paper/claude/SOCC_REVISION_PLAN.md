# Block Paper SoCC 2025 Revision Plan

## Overview

This document provides the comprehensive revision plan for the "Block" paper targeting SoCC 2025 submission. The paper was previously rejected from ACM Middleware 2026 with three "Weak Reject" scores. This revision addresses all reviewer concerns systematically.

**Deadlines:**
- **Feb 6, 2025**: Abstract deadline
- **Feb 13, 2025**: Paper deadline

---

## Executive Summary

**Block** is a predictive scheduling framework for multi-instance LLM serving that:
1. Predicts response length using fine-tuned RoBERTa-base (125M params)
2. Simulates latency using adapted Vidur simulator
3. Routes requests to instances with lowest predicted latency
4. Enables preemptive auto-provisioning

**Key Results (Middleware submission):**
- 48-52% E2E latency reduction
- 16-17% throughput improvement
- 64-68% preemption reduction
- 10-15% prediction error (relative ranking still accurate)
- Up to 13.8% capacity improvement under constrained configs

---

## Middleware Review Summary

### R12A (Weak Reject, Knowledgeable)
| Concern | Type | Solution |
|---------|------|----------|
| Scope hidden until §6.3 | Writing | State "commodity cluster without high-speed interconnects" in abstract/intro |
| Lacks focus/narrative | Writing | Single contribution: simulation-based predictive scheduling. Move auto-provisioning to "Extension" |
| Block-Len-Oracle preemption anomaly | Experiment | Remove Block-Len-Oracle from preemption study (explained by aggressive packing → less memory headroom) |
| Figure 7 unreadable | Figure | Redesign with subplots or table |
| Missing ablation | Experiment | Add Block-NoSim + Po2 ablation table |
| Distributed/stateless misleading | Writing | Remove framing, explain Po2 as scalability variant |
| Generalization to NVLink/InfiniBand | Writing | Add discussion paragraph |
| Sensitivity to prediction errors | Experiment | Error sensitivity heatmap (6×6 grid) |

### R12B (Weak Reject, Knowledgeable)
| Concern | Type | Solution |
|---------|------|----------|
| Novelty unclear vs Vidur | Writing | Explicit novelty statement |
| Background too long | Writing | Cut §2-3 by 30-40% |
| Block-Len-Oracle misleading | Writing | Reframe as oracle upper bound + structured output use case |
| RoBERTa fine-tuning details missing | Writing | Add 1 paragraph with training details |
| Why Llama-2 not Llama-3 | Writing | Explain: fair comparison with prior work (Vidur, Llumnix use Llama-2) |

### R12C (Weak Reject, Knowledgeable)
| Concern | Type | Solution |
|---------|------|----------|
| Novelty composition-heavy | Writing | Explicit contribution statement |
| Single model evaluation | Experiment | Llama-70B prediction accuracy on A100 cluster |
| No bursty workloads | Experiment | Burstiness study (0.25, 0.5, 1.0, 2.0) - **COMPLETED** |
| No scaling study (12→48→192) | Experiment | Po2 ablation + overhead breakdown (theoretical O(1)) |
| Capacity claims mismatch | Writing | Specify conditions for each claim |
| CPU overhead of predictors | Experiment | CPU tracking implementation |
| Snapshot consistency | Writing | Reframe existing overhead measurement as snapshot staleness |
| Simpler baseline without simulation | Experiment | Block-NoSim variant |
| Heterogeneous GPU robustness | Writing | Acknowledge as hardware limitation |

---

## Execution Plan

### Step 1: Implementation

| Task | Description | Files to Modify | Purpose |
|------|-------------|-----------------|---------|
| **1.1** Prediction error injection | Add `--length-error-pct` and `--latency-error-pct` params with Gaussian noise injection | `block/predictor/simulate_predictor.py`, `block/benchmark/benchmark_serving.py` | Error sensitivity heatmap |
| **1.2** CPU usage tracking | Add psutil monitoring + timing breakdown (simulation time, network RTT) | `block/predictor/api_server.py`, `block/predictor/simulate_predictor.py` | R12C overhead concern |
| **1.3** Block-NoSim scheduler | New scheduler variant: route to instance with minimum total unprocessed tokens (uses length prediction, no simulation) | `block/global_scheduler/api_server.py` | Ablation: isolate value of simulation |

#### 1.1 Error Injection Details
```python
# In simulate_predictor.py
def predict_with_noise(target_request, length_error_pct=0, latency_error_pct=0):
    # Inject Gaussian noise to length prediction
    if length_error_pct > 0:
        noise = np.random.normal(0, length_error_pct / 100)
        predicted_length = predicted_length * (1 + noise)

    # Inject Gaussian noise to latency prediction
    if latency_error_pct > 0:
        noise = np.random.normal(0, latency_error_pct / 100)
        predicted_latency = predicted_latency * (1 + noise)
```

#### 1.2 CPU Tracking Output
```python
{
    "cpu_percent": 12.5,           # CPU usage during prediction
    "simulation_time_ms": 45.2,    # Time spent in simulation
    "total_predict_time_ms": 52.1, # End-to-end prediction latency
    "network_overhead_ms": 6.9     # Difference = network/parsing overhead
}
```

#### 1.3 Block-NoSim Logic
```python
# For each instance, calculate:
total_unprocessed = pending_tokens + running_tokens + predicted_new_response_length

# Select instance with minimum total_unprocessed
selected_instance = min(instances, key=lambda i: i.total_unprocessed)
```

---

### Step 2: Create Experiment Scripts

| Script | Purpose | Parameters |
|--------|---------|------------|
| **2.1** `a100_40gb_profiling.sh` | **NEW: Profile Vidur on A100-40GB** | MLP, Attention, Collectives for Llama-70B |
| **2.2** `a100_llama70b_exp.sh` | Llama-70B latency prediction accuracy | Tensor parallel (4×A100-40GB), measure simulator accuracy |
| **2.3** `po2_ablation_exp.sh` | Po2 scalability ablation | N_SELECTED="2 4 6 12", QPS="24 28 30" |
| **2.4** `burstiness_exp.sh` | Burstiness study | burstiness="0.25 0.5 1.0 2.0", schedulers="Block Llumnix--" - **COMPLETED** |
| **2.5** `error_heatmap_exp.sh` | Prediction error sensitivity | 6×6 grid: length_error × latency_error (0%, 10%, 20%, 30%, 40%, 50%) |
| **2.6** `block_nosim_ablation_exp.sh` | Block-NoSim ablation | Compare Block vs Block-NoSim vs Llumnix- |

---

### Step 3: Run Experiments

| Cluster | Experiments | Est. Runs | Est. Time |
|---------|-------------|-----------|-----------|
| **A100-40GB** (d8545, 1 node, 4 GPUs) | Vidur profiling + Llama-70B prediction accuracy | ~3 QPS + profiling | ~2-3 hours |
| **A30** (12 nodes, 12 GPUs) | Po2 ablation | ~12 | ~2 hours |
| **A30** | Burstiness study | ~18 | ~3 hours |
| **A30** | Error heatmap | ~37 | ~6 hours |
| **A30** | Block-NoSim ablation | ~9 | ~1.5 hours |

**Total estimated experiment time:** ~13.5 hours

**Parallelization opportunity:** A100 and A30 experiments can run simultaneously.

#### Cluster Details

**A30 Cluster (Main Experiments):**
- Manifest: `a30_cl_manifest.xml`
- Platform: CloudLab
- Nodes: 12 × d7525 nodes
- GPU: NVIDIA A30 (24GB) per node
- CPU: 2 × 16-core AMD 7302 @ 3.00 GHz
- Memory: 128 GB ECC
- Network: 10 Gb/s

**A100-40GB Cluster (Llama-70B) - CloudLab d8545:**
- Manifest: `a100_cl_manifest.xml`
- Platform: CloudLab Wisconsin
- Node type: d8545 (only 4 nodes available)
- GPU: 4 × NVIDIA A100 SXM4 **40GB** per node (HGX A100 with NVLink)
- CPU: 2 × 24-core AMD 7413 @ 2.65 GHz
- Memory: 512 GB ECC
- Network: Mellanox ConnectX-6 100Gb
- Use: Tensor parallel (TP=4) for Llama-70B on single node

**IMPORTANT: A100-40GB Profiling Required**
The existing Vidur profiles are for A100-80GB. CloudLab d8545 has A100-40GB with different memory bandwidth (1.6 TB/s vs 2.0 TB/s). Must run profiling before experiment:
```bash
# On CloudLab d8545 node:
sh block/exp/end_to_end_exp_scripts/a100_40gb_profiling.sh
```
This generates profiles in `data/profiling/compute/a100_40gb/meta-llama/Llama-2-70b-hf/`

---

### Step 4: Analysis & Plots

| Task | Output | Figure/Table |
|------|--------|--------------|
| **4.1** Po2 tradeoff analysis | Capacity vs N, Overhead vs N | New figure: Po2 scalability curve |
| **4.2** Burstiness comparison | Block vs Llumnix- under bursty load | New figure: Burstiness study |
| **4.3** Error heatmap generation | 6×6 color grid, latency vs error levels | New figure: Error sensitivity heatmap |
| **4.4** Block-NoSim ablation | Table showing incremental value | New table: Ablation results |
| **4.5** Llama-70B accuracy | Prediction error comparison vs 7B | Add to generality study |
| **4.6** CPU overhead breakdown | Simulation time, network RTT, CPU % | New table or figure |
| **4.7** Redesign Figure 7 | Subplots, larger fonts, clearer lines | Replace existing Figure 7 |

---

### Step 5: Paper Revision

#### Structure Changes
| Task | Section | Action |
|------|---------|--------|
| **5.1** Abstract | Rewrite | State scope upfront, 15% prediction error as strength |
| **5.2** Introduction | Rewrite | Single contribution, remove auto-provisioning emphasis |
| **5.3** Background (§2) | Cut | Reduce by 30-40% |
| **5.4** Related Work (§3) | Cut | Reduce by 30-40% |
| **5.5** Design (§4) | Add | Po2 as scalability variant |
| **5.6** Evaluation (§6) | Restructure | See below |
| **5.7** Auto-provisioning | Move | To "Extension" section at end |

#### Evaluation Restructure
| Subsection | Content |
|------------|---------|
| 6.1 Setup | Keep, add Llama-70B setup |
| 6.2 Prediction Accuracy | Keep, add Llama-70B results |
| 6.3 Main Results | Block-Broadcast vs baselines (remove Block-Len-Oracle from preemption) |
| 6.4 **NEW: Ablation Study** | Block vs Block-NoSim vs Block-Len-Oracle vs Po2 table |
| 6.5 **NEW: Burstiness Study** | Block vs Llumnix- under bursty arrivals |
| 6.6 **NEW: Error Sensitivity** | Heatmap + discussion |
| 6.7 GPU Memory | Keep |
| 6.8 Overhead Analysis | **NEW: CPU tracking results** |
| 6.9 Generality | Keep, add Llama-70B |

#### New Ablation Table Structure
| Variant | Length Pred | Simulation | Capacity (QPS) | Overhead (ms) |
|---------|-------------|------------|----------------|---------------|
| Llumnix- | ❌ | ❌ | 30.7 | ~10 |
| Block-NoSim | ✅ | ❌ | TBD | ~15 |
| Block (Po2, N=2) | ✅ | ✅ | TBD | ~40 |
| Block (Broadcast) | ✅ | ✅ | 31.1 | ~80 |
| Block-Len-Oracle | ✅ (perfect) | ✅ | 32.0 | ~80 |

#### Key Framing Statements

**Novelty:**
> "Unlike prior work that uses length prediction for local batching [S³, ELIS] or offline simulation for cluster sizing [Vidur], Block is the first to combine online simulation with length prediction for real-time global scheduling decisions."

**Scope:**
> "We evaluate Block on a 12-GPU commodity cluster without high-speed interconnects, where migration-based rebalancing is impractical. Block's design is orthogonal to migration—in network-rich clusters, better initial placement would reduce migration frequency."

**Block-Len-Oracle Reframing:**
> "Block-Len-Oracle (with accurate length) serves as an oracle upper bound, showing the potential when length prediction improves. It also represents practical scenarios where output length is known, such as structured output generation with fixed schemas."

**Block-NoSim Ablation:**
> "Block-NoSim uses length prediction without simulation, routing to the instance with minimum total unprocessed tokens. Block's improvement over Block-NoSim demonstrates that simulation captures scheduling-relevant dynamics beyond token counting."

**Snapshot Staleness:**
> "The scheduling overhead (~80ms) represents the snapshot staleness—the age of instance state when dispatching. This overhead is independent of cluster size (bounded by slowest predictor) and remains below 3% of E2E latency."

#### Typo Fixes
- [ ] "came with" → "comes with" (§3)
- [ ] "and proposes" → fix grammar (§3)
- [ ] "is collection" → "is a collection" (§4.1)
- [ ] "at QPS as 32" → "at QPS=32"
- [ ] "Dynmaic Objects" → "Dynamic Objects" (Figure 3)
- [ ] Remove "distributed/stateless" misleading framing
- [ ] Full grammar pass

---

## Timeline

| Date | Milestone | Tasks |
|------|-----------|-------|
| **Jan 28** | Implementation | 1.1, 1.2, 1.3 |
| **Jan 29** | Scripts + Start Experiments | 2.1-2.5, begin 3.x |
| **Jan 30** | Complete Experiments | Finish 3.x |
| **Jan 31 - Feb 4** | Analysis + Writing | 4.x, 5.x |
| **Feb 5** | Abstract Finalization | Polish abstract |
| **Feb 6** | **Abstract Deadline** | Submit |
| **Feb 7-12** | Paper Polish | Final editing |
| **Feb 13** | **Paper Deadline** | Submit |

---

## Hardware Limitations (Acknowledged in Paper)

| Limitation | Reason | Mitigation |
|------------|--------|------------|
| No 48-192 instance scaling study | Don't have that many GPUs | Po2 + theoretical O(1) analysis |
| No heterogeneous GPU robustness | Only homogeneous A30/A100 clusters | Acknowledge as future work |
| No full Llumnix comparison | Requires NVLink/InfiniBand | Compare Llumnix- dispatcher only |

---

## Files to Create/Modify

### New Files
```
block/exp/end_to_end_exp_scripts/
├── a100_llama70b_exp.sh
├── po2_ablation_exp.sh
├── burstiness_exp.sh
├── error_heatmap_exp.sh
└── block_nosim_ablation_exp.sh

Block_paper/
├── figures/
│   ├── po2_scalability.pdf
│   ├── burstiness_study.pdf
│   ├── error_heatmap.pdf
│   └── figure7_redesigned.pdf
└── tables/
    └── ablation_table.tex
```

### Modified Files
```
block/predictor/simulate_predictor.py    # Error injection
block/predictor/api_server.py            # CPU tracking
block/global_scheduler/api_server.py     # Block-NoSim scheduler
block/benchmark/benchmark_serving.py     # Pass error params
block/exp/experiment.sh                  # Wire new params
```

---

## Success Criteria

1. **All new experiments complete** with reasonable results
2. **Ablation table** shows clear incremental value of each component
3. **Error heatmap** shows graceful degradation up to ~30% error
4. **Burstiness study** shows Block maintains advantage under bursty load
5. **Paper length** fits SoCC limit after cuts
6. **All reviewer concerns** addressed with concrete evidence or acknowledgment

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Experiments take longer than estimated | Parallelize A100/A30; prioritize heatmap last |
| Block-NoSim performs poorly | Frame as "lower bound" showing simulation value |
| Block-NoSim performs too well | Reduces novelty of simulation; focus on tail latency |
| CloudLab cluster unavailable | Have backup time slots reserved |
| Results don't support claims | Adjust claims to match evidence |

---

## Checklist

### Implementation
- [x] 1.1 Prediction error injection - **COMPLETED: `inject_noise()` in global_scheduler/api_server.py**
- [ ] 1.2 CPU usage tracking
- [x] 1.3 Block-NoSim scheduler - **DROPPED** (not needed)

### Experiment Scripts
- [x] 2.1 A100 Llama-70B script (`run_a100_llama70b.sh`)
- [x] 2.2 Po2 ablation script (`po2_ablation_exp.sh`)
- [x] 2.3 Burstiness script (`burstiness_exp.sh`)
- [x] 2.4 Error heatmap script (`error_heatmap_exp.sh`) - **Uses experiment.sh with error params**
- [x] 2.5 Block-NoSim ablation script (`block_nosim_ablation_exp.sh`)

### Experiments
- [x] 3.1 Llama-70B prediction accuracy (A100) - **COMPLETED: 16.6% MAE**
- [x] 3.2 Po2 ablation (A30) - **COMPLETED: N=2 achieves 97.7% capacity with 6x less overhead**
- [x] 3.3 Burstiness study (A30) - **COMPLETED: Block comparable to Llumnix-- (±1%)**
- [x] 3.4 Error heatmap (A30) - **COMPLETED: Block robust to 100% error (~2-3% degradation)**
- [x] 3.5 Block-NoSim ablation - **DROPPED** (not needed)

### Analysis & Plots
- [x] 4.1 Po2 tradeoff figure - **COMPLETED: po2_latency_comparison.pdf, po2_improvement.pdf**
- [x] 4.2 Burstiness comparison figure - **COMPLETED: burstiness_bar_comparison.pdf, burstiness_line_comparison.pdf**
- [x] 4.3 Error heatmap figure - **COMPLETED: heatmap_with_baseline.pdf, comparison_bar.pdf**
- [x] 4.4 Block-NoSim ablation table - **DROPPED** (not needed)
- [ ] 4.5 Llama-70B accuracy results
- [ ] 4.6 CPU overhead breakdown
- [ ] 4.7 Redesigned Figure 7

### Paper Revision
- [ ] 5.1 Rewrite abstract
- [ ] 5.2 Rewrite introduction
- [ ] 5.3 Cut background (30-40%)
- [ ] 5.4 Cut related work (30-40%)
- [ ] 5.5 Add Po2 to design section
- [ ] 5.6 Restructure evaluation
- [ ] 5.7 Add ablation study subsection
- [ ] 5.8 Add burstiness study subsection
- [ ] 5.9 Add error sensitivity subsection
- [ ] 5.10 Add overhead analysis subsection
- [ ] 5.11 Move auto-provisioning to extension
- [ ] 5.12 Fix all figures
- [ ] 5.13 Fix all typos
- [ ] 5.14 Full grammar pass

---

*Last updated: Feb 1, 2026*
*Document location: Block_paper/claude/SOCC_REVISION_PLAN.md*
