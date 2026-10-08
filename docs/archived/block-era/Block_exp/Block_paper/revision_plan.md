# Block Paper SoCC 2025 Revision Context

## Overview

This document provides context for the SoCC 2025 submission of the "Block" paper, a predictive scheduler for multi-instance LLM serving. The paper was previously rejected from ACM Middleware 2026 with three "Weak Reject" scores. This revision addresses reviewer concerns while targeting SoCC.

**Current Status**: SoCC revision complete. Five new experimental sections added. Mock review assessment: **Weak Accept (Borderline)**.

---

## Paper Summary

**Block** is a scheduling framework for LLM inference that:
1. Predicts response length using a fine-tuned RoBERTa regression model
2. Uses simulation (adapted from Vidur) to predict end-to-end latency
3. Routes requests to instances with lowest predicted latency
4. Achieves better load balancing and reduced preemptions vs heuristic baselines

**Key Results (SoCC Submission):**
- 13.8% capacity improvement (under suboptimal bs=24 configuration)
- 1.3% capacity improvement (optimal configuration, 31.1 vs 30.7 QPS)
- 8.03% P99 latency reduction (Block* at QPS=32)
- Robust to 100% prediction noise (≤3% degradation)
- Po2 variant achieves 97.7% capacity with 54% overhead reduction

---

## SoCC Revision: Completed Sections

### §6.7 `socc_llumnix_comparison.tex` - Comparison with Migration-based Scheduling

**Purpose**: Address concerns about missing Llumnix comparison

**Key Content**:
- Explains A30 cluster failure (single-GPU-per-host, 25Gbps inter-node NICs inadequate for KV-cache migration)
- Documents A100 cluster setup (2-node, 4 GPUs/node with NVLink)
- Shows Block ≈ Llumnix (scheduling-only) on A100 hardware
- Highlights chunked prefill incompatibility with Llumnix migration (assertion error)
- Llumnix+migration degrades 12× at high QPS due to disabled chunked prefill

**Key Finding**: Block achieves parity with Llumnix when both use optimal configurations. Chunked prefill is more critical than migration for high-load performance.

### §6.8 `socc_po2_ablation.tex` - Scalability: Power-of-Two Choices

**Purpose**: Address scalability concerns (O(N) message cost)

**Key Content**:
- Clarifies: O(1) scheduling latency, O(N) message cost
- Po2 achieves 97.7% capacity (30.2 vs 30.9 QPS)
- Po2 reduces overhead by 54% (55.4ms vs 120.2ms)
- Counterintuitive: Po2 has better latency at high QPS due to:
  1. Reduced scheduling overhead
  2. Reduced scheduling conflicts (race condition mitigation via randomness)
- Variance analysis: CoV identical across systems (~73%)

**Key Finding**: Po2 is a practical scaling strategy with minimal capacity loss.

### §6.9 `socc_burstiness.tex` - Robustness to Bursty Workloads

**Purpose**: Address R12C concern about bursty/heterogeneous workloads

**Key Content**:
- Gamma distribution arrivals: α = 0.25 (bursty) to 2.0 (regular)
- All differences within ±1% for mean latency, ±1.1% for P99
- Block slightly worse under high burstiness (+0.9%), slightly better under regular arrivals (-1.1%)
- Connected to Po2 analysis: scheduling overhead increases conflict likelihood during bursts

**Key Finding**: Block maintains comparable performance to Llumnix-- across all burstiness levels.

### §6.10 `socc_error_heatmap.tex` - Sensitivity to Prediction Errors

**Purpose**: Quantify robustness to prediction inaccuracies

**Key Content**:
- 4×4 grid: 0%, 25%, 50%, 100% Gaussian noise on length and latency predictions
- Clamping at [0.1v, 5.0v] (18% samples clamped at 100% error)
- Maximum degradation: 2.2% mean, 3.4% P99 (at 50%/50% error)
- Distinction between:
  - **Length errors**: Constant noise (same to all instances) → rankings preserved
  - **Latency errors**: Independent noise per instance → robustness from large underlying differences + power-of-choices effect
- References auto-extension mechanism for under-estimation

**Key Finding**: Block is remarkably robust to prediction errors. Even 100% noise causes only 1-3% degradation.

### §6.11 `socc_cpu_overhead.tex` - CPU Overhead Analysis

**Purpose**: Address R12C concern about CPU cost and interference

**Key Content**:
- Per-predictor CPU: 21-32% of single core
- 16 predictors × 25% ≈ 4 cores (12.5% of 32-core node)
- Memory: ~1.8GB per predictor (29GB total for 16 predictors)
- Llumnix-- has negligible overhead (no simulation)
- Scalability: Overhead scales with predictors per node, not cluster size
- References Po2 for constrained environments

**Key Finding**: CPU overhead is bounded and does not become a bottleneck.

---

## Mock Review Assessment: Remaining Issues

### High Priority (Should Address Before Submission)

| Issue | Description | Suggested Fix |
|-------|-------------|---------------|
| **W1. Marginal gains** | 1.3% capacity gain in optimal config is within noise | Emphasize gains under suboptimal configs are the realistic scenario; add statistical significance tests |
| **W2. Inconsistent configs** | Different configs across sections (A30 vs A100, bs=48 vs bs=96) | Add table summarizing all experimental configurations |
| **W4. Block*/Block disconnect** | If noise doesn't matter (§6.10), why does Block* outperform Block (§6.3)? | Add paragraph reconciling these findings |
| **W5. Statistical rigor** | No confidence intervals, single runs | Add variance from multiple runs or bootstrap CIs |
| **M1. Terminology** | "Llumnix-" vs "Llumnix--" inconsistent | Standardize to "Llumnix--" throughout main.tex |

### Medium Priority (Nice to Have)

| Issue | Description | Suggested Fix |
|-------|-------------|---------------|
| **W3. Incomplete A100** | No Block* or capacity numbers on A100 | Run additional A100 experiments if time permits |
| **W6. Missing comparisons** | No TetriServe, ServerlessLLM comparison | Add related work discussion, explain why not compared |
| **M2. Forward reference** | "appropriate multi-GPU hardware" undersells issue | Rephrase to acknowledge A30 is primary testbed |
| **M4. Section integration** | New sections feel appended | Consider reorganizing (Po2→scalability, error→accuracy) |
| **M5. Abstract accuracy** | "8.03% P99 reduction" is Block*, not Block | Clarify which variant achieves which gains |

### Low Priority (Polish)

| Issue | Description | Suggested Fix |
|-------|-------------|---------------|
| **M3. Figure quality** | Heatmap may not reproduce in grayscale | Test grayscale printing, add patterns if needed |
| Notation | Block*/Block naming could be clearer | Consider Block-Oracle/Block-Predicted |

---

## Questions to Address in Paper

1. **Why does Block* outperform Block if Block is robust to prediction errors?**
   - Suggested answer: Error sensitivity experiment shows *relative* rankings are preserved (§6.10), but *absolute* latency still matters. Block* achieves tighter predictions → better absolute placements, even if Block's rankings are mostly correct.

2. **Why didn't A100 cluster run full evaluation?**
   - Suggested answer: A100 cluster (2-node, 8-GPU) was provisioned specifically for Llumnix comparison. Main evaluation on A30 (12-node, 12-GPU) provides more instances for load balancing evaluation.

3. **What is Llumnix--'s actual scheduling overhead?**
   - Need to measure: Llumnix-- also queries all instances for metrics. Add comparison in §6.11.

---

## Middleware Reviewer Concerns: Status

### R12A (Weak Reject)

| Concern | Status | Notes |
|---------|--------|-------|
| Scope hidden | ✅ Fixed | A30 limitation stated in §6.3, detailed in §6.7 |
| Lacks focus | ✅ Fixed | Auto-provisioning de-emphasized |
| Block* preemption anomaly | ✅ Fixed | Removed from preemption study |
| Figure 7 unreadable | ⚠️ Partial | Improved but could use subplots |
| Missing ablation | ✅ Fixed | §6.8 Po2 ablation added |
| Distributed/stateless misleading | ✅ Fixed | Framing clarified |

### R12B (Weak Reject)

| Concern | Status | Notes |
|---------|--------|-------|
| Novelty unclear | ✅ Fixed | Explicit statement in intro |
| Background too long | ⚠️ Partial | Could still trim §2-3 |
| Block* misleading | ✅ Fixed | Reframed as oracle upper bound |
| RoBERTa details missing | ✅ Fixed | Details in §6.2 |
| Why Llama-2 | ✅ Fixed | Explained in §6.1 |

### R12C (Weak Reject)

| Concern | Status | Notes |
|---------|--------|-------|
| Novelty composition-heavy | ⚠️ Remains | Inherent to approach |
| Single model evaluation | ✅ Fixed | Qwen-2 added |
| No bursty workloads | ✅ Fixed | §6.9 burstiness study |
| No scaling study | ✅ Fixed | §6.8 Po2 ablation |
| Capacity claims mismatch | ⚠️ Partial | Need clearer specification |
| CPU overhead unknown | ✅ Fixed | §6.11 CPU analysis |
| Snapshot consistency | ⚠️ Missing | Need 1 paragraph |
| Simpler baseline | ✅ Fixed | INFaaS++ positioned as simpler signal |

---

## Writing Tasks Checklist (Updated)

### Completed ✅

- [x] Add Po2 as scalability variant (§6.8)
- [x] Burstiness study (§6.9)
- [x] Prediction error sensitivity heatmap (§6.10)
- [x] CPU overhead measurement (§6.11)
- [x] Llumnix comparison on A100 (§6.7)
- [x] State evaluation scope in intro
- [x] Llumnix positioning paragraph
- [x] Chunked prefill incompatibility documented
- [x] A100 cluster setup in Testbed paragraph (§6.1)
- [x] Llama-70B prediction accuracy (MAE 16.6%) in §6.2
- [x] Removed duplicate tables from burstiness/error sections
- [x] Condensed all SoCC revision sections

### Remaining 🔲

- [ ] Standardize "Llumnix--" terminology in main.tex
- [ ] Reconcile Block*/Block vs error-robustness findings
- [ ] Add statistical significance discussion
- [ ] Clarify abstract claims (which variant, which config)
- [ ] Add snapshot consistency paragraph
- [ ] Trim §2-3 Background further (optional)
- [ ] Add TetriServe/ServerlessLLM discussion in related work

### Figure Improvements 🔲

- [ ] Test heatmap in grayscale
- [ ] Consider reorganizing new sections for better flow

---

## Key Talking Points for Rebuttal

1. **"Marginal gains in optimal config"**: The 13.8% gain under suboptimal configs (bs=24) represents realistic deployment scenarios where configurations are not perfectly tuned. Block provides insurance against misconfiguration.

2. **"Why not compare with TetriServe?"**: TetriServe uses length prediction only for filtering (reject instances without sufficient memory), not for latency-based routing. Block's simulation provides richer information.

3. **"Different experimental configs"**: A30 cluster provides more instances (12) for load balancing evaluation; A100 cluster enables Llumnix migration comparison. Different configs test different aspects.

4. **"Block is noise-robust, so why does length accuracy matter?"**: Robustness to noise ≠ insensitivity to accuracy. Block tolerates noise without catastrophic failure, but better predictions still yield better absolute placements.

---

## Probability Assessment (Updated)

**55-65%** acceptance probability (improved from 50-60%)

| Strength | Remaining Risk |
|----------|----------------|
| Comprehensive new experiments | Marginal gains in optimal config |
| Po2 addresses scalability | Block*/Block disconnect unexplained |
| Error robustness is strong | No statistical significance tests |
| Chunked prefill insight is novel | Inconsistent configs across sections |
| CPU overhead is reasonable | "Composition-heavy" perception |

---

## File Structure

```
Block_paper/
├── main.tex                      # Main paper file
├── socc_llumnix_comparison.tex   # §6.7 - Llumnix comparison
├── socc_po2_ablation.tex         # §6.8 - Power-of-Two ablation
├── socc_burstiness.tex           # §6.9 - Burstiness study
├── socc_error_heatmap.tex        # §6.10 - Error sensitivity
├── socc_cpu_overhead.tex         # §6.11 - CPU overhead
├── figure/SoCC_revision/         # New figures
│   ├── llumnix_comparison.png
│   ├── po2_comparison.png
│   ├── burstiness_lines.png
│   ├── prediction_error_heatmap.png
│   └── cpu_overhead.png
├── reference.bib
└── revision_plan.md              # This file
```

---

## Quick Reference: Section Labels

| Section | Label | Title |
|---------|-------|-------|
| §6.7 | `\label{section:llumnix_comparison}` | Comparison with Migration-based Scheduling |
| §6.8 | `\label{section:po2_ablation}` | Scalability: Power-of-Two Choices |
| §6.9 | `\label{section:burstiness}` | Robustness to Bursty Workloads |
| §6.10 | `\label{section:error_sensitivity}` | Sensitivity to Prediction Errors |
| §6.11 | `\label{section:cpu_overhead}` | CPU Overhead Analysis |

---

## Cross-References Between Sections

- §6.8 (Po2) ← referenced by §6.9 (burstiness, scheduling conflicts)
- §6.8 (Po2) ← referenced by §6.11 (CPU overhead, constrained environments)
- §6.2 (Prediction Accuracy) ← referenced by §6.10 (error sensitivity, constant noise)
- §4.1 (Model Instance) ← referenced by §6.10 (auto-extension mechanism)

---

## Changelog

| Date | Changes |
|------|---------|
| Jan 2025 | Initial revision plan from Middleware rejection |
| Feb 2025 | SoCC revision: Added 5 new experimental sections |
| Feb 2025 | Mock review: Identified remaining issues |
