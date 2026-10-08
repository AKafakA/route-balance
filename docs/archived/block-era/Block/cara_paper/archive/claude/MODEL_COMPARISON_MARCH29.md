# Complete Model Comparison — March 29, 2026

All metrics on test set (3,642 examples). 14B target shown (representative).

## Length Prediction

### Bucket Classification (→ P(tokens ≤ B))

| Approach | 14B Acc | 7B Acc | 14B Adj Acc | 7B Adj Acc |
|----------|---------|--------|-------------|------------|
| Static majority | 39.2% | 31.0% | — | — |
| Online PFS | 15.1% | 10.3% | — | — |
| Online PFS + bins | 24.0% | 10.1% | — | — |
| KNN distribution | 44.5% | 34.8% | — | — |
| Per-model ModernBERT | 47.3% | 37.7% | 71.4% | 52.5% |
| **Fused ModernBERT** | **48.2%** | **38.0%** | **72.7%** | **55.6%** |
| Fused RoBERTa (512 ctx) | TBD (CSD3 running) | | | |
| Fused DeBERTa (512 ctx) | TBD (CSD3 running) | | | |
| LoRA bucket | Not feasible — output too complex for 0.5B | | | |

### Length Regression (→ E[length])

| Approach | 14B MAE | 7B MAE | 14B ρ | 7B ρ |
|----------|---------|--------|-------|------|
| Always predict median | 127.3 | 181.3 | 0.0 | 0.0 |
| KNN length | 95.2 | 158.6 | TODO | TODO |
| LoRA fused | 106.9 | 186.2 | 0.0* | 0.0* |
| ModernBERT MSE (5ep, underfitting) | — | 176.2 | — | — |
| **ModernBERT log-transform (5ep)** | **82.1** | **138.5** | **0.731** | **0.570** |
| E[length] from bucket probs | 84.9 | 143.3 | — | — |
| RoBERTa MSE (100ep) | TBD (CSD3 running) | | | |
| RoBERTa log-transform (20ep) | TBD (CSD3 running) | | | |
| ModernBERT MSE (100ep) | TBD (CSD3 running) | | | |

*LoRA length ρ=0.0 in full eval but ρ=0.364 on 100-sample test — discrepancy needs investigation.

## Quality Prediction (Judge Score)

| Approach | 14B MAE | 14B ρ | 14B Acc (10-class) | 14B Adj Acc | Best-Model Acc |
|----------|---------|-------|-------------------|-------------|----------------|
| Constant (always 0.8) | 0.191 | 0.0 | 46.3% | — | 25% (random) |
| KNN | 0.105 | TODO | TODO | TODO | **66.6%** |
| LoRA fused | 0.243 | 0.087 | 49.0% | 57.6% | — |
| **ModernBERT fused classifier** | — | TODO | **61.4%** | **72.7%** | — |

**Key finding**: LoRA judge (MAE=0.243) is **worse than constant median** (MAE=0.191). The 0.5B model cannot learn fine-grained per-model quality differences.

## Quality Prediction (Similarity)

| Approach | 14B MAE | 14B ρ | 72B MAE | 72B ρ |
|----------|---------|-------|---------|-------|
| Constant (always median) | 0.078 | 0.0 | 0.000 | 0.0 |
| LoRA fused | 0.072 | 0.473 | 0.000 | 0.000 |
| **ModernBERT fused** | **0.071** | **0.510** | 0.015 | 0.027 |

Note: 72B similarity ≈ 1.0 for all prompts (reference = itself), so all models trivially match.

## Quality Prediction (Reference Score)

| Approach | 14B MAE | 14B ρ | 7B MAE | 7B ρ |
|----------|---------|-------|--------|------|
| LoRA fused | 0.201 | 0.407 | 0.250 | 0.352 |
| **ModernBERT fused** | **0.174** | **0.538** | **0.183** | **0.581** |

## Latency Prediction

| Approach | MAPE (3B/A30) | MAPE (7B/A30) | MAPE (72B/A100) | MAPE (14B/V100) |
|----------|--------------|--------------|----------------|----------------|
| **XGBoost TTFT+TPOT+E2E** | **2.4%** | **2.1%** | **1.2%** | **5.6%** |
| Roofline | — | — | — | — |
| Static TPOT | — | — | — | — |

---

## Summary: Winner per Task

| Task | Best Approach | Runner-up | LoRA vs Constant |
|------|--------------|-----------|-----------------|
| Bucket classification | Fused ModernBERT (48.2%) | KNN distribution (44.5%) | N/A (not feasible) |
| Length regression | ModernBERT log (MAE=82, ρ=0.73) | KNN (MAE=95) | LoRA better on MAE, weak ρ |
| Judge quality | KNN (MAE=0.105, 66.6% best-model) | ModernBERT classifier (61.4% acc) | **LoRA worse than constant** |
| Similarity | ModernBERT (MAE=0.071, ρ=0.51) | LoRA (MAE=0.072, ρ=0.47) | LoRA slightly better |
| Reference score | ModernBERT (MAE=0.174, ρ=0.54) | LoRA (MAE=0.201, ρ=0.41) | LoRA better than constant |
| Latency | XGBoost (1.2-6.2% MAPE) | Roofline | — |

## Deployment Stack (Confirmed)

| Component | Model | Why |
|-----------|-------|-----|
| Budget compliance | Fused ModernBERT bucket | Best accuracy, probability distributions |
| Length E[length] | ModernBERT log-transform | Best MAE + Spearman by far |
| Quality | KNN (FAISS) | Best MAE + best-model accuracy, no GPU, sub-ms |
| Latency | XGBoost TTFT+TPOT+E2E | 1.2-6.2% MAPE |

## TODO: Missing Metrics to Recompute
1. KNN: ρ, 10-class accuracy, adj accuracy for judge/similarity/reference_score
2. ModernBERT judge classifier: ρ (convert predicted class → score, compute ρ vs GT)
3. LoRA length full eval: investigate ρ=0.0 discrepancy (full eval vs 100-sample ρ=0.364)
4. Constant baselines for reference_score
5. CSD3 running: RoBERTa + DeBERTa + ModernBERT long results
