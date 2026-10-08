# Model Training Plan — March 29, 2026

## Current Model Inventory

### Length Bucket Classification (→ P(tokens ≤ B))

| # | Approach | Status | Accuracy (14B/7B) | Notes |
|---|----------|--------|-------------------|-------|
| B1 | Static majority | ✅ Evaluated | 39%/31% | Baseline |
| B2 | Online PFS | ✅ Evaluated | 15%/10% | Sliding window median |
| B3 | Online PFS + bins | ✅ Evaluated | 24%/10% | Conditioned on input length |
| B4 | KNN distribution | ✅ Evaluated | 45%/35% | k=10, sentence embeddings |
| B5 | Per-model ModernBERT bucket | ✅ Trained (Vast) | 47%/38% | 4 separate models |
| B6 | **Fused ModernBERT bucket** | ✅ **Deployed** | **48%/38%** | 1 shared encoder, 4 heads |
| B7 | Qwen LoRA fused bucket | **MISSING** | — | Need to train |

### Length Regression (→ E[length] for cost/LPT/E2E scoring)

| # | Approach | Status | MAE (14B/7B) | MAPE | Epochs | Converged? |
|---|----------|--------|-------------|------|--------|------------|
| R1 | ModernBERT-base MSE | ✅ 5ep | —/176 | 187% | 5 | **NO** — still dropping |
| R2 | ModernBERT-base sMAPE | ✅ 5ep | —/168 | 164% | 5 | **NO** |
| R3 | ModernBERT-base Huber | ✅ 5ep | —/172 | 176% | 5 | **NO** |
| R4 | **ModernBERT-base log-transform** | ✅ **Deployed** | 82/139 | 85%/126% | 5 | **YES** — plateau at ep3 |
| R5 | ModernBERT-large log-transform | ✅ 5ep | —/135 | 135% | 5 | Likely yes |
| R6 | RoBERTa-base MSE | **BLOCKED** | — | — | 0 | transformers 5.3 bug |
| R7 | RoBERTa-base log-transform | **BLOCKED** | — | — | 0 | Same bug |
| R8 | Qwen LoRA fused regression | ✅ Trained (Vast) | 107/186 | 62%/66% | 3 | YES — flat at ep2 |
| R9 | E[length] from bucket midpoints | ✅ Available | 85/143 | — | — | Derived from B6 |
| R10 | DeBERTaV3-base MSE | **QUEUED** | — | — | 0 | CSD3 |
| R11 | DeBERTaV3-base log-transform | **QUEUED** | — | — | 0 | CSD3 |

### Quality Prediction

| # | Approach | Status | Best-Model Acc | MAE | Notes |
|---|----------|--------|---------------|-----|-------|
| Q1 | **KNN (FAISS)** | ✅ **Deployed** | **64%** | 0.08-0.11 | Sub-ms, no GPU |
| Q2 | Fused ModernBERT judge classifier | ✅ Trained (CSD3) | 58-64% | — | 10-class |
| Q3 | Qwen LoRA fused judge | ✅ Trained (Vast) | — | 0.22, Spearman=0.06-0.19 | Poor |
| Q4 | Qwen LoRA fused similarity | ✅ Trained (Vast) | — | 0.07, Spearman=0.45-0.49 | 72B broken (ref=self) |
| Q5 | Qwen LoRA fused reference_score | ✅ Trained (Vast) | — | 0.20, Spearman=0.35-0.41 | Decent |
| Q6 | MLP | ✅ Trained | — | 0.09-0.12 | |

### Latency Prediction

| # | Approach | Status | MAPE | Notes |
|---|----------|--------|------|-------|
| L1 | **XGBoost TTFT+TPOT+E2E** | ✅ **Deployed** | **1.2-6.2%** | Per-instance models |
| L2 | Roofline analytical | ✅ Trained | — | No contention modeling |
| L3 | Static TPOT lookup | ✅ Available | — | Median from calibration |
| L4 | LSTM | ✅ Trained | — | Off-by-one bug, needs re-run |

---

## What's Missing for the Paper

### Paper Claims to Verify

The paper (Table 3) claims regression results for "RoBERTa-base, ModernBERT-base, ModernBERT-large" and "50-130 epochs for regression convergence." Current state:
- **RoBERTa**: Never trained (transformers bug). **MUST train** to back the claim.
- **50-130 epochs**: MSE models only ran 5 epochs. **MUST run long** to either confirm or remove the claim.
- **ModernBERT-large**: Only log-transform was trained. Need MSE for complete table.

### Priority Training Jobs

#### Priority 1: Back Paper Claims (MUST DO)

| Job | Script | Encoder | Loss | Target | Epochs | Est. Time | Machine |
|-----|--------|---------|------|--------|--------|-----------|---------|
| J1 | train_bert_predictor.py | RoBERTa-base | MSE | length (7B) | 100 + early stop | ~10h | CSD3 |
| J2 | train_bert_predictor.py | RoBERTa-base | log-transform* | length (7B) | 20 | ~2h | CSD3 |
| J3 | train_bert_predictor.py | ModernBERT-base | MSE | length (7B) | 100 + early stop | ~10h | CSD3 |
| J4 | train_bert_predictor.py | ModernBERT-base | Huber | length (7B) | 100 + early stop | ~10h | CSD3 |
| J5 | train_bert_predictor.py | ModernBERT-base | sMAPE | length (7B) | 100 + early stop | ~10h | CSD3 |

*RoBERTa needs `transformers==4.50.3` venv. Log-transform converges fast so 20ep is enough.

**Note on train_bert_predictor.py**: This is the **per-model** trainer (not fused). It trains one (encoder, target_model) pair at a time. For the paper table we only need 7B target (hardest case). The fused trainer (`train_fused_bert_predictor.py`) trains all 4 models at once.

#### Priority 2: Missing Ablation Models

| Job | Script | Approach | Target | Epochs | Est. Time | Machine |
|-----|--------|----------|--------|--------|-----------|---------|
| J6 | train_fused_bert_predictor.py | Fused RoBERTa-base | length_bucket | 5 | ~1h | CSD3 |
| J7 | train_fused_bert_predictor.py | Fused RoBERTa-base | length (log) | 10 | ~2h | CSD3 |
| J8 | train_llm_predictor.py | Qwen LoRA fused | length_bucket | 3 | ~8h | Vast |
| J9 | train_llm_predictor.py | Qwen LoRA per-model | judge | 3 | ~2h | Vast |
| J10 | train_llm_predictor.py | Qwen LoRA per-model | reference_score | 3 | ~2h | Vast |
| J11 | train_llm_predictor.py | Qwen LoRA per-model | similarity | 3 | ~2h | Vast |

#### Priority 3: Nice-to-Have Ablation

| Job | Script | Approach | Target | Epochs | Est. Time | Machine |
|-----|--------|----------|--------|--------|-----------|---------|
| J12 | train_fused_bert_predictor.py | Fused DeBERTaV3-base | length_bucket | 5 | ~1h | CSD3 |
| J13 | train_fused_bert_predictor.py | Fused DeBERTaV3-base | length (log) | 10 | ~2h | CSD3 |
| J14 | train_bert_predictor.py | ModernBERT-large | MSE | length (7B) | 100 + early stop | ~15h | CSD3 |
| J15 | train_fused_bert_predictor.py | Fused ModernBERT-base | length (MSE) | 100 + early stop | ~12h | CSD3 |

---

## Execution Plan

### Vast.ai (24h remaining, RTX 8000 45GB)

**Total estimated: ~16h** — fits within 24h.

```bash
# J8: Qwen LoRA fused bucket classification (NEW — not trained yet)
python3 -m block.predictor.cara.offline_training.train_llm_predictor \
    --target length_bucket --mode fused --epochs 3 \
    --output-dir models/cara/baselines/qwen05b_lora/fused_bucket

# J9-J11: Qwen LoRA per-model (judge, reference_score, similarity)
for TARGET in judge reference_score similarity; do
    python3 -m block.predictor.cara.offline_training.train_llm_predictor \
        --target $TARGET --mode per_model --epochs 3 \
        --output-dir models/cara/baselines/qwen05b_lora/per_model_$TARGET
done
```

### CSD3 A100 (Q2 budget starts April 1)

**Phase A: INTR test jobs** (verify runnable, ~30 min each)

```bash
# Test J1: RoBERTa MSE (needs transformers==4.50.3)
# First create venv:
#   python3 -m venv /rds/user/wd312/hpc-work/venv_roberta
#   source /rds/user/wd312/hpc-work/venv_roberta/bin/activate
#   pip install transformers==4.50.3 torch accelerate datasets

python3 -m block.predictor.cara.offline_training.train_bert_predictor \
    --regression-model-name roberta-base \
    --loss-type mse \
    --target-model "Qwen/Qwen2.5-7B" \
    --epochs 3 \
    --output-dir models/cara/encoder_length/roberta_mse_test

# Test J3: ModernBERT MSE long
python3 -m block.predictor.cara.offline_training.train_bert_predictor \
    --regression-model-name answerdotai/ModernBERT-base \
    --loss-type mse \
    --target-model "Qwen/Qwen2.5-7B" \
    --epochs 3 \
    --output-dir models/cara/encoder_length/modernbert_mse_test

# Test J6: Fused RoBERTa bucket
python3 -m block.predictor.cara.offline_training.train_fused_bert_predictor \
    --encoder-name roberta-base \
    --target length_bucket \
    --epochs 1 \
    --output-dir models/cara/baselines/roberta_fused_bucket_test
```

**Phase B: Full SLURM jobs** (after INTR verification)

| SLURM Job | Contains | GPU-hours | Priority |
|-----------|----------|-----------|----------|
| `csd3_regression_long.sh` | J1 + J3 + J4 + J5 (sequential) | ~40h | **P1** |
| `csd3_roberta_fused.sh` | J2 + J6 + J7 (sequential) | ~5h | **P1** |
| `csd3_deberta_fused.sh` | J12 + J13 (sequential) | ~3h | P3 |
| `csd3_modernbert_long.sh` | J14 + J15 (sequential) | ~27h | P3 |

Total CSD3 budget needed: **~75 GPU-hours** (P1: ~45h, P3: ~30h)
Q2 budget: 3,000 GPU-hrs — plenty.

---

## Deployment Model Selection (Final)

No changes from current — all trained models confirm the same winners:

| Component | Deployed Model | Backup for Comparison |
|-----------|---------------|----------------------|
| Budget compliance | Fused ModernBERT bucket (B6) | KNN distribution (B4) |
| Length regression | ModernBERT log-transform (R4) | E[length] from buckets (R9) |
| Quality | KNN FAISS (Q1) | Fused ModernBERT judge (Q2) |
| Latency | XGBoost TTFT+TPOT+E2E (L1) | Roofline (L2) |

The remaining training (J1-J15) is for **paper tables only** — it won't change deployment.

---

## Paper Table Mapping

### Table 3: Regression Comparison (§5.2)
Need: J1 (RoBERTa MSE), J2 (RoBERTa log), J3-J5 (ModernBERT MSE/Huber/sMAPE long runs)
Currently: Only ModernBERT 5-epoch results. **Must have RoBERTa before submission.**

### Table 5: Bucket Classification (§5.2)
Have: B1-B6 all evaluated. J6 (RoBERTa bucket) and J8 (LoRA bucket) add ablation rows.

### Table 9: Judge Classification (§5.4)
Have: Q2 (fused ModernBERT), Q6 (MLP). J9 (LoRA per-model judge) adds comparison.

### Length Regression Approaches in Paper
Need to update paper to mention: R4 (log-transform), R8 (LoRA regression), R9 (bucket expectation) as three distinct E[length] methods with different trade-offs.
