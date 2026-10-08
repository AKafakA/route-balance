# Final Training Plan — March 29, 2026

## Complete Model Inventory

### A. Bucket Classification (→ P(tokens ≤ B) for budget compliance)

| ID | Approach | Encoder | Context | Epochs | Status | 7B Acc | 14B Acc |
|----|----------|---------|---------|--------|--------|--------|---------|
| B1 | Static majority | — | — | — | ✅ Done | 31.0% | 39.2% |
| B2 | Online PFS | — | — | — | ✅ Done | 10.3% | 15.1% |
| B3 | Online PFS + bins | — | — | — | ✅ Done | 10.1% | 24.0% |
| B4 | KNN distribution | — | — | — | ✅ Done | 34.8% | 44.5% |
| B5 | Per-model ModernBERT | ModernBERT-base | 1024 | 5 | ✅ Done (Vast) | 37.7% | 47.3% |
| B6 | **Fused ModernBERT** | ModernBERT-base | 1024 | 5 | ✅ **DEPLOYED** | **38.0%** | **48.2%** |
| B7 | Fused RoBERTa | RoBERTa-base | 512 | 5 | ❌ NEED | — | — |
| B8 | Fused DeBERTaV3 | DeBERTaV3-base | 512 | 5 | ❌ NEED | — | — |
| B9 | Qwen LoRA fused bucket | Qwen2.5-0.5B | 1024 | 3 | ❌ NEED | — | — |

### B. Length Regression (→ E[length] for cost, LPT sort, E2E latency)

| ID | Approach | Encoder | Loss | Context | Epochs | Status | 7B MAE | 7B MAPE |
|----|----------|---------|------|---------|--------|--------|--------|---------|
| R1 | ModernBERT-base | ModernBERT-base | MSE | 1024 | 5 | ✅ Done (underfitting) | 176 | 187% |
| R2 | ModernBERT-base | ModernBERT-base | sMAPE | 1024 | 5 | ✅ Done (underfitting) | 168 | 164% |
| R3 | ModernBERT-base | ModernBERT-base | Huber | 1024 | 5 | ✅ Done (underfitting) | 172 | 176% |
| R4 | **ModernBERT-base** | ModernBERT-base | **log-transform** | 1024 | 5 | ✅ **DEPLOYED** | **139** | **126%** |
| R5 | ModernBERT-large | ModernBERT-large | log-transform | 1024 | 5 | ✅ Done | 135 | 135% |
| R6 | ModernBERT-base MSE long | ModernBERT-base | MSE | 1024 | 100 | ❌ NEED | — | — |
| R7 | ModernBERT-base Huber long | ModernBERT-base | Huber | 1024 | 100 | ❌ NEED | — | — |
| R8 | ModernBERT-base sMAPE long | ModernBERT-base | sMAPE | 1024 | 100 | ❌ NEED | — | — |
| R9 | RoBERTa-base MSE | RoBERTa-base | MSE | 512 | 100 | ❌ NEED | — | — |
| R10 | RoBERTa-base log-transform | RoBERTa-base | log-transform | 512 | 20 | ❌ NEED | — | — |
| R11 | DeBERTaV3-base MSE | DeBERTaV3-base | MSE | 512 | 100 | ❌ NEED (P3) | — | — |
| R12 | DeBERTaV3-base log-transform | DeBERTaV3-base | log-transform | 512 | 20 | ❌ NEED (P3) | — | — |
| R13 | Qwen LoRA fused regression | Qwen2.5-0.5B | MSE | 1024 | 3 | ✅ Done (Vast) | 186 | 66% |
| R14 | E[length] from bucket probs | Derived from B6 | — | — | — | ✅ Available | 143 | — |

**Notes on context truncation**: RoBERTa (512) and DeBERTaV3 (512) will auto-truncate prompts >512 tokens. This loses context compared to ModernBERT (8192 native, 1024 used). This is expected and part of the comparison — the paper should note this as a limitation of older encoders.

### C. Quality Prediction

| ID | Approach | Status | Best-Model Acc | MAE |
|----|----------|--------|---------------|-----|
| Q1 | **KNN (FAISS)** | ✅ **DEPLOYED** | **64%** | 0.08-0.11 |
| Q2 | Fused ModernBERT judge (10-class) | ✅ Done (CSD3) | 58-64% | — |
| Q3 | Qwen LoRA fused judge | ✅ Done (Vast) | — | 0.22 (ρ=0.06-0.19) |
| Q4 | Qwen LoRA fused similarity | ✅ Done (Vast) | — | 0.07 (ρ=0.45-0.49) |
| Q5 | Qwen LoRA fused reference_score | ✅ Done (Vast) | — | 0.20 (ρ=0.35-0.41) |
| Q6 | Qwen LoRA per-model judge | ❌ NEED | — | — |
| Q7 | Qwen LoRA per-model reference_score | ❌ NEED | — | — |
| Q8 | Qwen LoRA per-model similarity | ❌ NEED | — | — |
| Q9 | MLP | ✅ Done | — | 0.09-0.12 |

### D. Latency Prediction

| ID | Approach | Status | MAPE |
|----|----------|--------|------|
| L1 | **XGBoost TTFT+TPOT+E2E** | ✅ **DEPLOYED** | **1.2-6.2%** |
| L2 | Roofline analytical | ✅ Done | — |
| L3 | Static TPOT lookup | ✅ Done | — |
| L4 | LSTM | ✅ Done | (off-by-one bug) |

---

## Training Jobs Required

### Priority 1: Paper Table Requirements (MUST before submission)

These are needed to back claims in the paper's Table 3 (regression comparison).

| Job | What | Script | Encoder | Loss | Target | Epochs | Venv | Machine | Est. Time |
|-----|------|--------|---------|------|--------|--------|------|---------|-----------|
| **J1** | R6 | train_bert_predictor.py | ModernBERT-base | MSE | length (7B) | 100 | main | CSD3 | ~10h |
| **J2** | R7 | train_bert_predictor.py | ModernBERT-base | Huber | length (7B) | 100 | main | CSD3 | ~10h |
| **J3** | R8 | train_bert_predictor.py | ModernBERT-base | sMAPE | length (7B) | 100 | main | CSD3 | ~10h |
| **J4** | R9 | train_bert_predictor.py | RoBERTa-base | MSE | length (7B) | 100 | roberta | CSD3 | ~10h |
| **J5** | R10 | train_bert_predictor.py | RoBERTa-base | log-transform | length (7B) | 20 | roberta | CSD3 | ~2h |
| **J6** | B7 | train_fused_bert_predictor.py | RoBERTa-base | CE (bucket) | length_bucket | 5 | roberta | CSD3 | ~1h |

**Total P1: ~43 GPU-hours on CSD3**

### Priority 2: Ablation Completeness

| Job | What | Script | Approach | Target | Epochs | Machine | Est. Time |
|-----|------|--------|----------|--------|--------|---------|-----------|
| **J7** | B9 | train_llm_predictor.py | Qwen LoRA fused | length_bucket | 3 | Vast | ~8h |
| **J8** | Q6 | train_llm_predictor.py | Qwen LoRA per-model | judge | 3 | Vast | ~2h |
| **J9** | Q7 | train_llm_predictor.py | Qwen LoRA per-model | reference_score | 3 | Vast | ~2h |
| **J10** | Q8 | train_llm_predictor.py | Qwen LoRA per-model | similarity | 3 | Vast | ~2h |
| **J11** | B8 | train_fused_bert_predictor.py | DeBERTaV3-base | length_bucket | 5 | CSD3 | ~1h |

**Total P2: ~14h Vast + ~1h CSD3**

### Priority 3: Nice-to-Have

| Job | What | Script | Approach | Target | Epochs | Machine | Est. Time |
|-----|------|--------|----------|--------|--------|---------|-----------|
| J12 | R11 | train_bert_predictor.py | DeBERTaV3-base MSE | length (7B) | 100 | CSD3 | ~10h |
| J13 | R12 | train_bert_predictor.py | DeBERTaV3-base log | length (7B) | 20 | CSD3 | ~2h |
| J14 | — | train_fused_bert_predictor.py | Fused ModernBERT MSE | length (all) | 100 | CSD3 | ~12h |

**Total P3: ~24h CSD3**

---

## Execution Plan

### Vast.ai (24h remaining, RTX 8000 45GB)

Run sequentially (GPU can only do one at a time):

```bash
# J7: LoRA fused bucket (~8h)
python3 train_llm_predictor.py \
    --input ~/data/train.jsonl --test-input ~/data/test.jsonl \
    --target length_bucket --mode fused --epochs 3 \
    --output-dir ~/models/qwen_lora_fused/length_bucket

# J8-J10: LoRA per-model quality (~6h total)
for TARGET in judge reference_score similarity; do
    python3 train_llm_predictor.py \
        --input ~/data/train.jsonl --test-input ~/data/test.jsonl \
        --target $TARGET --mode per_model --epochs 3 \
        --output-dir ~/models/qwen_lora_per_model/$TARGET
done
```

**Total: ~14h — fits in 24h window**

### CSD3 Phase A: INTR Verification (currently running)

Job 26507842 testing RoBERTa MSE, log-transform, bucket, fused bucket with 3 epochs each.

### CSD3 Phase B: SLURM Full Jobs

**SLURM Job 1: `csd3_modernbert_long.sh`** (main venv, ~30h)
- J1: ModernBERT MSE 100ep (early stopping patience=10)
- J2: ModernBERT Huber 100ep
- J3: ModernBERT sMAPE 100ep

**SLURM Job 2: `csd3_roberta_full.sh`** (roberta venv, ~13h)
- J4: RoBERTa MSE 100ep
- J5: RoBERTa log-transform 20ep
- J6: Fused RoBERTa bucket 5ep

**SLURM Job 3: `csd3_deberta.sh`** (main venv, ~1h P2 + 12h P3)
- J11: Fused DeBERTaV3 bucket 5ep
- J12: DeBERTaV3 MSE 100ep (P3)
- J13: DeBERTaV3 log-transform 20ep (P3)

**Total CSD3: ~56 GPU-hours** (P1+P2: ~44h, P3: ~12h)
Budget: 3,000 GPU-hrs/quarter — plenty.

### Common Training Args

All regression jobs:
```
--save-total-limit 2          # Keep best + last checkpoint
--early-stopping-patience 10  # Stop if no improvement for 10 epochs
--seed 42                     # Reproducibility
--lr 2e-5                     # Standard for fine-tuning
--batch-size 16               # Fits on A100-80GB
```

RoBERTa/DeBERTaV3 specific:
```
--max-length 512              # Auto-truncation (512 context limit)
--precision fp16              # RoBERTa works with fp16
```

DeBERTaV3 specific:
```
--precision bf16              # DeBERTaV3 requires bf16
```

---

## Expected Paper Tables After Training

### Table 3: Length Regression Comparison (7B target, hardest case)

| Encoder | Loss | Context | Epochs | MAE | MAPE | ρ |
|---------|------|---------|--------|-----|------|---|
| ModernBERT-base | MSE | 1024 | 100 | J1 | J1 | J1 |
| ModernBERT-base | sMAPE | 1024 | 100 | J3 | J3 | J3 |
| ModernBERT-base | Huber | 1024 | 100 | J2 | J2 | J2 |
| ModernBERT-base | log-transform | 1024 | 5 | **139** | **126%** | **0.57** |
| ModernBERT-large | log-transform | 1024 | 5 | 135 | 135% | — |
| RoBERTa-base | MSE | 512 | 100 | J4 | J4 | J4 |
| RoBERTa-base | log-transform | 512 | 20 | J5 | J5 | J5 |
| Qwen2.5-0.5B+LoRA | MSE (regression) | 1024 | 3 | 186 | 66% | — |

### Table 5: Bucket Classification Comparison

| Method | 3B | 7B | 14B | 72B |
|--------|----|----|-----|-----|
| Static majority | 30/180 | 31/222 | 39/137 | 28/162 |
| Online PFS | 11/148 | 10/177 | 15/127 | 14/129 |
| Online PFS + bins | 16/143 | 10/175 | 24/121 | 17/128 |
| KNN distribution | 35/111 | 35/159 | 45/97 | 34/111 |
| Per-model ModernBERT | 39/97 | 38/146 | 47/86 | 40/97 |
| **Fused ModernBERT** | **40/97** | **38/143** | **48/85** | **40/98** |
| Fused RoBERTa (512 ctx) | J6 | J6 | J6 | J6 |
| Fused DeBERTaV3 (512 ctx) | J11 | J11 | J11 | J11 |
| Qwen LoRA fused bucket | J7 | J7 | J7 | J7 |

### Table: E[length] Approaches

| Method | How E[length] is derived | 7B MAE |
|--------|-------------------------|--------|
| ModernBERT bucket → midpoint expectation | Σ P(bucket_i) × midpoint_i | 143 |
| ModernBERT log-transform regression | Direct regression output | 139 |
| Qwen LoRA fused regression | LLM generates number | 186 |
| Best per-model regression (from Table 3) | From long-run results | TBD |

---

## Deployment Stack (Unchanged)

| Component | Model | Reason |
|-----------|-------|--------|
| Budget compliance | Fused ModernBERT bucket (B6) | 48% top-1, probability distributions |
| Length regression | ModernBERT log-transform (R4) | Converged, MAE=82-139, ρ=0.57-0.75 |
| Quality | KNN FAISS (Q1) | 64% best-model, sub-ms, no GPU |
| Latency | XGBoost TTFT+TPOT+E2E (L1) | 1.2-6.2% MAPE |

Training results do NOT change deployment — they fill paper comparison tables.
