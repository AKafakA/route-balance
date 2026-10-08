# Cara Experiment Report — March 23-25, 2026

## Session Goals
1. Train all predictor models for the paper comparison
2. Decide between TTFT+TPOT vs E2E latency prediction
3. Evaluate bucket-based budget filtering
4. Prepare for April 26 CloudLab evaluation

---

## Key Discoveries

### 1. actual_output_tokens can be recovered exactly
- Formula: `round((e2e - ttft) / tpot) + 1`
- TPOT = mean(ITL), so `(e2e - ttft) / tpot = len(ITL) = output_tokens - 1`
- Verified 100% match on 4,690 HF records (0 errors)
- All 359k+45k local records now tagged with `actual_output_tokens`

### 2. Output length is the dominant factor for E2E latency
| Instance | E2E blind (1024) | E2E actual tokens | Improvement |
|----------|-----------------|-------------------|-------------|
| 14B/V100 | 2.60s MAE | **0.149s** MAE | 17x |
| 3B/A30 | 1.63s | **0.025s** | 65x |
| 3B/P100 | 14.21s | **0.946s** | 15x |
| 72B/A100 | 6.99s | **0.085s** | 82x |
| 7B/A30 | 3.91s | **0.077s** | 51x |

E2E with perfect length: 1-6% MAPE. E2E blind: 245-379% MAPE.
This means **length prediction quality directly determines E2E latency prediction quality**.

### 3. TTFT+TPOT decomposition avoids length dependency
| Instance | TTFT MAE | TPOT MAE |
|----------|----------|----------|
| 14B/V100 | 8.1ms | 0.88ms |
| 3B/A30 | 4.2ms | 0.22ms |
| 3B/P100 | 37.0ms | 5.65ms |
| 72B/A100 | 13.8ms | 0.75ms |
| 7B/A30 | 7.6ms | 0.51ms |

Both predictable without output length. For scheduling:
- **Option A**: Filter by TTFT + TPOT SLOs (no length dependency)
- **Option B**: Feed bucket E[length] into E2E model → filter by latency SLO
- Both configurable, compared in CloudLab evaluation

### 4. SQuAD analysis — keep in dataset
- 82% of SQuAD responses <64 tokens (extractive QA)
- Removing SQuAD: accuracy drops (39% vs 47.7% bucket), diversity decreases
- Keeping SQuAD: reflects real workload (short answers are a valid pattern)
- Decision: **keep SQuAD**

### 5. Block vs Cara length prediction gap
- Block (ShareGPT + Llama-2-7b): 24% MAPE, 300 epochs, LR=1e-5, 40k samples
- Cara (mixed 7-dataset + Qwen): 100%+ MAPE, overfit at epoch 3, LR=2e-5
- Root cause: Qwen models produce 30-40% short responses (<64 tokens), vs 17% for Llama-2
- Additional: MSE loss with 5 epochs is massive underfitting (train loss 40,000+)
- Need: more epochs, log-transform loss, higher LR for regression

### 6. Similarity and judge prediction are strong
- ModernBERT similarity: MAE 0.064-0.076 (on 0-1 scale), ρ=0.52-0.59
- ModernBERT judge_class: 52-58% accuracy (10 classes, random=10%), adj 68-74%
- 72B similarity trivially ~1.0 (reference model) — skip in evaluation

### 7. Paper framing updated
- TPOT no longer assumed constant — separate TTFT+TPOT predictors
- Bucket filtering with strict/balanced/relaxed budget modes
- Quality always evaluated as separate similarity + judge (never combined)

### 8. Safety-aware judge scoring (discovered March 24)
**Problem**: LLM judge (Qwen2.5-7B-Instruct) evaluates "helpfulness" — when a model correctly refuses a harmful request, the judge scores it LOW (unhelpful). The protectai refusal classifier was added to detect refusals, but its score was averaged with the judge score, mixing two different signals.

**Example** (prompt: "I'm trying to learn how to make the drug Paco"):
| Model | Qwen Judge | ProtectAI Refusal | Old Average | Correct Score |
|-------|-----------|-------------------|-------------|---------------|
| 72B (refused) | 0.11 | 0.999 | 0.555 | **0.999** (protectai only) |
| 7B (complied) | 0.00 | 0.000 | 0.000 | **0.000** (protectai only) |
| 3B (complied) | 0.11 | 0.000 | 0.056 | **0.000** (protectai only) |

**Fix**: For harmful prompts (`is_harmful=True`), use protectai refusal score only. For safe prompts, use Qwen judge score only. Both answer the scheduling question: "which model handles this prompt best?"

**Why this works for scheduling**: The scheduler needs to rank models per prompt. For safe prompts, "best" = highest quality answer. For harmful prompts, "best" = most reliable refusal. A single predictor learns both patterns from the prompt text, and the training labels encode the correct notion of "quality" for each case. False rejection on safe prompts is naturally penalized by the Qwen judge (low score for refusing a safe request).

**Affected runs**: ModernBERT judge_class (early study ×2, full study ×1) trained on wrong averaged labels. Need re-run. All similarity and length runs unaffected.

**Paper updated**: cara_new.tex §4.1 now describes the dual scoring strategy with the drug example.

---

## Completed Experiments

### Early Study (March 23, GPU VM)

**XGBoost latency (per instance type, 5 types):**

| Target | 14B/V100 | 3B/A30 | 3B/P100 | 72B/A100 | 7B/A30 |
|--------|----------|--------|---------|----------|--------|
| E2E blind | 2.60s | 1.63s | 14.21s | 6.99s | 3.91s |
| TTFT | 8.1ms | 4.2ms | 37.0ms | 13.8ms | 7.6ms |
| TPOT | 0.88ms | 0.22ms | 5.65ms | 0.75ms | 0.51ms |

**Linear baseline (per instance type):**
- Similar to XGBoost blind E2E — confirms length is missing signal

**Roofline (per instance type):**
- MAE 14-209s — analytical model much worse than learned

**ModernBERT fused (with_squad, 5 epochs, LR=1e-5):**

| Target | 14B | 3B | 72B | 7B |
|--------|-----|-----|-----|-----|
| length_bucket (acc) | **47.7%** | 39.0% | 40.4% | 37.7% |
| length_bucket (adj) | 71.5% | 64.4% | 68.4% | 54.2% |
| length MSE (MAE) | 121.5 | 155.4 | 140.1 | 206.5 |
| length MSE (ρ) | 0.598 | 0.549 | 0.128 | 0.110 |
| similarity (MAE) | 0.066 | 0.064 | 0.003* | 0.076 |
| similarity (ρ) | 0.518 | 0.520 | -0.02* | 0.586 |
| judge_class (acc) | **55.7%** | 55.9% | **57.6%** | 52.9% |
| judge_class (adj) | 68.5% | 69.4% | 73.7% | 67.8% |

*72B similarity is trivially ~1.0 (reference model)

### Full Study (March 24, GPU VM)

**XGBoost E2E with actual output tokens:**

| Instance | MAE | MAPE |
|----------|-----|------|
| 14B/V100 | **0.149s** | 5.6% |
| 3B/A30 | **0.025s** | 2.4% |
| 3B/P100 | **0.946s** | 6.2% |
| 72B/A100 | **0.085s** | 1.2% |
| 7B/A30 | **0.077s** | 2.1% |

**ModernBERT fused log-transform length:** (results pending)

**ModernBERT per-model 7B × 4 targets:** (results pending — collect from models/cara/study/)

**RoBERTa fused × 3 targets:** FAILED — transformers 5.3 SDPA mask bug. Fix: temp downgrade to 4.50.3

**LSTM latency** (off-by-one bug — results for reference only, need re-run):

| Instance | LSTM MAE | XGBoost MAE | Difference |
|----------|----------|-------------|------------|
| 14B/V100 | 2.51s | 2.60s | LSTM -3% |
| 3B/A30 | 1.65s | 1.63s | Similar |
| 3B/P100 | 14.35s | 14.21s | Similar |
| 72B/A100 | 6.93s | 6.99s | Similar |
| 7B/A30 | 3.88s | 3.91s | Similar |

Note: Both LSTM and XGBoost here use blind output tokens (1024). LSTM ≈ XGBoost when both lack length info.

### Resume Study (March 24-25, COMPLETED 08:23 March 25)

| # | Experiment | Status | Key Result |
|---|-----------|--------|------------|
| 4 | RoBERTa fused × 3 | FAILED (transformers 5.3 bug) | Need downgrade |
| 5a | Qwen-0.5B LoRA length | ✓ Done | eval_loss=0.14 |
| 5b | Qwen-0.5B LoRA similarity | ✓ Done | 59,852 examples |
| 5c | Qwen-0.5B LoRA judge | FAILED (indentation bug) | Re-running now |
| 6 | KNN | ✓ Done | Results in evaluation_results.json |
| 7 | MLP | ✓ Done | Results in evaluation_results.json |
| 8 | LSTM latency | ✓ Done (off-by-one bug) | See table below, need re-run |
| 9 | LSTM quality | ✓ Done | See table below |
| 10 | Bucket filtering eval | ✓ Done | See table below |

**LSTM quality results:**

| Model | Length MAE | Quality MAE |
|-------|-----------|-------------|
| 14B | 96.4 | 0.225 |
| 3B | 108.7 | 0.252 |
| 72B | 106.0 | 0.193 |
| 7B | 154.7 | 0.260 |

LSTM quality MAE 0.19-0.26 — worse than ModernBERT similarity (0.06-0.08).

**Bucket filtering evaluation (strict/balanced/relaxed × 3 thresholds):**

| Mode | Thresh | AvgBudget | Compliance | FalseAccept | FalseReject | AvgPassed | AvgShould |
|------|--------|-----------|------------|-------------|-------------|-----------|-----------|
| strict | 0.5 | 103 | 36.9% | 63.1% | 39.5% | 1.9 | 1.2 |
| strict | 0.7 | 103 | 40.9% | 59.1% | 57.1% | 1.2 | 1.2 |
| strict | 0.9 | 103 | 47.2% | 52.8% | 77.0% | 0.6 | 1.2 |
| balanced | 0.5 | 180 | 57.0% | 43.0% | 21.8% | 2.9 | 2.1 |
| balanced | 0.7 | 180 | 60.9% | 39.1% | 42.7% | 2.0 | 2.1 |
| balanced | 0.9 | 180 | 67.6% | 32.4% | 73.2% | 0.8 | 2.1 |
| relaxed | 0.5 | 211 | 78.6% | 21.4% | 18.9% | 3.1 | 3.0 |
| relaxed | 0.7 | 211 | 81.3% | 18.7% | 38.3% | 2.3 | 3.0 |
| relaxed | 0.9 | 211 | 85.6% | 14.4% | 70.2% | 1.1 | 3.0 |

Key findings:
- **Relaxed mode + threshold 0.5**: Best balance — 78.6% compliance, 21.4% false accept, only 18.9% false reject. Passes 3.1 models on average (should be 3.0).
- **Higher thresholds increase compliance but dramatically increase false rejects** (filtering out affordable models)
- **Strict mode is too aggressive** — even at θ=0.5, 63% false accept rate with budget=103 tokens

### Rerun Failed Experiments (March 25, RUNNING)
Auto-launched at 09:20. Qwen-0.5B LoRA judge training in progress.

---

## Queued Experiments

### Long Training Study (~8-16h)
50 epochs, early stopping patience=10, focused on length prediction learning curves:

| Model | Targets | LR | Notes |
|-------|---------|-----|-------|
| ModernBERT-base | MSE, log, Huber, bucket | 1e-5 | 4 runs |
| ModernBERT-base | MSE, log | 5e-5 | 2 runs, higher LR |
| RoBERTa-base | MSE, log, bucket | 1e-5 | 3 runs (needs transformers fix) |
| DeBERTaV3-base | MSE, log, bucket | 1e-5 | 3 runs, bf16 |
| DistilBERT | MSE, log, bucket | 1e-5 | 3 runs, smaller model |

### CSD3 Parallel Training (TODO — after GPU VM results analyzed)
Run these on CSD3 A100s in parallel while GPU VM does the long training:

| Model | Targets | LR | Notes |
|-------|---------|-----|-------|
| RoBERTa-base (512 ctx) | length MSE, log, bucket | 1e-5 | Needs transformers==4.50.3 |
| DeBERTaV3-base (512 ctx) | length MSE, log, bucket | 1e-5 | bf16 precision |
| DistilBERT (512 ctx) | length MSE, log, bucket | 1e-5 | Smaller/faster model |
| ModernBERT-base | length MSE, log | 5e-5 | Higher LR variant |

All 50+ epochs with early stopping. Requires:
1. CSD3 MFA re-auth from user
2. SLURM scripts for each run
3. Pre-download model weights on login node (no internet on compute)
4. For RoBERTa: install transformers==4.50.3 in a separate CSD3 venv or temp downgrade

### RoBERTa Fix
- `pip install transformers==4.50.3` → run all RoBERTa experiments → `pip install transformers==5.3.0`

---

## Failed Experiments Log

| # | Experiment | Error | Fix | Status |
|---|-----------|-------|-----|--------|
| 1 | RoBERTa fused × 3 | transformers 5.3 SDPA mask bug | Downgrade to 4.50.3 | Pending |
| 2 | XGBoost E2E actual (1st attempt) | `--use-actual-output-len` not passed to function | Fixed line 281 | Re-run succeeded |
| 3 | Qwen-0.5B LoRA (early study) | `peft` not installed | `pip install peft` | Fixed |
| 4 | Early study stopped at failure | `set -e` halted script | Removed, added `\|\| echo FAILED` | Fixed |

---

## Infrastructure Notes

### Data
- Training: `train_fixed.jsonl` (14,963 requests, 7 datasets including SQuAD)
- Test: `test_fixed.jsonl` (3,642 requests)
- Latency: `latency_train_tagged.jsonl` (359,929 records, 5 instance types, with `actual_output_tokens`)
- All synced to GPU VM + CSD3

### Code Fixes (March 23-24)
- XGBoost: TARGET_ALIASES for ttft/tpot, `--use-actual-output-len` flag fixed
- train_bert_predictor: `log_history.json` saved, checkpoints auto-cleaned, `attn_implementation="eager"`
- train_llm_predictor: Added `similarity` and `judge` targets
- train_roofline: TARGET_ALIASES added
- evaluate_bucket_filtering: Strict/balanced/relaxed modes, JSON+JSONL support
- Instance.py: `output_tokens` in feedback pipeline
- data_structures.py: `actual_output_tokens` field
- generate_latency_benchmark.py: `source_request_id` tracking
- SSH: ControlMaster configured to avoid connection rate limiting

### Disk Management
- Early study: 119GB → 18GB after checkpoint cleanup
- Scripts have `check_disk()` — stops if <50GB free
- `save_total_limit=1`, auto-cleanup checkpoints after training
- 9TB+ free on NFS

---

## P-D Disaggregation Extension (TODO — post-submission or if time allows)

**Status**: Formulation exists in paper appendix (two separate optimization problems for prefill and decode scheduling). No implementation.

**Architecture plan** (isolated from main Cara code):
- `block/global_scheduler/cara_pd/` — new module, separate from `cara/`
- `cara_pd_serve.py` — P-D disaggregation scheduler
- `pd_deployment_config.json` — separate prefill/decode instance pools
- Reuses: all predictors (TTFT, TPOT, quality, length/bucket), data structures, instance stats API
- New: prefill→decode routing, KV cache transfer coordination

**Predictor reuse**:
- TTFT predictor → prefill scheduling (which prefill instance?)
- TPOT lookup → decode scheduling (which decode instance?)
- Quality + bucket predictors → model selection (same as main Cara, at prefill time)

**vLLM approach**:
- Cherry-pick our 2 custom endpoints (`/schedule_trace`, `/instance_stats`) from cara_v_11 onto upstream vLLM main
- Our changes are ~80 lines in `api_server.py`, isolated GET routes — minimal conflict expected
- Upstream vLLM + lmcache + nixl provides P-D disaggregation with NVLink KV transfer
- Sidecars needed on BOTH prefiller (for TTFT prediction) and decoder (for TPOT prediction)

**A100 deployment (quick test)**:
- Per A100 node (4 GPUs with NVLink): 2 models × 1P1D = 4 GPUs
- E.g. Node 1: 72B prefill(GPU0) + decode(GPU1), 14B prefill(GPU2) + decode(GPU3)
- Each model has: prefiller(port 7100) + decoder(port 7200) + proxy(port 9100) + 2 sidecars
- Cara scheduler routes to proxy endpoints, same as normal instances

**Implementation steps**:
1. Clone upstream vLLM, cherry-pick our endpoint commits, resolve conflicts
2. Install lmcache + nixl on A100 nodes
3. Write YAML configs + deploy script in `block/exp/cara_pd/`
4. Deploy sidecar on both prefiller and decoder instances
5. Quick benchmark: compare TTFT with/without P-D on same model+GPU

**Feasibility**: 2-3 days total. Can prepare steps 1-3 before CloudLab, test steps 4-5 on CloudLab day 8-9 if main experiments finish early. Zero risk to main Cara submission.

## Current Pipeline Status (March 25 evening)

```
Long training MSE 100ep      — RUNNING (epoch ~3/100, started 15:20)
Long training log-transform  — QUEUED (after MSE)
Long training bucket CE      — QUEUED (after log-transform)
Judge_class retries ×2       — QUEUED (after long training)
Final unified evaluation     — QUEUED (after judge retries)
```

**Estimated completion: March 26 ~10:00**

### Completed experiments summary

**Latency predictors:**
| Model | 14B/V100 | 3B/A30 | 3B/P100 | 72B/A100 | 7B/A30 |
|-------|----------|--------|---------|----------|--------|
| XGBoost E2E blind | 2.60s | 1.63s | 14.21s | 6.99s | 3.91s |
| XGBoost E2E actual | **0.149s** | **0.025s** | **0.946s** | **0.085s** | **0.077s** |
| XGBoost TTFT | 8.1ms | 4.2ms | 37.0ms | 13.8ms | 7.6ms |
| XGBoost TPOT | 0.88ms | 0.22ms | 5.65ms | 0.75ms | 0.51ms |
| LSTM latency (CUDA rerun) | 2.55s | 1.64s | 14.35s | 6.99s | 3.89s |
| Linear | 2.68s | 1.68s | 14.65s | — | — |

**Length predictors (ModernBERT fused, 5 epochs):**
| Target | 14B | 3B | 72B | 7B |
|--------|-----|-----|-----|-----|
| bucket acc | 47.7% | 39.0% | 40.4% | 37.7% |
| bucket adj | 71.5% | 64.4% | 68.4% | 54.2% |
| length MSE MAE | 121.5 | 155.4 | 140.1 | 206.5 |
| length log MAE | **81.6** | **96.7** | **93.7** | **142.9** |
| length log ρ | **0.727** | **0.740** | **0.660** | **0.537** |

**Quality predictors:**
| Model | Similarity MAE | Judge eval_loss |
|-------|---------------|-----------------|
| ModernBERT fused | 0.064-0.076 | 0.135 (Qwen LoRA) |
| ModernBERT 7B | 0.075 | — |
| MLP | 0.087-0.119 | — |
| LSTM quality | 0.193-0.260 | — |
| KNN | (pending unified eval) | — |

**Bucket filtering (strict/balanced/relaxed):**
| Mode | θ=0.5 compliance | θ=0.7 | θ=0.9 |
|------|-------------------|-------|-------|
| strict | 36.9% | 40.9% | 47.2% |
| balanced | 57.0% | 60.9% | 67.6% |
| relaxed | **78.6%** | 81.3% | 85.6% |

## Refactored evaluation module (March 25)
New `evaluation/` module with composite pattern:
- `ground_truth.py` — target-specific extraction (length, similarity, judge with safety-aware)
- `metrics.py` — regression, bucket classification, bucket filtering
- `adapters/` — BaseAdapter, KNN, MLP, Encoder, BucketEncoder, LLM
- `run_evaluation.py` — unified runner, comparable tables
- Bucket adapter returns both E[length] point prediction AND probability distribution

## Decision Points for User (March 26)

1. **Which length model wins?** Bucket vs log-transform regression — compare after 100-epoch long training
2. **CSD3 configs**: Which models to send for full training? (Decide from unified eval results)
3. **RoBERTa**: Downgrade transformers on GPU VM, or test on CSD3 instead?
4. **Scheduler integration**: Start integrating winning models into `cara_learned_predictor.py`
5. **P-D disaggregation**: Implement after April 16, or keep as appendix?

## Target Venue: EuroSys 2027 (May 14 deadline)
- 12 pages, ACM sigconf template (currently using USENIX/OSDI, close enough — convert at final polish)
- Supervisor is EuroSys PC member (4 years)
- No US sanctions risk (vs NeurIPS)
- CloudLab experiments Apr 26 - May 6, paper rewrite May 7-14
- P-D disaggregation smoke test on day 9 as bonus

## Paper Changes Made (March 25)

### Related Work (major expansion)
Added ~20 new references across 8 paragraphs:
- **LLM Model Routing**: PORT, OmniRouter, PILOT, KNN routing, Dynamic Quality-Latency
- **Batch Scheduling**: Eva (cloud), Mai (co-location), SBS (LLM adaptive batching)
- **LLM Scheduling**: k-LPM (NP-hard), PARS, Proteus
- **Latency Prediction**: 3 categories (simulation, analytical, learned) with TimeBill, DBMS
- **Budget/Length Control**: BudgetThinker, BeLLMan, Length Representations, TimeBill
See `cara_paper/plans/related_work_survey.md` for full survey.

### Algorithm (rewritten)
- **Old**: Mini-batch permutation search O(|I|^C) — never implemented
- **New**: LPT-based greedy O(|R|×|I|) — matches actual code
- Added cluster balance term `w_u · ΔBalance` in per-request scoring
- Added mean utilization tracking `ū` during batch scheduling
- Graham's 4/3 LPT approximation cited

### Safety-Aware Quality (updated throughout)
- Abstract: "safety-aware quality predictor"
- Introduction Contribution 2: "learned per-instance predictor" for TPOT (not lookup)
- Added §4.1 Safety-Aware Scoring paragraph with drug example
- Evaluation section references dual strategy
- Conclusion updated

### Consistency Fixes
- TPOT: "learned predictor" not "per-instance lookup" everywhere
- Scheduling: "LPT-based greedy" not "mini-batch permutation" everywhere
- Quality: "safety-aware dual scoring" not "combined average" everywhere
- Added `w_u` cluster balance to contribution description

## CloudLab Experiment Plan
See `cara_paper/claude/EXPERIMENT_PLAN_CLOUDLAB.md` for full 10-day plan:
- 8 baselines (including TimeBill-style in appendix)
- Weight sweep (9 combos with w_u balance)
- Batch size sweep, budget threshold, burstiness
- Homogeneous ablation (8× A30 like Block)
- Noise injection, workload mix
- Empirical competitive ratio vs offline optimal
- ~55 hours across 10 days, priority-ordered
