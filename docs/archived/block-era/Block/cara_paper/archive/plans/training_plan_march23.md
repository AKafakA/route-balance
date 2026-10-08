# Cara Model Training & Development Plan (March 23 → April 25)

**Target**: NeurIPS May 6 deadline. CloudLab 18-node cluster reserved April 26 - May 6.

## Resources
| Resource | Capability | Best for |
|----------|-----------|----------|
| GPU VM (4× L40S) | Internet, permanent | Integration testing, Qwen-0.5B training, KNN, XGBoost/Linear/Roofline |
| CSD3 (A100-80GB) | 2,999 GPU-hrs to Apr 30 | ModernBERT batch training, LSTM |
| VPS (this machine) | CPU only, central hub | Scripts, paper, coordination |

## Data Alignment (verified March 23)
| Data | Local (desktop) | GPU VM | CSD3 |
|------|----------------|--------|------|
| `train_fixed.jsonl` (15k) | `9ce4...` | `9ce4...` ✓ | `9ce4...` ✓ |
| `test_fixed.jsonl` (3.6k) | `d5ef...` | `d5ef...` ✓ | `d5ef...` ✓ |
| `latency_train.jsonl` (359k) | `a8a0...` | `a8a0...` ✓ | `a8a0...` ✓ |

## Environment Notes
- **GPU VM**: `.venv/bin/python` has xgboost, torch 2.6, transformers 5.3, sklearn. `block` importable.
- **CSD3**: `.venv/bin/python` (Python 3.11) has xgboost, torch 2.5.1, transformers 5.3, sklearn. Needs `PYTHONPATH=.` for block imports. Partition: `ampere`, Account: `KALYVIANAKI-SL3-GPU`.
- **VPS**: No ML packages. Code-only hub.

---

## Week 1: Mar 23-29 — Model Training Sprint

### Day 1-2: CPU models (GPU VM) + GPU VM early study

**XGBoost E2E** (GPU VM, CPU, ~5 min):
```bash
cd ~/Code/llm/Block && PYTHONPATH=. .venv/bin/python -m block.predictor.cara.offline_training.train_xgboost \
    --data-dir data/cara/latency_data/all/ --output-dir models/cara/xgboost_e2e/ --target e2el
```

**XGBoost TTFT** (GPU VM, CPU, ~5 min):
```bash
cd ~/Code/llm/Block && PYTHONPATH=. .venv/bin/python -m block.predictor.cara.offline_training.train_xgboost \
    --data-dir data/cara/latency_data/all/ --output-dir models/cara/xgboost_ttft/ --target ttft
```

**XGBoost TPOT** (GPU VM, CPU, ~5 min):
```bash
cd ~/Code/llm/Block && PYTHONPATH=. .venv/bin/python -m block.predictor.cara.offline_training.train_xgboost \
    --data-dir data/cara/latency_data/all/ --output-dir models/cara/xgboost_tpot/ --target tpot
```

**Linear baseline** (GPU VM, CPU, ~1 min):
```bash
cd ~/Code/llm/Block && PYTHONPATH=. .venv/bin/python -m block.predictor.cara.offline_training.train_linear \
    --data-dir data/cara/latency_data/all/ --output-dir models/cara/linear/
```

**Roofline calibration** (GPU VM, CPU, ~1 min):
```bash
cd ~/Code/llm/Block && PYTHONPATH=. .venv/bin/python -m block.predictor.cara.offline_training.train_roofline \
    --data-dir data/cara/latency_data/all/ --output-dir models/cara/roofline/
```

### Day 1-2 (cont): GPU VM early study — fused models, 3-5 epochs

**Rationale:** Run fused models first to establish baselines before committing CSD3 hours.
Compare ModernBERT vs Qwen-0.5B LoRA across targets. Decide per-model vs fused from results.

**ModernBERT fused × 4 targets** (GPU VM, 1 L40S, ~30 min each):
```bash
cd ~/Code/llm/Block
# length_bucket (16-class, for budget filtering via P(affordable))
PYTHONPATH=. .venv/bin/python -m block.predictor.cara.offline_training.train_bert_predictor \
    --input data/cara/training_data/train_fixed.jsonl \
    --test-input data/cara/training_data/test_fixed.jsonl \
    --target length_bucket --lr 1e-5 --epochs 5 --device cuda \
    --output-dir models/cara/early_study/modernbert_fused_bucket/

# length (pointwise regression)
PYTHONPATH=. .venv/bin/python -m block.predictor.cara.offline_training.train_bert_predictor \
    --input data/cara/training_data/train_fixed.jsonl \
    --test-input data/cara/training_data/test_fixed.jsonl \
    --target length --lr 1e-5 --epochs 5 --device cuda \
    --output-dir models/cara/early_study/modernbert_fused_length/

# similarity (regression)
PYTHONPATH=. .venv/bin/python -m block.predictor.cara.offline_training.train_bert_predictor \
    --input data/cara/training_data/train_fixed.jsonl \
    --test-input data/cara/training_data/test_fixed.jsonl \
    --target similarity --lr 1e-5 --epochs 5 --device cuda \
    --output-dir models/cara/early_study/modernbert_fused_similarity/

# judge_class (10-class)
PYTHONPATH=. .venv/bin/python -m block.predictor.cara.offline_training.train_bert_predictor \
    --input data/cara/training_data/train_fixed.jsonl \
    --test-input data/cara/training_data/test_fixed.jsonl \
    --target judge_class --lr 1e-5 --epochs 5 --device cuda \
    --output-dir models/cara/early_study/modernbert_fused_judge/
```

**Qwen-0.5B LoRA fused × 3 targets** (GPU VM, 1 L40S, ~1-2h each):
```bash
# length (pointwise — showed promising results on 7B)
PYTHONPATH=. .venv/bin/python -m block.predictor.cara.offline_training.train_llm_predictor \
    --input data/cara/training_data/train_fixed.jsonl \
    --test-input data/cara/training_data/test_fixed.jsonl \
    --target length --epochs 3 \
    --output-dir models/cara/early_study/qwen05b_fused_length/

# similarity
PYTHONPATH=. .venv/bin/python -m block.predictor.cara.offline_training.train_llm_predictor \
    --input data/cara/training_data/train_fixed.jsonl \
    --test-input data/cara/training_data/test_fixed.jsonl \
    --target similarity --epochs 3 \
    --output-dir models/cara/early_study/qwen05b_fused_similarity/

# quality
PYTHONPATH=. .venv/bin/python -m block.predictor.cara.offline_training.train_llm_predictor \
    --input data/cara/training_data/train_fixed.jsonl \
    --test-input data/cara/training_data/test_fixed.jsonl \
    --target quality --epochs 3 \
    --output-dir models/cara/early_study/qwen05b_fused_quality/
```

**LSTM latency + quality** (GPU VM, CPU or 1 L40S):
```bash
PYTHONPATH=. .venv/bin/python -m block.predictor.cara.offline_training.train_lstm_latency \
    --data-dir data/cara/latency_data/all/ --output-dir models/cara/lstm_latency/

PYTHONPATH=. .venv/bin/python -m block.predictor.cara.offline_training.train_lstm_predictor \
    --input data/cara/training_data/train_fixed.jsonl \
    --test-input data/cara/training_data/test_fixed.jsonl \
    --output-dir models/cara/lstm_quality/
```

**KNN retrain** (GPU VM):
```bash
PYTHONPATH=. .venv/bin/python -m block.predictor.cara.offline_training.train_knn_estimator \
    --input data/cara/training_data/train_fixed.jsonl \
    --test-input data/cara/training_data/test_fixed.jsonl \
    --output-dir models/cara/knn/
```

### Day 3-4: Analyze early study results → decide CSD3 jobs

Compare all early study models. Key questions:
- Does per-model beat fused for length_bucket? (Run 1-2 per-model variants to check)
- ModernBERT vs Qwen-0.5B LoRA: which architecture wins per target?
- Is 5-epoch fused already sufficient, or does 20-epoch give meaningful gain?

### Day 5-6: CSD3 full runs (winning configs only)

Submit only configs that showed promise in early study. Use `intr` QoS (1h debug) for validation first.
```bash
# Example: if ModernBERT fused length_bucket won, run full 20 epochs
sbatch cara_bert_fused_bucket_20ep.sh   # intr first, then long
```

### Day 7: Evaluate + select
```bash
PYTHONPATH=. .venv/bin/python -m block.predictor.cara.offline_training.evaluate_predictors \
    --test-input data/cara/training_data/test_fixed.jsonl \
    --train-input data/cara/training_data/train_fixed.jsonl \
    --predictors knn:models/cara/knn modernbert:models/cara/early_study/modernbert_fused_bucket \
    --target length --device cuda
```

---

## Week 2: Mar 30 - Apr 5 — Integration & Experiment Scripts

### Scheduler enhancements (cara_serve.py, cara_learned_predictor.py)
- [ ] Data-driven normalization (replace hardcoded max values)
- [ ] Add LPT strategy (longest-predicted-time first)
- [ ] Runtime weight override via request param
- [ ] Fix target field naming (V2 Task #2)
- [ ] Real vs predicted output length switching (V2 Task #3)

### Integration testing (GPU VM)
- [ ] Load ModernBERT + XGBoost + KNN into CARALearnedPredictor
- [ ] Test predict_full() for all 4 model sizes
- [ ] Mini-benchmark: 50 requests, all 7 schedulers
- [ ] Profile predictor latency (target: <10ms)

### Experiment scripts (block/exp/cara/)
- [ ] `run_full_comparison.sh` — 7 schedulers × 5 QPS × 2000 requests
- [ ] `run_weight_sweep.sh` — 9 weight combos × 5 QPS
- [ ] `run_ablation.sh` — predictor combos
- [ ] `run_slo_sweep.sh` — budget × quality_min combos
- [ ] `plot_results.py` — CDF latency, Pareto, model distribution
- [ ] `collect_and_summarize.py` — paper tables

---

## Week 3: Apr 7-12 — Pre-CloudLab Validation + Paper
- Integration test: CSD3 (2 nodes, ~32 GPU-hrs) or P100 CloudLab on-demand
- Paper: USENIX → NeurIPS template, write methodology + related work

## Week 4: Apr 14-20 — Block Experiments + Cara Paper
- Block CloudLab Apr 16-20 (rebuttal)
- Cara paper sections in parallel

## Week 5: Apr 21-25 — Final Prep
- Fix issues from Week 3, update configs for 18-node cluster
- Experiment runbook, dry run on P100

---

## CloudLab 18-Node Deployment (Apr 26 - May 6)
- 2× A100 (d8545): Qwen2.5-72B (TP=4)
- 3× V100 (c4130): Qwen2.5-14B
- 5× A30 (d7525): Qwen2.5-7B (3) + Qwen2.5-3B (2)
- 5× P100 (c240g5): Qwen2.5-3B
- 3× A30: Coordinator + training

## Key Design Decisions (documented March 23)

### Bucket-based budget filtering
Pointwise length MAPE ~100% makes direct cost filtering useless. Instead:
1. Bucket classifier outputs P(bucket_k | prompt, model) for 16 × 64-token buckets
2. Given budget → compute max_affordable_output per model
3. P(affordable) = CDF up to max_affordable_bucket
4. Filter models where P(affordable) < 70% threshold
See: `cara_paper/codex/cara-objective-design.md` (Bucket-Based Budget Filtering section)

### TTFT + TPOT SLO decomposition
E2E latency SLO requires length prediction (high MAPE). Decompose into:
- TTFT SLO: depends on prompt length + queue state (no output length needed)
- TPOT SLO: depends on load + model/GPU (no output length needed)
XGBoost trains separate models for e2el, ttft, tpot targets.

### Training targets summary
| Predictor | Targets | Purpose |
|-----------|---------|---------|
| XGBoost (blind) | e2el (num_predicted=1024), ttft, tpot | Latency baselines |
| XGBoost (actual) | e2el (actual output tokens) | E2E with perfect length |
| ModernBERT fused | length_bucket, length, similarity, judge_class | Budget filtering + quality |
| ModernBERT fused log-transform | length | Regression baseline comparison |
| ModernBERT per-model 7B | length_bucket, length, similarity, judge_class | Per-model vs fused |
| RoBERTa fused | length (MSE, 512ctx), similarity, judge_class | Block baseline architecture |
| Qwen-0.5B LoRA | length, similarity, judge | Alternative architecture |
| KNN | length, similarity, judge + bucket dist from neighbors | Non-parametric baseline |
| MLP | length, similarity, judge | Embedding-based baseline |
| Linear/Roofline | e2el | Analytical baselines |
| LSTM | latency (e2el/ttft/tpot), similarity, judge | Sequence model ablation |

### Quality always separate
- **similarity**: cosine similarity vs 72B reference (regression)
- **judge**: LLM judge score (regression for Qwen/MLP/KNN/LSTM, 10-class for ModernBERT/RoBERTa)
- Never use combined "quality" score for training

### actual_output_tokens recovery
- Formula: `round((e2e - ttft) / tpot) + 1` — verified 100% exact on 4,690 HF records
- TPOT = mean(ITL), so (e2e-ttft)/tpot = len(ITL) = output_tokens - 1
- All 359k local records now have `actual_output_tokens` in tagged data

### Bucket filtering evaluation
- Budget modes: **strict** (min length), **balanced** (median), **relaxed** (2nd largest)
- Thresholds: 0.5, 0.7, 0.9
- 3×3 grid measuring false accept/reject rates per model per prompt

## CSD3 Budget
| Phase | GPU-hrs |
|-------|---------|
| Full 20-epoch runs (winning configs) | ~50-100 |
| Week 3 integration | ~32 |
| Buffer | ~50 |
| **Total estimate** | **~130-180 / 2,999** |

Use `intr` QoS (1h debug) for quick validation before committing `long` jobs.

## Progress Tracking

### Week 1 Status (updated March 23 evening)

**Early study (running on GPU VM, `run_gpu_vm_early_study.sh`):**
- [x] XGBoost E2E blind (1024) — done, 5 instance types
- [x] XGBoost TTFT — done, MAE 5-45ms per instance type
- [x] XGBoost TPOT — done, MAE 0.2-5.6ms per instance type
- [x] Linear baseline — done
- [x] Roofline calibration — done
- [x] ModernBERT fused length_bucket (with_squad) — done, 47.7% acc epoch 2
- [x] ModernBERT fused length (with_squad) — done
- [x] ModernBERT fused similarity (with_squad) — done
- [ ] ModernBERT fused judge_class (with_squad) — running
- [ ] ModernBERT fused × 4 targets (no_squad) — pending (will ignore results)
- [ ] Qwen-0.5B LoRA × 2 targets (old: length, quality) — pending
- [ ] KNN — pending

**Full study (queued, `run_gpu_vm_full_study.sh`, ~12h):**
- [ ] XGBoost E2E with actual output tokens
- [ ] ModernBERT fused log-transform length
- [ ] ModernBERT per-model 7B × 4 targets
- [ ] RoBERTa fused × 3 targets
- [ ] Qwen-0.5B LoRA × 3 targets (length, similarity, judge)
- [ ] KNN (with separate similarity/judge evaluation)
- [ ] MLP
- [ ] LSTM latency
- [ ] LSTM quality
- [ ] Bucket filtering evaluation (strict/balanced/relaxed × 0.5/0.7/0.9)

**Infrastructure fixes (March 23):**
- [x] Data aligned across all machines
- [x] Roofline calibration script created
- [x] Fixed target alias bug in XGBoost/Linear (ttft/tpot field names)
- [x] Tagged 405k latency records with instance_type (5 types)
- [x] Recovered actual_output_tokens via round((e2e-ttft)/tpot)+1
- [x] Fixed sidecar pipeline: output_tokens now flows through full chain
- [x] Fixed generate_latency_benchmark.py: source_request_id tracked
- [x] Updated paper: TPOT not assumed constant, separate TTFT+TPOT predictors
- [x] Documented bucket filtering + TTFT/TPOT SLO design
- [x] Added similarity/judge targets to train_llm_predictor.py
- [x] Created evaluate_bucket_filtering.py (strict/balanced/relaxed modes)
