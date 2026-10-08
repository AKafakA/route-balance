# Cara Predictor Training Infrastructure — Implementation Plan

## Context
18 CloudLab nodes are deployed (Steps 1-3 DONE). This plan covers: data collection, training data processing, and building training scripts for all predictors described in the paper design docs. The scope ends at "training infra ready to use" — after collecting broadcast data and stopping vLLM instances, we train models on the GPU nodes themselves. Scheduling integration and evaluation will be replanned separately.

## Design Alignment
Per `cara_paper/codex/cara-data-and-experiments.md` and `cara_paper/claude/TASK_1_4_DESIGN.md`:
- **Length/Quality predictors**: KNN-based (primary), MLP/BERT/LLM-based (ablation)
- **Latency predictor**: XGBoost (primary), LSTM (ablation), Roofline (analytical baseline), Linear (ablation)
- Existing code to reuse: `block/length_estimation/train_roberta.py` (RoBERTa finetuning), `similarity_scorer.py` + `llm_judge_scorer.py` (quality scoring)

## What Already Exists (code)
- `block/predictor/cara/offline_training/prepare_training_data.py` — broadcast JSON → scored training data
- `block/predictor/cara/offline_training/similarity_scorer.py` — embedding-based quality scoring
- `block/predictor/cara/offline_training/llm_judge_scorer.py` — LLM judge quality scoring
- `block/predictor/cara/offline_training/response_filter.py` — bad response filtering
- `block/length_estimation/train_roberta.py` — RoBERTa regression finetuning (simpletransformers)
- `block/length_estimation/eval_roberta.py` — RoBERTa evaluation
- `block/predictor/cara/dummy_cara_predictor.py` — heuristic predictor + training data collector
- `block/predictor/cara/training_data_collector.py` — online latency data collection
- Placeholder interfaces: `BaseOutputLengthEstimator`, `BaseQualityEstimator` (in TASK_1_4_DESIGN.md, not yet in code)

## What Already Exists (design docs only, NO code)
- XGBoost predictor design (LATENCY_PREDICTOR_DESIGN.md)
- Linear predictor design
- Roofline predictor design
- Multi-objective scorer design (cara-objective-design.md)
- RSO/SLO data structures design (TASK_1_4_DESIGN.md)

---

## Step 4: Data Collection (Day 1)

### 4a. Broadcasting Data (for length + quality predictors)

Use v3 dataset with 7 balanced datasets (LMSYS replaces ultrachat; ai2_arc and cnn_dailymail removed). Split 80/20 later for train/test.

**Dataset generation:**
```bash
# Full 20k dataset (with response-length pre-filtering on LMSYS)
python -m block.data.collect_data \
    --datasets reward_bench code_ultra_feedback beaver_tails mix_instruct lmsys gsm8k squad \
    --total-n 20000 --max-prompt-length 700 --tokenizer Qwen/Qwen2.5-3B \
    --chat-lang English --chat-max-response-tokens 1024 --chat-max-total-tokens 2048 \
    -o data/cara/best-route-v3.jsonl

# 500-sample test set (for quick ablation)
python -m block.data.collect_data \
    --datasets reward_bench code_ultra_feedback beaver_tails mix_instruct lmsys gsm8k squad \
    --total-n 500 --max-prompt-length 700 --tokenizer Qwen/Qwen2.5-3B \
    --chat-lang English \
    -o data/cara/best-route-v3-test-500.jsonl
```

**Broadcasting:**
```bash
# Validation (500 prompts, quick sanity check)
./block/exp/cara/run_broadcasting_remote.sh \
    asdwb@d8545-10s10301.wisc.cloudlab.us \
    "Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B Qwen/Qwen2.5-72B" \
    custom 500 inf initial_18node \
    data/cara/best-route-v3-test-500.jsonl

# Full run (all 20k prompts, rate-limited)
./block/exp/cara/run_broadcasting_remote.sh \
    asdwb@d8545-10s10301.wisc.cloudlab.us \
    "Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B Qwen/Qwen2.5-72B" \
    custom 20000 4.0 full_20k \
    data/cara/best-route-v3.jsonl
```

**Generation parameters** (from sweep ablation):
- `temperature=0.0, frequency_penalty=1.2, repetition_penalty=1.0`

### 4b. Latency Training Data (synthetic microbenchmark, Days 1-2)

Latency predictors don't need real prompts — use synthetic data with random (input_len, output_len) pairs and dummy tokens. This allows generating unlimited training data. Run continuously for 1-2 days to maximize data volume.

**Create:** `block/predictor/cara/offline_training/generate_latency_benchmark.py`

This script generates AND sends synthetic requests end-to-end:
1. Samples (input_len, output_len) from realistic distributions (e.g., log-normal matching ShareGPT/Cara dataset stats)
2. Generates dummy prompts with fake/repeated tokens at the desired input_len
3. Sets `max_tokens = sampled_output_len` to control response length
4. Sends requests through the live CARA scheduler (which routes to real vLLM instances)
5. The CARA scheduler must be running with `--enable-predictor-feedback --feedback-sample-rate 1.0`
6. Each instance's `TrainingDataCollector` sidecar automatically records (request_features, schedule_state, actual_latency) in JSONL

**Prerequisites**: CARA server + all vLLM instances running. Start scheduler with:
```bash
python -m block.global_scheduler.cara.cara_serve \
    --scheduling random \
    --enable-predictor-feedback --feedback-sample-rate 1.0 \
    --model_config_path block/config/cara/model_deployment.json \
    --host_config block/config/host_configs.json \
    --host 0.0.0.0 --port 8200
```

QPS sweep: 0.5x to 2x cluster size (18 instances), stepping by 3:
```bash
# QPS values: 9, 12, 15, 18, 21, 24, 27, 30, 33, 36
for qps in 9 12 15 18 21 24 27 30 33 36; do
    python -m block.predictor.cara.offline_training.generate_latency_benchmark \
        --host <coordinator> --port 8200 \
        --num-prompts 20000 \
        --request-rate $qps \
        --input-len-dist lognormal --input-len-mean 256 --input-len-std 2.0 \
        --output-len-dist lognormal --output-len-mean 128 --output-len-std 1.5
done
# One full sweep = 10 QPS levels × 20k prompts = 200k requests
# Repeat sweeps over ~1 day to accumulate more data (tune num-prompts down if too slow)
```

**Training data directory structure:**
```
latency_training_data/
├── qps_9/          # Data collected at 9 QPS
│   ├── instance_3b_a30_host1.jsonl
│   ├── instance_7b_a30_host2.jsonl
│   └── ...
├── qps_12/
├── qps_15/
├── ...
├── qps_36/
└── all/            # Merged from all QPS levels
    ├── instance_3b_a30_host1.jsonl  (concatenated)
    └── ...
```

**Latency training strategies (explored at training time):**
1. **Single model, all QPS**: Train one XGBoost per (model, GPU-type) on `all/` merged data. QPS is not an input feature — the model learns from queue state features which implicitly encode load level.
2. **Per-QPS models**: Train separate models using the target QPS bucket + its nearest neighbors. Neighbor selection is symmetric around the target:
   - Target QPS=21 → use {15, 18, 21, 24} (2 below + self + 1 above = 4 buckets)
   - Target QPS=18 → use {15, 18, 21} (1 below + self + 1 above = 3 buckets, since 18 is itself a bucket)
   - Target QPS=9 → use {9, 12} (edge case, no lower neighbor)
   - General rule: include target ± 2 steps, clamped to available range
3. **Ablation**: Compare single-model vs per-QPS-range to see if mixing all QPS introduces bias or if queue state features are sufficient.

Key advantages:
- Unlimited data — repeat sweeps until accuracy saturates or time runs out (1-2 days)
- Covers diverse load conditions (9-36 QPS = 0.5x-2x cluster capacity)
- Per-instance data automatically collected by `TrainingDataCollector` on each node
- Captures real scheduling overhead and queuing effects through the live scheduler
- Organized by QPS for flexible training strategy exploration

Retrieve per-instance JSONL from all nodes:
```bash
parallel-ssh -h block/config/hosts "ls -la Block/training_data/cara/"
# scp from each node, organize into qps_* directories
```

## Step 5: Process Training Data (Day 1-2)

### 5a. Process broadcast data for length + quality

```bash
scp asdwb@d8545-10s10301.wisc.cloudlab.us:Block/experiment_output/cara_broadcast_training_data/broadcast_*.json ./data/cara/

# Score with similarity (fast, primary)
python -m block.predictor.cara.offline_training.prepare_training_data \
    --input data/cara/broadcast_*.json \
    --scoring-method similarity \
    --reference-model Qwen/Qwen2.5-72B \
    --require-all-models \
    --output data/cara/best-route_similarity_training.json

# Score with LLM judge (slower, for comparison/ablation)
python -m block.predictor.cara.offline_training.prepare_training_data \
    --input data/cara/broadcast_*.json \
    --scoring-method llm_judge \
    --judge-model Unbabel/M-Prometheus-7B \
    --require-all-models \
    --output data/cara/best-route_llm_judge_training.json
```

### 5b. 80/20 train/test split

**Create:** `block/predictor/cara/offline_training/split_training_data.py`
- Deterministic split by hash on `request_id` (stable across reruns)
- Outputs: `*_train.json` (80%) and `*_test.json` (20%)
- Same split used for all predictor types (length, quality)

Output format per request: `{prompt, input_len, models: {model_name: {output_length, quality_score, ttft, server_latency}}}`

## Step 6: Build Training Scripts (Days 2-4)

### 6a. KNN-Based Length & Quality Predictor (Primary)
**Create:** `block/predictor/cara/estimators/knn_estimator.py`

Shared KNN infrastructure for both length and quality prediction:
- Precompute prompt embeddings using sentence-transformers (MiniLM-L6-v2 or MPNet)
- Build FAISS/sklearn NearestNeighbors index over training set embeddings
- For a new prompt: embed → find top-k neighbors → aggregate per-model values

**Length prediction**: `predict_length(prompt, model_id) -> (mean_output_len, p95_output_len)`
- Aggregate: distance-weighted mean of neighbors' output_lengths per model
- Also return Q90/Q95 for conservative bounds (MaxSafe)

**Quality prediction**: `predict_quality(prompt, model_id) -> float`
- Aggregate: distance-weighted mean of neighbors' quality_scores per model

**Create:** `block/predictor/cara/offline_training/train_knn_estimator.py`
- Loads processed training data (80% split from Step 5b)
- Computes embeddings for all training prompts (batch encode)
- Builds index, saves: embeddings + per-model label arrays + index metadata
- CLI: `--input <training_json> --embedding-model <model> --output-dir <dir> --k <neighbors>`
- Reports on 20% test split: MAE for length, best-model selection accuracy for quality

### 6b. MLP-Based Length Predictor (Ablation)
**Create:** `block/predictor/cara/offline_training/train_mlp_length.py`

Per-model MLP on frozen embeddings:
- Reuse embeddings from 6a
- Architecture: embedding_dim → 256 → 128 → 1 (per-model or shared with model_id feature)
- Optional quantile regression head for Q90/Q95
- Train with MSE loss, Adam, lr=1e-3

### 6c. RoBERTa-Based Length Predictor (Ablation)
**Adapt:** `block/length_estimation/train_roberta.py`

**Create:** `block/predictor/cara/offline_training/train_roberta_length.py`
- Loads Cara training data format (not ShareGPT conversations)
- Trains one RoBERTa regression model per model (or shared with model embedding)
- Reuses simpletransformers approach from Block's existing code
- Adapts `generate_regression_dataframe` to read `{prompt, models: {model: {output_length}}}` format

### 6d. LLM-Based Length/Quality Predictor (Ablation)
**Create:** `block/predictor/cara/offline_training/train_llm_predictor.py`
- Finetune Qwen2.5-0.5B for length prediction and/or quality prediction
- Input: prompt text → Output: predicted length (regression) or quality score
- Use HuggingFace Trainer with LoRA for parameter-efficient finetuning
- Can predict for specific model via prompt template: `"Predict output length for model {model_id}: {prompt}"`
- Trainable on A30/A100 nodes after stopping vLLM

### 6e. XGBoost Latency Predictor (Primary)
**Create:** `block/predictor/cara/estimators/xgboost_predictor.py`
**Create:** `block/predictor/cara/offline_training/train_xgboost.py`

Per the LATENCY_PREDICTOR_DESIGN.md, 25 features:
- Service rate signals (3): ema_decode_tok_per_s, ema_prefill_tok_per_s, ema_decode_iter_ms
- Context distribution (3): decode_ctx_p50/p95/max
- Batch state (3): num_running, num_active_decode_seqs, num_waiting
- Backlog (2): pending_prefill/decode_tokens
- Scheduler knobs (3): token_budget, prefill_chunk_size, max_num_seqs
- Memory pressure (3): kv_cache_util, kv_free_blocks, kv_evictions_per_s
- Request features (4): prompt_tokens, predicted_output_tokens, req_decode_ctx_avg/max
- Derived (4): backlog_seconds, prefill_time_est, decode_time_est, ctx_relative_to_p95

Train separate model per (model, GPU-type) combination:
- `qwen2.5-3b_a30.xgb`, `qwen2.5-3b_p100.xgb`, `qwen2.5-7b_a30.xgb`, etc.
- Training data: synthetic microbenchmark JSONL from Step 4b
- Can regenerate more data if accuracy is insufficient (run more sweeps)
- Hyperparameters: GridSearchCV with MAE/MAPE scoring
- Save models with hash-based versioning (follow `vidur/execution_time_predictor/sklearn_execution_time_predictor.py` pattern)

**Two training modes** (from Step 4b strategy):
1. `--data-dir latency_training_data/all/` → single model trained on all QPS data
2. `--data-dir latency_training_data/ --target-qps 21` → model trained on target + nearest neighbors (e.g., {15,18,21,24})
Compare both; expect single-model to work well since queue state features implicitly encode load.

### 6f. LSTM Latency Predictor (Ablation)
**Create:** `block/predictor/cara/estimators/lstm_predictor.py` (PyTorch model)
**Create:** `block/predictor/cara/offline_training/train_lstm.py`

- Config: hidden=128, layers=2, dropout=0.1 (from existing `LSTMPredictorConfig`)
- Input: sliding window of (schedule_state features, request features) ordered by timestamp
- Target: actual E2E latency
- Per (model, GPU-type) models like XGBoost
- Purpose: evaluate whether sequential/temporal information improves over point-in-time features

### 6g. Roofline Latency Predictor (Analytical Baseline) — via NVIDIA AIConfigurator
**Create:** `block/predictor/cara/estimators/roofline_predictor.py`

Use [ai-dynamo/aiconfigurator](https://github.com/ai-dynamo/aiconfigurator) as the roofline baseline. This is NVIDIA's open-source tool that decomposes LLM inference into granular operations (GEMMs, attention, all-reduce) and estimates latency from hardware-level roofline analysis — more principled and convincing than a hand-rolled formula.

Integration approach:
- Install AIConfigurator as a dependency
- Wrap its performance estimation API to match our predictor interface
- Configure per (model, GPU-type): model path, GPU count, TP config
- AIConfigurator predicts TTFT, TPOT, and E2E latency given (input_len, output_len, batch_state)
- No training needed — purely analytical, calibrated from hardware specs
- Falls back to simple formula if AIConfigurator doesn't support our exact GPU (e.g., P100):
  ```
  latency = backlog_time + prefill_time + decode_time + iteration_overhead
  ```
  with static rates calibrated per (model, GPU-type) from observed throughput data.

### 6h. Linear Latency Predictor (Ablation Baseline)
**Create:** `block/predictor/cara/estimators/linear_predictor.py`
**Create:** `block/predictor/cara/offline_training/train_linear.py`

```
latency = a*prompt_tokens + b*predicted_output_tokens + c*num_waiting + d
```
4 coefficients via sklearn OLS per (model, GPU-type). Simplest ML baseline.

---

## Step 7: Train Models on GPU Nodes (Days 3-4)

After collecting sufficient data, stop vLLM instances and use the GPU nodes for training:

| Node Type | GPU | Train What |
|-----------|-----|-----------|
| d8545 (A100x4) | A100 | LLM-based predictor (Qwen-0.5B finetuning), RoBERTa finetuning |
| d7525 (A30x1) | A30 | KNN index building, MLP training, XGBoost training |
| c4130 (V100x4) | V100 | LSTM training, LLM judge scoring |
| c240g5 (P100) | P100 | Linear/roofline (CPU-only, minimal GPU needed) |

Distribute training across nodes via parallel-ssh:
```bash
# Example: train KNN on a30 nodes
parallel-ssh -h block/config/a30_hosts "cd Block && python -m block.predictor.cara.offline_training.train_knn_estimator \
    --input data/cara/best-route_similarity_train.json \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --output-dir models/cara/knn/"

# Example: train XGBoost on a30 nodes
parallel-ssh -h block/config/a30_hosts "cd Block && python -m block.predictor.cara.offline_training.train_xgboost \
    --input training_data/cara/ \
    --output-dir models/cara/xgboost/"
```

---

## Step 8: Predictor Evaluation, Analysis & Plots (Day 4)

After training all models, evaluate them on the held-out 20% test set and latency test data to decide which predictors to use for online scheduling.

### 8a. Unified Evaluation Script
**Create:** `block/predictor/cara/offline_training/evaluate_predictors.py`

Runs all trained predictors on test data and produces a comparison report:

**Length predictor evaluation** (on 20% test split):
- Metrics: MAE, MAPE, Acc@50, Acc@100, R² per model
- Predictors compared: KNN, MLP, RoBERTa, LLM (Qwen-0.5B)
- Per-model breakdown (3B, 7B, 14B, 72B)

**Quality predictor evaluation** (on 20% test split):
- Metrics: MAE of quality score, best-model selection accuracy (does predictor pick the right model?), Spearman rank correlation
- Predictors compared: KNN, LLM (Qwen-0.5B)
- Per-model breakdown

**Latency predictor evaluation** (on held-out QPS buckets or time-split):
- Metrics: MAE, MAPE, RMSE, R², p50/p95/p99 error, under-prediction rate
- Predictors compared: XGBoost (all-QPS), XGBoost (per-QPS-range), LSTM, Roofline (AIConfigurator), Linear
- Per (model, GPU-type) breakdown
- Also measure: inference latency of the predictor itself (μs for XGBoost vs ms for LSTM — important for scheduling overhead)

### 8b. Plotting Script
**Create:** `block/predictor/cara/offline_training/plot_predictor_comparison.py`

Generates paper-ready comparison figures:

| Figure | Content |
|--------|---------|
| Length accuracy bar chart | MAE per predictor per model |
| Quality accuracy bar chart | Best-model selection accuracy per predictor |
| Latency prediction scatter | Predicted vs actual E2E latency, per predictor type |
| Latency error CDF | CDF of absolute error, all predictors overlaid |
| Latency error by QPS | MAE vs QPS level for each latency predictor |
| Predictor overhead bar chart | Inference time (μs) per predictor type |
| All-QPS vs per-QPS ablation | MAE comparison for XGBoost training strategies |

### 8c. Decision Support
The evaluation output includes a summary table recommending which predictors to use:

```
=== Predictor Selection Recommendation ===
Length:  KNN (MAE=X) vs MLP (MAE=Y) vs RoBERTa (MAE=Z)  → Best: ___
Quality: KNN (acc=X%) vs LLM (acc=Y%)                     → Best: ___
Latency: XGBoost (MAE=X) vs LSTM (MAE=Y) vs Roofline (MAE=Z)
         XGBoost overhead: ~10μs, LSTM overhead: ~500μs    → Best: ___
```

This directly informs the next planning phase: which predictors to wire into `cara_serve.py` for online scheduling.

---

## Files to Create (16 total)

| File | Purpose |
|------|---------|
| `block/predictor/cara/estimators/__init__.py` | Package init |
| `block/predictor/cara/estimators/knn_estimator.py` | KNN-based length + quality predictor |
| `block/predictor/cara/estimators/xgboost_predictor.py` | XGBoost latency predictor |
| `block/predictor/cara/estimators/lstm_predictor.py` | LSTM latency predictor (PyTorch) |
| `block/predictor/cara/estimators/roofline_predictor.py` | AIConfigurator wrapper + fallback roofline |
| `block/predictor/cara/estimators/linear_predictor.py` | Linear regression latency baseline |
| `block/predictor/cara/offline_training/generate_latency_benchmark.py` | Synthetic microbenchmark generator + sender |
| `block/predictor/cara/offline_training/split_training_data.py` | 80/20 deterministic train/test split |
| `block/predictor/cara/offline_training/train_knn_estimator.py` | KNN training + index building |
| `block/predictor/cara/offline_training/train_mlp_length.py` | MLP length predictor training |
| `block/predictor/cara/offline_training/train_roberta_length.py` | RoBERTa length predictor (adapts Block code) |
| `block/predictor/cara/offline_training/train_llm_predictor.py` | Qwen-0.5B finetuning for length/quality |
| `block/predictor/cara/offline_training/train_xgboost.py` | XGBoost latency training |
| `block/predictor/cara/offline_training/train_lstm.py` | LSTM latency training |
| `block/predictor/cara/offline_training/evaluate_predictors.py` | Unified evaluation across all predictor types |
| `block/predictor/cara/offline_training/plot_predictor_comparison.py` | Paper-ready comparison figures |

## Files to Modify (0 for training infra)
No existing files need modification for the training scripts themselves. The CARA server already handles dummy prompts (uses only `prompt_len`/`max_tokens`, never inspects prompt text). The benchmark already supports `--dataset-name random`.

**Pre-flight verification needed (Day 1, before latency collection):**
Verify `/schedule_trace` and `/instance_stats` endpoints respond on deployed vLLM instances:
```bash
# Test on one A100 instance
curl http://<d8545-host>:8000/schedule_trace
curl http://<d8545-host>:8000/instance_stats
# These were added in commit f9de671bd on cara_v_11 branch
# If broken, fix in vLLM repo at /auto/homes/wd312/Code/llm/llm-infer/vllm (cara_v_11 branch)
```

## Dependency Graph

```
=== Phase 1: Data Collection + Training Infra (Days 1-4) ===

Day 1:    Verify /schedule_trace & /instance_stats on deployed instances
Day 1:    Broadcasting 20k prompts (Step 4a) — model-related data first
Day 1:    Build training scripts in parallel while broadcasting runs
           │
Day 1-2:  Process broadcast data + split 80/20 (Step 5)
Day 1-2:  Build KNN estimator (6a) ─── primary, implement first
Day 2:    Build MLP length (6b), RoBERTa length (6c) ─── ablation scripts
           │
Day 2:    Build synthetic data generator script (Step 4b)
Day 2-4:  Synthetic latency microbenchmark runs continuously (~2 days)
           │                        (runs in background on cluster)
Day 2-3:  Build XGBoost latency (6e) ─── primary latency predictor
Day 3:    Build Roofline/AIConfigurator (6g) + Linear (6h) ─── baselines
Day 3:    Build LSTM latency (6f) ─── ablation
Day 3:    Build LLM predictor (6d) ─── ablation script
           │
Day 3-4:  Stop vLLM, train ALL models on GPU nodes (Step 7)
           (latency data collection finishes end of Day 3)
Day 4:    Evaluate all predictors + comparison plots (Step 8)
Day 4:    Write paper Sections 1-2: predictor/estimator studies
           → Decision: which predictors to use for online scheduling

=== Phase 2: Scheduling + Evaluation (Days 5-10, replanned separately) ===

Day 5-10: Wire selected predictors into cara_serve.py
           Implement quality-aware scheduling (cara_scorer.py)
           Run end-to-end experiments
           Write remaining paper sections
```

## Verification Checklist
1. After Step 4a: Broadcast JSON has `broadcast_results` with 4 model entries per request, ~20k requests
2. After Step 4b: Per-instance JSONL files on each node with latency data at multiple QPS levels
3. After Step 5: Processed JSON has quality scores [0-1] and output_lengths; 80/20 split files exist
4. After Step 6a: KNN predictor loads index, predicts length/quality for sample prompt, reasonable values
5. After Step 6e: XGBoost model trains without errors, reports MAE/MAPE on validation set
6. After Step 7: All trained model checkpoints saved and loadable on local machine
