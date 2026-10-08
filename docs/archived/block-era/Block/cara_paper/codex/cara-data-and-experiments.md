# CARA Data Pipeline and Experiment Plan (Codex)

This document captures the current end-to-end data collection, training, and experimental plan for CARA based on the latest code and paper draft. It is intended for future sessions to quickly reproduce and iterate.

## Context and Decisions

- Prior issue: safety-heavy prompts (e.g., BeaverTails) dominated earlier collections and broke similarity-based scoring when reference models refused harmful prompts. Also observed a repetition/cursor bug on Qwen 2.5 32B; we will exclude 32B.
- Fixes and updates:
  - Balanced dataset pool with well-known, diverse tasks; equal per-dataset ratios by default.
  - Token-length filtering before allocation using a real tokenizer (default `Qwen/Qwen2.5-3B`), no truncation.
  - Per-run distribution reporting and optional JSON stats.
  - Dropped `self_instruct` and `nq_open` from the collector; added ARC (Challenge+Easy) for capacity.

### Dataset Revision (March 2026)

Based on broadcasting ablation studies (stratified 499 prompts, max_output=1024, max_total=2048, `--require-all-models`), several datasets had poor survival rates:

| Dataset | Survival Rate | Action | Reason |
|---------|--------------|--------|--------|
| ai2_arc | 0% | **Removed** | Single-letter answers (2 tokens), always filtered as <3 tokens |
| cnn_dailymail | 64.9% | **Removed** | Summarization produces long outputs, high truncation |
| ultrachat | 73.7% | **Replaced with lmsys** | No response labels for pre-filtering; replaced by LMSYS-Chat-1M which has known response lengths |
| squad | 47.4% | Kept (under review) | Ground truth is short extractive answers, but LLMs generate verbose explanations |

**New dataset: LMSYS-Chat-1M** (`lmsys/lmsys-chat-1m`):
- 1M real user-LLM conversations with metadata
- Has `language` field for filtering (English default = 776k conversations)
- Has known assistant response text — enables pre-filtering by response length
- Requires gated dataset access (HF token: `cara`, see `cara_paper/claude/tools.md`)

**New CLI flags for conversation dataset filtering:**
- `--chat-lang` — Filter by language (default: English, "all" to disable)
- `--chat-max-response-tokens` — Filter prompts whose known response exceeds N tokens
- `--chat-max-total-tokens` — Filter prompts where prompt+response exceeds N tokens
- These only affect datasets with a `response` column (currently only lmsys)

**Best generation parameters** (from sweep_broadcasting.sh ablation):
- `temperature=0.0, frequency_penalty=1.2, repetition_penalty=1.0`
- This config minimizes repetition while keeping deterministic outputs

## Data Collection (block/data/collect_data.py)

- **Current default datasets** (7 total, as of March 2026):
  - `reward_bench` (AllenAI): `--reward-bench-split {filtered|raw}` and optional `--reward-bench-prefix` filters.
  - `code_ultra_feedback` (programming instructions).
  - `mix_instruct` (general instructions + inputs).
  - `beaver_tails` (harmful prompts; safety control, keep small share).
  - `lmsys` (LMSYS-Chat-1M; first user→assistant turn; yields `{id, prompt, response}`).
  - `gsm8k` (math word problems; prompts ask for a numeric final answer).
  - `squad` (QA with context; context included in prompt, filtered by tokens).

- **Legacy datasets** (still have loaders but removed from defaults):
  - `ultrachat` (replaced by lmsys — no response labels for pre-filtering).
  - `cnn_dailymail` (removed — 64.9% survival, high truncation from long summaries).
  - `ai2_arc` (removed — 0% survival, single-letter answers always filtered as <3 tokens).

- Key CLI flags:
  - `--datasets` names (defaults to all 7 current datasets).
  - `--total-n` target total samples (post-filter allocation enforces this).
  - `--ratios` relative per-dataset ratios (defaults to equal weights).
  - `--max-prompt-length` tokenizer-based max tokens (default 700); prompts exceeding this are discarded.
  - `--tokenizer` tokenizer for token counting (default `Qwen/Qwen2.5-3B`).
  - `--chat-lang` language filter for conversation datasets (default: English; "all" disables).
  - `--chat-max-response-tokens` filter conversation prompts by known response length.
  - `--chat-max-total-tokens` filter conversation prompts by prompt+response total.
  - `--reward-bench-split {filtered,raw}` and optional repeated `--reward-bench-prefix`.
  - Output: `--use-json` for array, otherwise JSONL; `--stats-output` writes capacities and final distribution.

- Behavior:
  - Filter by prompt token length first for each dataset.
  - For conversation datasets (lmsys): optionally filter by response token length.
  - Compute capacities post-filter; allocate by ratios; reallocate only when a dataset runs out.

- Example (v3 dataset, equal ratios, 20k, with response-length filtering):
```
python -m block.data.collect_data \
  --datasets reward_bench code_ultra_feedback beaver_tails mix_instruct lmsys gsm8k squad \
  --total-n 20000 \
  --max-prompt-length 700 \
  --tokenizer Qwen/Qwen2.5-3B \
  --chat-lang English \
  --chat-max-response-tokens 1024 \
  --chat-max-total-tokens 2048 \
  --stats-output data/cara/best-route.stats.json \
  -o data/cara/best-route-v3.jsonl
```

- **(retired Apr 15 PM revision 2)** Example (500-sample test set for ablation):
```
# RETIRED: produced data/cara/best-route-v3-test-500.jsonl which overlapped 289/500 prompts
# with the predictor's training split (data integrity audit, see design §5.14).
# REPLACEMENT: cara_paper/smoke_test_apr_13/scripts/resample_clean_bench.py samples 500 clean
# prompts from data/cara/scored/test_scored_filtered.jsonl (the held-out HF split with full
# 3B+7B deepeval coverage). Output: data/cara/best-route-v3-test-500-clean.jsonl. This is the
# canonical bench file for Phase 1 + Phase 2.
```

Expected distribution: ~14.2% per dataset (7 datasets, equal weights). LMSYS provides 776k English conversations.

## Training Data Preparation

**Script**: `block/predictor/cara/offline_training/prepare_benchmark_data.py` (unified preprocessing)

Computes per-model quality scores (similarity + LLM judge) and output lengths from broadcasting data. Latency fields are excluded — they are meaningless from broadcasting (concurrent contention across 4 models). Latency data is collected separately via `generate_latency_benchmark.py`.

**Recommended command** (both similarity + LLM judge, requires GPU, ~6.5h for 20k with --disable-rationale):
```bash
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data/cara/broadcast_v3_20k.json \
  --scoring-method all \
  --reference-model “Qwen/Qwen2.5-72B” \
  --judge-models Qwen/Qwen2.5-7B-Instruct \
  --include-response \
  --device cuda \
  --batch-size 32 \
  --disable-rationale \
  --dataset-name cara_v3
```

**With rationale** (higher quality scores but ~25h for 20k):
```bash
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data/cara/broadcast_v3_20k.json \
  --scoring-method all \
  --reference-model “Qwen/Qwen2.5-72B” \
  --judge-models Qwen/Qwen2.5-7B-Instruct \
  --include-response \
  --device cuda \
  --batch-size 32 \
  --dataset-name cara_v3
```

**Similarity only** (CPU-friendly, fast, ~2 min for 20k):
```bash
python -m block.predictor.cara.offline_training.prepare_benchmark_data \
  --input data/cara/broadcast_v3_20k.json \
  --scoring-method similarity \
  --reference-model “Qwen/Qwen2.5-72B”
```

**LLM Judge Performance** (tested on 50-prompt dev set, 196 model-response pairs):
| Config | Time | Per-pair | Projected 20k |
|--------|------|----------|---------------|
| fp16 + eager + batch=8 + rationale | 4m42s | 1.44s | ~32h |
| bf16 + sdpa + batch=32 + rationale | 3m41s | 1.13s | ~25h |
| bf16 + sdpa + batch=32 + no rationale | 57s | 0.29s | ~6.5h |

No-rationale vs rationale score correlation: 0.81-0.93 per model, MAE 0.06-0.10 on [0,1] scale.
100% parse success rate (Stage 1 number extraction) in all configurations.

**Output schema** (model-related fields only):
```json
{
  “request_id”: “...”, “prompt”: “...”, “input_len”: 128,
  “models”: {
    “Qwen/Qwen2.5-3B”: {
      “output_length”: 256, “response”: “...”,
      “similarity_score”: 0.85,
      “llm_judge_scores”: {“Qwen_Qwen2.5-7B-Instruct”: 7},
      “compression_ratio”: 0.45, “is_truncated”: false
    }
  }
}
```

**LLM Judge**: Best model is `Qwen/Qwen2.5-7B-Instruct` (Spearman ρ=0.843 on feedbackQA human ratings, 1-4 scale). Uses 3-stage score parsing in `llm_judge_scorer.py`. Replaced earlier `Unbabel/M-Prometheus-7B` which had poor parsing accuracy.

**Two-mode workflow**:
1. **Full mode** (judge model selection): Use `validate_judge_models.py` with rationale enabled to compare candidate judges against human ratings → select best model once
2. **Fast mode** (large-scale data tagging): Use `--disable-rationale --batch-size 32` for production scoring. Validated on 50-prompt dev set: 0 failures, 0.81-0.93 correlation with full mode, 5x faster
3. **Justification**: Small-scale test (196 pairs) showed 100% parse rate in both modes with high score correlation, confirming fast mode is reliable for training data generation

**Splits**: 80/20 by deterministic hash on `request_id` (`split_training_data.py`). After preprocessing, upload to HuggingFace: `hf upload asdwb/cara_model_estimator . --repo-type=dataset`

## Estimators for Routing

Quality (per model):
- KNN baseline: Precompute prompt embeddings (e.g., MPNet/MiniLM). For a new request, top-k neighbors → average per-model quality (distance-weighted). Fast to “train”, easy to refresh.
- Lightweight encoder head: Frozen embeddings + small MLP/GBDT per model (or a multi-head). Better generalization with small inference overhead.
- Evaluate: best-model selection accuracy vs 72B, monotonicity by model size, per-source breakdown.

Length (per model):
- KNN: Reuse embeddings; top-k neighbor output lengths → return E[output_len] (mean/median) and MaxSafe[output_len] (Q90–Q95) for constraints.
- Regressor: Frozen embeddings + small MLP/GBDT; optionally quantile regression for Q90/Q95. Add guardrails (monotonic by input_len, rolling bias correction, piecewise-linear fallback).

Latency (per model × GPU tier):
- Recommended: Parametric/GBDT with features: model_id, gpu_type, input_len, E[output_len], active_seq_count, simple state histogram of running lengths. Target TTFT/TPOT/server_latency. Fit Q90 for conservative bounds.
- Synthetic microbench data: sample input/output lengths from empirical distributions, sweep QPS/concurrency; collect per tier (A100/V100/A30/P100).
- Sequence/LSTM or KNN-on-state are optional ablations.

Integration:
- Use E[output_len] for cost/latency scoring; MaxSafe[output_len] to enforce 1024-token caps per model.
- Combine predicted latencies with quality in the scheduler’s scoring function (see `cara-objective-design.md`).

## Deployment and Instance Mix

- Drop Qwen 2.5 32B (repetition/cursor bug). Use: 72B×2, 14B×3, 7B×5, 3B×8.
- CloudLab mapping:
  - 72B on d8515 (A100×4): 1 instance per node, TP=4 in-node.
  - 14B on c4130 (V100×4): 1 per node; adjust TP/precision per VRAM.
  - 7B on d7525 (A30×1): 5 nodes.
  - 3B on 3×d7525 (A30×1) + 5×c240g5 (P100×1): use vLLM 0.4 and disable FlashAttention on P100.
- Keep tensor-parallel within node; pin vLLM versions per tier.

## Experiment Plan (superseded — current plan = Apr 15 PM **revision 3**)

**This SOSP-era plan is retired.** The current evaluation plan is in `cara_paper/codex/apr15_baseline_comparison_design.md` (revision 3). Summary:

- **Baselines (main table):** Cara-full + vllm_sr-wrapper + RouteLLM-wrapper(mf) + Avengers-Pro-wrapper + BEST-Route-wrapper. Wrappers paired with RR + shortest_queue dispatch.
- **Ablations:** integrated Cara ingredients (LPT toggle, filter on/off, per-term zero weights including w_balance, scoring vs naive assignment, quality_greedy_ablation).
- **Supporting studies:** routing-signal swap + filter swap inside fixed Cara shell.
- **Loads (rev-3):** Phase 1 (Apr 15-17 4-node) = preflight only at a single fixed λ = 10 req/s — metric/monitor/script validation, **NOT paper numbers**. Phase 2 (Apr 26 - May 6 18-node) = **paper-facing primary run with arrival-rate λ as the primary load axis** (pilot-justified grid ≈ {4, 8, 12, 16, 20, 24, 32}; sweep-scope layering in design §4.6).
- **Metrics:** Pareto on cost / quality / mean E2E latency (continuous); goodput @ {balanced, strict} SLO + throughput (constraint + raw work); balance CV time-series (scheduler-specific); routing overhead breakdown.
- **Quality:** pre-computed deepeval lookup (RouterBench-style).
- **Scale:** Phase 1 (preflight) = 4× A30 (Apr 15-17); Phase 2 (paper) = 18-node heterogeneous (Apr 26 - May 6).

## Repro Steps (Quick)

1) Collect balanced prompts (equal ratios, 20k) with token cap.
2) Prepare training sets: similarity (with and without safety) and LLM-judge.
3) Split 80/20 (deterministic).
4) Train estimators: quality (KNN + small head), length (KNN/regressor), latency (GBDT per model×tier).
5) Deploy 72B×2, 14B×3, 7B×5, 3B×8; pin per-tier vLLM.
6) Run experiments: baselines, loads, and scale; collect metrics.
7) Analyze: routing accuracy, quality-vs-latency trade-offs, efficiency.

## Change Log (Key)

- **March 2026**: Replaced ultrachat with LMSYS-Chat-1M (776k English conversations with response labels). Removed ai2_arc (0% survival) and cnn_dailymail (64.9% survival). Added `--chat-lang`, `--chat-max-response-tokens`, `--chat-max-total-tokens` flags. Best generation params: temp=0.0, freq_penalty=1.2, rep_penalty=1.0. Generated v3 test dataset (500 prompts, 7 datasets, even distribution).
- Added datasets: `ultrachat`, `squad`, `cnn_dailymail`, `ai2_arc` (Challenge+Easy), `gsm8k`.
- Removed datasets: `self_instruct`, `nq_open`.
- Filtering-first by tokenizer tokens; no prompt truncation; distribution reporting with optional JSON.
- Dropped Qwen 2.5 32B from experiments due to repetition/cursor bug.

