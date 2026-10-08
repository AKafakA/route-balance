# Cara Experiment Status — March 2026

## Overview
Cara (Context-Aware Resource Allocation): Learned, quality-aware scheduler for heterogeneous LLM serving.
Target: SoCC/ATC/EuroSys submission, paper draft by end of May 2026.

## Current Status: Phase 2 In Progress

### Phase 1 (COMPLETE) — Data Collection Infrastructure
- Broadcasting pipeline operational across all model-GPU combinations
- 405k latency records collected (QPS 8-24, mixed dataset, 18 instances)
- Dataset v3: 7 datasets (reward_bench, code_ultra_feedback, beaver_tails, mix_instruct, lmsys, gsm8k, squad)
- 20k broadcasting run complete (93.6% survival rate)
- Full deployment scripts tested on CloudLab (A30, V100, P100, A100)

### Phase 2 (IN PROGRESS) — Model Training & Latency Predictor
- KNN/MLP quality predictors: MAE 0.10-0.14, best-model accuracy 64%
- ModernBERT bucket classifier: 14B best at 41.9% acc (epoch 3), needs retry with LR=1e-5
- Safety scoring: 3,532 harmful prompts fixed
- XGBoost TTFT/TPOT: needs retraining with new latency data
- 100k LMSYS broadcast complete (3.4GB, on GPU VM)

### Phase 3 (PLANNED) — Evaluation & Optimization
Need heterogeneous CloudLab cluster:
- 8× d7525 (A30) — Qwen2.5-7B, 14B
- 3× c4130 (V100) — Qwen2.5-3B, 7B
- 5× c240g5 (P100) — Qwen2.5-3B
- 2× d8545 (A100) — Qwen2.5-72B with TP

Experiments needed:
1. Baseline comparisons: 7 scheduling policies
2. Ablation: LPT sorting, batch scheduling, quality routing, load awareness
3. Sensitivity: prediction error, batch size, load level, heterogeneity
4. Overhead breakdown
5. Scalability (simulation)

### Phase 4 (PLANNED) — Paper & Publication

## Key Resources
- **Code**: `/home/wd312/Code/llm/Block/` (cara branch: cara-small-a100)
- **GPU VM**: wd312@gxp-l40s-2 (4x L40S 48GB, permanent)
- **VPS**: wd312@openclaw (code sync, SSH relay)
- **CSD3 HPC**: A100-80GB, 3000 GPU-hrs/quarter — for training + script testing
- **Data**: broadcast_v3_20k.json on GPU VM

## Dependencies on Block
- Cara extends Block's predictor infrastructure
- Block's vLLM fork (AKafakA/vllm) with schedule_trace endpoint needed
- Block experiment scripts (setup.sh, predictor deploy) reused for Cara
- Fixes from Block March 2026 session apply to Cara too (subshell, port mapping, etc.)

## Next Steps
1. **CSD3**: Set up venv, test vLLM + Cara infrastructure
2. **Training**: ModernBERT with LR=1e-5, XGBoost retrain on GPU VM or CSD3
3. **CloudLab reservation**: Heterogeneous cluster for Phase 3 evaluation
4. **Paper draft**: Start writing after initial evaluation results
5. **Deadline**: End of May 2026
