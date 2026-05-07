#!/bin/bash
#SBATCH -J route_balance_refscore
#SBATCH -A KALYVIANAKI-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=INTR
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --output=/rds/user/anon/hpc-work/llm/RouteBalance/experiment_output/logs/refscore_study_%j.log

cd /rds/user/anon/hpc-work/llm/RouteBalance
source .venv/bin/activate
export PYTHONPATH=/rds/user/anon/hpc-work/llm/RouteBalance:$PYTHONPATH
export HF_HOME=/rds/user/anon/hpc-work/hf_cache

TRAIN=data/route_balance/training_data_with_ref/train_fixed.jsonl
TEST=data/route_balance/training_data_with_ref/test_fixed.jsonl
ENCODER=answerdotai/ModernBERT-base
EPOCHS=5
LR=2e-5
BATCH=32
MAX_LEN=1024
OUT_BASE=models/route_balance/refscore_study

echo "=== ROUTE_BALANCE reference_score study ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

# GPU 0: fused model + per-model 7B + per-model 3B
(
export CUDA_VISIBLE_DEVICES=0
echo "[GPU0] Starting fused reference_score..."
python route_balance/predictor/route_balance/offline_training/train_bert_predictor.py \
  --input $TRAIN --test-input $TEST \
  --regression-model-name $ENCODER --target reference_score \
  --epochs $EPOCHS --lr $LR --batch-size $BATCH --max-length $MAX_LEN \
  --output-dir $OUT_BASE/modernbert_fused_reference_score \
  --seed 42

echo "[GPU0] Starting per-model 7B reference_score..."
python route_balance/predictor/route_balance/offline_training/train_bert_predictor.py \
  --input $TRAIN --test-input $TEST \
  --regression-model-name $ENCODER --target reference_score \
  --target-models "Qwen/Qwen2.5-7B" \
  --epochs $EPOCHS --lr $LR --batch-size $BATCH --max-length $MAX_LEN \
  --output-dir $OUT_BASE/modernbert_7b_reference_score \
  --seed 42

echo "[GPU0] Starting per-model 3B reference_score..."
python route_balance/predictor/route_balance/offline_training/train_bert_predictor.py \
  --input $TRAIN --test-input $TEST \
  --regression-model-name $ENCODER --target reference_score \
  --target-models "Qwen/Qwen2.5-3B" \
  --epochs $EPOCHS --lr $LR --batch-size $BATCH --max-length $MAX_LEN \
  --output-dir $OUT_BASE/modernbert_3b_reference_score \
  --seed 42

echo "[GPU0] Done: $(date)"
) &

# GPU 1: per-model 14B + per-model 72B
(
export CUDA_VISIBLE_DEVICES=1
echo "[GPU1] Starting per-model 14B reference_score..."
python route_balance/predictor/route_balance/offline_training/train_bert_predictor.py \
  --input $TRAIN --test-input $TEST \
  --regression-model-name $ENCODER --target reference_score \
  --target-models "Qwen/Qwen2.5-14B" \
  --epochs $EPOCHS --lr $LR --batch-size $BATCH --max-length $MAX_LEN \
  --output-dir $OUT_BASE/modernbert_14b_reference_score \
  --seed 42

echo "[GPU1] Starting per-model 72B reference_score..."
python route_balance/predictor/route_balance/offline_training/train_bert_predictor.py \
  --input $TRAIN --test-input $TEST \
  --regression-model-name $ENCODER --target reference_score \
  --target-models "Qwen/Qwen2.5-72B" \
  --epochs $EPOCHS --lr $LR --batch-size $BATCH --max-length $MAX_LEN \
  --output-dir $OUT_BASE/modernbert_72b_reference_score \
  --seed 42

echo "[GPU1] Done: $(date)"
) &

wait
echo "=== All done: $(date) ==="
ls -la $OUT_BASE/*/training_results.json 2>/dev/null
