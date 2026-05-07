#!/bin/bash
#SBATCH -A KALYVIANAKI-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=INTR
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --output=logs/fused_intr_%j.log
#SBATCH --error=logs/fused_intr_%j.log
#SBATCH --job-name=route_balance_fused

# All 5 true fused multi-head ModernBERT models in 1 INTR job (2 GPUs)
# GPU 0: length_bucket + length_logtransform + similarity (~45 min)
# GPU 1: judge_class + reference_score (~30 min)
# Run: sbatch route_balance/exp/route_balance/csd3_fused_intr.sh

set -o pipefail
cd /rds/user/anon/hpc-work/llm/RouteBalance
export PYTHONPATH=.
export HF_HOME=/rds/user/anon/hpc-work/hf_cache
PYTHON=.venv/bin/python

TRAIN=data/route_balance/training_data/train_fixed.jsonl
TEST=data/route_balance/training_data/test_fixed.jsonl
TRAIN_REF=data/route_balance/training_data_with_ref/train_fixed.jsonl
TEST_REF=data/route_balance/training_data_with_ref/test_fixed.jsonl
ENCODER=answerdotai/ModernBERT-base
EPOCHS=5
LR=1e-5
BATCH=16  # match GPU VM per-model runs for fair comparison
MAX_LEN=1024

mkdir -p logs

echo "=== ROUTE_BALANCE Fused Training (2 GPUs) ==="
echo "Start: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# GPU 0: length_bucket + length_logtransform + similarity
(
export CUDA_VISIBLE_DEVICES=0
echo "[GPU0 $(date)] Fused length_bucket..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --encoder-name $ENCODER --target length_bucket \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH --max-length $MAX_LEN \
    --output-dir models/route_balance/length_bucket/deploy \
    --seed 42

echo "[GPU0 $(date)] Fused length (log-transform)..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --encoder-name $ENCODER --target length --log-transform \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH --max-length $MAX_LEN \
    --output-dir models/route_balance/length_regression/deploy \
    --seed 42

echo "[GPU0 $(date)] Fused similarity..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --encoder-name $ENCODER --target similarity \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH --max-length $MAX_LEN \
    --output-dir models/route_balance/quality/similarity/ablation/fused \
    --seed 42

echo "[GPU0 $(date)] Done"
) &

# GPU 1: judge_class + reference_score
(
export CUDA_VISIBLE_DEVICES=1
echo "[GPU1 $(date)] Fused judge_class..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --encoder-name $ENCODER --target judge_class \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH --max-length $MAX_LEN \
    --output-dir models/route_balance/quality/judge/ablation/fused \
    --seed 42

echo "[GPU1 $(date)] Fused reference_score..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor \
    --input $TRAIN_REF --test-input $TEST_REF \
    --encoder-name $ENCODER --target reference_score \
    --epochs $EPOCHS --lr $LR --batch-size $BATCH --max-length $MAX_LEN \
    --output-dir models/route_balance/quality/reference_score/ablation/fused \
    --seed 42

echo "[GPU1 $(date)] Done"
) &

wait
echo "=== All Complete: $(date) ==="
