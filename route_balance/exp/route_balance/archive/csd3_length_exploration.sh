#!/bin/bash
#SBATCH -J route_balance_length_exp
#SBATCH -A KALYVIANAKI-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=INTR
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/wd312/hpc-work/llm/Block/logs/route_balance_length_exp_%j.out
#SBATCH --error=/rds/user/wd312/hpc-work/llm/Block/logs/route_balance_length_exp_%j.err

# Quick exploration: test batch=64 + cosine scheduler for 10 epochs
# If it works + loss drops faster → submit full 300ep batch job
cd /rds/user/wd312/hpc-work/llm/Block
export PYTHONPATH=.
PYTHON=.venv/bin/python
TRAIN=data/route_balance/training_data/train_fixed.jsonl
TEST=data/route_balance/training_data/test_fixed.jsonl

echo "=========================================="
echo "Length prediction exploration — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=========================================="

# Test A: batch=64, lr=4e-5, cosine, 10 epochs
# Compare loss drop rate with our baseline (batch=16, lr=1e-5, polynomial)
echo "[$(date)] Test A: batch=64, lr=4e-5, cosine, 10ep (Qwen-14B only)..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target-models Qwen/Qwen2.5-14B \
    --target length --lr 4e-5 --epochs 10 --batch-size 64 --scheduler cosine --seed 42 --device cuda \
    --output-dir models/route_balance/exploration/modernbert_length_b64_lr4e5_cosine \
    || echo "[$(date)] Test A FAILED"

# Test B: batch=64, lr=4e-5, cosine_with_restarts, 10 epochs
echo "[$(date)] Test B: batch=64, lr=4e-5, cosine_with_restarts, 10ep..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target-models Qwen/Qwen2.5-14B \
    --target length --lr 4e-5 --epochs 10 --batch-size 64 --scheduler cosine_with_restarts --seed 42 --device cuda \
    --output-dir models/route_balance/exploration/modernbert_length_b64_lr4e5_cosine_restart \
    || echo "[$(date)] Test B FAILED"

echo "=========================================="
echo "[$(date)] DONE"
echo "=========================================="
