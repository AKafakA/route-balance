#!/bin/bash
#SBATCH -J route_balance_training
#SBATCH -A KALYVIANAKI-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/rds/user/anon/hpc-work/llm/RouteBalance/logs/route_balance_training_%j.out
#SBATCH --error=/rds/user/anon/hpc-work/llm/RouteBalance/logs/route_balance_training_%j.err

# Remaining training tasks for RouteBalance predictor models
# Submit: sbatch route_balance/exp/route_balance/csd3_remaining_training.sh

cd /rds/user/anon/hpc-work/llm/RouteBalance
export PYTHONPATH=.
PYTHON=.venv/bin/python
TRAIN=data/route_balance/training_data/train_fixed.jsonl
TEST=data/route_balance/training_data/test_fixed.jsonl

mkdir -p logs
mkdir -p data/route_balance/latency_data/tagged
ln -sf $(pwd)/data/route_balance/latency_data/all/latency_train_tagged.jsonl data/route_balance/latency_data/tagged/latency_train.jsonl
ln -sf $(pwd)/data/route_balance/latency_data/all/latency_test_tagged.jsonl data/route_balance/latency_data/tagged/latency_test.jsonl

echo "=========================================="
echo "CSD3 Remaining Training — $(date)"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

# 1. ModernBERT fused log-transform 100ep (~5h)
echo "[$(date)] 1/4: ModernBERT fused length log-transform 100ep..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target length --lr 1e-5 --epochs 100 \
    --early-stopping-patience 10 \
    --log-transform --device cuda --seed 42 \
    --output-dir models/route_balance/long_study/modernbert_fused_length_logtransform \
    || echo "[$(date)] Log-transform FAILED"

# 2. ModernBERT fused bucket CE 100ep (~5h)
echo "[$(date)] 2/4: ModernBERT fused length_bucket 100ep..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target length_bucket --lr 1e-5 --epochs 100 \
    --early-stopping-patience 10 \
    --device cuda --seed 42 \
    --output-dir models/route_balance/long_study/modernbert_fused_length_bucket \
    || echo "[$(date)] Bucket FAILED"

# 3. ModernBERT fused judge_class (safety-aware labels)
echo "[$(date)] 3/4: ModernBERT fused judge_class..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target judge_class --lr 1e-5 --epochs 5 --seed 42 --device cuda \
    --output-dir models/route_balance/study/modernbert_fused_judge_class \
    || echo "[$(date)] Judge fused FAILED"

# 4. ModernBERT 7B judge_class
echo "[$(date)] 4/4: ModernBERT 7B judge_class..."
$PYTHON -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target-models Qwen/Qwen2.5-7B \
    --target judge_class --lr 1e-5 --epochs 5 --seed 42 --device cuda \
    --output-dir models/route_balance/study/modernbert_7b_judge_class \
    || echo "[$(date)] Judge 7B FAILED"

echo "=========================================="
echo "[$(date)] ALL DONE"
echo "=========================================="
