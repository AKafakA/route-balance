#!/bin/bash
#SBATCH --job-name=rb_fused_bkt
#SBATCH --partition=ampere
#SBATCH --account=KALYVIANAKI-SL3-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/wd312/hpc-work/llm/Block/training_logs/fused_bucket_intr_%j.log

# Short fused bucket jobs: RoBERTa + DeBERTa (2 GPUs in parallel)
# J6: Fused RoBERTa bucket 5ep, J11: Fused DeBERTa bucket 5ep
# Run with: sbatch --qos=INTR route_balance/exp/route_balance/csd3_fused_bucket_intr.sh

set -e
cd /rds/user/wd312/hpc-work/llm/Block
export PYTHONPATH=/rds/user/wd312/hpc-work/llm/Block:$PYTHONPATH
source /rds/user/wd312/hpc-work/venv_roberta/bin/activate
mkdir -p training_logs models/route_balance/fused

echo "=== Fused Bucket INTR — $(date) ==="

TRAIN="data/route_balance/training_data/train_fixed.jsonl"
TEST="data/route_balance/training_data/test_fixed.jsonl"

# J6: Fused RoBERTa bucket 5ep (GPU 0)
echo "Starting J6: Fused RoBERTa bucket..."
CUDA_VISIBLE_DEVICES=0 python3 -m route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --encoder-name roberta-base \
    --target length_bucket --epochs 5 --lr 1e-5 \
    --max-length 512 --precision fp16 \
    --save-total-limit 2 --seed 42 \
    --output-dir models/route_balance/fused/roberta_bucket \
    > training_logs/j6_fused_roberta_bucket.log 2>&1 &
PID_J6=$!

# J11: Fused DeBERTa bucket 5ep (GPU 1)
echo "Starting J11: Fused DeBERTa bucket..."
CUDA_VISIBLE_DEVICES=1 python3 -m route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --encoder-name microsoft/deberta-v3-base \
    --target length_bucket --epochs 5 --lr 1e-5 \
    --max-length 512 --precision bf16 \
    --save-total-limit 2 --seed 42 \
    --output-dir models/route_balance/fused/deberta_bucket \
    > training_logs/j11_fused_deberta_bucket.log 2>&1 &
PID_J11=$!

echo "Waiting for J6 ($PID_J6) and J11 ($PID_J11)..."
wait $PID_J6 && echo "J6: SUCCESS" || echo "J6: FAILED"
wait $PID_J11 && echo "J11: SUCCESS" || echo "J11: FAILED"

echo ""
echo "=== Results ==="
tail -5 training_logs/j6_fused_roberta_bucket.log 2>/dev/null
tail -5 training_logs/j11_fused_deberta_bucket.log 2>/dev/null
echo ""
echo "=== Fused Bucket INTR Complete — $(date) ==="
