#!/bin/bash
#SBATCH --job-name=rb_mbert_long
#SBATCH --partition=ampere
#SBATCH --account=KALYVIANAKI-SL3-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=14:00:00
#SBATCH --output=/rds/user/anon/hpc-work/llm/RouteBalance/training_logs/modernbert_long_%j.log

# 4 ModernBERT regression jobs in parallel on 4 GPUs (main venv)
# J1: MSE 100ep, J2: Huber 100ep, J3: sMAPE 100ep, J14: Fused MSE 100ep
# Run with: sbatch route_balance/exp/route_balance/csd3_modernbert_long.sh

set -e
cd /rds/user/anon/hpc-work/llm/RouteBalance
export PYTHONPATH=/rds/user/anon/hpc-work/llm/RouteBalance:$PYTHONPATH
source /rds/user/anon/hpc-work/llm/RouteBalance/.venv/bin/activate
mkdir -p training_logs models/route_balance/encoder_length models/route_balance/fused

echo "=== ModernBERT Long Training — $(date) ==="
echo "4 jobs on 4 GPUs in parallel"
python3 -c "import transformers, torch; print(f'tf={transformers.__version__}, torch={torch.__version__}, GPUs={torch.cuda.device_count()}')"

TRAIN="data/route_balance/training_data/train_fixed.jsonl"
TEST="data/route_balance/training_data/test_fixed.jsonl"
COMMON="--save-total-limit 2 --early-stopping-patience 10 --seed 42 --lr 2e-5 --batch-size 16 --max-length 1024 --precision fp16"

# J1: ModernBERT MSE 100ep (GPU 0)
echo "Starting J1: ModernBERT MSE..."
CUDA_VISIBLE_DEVICES=0 python3 -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target-model "Qwen/Qwen2.5-7B" --target length \
    --loss-type mse --epochs 100 $COMMON \
    --output-dir models/route_balance/encoder_length/modernbert_mse_100ep \
    > training_logs/j1_modernbert_mse.log 2>&1 &
PID_J1=$!

# J2: ModernBERT Huber 100ep (GPU 1)
echo "Starting J2: ModernBERT Huber..."
CUDA_VISIBLE_DEVICES=1 python3 -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target-model "Qwen/Qwen2.5-7B" --target length \
    --loss-type huber --epochs 100 $COMMON \
    --output-dir models/route_balance/encoder_length/modernbert_huber_100ep \
    > training_logs/j2_modernbert_huber.log 2>&1 &
PID_J2=$!

# J3: ModernBERT sMAPE 100ep (GPU 2)
echo "Starting J3: ModernBERT sMAPE..."
CUDA_VISIBLE_DEVICES=2 python3 -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name answerdotai/ModernBERT-base \
    --target-model "Qwen/Qwen2.5-7B" --target length \
    --loss-type smape --epochs 100 $COMMON \
    --output-dir models/route_balance/encoder_length/modernbert_smape_100ep \
    > training_logs/j3_modernbert_smape.log 2>&1 &
PID_J3=$!

# J14: Fused ModernBERT MSE 100ep (GPU 3)
echo "Starting J14: Fused ModernBERT MSE..."
CUDA_VISIBLE_DEVICES=3 python3 -m route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --encoder-name answerdotai/ModernBERT-base \
    --target length --epochs 100 --lr 1e-5 \
    --save-total-limit 2 --early-stopping-patience 10 --seed 42 \
    --batch-size 16 --max-length 1024 --precision fp16 \
    --output-dir models/route_balance/fused/modernbert_mse_100ep \
    > training_logs/j14_fused_modernbert_mse.log 2>&1 &
PID_J14=$!

echo "All 4 jobs started. PIDs: J1=$PID_J1 J2=$PID_J2 J3=$PID_J3 J14=$PID_J14"
echo "Waiting..."

# Wait for all and report
FAIL=0
for PID_NAME in "J1:$PID_J1" "J2:$PID_J2" "J3:$PID_J3" "J14:$PID_J14"; do
    NAME="${PID_NAME%%:*}"
    PID="${PID_NAME#*:}"
    if wait $PID; then
        echo "$NAME (PID $PID): SUCCESS"
    else
        echo "$NAME (PID $PID): FAILED (exit $?)"
        FAIL=$((FAIL+1))
    fi
done

echo ""
echo "=== Results ==="
for LOG in training_logs/j{1,2,3,14}_*.log; do
    echo "--- $(basename $LOG) ---"
    tail -5 "$LOG" 2>/dev/null
done

echo ""
echo "=== ModernBERT Long Training Complete — $(date) ==="
echo "Failures: $FAIL/4"
