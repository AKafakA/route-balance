#!/bin/bash
#SBATCH --job-name=rb_roberta_long
#SBATCH --partition=ampere
#SBATCH --account=KALYVIANAKI-SL3-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=/rds/user/wd312/hpc-work/llm/Block/training_logs/roberta_deberta_long_%j.log

# 4 RoBERTa/DeBERTa regression jobs in parallel on 4 GPUs (roberta venv)
# J4: RoBERTa MSE 100ep, J5: RoBERTa log 20ep, J12: DeBERTa MSE 100ep, J13: DeBERTa log 20ep
# Run with: sbatch route_balance/exp/route_balance/csd3_roberta_deberta_long.sh

set -e
cd /rds/user/wd312/hpc-work/llm/Block
export PYTHONPATH=/rds/user/wd312/hpc-work/llm/Block:$PYTHONPATH
source /rds/user/wd312/hpc-work/venv_roberta/bin/activate
mkdir -p training_logs models/route_balance/encoder_length

echo "=== RoBERTa + DeBERTa Long Training — $(date) ==="
echo "4 jobs on 4 GPUs in parallel"
python3 -c "import transformers, torch; print(f'tf={transformers.__version__}, torch={torch.__version__}, GPUs={torch.cuda.device_count()}')"

TRAIN="data/route_balance/training_data/train_fixed.jsonl"
TEST="data/route_balance/training_data/test_fixed.jsonl"
COMMON_ROBERTA="--save-total-limit 2 --early-stopping-patience 10 --seed 42 --lr 2e-5 --batch-size 16 --max-length 512 --precision fp16"
COMMON_DEBERTA="--save-total-limit 2 --early-stopping-patience 10 --seed 42 --lr 2e-5 --batch-size 16 --max-length 512 --precision bf16"

# J4: RoBERTa MSE 100ep (GPU 0)
echo "Starting J4: RoBERTa MSE..."
CUDA_VISIBLE_DEVICES=0 python3 -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name roberta-base \
    --target-model "Qwen/Qwen2.5-7B" --target length \
    --loss-type mse --epochs 100 $COMMON_ROBERTA \
    --output-dir models/route_balance/encoder_length/roberta_mse_100ep \
    > training_logs/j4_roberta_mse.log 2>&1 &
PID_J4=$!

# J5: RoBERTa log-transform 20ep (GPU 1)
echo "Starting J5: RoBERTa log-transform..."
CUDA_VISIBLE_DEVICES=1 python3 -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name roberta-base \
    --target-model "Qwen/Qwen2.5-7B" --target length \
    --log-transform --epochs 20 $COMMON_ROBERTA \
    --output-dir models/route_balance/encoder_length/roberta_log_20ep \
    > training_logs/j5_roberta_log.log 2>&1 &
PID_J5=$!

# J12: DeBERTa MSE 100ep (GPU 2)
echo "Starting J12: DeBERTa MSE..."
CUDA_VISIBLE_DEVICES=2 python3 -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name microsoft/deberta-v3-base \
    --target-model "Qwen/Qwen2.5-7B" --target length \
    --loss-type mse --epochs 100 $COMMON_DEBERTA \
    --output-dir models/route_balance/encoder_length/deberta_mse_100ep \
    > training_logs/j12_deberta_mse.log 2>&1 &
PID_J12=$!

# J13: DeBERTa log-transform 20ep (GPU 3)
echo "Starting J13: DeBERTa log-transform..."
CUDA_VISIBLE_DEVICES=3 python3 -m route_balance.predictor.route_balance.offline_training.train_bert_predictor \
    --input $TRAIN --test-input $TEST \
    --regression-model-name microsoft/deberta-v3-base \
    --target-model "Qwen/Qwen2.5-7B" --target length \
    --log-transform --epochs 20 $COMMON_DEBERTA \
    --output-dir models/route_balance/encoder_length/deberta_log_20ep \
    > training_logs/j13_deberta_log.log 2>&1 &
PID_J13=$!

echo "All 4 jobs started. PIDs: J4=$PID_J4 J5=$PID_J5 J12=$PID_J12 J13=$PID_J13"
echo "Waiting..."

FAIL=0
for PID_NAME in "J4:$PID_J4" "J5:$PID_J5" "J12:$PID_J12" "J13:$PID_J13"; do
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
for LOG in training_logs/j{4,5,12,13}_*.log; do
    echo "--- $(basename $LOG) ---"
    tail -5 "$LOG" 2>/dev/null
done

echo ""
echo "=== RoBERTa + DeBERTa Long Training Complete — $(date) ==="
echo "Failures: $FAIL/4"
