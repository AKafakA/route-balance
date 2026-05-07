#!/bin/bash
#SBATCH --job-name=rb_lora_pm
#SBATCH --partition=ampere
#SBATCH --account=KALYVIANAKI-SL3-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/anon/hpc-work/llm/RouteBalance/training_logs/lora_permodel_intr_%j.log

# INTR test: per-model LoRA — 2 jobs on 2 GPUs (judge × 3B, judge × 7B)
# Run with: sbatch --qos=INTR route_balance/exp/route_balance/csd3_lora_permodel_intr.sh

set -e
cd /rds/user/anon/hpc-work/llm/RouteBalance
export PYTHONPATH=/rds/user/anon/hpc-work/llm/RouteBalance:$PYTHONPATH
source /rds/user/anon/hpc-work/llm/RouteBalance/.venv/bin/activate
mkdir -p training_logs models/route_balance/lora_per_model

echo "=== Per-Model LoRA INTR Test — $(date) ==="
python3 -c "import transformers, torch, peft; print(f'tf={transformers.__version__}, torch={torch.__version__}, peft={peft.__version__}, GPUs={torch.cuda.device_count()}')"

TRAIN="data/route_balance/training_data/train_fixed.jsonl"
TEST="data/route_balance/training_data/test_fixed.jsonl"

# Test: 1 epoch, judge target, 2 models in parallel
echo "Starting judge×3B on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python3 -m route_balance.predictor.route_balance.offline_training.train_llm_predictor \
    --input $TRAIN --test-input $TEST \
    --target judge --mode per_model --epochs 1 \
    --target-models "Qwen/Qwen2.5-3B" \
    --batch-size 16 --max-length 1024 \
    --output-dir models/route_balance/lora_per_model/test_judge_3B \
    > training_logs/lora_pm_judge_3B.log 2>&1 &
PID1=$!

echo "Starting judge×7B on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python3 -m route_balance.predictor.route_balance.offline_training.train_llm_predictor \
    --input $TRAIN --test-input $TEST \
    --target judge --mode per_model --epochs 1 \
    --target-models "Qwen/Qwen2.5-7B" \
    --batch-size 16 --max-length 1024 \
    --output-dir models/route_balance/lora_per_model/test_judge_7B \
    > training_logs/lora_pm_judge_7B.log 2>&1 &
PID2=$!

echo "Waiting for PID $PID1 and $PID2..."
FAIL=0
wait $PID1 && echo "judge×3B: SUCCESS" || { echo "judge×3B: FAILED"; FAIL=$((FAIL+1)); }
wait $PID2 && echo "judge×7B: SUCCESS" || { echo "judge×7B: FAILED"; FAIL=$((FAIL+1)); }

echo ""
echo "=== Results ==="
for LOG in training_logs/lora_pm_judge_*.log; do
    echo "--- $(basename $LOG) ---"
    tail -10 "$LOG" 2>/dev/null
done

echo ""
echo "=== Per-Model LoRA INTR Complete — $(date), Failures: $FAIL/2 ==="
