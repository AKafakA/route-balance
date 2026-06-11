#!/bin/bash
#SBATCH -J route_balance_qlora_ref
#SBATCH -A KALYVIANAKI-SL3-GPU
#SBATCH -p ampere
#SBATCH --qos=INTR
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=/rds/user/wd312/hpc-work/llm/Block/experiment_output/logs/qlora_refscore_%j.log

cd /rds/user/wd312/hpc-work/llm/Block
source .venv/bin/activate
export PYTHONPATH=/rds/user/wd312/hpc-work/llm/Block:$PYTHONPATH
export HF_HOME=/rds/user/wd312/hpc-work/hf_cache

TRAIN=data/route_balance/training_data_with_ref/train_fixed.jsonl
TEST=data/route_balance/training_data_with_ref/test_fixed.jsonl
OUT=models/route_balance/refscore_study/qwen05b_fused_reference_score

echo "=== Qwen-0.5B LoRA fused reference_score ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Check for checkpoint to resume from
RESUME_ARG=""
if [ -d "$OUT" ] && ls $OUT/checkpoint-* 1>/dev/null 2>&1; then
    LATEST_CKPT=$(ls -d $OUT/checkpoint-* | sort -V | tail -1)
    echo "Resuming from checkpoint: $LATEST_CKPT"
    RESUME_ARG="--resume-from-checkpoint $LATEST_CKPT"
fi

python route_balance/predictor/route_balance/offline_training/train_bert_predictor.py \
  --train-input $TRAIN --test-input $TEST \
  --encoder Qwen/Qwen2.5-0.5B --target reference_score \
  --epochs 3 --lr 2e-5 --batch-size 16 --max-length 1024 \
  --output-dir $OUT \
  --seed 42 $RESUME_ARG

echo "=== Done: $(date) ==="
