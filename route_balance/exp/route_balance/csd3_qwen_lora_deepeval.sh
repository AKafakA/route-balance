#!/bin/bash
#SBATCH -J rb_lora_de
#SBATCH -A KALYVIANAKI-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

# Qwen LoRA fused + per-model training on deepeval + ref_score targets
# Runs sequentially on 1 GPU

cd /rds/user/wd312/hpc-work/llm/Block
source .venv/bin/activate
export PYTHONPATH=/rds/user/wd312/hpc-work/llm/Block:$PYTHONPATH

DATA=/rds/user/wd312/hpc-work/llm/Block/data/route_balance/train_scored_filtered.jsonl
OUTBASE=/rds/user/wd312/hpc-work/llm/Block/models/route_balance

# Fused LoRA: judge
echo "=== Fused LoRA: judge ==="
python -u -m route_balance.predictor.route_balance.offline_training.train_llm_predictor \
    --input $DATA \
    --base-model Qwen/Qwen2.5-0.5B \
    --target judge --mode fused --epochs 3 --device cuda \
    --output-dir $OUTBASE/qwen_lora_fused/judge

# Fused LoRA: reference_score
echo "=== Fused LoRA: reference_score ==="
python -u -m route_balance.predictor.route_balance.offline_training.train_llm_predictor \
    --input $DATA \
    --base-model Qwen/Qwen2.5-0.5B \
    --target reference_score --mode fused --epochs 3 --device cuda \
    --output-dir $OUTBASE/qwen_lora_fused/reference_score

# Per-model LoRA: deepeval × 4 models
for MODEL in "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-14B" "Qwen/Qwen2.5-72B"; do
    SHORT=$(echo $MODEL | sed 's|.*/||')
    echo "=== Per-model LoRA: deepeval $SHORT ==="
    python -u -m route_balance.predictor.route_balance.offline_training.train_llm_predictor \
        --input $DATA \
        --base-model Qwen/Qwen2.5-0.5B \
        --target deepeval --mode per_model \
        --target-models "$MODEL" --epochs 3 --device cuda \
        --output-dir $OUTBASE/lora_per_model/deepeval_${SHORT}
done

# Per-model LoRA: reference_score × 4 models
for MODEL in "Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-14B" "Qwen/Qwen2.5-72B"; do
    SHORT=$(echo $MODEL | sed 's|.*/||')
    echo "=== Per-model LoRA: reference_score $SHORT ==="
    python -u -m route_balance.predictor.route_balance.offline_training.train_llm_predictor \
        --input $DATA \
        --base-model Qwen/Qwen2.5-0.5B \
        --target reference_score --mode per_model \
        --target-models "$MODEL" --epochs 3 --device cuda \
        --output-dir $OUTBASE/lora_per_model/reference_score_${SHORT}
done

echo "=== All Qwen LoRA jobs done ==="
