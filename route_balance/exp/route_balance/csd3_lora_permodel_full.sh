#!/bin/bash
#SBATCH --job-name=rb_lora_pm
#SBATCH --partition=ampere
#SBATCH --account=KALYVIANAKI-SL3-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=/rds/user/anon/hpc-work/llm/RouteBalance/training_logs/lora_permodel_full_%j.log

# Per-model LoRA: 4 targets × 4 models = 16 runs
# Batch 1: 4 GPUs × (judge×3B, judge×7B, judge×14B, judge×72B) — not feasible on 4 GPUs
# Instead: run 4 jobs at a time (1 per GPU), 4 batches
#
# With batch_size=16: each run ≈ 3h on A100
# 4 batches × 3h = ~12h total
#
# Run with: sbatch route_balance/exp/route_balance/csd3_lora_permodel_full.sh

set -e
cd /rds/user/anon/hpc-work/llm/RouteBalance
export PYTHONPATH=/rds/user/anon/hpc-work/llm/RouteBalance:$PYTHONPATH
source /rds/user/anon/hpc-work/llm/RouteBalance/.venv/bin/activate
mkdir -p training_logs models/route_balance/lora_per_model

echo "=== Per-Model LoRA Full Training — $(date) ==="
python3 -c "import transformers, torch, peft; print(f'tf={transformers.__version__}, torch={torch.__version__}, peft={peft.__version__}, GPUs={torch.cuda.device_count()}')"

TRAIN="data/route_balance/training_data/train_fixed.jsonl"
TEST="data/route_balance/training_data/test_fixed.jsonl"
COMMON="--mode per_model --epochs 3 --batch-size 16 --max-length 1024"

MODELS=("Qwen/Qwen2.5-3B" "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-14B" "Qwen/Qwen2.5-72B")
TARGETS=("judge" "similarity" "reference_score" "length")

run_batch() {
    local BATCH_NAME=$1; shift
    local PIDS=()
    local NAMES=()
    local GPU=0

    echo ""
    echo "--- Batch: $BATCH_NAME ---"
    for SPEC in "$@"; do
        TARGET="${SPEC%%:*}"
        MODEL="${SPEC#*:}"
        MODEL_SHORT=$(echo $MODEL | sed 's|Qwen/Qwen2.5-||')
        NAME="${TARGET}_${MODEL_SHORT}"
        OUTDIR="models/route_balance/lora_per_model/${NAME}"

        echo "  GPU $GPU: $NAME"
        CUDA_VISIBLE_DEVICES=$GPU python3 -m route_balance.predictor.route_balance.offline_training.train_llm_predictor \
            --input $TRAIN --test-input $TEST \
            --target $TARGET --target-models "$MODEL" \
            $COMMON \
            --output-dir $OUTDIR \
            > training_logs/lora_pm_${NAME}.log 2>&1 &
        PIDS+=($!)
        NAMES+=($NAME)
        GPU=$((GPU + 1))
    done

    # Wait for all
    local FAIL=0
    for i in "${!PIDS[@]}"; do
        if wait ${PIDS[$i]}; then
            echo "  ${NAMES[$i]}: SUCCESS"
        else
            echo "  ${NAMES[$i]}: FAILED (exit $?)"
            FAIL=$((FAIL + 1))
        fi
    done
    echo "  Batch $BATCH_NAME: $((${#PIDS[@]} - FAIL))/${#PIDS[@]} succeeded"
    return $FAIL
}

# Batch 1: judge × 4 models (4 GPUs)
run_batch "judge" \
    "judge:Qwen/Qwen2.5-3B" \
    "judge:Qwen/Qwen2.5-7B" \
    "judge:Qwen/Qwen2.5-14B" \
    "judge:Qwen/Qwen2.5-72B"

# Batch 2: similarity × 4 models (4 GPUs)
run_batch "similarity" \
    "similarity:Qwen/Qwen2.5-3B" \
    "similarity:Qwen/Qwen2.5-7B" \
    "similarity:Qwen/Qwen2.5-14B" \
    "similarity:Qwen/Qwen2.5-72B"

# Batch 3: reference_score × 4 models (4 GPUs)
run_batch "reference_score" \
    "reference_score:Qwen/Qwen2.5-3B" \
    "reference_score:Qwen/Qwen2.5-7B" \
    "reference_score:Qwen/Qwen2.5-14B" \
    "reference_score:Qwen/Qwen2.5-72B"

# Batch 4: length × 4 models (4 GPUs)
run_batch "length" \
    "length:Qwen/Qwen2.5-3B" \
    "length:Qwen/Qwen2.5-7B" \
    "length:Qwen/Qwen2.5-14B" \
    "length:Qwen/Qwen2.5-72B"

echo ""
echo "=== All Batches Complete — $(date) ==="
echo "Results:"
for d in models/route_balance/lora_per_model/*/; do
    name=$(basename $d)
    [ -f "${d}training_results.json" ] && echo "  $name: OK" || echo "  $name: MISSING"
done
