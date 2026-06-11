#!/bin/bash
# Run reference-grounded LLM judging with all three judges.
# Uses vLLM for fast inference on personal GPU VM (RTX 8000 48GB).
#
# Judges:
#   1. prometheus-eval/prometheus-7b-v2.0 (1-5, purpose-built)
#   2. meta-llama/Llama-3.1-8B-Instruct (1-10, unbiased)
#   3. Qwen/Qwen2.5-7B-Instruct (1-10, biased but best correlation)
#
# Each judge scores all 4 models × all prompts against dataset ground truth.
# Results stored in llm_judge_scores.{judge_key} per model entry.
#
# Usage:
#   bash route_balance/exp/route_balance/run_reference_judging.sh [smoke|test|train|all]
#
# Prerequisites:
#   - vLLM venv at /local/scratch/tmp/wd312/judge_venv
#   - Data at /local/scratch/tmp/wd312/data/{test,train}_with_reftext.jsonl
#   - HF cache at /local/scratch/tmp/wd312/hf_cache

set -e

MODE=${1:-smoke}
VENV=/local/scratch/tmp/wd312/judge_venv
DATA_DIR=/local/scratch/tmp/wd312/data
LOG_DIR=/local/scratch/tmp/wd312/logs
BLOCK_DIR=~/Code/llm/Block

export PYTHONPATH=$BLOCK_DIR:$PYTHONPATH
export HF_HOME=/local/scratch/tmp/wd312/hf_cache
export HF_TOKEN=${HF_TOKEN}
source $VENV/bin/activate

mkdir -p $LOG_DIR $DATA_DIR/scored

JUDGES=(
    "prometheus-eval/prometheus-7b-v2.0"
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

JUDGE_NAMES=(
    "prometheus"
    "llama"
    "qwen"
)

run_judge() {
    local judge_model=$1
    local judge_name=$2
    local input_file=$3
    local output_file=$4
    local split=$5
    local max_samples=$6

    echo ""
    echo "================================================================"
    echo "  Judge: $judge_name ($judge_model)"
    echo "  Split: $split"
    echo "  Input: $input_file"
    echo "  Output: $output_file"
    echo "  Max samples: $max_samples"
    echo "  Time: $(date)"
    echo "================================================================"

    local args="--input $input_file --output $output_file --judge-model $judge_model"
    if [ "$max_samples" -gt 0 ]; then
        args="$args --max-samples $max_samples"
    fi

    python -u -m route_balance.predictor.route_balance.offline_training.score_with_vllm $args \
        2>&1 | tee -a $LOG_DIR/${judge_name}_${split}.log

    echo ""
    echo "[$judge_name/$split] Done at $(date)"
    echo ""
}

if [ "$MODE" == "smoke" ]; then
    echo "=== SMOKE TEST (5 samples per judge) ==="
    for i in "${!JUDGES[@]}"; do
        run_judge "${JUDGES[$i]}" "${JUDGE_NAMES[$i]}" \
            "$DATA_DIR/test_with_reftext.jsonl" \
            "/tmp/smoke_${JUDGE_NAMES[$i]}.jsonl" \
            "smoke" 5
    done
    echo ""
    echo "=== SMOKE TEST COMPLETE ==="
    echo "Check outputs:"
    for name in "${JUDGE_NAMES[@]}"; do
        echo "  /tmp/smoke_${name}.jsonl"
    done

elif [ "$MODE" == "test" ]; then
    echo "=== SCORING TEST SET (3,642 entries) ==="
    # Chain: each judge reads the previous output so scores accumulate
    cp $DATA_DIR/test_with_reftext.jsonl $DATA_DIR/scored/test_scoring.jsonl

    for i in "${!JUDGES[@]}"; do
        run_judge "${JUDGES[$i]}" "${JUDGE_NAMES[$i]}" \
            "$DATA_DIR/scored/test_scoring.jsonl" \
            "$DATA_DIR/scored/test_scoring.jsonl" \
            "test" 0
    done
    # Final rename
    cp $DATA_DIR/scored/test_scoring.jsonl $DATA_DIR/scored/test_all_judges.jsonl
    echo "=== TEST SCORING COMPLETE: $DATA_DIR/scored/test_all_judges.jsonl ==="

elif [ "$MODE" == "train" ]; then
    echo "=== SCORING TRAIN SET (14,963 entries) ==="
    cp $DATA_DIR/train_with_reftext.jsonl $DATA_DIR/scored/train_scoring.jsonl

    for i in "${!JUDGES[@]}"; do
        run_judge "${JUDGES[$i]}" "${JUDGE_NAMES[$i]}" \
            "$DATA_DIR/scored/train_scoring.jsonl" \
            "$DATA_DIR/scored/train_scoring.jsonl" \
            "train" 0
    done
    cp $DATA_DIR/scored/train_scoring.jsonl $DATA_DIR/scored/train_all_judges.jsonl
    echo "=== TRAIN SCORING COMPLETE: $DATA_DIR/scored/train_all_judges.jsonl ==="

elif [ "$MODE" == "all" ]; then
    echo "=== SCORING ALL (test + train, 3 judges each) ==="
    echo "Start: $(date)"

    # Test first (smaller, faster)
    bash $0 test

    # Then train
    bash $0 train

    echo ""
    echo "=== ALL COMPLETE ==="
    echo "End: $(date)"
    echo "Results:"
    ls -lh $DATA_DIR/scored/test_all_judges.jsonl $DATA_DIR/scored/train_all_judges.jsonl

else
    echo "Usage: $0 [smoke|test|train|all]"
    exit 1
fi
