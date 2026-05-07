#!/bin/bash
# Overnight pipeline: DeepEval G-Eval scoring + HF upload + model training
#
# Steps:
#   1. Score train set with DeepEval G-Eval (Llama-3.1-8B, 1-10 reference-grounded)
#   2. Score test set
#   3. Merge scores into original data files + rename blind judge key
#   4. Upload to HuggingFace with updated datacard
#   5. Train fused models with new prometheus/deepeval target
#
# Prerequisites:
#   - vLLM server running on port 8000 with Llama-3.1-8B-Instruct
#   - judge_venv at /local/scratch/tmp/anon/judge_venv
#   - Data at /local/scratch/tmp/anon/data/{train,test}_with_reftext.jsonl
#
# Usage:
#   nohup bash route_balance/exp/route_balance/run_overnight_judging_pipeline.sh \
#       > /local/scratch/tmp/anon/logs/overnight_pipeline.log 2>&1 &

set -e

VENV=/local/scratch/tmp/anon/judge_venv
DATA_DIR=/local/scratch/tmp/anon/data
LOG_DIR=/local/scratch/tmp/anon/logs
BLOCK_DIR=~/Code/llm/RouteBalance
JUDGE_KEY="deepeval-llama3.1-8b-it_reference"

export PYTHONPATH=$BLOCK_DIR:$PYTHONPATH
export HF_HOME=/local/scratch/tmp/anon/hf_cache
export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN env var}
export OPENAI_API_KEY=dummy
export DEEPEVAL_DISABLE_TIMEOUTS=true
source $VENV/bin/activate

mkdir -p $LOG_DIR $DATA_DIR/scored

echo "============================================================"
echo "  Overnight Pipeline — $(date)"
echo "============================================================"

# Verify vLLM server is running
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "ERROR: vLLM server not running on port 8000!"
    echo "Start it first:"
    echo "  python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --dtype float16 --gpu-memory-utilization 0.85 --max-model-len 8192 --port 8000"
    exit 1
fi
echo "vLLM server: OK"

# ============================================================
# Step 1: Score TRAIN set
# ============================================================
echo ""
echo "============================================================"
echo "  Step 1: Scoring TRAIN set (14,963 entries × 4 models)"
echo "  Started: $(date)"
echo "============================================================"

python -u $BLOCK_DIR/route_balance/predictor/route_balance/offline_training/score_with_deepeval.py \
    --input $DATA_DIR/train_with_reftext.jsonl \
    --output $DATA_DIR/scored/train_scored.jsonl \
    --judge-key "$JUDGE_KEY" \
    --max-concurrent 8 \
    2>&1 | tee $LOG_DIR/deepeval_train.log

# Check for failures
TRAIN_FAILED=$(grep "failed in" $LOG_DIR/deepeval_train.log | grep -oP '\d+ failed' | grep -oP '^\d+')
if [ "$TRAIN_FAILED" != "0" ] && [ -n "$TRAIN_FAILED" ]; then
    echo "ERROR: Train scoring had $TRAIN_FAILED failures. STOPPING pipeline."
    echo "Fix the issue and rerun. Do NOT proceed to downstream steps."
    exit 1
fi
echo "Step 1 DONE (0 failures): $(date)"

# ============================================================
# Step 2: Score TEST set
# ============================================================
echo ""
echo "============================================================"
echo "  Step 2: Scoring TEST set (3,642 entries × 4 models)"
echo "  Started: $(date)"
echo "============================================================"

python -u $BLOCK_DIR/route_balance/predictor/route_balance/offline_training/score_with_deepeval.py \
    --input $DATA_DIR/test_with_reftext.jsonl \
    --output $DATA_DIR/scored/test_scored.jsonl \
    --judge-key "$JUDGE_KEY" \
    --max-concurrent 8 \
    2>&1 | tee $LOG_DIR/deepeval_test.log

# Check for failures
TEST_FAILED=$(grep "failed in" $LOG_DIR/deepeval_test.log | grep -oP '\d+ failed' | grep -oP '^\d+')
if [ "$TEST_FAILED" != "0" ] && [ -n "$TEST_FAILED" ]; then
    echo "ERROR: Test scoring had $TEST_FAILED failures. STOPPING pipeline."
    echo "Fix the issue and rerun. Do NOT proceed to downstream steps."
    exit 1
fi
echo "Step 2 DONE (0 failures): $(date)"

# ============================================================
# Step 3: Post-process — rename blind judge + merge into raw data
# ============================================================
echo ""
echo "============================================================"
echo "  Step 3: Post-processing"
echo "  Started: $(date)"
echo "============================================================"

python -u $BLOCK_DIR/route_balance/predictor/route_balance/offline_training/postprocess_judge_scores.py \
    --train-scored $DATA_DIR/scored/train_scored.jsonl \
    --test-scored $DATA_DIR/scored/test_scored.jsonl \
    --raw-data $BLOCK_DIR/data/route_balance/training_data/route_balance_v3_all_training_fixed.json \
    --output-dir $DATA_DIR/final \
    2>&1 | tee $LOG_DIR/postprocess.log

echo "Step 3 DONE: $(date)"

# ============================================================
# Step 4: Upload to HuggingFace
# ============================================================
echo ""
echo "============================================================"
echo "  Step 4: Upload to HuggingFace"
echo "  Started: $(date)"
echo "============================================================"

python -u $BLOCK_DIR/route_balance/predictor/route_balance/offline_training/upload_scored_data_to_hf.py \
    --train $DATA_DIR/final/train.jsonl \
    --test $DATA_DIR/final/test.jsonl \
    --repo anon/route_balance_model_estimator \
    --token $HF_TOKEN \
    2>&1 | tee $LOG_DIR/hf_upload.log

echo "Step 4 DONE: $(date)"

# ============================================================
# Step 5: Model training (switch to training venv)
# ============================================================
echo ""
echo "============================================================"
echo "  Step 5: Model training"
echo "  Started: $(date)"
echo "============================================================"

# Kill vLLM server to free GPU
pkill -f "vllm.entrypoints" || true
sleep 5
echo "vLLM server stopped, GPU freed"

# Switch to training venv (NFS shared .venv has training deps)
source $BLOCK_DIR/.venv/bin/activate
export PYTHONPATH=$BLOCK_DIR:$PYTHONPATH

TRAIN_DATA=$DATA_DIR/final/train.jsonl
TEST_DATA=$DATA_DIR/final/test.jsonl
MODEL_DIR=/local/scratch/tmp/anon/models

# 5a. KNN rebuild (~5 min)
echo "--- 5a: KNN rebuild ---"
python -u -m route_balance.predictor.route_balance.offline_training.train_knn \
    --train-data $TRAIN_DATA \
    --output-dir $MODEL_DIR/knn_with_deepeval \
    2>&1 | tee $LOG_DIR/train_knn.log
echo "KNN done: $(date)"

# 5b. Fused RoBERTa (deepeval target only — other targets already trained)
echo "--- 5b: Fused RoBERTa (deepeval) ---"
python -u -m route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor \
    --model-name roberta-base \
    --train-data $TRAIN_DATA \
    --val-data $TEST_DATA \
    --target deepeval \
    --output-dir $MODEL_DIR/roberta_fused/deepeval \
    --epochs 5 \
    --batch-size 8 \
    --lr 1e-5 \
    --scheduler polynomial \
    --device cuda \
    2>&1 | tee $LOG_DIR/train_roberta_deepeval.log
echo "RoBERTa deepeval done: $(date)"

# 5c. Fused ModernBERT (deepeval target only)
echo "--- 5c: Fused ModernBERT (deepeval) ---"
python -u -m route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor \
    --model-name answerdotai/ModernBERT-base \
    --train-data $TRAIN_DATA \
    --val-data $TEST_DATA \
    --target deepeval \
    --output-dir $MODEL_DIR/modernbert_fused/deepeval \
    --epochs 5 \
    --batch-size 8 \
    --lr 1e-5 \
    --scheduler polynomial \
    --device cuda \
    2>&1 | tee $LOG_DIR/train_modernbert_deepeval.log
echo "ModernBERT deepeval done: $(date)"

# 5d. Fused LoRA (deepeval target only)
echo "--- 5d: Fused LoRA (deepeval) ---"
python -u -m route_balance.predictor.route_balance.offline_training.train_llm_predictor \
    --train-data $TRAIN_DATA \
    --val-data $TEST_DATA \
    --target deepeval \
    --output-dir $MODEL_DIR/qwen_lora_fused/deepeval \
    --epochs 5 \
    --device cuda \
    2>&1 | tee $LOG_DIR/train_lora_deepeval.log
echo "LoRA deepeval done: $(date)"

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE — $(date)"
echo "============================================================"
echo "Results:"
ls -lh $DATA_DIR/final/*.jsonl
ls -lh $MODEL_DIR/*/
