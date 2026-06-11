#!/bin/bash
# ROUTE_BALANCE Comprehensive Smoke Test — validates ALL experiment types before full evaluation
# Covers: 7 strategies, weight/batch/LPT/budget/threshold/constraint sweeps,
#         normalization ablation, RSO density, estimator types, latency predictor types
# Usage: bash run_comprehensive_smoketest.sh [result_dir]
set -e

PORT=8200
NUM_REQ=20
DATASET="data/route_balance/best-route-v3-test-500.jsonl"
RESULT_DIR=${1:-"experiment_output/comprehensive_smoketest"}
DEPLOY_CONFIG="route_balance/config/route_balance/model_deployment_smoketest.json"
HOST_CONFIG="route_balance/config/host_configs.json"
PRED_CONFIG="route_balance/config/route_balance/predictor_config_smoketest_fused.json"
PRED_KNN="route_balance/config/route_balance/predictor_config_smoketest_knn.json"
PRED_PFS="route_balance/config/route_balance/predictor_config_smoketest_pfs.json"
PRED_ROOFLINE="route_balance/config/route_balance/predictor_config_smoketest_roofline.json"
PRED_STATIC_TPOT="route_balance/config/route_balance/predictor_config_smoketest_static_tpot.json"

# CloudLab nodes
NODE0="asdwb@d7525-10s10317.wisc.cloudlab.us"
NODE1="asdwb@d7525-10s10319.wisc.cloudlab.us"

cd ~/Block
export PYTHONPATH=~/Block:~/vllm:$PYTHONPATH
mkdir -p $RESULT_DIR experiment_output/logs experiment_output/traces

PASS=0; FAIL=0; SKIP=0

# === Helper functions ===

run() {
    local NAME=$1; shift
    echo "  [$NAME] running..."
    if python3 route_balance/benchmark/route_balance/benchmark_serving.py \
      --backend route_balance --host 127.0.0.1 --port $PORT \
      --dataset-name custom --dataset-path $DATASET \
      --num-prompts $NUM_REQ --trust-remote-code \
      --save-result --save-detailed \
      --result-dir $RESULT_DIR --result-filename "${NAME}.json" "$@" \
      2>&1 | tail -3; then
        if [ -f "$RESULT_DIR/${NAME}.json" ]; then
            echo "  [$NAME] PASS"; PASS=$((PASS+1))
        else
            echo "  [$NAME] FAIL (no output file)"; FAIL=$((FAIL+1))
        fi
    else
        echo "  [$NAME] FAIL (benchmark error)"; FAIL=$((FAIL+1))
    fi
}

start_sched() {
    local SCHED=$1
    local CFG=${2:-$PRED_CONFIG}
    pkill -f route_balance_serve 2>/dev/null || true; sleep 2
    local ARGS="--host 0.0.0.0 --port $PORT"
    ARGS="$ARGS --model_config_path $DEPLOY_CONFIG --host_config $HOST_CONFIG"
    ARGS="$ARGS --scheduling $SCHED --predictor-config $CFG --chat"
    if [ "$SCHED" = "route_balance" ] || [ "$SCHED" = "length_aware" ]; then
        ARGS="$ARGS --enable-predictor-feedback --feedback-sample-rate 0.0"
    fi
    nohup python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve $ARGS \
      > experiment_output/logs/route_balance_${SCHED}.log 2>&1 &
    sleep 8
    if pgrep -f route_balance_serve > /dev/null; then
        # Warmup estimator
        curl -s -X POST http://localhost:$PORT/v1/estimate \
          -H "Content-Type: application/json" -d '{"prompt":"warmup test"}' > /dev/null 2>&1
        echo "  Scheduler started: $SCHED (config: $(basename $CFG))"
    else
        echo "  FAIL: scheduler $SCHED did not start. Check experiment_output/logs/route_balance_${SCHED}.log"
        FAIL=$((FAIL+1))
        return 1
    fi
}

cfg() {
    local RESP=$(curl -s -X POST http://localhost:$PORT/v1/config \
      -H "Content-Type: application/json" -d "$1")
    echo "  Config updated: $(echo $RESP | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("status","?"))' 2>/dev/null || echo 'ok')"
}

stats() {
    echo "  Scheduling stats:"
    curl -s http://localhost:$PORT/v1/scheduling_stats 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
parts = [f'{k}={v}' for k,v in d.items() if v > 0]
print('    ' + ', '.join(parts) if parts else '    (all zero)')
" 2>/dev/null || echo "    (unavailable)"
}

echo "========================================================"
echo "ROUTE_BALANCE Comprehensive Smoke Test — $(date)"
echo "Nodes: node0=$NODE0 node1=$NODE1"
echo "Requests per run: $NUM_REQ"
echo "========================================================"

# ============================================================
echo ""
echo "===== PHASE 1: All Scheduling Strategies ====="
# ============================================================
for S in random round_robin shortest_queue quality_greedy cost_greedy length_aware route_balance; do
    start_sched $S || continue
    for Q in 2 5; do
        run "p1_${S}_qps${Q}" --request-rate $Q
    done
done

# ============================================================
echo ""
echo "===== PHASE 2: Weight Sweep (no restart) ====="
# ============================================================
start_sched route_balance
for entry in \
    'balanced:{"w_latency":0.3,"w_cost":0.2,"w_quality":0.3,"w_balance":0.2}' \
    'qual_focus:{"w_latency":0.1,"w_cost":0.1,"w_quality":0.7,"w_balance":0.1}' \
    'cost_focus:{"w_latency":0.1,"w_cost":0.6,"w_quality":0.2,"w_balance":0.1}' \
    'no_quality:{"w_latency":0.4,"w_cost":0.3,"w_quality":0.0,"w_balance":0.3}' \
    'no_balance:{"w_latency":0.35,"w_cost":0.25,"w_quality":0.4,"w_balance":0.0}' \
; do
    NAME="${entry%%:*}"; WEIGHTS="${entry#*:}"
    cfg "{\"scoring_weights\":$WEIGHTS}"
    run "p2_weight_${NAME}" --request-rate 5
done
# Reset weights
cfg '{"scoring_weights":{"w_latency":0.3,"w_cost":0.2,"w_quality":0.3,"w_balance":0.2}}'

# ============================================================
echo ""
echo "===== PHASE 3: Batch + Normalization + LPT Sweep (no restart) ====="
# ============================================================

echo "--- Batch size ---"
for BS_TO in "1:1" "8:50" "32:200"; do
    BS="${BS_TO%%:*}"; TO="${BS_TO#*:}"
    cfg "{\"slo_defaults\":{\"batch_config\":{\"max_batch_size\":$BS,\"batch_timeout_ms\":$TO,\"adaptive_sizing\":false}}}"
    run "p3_batch_${BS}" --request-rate 5
done
cfg '{"slo_defaults":{"batch_config":{"max_batch_size":8,"batch_timeout_ms":50,"adaptive_sizing":true}}}'

echo "--- Normalization mode ---"
for MODE in two_phase per_request topsis z_score; do
    cfg "{\"slo_defaults\":{\"normalization_mode\":\"$MODE\"}}"
    run "p3_norm_${MODE}" --request-rate 5
done
cfg '{"slo_defaults":{"normalization_mode":"two_phase"}}'

echo "--- LPT sort key ---"
for KEY in max min mean; do
    cfg "{\"slo_defaults\":{\"lpt_sort_key\":\"$KEY\"}}"
    run "p3_lpt_${KEY}" --request-rate 5
done
cfg '{"slo_defaults":{"lpt_sort_key":"max"}}'

# ============================================================
echo ""
echo "===== PHASE 4: Budget + RSO + Constraint Mode (no restart) ====="
# ============================================================

echo "--- Budget threshold ---"
for THRESH in "0.0" "0.5" "0.9"; do
    cfg "{\"slo_defaults\":{\"budget_confidence_threshold\":$THRESH}}"
    run "p4_thresh_${THRESH}" --request-rate 5
done
cfg '{"slo_defaults":{"budget_confidence_threshold":0.5}}'

echo "--- RSO density ---"
for RATE in "0.0" "0.3" "0.5"; do
    run "p4_rso_${RATE}" --request-rate 5 --rso-rate $RATE
done

echo "--- Constraint mode (tight SLO to trigger filtering) ---"
for MODE in STRICT TIERED RELAXED; do
    cfg "{\"slo_defaults\":{\"constraint_mode\":\"$MODE\",\"ttft_slo_ms\":50,\"tpot_slo_ms\":10}}"
    run "p4_constraint_${MODE}" --request-rate 5
    stats
done
cfg '{"slo_defaults":{"constraint_mode":"TIERED","ttft_slo_ms":10000,"tpot_slo_ms":200}}'

echo "--- Quality threshold ---"
for QMIN in "0.0" "0.5" "0.8"; do
    cfg "{\"slo_defaults\":{\"quality_min\":$QMIN}}"
    run "p4_qualmin_${QMIN}" --request-rate 5
done
cfg '{"slo_defaults":{"quality_min":0.0}}'

echo "--- Filter relax order (tight SLOs) ---"
cfg '{"slo_defaults":{"ttft_slo_ms":100,"tpot_slo_ms":30,"quality_min":0.4,"budget_tokens":128,"budget_confidence_threshold":0.5}}'
for entry in \
    'ttft_first:["ttft","tpot","quality","budget"]' \
    'budget_first:["budget","ttft","tpot","quality"]' \
    'quality_first:["quality","budget","ttft","tpot"]' \
; do
    NAME="${entry%%:*}"; ORD="${entry#*:}"
    cfg "{\"slo_defaults\":{\"relax_order\":$ORD}}"
    run "p4_order_${NAME}" --request-rate 5
    stats
done
cfg '{"slo_defaults":{"relax_order":["ttft","tpot","quality","budget"],"ttft_slo_ms":10000,"tpot_slo_ms":200,"quality_min":0.0,"budget_tokens":256,"budget_confidence_threshold":0.5}}'

echo "--- Per-filter impact (moderate SLOs) ---"
cfg '{"slo_defaults":{"ttft_slo_ms":200,"tpot_slo_ms":50,"quality_min":0.3,"budget_tokens":128,"budget_confidence_threshold":0.5}}'
run "p4_filter_all" --request-rate 5
cfg '{"slo_defaults":{"budget_confidence_threshold":0.0}}'
run "p4_filter_no_budget" --request-rate 5
cfg '{"slo_defaults":{"budget_confidence_threshold":0.5,"ttft_slo_ms":999999}}'
run "p4_filter_no_ttft" --request-rate 5
cfg '{"slo_defaults":{"ttft_slo_ms":200,"quality_min":0.0}}'
run "p4_filter_no_quality" --request-rate 5
cfg '{"slo_defaults":{"quality_min":0.3,"constraint_mode":"RELAXED"}}'
run "p4_filter_none" --request-rate 5
stats
cfg '{"slo_defaults":{"constraint_mode":"TIERED","ttft_slo_ms":10000,"tpot_slo_ms":200,"quality_min":0.0,"budget_tokens":256,"budget_confidence_threshold":0.5}}'

# ============================================================
echo ""
echo "===== PHASE 5: Model Estimator Ablation (requires restarts) ====="
# ============================================================

echo "--- Default (fused ModernBERT + KNN) ---"
start_sched route_balance $PRED_CONFIG
run "p5_estimator_default" --request-rate 5

echo "--- KNN-only ---"
if [ -f "$PRED_KNN" ]; then
    start_sched route_balance $PRED_KNN
    run "p5_estimator_knn" --request-rate 5
else
    echo "  SKIP: $PRED_KNN not found"; SKIP=$((SKIP+1))
fi

echo "--- PFS ---"
if [ -f "$PRED_PFS" ]; then
    start_sched route_balance $PRED_PFS
    run "p5_estimator_pfs" --request-rate 5
else
    echo "  SKIP: $PRED_PFS not found"; SKIP=$((SKIP+1))
fi

# ============================================================
echo ""
echo "===== PHASE 6: Latency Predictor Ablation (requires restarts) ====="
# ============================================================

echo "--- XGBoost (default learned) ---"
start_sched route_balance $PRED_CONFIG
run "p6_latency_xgboost" --request-rate 5

echo "--- Roofline ---"
if [ -f "$PRED_ROOFLINE" ]; then
    start_sched route_balance $PRED_ROOFLINE
    run "p6_latency_roofline" --request-rate 5
else
    echo "  SKIP: $PRED_ROOFLINE not found"; SKIP=$((SKIP+1))
fi

echo "--- Static TPOT ---"
if [ -f "$PRED_STATIC_TPOT" ]; then
    start_sched route_balance $PRED_STATIC_TPOT
    run "p6_latency_static_tpot" --request-rate 5
else
    echo "  SKIP: $PRED_STATIC_TPOT not found"; SKIP=$((SKIP+1))
fi

# LSTM — skip if no checkpoint
echo "  SKIP: LSTM (no trained checkpoint available)"; SKIP=$((SKIP+1))

# ============================================================
echo ""
echo "===== PHASE 7: Full Metric Verification ====="
# ============================================================
start_sched route_balance $PRED_CONFIG
# Enable trace saving
cfg '{"slo_defaults":{"trace_sample_rate":0.1}}'
run "p7_full_metrics" --request-rate 3 --rso-rate 0.3 --num-prompts 30

# Collect final stats
echo "--- Batch stats ---"
curl -s http://localhost:$PORT/v1/batch_stats 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "(unavailable)"
echo "--- Scheduling stats ---"
stats
echo "--- Batch traces ---"
ls experiment_output/traces/batch_*.json 2>/dev/null | wc -l
echo " trace files saved"
# Disable trace saving
cfg '{"slo_defaults":{"trace_sample_rate":0}}'

# ============================================================
echo ""
echo "========================================================"
echo "COMPREHENSIVE SMOKE TEST COMPLETE — $(date)"
echo "========================================================"
echo "PASS: $PASS  FAIL: $FAIL  SKIP: $SKIP"
echo "Results: $(ls -1 $RESULT_DIR/*.json 2>/dev/null | wc -l) files in $RESULT_DIR/"
echo ""
echo "Next: python3 route_balance/exp/route_balance/verify_smoketest.py --result-dir $RESULT_DIR"

pkill -f route_balance_serve 2>/dev/null || true
