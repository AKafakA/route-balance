#!/bin/bash
# Run remaining experiments (Parts A remainder + B-F)
set -e
cd ~/Block
export PYTHONPATH=~/Block:~/vllm:$PYTHONPATH
RESULT_DIR=experiment_output/comprehensive_results
PORT=8200
mkdir -p $RESULT_DIR

run() {
    local NAME=$1; shift
    echo "--- $NAME ---"
    python3 route_balance/benchmark/route_balance/benchmark_serving.py --backend route_balance --host 127.0.0.1 --port $PORT --dataset-name custom --dataset-path data/route_balance/best-route-v3-test-500.jsonl --num-prompts 50 --trust-remote-code --save-result --save-detailed --result-dir $RESULT_DIR --result-filename "${NAME}.json" "$@" 2>&1 | grep -E "saved|completed"
}

start() {
    pkill -f route_balance_serve 2>/dev/null || true; sleep 2
    local S=$1
    local ARGS="--host 0.0.0.0 --port $PORT --model_config_path route_balance/config/route_balance/model_deployment_smoketest.json --host_config route_balance/config/host_configs.json --scheduling $S --predictor-config route_balance/config/route_balance/predictor_config_smoketest.json --chat"
    if [ "$S" = "route_balance" ] || [ "$S" = "length_aware" ]; then
        ARGS="$ARGS --enable-predictor-feedback --feedback-sample-rate 0.0"
    fi
    nohup python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve $ARGS > experiment_output/logs/route_balance_${S}.log 2>&1 &
    sleep 6
    curl -s -X POST http://localhost:$PORT/v1/estimate -H "Content-Type: application/json" -d '{"prompt":"warmup"}' > /dev/null 2>&1 || true
    echo "Scheduler: $S"
}

cfg() { curl -s -X POST http://localhost:$PORT/v1/config -H "Content-Type: application/json" -d "$1" > /dev/null; }

echo "PART_A_REMAINING"
start length_aware; run length_aware_qps2 --request-rate 2; run length_aware_qps5 --request-rate 5
start route_balance; run route_balance_qps2 --request-rate 2; run route_balance_qps5 --request-rate 5

echo "PART_B_WEIGHTS"
cfg '{"scoring_weights":{"w_latency":0.6,"w_cost":0.1,"w_quality":0.2,"w_balance":0.1}}'; run weight_lat_focus --request-rate 5
cfg '{"scoring_weights":{"w_latency":0.1,"w_cost":0.1,"w_quality":0.7,"w_balance":0.1}}'; run weight_qual_focus --request-rate 5
cfg '{"scoring_weights":{"w_latency":0.1,"w_cost":0.6,"w_quality":0.2,"w_balance":0.1}}'; run weight_cost_focus --request-rate 5
cfg '{"scoring_weights":{"w_latency":0.4,"w_cost":0.3,"w_quality":0.0,"w_balance":0.3}}'; run weight_no_quality --request-rate 5
cfg '{"scoring_weights":{"w_latency":0.35,"w_cost":0.25,"w_quality":0.4,"w_balance":0.0}}'; run weight_no_balance --request-rate 5
cfg '{"scoring_weights":{"w_latency":0.3,"w_cost":0.2,"w_quality":0.3,"w_balance":0.2}}'

echo "PART_C_BATCH"
cfg '{"slo_defaults":{"batch_config":{"max_batch_size":1,"batch_timeout_ms":1,"adaptive_sizing":false}}}'; run batch_nobatch --request-rate 5
cfg '{"slo_defaults":{"batch_config":{"max_batch_size":4,"batch_timeout_ms":25,"adaptive_sizing":false}}}'; run batch_4_25ms --request-rate 5
cfg '{"slo_defaults":{"batch_config":{"max_batch_size":16,"batch_timeout_ms":100,"adaptive_sizing":false}}}'; run batch_16_100ms --request-rate 5
cfg '{"slo_defaults":{"batch_config":{"max_batch_size":8,"batch_timeout_ms":50,"adaptive_sizing":true}}}'

echo "PART_D_BUDGET"
for B in 64 128 256 512; do cfg "{\"slo_defaults\":{\"budget_tokens\":$B}}"; run "budget_$B" --request-rate 5; done
cfg '{"slo_defaults":{"budget_tokens":256}}'

echo "PART_E_CONSTRAINT"
cfg '{"slo_defaults":{"constraint_mode":"STRICT","ttft_slo_ms":3000}}'; run constraint_STRICT --request-rate 5
cfg '{"slo_defaults":{"constraint_mode":"TIERED","ttft_slo_ms":3000}}'; run constraint_TIERED --request-rate 5
cfg '{"slo_defaults":{"constraint_mode":"RELAXED","ttft_slo_ms":3000}}'; run constraint_RELAXED --request-rate 5
cfg '{"slo_defaults":{"constraint_mode":"TIERED","ttft_slo_ms":10000}}'

echo "PART_F_QPS"
for Q in 1 10 15; do run "route_balance_qps$Q" --request-rate $Q; done

echo "ALL_DONE"
ls -1 $RESULT_DIR/*.json | wc -l
echo "result files"
