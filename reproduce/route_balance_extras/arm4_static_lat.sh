#!/usr/bin/env bash
# Arm-4 (reviewer-W direct test): static per-tier latency prior in the model score.
# ROUTE_BALANCE_STATIC_LAT = nominal TPOT ms/tok (low-load medians): 3b=12.3 7b=20.6 14b=18.4 72b=47.4
# Compare to arm1 (learned T-hat): stationary l18 (arm1=2615) and 40/6 overload (arm1=3788).
# If arm4 reproduces arm1 in BOTH -> learned latency signal not load-bearing -> re-scope.
set -u
cd ~/RouteBalance; export PYTHONPATH=$PWD HF_HOME=/mydata/hf_cache
export ROUTE_BALANCE_STATIC_LAT="3b=12.3,7b=20.6,14b=18.4,72b=47.4"
MARK=~/ARM4_STATUS; OUT=~/RouteBalance/experiment_output/v2_sweep_jun06/review_extras; : > $MARK
pkill -f "[c]ara_serve" 2>/dev/null; pkill -f "[b]enchmark_serving" 2>/dev/null; sleep 5
for s in ovserve ovrun arm4serve; do tmux kill-session -t $s 2>/dev/null; done
post(){ curl -sS --max-time 15 -X POST localhost:8200/v1/config -H "Content-Type: application/json" -d "$1" -o /dev/null -w "POST %{http_code}\n" >> $MARK; }
bench(){ local name=$1 lam=$2 sq=$3; local SQENV=""; [ -n "$sq" ] && SQENV="ROUTE_BALANCE_SQUARE=$sq"; env $SQENV python3 -u -m route_balance.benchmark.route_balance.benchmark_serving --seed 5 --num-prompts $4 --dataset-name custom --dataset-path data/route_balance/best-route-v3-test-3534-eval.jsonl --custom-output-len 1024 --max-output-len 1024 --max-total-len 8192 --backend route_balance --base-url http://localhost:8200 --endpoint /v1/completions --request-rate $lam --burstiness 1.0 --save-result --save-detailed --result-dir $OUT --result-filename ${name}.json --rso-constraint-mode TIERED --model Qwen/Qwen2.5-3B --tokenizer Qwen/Qwen2.5-3B > $OUT/${name}.log 2>&1; echo "ARM4 ${name} e2e=$(python3 -c "import json;print(round(json.load(open(\"$OUT/${name}.json\")).get(\"mean_e2el_ms\",0)))" 2>/dev/null) fail=$(python3 -c "import json;print(json.load(open(\"$OUT/${name}.json\")).get(\"failed\"))" 2>/dev/null)" >> $MARK; }
# serve native with ROUTE_BALANCE_STATIC_LAT active
tmux new-session -d -s arm4serve "cd ~/RouteBalance && CUDA_VISIBLE_DEVICES='' ROUTE_BALANCE_STATIC_LAT='3b=12.3,7b=20.6,14b=18.4,72b=47.4' PYTHONPATH=\$PWD ROUTE_BALANCE_INPROC_PREDICTOR=1 HF_HOME=/mydata/hf_cache python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve --host 0.0.0.0 --port 8200 --model_config_path route_balance/config/route_balance/model_deployment.json --scheduling route_balance --chat --scheduler-config route_balance/config/route_balance/scheduler_config_qn_minmax.json --predictor-config route_balance/config/route_balance/predictor_xgboost_3model.json --host_config route_balance/config/host_configs.json >~/arm4serve.log 2>&1"
for t in $(seq 1 30); do curl -sf -m4 localhost:8200/health >/dev/null && break; sleep 5; done
grep -q "INPROC-XGB.*loaded" ~/arm4serve.log && echo "INPROC_OK" >> $MARK || { echo "INPROC_FAIL" >> $MARK; exit 1; }
post '{"scoring_weights":{"w_quality":0.33,"w_latency":0.33,"w_cost":0.33,"w_balance":0.0}}'
# SMOKE (N=50): verify it routes sanely (mix should resemble uniform ~57/10/32/1) before full runs
bench arm4_smoke 12 "" 50
echo "SMOKE_DONE" >> $MARK
# arm4 stationary l18 (compare arm1=2615) + arm4 40/6 overload (compare arm1=3788)
bench arm4_stat_l18 18 "" 3534
bench arm4_sq40 23 "6,40" 3534
echo "ARM4_DONE" >> $MARK
