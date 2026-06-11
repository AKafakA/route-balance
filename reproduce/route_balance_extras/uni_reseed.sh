#!/usr/bin/env bash
set -u
cd ~/RouteBalance; export PYTHONPATH=$PWD HF_HOME=/mydata/hf_cache
MARK=~/UNI_STATUS; OUT=~/RouteBalance/experiment_output/v2_sweep_jun06/review_extras; : > $MARK
pkill -f "[c]ara_serve" 2>/dev/null; pkill -f "[b]enchmark_serving" 2>/dev/null; sleep 5
for s in p3sqserve p3sqpipe uniserve; do tmux kill-session -t $s 2>/dev/null; done
tmux new-session -d -s uniserve "cd ~/RouteBalance && CUDA_VISIBLE_DEVICES='' PYTHONPATH=\$PWD ROUTE_BALANCE_INPROC_PREDICTOR=1 HF_HOME=/mydata/hf_cache python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve --host 0.0.0.0 --port 8200 --model_config_path route_balance/config/route_balance/model_deployment.json --scheduling route_balance --chat --scheduler-config route_balance/config/route_balance/scheduler_config_qn_minmax.json --predictor-config route_balance/config/route_balance/predictor_xgboost_3model.json --host_config route_balance/config/host_configs.json >~/uniserve.log 2>&1"
for t in $(seq 1 30); do curl -sf -m4 localhost:8200/health >/dev/null && break; sleep 5; done
grep -q "INPROC-XGB.*loaded" ~/uniserve.log && echo "INPROC_OK" >> $MARK || { echo "INPROC_FAIL" >> $MARK; exit 1; }
curl -sS --max-time 15 -X POST localhost:8200/v1/config -H "Content-Type: application/json" -d '{"scoring_weights":{"w_quality":0.33,"w_latency":0.33,"w_cost":0.33,"w_balance":0.0}}' -o /dev/null -w "POST %{http_code}\n" >> $MARK
bench(){ local name=$1 lam=$2 seed=$3; python3 -u -m route_balance.benchmark.route_balance.benchmark_serving --seed $seed --num-prompts 3534 --dataset-name custom --dataset-path data/route_balance/best-route-v3-test-3534-eval.jsonl --custom-output-len 1024 --max-output-len 1024 --max-total-len 8192 --backend route_balance --base-url http://localhost:8200 --endpoint /v1/completions --request-rate $lam --burstiness 1.0 --save-result --save-detailed --result-dir $OUT --result-filename ${name}.json --rso-constraint-mode TIERED --model Qwen/Qwen2.5-3B --tokenizer Qwen/Qwen2.5-3B > $OUT/${name}.log 2>&1; echo "UNI ${name} e2e=$(python3 -c "import json;print(round(json.load(open(\"$OUT/${name}.json\")).get(\"mean_e2el_ms\",0)))" 2>/dev/null) fail=$(python3 -c "import json;print(json.load(open(\"$OUT/${name}.json\")).get(\"failed\"))" 2>/dev/null)" >> $MARK; }
bench p2_uniform_l12_s6 12 6
bench p2_uniform_l12_s7 12 7
bench p2_uniform_l24_s6 24 6
bench p2_uniform_l24_s7 24 7
echo "UNI_DONE" >> $MARK
