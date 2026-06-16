#!/usr/bin/env bash
set -u
cd ~/RouteBalance; export PYTHONPATH=$PWD HF_HOME=/mydata/hf_cache ROUTE_BALANCE_SQUARE=6,30
MARK=~/P3SQ_STATUS; OUT=~/RouteBalance/experiment_output/v2_sweep_jun06/review_extras; : > $MARK
pkill -f "[c]ara_serve" 2>/dev/null; pkill -f "[b]enchmark_serving" 2>/dev/null; sleep 5
for s in wq6serve p3sqserve p3sqpipe; do tmux kill-session -t $s 2>/dev/null; done
post(){ curl -sS --max-time 15 -X POST localhost:8200/v1/config -H "Content-Type: application/json" -d "$1" -o /dev/null -w "POST %{http_code}\n" >> $MARK; }
bench(){ local name=$1; python3 -u -m route_balance.benchmark.route_balance.benchmark_serving --seed 5 --num-prompts 3534 --dataset-name custom --dataset-path data/route_balance/best-route-v3-test-3534-eval.jsonl --custom-output-len 1024 --max-output-len 1024 --max-total-len 8192 --backend route_balance --base-url http://localhost:8200 --endpoint /v1/completions --request-rate 18 --burstiness 1.0 --save-result --save-detailed --result-dir $OUT --result-filename ${name}.json --rso-constraint-mode TIERED --model Qwen/Qwen2.5-3B --tokenizer Qwen/Qwen2.5-3B > $OUT/${name}.log 2>&1; echo "P3SQ ${name} e2e=$(python3 -c "import json;print(round(json.load(open(\"$OUT/${name}.json\")).get(\"mean_e2el_ms\",0)))" 2>/dev/null) p99=$(python3 -c "import json,numpy as np;rd=[r for r in json.load(open(\"$OUT/${name}.json\"))['response_details'] if not r.get('error')];print(round(np.percentile([1000*float(r['e2el']) for r in rd],99)))" 2>/dev/null) fail=$(python3 -c "import json;print(json.load(open(\"$OUT/${name}.json\")).get(\"failed\"))" 2>/dev/null)" >> $MARK; }
# native uniform under square wave
tmux new-session -d -s p3sqserve "cd ~/RouteBalance && CUDA_VISIBLE_DEVICES='' ROUTE_BALANCE_SQUARE=6,30 PYTHONPATH=\$PWD ROUTE_BALANCE_INPROC_PREDICTOR=1 HF_HOME=/mydata/hf_cache python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve --host 0.0.0.0 --port 8200 --model_config_path route_balance/config/route_balance/model_deployment.json --scheduling route_balance --chat --scheduler-config route_balance/config/route_balance/scheduler_config_qn_minmax.json --predictor-config route_balance/config/route_balance/predictor_xgboost_3model.json --host_config route_balance/config/host_configs.json >~/p3sqserve.log 2>&1"
for t in $(seq 1 30); do curl -sf -m4 localhost:8200/health >/dev/null && break; sleep 5; done
grep -q "INPROC-XGB.*loaded" ~/p3sqserve.log && echo "INPROC_OK" >> $MARK || { echo "INPROC_FAIL" >> $MARK; exit 1; }
post '{"scoring_weights":{"w_quality":0.33,"w_latency":0.33,"w_cost":0.33,"w_balance":0.0}}'
bench p3sq_uniform_sq
echo "P3SQ_PHASE1_DONE" >> $MARK
# pipeline enhanced baselines under square wave
pkill -f "[c]ara_serve" 2>/dev/null; sleep 4; tmux kill-session -t p3sqserve 2>/dev/null
tmux new-session -d -s p3sqpipe "cd ~/RouteBalance && ROUTE_BALANCE_SQUARE=6,30 PYTHONPATH=\$PWD HF_HOME=/mydata/hf_cache python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve --host 0.0.0.0 --port 8200 --model_config_path route_balance/config/route_balance/model_deployment.json --scheduling pipeline --chat --scheduler-config route_balance/config/route_balance/scheduler_config_qn_minmax.json --predictor-config route_balance/config/route_balance/predictor_xgboost_3model.json --host_config route_balance/config/host_configs.json >~/p3sqpipe.log 2>&1"
for t in $(seq 1 24); do curl -sf -m4 localhost:8200/health >/dev/null && break; sleep 5; done
python3 -c "import json;print(json.dumps({'router':{'type':'best_route_4way','kwargs':{'checkpoint_path':'models/route_balance/best_route_4way_qwen','confidence_threshold':0.5}},'dispatch':{'type':'shortest_queue'},'filter':{'type':'route_balance_tiered'}}))" > /tmp/p3sq_br.json
post @/tmp/p3sq_br.json; curl -sS --max-time 90 -X POST localhost:8200/v1/completions -H "Content-Type: application/json" -d '{"prompt":"warmup","model":"Qwen/Qwen2.5-3B","max_tokens":1}' >/dev/null 2>&1
bench p3sq_br4enh_sq
echo "P3SQ_DONE" >> $MARK
