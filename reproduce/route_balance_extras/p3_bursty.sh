#!/usr/bin/env bash
set -u
cd ~/RouteBalance; export PYTHONPATH=$PWD HF_HOME=/mydata/hf_cache
MARK=~/P3_STATUS; OUT=~/RouteBalance/experiment_output/v2_sweep_jun06/review_extras; : > $MARK
pkill -f "[c]ara_serve" 2>/dev/null; pkill -f "[b]enchmark_serving" 2>/dev/null; sleep 5
for s in p2serve p3serve; do tmux kill-session -t $s 2>/dev/null; done
post(){ curl -sS --max-time 15 -X POST localhost:8200/v1/config -H "Content-Type: application/json" -d "$1" -o /dev/null -w "POST %{http_code}\n" >> $MARK; }
bench(){ local name=$1 lam=$2 b=$3; python3 -u -m route_balance.benchmark.route_balance.benchmark_serving --seed 5 --num-prompts 3534 --dataset-name custom --dataset-path data/route_balance/best-route-v3-test-3534-eval.jsonl --custom-output-len 1024 --max-output-len 1024 --max-total-len 8192 --backend route_balance --base-url http://localhost:8200 --endpoint /v1/completions --request-rate $lam --burstiness $b --save-result --save-detailed --result-dir $OUT --result-filename ${name}.json --rso-constraint-mode TIERED --model Qwen/Qwen2.5-3B --tokenizer Qwen/Qwen2.5-3B > $OUT/${name}.log 2>&1; echo "P3 ${name} e2e=$(python3 -c "import json;print(round(json.load(open(\"$OUT/${name}.json\")).get(\"mean_e2el_ms\",0)))" 2>/dev/null) p99=$(python3 -c "import json,numpy as np;rd=[r for r in json.load(open(\"$OUT/${name}.json\"))['response_details'] if not r.get('error')];print(round(np.percentile([1000*float(r['e2el']) for r in rd],99)))" 2>/dev/null) fail=$(python3 -c "import json;print(json.load(open(\"$OUT/${name}.json\")).get(\"failed\"))" 2>/dev/null)" >> $MARK; }
# PHASE 1: native uniform, bursty b=0.3, lambda18 mean (CUDA recipe)
tmux new-session -d -s p3serve "cd ~/RouteBalance && CUDA_VISIBLE_DEVICES='' PYTHONPATH=\$PWD ROUTE_BALANCE_INPROC_PREDICTOR=1 HF_HOME=/mydata/hf_cache python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve --host 0.0.0.0 --port 8200 --model_config_path route_balance/config/route_balance/model_deployment.json --scheduling route_balance --chat --scheduler-config route_balance/config/route_balance/scheduler_config_qn_minmax.json --predictor-config route_balance/config/route_balance/predictor_xgboost_3model.json --host_config route_balance/config/host_configs.json >~/p3serve.log 2>&1"
for t in $(seq 1 30); do curl -sf -m4 localhost:8200/health >/dev/null && break; sleep 5; done
grep -q "INPROC-XGB.*loaded" ~/p3serve.log && echo "INPROC_OK" >> $MARK || { echo "INPROC_FAIL" >> $MARK; exit 1; }
post '{"scoring_weights":{"w_quality":0.33,"w_latency":0.33,"w_cost":0.33,"w_balance":0.0}}'
bench p3_uniform_l18_b0.3 18 0.3
echo "PHASE1_DONE" >> $MARK
# PHASE 2: pipeline, enhanced br4 + enhanced avgpro, bursty b=0.3, lambda18
pkill -f "[c]ara_serve" 2>/dev/null; sleep 4; tmux kill-session -t p3serve 2>/dev/null
tmux new-session -d -s p3pipe "cd ~/RouteBalance && PYTHONPATH=\$PWD HF_HOME=/mydata/hf_cache python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve --host 0.0.0.0 --port 8200 --model_config_path route_balance/config/route_balance/model_deployment.json --scheduling pipeline --chat --scheduler-config route_balance/config/route_balance/scheduler_config_qn_minmax.json --predictor-config route_balance/config/route_balance/predictor_xgboost_3model.json --host_config route_balance/config/host_configs.json >~/p3pipe.log 2>&1"
for t in $(seq 1 24); do curl -sf -m4 localhost:8200/health >/dev/null && break; sleep 5; done
python3 -c "import json;print(json.dumps({'router':{'type':'best_route_4way','kwargs':{'checkpoint_path':'models/route_balance/best_route_4way_qwen','confidence_threshold':0.5}},'dispatch':{'type':'shortest_queue'},'filter':{'type':'route_balance_tiered'}}))" > /tmp/p3_br.json
post @/tmp/p3_br.json; curl -sS --max-time 90 -X POST localhost:8200/v1/completions -H "Content-Type: application/json" -d '{"prompt":"warmup","model":"Qwen/Qwen2.5-3B","max_tokens":1}' >/dev/null 2>&1
bench p3_br4enh_l18_b0.3 18 0.3
python3 -c "import json;print(json.dumps({'router':{'type':'avengers_pro','kwargs':{'artifact_dir':'models/route_balance/avengers_pro_qwen_v2_lc0.20'}},'dispatch':{'type':'shortest_queue'},'filter':{'type':'route_balance_tiered'}}))" > /tmp/p3_av.json
post @/tmp/p3_av.json; curl -sS --max-time 90 -X POST localhost:8200/v1/completions -H "Content-Type: application/json" -d '{"prompt":"warmup","model":"Qwen/Qwen2.5-3B","max_tokens":1}' >/dev/null 2>&1
bench p3_avgenh_l18_b0.3 18 0.3
echo "P3_DONE" >> $MARK
