#!/usr/bin/env bash
# P3 rigor fix: enhanced-AvengersPro under SQUARE WAVE (30/6 rps), matching p3sq.sh exactly.
# The square-wave avgpro cell was the one cell missing from Table 15 (br4enh + uniform already ran).
set -u
cd ~/RouteBalance; export PYTHONPATH=$PWD HF_HOME=/mydata/hf_cache ROUTE_BALANCE_SQUARE=6,30
MARK=~/P3SQAVG_STATUS; OUT=~/RouteBalance/experiment_output/v2_sweep_jun06/review_extras; : > $MARK
# process hygiene: kill any stale scheduler/bench (native reseeds are UNI_DONE)
pkill -f "[c]ara_serve" 2>/dev/null; pkill -f "[b]enchmark_serving" 2>/dev/null; sleep 5
for s in uni uniserve p3sqpipe p3sqavgpipe; do tmux kill-session -t $s 2>/dev/null; done
post(){ curl -sS --max-time 15 -X POST localhost:8200/v1/config -H "Content-Type: application/json" -d "$1" -o /dev/null -w "POST %{http_code}\n" >> $MARK; }
bench(){ local name=$1; python3 -u -m route_balance.benchmark.route_balance.benchmark_serving --seed 5 --num-prompts 3534 --dataset-name custom --dataset-path data/route_balance/best-route-v3-test-3534-eval.jsonl --custom-output-len 1024 --max-output-len 1024 --max-total-len 8192 --backend route_balance --base-url http://localhost:8200 --endpoint /v1/completions --request-rate 18 --burstiness 1.0 --save-result --save-detailed --result-dir $OUT --result-filename ${name}.json --rso-constraint-mode TIERED --model Qwen/Qwen2.5-3B --tokenizer Qwen/Qwen2.5-3B > $OUT/${name}.log 2>&1; echo "P3SQAVG ${name} e2e=$(python3 -c "import json;print(round(json.load(open(\"$OUT/${name}.json\")).get(\"mean_e2el_ms\",0)))" 2>/dev/null) p99=$(python3 -c "import json,numpy as np;rd=[r for r in json.load(open(\"$OUT/${name}.json\"))['response_details'] if not r.get('error')];print(round(np.percentile([1000*float(r['e2el']) for r in rd],99)))" 2>/dev/null) fail=$(python3 -c "import json;print(json.load(open(\"$OUT/${name}.json\")).get(\"failed\"))" 2>/dev/null)" >> $MARK; }
# pipeline serve under square wave (same launch as p3sq.sh PHASE 2)
tmux new-session -d -s p3sqavgpipe "cd ~/RouteBalance && ROUTE_BALANCE_SQUARE=6,30 PYTHONPATH=\$PWD HF_HOME=/mydata/hf_cache python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve --host 0.0.0.0 --port 8200 --model_config_path route_balance/config/route_balance/model_deployment.json --scheduling pipeline --chat --scheduler-config route_balance/config/route_balance/scheduler_config_qn_minmax.json --predictor-config route_balance/config/route_balance/predictor_xgboost_3model.json --host_config route_balance/config/host_configs.json >~/p3sqavgpipe.log 2>&1"
for t in $(seq 1 30); do curl -sf -m4 localhost:8200/health >/dev/null && break; sleep 5; done
curl -sf -m4 localhost:8200/health >/dev/null && echo "HEALTH_OK" >> $MARK || { echo "HEALTH_FAIL" >> $MARK; exit 1; }
# enhanced AvengersPro router (identical kwargs to p3_bursty.sh PHASE 2 avgenh cell)
python3 -c "import json;print(json.dumps({'router':{'type':'avengers_pro','kwargs':{'artifact_dir':'models/route_balance/avengers_pro_qwen_v2_lc0.20'}},'dispatch':{'type':'shortest_queue'},'filter':{'type':'route_balance_tiered'}}))" > /tmp/p3sq_av.json
post @/tmp/p3sq_av.json
curl -sS --max-time 90 -X POST localhost:8200/v1/completions -H "Content-Type: application/json" -d '{"prompt":"warmup","model":"Qwen/Qwen2.5-3B","max_tokens":1}' >/dev/null 2>&1
bench p3sq_avgenh_sq
echo "P3SQAVG_DONE" >> $MARK
