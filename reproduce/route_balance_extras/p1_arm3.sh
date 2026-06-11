#!/usr/bin/env bash
set -u
cd ~/RouteBalance; export PYTHONPATH=$PWD HF_HOME=/mydata/hf_cache
MARK=~/P1_STATUS; OUT=~/RouteBalance/experiment_output/v2_sweep_jun06/review_extras; : > $MARK
pkill -f "[c]ara_serve" 2>/dev/null; pkill -f "[b]enchmark_serving" 2>/dev/null; sleep 5
for s in reproserve ta2serve taserve p1serve; do tmux kill-session -t $s 2>/dev/null; done
# native recipe: CUDA hidden + INPROC + NO OMP pin + ROUTE_BALANCE_TIEBREAK=that (T-hat within-tier)
tmux new-session -d -s p1serve "cd ~/RouteBalance && CUDA_VISIBLE_DEVICES='' ROUTE_BALANCE_TIEBREAK=that PYTHONPATH=\$PWD ROUTE_BALANCE_INPROC_PREDICTOR=1 HF_HOME=/mydata/hf_cache python3 -u -m route_balance.global_scheduler.route_balance.route_balance_serve --host 0.0.0.0 --port 8200 --model_config_path route_balance/config/route_balance/model_deployment.json --scheduling route_balance --chat --scheduler-config route_balance/config/route_balance/scheduler_config_qn_minmax.json --predictor-config route_balance/config/route_balance/predictor_xgboost_3model.json --host_config route_balance/config/host_configs.json >~/p1serve.log 2>&1"
for t in $(seq 1 30); do curl -sf -m4 localhost:8200/health >/dev/null && break; sleep 5; done
grep -q "INPROC-XGB.*loaded" ~/p1serve.log && echo "INPROC_OK" >> $MARK || { echo "INPROC_FAIL" >> $MARK; exit 1; }
# w_lat=0 renorm to w_q=w_c=0.5 (IDENTICAL to arm 2, so same mix) + T-hat tiebreak (arm 3)
curl -sS --max-time 15 -X POST localhost:8200/v1/config -H "Content-Type: application/json" -d '{"scoring_weights":{"w_quality":0.5,"w_latency":0.0,"w_cost":0.5,"w_balance":0.0}}' -o /dev/null -w "POST %{http_code}\n" >> $MARK
bench(){ local name=$1 lam=$2 n=$3; python3 -u -m route_balance.benchmark.route_balance.benchmark_serving --seed 5 --num-prompts $n --dataset-name custom --dataset-path data/route_balance/best-route-v3-test-3534-eval.jsonl --custom-output-len 1024 --max-output-len 1024 --max-total-len 8192 --backend route_balance --base-url http://localhost:8200 --endpoint /v1/completions --request-rate $lam --burstiness 1.0 --save-result --save-detailed --result-dir $OUT --result-filename ${name}.json --rso-constraint-mode TIERED --model Qwen/Qwen2.5-3B --tokenizer Qwen/Qwen2.5-3B > $OUT/${name}.log 2>&1; local e=$(python3 -c "import json;print(round(json.load(open(\"$OUT/${name}.json\")).get(\"mean_e2el_ms\",0)))" 2>/dev/null); local d=$(python3 -c "import json;from collections import Counter;rd=[r for r in json.load(open(\"$OUT/${name}.json\"))['response_details'] if not r.get('error')];c=Counter(r.get('model','').split('/')[-1].replace('Qwen2.5-','') for r in rd);t=sum(c.values());print('/'.join(str(round(100*c.get(k,0)/t)) for k in ('3B','7B','14B','72B')))" 2>/dev/null); echo "P1 ${name} e2e=${e} 72Bmix=${d} fail=$(python3 -c "import json;print(json.load(open(\"$OUT/${name}.json\")).get(\"failed\"))" 2>/dev/null)" >> $MARK; echo $e; }
# SMOKE: l6 300 prompts, gate (mix must match arm2 ~38% 3B, serve healthy, e2e sane)
G=$(bench p1_smoke_arm3_l6 6 300)
echo "SMOKE_DONE e2e=$G" >> $MARK
# arm 3 full cells: w_lat=0 + T-hat tiebreak at l12/24/30
for LAM in 12 24 30; do bench p1_arm3_that_l${LAM} $LAM 3534 >/dev/null; done
echo "P1_DONE" >> $MARK
