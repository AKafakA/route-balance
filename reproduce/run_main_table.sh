#!/usr/bin/env bash
# run_main_table.sh — rev-3 5-row × knob × λ sweep.
#
# Design: route_balance_paper/codex/apr15_baseline_comparison_design.md §3.2, §4.6
#
# Rows:
#   r1 RouteBalance-full              knob = w_quality ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
#   r2 vllm_sr-wrapper        single configuration
#   r3 RouteLLM-mf-wrapper    knob = threshold α ∈ {0.05, 0.11593, 0.2, 0.3, 0.5}
#   r4 Avengers-Pro-wrapper   knob = performance_weight ∈ {0.25, 0.39, 0.53, 0.7, 0.8}
#   r5 BEST-Route-wrapper     knob = threshold t ∈ {0.3, 0.5, 0.6, 0.7, 0.8}
#
# Wrappers (r3, r4, r5) are paired with both round_robin and shortest_queue
# dispatchers per design §3.3; llumnix_minus dispatcher is optional (OPT_LLUMNIX=1).
#
# MODE (env):
#   preflight    5 rows × default knob × λ=10 (§7.A preflight)
#   lambda_axis  5 rows × default knob × full λ grid (F1-F3 curves)
#   pareto       5 rows × full knob × anchor rates (F4-F5 Pareto)
#
# Phase split because route_balance_serve --scheduling {route_balance|pipeline} requires restart:
#   PHASE=route_balance      r1 RouteBalance-full + r2 vllm_sr-wrapper (requires route_balance_serve --scheduling route_balance)
#   PHASE=pipeline  r3 + r4 + r5 wrappers            (requires route_balance_serve --scheduling pipeline)
#
# Total cluster time estimate (per handoff rev-3):
#   preflight    5 runs × 1 min    ≈  5 min
#   lambda_axis  35 runs × 1-2 min ≈ 35-70 min
#   pareto       50 runs × 1-2 min ≈ 50-100 min

set -uo pipefail
cd ~/RouteBalance

PHASE="${PHASE:-route_balance}"
MODE="${MODE:-preflight}"
NUM_PROMPTS="${NUM_PROMPTS:-500}"
OPT_LLUMNIX="${OPT_LLUMNIX:-0}"

# λ grids
LAMBDA_FULL="${LAMBDA_FULL:-4 8 12 16 20 24 32}"
LAMBDA_ANCHOR="${LAMBDA_ANCHOR:-12 24}"
LAMBDA_PREFLIGHT="${LAMBDA_PREFLIGHT:-10}"

# Default (per-router) knob values — used when MODE != pareto
ROUTE_BALANCE_DEFAULT="${ROUTE_BALANCE_DEFAULT_KNOB:-0.5}"
RLLM_DEFAULT="${RLLM_DEFAULT_KNOB:-0.11593}"
AVG_DEFAULT="${AVG_DEFAULT_KNOB:-0.7}"
BR_DEFAULT="${BR_DEFAULT_KNOB:-0.6}"

# Knob grids (5 values each) — used when MODE=pareto
ROUTE_BALANCE_KNOBS="${ROUTE_BALANCE_KNOBS:-0.1 0.3 0.5 0.7 0.9}"
RLLM_KNOBS="${RLLM_KNOBS:-0.05 0.11593 0.2 0.3 0.5}"
AVG_KNOBS="${AVG_KNOBS:-0.25 0.39 0.53 0.7 0.8}"
BR_KNOBS="${BR_KNOBS:-0.3 0.5 0.6 0.7 0.8}"

case "$MODE" in
  preflight)
    RATES="$LAMBDA_PREFLIGHT"
    ROUTE_BALANCE_K="$ROUTE_BALANCE_DEFAULT"; RLLM_K="$RLLM_DEFAULT"; AVG_K="$AVG_DEFAULT"; BR_K="$BR_DEFAULT"
    ;;
  lambda_axis)
    RATES="$LAMBDA_FULL"
    ROUTE_BALANCE_K="$ROUTE_BALANCE_DEFAULT"; RLLM_K="$RLLM_DEFAULT"; AVG_K="$AVG_DEFAULT"; BR_K="$BR_DEFAULT"
    ;;
  pareto)
    RATES="$LAMBDA_ANCHOR"
    ROUTE_BALANCE_K="$ROUTE_BALANCE_KNOBS"; RLLM_K="$RLLM_KNOBS"; AVG_K="$AVG_KNOBS"; BR_K="$BR_KNOBS"
    ;;
  *)
    echo "Unknown MODE='$MODE' (use preflight|lambda_axis|pareto)" >&2
    exit 2
    ;;
esac

# Wrapper dispatchers — RR always; shortest_queue always; llumnix_minus optional
DISPATCHERS="round_robin shortest_queue"
[[ "$OPT_LLUMNIX" == "1" ]] && DISPATCHERS="$DISPATCHERS llumnix_minus"

RESULTS="route_balance_paper/smoke_test_apr_13/results/main_table"
mkdir -p "$RESULTS"
STAMP=$(date -u +%Y%m%d_%H%M%S)
SUMMARY="$RESULTS/summary_${PHASE}_${MODE}_${STAMP}.jsonl"
: > "$SUMMARY"

ROUTE_BALANCE_URL="${ROUTE_BALANCE_URL:-http://127.0.0.1:8200}"
SR_URL="${SR_URL:-http://128.105.146.39:8899}"
# Default: rev-3 CLEAN bench (resampled Apr 15). Legacy contaminated file kept
# as an opt-in via DATASET env for audit only.
DATASET="${DATASET:-data/route_balance/best-route-v3-test-500-clean.jsonl}"
TOK="${TOK:-Qwen/Qwen2.5-3B}"

GOODPUT_FLAGS=(--goodput "ttft:1000" "tpot:50" "e2el:5000")
RSO_FLAGS=(--rso-rate 0.5 --rso-ttft-slo-ms 5000 --rso-tpot-slo-ms 200
           --rso-budget-cost 1.0 --rso-quality-min 0.5
           --rso-constraint-mode TIERED)
DETAIL_FLAGS=(--save-detailed)

# Warmup: trigger predictor cold-load before timed bench (Apr 16 finding).
warmup_estimator() {
  echo "  [warmup] trigger predictor cold-load..."
  curl -sS --max-time 30 -X POST "$ROUTE_BALANCE_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"warmup","model":"Qwen/Qwen2.5-3B","max_tokens":1}' > /dev/null 2>&1
  sleep 2
  echo "  [warmup] done."
}

emit_row() {
  local tag="$1"; local row="$2"; local system="$3"; local knob="$4"; local lam="$5"
  local disp="${6:-}"
  python3 -c "
import json, sys
d = json.load(open('$RESULTS/${tag}_${STAMP}.json'))
row = {
  'tag': '$tag', 'row': '$row', 'system': '$system',
  'knob': '$knob', 'lambda': float('$lam'),
  'dispatcher': '$disp',
  'ok': d.get('completed', 0), 'total': d.get('num_prompts', 0),
  'throughput_req_s': round(d.get('request_throughput', 0), 2),
  'output_tok_s': round(d.get('output_throughput', 0), 1),
  'ttft_p50_ms': round(d.get('median_ttft_ms', 0), 1),
  'ttft_p99_ms': round(d.get('p99_ttft_ms', 0), 1),
  'tpot_p50_ms': round(d.get('median_tpot_ms', 0), 1),
  'tpot_p99_ms': round(d.get('p99_tpot_ms', 0), 1),
  'e2el_p50_ms': round(d.get('median_e2el_ms', 0), 1),
  'e2el_p99_ms': round(d.get('p99_e2el_ms', 0), 1),
  'goodput_req_s': round(d.get('request_goodput', 0), 2) if 'request_goodput' in d else None,
}
print(json.dumps(row))
" | tee -a "$SUMMARY"
}

run_route_balance_full() {
  local knob="$1"; local lam="$2"
  local tag="r1_route_balance_full_wq${knob}_l${lam}"
  echo "=== RouteBalance-full ($tag) ==="
  local payload
  payload=$(python3 -c "import json; print(json.dumps({
    'scoring_weights':{'w_latency': round((1-${knob})/3,3),'w_cost': round((1-${knob})/3,3),'w_quality': ${knob},'w_balance': round((1-${knob})/3,3)},
    'slo_defaults':{'lpt_sort_key':'max','assignment_strategy':'scoring'},
    'filter':{'type':'route_balance_tiered'}
  }))")
  curl -sS --max-time 5 -X POST "$ROUTE_BALANCE_URL/v1/config" \
    -H "Content-Type: application/json" -d "$payload" > /dev/null
  python3 -m block.benchmark.route_balance.benchmark_serving \
    --backend route_balance --base-url "$ROUTE_BALANCE_URL" --endpoint /v1/completions \
    --model "Qwen/Qwen2.5-3B" --tokenizer "$TOK" \
    --dataset-name custom --dataset-path "$DATASET" \
    --num-prompts "$NUM_PROMPTS" --request-rate "$lam" \
    --result-dir "$RESULTS" --result-filename "${tag}_${STAMP}.json" \
    --save-result "${DETAIL_FLAGS[@]}" "${GOODPUT_FLAGS[@]}" "${RSO_FLAGS[@]}" \
    2>&1 | tail -30 > "$RESULTS/${tag}_${STAMP}.log"
  emit_row "$tag" "RouteBalance-full" "route_balance_full" "$knob" "$lam" "-"
}

run_sr_wrapper() {
  local lam="$1"
  local tag="r2_sr_wrapper_l${lam}"
  echo "=== vllm_sr-wrapper ($tag) ==="
  python3 -m block.benchmark.route_balance.benchmark_serving \
    --backend openai-chat --base-url "$SR_URL" --endpoint /v1/chat/completions \
    --model "MoM" --tokenizer "$TOK" \
    --dataset-name custom --dataset-path "$DATASET" \
    --num-prompts "$NUM_PROMPTS" --request-rate "$lam" \
    --result-dir "$RESULTS" --result-filename "${tag}_${STAMP}.json" \
    --save-result "${DETAIL_FLAGS[@]}" "${GOODPUT_FLAGS[@]}" \
    2>&1 | tail -30 > "$RESULTS/${tag}_${STAMP}.log"
  emit_row "$tag" "vllm_sr-wrapper" "sr_peer" "-" "$lam" "-"
}

# Shared runner for pipeline wrappers: given router type + kwargs + dispatcher,
# POST /v1/config and run the bench.
run_pipeline_wrapper() {
  local tag="$1"; local row="$2"; local router="$3"; local rkw="$4"
  local disp="$5"; local knob="$6"; local lam="$7"
  echo "=== $row ($tag dispatch=$disp) ==="
  local payload
  payload=$(python3 -c "import json; print(json.dumps({
    'router':{'type':'${router}','kwargs':${rkw}},
    'dispatch':{'type':'${disp}'},
    'filter':{'type':'route_balance_tiered'}
  }))")
  curl -sS --max-time 5 -X POST "$ROUTE_BALANCE_URL/v1/config" \
    -H "Content-Type: application/json" -d "$payload" > /dev/null
  # Per-router warmup: trigger lazy model loading (DeBERTa for BEST-Route,
  # MF checkpoint for RouteLLM, etc.) after config switch, before timed bench.
  curl -sS --max-time 60 -X POST "$ROUTE_BALANCE_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"warmup","model":"Qwen/Qwen2.5-3B","max_tokens":1}' > /dev/null 2>&1
  sleep 1
  python3 -m block.benchmark.route_balance.benchmark_serving \
    --backend route_balance --base-url "$ROUTE_BALANCE_URL" --endpoint /v1/completions \
    --model "Qwen/Qwen2.5-3B" --tokenizer "$TOK" \
    --dataset-name custom --dataset-path "$DATASET" \
    --num-prompts "$NUM_PROMPTS" --request-rate "$lam" \
    --result-dir "$RESULTS" --result-filename "${tag}_${STAMP}.json" \
    --save-result "${DETAIL_FLAGS[@]}" "${GOODPUT_FLAGS[@]}" "${RSO_FLAGS[@]}" \
    2>&1 | tail -30 > "$RESULTS/${tag}_${STAMP}.log"
  emit_row "$tag" "$row" "route_balance_pipeline" "$knob" "$lam" "$disp"
}

run_routellm_mf() {
  local knob="$1"; local lam="$2"; local disp="$3"
  local disp_short="${disp:0:2}"
  local tag="r3_routellm_mf_a${knob}_${disp_short}_l${lam}"
  local kw="{\"router_type\":\"mf\",\"checkpoint_path\":\"routellm/mf_gpt4_augmented\",\"threshold\":${knob}}"
  run_pipeline_wrapper "$tag" "RouteLLM-mf-wrapper" "routellm" "$kw" "$disp" "$knob" "$lam"
}

run_avengers_pro() {
  local knob="$1"; local lam="$2"; local disp="$3"
  local disp_short="${disp:0:2}"
  # Per-α artifact path (user decided: 5 separate artifacts)
  local ckpt_dir="models/route_balance/avengers_pro_qwen_pw${knob}"
  local tag="r4_avengers_pw${knob}_${disp_short}_l${lam}"
  local kw="{\"checkpoint_dir\":\"${ckpt_dir}\",\"performance_weight\":${knob}}"
  run_pipeline_wrapper "$tag" "Avengers-Pro-wrapper" "avengers_pro" "$kw" "$disp" "$knob" "$lam"
}

run_best_route() {
  local knob="$1"; local lam="$2"; local disp="$3"
  local disp_short="${disp:0:2}"
  local tag="r5_best_route_t${knob}_${disp_short}_l${lam}"
  # #62: trained DeBERTa-v3-small router. Until training lands, the wrapper
  # registration points at placeholder `best_route_wrapper` router; when #62 is
  # missing the POST /v1/config returns 400 and this call is skipped.
  local kw="{\"checkpoint_path\":\"models/route_balance/best_route_wrapper_qwen\",\"threshold\":${knob},\"strong_model\":\"Qwen/Qwen2.5-7B\",\"weak_model\":\"Qwen/Qwen2.5-3B\"}"
  run_pipeline_wrapper "$tag" "BEST-Route-wrapper" "best_route_wrapper" "$kw" "$disp" "$knob" "$lam"
}

# Top-level sweep
warmup_estimator  # prevent cold-start cascade (Apr 16 finding)
case "$PHASE" in
  route_balance)
    for lam in $RATES; do
      for k in $ROUTE_BALANCE_K; do run_route_balance_full "$k" "$lam"; done
      run_sr_wrapper "$lam"
    done
    ;;
  pipeline)
    for lam in $RATES; do
      for disp in $DISPATCHERS; do
        for k in $RLLM_K; do run_routellm_mf "$k" "$lam" "$disp"; done
        for k in $AVG_K;  do run_avengers_pro "$k" "$lam" "$disp"; done
        for k in $BR_K;   do run_best_route   "$k" "$lam" "$disp"; done
      done
    done
    ;;
  *)
    echo "Unknown PHASE='$PHASE' (use route_balance|pipeline)" >&2
    exit 2
    ;;
esac

echo
echo "=== DONE: mode=$MODE phase=$PHASE rates=\"$RATES\" summary=$SUMMARY ==="
