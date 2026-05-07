#!/usr/bin/env bash
# G5 (#55) — RouteBalance-full + integrated-system ablation sweep.
#
# Prereq: route_balance_serve launched with `--scheduling route_balance` (NOT pipeline).
# Runs RouteBalance-full (row 1) then 10 ablation rows toggled via /v1/config:
#   row 2  RouteBalance-no-LPT            slo_defaults.lpt_sort_key = "none"
#   row 3  RouteBalance-no-filter         filter.type = "none"
#   row 4  RouteBalance-point-filter      filter.type = "route_balance_hard_reject"  (planned; falls back to route_balance_tiered today)
#   row 5  RouteBalance-no-quality-term   scoring_weights.w_quality = 0
#   row 6  RouteBalance-no-latency-term   scoring_weights.w_latency = 0
#   row 7  RouteBalance-no-cost-term      scoring_weights.w_cost    = 0
#   row 7b RouteBalance-no-balance-term   scoring_weights.w_balance = 0
#   row 8a RouteBalance-simple-shortestQ  slo_defaults.assignment_strategy = "shortest_queue"
#   row 8b RouteBalance-simple-llumnix    slo_defaults.assignment_strategy = "llumnix_minus"
#
# Uses the customized vLLM bench (route_balance/benchmark/route_balance/benchmark_serving.py).
# Emits per-row result JSON + a summary JSONL for the paper table.
#
# Supporting studies (router-signal swap, filter swap) live in full_bench_sweep.sh
# and require `--scheduling pipeline` mode — launch separately.

set -uo pipefail
cd ~/RouteBalance

NUM_PROMPTS="${1:-200}"
QPS="${2:-10}"
RESULTS=route_balance_paper/smoke_test_apr_13/results/ablation
mkdir -p "$RESULTS"
STAMP=$(date -u +%Y%m%d_%H%M%S)
SUMMARY="$RESULTS/summary_$STAMP.jsonl"
: > "$SUMMARY"

ROUTE_BALANCE_URL="${ROUTE_BALANCE_URL:-http://127.0.0.1:8200}"
DATASET="${DATASET:-data/route_balance/best-route-v3-test-500-clean.jsonl}"
TOK="${TOK:-Qwen/Qwen2.5-3B}"

# SLO / goodput: DistServe-style headline (ttft, tpot, e2el).
GOODPUT_FLAGS=(--goodput "ttft:1000" "tpot:50" "e2el:5000")
# 50% requests carry random per-request RSO; lenient tier for main table.
RSO_FLAGS=(--rso-rate 0.5 --rso-ttft-slo-ms 5000 --rso-tpot-slo-ms 200
           --rso-budget-cost 1.0 --rso-quality-min 0.5
           --rso-constraint-mode TIERED)
DETAIL_FLAGS=(--save-detailed)

# Starting full-RouteBalance defaults (row 1).  The server already has these; we POST
# them explicitly so each row is reproducible from scratch.
FULL_ROUTE_BALANCE_PAYLOAD='{
  "scoring_weights": {"w_latency":0.3,"w_cost":0.2,"w_quality":0.3,"w_balance":0.2},
  "slo_defaults": {"lpt_sort_key":"max","assignment_strategy":"scoring"},
  "filter": {"type":"route_balance_tiered"}
}'

snapshot_filter_stats() {
  local outfile="$1"
  curl -sS --max-time 3 "$ROUTE_BALANCE_URL/v1/scheduling_stats" > "$outfile" 2>/dev/null || echo "{}" > "$outfile"
}

# Warmup: send a single small request to trigger model estimator (RoBERTa)
# cold-load + JIT compilation before the timed benchmark. Without this,
# the first batch takes 8s for model loading and causes a cascading
# queue backlog that inflates E2E 10×. (Discovered during Apr 16 preflight.)
warmup_estimator() {
  echo "  [warmup] trigger predictor cold-load..."
  curl -sS --max-time 30 -X POST "$ROUTE_BALANCE_URL/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"warmup","model":"Qwen/Qwen2.5-3B","max_tokens":1}' > /dev/null 2>&1
  sleep 2
  echo "  [warmup] done."
}

WARMED_UP=0

# Usage: run_row <tag> <row_name> <override_json>
run_row() {
  local tag="$1"; local row="$2"; local override="$3"
  echo ""
  echo "=== $row ($tag) ==="

  # 1. Restore full-RouteBalance defaults, then apply row-specific override.
  curl -sS --max-time 5 -X POST "$ROUTE_BALANCE_URL/v1/config" \
    -H "Content-Type: application/json" \
    -d "$FULL_ROUTE_BALANCE_PAYLOAD" > /dev/null

  # Warmup once per ablation run (first row triggers predictor load).
  if [[ "$WARMED_UP" -eq 0 ]]; then
    warmup_estimator
    WARMED_UP=1
  fi
  curl -sS --max-time 5 -X POST "$ROUTE_BALANCE_URL/v1/config" \
    -H "Content-Type: application/json" \
    -d "$override" > /dev/null

  # 2. Snapshot pre-run counters.
  snapshot_filter_stats "$RESULTS/${tag}_${STAMP}.filter_pre.json"

  # 3. Run bench against /v1/completions (RouteBalance routes internally).
  python3 -m block.benchmark.route_balance.benchmark_serving \
    --backend route_balance --base-url "$ROUTE_BALANCE_URL" --endpoint /v1/completions \
    --model "Qwen/Qwen2.5-3B" --tokenizer "$TOK" \
    --dataset-name custom --dataset-path "$DATASET" \
    --num-prompts "$NUM_PROMPTS" --request-rate "$QPS" \
    --result-dir "$RESULTS" --result-filename "${tag}_${STAMP}.json" \
    --save-result "${DETAIL_FLAGS[@]}" "${GOODPUT_FLAGS[@]}" "${RSO_FLAGS[@]}" \
    2>&1 | tail -30 > "$RESULTS/${tag}_${STAMP}.log"

  snapshot_filter_stats "$RESULTS/${tag}_${STAMP}.filter_post.json"

  # 4. Emit summary row.
  python3 -c "
import json
d = json.load(open('$RESULTS/${tag}_${STAMP}.json'))
row = {
  'tag': '$tag', 'row': '$row', 'system': 'route_balance_full',
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

# Row 1 — RouteBalance-full (paper system)
run_row "row1_route_balance_full" "RouteBalance-full" '{}'

# Row 2 — RouteBalance-no-LPT (FIFO)
run_row "row2_no_lpt" "RouteBalance-no-LPT" \
  '{"slo_defaults":{"lpt_sort_key":"none"}}'

# Row 3 — RouteBalance-no-filter
run_row "row3_no_filter" "RouteBalance-no-filter" \
  '{"filter":{"type":"none"}}'

# Row 4 — RouteBalance-point-filter (route_balance_hard_reject approximates point-feasibility;
# true per-design unified point filter still planned — see design §5.11).
run_row "row4_point_filter" "RouteBalance-point-filter" \
  '{"filter":{"type":"route_balance_hard_reject"}}'

# Row 5 — w_quality = 0
run_row "row5_no_quality" "RouteBalance-no-quality-term" \
  '{"scoring_weights":{"w_quality":0.0}}'

# Row 6 — w_latency = 0
run_row "row6_no_latency" "RouteBalance-no-latency-term" \
  '{"scoring_weights":{"w_latency":0.0}}'

# Row 7 — w_cost = 0
run_row "row7_no_cost" "RouteBalance-no-cost-term" \
  '{"scoring_weights":{"w_cost":0.0}}'

# Row 7b — w_balance = 0
run_row "row7b_no_balance" "RouteBalance-no-balance-term" \
  '{"scoring_weights":{"w_balance":0.0}}'

# Row 8a — LPT + shortest_queue (scoring-based joint assignment dropped)
run_row "row8a_shortest_q" "RouteBalance-simple-shortestQ" \
  '{"slo_defaults":{"assignment_strategy":"shortest_queue"}}'

# Row 8b — LPT + llumnix_minus (scoring-based joint assignment dropped)
run_row "row8b_llumnix" "RouteBalance-simple-llumnix" \
  '{"slo_defaults":{"assignment_strategy":"llumnix_minus"}}'

# Row 9 — quality_greedy_ablation = RouteBalance-full with w_quality=1, all other weights=0.
# Structurally equivalent to the legacy `best_route.py` quality-argmax policy;
# per user decision (Apr 15 PM) realized as a zero-weight config instead of a
# standalone router. Distinct from the trained BEST-Route-wrapper (task #62)
# which ships as main-table row 5.
run_row "row9_quality_greedy" "quality_greedy_ablation" \
  '{"scoring_weights":{"w_quality":1.0,"w_latency":0.0,"w_cost":0.0,"w_balance":0.0}}'

# Restore full-RouteBalance defaults so server is left in the paper-system state.
curl -sS --max-time 5 -X POST "$ROUTE_BALANCE_URL/v1/config" \
  -H "Content-Type: application/json" -d "$FULL_ROUTE_BALANCE_PAYLOAD" > /dev/null

echo ""
echo "============ ROUTE_BALANCE-FULL ABLATION SUMMARY ============"
cat "$SUMMARY"
echo ""
echo "Master JSONL: $SUMMARY"
echo "Per-row details: $RESULTS/{row*_${STAMP}}.json"
