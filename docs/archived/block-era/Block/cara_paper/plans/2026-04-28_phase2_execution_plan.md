# Cara — Phase 2 Execution Plan (Apr 28 → Apr 29)

**Status**: drafted at end of Apr 28 latency-predictor work.
**Author**: autonomous overnight session, executing while user reviews tomorrow.
**Target deadline**: NeurIPS 2026 May 6 — 8 days remaining.

## What just landed (Apr 28)

- Phase A/B/C distributed (5 type-heads): 250 training cells + 50 test cells, 1.93M records.
- XGBoost iterated predictors per (model, GPU): MAE 0.51-12.69s, MAPE 5.6-10.5%.
- LSTM v2 predictors per (model, GPU): MAE 0.54-1.06s on 4/5 (p100 in flight).
- Models backed to VPS at `Block/models/cara/latency_predictors_apr28/`.
- Consolidated metrics at `Block/models/cara/latency_predictors_apr28/ALL_METRICS_SUMMARY.json`.

## Execution order (autonomous overnight)

### S0 — Block-wait for p100 LSTM (~15 min)
Monitor `bmyc6c02y` polls c240g5 until `lstm_v2_p100_3b.metrics.json` exists, then rsyncs model + metrics back to VPS.

### S1 — Restart cluster (~30 min wall)
On each head, restart vLLM (model load 5-10 min for 72B/14B; <2 min for 3B/7B) and re-launch
cara_serve mini-coordinator (port per head topology). Verify `/health` returns 200 + `/v1/models`
returns expected model.

### S2 — Wire iterated XGBoost into cara_serve (~30 min)
Each head's cara_serve currently loads predictors via `predictor_deployment_config.json` — points at
old (Mar 23) models. Update each head's config to point at `/mydata/predictors/xgboost_iterated/xgboost_<type>.json`,
restart cara_serve, smoke-test that `/v1/estimate` returns sensible per-instance latency.

Acceptance:
- `/v1/estimate` for each model returns latency estimate matching XGBoost prediction within 5% on
  10 sample prompts.
- No errors in cara_serve log.

### S3 — Predictor side-tests (~1.5h, parallel-able)
- **Inference latency**: time `xgboost.predict` per request on each head; verify <0.5ms (cara_serve
  per-request budget). Single Python script per head.
- **Per-cell saturation MAE**: compute MAE conditioned on QPS-bucket (under-saturated, near-knee,
  over-saturated). Each head writes `Block/models/cara/latency_predictors_apr28/saturation_bucket_mae_<type>.json`.
- **Held-out QPS-level extrapolation**: re-train XGBoost on multipliers ≤0.9, evaluate on 1.0/1.1/1.2.
  Tests over-saturation extrapolation. Write to `extrapolation_<type>.json`.

### S4 — §7.A E2E Preflight (~2h)
Run 5 main-table rows at default knob × λ=10 × 100 requests on the clean bench (`best-route-v3-test-500-clean.jsonl`).

Pass criteria:
- All 5 rows return 100 success responses.
- deepeval coverage 100% on each row.
- Scheduler decisions not pathological (no all-to-one routing).
- E2E latency reasonable (mean < 30s for λ=10).

If FAIL: log to handoff doc, halt main run launches.

### S5 — Coarse λ pilot (~3h)
λ ∈ {2, 5, 10, 15, 20, 30}, 5 rows × 1 default knob × 200 requests = 30 runs. Find each row's
saturation knee. Lock final λ grid.

### S6 — Main λ-axis curves (~5h)
5 rows × 7 λ × 1 default knob × 500 requests = 35 runs. The headline figure data.

Output: `cara_paper/exp/phase2_lambda_axis/{row}_{lambda}_{knob}.jsonl`.

### S7 — IF S6 finishes overnight: Pareto sweeps (~5h)
5 rows × 5 knobs × 2 anchor λ = 50 runs.

Output: `cara_paper/exp/phase2_pareto/{row}_{lambda}_{knob}.jsonl`.

### Defer to next session
- Ablations at 3 anchors (30 runs)
- Supporting studies (~12 runs)
- vLLM-SR Docker setup (Day-3 deferred)
- Aggregation + paper figures

## Halt-and-page conditions

If any of these happen, the autonomous session halts and writes a `BLOCKER` section to the handoff:
1. Predictor wiring smoke fails (S2 acceptance not met).
2. Preflight (S4) fails on >2 rows.
3. Coarse λ pilot reveals broken baseline (e.g., RouteLLM-mf collapsed to all-3B again).
4. Disk on any head exceeds 90% (mitigation: pause writes, alert).
5. Cluster process count drops without reason.

## Manual review checkpoints

User asked to review tomorrow morning (Apr 29). Expected to find:
- Handoff doc with hourly progress sections
- Predictor wiring results (S2)
- Preflight pass/fail (S4)
- λ pilot results (S5)
- Optionally: main λ-axis curves (S6) or partial Pareto data (S7)

## Storage discipline (post-disk-full incident Apr 27)

- All cluster experiment writes go to `/mydata/...` — NEVER `/tmp/...`.
- VPS only stores models + metrics + plan docs (small, <100MB total).
- Disk audit in handoff cron tick reports `/mydata` and `/` usage per head.
- If `/` on any head crosses 92% — halt training/runs, surface to user.

## Open questions for tomorrow's review

1. Is the iterated XGBoost MAE acceptable for paper submission? (Mar 23 baseline target was MAE within 20% — we're well within that.)
2. Should we commit the patched code (`generate_latency_benchmark.py`, `train_lstm_latency_v2.py`,
   `cara_serve.py uvloop`, `offline_training/__init__.py`) to git?
3. Phase 2 runs use bench `best-route-v3-test-500-clean.jsonl` (500 prompts × 4 models). Sufficient
   sample size, or pull more?
4. Some §7.A baselines (vLLM-SR) need Docker setup that's deferred — ok to skip for first paper draft?
