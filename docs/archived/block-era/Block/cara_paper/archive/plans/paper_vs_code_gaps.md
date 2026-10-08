# Paper vs Code Architecture Gaps — March 26, 2026

## STATUS: All 7 gaps IMPLEMENTED (March 26, 2026)

All gaps closed in a single implementation pass. Code changes:
- `block/predictor/cara/model_estimator.py` (NEW) — ModelEstimator interface + DefaultModelEstimator
- `block/predictor/cara/estimators/knn_estimator.py` — Added predict_similarity(), predict_judge()
- `block/global_scheduler/cara/cara_serve.py` — BatchQueue, batch loop, LPT sort, 4-term scoring, normalization, tiered constraints, sidecar latency
- `block/predictor/cara/cara_predictor_api_server.py` — Added /predict_latency sidecar endpoint
- `block/config/cara/predictor_config_learned.json` — model_estimator, w_balance, batch_config, instance_capacity
- `tests/global_scheduler/cara/test_batch_scheduling.py` — 26 unit tests (all passing)

### Design decisions:
- **LPT sort key**: max predicted output length across ALL candidate models (`max_m E[len_m(r)]`). Conservative — ensures longest potential output scheduled first regardless of final instance assignment. Updated in paper Algorithm 1 and Section 4 text.
- **ModelEstimator**: quality + length predictions run ONCE per prompt at scheduler side (not per-instance). Sidecar handles latency only.
- **Score type configurable**: "judge" (default), "similarity", or "combined" via config
- **Fused bucket classifier**: single ModernBERT checkpoint works for all target LLMs

---

## ORIGINAL GAPS (for reference)

### 1. No batch scheduling at all — FIXED
**Paper**: Requests collected into batches, LPT sorted, processed with batch-level state
**Code**: ~~Per-request online dispatch, no queue, no batching, no sorting~~ → BatchQueue + LPT + local state updates
**Impact**: Claims C, E, F, G, H, L all depend on this

### 2. Predictor architecture mismatch
**Paper**: Sidecar queries instance_stats locally, runs XGBoost TTFT/TPOT, returns to scheduler
**Code**: Scheduler loads learned_predictors locally, fetches instance_stats remotely
**Impact**: Paper's scalability argument (local prediction, no network overhead) is not true in code
**Fix**: Two options:
- Option A: Move XGBoost to sidecar, scheduler calls sidecar /predict
- Option B: Keep scheduler-side but fetch instance_stats via sidecar (sidecar as proxy)
- Option A matches paper better. Estimated: 1-2 days

### 3. No cluster balance term (w_u)
**Paper**: Score includes `w_u · ΔBalance(i, r, S)` with mean utilization tracking
**Code**: Score only has w_lat, w_cost, w_qual
**Fix**: Add w_u weight, track mean utilization during batch, penalize above-average assignments
**Depends on**: #1 (batch scheduling) — balance tracking only makes sense within a batch
**Estimated**: 0.5 day (after #1)

### 4. Hardcoded normalization instead of batch-level
**Paper**: Per-signal min-max normalization within batch
**Code**: `ttft / 10.0` and `cost / 0.05` hardcoded
**Fix**: After batch collection, compute min/max per signal across all candidates, normalize to [0,1]
**Depends on**: #1 (batch scheduling)
**Estimated**: 0.5 day (after #1)

### 5. No E2E soft scoring (two-tier incomplete)
**Paper**: Tier 1 = TTFT+TPOT for SLO filtering, Tier 2 = E2E for soft scoring
**Code**: Uses TTFT for both filtering AND scoring, no E2E predictor
**Fix**: Load XGBoost E2E model, use bucket E[length] as input, add to scoring
**Estimated**: 0.5 day

### 6. No tiered constraint handling
**Paper**: STRICT/RELAXED/TIERED modes with relax_order
**Code**: Only falls back to shortest_queue when all filtered
**Fix**: Parse relax_order from RSO, implement relaxation logic
**Estimated**: 1 day

### 7. No adaptive batch sizing
**Paper**: Dynamic batch size based on cluster load (busy instances → larger batch)
**Code**: No batching at all
**Fix**: Part of #1 — adjust max batch size / timeout based on fraction of busy instances
**Estimated**: Included in #1

## ALREADY MATCHES

| Claim | Status |
|-------|--------|
| D. Filter-then-rank | ✓ All 4 filters + weighted scoring |
| I. Safety-aware quality | ✓ In KNN training pipeline |
| #9 sampling override | ✓ Fixed (respects client params) |
| #8 deploy argparse | ✓ Fixed |
| #26 QPS control | ✓ Fixed (asyncio.create_task) |

## Implementation Priority Order

1. **Batch scheduling core** (#1) — everything else depends on this
2. **Predictor architecture** (#2) — aligns code with paper's scalability claim
3. **Balance + normalization** (#3, #4) — quick after batch is done
4. **E2E soft scoring** (#5) — quick, model already trained
5. **Tiered constraints** (#6) — nice to have for evaluation
6. **Adaptive batch sizing** (#7) — integrated with #1

## Timeline (April 1-12)

| Week | Tasks |
|------|-------|
| Apr 1-3 | #1 Batch scheduling + #7 adaptive sizing |
| Apr 4-5 | #2 Predictor architecture (sidecar XGBoost) |
| Apr 6-7 | #3 Balance + #4 Normalization + #5 E2E scoring |
| Apr 8-9 | #6 Tiered constraints |
| Apr 10-12 | CSD3 dry run to validate everything |

## Total Estimated Effort: ~8-10 days
