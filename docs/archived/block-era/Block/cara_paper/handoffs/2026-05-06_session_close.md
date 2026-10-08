# 2026-05-06 Session Close — Cluster Released, Paper-Side Resume

**Status**: Cluster released ~08:06 UTC May 6. NeurIPS submission today (May 6). vLLM-SR investigation terminated; deferred to June 10 follow-up sessions.

## Final Data State on VPS (`/home/wd312/Code/llm/Block/`)

### Experimental cells — 215 total across all tracks

Path: `cara_paper/smoke_test_apr_13/results/main_table_full_3534/cara_final_e2e/`

| Track | Subdir | Cells | Status | INDEX |
|---|---|---|---|---|
| Headline | `cara_weight_sweep/` | 56 | ✅ complete | INDEX_overnight_20260504.md |
| Headline | `dispatcher_only_baselines/` | 21 | ✅ complete | INDEX_overnight_20260504.md |
| T2a | `best_route_4way_threshold/` | 42 | ✅ complete | INDEX_t2_20260505.md |
| T2b | `avengers_pro_4m_pw_sweep/` | 56 | ✅ complete | INDEX_t2_20260505.md |
| T3 | `budget_calibrated_demo/` | 9 | ✅ complete | INDEX_t3fix_t4_20260506.md |
| T4a | `ablation_lpt/` | 7 | ✅ complete | INDEX_t3fix_t4_20260506.md |
| T4b/c | `ablation_batch/` | 19 | ✅ complete | INDEX_t3fix_t4_20260506.md |
| T5 | `slo_ablation/` | 5 | ⚠️ partial (planned 14) | INDEX_t5_20260506.md |

**`cara_final_e2e/_aggregate/` is EMPTY** — aggregator only ran on the overnight grid (105 cells). Post-overnight tracks (T2/T3/T4/T5 = 138 cells) have NOT been aggregated yet. This is the highest-priority paper-side blocker.

### Predictor models (all on VPS, byte-verified)

Path: `models/cara/latency/`

- `deploy_tpot/`: 5 .xgb files (qwen2.5-{3b_a30, 3b_p100, 7b_a30, 14b_v100, 72b_a100}) + training_metrics.json + xgboost_metadata.json
- `deploy_ttft/`: same 5 instance types
- `deploy_e2e/`: same 5 instance types
- `block/config/cara/predictor_xgboost_3model.json` — predictor config (paths reference `/users/asdwb/Block/...` cluster paths, future sessions need remap to local path)

### Test data (on VPS)

- `data/cara/best-route-v3-test-3534-eval.jsonl` — 3534 prompts
- `data/cara/scored/test_scored_filtered.jsonl` — 22MB, full deepeval-scored prompt set

### Cluster archive

Path: `cara_paper/handoffs/_cluster_archive_20260506/`

- `_session_archive_d8545.tgz` (21M) — scheduler /tmp scratch dir (analysis scripts, build_calibrated_budgets.py, audit_errors.py, etc.)
- `_p100_essentials.tgz` (527K) — P100 vllm-sr router + Envoy YAMLs + patched p100_full_data_eval.py + λ=30 result jsonl
- `scheduler_config_carafast.json` — cara_serve boot config

## Predictor Accuracy — MAE Framing (NEW DECISION)

**Decision locked 2026-05-06**: Report TPOT/TTFT predictor accuracy as **MAE in seconds**, NOT MAPE. MAPE blows up on tiny TPOT denominators (sub-millisecond per-token). E2E MAPE is fine to report (<10% all instances).

### TPOT — `models/cara/latency/deploy_tpot/training_metrics.json`

| instance | MAE (s/tok) | MAPE | RMSE |
|---|---|---|---|
| qwen2.5-72b_a100 | 0.000750 | 1.82% | 0.00152 |
| qwen2.5-3b_a30 | 0.000222 | 38.44% | 0.00054 |
| qwen2.5-7b_a30 | 0.000514 | 64.68% | 0.00096 |
| qwen2.5-14b_v100 | 0.000879 | 152.88% | 0.00146 |
| qwen2.5-3b_p100 | 0.005648 | 6.85% | 0.00818 |

MAE << inter-instance routing gap (3B-on-A30 vs 72B-on-A100 differs ~50× in TPOT), so predictor reliably distinguishes routing targets despite MAPE looking alarming.

### TTFT — `models/cara/latency/deploy_ttft/training_metrics.json`
All 5 instances: MAE 4-37 ms, MAPE 14-20%.

### E2E — `models/cara/latency/deploy_e2e/training_metrics.json`
All 5 instances: MAPE 1.22%–6.25% (well below 10%) — OK to report as MAPE.

## vLLM-SR — DEFERRED to June 10

**Investigation terminated 2026-05-06 07:42 UTC.** Reason: λ=30 cell ran 357s with 373/3534 errors (10.6%): 217 timeouts + 156 HTTP 503. Mean latency 23.4s, p99 155s; throughput plateaued at 9.9 req/s vs 30 req/s target — saturation symptom but ALSO had a confounder.

### Confounder discovered: internal-IP routing dead NIC

User reminded mid-investigation: **"use public ip always"** (rule documented elsewhere, missed by me). Original Envoy used internal 10.10.1.x; `10.10.1.3` (c4130-110433's experiment-LAN NIC) was unreachable from peers ("No route to host") even though vLLM was healthy at 127.0.0.1:8000 on that node. Rewrote Envoy to public 128.105.146.x — all 13 backends reachable.

λ=24 retry with public IPs + timeout=60s (lowered from 120s) gave 1080/3534 errors (30.6%). Worse — likely from the timeout reduction catching tail latencies (p99=46s, max=59s near the 60s line) AND public-IP NIC having lower throughput than experiment LAN. User called it: "leave for next run".

### What's saved for June 10 resume

- `_cluster_archive_20260506/_p100_essentials.tgz`: working router + Envoy YAMLs (post public-IP fix), patched `p100_full_data_eval.py` with Poisson + cara penalties (`frequency_penalty=1.2`, `repetition_penalty=1.0`), 1× λ=30 result jsonl as failure baseline.
- For June 10: keep public IPs, raise timeout back to 120s, try λ ∈ {6, 8, 10} only (offered load below cluster ceiling).

### Paper framing for vLLM-SR

Same posture as RouteLLM-mf — listed in related work, NOT in headline main table. Reason: cross-cluster Envoy → main vLLM couldn't sustain cara's open-loop Poisson axis for apples-to-apples comparison.

## Tasks Status

- T1 (vLLM-SR): DEFERRED to June 10 (#10, #39, #60 deleted)
- T2 (AvengersPro pw + BEST-Route threshold): COMPLETE (#81 #82 done — update needed)
- T3 (budget control): COMPLETE (#83 #84 #85)
- T4 (LPT + batch ablation): COMPLETE (#86 #87)
- T5 (SLO filter ablation): PARTIAL 5/14 cells (#88)

## Highest-Priority Paper-Side Work (next session)

1. **Run aggregator on T2/T3/T4/T5 cells** — `cara_final_e2e/_aggregate/` empty for 138 cells. Use `cara_paper/smoke_test_apr_13/scripts/aggregate_overnight_grid.py` as template; extend per-track. Output: `_aggregate/AGGREGATE.{jsonl,csv}` + per-track figs.
2. **Regenerate paper tables with MAE framing** — predictor accuracy section.
3. **Main-table rows lock** — verify cara wins 3-axis Pareto on the broader sweep (8 weight tuples × 7 λ + AvengersPro pw {0.25, 0.4, 0.53, 0.7} × 2 disp × 7 λ + BEST-Route argmax+thresholds × 2 disp × 7 λ + dispatcher-only × 3 disp × 7 λ).
4. **Paper text edits** — update §experimental setup (predictor MAE framing), §main results (broader sweep numbers), drop vLLM-SR from headline.

## VPS Storage State

- After cleanup (deleted 960M of `_invalid_*` + smoke + SoCC_revision): 1.8GB free
- If more space needed: `models/cara/roberta_fused_5ep/` (1.9GB) is archived per MEMORY (RoBERTa-fused deferred), `models/cara/length_regression/` (1.2GB) and `length_bucket/` (1.2GB) similarly archived.

## CLAUDE Memory Notes (for refresh)

If user removes the auto-memory hook on session refresh, this doc + `cara_paper/handoffs/2026-04-28_handoff.md` (rolling handoff) + the INDEX_*.md files in `cara_final_e2e/` are the durable resume points. Critical memories that should survive somewhere:

- **MAE not MAPE** for TPOT/TTFT framing
- **Public IP always** for cluster routing (10.10.1.x has dead-NIC failure modes)
- **vLLM-SR deferred to June 10**, not part of NeurIPS submission
- **Aggregator gap on 138 post-overnight cells** is the headline blocker
