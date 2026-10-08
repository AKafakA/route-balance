# Cara 4× A30 Integration Smoke Test — 3-Day Plan (Apr 13–15, 2026)

> **Workspace:** `cara_paper/smoke_test_apr_13/` — single root for all configs, scripts, logs, results, handoff. All host & deployment configs are **auto-generated from the manifest** (`manifests/cl_manifest_a30_apr_12.xml`) via `scripts/generate_*.py`. Never hand-edit files under `configs/`. See `README.md` for layout and `TRACKER.md` for live status.

## Context

CSD3 RoBERTa-fused 200ep training (job 27444574) is pending Priority until **Apr 15 02:35** — the upgraded predictor will not be available for the first two days of this window. Personal GPU VM is occupied with other experiments, so the VPS's current 5-epoch short-trained predictors (`models/cara/roberta_fused_5ep/{length,length_bucket,judge_class,reference_score}`, `models/cara/latency/deploy_{e2e,ttft,tpot}`) are the only option for online testing.

A **new** 4× A30 CloudLab reservation (`asdwb-299799`, manifest `cara_paper/smoke_test_apr_13/manifests/cl_manifest_a30_apr_12.xml`, nodes d7525-10s10315/17/11/39) gives us a 3-day window to **re-validate features built since the last P100 smoke test (Apr 2)** before the full 18-node NeurIPS run on Apr 26 – May 6. The features to re-validate are non-trivial and have never run together on a real cluster at rate-4-node scale:

1. **New broadcasting pipeline** (`block/exp/cara/test_broadcasting.sh`, `sweep_broadcasting.sh`, `run_broadcasting_remote.sh`) — added `broadcast_results[]` field per request.
2. **New latency collection** (`/schedule_trace` now emits 6 fields incl. `actual_output_tokens` + `pending_decode_tokens`, `generate_latency_benchmark.py` passes actual output lengths) — data must conform to the HF `cara_latency_prediction` schema as defined by `block/predictor/cara/offline_training/upload_latency_v2.py:flatten_record` so we can re-upload without the concurrency-cap bug.
3. **Batch scheduling framework** (LPT sort with max/min/mean keys, BatchQueue, tiered SLO relaxation) — 26 unit tests pass, never run online.
4. **5 SLO filters** (none, SLOs-Serve, PolyServe, QLM, CaraCDF hard, CaraCDF tiered) — P100 test covered 6 modes but only at 2 QPS × 30 reqs.
5. **P-D disaggregation** (`block/cara_pd/cara_pd_serve.py`, `block/config/cara/pd_config_smoketest.json`) — feasibility validated Mar 28 on A30 TCP (1.67 Gbps), TTFT/TPOT not propagated to benchmark client (known gap).

**Goal:** produce meaningful early online numbers (*not* paper numbers — 5ep predictors understate quality), validated latency samples conforming to HF schema, and a punch-list of bugs to fix before Apr 26.

User decisions locked in (from AskUserQuestion):
- **Layout:** 2× Qwen-2.5-3B (node0, node1) + 2× Qwen-2.5-7B (node2, node3)
- **P-D:** Day 3 reconfigure (node0 → Prefill-7B, node1 → Decode-7B; node2, node3 remain regular Cara for A/B)
- **QPS sweep:** Low=4 / Med=10 / High=20 QPS

Hard constraint: d7525 has **no `/mydata`** — only 63 GB root. Must fit HF cache + 2×3B weights + 2×7B weights + logs carefully; no 32B, no full broadcasting sweep.

## Day 1 (Apr 13) — Bring-up + Functional Sanity

### 1.1 Bring-up (must use `setup.sh`, rule R2)
- **Auto-generate hosts files** from manifest: `python3 cara_paper/smoke_test_apr_13/scripts/generate_hosts.py` → writes `cara_paper/smoke_test_apr_13/configs/{hosts,ampere_hosts}`.
- Stage configs + run setup: `bash cara_paper/smoke_test_apr_13/scripts/run_setup.sh` — copies generated `configs/hosts` and `configs/ampere_hosts` into `block/config/` (with backup), then runs the authoritative `bash block/exp/cara/setup.sh`. Installs CUDA 12.8, vLLM `cara_v_11` branch, transformers==4.50.3 pin, torch, sentence-transformers. **Reboot is required** after install — handled by the wrapper.
- Sync short-trained predictors from VPS → each node's root (`scp -r models/cara/roberta_fused_5ep`, `models/cara/latency/deploy_{e2e,ttft,tpot}`). Target: `~/Block/models/cara/` on each node.
- Sync `data/cara/best-route-v3-test-500.jsonl` to node0.

### 1.2 Auto-generated deployment config
- `python3 cara_paper/smoke_test_apr_13/scripts/generate_deployment_config.py` — writes `cara_paper/smoke_test_apr_13/configs/model_deployment_a30_smoketest.json` from the manifest + locked role map (node0/1 = Qwen-2.5-3B, node2/3 = Qwen-2.5-7B).
- Scheduler launches on node0:8200 (config pointer).
- Role map is defined at the top of the generator; changing it requires editing PLAN.md first.

### 1.3 Instance-type symlink (guards against d7525/a30 mismatch, CODE_PAPER_AUDIT item)
In `block/predictor/cara/models/latency/deploy/`, verify `d7525/` symlink → `a30/` exists. If not, create it (this is model-directory convention, not hyperparameter — allowed per R3).

### 1.4 Functional sanity (adapt `block/exp/cara/p100_smoke_test.sh` template → NEW `cara_paper/smoke_test_apr_13/scripts/a30_smoke_test.sh`)
- Phase 1: single `curl /v1/completions` on each of 4 nodes returns 200
- Phase 2: launch `cara_serve.py --scheduling random`, 20 reqs, verify all 4 backends receive traffic
- Phase 3: `--scheduling cara` with default scheduler config, 50 reqs @ 4 QPS
- Phase 4: 5-filter sweep (none, slos_serve, polyserve, qlm, cara_tiered) × 50 reqs each @ 4 QPS
- **Verification A (CRITICAL):** GET `/v1/schedule_trace` on any backend — response must contain all 6 fields: `num_output_tokens`, `predicted_decode_tokens`, plus the prior 4. `pending_decode_tokens > 0` during Phase 3 under 4 QPS load.
- **Verification B:** `cara_serve` `/v1/scheduling_stats` — batch LPT sort + tiered relaxation events > 0.

**Day 1 deliverable:** sanity log + schedule_trace sample; go/no-go decision for Day 2.

## Day 2 (Apr 14) — Broadcasting + Latency Collection + Baseline Sweep

### 2.1 Broadcasting (reduced sweep — NOT full 6×500)
Run `block/exp/cara/run_broadcasting_remote.sh` with 2 of the 6 generation configs (`temp=0`, `temp=0.1,rep_pen=1.15`) × 100 prompts from `best-route-v3-test-500.jsonl`. All 4 models receive each prompt in parallel (broadcasting mode).
- **Verification:** output JSON has `broadcast_results[]` list with one entry per model, each containing `{model, ttft, e2el, output_len, generated_text}`. Field presence + non-null for all 4 models.
- Downstream consumer (`block/benchmark/cara/cara_end_point_func.py:119`) parses it without error.

### 2.2 Latency dataset collection @ low / med / high QPS
Drive load via `block/benchmark/cara/benchmark_serving.py`:
- **Low = 4 QPS × 50 reqs** (~13s soak)
- **Med = 10 QPS × 200 reqs** (20s soak)
- **High = 20 QPS × 200 reqs** (10s soak — overloaded, forces queuing)

Between runs, `curl /v1/schedule_trace` to dump records. Per rate, save to `cara_paper/smoke_test_apr_13/results/day2/latency_{low,med,high}.jsonl`.

### 2.3 HF schema validation (DRY RUN — no upload)
Pipe each of the 3 jsonl files through `block/predictor/cara/offline_training/upload_latency_v2.py:flatten_record` offline (write a small validation script, or run the existing function in a python REPL locally on VPS). Confirm:
- All 11 top-level fields present (`request_id`, `instance_id`, `instance_type`, `num_prompt_tokens`, `num_predicted_output_tokens`, `actual_output_tokens`, `actual_e2e_latency`, `actual_ttft`, `actual_tpot`, `prediction_timestamp`, `completion_timestamp`)
- 17 flattened schedule-state fields present
- `actual_output_tokens` ≠ null (proves new pipeline works)
- `pending_decode_tokens > 0` at high QPS (proves LSTM v2 would have non-trivial signal)
- Also produce `train_queue_details.jsonl`-shape output with `running_requests[]` / `waiting_requests[]`.

If schema matches → **defer actual HF re-upload to post-smoke** (requires user sign-off per project rules).

### 2.4 Baseline comparison at med rate (10 QPS × 200 reqs)
Run the 4 ready baselines from `cara_serve.py`:
1. `random` 2. `round_robin` 3. `shortest_queue` 4. `cara` (bucket+deepeval predictor, CaraCDF tiered filter)

Report TTFT/TPOT/E2E P50/P95, SLO goodput, avg predicted quality, balance CV. **5ep predictors → treat numbers as lower bound.**

**Day 2 deliverable:** validated latency sample dataset + baseline diff table + broadcasting schema confirmation.

## Day 3 (Apr 15) — P-D Smoke + Predictor Upgrade (if CSD3 finishes)

### 3.1 Reconfigure node0 & node1 as P-D pair (keep node2/node3 regular)
- Tear down `vllm_instance.py` on node0 & node1 (graceful, verify via `pgrep` per project rule).
- Launch per `block/config/cara/pd_config_smoketest.json`: node0:7100 = Prefill-Qwen-2.5-7B, node1:7200 = Decode-Qwen-2.5-7B (NIXL KV transport, A30 TCP 1.67 Gbps).
- Launch `block/cara_pd/cara_pd_serve.py --pd-config pd_config_smoketest.json --port 8200`. Node2 & node3 still serve 7B under regular `cara_serve.py` on port 8201 for A/B.

### 3.2 Fix benchmark-client P-D metrics gap
Edit `block/benchmark/cara/cara_end_point_func.py` to extract `ttft` and `tpot` from the `pd_info` subfield of the P-D response (currently line 326 in P-D returns prefill-only TTFT). **This is a code change — requires user confirmation at start of Day 3 per rule R3 ambiguity check.** Reason: it's not a hyperparameter/data change — it's metrics plumbing — but user should confirm scope.

### 3.3 P-D smoke vs regular A/B
- Run 100 prompts through P-D endpoint (port 8200) @ 4 QPS, then @ 10 QPS.
- Run same 100 prompts through regular Cara endpoint (port 8201) @ same rates.
- Compare: TTFT, TPOT, E2E, measured KV-transfer time vs analytical model (`t = prompt_tokens × 114688 bytes / 1.67 Gbps`, `cara_pd_serve.py:63-67`).
- **Verification:** analytical transfer time within ±20% of measured on 100-token prompts (~6.9 ms expected).

### 3.4 CSD3 predictor upgrade (OPPORTUNISTIC)
If job 27444574 finishes by ~14:00 Apr 15:
- Pull RoBERTa-fused 200ep weights from CSD3 `/rds/user/wd312/hpc-work/llm/Block/models/` to VPS, then to nodes.
- Re-run the Day 2 baseline comparison (Section 2.4) with upgraded predictor.
- If it doesn't finish: keep 5ep results as-is, note in handoff.

### 3.5 Handoff doc
Apr 15 handoff already landed at `cara_paper/handoffs/2026-04-15_handoff.md` (revision-3 methodology lock + open hard tasks + cluster state). Also indexed in `cara_paper/INDEX.md`. The previous `cara_paper/smoke_test_apr_13/handoff/` subdir was retired during the Apr 15 reorg — all handoffs now live in `cara_paper/handoffs/YYYY-MM-DD_handoff.md`.

**Day 3 deliverable:** P-D A/B numbers, opportunistic 200ep refresh, handoff doc.

## Files — READ-only vs NEW/EDIT

### Read only (templates and configs to reuse, not modify):
- `block/exp/setup.sh` — R2 compliant installer
- `block/exp/cara/p100_smoke_test.sh` — template for Day 1 sanity
- `block/exp/cara/run_broadcasting_remote.sh`, `test_broadcasting.sh`, `sweep_broadcasting.sh`
- `block/exp/cara/benchmark_estimator_overhead.py`
- `block/global_scheduler/cara/cara_serve.py` (launch, don't edit)
- `block/cara_pd/cara_pd_serve.py`
- `block/config/cara/pd_config_smoketest.json` (Day 3 as-is)
- `block/config/cara/scheduler_config_default.json`
- `block/benchmark/cara/benchmark_serving.py`
- `block/predictor/cara/offline_training/upload_latency_v2.py` — schema reference
- `data/cara/best-route-v3-test-500.jsonl`

### NEW files (all under `cara_paper/smoke_test_apr_13/`):
- `manifests/cl_manifest_a30_apr_12.xml` — copied from `block/cl_manifest_a30_apr_12.xml`
- `scripts/parse_manifest.py` — XML → normalized JSON records
- `scripts/generate_hosts.py` — manifest → `configs/{hosts,ampere_hosts}`
- `scripts/generate_deployment_config.py` — manifest + role map → `configs/model_deployment_a30_smoketest.json`
- `scripts/run_setup.sh` — stage + invoke `block/exp/cara/setup.sh`
- `scripts/a30_smoke_test.sh` — Day 1 runner (adapted from `p100_smoke_test.sh`)
- `scripts/a30_broadcasting.sh`, `scripts/a30_latency_sweep.sh`, `scripts/a30_baselines.sh`
- `scripts/a30_pd_smoke.sh` — Day 3 P-D runner
- `scripts/validate_latency_schema.py` — dry-run against `flatten_record`
- `configs/` — GENERATED outputs (hosts, ampere_hosts, model_deployment_a30_smoketest.json)
- `results/{day1,day2,day3}/` — experiment outputs
- `logs/` — timestamped stdout/stderr
- `handoff/HANDOFF_APRIL15.md` — end-of-window handoff

### Files staged into the main tree (with backup) by `scripts/run_setup.sh`:
- `block/config/hosts` — replaced from `configs/hosts`; previous version backed up with timestamp
- `block/config/ampere_hosts` — replaced from `configs/ampere_hosts`; previous backed up
- `block/config/cara/model_deployment_a30_smoketest.json` — copied from `configs/`

### Existing files to EDIT (only after explicit user confirmation on Day 3):
- `block/benchmark/cara/cara_end_point_func.py` — extract TTFT/TPOT from P-D `pd_info` (Section 3.2)

## Verification Checklist (end-of-window pass/fail)

- [x] Day 1: `setup.sh` returns 0 on all 4 nodes; `nvidia-smi` shows A30 24 GB free pre-launch (actual HW: A30 24 GB, not 48 GB)
- [ ] Day 1: 5-filter sweep at 4 QPS × 50 reqs completes without error
- [ ] Day 1: `/schedule_trace` response contains `actual_output_tokens` and `pending_decode_tokens` fields
- [ ] Day 2: `broadcast_results[]` has 1 entry per deployed model per request
- [ ] Day 2: `flatten_record` consumes all 3 latency files without schema error
- [ ] Day 2: `pending_decode_tokens > 0` in ≥50% of high-QPS records
- [ ] Day 2: 4 baselines produce distinguishable TTFT/TPOT/E2E P50/P95 numbers
- [ ] Day 3: P-D analytical transfer time within ±20% of measured
- [ ] Day 3: benchmark client reports non-zero TTFT *and* TPOT for P-D requests
- [ ] Day 3: `HANDOFF_APRIL15.md` written with open-bug list for Apr 26

## Risks & Constraints

1. **Disk (63 GB root, no `/mydata`)** — 2×3B (~6 GB) + 2×7B (~14 GB) weights + HF cache already ~45 GB. Monitor `df -h /` between phases; reduce logging verbosity if needed. Do NOT pull 200ep RoBERTa to node local disk until Day 3 Section 3.4.
2. **5ep predictor floor** — quality numbers are a *lower bound*. Communicate clearly in handoff; do not present as paper results. Per R7, hyperparameters remain locked.
3. **Instance-type mismatch** (d7525 vs a30) — model-dir symlink is the agreed workaround. If inference fails due to missing `d7525` dir, report and wait (per CLAUDE.md "report and wait" rule) rather than retraining.
4. **CSD3 predictor may not finish by Day 3** — Section 3.4 is opportunistic; baseline results stand on 5ep predictors if it slips.
5. **Broadcasting sweep reduced from 6×500 to 2×100** — documents this explicitly in handoff so the April 26 run knows to repeat full sweep.
6. **Rule R3 (no silent design decisions):** any deviation from the locked layout (2×3B+2×7B), P-D split, or QPS sweep must be escalated to user, not auto-resolved.
7. **Rule R4 (smoke before long run):** every phase runs 10-prompt dry-run first before the headline request count.
8. **Process hygiene:** before each launch, `pgrep -u $(whoami) -f 'vllm|cara_serve|cara_pd' -a` on every node, kill stale, confirm 0 before starting.

## Cadence

- Day 1 checkpoint: end-of-day go/no-go on sanity + schema fields (Verification A & B)
- Day 2 checkpoint: midday schema-match confirmed before baseline sweep
- Day 3 checkpoint: before editing `cara_end_point_func.py`, confirm with user (Section 3.2)
