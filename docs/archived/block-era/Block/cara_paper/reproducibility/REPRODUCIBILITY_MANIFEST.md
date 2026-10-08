# v2 Reproducibility Manifest (2026-06-09)
Everything needed to reproduce the v2 results in a future CloudLab window. All artifacts are on
**openclaw** (`wd312@openclaw`, the persistent code+data hub) in the local Block repo; the cluster
itself is ephemeral. Pair this with `REPRODUCE_GUIDE.md` (step-by-step) and `v2_deployed_md5.txt` (checksums).

## 1. CODE — git (local repo, no remote)
- Checkout commit **`9ae1f20`** (master). Lineage: `2074f4a` (v2 stack code) → `2036690` (orchestration
  scripts + changelog lock) → `8107790` (consolidation + aggregator random-dispatcher fix) → `9ae1f20`
  (review closeout: producer `consolidate_v2.py`, latent-bug quarantine, §6.3 honesty fix).
- Online AKafakA/Block is STALE — do NOT use it. This local repo is authoritative.

## 2. MODELS
### 2a. Re-pull from HuggingFace (public/ungated — NOT stored on openclaw; too large)
| model | role | gated? |
|---|---|---|
| `Qwen/Qwen2.5-3B` / `-7B` / `-14B` / `-72B` | served LLM pool (the 4 tiers) | no |
| `unsloth/gemma-3-12b-it` | judge-alt model (#15c) | **no** (ungated mirror) |
| `meta-llama/Llama-3.1-8B-Instruct` | MAIN deepeval judge | **GATED** (need HF license) |
Note: `google/gemma-4-12B-it` is ungated but needs CUDA-13/driver≥580 — the CloudLab A100s are driver
570/CUDA-12.8, so the judge MUST be gemma-3 with **vllm 0.10.2 (cu128) + transformers 4.57.6** (see Guide).

### 2b. Custom predictor/baseline models — ON openclaw (`models/cara/`), md5s in `v2_deployed_md5.txt`
- `models/cara/latency/deploy_{ttft,tpot,e2e}/*.xgb` — v2 1024-retrained XGBoost latency predictors (17 files;
  per (model,hw) tier: 3b_a30, 3b_p100, 7b_a30, 14b_v100, 72b_a100). Referenced by `predictor_xgboost_3model.json`.
- `models/cara/quality/` — deepeval KNN k=10 quality predictor (minmax norm) + MiniLM embedder.
- `models/cara/best_route_4way_qwen/` — BEST-Route DeBERTa checkpoint (deepeval, v1-kept, no retrain).
- `models/cara/avengers_pro_qwen_v2_lc{0.20,0.30,0.47,0.61,0.75,1,3,10,30}/` — v2 AvengersPro artifacts
  (lc = 1-pw; the sweep used pw{.25,.39,.53,.70,.80}).
- KEEP-DEPLOYED rule: these are the EXACT models that produced the v2 results. Do not rebuild — restore from here.

## 3. DATA — ON openclaw (`data/cara/`)
| file | md5 (see v2_deployed_md5.txt) | use |
|---|---|---|
| `data/cara/best-route-v3-test-3534-eval.jsonl` | — | the 3534-prompt bench (all sweeps) |
| `data/cara/scored/test_scored_filtered.jsonl` | 0318916… | per-(prompt,model) deepeval-LLAMA scores (quality lookup of record) |
| `data/cara/scored/gemma3_scored_full.jsonl` | cd0db7… | per-(prompt,model) deepeval-GEMMA3 scores (judge-alt) |
| `data/cara/budget_table_cal_{75,45,30}.jsonl` | — | budget mixes (§5.4); rebuild via `_recovered_budget/build_calibrated_budgets.py` |

## 4. COLLECTED RESULTS (ground truth to compare against) — ON openclaw
`experiment_output/v2_sweep_jun06/` (8.2G, **442 cells**):
- `cara/` (113: 8 simplex + 3 cost-extreme + 5 scalar-q × 7λ) · `baselines_maygrid/` (178) ·
  `batching_ablation{,_v2}/` (54) · `budget/` (18) · `multiseed/` (47) · `vllm_sr/` (9) · `probe_timeout/` (7).
- `_aggregate/AGGREGATE.{csv,jsonl}` (287 rows, FIXED aggregator).
Each cell = a `benchmark_serving` JSON with `response_details` (prompt, served model, output_len, e2el, error,
scheduling_overhead_breakdown). Quality is JOINED offline from the scored data (NOT in the cell).

## 5. CONFIGS (in repo, `block/config/cara/`)
- `scheduler_config_qn_minmax.json` — **v2 deployed scheduler config** (deepeval routing, KNN k10, minmax).
- `predictor_xgboost_3model.json` — latency XGB paths (per-tier). `model_deployment.json` — 13-node topology
  (Qwen pool, TP, ports). `host_configs.json` — host map.
- vLLM-SR: `cara_paper/.../v2_orchestration/vllm_sr/{config.yaml,envoy_qwen.yaml,vsr_relaunch.sh}`.

## 6. CLUSTER TOPOLOGY (CloudLab, ephemeral — re-acquire + setup.sh)
13-node: 2×A100(72B, d8545, TP=4, port7000) · 3×V100(14B, c4130) · 5×A30(7B) · 3×A30(3B, d7525).
Scheduler `cara_serve` on the A100 node, port 8200. vLLM backends port 8000 (host_configs) / 7000 (deploy).
For the judge-alt, free both A100 nodes (8 GPU) — gemma-3 fits 1 A100 bf16; 72B redeploy from model_deployment.json.

## 7. TWO REPRODUCTION LEVELS (see REPRODUCE_GUIDE.md)
- **L1 analysis smoke (NO cluster):** `python3 cara_paper/smoke_test_apr_13/scripts/consolidate_v2.py`
  re-derives EVERY §6 number from the collected cells + scored data. Verifies analysis + data integrity instantly.
- **L2 full cluster re-run:** redeploy stack (setup.sh + manifest models/configs) → run `v2_orchestration/*.sh`
  → `aggregate_v2_grid.py` + `consolidate_v2.py` → compare to the collected `experiment_output/` ground truth.
