# v2 Reproduction & Evaluation Guide (2026-06-09)
How to reproduce EVERY v2 result (figures, tables, key numbers). Read `REPRODUCIBILITY_MANIFEST.md` first.

There are **two levels**. Do **L1 first** (instant, no cluster) — it proves the analysis + collected data
reproduce the paper numbers. Do **L2** only when a fresh CloudLab window is up — it re-runs the serving
experiments and compares against the L1 ground truth.

All commands run from the repo root `~/Code/llm/Block` on **openclaw** (L1) or the **scheduler node** (L2),
after `git checkout 9ae1f20`.

---
# L1 — ANALYSIS SMOKE (no cluster; ~2 min). The fast reproducibility check.
This re-derives all §6 numbers from the COLLECTED cells (`experiment_output/v2_sweep_jun06/`) + scored data.

```bash
# 1. (Re)build the main aggregate from the 442 collected cells with the FIXED aggregator:
python3 cara_paper/smoke_test_apr_13/scripts/aggregate_v2_grid.py
#    -> experiment_output/v2_sweep_jun06/_aggregate/AGGREGATE.{csv,jsonl}  (expect 287 rows;
#       dispatcher_only MUST be 21 = rr+sq+random; cara 112, avgpro 70, br4 84)

# 2. Producer-of-record for EVERY §6 number (peak-q, frontier, load-robustness, multi-seed CI,
#    bootstrap CI, overhead, vLLM-SR head-to-head, judge-alt consistency):
python3 cara_paper/smoke_test_apr_13/scripts/consolidate_v2.py
```
**PASS criteria** — the printed numbers must match `V2_CHANGELOG_AND_SUMMARY.md §6`, notably:
- §6.1 peak deepeval: cara **0.4190** > br4 0.4056 > avgpro 0.3760 > dispatcher 0.3709.
- §6.2 cara cost-extreme cheapest **1.68e-5/req**; cara peak cost 6.9e-5; br4 0.4056, avgpro 0.3760.
- §6.3 λ30 e2e (peak-q): cara 11835ms, **br4 64204ms (collapse 21×)**, avgpro 7205ms; cara cheap cell 2389ms.
- §6.4 cara wq0.8 multi-seed **0.4184 [0.4182,0.4187]** (n=4).
- §6.5 bootstrap cara wq0.8 λ12 **0.4187 [0.4089,0.4288]**.
- §6.6 overhead total **134ms** (batch_wait 48.6 + KNN 28.2 + stats 14.8 + …).
- §6.7 vLLM-SR: λ≤10 0 fails; λ18 3161 fail / 236s; λ30 3411 fail / 344s.
- §6.8 judge-alt: per-model gemma 3B .585/7B .626/14B .677/72B .706; **Pearson r=0.555** (n=14,427).
If any number drifts, the analysis or the collected data changed — STOP and diff.

---
# L2 — FULL CLUSTER RE-RUN (fresh CloudLab window). Compares against L1 ground truth.

### Step 0 — bring up the cluster (NEVER manual; use setup.sh — R2)
Acquire the 13-node topology (Manifest §6), run `setup.sh` on each, restore the deployed models from
openclaw (`rsync openclaw:~/Code/llm/Block/models/cara/ ./models/cara/`), and the configs
(`scheduler_config_qn_minmax.json`, `predictor_xgboost_3model.json`, `model_deployment.json`, `host_configs.json`).
Launch `cara_serve` (port 8200): native mode for cara, pipeline mode for baselines (see scripts). Re-pull the
Qwen pool from HF onto the GPU nodes per `model_deployment.json`.

### Each paper artifact → script → aggregate → compare
| Paper artifact (§) | L2 run script (`cara_paper/smoke_test_apr_13/scripts/v2_orchestration/`) | mode |
|---|---|---|
| **fig:pareto, fig:radar, tab:extremes, tab:radar_picks (§5.2)** | `v2_baseline_maygrid.sh` (avgpro pw×2disp + br4 t×2disp + passthru×3disp × 7λ) **then** `v2_phase2.sh` (cara 8 simplex tuples) **then** `v2_cara_costextreme.sh` (3 cost-extreme tuples) | baselines=`--scheduling pipeline`; cara=`--scheduling cara` (restart between) |
| **fig:batching (§5.5)** | `v2_phase3_batching.sh` / `v2_batching_redo.sh` (default/LPT-off/adaptive-off + bs sweep; POST **top-level** batch_config, smoke-gate `/v1/batch_stats`) | native |
| **tab:budget_violations (§5.4)** | `v2_budget_cara_native.sh` (cara_filter/nofilter × 3 mixes @λ16) + br4 pipeline; needs `budget_table_cal_{75,45,30}.jsonl` | cara native (filter only engages here) |
| **multi-seed (App)** | `v2_multiseed_cara.sh` (cara wq0.8+uniform & baselines × λ{12,18,24} × seeds 2,3,4) | both |
| **vLLM-SR e2e (head-to-head)** | `v2_orchestration/vllm_sr/vsr_relaunch.sh` (docker vllm-sr:latest + Envoy, configs in same dir; ext_proc `message_timeout: 300s`) then `v2_vllm_sr_grid.sh` (`--backend vllm_sr` Envoy:8801, all 7λ, option A) | Envoy |
| **judge-alt (App)** | free 2 A100 nodes → new venv `uv pip install "vllm==0.10.2" --torch-backend=cu128` + `transformers>=4.56,<5`; serve `unsloth/gemma-3-12b-it` bf16 (4 instances/GPUs); shard dataset; `block.predictor.cara.offline_training.score_with_deepeval --model unsloth/gemma-3-12b-it --base-url localhost:800x/v1 --judge-key deepeval-gemma3-12b-it_reference` | A100 |

### After each track:
```bash
# main + cost-extreme:
python3 cara_paper/smoke_test_apr_13/scripts/aggregate_v2_grid.py     # -> fresh AGGREGATE.csv
python3 cara_paper/smoke_test_apr_13/scripts/consolidate_v2.py        # -> all §6 numbers
```
**Compare vs ground truth:** diff the new `AGGREGATE.csv` / consolidate output against the COLLECTED
`experiment_output/v2_sweep_jun06/` (synced on openclaw). Quality is deterministic per (prompt,model) via
the deepeval lookup, so quality numbers should match closely; **latency (e2e) varies run-to-run** with load —
expect the SHAPE (br4 collapse, cara bounded, vLLM-SR saturation) to reproduce, not the exact ms.

### Sanity gates (catch the v1-class bugs)
- Aggregator: `dispatcher_only == 21` (random dispatcher included), no `a or b` zero-score drop
  (cara-peak λ12 quality ≈ 0.419, NOT ~0.52 — that inflation = the v1 bug resurfacing).
- Routing verified by DISTRIBUTION not e2e (AvengersPro must NOT flood all-72B; restart cara_serve if shim cached).
- vLLM-SR served-model from response `model` field; judge-alt index-map is chunk-local (do not "fix").

---
# WHAT EACH KEY NUMBER MEANS (for the figures/tables)
- **fig:pareto / tab:extremes** ← §6.1+6.2 (quality vs cost vs e2e per cell at λ12; cara spans cheapest→highest-q).
- **fig:radar** ← per-cell {quality, cost, latency, balance} (radar via MoE-CAP helpers; cells in AGGREGATE).
- **load-robustness fig/table** ← §6.3 (peak-q e2e vs λ; br4 collapse 21×).
- **tab:budget_violations** ← budget cells (cara+filter 32.8/19.2/12.6 vs nofilter/br4).
- **fig:batching** ← batching cells (LPT-off≈default; adaptive-off +3.5%; bs=1 collapses at λ≥16).
- **App.A tab:bootstrap** ← §6.5. **App.B tab:sched_combined** ← §6.6. **multi-seed App** ← §6.4.
- **vLLM-SR head-to-head** ← §6.7. **judge-alt robustness App** ← §6.8 (cara #1 under both judges).
