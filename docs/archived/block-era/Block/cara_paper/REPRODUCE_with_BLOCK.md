# INSTRUCTION DOC 1 — Reproduce v2 via the Block repo
Uses `~/Code/llm/Block` (full serving + analysis stack) against this pack's `models/`, `data/`, `results/`.
`PK=/home/wd312/Code/llm/rb-exp/v2-exp`. Checkout: `cd ~/Code/llm/Block && git checkout 9ae1f20`.

## L1 — ANALYSIS SMOKE (openclaw, no cluster, ~2 min) — the fast reproducibility check
The analysis scripts read the LOCAL Block paths, which already point to the same files this pack bundles.
```bash
cd ~/Code/llm/Block
python3 cara_paper/smoke_test_apr_13/scripts/aggregate_v2_grid.py   # -> _aggregate/AGGREGATE.csv (287 rows; dispatcher_only=21)
python3 cara_paper/smoke_test_apr_13/scripts/consolidate_v2.py      # -> ALL §6 numbers
```
PASS = printed numbers match `V2_CHANGELOG_AND_SUMMARY.md §6` (cara 0.4190 #1; cost-extreme 1.68e-5;
br4 64204ms vs cara 2389ms @λ30; bootstrap [0.4089,0.4288]; overhead 134ms; judge-alt r=0.555).
To run against THIS pack instead of Block's copies: point `aggregate_v2_grid.py` BASE/SCORED at
`$PK/results` and `$PK/data/cara/scored/test_scored_filtered.jsonl` (or symlink them).

## L2 — FULL CLUSTER RE-RUN (fresh CloudLab window)
### 0. Bring up cluster (R2: setup.sh only) + restore from pack
```bash
# acquire the 13-node topology (configs/cara/model_deployment.json), run setup.sh per node.
rsync -a $PK/models/cara/ ~/Code/llm/Block/models/cara/         # deployed predictors+baselines
cp $PK/configs/cara/* ~/Code/llm/Block/block/config/cara/        # scheduler_config_qn_minmax + predictor + deployment + host
# re-pull Qwen pool from HF onto GPU nodes per model_deployment.json (Qwen/Qwen2.5-{3B,7B,14B,72B}).
# launch cara_serve port 8200:  python3 -m block.global_scheduler.cara.cara_serve --model_config_path block/config/cara/model_deployment.json --scheduler-config block/config/cara/scheduler_config_qn_minmax.json --predictor-config block/config/cara/predictor_xgboost_3model.json --host_config block/config/cara/host_configs.json --scheduling cara|pipeline
```
### Each artifact → script (in `code/orchestration/` = Block's `cara_paper/smoke_test_apr_13/scripts/v2_orchestration/`)
| artifact (§) | script | scheduling |
|---|---|---|
| fig:pareto/radar, tab:extremes (§5.2) | `v2_baseline_maygrid.sh` → `v2_phase2.sh` → `v2_cara_costextreme.sh` | pipeline (baselines) / cara (restart between) |
| fig:batching (§5.5) | `v2_phase3_batching.sh` (POST top-level batch_config, smoke /v1/batch_stats) | cara native |
| tab:budget (§5.4) | `v2_budget_cara_native.sh` (+ budget_table_cal_{75,45,30}.jsonl) | cara native (filter only here) |
| multi-seed | `v2_multiseed_cara.sh` (λ{12,18,24} × seeds 2,3,4) | both |
| vLLM-SR | `vllm_sr/vsr_relaunch.sh` (docker vllm-sr:latest + Envoy; ext_proc message_timeout 300s) → `v2_vllm_sr_grid.sh` (--backend vllm_sr, Envoy:8801, all 7λ) | Envoy |
| judge-alt | free 2 A100 → `uv pip install "vllm==0.10.2" --torch-backend=cu128` + `transformers>=4.56,<5`; serve `unsloth/gemma-3-12b-it` bf16 (4 GPUs); shard; `python3 -m block.predictor.cara.offline_training.score_with_deepeval --model unsloth/gemma-3-12b-it --base-url localhost:800x/v1 --judge-key deepeval-gemma3-12b-it_reference` | A100 |
### After each track
```bash
python3 cara_paper/smoke_test_apr_13/scripts/aggregate_v2_grid.py
python3 cara_paper/smoke_test_apr_13/scripts/consolidate_v2.py
# COMPARE new output vs $PK/results ground truth: quality (deterministic per prompt,model) should match;
# latency varies run-to-run — verify the SHAPE (br4 collapse / cara bounded / vLLM-SR saturation).
```
### Sanity gates (catch v1-class bugs)
- `dispatcher_only == 21` (random dispatcher in); cara-peak λ12 ≈ 0.419 NOT ~0.52 (the v1 zero-score inflation).
- Verify routing by DISTRIBUTION, not e2e; restart cara_serve if AvengersPro floods all-72B (stale shim).
- No `a or b` on deepeval scores (0.0 is valid); `compute_ci_and_tail.py` is May-path only (quarantined).

## CRITICAL: native-mode (cara) serving requirements (discovered 2026-06-10; undocumented before)
Native (`--scheduling cara`) latency results are only valid when ALL of the following hold — without them the
serve SILENTLY degrades (no error) to ~15-18 req/s scheduler throughput and collapses at λ≥18:
0. **`CUDA_VISIBLE_DEVICES=""` in the serve environment (CRITICAL — discovered 2026-06-11).** The scheduler is
   CPU-only, but it runs on a GPU node co-located with vLLM. XGBoost >=2.x probes the visible CUDA device on every
   `inplace_predict`, making the predictor ~600x slower (0.17ms -> ~930ms/call) and collapsing native E2E at
   load (uniform lambda12 2.5s -> 26s). Hiding the GPU restores grid speed. This was the root cause of the
   2026-06-10 "native cannot reproduce" scare; the grid ran with CUDA hidden. (Also pinned in code:
   xgboost_predictor.py set_param device=cpu,nthread=1.)
1. `CARA_INPROC_PREDICTOR=1` in the serve environment (cara_serve.py gates the in-process batched TPOT XGBoost
   on it; without it every request makes a ~200ms HTTP /predict_latency call to the local sidecar).
2. One predictor sidecar per backend node:
   `cd ~/Block && PYTHONPATH=$PWD python3 -u -m block.predictor.cara.cara_predictor_api_server --host 0.0.0.0
   --port 8300 --backend-port 8000 --hostname $(hostname -f) --config-path block/config/cara/predictor_xgboost_3model.json
   --instance-type <tier, e.g. qwen2.5-3b_a30>`  — gate on `curl :8300/health` per node (a dead sidecar distorts
   ROUTING: that node's tier gets starved). Sidecars die when their backend restarts — re-gate after any restart.
3. Backends = the fork vLLM launched as `python3 -u -m vllm.entrypoints.openai.api_server ... --trust-remote-code`
   FROM A CWD WITHOUT the vllm source checkout (running from $HOME picks up ~/vllm via sys.path → ImportError).
   A100 72B adds `--tensor-parallel-size 4 --enforce-eager --gpu-memory-utilization 0.95`.
4. Verify before any long run: serve log prints `[INPROC-XGB] tpot model loaded`; a smoke cell's
   `scheduling_overhead_breakdown` shows `xgb_batched_tpot_ms` > 0 and `sidecar_ms` < 1ms.
NOTE (2026-06-10): with 1-3 satisfied a residual fault still prevented grid-level native λ24 latency on the
rebuilt cluster (VAL3: completions ~0.45/s); root-causing that is first item for the next slot. Pipeline-mode
(baseline) cells are unaffected by all of the above and reproduce exactly (avgpro 0.37598 verified same-day).

## COMPLETE ARTIFACT → REPRODUCE MAP (every paper number/table/figure)
Fig 1 (architecture, `RouteBalance.png`) is a manual diagram with no data producer. `consolidate_v2.py` and `aggregate_v2_grid.py` hardcode Block paths (`experiment_output/v2_sweep_jun06`, `data/cara/...`); repoint BASE/SCORED to run against the pack. All other numbers trace to `results/_aggregate/AGGREGATE.csv` (287 cells) + the named analysis JSONs, via the
producer scripts in `cara_paper/smoke_test_apr_13/scripts/`. To reproduce a value: run the producer, then
compare to the committed JSON/CSV of record. `qcompute.py <cell.json> <scored.jsonl> <gt> <label>` recomputes
any cell's zero-safe DeepEval quality + routing distribution.

### Main (§5)
| paper artifact | producer | data (pack `results/`) | compare-to |
|---|---|---|---|
| Fig 2 pareto, Fig 3 radar, Fig planes | `make_fig2_fig3_final.py` | `cara/`, `baselines_maygrid/` via AGGREGATE.csv | figures/*.pdf byte-regen |
| Table: heterogeneous cluster | static (model_deployment.json + PRICING_PER_1M in aggregate_v2_grid.py) | configs/ | — |
| tab_extremes (peak/cost/tput) | `aggregate_v2_grid.py` → AGGREGATE.csv | `cara/`,`baselines_maygrid/` | AGGREGATE.csv columns |
| headline 0.419 / +0.013 CI / +0.043 | `consolidate_v2.py` (§6.5 10k paired bootstrap) | `cara/cara_wq0.8*l12`, `baselines_maygrid/br4_t0.5_sh_l12` | App A numbers |
| 23× collapse (serial) / 4× (enhanced) | AGGREGATE + `fixed_baselines/` | `baselines_maygrid/*_l30`, `fixed_baselines/br4fix_*_l30` | tab_residence_vs_e2e |

### Baselines + scorecard
| tab_dominance (49-check scorecard) | `compute_dominance_counts.py` | AGGREGATE.csv | tab_dominance.tex |
| tab_sched_combined / tab_residence_vs_e2e / tab_sched_breakdown | from cell `response_details` scheduling_overhead_breakdown | `cara/`,`baselines_maygrid/`,`fixed_baselines/` | the .tex values |

### Appendices
| App A bootstrap CI + multiseed | `consolidate_v2.py` (§6.4 multiseed, §6.5 bootstrap) | `multiseed_v2/` | tab_multiseed |
| App A OOD (per-domain ID/OOD, BestAcc 0.348) | `appA_canonical.py` (deployed predictor, deepeval GT) | `data/cara/train_scored_filtered.jsonl` (non-flat, in pack) + `data/cara/scored/test_scored_filtered.jsonl` | `_aggregate/appA_canonical_bestacc.txt` + LAB_NOTE |
| App B overhead breakdown | per-cell `scheduling_overhead_breakdown` | `cara/` | App B prose |
| App B deployment ladder (serial/microbatch/enhanced) | `fixed_baselines/` + `batched_br/` ; per-cell residual | `fixed_baselines/{br4fix,avgfix}_*`, `batched_br/` | App B / tab_residence_vs_e2e enhanced rows |
| App B Task A latency-term ablation (tab_latablation) | `review_extras/taskA2_*` ; `taskA_latency_ablation.json` | `review_extras/taskA2_{uniform,wlat0}_l{12,24,30}` | `_aggregate/taskA_latency_ablation.json` |
| App B tail (tab_tail) | p95/p99 from `response_details` | `cara/`,`baselines_maygrid/` | `_aggregate/tail_metrics_e4.json` |
| App C vLLM-SR (tab_vllm_sr) | `vllm_sr/` cells | `vllm_sr/` | tab_vllm_sr.tex |
| App D judge-alt (tab_judge_alt) | `score_with_deepeval --judge-key deepeval-gemma3-12b-it_reference` ; gemma bootstrap | `judge_t05_out/` | `_aggregate/gemma_rejudge_bootstrap.json` |
| App E predictor table (tab_predictor) | deployed `training_metrics.json` per .xgb | `models/cara/latency/deploy_{tpot,ttft,e2e}/training_metrics.json` | tab_predictor.tex |
| App E oracle (0.582/0.376/0.428) | `oracle_headroom.py` (verified 2026-06-11) | `data/cara/scored/test_scored_filtered.jsonl` | App E prose / LAB_NOTE |
| App E/G cost-sensitivity (5 price vectors) | recompute per vector | `_aggregate/cost_sensitivity_e8.json` | the JSON |
| App E tier-loss | `tier_removal/` clean cells | `tier_removal/clean_no72b_*` | App tier-loss prose |

### Ablation + extra
| batching ablation (fig_batching_ablation) | `make_v2_paper_figs.py` (fig) + `batching_ablation_v2/` (data) | `batching_ablation_v2/` | the figure |
| budget §5.4 (tab) | `budget/` + `budget_rescore/` (DeepEval re-score) | `budget/`,`budget_rescore/` | §5.4 numbers |
| safety (E11, is_harmful routing) | `safety_e11.json` | `_aggregate/safety_e11.json` | App predictor safety ¶ |
| scaling microbench (E10) | `scripts/v2_orchestration/node_e10_microbench.py` | `_aggregate/e10_microbench_result.json` | App B scaling sentence |

### Env-gated extra experiments (reviewer-driven, 2026-06-11; all gates DEFAULT-OFF)
Every extra cell below is produced by an **env/flag toggle that is off by default**, so the main grid and
all deployed cells are byte-unaffected (gates: `CARA_SQUARE`→unset, `CARA_TIEBREAK`→"queue",
`CARA_STATIC_LAT`→unset). Trigger scripts live in `Block/cara_paper/smoke_test_apr_13/scripts/v2_orchestration/`
and are mirrored in the pack under `scripts/v2_orchestration/`. Each is N=3534 on `data/cara/best-route-v3-test-3534-eval.jsonl`, seed 5.

| experiment | gate / flag (code site) | trigger script | cells (`review_extras/`) | maps to | expected |
|---|---|---|---|---|---|
| P1 fusion-isolation **arm 3** (decoupled-predictive) | `CARA_TIEBREAK=that` (cara_serve.py ~L1628; default "queue") | `p1_arm3.sh` | `p1_arm3_that_l{12,24,30}` | Table 10 arm 3 row | 3.42/3.50/3.75 s, 72B 14%, q0.385 |
| **Arm 4** static-prior latency (reviewer-W) | `CARA_STATIC_LAT="3b=12.3,7b=20.6,14b=18.4,72b=47.4"` ms/tok (cara_serve.py ~L1483; default unset → learned T̂). Constants = low-load (λ6) per-tier TPOT medians | `arm4_static_lat.sh` (smoke→stat→overload) | `arm4_stat_l18`, `arm4_sq40` | App G arm-4 (static prior vs learned) | vs arm 1: stat l18 2615 / 40-6 overload 3788 |
| Non-stationary **square wave** | `CARA_SQUARE="lo,hi"` on BOTH serve+bench (benchmark_serving.py ~L153; ~30s phases; default unset → Poisson) | `p3sq.sh` (uniform+br4enh) + `p3sq_avg.sh` (avgpro) | `p3sq_{uniform,br4enh,avgenh}_sq` (30/6) | Table 15 square col | 2577 / 7107 / 2733 |
| Non-stationary **Gamma burst** | `--burstiness 0.3` (existing bench flag; no code) | `p3_bursty.sh` | `p3_{uniform,br4enh,avgenh}_l18_b0.3` | Table 15 gamma col | 2524 / 4392 / 2473 |
| **Overload probe** (40/6, peak>27.6 sustained max) | `CARA_SQUARE=6,40` | `ov_arm1_sq40.sh` | `ov_arm1_sq40` | App G Q3(a) per-phase mix | e2e 3788; mix flat across phases |
| **n=3 latency reseed** (uniform) | seed variation only (no gate) | `uni_reseed.sh` | `p2_uniform_l{12,24,30}_s{6,7}` | App A multiseed latency | l12/24/30 ±1.4/2.7/2.0% |

**Stationary references** for Table 15 (b=1, λ18): RB-uniform 2615 (grid l18), enh-AvgPro 2400 (`fixed_baselines/avgfix_pw0.8_sh_l18_s1`, same lc0.20 artifact), enh-BR4 4079 (enhanced grid l18).

**Log-only analyses** (no serving; recompute from `review_extras/` + `cara/` `response_details`):
`_aggregate/uniform_mix_vs_lambda.json` (per-λ tier mix, Table mixvslambda), `_aggregate/Q3_static_vs_live_probe.json`
(per-phase mix under 30/6 and 40/6; output-length-quartile mix), `_aggregate/P1_fusion_triangle.json`,
`_aggregate/P3_nonstationary_3x3.json`, `_aggregate/uniform_latency_multiseed.json`. Tier from `model` field; square-wave
phase from `request_id` index (`idx % ((lo+hi)*30)` < `hi*30` ⇒ high phase).

### Native-mode env contract (REQUIRED for any native/cara cell — see §native-mode above)
`CUDA_VISIBLE_DEVICES=""` (XGBoost else probes the co-located GPU, 600× predict slowdown) +
`CARA_INPROC_PREDICTOR=1` + per-node :8300 sidecars + fork-vLLM api_server entrypoint from a non-source CWD;
do NOT set `OMP_NUM_THREADS=1` (throttles MiniLM). Verified: native cells reproduce within 3% (quality exact:
wq0.8 0.4184 vs grid 0.4187; uniform l12/24/30 2373/2597/2780 vs 2311/2684/2783).

## PRE-BUILD QUALITY-CONSISTENCY GATE (prevents stale-quality divergence)
Before any pdflatex build, run `python3 cara_paper/smoke_test_apr_13/scripts/check_quality_consistency.py` —
it recomputes every DeepEval-quality number quoted in the manuscript from the released `response_details`
via the canonical zero-safe chat-unwrap path and diffs vs the manuscript (exit nonzero on |Δ|>0.01).
Catches the recurring inflated/stale-quality class (v1 Fig2, v4 OOD acc, v6 Table 10). Add new quoted
quality numbers to its CLAIMS list.
