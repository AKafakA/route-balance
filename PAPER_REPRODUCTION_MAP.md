# Paper-element → reproduction MASTER MAP (v2, 2026-06-11)

Every figure / table / headline number in `paper/neurips_2026_v2.tex` → the experiment that produces it,
its trigger script, data/cells, and where it is (pack `results/` + repo). **Audit 2026-06-11: 0 missing —
all 7 inline tables + 10 `\input` tables + 5 figures have their data source in this pack.**

- **Repos**: internal `Block` @ `<BLOCK_COMMIT>` (full serving+analysis stack); public `route-balance` @ `<RB_COMMIT>`
  (mirror, 0 cara). Paper repo `route-balance-paper` @ `<PAPER_COMMIT>`.
- **Pack** `PK=~/Code/llm/v2-exp`: `results/` (cells+`_aggregate/` JSONs), `models/`, `data/`, `configs/`,
  `paper/` (tex+tables_v2+figures+pdf), `scripts/v2_orchestration/` (trigger scripts).
- **Two reproduction levels** (see `REPRODUCE_with_BLOCK.md`): **L1** analysis-only (openclaw, ~2 min, recompute every
  number from committed `results/`); **L2** full cluster re-run (fresh CloudLab 13-node slot).
- **Fast check**: `aggregate_v2_grid.py` + `consolidate_v2.py` reproduce all §6 numbers from `results/`.
- **Gate before any paper build**: `check_quality_consistency.py` (recomputes 7 quality numbers, fails on |Δ|>0.01).

## Body

| Paper element | Claim / conclusion | Trigger script (L2) | Data / cells (pack) | Analysis |
|---|---|---|---|---|
| **Fig 1** `fig:route_balance` (RouteBalance.png) | architecture (static, author-fixed) | — | — | — |
| **Tab 1** `tab:model_notation` (§3) | notation (static) | — | — | — |
| **Tab 2** `tab:heterogeneous_cluster` (§5.1) | 13-instance/28-GPU topology | cluster bring-up (setup.sh) | `configs/cara/model_deployment.json` + `host_configs.json` | static |
| **Fig 2** `fig:pareto` (pareto_multilambda.pdf) (§5.2) | quality-knob frontier; BR 23× collapse; RB dominates q-vs-tput | `v2_baseline_maygrid.sh`→`v2_phase2.sh`→`v2_cara_costextreme.sh` | `results/cara/`, `cara_edge/`, `baselines_maygrid/`, `fixed_baselines/` → `_aggregate/AGGREGATE.csv` | `make_fig2_fig3_final.py` |
| **Fig 3** `fig:radar` (cap_radar_multilambda.pdf) (§5.2) | 3-axis frontier; one stack holds all rims | same as Fig 2 | same + `tables_v2/tab_radar_picks` | `make_v2_paper_figs.py` |
| **Tab** `tab_extremes` (§5.2) | peak quality 0.4190 / cost-extreme 1.68e-5 | `v2_cara_tuples.sh`,`v2_cara_costextreme.sh` | `cara/`, `cara_edge/` | `consolidate_v2.py` |
| **Tab** `tab_sched_breakdown`/`tab_sched_combined` (§5.3,App B) | per-request residual sub-linear; compute ≈33ms | (from main grid) | `cara/` `response_details.scheduling_overhead_breakdown` | `consolidate_v2.py` |
| **Tab** `tab_residence_vs_e2e` (§5.3) | BR serial collapse; enhanced still trails 2.6–4.5× | `v2_bestroute.sh` (serial) + `fixed_baselines/`,`batched_br/` (enhanced) | `baselines_maygrid/`, `fixed_baselines/`, `batched_br/` | `consolidate_v2.py` |
| **Tab** `tab:budget_violations` (§5.4) | predictive filter cuts exhaustion | `v2_budget_cara_native.sh` (+budget cal jsonl) | `budget/`, `budget_rescore/` | `consolidate_v2.py` |
| **Fig** `fig:batching` (fig_batching_ablation.pdf) (§5.5) | bs=1 doesn't collapse; LPT within ±2.3% | `v2_phase3_batching.sh` | `batching_ablation_v2/` | `make_v2_paper_figs.py` |

## Appendices

| Paper element | Claim | Trigger script | Data / cells | Analysis |
|---|---|---|---|---|
| **Tab** `tab:bootstrap`+`tab_multiseed` (App A) | quality CI excludes 0; seed-stable | `v2_multiseed_cara.sh` | `multiseed_v2/` | `consolidate_v2.py` (bootstrap+multiseed) |
| App A latency seed-stability (n=3) | uniform ±1.4/2.7/2.0% @λ12/24/30 | `uni_reseed.sh` | `review_extras/p2_uniform_l{12,24,30}_s{6,7}` | `_aggregate/uniform_latency_multiseed.json` |
| App A OOD (BestAcc 0.348) | per-domain ID/OOD, no systematic collapse | `appA_canonical.py` | `data/cara/train_scored_filtered.jsonl`+`scored/test_scored_filtered.jsonl` | `_aggregate/appA_canonical_bestacc.txt` + `LAB_NOTE` |
| **Tab** `tab_tail` (App B) | p95/p99 tails | (main grid) | `cara/`,`baselines_maygrid/` | `_aggregate/tail_metrics_e4.json` |
| App B scaling microbench (E10) | scheduler compute scaling | `node_e10_microbench.py` | — | `_aggregate/e10_microbench_result.json` |
| **Tab** `tab_vllm_sr` (App C) | vLLM-SR collapses from λ18 | `vllm_sr/vsr_relaunch.sh`→`v2_vllm_sr_grid.sh` | `vllm_sr/` | `tab_vllm_sr.tex` |
| **Tab** `tab_judge_alt` (App D) | system order stable under gemma judge (r=0.555) | `node_g12_serve_judge.sh` + `score_with_deepeval --judge-key ...gemma3-12b...` | `judge_t05_out/` | `_aggregate/gemma_rejudge_bootstrap.json` |
| **Tab** `tab_predictor` (App E) | deployed predictor MAE/MAPE; KNN 0.348 | (deployed artifacts) | `models/cara/latency/deploy_{tpot,ttft,e2e}/training_metrics.json` | `tab_predictor.tex` |
| App E oracle (0.582/0.376/0.428) | headroom; served 0.419 closes ~1/5 gap | `oracle_headroom.py` | `data/cara/scored/test_scored_filtered.jsonl` | `LAB_NOTE` |
| App E/G cost-sensitivity (5 vectors) | orderings invariant | (recompute) | `_aggregate/cost_sensitivity_e8.json` | the JSON |
| App E tier-loss | graceful degrade; uniform q unchanged | `node_e7_tier_removal.sh` | `tier_removal/clean_no72b_*` | prose |
| App E safety (E11) | no preset dumps harmful prompts | (from grid) | `_aggregate/safety_e11.json` | the JSON |
| **Tab** `tab_dominance` (App F) | per-baseline scorecard (21/21, 46/49, 12/49) | (main grid) | `_aggregate/AGGREGATE.csv` | dominance in `consolidate_v2.py` |
| **Fig** `fig:planes` (pairwise_planes_l12.pdf) (App F) | 3 pairwise trade-off planes | (main grid) | `_aggregate/AGGREGATE.csv` | `make_v2_paper_figs.py` |
| **Tab** `tab:latablation` 3-arm (App G) | within-tier T̂≈queue; fusion=cross-tier 26–31% | `p1_arm3.sh` (`CARA_TIEBREAK=that`) + `taskA*` (arm1/2) | `review_extras/p1_arm3_that_l{12,24,30}`, `taskA2_*` | `_aggregate/P1_fusion_triangle.json`,`taskA_latency_ablation.json` |
| **Tab** `tab:mixvslambda` (App G) | uniform mix rate-independent (72B pinned 1%) | (from main grid `cara/`) | `cara/cara_wq0.33_wl0.33_wc0.33_l{6,12,18,24,30}` | `_aggregate/uniform_mix_vs_lambda.json` |
| **Tab** `tab:nonstationary` 3×3 (App G) | amortized-scoring (RB+enh-AvgPro) bounded; serial BR degrades +74% | `p3_bursty.sh`(`--burstiness 0.3`) + `p3sq.sh`+`p3sq_avg.sh`(`CARA_SQUARE`) | `review_extras/p3_*`, `p3sq_*`; stat = `fixed_baselines/avgfix_pw0.8_sh_l18_s1` | `_aggregate/P3_nonstationary_3x3.json` |
| **App G arm-4** static-prior latency | learned latency NOT load-bearing — static prior matches everywhere | `arm4_static_lat.sh` (`CARA_STATIC_LAT`) + `ov_arm1_sq40.sh` (overload `CARA_SQUARE=6,40`) | `review_extras/arm4_{stat_l18,sq40,smoke}`, `ov_arm1_sq40` | `_aggregate/arm4_static_prior_result.json`,`Q3_static_vs_live_probe.json` |

## Reproduction guarantees / notes for next slot
1. **All env-gated extras are DEFAULT-OFF** (`CARA_SQUARE`/`CARA_TIEBREAK`/`CARA_STATIC_LAT`) → main grid byte-unaffected.
   Gating + per-experiment expected numbers: `REPRODUCE_with_BLOCK.md` §"Env-gated extra experiments".
2. **Native-mode contract is MANDATORY** for any `--scheduling cara` cell (`CUDA_VISIBLE_DEVICES=""`,
   `CARA_INPROC_PREDICTOR=1`, no `OMP_NUM_THREADS=1`) — else silent 600× predictor slowdown. See doc §native-mode.
3. **Quality is deterministic** per (prompt,model) via DeepEval lookup → must match exactly on re-run; **latency varies
   run-to-run** → verify the SHAPE (BR collapse / RB+AvgPro bounded / vLLM-SR saturation), not absolute ms.
4. **Figures**: `make_fig2_fig3_final.py` (Fig 2/3) + `make_v2_paper_figs.py` (planes, batching, radar) read `_aggregate/AGGREGATE.csv`.
5. **OPEN next-slot experiment** (App G arm-4 follow-through): re-scope decision pending — if the paper keeps the
   "live/learned latency" framing, the discriminating test is arm-4 vs arm-1 at a phase that *exceeds* sustained max
   under a longer horizon + multi-seed (already shows static matches; confirm at scale). Else re-scope per arm-4 result.
