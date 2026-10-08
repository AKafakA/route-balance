# RouteBalance/Cara — v2 Comprehensive Rerun Plan (remade 2026-06-07)

Supersedes the Jun-6 5-day "Phase-0-then-v2" plan. Authoritative operational tracker = the
session task list (#9, #11–#15). This doc is the narrative source of truth: scope, findings,
decisions, execution order. Cluster: `asdwb-307592` (13-node), scheduler `asdwb@d8545-10s10301`
(node4), port 8200.

## 0. Scope decision (pivot)
Phase-0 rebuttal-on-v1-stack is **dropped** (user: "rebuttal makes no sense, it builds on buggy
code — start v2 now"). We run **one comprehensive v2 campaign**: rerun *every experiment that
appears in the paper* on the fixed v2 stack, plus the new experiments below. v1/v2 never mixed.

## 1. The v2 stack = submitted stack + intended fixes ONLY
Reference for "submitted": `route-balance/reproduce/run_main_table.sh` + submitted cells
`cara_paper/.../main_table_full_3534/cara_final_e2e/` + deployed `_cluster_archive_20260506/scheduler_config_carafast.json`.

Intended fixes (keep): output cap **256→1024**; routing `score_type` **judge→deepeval** (submitted
routed on judge, reported deepeval); KNN **k=50→k=10**; `quality_norm` raw→**minmax**; a30 latency
**hardware-aware tier** (first-match p100 bug); **1024 latency-XGB retrain**; batched-FAISS KNN.
Config diff otherwise identical (slo_defaults: lpt_sort_key=max, batch_config, instance_capacity,
kv_config, relax_order all match).

## 2. Findings this session (all evidence-cited, in memory/)
1. **Param grids silently didn't match May** → fixed. cara = 8 simplex tuples (not scalar-q);
   AvengersPro = pw{0.25,0.39,0.53,0.70,0.80} (not lambda_cost{1,3,10,30}); BEST-Route = argmax +
   t{0.3,0.5,0.7} (v2 supersets to {0,0.3,0.5,0.6,0.7,0.8}); dispatcher-only adds **random**.
   Paper §5.2 names exactly AvengersPro pw{.25,.39,.53,.8}, BR t{.3,.5,.7}, passthru{rr,sq,ra}.
2. **AvengersPro all-72B flood** = upstream `SimpleClusterRouter` interface drift (`embedder.get`
   + `tokenizer`); shim fixed. **Then** the live flood persisted because the long-running
   `cara_serve` had the STALE module cached → required restart. Verified by distribution after fix.
3. **Aggregator zero-score bug** (`a or b` drops deepeval==0.0) inflated quality by **+0.116**
   (submitted May cell 0.512→0.397, coverage 0.774→1.0). Fixed in `aggregate_v2_grid.py`; the May
   `aggregate_overnight_grid.py` still has it. System-dependent (cheap-routing inflated more →
   correcting it *widens* cara's lead over AvengersPro). [[project_aggregator_zero_score_bug]]
4. **Scheduling-mode phase-split** (gotcha): cara native = `--scheduling cara`; baselines =
   `--scheduling pipeline`. Bare `scoring_weights` POST in pipeline mode RAISES (cara_serve.py:474).
   Rerun must restart between phases. [[project_v2_rerun_match_may]]
5. λ=12 is NOT a clean cara win on its own (shared frontier; cara-exclusive only at q>0.41). The
   real claim is **load robustness at high λ** (baselines collapse) — pending λ=18/24/30 cells.

## 3. Locked decisions (2026-06-07)
- Route + report on **deepeval** (all three routers: cara KNN k=10 / AvengersPro deepeval-flat /
  br4 deepeval checkpoint — no br4 retrain needed, already deepeval).
- Output cap **1024**, quality_norm **minmax**.
- Alt-judge = **google/gemma-3-12b-it** (offline, dev-gpu, `--judge-key deepeval-gemma3-12b-it_reference`).
- Multi-seed "best cells" = **peak-quality + uniform per system** (cara wq=0.8 & uniform(⅓);
  AvengersPro pw=0.8; BEST-Route t=0.3; + best dispatcher per baseline) × λ{12,18,24} × **3 more seeds**.
- vLLM-SR = **after main + ablations**.
- **Ignore** (not in paper): SLO-filter ablation, KNN-vs-RoBERTa predictor comparison.
- Quality framing stays confident (Rule 1); all scoring via the **fixed** aggregator.

## 4. Experiment matrix (paper artifact → track → status)
| Paper artifact | Experiment | Track | Status |
|---|---|---|---|
| fig:pareto, fig:radar, tab:extremes, tab:radar_picks (§5.2) | cara simplex + AvengersPro pw + BR t + passthru{rr,sq,ra} × λ{6..30}, seed1 | #9 | **running** |
| tab:sched_combined, App.B (§5.3) | off-instance residual + compute breakdown | #15 | offline (from cells) |
| tab:budget_violations (§5.4) | 3 configs × 3 mixes(75/45/30%) @ λ16 + rescore | #12 | **blocked: budget table** |
| fig:batching (§5.5) | default/LPT-off/adaptive-off ×λ; bs{1,16,32}×λ{8,16,24} | #11 | **chained (phase3)** |
| tab:bootstrap (App.A) | 95% CI on DeepEval mean @ λ12, wq=0.8 | #15 | offline |
| — (NEW) | multi-seed best cells × λ{12,18,24} × 3 seeds | #13 | after main agg |
| — (NEW) | vLLM-SR e2e | #14 | after main+ablations |
| — (NEW) | judge-alt (gemma) on multi-seed cells | #15 | offline, after cells |
| — (NEW, 2026-06-08) | **cost-extreme cara tuples (0,0,1)/(.05,.05,.9)/(.1,.05,.85) ×7λ** | #18 | **chained after batching redo** |

**FRAMING DECISION (added 2026-06-08):** v2 shows global Pareto-dominance does NOT hold as submitted —
frontier shared at all rates (AvgPro owns cheap corner, cara owns high-quality + load-robustness; radar
4-axis area still favors cara). cara's May sweep caps wc=0.8 → never pure-3B → cedes cheap corner. **Track
#18** adds cost-extreme tuples to test if cara reclaims the cheap corner → if yes, **Pareto-optimality
returns as the core v2 headline**; if no (scheduling-overhead floor), use scoped framing (cara owns high-q
frontier + load-robust + largest radar area). Decide framing AFTER #18 re-aggregates. Rewrite waits for ALL exp.

## 5. Execution order & dependencies
- **Pipeline-mode block (running):** main baselines + passthru random (phase2a). cara_serve `--scheduling pipeline`.
- **Native-mode block (chained):** phase2b cara tuples → phase3 batching ablation. cara_serve restarts `--scheduling cara`.
- **After main aggregation:** identify best cells/dispatchers → multi-seed (#13, mode-batched) → vLLM-SR (#14).
- **Budget (#12):** rebuild `budget_table_calibrated_n3534.jsonl` (tier-feasibility scheme via
  `build_calibrated_budgets.py` — NOT `build_budget_dataset.py`'s uniform scheme; or recover from
  `_session_archive_d8545.tgz`) → serving (cara native + br4 pipeline) → offline rescore.
- **Offline (#15):** overhead breakdown, bootstrap CI, judge-alt — anytime cells exist, fixed aggregator.

## 6. Live status (2026-06-07)
- Main grid: λ=12 done, 46/168 baseline cells, 0 failures (pid on node).
- Chains armed: phase2 (passthru_ra + native cara tuples) waits for baselines; phase3 (batching) waits for phase2.
- cara scalar-q cells (35) = valid native-mode sensitivity extra; cara tuple cells = May-comparable headline.
- Completion waiters armed (harness background tasks).

## 7. Blockers / open
- **Budget table missing** (#12) — locate `build_calibrated_budgets.py` or recover archive; tier-feasibility 75/45/30%.
- **Best-cell dispatcher** per baseline — pick from main aggregation (likely shortest_queue).
- vLLM-SR Envoy bring-up (#14) — public IPs, timeout 120s; May-6 SR failed at λ24/30 (watch saturation).

## 8. Guardrails (carried forward)
R3/R7 no silent param changes; R4 smoke before long runs; R12 two-location sync + md5 before lease
release; no fabricated numbers/mechanisms (cite file/log); verify routing by DISTRIBUTION not e2e.

## 9. Key paths
- Aggregator (fixed): `cara_paper/smoke_test_apr_13/scripts/aggregate_v2_grid.py`
- Results: `experiment_output/v2_sweep_jun06/{cara,baselines_maygrid,batching_ablation}/`
- Node scripts: `/tmp/v2_baseline_maygrid.sh`, `/tmp/v2_phase2.sh`, `/tmp/v2_phase3_batching.sh`
- Reproduce ref: `route-balance/reproduce/{run_main_table.sh,build_budget_dataset.py,compute_bootstrap_ci.py}`
- Paper: `route-balance-paper/neurips_2026.tex` (+ `tables/`, `PAPER_TODO.md`)
