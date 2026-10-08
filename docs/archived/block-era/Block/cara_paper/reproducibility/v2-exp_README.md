# v2-exp — Self-Contained Reproduction Pack (Cara/RouteBalance v2)
**2026-06-09. This directory is THE single source to reproduce every v2 result.**

It bundles everything: deployed models, data, the collected results (ground truth), configs, code, and
two instruction documents — one per code repo. Pair with a fresh CloudLab window for the full re-run, or
run the analysis-only smoke immediately on openclaw.

## Contents
```
v2-exp/
  models/cara/        deployed v2 predictors+baselines (latency XGB, quality KNN-k10, best_route_4way, avengers_pro_qwen_v2_lc*)
  data/cara/          scored data (llama + gemma-3 deepeval), 3534-prompt bench, budget tables
  results/            COLLECTED GROUND TRUTH — experiment_output/v2_sweep_jun06 (442 cells + _aggregate/AGGREGATE.csv)
  configs/cara/       scheduler_config_qn_minmax.json, predictor_xgboost_3model.json, model_deployment.json, host_configs.json
  code/scripts/       aggregate_v2_grid.py (FIXED), consolidate_v2.py (producer-of-record), score_with_deepeval.py
  code/orchestration/ v2_*.sh node run scripts + vllm_sr/ configs (config.yaml, envoy_qwen.yaml, relaunch)
  code/serving/cara/  the cara serving stack (global_scheduler/cara: routers, cara_serve.py, ...)
  V2_CHANGELOG_AND_SUMMARY.md   the v2 changes/fixes + §6 numbers (the targets to reproduce)
  v2_deployed_md5.txt           checksums of deployed models/data/configs
  REPRODUCE_with_BLOCK.md       INSTRUCTION DOC 1 — reproduce via the Block repo
  REPRODUCE_with_ROUTEBALANCE.md INSTRUCTION DOC 2 — reproduce via the route-balance repo
```

## Two reproduction levels (both detailed in the two instruction docs)
- **L1 — analysis smoke (no cluster, ~2 min):** re-derive EVERY §6 number from `results/` + `data/`.
  `python3 code/scripts/aggregate_v2_grid.py && python3 code/scripts/consolidate_v2.py` → must match §6.
- **L2 — full cluster re-run (fresh CloudLab window):** redeploy stack from `models/`+`configs/`, run
  `code/orchestration/*.sh`, re-aggregate, compare to `results/` ground truth.

## The two code repos (pick one for L2)
- **Block** (`~/Code/llm/Block`, commit `9ae1f20`): the full serving + analysis stack. → `REPRODUCE_with_BLOCK.md`.
- **route-balance** (`~/Code/llm/route-balance`, `v2-reproduce/`): the public paper repo with v2 scripts synced
  in. → `REPRODUCE_with_ROUTEBALANCE.md`.
Both drive the SAME experiments against the SAME `models/`+`data/`+`results/` in this pack.

## Targets (what must reproduce — see V2_CHANGELOG_AND_SUMMARY.md §6)
cara peak deepeval **0.4190 #1**; cost-extreme cheapest **1.68e-5/req**; load-robustness br4 collapse 21×
(64204ms) vs cara bounded 2389ms @λ30; bootstrap [0.4089,0.4288]; overhead 134ms; vLLM-SR collapse 89–97%
@λ≥18; judge-alt cara #1 under both judges, Pearson r=0.555. dispatcher_only MUST = 21 (random included).
