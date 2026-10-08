{
 "findings": [
  {
   "item": "(a) FOUR ARMS / N2 Decoupled-SameEstimator",
   "status": "ASSET-EXISTS",
   "evidence": "cara_serve.py:1624-1631 defines the arms via env gates: 'CARA_TIEBREAK env: \"queue\" (default, reactive shortest-queue) or \"that\" (predictive T-hat = pred[\"e2e_latency\"], for the decoupled-predictive ablation arm 3 ...)'; `if os.environ.get(\"CARA_TIEBREAK\",\"queue\")==\"that\": scored_lru.sort(key=lambda x:(-x[0],x[4][\"e2e_latency\"],x[2])) else ...(-x[0],x[1],x[2])`. cara_serve.py:1476-1497 defines arm4 static prior: 'optional STATIC per-tier latency prior ... CARA_STATIC_LAT=\"3b=12.3,7b=20.6,14b=18.4,72b=47.4\" ... TPOT_nominal(tier)*predicted_length ... zero telemetry, no XGBoost'. _aggregate/P1_fusion_triangle.json: arm1_uniform_fullobj (w_lat=0.33), arm2_wlat0_reactiveSQ (w_lat=0.0, tiebreak queue, mix 38/19/30/14), arm3_wlat0_predictiveThat (w_lat=0.0, tiebreak that). The review's 'same KNN estimator, pick model first, then shortest-queue within tier' = arm2 EXACTLY (same native machinery, w_lat=0 -> model tier by quality/cost, then reactive shortest-queue dispatch). Arm4 adds the optional static tier-latency prior. Differs from arm3 (predictive T-hat within-tier instead of reactive queue).",
   "file": "block/global_scheduler/cara/cara_serve.py",
   "line": 1624
  },
  {
   "item": "(a) N2 full weight-space sweep of decoupled arm",
   "status": "ASSET-MISSING",
   "evidence": "P1_fusion_triangle.json only ran arm2/arm3 at uniform weights and lambdas l12/24/30 (arm2 'l12:3329,l24:3525,l30:3887'); arm4_static_prior_result.json only 'stat_l18' + 'overload_40_6'. No full weight-space sweep of arm2/arm4 across all principal rates exists \u2014 those require new cara_serve serving runs (live queue), not offline.",
   "file": "experiment_output/v2_sweep_jun06/_aggregate/P1_fusion_triangle.json",
   "line": 1
  },
  {
   "item": "(a) baseline estimator separation",
   "status": "REVIEW-CORRECT",
   "evidence": "passthrough.py:7 docstring references (no) 'quality-aware model selection'; class PassthroughRouter (no estimator). avengers_pro.py:11-20 loads its OWN kmeans artifact ('Load the trained artifact (kmeans / rankings / normalizer ...'). best_route.py:13 note: 'where Q\u0302 comes from the loaded ModelEstimator (same estimator Cara's native ...)' \u2014 so best_route can share Cara's estimator standalone (nuance: not strictly its 'own'), but avengers_pro uses its own kmeans and passthrough has no quality.",
   "file": "block/global_scheduler/cara/routers/best_route.py",
   "line": 13
  },
  {
   "item": "(b) STEADY-STATE fields / N5",
   "status": "ASSET-EXISTS",
   "evidence": "cara/cara_q0.20_l10.json top-level: num_prompts 3534, completed 3534, failed 0, duration 362.59, request_rate 10.0 (offered vs completed + duration present). response_details is a per-request list (len 3534) with keys request_id, input_len, output_len, ttft, client_ttft, itl, e2el, scheduling_overhead, model, instance_id. Completed-throughput, offered!=completed overload labeling, and per-request-latency-vs-index queue-growth slope are all computable OFFLINE. CAVEAT: no absolute wall-clock arrival/first-token/completion timestamp field is stored, so exact queue-length-over-time is only reconstructable approximately (arrival schedule = request_rate/burstiness seed; completion = arrival + e2el).",
   "file": "experiment_output/v2_sweep_jun06/cara/cara_q0.20_l10.json",
   "line": 1
  },
  {
   "item": "(b) run length per cell",
   "status": "ASSET-EXISTS",
   "evidence": "Confirmed 3534 prompts at request_rate=10 gave duration 362.59s (num_prompts 3534, request_rate 10.0, duration 362.5904462529943); run length ~= num_prompts/lambda.",
   "file": "experiment_output/v2_sweep_jun06/cara/cara_q0.20_l10.json",
   "line": 1
  },
  {
   "item": "(c) PER-DATASET quality / N7",
   "status": "ASSET-EXISTS",
   "evidence": "data/cara/scored/test_scored_filtered.jsonl record keys include 'dataset','is_harmful','source','reference_text'; `grep -oE '\"dataset\":'` = 3634 occurrences, '\"source\":' = 3634 (every test row tagged). deepeval score under models[*]['llm_judge_scores']['deepeval-llama3.1-8b-it_reference']. Served response_details carry request_id (e.g. 'bench-fd5c611f-0') joinable to scored rows (also 'bench-...-N'). A per-dataset table is offline-computable by joining request_id -> dataset. NOTE: aggregate_v2_grid.py has NO per-dataset support (grep for dataset/by_dataset/source returned nothing), so a small new offline join script is needed \u2014 no GPU.",
   "file": "data/cara/scored/test_scored_filtered.jsonl",
   "line": 1
  },
  {
   "item": "(d) SERVED-vs-STORED re-judge / N7",
   "status": "ASSET-EXISTS",
   "evidence": "R1_served_vs_lookup.json gives exact-match fractions per cell (RB_wq0.8_l12 exact_pct 10.27, normalized 0.25, prefix 2.49, match_total 13.02, diverged 86.98; also BR/AP/PT cells). R2_servedtext_primaryjudge.json gives paired served-vs-stored delta (rb_wq08_l12 paired_delta_served_minus_stored -0.0748, ci95 [-0.1014,-0.0465], n_paired 492), cross-system paired gaps (rb-br4 served_gap 0.0237 ci95[0.0102,0.0371]), and judge_drift_identical_text mean -0.0016 median 0.0 (the 'drift=0 on identical text' claim). Scripts: smoke_test_apr_13/scripts/r1_served_vs_lookup.py, r2_aggregate.py, r2_offset_diagnostics.py. Fully offline.",
   "file": "experiment_output/v2_sweep_jun06/_aggregate/R2_servedtext_primaryjudge.json",
   "line": 1
  },
  {
   "item": "(e) COST price table / N8",
   "status": "ASSET-EXISTS",
   "evidence": "Per-token price table lives in scheduler_config_qn_minmax.json:87-112 (cost_per_input_token/cost_per_output_token per tier: 6e-08, 6e-08, 7e-08, 1.5e-07, 3.8e-07/4e-07). Consumed in cara_serve.py:1119-1121 (cost_per_token = meta.get('cost_per_token',0.01); cost_in/out_per_tok). A SECOND systems-cost metric already exists offline: cost_sensitivity_e8.json has a 'gpuhour' row (RB 1.38, AP 1.38, BR4 2.31, PASS 3.03). Per-request instance_id+model+output_len+e2el in response_details make GPU-seconds/request computable offline.",
   "file": "block/config/cara/scheduler_config_qn_minmax.json",
   "line": 87
  },
  {
   "item": "(f) NORMALIZATION / N9",
   "status": "ASSET-EXISTS",
   "evidence": "Deployed v2 config scheduler_config_qn_minmax.json:54 '\"quality_norm\": \"minmax\"' (quality sub-score minmax across candidates). cara_serve.py:1464-1473 header 'Normalization + Scoring (Avengers-Pro-style max-divide) ... Replaces legacy min-max'; cara_serve.py:1565-1567 cost_score = 1.0-(cost/(max_cost+eps)); lat_score = 1.0-(lat/(max_lat+eps)); bal_score = 1.0-(util/(max_util+eps)) \u2014 i.e. cost/lat/bal use MAX-DIVIDE (max-only), quality uses MINMAX. So the review's 'max-normalized' and 'minmax' are BOTH partially right (mixed).",
   "file": "block/global_scheduler/cara/cara_serve.py",
   "line": 1464
  },
  {
   "item": "(g) k=10 confirmed",
   "status": "REVIEW-CORRECT",
   "evidence": "scheduler_config_qn_minmax.json:73-76 knn estimator 'model_dir\":\".../deploy_knn_k10\",\"k\":10'. R5_k_sensitivity.json _method: 'deployed k=10' and reports k in {5,10,20,50}. Paper deployed value k=10 is confirmed.",
   "file": "block/config/cara/scheduler_config_qn_minmax.json",
   "line": 76
  },
  {
   "item": "(g) validation split / N6",
   "status": "ASSET-MISSING",
   "evidence": "wc -l: data/cara/train_scored_filtered.jsonl = 14919, data/cara/scored/test_scored_filtered.jsonl = 3634 (matches review's 14919/3634). No dedicated validation split used for PRESET selection was found; dev files exist only for predictor training (data/cara/broadcast_v3_50_dev.json, cara_v3_dev_norat.json). The 8 weight tuples were a FIXED grid, not validation-selected: handoff 2026-06-07 line 41 'Param grids silently != May -> fixed (cara 8 tuples ...)'. A train/val/test preset-selection redo needs new offline aggregation (data present) but presets themselves were swept/reported on test.",
   "file": "cara_paper/handoffs/2026-06-07_v2_rerun_handoff.md",
   "line": 41
  },
  {
   "item": "(h) PUBLIC k=50 vs k=10 / P6",
   "status": "REVIEW-CORRECT",
   "evidence": "route-balance/REPRODUCE.md:227 '--k 50 --weighted' with out-dir 'models/route_balance/knn_k50/', while paper/deployment use k=10 (scheduler_config_qn_minmax.json:76). Genuine inconsistency confirmed.",
   "file": "route-balance/REPRODUCE.md",
   "line": 227
  },
  {
   "item": "(h) PUBLIC bootstrap 1000 vs 10000 / P6",
   "status": "REVIEW-CORRECT",
   "evidence": "route-balance/REPRODUCE.md:106 '--B 1000' and :110 'B=1000 resamples'; reproduce/compute_bootstrap_ci.py:62 'def bootstrap_ci(values, n=1000,...)' and :90 print '=== Bootstrap CIs at lambda=12 (B=1000) ==='. Paper claims 10,000 resamples (handoff 2026-06-07 line 120 confirms internal '10k resamples' for the App.A CI). Public command uses 1,000 \u2014 inconsistency confirmed.",
   "file": "route-balance/reproduce/compute_bootstrap_ci.py",
   "line": 62
  },
  {
   "item": "(h) single-seed ambiguity / P6",
   "status": "REVIEW-CORRECT",
   "evidence": "route-balance/REPRODUCE.md:339 'Single seed. Headline cells were run with a single per-cell PRNG seed; bootstrap CIs are over the N=3534 workload, not over re-runs.' Confirms the single-seed-vs-multiseed ambiguity the review flags.",
   "file": "route-balance/REPRODUCE.md",
   "line": 339
  },
  {
   "item": "(i) burst / square-wave / non-stationary / tier-removal / overload assets / N3,N5",
   "status": "ASSET-EXISTS",
   "evidence": "_aggregate/P3_bursty_results.json (gamma b=0.3), P3sq_wq6_results.json (square-wave 30s@30rps/30s@6rps), P3_nonstationary_3x3.json (3x3 stationary/gamma/square), Q3_static_vs_live_probe.json (square-wave mix by phase + outlen-quartile). tier_removal/ (no72b_uniform_l12, clean_no72b_*). Overload discriminator 40/6 exists: arm4_static_prior_result.json 'overload_40_6' + review_extras/arm4_sq40.json, ov_arm1_sq40.json. All offline-derived from existing traces.",
   "file": "experiment_output/v2_sweep_jun06/_aggregate/P3_nonstationary_3x3.json",
   "line": 1
  },
  {
   "item": "(i) gsm8k / OOD held-out-domain routing / N7",
   "status": "ASSET-MISSING",
   "evidence": "find over experiment_output/v2_sweep_jun06 for *gsm*/*ood*/*heldout*/*held_out* returned NO matches. An end-to-end held-out-domain routing serving experiment is not present (needs new run). Only offline scored data exists per dataset tag (see item c). NOTE: gemma3_scored_full.jsonl + gemma_rejudge_bootstrap.json exist but are judge-alt/re-judge robustness, not a held-out-domain routing run.",
   "file": "experiment_output/v2_sweep_jun06",
   "line": 1
  },
  {
   "item": "N3 STATIC PRIOR retains benefit",
   "status": "REVIEW-CORRECT",
   "evidence": "arm4_static_prior_result.json finding: 'Arm4 reproduces arm1 on ALL axes (mix within 1-2pp, quality 0.369 vs 0.370, latency same-or-better) under BOTH stationary and 40/6 overload. The learned/live latency signal is NOT load-bearing for cross-tier shaping; a static per-tier prior ... suffices.' implication: 'no decoupled combination achieves (intro L91) must soften: a static decoupled latency prior matches.' The review's claim that the static prior retains nearly all headline benefit is supported by existing data.",
   "file": "experiment_output/v2_sweep_jun06/_aggregate/arm4_static_prior_result.json",
   "line": 1
  },
  {
   "item": "N1 RouterWise/BOute not cited",
   "status": "REVIEW-CORRECT",
   "evidence": "grep -rniE 'routerwise|boute' over cara_paper/*.tex and sample.bib returned NO matches \u2014 neither is cited. The joint-novelty claim exists: cara_neurips.tex:42 'No prior system jointly considers all three axes across a heterogeneous multi-model cluster under realistic open-loop arrival patterns.'",
   "file": "cara_paper/cara_neurips.tex",
   "line": 42
  },
  {
   "item": "N1/P5 OmniRouter 'future Dec-2026 date'",
   "status": "REVIEW-WRONG",
   "evidence": "sample.bib:597 '@misc{omnirouter2025, title={{OmniRouter}...}, author={Various}, year={2025}, eprint={2502.20576}, archivePrefix={arXiv}}'. Year is 2025 (eprint 2502 = Feb 2025), NOT a future Dec-2026 date in this repo's bib. UNVERIFIABLE whether a different PDF the review saw differs.",
   "file": "cara_paper/sample.bib",
   "line": 597
  },
  {
   "item": "P1 LPT / P2 simplex terminology",
   "status": "REVIEW-CORRECT",
   "evidence": "cara_neurips.tex uses 'LPT' at lines 48,87,89,90,93,114,189,212,226,233 and 'simplex' at lines 82,87,128,132,193 \u2014 confirming the paper invokes classical LPT and simplex terminology the review objects to (the '3-simplex' wording and the LPT approximation-guarantee framing are present to review).",
   "file": "cara_paper/cara_neurips.tex",
   "line": 87
  },
  {
   "item": "C1 double-blind / anonymization",
   "status": "PARTIALLY-CORRECT",
   "evidence": "cara_neurips.tex is ALREADY anonymized: line 24 '\\author{Anonymous Authors}', line 109 'released at an anonymous repository: \\url{https://anonymous.4open.science/r/cara-neurips-2026}'. grep for 'Conference''17'/DOI/real-arxiv-id found NONE in this tex. So C1's claim that the current artifact 'reveals authors/institution/arXiv id/public repo' and has a 'Conference''17' placeholder is NOT supported by cara_neurips.tex; it may target a different EuroSys-formatted PDF not located here. UNVERIFIABLE for that other file.",
   "file": "cara_paper/cara_neurips.tex",
   "line": 24
  },
  {
   "item": "FACT 13 instances / testbed",
   "status": "PARTIALLY-CORRECT",
   "evidence": "cara_neurips.tex:58 contains '13 instances'; A100/A30/V100 all appear (e.g. line 31). Consistent with 'four Qwen2.5 tiers, 13 instances'. The 28-GPU count (2xA100 TP4 + 3xV100 TP4 + 8xA30 = 28) is arithmetic-consistent but not stated as '28-GPU' verbatim in this tex.",
   "file": "cara_paper/cara_neurips.tex",
   "line": 58
  },
  {
   "item": "FACT 442 configurations vs 215",
   "status": "REVIEW-WRONG",
   "evidence": "cara_neurips.tex says '215-cell' / '215 cell' evaluation corpus (lines 48,109,114,208: 'The 215-cell evaluation corpus (raw per-request traces + AGGREGATE_FULL.csv)'). No '442' appears in this tex; the review's '442 configurations' does not match this version (may reflect a larger later v2 sweep).",
   "file": "cara_paper/cara_neurips.tex",
   "line": 109
  },
  {
   "item": "FACT +0.013 (0.406->0.419)",
   "status": "PARTIALLY-CORRECT",
   "evidence": "handoff 2026-06-07 line 251 multi-seed: 'cara wq0.8 q=0.4180... br4 t0.5 q=0.4056'; delta = 0.0124 ~ +0.012-0.013. cara_neurips.tex:264-265 contain '0.013'. Peak cara CI 0.4187 [0.4089,0.4288] (handoff:120). Numbers round to the review's 0.406->0.419 / +0.013 approximately; exact literals 0.406/0.419 not grep-matched in tex.",
   "file": "cara_paper/handoffs/2026-06-07_v2_rerun_handoff.md",
   "line": 251
  },
  {
   "item": "N4 second family / second topology",
   "status": "ASSET-MISSING",
   "evidence": "Only Qwen2.5 four-tier family is served (all_model_scores keys: Qwen2.5-3B/7B/14B/72B in response_details). gemma assets are offline scored/re-judge only (data/cara/scored/gemma3_scored_full.jsonl, _aggregate/gemma_rejudge_bootstrap.json {s1 delta 0.0197, s2 delta 0.0120}) \u2014 not an end-to-end Gemma/Llama serving sweep. tier_removal/ (no72b) and cara_edge/ (cost-skewed weights) exist but are not a full capacity-balanced-vs-cost-skewed replica-allocation topology sweep. A second-family or second-topology serving campaign needs new GPU runs.",
   "file": "data/cara/scored/gemma3_scored_full.jsonl",
   "line": 1
  },
  {
   "item": "N9 objective sensitivity (minmax/max/z-score, candidate add-remove, weight transfer)",
   "status": "ASSET-EXISTS",
   "evidence": "Per-request response_details store all_model_scores and raw predicted_quality/predicted_length/actual_cost per candidate (cara_q0.20_l10.json response_details[0]: all_model_scores{4 tiers}, predicted_quality, predicted_length, budget_cost/actual_cost), so re-scoring under minmax vs max-only vs z-score and candidate add/remove is recomputable OFFLINE from stored per-candidate scores. cost_sensitivity_e8.json already shows price-perturbation robustness (72Bx0.5/x2, 14Bx0.5/x2). Weight-tuple transfer across rates: uniform_mix_vs_lambda.json shows mix flat across lambda 6->30.",
   "file": "experiment_output/v2_sweep_jun06/cara/cara_q0.20_l10.json",
   "line": 1
  }
 ],
 "summary": "I audited the private repo /home/wd312/Code/llm/Block and public mirror /home/wd312/Code/llm/route-balance to determine which review asks are answerable from existing offline data (no GPU) versus needing new serving runs. (a) The review's 'Decoupled-SameEstimator' baseline already exists mechanically as arm2 (cara_serve.py env gate CARA_TIEBREAK=queue with w_lat=0: same KNN estimator, model tier by quality/cost, then reactive shortest-queue dispatch); arm3 is the predictive-T-hat variant, arm4 (CARA_STATIC_LAT) is the static per-tier latency prior \u2014 all captured in _aggregate/P1_fusion_triangle.json and arm4_static_prior_result.json. What is missing is a FULL weight-space sweep of the decoupled arm across all rates (needs new serving runs). (b) Steady-state analysis is largely offline-computable: per-cell JSONs store num_prompts/completed/failed/duration and per-request e2el/ttft/itl, enabling completed-throughput and overload labeling; only absolute arrival/completion wall-clock timestamps are absent, so exact queue-length-time-series is approximate. (c) Per-dataset quality IS offline-buildable \u2014 test_scored_filtered.jsonl tags every row with dataset/source and request_id joins to served traces \u2014 though aggregate_v2_grid.py currently lacks per-dataset code. (d) Served-vs-stored re-judge assets fully exist (R1/R2 JSONs with exact-match fractions, paired cross-system CIs, judge-drift ~0 on identical text). (e) Cost uses a per-token price table in scheduler_config_qn_minmax.json; a GPU-hour metric already exists (cost_sensitivity_e8.json) and GPU-seconds/request is offline-computable. (f) Normalization is mixed: quality=minmax (config line 54), cost/lat/bal=max-divide (code 1565-1567), so the review's minmax-vs-max claim is partially right. (g) k=10 is confirmed deployed; there is NO dedicated validation split for preset selection (the 8 tuples were a fixed grid). (h) The public REPRODUCE.md genuinely diverges from the paper: k=50 vs k=10, B=1000 vs 10000 resamples, and an explicit single-seed note \u2014 all confirmed. (i) Burst/square-wave/non-stationary/tier-removal/40-6-overload assets exist offline, but gsm8k/OOD held-out-domain routing and a second model family/topology serving campaign do NOT exist and need new runs. On review fact-checks: RouterWise/BOute are indeed uncited and the joint-novelty claim is present (cara_neurips.tex:42); but OmniRouter's bib year is 2025 (not a future Dec-2026 date), the tex says '215-cell' not '442 configurations', and cara_neurips.tex is already anonymized with no Conference'17 placeholder \u2014 so several review assertions are wrong or target a different PDF than the one in-repo."
}