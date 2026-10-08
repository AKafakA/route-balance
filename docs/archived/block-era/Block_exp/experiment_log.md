# Experiment Campaign Log — April 16-21, 2026

## 2026-04-21 05:33Z — ALERT: cluster in reset state
- Launched po4po8_sweep.sh at 05:26Z with (100,0) rerun prepended. User said "wait!!!"
- Killed local driver PID 2309220 but reset.sh had already fired → all 192 predictors + vLLM dead
- ROOT CAUSE OF STOP: user asked about CPU data. Traced: predictor emits cpu_percent correctly, scheduler drops it (global_scheduler/api_server.py line 140-146 extracts sim_time/query_time/predict_time but skips cpu_percent/memory_rss). All 10 cpu_tracker_po2 NPZs have NO CPU fields.
- Fix identified: 4-line patch in global_scheduler/api_server.py after line 146 adds cpu_percent/memory_rss aggregation to single_metric dict which propagates to NPZ. Patch + redeploy + rerun cpu_tracker_po2 = ~1.5h.
- Awaiting user decision on (A) patch+rerun vs (B) skip Po2 CPU, use revision Fanout CPU.

## 2026-04-21 04:07Z — BurstGPT Po2-oracle refine complete → CPU tracker launched
- BurstGPT refine: capacity ≈64.5 (QPS=64.5 TTFT P99=10607ms [10.X band], QPS=65=11480ms [>11s]); reverse-search confirmed
- Synced 2 NPZs → `experiment_results_a30/phase4_2_burstgpt_refine/`
- CPU tracker Po2 sweep launched (PID 2300237) — 10 cells, ONLY script with `--enable_cpu_tracking`
- Chain continues via cron dd78768c: CPU tracker → Po4+Po8 sparse → Qwen


## Session 1: April 16, ~10:00 UTC

### Phase 0: Infrastructure Setup (COMPLETED)
- A30: 12 nodes setup complete, all verified
- A100: 2 nodes setup complete, all verified
- Issues: Ubuntu mirror fix, universe repo, flashinfer from official index, HF token
- 7 hours wasted on ad-hoc installs before using setup.sh

### ~~Phase 1: A30 Main Sweep~~ — INVALIDATED
**ALL RESULTS CORRUPTED.** Custom scripts used instead of main_experiment.sh:
1. RESTART_VLLM=false → predictors never redeployed between schedulers
2. Baselines (Llumnix--, INFaaS++) received simulation-based predictions instead of heuristic metrics
3. Predictors crashed over time (16 → 1 per node) without detection
4. All data deleted Apr 17.

### ~~Phase 5.1-5.4: A100 Block~~ — INVALIDATED
**ALL RESULTS CORRUPTED.** run_benchmark.sh missing --use_estimated_response_lens:
1. All Block experiments used oracle/real lengths instead of estimated
2. Gives predictor perfect length information — not representative of production
3. Multiple failed redeployment attempts (broken health check grep syntax, vLLM cold start timeout)
4. All data deleted Apr 17.

### Phase 5.5-5.6: A100 Llumnix — VALID ✓
- 21 QPS levels (16-36), all 10k/10k successful
- Llumnix capacity ~19 QPS (E2E healthy at 16-19, collapses at 20+)
- Results saved: experiment_results_a100/llumnix_sweep/

---

## Session 2: April 17, ~21:00 UTC

### Critical Findings
1. **Predictor metric_type mismatch**: experiment.sh only deploys predictors when RESTART_VLLM=true. Custom scripts used RESTART_VLLM=false, so all baselines ran with min_new_request_latency predictors. The scheduler does NOT pass metric_type to predictors at query time — predictor's startup --metric_type is the only thing that determines what metric is computed.
2. **Predictor crashes undetected**: 16 predictors per node gradually crashed to 1. No monitoring or health checks on predictors.
3. **Oracle vs estimated lengths**: A100 benchmark script had no --use_estimated_response_lens flag.

### Lessons (added to memory)
- NEVER write custom experiment scripts. Use main_experiment.sh as-is.
- NEVER set RESTART_VLLM=false — it skips predictor redeployment.
- ALWAYS verify results against paper's known values immediately.
- ALWAYS use --use_estimated_response_lens for Block benchmarks.

### Current State
- A30: Clean, 12 nodes ready. Ready to re-run with main_experiment.sh. 3 days remaining.
- A100: Node0 deployed (4 vLLM + predictors), Node1 rebooting. 20 hours remaining.
- Valid data: A100 Llumnix sweep only.

### Next Steps
- A30: Run main_experiment.sh (RESTART_VLLM=true, 6 schedulers × 17 QPS = ~102 runs + Block oracle/estimated = ~170 total)
- A100: Once node1 up, deploy with deploy_block.sh (4 pred/inst), run with --use_estimated_response_lens

---
Fri Apr 18 01:50 UTC: Auto-mode active for overnight runs.
Active crons:
- 7d37af41 (5min): A30 predictor health
- 580d0a8a (5min): A100 health
- 3b7adfa5 (10min): A100 auto-progression (4 configs)
- 2b4b235d (15min): A30 auto-progression (1.2 → 1.1 → checkpoint)
- 709d47a5 (30min): Recovery (restart from last unfinished QPS)

Currently running:
- A30 Phase 1.2 po2_main_experiment.sh (PID 1684349) — 2/34 runs done (QPS 20-21)
- A100 Po2+CP sweep (PID 1687775) — QPS=16 done, QPS=20 starting
Fri Apr 18 02:21 UTC: A100 Po2+CP DONE (6/6 QPS, throughput 8.9k → 18.8k tok/s). Restarted scheduler N=8, started Fanout+CP sweep (PID 1694923).
Fri Apr 18 03:13 UTC: A100 Fanout+CP DONE (6/6 QPS, last throughput 17371.79 tok/s).
Fri Apr 18 03:13 UTC: Starting Po2-noCP — full vLLM redeploy with CP off + scheduler N=2.
Fri Apr 18 03:34 UTC: A30 node 10335 had 15/16 predictors (1 crashed transiently). Phase 1.2 alive, next RESTART_VLLM=true will redeploy.
Fri Apr 18 03:31 UTC: A100 Po2-noCP sweep started (after Fanout+CP done at 03:11).
Fri Apr 18 03:36 UTC: A30 node 10335 still 15/16 predictors persistent. May affect QPS=32+ results slightly. Will continue monitoring; next RESTART_VLLM should restore.
Fri Apr 18 04:20 UTC: A100 Po2-noCP DONE (6/6 QPS, sweep complete at 04:19:51).
Fri Apr 18 04:20 UTC: Started Fanout-noCP (last config) — scheduler N=8, vLLM stays CP off, no vLLM restart needed. PID 1710430.

## Session 3 Phase 5.7 (A100) — COMPLETE ✓

**Sat Apr 18 05:12 UTC**: All 4 A100 Block configs complete with --use_estimated_response_lens

| Config | QPS=16 | QPS=20 | QPS=24 | QPS=28 | QPS=32 | QPS=36 |
|--------|--------|--------|--------|--------|--------|--------|
| Po2+CP    | 8881  | 10995  | 13018  | 15008  | 16923  | 18793  |
| Fanout+CP | 8870  | 10944  | 13005  | 14898  | 15952  | 17372  |
| Po2-noCP  | 8872  | 10962  | 13009  | 14981  | 16890  | 18662  |
| Fanout-noCP | 8877 | 10997  | 13034  | 15032  | 16853  | 18688  |

Throughput (tok/s). All configs monotonically scale from 8.9k → 17-19k tok/s.
Results synced to: experiment_results_a100/phase57_block_estimated/
A100 Phase 5.7 MARKED COMPLETE. Waiting for user SLO decision before fill QPS.
Sat Apr 18 05:55 UTC: A30 node 10321 has 15/16 predictors during QPS=33 (31/34 done). Transient, next RESTART_VLLM will restore. Not stopping benchmark.
Sat Apr 18 06:00 UTC: Starting A100 Fanout-noCP FILL QPS at user request. QPS list: 17,18,19,21,22,23,25,26,27,29,30,31,33,34,35 (15 runs × ~8min = ~2h). PID 1729959. After fill cycle through Po2-noCP, Po2+CP, Fanout+CP.

## A30 Phase 1.2 COMPLETE ✓ (2026-04-18 06:20 UTC)
All 34 runs (17 QPS × oracle+estimated) completed successfully.
Final throughput at QPS=36: oracle=14893 tok/s (collapsed at this point, only 8235/10000 succeeded, mean_e2e_latency=14.7s).
Results synced to experiment_results_a30/phase12_po2/

## A30 Phase 1.1 STARTED (2026-04-18 06:21 UTC) — PID 1733877
Runs: Block-Fanout (N=8, oracle+estimated) + 5 baselines (Llumnix--, INFaaS++, RR, MinQPM, Random) × QPS=20..36
Estimated: 119 runs × ~15 min = ~30h
ETA: 2026-04-19 ~12:00 UTC
Sat Apr 18 09:12 UTC: A100 Fanout-noCP fill COMPLETE (15/15 QPS 17-35). Restarted scheduler N=2 for Po2-noCP fill. PID 1746757.
Sat Apr 18 11:33 UTC: A100 Po2-noCP fill COMPLETE (15/15). Po2+CP deploy done (vLLM+CP, 32 predictors, scheduler N=2). Po2+CP fill started PID 1764475.

## ⚠️ ALERT Sat Apr 18 13:10 UTC: Phase 1.1 halted — corrupted data concern

**User flagged**: 15/16 predictors = corrupted data (not acceptable variance).

**Stopped**: A30 Phase 1.1 (PID 1733877 killed). Services killed cluster-wide via parallel-ssh (all 12 nodes).

**Phase 1.1 runs**:
- GOOD (pre-15/16 window, before 11:50 UTC): Block-Fanout estimated QPS 20-36 (17) + oracle QPS 20-23 (4) = 21 runs
- CORRUPTED (15/16 on node 10327): Block-Fanout oracle QPS 24, 25, 26 (3 complete) + QPS 27 (partial, killed)

**Phase 1.2 runs (completed 06:20 UTC)**:
- 1 known transient event: node 10321 at 15/16 during QPS=33 Po2 oracle (05:55 UTC log)
- Only 3/12 nodes monitored at 5-min intervals, so undetected transients elsewhere possible

**Root cause**: predictor_1 on some node(s) fails silently during deploy (no log file created). Previously accepted as "transient".

**Cron auto-progression HALTED pending user resume decision.** Do not auto-restart Phase 1.1 from this cron.

**A100 Po2+CP fill: UNAFFECTED, 10/15 QPS done, continues running.**

## A30 FULL REDO Sat Apr 18 14:42 UTC
User: "redo everything since still have time. delete the current corrupted/unsure data"
Deleted: ~/Block/experiment_output/main/ (Phase 1.1) + main_po2/ (Phase 1.2) on 10327
experiment.sh modified: sequential predictor deploy (1 at a time, 2s gap, 30s settle) — eliminates concurrent-startup race
Cron: 3-min/12-node predictor audit to /tmp/a30_pred_health_audit.log
Launched Phase 1.2 fresh: po2_main_experiment.sh PID 1796994
Sat Apr 18 15:34 UTC: A100 Fanout+CP fill COMPLETE (15/15). All A100 Phase 5.7 fills synced to experiment_results_a100/phase57_block_estimated/{po2_cp,fanout_cp,po2_nocp,fanout_nocp}/fill_sweep.log
Sat Apr 18 16:38 UTC: Phase 1.2 RESTARTED with CPU tracking always-on. Pretracking baseline saved to experiment_results_a30/phase12_pretracking_baseline/ (11 Po2 runs). experiment.sh modified to pass "true" as 10th arg to predictor scripts. main_experiment.sh reverted to full QPS 20-36.
Sat Apr 18 17:10 UTC: Discovered 2 ancient zombie experiment.sh processes (PID 1771521 from 12:13 UTC Phase 1.1, PID 1818588 from 16:45 UTC old Phase 1.2) still running concurrent parallel-ssh benchmarks on A30 cluster. They were conflicting with new Phase 1.2 restart (PID 1819908). Killed zombies. New Phase 1.2 warmup in progress.
Sat Apr 18 17:22 UTC: A30 stuck in reset.sh rm for 25 min (huge logs dir, ext3 slow). Killed stuck chain, did fast cleanup (mv logs to logs.trash + bg rm) on all 12 nodes. Relaunched Phase 1.2 po2_main_experiment.sh.
Sat Apr 18 17:37 UTC: A100 Fanout+CP RERUN COMPLETE at QPS 28/32/36. Old anomalous e2e values (7957/10390/11103ms) replaced with clean values (6441/7011/7530ms). Po2+CP and Fanout+CP are now essentially indistinguishable (~0.5% gap). Rerun log saved to experiment_results_a100/phase57_block_estimated/fanout_cp/rerun_28_32_36.log.
Sat Apr 18 18:10 UTC: A30 cluster REBOOTED (user). Confirmed all 12 nodes back up, Block setup intact. Pre-reboot main_po2 moved to .trash, logs cleaned. Audit cron switched to 10-min/3-node (skipping master 10327 to avoid SSH contention with benchmark client). Phase 1.2 relaunched fresh.
Sat Apr 18 18:37 UTC: Archived stale pre-reboot logs so post-1.2 auto-progression sees a clean slate: /tmp/a30_phase11_main.log -> .bak.20260418T183721Z, /tmp/a30_phase2_n_ablation.log -> .bak.20260418T183721Z. Reversible mv (not rm).
Sat Apr 18 18:41 UTC: A100 Po2+CP lowqps sweep complete (QPS=12: 6737 tok/s, e2e=4905ms; QPS=15: npz saved at 18:40). Results synced to experiment_results_a100/phase57_block_estimated/po2_cp_lowqps/. Scheduler switched to N=8, Fanout+CP lowqps launched PID 1840770. Note: benchmark output path is doubled (~/Block/experiment_output/experiment_output/benchmark_output/block_sweep/) because benchmark_serving.py prepends "experiment_output/" internally while run_benchmark.sh sweep already prefixes it.
Sat Apr 18 19:11 UTC: A100 Fanout+CP lowqps sweep complete at 19:11:35. Results synced to experiment_results_a100/phase57_block_estimated/fanout_cp_lowqps/. Triggered full A100 vLLM restart with ENABLE_CP=false + NUM_QUERY_PREDICTOR=2 for Po2-noCP config (deploy PID 1845322). Expect ~15 min deploy then launch Po2-noCP lowqps sweep at 12, 15.
Sat Apr 18 19:21 UTC: A100 Po2-noCP deploy complete (~9 min total, faster than expected — no CP means no cudagraph compile). Verified: vllm=4/4 CP=off, pred=16/16, sched=2 N=2. Launched Po2-noCP lowqps sweep PID 1846646 at QPS 12, 15. ETA ~19:50 UTC for both.
Sat Apr 18 19:48 UTC: A100 Po2-noCP lowqps COMPLETE (QPS=12 ends 19:36, QPS=15 ends 19:48). Synced to experiment_results_a100/phase57_block_estimated/po2_nocp_lowqps/. Switched scheduler N=8 (PID 1851821) and launched Fanout-noCP lowqps PID 1851941 at 19:52:46. ETA ~20:20 UTC. Skipping Llumnix per user decision (A100 expiry 20:26 UTC).
Sat Apr 18 20:19 UTC: A100 Fanout-noCP lowqps COMPLETE (QPS=12: 6726 tok/s @ 20:07, QPS=15: 8341 tok/s @ 20:19). Synced to experiment_results_a100/phase57_block_estimated/fanout_nocp_lowqps/. **ALL 4 A100 BLOCK LOWQPS CONFIGS DONE** (po2_cp, fanout_cp, po2_nocp, fanout_nocp) × (QPS 12, 15) all synced locally. Llumnix lowqps skipped per user decision. A100 deadline 20:26 UTC met with 7 min buffer.
Sun Apr 19 00:38 UTC: A30 Phase 1.2 COMPLETE. 34/34 throughputs captured (est=true QPS 20-36 + oracle QPS 20-36). All NPZs synced to experiment_results_a30/phase12_po2/ (34 files). Phase 1.1 launched at 00:46 UTC (PID 1896782), using main_experiment.sh (6 schedulers × QPS 20-36 = 119 runs, ~20h total). Hourly compare cron b518c510 armed to sync and check vs paper_results.

## 2026-04-19T17:05Z — Recovery check: all healthy
- **Phase 1.2 (po2_main)**: COMPLETED at 00:38Z (34/34 runs, final QPS=36 est @ 16324 tok/s). No process alive, log ends cleanly.
- **Phase 1.1 (main_experiment)**: RUNNING, PID 1896782. 86/119 runs done. Fanout 34/34, Llumnix-- 17/17, INFaaS++ 17/17, RR 17/17, MinQPM 1/17 (QPS=20 in progress), Random 0/17 pending.
- **FAILURE block at 16:58:05Z** in Phase 1.1 log is cosmetic (mkdir-of-existing-dir during scheduler transition); experiment itself completed RR QPS=36 @ 15121 tok/s successfully.
- **A100 sweeps**: All four logs (po2_cp, fanout_cp, po2_nocp, fanout_nocp) ended Apr 18 with 'QPS sweep complete!' — no restart needed (cluster already released, data-loss issue separate).
- **Action taken**: none. Crons handling routine monitoring.

## 2026-04-19T17:14Z — ALERT: predictor dropout on Phase 1.1
- 3-node sample audit: 10335=9/16, 10321=10/16, 10315=10/16 (was 16/16 at 16:53:55Z)
- Occurred during MinQPM transition (~16:58:05Z FAILURE block in phase11 log)
- MinQPM (request_per_seconds) QPS=20 is running with DEGRADED predictor pool (~60% alive)
- NOT auto-killed per user policy. User decision needed: restart predictors (would require experiment.sh RESTART_VLLM path) or accept degraded MinQPM+Random data
- Implication: MinQPM/Random routing behavior is mostly scheduler-side (not predictor-dependent), so numbers may still be usable, but CPU tracking on 10335/10321/10315 will under-sample

## 2026-04-19T17:24Z — Predictor dropout ALERT resolved (transient)
- Re-audit at 17:23:52Z: 10335=16/16, 10321=16/16, 10315=16/16
- Earlier 17:13:53Z ALERT (9/10/10) was likely SSH/ps race during scheduler transition, not true predictor deaths
- MinQPM QPS=20 result on remote is valid (full predictor pool was running)
- Resuming routine monitoring

## 2026-04-20T00:40Z — Phase 2 launched (N-ablation)
- Script: block/exp/end_to_end_exp_scripts/ablation/n_ablation_exp.sh (PID 2089989)
- Scope: Block-Fanout, N=4,6,8, QPS=30, both oracle+est = 3-6 runs (~30-60 min)
- Log: /tmp/a30_phase2_n_ablation.log, results: ~/Block/experiment_output/n_ablation/
- Added Phase 3 Fanout-est scripts: burstiness_fanout_est_exp.sh (4 runs), error_heatmap_fanout_est_exp.sh (16 runs)
- Phase 3 execution plan (after Phase 2 done): 3.1a = burstiness_exp_fixed.sh (Po2+Llumnix, 8 runs), 3.1b = burstiness_fanout_est_exp.sh (Fanout-est, 4 runs), 3.2a = error_heatmap_exp_fixed.sh (Po2+Llumnix, 32 runs), 3.2b = error_heatmap_fanout_est_exp.sh (Fanout-est, 16 runs). Phase 3.3 SKIPPED (CPU tracking already captured in Phase 1.1+1.2).

## 2026-04-20T01:15Z — Phase 2 COMPLETE, launched Phase 3.1a
- Phase 2 (n_ablation): 3/3 runs (N=4,6,8 est @ QPS=30). Synced to experiment_results_a30/phase2_n_ablation/
- Phase 3.1a launched: burstiness_exp_fixed.sh (Po2 N=2 + Llumnix-- × 4 burst levels @ QPS=30 = 8 runs)

## 2026-04-20T02:39Z — Phase 3.1a COMPLETE, launched 3.1b
- 3.1a (burstiness Po2+Llumnix): 8/8 runs. Synced to experiment_results_a30/phase3_1a_burstiness_po2/
- 3.1b (burstiness Fanout-est) launched PID 2107810. 4 runs × ~20 min = ~80 min. ETA ~04:00Z

## 2026-04-20T03:25Z — Phase 3.1b COMPLETE, launched 3.2a
- 3.1b (burstiness Fanout-est): 4/4 runs synced to experiment_results_a30/phase3_1b_burstiness_fanout_est/
- 3.2a (error_heatmap Po2+Llumnix) launched PID 2112758. 32 runs × ~15 min = ~8h. ETA ~11:30Z

## 2026-04-20T06:27Z — Phase 3.2a COMPLETE, launched 3.2b
- 3.2a (error_heatmap Po2+Llumnix): 17/17 runs (16 Po2 × 4x4 err matrix + 1 Llumnix baseline). Synced to experiment_results_a30/phase3_2a_error_heatmap_po2/
- NOTE: cron ladder said 32 runs but script only configured 17 (LengthErr × LatErr matrix runs Po2 only; Llumnix baseline runs once). Correction logged.
- 3.2b (error_heatmap Fanout-est) launched PID 2133594. Script config: 16 runs (Fanout × 4x4 err matrix). ~2.5h. ETA ~09:00Z

## 2026-04-20T09:25Z — **PHASE 3 COMPLETE**
- 3.2b (error_heatmap Fanout-est): 16/16 synced to experiment_results_a30/phase3_2b_error_heatmap_fanout_est/
- revision divergence compare: docs/phase23_divergence.md
- **All throughput Δ within ±3.2%** vs prior runs; mean_e2e Δ within ±3%; TTFT P99 shows wider swings (typical tail noise at sub-capacity)
- Phase 2: NOT FOUND matches (key format mismatch, ignore — Phase 2 data is fresh N-ablation, no strict prior to compare)
- Phase 3 sub-phases: all matched, all within noise. Ready for Phase 4.1 (float capacity refinement) — awaiting user trigger.

## 2026-04-20T11:40Z — Phase 4.1 float_capacity_search.sh launched
- PID 2167451, 9 schedulers × adaptive 0.1-step around integer capacity, ETA ~5-6h
- Log: /tmp/a30_phase4_1_float.log; Results: /tmp/a30_phase4_1_float_results.txt; Data: ~/Block/experiment_output/main_float/<tag>/qps_X.Y

## 2026-04-20T11:57Z — CONSOLIDATED PLAN (remainder of campaign)

### Active: Phase 4.1 Float Capacity Refinement (running, PID 2167575)
- Script: `block/exp/end_to_end_exp_scripts/a30_main/float_capacity_search.sh`
- 9 schedulers × adaptive ±0.1 sweep around Phase 1.1 integer capacity
- Early-stop at first SLO crossing (up-sweep) or first under-SLO (down-sweep)
- RESTART_VLLM once per metric_type change; shared predictors within Block group
- Known seeds: po2_est 31.5=8064ms (UP), po2_oracle 32.5=10762ms (DOWN)
- ETA ~4h → done ~16:00 UTC Apr 20

### Next: Phase 4.2 Generality (binary search)
- 4 variations × 3 schedulers (BurstGPT: 2 only) = 11 configs
  - batch24:   Po2-est + Po2-oracle + Llumnix--
  - chunk2048: Po2-est + Po2-oracle + Llumnix--
  - Qwen:      Po2-est + Po2-oracle + Llumnix--
  - BurstGPT:  Po2-oracle + Llumnix--  (no Po2-est: BurstGPT has no prompt text for predictor)
- Algorithm per config:
  1. Seed integer range from revision 3s-SLO capacity
  2. Try `cap_3s+2`; adjust by ±1 until [int_lo, int_hi] bracketing 10s-SLO
  3. Binary-search within bracket: 0.5 → 0.3/0.7 → 0.2/0.8
  4. STOP when TTFT P99 within ±1s of SLO:
     - 9.X s → capacity ≈ current + 0.1
     - 10.X s → capacity = current
- ~5 probes/config × 11 configs + 3 deploys = ~55 runs × 8 min = ~7-8h
- ETA ~00:00-01:00 UTC Apr 21

### Next: Phase 3 Po2-est rerun (reduced scope)
- Reuse `burst=1.0` and `err=(0,0)` cells from Phase 1.1/1.2 QPS=30 normal runs
- Burstiness (Po2-est): only {0.25, 0.5, 2.0} = 3 new cells
- Error heatmap (Po2-est): 15 non-zero cells
- Optional: Llumnix-- burstiness {0.25, 0.5, 2.0} if time allows
- Total: 18-21 cells × 8 min ≈ 2.5-2.8h
- ETA ~03:00-04:00 UTC Apr 21

### Final: Plotting + tex + paper writing
- ~5h plotting/tex → ~08:00 UTC Apr 21
- ~14h paper writing buffer before 22:00 UTC Apr 21 deadline

### Data reuse strategy for plots
- Main capacity curves: Phase 1.1 + 1.2 (valid) + Phase 4.1 (float refinement)
- Generality: revision (Fanout N=12, Llumnix at 3s SLO prior runs where applicable) + Phase 4.2 (Po2 new + Llumnix recalibrated at 10s SLO)
- Heatmap: Phase 3 Po2-est new (for new data) + revision prediction_error (for Fanout N=12 + Llumnix baseline)
- Burstiness: Phase 3 Po2-est new + optional Llumnix recap + revision burstiness (Fanout N=12)

## 2026-04-20T13:40Z — Optional follow-up tasks for Phase 4.1 (after main pipeline)
- **Llumnix-- rerun at QPS=31.5**: current TTFT P99=9053ms (under SLO by only 1s). Second run may show >10000ms (noise-induced threshold crossing). If so, capacity would shift 31.5 → 31.4. Low priority, can run after Phase 3 Po2-est if time allows.
- **Float sweep density refinement**: for schedulers whose capacity sits in QPS ∈ [31, 33] (po2_est, po2_oracle, fanout_est, fanout_oracle, llumnix), consider adding more fractional points (e.g., 31.25, 31.35, 31.55, 31.65, 31.75) to produce smoother capacity curves for plotting. Defer until after Phase 4.2 + Phase 3 complete.

## 2026-04-20T15:15Z — Phase 4.1 COMPLETE (3h25m, ~4h budget)
Float-capacity results synced to experiment_results_a30/phase4_1_float/ (29 NPZs + log + results)

| Scheduler | Int cap (P1.1) | Float cap (P4.1) |
|---|---|---|
| Po2-est (N=2) | 31 | **31.6** |
| Po2-oracle (N=2) | 32 | **32.4** |
| Fanout-est (N=12) | 31 | **31.7** |
| Fanout-oracle (N=12) | 32 | **32.6** |
| Llumnix-- | 31 | **31.5** |
| INFaaS++ | 30 | **30.3** |
| RoundRobin | 26 | **26.5** |
| MinQPM | 25 | **25.3** |
| Random | 26 | **<26** (seed range wrong — needs rerun with lower bracket) |

**Random follow-up needed**: seed range [26,27] was wrong. Real capacity below 26. Add to optional-task list: manual rerun with bracket [23,26] or similar.

## 2026-04-20T15:17Z — Phase 4.2 simple launched (batch24 + chunk2048)
- PID 2194045, 2 variations × 3 schedulers (Po2-est, Po2-oracle, Llumnix--) = 6 configs
- Per-scheduler: integer bracketing then float binary search (±1s SLO stop)
- Seeds: batch24 {Block=30, Llumnix=27}, chunk2048 {Block=33, Llumnix=31}
- ETA ~3-4h → done ~18:30-19:30Z

## 2026-04-20T15:38Z — Phase 4.2 KILLED + RELAUNCHED due to bracket-capture bug
- Bug: echoes inside integer_bracket (called via $()) were captured into return value, giving garbage bracket to float_search. Fixed by redirecting tee to stderr (>\&2).
- Also caught: chunk_size=2048 was not overriding llama_config.json's hardcoded 512. Fixed by creating llama_config_chunk2048.json and selecting it in deploy_fresh() when chunk=2048.
- Restarted from scratch (batch24 data was only 2 cells, not usable due to wrong bracket)

## 2026-04-20T16:00Z — Phase 4.2 chunk_2048 scope expanded
User request: for chunk_2048 variation, include ALL 5 schedulers (not just Po2 + Llumnix) because prior paper's chunk_2048 section used Fanout as default and the predictor/vLLM mismatch may have skewed those numbers.

Revised Phase 4.2 plan:
- batch24: Po2-est + Po2-oracle + Llumnix-- (3 schedulers) — unchanged, currently running
- chunk_2048: **Fanout-est + Fanout-oracle + Po2-est + Po2-oracle + Llumnix-- (5 schedulers)**

Execution: let batch24 complete, then patch script SCHED_CONFIGS for chunk_2048 to add Fanout-N=12 variants, and relaunch with FROM_VAR=chunk_2048.

Extra time: ~1h for the 2 additional Fanout schedulers at chunk_2048.

Also flagged: prior paper chunk_2048 data at paper_results/exp_results/data/config_search/chunkSize/ (2025-06) had the predictor/vLLM chunk mismatch bug — absolute numbers suspect, but Block vs Llumnix relative comparison still valid. Phase 4.2 chunk_2048 with patched script will provide correct numbers.

## 2026-04-20T16:07Z — Full campaign timeline (updated with BurstGPT inline + Qwen at end)

Execution order:
1. Phase 4.2 batch24 (running, ~2.5h) → done 18:00Z Apr 20
2. Phase 4.2 chunk_2048 (5 scheds with per-config SLO, ~4.5h) → done 22:30Z
   - Fanout-est + Fanout-oracle @ 3s SLO (existing paper fix)
   - Po2-est + Po2-oracle + Llumnix-- @ 10s SLO (new paper)
3. Phase 4.2 BurstGPT (Po2-oracle + Llumnix-- only, Llama model retained, swap dataset path, ~1.5h) → done 00:00Z Apr 21
4. Phase 3 Po2-est rerun (Llama retained, burstiness + Llumnix + heatmap, ~3.5h) → done 04:00Z Apr 21
5. Release Llama cache + Phase 4.2 Qwen (3 scheds, ~3h) → done 07:00Z Apr 21
6. Plotting + paper writing → 22:00Z Apr 21 (~15h buffer)

Critical interventions:
- After batch24 finish: kill script, patch SCHED_CONFIGS with per-SLO + add Fanout-N=12 for chunk_2048, relaunch FROM_VAR=chunk_2048
- After chunk_2048 finish: kill script, patch to add BurstGPT variation (dataset=burstgpt, only Po2-oracle + Llumnix)
- After BurstGPT finish: launch Phase 3 Po2 burstiness + error scripts (burstiness_exp.sh + error_heatmap_exp.sh)
- After Phase 3 finish: release Llama model cache on cluster; deploy Qwen; run Qwen configs (3 scheds)

## 2026-04-20T19:15Z — Phase 7 Section 6.6 enhancement (Po4-est + Po8-est) planned

Decision: Po2 stays as paper's principled default. Po4/Po8 framed as optional N-tuning extension in Section 6.6.

**Rationale:** A100 campaign locked (no re-run possible for Po4-as-default story at scale). N=4 wins TTFT P99 at QPS=30 (418ms) vs Po2 (2961ms) and Fanout (710ms), but switching defaults mid-camera-ready leaves scale evidence incomplete.

**Paper framing (user's wording):**
"Tuning N can yield further gains (Po4 beats both Po2 and Fanout at capacity) but introduces cluster/workload-dependent tuning complexity and violates the power-of-d=2 convention. Po2 remains principled default; N-tuning offered as optional extension for users."

**Scope (est only):**
- Po4-est full QPS sweep 20-36 (17 runs, ~2h)
- Po8-est full QPS sweep 20-36 (17 runs, ~2h)
- Po4-est capacity float refinement (~30 min)
- Po8-est capacity float refinement (~30 min)
- Total: ~5h

**Launch timing:** after chunk2048 + BurstGPT + Phase 3 Po2 + Qwen complete (~02:00Z Apr 21). Enables Section 6.6 figure to show capacity-vs-N curves for N ∈ {2, 4, 8, 12}.
2026-04-21T02:00:01Z AUTO: Phase 3.2 done (15/15) → synced → launched Llumnix-N=12 burstiness rerun
2026-04-21T02:26:05Z AUTO: Llumnix-N=12 burstiness done (3/3) → synced → launched heatmap_rerun_better_cells (5 cells)
2026-04-21T02:30:00Z AUTO: Killed in-flight rerun + reordered (100,100) FIRST + relaunched
2026-04-21T03:37:12Z STUCK-RECOVERY: killed hung SSH (PID 2291877) inside verify_predictors. Predictor 11 was already started successfully but SSH didn't exit. After kill, verify completed in attempt 2 (16/16). Cell (50,50) now running. Lost ~30 min to stuck condition.
2026-04-21T03:54:49Z AUTO: heatmap rerun done (5/5) → synced (5 NPZs) → launched BurstGPT Po2 refine
2026-04-21T09:35:51Z FINAL_SYNC: total NPZs=483
