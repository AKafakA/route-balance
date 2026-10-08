# SoCC 2026 Camera-Ready Campaign — Lessons Learned

April 16-21, 2026 (6 days, ~648+ GPU-hours of experiments)

## Cost summary

| # | Issue | Time/GPU lost | Severity |
|---|---|---|---|
| 1 | A100 manual installs instead of setup.sh | 10h | Critical |
| 2 | Custom A30 scripts with RESTART_VLLM=false | ~2 days | Critical |
| 3 | A100 oracle lengths (missing `--use_estimated_response_lens`) | 8h overnight | Critical |
| 4 | A100 fixed output_dir overwrote 4 configs | **Permanent data loss** + 1 day | Catastrophic |
| 5 | "15/16 predictors is transient" silent acceptance | 5+ hours data discarded | Critical |
| 6 | Setup.sh failure handling — improvised fixes | 7h | Critical |
| 7 | Failed deploy with broken grep escaping (`\|` vs `|`) | 3+ failed redeploys | Major |
| 8 | _fixed ablation scripts had broken --burstiness wiring | Phase 3 invalidated multiple times | Major |
| 9 | Llumnix N=2 instead of N=12 in burstiness | Bogus burst=1.0 result | Major |
| 10 | Heatmap single-deploy variance (~±4% e2e noise) | 5 cells appeared "better than baseline" (false) | Major |
| 11 | CPU tracking patch incomplete (scheduler patched, benchmark wasn't) | ~1.5h, 10 NPZs missing CPU | Major |
| 12 | OOM during concurrent Qwen predictor + vLLM deploy | ~30 min recovery + cluster reset | Major |
| 13 | verify_predictors infinite SSH hang from `nohup &` | 12+ min × multiple recurrences | Major |
| 14 | Killed script left vLLM with stale KV cache | User caught before bias | Near-miss |
| 15 | Wrong Po2-oracle seed (started 6 below true cap) | 16-20 min saved by user intervention | Near-miss |
| 16 | State-machine cron auto-relaunching after manual kill | Conflicting state | Minor |

**Estimated total lost compute/wall-clock: ~5-6 days of GPU time + permanent A100 NPZ loss for 3 of 4 configs.**

---

## Lesson 1: Installation order & environment setup

**Bug**: Manual ad-hoc installs on CloudLab nodes after setup.sh failed (Apr 16). Mirror/repo issues "fixed" by improvising → corrupted environments, ABI mismatches, SSH exhaustion.

**Cost**: 10 hours overnight wasted. Real fix was 2 lines in setup.sh (Ubuntu mirror + universe repo).

**Rule**: When setup.sh fails on CloudLab:
1. Fix the script (not the nodes)
2. Reload nodes (returns clean state)
3. Re-run setup.sh once
4. NEVER run ad-hoc `apt`, `pip`, `add-apt-repository` on cluster nodes
5. NEVER re-run setup.sh on dirty nodes

---

## Lesson 2: Never write custom experiment scripts

**Bug**: Wrote custom scripts that set `RESTART_VLLM=false` to "save time" — skipped predictor redeployment between schedulers (Apr 17).

**Consequences**:
- Baselines (Llumnix--, INFaaS++) got `min_new_request_latency` predictor instead of their own metric → all baselines used WRONG metric
- Predictors crashed silently from 16/node → 1/node over hours, no monitoring
- All Phase 1 + Phase 2 A30 results corrupted

**Cost**: 2 days wasted, full re-run required.

**Rule**:
- For main sweep: use `main_experiment.sh` as-is
- For ablations: use existing top-level scripts
- NEVER write custom orchestration scripts when tested ones exist
- NEVER set `RESTART_VLLM=false` (it skips critical predictor metric/state reset)

---

## Lesson 3: Output directory MUST encode every config axis

**Bug**: A100 `run_benchmark.sh` writes to fixed path `experiment_output/benchmark_output/block_sweep/block_qps${qps}` — only QPS in path, no scheduler/CP/N axis. Successive configs (Po2+CP, Fanout+CP, Po2-noCP, Fanout-noCP) all overwrote each other.

**Consequences**: Only the LAST config's NPZ files survived. Earlier configs' raw per-request data **permanently lost** (only `sweep.log` text aggregates remained).

**Cost**: 1 full day of A100 GPU time + permanent loss of 3 configs' raw arrays. Paper p99 / CDF figures couldn't be regenerated. A100 lease expired before recovery possible.

**Rule**:
- Output path template MUST include every config axis (scheduler, N, CP on/off, estimation mode, batch, chunk, model, etc.)
- A30's `experiment.sh` does this correctly: paths like `qps_X_num_queries_Y_n_Z_chunked_W_predictor_A_global_B_len_estimated_C_max_slo_D_..._batch_E_chunk_F`
- For scripts that have fixed paths: rsync per-config IMMEDIATELY after each QPS, before advancing to next config

---

## Lesson 4: Predictor count is binary — 16/16 or corrupted

**Bug**: Accepted 15/16 predictors as "transient variance" on 3 occasions (nodes 10335, 10321, 10327) Apr 18.

**Consequences**: 5+ hours of Phase 1.2 + 1.1 data discarded. Full redo.

**Root cause**: `experiment.sh` originally launched 7 predictors in parallel + 9 in parallel. 16 concurrent pandas dataframe loads → ~8% failure rate per node. Fixed by **sequential deploy** (16 × 2s apart, 30s settle).

**Rule**:
- Pre-deploy: verify all 12 nodes show 16/16 BEFORE running benchmark
- Mid-run: audit every 3 min; if any drops, kill benchmark, mark QPS invalid, re-run
- Never say "transient, next RESTART_VLLM will fix" — escalate immediately
- Hard gate: `verify_predictors.sh` after deploy with ABORT-on-failure

---

## Lesson 5: A100 oracle vs estimated lengths

**Bug**: A100 `run_benchmark.sh` was missing `--use_estimated_response_lens`. Block experiments used oracle lengths (perfect prediction = cheating).

**Cost**: All A100 Block phases 5.1-5.4 invalidated, 8h overnight + recovery time.

**Rule**: ALWAYS verify the EXACT benchmark command matches paper methodology BEFORE starting any timed run. Smoke-test the command output with 100 samples and check NPZ field validity first.

---

## Lesson 6: _fixed scripts had broken parameter plumbing

**Bug**: Earlier session created `_fixed` variants of error_heatmap and burstiness scripts under `/ablation/`. They had broken `--burstiness` and `--length_error_pct` arg passing — values never reached the benchmark/scheduler.

**Cost**: Phase 3 results invalidated multiple times. Confusion for hours determining which script was canonical.

**Rule**: Top-level scripts in `block/exp/end_to_end_exp_scripts/` are canonical. `_fixed` and `_fanout_est` variants under `/ablation/` are quarantined and broken — never use them.

---

## Lesson 7: Llumnix N value bug in shared scripts

**Bug**: `burstiness_exp.sh` used `N_SELECTED=2` for both Po2 (correct) and Llumnix (WRONG — Llumnix should be N=12 broadcast).

**Cost**: At burst=1.0 (Poisson), Po2 appeared dramatically better than Llumnix — but it was an artifact of comparing N=2 vs N=2 (not the real Llumnix). User caught this. ~20 min waste, 3 cells re-run.

**Rule**: Llumnix is fundamentally a fanout/broadcast scheduler — must always use N=12. Add per-scheduler N branching:
```bash
if [ "$scheduler" = "min_new_request_latency" ]; then
    N_THIS_RUN="$N_SELECTED"  # Po2 = 2
else
    N_THIS_RUN="12"  # Llumnix uses broadcast
fi
```

---

## Lesson 8: Single-deploy variance ≈ ±4% — invalid for absolute comparisons

**Bug**: Original heatmap ran all 15 error cells with one vLLM deploy (deploy-per-scheduler). Run-to-run e2e_mean variance is ~±4% under load.

**Symptom**: 5 cells appeared "better than baseline" (delta -0.7% to -3.2%). This is physically impossible — adding error to predictions should never improve performance.

**Cost**: User noticed anomaly. 5 fresh-deploy reruns took 50 min. 4/5 cells flipped to expected positive degradation; 1 stayed near zero.

**Rule**: For absolute-comparison experiments (heatmap, burstiness reruns), do **fresh deploy per cell**. For capacity-curve experiments, single-deploy is acceptable since we look at relative trends across many QPS.

---

## Lesson 9: When patching new fields, trace the full data pipeline

**Bug**: Patched scheduler to aggregate `cpu_percent` into `single_metric`, but benchmark_serving.py uses a **whitelist** for NPZ save fields. CPU fields fell through.

**Cost**: 10 cpu_tracker_po2 NPZs collected with no CPU data. Had to re-run after second patch.

**Rule**: When adding a metric, trace it end-to-end:
1. Source: predictor returns it
2. Aggregator: scheduler `single_metric`
3. Per-request collector: benchmark `output['xxx']` extraction
4. Persistence: benchmark `data` dict for `np.savez`

If any link is missing, the field disappears silently.

---

## Lesson 10: OOM during concurrent deploy with new model cache

**Bug**: When deploying Qwen first time, `qwen_capacity_search.sh` launched 16 predictors in parallel. Each predictor needed to train ~10 RandomForest models (Qwen had no cache). Combined with vLLM holding 14GB Qwen weights, hit 85-88% memory + 84-88% swap. OOM killer took out vLLM processes on most nodes.

**Cost**: ~30 min recovery + cluster reset.

**Rule**: For new model deployments:
1. First run `warmup.sh <model>` — uses `PREDICTOR_WORKERS=1`, `RUN_EXP=false`. Trains ONE predictor, populates `~/Block/cache/*.pkl`.
2. THEN do regular deploy with all 16 — they all hit cache, ~30s startup, no memory contention.

For cached models, regular deploy is safe.

---

## Lesson 11: SSH-from-shell with `nohup &` can hang indefinitely

**Bug**: `verify_predictors.sh` used `ssh -n "$host" "nohup python ... &"` to spawn missing predictors. The remote shell sometimes doesn't exit cleanly after backgrounding python (stdin not closed). SSH hangs. `wait` blocks forever.

**Cost**: Multiple incidents of 12+ min infinite hang. Triggers chain delays.

**Fix applied** (3 layers of safety):
```bash
timeout 30 ssh -n -o ServerAliveInterval=5 -o ServerAliveCountMax=2 "$host" \
    "nohup python ... > log 2>&1 < /dev/null &" &
# Bounded wait with 60s cap + pkill stuck ssh
end_time=$(($(date +%s) + 60))
while [ -n "$(jobs -p)" ] && [ "$(date +%s)" -lt "$end_time" ]; do
    sleep 2
done
[ -n "$(jobs -p)" ] && kill $(jobs -p) 2>/dev/null
wait 2>/dev/null || true
```

**Rule**: For any backgrounded SSH-spawning-process pattern, use `< /dev/null`, explicit `timeout`, and bounded `wait`. Never trust `nohup &` to release SSH cleanly.

---

## Lesson 12: KV cache + in-flight requests bias new measurements

**Bug**: After killing a benchmark mid-run, the next benchmark's first ~30s saw stale requests draining + KV cache from old workload. Biases TTFT measurement.

**Caught by user** before bias propagated.

**Rule**: Before any fresh measurement after killing a previous benchmark:
1. Kill benchmark + scheduler
2. Wait for vLLM to drain: query `schedule_trace`, ensure `running=0, waiting=0` on all 12 nodes
3. Then relaunch scheduler + new benchmark

vLLM drains in ~30s naturally. Do NOT skip this drain check.

---

## Lesson 13: Capacity search seed matters — saves probes

**Bug**: Initial Qwen po2_oracle seed was 70 in script. Po2-est converged at 73.9. Oracle should start near 76 (cap_est + 2-3 typical). Walking from 70 wastes 4-5 probes (~20 min).

**Caught and intervened**: User flagged. Saved ~16 min by manual relaunch with seed=76.

**Rule**: For multi-scheduler capacity search, use the previous scheduler's converged capacity to inform the next scheduler's seed:
- Po2-oracle seed = Po2-est cap + 2-4
- Llumnix seed = Po2-est cap - 2-5

---

## Lesson 14: Tokenizer-dependent dataset filtering (not a bug)

**Observation**: Qwen-tokenized sharegpt yielded 9963/10000 valid samples consistently across all schedulers. ~37 requests filtered by Qwen tokenizer (likely length out-of-bounds).

**Not a bug** — uniform across all 3 schedulers (Po2-est, Po2-oracle, Llumnix) → comparison stays fair. Worth noting in paper to preempt reviewer questions.

---

## Lesson 15: Cron auto-launchers must be tightly scoped

**Bug**: State-machine cron `dd78768c` was set to auto-advance through Qwen → CPU tracker → Po4+Po8. After I manually killed Qwen for re-seed intervention, cron immediately relaunched it.

**Fix**: Deleted the cron via `CronDelete` before manual intervention.

**Rule**: When intervening manually mid-pipeline, **disable the auto-progression cron first**. State machines must check more conditions than just "X not running" — e.g., a sentinel file that indicates manual takeover is active.

---

## Aggregate root-cause patterns

1. **Silent degradation > loud failure**: Most painful bugs were silent — RESTART_VLLM=false, oracle lengths, fixed output dir, 15/16 predictors. The system kept producing data that LOOKED valid but was wrong. Fail-loud is much cheaper than recover-from-silent-corruption.

2. **Pipeline gap blindness**: When a feature spans predictor → scheduler → benchmark → NPZ → analysis (CPU tracking), every layer must be wired. Patching one layer in isolation produces NPZ that looks complete but is empty.

3. **Single-source-of-truth violations**: Multiple variants of the same script (`_fixed`, `_fanout_est`, custom one-offs) caused confusion about which is canonical. Should have quarantined or deleted broken variants immediately.

4. **Race conditions hide under "usually works"**: Concurrent predictor launches mostly worked but had ~8% failure rate. Sequential deploy was the deterministic fix.

5. **Time pressure → corner cutting → bigger waste**: Every "let me skip the smoke test / verify / drain" attempt cost more than the skipped step would have.

---

## Process improvements that worked

- **Hard gates with abort**: `verify_predictors.sh` blocking subsequent benchmarks until 192/192 confirmed.
- **Fresh-deploy-per-cell** for absolute comparisons.
- **Per-cell rsync** so partial sweeps preserve completed cells.
- **9.X / 10.X early-stop bands** (TTFT P99 within ±1s of SLO) for capacity float refinement — saves probes vs exhaustive bisect.
- **State-machine cron** for autonomous overnight progression — but with manual disable when intervening.
- **Two-location final sync** (`experiment_results_a30/` + `experiment_results_a30_final_sync/`) for redundancy before lease release.
- **md5 content verification** post-sync to confirm bit-exact local copy.
