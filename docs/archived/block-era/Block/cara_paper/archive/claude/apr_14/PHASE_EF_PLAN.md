# Phase E + F Execution Plan — Apr 14

## Phase E — P-D feasibility at the upper regime

**Goal:** extend F1/F4 to test P-D where KV transfer actually matters: longer prompts + higher concurrent load.

### E1. Long-prompt sweep
Test dataset maxes at 442 words; synthesize longer prompts by concatenating.
- Target buckets: ~500 / 1000 / 2000 / 4000 tokens
- N=15 prompts/bucket, 1 QPS serial
- Measure: `transfer_est_ms`, `prefill_ms`, `decode_ms` scaling
- **Expected:** transfer_est_ms grows linearly past ~100 ms at 2000 tokens, ~200 ms at 4000 tokens; prefill_ms grows super-linearly (attention is O(n²)); decode_ms stable

### E2. P-D under long-prompt concurrent load
Send 100 prompts of 1000+ tokens through P-D at 8 QPS (the 2×7B cluster can't handle monolithic 1000-token serving at 8 QPS, but P-D can pipeline).
- Measures feasibility of P-D absorbing the regime where regular vLLM would preempt + thrash.

**Duration:** ~5 min runtime. No cluster reconfig — uses existing P-D on node2/3.

## Phase F — Comprehensive regular-cluster pipeline

**Goal:** full coverage on the 4-node regular cluster (2×3B + 2×7B) with all fixes applied (B10 int-cast, B15 pkg pins, B11 launcher pattern, ...). Re-creates the paper-layout cluster and runs every experiment class.

### F1. Teardown P-D, relaunch regular cluster
- pkill vllm on node2/3 (prefill + decode) and cara_pd_serve
- Relaunch regular vllm on node2/3 with `max-model-len=8192, gpu-mem=0.9` (same as Day 1/2)
- Relaunch regular vllm on node0/1 with Qwen-2.5-3B (HF cache warm)
- Launch `cara_serve` on node0:8200 with `--scheduling cara` using `model_deployment_a30_smoketest.json`
- Verify all 4 backends + scheduler healthy

### F2. Comprehensive experiment matrix

| Experiment | QPS | n | notes |
|------|-----|----|-------|
| 5-filter sweep @ 4 QPS | 4 | 50×5 | re-run with all fixes (Day 1.4 redux) |
| 4-scheduler comparison @ 10 QPS | 10 | 200×4 | Day 2.4 redux |
| Cara QPS sweep | 2/4/8/12/16/20 | 50/50/100/100/200/200 | extends Day 2.2 with 6 points for finer curve |
| Broadcasting 2-config | — | 100×2 | Day 2.1 redux |

Run serial via `scripts/a30_comprehensive_pipeline.sh` (new). Each phase uses the remote-launcher pattern (B11 fix).

### F3. Consolidated analysis
Rerun `analyze_results.py` + `analyze_sched_overhead.py` to pick up the new runs. Should land at ~50 total runs consolidated.

**Duration:** ~30-45 min (most time is scheduler relaunches for 4-scheduler comparison).

## Execution order
1. E1 (long-prompt sweep) — 4 buckets, <5 min
2. E2 (P-D under concurrent long-prompt load) — 2 min
3. Teardown + relaunch regular (F1) — 5-10 min depending on vLLM startup
4. F2 experiments — ~30 min
5. F3 analysis + update APR14_EXTENSIONS.md with F5/F6/F7 sections

Total: ~45-60 min wall clock.
