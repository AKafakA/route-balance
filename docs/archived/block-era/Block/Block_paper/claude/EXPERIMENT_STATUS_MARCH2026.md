# Block Experiment Status — March 2026

## Overview
Running Block Po2 (Power-of-Two, N=2) experiments to reframe Po2 as default scheduling strategy.
Working repo: `/home/wd312/Code/llm/Block_exp/` (GitHub: AKafakA/Block, main branch)

## Completed Experiments (A30, March 19)

### Po2 Oracle Full QPS Sweep (ShareGPT / Llama-7B)
- 17 QPS levels (20-36), all 10000/10000 success
- **Capacity: 31.2 QPS** (vs Block* broadcasting 32.0, Block broadcasting 31.1)
- Results: `experiment_results_a30/po2_len_oracle/`

### Po2 Estimated
- QPS=30: TTFT P99 = 2,820ms | QPS=32: TTFT P99 = 11,248ms
- Results: `experiment_results_a30/po2_estimated/`

### Llumnix-- Comparison (correct min_lunmnix_load predictors)
- QPS=30: TTFT P99 = 1,171ms | QPS=32: TTFT P99 = 13,797ms
- Results: `experiment_results_a30/llumnix_verify/`

### Burstiness (Block Po2 N=2, QPS=30)
- 4 levels (0.25, 0.5, 1.0, 2.0), all 100% success
- Results: `experiment_results_a30/burstiness_po2/`

### Error Heatmap (Block Po2 N=2, QPS=30)
- 4×4 grid (0%, 25%, 50%, 100%), all 100% success
- Results: `experiment_results_a30/error_heatmap_po2/`

### Generality Capacity Search
| Config | Po2 Oracle (N=2) | Block* Oracle (N=12) | Po2 Estimated (N=2) | Block Estimated (N=12) | Llumnix-- |
|--------|-----------------|---------------------|--------------------|-----------------------|-----------|
| ShareGPT | **31.2** | 32.0 | ~30 | 31.1 | 30.6 |
| BurstGPT | **60.1** | 59.0 | N/A | / | 55.1 |
| batch_cap=24 | — | 27.9 | **27.2** | 27.2 | 23.9 |
| chunk2048 | — | 31.5 | needs rerun | 30.8 | 29.8 |
| Qwen2-7B | — | 68.3 | pending | 67.9 | 62.0 |

## A100 Status
- Setup and scripts verified (100/100 smoke test passed, 0 errors)
- Full experiments not completed — need separate reservation
- All fixes pushed to GitHub

## Remaining for April 16-20 (A30, 5 days)
1. **Full main sweep**: all schedulers × QPS 20-36 (including float capacity)
2. **N ablation**: N=2,4,6,8 estimated (new study)
3. **Generality**: chunk2048 rerun + Qwen capacity
4. **Llumnix-- burstiness**: redeploy with min_lunmnix_load predictors
5. **CPU overhead**: with --enable_cpu_tracking
6. Consider: increase SLO to show Po2 tail latency advantage

## Remaining A100
- 3 configs × 6 QPS: Po2 chunked, Po2 no-chunked, Broadcasting no-chunked
- Scripts ready and tested

## Infrastructure Fixes (all in GitHub)
See experiment_report_march19_block.md in memory for full list of 52 bugs found and fixed.
Key fixes: predictor path, subshell detach, port 8200 skip, profiling=0.0, predictor_timeout, transformers pin, warmup parameter order.
