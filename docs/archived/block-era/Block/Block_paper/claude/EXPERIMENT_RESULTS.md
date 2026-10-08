# Block vs Llumnix Comprehensive Experiment Results

*Last Updated: February 2, 2026*

This document contains the complete experimental results comparing Block and Llumnix scheduling systems on a 2-node A100-40GB cluster.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Experimental Setup](#experimental-setup)
3. [Full Results Tables](#full-results-tables)
4. [Head-to-Head Comparison](#head-to-head-comparison)
5. [Key Findings](#key-findings)
6. [Data Locations](#data-locations)
7. [Reproduction Guide](#reproduction-guide)

---

## Executive Summary

### Three Configurations Tested

| Config | Attention Backend | Chunked Prefill | Migration | Notes |
|--------|------------------|-----------------|-----------|-------|
| **Block 16pred** | FlashInfer | Yes (512) | N/A | Predictive scheduling |
| **Llumnix Scheduling Only** | xFormers | Yes (512) | No | Best Llumnix config |
| **Llumnix Migration** | FlashInfer | No | Yes (rayrpc) | Migration enabled |

### Block vs Llumnix Scheduling Only at QPS=28

| Metric | Block | Llumnix Sched Only | Block Advantage |
|--------|-------|-------------------|-----------------||
| **Throughput** | 15,009 tok/s | 14,872 tok/s | +0.9% |
| **Mean E2E Latency** | 6,599 ms | 6,423 ms | Similar |
| **Mean Token Latency** | 20.97 ms | 20.40 ms | Similar |

**Conclusion**: Block and Llumnix Scheduling Only achieve comparable performance at all QPS levels. Both systems scale well to QPS=36 (~19k tok/s) with similar latency.

---

## Experimental Setup

### Hardware
- **Cluster**: 2x CloudLab d8545 nodes
- **GPUs**: 8x NVIDIA A100-40GB (4 per node)
- **Network**: InfiniBand interconnect

### Software Configuration

| Parameter | Block 16pred | Llumnix Sched Only | Llumnix Migration |
|-----------|-------------|-------------------|-------------------|
| Framework | vLLM 0.5.4 | vLLM 0.6.3.post1 | vLLM 0.6.3.post1 |
| Model | Llama-2-7B | Llama-2-7B | Llama-2-7B |
| Instances | 8 (TP=1) | 8 (TP=1) | 8 (TP=1) |
| Batch Size | 96 | 96 | 96 |
| Max Seq Length | 4096 | 4096 | 4096 |
| Chunked Prefill | Yes (512) | Yes (512) | **No** |
| Attention Backend | FlashInfer | xFormers | FlashInfer |
| Migration | N/A | Disabled | **Enabled** |

### Benchmark Parameters
- **Dataset**: ShareGPT
- **Requests**: 10,000 per experiment
- **QPS Values**: 16, 20, 24, 28, 32, 36
- **Distribution**: Gamma (burstiness=1.0)
- **Request Length**: Max 4096 tokens

---

## Full Results Tables

### Block 16pred Results

| QPS | Throughput (tok/s) | Duration (s) | Mean E2E (ms) | Mean Token (ms) | Sched Overhead (ms) |
|-----|-------------------|--------------|---------------|-----------------|---------------------|
| 16 | 8,874 | 642.04 | 5,299 | 16.77 | 46.25 |
| 20 | 10,986 | 518.62 | 5,647 | 17.94 | 49.76 |
| 24 | 12,965 | 439.43 | 6,024 | 19.15 | 55.88 |
| 28 | 15,009 | 379.61 | 6,599 | 20.97 | 62.89 |
| 32 | 16,784 | 339.47 | 7,048 | 22.50 | 73.37 |
| 36 | 18,798 | 303.09 | 7,505 | 24.03 | 85.10 |

### Llumnix Scheduling Only Results (xFormers + Chunked Prefill, No Migration)

| QPS | Throughput (tok/s) | Duration (s) | Mean E2E (ms) | Mean Token (ms) | Sched Overhead (ms) |
|-----|-------------------|--------------|---------------|-----------------|---------------------|
| 16 | 8,856 | 643.35 | 5,240 | 16.62 | 28.49 |
| 20 | 10,940 | 520.77 | 5,584 | 17.73 | 31.44 |
| 24 | 13,010 | 437.93 | 5,971 | 18.94 | 32.63 |
| 28 | 14,872 | 383.09 | 6,423 | 20.40 | 34.69 |
| 32 | 16,971 | 335.72 | 6,944 | 22.08 | 38.12 |
| 36 | 18,843 | 302.37 | 7,458 | 23.72 | 42.00 |

### Llumnix Migration + FlashInfer Results (No Chunked Prefill)

| QPS | Throughput (tok/s) | Duration (s) | Mean E2E (ms) | Mean Token (ms) | Sched Overhead (ms) |
|-----|-------------------|--------------|---------------|-----------------|---------------------|
| 16 | 8,886 | 641.15 | 5,578 | 17.80 | 5.58 |
| 20 | 10,973 | 519.21 | 6,013 | 19.52 | 22.62 |
| 24 | 13,023 | 437.49 | 6,618 | 23.08 | 55.99 |
| 28 | 15,047 | 378.65 | 7,635 | 30.15 | 42.99 |
| 32 | 14,278 | 399.04 | **47,305** | **466.85** | 74.33 |
| 36 | 12,936 | 440.42 | **89,027** | **919.84** | 104.25 |

---

## Head-to-Head Comparison

### Throughput Comparison (tokens/s)

```
QPS    | Block 16pred | Llumnix Sched Only | Llumnix Migration
-------|--------------|--------------------|-----------------
16     |    8,874     |      8,856         |      8,886
20     |   10,986     |     10,940         |     10,973
24     |   12,965     |     13,010         |     13,023
28     |   15,009     |     14,872         |     15,047
32     |   16,784     |     16,971         |     14,278  <-- DEGRADED
36     |   18,798     |     18,843         |     12,936  <-- DEGRADED
```

### Mean E2E Latency Comparison (ms) - Lower is Better

```
QPS    | Block 16pred | Llumnix Sched Only | Llumnix Migration
-------|--------------|--------------------|-----------------
16     |    5,299     |      5,240         |      5,578
20     |    5,647     |      5,584         |      6,013
24     |    6,024     |      5,971         |      6,618
28     |    6,599     |      6,423         |      7,635
32     |    7,048     |      6,944         |     47,305  <-- EXPLODED
36     |    7,505     |      7,458         |     89,027  <-- EXPLODED
```

### Mean Token Latency Comparison (ms) - Lower is Better

```
QPS    | Block 16pred | Llumnix Sched Only | Llumnix Migration
-------|--------------|--------------------|-----------------
16     |   16.77      |     16.62          |     17.80
20     |   17.94      |     17.73          |     19.52
24     |   19.15      |     18.94          |     23.08
28     |   20.97      |     20.40          |     30.15
32     |   22.50      |     22.08          |    466.85  <-- EXPLODED
36     |   24.03      |     23.72          |    919.84  <-- EXPLODED
```

---

## Key Findings

### 1. Block and Llumnix Scheduling Only Have Comparable Performance

At all QPS levels (16-36), both Block 16pred and Llumnix Scheduling Only achieve:
- Similar throughput (within 1-2%)
- Similar latency (within 5%)
- Both scale smoothly to QPS=36 (~19k tok/s)

### 2. Chunked Prefill is Critical for High-Load Performance

Llumnix Migration (FlashInfer, NO chunked prefill):
- Performs well at QPS 16-28
- **Collapses at QPS 32+** with latency explosion (7x-12x increase)
- Throughput degrades instead of improving

This demonstrates that **chunked prefill is more important than migration** for handling high load.

### 3. Attention Backend Difference is Minor

- Block uses FlashInfer, Llumnix Scheduling Only uses xFormers
- Performance difference is negligible (<2%)
- Both backends work well with chunked prefill

### 4. Migration Without Chunked Prefill is Harmful at High Load

The migration experiment shows that:
- Migration overhead increases latency
- Without chunked prefill, long prefills block the system
- At high QPS, the system becomes overloaded and latency explodes

---

## Data Locations

### Raw Results
```
Block 16pred:           experiment_output/benchmark_output/block_16pred/block_16pred_qps*/
Llumnix Scheduling Only: experiment_output/benchmark_output/llumnix_scheduling_only/batch96_qps*/
Llumnix Migration:      experiment_output/benchmark_output/llumnix_migration_flashinfer/qps*/
```

### Key Files Per Experiment
```
<experiment_dir>/
  ├── *_logs.txt           # Human-readable results summary
  └── *_all_metrics.npz    # Full metrics (NumPy format)
```

### NPZ File Contents
- `e2e_latencies`: End-to-end latency per request
- `token_latencies`: Per-token latency
- `prefill_latencies`: First token latency
- `response_lens`: Response length per request
- `prompt_lens`: Prompt length per request

---

## Reproduction Guide

### Quick Start

1. **Deploy Block**:
   ```bash
   ./block/exp/end_to_end_exp_scripts/a100_supplementary/deploy_block.sh start
   ```

2. **Run Block benchmark**:
   ```bash
   ./block/exp/end_to_end_exp_scripts/a100_supplementary/run_benchmark.sh block 28 10000
   ```

3. **Stop Block, Deploy Llumnix**:
   ```bash
   ./block/exp/end_to_end_exp_scripts/a100_supplementary/deploy_block.sh stop
   ./block/exp/end_to_end_exp_scripts/a100_supplementary/deploy_llumnix.sh start
   ```

4. **Run Llumnix benchmark**:
   ```bash
   ./block/exp/end_to_end_exp_scripts/a100_supplementary/run_benchmark.sh llumnix 28 10000
   ```

### Full Comparison
```bash
./block/exp/end_to_end_exp_scripts/a100_supplementary/full_comparison.sh both
```

### Scripts Reference
| Script | Purpose |
|--------|---------|
| `deploy_block.sh` | Deploy Block (vLLM + predictors + scheduler) |
| `deploy_llumnix.sh` | Deploy Llumnix (Ray cluster + Llumnix server) |
| `run_benchmark.sh` | Run benchmarks for either system |
| `full_comparison.sh` | Full end-to-end comparison experiment |

See [LLUMNIX_COMPARISON_GUIDE.md](LLUMNIX_COMPARISON_GUIDE.md) for detailed deployment instructions.

---

*Document location: Block_paper/claude/EXPERIMENT_RESULTS.md*
