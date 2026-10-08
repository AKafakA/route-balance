# A100 Experiment Log — SoCC 2026 Camera-Ready

**Cluster**: 2x CloudLab d8545 (4x A100-SXM4-40GB each, 8 GPUs total)
**Nodes**: `d8545-10s10301.wisc.cloudlab.us` (node0), `d8545-10s10305.wisc.cloudlab.us` (node1)
**Model**: meta-llama/Llama-2-7b-hf, TP=1, 8 instances (4 per node)
**Dataset**: ShareGPT (10,000 requests per QPS level)

---

## ~~Phases 5.1-5.4: Block Experiments~~ — INVALIDATED, DATA DELETED

**Root cause**: run_benchmark.sh missing `--use_estimated_response_lens`. All Block runs used oracle lengths. Results invalid for paper comparison.

---

## Phase 5.5-5.6: Llumnix with Migration — VALID ✓

**Config**: 8 instances, max_num_seqs=96, migration_backend=rayrpc, FlashInfer
**Capacity**: ~19 QPS (collapses at 20+)
**Files**: `experiment_results_a100/llumnix_sweep/benchmark_llumnix_qps{16..36}.log`

---

## Phase 5.7: Block Re-run — PENDING

Must use: `--use_estimated_response_lens`, deploy_block.sh (4 pred/inst), RESTART_VLLM=true equivalent.
Node1 rebooting. ~20 hours remaining.
