# A100 Phase 5.7 Runbook — Block Fill Sweep

## Goal
Run all 4 Block configs at 15 missing QPS levels (17,18,19,21,22,23,25,26,27,29,30,31,33,34,35).
QPS 16,20,24,28,32,36 already collected in Phases 5.1-5.4.

## Nodes
- node0: `asdwb@d8545-10s10301.wisc.cloudlab.us` (internal: 10.10.1.1)
- node1: `asdwb@d8545-10s10305.wisc.cloudlab.us` (internal: 10.10.1.2)

## Step 1: Setup (after reload)
```bash
# From local machine
sh block/exp/end_to_end_exp_scripts/a100_supplementary/setup_a100.sh
# OR if setup_a100.sh doesn't exist, run setup.sh on each node:
# scp block/exp/setup.sh asdwb@d8545-10s10301.wisc.cloudlab.us:~/
# ssh asdwb@d8545-10s10301.wisc.cloudlab.us "bash ~/setup.sh"
# (repeat for node1)
```

## Step 2: Deploy Block (chunked prefill ON — for configs 1 & 3)
```bash
sh block/exp/end_to_end_exp_scripts/a100_supplementary/deploy_block.sh start
```
This deploys: 8 vLLM instances (with --enable-chunked-prefill), 32 predictors (with --enable_chunked_prefill), global scheduler on port 8200.

## Step 3: Run Config 1 — Block-Po2+CP (N=2)
The scheduler is already set with `num_query_predictor=8` (queries all 8).
Need to change to N=2 for Po2. Either:
- Restart scheduler with `--num_query_predictor 2`
- OR use the benchmark script which handles N via the global scheduler

Actually, the deploy_block.sh sets `NUM_INSTANCES=8` for `num_query_predictor`.
For Po2 (N=2), need scheduler with `--num_query_predictor 2`.
For Fanout (N=8), need scheduler with `--num_query_predictor 8`.

### Scheduler restart for different N:
```bash
# Kill old scheduler
ssh asdwb@d8545-10s10301.wisc.cloudlab.us "pkill -f global_scheduler"
sleep 3

# Start with N=2 (Po2)
ssh asdwb@d8545-10s10301.wisc.cloudlab.us "cd ~/Block && export PYTHONPATH=. && \
    nohup python block/global_scheduler/api_server.py \
        --config_path block/config/a100_8x7b_host_configs.json \
        --metrics_type min_new_request_latency \
        --num_query_predictor 2 \
        --num_required_predictor 1 \
        --workers 1 \
        --num_predictor_ports 4 \
        --profiling_sampling_rate 0.0 \
        --predictor_timeout 2000 \
        --backend_timeout 3600 \
        --initial_available_instance 8 \
        --max_slo_in_seconds 0 \
        > experiment_output/logs/global_scheduler.log 2>&1 &"
```

### Run benchmark sweep:
```bash
sh block/exp/end_to_end_exp_scripts/a100_supplementary/run_benchmark.sh sweep block 10000 "17 18 19 21 22 23 25 26 27 29 30 31 33 34 35"
```
Save results to: `experiment_output/benchmark_output/block_po2_chunked_fill/`

## Step 4: Config 2 — Block-Po2 no CP (N=2)
Need to restart vLLM WITHOUT chunked prefill:
```bash
# Kill vLLM on both nodes
ssh asdwb@d8545-10s10301.wisc.cloudlab.us "pkill -f 'vllm.entrypoints.api_server'"
ssh asdwb@d8545-10s10305.wisc.cloudlab.us "pkill -f 'vllm.entrypoints.api_server'"
sleep 5

# Restart vLLM WITHOUT --enable-chunked-prefill on both nodes
for host in asdwb@d8545-10s10301.wisc.cloudlab.us asdwb@d8545-10s10305.wisc.cloudlab.us; do
    ssh $host "cd ~/Block && \
        export HF_HOME=/mydata/huggingface && \
        export HF_TOKEN=\$(cat ~/.hf_token) && \
        export VLLM_USE_V1=0 && \
        for gpu in 0 1 2 3; do \
            port=\$((8000 + gpu)); \
            CUDA_VISIBLE_DEVICES=\$gpu nohup python -m vllm.entrypoints.api_server \
                --model meta-llama/Llama-2-7b-hf \
                --port \$port \
                --max-num-seqs 96 \
                --max-model-len 4096 \
                > experiment_output/logs/vllm_gpu\${gpu}.log 2>&1 & \
        done"
done
```
Also restart predictors WITHOUT --enable_chunked_prefill (critical — Phase 5.2 had a mismatch bug).
Scheduler stays N=2.

## Step 5: Config 3 — Block-Fanout+CP (N=8)
Restart vLLM WITH chunked prefill (same as Step 2).
Restart scheduler with `--num_query_predictor 8`.
Restart predictors WITH --enable_chunked_prefill.

## Step 6: Config 4 — Block-Fanout no CP (N=8)
Restart vLLM WITHOUT chunked prefill.
Scheduler stays N=8.
Restart predictors WITHOUT --enable_chunked_prefill.

## Critical Rules
- profiling_sampling_rate=0.0 ALWAYS
- VLLM_USE_V1=0 ALWAYS
- batch_cap=96 (max_num_seqs) for A100
- Deploy predictors in batches (max 8 concurrent startups, sleep 10s)
- Warmup before benchmarks
- Predictor CP flag MUST match vLLM CP setting (Phase 5.2 bug)

## Results Location
- Local: `experiment_results_a100/{block_po2_chunked_fill,block_po2_no_chunked_fill,block_fanout_chunked_fill,block_fanout_no_chunked_fill}/`
