# 72B A100 teardown — 2026-06-09 (for gemma-4 alt-judge)
User authorized freeing A100s for the gemma-4-12B-it judge run.
## Deployed 72B (PRESERVED — redeployable):
- Model: Qwen/Qwen2.5-72B (public HF), precision fp16, backend vllm
- vllm_params: tensor-parallel-size=4, port=7000, max-model-len=8192, gpu-memory-utilization=0.95, enforce-eager=true
- node_hosts: d8545-10s10301.wisc.cloudlab.us, d8545-10s10305.wisc.cloudlab.us
- Source of truth: block/config/cara/model_deployment.json (in git) -> redeploy via cara serving stack.
- Node4 GPU worker pids at teardown: 8037-8040 (VLLM::Worker_TP0-3, 40GB each).
## Teardown executed 2026-06-09 (user: "free both A100 nodes"):
- node4 (d8545-10s10301): killed 72B main pid 7804 + workers 8037-8040 -> 4x A100 free (1 MiB). ✓
- node5 (d8545-10s10305, internal "node1"): killed 72B workers 4214-4217 -> 4x A100 free (1 MiB). ✓
- Total 8x A100-40GB free; both nodes /mydata=1.5T (1.3T free).
## Gemma-4 judge: NEW venv /mydata/gemma_judge/venv (R1; system py3.10 = old cara-fork vLLM, no gemma-4).
  install w/ TMPDIR=/mydata (root 92% full -> Errno 28 first try). Serve google/gemma-4-12B-it (UNGATED) bf16; key deepeval-gemma4-12b-it_reference.
## REDEPLOY 72B: cara serving stack + model_deployment.json (TP=4, port 7000, both nodes).

## Reason: all v2 experiments COMPLETE; A100s repurposed for gemma-4-12B-it bf16 judge (full judge-alt scoring).
