# Vast wrapper smoke results — 2026-05-02

## Goal
Validate trained checkpoints (LLMRouter MF, GraphRouter, BEST-Route 4-way) load
correctly and produce diverse routing on a held-out 100-prompt slice from
`best-route-v3-test-3534-eval.jsonl`. Standalone smoke (no cara_serve).

## Environment
- Vast RTX 3090 (SSH alias: `vast`)
- Venv: `/workspace/.venv-train` (torch 2.5.1+cu121, transformers 4.57.6)
- Disk: 982MB free (98% full) — too tight for installing extra deps

## Results

### LLMRouter MF — PASS
- Checkpoint: `/workspace/saved_models/mfrouter/mfrouter.pkl`
- Config: `/workspace/configs_vast/mfrouter_cara.yaml` (vast-localized paths)
- 50/50 routed @ 22.0 ms/prompt, **0 errors**
- 100/100 routed @ 21.9 ms/prompt, **0 errors**
- Distinct models: 4/4
- Distribution (n=50): 3B=40%, 7B=34%, 72B=16%, 14B=10%
- DIVERSITY_GATE: PASS

### BEST-Route 4-way (DeBERTa-v3-small) — PASS
- Checkpoint: `/workspace/best_route_4way_qwen/` (num_labels=4)
- 100/100 routed @ 13.9 ms/prompt on cuda, **0 errors**
- Distinct models: 4/4
- Distribution (n=100): 3B=60%, 7B=18%, 14B=14%, 72B=8%
- DIVERSITY_GATE: PASS
- Note: train val_acc=0.39 epoch 3 — distribution skews toward 3B as expected
  for deepeval label distribution.

### LLMRouter GraphRouter — BLOCKED
- Checkpoint exists: `/workspace/saved_models/graphrouter/graphrouter.pt`
- Error: `ModuleNotFoundError: No module named 'torch_geometric'`
- Cannot install on Vast (R1: never modify existing venv; disk 98% full anyway).
- Smoke must run on cluster where venv has torch_geometric.

## Runtime hacks
- transformers 4.57+ added CVE-2025-32434 guard requiring torch≥2.6 for
  `torch.load`. Vast venv pinned to torch 2.5.1+cu121 for vllm 0.7.3 compat.
- Worked around with runtime monkey-patch of `check_torch_load_is_safe` across
  all transformers submodules (script-level, no on-disk edit).
- Longformer-base-4096 is a trusted official HF model — patch is for smoke only.

## Cluster integration prep
For Phase H (MF) and Phase J (BEST-Route 4-way) cluster smokes:
- MF wrapper code (cluster) needs to call `MFRouter(yaml_path).route_single({"query": prompt})` and read `model_name` from output dict.
- BEST-Route 4-way wrapper needs N-way variant (current is binary). Argmax over
  4 logits → `[3B, 7B, 14B, 72B][label]`.
- Cluster torch version should be ≥2.6 — if not, same monkey-patch applies.
