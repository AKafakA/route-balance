# CARA Experiment Playbook (P100 & A100)

This note captures the recent changes and the exact steps to reproduce our results across P100 and A100 nodes. It covers: dataset preprocessing with soft length control, benchmark runner updates, A100 deployment/environment fixes, and a concise command cheat sheet.

---

## 1) Dataset Preprocessing: Clamp + Soft Length Control

Tool: `block/benchmark/cara/tools/normalize_word_limits.py`

Key behavior
- Clamps explicit word-count phrases in prompts to fit a per-request token budget (preserves phrasing; only numbers change when needed).
- Appends a soft guidance when no explicit word counts exist using the template:
  - "Try to answer within {n} words instead."
- Computes the per-prompt budget so that `prompt_len + response_len ≤ 1024` (Qwen 2.5 setup), with tokenizer-aware refinement of the guidance overhead.

Defaults and knobs
- `--max-total-len 1024` (total = prompt + output) — default.
- `--output-cap 1024` — default upper bound considered.
- `--soft-clamp` — enable soft guidance append (applies only when no explicit counts are present).
- Tokenizer and TPW estimation:
  - `--tokenizer Qwen/Qwen2.5-3B-Instruct` by default.
  - `--infer-tokens-per-word` enabled by default; backs off to `--tokens-per-word` when tokenizer is not available.
- Rounding and stability:
  - `--soft-round 50` — down-round the soft budget to a "nice" number.
  - Overhead is inferred iteratively; no manual overhead flag needed.

Recommended usage
```bash
python block/benchmark/cara/tools/normalize_word_limits.py \
  --infile data/cara/best-route-extension.jsonl \
  --outfile data/cara/best-route-extension.clamped.jsonl \
  --soft-clamp
```
Notes
- For strict enforcement at runtime, we also cap `max_new_tokens = max(0, 1024 − prompt_len)` in the benchmark client (see Section 2).

---

## 2) Benchmark Runner Changes (Consistency + Readiness + Hard Cap)

Files
- `block/benchmark/cara/benchmark_serving.py`
- `block/benchmark/cara/dataset.py`

What changed
- Fast readiness for CARA: The ready check now polls `GET /health` (no generation) so it doesn’t block or 400 when length limits are large.
- Consistent custom dataset across nodes: A local `CaraCustomDataset` is always used for `--dataset-name custom` (JSONL). Identical behavior on A100 and P100.
- **Output-length setting (corrected 2026-05-06)**: bench inherits **vLLM upstream default `--custom-output-len 256`**. An earlier version of this doc claimed defaults were "aligned to 1024" — that change was aspirational and **never landed in the wrapper**; all paper runs (2026-04-15 → 2026-05-06) used 256. We retain 256 as the standard benchmark setting (it matches vLLM's published bench default) and frame paper results accordingly.
- Client-side hard cap: For each request, the client enforces `max_tokens = min(expected_output_len, max(0, max_total_len − prompt_len))`. With `custom_output_len=256` and `max_total_len=2048`, max_tokens=256 for typical prompts.

Implications
- A100 and P100 now sample and cap identically for custom JSONL datasets.
- The ready check is fast and independent of token lengths.

---

## 3) A100 Deployment/Environment Fixes

Symptom
- ImportError for CUDA libraries when running vLLM or benchmarks, e.g. `libcudnn.so.9` or `libcusparseLt.so.0` not found.

Root cause
- cuDNN/NCCL/cuSPARSELt installed via pip don’t live under `/usr/local/cuda`; you must add the Python site-packages paths to `LD_LIBRARY_PATH`.

Fixes we added
- Deployment scripts configure `CUDA_HOME`, `PATH`, and `LD_LIBRARY_PATH` to include CUDA libs and pip-installed NVIDIA libraries:
  - `block/exp/cara/deploy_cara.py` (for vLLM and Ollama branches)
  - `block/exp/cara/setup.sh` and `block/exp/setup.sh` (persistent shell env)

Manual one-liners (if needed in a fresh shell)
```bash
# Install wheels into the active Python env
python -m pip install -U nvidia-cudnn-cu12 nvidia-nccl-cu12 nvidia-cusparselt-cu12

# Find site-packages path
SITE=$(python -c "import site; s=getattr(site,'getsitepackages',lambda:[])() or []; print((s+[site.getusersitepackages()])[0])")

# Export runtime paths
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE/nvidia/cudnn/lib:$SITE/nvidia/nccl/lib:$SITE/nvidia/cusparselt/lib

# Optional: user site fallback
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/nccl/lib:$HOME/.local/lib/python3.10/site-packages/nvidia/cusparselt/lib

# Verify
python -c "import ctypes; ctypes.CDLL('libcudnn.so.9'); ctypes.CDLL('libcusparseLt.so.0'); print('CUDA libs OK')"
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Backend max length
- Our template sets vLLM `--max-model-len 1024` (see `block/config/cara/model_config_template.json`).
- You can raise this (e.g., 2048), but it increases KV cache and may reduce batch capacity.

---

## 4) Commands and Process (P100 & A100)

Assumptions
- CARA server running at `http://127.0.0.1:8200` (see Section 5 for broadcast mode / E2E).
- Custom dataset JSONL at `data/cara/best-route-extension.jsonl`.
- Defaults (corrected 2026-05-06): `custom_output_len=256` (vLLM upstream default, inherited unchanged), `max_total_len=2048`, `ignore_eos=False`. **All paper-corpus cells use these values.**

Plain vs Clamped runs (3B, A100/P100 the same)
```bash
# Plain
python block/benchmark/cara/benchmark_serving.py \
  --backend cara --base-url http://127.0.0.1:8200 \
  --dataset-name custom --dataset-path data/cara/best-route-extension.jsonl \
  --num-prompts 10 --request-rate 1.0 \
  --model Qwen/Qwen2.5-3B --tokenizer Qwen/Qwen2.5-3B --trust-remote-code \
  --save-detailed --save-result --result-dir cara_result

# Clamped (preprocess first with Section 1 tool)
python block/benchmark/cara/benchmark_serving.py \
  --backend cara --base-url http://127.0.0.1:8200 \
  --dataset-name custom --dataset-path data/cara/best-route-extension.clamped.jsonl \
  --num-prompts 10 --request-rate 1.0 \
  --model Qwen/Qwen2.5-3B --tokenizer Qwen/Qwen2.5-3B --trust-remote-code \
  --save-detailed --save-result --result-dir cara_result
```

72B example (A100) — longer contexts
```bash
python block/benchmark/cara/benchmark_serving.py \
  --backend cara --base-url http://127.0.0.1:8200 \
  --dataset-name custom --dataset-path data/cara/best-route-extension.jsonl \
  --num-prompts 10 --request-rate 1.0 \
  --model Qwen/Qwen2.5-72B --tokenizer Qwen/Qwen2.5-72B --trust-remote-code \
  --save-detailed --save-result --result-dir cara_result
```
Notes
- The client enforces `max_tokens ≤ (max_total_len − prompt_len)`. With default `max_total_len=1024`, this prevents 400s from the backend.
- Ready check uses `GET /health` for CARA; no delays from large test generations.

Optional flags
- `--ignore-eos` to avoid early stop and generate up to the cap.
- `--ready-check-timeout-sec 0` to skip readiness (useful in scripting).

---

## 5) CARA Server + Broadcast Mode (reference)

Start CARA server (single model; ready check will hit `/health`)
```bash
python -m block.global_scheduler.cara.cara_serve \
  --host 127.0.0.1 --port 8200 \
  --model_config_path block/config/cara/model_deployment.json \
  --host_config block/config/host_configs.json \
  --scheduling random
```

Broadcasting test (multi-model data collection)
- See `block/exp/cara/test_broadcasting.sh` for full flow.

---

## 6) Quick Troubleshooting

- vLLM not ready / long wait: make sure you pulled the readiness change (should call `/health`). Otherwise use `--ready-check-timeout-sec 0`.
- Backend rejections (HTTP 400): ensure `max_total_len` matches server-side `--max-model-len`. Client now caps per-request length automatically to avoid this.
- CUDA ImportErrors: verify cuDNN and cuSPARSELT in `LD_LIBRARY_PATH` (Section 3). Install NVIDIA cu12 wheels if missing.
- A100 vs P100 consistency: ensured by `CaraCustomDataset` (local JSONL loader), 1024 defaults, and per-request cap.

---

## 7) File Map (for code readers)

- Dataset preprocessing: `block/benchmark/cara/tools/normalize_word_limits.py`
- Local custom dataset (always used for `--dataset-name custom`): `block/benchmark/cara/dataset.py`
- Benchmark client (ready check, defaults, hard cap): `block/benchmark/cara/benchmark_serving.py`
- Deployment env fixes: `block/exp/cara/deploy_cara.py`, `block/exp/cara/setup.sh`, `block/exp/setup.sh`
- Server health endpoint: `block/global_scheduler/cara/cara_serve.py` (`/health`)

---

That’s it. Use the commands in Section 4 as your daily driver for consistent experiments across nodes.

