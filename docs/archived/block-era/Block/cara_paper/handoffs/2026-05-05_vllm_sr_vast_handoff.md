# vLLM-SR Vast Feasibility — Handoff for parallel session

**Date**: 2026-05-05 17:00 UTC
**Author**: main session (now refocused on CloudLab T2/T3/T4 monitoring)
**Reader**: parallel session continuing this debug

## TL;DR — what you're solving

**Goal**: get vLLM-SR (`github.com/vllm-project/semantic-router`) producing **real per-prompt complexity-driven routing** on the Vast.ai test rig — i.e., the `complexity` signal with embedding-similarity to hard/easy candidate phrases, not category-conditional/static routing. Once working, port the recipe to CloudLab post-T4 to add a **vllm-sr column** to the NeurIPS paper main table.

**Symptom blocking us**: the runtime fails at startup with `ModelFactory not initialized` + `Failed to compute text hard embedding ... status: -1` for every candidate phrase. The complexity classifier degrades to a placeholder service that returns constant `confidence=0.5498` for all prompts — no per-prompt differentiation. **Reproduced on 2 independent hosts** (CloudLab via Docker image, Vast via extracted binary).

**Key uncertainty**: I'm ~50% confident it's an environment / system-library mismatch rather than a config error or upstream bug. **Eliminate user error first** before claiming upstream bug.

---

## Verified facts (do NOT redo)

1. **The ML primitives work in isolation.** `vast_classifier_test.py` (at `/root/vast_classifier_test.py` on Vast) loads the classifier + 3 embedders directly via `transformers` and shows real differentiation:
   - Intent classifier: 8/12 distinct categories on varied prompts ✓
   - mmbert-32k-yarn embedder: spread=0.324 ✓
   - **mmbert-embed-32k-2d-matryoshka** (the one bundled in the Docker image): **spread=0.343 (strongest!)** ✓
   - ModernBERT-base embedder: spread=0.138 ✓
   - All 4 prompt-difficulty test cases produce sensible cosine deltas.
   - **So the model files work — the bug is in vllm-sr's runtime layer, not the ML weights.**

2. **vllm-sr 0.3.0 is only on git main**, not pip. PyPI's released `vllm-sr 0.2.0` generates v0.1 schema (no `signals.complexity`, no `routing.signals/projections`). To get the v0.3 validator + complexity support:
   ```bash
   pip install --break-system-packages git+https://github.com/vllm-project/semantic-router.git@main#subdirectory=src/vllm-sr
   # → installs vllm-sr 0.3.0
   ```
   This is already done on Vast.

3. **Docker is unavailable inside the Vast container** (no `/var/run/docker.sock`, capabilities limited). `vllm-sr serve` requires Docker. Workaround: extract the published image's binary + cgo bindings via `skopeo copy docker://ghcr.io/vllm-project/semantic-router/vllm-sr:latest dir:/root/vsr_image`, then run the binary directly.

4. **The router auto-downloads its required models on first run** (mmbert-embed-32k-2d-matryoshka with ONNX layers 6/11/16/22 + safetensors), but the `complexity_candidates_preload_failed` error fires **before** the download completes, so it FAILS the first run. On second run with files present, it STILL fails identically — so it's not a download race.

5. **Schema migration done**: my v0.3 config validates cleanly with `vllm-sr validate`. Config at `/root/vllm_sr_test/config.yaml` on Vast, also at `/tmp/vllm_sr_config_v3_real.yaml` on the VPS.

## Vast environment

- **Host**: `vast` SSH alias (74.48.140.178:60877), root user.
- **Hardware**: 2× RTX 5070 Ti (16 GB each = 32 GB VRAM), 130 GB disk (29% used = 94 GB free), 125 GB RAM, Ubuntu 24.04, CUDA driver 580.126.09 (CUDA 13.1 likely).
- **Caveat**: it's a Docker container itself (`bd5f19117b38`); we cannot run Docker inside.
- **Time budget**: ~10h paid (started ~14:30 UTC May 5; runs until ~00:30 UTC May 6).

## File layout on Vast (already in place)

```
/root/vllm_sr_test/
├── config.yaml                            # v0.3 config (validates ✓), complexity-only (no classifier block)
├── config_v3_attempted.yaml.bak           # earlier draft (same content)
├── config_v01_init.yaml.bak               # auto-generated v0.1 sample (rejected by v0.3 runtime)
├── intent_clf/                            # llm-semantic-router/mmbert32k-intent-classifier-merged (1.2GB)
├── embedder/                              # llm-semantic-router/mmbert-32k-yarn (588M)
├── modernbert_base/                       # answerdotai/ModernBERT-base (571M)
└── mmbert_embed_2d/                       # llm-semantic-router/mmbert-embed-32k-2d-matryoshka (586M)

/root/vsr_image/                           # skopeo-pulled Docker image
└── extracted/
    ├── usr/local/bin/router               # 203MB Go binary
    ├── usr/local/lib/libcandle_semantic_router.so  # 16M (Rust, ONNX inference)
    ├── usr/local/lib/libml_semantic_router.so      # 674K
    ├── usr/local/lib/libnlp_binding.so             # 1.8M
    └── app/
        ├── config/
        │   └── knowledge_bases/           # KB manifests (router-defaults references these)
        ├── models/
        │   └── (intentionally NOT bundled — auto-downloaded at runtime)
        └── start-router.sh                # entrypoint

/tmp/vsr_run/                              # working dir last used
└── models/
    └── mmbert-embed-32k-2d-matryoshka/    # auto-downloaded by router on first run, 4.6GB
        ├── model.safetensors              # 586MB
        ├── tokenizer.json
        ├── config.json
        └── onnx/
            ├── layer-{6,11,16,22}/        # each has model.onnx + model.onnx.data + model_fa_fp16.onnx
            └── model_config.json

/root/semantic-router-src/                 # full git clone of upstream (depth=1)
└── src/semantic-router/                   # Go source (cmd/main.go, pkg/**)
└── candle-binding/                        # Rust source for libcandle_semantic_router

/root/vast_classifier_test.py              # standalone direct-transformers test (V1 PASS)
/usr/local/go/                             # Go 1.23.4 (installed earlier, but build target is 1.24.1 — minor mismatch)
/usr/local/bin/{vllm,vllm-sr}              # vllm 0.20.1 + vllm-sr 0.3.0 (from git main)
```

**Qwen models** (on Vast, ready for the eventual 2-tier serve test):
```
/workspace/.hf_home/hub/models--Qwen--Qwen2.5-3B-Instruct/   # 5.8G complete
/workspace/.hf_home/hub/models--Qwen--Qwen2.5-7B-Instruct/   # 9.7G partial — 7B download was killed; resume via `hf download Qwen/Qwen2.5-7B-Instruct` (~2-3 min more)
```

## How to launch the router (current best invocation)

```bash
# Stop any prior:
pkill -f /usr/local/bin/router
sleep 2

cd /tmp/vsr_run    # wd matters — router resolves `models/...` paths relative to cwd
LD_LIBRARY_PATH=/root/vsr_image/extracted/usr/local/lib \
  /root/vsr_image/extracted/usr/local/bin/router \
  -config /root/vllm_sr_test/config.yaml \
  -port 50051 \
  -api-port 8081 \
  > /tmp/router_run.log 2>&1 &

sleep 30
grep -iE 'complexity|embedding|ModelFactory|preload|status: -1|fail' /tmp/router_run.log | head -25
```

## The exact failure mode (verbatim from `/tmp/router_run2.log`)

```
{"msg":"embedding_models_init_started","mmbert_configured":true,"use_cpu":true,...}
{"msg":"Unified classifier auto-discovery failed: model validation failed: no valid models found (neither LoRA nor legacy)"}
{"msg":"knowledge_base_classifier_create_failed","knowledge_base":"mmlu_kb","error":"failed to load KB manifest from knowledge_bases/mmlu/: open /root/vllm_sr_test/knowledge_bases/mmlu/labels.json: no such file or directory"}
{"msg":"complexity_classifier_initialized","model_type":"mmbert","rules":1}
Error: ModelFactory not initialized                       <-- repeated 14 times for 10 candidates × maybe 1-2 retries
{"msg":"Failed to compute text hard embedding for candidate 'explain in detail with multiple steps and tradeoffs': failed to generate embedding with model type mmbert (status: -1)"}
... (one per candidate)
{"msg":"complexity_candidates_preload_failed","model_type":"mmbert"}
{"msg":"complexity_classifier_create_failed","stacktrace":"...buildComplexityClassifierOption..."}
{"msg":"Legacy classifier initialization failed: ..."}
{"msg":"No classifier initialized. Using placeholder service."}
...
{"msg":"startup_complete","embedding_ready":false,"decisions":"simple_route,complex_route,fallback_route"}
```

The router DOES start serving on the API/extproc ports — but **`embedding_ready: false`**, and any `/v1/chat/completions` request returns the default model (3B) with no real classification.

## Likely causes — diagnose in this order

### Hypothesis 1 (most likely): missing knowledge_base manifests cause init cascade

The log shows `knowledge_base_classifier_create_failed` for `mmlu_kb` and `privacy_kb` BEFORE `complexity_classifier_initialized`. Even though our config doesn't use KB signals, the runtime may treat KB-init failure as a hard error that taints downstream `ModelFactory`. Try:

```bash
mkdir -p /tmp/vsr_run/knowledge_bases
cp -r /root/vsr_image/extracted/app/config/knowledge_bases/* /tmp/vsr_run/knowledge_bases/ 2>/dev/null || true
ls /tmp/vsr_run/knowledge_bases/    # should show mmlu/, privacy/, etc.
```
Re-launch the router and see if `ModelFactory not initialized` disappears.

### Hypothesis 2: the `embedding_models` block isn't being auto-injected

`router-defaults.yaml` (inside Docker image at `/app/.vllm-sr/router-defaults.yaml`) has:
```yaml
embedding_models:
  mmbert_model_path: "models/mom-embedding-ultra"
  use_cpu: true
  hnsw_config: {model_type: mmbert, target_dimension: 768, target_layer: 22, ...}
```
But the user-config schema rejects `embedding_models` at top level (we tested). Maybe this block needs to be merged into the runtime config in a way I missed. Check upstream's config-loading code:

```bash
grep -rn 'embedding_models' /root/semantic-router-src/src/semantic-router/pkg/ | head
grep -rn 'router-defaults' /root/semantic-router-src/ | head
grep -rn 'ModelFactory' /root/semantic-router-src/ | head
```

If the runtime expects to find `router-defaults.yaml` next to the binary or at `/app/.vllm-sr/`, but doesn't, the embedding_models block is unset → factory init no-ops → all subsequent calls return -1.

### Hypothesis 3: cgo-bound Rust library can't dlopen its CUDA deps

`libcandle_semantic_router.so` was compiled inside the Docker image's exact env (Ubuntu + CUDA 12.x). Run:
```bash
ldd /root/vsr_image/extracted/usr/local/lib/libcandle_semantic_router.so 2>&1 | grep -E 'not found|=>'
ldd /root/vsr_image/extracted/usr/local/bin/router 2>&1 | grep -E 'not found|libcandle'
```
If anything says `not found`, the library failed to load and the symbol behind ModelFactory is null. Fix by installing missing libs (apt or downloading from canonical CUDA repo).

### Hypothesis 4: `use_cpu: true` is overridden somewhere, requiring CUDA that isn't compiled-in

The Rust binding's default feature is `cuda`. If the bundled .so only has CUDA codepath and we're trying CPU init, init may fail. Try:
```bash
# strace to see what the lib tries to load
strace -e trace=openat -f -o /tmp/strace.log \
  bash -c 'cd /tmp/vsr_run; LD_LIBRARY_PATH=/root/vsr_image/extracted/usr/local/lib /root/vsr_image/extracted/usr/local/bin/router -config /root/vllm_sr_test/config.yaml -port 50053 -api-port 8082' &
sleep 20; pkill router
grep -E 'mmbert|onnx|cuda|libnvinfer' /tmp/strace.log | head -30
```

### Hypothesis 5: my config is wrong despite passing `vllm-sr validate`

Try the upstream's published "balance" recipe verbatim:
```bash
cd /tmp/vsr_run
cp /root/semantic-router-src/deploy/recipes/balance.yaml /tmp/vsr_run/balance.yaml
# Edit only providers.models[].backend_refs[].endpoint to point at localhost:8001 / 8002
# Leave signals/projections/decisions UNCHANGED
LD_LIBRARY_PATH=... /root/vsr_image/extracted/usr/local/bin/router -config /tmp/vsr_run/balance.yaml ...
```
If `balance.yaml` ALSO fails with `ModelFactory not initialized`, the bug is upstream / environmental — NOT my config.

## Final target / success criterion

Real per-prompt routing observable in HTTP response headers:
```
x-vsr-selected-confidence: <varies per prompt>          ← NOT constant 0.5498
x-vsr-matched-complexity: prompt_complexity:{easy|medium|hard}   ← varies
x-vsr-matched-projection: tier_{simple|complex}                  ← varies
x-vsr-selected-model: Qwen/Qwen2.5-{3B|7B}-Instruct              ← splits across both
```

Test with 8 varied prompts (greeting / fact / math / coding / proof / chat / philosophy / health) — should produce ≥2 distinct tier-band assignments and 2 model selections.

Once that passes:
1. Document the config + launch sequence (post here)
2. Port to CloudLab post-T4 (the 4-tier expansion is mechanical: just add 14B + 72B endpoints + extend tier mapping)

## What to leave alone

- **CloudLab cluster**: T2/T3/T4 chained orchestrator running, ~03:00 UTC done. Don't touch cara_serve, sweep_t2_t2a_t2b.sh, sweep_t3_t4.sh.
- **The 105-cell overnight grid results** in `cara_paper/.../cara_final_e2e/{cara_weight_sweep,trained_router_baselines,dispatcher_only_baselines}/`. Already aggregated.
- **VPS cron**: `cron_overnight.sh` still disabled (commented out in crontab). DO NOT re-enable until all experiments done — its spec audit will fire pkill on benchmark_serving when deployment doesn't match the frozen spec.

## Useful files on VPS

- `/home/wd312/Code/llm/Block/cara_paper/handoffs/2026-04-28_handoff.md` — master ops log, append your findings
- `/tmp/vllm_sr_config_v3_real.yaml` — v0.3 config (mirror of Vast)
- `/tmp/envoy_qwen_v2.yaml` — 13-host Envoy template (for CloudLab port-over later)
- `/tmp/vast_classifier_test.py` — standalone V1 test (proves models work)

## Time budget guidance

- **30 min**: try Hypothesis 1 + 5 (knowledge_bases dir + balance.yaml) — most likely fixes.
- **+30 min**: Hypothesis 3 (ldd + strace) if 1+5 don't work.
- **+1h**: read upstream Go source for ModelFactory init path.
- **Hard cap**: ~2-3h total. Beyond that, diminishing returns vs paper deadline.

If you find the fix, message back to main session with: (a) the working config, (b) the launch invocation, (c) the smoke-test trace showing varied confidences. Main session will then port to CloudLab and add the vllm-sr cells to the paper.

If after 3h still blocked, document the precise failure mode + file an upstream issue. Defer vllm-sr to camera-ready.
