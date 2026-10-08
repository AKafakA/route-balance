# SPEC F — Environment, Deployment, Tests, Repo Scaffolding (port spec for the from-scratch rebuild)

Survey date 2026-09-24. Strictly read-only survey; nothing in any source tree was modified.

**Source abbreviations used below**

| tag | what | state surveyed |
|---|---|---|
| `V3` | `/home/wd312/Code/llm/Block`, branch `v3-campaign` | HEAD `984ea18` (block/ subtree git-clean) |
| `V3X` | `V3/cara_paper/handoffs/window_setup_work/v3-exp/` (the v3 reproduction bundle docs) | working copy (note: `env/ENVIRONMENT.md` has an UNCOMMITTED edit) |
| `WSW` | `V3/cara_paper/handoffs/window_setup_work/` (v3_scripts, setup_jul25.sh, ENVIRONMENT/) | working copy |
| `FORK` | `/home/wd312/Code/llm/vllm-cara` at `2894a98722733c76cdb549c8b5e5068594f12bef` (read via `git show`/`git grep` only) | upstream base = `70d5953f8` (upstream main, 2025-11-26) |
| `OLD` | previous public repo `/home/wd312/Code/llm/route-balance` HEAD `671db9f` (read via `git show` only; working tree intentionally empty) | 294 tracked files, 12 commits |

"The codename" = the internal codename of RouteBalance (the 4-letter word in `V3` paths). It must never
appear in the new repo; this spec writes it only when quoting a v3 path/identifier that the porter has to find.

---

## 0. Decisions the owner must make before/while porting (R3: nothing below is decided silently)

1. **Package/import name** for the new repo (OLD used `route_balance`; suggestion `routebalance`). Env-var prefix likewise (OLD `ROUTE_BALANCE_*`, V3 `<CODENAME>_*`).
2. **Python on the cluster**: reproduce v3 exactly (system `/usr/bin/python3` 3.10.12, `sudo pip`, no venv — `V3X/env/ENVIRONMENT.md:14`) or move to a venv (global R1 says new venv; v3 did not use one). Changing it is an execution difference and must be declared in the parity manifest (§6 FIRST RULE).
3. **vLLM fork reference**: pin by SHA `2894a98722733c76cdb549c8b5e5068594f12bef` only. The branch name `cara_v_11` contains the codename (a codename-free tag on `AKafakA/vllm` would be an owner action).
4. **Scheduler head placement** (open CR item: v3 ran the scheduler co-located with `14B_0` on a 16-core V100 node; see §2.1, §7-H3).
5. **Where the fake-backend / equivalence fixtures come from** (they must be generated from the V3 tree once; §5.4).
6. **dev-gpu-wd312 smoke deviations** (TRITON attention backend, co-located small tiers with reduced `gpu-memory-utilization`): these are smoke-only and need explicit approval (R7) and must never produce reported numbers.
7. **Fate of dead config keys** (`model_kv_config`, `vllm_params.port`, `vllm_params.tensor-parallel-size`, `embedding_model`): keep-as-inert (documented) vs delete. Wiring any of them up changes behaviour vs v3 (§7-H9).

---

## 1. Exact software environment of the v3 run

### 1.1 Platform (authoritative: captured live from lease `asdwb-312083`, 2026-08-04)

| item | value | source |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS (Jammy), kernel 5.15.0-186-generic | `V3X/env/ENVIRONMENT.md:12-13`, `V3X/env/head_snapshot.txt:5-8` |
| Python | **3.10.12, system `/usr/bin/python3.10`, `venv=/usr` (no venv)** | `ENVIRONMENT.md:14`, `head_snapshot.txt:2-4`, `serve_a100_snapshot.txt:3-5` |
| CUDA toolkit | 12.8.0 (`cuda-toolkit-12-8` from the NVIDIA local-installer deb `cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10`) at `/usr/local/cuda-12.8` | `ENVIRONMENT.md:15`, `WSW/setup_jul25.sh:45-49` |
| Driver | 570.86.10 (ampere: `nvidia-open`; volta: `cuda-drivers` + **reboot**, nouveau holds V100s) | `ENVIRONMENT.md:16`, `setup_jul25.sh:50-61`, `WSW/ENVIRONMENT/README.md` trap 5 |
| GPUs | V100-SXM2-16GB (c4130), A30 24GB (d7525), A100-SXM4-40GB (d8545) | snapshots; `head_snapshot.txt:11-12` lists only **2** V100 lines (docs elsewhere say V100x4 — verify, §7-H4) |
| torch wheel | `torch==2.9.0` from **PyPI** (no `+cu128` local suffix in the freeze; bundles CUDA 12.8 runtime `nvidia-cuda-runtime-cu12==12.8.90`) — `WSW/ENVIRONMENT/README.md` prints it as `2.9.0+cu128` | `head_pip_freeze.txt`, README verification gate |

### 1.2 Key package versions (identical on head + A30 + A100: `diff` of `head_pip_freeze.txt` vs `serve_a100_snapshot.txt` freeze section is empty; A100 vs A30 freeze identical)

| package | v3 version | where it is pinned authoritatively | stale/wrong pins elsewhere |
|---|---|---|---|
| vLLM | `-e git+https://github.com/AKafakA/vllm.git@2894a98722733c76cdb549c8b5e5068594f12bef#egg=vllm`; import version `0.1.dev11709+g2894a9872.precompiled` | `head_pip_freeze.txt` last-but-2 line; `setup_jul25.sh:73-75` (SHA checkout + `sudo VLLM_USE_PRECOMPILED=1 pip install --editable .`); version string `WSW/env_manifests/pipfreeze_node*.txt` | OLD `README.md:124` / `REPRODUCE.md:7` say "vLLM ≥ 0.6" (WRONG: upstream lacks `/instance_stats`) |
| torch / torchvision / torchaudio / triton | 2.9.0 / 0.24.0 / 2.9.0 / 3.5.0 | FORK `requirements/cuda.txt` (`torch==2.9.0`), `setup_jul25.sh:42` | V3 `requirements.txt:14` `torch==2.6.0` (stale); OLD `requirements-training.txt:6` `torch>=2.5.0` |
| transformers | 4.56.0 | `setup_jul25.sh:78` (user+system); FORK `requirements/common.txt` `>=4.56.0,<5` | V3 `requirements.txt:30` and global CLAUDE.md "Version Pins" `4.50.3` (applies to the OLD P100 fork only — `ENVIRONMENT.md:23-25`); OLD `requirements-training.txt:5` `5.4.0` (training env, incompatible with serving) |
| tokenizers / huggingface_hub / safetensors | 0.22.2 / 0.36.2 / 0.8.0 | freeze | — |
| numpy | 2.2.6 | `setup_jul25.sh:80` ("May-faithful" override) | V3 `requirements.txt:1` `numpy~=1.26.4` (never ran on the cluster, `setup_jul25.sh:79`) |
| scipy / scikit-learn / pandas / joblib | 1.13.1 / 1.5.2 / 2.2.3 / 1.5.3 | freeze | — |
| xgboost | 3.2.0 (latency heads, CPU) | `setup_jul25.sh:80` | OLD `requirements-training.txt:15` `>=2.0.0` |
| faiss-cpu | 1.13.2 (KNN index) | `setup_jul25.sh:80` | — |
| sentence-transformers | 5.4.1 (MiniLM embedder) | `setup_jul25.sh:80` | — |
| fastapi / starlette / uvicorn / uvloop / httptools | 0.141.1 / 1.3.1 / 0.32.1 / 0.22.1 / 0.8.0 | freeze (fastapi pulled by FORK `fastapi[standard]>=0.115.0`) | V3 `requirements.txt:16` `fastapi~=0.115.5` |
| pydantic / pydantic_core | 2.13.4 / 2.46.4 | freeze | — |
| aiohttp | 3.10.11 (scheduler→backend HTTP) | freeze | V3 `requirements.txt:19` `~=3.10.5` (compatible) |
| requests / orjson / psutil / PyYAML / tqdm | 2.32.5 / 3.11.9 / 5.9.8 / 6.0.3 / 4.67.3 | freeze | — |
| ray | 2.55.1 (installed, TP uses vLLM's default executor) | `setup_jul25.sh:80` | V3 `requirements.txt:11` `2.43.0` |
| flashinfer-python / xgrammar | 0.5.2 / 0.1.27 | FORK `requirements/cuda.txt`, `common.txt` | `xgrammar==0.1.16` was deleted from PyPI (`WSW/ENVIRONMENT/README.md` trap 2) |
| datasets / pyarrow | 4.8.5 / 24.0.0 | `setup_jul25.sh:80` | V3 `requirements.txt:29,31` `~=2.18.0` / `~=15.0.0` |
| cupy-cuda12x / opencv-python-headless | 14.0.1 / 4.13.0.92 | `setup_jul25.sh:80` | — |
| nvidia-ml-py (pynvml, monitor) | 13.610.43 | freeze | — |
| nvidia-nccl-cu12 / cudnn-cu12 / cublas-cu12 | 2.27.5 / 9.10.2.21 / 12.8.4.1 | freeze | — |
| dacite / jsonlines / ddsketch / fasteners / shapely | 1.9.2 / 4.0.0 / 3.0.1 / 0.18 / 2.1.1 | freeze / V3 `requirements.txt` | — |
| NOT installed in v3 | pytest, routellm, deepeval, nixl, onnxruntime, xformers, flash-attn (standalone) | absent from freeze | V3 `block/exp/<codename>/setup.sh:100-120` installs nixl/onnx — that script was NOT what built v3 |

Full lists (the real lock): `V3X/env/head_pip_freeze.txt` (232 lines), `serve_a100_snapshot.txt` / `serve_a30_snapshot.txt` (freeze sections). Copy them verbatim into the new repo as constraints files (§5.2).

### 1.3 How the v3 env was actually built (the canonical recipe is `WSW/setup_jul25.sh`, NOT `V3/block/exp/<codename>/setup.sh`)

`setup_jul25.sh` steps (line refs): S0 `/mydata` mkfs+mount (`:23-25`; d8545/d7525 `nvme0n1`, c4130 `sdb`; volta `/mydata` NOT in fstab, remounted after reboot `:60-61`); S1 apt full-upgrade (`:27-28`); S2 `python3-pip python3-venv ccache ninja-build cmake` (`:29-30`, ninja must come from APT); S3 pip upgrade; S4 rsync of the code tree from the VPS path + `pip install -r requirements.txt` (`:33-40`); S5 `torch==2.9.0 torchvision==0.24.0 dacite` (`:42`); S6 CUDA 12.8.0 toolkit (`:45-49`); S7 drivers (`:50-52`); S7.5 volta reboot + wait (`:53-61`); S8 `pip install --upgrade "ray[cgraph]"` (`:63`, unpinned — later overridden by S13.5); S9 `setup_env.sh` (bashrc CUDA/LD/PYTHONPATH/HF_TOKEN) (`:65`); S9.5 neutralise expired HF_TOKEN export + install stored token file (`:66-68`); S10 `nvidia-smi -mig 0`; S12 fork clone, `git checkout 2894a9872`, `sudo VLLM_USE_PRECOMPILED=1 pip install --editable .` (`:73-75`); S13 `sentence-transformers faiss-cpu` + `transformers==4.56.0` user AND system (`:76-78`); S13.5 numeric-stack overrides (`:79-81`); S14 verify-by-execution: SHA assert, imports, versions, per-node pip freeze (`:82-85`).

Order-dependence hazards: S4 installs the stale `requirements.txt` (torch 2.6, numpy 1.26, transformers 4.50.3...) and later steps overwrite them; the final env is only correct because S5/S12/S13/S13.5 run after S4. A rebuild must install from the frozen constraints in one resolution, not replay this sequence.

### 1.4 What dev-gpu-wd312 (1x Quadro RTX 8000 48 GB, Turing sm_75) can run for a smoke test

Known box facts (global CLAUDE.md; memory `reference_personal_gpu_vm.md`; vPipe memory `devgpu-sm75-deployment-state.md`): 79 GB root (~45 GB free), NFS home (kimana, shared with gxp-l40s-2), `/local/scratch/tmp/wd312/` 63 GB local SSD **that a reboot wiped once** (treat as ephemeral, R14 storage triage), start/stop via xo.cl.cam.ac.uk. Python version, driver version and CUDA toolkit presence on the box are NOT recorded anywhere I could read → must be probed (`python3 -V`, `nvidia-smi`, `nvcc -V`) before building. A vPipe sm75 venv lives under `/local/scratch/tmp/wd312/` — never touch it (R1; `readlink -f` before any delete, universal R14).

| item | sm_75 status | evidence / action |
|---|---|---|
| torch 2.9.0 (PyPI cu12.8) | supported arch | needs driver >= 570 for exact v3 parity; 525-569 only via CUDA minor-version compatibility (probe first) |
| vLLM fork, precompiled | Python-only diff vs upstream except `CMakeLists.txt` arch lists (adds 6.0/6.1 for P100; `git diff 70d5953f8 2894a98 --stat`: 18 files, +339/-22) → `VLLM_USE_PRECOMPILED=1` uses the upstream wheel of the merge base, which includes sm_75 for CUDA 12.8 | risk: the precompiled wheel for a Nov-2025 upstream commit may no longer be hosted (§7-H14) |
| **bf16** | **unsupported** (Turing). FORK `vllm/platforms/cuda.py:456-469` raises for bfloat16 below capability 8.0. Qwen2.5 `config.json` defaults to bf16 → `--dtype auto` would fail | v3 always passes `--dtype float16` (`deploy_<codename>.py:125,137` for `precision: fp16`) → OK. Keep fp16 on every tier (also on Ampere: parity) |
| attention backend | FORK backend priority (non-Blackwell): FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX (`cuda.py:81-84`). FLASH_ATTN requires >= 8.0 (`v1/attention/backends/flash_attn.py:163-164`); FLASHINFER accepts 7.5-12.1 (`flashinfer.py:348-351`); TRITON_ATTN accepts all (`triton_attn.py:205-206`) | so: A30/A100 → FLASH_ATTN; **V100 (sm_70) → TRITON_ATTN**; RTX 8000 (sm_75) → **FLASHINFER by default** — vPipe attested "flashinfer prefill unusable on sm_75 (48 KB smem; seq>=32 fails; boots-then-dies)" (for SGLang; unverified for vLLM). Smoke proposal (needs approval): `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (env honoured, `vllm/envs.py:649`), which is also the V100 tier's code path |
| flash-attn (standalone pip) | not used by v3, not installable for sm_75 | do not add |
| flashinfer JIT | needs nvcc + ninja if FLASHINFER is ever selected | another reason to force TRITON_ATTN on the box |
| Qwen2.5-3B fp16 (~6.2 GB) / 7B (~15.2 GB) | fit on 48 GB individually at v3 settings (`gpu-memory-utilization 0.9`, `max-model-len 8192`) | two instances on one GPU need the utilisation split (e.g. 0.40/0.50) — a smoke-only deviation from 0.9 (R7) |
| Qwen2.5-14B fp16 (~29.5 GB) | fits at TP=1, 0.9, 8192 (KV headroom ~10 GB) | v3 ran it TP=$GPU_COUNT on V100s — latency not comparable |
| Qwen2.5-72B | does not fit | cannot smoke the 72B tier or `--enforce-eager` behaviour at scale |
| scheduler (CPU) | runs with `CUDA_VISIBLE_DEVICES=""` exactly as v3 (`WSW/v3_scripts/lib_serve_v3.sh:158`) | KNN (MiniLM on CPU) + XGBoost `inplace_predict` on CPU |
| XGBoost latency heads | exist only for `qwen2.5-{3b,7b}_a30`, `qwen2.5-14b_v100`, `qwen2.5-72b_a100` (`V3X/deployed_config/predictor_xgboost_3model.json:7-27`) | an RTX 8000 instance has NO head. For a smoke, map the host prefix to an existing tier via the GPU-map override (V3 `CARA_HOST_GPU_MAP`, `cara_serve.py:183-195`) and label the run "wiring-only" |
| disk | weights 3B+7B+14B ~51 GB + venv (~10-12 GB with torch/flashinfer/cupy) ~= the 63 GB scratch | put `HF_HOME` on scratch (re-downloadable), keep the venv reproducible from the constraints file; never store the only copy of anything there |

What a dev-gpu smoke CAN validate: install/lock correctness, fork endpoints (`/instance_stats`, `/schedule_trace`, `predicted_decode_tokens` echo), scheduler startup + EFFECTIVE_CONFIG line, request path (chat unwrap → `/v1/chat/completions` streaming), decision-log schema, benchmark client output schema, verify_output gate. What it CANNOT validate: any latency/throughput number, TP>1, `--enforce-eager` on 72B, cross-host networking, the fleet gates, the 4-tier mix.

---

## 2. Deployment

### 2.1 Topology of the v3 S2 run (13 CloudLab Wisconsin nodes, one vLLM instance per node)

| tier (instance_type) | model (HF, base not Instruct) | hw type | nodes | GPU/node | vLLM params (deployed) | cost in/out ($/token) | capacity |
|---|---|---|---|---|---|---|---|
| `qwen2.5-3b_a30` | `Qwen/Qwen2.5-3B` | d7525 | 3 | A30 24 GB | fp16, util 0.9, len 8192 | 6e-08 / 6e-08 | 10 |
| `qwen2.5-7b_a30` | `Qwen/Qwen2.5-7B` | d7525 | 5 | A30 | fp16, util 0.9, len 8192 | 7e-08 / 7e-08 | 6 |
| `qwen2.5-14b_v100` | `Qwen/Qwen2.5-14B` | c4130 | 3 | V100-SXM2-16GB | fp16, util 0.85, len 8192, ("TP 4" declared, **inert**) | 1.5e-07 / 1.5e-07 | 4 |
| `qwen2.5-72b_a100` | `Qwen/Qwen2.5-72B` | d8545 | 2 | A100-SXM4-40GB | fp16, util 0.95, len 8192, **enforce-eager**, ("TP 4" declared, inert) | 3.8e-07 / 4e-07 | 2 |

Sources: `V3X/deployed_config/model_deployment.json:1-70`, `scheduler_config_s2.json:72-77,110-134` (MD5-identical to `V3/block/config/<codename>/s2/*.json`, verified: dfec4da…, 002413d…, e57e13d…, 8ae030a…). Host→GPU class map `{"d7525":"a30","c4130":"v100","d8545":"a100"}` lives in THREE places: `V3/block/global_scheduler/<codename>/<codename>_serve.py:180`, `V3/block/exp/<codename>/deploy_<codename>.py:752`, `WSW/v3_scripts/make_s2_configs.py:37-42` (+`c240g5:p100` "report only").

Scheduler head: `c4130-110433` (node0, 16 cores), **co-located with `Qwen-2.5-14B_0`** — the committed `ENVIRONMENT.md` (fa3eff6) says "head serves none", the uncommitted working copy (`ENVIRONMENT.md:49-51`) says co-located "verified from registration logs"; `V3/cara_paper/handoffs/2026-08-02_CR_CORRECTION_LIST.md:111-114` confirms co-location. v1 ran the scheduler on an A100 node (`V3/cara_paper/reproducibility/REPRODUCIBILITY_MANIFEST.md:55`) → cross-era numbers are confounded by scheduler-host CPU.

### 2.2 `model_deployment.json` schema (two accepted shapes)

A. **Allocation template** (`V3/block/config/<codename>/model_config_template.json`): `{ "<model_key>": { hf_model_name, precision, backend: "vllm"|"ollama", node_type: { "<hwprefix>": int | {count, <vllm params...>} } } }`. `deploy_<codename>.py:646-669` pops `count` nodes of that hw prefix from the host file and rewrites the entry to `node_hosts` + optional embedded `instances` (`:684-696`). Note the template's 72B/14B/7B/3B params are NOT identical to s2 (e.g. c240g5 P100 3B entry, 32B tier).

B. **Placed** (what v3 ran, `s2/model_deployment.json`): `{ "<model_key>": { backend, hf_model_name, node_hosts: ["user@host", ...], precision: "fp16", vllm_params: {...} } }`.

Per-field consumer (effect, not declaration):

| field | read by | effect |
|---|---|---|
| model key (`Qwen-2.5-3B`) | scheduler: instance ids `"<key>_<idx>"` (`<codename>_serve.py:3518`) | idx order = `node_hosts` order |
| `hf_model_name` | deploy `--model`; scheduler `VllmInstance(model_name=…)` → the `"model"` field of every backend request | must equal the served model id |
| `backend` | deploy + scheduler (`:3519-3546`); anything else raises | — |
| `precision` | deploy: `fp16` → `--dtype float16`, else `auto` (`deploy:121-125`); `vllm_params.dtype` overrides | — |
| `node_hosts` | deploy (ssh targets), scheduler (hostname key into host_configs) | `user@` prefix + hw prefix regex `@([a-zA-Z0-9]+)-` (`deploy:27`) is **load-bearing** for GPU-class inference |
| `vllm_params.gpu-memory-utilization` / `max-model-len` / `enforce-eager` | deploy (`:143-153`) | silent defaults 0.9 / 8192 / False if absent |
| `vllm_params.disable-prefix-caching` (or CLI `--disable-prefix-caching`) | deploy (`:160-161`, CLI re-merge fix `:634-680`) | APC ON by default (vLLM default) |
| `vllm_params.use_hf_cache` | deploy: `export HF_HOME=/mydata/hf_cache` (`:174-179`) | vLLM process only |
| `vllm_params.dtype`, `attention_backend` | deploy (`:122-123`, `:181-185` → `VLLM_ATTENTION_BACKEND`) | not set in s2 |
| **`vllm_params.port` (7000)** | **nobody** — deploy never passes `--port`; vLLM default 8000 (FORK `cli_args.py:78`); health check hardcodes 8000 (`deploy:199-203`) | DEAD. Docs that say "port 7000" (`V3X/REPRODUCE.md:36`, `REPRODUCIBILITY_MANIFEST.md:54`, OLD template) are wrong |
| **`vllm_params.tensor-parallel-size` (4)** | **nobody** — deploy always emits `--tensor-parallel-size $GPU_COUNT` with `GPU_COUNT=$(python3 -c 'import torch; print(torch.cuda.device_count())')` (`deploy:138,192`) | DEAD. Actual TP = all visible GPUs on the host |
| `instances` (embedded v2 form) | scheduler prefers it over host_configs (`:3494`) | not present in s2 file |

### 2.3 `host_configs.json` schema (how the scheduler finds instances)

`{ "<fqdn hostname>": { "backend_port": 8000, "ip_address": "<public IPv4>", "predictor_ports": [8300] } }` (`V3X/deployed_config/host_configs.json`). Join: for each `node_hosts` entry, `hostname = host.split("@")[-1]`, `hc = host_config[hostname]` (KeyError if absent; `<codename>_serve.py:3496-3509`, `:3502`), URL = `http://{ip_address}:{backend_port}/v1/{completions|chat/completions}` (`vllm_instance.py:78-79`), stats URL `…/instance_stats` (`<codename>_serve.py:976`). Silent defaults inside the join: `ip_address ""`, `backend_port 8000`, `predictor_ports [8300]` (`:3506-3508`) — a missing field produces a wrong-but-running instance (CLASS 3 defect). `predictor_ports` is vestigial in v3: the latency predictor is in-process (`<CODENAME>_INPROC_PREDICTOR` defaults to "1", `:1377`, `:3061`) and v3 deployed only `-d model_instance monitor` (`WSW/v3_scripts/s2_driver.sh:811-819`). IPs must be the PUBLIC rspec IPv4 (128.105.x), never LAN `10.10.1.x` (`V3X/REPRODUCE.md:18-20`; `make_s2_configs.py:1-24`; `V3/block/config/<codename>/jul25/host_configs.json` still holds the LAN map that did not run).

### 2.4 vLLM launch (exact, reconstructed from `deploy_<codename>.py:119-207`)

Per serving host, one ssh command chain (`run_ssh_cmd`, `StrictHostKeyChecking=no`, `UserKnownHostsFile=/dev/null`, 900 s timeout, `:46-55`):

```bash
[ -d ~/vllm ] || exit 1; cd ~/vllm
export HF_TOKEN=<from --hf-token, default "">          # token on the command line (hygiene hazard)
export HF_HOME=/mydata/hf_cache                         # only if use_hf_cache
export LD_LIBRARY_PATH=$(python3 -c '<glob site-packages/nvidia/*/lib>'):$LD_LIBRARY_PATH
export GPU_COUNT=$(python3 -c 'import torch; print(torch.cuda.device_count())')
sh -c 'cd ~/vllm && nohup python3 -u -m vllm.entrypoints.openai.api_server \
   --model Qwen/Qwen2.5-<N>B --dtype float16 --tensor-parallel-size $GPU_COUNT --trust-remote-code \
   --gpu-memory-utilization <0.9|0.85|0.95> --max-model-len 8192 [--enforce-eager  # 72B only] \
   [--no-enable-prefix-caching  # only with --disable-prefix-caching] \
   > vllm_server.log 2>&1 < /dev/null &'
sleep 10            # 600 with --fresh-deploy
for i in 1..36: curl -sf http://localhost:8000/health && break; sleep 5    # then exit 1 if still down
```

Preceded by cleanup `pkill -f '^python.*vllm\.entrypoints\.openai\.api_server'` (`:87-95`). Host = all interfaces (FORK `api_server.py:2127` `(args.host or "", args.port)`), port 8000. Every other engine knob is a vLLM default resolved at start-up (max_num_seqs, max_num_batched_tokens, chunked prefill, CUDA-graph capture sizes, attention backend per arch, APC ON). Those resolved values are the execution facts the FIRST RULE is about: record them per instance from `vllm_server.log` and from `/instance_stats` (`token_budget_per_iter`, `max_num_seqs`), not from the launch args.

### 2.5 Scheduler and benchmark launch (exact v3 form, `WSW/v3_scripts/lib_serve_v3.sh`)

```bash
cd $TREE && setsid --fork env CUDA_VISIBLE_DEVICES="" <CODENAME>_INPROC_PREDICTOR=1 $extra \
  nohup python3 -u -m block.global_scheduler.<codename>.<codename>_serve --host 0.0.0.0 --port 8200 \
  --model_config_path block/config/<codename>/s2/model_deployment.json --scheduling <mode> --chat \
  --scheduler-config block/config/<codename>/s2/scheduler_config_s2.json \
  --predictor-config block/config/<codename>/predictor_xgboost_3model.json \
  --host_config block/config/<codename>/s2/host_configs.json --decision-log $DECISION_LOG \
  > $SCHED_STDOUT_LOG 2>&1 < /dev/null &                                     # :158-167
# RB = --scheduling <codename-mode>; ALL baselines = --scheduling pipeline (V3X/REPRODUCE.md:60-62)
python3 -u -m block.benchmark.<codename>.benchmark_serving --backend <codename> --base-url http://127.0.0.1:8200 \
  --endpoint /v1/completions --model Qwen/Qwen2.5-3B --tokenizer Qwen/Qwen2.5-3B \
  --dataset-name custom --dataset-path data/<codename>/best-route-v3-test-3534-eval.jsonl \
  --num-prompts $N --request-rate $LAMBDA --seed 1 --max-output-len 1024 --metadata cap=1024 \
  --result-dir … --result-filename … --save-result --save-detailed         # :222-231
```

Execution facts hidden in defaults (all must be parity-checked, §6): scheduler sampling injection `temperature 0.0`, `repetition_penalty 1.0`, `frequency_penalty 1.2` (CLI defaults `<codename>_serve.py:135-137,3836-3845`, injected for every mode at `:423-428`); `--workers` must be 1 (`:3800`); `logging.basicConfig(filemode='a+')` to a FIXED file `experiment_output/logs/<codename>_serve.log` (`:129`); output cap `V3_MAX_OUTPUT_LEN=1024` (`lib_serve_v3.sh:41`; v1/rebuttal ran at the implicit 256); benchmark `--max-total-len` 2048 and `is_valid_sequence` `max_prompt_len=1024` (FORK `vllm/benchmarks/datasets.py:263-287`, `--max-total-len` is a FORK-added flag); custom JSONL datasets go through the FORK's upstream `CustomDataset` (`V3/block/benchmark/<codename>/dataset.py:114-115`): `random.seed(seed); random.shuffle` unless `--disable-shuffle`, and the **benchmark applies the Qwen2.5-3B chat template** unless `--skip-chat-template`; the scheduler then unconditionally unwraps it (`<codename>_serve.py:80,114,3059`) and, with `--chat`, calls `/v1/chat/completions` on base models.

### 2.6 `deploy_<codename>.py` flow and v3 invocation

v3 used: `python3 block/exp/<codename>/deploy_<codename>.py --hosts $S2DIR/hosts_all --output $S2DIR/deployed_model_deployment.json --config $CFGDIR/model_deployment.json --host-config $CFGDIR/host_configs.json --models "$MODELS" -d model_instance monitor --parallel-deploy 8` (`s2_driver.sh:813-819`), followed by an effect gate that polls every declared (host, port) (`:821-830`). Services: `model_instance` (cleanup + vLLM), `predictor` (HTTP sidecars on `predictor_ports`, `CUDA_VISIBLE_DEVICES=` + OMP/MKL=1, `:275-292`), `monitor` (`monitor.py`, NVML + `/metrics`), `scheduler` (`:377-435`, health = `GET /v1/batch_stats`). Final config written to `--output`; any failed service → exit 1 (`:864-883`).

Known defects to NOT port: default paths point at a non-existent codename tree (`--hosts cara/config/hosts`, `--config cara/config/cara/…`, `:477-548`); monitor and scheduler launchers `cd ~/<Codename>` and run `cara/exp/…/monitor.py` (`:336-361`, `:394-424`) while v3 lived in `~/Block` → monitor almost certainly failed silently: the verify step prints `Monitor FAILED` but exits 0 (`:367-371`); scheduler deploy hardcodes `--host_config cara/config/host_configs.json` ignoring `--host-config` (`:409`); `host_config[hostname]` KeyError for any host absent from the map (`:738`, the reason v3 skipped `-d predictor`); `--predictor-type` defaults to `dummy`; `--hf-token` comment "Hardcoded for local testing. Remove before committing" (`:482-484`); `except:` bare fallback when reading an existing output (`:600-601`); `--parallel-deploy` ThreadPool contradicts universal R4 unless measured.

### 2.7 Ports

| port | service | source |
|---|---|---|
| 8000 | vLLM OpenAI server (all instances) | vLLM default; `host_configs.json` `backend_port`; `deploy:199` |
| 8200 | scheduler (`/v1/completions`, `/health`, `/v1/batch_stats`, `/v1/config` GET/POST, `/v1/estimate`, router info) | `lib_serve_v3.sh:160`, `<codename>_serve.py:411,658,736,765,889` |
| 8300 | predictor HTTP sidecar (not used in v3; in-process) | `host_configs.json` |
| 11434 | Ollama (testing only) | `deploy:96-113` |
| (7000) | declared in `vllm_params.port`, never used | §2.2 |

### 2.8 CloudLab provisioning, disk layout

- `/mydata`: ext4 on `/dev/nvme0n1` (d8545, d7525 per `setup_jul25.sh:23-24`) or `/dev/sdb` (c4130, `:25`), `chmod 777`, NOT in fstab (remount after every reboot). Conflicting docs: `V3/CLAUDE.md:45-58` and `WSW/ENVIRONMENT/README.md` "Storage" say A30 d7525 has NO `/mydata`; `head_scripts/_verify_env.sh:9` requires `/mydata` mounted on every host. Resolve by probing `lsblk` per hw type at lease start.
- Layout used: `/mydata/hf_cache` (HF_HOME for vLLM weights; `_verify_weights_14b.sh:2` checks `models--Qwen--Qwen2.5-14B` has >= 1 safetensors), `/mydata/<campaign>/…` for cells/logs; code at `~/Block` (rsync copy, NO `.git` — `V3X/manifest/AUDIT.md:129-134`), fork at `~/vllm`; model artifacts at `/users/<user>/Block/models/<codename>/latency/deploy_{ttft,tpot,e2e}/*.xgb` (absolute paths in `predictor_xgboost_3model.json:9-44`) and `models/<codename>/knn/deploy_v3_k50_14919/knn_estimator.pkl` (relative, `scheduler_config_s2.json:104`); dataset `data/<codename>/best-route-v3-test-3534-eval.jsonl`. Head root disk was 92% full at release (`AUDIT.md:121`) — the scheduler's sentence-transformers download goes to the default `~/.cache/huggingface` (root), not `/mydata`.
- Root FS is ~63 GB on every hw type.
- The OLD repo's `cloudlab/` is NOT a RouteBalance profile: `cloudlab/setup.sh:19-28` clones and builds `AKafakA/dodoor` (a different project) and installs Java/Maven/thrift; `cloudlab/README.md:1` says "profile for dodoor"; `cloudlab/profile.py:18-45` requests `{c4130:3, d7525:8, d8545:2}` but assigns hardware types to `node0..node12` with `random.choice` (non-deterministic node→hw mapping; node0 = scheduler head is random), no disk image, no blockstore, no `/mydata`. OLD `REPRODUCE.md:15` claims it "runs cloudlab/setup.sh per node to install vLLM + weights, emitting a ready host_configs.json" — false.

### 2.9 What the scheduler relies on from the vLLM fork (FORK @ 2894a98)

| feature | FORK location | scheduler consumer | if run on upstream vLLM |
|---|---|---|---|
| `GET /instance_stats` → `get_aggregated_stats()`: `num_running, num_waiting, num_active_decode_seqs, decode_ctx_{mean,p50,p95,max}, total_decode_context_tokens, pending_prefill_tokens, pending_decode_tokens, kv_cache_utilization, kv_free_blocks, token_budget_per_iter, max_num_seqs, num_preempted, timestamp` (16 fields) | `vllm/entrypoints/openai/api_server.py:414-429`; `vllm/v1/core/sched/scheduler.py:1281-1375`; plumbing `engine/protocol.py:118`, `v1/engine/async_llm.py:737`, `core.py:492`, `core_client.py:147,217,308,778,982` | `_get_instance_stats` (3 s timeout, raises `InstanceStateUnavailable` on non-200/empty/exception unless `slo_defaults.stats_blind_absorb`) `<codename>_serve.py:928-1000`; XGBoost feature vector = the raw body (`estimators/xgboost_predictor.py:19-53,106-107`); balance term uses `kv_cache_utilization` (`:1073-1095`); dispatchers (sq/po2) use `num_running/num_waiting` | 404 → every dispatch fails loud (good) — unless the absorb flag is set, then silently `{}` (idle state) |
| `predicted_decode_tokens` request field (soft estimate, defaults to `max_tokens`) | `entrypoints/openai/protocol.py:723,1152`; `serving_chat.py:338`; `serving_completion.py:229`; `serving_engine.py:1210,1226`; `v1/engine/processor.py:397,526`; `v1/engine/__init__.py:76`; `v1/request.py:47,87-89` | scheduler sends `int(round(num_predicted_output_tokens or max_tokens))` (`vllm_instance.py:109-110,123,139`) | upstream pydantic model ignores/rejects the extra field → `pending_decode_tokens` unavailable |
| `pending_decode_tokens` semantics: Σ over running+waiting of `predicted_decode_tokens − num_output_tokens`, with **auto-extend** `predicted = num_output + 10` once exceeded; **the read endpoint mutates request state** | `scheduler.py:1291-1330` | feature `pending_decode_tokens`, `backlog_seconds` derived | — |
| `GET /schedule_trace` (running/waiting flattened `[request_id, num_prompt_tokens, num_computed_tokens, total_num_tokens, num_output_tokens, predicted_decode_tokens]` per request + `free_gpu_blocks`, `num_preempted`) | `api_server.py:379-411`; `scheduler.py:54-68,1274-1279`; non-OpenAI `entrypoints/api_server.py:40` has a different 6-field schema and an in-place list-mutation bug | sidecar predictor / `schedule_trace_client.py` / training-data collection (not the v3 in-process path) | — |
| `num_preempted_requests` counter | `scheduler.py` (+2 lines) | `num_preempted` feature/monitoring | — |
| benchmark `--max-total-len` (default 2048) | `vllm/benchmarks/datasets.py` diff (+10 lines) | benchmark dataset filter | flag missing → argparse error |
| benchmark library internals: `vllm.benchmarks.datasets` (`SampleRequest`, `add_dataset_parser`, `CustomDataset`, `is_valid_sequence`, `get_samples`), `vllm.benchmarks.lib.endpoint_request_func`, `lib.ready_checker`, `lib.utils`, `vllm.transformers_utils.tokenizer.get_tokenizer`, `vllm.utils.gc_utils.freeze_gc_heap`, `vllm.utils.network_utils.join_host_port` | upstream code at the fork base | `V3/block/benchmark/<codename>/benchmark_serving.py:29-39`, `dataset.py:8,114`, `<codename>_end_point_func.py:9` | version-coupled: the benchmark host needs the fork installed |
| `request_id` honoured in body + `X-Request-Id` header; `Authorization: Bearer $OPENAI_API_KEY` | upstream | `vllm_instance.py:128,145,148-151` | — |
| `/health` 200/503 | `api_server.py:369-376` | deploy health loop | — |

XGBoost features the fork does NOT provide and that are therefore always 0.0 via `schedule_state.get(f, 0.0) or 0.0` (`xgboost_predictor.py:106-107`): `ema_decode_tok_per_s`, `ema_prefill_tok_per_s`, `ema_decode_iter_ms`, `prefill_chunk_size`, `kv_evictions_per_s`. The trained boosters and the v3 results depend on those zeros; an "improvement" that fills them changes every prediction.

---

## 3. Existing tests (`V3/tests/`, 15 files)

**Bottom line: none of the 12 test modules can even be collected against the v3 tree.** They were written for the Jan–Mar 2026 architecture (policy/registry/state-manager objects) and last touched in `2074f4a` (2026-06-07). Static import check (AST, read-only): every module imports at least one symbol or module that no longer exists (`SimpleScheduleState`, `CARABasePredictor` [now `CARA_BasePredictor`], `StateApiType`, `AdaptiveBatcherConfig`, `CARASchedulerConfig`, `SchedulerConfig`, `RooflinePredictorConfig`, `LinearPredictorConfig`, `_schedule_cara_batch`; modules `adaptive_batcher`, `<codename>_scheduler_loop`, `multi_objective_scorer`, `rso`, `scheduling_policy`, `service_rate_tracker`, `state_manager`, `predictor_registry`, `offline_training.evaluate_predictor` — none exist and none are in git history under those paths). pytest is not in the v3 freeze; the tests never ran on the cluster.

| file (lines) | covered (by intent) | v3 validity | port value |
|---|---|---|---|
| `tests/__init__.py`, `tests/predictor/__init__.py`, `tests/predictor/<c>/__init__.py`, `tests/global_scheduler/<c>/__init__.py` (1 each) | package markers | — | drop |
| `global_scheduler/<c>/test_adaptive_batcher.py` (225) | load-ratio → batch size interpolation, busy detection | module gone | concept only (adaptive batch sizing still exists as `batch_config.adaptive_sizing`); rewrite against `BatchQueue` |
| `global_scheduler/<c>/test_batch_scheduling.py` (636) | `BatchQueue.adapt_params`, LPT ordering, local-state update accumulation, mean utilisation, normalisation, tiered constraints, balance term, E2E soft scoring, estimator dataclass, KNN extension methods | `BatchQueue`, `BatchRequest`, `_compute_mean_utilization`, `_update_local_stats`, `_fetch_all_instance_stats` still exist; `_schedule_cara_batch` gone → whole module fails import | **highest value**: the surviving targets are core; rewrite as unit tests of the ported scorer/LPT/local-update code |
| `global_scheduler/<c>/test_<c>_scheduler_loop.py` (380) | pending-request loop, service-rate integration | modules gone | drop |
| `predictor/<c>/test_data_structures.py` (216) | `PredictRequest`, `RequestInfo.from_list` (+offset), `ScheduleState.from_response`, backlog seconds, staleness, `TrainingExample` | `PredictRequest`/`RequestInfo`/`ScheduleState`/`TrainingExample` exist; `SimpleScheduleState` gone | port the `RequestInfo`/`ScheduleState` parsing tests — they are the `/schedule_trace` contract (note `_detect_fields_per_request` handles 4- vs 6-field schemas, `data_structures.py:15`) |
| `predictor/<c>/test_evaluate_predictor.py` (358) | MAPE/percentile metrics, test-data filtering, baseline eval | module gone | drop (offline) |
| `predictor/<c>/test_multi_objective_scorer.py` (331) | constraint/RSO violations, tiers, relaxed/strict/tiered selection, minmax/regret normalisation | modules gone | concept only; scorer now lives in `_score_and_pick_instance` (`<codename>_serve.py:1548`) — rewrite |
| `predictor/<c>/test_predictor_registry.py` (195) | instance-type derivation, registry fallback | module gone | the instance-type derivation concept matters (single GPU map); rewrite |
| `predictor/<c>/test_predictors.py` (371) | roofline/linear/XGBoost predictor API, XGBoost feature names/count | base class renamed, configs gone | port only the XGBoost **feature order/count** assertion (29 features, `xgboost_predictor.py:53`) |
| `predictor/<c>/test_scheduling_policy.py` (571) | bootstrap vs batch policy factory | modules gone | drop |
| `predictor/<c>/test_service_rate_tracker.py` (313) | EMA rate tracker, thread safety | module gone | drop (ema_* features are 0 in v3) |
| `predictor/<c>/test_state_manager.py` (248) | state cache, capacity, health | module gone | drop |

**The real v3 test assets are the gate scripts in `WSW/v3_scripts/`** (offline unless noted), which the rebuild should port as `tests/gates/` + `routebalance/gates/`:
`gate_selftest.py` (1555 lines; "every gate must FAIL on demand"; expected 162/89/0, `V3X/REPRODUCE.md:82`), `static_defect_gate.py` (4 bug classes: falsy-`or`, hardcoded model ladders / duplicated host→GPU maps, silent defaults, dead config keys; default `ROOT` is a hardcoded personal path), `runtime_gates.py` (tiers/manifest/orphans/arms), `preflight_gate.py` (declared vs resolved vs behaviour), `assert_effective_config.py` (reads the scheduler's `EFFECTIVE_CONFIG` log line), `tpot_gate.py` + `tpot_zero_gate.py` (arm gate on the decision log: static arm ⇒ `lat_s/exp_len` constant per tier), `verify_pairing.py` (same seed ⇒ identical arrival process), `check_zero_safe.py` (0.0 quality must not be dropped; hardcoded `/users/<user>/Block/...` AGG path), `test_cost_calc.py` (decode-only vs full cost arithmetic), `check_quality_determinism.py`, `expected_cells.py`, `dominance.py`, `cell_audit_v3.sh`/`cell_audit_s2.sh` (RULE-0 per-cell audit), `head_scripts/_verify_{env,tree,artifacts,weights_14b}.sh`, `_pin.sh`.

**Gaps (no test anywhere)**: (1) deploy command generation (flags per tier, TP source, dtype, prefix-caching merge, port/health); (2) config loaders and the model_deployment ⋈ host_configs join (missing keys must raise); (3) a contract test of the fork API (`/instance_stats` 16-field schema, `/schedule_trace` 6-field schema, `predicted_decode_tokens` accepted + reflected in `pending_decode_tokens`) — needs a fake backend (OLD `dummy_backend.py` implements only `/schedule_trace`); (4) sampling-param injection and budget-clamp parity between `<codename>`-mode and `pipeline` mode (`<codename>_serve.py:423-428`, `:560-582` vs `:1123-1146`); (5) prompt unwrap round-trip (benchmark chat template → scheduler unwrap → chat endpoint); (6) KNN determinism (same prompt → same `exp_len`/quality; v3 decision logs carry `exp_len` per request, e.g. `rb-exp/v3-exp/cells/s2_p1_l12/decisions_*.jsonl` keys `req, static_active, static_cfg_ms, cand[inst,itype,lat_s,exp_len,learned_e2e_s]`); (7) XGBoost feature-vector golden test (input stats dict → exact float32 vector); (8) benchmark sampling (seeded shuffle, `is_valid_sequence` pruning, resulting N); (9) decision-log schema; (10) output-path builder includes every config axis (universal R6); (11) verify_output per cell (P5); (12) dead env-var detector (an exported env var with no reader: e.g. `<CODENAME>_UNWRAP_PROMPT` is exported by 4+ v3 launchers — `phase2_cells.sh:13`, `gemma_pt.sh:11`, `rebuttal_fix_phase1.sh:35`, `w1b_cells.sh:19` — but has no reader; unwrap is unconditional, `<codename>_serve.py:3059`); (13) equivalence against v3 decisions (the decision logs record outputs but NOT the `/instance_stats` inputs, so replay equivalence needs a new input capture, §5.4).

---

## 4. Repo-hygiene hazards found in the OLD public repo (HEAD `671db9f`; file names/lines only)

**4.1 Cluster identity / infra addresses in tracked files**
- Hardcoded public cluster IPs: `reproduce/run_main_table.sh:85` (`SR_URL` default), `route_balance/exp/route_balance/run_smoketest.sh:15` (`SCHEDULER_IP`).
- LAN IPs: `route_balance/route_balance_pd/config/pd_config_smoketest.json:10,19`, `route_balance/route_balance_pd/exp/deploy_pd.sh:19-20`.
- CloudLab node names (d8545-10s10301, d7525-10s10317) in comments/defaults: `route_balance/exp/route_balance/sweep_latency_benchmark.sh:7`, `deploy_route_balance.py:26`, `predictor/route_balance/offline_training/tag_latency_instance_type.py:40,75`, `train_fused_bert_predictor.py:79`, `route_balance_predictor_api_server.py:298`.
- CloudLab/HF account name `asdwb` as a default: `route_balance/exp/generate_config.py:93`; as HF dataset namespace in `README.md:87`, `REPRODUCE.md:216-225`, `run_overnight_judging_pipeline.sh:130`, `fix_safety_scores.py:7,23,28`, `update_hf_datacard.py:7,119,154`, `upload_latency_to_hf.py:21,186`, `upload_latency_v2.py:107`, `upload_scored_data_to_hf.py:14,31` (the HF namespace may be intended-public; the CloudLab default user is not).
- Internal paths/doc leakage: `REPRODUCE_with_ROUTEBALANCE.md` (references `~/Code/llm/route-balance/v2-reproduce/`, `~/Block`, the private `block.*` package, pack layout), `CODE_TOUR.md` (lists a `route_balance_paper/` tree, `smoke_test_apr_13/`, apr26 launcher names), `.gitignore:76-77` (names `scripts/transform_block_to_route_balance.sh`, `PAPER_REPRODUCTION_MAP.md`), templates mention "dodoor-cl-profile" (`route_balance/config/host_configs.template.json:2`, `route_balance/config/route_balance/model_deployment.template.json:2`).
- No personal identifiers (wd312, cam.ac.uk hosts, VPS IP, CSD3 account) found in HEAD.

**4.2 Codename.** Zero hits in code/docs. Only hits: `data/route_balance/scored/test_scored_filtered.jsonl` and `data/route_balance/best-route-v3-test-3634-full.jsonl` (2 each) — a GSM8K word problem about a person named "Cara". A codename gate must scan code/docs/configs/filenames/env-var names and **allowlist dataset text**, or it false-positives on the eval set.

**4.3 Backup / stale / dead files tracked despite `.gitignore:73-74` (`*.bak`, `*.bak.*`) — ignore rules do not untrack:**
`route_balance/config/{ampere_hosts,hosts,pascal_hosts,volta_hosts}.bak.20260413_163144`, `…bak.20260606_030233` (8 files), `route_balance/config/route_balance/model_deployment_apr26_phase2.json.bak.20260606_030233`, `route_balance/config/route_balance/scheduler_config_route_balancefast.json.STALE_v1`; P100-only configs (`p100_smoke/*`, `p100_smoketest_hosts`, `pascal_hosts`, `model_config_p100_vsr.json`) for a tier that no longer exists (the "zombie p100 tier" defect); 16 near-duplicate `route_balance/exp/run_exp_predictor_{1..16}.sh`; `setup_llumnix.sh`; `data/dataset_mixer.ipynb`.

**4.4 Secrets.** No token-shaped literals (HF `hf_…`, OpenAI/Anthropic `sk-…`, GitHub `ghp_`/`github_pat_`, Slack, AWS) in OLD HEAD or in any of its 12 commits (pickaxe over non-data paths). Hazards that remain for the rebuild:
- `V3/block/exp/<codename>/setup_env.sh:28,40` writes `export HF_TOKEN=…` into every node's `~/.bashrc` (an expired token there caused the jul25 401s; `setup_jul25.sh:66-68` had to sed it out); `deploy_<codename>.py:171` puts the token on the ssh command line (visible in `ps`, shell history, deploy logs).
- The V3 private repo has token-shaped literals in 11 tracked files OUTSIDE the port scope (names only): `Block_paper/claude/tokens.md`, `cara_paper/active/APR26_PREFLIGHT.md`, `cara_paper/claude/tools.md`, `cara_paper/handoffs/2026-04-15_handoff_v2.md`, `scripts/fix_quality_scores.py`, `scripts/fix_safety_scores.py`, `update_latency_readme.py`, `update_latency_readme_v2.py`, `update_model_estimator_readme.py`, `update_model_estimator_readme_v2.py`, `validate_judges_example.sh`. Never copy these. Untracked, git-ignored secret file exists: `WSW/.hf_token_jul25` (mode 600). The port-scope directories (`V3/block`, `V3/tests`, `WSW/v3_scripts`, `V3X`) are clean.

**4.5 Docs that do not match the tree (a reader following them fails):** `README.md:49` `pip install -e .` (no pyproject/setup.py tracked) and `-r requirements.txt` (not tracked); quick-start configs `model_deployment_smoketest.json`, `predictor_config_smoketest_knn.json`, `scheduler_config_smoketest.json` and `data/best-route-v3-test-500-clean.jsonl` not tracked (`README.md:58-67`); `REPRODUCE.md` references ~20 missing scripts/files (`reproduce/run_overhead_study.sh`, `run_budget_control.sh`, `aggregate_budget.py`, `make_ablation_figures.py`, `smoke_one_cell.sh`, `offline_training/prepare_xgboost_dataset.py`, `train_xgboost_3model.py`, `data/best-route-v3-test-3534-eval.jsonl` — only a **3634**-line file is tracked, v3 ran the 3534 eval set); `README.md:77-80`/`REPRODUCE.md:43-46` use `deploy_route_balance.py --model-config` (the flag is `--config`); hardware claims `REPRODUCE.md:5` "2x A100-80GB, 3x V100-32GB" vs v3 A100-40GB / V100-16GB; `README.md:10` KNN k=10 and `:95` weights 1/3 vs v3 k=50 and (0.3, 0.2, 0.3, 0.2) with `w_balance`; "Apache-2.0" (`README.md:150`) with no LICENSE file tracked; `README.md:37` lists a `csd3/` dir that `.gitignore:70` excludes.

**4.6 Templates that would silently change execution:** `model_deployment.template.json` 72B has util 0.85 and NO `enforce-eager` (v3: 0.95 + enforce-eager), 3B util 0.85 (v3 0.9); node names `node-N.example.com` make `parse_host_file`'s hw-prefix regex yield `node` → GPU class `node` → no XGBoost head; `${CLOUDLAB_USER}` / `${CLOUDLAB_HOST}` are literal strings inside JSON and inside every host file (`route_balance/config/hosts` etc. contain 13 literal `${CLOUDLAB_HOST}` lines) — the genericising sed broke the load-bearing host naming.

**4.7 OLD code is pre-v3.** `route_balance_serve.py` (3221 lines) lacks every v3 fix marker present in V3 (`_HOST_GPU_DEFAULT`, `--decision-log`, `EFFECTIVE_CONFIG`, static-TPOT-head reader, unconditional unwrap, `InstanceStateUnavailable` fail-loud) and carries App-G gates (`ROUTE_BALANCE_TIEBREAK`, `ROUTE_BALANCE_SQUARE`) that V3 does not have. **Do not port from OLD; port from V3 and re-add App-G gates deliberately.**

---

## 5. Proposed new repo

### 5.1 Layout

```
routebalance/                      # package (name = owner decision §0.1)
  serve/        app.py (FastAPI entry), config.py (typed loaders, fail-loud), instances/ (vllm.py, fake.py),
                scoring/ (score, normalise, LPT, local-state update, budget clamp), routers/, dispatch/, filters/,
                decision_log.py, effective_config.py
  predict/      knn.py, xgb_latency.py, features.py (the 29-feature vector, frozen order), static_prior.py
  hw.py         THE single host-prefix → GPU-class map + instance_type derivation (replaces 3 copies)
  bench/        client.py, dataset.py (pins the fork's CustomDataset semantics), results.py
  deploy/       plan.py (model_deployment × hosts → launch specs, pure), render.py (vLLM/scheduler command strings),
                ssh.py (bounded ssh: timeout + -n + ServerAlive, universal R5/T4), deploy.py (CLI)
  gates/        static.py (4 defect classes + codename/hygiene/secrets), runtime.py (tiers, orphans, manifest),
                preflight.py, effective_config.py, arm.py, pairing.py, zero_safe.py, parity.py (FIRST RULE),
                verify_output.py (T1), selftest.py
  analysis/     aggregate.py, dominance.py, cost.py, bootstrap_ci.py, figures/
configs/
  schema/       model_deployment.schema.json, host_configs.schema.json, scheduler.schema.json, predictor.schema.json
  v3/           scheduler_config.json, predictor_xgboost.json (artifact paths RELATIVE to $RB_MODELS),
                model_deployment.template.json (== s2 values, node_hosts empty), FLEET.md (tier table)
  smoke-devgpu/ (explicitly labelled smoke-only deviations)
constraints/    v3-head.txt, v3-serve.txt   (verbatim freezes, §1.2)
scripts/        node_setup.sh (from setup_jul25.sh, parameterised, no personal paths), verify_node.sh (S14 + _verify_*),
                make_configs_from_rspec.py, run_cell.sh, final_sync.sh (T7), pin.sh
cloudlab/       profile.py (deterministic node0=head, explicit hw per node, disk image, /mydata blockstore), README.md
tests/          unit/ contract/ equivalence/ gates/ fixtures/ (scrubbed, small)
docs/           README.md, REPRODUCE.md, ENVIRONMENT.md, DEPLOYMENT.md, CODE_TOUR.md, PARITY.md
Makefile  pyproject.toml  .gitignore  LICENSE  CITATION.cff
```

Principles: no absolute paths anywhere (`$RB_ROOT`, `$RB_MODELS`, `$RB_DATA` required env or CLI, missing ⇒ exit non-zero); host files carry real `user@<hwprefix>-…` names only in untracked `local/`; every config key has a reader or is rejected by the schema (`additionalProperties: false`); no `_v2`/`.bak`/`archive/` (universal R8).

### 5.2 `pyproject.toml` (pins copied verbatim from `V3X/env/head_pip_freeze.txt`; the constraints files carry the remaining ~200 transitive pins)

```toml
[build-system]
requires = ["setuptools>=64"]          # build tool only; not part of the v3 execution env
build-backend = "setuptools.build_meta"

[project]
name = "routebalance"                  # owner decision
version = "0.0.0"
requires-python = "==3.10.*"           # v3 ran 3.10.12
dependencies = [                       # scheduler (CPU) — exactly the v3 head
  "numpy==2.2.6", "scipy==1.13.1", "pandas==2.2.3", "scikit-learn==1.5.2", "joblib==1.5.3",
  "xgboost==3.2.0", "faiss-cpu==1.13.2",
  "torch==2.9.0",                      # PyPI wheel (cu12.8 libs bundled), as in v3
  "transformers==4.56.0", "tokenizers==0.22.2", "huggingface_hub==0.36.2", "safetensors==0.8.0",
  "sentence-transformers==5.4.1",
  "fastapi==0.141.1", "starlette==1.3.1", "uvicorn==0.32.1", "uvloop==0.22.1", "httptools==0.8.0",
  "pydantic==2.13.4", "pydantic_core==2.46.4",
  "aiohttp==3.10.11", "requests==2.32.5", "orjson==3.11.9",
  "psutil==5.9.8", "PyYAML==6.0.3", "tqdm==4.67.3", "dacite==1.9.2", "jsonlines==4.0.0",
  "nvidia-ml-py==13.610.43",
]

[project.optional-dependencies]
serve = [                              # serving hosts + benchmark host; install with VLLM_USE_PRECOMPILED=1
  "vllm @ git+https://github.com/AKafakA/vllm.git@2894a98722733c76cdb549c8b5e5068594f12bef",
  "torchvision==0.24.0", "torchaudio==2.9.0", "triton==3.5.0",
  "flashinfer-python==0.5.2", "xgrammar==0.1.27", "ray==2.55.1", "cupy-cuda12x==14.0.1",
  "nvidia-nccl-cu12==2.27.5", "nvidia-cudnn-cu12==9.10.2.21", "nvidia-cublas-cu12==12.8.4.1",
  "nvidia-cuda-runtime-cu12==12.8.90",
]
bench    = ["datasets==4.8.5", "pyarrow==24.0.0"]            # + the fork (bench imports vllm.benchmarks.*)
analysis = ["matplotlib==3.9.4", "seaborn==0.13.2", "plotly==5.22.0"]
dev      = []   # pytest / ruff / jsonschema: NOT in the v3 env — choose + pin at repo creation (owner, R3)
# training (transformers==5.4.0 per OLD requirements-training.txt) MUST be a separate environment:
# it conflicts with the serving pin transformers>=4.56,<5.
```

Install recipe: `pip install -c constraints/v3-head.txt -e .` (scheduler/VPS) and `VLLM_USE_PRECOMPILED=1 pip install -c constraints/v3-serve.txt -e ".[serve,bench]"` (nodes). `pip check` + an import-and-print gate after install (numpy 2.4.4 once "installed" but left the old one in place — `WSW/ENVIRONMENT/README.md` trap 3). Pin third-party model revisions too (not done in v3): `Qwen/Qwen2.5-{3B,7B,14B,72B}` and `sentence-transformers/all-MiniLM-L6-v2` (the embedder name comes from the pickled KNN state, `estimators/knn_estimator.py:70,435,481`; config key `embedding_model` is unread) by HF commit hash, recorded in `ENVIRONMENT.md` and asserted at start-up.

### 5.3 `make check` (all offline, CPU-only, no network; each target exits non-zero on failure)

| target | contents |
|---|---|
| `make lint` | formatter/linter check; `python -m compileall`; shellcheck on `scripts/*.sh` |
| `make hygiene` | codename scan (code/docs/configs/file names/env-var names; dataset text allowlisted), host/IP scan (`cloudlab.us`, `wisc.`, `hyperion`, hw-node patterns, RFC1918 + 128.105.x literals, personal host names), secret scan (token regexes; report file:line only), forbidden files (`*.bak*`, `*.STALE*`, `*_v2*`, `archive/`), absolute-path scan (`/users/`, `/home/`, `/mydata/` outside documented defaults) |
| `make static` | port of `static_defect_gate.py`: falsy-`or` chains on numeric paths, `.get(k, <fabricated>)` in scoring/feature paths, duplicated host→GPU maps (exactly one allowed: `routebalance/hw.py`), hardcoded model ladders, config keys without readers (schema ⇄ code), exported env vars without readers (scripts ⇄ code), hardcoded `output_dir` without all axes (universal R6) |
| `make unit` | `tests/unit`: config loaders (missing key ⇒ raise), deploy plan/render (golden command strings per tier, TP source explicit, dtype, APC merge), XGBoost 29-feature golden vector, scorer/normaliser/LPT/local-update, budget clamp parity (both modes), sampling injection, unwrap round-trip, cost arithmetic (port `test_cost_calc.py`), output-path builder |
| `make contract` | `tests/contract`: fake fork backend (implements `/health`, `/instance_stats` 16 fields, `/schedule_trace` 6-field, streaming `/v1/{chat/,}completions` accepting `predicted_decode_tokens` and reflecting it in `pending_decode_tokens` incl. auto-extend +10); scheduler must RAISE on 404/empty/timeout; schema of decision log + EFFECTIVE_CONFIG line + benchmark result JSON |
| `make equivalence` | `tests/equivalence`: replay golden fixtures produced once from the V3 tree (§5.4) — per request: KNN `exp_len`/quality equal to 1e-9, XGBoost `lat_s`/`learned_e2e_s` equal, chosen instance equal; benchmark sampler reproduces the exact v3 request list (ids, prompt_len, order) for seed 1 |
| `make gates` | `tests/gates`: gate self-test (every gate has MUST-FAIL fixtures and fails on them — port `gate_selftest.py` suites), `verify_output` on fixture cells, arm gate, pairing, zero-safe, **parity gate** (§6.1) on two fixture execution manifests |
| `make check` | `lint hygiene static unit contract equivalence gates` |
| (not in check) `make env-verify`, `make preflight`, `make smoke-devgpu` | on nodes / the box: SHA + imports + versions + `/mydata` + GPU + weights shards; preflight T3; 100-sample smoke |

### 5.4 Equivalence fixture generation (one-time, in the private tree, owner-approved)

Run the V3 scheduler code (unchanged) against the fake backend with scripted `/instance_stats` sequences and a fixed prompt subset (e.g. 200 prompts incl. zero-quality and chat-wrapped cases); record `(prompt, stats_map snapshot, config) → (per-candidate exp_len, quality, lat_s, score terms, chosen instance, max_tokens after clamp, sampling params sent)`. Scrub (no hosts, no codename) and store under `tests/fixtures/`. The existing v3 decision logs (e.g. `rb-exp/v3-exp/cells/s2_p1_l12/decisions_cara_1785790227.jsonl`, 28,320 rows) can cross-check `exp_len` per request id but cannot replay latency (no input stats recorded).

---

## 6. Code-relevant rules → checklist items the rebuild's gates must enforce

### 6.1 FIRST RULE (baseline execution parity; global CLAUDE.md top)
- [ ] **Execution manifest per arm, per cell**: for every instance record the RESOLVED backend config (vLLM version string + fork SHA, dtype, TP actually used, attention backend, `max_num_seqs`, `max_num_batched_tokens`/`token_budget_per_iter`, APC on/off, chunked prefill, CUDA-graph capture sizes / enforce-eager, gpu-mem util, max-model-len) from `vllm_server.log` + `/instance_stats`; for the scheduler record the EFFECTIVE_CONFIG line, sampling params actually sent (temperature, repetition/frequency penalty), `max_tokens` cap and clamp rule, chat-vs-completions endpoint, unwrap, benchmark cap/max-total-len/max-prompt-len/shuffle/seed/tokenizer/chat-template, scheduler host + CPU pinning.
- [ ] **Parity gate** diffs treatment vs every baseline manifest and exits non-zero on ANY difference not listed in the pre-registered "treatment-defining" allowlist (the only allowed difference is the scheduling policy: `--scheduling <rb>` vs `pipeline` + its router/dispatch/filter payload). A config diff is not sufficient; the gate reads execution evidence.
- [ ] Any difference a stock flag could give the baseline is either shared or has a control arm; a disclosure sentence is not a control.
- [ ] Cross-era comparisons (v1 256-cap, A100 head; v3 1024-cap, V100 co-located head) are forbidden unless remeasured.
- [ ] A lane/campaign is "finished" only when every owed control has run and been mirrored (two-location, T7), with the check output shown.

### 6.2 Global R1–R7 (as code rules)
- [ ] R1: setup never `pip install`s into an existing working env; new envs are built from `constraints/*.txt`; configs used by a running process are never edited (write new versioned files).
- [ ] R2: exactly one node-setup script; a failure stops (no improvised fixes chained on top).
- [ ] R3/R7: no default for any experiment parameter in code or scripts — required CLI/config keys, schema-validated; the deploy renderer refuses when `gpu-memory-utilization`/`max-model-len`/`enforce-eager`/`dtype` are unspecified instead of defaulting (v3 defaults at `deploy:143-145`); no combined metrics.
- [ ] R4/data-integrity R3: smoke (10–100 samples) is the first cell of every sweep (P3) and verifies the EXACT config.
- [ ] R5: use the fork's own benchmark/dataset code and official libs; do not re-implement.
- [ ] R6: fail loud — no `except: pass`, no `return {}` on telemetry failure (keep `InstanceStateUnavailable`), missing model/artifact ⇒ startup failure, >1% request failures ⇒ cell INVALID.

### 6.3 data_integrity_rules.md
- [ ] (R1) no silent parameter/default/scoring changes; any field available but not piped through is REPORTED, not decided.
- [ ] (R2) never overwrite source data; versioned copies; data ops are scripts.
- [ ] (R3) output schema + failure-rate check after the first batch.
- [ ] (R4) code↔paper equation map: each scoring term has a test that names the paper equation it implements; function names match behaviour (`predict_ttft` returns TTFT).
- [ ] (R6) never report success when the underlying model was not loaded: the scheduler logs and the gate asserts loaded tiers == deployed tiers (`runtime_gates.py tiers`), KNN k and corpus size == config (`EXPECT_K`, `_pin.sh:12`).

### 6.4 universal_rules.md (+ patterns, tools)
- [ ] R1/T3 preflight before any sweep: patches present, synced (tree manifest per file, not a bare hash — `TREE_MANIFEST_V3.md5` is a 32-byte bare md5 that failed its own gate and does not match the current tree under `_pin.sh`'s recipe), N/N processes, 100-sample smoke with schema-valid output.
- [ ] R2 `assert_invariants()` at every cell tail (sample count == N, required fields, finite TTFT P99, capacity in known range) ⇒ non-zero exit.
- [ ] R3 new cross-service metric ⇒ 1-cell smoke asserting the field reaches the final file.
- [ ] R4 sequential deploy by default; parallelism only with a measured reason.
- [ ] R5/T4 every remote spawn: `timeout`, `ssh -n`, `< /dev/null`, ServerAlive, bounded wait.
- [ ] R6/T5 one output-path builder including every config axis; static gate rejects hardcoded result dirs.
- [ ] R7/R12/T7 per-cell immediate sync; two-location copy + md5 sample verify before any lease release.
- [ ] R8 no `_fixed/_v2/_backup/.bak` files; git is the archive.
- [ ] R10 hard gates (`exit 1`), never warnings (e.g. v3 `deploy` prints "Monitor FAILED" and exits 0 — forbidden).
- [ ] R11/T6 state machine with completion markers + non-empty output check + PAUSED sentinel.
- [ ] R14 (symlinks) resolve `readlink -f` before any `rm -rf` on shared hosts; (SSH) no retry loop faster than 1/180 s, one ControlMaster per host; storage triage: nothing only on ephemeral scratch.
- [ ] P2 declare fresh-deploy-per-cell vs deploy-once per experiment (APC state carries across cells otherwise).
- [ ] P4/P5 md5 post-rsync; per-cell output verify (T1 `verify_output_archive.py`).
- [ ] v3 lessons: gates read EFFECTS not declarations; every gate has a MUST-FAIL self-test; a gate that cannot obtain its evidence FAILS; one tree only (sibling `Block_*` trees ⇒ fatal, `lib_serve_v3.sh:20-23`, `_verify_tree.sh:4-5`); contamination monitor absent ⇒ not clear.

---

## 7. Hazards for a porter

- **H1 Two different "setup.sh" and neither is in OLD.** The v3 env came from `WSW/setup_jul25.sh`; `V3/block/exp/<codename>/setup.sh` is an older recipe (anonymised clone URLs `https://github.com/anon/…` at `:1-3`, unpinned `pip3 install torch torchvision` `:48`, `pip install --upgrade torch` `:57-58`, `git checkout <codename>` branch of the code repo, nixl/onnx installs `:111-120`) that did not produce the v3 freeze; OLD `cloudlab/setup.sh` is for a different project (dodoor). `V3X/REPRODUCE.md:31` itself says "setup.sh was not captured".
- **H2 Wrong-branch trap.** A checkout of a scrubbed/nonexistent branch name silently fell through to `main` for a whole night (`WSW/ENVIRONMENT/README.md` trap 1). Always `git checkout <SHA>` and assert `git rev-parse HEAD`.
- **H3 Scheduler head co-location.** v3 scheduler shared a 16-core V100 node with `14B_0`; scheduling overhead 55–500 ms per request depends on it; BEST-Route sq/po2 collapse at λ>=18 may be partly a head-CPU artefact (`2026-08-02_CR_CORRECTION_LIST.md:111-130`, `2026-09-01_NEXT_CAMPAIGN_HANDOFF.md` §3). The deploy layer must make the head host explicit (and optionally pin CPUs), and the parity manifest must record it.
- **H4 TP is not what the config says.** `tensor-parallel-size` in `model_deployment.json` is ignored; TP = `torch.cuda.device_count()` on the host (`deploy:138,192`). The v3 head snapshot shows 2 V100 lines (`head_snapshot.txt:11-12`) while the docs say V100x4; 14B fp16 (~29.5 GB) cannot fit 2x16 GB at util 0.85, so either the snapshot is truncated or that node differed. Record the actual per-host GPU count and TP in the execution manifest; make TP explicit in the new renderer (owner decides whether to keep "all visible GPUs").
- **H5 Port 7000 is fiction; 8000 is real.** See §2.2/§2.7.
- **H6 Silent defaults at the join.** host_configs defaults (`""`, 8000, [8300]), deploy defaults (util 0.9, len 8192), feature defaults (0.0), `predictor_ports` filtering — all silent. The new loaders must reject missing keys; but equivalence requires reproducing the v3 VALUES (write them explicitly into `configs/v3/`).
- **H7 Benchmark semantics live in the fork.** Shuffle-by-seed, chat templating by the benchmark client, `is_valid_sequence` pruning (prompt > 1024 tokens or prompt+output > 2048 dropped, tokenized with the Qwen2.5-3B tokenizer), `--max-total-len` (a fork flag), `--max-output-len` alias. Rewriting the client "cleanly" changes the workload. The OLD repo tracks a 3634-prompt file; v3 used the 3534-prompt eval file (`data/<codename>/best-route-v3-test-3534-eval.jsonl`, MD5 pinned by `_pin.sh:10`).
- **H8 Prompt-format round trip.** benchmark wraps with the chat template → scheduler unwraps unconditionally (`<codename>_serve.py:80,114,3059`) → `--chat` sends `messages` to `/v1/chat/completions` of BASE (non-Instruct) Qwen2.5 models. The earlier "chat skew" defect shared a bias across arms (`preflight_gate.py` docstring). Keep the round-trip identical and tested.
- **H9 Dead config keys that look alive.** `model_kv_config.kv_bytes_per_token` has 0 readers in V3 and its values are internally inconsistent with the Qwen2.5 architectures (3B 55,296 vs 36,864 computed; 7B 114,688 vs 57,344; 14B 393,216 vs 196,608; 72B 327,680 matches) — wiring it up (the README's "projected memory gain") would change the balance term; `embedding_model` unread (pkl decides); `tpot_lookup` read only on the static/fallback path and refuses a 0.05 default (`<codename>_learned_predictor.py:324-338`); `vllm_params.port`/`tensor-parallel-size` unread. Report to the owner; do not "fix".
- **H10 The fork's `/instance_stats` mutates state.** Reading it auto-extends `predicted_decode_tokens` (`scheduler.py:1309-1310,1326-1327`), so polling frequency changes subsequent `pending_decode_tokens`. Any change in how often the scheduler (or a monitor) polls is an execution change that differs between arms (sq/po2 poll per request, rr does not).
- **H11 Fail-loud vs era masking.** `slo_defaults.stats_blind_absorb` reverts telemetry failures to `{}` for ALL fetch sites (`<codename>_serve.py:968-975,2737-2742`); it was used only for the "brfill" exploratory cells. Port it as an explicit, logged, parity-manifest-visible mode or not at all.
- **H12 Env-var gates and renames.** V3 behaviour switches on env vars (`<CODENAME>_INPROC_PREDICTOR` default ON, `_STATIC_LAT`, `_STATIC_TPOT_HEAD`, `_TPOT_DIR`, `_MODELS_DIR`, `_HOST_GPU_MAP`, `_BATCH_WORKERS`, `_PREPPER_WORKERS`, `_DUMMY_ESTIMATOR`, `_DUMMY_SIDECAR`, `_DEBUG_TPOT`, `_ASYNC_REQUEST_FUNCS`, `_ROUTELLM_SRC`, `_AVENGERS_REPO`). Renaming the prefix makes every old launcher's export inert — exactly the "static arm exported to a tree with no reader" defect that invalidated three campaigns (`tpot_gate.py` docstring). Prefer CLI/config over env; where env stays, the static gate must prove every exported var has a reader and the EFFECTIVE_CONFIG line must echo it. `<CODENAME>_UNWRAP_PROMPT` is already an inert export in v3 launchers.
- **H13 Artifacts are outside git and paths are absolute.** XGBoost boosters at `/users/<user>/Block/models/...` (absolute), KNN pkl (24 MB, was missing locally until the audit — `AUDIT.md:138-140`), `best_route_4way_qwen` (552 MB), `avengers_4model/`; bundle `v3_repro_bundle.tar.zst` + Google Drive pack. The new repo must fetch/verify them by MD5 (`_pin.sh`, `_verify_artifacts.sh`) from a documented location; never embed absolute paths.
- **H14 Precompiled vLLM wheel availability.** `VLLM_USE_PRECOMPILED=1` downloads the upstream wheel for the fork's merge base (2025-11-26). If it is no longer hosted, the fallback is a full CUDA source build (hours per node, needs nvcc 12.8 + ninja; `TORCH_CUDA_ARCH_LIST` must cover 7.0/7.5/8.0). Test this on dev-gpu-wd312 BEFORE the 2026-09-28 lease.
- **H15 System Python + sudo pip.** v3 installed into system Python (both `pip` and `sudo pip`, `setup_jul25.sh:75,78`). Mixed user/system site-packages can shadow each other (transformers pinned at both levels for that reason). A venv move is cleaner but is an execution change to record.
- **H16 XGBoost must stay on CPU.** With CUDA visible, 1-row predicts dispatch to the GPU and wait ~800 ms behind vLLM (`deploy:275-280`); v3 sets `CUDA_VISIBLE_DEVICES=""` for the scheduler (`lib_serve_v3.sh:158`) and OMP/MKL/OPENBLAS=1 for sidecars. On a co-located head, a missing `CUDA_VISIBLE_DEVICES=""` also steals GPU memory from `14B_0`.
- **H17 Log file append + fixed name.** `logging.basicConfig(filemode='a+', filename=experiment_output/logs/<codename>_serve.log)` accumulates across restarts; gates that grep "the" log can read a previous run's EFFECTIVE_CONFIG. Use a per-run log path.
- **H18 Tests give false comfort.** The 12 V3 unit-test files cannot import; do not count them as coverage; the v3 gate scripts are the real regression suite and several hardcode personal/cluster paths (`static_defect_gate.py` `ROOT`, `check_zero_safe.py` `AGG`).
- **H19 Docs disagree with each other.** Examples: `/mydata` on A30 (yes in `setup_jul25.sh`, no in `V3/CLAUDE.md`), head role (committed vs uncommitted `ENVIRONMENT.md`), hardware sizes (OLD 80 GB/32 GB vs v3 40 GB/16 GB), KNN k (OLD 10 vs v3 50), transformers pin (global CLAUDE.md 4.50.3 vs v3 4.56.0). Trust the captured freeze/configs/logs; write one `docs/ENVIRONMENT.md` that cites them.
- **H20 Codename leaks through identifiers, not prose.** Module paths, env prefixes, log file names, decision-log file names (`decisions_<codename>_*.jsonl`), backend name `--backend <codename>`, scheduling mode `--scheduling <codename>`, instance-type strings, the fork branch name, `vllm-<codename>` directory name, HF token names (`token_for_<codename>_testing` appears in trap 4 text). The hygiene gate must scan identifiers and filenames, and fixtures generated from V3 must be scrubbed.
- **H21 Secrets on the wire.** HF token via ssh command line and `.bashrc` (§4.4). New deploy must read tokens from a 0600 file on the node (or HF cache login) and never echo/log them; hygiene gate scans logs committed as fixtures.
- **H22 Imminent lease.** Next CloudLab booking 2026-09-28 → 2026-10-08 (`2026-09-01_NEXT_CAMPAIGN_HANDOFF.md` §1); anything not smoke-validated (install, fork contract, benchmark sampler equivalence) before then will be debugged on the lease.
