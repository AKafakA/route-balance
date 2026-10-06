# Stage 1: environment

## Analysis (CPU only)

Stage 3 needs Linux, Bash and CPython 3.10. Create a new virtual environment for it. Never install into an
environment that serves runs.

```bash
uv venv --python 3.10 .venv-uvicorn
uv pip install --python .venv-uvicorn/bin/python -e ".[dev]"
make check
```

`make check` runs offline, on the CPU, with no network: lint, the hygiene gate, the static defect gate, the unit
tests, the contract tests against a fake vLLM, and the campaign gates' tests (`gates.md`). It must pass before
anything else is trusted.

The paper data pack carries its own pinned analysis environment (`requirements.txt` in the pack) and needs only
that, not this repository.

## Serving cluster (stage 2 only)

The runs used 13 bare-metal CloudLab nodes of three types. `routebalance/hw.py` is the only map from node type to
GPU, and the fleet files place every instance:

| node type | GPUs | Qwen fleet (`configs/fleet_qwen.json`) | Gemma fleet (`configs/fleet_gemma.json`) |
|---|---|---|---|
| d8545 (2 nodes) | 4x A100 | Qwen2.5-72B, one instance per node | Gemma 3 27B and 12B |
| c4130 (3 nodes) | 4x V100 | Qwen2.5-14B, one instance per node | |
| d7525 (8 nodes) | 1x A30 | Qwen2.5-7B (5) and Qwen2.5-3B (3) | Gemma 3 4B and 1B |

The scheduler runs on one of the fleet's nodes, on its CPUs. Every node runs Ubuntu 22.04.

### Hosts

Copy `configs/hosts.example.json` to an untracked file and give each node its address, its node type, and the
address of the experiment network the scheduler reaches the engines on (`service_address`). Every script and tool
takes this file as `--hosts`. The engines and the scheduler listen on every interface (vLLM's default, and the
launch file's `host`). Every run's traffic check compares each engine's serving counters with the run's own
requests, so a request from outside makes the run invalid (`gates.md`). In the paper's campaign it found outside
requests in three runs, which were rerun, and from 3 October a firewall rule on every node admitted the engines'
and the scheduler's ports only from the experiment network. New runs should restrict them the same way.

### Node setup

`scripts/setup.sh` prepares one node and `scripts/setup_fleet.sh` runs it on every node of a fleet, from the
operator's machine. The operator scripts run this repository's tools with the Python that `PY` names:

```bash
PY=.venv-uvicorn/bin/python scripts/setup_fleet.sh --fleet configs/fleet_qwen.json --hosts HOSTS.json \
  --engine-args configs/engine_args_qwen.json --scheduler-node <node> --hf-token-file <token file> \
  --tree <repository path on the nodes> --log-dir <local log dir> --hub configs/scheduler_hub.json
```

- **Engine nodes:** the NVIDIA driver and CUDA 12.8.0, then official vLLM 0.18.1 in a fresh virtual environment,
  and the weights of every model the node serves at the revision `configs/engine_args_*.json` pins. The Gemma 3
  weights are gated on the Hugging Face Hub: the token's account must have accepted their licence.
- **The scheduler node:** this repository with its pinned dependencies (`pyproject.toml`) in a fresh virtual
  environment, the pinned upstream AvengersPro and BEST-Route checkouts, and the encoders and tokenizers
  `configs/scheduler_hub.json` lists.
- **The vLLM Semantic Router** (`setup.sh --role vsr`, on the scheduler's node): Docker, vLLM-SR 0.4.0 and its
  images pinned by digest. `vllm-sr-bench/README.md` documents that study.

Every role ends with a self-check that prints the versions and exits non-zero on any mismatch. A node whose driver
needs a reboot exits 3: reboot it and run the same command again.

Next: [run the experiments](02-run-experiments.md).
