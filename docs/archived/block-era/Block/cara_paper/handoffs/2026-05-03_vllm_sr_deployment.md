# vLLM-SR Deployment Guide — Cara Cluster (CloudLab asdwb-302610)

**Last verified working: 2026-05-03 00:11 UTC, node5 (`d8545-10s10301`)**

This document captures the canonical deployment we hit running vllm-project/semantic-router (real upstream, not the proxy stand-in) on our 4-Qwen / 18-endpoint cluster. **Read this before re-attempting vllm-sr setup** — there are several non-obvious gotchas in the upstream tooling.

---

## TL;DR

Three components must run, all on the same Docker network:
1. **vllm-sr-container** — router decision service (gRPC ext_proc on `:50051`, HTTP API on `:8080`)
2. **Envoy** — proxy on `:8801` that intercepts client requests, calls router via ext_proc, then forwards to chosen Qwen vLLM cluster
3. Optional observability (grafana, prometheus, jaeger) — not needed for benchmarking

The pip-installed `vllm-sr` CLI v0.2.0 starts (1) but **NOT (2)** — Envoy must be launched manually via `docker run`. This is canonical upstream behavior at v0.2.0.

---

## Prerequisites on cluster scheduler node (e.g. node5 = `d8545-10s10301`)

```bash
# 1. Docker installed (Ubuntu 22.04)
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
sudo usermod -aG docker asdwb
# Point Docker storage to /mydata (root disk is only 63 GB; /mydata is 1.5 TB)
sudo mkdir -p /mydata/docker
echo '{"data-root":"/mydata/docker"}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker

# 2. Pip-installed vllm-sr (Python venv on /mydata to save root disk)
mkdir -p /mydata/vllm_sr
python3 -m venv /mydata/vllm_sr/venv
source /mydata/vllm_sr/venv/bin/activate
pip install --upgrade pip
pip install vllm-sr   # v0.2.0 currently
```

---

## Critical gotcha #1: Schema is v0.3, but `vllm-sr init` generates v0.1

The CLI's `init` command writes a v0.1-schema config.yaml to disk. **The Docker image that gets pulled is v0.3 and rejects v0.1 fields:**

```
runtime_config_load_failed: deprecated config fields are no longer supported:
  providers.default_model, providers.models[*].endpoints
```

There's also a CLI/image schema mismatch: `vllm-sr validate` validates against v0.1 (CLI) but the runtime container expects v0.3.

**Workaround**: write a v0.3 config from scratch (see template below). Skip `vllm-sr validate`. The container will still accept it.

---

## Critical gotcha #2: Pip CLI v0.2.0 doesn't start Envoy

In `/mydata/vllm_sr/venv/lib/python3.10/site-packages/cli/commands/serve.py:253`:

```python
# TODO: Start services
log.info("Next steps (to be implemented):")
log.info("  1. Start Envoy proxy")
log.info("  2. Start Router service")
log.info("  3. Wait for health checks")
```

`vllm-sr serve` only launches the router-decision container (vllm-sr-container). **Envoy must be launched separately**.

The upstream `deploy/docker-compose/docker-compose.yml` referenced in their `installation/docker-compose.md` docs **does not exist on the current main branch** (only in v0.1 archived docs). `deploy/local/envoy.yaml` exists but is hardcoded for cloud APIs (Anthropic, OpenAI), not local Qwen pools.

So we wrote our own Envoy config + launched Envoy as a Docker container manually.

---

## Critical gotcha #3: Disk pressure on root

Docker images + auto-downloaded ML models (~1.5 GB) + observability stack add up. **Always set `data-root: /mydata/docker`** before starting Docker. Without this, root will run out of space (we hit 89% before the cleanup; the CUDA repo `/var/cuda-repo-ubuntu2204-12-8-local` was also deletable for 3.9 GB recovery).

---

## Working configuration files

### `/mydata/vllm_sr/config.yaml` (vllm-sr router config, v0.3)

```yaml
version: v0.3

listeners:
  - name: http-8888
    address: 0.0.0.0
    port: 8888
    timeout: 1200s

providers:
  defaults:
    default_model: Qwen/Qwen2.5-3B
    default_reasoning_effort: low
  models:
    - name: Qwen/Qwen2.5-3B
      provider_model_id: Qwen/Qwen2.5-3B
      pricing: {prompt_per_1m: 2, completion_per_1m: 2, currency: USD}
      backend_refs:
        - {name: n3,  endpoint: "10.10.1.4:8000",  protocol: http, weight: 1}
        # ... 7 more 3B endpoints (10.10.1.{15,18,16,17,13,11,14}:8000)
    - name: Qwen/Qwen2.5-7B
      pricing: {prompt_per_1m: 3, completion_per_1m: 3, currency: USD}
      backend_refs:
        # ... 5 endpoints (10.10.1.{2,9,7,8,5}:8000)
    - name: Qwen/Qwen2.5-14B
      pricing: {prompt_per_1m: 5, completion_per_1m: 5, currency: USD}
      backend_refs:
        # ... 3 endpoints (10.10.1.{12,1,3}:8000)
    - name: Qwen/Qwen2.5-72B
      pricing: {prompt_per_1m: 20, completion_per_1m: 20, currency: USD}
      backend_refs:
        - {name: n5, endpoint: "10.10.1.6:8000",  protocol: http, weight: 1}
        - {name: n9, endpoint: "10.10.1.10:8000", protocol: http, weight: 1}

routing:
  modelCards:
    - {name: Qwen/Qwen2.5-3B,  param_size: 3B,  capabilities: [chat]}
    - {name: Qwen/Qwen2.5-7B,  param_size: 7B,  capabilities: [chat, reasoning]}
    - {name: Qwen/Qwen2.5-14B, param_size: 14B, capabilities: [chat, reasoning]}
    - {name: Qwen/Qwen2.5-72B, param_size: 72B, capabilities: [chat, reasoning, long-context]}
  decisions:
    - name: default-route
      priority: 100
      rules: {operator: AND, conditions: []}
      modelRefs:
        - {model: Qwen/Qwen2.5-3B,  use_reasoning: false}
        - {model: Qwen/Qwen2.5-7B,  use_reasoning: false}
        - {model: Qwen/Qwen2.5-14B, use_reasoning: false}
        - {model: Qwen/Qwen2.5-72B, use_reasoning: true}

global:
  router:
    strategy: priority
```

### `/mydata/vllm_sr/envoy_qwen.yaml` (custom Envoy config, full file at `~/Code/llm/Block/cara_paper/handoffs/2026-05-03_envoy_qwen.yaml.template`)

Key sections:
- `listeners[0]` on port 8801
- `http_filters: [ext_proc → router gRPC at vllm-sr-container:50051, router]`
- `clusters`: 4 STATIC clusters (qwen{3b,7b,14b,72b}_cluster) each listing the corresponding 8/5/3/2 endpoints by `10.10.1.X:8000`
- `clusters[ext_proc_cluster]` with `http2_protocol_options: {}` (REQUIRED for gRPC) pointing to `172.18.0.5:50051` (vllm-sr-container's IP on `vllm-sr-network`)
- Routes match on `x-selected-model` header value → cluster

The router (via ext_proc gRPC) sets the `x-selected-model` response header on each request, which Envoy reads to pick the cluster.

---

## Launch sequence (verified working)

```bash
# A. Start vllm-sr router stack (CLI launches router + observability containers, NOT envoy)
cd /mydata/vllm_sr
nohup /mydata/vllm_sr/venv/bin/vllm-sr serve --config config.yaml > /tmp/vllm_sr_serve.log 2>&1 &
# Wait ~60s for image pull + ML model download. Verify:
docker ps   # should show: vllm-sr-container, vllm-sr-grafana, vllm-sr-prometheus, vllm-sr-jaeger

# B. Find vllm-sr-container's IP on the network
docker network inspect vllm-sr-network --format '{{json .Containers}}' \
  | python3 -m json.tool | grep -A1 vllm-sr-container | grep IPv4
# Example output: "IPv4Address": "172.18.0.5/16"
# UPDATE the ext_proc_cluster endpoint in envoy_qwen.yaml to match this IP.

# C. Start Envoy as a separate Docker container on the same network
docker run -d --name vllm-sr-envoy \
  --network vllm-sr-network \
  -p 8801:8801 -p 19000:19000 \
  -v /mydata/vllm_sr/envoy_qwen.yaml:/etc/envoy/envoy.yaml:ro \
  envoyproxy/envoy:v1.31-latest \
  /usr/local/bin/envoy -c /etc/envoy/envoy.yaml \
    --component-log-level ext_proc:info,http:warn

# D. Verify
curl -s --max-time 10 http://127.0.0.1:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MoM","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":30}'
# Expected response includes "model":"Qwen/Qwen2.5-3B" + actual generated text.
```

---

## Metric gap with bench (`benchmark_serving.py`) — what's actually required

`--backend openai-chat --base-url http://localhost:8801 --endpoint /v1/chat/completions` works for vLLM-SR but misses several fields that the cara backend captures. **Audit against the paper draft (`route-balance/nips_draft/neurips_2026.tex`):**

| Field | openai-chat populates? | Required by paper figures/tables? | Action |
|---|---|---|---|
| ttft / itl / e2el | ✅ | ✅ Pareto plots (`fig:pareto`), CAP radar (`fig:radar`), tab:extremes | none |
| **model** | ❌ EMPTY | ✅ **REQUIRED** — without it we can't compute DeepEval quality (need to look up `models[chosen].deepeval_score` by request_id × chosen model) or cost (`cost_per_token[chosen] × output_len`). The Pareto y-axis (DeepEval) and the cost axis on the radar both depend on this. | **MUST fix** |
| scheduling_overhead | ❌ 0.0 | 🟡 nice-to-have per user direction. No dedicated figure in current draft, but useful for §App overhead discussion. | derive |
| host / instance_id | ❌ EMPTY | ❌ not in any current paper figure/table | skip |
| predicted_quality / predicted_best_model / all_model_scores | ❌ | ❌ used only in cara-internal ablation tables (`tab:bucket_results`, `tab:calibration`, `tab:routing_analysis`); vllm-sr doesn't have these predictors anyway → not comparable, not relevant | skip |

### Required: capture the `model` field for vLLM-SR cells

**Option 2 (chosen path) — extract `x-upstream-host` Envoy header + static host→model lookup**:

In our deployment, each of the 18 vLLM endpoints serves exactly one Qwen size (per `block/config/cara/model_deployment_apr26_phase2_deployed.json`):

| host (10.10.1.X:8000) | Qwen size |
|---|---|
| .4, .15, .18, .16, .17, .13, .11, .14 | 3B (8 instances) |
| .2, .9, .7, .8, .5 | 7B (5 instances) |
| .12, .1, .3 | 14B (3 instances) |
| .6, .10 | 72B (2 instances) |

So if we add `x-upstream-host` to Envoy's response headers, we get **both** which instance handled the request AND which model it was — by static lookup. Two pieces of info from one header, no body parsing.

**Steps:**
1. In `envoy_qwen.yaml`, add `response_headers_to_add` to each route to capture upstream host:
   ```yaml
   route:
     cluster: qwen3b_cluster
     timeout: 300s
     # %UPSTREAM_HOST% gets filled by Envoy with the actual upstream IP:port chosen by load-balancer
   response_headers_to_add:
     - header: {key: "x-upstream-host", value: "%UPSTREAM_HOST%"}
       append_action: OVERWRITE_IF_EXISTS_OR_ADD
   ```
2. Add a custom `vllm-sr` backend in `block/benchmark/cara/cara_end_point_func.py` (clone `async_request_openai_chat_completions`):
   - Read `response.headers.get("x-upstream-host")` → "10.10.1.X:8000"
   - Static dict `HOST_TO_MODEL = {...}` (built from the deployment config) maps host → Qwen model name
   - Write both into `RequestFuncOutput.host` and `RequestFuncOutput.model`
3. Run cells with `--backend vllm-sr`

**Option 1 (alternative, body-parse)**: parse `response.json()["model"]` from the chat-completion body. Captures `model` only, not `host`. Simpler but loses the upstream-host info.

**Option 3 (post-process)**: Envoy access logs already have `upstream_host` in the JSON format we configured. Join by request_id after the fact. Avoids any bench changes but requires a post-processor stage.

### Nice-to-have: scheduling_overhead for vLLM-SR

vLLM-SR doesn't expose this through the OpenAI-compat API — the router's internal timing stays in its logs. **Derive it post-hoc**: `scheduling_overhead ≈ e2el − vllm_inference_time`, where `vllm_inference_time = ttft + (output_tokens − 1) × mean(itl)`. The bench already captures all the inputs needed; the post-processor computes overhead per request.

Alternative: the router exposes Prometheus metrics on `:9190/metrics` that include per-request routing decision time. If the experiment includes Prometheus scraping, those are available. Lower priority — the derived approach is sufficient for paper-grade reporting.

---

## When something goes wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| `vllm-sr serve` works but `/v1/chat/completions` 8888 returns nothing | Envoy not running | manually launch envoy Docker per step C above |
| Envoy container exits immediately | YAML parse error | `docker logs vllm-sr-envoy` to see Envoy startup error |
| Envoy starts but ext_proc 503s | router IP wrong in `ext_proc_cluster` | re-check with `docker network inspect vllm-sr-network` |
| `runtime_config_load_failed: deprecated config fields` | v0.1 schema | rewrite as v0.3 (see template above) |
| Disk fills up rapidly | `data-root` not set | `/etc/docker/daemon.json` → `{"data-root":"/mydata/docker"}` then `systemctl restart docker` (will lose existing images) |

---

## Stop / cleanup

```bash
docker stop vllm-sr-envoy && docker rm vllm-sr-envoy
/mydata/vllm_sr/venv/bin/vllm-sr stop   # stops router + observability containers
```

To reclaim disk:
```bash
docker system prune -a   # nukes images + caches
```
