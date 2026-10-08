# Cara Cluster Deployment Guide

**Last verified working**: 2026-05-03 ~17:30 UTC, cluster `asdwb-302610`, 18 nodes.

This guide captures the *exact* invocation that produced a clean cara stack
after the May-3 multi-fix session. Run from the **VPS** (`openclaw`), not
from the cluster scheduler.

---

## 1. Pre-flight assumptions

- `~/Code/llm/Block/` on VPS has the latest patched code (cara_serve.py,
  cara_learned_predictor.py, deploy_cara.py) and the latest configs
  (`scheduler_config_caraknn.json`, `scheduler_config_cararoberta.json`,
  `predictor_xgboost_3model.json`, `model_deployment_apr26_phase2.json`).
- VPS user key is in each cluster host's `~/.ssh/authorized_keys` (CloudLab
  default for the experiment owner).
- All 18 hosts in `block/config/hosts` have:
  - `/users/asdwb/Block/` cloned (per host, via setup.sh on first acquire).
  - `/mydata/` mounted (per CloudLab profile: `/dev/sdb` for c4130/c240g5/d7525,
    `/dev/nvme0n1` for d8545; `mkfs.ext4` once on first acquire).
  - `~/vllm/` upstream-Block fork checkout.
  - vLLM `pip install` already done (CLI at `/usr/local/bin/vllm`).
- Scheduler hostname: `d8545-10s10301.wisc.cloudlab.us` (one of the 2 A100
  nodes; doubles as 72B serving host — `cara_serve` is CPU-only so no GPU
  contention).

## 2. Configs to verify before deploy

Source-of-truth: `cara_paper/active/deployment_spec.json`. Run
`scripts/audit_deployment_vs_spec.py` after deploy; expect 0 FAIL.

### Sidecar predictor: `block/config/cara/predictor_xgboost_3model.json`

```json
{
  "predictor_type": "xgboost_3model",
  "schedule_trace_timeout": 5,
  "filter_enabled": false,
  "xgboost_3model": {
    "max_batch": 128,
    "model_paths_by_instance": {
      "qwen2.5-3b_a30":   {"ttft": "...deploy_ttft/qwen2.5-3b_a30.xgb",  "tpot": "...", "e2e": "..."},
      "qwen2.5-3b_p100":  {...},
      "qwen2.5-7b_a30":   {...},
      "qwen2.5-14b_v100": {...},
      "qwen2.5-72b_a100": {...}
    }
  }
}
```

When `filter_enabled=false`: sidecars skip TTFT+TPOT loading entirely; only E2E
XGBoost is loaded per replica.

### Scheduler caraknn: `scheduler_config_caraknn.json`

- `model_estimator.bucket_config.fused_model_dir =
  /users/asdwb/Block/models/cara/length_bucket/deploy` (RoBERTa fused, encoder
  must be `roberta-base`).
- `model_estimator.quality_config = {type: knn, model_dir: quality/judge/deploy_knn,
  embedding_model: sentence-transformers/all-MiniLM-L6-v2, k: 10}`.
- `slo_defaults.filter_enabled = false`, `constraint_mode = NONE`.
- `slo_defaults.instance_capacity` and `instance_metadata` cover all 5 types
  (3b_p100, 3b_a30, 7b_a30, 14b_v100, 72b_a100). Each `instance_metadata`
  entry has `cost_per_input_token` + `cost_per_output_token` per Novita/Together
  rates.
- `bucket_config.model_map` covers all 4 Qwen sizes.

### Scheduler cararoberta: `scheduler_config_cararoberta.json`

Same as caraknn except `quality_config.type = fused_deepeval` and
`quality_config.model_dir = quality/judge/deploy_roberta`.

### Model deployment: `model_deployment_apr26_phase2.json`

- 14B has 3 hosts (c4130-110133, c4130-110233, c4130-110433).
- Pricing block per Qwen size: 3B Together, 7B/14B/72B Novita, 72B asymmetric
  (`input_per_1m=0.38, output_per_1m=0.40`).
- All 4 models have `pricing` block — this is what `cara_serve.py:2259-2280`
  injects into the per-instance cost computation.

## 3. Deploy command (run from VPS)

```bash
cd ~/Code/llm/Block
nohup python3 -u block/exp/cara/deploy_cara.py \
  --hosts block/config/hosts \
  --config block/config/cara/model_deployment_apr26_phase2.json \
  --scheduler-host asdwb@d8545-10s10301.wisc.cloudlab.us \
  --scheduler-config block/config/cara/scheduler_config_caraknn.json \
  --predictor-type learned \
  --learned-predictor-config block/config/cara/predictor_xgboost_3model.json \
  --host-config block/config/host_configs.json \
  -d model_instance predictor monitor scheduler \
  --parallel-deploy 18 \
  > /tmp/deploy_cara_$(date +%Y%m%d_%H%M).log 2>&1 &
disown
```

Expected runtime: 5-15 min (HF cache cold-start dominates first launch; warm
cache reduces to ~3 min).

## 4. What deploy_cara does, in order

1. Loads `host_configs.json` (per-host backend_port + ip_address +
   predictor_ports).
2. Loads `predictor_deployment_config.json` for sidecar misc settings
   (`num_predictors_per_instance: 4`, `data_collection_sample_rate`, etc.).
3. Loads `model_deployment_apr26_phase2.json` and processes each model
   sequentially. Within each model, all hosts deploy in parallel (limited by
   `--parallel-deploy`).
4. Per (model, host) tuple: cleanup → deploy vLLM → deploy 4 predictor
   sidecars → deploy monitor.
5. Finally deploys cara_serve scheduler on the host given by
   `--scheduler-host`. The scheduler launch script lives at
   `/tmp/start_cara_scheduler.sh` on the scheduler node — uses `nohup` +
   detach + `CUDA_VISIBLE_DEVICES=` (CPU-only RoBERTa) + writes log to
   `experiment_output/logs/cara_scheduler.log`.

## 5. Common pitfalls (and what was fixed in the May-3 session)

| Symptom | Root cause | Fix |
|---|---|---|
| Sidecars crash with `FileNotFoundError: predictor_xgboost_3model.json` | New configs not synced to all workers | Add rsync step before deploy_cara, OR push via git pull on each worker |
| Sidecars log `xgboost_3model ttft_model not found:` | Schema mismatch between config (uses `xgboost_ttft.model_dir`) and loader (expects `xgboost_3model.ttft_model`). Or instance_type doesn't match .xgb filename | Use `predictor_xgboost_3model.json` schema with `model_paths_by_instance` map. Confirm `--instance-type` arg matches `<model>_<gpu>` format |
| cara_serve dispatches to `http://:8000/...` (empty IP) | Missing `--host_config` arg | deploy_cara now appends it automatically (line 391) |
| cara_serve crashes with `unsupported operand type(s) for *: 'NoneType' and 'int'` on TTFT | Filter checks ttft when filter_enabled=false (sidecar returned no ttft) | `cara_serve.py:1054-1085` wrapped in `if _filter_on:` with None-guard |
| cara_serve OOM on RoBERTa load | A100 GPUs saturated by 72B vLLM (gpu-mem-util=0.95) | `CUDA_VISIBLE_DEVICES=` exported before launch — RoBERTa runs on CPU |
| Scheduler lands on c4130-110233 (first host in `block/config/hosts`) instead of d8545-10s10301 | Default `--scheduler-host` is first host in file | Always pass `--scheduler-host asdwb@d8545-10s10301.wisc.cloudlab.us` explicitly |
| 14B deploy only on 2 hosts | `model_deployment.json` had 2 hosts | Verify `node_hosts` has 3 entries before deploy (110133, 110233, 110433) |

## 6. Switching between caraknn and cararoberta phases

After cara-knn-full phase completes, switch quality predictor without
redeploying vLLM/sidecars:

```bash
ssh asdwb@d8545-10s10301.wisc.cloudlab.us \
  "curl -X POST http://127.0.0.1:8200/v1/config \
    -H 'Content-Type: application/json' \
    -d '{\"scheduler_config_path\": \"block/config/cara/scheduler_config_cararoberta.json\"}'"
```

Or just relaunch scheduler via `deploy_cara -d scheduler --scheduler-config
.../scheduler_config_cararoberta.json`.

## 7. Verifying sidecar models loaded correctly

After deploy:

```bash
ssh asdwb@d7525-10s10311.wisc.cloudlab.us \
  'grep "loaded: " /users/asdwb/Block/experiment_output/logs/predictor_8100.log | tail -1'
```

Expected (filter off, e2e only):
```
XGBoost3ModelSidecarPredictor[qwen2.5-7b_a30] loaded: ttft=False tpot=False e2e=True max_batch=128
```

For all 18 hosts in one shot:
```bash
for h in $(grep -oE "[a-z0-9-]+\.wisc\.cloudlab\.us" block/config/hosts | sort -u); do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=4 asdwb@$h \
    'grep "loaded: " /users/asdwb/Block/experiment_output/logs/predictor_8100.log | tail -1')
  echo "$h: $out"
done
```

## 8. Pre-bench warmup

`cara_paper/smoke_test_apr_13/scripts/warmup_cell.sh` sources a 100-prompt
static set (disjoint from the 3534-prompt eval). Run before each timed cell:

```bash
ssh asdwb@d8545-10s10301.wisc.cloudlab.us \
  'cd /users/asdwb/Block && \
   source cara_paper/smoke_test_apr_13/scripts/warmup_cell.sh && \
   warmup_cell http://127.0.0.1:8200 12'  # last arg = upcoming λ
```

Warms RoBERTa (CPU first-inference ~8s), KNN, sidecar batchers, vLLM
prefix-cache.

## 9. Smoke + λ-grid bench

After warmup, run a single cell on the cluster scheduler (loopback to
cara_serve, no VPS-cluster network in the latency path):

```bash
ssh asdwb@d8545-10s10301.wisc.cloudlab.us \
  'cd /users/asdwb/Block && \
   python3 -m block.benchmark.cara.benchmark_serving \
     --backend cara --base-url http://127.0.0.1:8200 --endpoint /v1/completions \
     --model Qwen/Qwen2.5-3B --tokenizer Qwen/Qwen2.5-3B \
     --dataset-name custom --dataset-path data/cara/best-route-v3-test-3534-eval.jsonl \
     --num-prompts 3534 --request-rate <LAMBDA> --seed <SEED> \
     --result-dir <DIR> --result-filename <NAME>.json \
     --save-result --save-detailed'
```

Check `errors=0` in the resulting JSON before promoting to the active
results tree. The cron strict-halt at
`cara_paper/smoke_test_apr_13/scripts/cron_overnight.sh` enforces this.

## 10. Tearing down

```bash
ssh asdwb@d8545-10s10301.wisc.cloudlab.us 'pkill -f cara_serve'
for h in $(grep -oE '[a-z0-9-]+\.wisc\.cloudlab\.us' block/config/hosts | sort -u); do
  ssh asdwb@$h 'pkill -f cara_predictor_api_server; pkill -f vllm.entrypoints' &
done; wait
```

Tearing down the cluster reservation? Make sure to rsync `/mydata/phase_*/`
off-cluster first if you want to keep training data — `/mydata/` is wiped on
release (not reboot, but release).
