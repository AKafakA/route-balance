> **SUPERSEDED 2026-05-16 by `model_training_index.md`.** This first-pass index has known errors (e.g. it lists GPU-VM `length_bucket/` as "keep — referenced by code"; that path was a Mar-27 fallback stub — the real deployed length_bucket is VPS-side). Trust `model_training_index.md` + `gpu_vm_ARCHIVE_MANIFEST.txt` for authoritative state.

# GPU VM `cara_exp` Index — dev-gpu-wd312

Generated 2026-05-16 (UTC) during a conservative cleanup pass.

## Machines & paths

- **GPU VM (training/eval host)**: `wd312@dev-gpu-wd312.cl.cam.ac.uk`
  - Repo root: `/home/wd312/Code/llm/Block` (on `kimana` NFS — was **97% full**, hence this cleanup)
  - Local scratch (empty, 60 GB free): `/local/scratch/tmp/wd312/`
- **VPS (code hub)**: `wd312@openclaw` — `/home/wd312/Code/llm/Block` (this index mirrored here under `cara_paper/claude/`)
- The **deployment** machine is the CloudLab cluster, *not* this GPU VM. The GPU VM holds *training-side* artifacts; deployed model paths in `block/config/cara/predictor_deployment_config.json` (e.g. `models/cara/latency/deploy_{e2e,tpot,ttft}`) point at cluster paths under `/users/asdwb/Block/...`.

## `models/cara/` — REQUIRED (kept in place)

Determined from `block/config/cara/predictor_deployment_config.json` + route_balance predictor code grep (`models/cara/latency/deploy_tpot`, `length_bucket/deploy/fused_config.json`, `quality/judge/deploy/fused_config.json`) and `MEMORY.md`:

| dir | size | why kept |
|---|---|---|
| `deepeval_models/` | 5.2 G | multi-target deepeval predictors (MEMORY: predictor of record family) |
| `roberta_fused_5ep/` | 1.9 G | paper's pre-refactor predictor baseline (predictor-arch table); MEMORY says archived-but-cited |
| `length_bucket/` | 571 M | **referenced by route_balance code** (`length_bucket/deploy/fused_config.json`) |
| `length_regression/` | 570 M | recent (May 6); kept pending confirmation it is/ isn't superseded |
| `lora_per_model/` | 94 M | LoRA adapters — slow to retrain, MEMORY says keep |
| `latency/` | 3.6 M | **referenced by deployed config** (`latency/deploy_{e2e,tpot,ttft}`) |
| `xgboost_enriched/`, `_tpot/`, `_ttft/` | ~3 M | XGBoost latency variants (Apr 9) |

## `models/cara/` — ARCHIVED (moved to `cara_exp/archive/models/cara/`, NOT deleted)

| dir | size | reason |
|---|---|---|
| `lstm_v2_enriched/` | 762 K | LSTM predictor not used (MEMORY) |
| `knn_with_ref/` | 23 M | superseded Mar-27 KNN-with-reference variant |
| `qwen_lora_fused/` | 36 M | superseded by `lora_per_model/` |
| `knn/` | 23 M | old Mar-27 KNN (deployed estimator is the k=50-weighted head inside length_bucket/quality, not this dir) |

Full record: `cara_exp/ARCHIVE_MANIFEST.txt` on the GPU VM.
**Restore any item:** `mv cara_exp/archive/models/cara/<name> models/cara/<name>`

## `data/cara/` — kept in place (not touched this pass)

- `scored/` 102 M (Apr 7) — likely reproducible from HF `asdwb/cara_model_estimator`
- `latency_data/` 215 M (Apr 11) — **probably the only good copy** (HF `asdwb/cara_latency_prediction` is buggy/needs re-upload per MEMORY) — do NOT archive without re-upload
- `best-route-v3-test-500.jsonl` 207 K (Apr 2) — clean bench

## Scripts (kept in place, not touched)

- `scripts/compute_mape.py`, `scripts/fix_safety_scores.py`
- `experiments_analysis/{auto_provision_plot,experiment_plot,prediction_plot}.py`

## Status of this pass

- **Conservative only**: 4 clearly-dead model dirs archived (~82 MB), nothing deleted, all reversible.
- Deferred (ambiguous, left in place pending user call): `roberta_fused_5ep` (1.9 G), `length_regression` vs `length_bucket` which-is-canonical, `data/cara/scored` HF-reproducibility.
- NFS still ~97% full — a moderate pass on `roberta_fused_5ep` / `deepeval_models` would reclaim real space but needs explicit per-artifact sign-off.
