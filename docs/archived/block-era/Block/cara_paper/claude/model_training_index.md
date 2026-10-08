# Cara Model Training Index — params, time, paper-usage, keep/remove

Generated 2026-05-16 (UTC). Source: each model's `training_results.json` / `log_history.json` on `dev-gpu-wd312`.
Policy (user, 2026-05-16): keep paper-used models; for NOT-used models remove only if (a) training params recorded here AND (b) training short. Keep long-trained models even if unused (e.g. Qwen LoRA).

| model dir | size | train time | key params (recorded → retrainable) | in paper? | verdict |
|---|---|---|---|---|---|
| `lora_per_model/` (reference_score 7B+72B) | 94 M | **48 min** (2865 s) | base `Qwen2.5-0.5B`, `lora_r=16`, `lora_alpha=32`, epochs 3, 14 919 ex, target `reference_score`, mode per_model | not deployed (KNN is) | **KEEP — long train (48 min); user rule: Qwen LoRA stays** |
| `roberta_fused_5ep/` | 1.9 G | **9 min** | encoder RoBERTa-base, fused 4-head, 5 epochs, lr warmup ~3.4e-6; eval_loss: reference_score 0.071 / length 0.767 / length_bucket 1.73 / judge_class 1.17 | cited as pre-refactor predictor baseline (predictor-arch table) + #107 in-progress eval | **KEEP pending #107 — short (9 min) but actively in the in-progress predictor eval; remove only after #107 closes** |
| `deepeval_models/modernbert_fused_deepeval/` | 3.8 G (final .pt 596 MB; ~3.2 G is checkpoint bloat) | part of ~26 min | encoder `answerdotai/ModernBERT-base`, fused, deepeval target | not deployed; #107 uses it | KEEP final .pt + JSONs (used by #107); **bloat (~3.2 G non-final) removable** |
| `deepeval_models/roberta_fused_deepeval/` | 478 M | part of ~26 min | RoBERTa-base fused, deepeval target, params recorded | not deployed; #107 ref | keep final .pt until #107 closes |
| `deepeval_models/roberta_multitarget_A/` | 479 M | recorded `train_time_s` | RoBERTa multitarget (length_bucket+length_log+deepeval), encoder/targets recorded | **no** (abandoned multitarget experiment) | **REMOVE-eligible — not in paper, params recorded, short** |
| `deepeval_models/roberta_multitarget_B/` | 478 M | recorded `train_time_s` | RoBERTa multitarget (length_bucket+deepeval), recorded | **no** | **REMOVE-eligible — not in paper, params recorded, short** |
| `length_regression/deploy/` | 570 M | **13.5 min** (810 s) | encoder `ModernBERT-base`, fused, target length; MAE per Qwen recorded; **byte-identical fused_model.pt to VPS copy** | superseded by VPS | **REMOVE-eligible on GPU VM — redundant (VPS has identical copy), params recorded, 13 min** |
| `length_bucket/` (GPU VM) | 571 M | — | only `.fallback_modernbert_mar27` stub, no deploy/ | VPS canonical | **REMOVE-eligible — Mar-27 fallback stub, real one on VPS** |
| `latency/` (GPU VM) | 3.6 M | seconds (XGBoost) | old Mar-27; VPS has May-3 deployed | superseded | REMOVE-eligible (tiny, regenerable) |
| `xgboost_enriched{,_tpot,_ttft}/` | ~3 M | seconds | XGBoost latency variants Apr-9 | superseded by VPS latency_predictors_apr28 | REMOVE-eligible (tiny, fast retrain) |
| **archived already** `knn/`,`knn_with_ref/`,`qwen_lora_fused/`,`lstm_v2_enriched/` | 82 M | — | in `cara_exp/archive/` (manifest) | no | already moved (reversible) |

## Net reclaim if all REMOVE-eligible deleted (after #107 closes for the roberta_fused/modernbert items)
- multitarget_A+B: ~0.96 G
- length_regression (GPU, redundant): 0.57 G
- length_bucket stub + latency + xgboost_enriched*: ~0.58 G
- modernbert checkpoint bloat: ~3.2 G
- → ~5.3 G off the 97%-full `kimana` NFS. (Archiving on same NFS does NOT free space — only `rm` does.)

## Hard keeps
- `lora_per_model/` — 48 min retrain, user rule.
- VPS `models/cara/{latency_predictors_apr28, latency, length_bucket, best_route_4way_qwen, avengers_4model*}` — canonical deployed/baseline artifacts (NOT on GPU VM; do not touch).
- `data/cara/latency_data/` — likely only good copy (HF `cara_latency_prediction` buggy).

---

## Reproduce recipes (verified 2026-05-16 — scripts git-tracked in public route-balance repo)

All trainers: `route_balance/predictor/route_balance/offline_training/` (committed to `github.com/AKafakA/route-balance`).
Training data: `data/cara/scored/train_scored_filtered.jsonl` (GPU VM + VPS test split + HF `asdwb/cara_model_estimator`).

| model | command (run from offline_training/) |
|---|---|
| `roberta_fused_5ep` | `python train_fused_bert_predictor.py --encoder-name roberta-base --epochs 5 --lr 1e-5 --batch-size 16 --max-length 1024 --seed 42 --target {reference_score,length,length_bucket,judge_class}` (one run per target, 4-head fused) |
| `roberta_fused_deepeval` | `python train_fused_bert_predictor.py --encoder-name roberta-base --epochs 5 --lr 1e-5 --batch-size 16 --max-length 1024 --seed 42 --target deepeval` |
| `modernbert_fused_deepeval` | `python train_fused_bert_predictor.py --encoder-name answerdotai/ModernBERT-base --epochs 5 --lr 1e-5 --batch-size 16 --max-length 1024 --seed 42 --target deepeval` (defaults) |
| `length_regression` | `python train_fused_bert_predictor.py --encoder-name answerdotai/ModernBERT-base --target length` (all defaults) |
| `roberta_multitarget_A/B` | `python train_fused_multitarget.py --encoder-name roberta-base --targets <see each dir's training_results.json `targets`> --epochs 5 --lr 1e-5 --batch-size 16 --max-length 512` |
| `lora_per_model` (KEEP) | `python train_lora_bert_predictor.py --encoder-name roberta-base --target reference_score --epochs 3 --lr 1e-4 --batch-size 16 --max-length 512` (lora_r=16, lora_alpha=32; per training_results.json) |
| KNN (deployed, on VPS) | `python train_knn_estimator.py` (k=50 distance-weighted, non-parametric — no transformer training) |

Per-model exact lr schedule + final epoch are in each dir's `log_history.json`; encoder/target/results in `training_results.json`. With script(git)+HP(here)+data(HF/VPS), every REMOVE-eligible model is reproducible.

**Reproducibility status: LOCKED.** Safe to delete REMOVE-eligible models per user policy once #107 scope is decided.

---
## ~root baseline cleanup (2026-05-16) — DONE
- `best_route_4way_faithful` trimmed 2.7G→545M (checkpoint-best kept; deployed copy also on VPS `models/cara/best_route_4way_qwen`)
- removed: `.bak` dup (2.7G), 4 intermediate BEST-Route ckpts, `best-route-llm` (re-cloneable), `llmrouter_out` (RouteLLM-mf, not in submitted paper)
- preserved scripts → `cara_exp/scripts/` (recluster + data-prep + train .sh, none were git-tracked)
- full record: `cara_exp/ARCHIVE_MANIFEST.txt` (mirrored here as `gpu_vm_ARCHIVE_MANIFEST.txt`)
- net: GPU-VM `models/cara` 8.3G→105M; `~root` 41G→36G; nothing paper-critical or non-reproducible removed.
