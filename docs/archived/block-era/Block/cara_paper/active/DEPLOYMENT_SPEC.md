# Deployment Spec — single source of truth for cron audit

**Last updated**: 2026-05-03 12:34 UTC. Owner: experiment runner.

**Cluster delta**: As of 2026-05-03 12:34 UTC, **node2 (c4130-110433, 10.10.1.3)** is hardware-down. 14B cluster reduced from 3 → 2 vLLM instances (10.10.1.12, 10.10.1.1). Removed from `envoy_qwen.yaml:qwen14b_cluster` and `model_deployment_apr26_phase2.json:Qwen-2.5-14B.node_hosts`. All baselines (vllm-sr / AvengersPro / BEST-Route / cara) operate on the reduced fabric. 14B aggregate capacity is 2× single-instance throughput; saturation may appear at lower λ for routing strategies that concentrate on 14B (e.g. vllm-sr AutoMix-default, AvengersPro pw=0.7).
**Purpose**: every cell collected for the paper must match these specs. Cron compares **deployed reality** against this doc each sweep and flags mismatches loudly. If a mismatch appears, do not aggregate that cell and do not kick a new grid until the deployment matches the spec.

This doc is paired with the machine-readable `cara_paper/active/deployment_spec.json` consumed by `cara_paper/smoke_test_apr_13/scripts/audit_deployment_vs_spec.py` (called from `cron_overnight.sh`).

---

## Vendor pricing (single source)

All systems read these rates. Source: Together AI / Novita AI public pricing pages (May 2026). See `cara_paper/active/PAPER_UPDATES_TODO.md` Table A for citations.

| Model | Input $/1M | Output $/1M | Provider |
|---|---|---|---|
| Qwen 2.5 3B  | 0.06 | 0.06 | Together AI |
| Qwen 2.5 7B  | 0.07 | 0.07 | Novita AI |
| Qwen 2.5 14B | 0.15 | 0.15 | Novita AI |
| Qwen 2.5 72B | **0.38** | **0.40** | Novita AI (asymmetric) |

**Where stored**: `block/config/cara/model_deployment_apr26_phase2.json` — `pricing` block per model, with `provider` + `source` URL fields.

---

## Cara

| Spec field | Expected value | How to verify |
|---|---|---|
| Predictor type | `learned` (XGBoost 3-model per (model, GPU)) | `cat block/config/cara/predictor_config_learned.json` field `predictor_type` |
| **NOT** | `dummy` (heuristic min_requests). Last seen May 2-3 incident; do not deploy. | `predictor_deployment_config.json` IS the dummy config — must NOT be loaded |
| TTFT predictor | XGBoost; model_dir = `/users/asdwb/Block/models/cara/latency/deploy_ttft/` | `predictor_config_learned.json` `xgboost_ttft.model_dir` |
| TPOT predictor | XGBoost (NOT static lookup) | `predictor_config_learned.json` SHOULD have `xgboost_tpot.model_dir`, NOT just `tpot_lookup` |
| E2E predictor | XGBoost dedicated (NOT derived from `ttft + N×tpot`) | `predictor_config_learned.json` SHOULD have `xgboost_e2e.model_dir` |
| XGBoost training data | Apr 28-29 Phase 2 cluster per-instance latency sweep | training_metrics.json mtime ≥ 2026-04-28 |
| Bucket encoder | **RoBERTa** (per Apr 19-20 Vast HP-sweep, target-specific best `length_bucket/cosine` + `deepeval/warmup0.1`) | `cat /mydata/models/cara/bucket_classifier/fused_config.json` field `encoder_name` ≠ `answerdotai/ModernBERT-base` |
| **NOT** | ModernBERT (May 1 incident — 3-day loss when stale ModernBERT was deployed) | If `encoder_name == "answerdotai/ModernBERT-base"` → STOP |
| Quality predictor | KNN over MiniLM-L6-v2 embeddings (per `predictor_config_learned.json` `quality_predictor.type=knn`) | `cat predictor_config_learned.json` field `quality_predictor.type == "knn"` |
| Sidecar predictors | xgboost_3model on ports 8100-8103 of all 5 head nodes | curl `http://<host>:8300/health` for each head's predictor service; expect 200 |
| Scoring weights | `{w_latency: 0.3, w_cost: 0.2, w_quality: 0.3, w_balance: 0.2}` (sum = 1.0) | `cat scheduler_config_caraknn.json` (and `cararoberta.json`) — confirm scoring_weights block |
| Cost formula | `prompt_tokens × cost_per_input_token + expected_output × cost_per_output_token` (split-aware) | `cara_serve.py:838-845, 1321-1324` — present after 2026-05-03 fix |
| Pricing source | Injected from `model_deployment_apr26_phase2.json:pricing` (NOT from predictor config) | `cara_serve.py:2259-2280` does the injection at boot |
| Filter mode | `slo_defaults.filter_enabled = false` for main run; `true` for appendix | `cat scheduler_config_*.json` `slo_defaults.filter_enabled` |
| λ grid | {6, 8, 10, 12} | grid script header |
| Eval set | `data/cara/best-route-v3-test-3534-eval.jsonl` (3534 prompts, no train leak) | grid script `--dataset-path` |
| Knob (paper headline) | `w_quality ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` (5 values) | grid script |

### Cara overnight phases

| Phase | Dir | Cells | Config tag |
|---|---|---|---|
| A. cara-knn-full | `cara-knn-full/` | 20 (5 wq × 4 λ) | `scheduler_config_caraknn.json` |
| B. cara-roberta-full | `cara-roberta-full/` | 20 | `scheduler_config_cararoberta.json` |
| C. budget_control | `budget_control/` | 30 | (smaller N=706) |
| D. filter_ablation | `filter_ablation/` | ~10 | filter_enabled=true |
| E. weights_ablation | `weights_ablation/` | ~10 | varied scoring_weights |

---

## AvengersPro 4-model

| Spec field | Expected value | How to verify |
|---|---|---|
| Router type | `avengers_pro` | grid script POST `/v1/config` payload |
| Artifact dir | `/users/asdwb/Block/models/cara/avengers_4model_v3_novita/avengers_pro_qwen_pw0.7` | grid script `artifact_dir` kwarg |
| Cost formula at training | **Input + output split** (`mean_inlen × cost_in + mean_outlen × cost_out`) per (cluster, model) | `cat avengers_pro_qwen_pw0.7/config.json` field `cost_model` includes "input+output split" |
| **NOT** | Output-only `mean_outlen × cost_per_token` (was v2 path until 2026-05-03 PM) | Reject `config.json` if `cost_model` is missing or says "output-only" |
| Pricing values used | Novita 2026-05 ($0.06/$0.07/$0.15/(0.38 in, 0.40 out) per 1M) | `config.json` `cost_per_input_token_1m` + `cost_per_output_token_1m` blocks |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | `config.json` `embedding_model` |
| Quality target | `deepeval-llama3.1-8b-it_reference` | `config.json` `judge_key` |
| Trained on | `train_scored_filtered.jsonl` (14,891 prompts) | `config.json` `n_train_prompts` |
| Dispatchers | `round_robin` + `shortest_queue` | grid script POST `dispatch.type` per cell |
| pw values | `{0.25, 0.39, 0.53, 0.7, 0.8}` (5 artifacts) | output_dir `avengers_pro_qwen_pw{X}/` exists for all |
| λ grid | {6, 8, 10, 12} | grid script |
| Cells per pw | 4 λ × 2 disp = 8 (or 1 × 4 if only default pw=0.7) | dir cell count |

---

## vLLM-SR (Two distinct routing paths — pick the right one)

**2026-05-03 finding**: vLLM-SR has TWO routing layers. The May-3 prior runs invoked `algorithm.type=confidence` (AutoMix-style logprob cascade), NOT the documented semantic router. Real semantic router uses ModernBERT.

### Path A (HEADLINE — what upstream documents): ModernBERT category classifier

Per `vllm-semantic-router.com/docs/training/training-overview/` (read 2026-05-03): ModernBERT is the **shared backbone** for the semantic router with multiple fine-tuned classification heads:
- Category Classifier (10 MMLU-Pro academic domains)
- PII Detection (6 entity types)
- Jailbreak Detection (binary)
- Intent Classifier (8 function categories)

Routing decision = ModernBERT classifies prompt → maps category to model.

| Spec field | Expected value | How to verify |
|---|---|---|
| Algorithm type | `classification` (NOT `confidence`) | `config.yaml` `decisions[*].algorithm.type` |
| Classifier model | upstream-pretrained ModernBERT category classifier from `huggingface.co/LLM-Semantic-Router/...` | `routing.modelCards.path` |
| Routing rule | 10-category mapping → 4 Qwen sizes (math/code/science → 72B, general/trivia → 3B, etc.) | `config.yaml` `routing.decisions` |
| Smoke ≥3 distinct sizes | required: 100 prompts hit ≥3 of {3B, 7B, 14B, 72B} | bench `model` field |

**Status 2026-05-03**: NOT YET DEPLOYED. Estimated 2-3h to download pretrained ModernBERT classifier + write routing.modelCards + smoke.

### Path B (ALTERNATIVE used in May-3 prior runs): AutoMix logprob cascade

This is a separate algorithm in vllm-sr that uses logprob/margin from the serving model's own completion to decide whether to escalate to a larger model. It does NOT invoke ModernBERT for the routing decision (BERT classifier may still run but its output is unused for the route choice). Suitable as a documented "AutoMix-cascade" baseline, not as "vllm-sr semantic router".

| Spec field | Expected value | How to verify |
|---|---|---|
| Algorithm type | `confidence` | `cat config.yaml` `decisions[*].algorithm.type` |
| escalation_order | `automix` | `cat config.yaml` `decisions[*].algorithm.confidence.escalation_order` |
| Looper.endpoint | `http://vllm-sr-envoy:8899/v1/chat/completions` | `cat config.yaml` `global.integrations.looper.endpoint` |
| Per variant | `{avg_logprob, margin, hybrid, self_verify}` with upstream defaults `{-1.0, 0.5, 0.5, 0.7}` | grid script |

**Common fields (both paths):**

| Spec field | Expected value | How to verify |
|---|---|---|
| Stack | 5 Docker containers up (vllm-sr-container, envoy, grafana, prometheus, jaeger) | `docker ps` |
| Pricing block | Novita 2026-05 in `providers.models[*].pricing` | `cat /mydata/vllm_sr/config.yaml` |
| Envoy listener_1 | port 8899, Lua filter sets x-selected-model from body | `cat /mydata/vllm_sr/envoy_qwen.yaml` |
| Envoy listener_0 | port 8801, ext_proc with `BUFFERED` body mode + `allow_mode_override: true` | yaml |
| Bench backend | `--backend vllm-sr` (registered in `cara_end_point_func.py`) | grid script |
| Captured fields | `model`, `host`, `instance_id`, `ttft`, `e2el`, `output_len`, `scheduling_overhead_breakdown.vllm_sr_looper_*` | response_details schema |

**Order in main experiments**: vllm-sr (Path A, real ModernBERT) runs **AFTER BEST-Route 4-way** in the priority queue. If time-budget runs out, drop to Path B as a documented AutoMix-cascade baseline with explicit §3.5 framing.

---

## BEST-Route 4-way (Ding 2025 §3 faithful)

| Spec field | Expected value | How to verify |
|---|---|---|
| Router type | `best_route_4way` (registered in `factory.py:90-93`) | factory.py |
| Wrapper class | `BestRoute4WayRouter` using upstream `NClassReranker.predict()` | `cat best_route_wrapper.py` — must import `from hybrid_llm.pair_ranker.ranker import NClassReranker` |
| **NOT** | `AutoModelForSequenceClassification` (silently swaps trained N-way head for random 2-class) | If wrapper uses AutoModelForSequenceClassification → STOP |
| Checkpoint path | `/mydata/best_route_4way_faithful_novita/checkpoint-best/` (Novita-priced retrain) | grid script `checkpoint_path` kwarg |
| Loss type | `prob_nlabels` | `cat checkpoint-best/config.json` `loss_type` |
| n_candidates | 4 | `config.json` |
| Quality metric | `armoRM_scores` | `config.json` |
| Cost dict in training | `model2prompt_cost = {3B: 0.06e-6, 7B: 0.07e-6, 14B: 0.15e-6, 72B: 0.38e-6}` and `model2output_cost = {72B: 0.40e-6}` (asymmetric) | `cat /home/wd312/best-route-llm/hybrid_llm/pair_ranker/data.py:16-17` |
| Source prefix token | `<|source|>` added during training (vocab 128002 vs DeBERTa default 128001) | wrapper's resize logic at boot |
| Decision rule | Ding 2025 §3 M_ref vs M abstention: `if p_72b ≥ threshold → 72B`, `else argmax over {3B,7B,14B}` | wrapper's `choose_model` logic |
| Knob | threshold `t ∈ {0.3, 0.5, 0.6, 0.7, 0.8}` (paper sweep); default 0.5 for first run | grid script `threshold` kwarg |
| Output dir | `baselines/best_route_4way/` | grid script |

---

## Cron audit responsibilities (2026-05-03 update)

The cron at `/home/wd312/Code/llm/Block/cara_paper/smoke_test_apr_13/scripts/cron_overnight.sh` MUST:

1. Each sweep, before any sync/launch decision, call `audit_deployment_vs_spec.py` which compares live deployment to `deployment_spec.json` and emits per-system PASS/FAIL.
2. If any spec FAIL → write a 🚨 section to handoff log and DO NOT sync. User must fix the deployment to match spec before next sweep.
3. Spec is authoritative; if spec needs a change, edit this doc + `deployment_spec.json` first, then re-run cron.

---

## Paper claims this spec supports

- **§3.5** Baseline framing: 4 baselines (cara-full + AvengersPro + vLLM-SR + BEST-Route) at upstream-default settings + Novita 2026-05 vendor pricing
- **§5** Main results: Pareto plots over cost-quality-latency at λ ∈ {6, 8, 10, 12}
- **App G** Scheduling overhead: cara reports per-component breakdown; vLLM-SR reports looper iterations only (per-token timing not exposed by upstream)
- **App F** Asymmetric pricing sensitivity: cara + BEST-Route honor 72B's $0.38/$0.40 split; AvengersPro now honors after v3 retrain; vLLM-SR routing reads only PromptPer1M (upstream limitation, document)
