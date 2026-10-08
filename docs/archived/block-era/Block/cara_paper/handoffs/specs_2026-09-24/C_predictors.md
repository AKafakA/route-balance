# C — Port spec: RouteBalance prediction stack (KNN quality/length + TPOT latency)

Source of truth: Block, branch `v3-campaign`, HEAD `984ea18` (last code commit touching these files:
`9af80f2`, 2026-08-04). Read-only survey, 2026-09-24. No model was loaded; pickles were opened with
numpy only to measure array properties.

Citation keys (all paths absolute under `/home/wd312/Code/llm/Block/`):

| key | file | lines | md5 (local HEAD) |
|---|---|---|---|
| ME  | `block/predictor/cara/model_estimator.py` | 1,142 | `c702ee99b83dc88af152db2cd4974c98` |
| KNN | `block/predictor/cara/estimators/knn_estimator.py` | 498 | `d17e5e239ffda84fddd59acad4860846` |
| XGB | `block/predictor/cara/estimators/xgboost_predictor.py` | 231 | `7a831be8b53aba0220f06cab2b89dc09` |
| CLP | `block/predictor/cara/cara_learned_predictor.py` | 895 | — |
| CFG | `block/predictor/cara/cara_predictor_config.py` | — | — |
| CS  | `block/global_scheduler/cara/cara_serve.py` | 3,873 | `2144a2f9ae660c6f9b2d1d51e8d6e2e1` |
| TK  | `block/predictor/cara/offline_training/train_knn_estimator.py` | 237 | — |

Provenance note: `cara_paper/handoffs/window_setup_work/v3_preservation/RERUN_PINS.md` (captured
2026-07-31) lists different md5s for CS/ME/KNN; those predate the 08-01..08-04 commits.
`v3-exp/manifest/AUDIT.md` (2026-08-04) states the S2 head copy matched local. Pin the golden oracle to
the local HEAD md5s above, not to RERUN_PINS.

---

## 0. Read first: nine findings that change what "port faithfully" means

1. **Two different "static TPOT" arms have been executed, and the MLSys target matches neither v3 run.**
   * **v3 Q1 arm (config key `slo_defaults.static_tpot_ms_per_tok`)**: the value that enters the score is
     `const[tier] × E[len_model]`, with no queue term (CS:1865-1871). The queue-aware envelope is also
     computed with the constant TPOT (CS:1275-1282 then CS:1389-1402) into `pred["e2e_latency"]`, but
     with `filter_enabled=false` nothing reads it except the decision log (`learned_e2e_s`). The per-cell
     gate `tpot_gate.py` asserts exactly this: `lat_s / exp_len` is constant per tier.
   * **v2-tree arm (`CARA_STATIC_TPOT_HEAD` env, used for the posted Gemma W1-B cells)**: the constant
     was injected at the batched-TPOT step and "leaves the analytic queue envelope byte-identical"
     (`window_setup_work/v3_scripts/gemma_cells_w1b.sh:37-49`). So the scored value was
     `(queue_iters + own) × const`. That reader existed only in the non-git `~/Block` cluster tree; no
     branch in the Block repo contains it (`git log --all -S STATIC_TPOT_HEAD` only finds comments).
   * The MLSys target, "static TPOT × queue-aware decode iterations", is the v2 semantics. Building
     it as the default is a design change relative to the v3 Q1 evidence. **The owner has to decide
     which envelope the static arm uses (R3).** The rebuild should make the envelope a required enum
     with no default, and golden-test both (§7, §8).
2. **The learned XGBoost TPOT heads get features at serving time that differ from their training
   features.** The in-process path passes the raw `/instance_stats` body into `build_feature_vector`
   (CS:1299-1300). The vLLM fork `AKafakA/vllm@2894a98` (`vllm/v1/core/sched/scheduler.py`,
   `get_aggregated_stats`) emits 12 of the 17 state features. `ema_decode_tok_per_s`,
   `ema_prefill_tok_per_s`, `ema_decode_iter_ms`, `prefill_chunk_size` and `kv_evictions_per_s` are
   absent, so they become `0.0` (XGB:107). Because the EMAs are missing, three derived features fall
   back to the 1.0 tok/s floor (XGB:71-74, 82-84). The training rows do have these fields: in the first
   60,000 records of `data/cara/latency_data/all/latency_train_tagged.jsonl`, all three `ema_*` fields
   are non-zero in 60,000/60,000. They were produced by the data-collection sidecar
   (`dummy_cara_predictor.py:98-102`). No commit in the fork ever emits them.
   Scope: this affects the learned arm only. The static arm is unaffected. It matters most for the 72B
   head, whose top features by importance include `ema_decode_iter_ms` (#2) and
   `request_decode_time_est` (#3) (`models/cara/latency/deploy_tpot/training_metrics.json`). For a
   faithful port, reproduce the zero-fill. Fixing it creates a new arm that needs owner approval and
   re-measurement.
3. **The v3 static arm still depends on the XGBoost artifacts.** At startup the code loads
   `deploy_tpot` and refuses to start unless it covers every fleet tier (CS:3636-3686).
   `_batched_tpot_per_request` returns `{}` before it reaches the static branch when no booster set is
   loaded (CS:1267-1268). Gemma has no heads, so RB/Gemma cannot start on the v3 tree with the
   in-process predictor. In the rebuild, the static arm must need no XGBoost artifact at all.
4. **The TPOT head is fed the maximum predicted length across models, not the candidate's own
   length.** `num_predicted_output_tokens = max(1, int(max_m E[len_m]))` is the same value for every
   tier (CS:2266-2272, CS:2481-2486). The envelope then uses the per-model `int(E[len_m])` (CS:1393),
   and the static score uses the per-model *float* `E[len_m]` (CS:1871). Replicate all three exactly.
5. **Several deployed config keys are never read.** `--predictor-config`
   (`predictor_xgboost_3model.json`) is not loaded when `--scheduler-config` is also given (if/elif at
   CS:3575-3579), and the S2 launcher passes both (`v3_scripts/lib_serve_v3.sh:158-166`). Its
   `tpot_lookup` (25/35/55/65 ms), `model_paths_by_instance` and `/users/asdwb/...` paths were never
   read in S2. The heads actually used come from `CARA_TPOT_DIR`, which defaults to
   `<tree>/models/cara/latency/deploy_tpot` (CS:3649-3652). Also dead: `quality_config.embedding_model`
   and `quality_config.type` (the encoder name comes from the pickle, KNN:481), and
   `instance_metadata.*.static_tpot_ms_per_token` (Gemma config).
6. **KNN `k`: the value baked into the artifact wins.** The config `k` must match it or startup raises
   (ME:178-195). `KNNModelEstimator` (type `knn`, unused) does the opposite: the config overrides the
   baked value, with a default of 50 (ME:995, ME:1002). The v3 ablation configs `cfg_{de,ju}_f50.json`
   and `cfg_w2a_promptblind.json` declare a k that differs from what their artifacts bake, so they
   raise on the current code.
7. **Missing DeepEval labels are stored as 0.0 in the index** (KNN:153-154,
   `float(js.get(jk, 0.0) or 0.0)`). Measured against `data/cara/train_scored_filtered.jsonl`, which
   is the exact source of `deploy_v3_k50_14919` (row order, lengths and labels verified identical):
   7/11/9/13 missing labels for 14B/3B/72B/7B are stored as 0.0. They cannot be told apart from
   3,137/3,850/2,930/4,869 genuine 0.0 scores. Reproduce this for equivalence.
8. **Silent defaults remain on the serving path**: see §4 and §9 (H8). The worst one: if no
   `ModelEstimate` exists for an instance's model (a name mismatch, or an empty KNN dir), the request is
   scored with `quality=0.5` and `length=max_output_tokens` (CS:1358-1361), with no error.
9. **Ablation artifacts.**
   * *50% labels*: the v3 artifact is `deploy_v3_f50_k50` (7,459 rows). It is an order-preserving subset
     of the 14,919-row index with identical embeddings and labels (verified, including 14 ambiguous
     duplicate rows), and bakes k=50.
   * *Prompt-blind*: no v3/DeepEval artifact exists. The only one is the v1
     `p4_indexes/f0_uniformprior` (14,966 rows, k=10 baked). It has the old schema: `llm_judge_scores`
     is exactly 0.5 for every model and row, and there are no `per_judge_scores`. Only quality is
     blinded; lengths are still retrieved from the prompt's neighbours. With the default `minmax`
     normalisation, a constant quality gives `qual_score = 0` for every candidate (CS:1968), so the
     quality term drops out.

---

## 1. Public API the scheduler actually uses

### 1.1 Construction (startup, synchronous, main thread)

| call site | what runs | notes |
|---|---|---|
| CS:3202-3219 `_load_scheduler_config` | `DefaultModelEstimator(me_config)` when `model_estimator.type` is `"default"` or any value other than `"pfs"`/`"knn"`; `KNNModelEstimator` for `"knn"`; `PFSModelEstimator` for `"pfs"` | v3 configs all use `"default"`. A load exception becomes `SchedulerConfigError` (CS:3271-3279). |
| CS:3282-3396 `_load_learned_predictors` | legacy path; also builds per-instance `CaraLearnedPredictor` into `learned_predictors` | **Not executed in v3**, because `--scheduler-config` takes precedence (CS:3575-3579). It also swallows estimator load errors (CS:3325-3326). |
| CS:3584-3588 | `cara` / `length_aware` with `model_estimator is None` → raise | Does **not** catch an estimator object whose `_knn is None` (empty `model_dir`, ME:140-142). |
| CS:3635-3690 | `_TPOT_BATCH_MODEL = XGBoostLatencyPredictor.load(CARA_TPOT_DIR)` + fleet-coverage check | Runs in **both** arms (see §0.3). |

`DefaultModelEstimator.__init__(config: dict, device: str = "cpu")` (ME:86-121) loads the KNN index
(ME:123-224), then bucket classifiers (ME:226-293) and the length regressor (ME:295-327). Both of the
latter return early in v3 because no config for them is present.

### 1.2 Per-batch / per-request calls

| # | call site | method | signature → return | units | batching | thread / executor |
|---|---|---|---|---|---|---|
| A | CS:2348-2363 (`_batch_scheduler_loop`, the default mode) and CS:2130-2138 (`_prepper_loop`, only when `CARA_PREPPER_WORKERS>=1`) | `model_estimator.estimate_batch(prompts, budget)` | `(List[str], int) → List[Dict[str, ModelEstimate]]`, same order as `prompts` | `length_expected`: tokens (float); `score`: quality in [0,1]; `p_under_budget`: probability; `length_bucket_probs`: (16,) float32 | one call per scheduler batch; the batch size adapts over 4/16/32 (CS:345-381); `budget = batch[0].budget_tokens` for the whole batch (CS:2359) | **`loop.run_in_executor(None, …)`** (default ThreadPoolExecutor). In S2 there is one batch worker (`CARA_BATCH_WORKERS` default 1, `CARA_PREPPER_WORKERS` unset), so calls never overlap. |
| B | CS:2379-2391, CS:2147-2152 | `model_estimator.estimate(_estimator_prompt(p), req.budget_tokens)` | `(str, int) → Dict[str, ModelEstimate]` | as above | per request | **Inline on the event loop.** Only reached if the estimator has no `estimate_batch`, which never happens for Default. Counted by `_SCORE_PERREQ_CALLS` (CS:1203). |
| C | CS:664-704 `POST /v1/estimate` | `estimate(prompt, budget_tokens=256)` | JSON `{model: {length_expected, p_under_budget, score, score_type, bucket_probs}}` | | per call | inline on the event loop (blocks it). Used by preflight probes; its `p_under_budget` and `bucket_probs` semantics differ from A (§2.7). |
| D | CS:3009-3010 `_schedule_cara` | `estimate(...)` | | | per request | inline; used only when `batch_queue is None` |
| E | `routers/cara_native.py:65` (pipeline mode) | `estimate(req.prompt, req.budget_tokens)` | | | per request | inline; the prompt is unwrapped inside `_find_neighbors` |
| F | CS:1256-1322 `_batched_tpot_per_request(req, instances, stats_map, expected_length)` | `booster.inplace_predict(X)` per tier | → `{instance_id: tpot_seconds_per_token}` | s/token, clamped ≥0.001 | once per request, one call per tier (4 Qwen tiers) | **each tier runs in `loop.run_in_executor(None, _predict_tier, …)`** (CS:1311), gathered with `asyncio.gather`; the per-request loop awaits it in sequence (CS:2488) |
| G | CS:1325-1517 `_predict_one_for_req` | reads `req.model_estimates[model_name]` and `req._batched_tpot[inst_id]`; computes the envelope | → `(inst, pred_dict, ms)`; `pred_dict` keys: `quality, expected_length, p_under_budget, ttft, tpot, e2e_latency, cost…, instance_type, xgb_predict_ms` | e2e in seconds | per (request × instance), gathered | event loop |
| H | CS:1548-2093 `_score_and_pick_instance` | `_lat_val` (static or learned), quality normalisation | picks the argmax | | per request | event loop |

`ModelEstimate` (ME:35-43), a plain mutable dataclass: `model_name: str`, `length_expected: float`,
`length_bucket_probs: Optional[np.ndarray]`, `p_under_budget: float`, `score: float`,
`score_type: str`.

Fields consumed downstream:
* `score` → `pred["quality"]` (CS:1346)
* `length_expected` → envelope, cost and LPT (CS:1347, 2142-2146, 2367-2378)
* `p_under_budget` and `length_bucket_probs` → only the legacy filter, and only with
  `filter_enabled=true` (CS:1696-1714); every deployed config sets it false
* Response trace fields written from the estimates (CS:1166-1187): `predicted_quality`,
  `predicted_length`, `p_under_budget`, `predicted_best_model/score/hit`, `all_model_scores`. These are
  the golden oracle (§8).

Observables the rebuild must keep: `GET /v1/config` → `score_batched_calls`, `score_perreq_calls`,
`tpot_predict_calls`, `static_tpot_ms_per_tok` (CS:746-761); the `EFFECTIVE_CONFIG` log line
(CS:3028-3072: `estimator_type, score_type, knn_k, knn_corpus, quality_norm, static_tpot_ms_per_tok,
inproc_predictor, …`); the decision log (CS:62-77, 1876-1890: per candidate `inst, itype, lat_s,
exp_len, learned_e2e_s`; `req` is the scheduler arrival counter, not the client id).

### 1.3 Thread-safety facts

* `KNNEstimator._get_encoder` (KNN:94-102) and `_build_faiss_index` (KNN:328-376, called from ME:646-647
  and KNN:395-396) initialise lazily without a lock. The first `estimate_batch` therefore loads MiniLM
  inside an executor thread, and the cold start lands in the first batch's `estimator_ms`.
* FAISS `IndexFlatIP.search` supports concurrent reads. Concurrent `SentenceTransformer.encode` calls are
  not guaranteed safe, but they never happen in S2.
* The four per-tier `inplace_predict` calls run on **different** boosters at the same time (safe).
  `_TPOT_PREDICT_CALLS += 1` runs inside those threads (CS:1304-1305) without a lock, so under
  contention it can lose increments. Only `delta > 0` (learned) and `delta == 0` (static) are ever
  asserted.

---

## 2. KNN estimator internals (the executed path is `DefaultModelEstimator.estimate_batch`)

### 2.1 Artifact layout and loading

* `model_dir/knn_estimator.pkl`, a `pickle` of a **dict** (KNN:434-446) with keys
  `embedding_model_name, k, embeddings, model_names, output_lengths, similarity_scores, llm_judge_scores,
  reference_similarity_scores, per_judge_scores`. `knn_metadata.json` is written by `save()` but never
  read by `load()`; the v3 dir has no metadata file.
* `KNNEstimator.load(model_dir, device, index_type)` (KNN:461-498) requires `embedding_model_name`, `k`,
  `embeddings`, `model_names` and `output_lengths`. The other arrays default to `{}` without a warning.
* **No md5/sha check exists in code.** Integrity is enforced only outside the code: cell provenance
  `quality.knn_pkl_md5` (e.g. `d2868fdc21b26b9e75dbb79d634dda19`) and the cell_audit `PKL_MD5_EXP`.
  `_log_and_assert_deployed_predictors` (CS:3399-3456) audits `fused_config.json` of the ModernBERT
  dirs, not the KNN; if they are missing it only warns.
* Startup checks in `_load_knn` (ME:123-224):
  * empty `model_dir` → warn, KNN disabled (ME:140-142)
  * path missing → raise (ME:143-148)
  * load exception → raise (ME:157-161)
  * k mismatch → raise unless `k_mismatch_ack` (ME:178-195)
  * zero models → raise (ME:200-204)
  * `score_type=="deepeval"` and any model lacking a `DEEPEVAL_KEY` array → raise (ME:209-219)
* Measured `deploy_v3_k50_14919` (24,349,380 B):
  * `embeddings` (14919, 384) float32, every row L2-norm = 1.000000
  * 17 duplicate-embedding pairs, **all with differing labels**
  * `model_names = ['Qwen/Qwen2.5-14B','Qwen/Qwen2.5-3B','Qwen/Qwen2.5-72B','Qwen/Qwen2.5-7B']`
    (sorted, KNN:124)
  * `per_judge_scores[m]` keys: `Qwen_Qwen2.5-7B-Instruct`, `deepeval-llama3.1-8b-it_reference`,
    `protectai_distilroberta-base-rejection-v1`
  * `llm_judge_scores[m]` is bitwise equal to `per_judge_scores[m]['Qwen_Qwen2.5-7B-Instruct']`
    (KNN:164-166)
  * all arrays float32, `k=50`

### 2.2 Prompt preprocessing (the chat-template skew fix)

`unwrap_chat_prompt(p)` (KNN:30-44) is the single point where text becomes an embedding query:

```
if not p: return p
if "<|im_start|>user" in p:   m = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", p, re.S); return (m.group(1) if m else p).strip()
if "<start_of_turn>user" in p: m = re.search(r"<start_of_turn>user\n(.*?)<end_of_turn>", p, re.S); return (m.group(1) if m else p).strip()
return p.strip()
```

* Applied in `estimate_batch` (ME:635-643) and in `_find_neighbors` (KNN:388-390). The scheduler
  already applies a byte-identical copy, `_estimator_prompt` (CS:80-81, CS:107-120), at CS:2133, 2152,
  2358, 2390 and 3010. The two applications are idempotent.
* Non-greedy, first match: only the **first** user turn is kept and the system block is dropped. If the
  end tag is missing, the whole text (system block included) is returned stripped.
* Index side: `build_index` encodes `d["prompt"]` raw, without strip (KNN:116-120). None of the 14,919
  corpus prompts has leading or trailing whitespace (measured), so the query-side `.strip()` does not
  cause a mismatch for the Qwen corpus.
* Banked serving prompts look like
  `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n…` (Qwen) and
  `<bos><start_of_turn>user\n…` (Gemma), taken from the S2 and W1-B cells.
* Evidence for why this matters: `cara_paper/handoffs/2026-07-31_KNN_CHAT_TEMPLATE_SKEW.md`. Recorded
  scores reproduce the wrapped prediction to 0.000000; the unwrapped one differs by 0.100; the wrapped
  ordering inverts 3B/14B.

### 2.3 Embedding

* `SentenceTransformer(state["embedding_model_name"], device=quality_config.device or "cpu")`
  (KNN:94-102). Version pins: `sentence-transformers/all-MiniLM-L6-v2` @ revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, sentence-transformers 5.4.1, torch 2.9.0, transformers
  4.56.0 (`v3_preservation/RERUN_PINS.md`, `v3-exp/env/ENVIRONMENT.md`). v3 runs with
  `CUDA_VISIBLE_DEVICES=""`, so on CPU.
* Batch path: `encoder.encode([unwrap(p)…], normalize_embeddings=True, batch_size=max(n, 32))`
  (ME:641-644), which is always a single encode call for n ≤ 32. Single path:
  `encoder.encode([unwrap(p)], normalize_embeddings=True)` (KNN:388-390). The model's own
  `max_seq_length` truncates the input; the code does not set it.

### 2.4 Search

* `faiss.IndexFlatIP(384)` over `embeddings.astype(float32)` (KNN:345-349). Inner product on
  normalised vectors is cosine similarity. faiss-cpu 1.13.2.
* `k = min(self._knn.k, N)` (ME:649).
* Batch path (D19): one `search(Q, k)` over the (n, 384) float32 matrix (ME:668-674); `idx` int64;
  `dist = (1.0 - sims).astype(float64)` (ME:674). The float32 subtraction happens before the cast.
* Single path: `search(query_f32, k)`, then `.squeeze()`, then the same distance formula (KNN:398-406).
  `.squeeze()` would break if k=1.
* Numpy fallback (only if faiss is missing, or `index_type="numpy"`/unknown): `embeddings @ Q.T`,
  `argpartition`, `argsort(-sim)`. Its distances stay **float32** (KNN:409-418; ME:678, 685-688), and
  tie order can differ from FAISS.
* `faiss_ivfpq` exists (KNN:351-373) but is dead.

### 2.5 Neighbour weights

`_distance_weights(d)` (KNN:420-427): `w = 1/(d + 1e-6); w /= w.sum()` (float64 on the FAISS path).
`weighted=False` would give uniform weights, but no serving caller uses it. When a query is an exact
duplicate of a corpus prompt, `d≈0` and that one neighbour gets weight ≈ 1e6. If a float32 rounding
error made `d == -1e-6` the weight would be infinite; this has not been observed.

### 2.6 Quality aggregation by `score_type` (batch path ME:707-746, single path ME:445-496)

`score[m] = float(np.average(arr[m][idx], weights=w))`. Note that `np.average` divides by `Σw` again
(the weights sum to about 1, not exactly 1). The dtype is float64.

| `score_type` | array read | missing array |
|---|---|---|
| `"deepeval"` (**v3 Qwen**) | `per_judge_scores[m]["deepeval-llama3.1-8b-it_reference"]` (key `DEEPEVAL_JUDGE_KEY`, KNN:51; aliased ME:28-30) | raise (ME:725-734, 478-484) |
| `"judge"` (**Gemma; also the v3 `ju` ablation**) | `llm_judge_scores[m]` (Qwen2.5-7B-Instruct judge, KNN:164-170) | batch: raise (ME:739-744); single: **0.0** (ME:461-466, still silent) |
| `"similarity"` | `similarity_scores[m]` | batch raise / single 0.0 |
| `"reference"` | `reference_similarity_scores[m]` | batch raise / single 0.0 |
| anything else (`"combined"`, typos) | `reference_similarity_scores` (batch ME:736-737). Single path: reference, then judge, then **0.0** (ME:487-495) | **silent signal substitution for unknown strings** |

### 2.7 Length aggregation (batch path, ME:692-706 and ME:787-806)

* `length_expected[m] = float(np.average(output_lengths[m][idx], weights=w))`: a distance-weighted
  **mean** in tokens. It is *not* the bucket-midpoint expectation.
* `length_bucket_probs[m]`: float32 array of length `max_buckets` (16). For each neighbour,
  `b = min(int(len/bucket_size), 15)` and `probs[b] += w_f32`, then normalise. `bucket_size=64`; the last
  bucket is open-ended.
* `p_under_budget[m] = Σ_{len_i ≤ budget} w_f32_i / max(Σ w_f32, 1e-9)`, with `budget = batch[0].budget_tokens`
  (256 by default).
* **The single path `estimate()` differs** (ME:555-571): `length_bucket_probs=None` and
  `p_under_budget = 1.0 if E[len] ≤ budget else 0.0`, a hard step. `score` and `length_expected` agree
  with the batch path, up to encoder/FAISS numerical noise.
* `KNNModelEstimator` (dead) uses yet another definition: the bucket-midpoint expectation (ME:1105-1110).

### 2.8 Per-model outputs

The result is a dict keyed by the KNN `model_names` in pickle order (sorted). The scheduler looks it up
by `instance_meta[inst].model_name or inst._model_name` (CS:1338-1342), which must match the HF name
string **exactly**. On a mismatch, `quality=0.5` and `length=max_output_tokens` are used silently
(CS:1358-1361). `predicted_best_model` takes the first maximum in dict order (CS:1177-1180).

### 2.9 Quality normalisation (scheduler side, CS:1936-1972; validation CS:3025, CS:3075-3112)

* `quality_norm` is read from `slo_defaults` only. Default `"minmax"`. A top-level key raises; an
  unknown value raises; `"raw"` requires `slo_defaults.quality_norm_raw_ack`.
* `minmax`: `(q − q_min)/(q_max − q_min + 1e-8)`, with min and max taken over the **candidate instances**
  of this request (CS:1962-1968).
* `scaled`: `min(1, q·quality_gain)`, with `quality_gain` defaulting to 1.0.
* `raw`: `q`.
* A candidate whose quality is `None` raises (CS:1955-1961). A constant quality under minmax gives 0 for
  every candidate (the prompt-blind behaviour).
* `norm_ctx` batch ranges (CS:2159-2173, 2227-2244, 2407-2455) are computed but **never read** by the
  scorer (dead).

### 2.10 How DeepEval scores are read

They are an index-time table: `per_judge_scores[m][DEEPEVAL_KEY]` float32 in [0,1], built by
`build_index` from `models[m].llm_judge_scores[DEEPEVAL_KEY]` of the training JSONL (KNN:136-161).
Nothing is scored at runtime. The v3 source file is `data/cara/train_scored_filtered.jsonl`
(93,622,480 B, md5 `0171225264aa4746af255143a5054fcf`, 14,919 rows); the alignment is verified exactly.

### 2.11 Missing labels and zero scores (code side)

* The index still maps missing labels to 0.0 (KNN:148-154), see §0.7. `output_lengths` has no zeros in
  the v3 index.
* Code-side v3 fixes that must survive:
  * deepeval absent → raise instead of 0.0 (ME:474-484, 707-744)
  * `KNNModelEstimator._get_score` refuses 0.0 (ME:1077-1086)
  * `predict_all_models` now returns `deepeval_score` and documents the `quality_score` bug (KNN:299-325)
  * `train_knn_estimator.evaluate` has no cross-metric fallback and raises on a missing deepeval score
    (TK:59-100)
* The "aggregator zero-score bug" (D2) itself is on the reporting side
  (`score = get(m) or get(stripped)` drops valid 0.0). It is not in these modules, but any rebuild
  aggregator must test `is None`, not truthiness.

### 2.12 Determinism

* The same query matrix and FAISS version give bitwise-identical indices and similarities.
* Sources of noise at roughly 1e-7:
  1. Sentence-transformers batch composition and padding. The adaptive batches (4/16/32) mean the same
     prompt was embedded alongside different batch-mates.
  2. FAISS switches to a BLAS sgemm kernel for `n ≥ distance_compute_blas_threshold` (default 20)
     queries, so batches of 4/16 and batches of 32 use different kernels.
  3. torch CPU thread count.
* These can only change the neighbour **set** when a similarity tie or near-tie falls at the k-th rank.
  The 17 duplicate pairs with differing labels make such flips visible.
* The summation order of `np.average` follows FAISS rank order; keep that order for bitwise equality.

### 2.13 Index variants (all measured)

| artifact (path) | size | md5 | N | baked k | signals | use |
|---|---|---|---|---|---|---|
| `Block/models/cara/knn/deploy_v3_k50_14919/knn_estimator.pkl` | 24,349,380 | `d2868fdc21b26b9e75dbb79d634dda19` | 14,919 | 50 | deepeval + Qwen7B judge + protectai + sim + ref | **v3 Qwen main** |
| `rb-exp/v3-exp/rebuttal_jul31_v3/v3/models/deploy_v3_k10_14919/` | 24,349,380 | `9cbbc30d15a35b3297c2e7cb88015144` | 14,919 | 10 | same arrays as k50 (embeddings bitwise equal) | the k ablation / pre-07-31 cells |
| `rb-exp/…/models/deploy_v3_f50_k50/` | 12,294,114 | `0467b12e8bca35a13f213075d9e0cd68` | 7,459 | 50 | deepeval etc. | **v3 50% labels** |
| `Block/models/cara/knn_gemma3/` | 12,938,770 | `9a09557ff5e4ac088c6e8347de334f3e` | 8,086 (18 dup rows) | 10 | Qwen7B judge **only** (no deepeval) | Gemma (July) |
| `rb-exp/…/models/knn_gemma3_k50/` | 12,938,770 | `94a28c195546831be1d131a67e66176d` | 8,086 | 50 | identical arrays to the k10 file | **Gemma v3 W1-B** (`cfg_gemma_k50.json`) |
| `Block/models/cara/_AS_SERVED_v1exact/quality/judge/p4_indexes/f0_uniformprior/` | 24,186,188 | `c651d1bd42aac78749e8d56660b38edd` | 14,966 | 10 | old schema: `llm_judge_scores ≡ 0.5`; `quality_scores`; no per_judge | **prompt-blind** (v1 only) |
| `…/p4_indexes/f50/` | 11,973,853 | `b1407f742d824ff5be887ae2eb137e2c` | 7,483 | 10 | judge only | v1 50% labels (a different corpus from v3) |
| `…/quality/judge/deploy_knn/` | — | `b9c5c606f39d7d8b132546abe8be9316` | 14,966 | 10 | judge | v1 as-served (NeurIPS submission) |

Gemma `model_names = ['google/gemma-3-12b-it','google/gemma-3-1b-it','google/gemma-3-27b-it','google/gemma-3-4b-it']`.
Gemma quality sits around 0.60 (Qwen judge) against about 0.40 for Qwen DeepEval. The two families are
on different scales and must never be mixed.

---

## 3. Latency path

### 3.1 Per-request flow in the batch worker (CS:2477-2509, identical in the prepper mode CS:2256-2294)

1. `expected_len = max(1, int(max_m est.length_expected))`, or `max(1, int(max_output_tokens or 256))`
   when there are no estimates (CS:2481-2486).
2. `req._batched_tpot = await _batched_tpot_per_request(req, instances, stats_map, expected_len)`.
   `xgb_batched_tpot_ms` is logged per request (CS:2487-2491, 2526).
3. `_predict_one_for_req` for every instance (CS:2492-2495) gives `tpot` and `e2e_latency`.
4. `_score_and_pick_instance`: `_lat_val` → `lat_score = 1 − lat/(max_lat + 1e-8)` (CS:1873, 1892, 1984).
5. `_update_local_stats(stats_map, inst, req)` (CS:1097-1111), which updates in-batch state:
   `num_waiting += 1`, `pending_prefill_tokens += num_prompt_tokens`,
   `pending_decode_tokens += int(req.predicted_tokens)`. `predicted_tokens` is the LPT key value, i.e.
   the max across models.

`stats_map` is fetched once per batch from `GET http://ip:port/instance_stats`, with a 3 s timeout, and
raises on any failure (CS:963-1019). The `stats_blind_absorb` mask is off for the Q1 and grid cells.

### 3.2 Instance-state keys (vLLM fork `2894a98`, `get_aggregated_stats`)

Emitted: `num_running, num_waiting, num_active_decode_seqs, decode_ctx_mean, decode_ctx_p50,
decode_ctx_p95, decode_ctx_max, total_decode_context_tokens, pending_prefill_tokens,
pending_decode_tokens, kv_cache_utilization, kv_free_blocks, token_budget_per_iter, max_num_seqs,
num_preempted, timestamp`.

`pending_decode_tokens` sums the per-request `predicted_decode_tokens − num_output_tokens`, auto-extended
by 10 whenever output passes the prediction. RB sends that prediction as
`request_json["num_predicted_output_tokens"] = req.predicted_tokens` (CS:1120-1121). The queue term
therefore feeds on RB's own KNN length predictions through the backend.

### 3.3 XGBoost feature vector (25 float32, order fixed; XGB:19-53, 97-113)

The booster files store the same 25 names in the same order (checked in the UBJ header), and
`xgboost_metadata.json.feature_names` matches. `inplace_predict` on a numpy array **does not validate
names**, so order is the contract.

| # | name | v3 serving source | value when absent |
|---|---|---|---|
| 0 | ema_decode_tok_per_s | **absent from /instance_stats** | 0.0 |
| 1 | ema_prefill_tok_per_s | **absent** | 0.0 |
| 2 | ema_decode_iter_ms | **absent** | 0.0 |
| 3 | decode_ctx_p50 | stats | 0.0 |
| 4 | decode_ctx_p95 | stats | 0.0 |
| 5 | decode_ctx_max | stats | 0.0 |
| 6 | num_running | stats | 0.0 |
| 7 | num_active_decode_seqs | stats | 0.0 |
| 8 | num_waiting | stats + dead-reckoning | 0.0 |
| 9 | pending_prefill_tokens | stats + dead-reckoning | 0.0 |
| 10 | pending_decode_tokens | stats + dead-reckoning | 0.0 |
| 11 | token_budget_per_iter | stats | 0.0 |
| 12 | prefill_chunk_size | **absent** | 0.0 |
| 13 | max_num_seqs | stats | 0.0 |
| 14 | kv_cache_utilization | stats | 0.0 |
| 15 | kv_free_blocks | stats | 0.0 |
| 16 | kv_evictions_per_s | **absent** | 0.0 |
| 17 | num_prompt_tokens | `req.num_prompt_tokens` = `request_json["prompt_len"]` (client-supplied; **0 if missing**, CS:438) | — |
| 18 | num_predicted_output_tokens | `int(expected_len)`, the max over models (§3.1 step 1) | — |
| 19 | req_decode_ctx_avg | `P + (O−1)/2` | — |
| 20 | req_decode_ctx_max | `P + O − 1` | — |
| 21 | backlog_seconds | `pending_prefill/ema_prefill + pending_decode/ema_decode`, EMAs floored to 1.0 when 0 or missing (XGB:71-73, 82) | — |
| 22 | request_prefill_time_est | `P/ema_prefill` (= P at serving) | — |
| 23 | request_decode_time_est | `O/ema_decode` (= O at serving) | — |
| 24 | ctx_relative_to_p95 | `req_decode_ctx_max/(decode_ctx_p95 or 1.0)` (XGB:74, 85) | — |

Missing state keys **and** falsy values both become 0.0 (`float(ss.get(f, 0.0) or 0.0)`, XGB:107). The
1.0 floor is a divide-by-zero floor, documented as an open question (XGB:60-70), not a
missing-value default. The same `compute_derived_features` was used in training
(`train_xgboost.py:135-160`), so the floor itself is not a skew; the missing EMA inputs are (§0.2).

### 3.4 Model file layout

`CARA_TPOT_DIR` (default `<tree>/models/cara/latency/deploy_tpot`, CS:3649-3652) contains:

* `xgboost_metadata.json` (820 B, md5 `8d66b0e33f70a4eb5b788ee79d6b46d0`):
  `instance_types = [qwen2.5-14b_v100, qwen2.5-3b_a30, qwen2.5-72b_a100, qwen2.5-7b_a30]`, 25
  `feature_names`
* `{instance_type}.xgb` in UBJSON, objective `reg:squarederror`, target = TPOT in **seconds/token**, no
  log target:

| file | bytes | md5 |
|---|---|---|
| `qwen2.5-14b_v100.xgb` | 317,310 | `63b1129ffe690a3b67e422e79ed24d6c` |
| `qwen2.5-3b_a30.xgb` | 120,803 | `c5caf54d5dc04d238d0126240f92390c` |
| `qwen2.5-72b_a100.xgb` | 161,418 | `da715c2cb7a3f22302c753383acf8285` |
| `qwen2.5-7b_a30.xgb` | 277,420 | `45eaaf9895608892333004cf0ff04d21` |

* `training_metrics.json` (still lists the removed `qwen2.5-3b_p100` and `unknown_unknown` tiers).
  Validation MAE: 3B 0.22 ms, 7B 0.51 ms, 14B 0.88 ms, 72B 0.75 ms.

`XGBoostLatencyPredictor.load` (XGB:212-231) → `xgb.Booster().load_model(path)` for each listed tier.
The metadata was edited on 2026-07-31 to drop the P100 zombie tier; its md5 differs from the v1
as-served manifest value `d22970291997398cecdee0b30d79f2ba`. xgboost 3.2.0.

### 3.5 Learned-arm prediction (CS:1283-1322)

* Group instances by `instance_meta[id].instance_type`; instances whose tier has no booster are skipped.
* Per tier: `X = stack(build_feature_vector(stats, P, O))`, `_TPOT_PREDICT_CALLS += 1`,
  `preds = booster.inplace_predict(X)`.
* Clamp: `tpot = max(0.001, p)` seconds/token.
* An instance left out of `out` falls to the legacy branch. With `--scheduler-config` there is no
  `learned_predictors` entry, so the result is `ttft=1.0, tpot=0.05`,
  `e2e = expected_length × 0.05` (CS:1403-1406, 1446-1449, 1491-1492). **Silent.** The fleet check at
  CS:3669-3679 only covers instances that *have* metadata.

### 3.6 Queue-aware envelope (CS:1389-1402; also CLP:379-407, 850-895)

```
running = int(ss.num_running or 0); waiting = int(ss.num_waiting or 0)
max_seqs = int(ss.max_num_seqs or 256)            # 256 = silent default
own = max(1, int(E[len_model] or 1))
if running < max_seqs and waiting == 0:  e2e = own * tpot
else:                                    e2e = (pending_decode_tokens / max(running, 1) + own) * tpot
```

After the first assignment to an instance within a batch, `num_waiting ≥ 1` (dead-reckoning), so that
instance is in the queued regime for every later request in the batch.

### 3.7 Static arm (v3, config-driven)

* **Config**: `slo_defaults.static_tpot_ms_per_tok = {tier_key: ms_per_token}`, set through the
  scheduler config or `POST /v1/config`. The payload `"static_tpot_ms_per_tok": null` selects the
  learned arm.
* **Resolution** (CS:1210-1236, "ONE resolution for both consumers"): returns `None` for the learned arm.
  A non-dict or empty dict raises; a non-number raises. Keys are `str(k).strip().lower()`; values are
  converted to s/token as `float(v)/1000.0`.
* **Matcher** `_static_tpot_for(name, smap)` (CS:1239-1253): tries keys longest first, returns the first
  key that is a **substring** of `name.lower()`, and raises if none matches (D13).
* **Predict boundary** (CS:1269-1282): when static is active, returns
  `{id: max(0.001, const(instance_type or inst._model_name))}`. **The booster is never called**, so
  `_TPOT_PREDICT_CALLS` stays unchanged. This is the ASSERT-5 zero-call gate
  (`v3_scripts/tpot_zero_gate.py`: static requires calls-delta == 0 and every batch's
  `xgb_batched_tpot_ms ≤ 1.0`; learned requires delta > 0 and a mean above 1.0 ms).
* **Scoring boundary** (CS:1860-1871): `_lat_val = _static_tpot_for(pred["instance_type"]) × pred["expected_length"]`.
  This is **queue-blind** and uses the float E[len]. Fallbacks differ between the two consumers:
  CS:1280 uses `instance_type or _model_name`, while CS:1868 uses `pred.get("instance_type", …)`, which
  holds `""` when metadata is missing, and `""` raises.
* **Presets on record**:
  * v3 Q1 (recalibrated): `{"3b":10.7,"7b":19.9,"14b":16.5,"72b":42.6}` ms/tok
    (`v3_scripts/s2_q1.sh:79-80`; cell provenance `posted_payload`)
  * older nominal: `3b=12.3,7b=20.6,14b=18.4,72b=47.4` (route-balance public
    `ROUTE_BALANCE_STATIC_LAT`, the queue-blind `_lat_val` form)
  * Gemma measured median ITL: `gemma-3-1b_a30=5.02, gemma-3-4b_a30=12.88, gemma-3-12b_a100=21.55,
    gemma-3-27b_a100=26.28` (`gemma_cells_w1b.sh:48`)
  * dead Gemma `instance_metadata.static_tpot_ms_per_token`: 4.799/12.933/21.376/25.47
* **Merge hazard**: `POST /v1/config` merges `slo_defaults` (CS:780-781), so a static prior survives into
  the next cell unless it is explicitly nulled (CS:851-860). The runners assert the resolved value
  before each cell.

### 3.8 The value that enters the objective

| arm | `pred["tpot"]` | `pred["e2e_latency"]` | **scored `lat_i`** |
|---|---|---|---|
| learned (v3 default) | XGBoost(tier, state, P, O_max) | envelope(tpot, state, E[len_m]) | = `e2e_latency` |
| static (v3 Q1) | const[tier] | envelope(const, state, E[len_m]) | **`const[tier] × E[len_m]`** (no queue term) |
| static (v2 W1-B, not in any git tree) | const[tier] | envelope(const, state, E[len_m]) | = `e2e_latency` (queue-aware) |

### 3.9 Legacy and sidecar latency paths (dead in v3)

`CaraLearnedPredictor.predict_tpot/predict_ttft/predict_e2e` (CLP:291-407),
`XGBoost3ModelSidecarPredictor` (CLP:727-895), `LSTMSidecarPredictor`, `RooflineSidecarPredictor`,
`_OpportunisticBatcher`, and the HTTP sidecar `_call_sidecar_predict_latency` (CS:1022-1070, reachable
only with `CARA_INPROC_PREDICTOR=0`). The `CARA_DUMMY_SIDECAR` and `CARA_DUMMY_ESTIMATOR` debug gates
(CS:1370-1376, 2120-2129) are also dead. Do not port any of these.

---

## 4. Every config key read, with its default (SILENT = applied without an error or warning)

### 4.1 `model_estimator` block (scheduler config)

| key | default | where | behaviour |
|---|---|---|---|
| `model_estimator` | — | CS:3203 | absent → no estimator → startup raise for `cara` (CS:3584-3588) |
| `.type` | `"default"` | CS:3205 | **SILENT**: any unknown string → DefaultModelEstimator |
| `.score_type` | `"deepeval"` | ME:95 | **SILENT** default (validated at startup only for deepeval). Legacy-path logs print a `'judge'` default (CS:3316, 3323), which is misleading. |
| `.bucket_config.bucket_size` | 64 | ME:100 | **SILENT**; drives the neighbour histogram |
| `.bucket_config.max_buckets` | 16 | ME:101 | **SILENT** |
| `.bucket_config.max_length` | 1024 | ME:102 | dead in scope |
| `.bucket_config.fused_model_dir`, `.model_dir`, `.model_map` | `""`, `""`, `{}` | ME:233, 256-257 | dead in scope (a configured fused dir that fails raises) |
| `.length_config.fused_model_dir` | `""` | ME:297 | dead in scope |
| `.quality_config.model_dir` | `""` | ME:139 | `""` → **warning only, KNN disabled** → every estimate `{}` → the silent 0.5 path. Missing path → raise. |
| `.quality_config.device` | `"cpu"` | ME:151 | SILENT |
| `.quality_config.index_type` | `"faiss_flat"` | ME:152 | SILENT default. Unknown value → warning and numpy (KNN:374-376). faiss ImportError → warning and numpy (KNN:335-340). |
| `.quality_config.k` | baked k | ME:179 | absent → baked k; mismatch → raise |
| `.quality_config.k_mismatch_ack` | `""` | ME:183 | non-empty → warn and serve the baked k |
| `.quality_config.embedding_model` | — | never read | **DEAD**: the pickle's `embedding_model_name` is used |
| `.quality_config.type` | — | never read | **DEAD** |

Pickle keys read by `KNNEstimator.load` (KNN:476-492): the similarity, reference, judge and per-judge
arrays default to `{}` (SILENT).

### 4.2 `slo_defaults` keys on the predictor path (all read in CS)

| key | default | where | behaviour |
|---|---|---|---|
| `static_tpot_ms_per_tok` | None (learned) | CS:1219, 1860 | absent/None = learned arm by design; malformed → raise |
| `quality_norm` | `"minmax"` | CS:1950, 3056-3058 | validated (CS:3075-3112) |
| `quality_gain` | 1.0 | CS:1951 | SILENT |
| `quality_norm_raw_ack` | `""` | CS:3106 | required for raw |
| `lpt_sort_key` | `"max"` | CS:2118, 2346 | SILENT. An unknown value maps to `max` on the batched path (CS:2146, 2378) but to `mean` on the prepper per-request path (CS:2155-2156), which is inconsistent. |
| `filter_enabled` | False | CS:1366, 1713 | SILENT (every deployed config sets false) |
| `budget_tokens` | 256 | CS:469 | request RSO overrides |
| `bucket_size` | 64 | CS:1700 | legacy filter only |
| `instance_capacity[tier]` | 8 | CS:1091, 1914-1916 | SILENT; balance term only when kv stats are absent |
| `instance_config[id].kv_block_size_tokens` | 16 | CS:1903-1905 | SILENT (balance term) |
| `batch_config.{max_batch_size,batch_timeout_ms,adaptive_sizing}` | 16 / 100 / True | CS:3623-3631 | overridden by adapt_params 4/16/32 (CS:373-381) |

### 4.3 `instance_metadata[tier]` (scheduler config)

* `model_name`, `gpu_type` → the D3/D4 tier mapping (CS:3236-3262)
* `cost_per_token`: **0.01 SILENT** (CS:1464)
* `cost_per_input_token` / `cost_per_output_token`: fall back to `cost_per_token`. They are overridden
  by the `model_deployment.json` `pricing` block (CS:3594-3611).
* `static_tpot_ms_per_token`: **DEAD**

### 4.4 Environment variables and CLI flags

| name | default | where | note |
|---|---|---|---|
| `CARA_INPROC_PREDICTOR` | `"1"` | CS:1377, 2265, 2480, 3636 | `0` → legacy/sidecar path with silent 0.05 defaults |
| `CARA_TPOT_DIR` | `<tree>/models/cara/latency/deploy_tpot` | CS:3650-3652 | missing dir → raise |
| `CARA_PREPPER_WORKERS` | 0 | CS:3704 | ≥1 → preppers + causal worker |
| `CARA_BATCH_WORKERS` | 1 | CS:3705 | >1 breaks causality |
| `CARA_HOST_GPU_MAP` | — | CS:183-195 | host→GPU override |
| `CARA_MODELS_DIR`, `EXPECTED_*_ENCODER` | — | CS:3419-3456 | ModernBERT audit only |
| `CARA_DUMMY_SIDECAR`, `CARA_DUMMY_ESTIMATOR` | — | CS:1370, 2120 | debug only |
| `CARA_UNWRAP_PROMPT`, `CARA_STATIC_TPOT_HEAD`, `CARA_STATIC_LAT` | — | none in v3 | **no reader**; still exported by some runners (phase1, w1b) |
| `--scheduler-config` | — | CS:3565-3576 | missing file → raise |
| `--predictor-config` | — | CS:3577-3579 | **ignored when `--scheduler-config` is given** |
| `--decision-log` | — | CS:3858-3869 | gate evidence |
| `--chat`, `--model-family` | — | CS:3471-3472 | the prompt is templated client-side; the estimator unwraps it |

### 4.5 Predictor config (`predictor_xgboost_3model.json`)

Read only by `CaraBasePredictorConfig.from_dict` (CFG) on the legacy path and by the sidecar server.
In S2 none of its keys is read: `predictor_type, schedule_trace_timeout, enable_data_collection,
xgboost_3model.{max_batch, model_paths_by_instance}, tpot_lookup, filter_enabled, xgboost_{ttft,tpot,e2e}.model_dir`.
On the legacy path the defaults are also silent: `LearnedPredictorConfig.bucket_config` reads
`raw["bucket_classifier"]` and `.quality_config` reads `raw["quality_predictor"]` (CFG), which are
different key names from the scheduler config.

---

## 5. Executed vs dead (MLSys scope: `--scheduling cara`, `type=default`, in-process predictor)

**`model_estimator.py`**

| symbol | status |
|---|---|
| `DEEPEVAL_KEY` import, `ModelEstimate`, `ModelEstimator` ABC | EXECUTED |
| `DefaultModelEstimator.__init__`, `_load_knn`, `model_names`, `_has_fused_*` (→ False) | EXECUTED |
| `_load_bucket_classifiers`, `_load_length_regressor` | EXECUTED but return early; bodies DEAD |
| `_predict_buckets_fused(_batch)`, `_predict_buckets(_batch)`, `_expected_length`, `_budget_compliance` | DEAD (ModernBERT/RoBERTa) |
| `estimate_batch` | EXECUTED: the FAISS branch and the neighbour-histogram branch. The numpy branch (ME:676-678, 684-688), bucket-model branches (ME:753-770, 783-786) and no-KNN branch (ME:807-812) are DEAD. |
| `estimate`, `_get_scores_all_models` | EXECUTED only by `/v1/estimate`, `cara_native` and `_schedule_cara`; not on the RB batch path |
| `estimate_length` | DEAD (no callers) |
| `PFSModelEstimator`, `KNNModelEstimator` | DEAD |

**`knn_estimator.py`**

| symbol | status |
|---|---|
| `unwrap_chat_prompt`, `DEEPEVAL_JUDGE_KEY`, `KNNEstimator.__init__/load/_get_encoder/_build_faiss_index(flat)/_find_neighbors/_distance_weights` | EXECUTED |
| `build_index`, `save` | OFFLINE (index build via TK) |
| `predict_all_models` | OFFLINE (TK evaluate) |
| `predict_length/quality/reference_similarity/similarity/judge/by_judge_key/bucket`, the IVF-PQ branch | DEAD for serving (`predict_quality` only via the legacy CLP) |

**`xgboost_predictor.py`**

| symbol | status |
|---|---|
| feature lists, `compute_derived_features`, `build_feature_vector` | EXECUTED (learned arm) |
| `XGBoostLatencyPredictor.__init__/load` | EXECUTED in **both** arms (startup) |
| `.predict`, `.predict_batch` | DEAD (the in-process path uses `booster.inplace_predict` directly) |
| `.save` | OFFLINE |

**Imported package modules**

| symbol | status |
|---|---|
| `cara_learned_predictor.CaraLearnedPredictor` | DEAD (legacy loader not called) |
| `LSTMSidecarPredictor`, `RooflineSidecarPredictor`, `XGBoost3ModelSidecarPredictor`, `_OpportunisticBatcher` | DEAD (sidecar server). Latent bugs: the sidecars call `self._static_tpot_or_die`, which only `CaraLearnedPredictor` defines (CLP:576, 579, 838, 846) → AttributeError; LSTM and Roofline pass `hostname=`/`port=` to `ScheduleTraceClient(backend_host, backend_port, …)` (CLP:532-535, 620-623 vs `schedule_trace_client.py:22`) → TypeError. |
| `cara_predictor_config.*`, `base_predictor.CARA_BasePredictor`, `data_structures.*`, `estimators/{pfs,lstm,lstm_v2,mlp,roofline,linear}_*`, `offline_training/train_fused_bert_predictor.load_fused_model` | DEAD for scope |

---

## 6. Bug-fix markers that must survive (as behaviour and as comments or tests)

| where | marker | meaning to preserve |
|---|---|---|
| KNN:18-44; ME:635-643; CS:31-46, 107-120, 2384-2388, 3059 | CHOKE POINT / chat-template skew (2026-07-31) | unwrap unconditionally at the embedding boundary, idempotent; no switch |
| KNN:47-51; ME:23-30 | v3 FIX single `DEEPEVAL_KEY` (D13/D25 hardcoded-literal class) | one constant. Violated by ME:1072 (dead class). |
| KNN:299-305 | v3 FIX `predict_all_models` `quality_score` phantom key | return `deepeval_score`; the caller must not default it |
| ME:126-161 | v3 FIX fail-closed load (D30 guard, F1) | configured-but-missing/unloadable → raise |
| ME:163-199 | dead `k` key; baked-k authority; mismatch raises (2026-07-31) | baked k wins; disagreement is fatal |
| ME:200-219 | zero-model raise; D29 startup deepeval assertion | fail at startup, not mid-cell |
| ME:474-484 | v3 FIX D29 (single path) / D1 | no 0.0 for an absent deepeval array |
| ME:651-667 | v3 FIX D19 batched FAISS + "CORRECTED 2026-07-30" | one `search(Q,k)`; distances may differ about 1.2e-7 from numpy; the neighbour set can differ on duplicates |
| ME:707-746 | v3 FIX D29 (batch path) / D1 | branches mirror estimate(); a missing signal raises |
| ME:1077-1086 | never fabricate 0.0 | raise on a missing array |
| TK:59-100 | v3 FIX metric-substitution chain (24.72% of values substituted) and `quality_score` constant-0 | deepeval only; explicit `is None` |
| XGB:60-74 | divide-by-zero floor 1.0; "OPEN QUESTION … deliberately NOT changed" | keep the floor bit-exact; an owner decision is pending |
| CLP:320-342 | v3 FIX `_static_tpot_or_die` (Gemma collapsed to 0.05 s/tok) | never default a TPOT prior |
| CLP:139-145 | `raw_config` AttributeError fix | legacy only |
| CS:1196-1207 | `_TPOT_PREDICT_CALLS` measured per cell | expose the counter |
| CS:1210-1253 | one static resolution; longest-key-first; raise on no match (v3/D13) | |
| CS:1269-1274 | ZERO-CALL STATIC ARM (author 2026-08-03) | static never calls the booster |
| CS:1474-1488 | v3 FIX G4 / D20 | fallback e2e = `E[len]·tpot`, no ttft term; E2E is analytic |
| CS:1848-1859 | static prior is config-driven, never an env var (2026-07-31) | |
| CS:1865-1871, 1974-1983 | v3 FIX D13 shared matcher; "arm-4 correctness": the numerator uses the same `_lat_val` as `max_lat` | |
| CS:1936-1961, 3025, 3075-3112 | `quality_norm` minmax default flip; top-level key fatal; raw needs ack; quality None → raise | |
| CS:47-77, 1876-1890 | decision log always on | |
| CS:157-195, 3226-3262, 3329-3363 | v3 FIX D26 host→GPU map; D3/D4 (model, GPU) tier match; ambiguity raises | |
| CS:168-177, 3269-3279, 3392-3393 | v3 FIX D31: never swallow `SchedulerConfigError` | |
| CS:3412-3419 | F2: tree-relative artifact root | |
| CS:3566-3588 | missing `--scheduler-config` raises; FAIL CLOSED with no estimator (2026-08-01) | |
| CS:3637-3686 | FAIL-CLOSED in-process XGB load; fleet-coverage raise; D4 zombie-tier warning | keep for the learned arm; do **not** require it for static (§0.3) |
| CS:851-860 | ARM LEAK: `/v1/config` merge persists the static prior | |

(D2, D5 and D12′ are reporting-side or out-of-module defects that the rebuild's aggregation and
serving cap must also respect: zero-safe `is None` aggregation, and a KNN corpus length cap of 1023
against the serving cap. v3 cells use cap=1024.)

---

## 7. Proposed rebuild interfaces

```python
# rb/predict/types.py
ScoreType = Literal["deepeval", "judge", "similarity", "reference"]   # no "combined"/fallthrough

@dataclass(frozen=True, slots=True)
class ModelEstimate:
    model_name: str
    length_expected: float            # tokens; Σ w_i·len_i / Σ w_i (float64), FAISS rank order
    length_bucket_probs: np.ndarray   # (max_buckets,) float32; b=min(int(len/bucket_size), max_buckets-1)
    p_under_budget: float             # Σ_{len_i<=budget} w32_i / max(Σ w32, 1e-9)
    score: float                      # Σ w_i·q_i / Σ w_i, q = array selected by score_type
    score_type: ScoreType

@dataclass(frozen=True, slots=True)
class QualityOverride:                # prompt-blind ablation, explicit and logged
    mode: Literal["constant", "per_model_prior"]
    value: float | None = None        # v1 artifact semantics = constant 0.5; per_model_prior = corpus mean per model (owner to choose)

@dataclass(frozen=True, slots=True)
class KnnIndexSpec:
    model_dir: Path                   # REQUIRED, no "" = disabled
    pkl_md5: str                      # REQUIRED; verified before unpickling (new)
    score_type: ScoreType             # REQUIRED (Qwen: deepeval; Gemma: judge)
    k: int                            # REQUIRED; must equal the baked k, else raise
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"      # must equal the pickle, else raise
    encoder_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    device: str = "cpu"
    index_type: Literal["faiss_flat"] = "faiss_flat"                  # numpy only via an explicit test flag; faiss ImportError = raise
    bucket_size: int = 64
    max_buckets: int = 16
    quality_override: QualityOverride | None = None
    label_subset: Literal["full", "f50"] = "full"   # informational; the artifact defines it

class QualityLengthEstimator(Protocol):
    model_names: tuple[str, ...]      # pickle order (sorted)
    k: int; corpus_size: int; score_type: ScoreType
    def estimate_batch(self, prompts: Sequence[str], budget_tokens: int) -> list[dict[str, ModelEstimate]]: ...
    def estimate(self, prompt: str, budget_tokens: int) -> dict[str, ModelEstimate]: ...  # := estimate_batch([prompt])[0]

class KnnQualityLengthEstimator(QualityLengthEstimator):
    def __init__(self, spec: KnnIndexSpec) -> None: ...   # eager: md5 → load → validate every model has the
                                                          # score_type array → build IndexFlatIP → load encoder
    def encode(self, prompts: Sequence[str]) -> np.ndarray: ...     # unwrap_chat_prompt applied here ONLY; (n,384) f32, normalized
    def search(self, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...  # (n,k) int64, (n,k) float64 = (1.0 - sims_f32).astype(f64)
    def aggregate(self, idx: np.ndarray, dist: np.ndarray, budget_tokens: int) -> dict[str, ModelEstimate]: ...

def unwrap_chat_prompt(p: str) -> str: ...   # verbatim KNN:30-44 (Qwen + Gemma regexes)
```

```python
# rb/predict/latency.py
TPOT_FEATURES: tuple[str, ...] = (...25 names, XGB:19-53 order...)

@dataclass(frozen=True, slots=True)
class TierQuery:
    instance_id: str
    tier: str                         # instance_type, e.g. "qwen2.5-3b_a30" (REQUIRED; never fall back to the model name)
    state: Mapping[str, float]        # /instance_stats body + in-batch dead-reckoning
    num_prompt_tokens: int            # request prompt_len (raise if missing: v3 defaulted to 0)
    feature_output_tokens: int        # v3: max(1, int(max_m E[len_m])), shared by all tiers
    own_expected_length: float        # E[len] of THIS instance's model (float)

@dataclass(frozen=True, slots=True)
class LatencyEstimate:
    tpot_s: float                     # seconds/token, >= 0.001
    e2e_queue_aware_s: float          # envelope(tpot_s, state, int(own_expected_length))
    score_latency_s: float            # the value that enters the objective (see envelope)
    source: Literal["static", "xgboost"]

class TpotModel(Protocol):
    kind: Literal["static", "xgboost"]
    predict_calls: int                # booster invocations; static: always 0
    async def tpot_s(self, queries: Sequence[TierQuery]) -> dict[str, float]: ...

class StaticTPOT(TpotModel):          # needs NO xgboost artifact, NO xgboost import
    def __init__(self, ms_per_token: Mapping[str, float]) -> None: ...   # malformed/empty → raise; stored as float(v)/1000.0
    def lookup(self, tier: str) -> float: ...                            # longest-key-first substring; raise on no match

class XGBoostTPOT(TpotModel):
    def __init__(self, model_dir: Path, fleet_tiers: set[str], md5: Mapping[str, str]) -> None: ...  # coverage check = raise
    def features(self, q: TierQuery) -> np.ndarray: ...   # float32 (25,), bit-identical to build_feature_vector,
                                                          # INCLUDING zero-fill of absent EMA keys (v3 behaviour)
    # tpot_s(): group by tier; one booster.inplace_predict per tier via run_in_executor; clamp max(0.001, p)

@dataclass(frozen=True, slots=True)
class LatencyPolicy:
    tpot: TpotModel
    score_envelope: Literal["queue_aware", "length_only"]   # REQUIRED, no default:
    # learned (v3)       = XGBoostTPOT + queue_aware
    # static  (v3 Q1)    = StaticTPOT  + length_only   -> score = const × own_expected_length (float)
    # static  (v2 W1-B / MLSys target) = StaticTPOT + queue_aware

def queue_aware_e2e_s(tpot_s: float, state: Mapping[str, float], own_output_tokens: float) -> float: ...  # CS:1389-1402 verbatim
```

Counters and observables (keep): `score_batched_calls`, `score_perreq_calls`, `tpot_predict_calls`
(guarded by a lock, or incremented on the event loop after `gather`), and `EFFECTIVE_CONFIG` including
`knn_pkl_md5`, `score_envelope` and `tpot_kind`. Decision-log records should add the per-candidate
`state` snapshot. That adds observability without changing behaviour, and it lets learned-arm
decisions be replayed later.

---

## 8. Golden-test recipe (v3 oracle ↔ rebuild)

Environment: a **new** venv (R1). Python 3.10.12, numpy 2.2.6, faiss-cpu 1.13.2, xgboost 3.2.0,
sentence-transformers 5.4.1, torch 2.9.0 (CPU), transformers 4.56.0, MiniLM @ `1110a24…`. Use
`OMP_NUM_THREADS` and `torch.set_num_threads` at a fixed value. Oracle = the Block tree at `984ea18` on
`sys.path`, imported read-only. Importing CS has side effects (uvloop policy, FastAPI app,
`logging.basicConfig(filename='experiment_output/logs/cara_serve.log')`), so run it from a scratch cwd
that contains that directory, or vendor CS:1210-1253 and CS:1389-1402 verbatim into the test (with an
md5 check of the source file).

**T1 — retrieval and aggregation, encoder-free, bitwise.**
* Build a text→vector table of query embeddings:
  * (a) 2,000 corpus rows perturbed with seeded N(0, 0.05²) noise and renormalised
  * (b) all 17 duplicate-pair rows exactly, to exercise ties
  * (c) the 3,534 real eval-prompt embeddings produced once by T2
* Replace `me._knn._encoder` with a stub whose `.encode(texts, **kw)` returns
  `np.stack(table[unwrap(t)])`.
* Run v3 `DefaultModelEstimator({"score_type":…, "quality_config":{"model_dir":…, "k":50, "index_type":"faiss_flat"}}).estimate_batch`
  and the rebuild on **identical chunkings** of 1, 4, 16 and 32. Chunking must match because of the
  FAISS BLAS threshold at 20.
* Assert, bitwise: `idx`; `dist`; `score`; `length_expected`; `length_bucket_probs`; `p_under_budget`.
* Repeat for: `deploy_v3_k50_14919` × {deepeval, judge}; `deploy_v3_f50_k50` × deepeval;
  `knn_gemma3_k50` × judge; `f0_uniformprior` × judge (baked k=10, so the config must say 10).
* Also check `estimate()` against `estimate_batch([p])`, on `score` and `length_expected` only.

**T2 — encoder equivalence** (loads MiniLM; run on the GPU VM or CPU per the resource policy).
* Unwrapped eval prompts from `Block/data/cara/best-route-v3-test-3534-eval.jsonl` (1,577,938 B, md5
  `15ed0e67ffa50e0d17955127c144a566`, 3,534 rows, keys `id, prompt, source`).
* v3 `_get_encoder().encode(batch, normalize_embeddings=True, batch_size=max(n,32))` against the rebuild
  with the same batch composition → bitwise.
* Also record max|Δ| between single and batched encodes, to size the tolerance for T3.

**T3 — end-to-end oracle against banked cells** (the prompts are templated, so this also covers unwrap).
* For each `response_details[i]`: `rebuild.estimate(prompt)[m].score` vs `all_model_scores[m]` for all 4
  models, and `[model].length_expected` vs `predicted_length`.
* Pass criteria: |Δscore| ≤ 1e-6 and |Δlen| ≤ 1e-3 on ≥ 99.5% of rows. Every exceedance must show a
  similarity gap between rank k and rank k+1 below 1e-5 (neighbour-set flip); anything else fails.
* Cells to use:
  * `/home/wd312/Code/llm/rb-exp/v3-exp/cells/s2_q1_PROD_zerocall/q1_{static,learned}_s1_l{12,24}_n3534.json`
    (≈20 MB each; Qwen, deepeval, k50, cap 1024; `provenance.json` gives `knn_pkl_md5 d2868fdc…` and
    the posted payload)
  * `/home/wd312/Code/llm/rb-exp/v3-exp/cells/s2_p1_l{6,12,18,24,30}/*` (RB grid)
  * `/home/wd312/Code/llm/rb-exp/v3-exp/rebuttal_jul31_v3/v3/rebuttal_fix/w2a_{full,f50}_{de,ju}_l12_n3534.json`
    (50% labels, and the judge signal; the Jul-31 tree ran with the `CARA_UNWRAP_PROMPT=1` era gate)
  * `…/rebuttal_fix/w1b_rb{532,811,uni}_l24_n1979.json` (Gemma: `knn_gemma3_k50`, judge, Gemma template)
* Do **not** use pre-07-31 cells, `post3_w1b`, or `*_RAW_SUPERSEDED`. Their scores match the wrapped
  embedding.

**T4 — TPOT features and heads.**
* Inputs:
  * (a) a synthetic grid of `/instance_stats`-shaped dicts containing only the 12 keys the fork emits:
    idle, all zeros, `running == max_num_seqs`, `waiting > 0`, `decode_ctx_p95 = 0`, key absent vs
    value 0
  * (b) 5,000 `schedule_state` rows from `Block/data/cara/latency_data/all/latency_test_tagged.jsonl`
    (88,526,496 B), which do carry the EMA fields
  * each crossed with `P ∈ {0, 1, 88, 4000}` and `O ∈ {1, 97, 1023}`
* Assert v3 `build_feature_vector` == rebuild `features` (float32, bitwise).
* Load `Block/models/cara/latency/deploy_tpot` in both (`XGBoostLatencyPredictor.load` against the
  rebuild); `inplace_predict` must be bitwise equal. Cross-check against `Booster.predict(DMatrix)` with
  |Δ| ≤ 1e-7. Check the clamp at 0.001.

**T5 — envelope and static arm.**
* Table tests of `_static_tpot_map`/`_static_tpot_for` against the rebuild `StaticTPOT`:
  * Qwen short keys (`3b/7b/14b/72b`) against the tiers `qwen2.5-{3b_a30,7b_a30,14b_v100,72b_a100}`
  * Gemma keys, including a `"2b"`/`"12b"` and `"7b"`/`"27b"` longest-first case
  * unmatched → raise; `{}`/`"x"`/non-numeric → raise; `None` → learned
* `queue_aware_e2e_s` against CS:1389-1402 on the T4 grid (bitwise).
* Replay `q1_static_s1_l12_n3534.decisions.jsonl`: for every candidate, `lat_s == const[tier] × exp_len`
  bitwise with `const = 16.5/1000` etc. The rebuild's `length_only` envelope must reproduce it. For the
  learned logs, only `lat_s == learned_e2e_s` is checkable, since state is not logged.
* **Zero-call gate**: run the rebuild static arm over 1,000 synthetic requests with booster stubs that
  raise when called. Expect `predict_calls == 0` and no exception, and no xgboost import in a
  static-only process (check `sys.modules`).

**T6 — startup parity.** Each of the following must raise in both the oracle and the rebuild:
* k mismatch (`cfg_de_f50.json` as preserved, k=10 against a baked 50)
* missing `model_dir`
* `score_type="deepeval"` against the Gemma index
* a zero-model pickle
* a malformed static map
* a learned arm whose fleet tier has no head

Declared changes, each listed in the spec with owner approval:
* v3 warns on an empty `model_dir`; the rebuild raises
* faiss ImportError: v3 falls back to numpy; the rebuild raises
* unknown `score_type`: v3 falls through to reference; the rebuild raises
* a model with no estimate: v3 uses 0.5/max; the rebuild raises

Artifact inventory (sizes and md5s in §2.13 and §3.4). The ground-truth test labels are in
`Block/data/cara/scored/test_scored_filtered.jsonl` (22,356,273 B; for offline best-model accuracy only;
not used by the serving oracle).

---

## 9. Hazards for a porter

* **H1**: The static envelope is ambiguous across campaigns (§0.1). Put it in config, make it required,
  record it in every cell, and get the owner's decision before the window.
* **H2**: The learned heads' EMA features are zero at serving (§0.2). Reproduce first; any fix is a new
  arm.
* **H3**: v3's static arm depends on XGBoost artifacts, and Gemma RB cannot start in-process on the v3
  tree (§0.3).
* **H4**: Three different length values feed the latency path: the feature uses `int(max_m E[len])`, the
  envelope uses `int(E[len_m])`, and the static score uses float `E[len_m]` (§0.4). Dead-reckoning adds
  `int(req.predicted_tokens)` (the max) to `pending_decode_tokens`.
* **H5**: Dead config keys and ignored files: `--predictor-config`, `tpot_lookup`,
  `quality_config.embedding_model/type`, `instance_metadata.static_tpot_ms_per_token`, and the
  `CARA_UNWRAP_PROMPT`/`CARA_STATIC_TPOT_HEAD`/`CARA_STATIC_LAT` env vars. A porter who "honours" them
  changes behaviour.
* **H6**: k authority differs by class (§0.6). `fix_config_k.sh` reads `getattr(obj, "k")` on the
  unpickled **dict**, which always returns None, so it cannot report the baked k. Read `state["k"]`.
* **H7**: 40 missing labels are stored as 0.0 (§0.7). The rebuild index builder should store a mask. For
  equivalence, serve the v3 pickles as they are.
* **H8**: Silent defaults on the path:
  * `quality=0.5` / `length=max_output_tokens` when an estimate is missing (CS:1358-1361)
  * TPOT 0.05 s and TTFT 1.0 for instances without metadata or head (CS:1446-1449)
  * `prompt_len` → 0 (CS:438)
  * `max_num_seqs` → 256 (CS:1392)
  * `cost_per_token` → 0.01 (CS:1464)
  * `score_type` unknown → reference (ME:736-737)
  * `index_type` unknown or faiss missing → numpy (KNN:335-340, 374-376)
  * the single-path `judge/similarity/reference` still returns 0.0 when the array is missing
    (ME:455-472)
* **H9**: `estimate()` and `estimate_batch()` differ on `p_under_budget` and `length_bucket_probs`
  (§2.7). The preflight probe (`/v1/estimate`) therefore does not observe the serving path
  (`2026-07-31_GATES.md:614`).
* **H10**: Unwrap edge cases: first user turn only; a missing end tag returns the full templated text; a
  nested template inside user content is not handled; `.strip()`. Keep exactly one implementation (v3
  has two identical copies, CS:80-120 and KNN:30-44).
* **H11**: Model-name matching between KNN `model_names`, `instance_metadata.model_name` and
  `model_deployment.hf_model_name` is exact-string. Qwen uses base names (`Qwen/Qwen2.5-3B`); Gemma uses
  `-it` names (a `-pt` name leaves every lookup unresolved; `gemma_deploy_v3.sh:28-30`).
* **H12**: Ties and numerics: 17 duplicate pairs with conflicting labels; FAISS BLAS threshold at 20;
  sentence-transformers batch composition; the numpy fallback computes float32 weights. Match batching
  exactly in bitwise tests; use tolerances only against banked cells.
* **H13**: The weight singularity `1/(d+1e-6)` makes an exact corpus duplicate dominate its query.
* **H14**: Counters are cumulative per process and advanced by warmup requests. The gates use deltas.
  The decision-log `req` is the server arrival counter, not the client `request_id`. `/v1/config`
  merges `slo_defaults` (the arm-leak hazard); the rebuild should replace the arm block per POST, or keep
  the merge plus the pre-cell assertion (owner call).
* **H15**: Lazy encoder and FAISS initialisation happen in the first executor call. Eager loading changes
  only first-batch timing; state it if you change it.
* **H16**: `pickle.load` of the artifacts means arbitrary code on load. Verify the md5 first, or convert
  once to `.npz` plus a manifest, with a test that the arrays are bitwise equal.
* **H17**: `xgboost_metadata.json` was edited (P100 removed); `training_metrics.json` and `deploy_ttft/`
  still carry the `3b_p100`/`unknown_unknown` tiers. Load only the tiers the metadata lists and the fleet
  needs, and warn on extras (the D4 shape).
* **H18**: Gemma: judge signal only, and two pickles differ only in baked k. The posted W1-B used k=50
  with the v2-tree static head (queue-aware). The July configs used k=10 with raw normalisation.
* **H19**: Mixing corpora: the v1 prompt-blind and f50 indexes are on the 14,966-row judge corpus, and
  the v3 indexes on the 14,919-row deepeval corpus. Pairing a v1 ablation index with v3 deepeval cells
  confounds corpus and signal (`rebuttal_fix_phase1.sh:17-20`). The prompt-blind arm needs a v3-corpus
  artifact, or a `QualityOverride` built from `deploy_v3_k50_14919`, with the constant vs per-model-prior
  semantics chosen by the owner.
* **H20**: E4 (MLSys plan): RB sends `num_predicted_output_tokens` = its KNN max length, and baselines
  send `max_tokens`. That value drives the fork's `pending_decode_tokens`, which RB's own queue term
  reads back. Keep sending it; baselines' dispatchers must be checked for reading it.
* **H21**: The dead `norm_ctx` batch ranges (CS:2159-2173, 2407-2455) price input+output cost, while the
  live cost is decode-only (CS:84-104). Do not revive them as live logic.
