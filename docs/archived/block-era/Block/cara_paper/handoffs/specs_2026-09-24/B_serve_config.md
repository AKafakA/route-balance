# Port spec B: serving layer, runtime config, stats fetch, pipeline host

Source of truth: `/home/wd312/Code/llm/Block/block/global_scheduler/cara/cara_serve.py` (3,873 lines, v3 campaign tree).
Ranges covered (read in full): **1–236, 398–1072, 2584–3025, 3025–3873**. Other lines were skimmed only to resolve references; they are cited as `[xref L…]` and belong to other spec parts.

Supporting evidence read:
- deployed v3 configs: `.../v3-exp/deployed_config/{scheduler_config_s2.json, model_deployment.json, host_configs.json, predictor_xgboost_3model.json}` (`scheduler_config_s2.json` is byte-identical after JSON normalisation to `block/config/cara/s2/scheduler_config_s2.json`, which `lib_serve_v3.sh:27` uses as `V3_CFG`)
- payload scripts: `.../v3_scripts/payloads_v3.sh`, `lib_post_config.sh`, `p1_pareto.sh`, `s2_q1.sh`, `lib_serve_v3.sh`
- factories and baselines: `routers/{factory,passthrough,best_route_4way,avengers_pro}.py`, `dispatch/{factory,base,round_robin,random_dispatch,shortest_queue,po2_queue}.py`, `filters/{factory,base}.py`
- instance client: `cara_instance/{Instance,vllm_instance}.py`
- benchmark client: `block/benchmark/cara/cara_end_point_func.py`, `benchmark_serving.py`

Legend: **IN-SCOPE** means the MLSys scope executes it. **DEAD** means the scope never executes it and the rebuild can drop it. **OBS** means it does not change routing but gates or aggregators read it, so it must be kept.

---

## 0. The v3 launch this spec reproduces

From `lib_serve_v3.sh:158-166`:

```
env CUDA_VISIBLE_DEVICES="" CARA_INPROC_PREDICTOR=1 $extra \
python3 -u -m block.global_scheduler.cara.cara_serve \
  --host 0.0.0.0 --port 8200 --model_config_path $MODEL_CFG \
  --scheduling {cara|pipeline} --chat \
  --scheduler-config $V3_CFG \
  --predictor-config block/config/cara/predictor_xgboost_3model.json \
  --host_config $HOST_CFG --decision-log $DECISION_LOG
```

What follows from this launch:
- Only two modes are launched: `cara`, 5 call sites, and `pipeline`, 7 call sites (`grep restart_v3`). No legacy `--scheduling shortest_queue|random|...` launch exists.
- `--predictor-config` is passed but never read, because `--scheduler-config` wins (if/elif at L3575-3579).
- `CUDA_VISIBLE_DEVICES=""` puts every scheduler-side model on CPU: the KNN MiniLM, the RB length head, the BEST-Route DeBERTa and the AvengersPro embedder.
- `CARA_PREPPER_WORKERS` and `CARA_BATCH_WORKERS` are unset, so native mode runs the legacy single `_batch_scheduler_loop` (L3704-3724).
- No `--workers` is passed (default 1). No `--root-path`, no SSL.

---

## 1. Request lifecycle of `POST /v1/completions` (L411-655)

### 1.1 Common prologue (all modes)

| Step | Line | Behaviour |
|---|---|---|
| 1 | 414 | `request_json = await request.json()`. No schema validation. |
| 2 | 415-417 | `num_requests += 1`. `request_id = request_json.get("request_id", str(num_requests))` and it is written back into `request_json`. The client always sends `request_id`. |
| 3 | 418 | `served_requests.append(request_id)`. Write-only list that grows without bound (DEAD). |
| 4 | 423-428 | Server sampling defaults are applied only if the key is absent: `repetition_penalty` (CLI default 1.0), `frequency_penalty` (1.2), `temperature` (0.0), each cast with `float()`. The client always sends 1.0 / 1.2 / 0.0, so these defaults matter only for curl warmups. |
| 5 | 430-434 | If `--chat`, set `request_json["use_chat_endpoint"] = True`. `VllmInstance.query_backend` then posts `/v1/chat/completions` with `messages=[{"role":"user","content":prompt}]` and `max_completion_tokens=max_tokens` (vllm_instance.py:96-131). |
| 6 | 438 | `num_prompt_tokens = request_json.get("prompt_len", 0)`. This is the **client-computed** token count. The server never tokenizes. |
| 7 | 439 | `max_output_tokens = request_json.get("max_tokens", 256)`. |
| 8 | 441-465 | Predictor-feedback fan-out, only if `--enable-predictor-feedback`. DEAD in scope, and `predictions` is never used. |
| 9 | 468 | `rso = request_json.get("rso") or request_json.get("request_specific_objective") or {}`. The benchmark sends `request_specific_objective`. |
| 10 | 469 | `budget_tokens = rso.get("budget_tokens", rso.get("budget", slo_defaults.get("budget_tokens", 256)))` |
| 11 | 470-472 | `ttft_slo_ms`, `tpot_slo_ms` and `quality_min` are taken from rso, else `slo_defaults`, else literals 5000 / 200 / 0.0. The literal 0.0 for `quality_min` is unreachable because the module dict has 0.3, see S2. |
| 12 | 473 | `budget_cost = float(rso.get("budget_cost", request_json.get("budget_cost", 0.0)) or 0.0)`. **RSO wins** here. `BatchRequest.__init__` L259-260 uses the **opposite precedence**: top-level wins, and it has no `or 0.0` guard (see H12). |
| 13 | 474 | `prompt_text = request_json.get("prompt", "")`. This is the chat-templated text the client sends (see H9). |

### 1.2 Mode dispatch (evaluated in this order)

| Order | Condition | Line | Path |
|---|---|---|---|
| a | `broadcasting_enabled and broadcast_model_list` | 478 | Broadcast (DEAD) |
| b | `scheduling == "pipeline" and router is not None and dispatcher is not None` | 546 | Pipeline (IN-SCOPE, baselines) |
| c | `scheduling == "cara" and batch_queue is not None` | 597 | Native batch (IN-SCOPE, RB) |
| d | else, and `scheduling == "pipeline"` | 615-620 | **raise RuntimeError** ("router/dispatcher are not initialized"). Traffic before `POST /v1/config` fails loudly with a 500. |
| e | else | 622-635 | Per-request legacy `_select_instance` (DEAD) |

### 1.3 Native batch path (`--scheduling cara`), L597-610

1. Build `BatchRequest(request_json, prompt_text, num_prompt_tokens, max_output_tokens, budget_tokens, ttft_slo_ms, tpot_slo_ms, quality_min, request_id)` at L599-609. `BatchRequest.__init__` [xref L247-279] also reads `budget_cost` (L259), `constraint_mode` (rso, then `slo_defaults.constraint_mode`, then "STRICT"), `relax_order`, `arrival_time=time.monotonic()`, and `predicted_tokens=float(max_output_tokens)`. It creates the future with `asyncio.get_event_loop().create_future()`.
2. `response_dict = await batch_queue.enqueue(breq)` at L610. This puts the request on an `asyncio.Queue` and awaits the future.
3. The worker [xref `_batch_scheduler_loop` L2330+] runs collect_batch, then the batched estimator (in `run_in_executor`), then the LPT sort, then `_fetch_all_instance_stats(instances)` once per batch, then `adapt_params` (sizes the NEXT batch), then per request `_score_and_pick_instance`, then `_dispatch_and_resolve` [xref L1114-1193].
4. `_dispatch_and_resolve` sets `request_json["num_predicted_output_tokens"] = req.predicted_tokens or max_output_tokens`, applies the budget clamp (same formula as pipeline, see §5.4), calls `inst.query_instance`, and attaches the budget fields, `scheduling_overhead_breakdown` (only if `overhead` is truthy), `predicted_quality`, `predicted_length`, `p_under_budget`, `predicted_best_model`, `predicted_best_score`, `predicted_best_hit` and `all_model_scores` (only if `req.model_estimates`). It then resolves the future.
5. A worker exception calls `req.future.set_exception(e)` [xref L2574-2581]. The await at L610 re-raises it and the generic handler returns a 500 (L638-655). `selected_instance` stays `None` in native mode, so the error body says `"Unknown"`.

### 1.4 Pipeline path (`--scheduling pipeline` + POST `/v1/config`), L546-596

1. `selected_instance, predicted_output, sched_breakdown = await _select_via_pipeline(instances, ...)` at L548-558. Details are in §5.
2. `request_json["num_predicted_output_tokens"] = predicted_output` at L559. `predicted_output` is the **pre-clamp** `max_output_tokens` (see H10).
3. Budget clamp at L563-582, only when `budget_cost > 0` **and** `pl_c_out > 0`. See §5.4.
4. `response_dict = await selected_instance.query_instance(request_json, predicted_num_decode_tokens=predicted_output)` at L583-586.
5. `response_dict["scheduling_overhead_breakdown"] = sched_breakdown` is **always** attached (L587).
6. If `budget_cost > 0`, attach `budget_cost`, `actual_cost`, `affordable_output_tokens` (`pl_affordable_out or 0`) and `budget_exhausted` (L588-596).
7. No `predicted_quality` or other estimator fields are attached. The client defaults them to 0.0 or "".

### 1.5 Per-request legacy path, L611-635 (DEAD for scope)

`_select_instance(scheduling, instances, num_requests, request_json, ...)` returns `(inst, predicted_output)`. The path then sets `num_predicted_output_tokens` and calls `query_instance`. It attaches no breakdown and no budget fields, and it performs no budget clamp. It is reachable only for `random|round_robin|shortest_queue|quality_greedy|cost_greedy|length_aware`, or for `cara` without a batch queue, which cannot happen because `init_app` always builds the queue for `cara` (L3621-3631).

### 1.6 Broadcast path, L478-543 (DEAD for scope: data collection)

- Model names are normalised with `to_ollama_tag`.
- One random instance is picked per listed model (global `random.shuffle`).
- Each gets a deep copy of the payload (`json.loads(json.dumps(...))`) and the queries are gathered with `return_exceptions=True`.
- One result is picked at random (`random.choice`) and the full list goes in `broadcast_results`.
- This path never sets `num_predicted_output_tokens`, so the client's oracle value reaches vLLM (see H10).

### 1.7 Error path, L638-655

Any exception on any path returns HTTP 500 with `{"success": False, "error", "error_traceback", "request_id", "instance_id", "host", "model"}`. That includes `InstanceStateUnavailable`, router errors, and a missing router/dispatcher. A vLLM non-200 is **not** an exception: `query_backend` returns `success: False` inside an HTTP 200 (vllm_instance.py:201-226).

### 1.8 Response fields and who produces them

| Field | Producer | Native | Pipeline | Client reads (cara_end_point_func.py) |
|---|---|---|---|---|
| `success` | vllm_instance.py:220 (return dict 215-226) | yes | yes | yes. `success==False` counts as a failure even on HTTP 200 (cara_end_point_func.py:118) |
| `generated_text`, `ttft`, `itl`, `output_tokens`, `model`, `server_latency` | vllm_instance.py:215-226 | yes | yes | yes. `output_tokens` can be `None` if usage has no `completion_tokens` |
| `serving_time`, `instance_id`, `host` | Instance.py:149-151 | yes | yes | `instance_id` and `host` are read |
| `scheduling_overhead_breakdown` | native [xref L1163-1164] / pipeline L587 | only if truthy | always | yes (`{}` default) |
| `budget_cost`, `actual_cost`, `affordable_output_tokens`, `budget_exhausted` | [xref L1153-1161] / L588-596 | if `budget_cost>0` | if `budget_cost>0` | yes, coerced with `float(... or 0.0)`, `int(... or 0)` and `bool` |
| `predicted_quality`, `predicted_length`, `p_under_budget`, `predicted_best_model`, `predicted_best_score`, `predicted_best_hit`, `all_model_scores` | [xref L1166-1187] | if estimates exist | **never** | yes (defaults 0.0, "", False, {}) |
| `broadcast_results` | L523 | no | no | yes (`[]`) |
| `error`, `error_traceback` | L646-654 | on 500 | on 500 | `error` is read |

The client derives `scheduling_overhead = client_latency - server_latency` (cara_end_point_func.py:142). `server_latency` spans only the backend POST and stream (vllm_instance.py:84, 213). So everything from the client's POST to the backend POST counts as scheduling overhead: batch wait, estimator, stats fetch, router and dispatcher.

### 1.9 What the benchmark client sends (cara_end_point_func.py:73-100)

```
{"request_id", "model": "cara", "prompt", "prompt_len", "temperature": 0.0,
 "repetition_penalty": 1.0, "frequency_penalty": 1.2, "max_tokens": output_len,
 "num_predicted_output_tokens": output_len,   # ORACLE length -- server MUST overwrite
 "stream": False,
 "request_specific_objective": {...}}          # only when RSO sampled / budget table
```

- Budget cells use `--rso-budget-table`, which sends `request_specific_objective = {"budget_cost": X, "constraint_mode": "TIERED"}` (benchmark_serving.py:805-812, 1491-1497).
- The server response must be a single JSON object. The server never streams.
- The client expects HTTP 200 with `success` truthy.

---

## 2. Module-level globals

| Name | Line | Written by | Read by | Scope |
|---|---|---|---|---|
| uvloop policy | 3-7 | import time (silently skipped if uvloop is missing) | the event loop | IN-SCOPE (execution parity, H20) |
| `STOP_WORD_MAPS` | 27 | – | nobody | DEAD |
| `_DECISION_LOG_PATH` | 62 | `__main__` L3866. Set to None by `_decision_log` on a write failure (L77) | `_decision_log` L68,72,73 | IN-SCOPE/OBS: arm-gate evidence. Written by native scoring [xref L1875] |
| `_DECISION_LOG_FH` | 63 | `_decision_log` L73 (opened append, line-buffered, never closed) | `_decision_log` | IN-SCOPE/OBS |
| `_QWEN_USER_RE`, `_GEMMA_USER_RE` | 80-81 | – | `_estimator_prompt` | IN-SCOPE |
| `TIMEOUT_KEEP_ALIVE=600` | 122 | – | `run_server` L3754 (uvicorn `timeout_keep_alive`) | IN-SCOPE (H14) |
| `app` | 123 | `build_app` sets `root_path` (L3461) | uvicorn | IN-SCOPE |
| `instances` | 124 | `init_app` L3487/3547 | completion, pipeline, batch loops, `run_server` L3744/3773 | IN-SCOPE. **Order matters** (H3) |
| `num_requests` | 125 | completion L415 | L416 (request_id default), L623 (legacy RR) | IN-SCOPE (request-id fallback only) |
| `start_time` | 126 | `init_app` L3549 | nobody | DEAD |
| `scheduling` | 127 | `init_app` L3550 | L546/597/615, `/v1/router` L731, GET `/v1/config` L751, POST L879, `init_app` L3558-3621 | IN-SCOPE |
| `served_requests` | 128 | completion L418 | nobody | DEAD (unbounded memory growth) |
| logging.basicConfig | 129-131 | import time: **append** to fixed relative path `experiment_output/logs/cara_serve.log` | – | IN-SCOPE/OBS (H18) |
| `logger` | 132 | – | everywhere | IN-SCOPE |
| `chat` | 133 | `init_app` L3471 | completion L430 | IN-SCOPE (v3 runs `--chat`) |
| `model_family` | 134 | `init_app` L3472 | nobody | DEAD |
| `repetition_penalty`, `frequency_penalty`, `temperature` | 135-137 | `init_app` L3473-3475 | completion L423-428 | IN-SCOPE (benign) |
| `broadcasting_enabled`, `broadcast_model_list` | 138-139 | `init_app` L3477-3478 | completion L478-541 | DEAD |
| `enable_predictor_feedback` | 140 | `init_app` L3479 | completion L441 (`run_server` uses `args.`) | DEAD (off) |
| `learned_predictors` | 142 | `_load_learned_predictors` L3373 only | [xref L1343], L2947 | DEAD. It is always `{}` in scope, because the scheduler-config path never populates it. The native fallback branch [xref L1349] therefore always sees `None` |
| `scoring_weights` | 144-146 | POST L779, `_load_scheduler_config` L3131, legacy L3295 (all `dict.update`, **in place**) | scoring [xref L1571-1574], GET/POST echo, passed **by reference** to `create_router`/`create_dispatcher` (L793, 813, 3171, 3183) | IN-SCOPE |
| `slo_defaults` | 147-155 | POST L781, loader L3133, legacy L3296 (`dict.update`, **merge**) | ~25 keys, see §4.4 | IN-SCOPE |
| `SchedulerConfigError` | 168 | – | loaders, `init_app` | IN-SCOPE |
| `_HOST_GPU_DEFAULT` | 180 | – | `_host_gpu_map` | IN-SCOPE |
| `instance_meta` | 199 | loaders L3260/3360, pricing injection L3605-3608 (mutates in place) | pipeline clamp L569, pipeline filter L2770, native [xref L1090,1129,1279,1288,1332,1615,1651,1697,…], TPOT tier check L3670 | IN-SCOPE |
| `model_estimator` | 201 | loaders L3208-3214 / L3305-3320 | `/v1/estimate`, GET `/v1/config`, POST (`create_router`), `_log_effective_config`, batch loops, fail-closed L3584 | IN-SCOPE |
| `batch_queue` | 203 | `init_app` L3627 (cara only) | completion L597/610, POST L837-844, `/v1/batch_stats`, loops | IN-SCOPE |
| `slo_filter` | 205 | POST L825, loader L3139/3142/3162 | native [xref L1602/1678], pipeline L2733/2755 | IN-SCOPE (H7) |
| `router` | 210 | POST L790, loader L3168/3179, estimator back-fill L3222-3223 | L546, `/v1/router`, GET/POST echo, L2702 | IN-SCOPE |
| `applied_router_config` | 216 | POST L796 (success only), loader L3173 | `/v1/router`, GET, POST echo and error body | IN-SCOPE/OBS |
| `dispatcher` | 217 | POST L811, loader L3181/3190 | L546, L2732, L2831, POST echo | IN-SCOPE |
| `scheduling_counters` | 220-234 | pipeline L2806-2815, L2836; native [xref L1645,1715-1832,2037-2051] | `/v1/scheduling_stats` | OBS. **Never reset**: cumulative since boot, so runners must snapshot deltas |
| `_HTTP_SESSION` | 906 | `_get_http_session` | stats fetch, sidecar | IN-SCOPE |
| `_TPOT_BATCH_MODEL` | [xref 1200] | `init_app` L3662 | [xref L1267-1303], L3636 | IN-SCOPE (cara) |
| `_SCORE_BATCHED_CALLS`, `_SCORE_PERREQ_CALLS`, `_TPOT_PREDICT_CALLS` | [xref 1202-1204] | batch loops / TPOT predict | GET `/v1/config` L757-759, `_log_effective_config` L3049-3051 | OBS. Gates read the deltas (`s2_q1.sh:163-165`, `s2_q1_burst.sh:91`, `tpot_zero_gate.py`) |
| `prepped_queue`, `_batch_id_counter` | [xref 2096-2097] | `init_app` L3708 | Option-A loops | DEAD (`CARA_PREPPER_WORKERS` unset) |

---

## 3. HTTP / CLI / env contracts

### 3.1 `POST /v1/config` (L764-886)

**Accepted keys**: `scoring_weights`, `slo_defaults`, `router`, `dispatch`, `filter` and `batch_config`, all optional. Unknown top-level keys are silently ignored.

**Execution order. It is not atomic, and the order matters:**

| # | Line | Key | Action | Validation |
|---|---|---|---|---|
| 1 | 778-779 | `scoring_weights` | `scoring_weights.update(v)` in place | **None**: no key whitelist, no type check (strings are accepted), no sum check |
| 2 | 780-781 | `slo_defaults` | `slo_defaults.update(v)` **merge**. JSON `null` stores `None`, which is how `static_tpot_ms_per_tok: null` resets the learned arm (`s2_q1.sh:92`) | **None**. `_validate_quality_norm` is **not** called on POST (H5). A malformed `static_tpot_ms_per_tok` is only detected per request at scoring [xref L1222-1235] and produces 500s |
| 3 | 787-804 | `router` | `create_router(data["router"], model_estimator=<live>, scoring_weights=<live ref>)`. On success: `router = …`, `applied_router_config = data["router"]` | On any exception: **HTTP 400** `{"status":"error","where":"router","error","router_still_live": applied_router_config}`. **Steps 1-2 are already applied** and are not rolled back |
| 4 | 806-820 | `dispatch` | `create_dispatcher(data["dispatch"], scoring_weights=<live ref>)` | On exception: 400 `where:"dispatch"`. The router from step 3 **stays swapped**. The error body has no `router_still_live` |
| 5 | 822-831 | `filter` | `slo_filter = create_filter(data["filter"])` | On exception: 400 `where:"filter"`. Steps 1-4 stay applied |
| 6 | 837-844 | `batch_config` (**top level**) | Applied only `if batch_queue is not None`: `max_batch_size=int()`, `batch_timeout_ms=float()`, `adaptive_sizing=bool()` | Silently ignored in pipeline mode. `bool("false") == True` (H13). `slo_defaults.batch_config` does **not** reach the live queue (H4) |
| 7 | 862-867 | – | Re-emit `EFFECTIVE_CONFIG` via `_log_effective_config(getattr(model_estimator,"type",None) or "default", {}, model_estimator)`. Exceptions are logged and swallowed | `ModelEstimator` has no `.type` attribute (model_estimator.py), so the re-emit always says `estimator_type:"default"` and `knn_dir: null` (H17) |

**What a POST resets:**
- A new router object replaces the old one's internal state.
- A new dispatcher object means **the RR counters and SQ/po2 RNG are reset**: `Po2QueueDispatch` and `ShortestQueueDispatch` get `random.Random(1)` when no seed is given (po2_queue.py:23, shortest_queue.py:20).
- A new filter object replaces the old one.

**What a POST never resets:** `scheduling_counters`, `batch_queue` totals, the `_SCORE_*`/`_TPOT_*` counters, `slo_defaults` keys that are not sent, `scoring_weights` keys that are not sent, and `instance_meta`.

**Success response (L869-886):**
```
{"status":"updated","scoring_weights":{…},"static_tpot_ms_per_tok":…,
 "slo_defaults":{non-dict values only},            # lists (relax_order) included; dicts excluded
 "pipeline":{"router":type-or-classname,"dispatch":…,"filter":…,"mode":scheduling},
 "router_config": applied_router_config, "router_sha": _router_sha(applied_router_config)}
```

**`_router_sha` (L707-720):**
```
sha256(json.dumps(cfg, sort_keys=True, separators=(",",":")).encode()).hexdigest()[:16]
```
It returns "" for None. `lib_post_config.sh:_router_sha_of` recomputes it client-side from the POSTed `router` sub-object and compares. The rebuild must produce **byte-identical** JSON serialisation: Python `json.dumps` defaults, `ensure_ascii=True`, and `0.0` kept as a float.

**Runner-side verification** (`lib_post_config.sh`):
- The POST must return HTTP 200.
- `router_sha` must equal the local digest.
- For payloads with `scoring_weights`, `GET /v1/config` is read back up to 3 times and every **sent** weight key must satisfy `|live - sent| ≤ 1e-9` (float comparison).
- `s2_q1.sh:137` also reads `GET /v1/config .static_tpot_ms_per_tok` and asserts the arm.

### 3.2 Exact key sets the v3 scripts POST

| System | Payload (verbatim shape) | Source |
|---|---|---|
| RB native, 8-tuple Pareto | `{"scoring_weights":{"w_quality":q,"w_latency":l,"w_cost":c,"w_balance":0},"slo_defaults":{"lpt_sort_key":"max","assignment_strategy":"scoring"},"filter":{"type":"none"}}`, q:l:c ∈ {0.1:0.1:0.8, 0.1:0.8:0.1, 0.2:0.2:0.6, 0.2:0.6:0.2, 0.33:0.33:0.33, 0.5:0.3:0.2, 0.6:0.2:0.2, 0.8:0.1:0.1} | `p1_pareto.sh:49,125` |
| RB Q1 learned TPOT arm | `{"scoring_weights":{"w_quality":0.5,"w_latency":0.3,"w_cost":0.2,"w_balance":0},"slo_defaults":{"lpt_sort_key":"max","assignment_strategy":"scoring","static_tpot_ms_per_tok":null},"filter":{"type":"none"}}` | `s2_q1.sh:89-92` |
| RB Q1 static TPOT arm | same, with `"static_tpot_ms_per_tok":{"3b":10.7,"7b":19.9,"14b":16.5,"72b":42.6}` (ms/token) | `s2_q1.sh:76,93` |
| RB ablations: no-LPT / no-term | `{"slo_defaults":{"lpt_sort_key":"none"}}`; `{"scoring_weights":{"w_quality":0.0}}` (and likewise `w_latency`, `w_cost`, `w_balance`), each after re-posting the full payload | `route-balance@HEAD reproduce/run_full_ablation.sh:48-49,134-159` |
| RB fixed batch / adaptive off | `{"batch_config":{"max_batch_size":N,"adaptive_sizing":false,"batch_timeout_ms":100}}` at **top level**. The v2 redo exists because the `slo_defaults.batch_config` form "did NOT register" | `v2_orchestration/v2_batching_redo.sh:12-37` (the older `v2_phase3_batching.sh`, `run_all_experiments.sh:88-95` and `batch_sweep.sh` use the **ineffective** `slo_defaults.batch_config` form) |
| BEST-Route | `{"router":{"type":"best_route_4way","kwargs":{"checkpoint_path":"models/cara/best_route_4way_qwen","confidence_threshold":T,"fallback_model":"Qwen/Qwen2.5-3B"}},"dispatch":{"type":D},"filter":{"type":"none"}}`, T ∈ {0.0,0.3,0.5,0.7}, D ∈ {round_robin, shortest_queue, po2_queue} | `payloads_v3.sh:19`, `p1_pareto.sh:51-52,139` |
| AvengersPro | `{"router":{"type":"avengers_pro","kwargs":{"artifact_dir":"models/cara/avengers_4model/avengers_pro_qwen_pw{P}"}},"dispatch":{"type":D},"filter":{"type":"none"}}`, P ∈ {0.25,0.39,0.53,0.7,0.8} (Gemma: `avengers_pro_gemma_pw0.7`) | `p1_pareto.sh:50,166`, `w1b_cells.sh:48` |
| Dispatcher-only (passthrough) | `{"router":{"type":"passthrough","kwargs":{}},"dispatch":{"type":D},"filter":{"type":"none"}}`, D ∈ {round_robin, shortest_queue, random, po2_queue} | `p1_pareto.sh:53,153`, `gemma_pt.sh:37-38` (no `kwargs`, so the digest differs) |
| BR fill rerun (conditional) | adds `"dispatch":{"type":D,"blind_policy":"counter"|"random"}` and `"slo_defaults":{"stats_blind_absorb":true}` | `a_cell.sh:35`, `br_fill_rerun.sh:47` |

Budget control is **not** a `/v1/config` key. It is carried per request in `request_specific_objective.budget_cost` (§1.9).

### 3.3 Other endpoints

| Endpoint | Lines | Contract | Scope |
|---|---|---|---|
| `GET /health` | 658-661 | `{"status":"ok"}`. It **does not** reflect router/dispatcher readiness in pipeline mode | IN-SCOPE (`wait_health`) |
| `GET /v1/router` | 723-732 | `{"router": classname|None, "router_config", "router_sha", "mode"}` | IN-SCOPE/OBS (`require_router`) |
| `GET /v1/config` | 735-761 | `{"scoring_weights", "static_tpot_ms_per_tok", "slo_defaults"(non-dict values), "scheduling", "router", "router_sha", "model_estimator": classname, "score_batched_calls", "tpot_predict_calls", "score_perreq_calls", "stats_blind_absorb"}` | IN-SCOPE/OBS |
| `POST /v1/estimate` | 664-704 | `{"prompt", "budget_tokens"=256}` returns `{model: {length_expected, p_under_budget, score, score_type, bucket_probs}}`. 503 if no estimator, 400 if the prompt is empty. It does **not** apply `_estimator_prompt`, and the estimator call is synchronous on the event loop | OBS (`preflight_gate.py:108`) |
| `GET /v1/batch_stats` | 889-897 | `BatchQueue.stats`, 404 if no queue | OBS (the v2 batching redo read `max_batch_size` from it) |
| `GET /v1/scheduling_stats` | 900-903 | `scheduling_counters` | OBS (`run_full_ablation.sh` snapshots it). No v3_scripts caller |

### 3.4 CLI args (L3797-3863), passed into `init_app` / `run_server`

| Arg | Default | v3 value | Used in scope? |
|---|---|---|---|
| `--host` | None | 0.0.0.0 | yes |
| `--port` | 8200 | 8200 | yes |
| `--workers` | 1 | 1 | must stay 1 (module state). uvicorn ignores >1 with an app object |
| `--ssl-keyfile/--ssl-certfile/--ssl-ca-certs` | None | – | no |
| `--ssl-cert-reqs` | `int(ssl.CERT_NONE)` | – | no |
| `--root-path` | None | – | no |
| `--model_config_path` | `"cara/config/cara/model_deployment.json"` (**stale path**) | `block/config/cara/s2/model_deployment.json` | yes. Must be passed (S15) |
| `--host_config` | None | `block/config/cara/s2/host_configs.json` | yes, for the V1 node_hosts form. A missing file is silently ignored (S13) |
| `--scheduling` | `"random"` | `cara` / `pipeline` | yes. **Silent default "random"** (S14) |
| `--debugging_logs` | off | off | no |
| `--chat` | off | **on** | yes |
| `--model-family` | "Qwen" | – | DEAD |
| `--repetition-penalty` | 1.0 | – | benign |
| `--frequency-penalty` | 1.2 | – | benign |
| `--temperature` | 0.0 | – | benign |
| `--enable-predictor-feedback` | off | off | DEAD |
| `--feedback-sample-rate` | 1.0 | – | DEAD |
| `--broadcasting` | off | off | DEAD |
| `--selected-broadcasted-models` | [] | – | DEAD |
| `--predictor-config` | None | passed | **ignored** because `--scheduler-config` is given (L3575-3579) |
| `--scheduler-config` | None | `scheduler_config_s2.json` | yes. Required for scope |
| `--decision-log` | None | per-session JSONL | yes/OBS. Missing only **warns** (L3868-3870) |

### 3.5 Environment variables read in these ranges

| Var | Line | Default | v3 | Effect |
|---|---|---|---|---|
| `CARA_HOST_GPU_MAP` | 187 | unset | unset | `prefix=gpu,…` overrides. Malformed parts are silently skipped |
| `CARA_INPROC_PREDICTOR` | 3061, 3636 [xref 1374] | "1" | "1" | cara mode loads `_TPOT_BATCH_MODEL` (fail-closed). "0" selects the HTTP sidecar path (DEAD) |
| `CARA_TPOT_DIR` | 3650 | `<tree>/models/cara/latency/deploy_tpot` | unset | TPOT booster directory |
| `CARA_MODELS_DIR` | 3419 | `<tree>/models/cara` | unset | predictor audit root |
| `EXPECTED_LENGTH_BUCKET_ENCODER`, `EXPECTED_QUALITY_JUDGE_ENCODER` | 3421-3422 | unset | unknown | audit assertion |
| `CARA_PREPPER_WORKERS` | 3704 | "0" | unset | ≥1 selects Option A (DEAD) |
| `CARA_BATCH_WORKERS` | 3705 | "1" | unset | >1 breaks causality (DEAD) |

Outside these ranges, for cross-reference: `CARA_DUMMY_SIDECAR`, `CARA_DUMMY_ESTIMATOR` (test only), and `CARA_STATIC_LAT`/`CARA_STATIC_TPOT_HEAD`. The last two have **no reader** in this tree [xref L1848-1855]. `w1b_cells.sh:20` exports `CARA_STATIC_TPOT_HEAD` for the Gemma static arm, so in this tree that export is a no-op. The static arm in this tree is configured **only** through `slo_defaults.static_tpot_ms_per_tok`.

---

## 4. Config loading

### 4.1 `init_app` order (L3465-3736). This order is load-bearing.

1. `build_app` (L3469) sets `root_path`.
2. Copy CLI args into globals (L3470-3479).
3. `model_dict = json.load(model_config_path)` (L3481). `host_config` is loaded only if the path is given **and exists** (L3484-3485).
4. Build the instances (L3486-3547). Iterate `model_dict.items()` in **JSON order**. For each model:
   - **V2** form: `"instances":[{hostname, ip_address, predictor_ports, backend_port}]`.
   - **V1** form: `node_hosts` + `host_config[hostname]`. `hostname = host.split("@")[-1]`. `hc` defaults to `{}` when host_config is None. Missing keys default to `ip_address:""`, `backend_port:8000`, `predictor_ports:[8300]`.
   - The backend port is removed from `predictor_ports` (L3517).
   - `instance_id = f"{model_key}_{idx}"`, e.g. `Qwen-2.5-3B_0`.
   - `vllm` gives a `VllmInstance(model_name=hf_model_name)`. `ollama` gives an `OllamaInstance` with the `to_ollama_tag` name (DEAD). Any other backend raises `ValueError`.
5. `start_time`, `scheduling` (L3549-3550). Validate against `_VALID_STRATEGIES` (L3554-3562).
6. Load the scheduler config (L3565-3579): a missing path raises `SchedulerConfigError`. Then `_load_scheduler_config`, else, for cara/length_aware with `--predictor-config` only, `_load_learned_predictors`.
7. **FAIL CLOSED** (L3584-3588): cara or length_aware with `model_estimator is None` raises `SchedulerConfigError`.
8. Pricing injection (L3594-3611): for each `model_dict[*].pricing = {input_per_1m, output_per_1m}`, override `instance_meta[*].cost_per_input_token/cost_per_output_token/cost_per_token (=out)/pricing`. It matches on `meta["model_name"] == hf_model_name`. The v3 `model_deployment.json` has **no** `pricing` block, so this is a no-op and prices come from `instance_metadata`.
9. `_log_and_assert_deployed_predictors()` (L3614-3618). Any exception is logged and re-raised.
10. cara only: `BatchQueue(max_batch_size=bc.get(16), batch_timeout_ms=bc.get(100.0), adaptive_sizing=bc.get(True))` with `bc = slo_defaults.get("batch_config", {})` (L3621-3631). Then the in-process TPOT load (L3635-3690, fail-closed). Then worker task creation (L3704-3724).
11. Log the instances and routes (L3727-3734).
12. `run_server` calls `set_ulimit()` **after** init (L3743; the print shows "None set limits file"), then `assert len(instances) > 0` (L3744), then `serve_http(..., timeout_keep_alive=600)`.
13. On shutdown, the predictor flush (L3764-3793) is skipped because feedback is off (DEAD).

### 4.2 `_load_scheduler_config(path, instances)` (L3115-3279). IN-SCOPE.

Fields read:

| Key | Line | Handling |
|---|---|---|
| top-level `quality_norm` | 3091 | **raise** (misplaced) |
| `slo_defaults.quality_norm` | 3098-3112 | validated: must be in `("minmax","raw","scaled")`; `raw` needs a non-empty `quality_norm_raw_ack` |
| `scoring_weights` | 3130-3131 | merged into the module defaults |
| `slo_defaults` | 3132-3133 | merged into the module defaults |
| `slo_defaults.filter` | 3136-3143 | present: `create_filter(cfg)`. Absent: `slo_filter = None`, which means the **native legacy inline filter** is used [xref L1678] |
| `scheduling_pipeline.{router,dispatch,filter}` | 3149-3200 | absent in v3. If present, the router and dispatch default to `{"type":"cara_native"}`. `NotImplementedError` sets it to None and logs (pipeline disabled). Other exceptions are wrapped as `SchedulerConfigError` |
| `model_estimator.type` | 3203-3219 | `"pfs"` gives PFS, `"knn"` gives KNN, **anything else** gives `DefaultModelEstimator` (v3: `"default"` + `quality_config.type="knn"`, `k=50`, `score_type="deepeval"`). Then `_log_effective_config(me_type, me_config, est)`. Back-fill the router's `_model_estimator` (L3222-3223) |
| `instance_metadata` | 3236-3262 | per instance: `prefix = hostname.split("-")[0].split(".")[0]`, `gpu = _host_gpu_map().get(prefix,"")`. `same_model` lists the tiers whose `model_name == inst._model_name`. Pick the tier matching `gpu_type == gpu`; otherwise the unique model-only match; otherwise **raise** if >1. **If 0 matches, the instance is silently unmapped** (S7). `instance_meta[iid] = {**meta, "instance_type": tier_key}` |

v3 tier keys (`scheduler_config_s2.json`): `qwen2.5-3b_a30`, `qwen2.5-7b_a30`, `qwen2.5-14b_v100`, `qwen2.5-72b_a100`. Each has `model_name`, `gpu_type`, `cost_per_input_token` and `cost_per_output_token`, and **no** `cost_per_token`. Host prefixes: `d7525` is a30, `c4130` is v100, `d8545` is a100.

### 4.3 `_load_learned_predictors` (L3282-3395). DEAD for scope.

Reached only without `--scheduler-config`. It differs from 4.2 in several ways:
- It builds `CaraBasePredictorConfig`.
- It never loads a filter or pipeline, so `slo_filter` keeps its module default of None.
- A ModelEstimator load failure is **swallowed** with a warning (L3325-3326).
- Per-instance `CaraLearnedPredictor` failures are swallowed (L3378-3381).
- Every non-`SchedulerConfigError` exception is **logged and swallowed** (L3394-3395). The FAIL CLOSED check at L3584 is the only backstop.

The rebuild can drop it, but must not re-create the "two loaders" divergence (see the D3 legacy note).

### 4.4 `slo_defaults` keys: every reader in the file

| Key | Default at read site | Readers | Scope |
|---|---|---|---|
| `ttft_slo_ms` | module 5000; read-site 5000 | L470 | carried; no filter in scope |
| `tpot_slo_ms` | 200 / 200 | L471 | carried |
| `budget_tokens` | 256 / 256 | L469, L3068 | carried (native legacy filter uses it) |
| `quality_min` | **module 0.3**; read-site 0.0 | L472 | carried |
| `budget_confidence_threshold` | 0.5 | [xref 1575] | native |
| `filter_enabled` | False | [xref 1366, 1713], L3064 | native (v3 False) |
| `constraint_mode` | "STRICT" | L274 (BatchRequest) | native (v3 config "NONE") |
| `relax_order` | `["ttft","tpot","quality","budget"]` | L278 | native |
| `lpt_sort_key` | "max" | [xref 2118, 2225, 2346], L3067 | native: `max|min|mean`, and `none|fifo` disables LPT |
| `assignment_strategy` | "scoring" | [xref 1581] | native |
| `batch_config` | {} | L3623 (**boot only**) | native |
| `instance_capacity` | {} (per-tier default 8) | L1091, [xref 1914] | native |
| `instance_config` | {} | [xref 1903] | native |
| `model_kv_config` | – | [xref balance] | native |
| `global_slo` | {} | [xref 1722] | native |
| `quality_norm` | "minmax" | [xref 1950], L3057 | native |
| `quality_gain` | 1.0 | [xref 1951] | native |
| `static_tpot_ms_per_tok` | None (learned arm) | [xref 1219, 1860], L748, L872, L3062 | native: arm switch |
| `stats_blind_absorb` | falsy | L760, L975, L2742 | conditional (§3.2 last row) |
| `bucket_size` | 64 | [xref 1700] | native legacy filter |
| `trace_sample_rate` | 0 | [xref 2540] | OBS |
| `scoring_debug_sample_every` | 0 | [xref 2050] | OBS |
| `filter` | – | L3136 (boot only) | selects the filter |

### 4.5 Fail-loud checks (each raise, with its line)

| Line | Condition | Type |
|---|---|---|
| 615-620 | pipeline traffic before router/dispatcher are set | RuntimeError (per request, 500) |
| 958-960 | `_raise_if_state_missing`: any exception or empty stats | InstanceStateUnavailable |
| 985-986, 993, 997 | `/instance_stats` non-200, transport/parse error, empty body (unless `stats_blind_absorb`) | InstanceStateUnavailable |
| 1014-1018 | `_fetch_all_instance_stats`: any failure or empty body. **Even under `stats_blind_absorb`**, because the masked `{}` counts as missing (H6) | InstanceStateUnavailable |
| 2635-2641 | unknown legacy strategy | RuntimeError |
| 2683-2684 | pipeline: no instances | ValueError |
| 2713-2717 | router chose a model that no instance serves | ValueError |
| 3091-3096 | top-level `quality_norm` | RuntimeError (wrapped as SchedulerConfigError by L3277) |
| 3102-3105 | unknown `quality_norm` | same |
| 3106-3112 | `raw` without `quality_norm_raw_ack` | same |
| 3250-3257 | ambiguous tier mapping | SchedulerConfigError (re-raised L3269-3270) |
| 3271-3279 | **any** other loader exception (bad JSON, estimator load, filter/router construction) | SchedulerConfigError, chained |
| 3350-3357 | ambiguous tier mapping (legacy) | SchedulerConfigError |
| 3427-3433 | audit: env var set but `fused_config.json` missing | RuntimeError |
| 3452-3456 | audit: encoder mismatch | RuntimeError |
| 3546 | unsupported backend | ValueError |
| 3558-3562 | invalid `--scheduling` | RuntimeError |
| 3571-3574 | `--scheduler-config` path does not exist | SchedulerConfigError |
| 3585-3588 | cara or length_aware without an estimator | SchedulerConfigError |
| 3654-3660 | TPOT dir missing (cara, inproc) | RuntimeError |
| 3664-3668 | TPOT load failure | RuntimeError |
| 3674-3679 | loaded TPOT tiers do not cover every mapped `instance_type` | RuntimeError |
| 3744 | zero instances | AssertionError (after init) |
| POST 797-831 | router/dispatch/filter construction failure | HTTP 400 (partial state, H1) |

### 4.6 SILENT defaults (a default applied with no error). Every item below is flagged.

- **S1** `scoring_weights` module defaults `{w_latency:.3, w_cost:.2, w_quality:.3, w_balance:.2}` (L144-146) apply to any key the config or a POST omits. For example, a payload without `w_balance` inherits whatever value was last set.
- **S2** `slo_defaults` module defaults (L147-155) are `ttft 5000`, `tpot 200`, `budget_tokens 256`, `quality_min 0.3`, `budget_confidence_threshold 0.5` and `filter_enabled False`. Note that `quality_min` is **0.3** at module level while every read-site literal says 0.0. v3 config sets 0.0 explicitly.
- **S3** Per-request RSO fallbacks (L469-473) are silent by design. `prompt_len` absent gives 0, which makes the clamp's input cost 0 (L438). `max_tokens` absent gives 256 (L439).
- **S4** Server sampling defaults (L423-428).
- **S5** No `slo_defaults.filter` gives `slo_filter=None` (L3142), and native then uses the legacy inline filter until the first POST that carries `filter` (H7).
- **S6** A host prefix missing from the GPU map gives `gpu=""`, and the unique model-only match is then taken silently (L3241-3248).
- **S7 (critical)** If no `instance_metadata` entry has the instance's `model_name`, the instance is **silently absent from `instance_meta`** (L3246-3262 has no else branch). Consequences:
  - The pipeline clamp is skipped (`c_out = 0`, L570-573), yet the budget fields are still reported.
  - The filter sees `gpu_type` as "unknown" (L2770-2772).
  - The TPOT tier-coverage check ignores the instance (L3670-3671 discards "").
  - The native TPOT batch skips it [xref L1290].
  - Cost falls to 0 in native scoring.
  - A model-name typo (for example Gemma `-pt` vs `-it`) produces a clean-looking, wrong cell.
- **S8** Malformed `CARA_HOST_GPU_MAP` parts (no "=") are skipped (L191-194).
- **S9** `model_estimator.type` values other than `"pfs"`/`"knn"`, including typos, give `DefaultModelEstimator` (L3212-3214).
- **S10** A `scheduling_pipeline` block present without router/dispatch gives `cara_native` (L3155-3156). `NotImplementedError` disables the pipeline with only an error log (L3174-3190), so it becomes loud at the first request.
- **S11** `batch_config` defaults of 16 / 100.0 / True (L3628-3630). With `adaptive_sizing=True`, `adapt_params` [xref L345-381] **overwrites** `max_batch_size` and `batch_timeout_ms` after every batch with hard-coded tiers (4/50, 16/100, 32/200). The configured `max_batch_size: 8` in `scheduler_config_s2.json` therefore governs **only the first batch**.
- **S12** A nonexistent `--host_config` path is silently ignored (L3484). V1 instances then get `ip_address=""`, `backend_port 8000`, `predictor_ports [8300]` (L3502-3509) and fail only at request time. A hostname **missing** from an existing host_config raises `KeyError` (loud).
- **S13** Pricing is absent from `model_deployment`, so costs come from `instance_metadata`. The cost lookup chain `cost_per_output_token → cost_per_token → 0.0` (L570-572) is silent.
- **S14** `--scheduling` defaults to `"random"`, a valid strategy. Forgetting the flag serves random routing with no error.
- **S15** The `--model_config_path` default points at a stale relative path.
- **S16** A missing `--decision-log` only warns (L3869). Arm gates fail later.
- **S17** `_decision_log` write failure disables the log with a warning (L75-77). Instrumentation never breaks serving.
- **S18** POST applies **no validation** to `scoring_weights` or `slo_defaults`. Unknown keys and typos (e.g. `"w_qual"`) are stored silently. `_validate_quality_norm` is not applied (H5).
- **S19** POSTed `slo_defaults.batch_config` is silently not applied to the live queue (H4). Top-level `batch_config` in pipeline mode is silently ignored (L837).
- **S20** A pipeline SLO filter exception is swallowed at DEBUG (L2817-2818), and the candidates stay unfiltered.
- **S21** Factory-level silent defaults (outside this file but reached from it):
  - `create_filter` without `type` gives `cara_tiered`.
  - `create_dispatcher` without `type` gives `cara_native`.
  - `blind_policy` defaults to `"counter"`.
  - `seed=None` makes SQ/po2 use `Random(1)`, while random dispatch uses the **global** `random` module.
- **S22** Missing `uvloop` silently falls back to the stock asyncio loop (L3-7).
- **S23** `_estimator_prompt`: when the regex misses, the full text is returned stripped (L115-119). Only the **first** user turn is extracted.
- **S24** `_log_effective_config` swallows its own exceptions (L3071-3072). A missing `EFFECTIVE_CONFIG` line is caught by the runner (`lib_serve_v3.sh:178-181`).
- **S25** `_log_and_assert_deployed_predictors`: a missing `fused_config.json` with no env var only warns (L3434-3435). An **unparsable** `fused_config.json` warns and continues **even when the env var is set** (L3438-3440), so the audit passes silently.
- **S26** Extra TPOT tiers not deployed only warn (L3680-3686).

---

## 5. Pipeline mode: `_select_via_pipeline` (L2657-2837)

Signature: `(all_instances, *, prompt_text, num_prompt_tokens, max_output_tokens, budget_tokens, ttft_slo_ms, tpot_slo_ms, quality_min, request_id) -> (Instance, predicted_output, breakdown)`.

### 5.1 Steps

1. The imports `RouterRequest` and `DispatchRequest` are **inside** the function (L2677-2678).
2. `t_total = time.time()` (wall clock, L2681). Raise if there are no instances (L2683-2684).
3. `pool = _unique_model_pool(all_instances)` (L2686). This keeps first-seen order over `instances`, which is **JSON order** of model_deployment. In v3 Qwen that is `[14B, 3B, 72B, 7B]`.
4. `rreq = RouterRequest(prompt=_estimator_prompt(prompt_text), num_prompt_tokens, max_output_tokens, budget_tokens, ttft_slo_ms, tpot_slo_ms, quality_min, request_id)` (L2691-2700). All routers see the **unwrapped** prompt (FAIRNESS, L2687-2690). There is **no `budget_cost`** field, so routers never see the budget.
5. `decision = await router.choose_model(rreq, pool)` gives `router_ms` (L2701-2703). BEST-Route (DeBERTa forward) and AvengersPro (`SentenceTransformer.encode` + `route_queries_batch`) run **synchronously on the event loop** inside an `async def` (best_route_4way.py:123-134, avengers_pro.py:239-262). See H2.
6. `chosen_model == "__ALL__"` (passthrough) gives `candidates = list(all_instances)`. Otherwise candidates are the instances with `i._model_name == chosen_model`, and an empty list raises ValueError (L2709-2717).
7. **`predicted_output = max_output_tokens`**, and `breakdown["estimator_ms"] = 0.0` (L2725-2726). No length estimator runs on this path; it was removed as a ~70-90 ms loop-blocking tax (L2719-2724).
8. Stats (L2732-2752):
   - `needs_stats = getattr(dispatcher,"requires_stats",True) or (slo_filter is not None and getattr(slo_filter,"requires_stats",True))`.
   - `requires_stats` per class: `RoundRobinDispatch` False; **`RandomDispatch` inherits True** (random_dispatch.py has no override, base.py:42); `ShortestQueueDispatch` True; `Po2QueueDispatch` True; `_NoFilter` False (filters/factory.py:75).
   - If `needs_stats`, the stats fetch covers **only `candidates`** (the chosen model's pool, or all instances for passthrough), per request:
     - default: `_fetch_all_instance_stats(candidates)`, fail-loud.
     - `slo_defaults.stats_blind_absorb` truthy: gather `_get_instance_stats` per candidate and **drop** exceptions and empty results, which leaves a partial `stats_map`. The dispatcher's `blind_policy` then handles the blind instances.
   - Otherwise `stats_map = {}`.
   - `stats_fetch_ms` is recorded either way.
9. Filter (L2755-2819), only if `slo_filter is not None`. With the v3 payloads this is `_NoFilter`, so it always runs:
   - Build `InstanceState(instance_id, model_name, gpu_type from instance_meta or "unknown", num_running, num_waiting, kv_cache_utilization)` with coercions `int(x or 0)` and `float(x or 0.0)`.
   - Build `SLOConstraints(ttft_ms, tpot_ms, budget_tokens, quality_min)`. There is **no `budget_cost`, `num_prompt_tokens` or `max_output_tokens`**, unlike native [xref L1604-1612].
   - `eligible = slo_filter.get_eligible(...)`. If it is non-empty, narrow the candidates, falling back to the original list if the narrowed list would be empty.
   - If anything was filtered, call `slo_filter.filter(...)` **again** and bucket the reasons into `filtered_ttft|tpot|quality|budget|other`.
   - Exceptions are swallowed at DEBUG (S20).
   - `filter_ms` is recorded **only in this branch**.
10. `dreq = DispatchRequest(prompt=prompt_text (WRAPPED), num_prompt_tokens, max_output_tokens, predicted_output_tokens=predicted_output, ttft_slo_ms, tpot_slo_ms, request_id)` (L2821-2829).
11. `inst = await dispatcher.choose_instance(candidates, dreq, stats_map=stats_map)` gives `dispatcher_ms` (L2830-2834).
12. `total_sched_ms` (L2835). `scheduling_counters["total_scheduled"] += 1` (L2836, success only).

### 5.2 Dispatcher semantics the rebuild must reproduce (from the dispatch modules, verbatim behaviour)

- **round_robin**: per-model counter keyed on `candidates[0]._model_name`: `idx = counter % len(candidates)`. Under passthrough the key is the first instance's model, which is effectively one global rotation over `instances` order.
- **shortest_queue** (`counter`): argmin of `num_running + num_waiting` over `candidates` in order. **Ties keep the first candidate**. An instance with missing stats uses `inst.total_request`, the local count of dispatched requests incremented in `Instance.query_instance` (Instance.py:141).
- **po2_queue** (`counter`): `pool = rng.sample(candidates, 2)` with `rng = Random(seed or 1)` recreated per POST. Pick the smaller queue, and on a tie keep the first sampled.
- **random**: `random.choice(candidates)` from the global RNG (unseeded) unless `seed` is given.

### 5.3 Overhead breakdown keys (pipeline)

`router_ms`, `estimator_ms` (always 0.0), `stats_fetch_ms` (0.0 when skipped), `filter_ms` (present only when `slo_filter` is not None, which with `filter:none` means always), `dispatcher_ms` and `total_sched_ms`. All are wall-clock ms from `time.time()`. The budget clamp and the backend call are **excluded**. The native breakdown keys differ [xref `_score_and_pick_instance`/batch loop], and aggregators must not assume the same key set.

### 5.4 Budget clamp parity (pipeline L563-596 vs native [xref L1129-1161])

Common formula:

```
cpt   = meta.get("cost_per_token", 0.0)
c_in  = meta.get("cost_per_input_token",  cpt)
c_out = meta.get("cost_per_output_token", cpt)
if budget_cost > 0 and c_out > 0:
    remaining = budget_cost - num_prompt_tokens * c_in
    affordable = 1 (and exhausted=True) if remaining <= 0 else max(1, int(remaining / c_out))   # floor
    request_json["max_tokens"] = min(request_json.get("max_tokens", max_output_tokens), affordable)
after the response:
    actual_out = int(resp.get("output_tokens", 0) or 0)
    actual_cost = input_cost + actual_out * c_out
    budget_exhausted = exhausted or actual_cost >= budget_cost - 1e-9
```

| Aspect | Pipeline | Native |
|---|---|---|
| `budget_cost` source | L473: rso first, then top-level, then `or 0.0` | L259: top-level first, then rso, with no `or 0.0` |
| `input_cost` in `actual_cost` when `c_out == 0` | 0 (`pl_input_cost` is set only inside `if pl_c_out > 0`) | `num_prompt_tokens * c_in` |
| `affordable_output_tokens` when `c_out == 0` | 0 (`or 0`) | `None`, serialised as JSON null. The client coerces it to 0 |
| Clamp mutates | `request_json["max_tokens"]` | `req.request_json["max_tokens"]` (same dict object) |
| `num_predicted_output_tokens` sent to vLLM | **pre-clamp `max_output_tokens`** | `req.predicted_tokens` (estimator max/min/mean length, float, rounded in vllm_instance.py:110) |
| Budget awareness in selection | **none**: router and dispatcher never see `budget_cost`. Budget control for baselines is a post-selection clamp only | cost term / filter [xref] |

For the v3 scope every tier has `c_out > 0` and the client never sends top-level `budget_cost`, so the first three rows are **observationally identical**. The rebuild should use **one shared clamp function** (see §8), and it must not change which instance is chosen.

The reported `actual_cost` includes **input** tokens, while RB's scoring cost is **decode-only** (`_objective_cost`, L84-104). These are two different quantities. Keep both exactly as they are, and see H11.

---

## 6. Executed vs dead (these ranges)

| Item | Lines | Status | Reason |
|---|---|---|---|
| uvloop try-import | 3-7 | IN-SCOPE | loop implementation (H20) |
| `STOP_WORD_MAPS` | 27-29 | DEAD | never read |
| `_decision_log` | 66-77 | IN-SCOPE/OBS | native scoring evidence [xref L1875] |
| `_objective_cost` | 84-104 | IN-SCOPE | native scoring [xref L1467]. Decode-only, **no switch** |
| `_estimator_prompt` + regexes | 80-81, 107-120 | IN-SCOPE | native estimator [xref L2133,2152,2358,2390] and pipeline router L2692 |
| `TIMEOUT_KEEP_ALIVE`, `app`, globals | 122-234 | mixed | see §2 |
| `SchedulerConfigError` | 168-177 | IN-SCOPE | |
| `_HOST_GPU_DEFAULT`, `_host_gpu_map` | 180-195 | IN-SCOPE | |
| `to_ollama_tag` | 398-406 | DEAD | ollama backend and broadcast only |
| completion: prologue | 411-474 | IN-SCOPE | step 8 (predictor feedback, L441-465) is DEAD |
| completion: broadcast branch | 478-543 | DEAD | data collection |
| completion: pipeline branch | 546-596 | IN-SCOPE | baselines |
| completion: cara batch branch | 597-610 | IN-SCOPE | RB |
| completion: pipeline-not-ready guard | 615-620 | IN-SCOPE | fail-loud |
| completion: legacy per-request | 621-635 | DEAD | never launched |
| completion: error handler | 638-655 | IN-SCOPE | |
| `/health` | 658-661 | IN-SCOPE | |
| `/v1/estimate` | 664-704 | OBS | preflight probe |
| `_router_sha`, `/v1/router` | 707-732 | IN-SCOPE/OBS | digest contract |
| GET `/v1/config` | 735-761 | IN-SCOPE/OBS | runner readback |
| POST `/v1/config` | 764-886 | IN-SCOPE | all keys used (`batch_config` for the fixed-batch ablation) |
| `/v1/batch_stats` | 889-897 | OBS | |
| `/v1/scheduling_stats` | 900-903 | OBS | |
| `_get_http_session` | 906-925 | IN-SCOPE | stats fetch |
| `InstanceStateUnavailable` | 928-944 | IN-SCOPE | |
| `_raise_if_state_missing` | 947-960 | DEAD | only the legacy strategies call it (L2850, 2888, 2915, 2944) |
| `_get_instance_stats` | 963-998 | IN-SCOPE | the `_mask` branch is conditional (`stats_blind_absorb`) |
| `_fetch_all_instance_stats` | 1001-1019 | IN-SCOPE | native batch loop + pipeline |
| `_call_sidecar_predict_latency` | 1022-1070 | DEAD | only reached when `CARA_INPROC_PREDICTOR=0` [xref L1408]. v3 runs with =1 |
| `_select_instance` | 2584-2641 | DEAD | legacy per-request |
| `_unique_model_pool` | 2644-2654 | IN-SCOPE | pipeline pool **order** |
| `_select_via_pipeline` | 2657-2837 | IN-SCOPE | the `stats_blind_absorb` branch L2742-2746 is conditional |
| `_schedule_shortest_queue` | 2840-2857 | DEAD | legacy (the pipeline SQ is `dispatch/shortest_queue.py`) |
| `_model_size_key` | 2860-2867 | DEAD | legacy |
| `_schedule_quality_greedy`, `_schedule_cost_greedy` | 2870-2923 | DEAD | legacy |
| `_schedule_length_aware` | 2926-2974 | DEAD | legacy |
| `_schedule_cara` | 2977-3022 | DEAD | the cara batch queue always exists in cara mode |
| `_VALID_QUALITY_NORM`, `_validate_quality_norm` | 3025, 3075-3112 | IN-SCOPE | boot validation |
| `_log_effective_config` | 3028-3072 | IN-SCOPE/OBS | gates (`assert_effective_config.py`, last line wins) |
| `_load_scheduler_config` | 3115-3279 | IN-SCOPE | the `scheduling_pipeline` block branch (L3149-3200) is DEAD in v3 (the config has none), but it is cheap to keep |
| `_load_learned_predictors` | 3282-3395 | DEAD | `--scheduler-config` always wins |
| `_log_and_assert_deployed_predictors` | 3399-3456 | IN-SCOPE/OBS | startup audit |
| `build_app` | 3459-3462 | IN-SCOPE | |
| `init_app`: V1 node_hosts path | 3497-3509 | IN-SCOPE | Qwen v3 deployment |
| `init_app`: V2 instances path | 3494-3495 | IN-SCOPE | Gemma 12B/27B on A100 (`gemma_deploy_v3.sh:6-7`) |
| `init_app`: ollama | 3519-3532 | DEAD | |
| `init_app`: pricing injection | 3594-3611 | IN-SCOPE but no-op in v3 | keep: a future `pricing` block would override |
| `init_app`: batch queue + TPOT load | 3621-3690 | IN-SCOPE (cara) | |
| `init_app`: Option-A preppers | 3707-3716 | DEAD | env unset |
| `init_app`: legacy worker loop | 3717-3724 | IN-SCOPE | `n_workers=1` |
| `run_server` serve | 3739-3763 | IN-SCOPE | |
| `run_server` feedback flush | 3764-3793 | DEAD | feedback off |
| `__main__` argparse + decision-log wiring | 3796-3873 | IN-SCOPE | |

---

## 7. Bug-fix markers in these ranges. All of them must survive the rebuild.

| Line | Marker | Meaning |
|---|---|---|
| 31-46 | KNN train/serve skew fix (2026-07-31) | Always unwrap the chat template before the quality estimator. There is **no switch**. The wrapped prompt inverted 3B/14B quality ordering (accuracy 29.7% against a 25% chance floor). |
| 47-61 | DECISION LOG, "Always on" | Per request, per candidate, log the latency that entered the score. There is no env gate, because opt-in evidence produced three inert static arms. |
| 84-104 | `_objective_cost` (2026-07-31) | The cost objective is **decode-only**, `expected_length * cost_out`, as in the paper's Table 2. There is no switch; the old env var let the cost model change between cells. |
| 107-111 | `_estimator_prompt` "No switch" | Unwrap is unconditional. It is also enforced at `knn_estimator.unwrap_chat_prompt`. |
| 122 | keep-alive "Was 5" | `timeout_keep_alive=600`. A value of 5 caused a localhost RST race against the benchmark's 60 s keepalive. |
| 157-167 | **v3 FIX (D26)** | One host-prefix→GPU map. The P100 (`c240g5`) entry was removed. Other copies (`deploy_cara.py`, …) still exist and must be kept in sync by hand. Override with `CARA_HOST_GPU_MAP`. |
| 171-176 | **v3 FIX (D31)** | `SchedulerConfigError` is re-raised explicitly, so fail-loud checks are never swallowed by the loaders' `except Exception`. |
| 184-186 | D3 / zombie P100 note | P100 is deliberately absent from the map. Re-add it only with a real P100 host. |
| 211-216 | **v3 FIX** (`applied_router_config`) | Echo the live router config, so a 400'd POST cannot leave a mislabelled cell served by the previous router. |
| 560-562 | Budget clamp parity | The pipeline clamp is identical to native, so baseline budget cells are comparable to RB budget cells. |
| 612-620 | fail-loud pipeline guard | No silent random fallback when the router or dispatcher is missing (data_integrity rule #6). |
| 741-745 | GET `/v1/config` (2026-08-01) | Read-only view of resolved state. A leaked static prior was previously undetectable. |
| 800-802 | `router_still_live` | A 400 body names the router that is still live. |
| 833-836 | T4 batching ablation | Hot-swap `batch_config` via POST. It must be the **top-level** key. |
| 846-861 | EFFECTIVE_CONFIG re-emit (2026-08-01) | Re-log the resolved config after **every** POST. Merge semantics are deliberately kept (an author decision). The re-emit makes a leaked `static_tpot_ms_per_tok` visible. |
| 906-913 | pooled session | One `TCPConnector` (4096 / 256 per host / keepalive 600) instead of a session per call. |
| 928-944 | **AUTHOR DECISION 2026-08-02** | A stats failure **raises** and never becomes `{}`. `{}` had meant `slot_open=True`, which turned the queue-conditioned learned arm into an idle-state estimate. |
| 947-953 | `_raise_if_state_missing` | No substituted rule on partial telemetry. |
| 968-974 | **ERA-MODE MASKING (author-directed 2026-08-04)** | `stats_blind_absorb` reverts `_get_instance_stats` to return `{}`. Default OFF. **Its comment's claim "reverts ALL fetch sites" is false** (H6). |
| 1008-1011 | "DO NOT MASK" | `return_exceptions=True` is kept only for socket hygiene. Every exception is re-raised. |
| 2635-2641 | unknown strategy fail-loud | No random fallback. |
| 2687-2690 | FAIRNESS | Baseline routers get the same unwrapped prompt as RB's KNN. |
| 2719-2725 | pipeline estimator removal | No sync RoBERTa forward on the pipeline path (it was a 70-90 ms loop-blocking cap). `predicted_output = max_tokens`. |
| 2728-2731 | stats skip | Skip `/instance_stats` when the dispatcher and filter do not need it (it was an 87-255 ms unfair tax). |
| 2737-2741 | `stats_blind_absorb` (author-ordered 2026-08-03) | Pipeline-only A/B apparatus. Default OFF, so fail-loud governs production cells. |
| 3036-3041 | EFFECTIVE_CONFIG rationale | Assert the resolved state, not the declared state: static TPOT was a dead key, chat skew went unnoticed, the P100 zombie survived. |
| 3059-3060 | `unwrap_prompt:true`, `cost_decode_only:true` | Unconditional, and reported as such. |
| 3076-3090 | `_validate_quality_norm` | Fail loud on a misplaced key, an unknown value, or `raw` without an ack. A whole cell set once ran on `raw` by accident. |
| 3220-3223 | estimator back-fill | Routers built before estimator load get the handle afterwards. |
| 3226-3235 | **v3 FIX (D3/D4)** | Tier mapping matches (model, GPU). A model-only match is allowed only when unambiguous, and ambiguity **raises**. The v1 first-match gave A30 3B the P100 tier, and fixing it moved the λ6 mix to 74.8% 3B. |
| 3269-3279 | **D31** + "Was: log and continue" | Loader failures become `SchedulerConfigError` (R6). |
| 3329-3334 | **v3 FIX (D3, legacy path)** | Same matching in the legacy loader. |
| 3382-3385 | NOTE: no `break` | A `break` would exit the per-instance loop after the first instance. |
| 3392-3393 | D31 (legacy) | Re-raise `SchedulerConfigError`. |
| 3412-3417 | **F2 (2026-07-31)** | The models dir is derived from this module's own tree, not a hard-coded `/users/asdwb/Block`. The audit must inspect the running tree. |
| 3427-3433 | audit must not pass without its artifact | Raise if the env var is set and the config is missing. |
| 3552-3562 | strategy validation | Fail loud at startup. |
| 3567-3574 | (2026-07-31) missing `--scheduler-config` path | Raise instead of falling through to a healthy server with no estimator. |
| 3581-3588 | **FAIL CLOSED (author-directed 2026-08-01)** | cara without an estimator refuses to start. |
| 3590-3593 | pricing canonical source | `model_deployment.pricing` overrides `instance_metadata` costs. |
| 3637-3644 | **FAIL-CLOSED (2026-08-01)** in-process TPOT | A missing or failed TPOT load raises; no silent legacy path while `inproc_predictor:true` is reported. The path is derived from the running tree. |
| 3669-3686 | tier coverage / **D4** zombie warning | Missing tier heads raise. Extra heads only warn. |
| 3858-3870 | `--decision-log` REQUIRED evidence | Warn if absent. |

Referenced but outside these ranges: **D13** (static prior tier matcher raises on no match, longest key first [xref L1239-1253]) and **D30** (AvengersPro raises instead of falling back to the largest model, avengers_pro.py:245-262). The BEST-Route `fallback_model` is mandatory for T>0 (best_route_4way.py:38-48, `payloads_v3.sh:5-13`).

---

## 8. Proposed interfaces for the rebuild (no module globals)

```python
# ---------- config schema + loader ----------
from dataclasses import dataclass, field
from typing import Literal, Optional, Mapping

LptKey = Literal["max", "min", "mean", "none", "fifo"]
QualityNorm = Literal["minmax", "raw", "scaled"]

@dataclass(frozen=True)
class TierMeta:                     # one instance_metadata entry
    tier: str                       # e.g. "qwen2.5-3b_a30"
    model_name: str                 # HF name, must equal Instance.model_name
    gpu_type: str
    cost_per_input_token: float
    cost_per_output_token: float
    pricing: Optional[Mapping[str, float]] = None

@dataclass
class ScoringWeights:
    w_latency: float; w_cost: float; w_quality: float; w_balance: float
    def merge(self, patch: Mapping[str, float]) -> "ScoringWeights": ...
    # merge() keeps v3 semantics (unsent keys persist) but REJECTS unknown keys / non-numbers.

@dataclass
class BatchConfig:
    max_batch_size: int; batch_timeout_ms: float; adaptive_sizing: bool
    # bool parsing: accept JSON bool only (reject "false" strings)

@dataclass
class SloDefaults:                  # every key in §4.4, typed; unknown keys rejected
    ttft_slo_ms: float; tpot_slo_ms: float; budget_tokens: int; quality_min: float
    budget_confidence_threshold: float; constraint_mode: str; relax_order: list[str]
    lpt_sort_key: LptKey; assignment_strategy: str; filter_enabled: bool
    quality_norm: QualityNorm = "minmax"; quality_norm_raw_ack: str = ""
    quality_gain: float = 1.0
    static_tpot_ms_per_tok: Optional[dict[str, float]] = None     # None == learned arm
    instance_capacity: dict[str, int] = field(default_factory=dict)
    model_kv_config: dict = field(default_factory=dict)
    instance_config: dict = field(default_factory=dict)
    global_slo: dict = field(default_factory=dict)
    stats_blind_absorb: bool = False
    batch_config: BatchConfig = ...
    trace_sample_rate: float = 0; scoring_debug_sample_every: int = 0; bucket_size: int = 64
    def merge(self, patch: Mapping) -> "SloDefaults": ...   # dict-merge semantics, validated
    def validate(self) -> None: ...                           # _validate_quality_norm + static map shape

@dataclass(frozen=True)
class EstimatorConfig: type: str; score_type: str; quality_config: Mapping; raw: Mapping

@dataclass(frozen=True)
class SchedulerConfig:
    scoring_weights: ScoringWeights
    slo_defaults: SloDefaults
    model_estimator: EstimatorConfig
    tiers: Mapping[str, TierMeta]
    filter: Optional[Mapping] = None            # slo_defaults.filter (boot)
    pipeline: Optional[Mapping] = None          # scheduling_pipeline block

class SchedulerConfigError(RuntimeError): ...

def load_scheduler_config(path: str) -> SchedulerConfig: ...        # strict; raises SchedulerConfigError
def host_gpu_map(env: Mapping[str, str]) -> dict[str, str]: ...     # D26; reject malformed parts
def resolve_instance_meta(instances, tiers, gpu_map) -> dict[str, "InstanceMeta"]:
    """D3/D4 rule verbatim; ambiguity raises. PROPOSAL (needs owner OK, R3): an
    instance with ZERO matching tier also raises (closes S7). For the v3 fleet every
    instance matches, so this changes no banked cell."""

@dataclass(frozen=True)
class InstanceMeta:
    instance_id: str; model_name: str; gpu_type: str; instance_type: str
    cost_in: float; cost_out: float

def load_deployment(model_config_path: str, host_config_path: Optional[str]) -> list["InstanceSpec"]:
    """Preserve JSON key order x node_hosts order (H3). Missing host_config path
    when node_hosts form is used -> raise (closes S12; needs owner OK)."""

# ---------- application state ----------
@dataclass
class ObservedCounters:            # exposed on GET /v1/config, EFFECTIVE_CONFIG
    score_batched_calls: int = 0; score_perreq_calls: int = 0; tpot_predict_calls: int = 0

@dataclass
class AppState:
    mode: Literal["cara", "pipeline"]
    instances: list["Instance"]                  # ordered
    instance_meta: dict[str, InstanceMeta]
    weights: ScoringWeights
    slo: SloDefaults
    estimator: Optional["ModelEstimator"]
    slo_filter: Optional["SLOFilter"]            # None == native legacy inline filter (H7)
    router: Optional["RouterBase"] = None
    router_cfg: Optional[dict] = None            # applied_router_config
    dispatcher: Optional["DispatchBase"] = None
    batch_queue: Optional["BatchQueue"] = None
    tpot_model: Optional[object] = None
    counters: dict = field(default_factory=dict)      # scheduling_counters (cumulative)
    observed: ObservedCounters = field(default_factory=ObservedCounters)
    decision_log: "DecisionLog" = ...
    stats: "InstanceStatsClient" = ...
    sampling_defaults: dict = field(default_factory=lambda: {"repetition_penalty": 1.0,
                                     "frequency_penalty": 1.2, "temperature": 0.0})
    chat: bool = True
    request_seq: int = 0

async def build_state(args: "ServeArgs", env: Mapping[str, str]) -> AppState: ...
    # performs §4.1 steps 3-10 in the SAME order; all fail-loud checks of §4.5.

# ---------- HTTP API layer ----------
def create_app(state: AppState) -> "FastAPI": ...
    # routes: POST /v1/completions, GET /health, POST /v1/estimate, GET /v1/router,
    #         GET|POST /v1/config, GET /v1/batch_stats, GET /v1/scheduling_stats
    # handlers read request.app.state.rb; no module globals.

@dataclass
class ConfigUpdate:                 # parsed POST body; every field optional
    scoring_weights: Optional[dict] = None; slo_defaults: Optional[dict] = None
    router: Optional[dict] = None; dispatch: Optional[dict] = None
    filter: Optional[dict] = None; batch_config: Optional[dict] = None

@dataclass
class ConfigUpdateResult:
    ok: bool; where: Optional[str]; error: Optional[str]; body: dict; http_status: int

def apply_config_update(state: AppState, upd: ConfigUpdate) -> ConfigUpdateResult:
    """v3 order: weights -> slo -> router -> dispatch -> filter -> batch_config -> re-emit.
    PROPOSAL (owner decision): build router/dispatcher/filter FIRST, then commit all
    or nothing. Observationally equivalent for every v3 run, because post_config aborts
    the cell on any non-200. Merge semantics of weights/slo MUST be preserved."""

def router_sha(cfg: Optional[dict]) -> str: ...   # byte-identical to L718-720

def effective_config(state: AppState, me_type: str, me_cfg: Mapping) -> dict: ...
    # same keys as L3045-3069; logged as "EFFECTIVE_CONFIG <json sort_keys>"

# ---------- instance stats ----------
class InstanceStateUnavailable(RuntimeError): ...

class InstanceStatsClient:
    def __init__(self, *, per_call_timeout_s: float = 3.0, session_timeout_s: float = 30.0,
                 limit: int = 4096, limit_per_host: int = 256, keepalive_s: float = 600): ...
    async def fetch_one(self, inst, *, absorb: bool) -> dict:
        """GET http://{ip}:{backend_port}/instance_stats. Non-200 / exception / empty
        body -> raise InstanceStateUnavailable, unless absorb -> return {}."""
    async def fetch_all(self, insts, *, absorb: bool = False) -> dict[str, dict]:
        """gather(return_exceptions=True); absorb=False -> raise on ANY missing/empty
        (v3 L1012-1018); absorb=True -> drop missing (v3 pipeline L2743-2746)."""
    async def close(self) -> None: ...

# ---------- pipeline host ----------
@dataclass(frozen=True)
class CompletionContext:            # what completion() extracts (§1.1)
    request_id: str; prompt_text: str; num_prompt_tokens: int; max_output_tokens: int
    budget_tokens: int; ttft_slo_ms: float; tpot_slo_ms: float; quality_min: float
    budget_cost: float; rso: Mapping

def parse_completion(req_json: dict, slo: SloDefaults, seq: int) -> CompletionContext: ...
    # ONE budget_cost precedence for both modes (rso first, then top-level, then 0.0; None->0.0)

@dataclass
class PipelineDecision:
    instance: "Instance"; predicted_output: int; breakdown: dict[str, float]

async def select_via_pipeline(state: AppState, ctx: CompletionContext) -> PipelineDecision: ...
    # §5.1 verbatim, including: unwrapped prompt to router, wrapped prompt to dispatcher,
    # predicted_output = max_output_tokens, requires_stats semantics (random == True),
    # stats over candidates only, filter_ms only when a filter object exists.

@dataclass(frozen=True)
class BudgetClamp:
    max_tokens: int; affordable_out: Optional[int]; exhausted: bool
    input_cost: float; c_in: float; c_out: float

def budget_clamp(meta: Optional[InstanceMeta], budget_cost: float,
                 num_prompt_tokens: int, requested_max: int) -> Optional[BudgetClamp]: ...
def budget_fields(clamp: BudgetClamp, budget_cost: float, output_tokens) -> dict: ...
    # SHARED by native _dispatch_and_resolve and pipeline -> parity by construction (§5.4).
```

---

## 9. Hazards: things a porter can easily get wrong

**H1. `POST /v1/config` is non-atomic.** Weights and SLO defaults are merged **before** router construction. A 400 from the router, dispatch or filter leaves earlier steps applied, and a dispatch or filter 400 leaves the new router live. Runners abort on non-200, so no banked cell depended on this. The rebuild may make it atomic, but this is an owner decision (R3); it must not happen silently.

**H2. Execution asymmetry between RB and the baselines (FIRST-RULE item).** These are code-level differences, not config differences:
- (a) RB's estimator runs in `run_in_executor` [xref L2360-2362]. The BEST-Route and AvengersPro router models run synchronously on the event loop (§5.1 step 5) and block all other in-flight handlers while they run.
- (b) The RB native path fetches stats once per batch for all instances. Pipeline SQ/po2/random fetch per request over their candidates.
- (c) `random` dispatch fetches stats that it never reads, because `requires_stats` is inherited as True. This adds overhead and exposes it to fail-loud telemetry blips.
- (d) All scheduler-side models run on CPU (`CUDA_VISIBLE_DEVICES=""`).

To reproduce v3 numbers, keep all four exactly as they are. Changing any of them changes what the baselines execute, needs owner approval, and requires the affected control arms to be re-run. List them in the campaign's execution-difference table.

**H3. Instance ordering is semantics.** `instances` follows model_deployment JSON key order × `node_hosts` order. It drives:
- `_unique_model_pool` order, so the router pool is `[14B, 3B, 72B, 7B]` for v3 Qwen. v1 BEST-Route's `model_pool[0]` fallback was 14B for this reason.
- The SQ tie-break (first candidate wins).
- The passthrough RR rotation.
- The po2 `rng.sample` population order.

A rebuild that sorts instances, or builds them from a set or dict with different ordering, changes routing.

**H4. `batch_config` lives at top level in POST.** `{"slo_defaults":{"batch_config":…}}` merges into `slo_defaults` but is read **only at boot** (L3623). The live queue is untouched (the v2 redo exists for this reason). `adaptive_sizing=True` also overwrites the size and timeout after every batch using hard-coded tiers [xref L373-381], so a "fixed batch size" ablation requires `adaptive_sizing:false`, and the configured `max_batch_size:8` governs only the first batch (S11).

**H5. POST skips all validation.** `_validate_quality_norm` runs only at boot. A POSTed `quality_norm:"raw"` without an ack, or an unknown value, is accepted. A malformed `static_tpot_ms_per_tok` fails later, per request, as 500s [xref L1222-1235]. String weights pass the POST and fail in scoring.

**H6. Masking under `stats_blind_absorb`.** When set, `_get_instance_stats` returns `{}`. But `_fetch_all_instance_stats` (L1012-1013) and `_raise_if_state_missing` (L955) treat `{}` as missing and **still raise**. Only the pipeline absorb branch (L2742-2746) actually tolerates a missing instance, and the native batch loop still fails loud. The comment at L971-973 ("reverts ALL fetch sites") is inaccurate. Port the behaviour, not the comment, and only if the BR fill-rerun cells are in scope.

**H7. The native filter branch depends on POST history.** At boot with `scheduler_config_s2.json`, `slo_filter=None`, so the native legacy inline filter is used [xref L1678-1743]. After any POST containing `"filter":{"type":"none"}` (every v3 RB payload), `slo_filter=_NoFilter` and the pluggable branch runs [xref L1602-1676]. Both accept everything when `filter_enabled=False`, but they are different code. The rebuild must model this state explicitly, and the scoring-spec owner must confirm the two branches are equivalent for scope (e.g. any use of `p_budget` downstream).

**H8. `slo_defaults` and `scoring_weights` merges persist across cells.** An arm key that is not re-sent leaks from the previous cell. This is the measured 2026-08-01 static-prior leak: 3534/3534 identical routes. v3 payloads send `static_tpot_ms_per_tok: null` for the learned arm, and JSON `null` must be stored as None, not dropped. `p1_pareto.sh` RB payloads do **not** send it and rely on a fresh `restart_v3 cara`. Keep the merge semantics (an author decision, L858-860), and keep the GET readback plus the EFFECTIVE_CONFIG re-emit.

**H9. Double chat templating.** Per L32-35, the benchmark client sends the chat-templated prompt (vLLM `CustomDataset` applies the tokenizer template unless `--skip-chat-template`). With `--chat`, `vllm_instance` then posts it as a `user` message to `/v1/chat/completions`, so vLLM templates it a second time. The KNN and the routers see the unwrapped text (`_estimator_prompt`), while the backend sees double-wrapped text. This is identical across all arms. Reproduce it byte-for-byte, or change it for every arm at once with the owner's approval (verify the client-side templating with the actual vLLM version before relying on this).

**H10. `num_predicted_output_tokens` must be overwritten.** The client sends the **oracle** output length (cara_end_point_func.py:94). Every scheduling path overwrites it (pipeline L559, native [xref L1121]). Only broadcast leaks it. vLLM uses it as `predicted_decode_tokens` (vllm_instance.py:109-110, `or max_tokens`), which feeds `/instance_stats.pending_decode_tokens` and therefore the RB TPOT features. In pipeline mode the value written is the **pre-clamp** `max_tokens`, even when the clamp lowered `max_tokens`. Keep this as is.

**H11. Two different "cost" quantities.** The scoring cost is decode-only (`_objective_cost`). The budget clamp and the reported `actual_cost` include input tokens × `c_in`. Aggregators must know which one they read. Do not unify them silently.

**H12. `budget_cost` precedence and `None` handling differ.** `completion()` (L473) reads RSO first with an `or 0.0` guard. `BatchRequest` (L259) reads top-level first with no guard, so `float(None)` raises `TypeError`. They agree for the v3 client, which sends only `request_specific_objective`. Unify them in `parse_completion` and document the change.

**H13. Coercions.** `bool(bc["adaptive_sizing"])` turns the string `"false"` into True. `int(bc["max_batch_size"])` truncates floats. Stats values are coerced with `int(x or 0)` / `float(x or 0.0)` in the pipeline filter, but shortest_queue and po2 use raw `s.get("num_running",0) + s.get("num_waiting",0)`, which does no coercion and would raise on `None`. `int(remaining / c_out)` floors, and `max(1, …)` means a clamp never produces 0 tokens. vLLM receives `int(round(predicted))`.

**H14. Timeouts and keepalive values:**
- uvicorn `timeout_keep_alive=600` (5 caused RSTs).
- Pooled stats session: connector limit 4096, 256 per host, `keepalive_timeout=600`, session `total=30`, per stats call `total=3`, sidecar `total=10`.
- **The backend call uses a new `ClientSession` per request** (vllm_instance.py:152) with a 30-minute timeout and no connection reuse.
- `lib_post_config.sh` uses `--max-time 15` on the POST and 10 on readbacks.

Changing the backend session pooling changes latency for every arm, so treat it as an execution change (H2 rules).

**H15. Stats key spelling.** The pipeline and dispatchers read `num_running`, `num_waiting` and `kv_cache_utilization`. Native paths also read `max_num_seqs`, `pending_decode_tokens`, `pending_prefill_tokens`, `kv_free_blocks`, `ema_prefill_tok_per_s`, `ema_decode_tok_per_s` and `decode_ctx_p95`, plus the XGBoost `INSTANCE_STATE_FEATURES`. The endpoint is `GET http://{ip}:{backend_port}/instance_stats` on the **vLLM backend port** (8000), not the predictor port. An empty dict is treated as **missing**.

**H16. Router digest serialisation.** It must match Python `json.dumps(sort_keys=True, separators=(",",":"))` exactly, including float repr (`0.0` vs `0`) and `ensure_ascii`. `gemma_pt.sh` omits `"kwargs":{}` while `p1_pareto.sh` sends it, so the same router gets two digests. That is fine, because the digest is compared against the payload each runner sent.

**H17. EFFECTIVE_CONFIG after a POST is degraded.** The re-emit passes `me_config={}` and a missing `.type` attribute, so it reports `estimator_type:"default"` and `knn_dir:null`. `assert_effective_config.py` takes the **last** line. An assertion on `knn_dir`, or on a non-default estimator type, made after the first POST would fail spuriously. The rebuild should retain the boot `me_type` and `me_cfg` in `AppState` so the re-emit is faithful. This changes only the audit line, never routing.

**H18. Import-time logging side effect.** `logging.basicConfig(filename='experiment_output/logs/cara_serve.log', filemode='a+')` at import (L129-131) opens a CWD-relative path in append mode. If the directory is missing, the import fails. Sessions accumulate unless the runner rotates the file (`lib_serve_v3.sh:132-135` does). EFFECTIVE_CONFIG, PREDICTOR_AUDIT and scoring logs go to **this** file, not stdout, while `[INPROC-XGB]` tier lines go to stdout (`print`, L3687-3690). `runtime_gates.py tiers` reads the **stdout** log.

**H19. Exceptions that are swallowed:**
- the pipeline filter (DEBUG, L2817)
- `_decision_log` (self-disables, L75-77)
- `_log_effective_config` (L3071)
- the POST re-emit (L866)
- the broadcast gather
- the legacy loader (L3394-3395)
- the predictor audit parse failure (L3438-3440)

Everything else propagates to a 500 (per request) or aborts startup.

**H20. Event-loop details:**
- uvloop is used if importable. Pin it or record it, since the loop implementation is part of what executes.
- `BatchRequest` uses the deprecated `asyncio.get_event_loop().create_future()`.
- Tasks are created in `init_app` before uvicorn starts. `set_ulimit` runs **after** init (L3743).
- `/v1/estimate` and the router construction in POST (DeBERTa / SentenceTransformer loads) are synchronous on the loop. That is acceptable only because they happen before traffic.

**H21. `/health` returns ok before the pipeline is configured.** Traffic sent after `/health` but before POST fails loudly per request (L615-620). Warmup must come after `post_config`.

**H22. Counters are cumulative since boot.** This covers `scheduling_counters`, the `_SCORE_*`/`_TPOT_*` counters and `batch_queue` totals. Warmup requests increment them. Gates compute deltas around a cell (`s2_q1.sh:163-165`), and the rebuild must not reset them on POST unless the gates change too.

**H23. Every scheduler-side instance of `num_prompt_tokens` is client-supplied (`prompt_len`).** The server never tokenizes. It feeds cost, the clamp, and the XGBoost features. A rebuild that tokenizes server-side changes inputs for every arm.

**H24. Static TPOT through an env var is dead in this tree.** `CARA_STATIC_TPOT_HEAD` and `CARA_STATIC_LAT` have no reader [xref L1848-1855]. The Gemma W1-B runner (`w1b_cells.sh:20`) exports `CARA_STATIC_TPOT_HEAD`. In this tree that export is inert, so any Gemma static cell must set `slo_defaults.static_tpot_ms_per_tok` with keys that match `instance_type` by longest substring (e.g. `1b, 4b, 12b, 27b`; `12b` must beat `2b`). Check which tree produced any banked Gemma static cell before porting its numbers.
