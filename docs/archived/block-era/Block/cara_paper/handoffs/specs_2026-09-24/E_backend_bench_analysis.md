# PORT SPEC E — backend client, load generator / benchmark client, result analysis + gates

Source of truth: repo `/home/wd312/Code/llm/Block`, branch `v3-campaign` (HEAD 7af63c7 family; "cara" = RouteBalance).
All paths below are relative to that repo unless absolute. `BS` = `block/benchmark/cara/benchmark_serving.py`,
`EP` = `block/benchmark/cara/cara_end_point_func.py`, `VI` = `block/global_scheduler/cara/cara_instance/vllm_instance.py`,
`IN` = `.../cara_instance/Instance.py`, `CS` = `block/global_scheduler/cara/cara_serve.py`,
`V3S` = `cara_paper/handoffs/window_setup_work/v3_scripts/`, `CA2` = `cara_paper/handoffs/window_setup_work/cell_audit_s2.sh`,
`VFORK` = vLLM fork `AKafakA/vllm@2894a98722733c76cdb549c8b5e5068594f12bef` (read locally via
`git -C /home/wd312/Code/llm/vllm-cara show 2894a98:<path>`), `CELLS` = `/home/wd312/Code/llm/rb-exp/v3-exp/cells/`.

Verification legend: **[V]** = I re-derived it from data or by running a read-only script in this session;
**[C]** = read from code; **[D]** = stated in docs/handoffs only.

---

## 1. Backend client (scheduler → vLLM)

### 1.1 Classes and what actually runs
- `IN` (172 lines) abstract `Instance`; `VI` (226 lines) `VllmInstance`; `ollama_instance.py` (227 lines) `OllamaInstance` (dead, §6).
- Instances are built in `CS:3511-3547` from `model_deployment.json` (`backend: "vllm"` for all four Qwen tiers) +
  `host_configs.json` (`ip_address`, `backend_port` = **8000**, `predictor_ports` = [8300]). `instance_id = f"{model}_{idx}"`
  e.g. `Qwen-2.5-14B_0`. Constructor defaults used: `query_predictor_timeout=10`, `query_backend_timeout=30*60` (`VI:62-63`).
  NOTE: `model_deployment.json` `vllm_params.port = 7000` is never read; `deploy_cara.py` launches vLLM on its default 8000
  (health probe hardcodes `localhost:8000`, `block/exp/cara/deploy_cara.py:200-203`). `backend_port` from host_configs is authoritative.

### 1.2 Request send path (per request)
`CS` `/v1/completions` handler (`CS:411`) → router/dispatcher (pipeline, `CS:546-600`) or batch queue (cara mode,
`CS:597-610` → `_dispatch_and_resolve` `CS:1114+`) → `Instance.query_instance(request_json, predicted_num_decode_tokens)`
(`IN:137-164`) → `VllmInstance.query_backend(payload, headers={})` (`VI:81-226`).

Scheduler-side mutations of the client JSON before it reaches `VI`:
- `CS:423-428`: sampling defaults only if absent (bench always sends them: temperature 0.0, repetition_penalty 1.0, frequency_penalty 1.2).
- `CS:432-434`: `--chat` (always passed by `lib_serve_v3.sh:162`) sets `request_json["use_chat_endpoint"]=True`.
- `CS:559` (pipeline) / `CS:1119-1121` (cara batch): `num_predicted_output_tokens` is **overwritten** with the scheduler's
  predicted output length (`predicted_tokens or max_output_tokens`). Budget clamp of `max_tokens` exists (`CS:568-583`,
  `CS:1153+`) but is inert in v3 (no RSO ⇒ `budget_cost=0`).

`VI.query_backend` exactly:
- Endpoint: `http://{ip}:{backend_port}/v1/chat/completions` when `use_chat_endpoint` (always in v3), else
  `/v1/completions` (`VI:78-79, 96-97`).
- `_pdt = payload.get("num_predicted_output_tokens") or payload.get("max_tokens")`;
  `predicted_decode = int(round(_pdt))` (`VI:109-110`) — falsy-`or` on an int: a prediction of 0 falls back to `max_tokens`.
- Chat payload (`VI:113-129`): `{"model": hf_model_name, "messages":[{"role":"user","content": payload["prompt"]}],
  "temperature", "repetition_penalty", "frequency_penalty", "max_completion_tokens": payload["max_tokens"],
  "predicted_decode_tokens": predicted_decode, "stream": True, "stream_options":{"include_usage":True},
  "request_id": str(request_id)}`. Completions variant (`VI:131-146`) uses `"prompt"`, `"max_tokens"`, `"logprobs": None`.
  `predicted_decode_tokens` is a **fork-only** field (feeds vLLM `/instance_stats` `pending_decode_tokens`).
- Headers (`VI:148-151`; `query_instance` passes `{}` which is falsy ⇒ always these): `Authorization: Bearer {OPENAI_API_KEY}`
  (literally `Bearer None` when unset) and `X-Request-Id`.
- Transport: a **new `aiohttp.ClientSession` per request** (`VI:152`) with `ClientTimeout(total=1800)`; `ssl=False`.
  No connection pooling, **no retries** anywhere in this path.
- Streaming parse: `response.content.iter_any()` (`VI:157`), each network read `lstrip()`ed (`VI:159`), SSE messages split on
  `\n\n` by `StreamedResponseHandler` (`VI:10-52`, incremental UTF-8 decoder `VI:14-17`, trailing-buffer JSON fallback `VI:38-50`);
  `:`-prefixed ping lines skipped; `[DONE]` ignored.

### 1.3 Timing (backend-client clock, `time.perf_counter`, on the scheduler host)
- `st` = first statement of `query_backend` (`VI:84`) — **before** payload build and before the per-request session/TCP connect.
- `ttft = perf_counter() - st` at the first SSE message whose `choices` is non-empty (`VI:187-191`). Under the chat endpoint the
  first such message is vLLM's role-only delta, emitted together with the first token. **[V]** `itl[0]` median = 0.015 ms vs
  17 ms for later gaps (`CELLS/s2_p1_l30/pt_sq_l30_n3534.json`); `len(itl) == output_len` for 3512/3534 requests.
- `itl.append(ts - most_recent_ts)` for every later message with non-empty `choices` (`VI:192-196`); timestamps are per SSE
  message, so messages from one network read get ~equal timestamps.
- `output_tokens = usage["completion_tokens"]` from the final usage chunk (`VI:198-199`).
- `server_latency = perf_counter() - st` after the session closes (`VI:213`).
- `IN.query_instance` additionally measures `serving_time` with `time.time()` (`IN:142-149`) — returned but unused downstream.
- Return dict (`VI:215-226`): `generated_text, ttft, itl, output_tokens, success, error, model, server_latency, instance_id, host`.
  `CS` adds `scheduling_overhead_breakdown` (`CS:587`, `CS:1164`) and, in cara mode, `predicted_quality`, `predicted_length`,
  `predicted_best_model`, `predicted_best_hit`, `all_model_scores` (`CS:1171-1185`).

### 1.4 Error handling
- HTTP 200 but no chunk with choices ⇒ `success=False`, error "Never received a valid chunk…" (`VI:200-207`).
- Non-200 ⇒ `success=False`, `error=response.reason` (`VI:208-210`).
- Exceptions (connect refused, timeout, JSON decode) are **not caught in `VI`**; they propagate to the `CS` handler which returns
  HTTP 500 `{"success":False,"error","error_traceback","request_id","instance_id","host","model"}` (`CS:637-655`).
- Ollama client catches everything itself (`ollama_instance.py:179-212`) — dead.

### 1.5 `/instance_stats` (fork endpoint; fetched by the scheduler, not by `VI`)
- `_get_instance_stats` `CS:963-998`: `GET http://{ip}:{backend_port}/instance_stats`, shared session, `ClientTimeout(total=3)`.
  Fail-loud: non-200, exception, or empty body ⇒ `raise InstanceStateUnavailable` (`CS:928`) — unless
  `slo_defaults.stats_blind_absorb` is set (ERA-MODE MASKING, `CS:970-977`), then returns `{}`.
- `_fetch_all_instance_stats` `CS:1001-1018`: `gather(return_exceptions=True)` then re-raises if any instance missing ("DO NOT MASK").
- Keys (format documented at `block/predictor/cara/data_structures.py:169-200`): `num_running, num_waiting, num_active_decode_seqs,
  decode_ctx_p50, decode_ctx_p95, decode_ctx_max, pending_prefill_tokens, pending_decode_tokens, kv_cache_utilization,
  kv_free_blocks, token_budget_per_iter, prefill_chunk_size, max_num_seqs, num_preempted`. Consumers in `CS`/`dispatch/` read
  mostly `num_running`, `num_waiting`, `kv_cache_utilization`, `kv_free_blocks`, `max_num_seqs`, `pending_*_tokens`.
- Who fetches: cara-mode batch scheduler (`CS:2235`, `CS:2442`) and pipeline dispatchers that need state (sq/po2; `CS:2728-2748`);
  rr/random do not. This asymmetry is why only BR/AP/PT **sq/po2** cells at λ≥18 died (§4.6).

---

## 2. Benchmark client / load generator

Invocation (grid, `V3S/lib_serve_v3.sh:219-231`), run from `$V3TREE` on the scheduler head node:
```
python3 -u -m block.benchmark.cara.benchmark_serving --backend cara --base-url http://127.0.0.1:8200 \
  --endpoint /v1/completions --model Qwen/Qwen2.5-3B --tokenizer Qwen/Qwen2.5-3B \
  --dataset-name custom --dataset-path data/cara/best-route-v3-test-3534-eval.jsonl \
  --num-prompts 3534 --request-rate <λ> --seed 1 --max-output-len 1024 --metadata cap=1024 \
  --result-dir <dir> --result-filename <cell>.json --save-result --save-detailed  > <cell>.bench.log 2>&1
```
Burst (`V3S/s2_q1_burst.sh:94-108`) adds `--seed <s> --burstiness <b>` and
`--metadata arm=<a> pattern=<p> seed=<s> burst=<b> cap=1024 q1burst=1 preset=<532|uniform>`. Q1 steady
(`V3S/s2_q1.sh:105-118`) adds `--seed <s>` and `--metadata cap=1024 arm=<a> seed=<s> q1=1`.

### 2.1 Seeding and RNG
- `BS:1648-1649`: `random.seed(args.seed); np.random.seed(args.seed)` (global legacy MT19937 streams).
- Grid cells always `--seed 1` (hardcoded `lib_serve_v3.sh:226`). Burst/Q1 seeds 1..6.
- The seed affects **only arrivals** in practice (numpy stream). The python stream is consumed only by `_sample_rso` (off,
  `--rso-rate 0.0`) and the LoRA picker (off) — enforced by `verify_pairing.py --check-scope` (`V3S/verify_pairing.py:177-258`).
  The dataset shuffle re-seeds python `random` to a **fixed 0** (§2.3), so it does not depend on `--seed`.

### 2.2 Arrival process (exact) — `get_request` `BS:169-266`
```
for i in range(N):                       # ramp-up unused
    if rate == inf: d_i = 0
    elif burstiness == inf: d_i = 1/rate
    else: d_i = np.random.gamma(shape=burstiness, scale=1.0/(rate*burstiness))   # BS:235-239
cum: D_i = sum_{j<=i} d_j                # BS:242-243
normalise: D_i *= (N/rate) / D_{N-1}     # BS:244-257  (last arrival lands exactly at N/rate)
pacing: start_ts = time.time(); for each i: sleep(start_ts + D_i - time.time()) if >0; yield   # BS:259-266
```
- Poisson = `burstiness 1.0` (exponential gaps). Bursty = `burstiness 0.25` ⇒ gamma shape 0.25, **CV² = 1/0.25 = 4** (pattern `gam025`).
- First request is sent at `D_0 > 0`, not at t=0. Pacing uses wall clock `time.time()`; durations use `perf_counter`.
- Open loop: one `asyncio.create_task` per arrival, no concurrency cap (`--max-concurrency` unset ⇒ `nullcontext`,
  `TCPConnector(limit=0)` `BS:612-621`).
- **[V] Golden reproduction:** a standalone generator (legacy `np.random.seed(s)`, list of `np.random.gamma` draws, cumsum,
  normalise) reproduces all 73 lines of `CELLS/s2_q1burst_PROD/ARRIVAL_DIGESTS.txt` **only if** the digest is taken over
  `x' = (1_000_000.0 + D_i) - 1_000_000.0` (the frozen-clock arithmetic in `verify_pairing.py:134-152`), formatted
  `f"{x':.9f}|"`, sha256, first 16 hex (`verify_pairing.py:155-159`). Digesting raw `D_i` differs at ~1e-10 and fails every line.
  Verified with numpy 2.4.3 locally; head ran numpy 2.2.6 (legacy RandomState stream is frozen across versions).

### 2.3 Dataset loading and the N=3534 set
- `get_samples` (`dataset.py:97-115`): the path does not end in `lmsys`, so it delegates to vLLM `get_samples`; custom branch
  `VFORK vllm/benchmarks/datasets.py:1649-1660` builds `CustomDataset(dataset_path, disable_shuffle=args.disable_shuffle)` —
  **no `random_seed` passed ⇒ `DEFAULT_SEED = 0`** (`datasets.py:93, 116`).
- `CustomDataset.load_data` (`datasets.py:1947-1979`): `pd.read_json(lines=True)` → list of row dicts; requires a `prompt`
  column; `random.seed(0); random.shuffle(self.data)`.
- `CustomDataset.sample` (`datasets.py:1981-2031`): takes the first `num_prompts` rows; applies
  `tokenizer.apply_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True, tokenize=False)` (Qwen2.5-3B
  template ⇒ `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`);
  `prompt_len = len(tokenizer(templated).input_ids)`; `expected_output_len = custom_output_len` (= 1024 via `--max-output-len`,
  `BS:1635-1636`); `request_id = request_id_prefix + str(i)`. **No length filter** (no `is_valid_sequence`), no oversampling.
- File `data/cara/best-route-v3-test-3534-eval.jsonl`: 3534 rows, keys `id, prompt, source`, md5 `15ed0e67ffa50e0d17955127c144a566`
  (matches cell provenance). N=3534 = the whole file. All 3534 prompts exist in the 3634-row scored lookup **[V]**.
- **[V] Send order** = `random.seed(0); random.shuffle(rows)` of the JSONL, identical for grid (seed 1) and burst (seed 2) cells;
  `request_id` suffix == index (checked on `pt_sq_l30` and `qb_learned_gam025_s2_l24_puniform`).
- `--request-id-prefix` default `bench-{uuid4().hex[:8]}-` (`BS:1450-1455`) — random per run, not seeded; never join on request_id.

### 2.4 Request payload bench → scheduler (`EP:73-97`, POST `{base_url}/v1/completions`, `stream: False`)
`{"request_id", "model":"cara" (ignored), "prompt": <templated>, "prompt_len", "temperature":0.0, "repetition_penalty":1.0,
"frequency_penalty":1.2, "max_tokens": 1024, "num_predicted_output_tokens": 1024, "stream": False}` (+ `request_specific_objective`
only if RSO, dead). Headers `Content-Type: application/json`, `x-request-id` (`EP:14-21`). The comment at `EP:87-93` calls
`output_len` an "oracle predicted output length" — for the custom dataset it is the **cap (1024)**; the scheduler overwrites it (§1.2).
Consequence: the backend receives a **double-templated** prompt (client applied the Qwen template; `--chat` then wraps it again as a
user message). The quality estimator and all aggregators unwrap (`CS:107-120`, `dominance.py:63-76`); generation does not.

### 2.5 Client-side E2E and per-request record
- Session: one shared `aiohttp.ClientSession(connector, trust_env=True, timeout=ClientTimeout(total=1800))` (`BS:626-630`). No retries.
- Ready check before timing: `wait_for_endpoint` sends `input_requests[0]` through the scheduler until success (≤600 s) (`BS:662-678`).
  `--num-warmups 0`. The runner also sends 5 curl warmups (`lib_serve_v3.sh:210-216`). **[V]** these 6 requests appear in the
  session decision log (7080 lines for 2 RB cells = 2×(3534+6)).
- `st = perf_counter()` just before `session.post` (`EP:111`); on `status==200 and response.success`: `latency = perf_counter()-st`
  after `await response.json()` (`EP:118-124`). `scheduling_overhead = latency - server_latency` (`EP:139-142`).
  Failure paths leave `latency = 0.0`, fill `error` (server error string, `HTTP <code>: <reason>`, or full traceback `EP:177-180`).
- `benchmark_duration = perf_counter()` from just before the arrival loop (`BS:762`) to after `gather` (`BS:844`).
- Per-request dict (`BS:929-961`), keys: `request_id, prompt (templated), input_len (=prompt_len), output_len (=actual_output_lens[i]),
  response, ttft, itl, e2el (client latency s), scheduling_overhead (s), model, host, instance_id, error, broadcast_results,
  scheduling_overhead_breakdown, predicted_quality, predicted_length, predicted_best_model, predicted_best_hit, all_model_scores,
  budget_cost, actual_cost, budget_exhausted, affordable_output_tokens`. `actual_output_lens[i]` = `output_tokens` from vLLM usage, else
  tokenizer count of `generated_text` (`BS:348-361`), 0 for failures (`BS:374-375`).
- `scheduling_overhead_breakdown` keys differ by mode **[V]**: cara batch path `{batch_wait_ms, estimator_ms (÷batch), stats_fetch_ms
  (÷batch), sidecar_ms, xgb_predict_ms, xgb_batched_tpot_ms, predict_round_ms, scoring_ms, total_scheduling_ms, batch_id, batch_size}`
  (`CS:2303-2315`, monotonic clock, arrival→assignment); pipeline `{router_ms, estimator_ms, stats_fetch_ms, filter_ms, dispatcher_ms,
  total_sched_ms}` (`CS:2703-2835`, wall clock).

### 2.6 Per-cell output
- `<dir>/<cell>.json` — single JSON object (`BS:1830-1920`): `date` (`%Y%m%d-%H%M%S`, local time of the head), `endpoint_type`,
  `backend`, `label`(None), `model_id`, `tokenizer_id`, `num_prompts`, **metadata merged at top level as strings** (`BS:1843-1852`:
  `cap:"1024"`, burst: `arm, pattern, seed, burst, q1burst, preset`), `request_rate` (float or `"inf"`), `burstiness` (float),
  `max_concurrency`, then `duration, completed, failed, total_input_tokens, total_output_tokens, request_throughput,
  request_goodput(None), output_throughput, total_token_throughput, max_output_tokens_per_s, max_concurrent_requests`,
  `{mean,median,std,p50,p99}_{ttft,tpot,itl,e2el}_ms`, `{mean,median,std,p50,p99}_scheduling_overhead_ms`,
  `p50/p99_prompt_len`, `p50/p99_output_len`, `mean/median/std_prompt_len`, `mean/median/std_output_len`,
  `prompt_response_correlation`, `response_details` (list, §2.5). `source` injection (`BS:1719-1731,1869-1874`) never fires
  (keys `prefix+item["id"]` vs `prefix+index` never match).
- `<cell>.bench.log` (stdout/stderr); `failed_requests_<ts>.log` written into the bench CWD (`$V3TREE`) when any request failed (`BS:418-472`).
- `.pytorch.json` sidecar only if env `SAVE_TO_PYTORCH_BENCHMARK_FORMAT` is set (`VFORK vllm/benchmarks/lib/utils.py:20`) — never in v3.

---

## 3. Metric definitions (exactly as computed)

### 3.1 Bench-side (per cell; successful requests only) — `calculate_metrics` `BS:316-571`
- `request_throughput = completed / duration`; `output_throughput = Σ output_len / duration`; `total_token_throughput = (Σ input_len + Σ output_len)/duration`.
- E2E: `mean_e2el_ms = np.mean(e2el)*1000`, `p99_e2el_ms = np.percentile(e2el, 99)*1000` (numpy default linear interpolation);
  median/std likewise; percentiles from `--metric-percentiles 50,99` (`BS:1429-1437`).
- TTFT: stats of `ttft` (**server-side**, §1.3 — excludes scheduler queueing/batch wait).
- TPOT: `(e2el_client - ttft_server)/(output_len-1)` over `output_len>1` (`BS:363-367`) — **mixed clocks**, includes scheduling overhead.
- ITL: flattened `itl` lists (includes the ~0 ms role-chunk gap per request).
- Scheduling overhead (`BS:1022-1039`, only for `--backend cara`): stats of `scheduling_overhead` over successes →
  `mean/median/std/p50/p99_scheduling_overhead_ms`. This per-request `e2el − server_latency` is what the paper calls the
  **off-instance residual** (handoffs `2026-07-23_neurips_rebuttal_audit.md:401-406`): client→scheduler network, FastAPI parse,
  batch wait, scoring/prediction, dispatch, per-request session creation is NOT in it (inside server_latency), response JSON back.
  The decision-compute split comes from `scheduling_overhead_breakdown` (e.g. `total_scheduling_ms`, `estimator_ms`, `xgb_batched_tpot_ms`).
- `max_output_tokens_per_s`, `max_concurrent_requests` need `start_time`, which is not persisted — not recomputable from the cell.
- **[V]** For `wq0.5_wl0.3_wc0.2_lambda30` and `pt_sq_l30` every summary field above (mean/p99 e2el/ttft/tpot/itl, throughputs,
  token totals, scheduling-overhead mean/p99, p99_output_len) recomputes exactly (≤1e-9 rel) from `response_details` + `duration`.

### 3.2 Analysis-side — `V3S/dominance.py` `summarize()` `:157-263` (the basis of record)
- Rows used: `ok = [r for r in response_details if not r.get("error")]` (`:160`); RULE-0 post already required `failed==0`.
- **Quality** (`:79-96, 165-204, 222`): lookup `SCORED_DEFAULT=/home/wd312/Code/llm/Block/data/cara/scored/test_scored_filtered.jsonl`
  (3634 rows, md5 `0318916065c1c7713cd20c0002574d0e`), key `DEEPEVAL_KEY="deepeval-llama3.1-8b-it_reference"` inside
  `models[<name>].llm_judge_scores`; table keyed by `prompt.strip()`; per-model entries stored under both `name` and `"Qwen/"+name`
  (scored names are already `Qwen/Qwen2.5-*`, so the served model matches `name`). Response prompt → `unwrap_chat` (regex
  `<\|im_start\|>user\n(.*?)(?:\n?<\|im_end\|>|\Z)`, DOTALL, `.strip()`). Score lookup zero-safe: `s = pm.get(m); if s is None: s = pm.get(m.replace("Qwen/",""))`.
  Three outcomes: prompt absent ⇒ `unmatched` (**fails the cell** unless `--allow-partial-coverage`); prompt present, model unscored ⇒
  `corpus_gap` (NOTE only); else `matched`, value appended (0.0 counted, `n_zero` incremented).
  `q = mean(matched values)` (NaN if none); `coverage = matched / n`, `n = len(ok)`.
  **[V]** the scored file has 4 corpus-gap pairs, all in the eval set: (Perl-rewrite×14B), ("wetback"×7B), (neighbour×72B),
  (Enhance-snippet×14B) — `2026-07-31_GATES.md:1040-1078` says "exactly one", which is stale. AP pw0.70 cells show `matched 3531/3534`.
- **Cost** (`:56-61, 171-178, 225-226`): USD per 1M tokens `PRICE_OUT = {3B:0.06, 7B:0.07, 14B:0.15, 72B:0.40}`,
  `PRICE_IN = {…, 72B:0.38}` (source: paper `tab:heterogeneous_cluster`, `neurips_2026.tex:115-118`).
  `cost_decode = Σ_ok output_len × PRICE_OUT[m] / 1e6 / n × 1000` (**USD per 1k requests**); `cost_full` adds `input_len × PRICE_IN[m]`.
  A served model without a price ⇒ problem (fail) and the row is skipped for cost and quality (`:173-176`). Decode is the basis of
  record (author decision 2026-07-31; paper `neurips_2026.tex:221`). The scheduler's own objective is also decode-only
  (`CS:84-104`, EFFECTIVE_CONFIG `cost_decode_only: true`). Cross-check tests: `V3S/test_cost_calc.py`.
- **Latency / throughput axes** read from the cell top-level, never defaulted: `mean_ms = mean_e2el_ms`, `p99_ms = p99_e2el_ms`,
  `thru = request_throughput` via `_need_metric` (missing / non-numeric / non-finite / negative ⇒ `MetricUnavailable` ⇒ refusal) (`:135-150, 234-236`).
- Provenance fields: `date`, `fill` (top-level metadata `fill` not in {"0","None",""}), `cap` (top-level string `cap`, or `max_output_len`,
  or `metadata.cap`), `cap_observed = p99_output_len`, `src` = parent dir, `mix` = Counter(model short name) (`:241-263`).

### 3.3 Two quality scales (do not mix)
Zero-safe mean (above) ≈ 0.42 — dominance basis. "Zeros-excluded" ≈ 0.51–0.52 = mean over non-zero scores, numerically what the old
`a or b` aggregator produced (posted rebuttal series per `rb-exp/v3-exp/docs/EXPERIMENTS.md:5-6`). **[V]** arithmetic:
λ30 RB(0.5,0.3,0.2) q=0.4156, n_zero=686 ⇒ 0.4156·3534/2848 = 0.516. Open item from `2026-08-04_V3_CLOSEOUT_HANDOFF.md:44-48`:
verify which aggregator produced the posted rebuttal numbers — unresolved.

### 3.4 Burst statistic — `V3S/burst_analysis.py`
Per (arm, pattern, seed, λ, preset) take `--metric` (default `p99_e2el_ms`) from each `qb_*.json` (name regex `:23`, preset default 532);
verify cell `burstiness` equals the name (`:69-75`, mismatch ⇒ UNUSABLE ⇒ rc 2). Ratio = gam025 / pois paired by (λ, seed, preset);
mean ± t_{0.975,df}·sd/√n (table `:32-46`); verdict = paired Δratio (learned − static) t-test (`:118-149`).

---

## 4. Aggregation, dominance, expected cells, gates

### 4.1 Which aggregator is "the fixed one"
- **v3 grid producer = `V3S/dominance.py`** (zero-safe, decode cost, fail-loud axes). It writes `DOMINANCE_l<λ>.csv` with
  columns `system,knob,disp,lam,cell,n,err,q,coverage,n_zero,cost_decode,cost_full,mean_ms,p99_ms,thru` (`:628-637`).
- Zero-bug history: `cara_paper/smoke_test_apr_13/scripts/aggregate_overnight_grid.py:129` — on branch `master` still
  `score = per_model.get(m) or per_model.get(m.replace("Qwen/",""))` (**buggy**); on `v3-campaign` fixed at `:129-138`
  ("v3 FIX (D2)", commit e905123). `aggregate_v2_grid.py:123-131` is the first fixed copy (identical md5 in
  `rb-exp/v2-exp/code/scripts/`). Both use v1/v2 filename schemas and **full** cost with a single price table (72B 0.40 in and out) — not
  usable for v3 cells. Still buggy elsewhere: `rb-exp/v2-exp/compute_bootstrap_ci.py:58`, and the public repo
  `route-balance` HEAD `reproduce/compute_bootstrap_ci.py:58`. `V3S/agg_w1b.py:13-21` uses the or-chain **deliberately** (posted basis,
  and on predicted `all_model_scores`) — Gemma W1-B only. `V3S/check_zero_safe.py:74` reproduces the or-chain on purpose as a comparator.

### 4.2 dominance.py rules
Inputs: `--cells DIR…`, `--lambda N | --all-lambdas`, `--scored`, `--cost-basis decode|full` (default decode), `--suggest`,
`--allow-partial-coverage`, `--fill-min-lambda 12`, `--max-span-hours 12`, `--rehearsal`, `--csv`.
- File discovery: `*.json` minus `*.provenance.json` (`:547-553`); `.json.INVALID`, `.bench.log`, `.decisions.jsonl`, `.csv`, `.md` ignored.
- Name → system/knob (`:102-121`): RB `wq{q}_wl{l}_wc{c}_lambda{λ}_n{N}` (disp "lpt"); BR `br4_t{dd}_{rr|sq|po2|rand}_l{λ}_n{N}`
  (t00 ⇒ "argmax", else `t{int/10^(len-1):.1f}`); AP `ap_pw{dd}_{disp}_l{λ}_n{N}` (knob `pw{…:.2f}`); Dispatcher-only `pt_{disp}_l{λ}_n{N}`.
- Own-best per baseline system = **all** co-maximizers of `q × thru` within 1e-9 (`:411-444`); RB must dominate every one.
- `dominates(rb, base)`: strict — ≥ on all four axes (q higher-better; cost, mean, p99 lower-better), > on at least one; any NaN axis ⇒
  refusal (`:310-345`).
- Reported: cheapest dominating RB tuple; "dominated-with-fill" if only fill tuples dominate (`:445-480`); else NOT DOMINATED with
  the closest tuple (fewest deficit axes, then smallest worst relative deficit) and optional `suggest()` next w_q (disabled below λ12) (`:481-507`).
- Cost-basis sensitivity block (`:597-626`) — **uses `max()` (first of ties), not the co-max set** (inconsistent with main verdict).

### 4.3 Every check that refuses (exit 2, `PROBLEMS`) in dominance.py
1 REHEARSAL marker in a cells dir without `--rehearsal` (`:557-564`, immediate rc 2). 2 unparseable file name (`:583-590`).
3 cell with no successful responses (`:161-163`). 4 served model without price (`:173-176`). 5 unmatched prompt with full coverage
required (`:211-214`). 6 missing/null/non-numeric/non-finite/negative `mean_e2el_ms`/`p99_e2el_ms`/`request_throughput` (`:135-150`).
7 differing declared caps at one λ (`:279-282`). 8 cells without a parseable `date` (`:293-295`). 9 date span > `--max-span-hours` (`:303-306`).
10 no cells at the λ (`:385-393`). 11 no RB cell (`:404-406`). 12 all own-best scores NaN for a system (`:427-431`). 13 no own-best
selectable (`:435-437`). 14 NaN on any axis inside `dominates` (`:322-330`).
Non-failing notes: corpus gap, "no declared cap", tie report, fill notes, cost-basis flip warning.
Driver treats rc≠0 as a refusal and also requires a `DOMINATED|NOT DOMINATED` line (`V3S/s2_driver.sh:977-994`).
**[V]** `dominance.py --cells CELLS/s2_p1_l30 --lambda 30` ⇒ rc 0, 31 cells, table identical to `DOMINANCE_l30.csv`; verdicts:
AP own-best pw0.25/po2 dominated by RB (0.1,0.8,0.1); BR own-best t0.7/rr by RB (0.1,0.1,0.8); Dispatcher rr by RB (0.1,0.1,0.8);
BR flips to not-dominated on `full`.

### 4.4 Expected-cells manifest — `V3S/expected_cells.py`
Knob lists (must equal `p1_pareto.sh`): RB 8 tuples `0.1:0.1:0.8, 0.1:0.8:0.1, 0.2:0.2:0.6, 0.2:0.6:0.2, 0.33:0.33:0.33, 0.5:0.3:0.2,
0.6:0.2:0.2, 0.8:0.1:0.1`; BR t ∈ {0.0,0.3,0.5,0.7} × {rr,sq,po2}; AP pw ∈ {0.25,0.39,0.53,0.7,0.8} × {rr,sq,po2}; Dispatcher-only
{rr,sq,po2,rand}; ⇒ **39 per rate × 5 rates {30,24,18,12,6} = 195**. Q1-burst: presets {532,uniform} × λ {12,24,30} × {pois,gam025}
× seeds 1-6 × arms {learned,static} = **144** `qb_{arm}_{pat}_s{s}_l{λ}_p{preset}_n3534.json` (`:26-77`). `--check DIR` reports MISSING,
UNEXPECTED, and `.INVALID` count; rc 0 only if exactly on target (`:101-134`).
**[V]** λ6, λ12: ON TARGET 39/39. λ18/24/30: 31 present, MISSING = the 8 BR sq/po2 cells, 8 `.INVALID` each (rc 2 by design — there is no
"accounted INVALID" state). Q1 on `s2_q1burst_PROD`: 144/144 present but 12 `base_{a,b,static}_*` cells flagged UNEXPECTED (rc 2).

### 4.5 Gates — inventory, where they run, what fails a cell
Launch-time (per scheduler session, `restart_v3`, `V3S/lib_serve_v3.sh:107-208`): single `cara_serve` after pkill (`:139-140,175-176`),
`/health` within 240 s (`:45-51,169-173`), `EFFECTIVE_CONFIG` line present in `cara_serve.log` (`:177-182`), **G-TIERS**
`runtime_gates.py tiers` (non-pipeline only; every loaded predictor tier `…_<hw>` in `ALLOWED_HW`, no duplicate tier per model) (`:195-206`).

Per grid cell (`run_cell_v3`, `lib_serve_v3.sh:266-312`) — a cell is refused/INVALID when:
1. **SKIP-INCOMPATIBLE (rc 2)**: payload is BR/AP and `model_deployment.json` lacks any of 3b/7b/14b/72b (`:250-278`).
2. **CONTAMINATED**: `$CONTAM_STATUS` (default `~/S2_CONTAM_STATUS`) missing (unless `MONITOR_OPTIONAL=1`), unreadable, older than
   `CONTAM_MAX_AGE` (2100 s), or `n_findings>0` (`:61-104`).
3. **RULE0-PRE-FAIL** (`CA2 pre <arm>`, `CA2:35-203`): pins file `S2_PINS.env` present with 6 non-empty pins (`:27-33`); exactly one
   scheduler proc; proc cwd == `$S2_ROOT`; no sibling `Block_*`; tree md5 (`find block -name *.py|*.json|*.sh|*.yaml|*.yml`, sorted,
   md5 of md5s) == pin; `--scheduler-config` from `/proc/<pid>/cmdline` md5 == pin; config semantics: `score_type=="deepeval"`,
   `quality_config.k==EXPECT_K` (50), no p100 in model_dir/instance_capacity/instance_metadata, capacity keys == metadata keys,
   `budget_tokens==256`, `lpt_sort_key=="max"`, `filter_enabled is False` (`:60-78`); KNN pkl md5, eval dataset md5, TPOT-head aggregate
   md5 == pins (`:80-88`); env `CARA_INPROC_PREDICTOR=1`, `CUDA_VISIBLE_DEVICES=` empty; `EFFECTIVE_CONFIG` present and its
   `static_tpot_ms_per_tok` is None for arms learned/arm1/pipeline, non-None for static/arm4 (**G-EFFECTIVE**) (`:94-111`); for
   non-pipeline arms the stdout log contains `[INPROC-XGB] tpot model loaded` and no fallback marker (`:113-129`). Writes
   `<cell>.provenance.json` (`:135-202`).
4. **CONFIG-FAIL** (`post_config`, `V3S/lib_post_config.sh:37-111`): POST `/v1/config` HTTP≠200 (previous router stays live); router
   digest (sha256 of sorted-keys compact JSON of `router`, 16 hex) ≠ live `router_sha`; for RB payloads, **G-WEIGHTS** readback of
   `GET /v1/config .scoring_weights`, 3 attempts 2 s apart, any sent key differing by >1e-9 or unreadable ⇒ fail.
5. **BENCH-FAIL**: benchmark process non-zero (e.g. ready check failed).
6. **RULE0-POST-FAIL ⇒ `mv <cell>.json <cell>.json.INVALID`** (`CA2:205-229`): `completed == N_PROMPTS` (3534), `failed == 0`,
   `response_details` non-empty, first record has `model` and `e2el`; optional dominant-tier assertion (grid passes none).
   (Implemented with Python `assert` — would be stripped under `-O`.)
7. `ARM_EXPECT` branch (tpot_gate on the cell's decision-log slice) — **no grid runner sets it**.
After a failed cell, `p1_pareto.sh:89-98` restarts the scheduler if `/v1/config` is unhealthy.

Per burst cell (`V3S/s2_q1_burst.sh:110-240`), additionally: ASSERT-1 live `static_tpot_ms_per_tok` matches arm right after POST
(`:132-139`, ARM-LEAK); RULE-0 pre runs **after** the POST (`:141-149`); ASSERT-2 cell `burstiness` == requested (`:165-169`);
ASSERT-3 `score_batched_calls` advanced and `score_perreq_calls` unchanged, unreadable/-1 ⇒ fail (`:179-201`); ASSERT-5 zero-call
(`tpot_zero_gate.py`: static ⇒ every batch `xgb_batched_tpot_ms ≤ 1.0` and `tpot_predict_calls` delta 0; learned ⇒ mean > 1.0 and delta > 0;
delta "NA" ⇒ fail) (`:206-213`); ASSERT-4 `tpot_gate.py` on the decision-log slice (static: one `lat_s/exp_len` ratio per tier, ratios differ
across tiers, each within 0.05 ms of requested; learned: ≥50 distinct ratios per tier, ≥100 rounds, ≥20 distinct exp_len; tiers inside
`ALLOWED_HW`; `static_active` flag agrees) (`:221-237`). Pre-sweep: `BURST_FLEET` declared and consistent with `ALLOWED_HW` (`:249-262`),
seed scope (`:267-268`), arrival pairing digests reproducible for every (seed, λ, pattern) (`:273-296`), per-preset same-config baseline
+ `runtime_gates.py arms` divergence (`:320-358`; a preset whose arms agree ≥ baseline% is skipped and logged to `PRESET_SKIPS.txt`).

Driver-level (`V3S/s2_driver.sh`): P4_GATES runs `gate_selftest.py` and `static_defect_gate.py --baseline static_gate_baseline.json`
(`:859-885`); per rate P6_GRID pulls cells, runs Phase E′ dominance and marks the rate done only if the count ≥ 39 (`:915-1018`).

Offline gates:
- `gate_selftest.py` — 12 suites (`:1517-1524`); **[V] static AST count of the current source = 162 cases, 89 MUST-FAIL** (matches the
  required "162 / 89 / 0 wrong"; "0 wrong" needs execution, not done here — read-only). The only recorded run log
  (`window_setup_work/s2_cluster/logs/gate_selftest.log`, Aug 3 01:27) shows an older 153/84/0; source was modified Aug 3 19:24.
  Suite `static` hardcodes root `/home/wd312/Code/llm/Block` (`:563`).
- `static_defect_gate.py` — classes 1 falsy-or on score/quality/cost lookups (AST), 2 hardcoded Qwen ladders + duplicated host→GPU map,
  3 fabricated defaults / `[False]*len(...)`, 4 watched config keys never read; exit 2 on any finding not in the triaged baseline
  (4 known findings in `static_gate_baseline.json`). Scans `block/{global_scheduler,predictor,benchmark,exp}` + public `reproduce/`
  — **not `v3_scripts/`**.
- `preflight_gate.py`, `preflight_tpot.py`, `assert_effective_config.py` — built, **no caller** in S2 (`s2_driver.sh:875-877` says so);
  preflight has three documented hollow checks (`2026-07-31_GATES.md:598-618`).
- `check_zero_safe.py` / `check_quality_determinism.py` — v3-era GATE-4/GATE-3, called only by the dead `gate_v3.sh`; load the aggregator
  from `/users/asdwb/Block/...` (`:23-24`). `check_quality_determinism` fails on corpus gaps (inconsistent with dominance) and its
  claimed check 3 (float equality) is not implemented.

### 4.6 The 24 INVALID cells
BR sq/po2 at λ18/24/30 (8 per rate): **[V]** e.g. `br4_t00_sq_l30`: completed 894, failed 2640, errors are aiohttp tracebacks
(`selector_events.py`, bad fd) — scheduler overwhelmed by the per-request `/instance_stats` fan-out; fail-loud refused, RULE-0 post
renamed them. "Filled" separately under `stats_blind_absorb` into `CELLS/brfill/*_BRFILL/` with names `br4_t<t>_<disp>_{A|B}_l<λ>` —
unparseable by `dominance.py` by design (never mixed into the grid).

---

## 5. Launch flow for one grid cell (end-to-end) and every env var

Head launcher (`v3-exp/head_scripts/_run_p1_l30.sh`): `cd $HOME/Block/.../v3_scripts; export CONTAM_STATUS=$HOME/S2_CONTAM_STATUS
ALLOWED_HW=a30,v100,a100 N_PROMPTS=3534 S2_MODE=production OUT=$HOME/Block/experiment_output/s2_p1_l30 LAMBDAS_OVERRIDE=30; exec bash p1_pareto.sh`.

1. `p1_pareto.sh` sources `lib_post_config.sh`, `payloads_v3.sh` (`BR_FALLBACK=Qwen/Qwen2.5-3B`, `BR_CKPT=models/cara/best_route_4way_qwen`),
   `lib_serve_v3.sh` (resolves `V3TREE` = tree containing the script, refuses sibling `Block_*`, sets `CARA_URL=http://127.0.0.1:8200`,
   `V3_CFG`, `MODEL_CFG`, `HOST_CFG`, `DATASET`, `TOK=Qwen/Qwen2.5-3B`, `GATE=cell_audit_s2.sh`, `V3_MAX_OUTPUT_LEN=1024`).
2. Phase restart: `restart_v3 cara` (RB) — later `restart_v3 pipeline` (BR, PT, AP) (`p1_pareto.sh:121-122,134-135`):
   `DECISION_LOG=${DECISION_LOG_DIR:-$OUT}/decisions_<mode>_<epoch>.jsonl`; `pkill -9 -f "cara[_]serve"`; rotate
   `$V3TREE/experiment_output/logs/cara_serve.log`; launch
   `setsid --fork env CUDA_VISIBLE_DEVICES="" CARA_INPROC_PREDICTOR=1 nohup python3 -u -m block.global_scheduler.cara.cara_serve
   --host 0.0.0.0 --port 8200 --model_config_path block/config/cara/s2/model_deployment.json --scheduling <mode> --chat
   --scheduler-config block/config/cara/s2/scheduler_config_s2.json --predictor-config block/config/cara/predictor_xgboost_3model.json
   --host_config block/config/cara/s2/host_configs.json --decision-log $DECISION_LOG > logs/cara_<mode>_<epoch>.log`;
   health; one proc; EFFECTIVE_CONFIG; G-TIERS.
3. Per cell: skip if `<OUT>/<cell>.json` exists (`p1_pareto.sh:81`) → fleet-compat → contamination → RULE-0 pre (+provenance) →
   `post_config` (payloads below) → warmup ×5 + `sleep 2` → note decision-log offset → `bench_v3` → RULE-0 post (or `.INVALID`) → append
   `INDEX_p1.md` row `| block | cell | lambda | OK|FAIL|SKIP-INCOMPATIBLE |`.
   Payloads (`p1_pareto.sh:125,141,153,166`):
   RB `{"scoring_weights":{"w_quality":q,"w_latency":l,"w_cost":c,"w_balance":0},"slo_defaults":{"lpt_sort_key":"max","assignment_strategy":"scoring"},"filter":{"type":"none"}}`;
   BR `{"router":{"type":"best_route_4way","kwargs":{"checkpoint_path":BR_CKPT,"confidence_threshold":t,"fallback_model":BR_FALLBACK}},"dispatch":{"type":disp},"filter":{"type":"none"}}`;
   PT `{"router":{"type":"passthrough","kwargs":{}},"dispatch":{"type":disp},"filter":{"type":"none"}}`;
   AP `{"router":{"type":"avengers_pro","kwargs":{"artifact_dir":"models/cara/avengers_4model/avengers_pro_qwen_pw<pw>"}},"dispatch":{"type":disp},"filter":{"type":"none"}}`.
   Dispatcher labels via exhaustive `dlabel` (rr/sq/po2/rand; unknown ⇒ return 1 ⇒ FATAL) (`:60-73`).
4. Driver (VPS) after the runner: count `ls $RATE_REMOTE/*.json`, rsync to `s2_cluster/cells/s2_p1_l<λ>/`, run
   `dominance.py --cells <dir> --lambda <λ> --suggest --csv <dir>/DOMINANCE_l<λ>.csv`, require rc 0 + verdict, mark `P6_GRID_<λ>` (`s2_driver.sh:925-1011`).

Env vars (all runners): `V3TREE, V3_CFG, MODEL_CFG, HOST_CFG, GATE, OUT, DECISION_LOG_DIR, DECISION_LOG, SCHED_LOG, SCHED_STDOUT_LOG,
CONTAM_STATUS, CONTAM_MAX_AGE, MONITOR_OPTIONAL, ALLOWED_HW, N_PROMPTS, ARM_EXPECT, LAMBDAS_OVERRIDE, CURRENT_SERVE_MODE, BR_CKPT,
BR_FALLBACK, S2_MODE, S2_ROOT, S2_PINS, PROV_OUT, CELL_PAYLOAD, CELL_LABEL, CUDA_VISIBLE_DEVICES (=""), CARA_INPROC_PREDICTOR (=1),
OPENAI_API_KEY (backend header)`. Burst/Q1: `SEEDS, BURST_LAMBDAS, LAMBDAS, PATTERNS, PRESETS, BURST_FLEET`. brfill/acell/retune:
`OUTROOT, CUTOFF_UTC`. Driver: `S2DIR, RSPEC, CFGDIR, LAMBDAS (30 24 18 12 6), N_PROMPTS, S2_MODE, S2_TOPOLOGY, WD_REMOTE_STEP,
WD_CELL_RUN, WD_SETUP, WD_WEIGHTS, WD_DEPLOY, KNN_SRC, TPOT_SRC, TPOT_SUBSET, GRID_OUT, GRID_CELLS_PER_RATE (39), REMOTE_HOME,
NO_MONITOR, MONITOR_INTERVAL (600), Q1_LAMBDAS`. Optional/dead: `SAVE_TO_PYTORCH_BENCHMARK_FORMAT`.

Scheduler read-back API the flow depends on: `GET /v1/config` (`CS:735-761`) → `scoring_weights, static_tpot_ms_per_tok, slo_defaults,
scheduling, router, router_sha, model_estimator, score_batched_calls, tpot_predict_calls, score_perreq_calls, stats_blind_absorb`;
POST merges (`slo_defaults.update`, `scoring_weights.update`) and re-emits EFFECTIVE_CONFIG (`CS:846-867`). EFFECTIVE_CONFIG keys seen
in provenance: `batched_path, budget_tokens, cost_decode_only, estimator_type, filter_enabled, inproc_predictor, knn_corpus, knn_dir,
knn_k, lpt_sort_key, quality_norm, score_batched_calls, score_perreq_calls, score_type, static_tpot_ms_per_tok, tpot_predict_calls, unwrap_prompt`.

---

## 6. Executed vs dead in the v3 S2 campaign

Executed: `VllmInstance` chat path; `Instance.query_instance` (feedback off); `StreamedResponseHandler`; `EP.async_request_cara_openai_completions`;
`BS.get_request` (Poisson; gamma 0.25 only in Q1-burst at λ12/24/30, RB-only), `calculate_metrics`, `benchmark`, `main_async`; vLLM
`CustomDataset`; `lib_serve_v3.sh` (all functions except the ARM_EXPECT branch); `lib_post_config.post_config`; `payloads_v3.sh` (only
`BR_CKPT/BR_FALLBACK` — `p1_pareto.sh` inlines its JSON, the `BR4*` vars are unused); `p1_pareto.sh`; `s2_q1.sh`; `s2_q1_burst.sh`;
`s2_driver.sh`; `CA2`; `runtime_gates.py` tiers/manifest/orphans/arms; `orphan_monitor.py`; `tpot_gate.py`; `tpot_zero_gate.py`;
`verify_pairing.py`; `dominance.py`; `gate_selftest.py`; `static_defect_gate.py`; `burst_analysis.py` and `expected_cells.py` (manual, post-hoc).
Exploratory (not grid): `a_cell.sh` (accepts ≥99% completion, no RULE-0), `br_fill_rerun.sh`, `l6_retune.sh`.

Dead: `OllamaInstance` (no ollama backend deployed); `VI` `/v1/completions` branch (always `--chat`); predictor feedback
(`IN:52-78, 84-135`); `LmsysDataset` (`dataset.py:15-91`); `vllm_sr` backend (`EP:187-273`); RSO sampling + `--rso-budget-table`
(`BS:54-85, 803-816, 1733-1748`; also references a global `args` inside `benchmark()`, NameError if imported); goodput; ramp-up; profile;
LoRA; `--max-concurrency`; embeddings metrics; `source` injection; pytorch-format sidecar; `CARA_SQUARE` square wave and `client_ttft`
(removed from the v3 tree, `git diff 1f86486 7af63c7`); `preflight_gate.py`, `preflight_tpot.py`, `assert_effective_config.py`;
`gate_v3.sh` + `check_zero_safe.py` + `check_quality_determinism.py` (v3-era Block_v3 paths); `agg_w1b.py`, `gemma_quality_join.py`
(Gemma W1-B, `/mydata/v3/...` paths); `aggregate_v2_grid.py`, `aggregate_overnight_grid.py`, `aggregate_full_215.py` (other filename
schemas); `CA2 post` dominant-tier argument (grid passes none); `set_ulimit()` return value (returns None ⇒ always prints the warning).

---

## 7. Bug-fix markers that MUST survive (file:line → meaning)

Backend / bench (no D-ids in these files; dated fixes):
- `VI:14-17` incremental UTF-8 decoder — prevents UnicodeDecodeError when a chunk splits a multi-byte char.
- `VI:158-159` strip only leading whitespace — preserves SSE `\n\n` delimiters.
- `VI:105-110` cast predicted length to int at the vLLM boundary.
- `EP:81-85` (Apr 26) `frequency_penalty=1.2` identical on training and inference paths.
- `EP:87-94` (Apr 26) send `num_predicted_output_tokens` (schema parity with `generate_latency_benchmark.py`).
- `BS:623-630` (Apr 26) client timeout 30 min (was 6 h).
- `CS:30-45`, `CS:107-120` KNN train/serve skew fix (2026-07-31): unwrap chat template before the estimator; no switch.
- `CS:84-104` decode-only objective cost; no env switch.
- `CS:963-998` fail-loud `/instance_stats` + ERA-MODE MASKING (`stats_blind_absorb`, author-directed 2026-08-04); `CS:1007-1018` DO NOT MASK.
- `CS:846-867` re-emit EFFECTIVE_CONFIG after every POST (merge-leak visibility). Scheduler-side "v3 FIX" markers (other spec's scope):
  `CS:157` (D26 host→GPU map), `:171` (D31), `:211` (live router config echo → `router_sha`), `:1474` (G4), `:1869` (D13), `:1974`
  (arm-4), `:3226` (D3/D4), `:3329` (D3 legacy).
Analysis:
- `aggregate_overnight_grid.py:129-138` "v3 FIX (D2)" zero-safe lookup (v3-campaign only); `aggregate_v2_grid.py:123-131` same.
- `dominance.py:37-38` zero-safe by construction; `:187-199` corpus gap ≠ aggregation failure (G-COVERAGE); `:207-214` report counts not
  rounded %; `:227-236` no `or 0.0` on comparison axes (author decision 2026-08-02); `:242-259` cap read from top-level string metadata;
  `:252-257` FILL stamp (G-FILL); `:277-306` provenance/span (G-SPAN); `:314-330` NaN is not a tie; `:373-375` suggest honours basis;
  `:385-393` no cells is a refusal; `:411-444` co-maximizer ties (author decision 2026-08-02); `:493-500` no fill below λ12;
  `:538-548` provenance sidecar exclusion; `:554-564` REHEARSAL refusal.
- `burst_analysis.py:18-23` preset is identity (G-PRESET); `:26-34` t not z; `:57-62` sidecar exclusion; `:94-100` falsy-zero fix.
- `verify_pairing.py:134-141` frozen clock; `:177-258` seed scope (G-SEEDSCOPE).
- `expected_cells.py:28-29` knob lists must mirror runners.
Launch / gates:
- `payloads_v3.sh:6-15` BEST-Route explicit `fallback_model=Qwen/Qwen2.5-3B` (v1 fell through to `model_pool[0]`=14B; F1).
- `lib_post_config.sh:5-19,46-62` silent-400 / router digest; `:63-109` G-WEIGHTS readback, retry the read not the POST.
- `lib_serve_v3.sh:12-23` ONE TREE; `:34-41` cap 256→1024 explicit (AUTHOR DECISION 2026-07-30); `:53-104` contamination fail-closed
  (reads local published status); `:114-118` bracketed pkill; `:120-137` log rotation; `:141-167` `setsid --fork` (G-DETACH);
  `:184-206` G-TIERS has a caller (D3/D4); `:241-278` G-FLEETCOMPAT rc 2.
- `p1_pareto.sh:33-35` N_PROMPTS overridable; `:56-73` exhaustive `dlabel` via return code; `:89-98` scheduler recovery after a failed
  cell (author-approved 2026-08-03); `:106-110` LAMBDAS_OVERRIDE respected.
- `CA2:8-19` generated pins; `:54-58` config discovered from the live process; `:90-111` G-EFFECTIVE; `:113-129` liveness from the
  stdout log; `:131-202` persisted provenance.
- `s2_q1_burst.sh:71-73` learned arm sends explicit `null` prior (G-ARMLEAK); `:141-145` RULE-0 pre after POST; `:173-194` G-BATCHPATH
  non-vacuous; `:203-213` ASSERT-5 zero-call (author decision 2026-08-03); `:215-226` no `if` around the arm gate.
- `s2_driver.sh:328-337` production must name ALLOWED_HW; `:88-97` mode-scoped state (G-STATE); `:299-319` no VPS-local path
  (G-LAUNCHER); `:548-558` re-record pins after P1_SETUP; `:981-994` refusal ≠ verdict (G-DOMINANCE).
- `static_defect_gate.py:10-16` classes named by D2, D14, D27, G1 / D13, D3, D26 / G2 / D10; `:94-100` Codex #8; `:144-152` Codex #9.
- `runtime_gates.py:21-62` D4 tiers / D3 first-match ambiguity; `:196-202,272-282` G-ARMS baseline is fleet-shaped (measure it).

---

## 8. Proposed rebuild interfaces

```python
# backend/client.py
@dataclass(frozen=True)
class BackendRequest:
    request_id: str; prompt: str; max_tokens: int                     # max_tokens -> max_completion_tokens (chat)
    temperature: float = 0.0; repetition_penalty: float = 1.0; frequency_penalty: float = 1.2
    predicted_decode_tokens: int | None = None; use_chat: bool = True
@dataclass
class BackendResult:
    generated_text: str; ttft_s: float; itl_s: list[float]; output_tokens: int
    success: bool; error: str; model: str; server_latency_s: float; instance_id: str; host: str
class VllmClient:
    def __init__(self, instance_id, host, ip, port, model, timeout_s=1800, session_mode="per_request"): ...
    async def generate(self, req: BackendRequest) -> BackendResult    # clock starts at call entry (parity with VI:84)
# backend/stats.py
@dataclass
class InstanceStats: num_running:int; num_waiting:int; num_active_decode_seqs:int; decode_ctx_p50:float; decode_ctx_p95:float
    decode_ctx_max:float; pending_prefill_tokens:int; pending_decode_tokens:int; kv_cache_utilization:float; kv_free_blocks:int
    token_budget_per_iter:int; prefill_chunk_size:int; max_num_seqs:int; num_preempted:int; raw: dict
async def fetch_instance_stats(ip, port, timeout_s=3.0, blind_absorb=False) -> InstanceStats | None   # raises InstanceStateUnavailable
# bench/arrivals.py
def arrival_schedule(n:int, rate:float, burstiness:float, seed:int) -> list[float]   # legacy np.random.RandomState(seed).gamma == global seed
def schedule_digest(times:list[float]) -> str                                        # ((1e6+t)-1e6) '%.9f|' sha256[:16]
# bench/dataset.py
def load_eval_prompts(path, tokenizer_id="Qwen/Qwen2.5-3B", shuffle_seed=0, chat_template=True) -> list[Sample]  # Sample(prompt, prompt_len, raw_prompt, index)
# bench/runner.py
@dataclass
class RequestRecord:   # exact v3 per-request keys (§2.5), e2el/ttft/itl in seconds
@dataclass
class CellRecord:      # exact v3 top-level keys (§2.6) + metadata: dict[str,str] flattened to top level on write
async def run_cell(base_url, samples, arrivals, max_tokens=1024, metadata={}, ready_check=True) -> CellRecord
# analysis/quality.py
class QualityLookup: def score(self, prompt:str, model:str) -> tuple[float|None, Literal["matched","unmatched","corpus_gap"]]
# analysis/cost.py   PRICE_IN/PRICE_OUT tables; def cost_per_1k(rows, basis="decode") -> float
# analysis/summary.py
@dataclass
class CellSummary: system; knob; disp; lam; cell; n; err; q; coverage; matched; unmatched; corpus_gap; n_zero
    cost_decode; cost_full; mean_ms; p99_ms; thru; date; fill:bool; cap:float; cap_observed:float; src; mix:dict
    q_nonzero: float     # explicitly labelled zeros-excluded scale, never used for dominance
# analysis/names.py   parse_cell_name(stem) -> CellId | None (the 4 grid patterns + qb_/base_ patterns); a name that does not parse FAILS
# analysis/dominance.py  own_best(cands) -> list[CellSummary] (co-max, tol 1e-9); dominates(a,b,basis) -> (bool, deficits); judge_lambda(...)
# analysis/manifest.py  expected_grid(rates)->list[ExpectedCell]; expected_q1()->...; check(dir) -> Coverage(missing, unexpected, invalid, accounted_invalid)
# gates/  every gate: run(ctx) -> GateResult(ok:bool, code:int, evidence:dict, message:str); "cannot obtain evidence" => ok=False
#   rule0_pre, rule0_post, config_verify (router_sha + weights readback), contamination, tiers, effective_config, fleet_compat,
#   burst_assert, batchpath, zero_call, tpot_arm, pairing, seed_scope, static_defect, arms_divergence, selftest
```
Design constraints: keep the v3 JSON schema byte-compatible so v3 cells load unchanged; the aggregator must accept both old cells and new
ones; count cells from the manifest, never from `*.json` globs.

---

## 9. Golden-test recipe (re-aggregate v3 cells, compare number-for-number)

All v3 cells are ~19–21 MB each (3534 records); there are no small ones. Recommended fixture set (`CELLS/s2_p1_l30/`):
`wq0.5_wl0.3_wc0.2_lambda30_n3534.json` (20,507,998 B, RB cara mode), `pt_sq_l30_n3534.json` (20,858,733 B, pipeline sq),
`pt_rand_l30_n3534.json` (21,055,200 B), `ap_pw07_rr_l30_n3534.json` (20,540,450 B, AP with 3 corpus gaps),
`br4_t07_rr_l30_n3534.json` (21,187,009 B). Reference table: `CELLS/s2_p1_l30/DOMINANCE_l30.csv` (md5 `bd137330bca45abca9100c94165db396`),
plus `DOMINANCE_l{6,12,18,24}.csv` in their rate dirs.
1. **Per-cell summary:** rebuild `summarize` on each fixture; compare `n, err, q, coverage, n_zero, cost_decode, cost_full, mean_ms, p99_ms, thru`
   to the CSV row (exact float equality is achievable: same inputs, same arithmetic order; accept ≤1e-12 relative).
   Spot values (λ30): RB(0.5,0.3,0.2) q 0.4156196943972835, cost_decode 0.02308877475947936, mean 4531.839192999472, p99 23469.977680301832,
   thru 22.53493952760076; AP pw0.70/rr q 0.4272727272727273, coverage 0.9991511035653651. Sanity: AP q is identical across dispatchers and rates.
2. **Full-rate verdicts:** point the rebuild at each `s2_p1_l<λ>` dir; λ30 must give rc 0, 31 cells, and the three DOMINATED verdicts in §4.3;
   λ6/λ12 near-parity numbers must match `v3-exp/EXPERIMENTS.md:49-54` (AP own-best pw0.70/**po2**: λ6 0.4273/0.0379/4352/31062).
   Reference run: `python3 -B V3S/dominance.py --cells <dir> --lambda <λ>` (read-only without `--csv`; ~1 min per rate).
3. **Bench metric code:** recompute every summary field from `response_details`+`duration` (§3.1) for the fixtures — exact [V].
4. **Arrivals:** `schedule_digest(arrival_schedule(3534, λ, b, s))` for each line of `CELLS/s2_q1burst_PROD/ARRIVAL_DIGESTS.txt`
   (36 unique (seed ∈1-6, λ ∈{12,24,30}, pattern), 73 lines) — all must match [V]. Grid cells use seed 1 Poisson at λ ∈{6,…,30}; generate their
   digests with the v3 `verify_pairing.py --digest` for the new rates.
5. **Prompt order:** `load_eval_prompts(...)[i].raw_prompt == unwrap(cell.response_details[i].prompt)` for all i on any grid or burst cell [V].
6. **Burst analysis:** rebuild `burst_analysis` on `CELLS/s2_q1burst_PROD` (156 cells; `qb_*` 144 used) vs `ANALYSIS_FULL_144.txt`
   (e.g. preset 532 λ24 learned 1.048 ± 0.035, static 1.002 ± 0.040).
7. **Manifest:** λ6/λ12 on target; λ18/24/30 → 8 MISSING + 8 INVALID each; Q1 → 144 present + 12 `base_*` (must be classed as known baselines, not UNEXPECTED).
8. **Gate self-test:** port the 162 cases / 89 MUST-FAIL; execute and require 0 wrong.
Do not use `*.PRESWAP_ORIG` (two λ30 RB cells retired by the span gate) or `*.BACKLOG_ORIG` (one burst cell) as fixtures.

---

## 10. Hazards for a porter

1. **Shuffle seed is fixed at 0**, not `--seed` (vLLM fork `get_samples` omits `random_seed`). Passing the seed to the shuffle changes
   the prompt order per seed and breaks parity. `verify_pairing.check_scope` says the python RNG is "never consumed" — the dataset
   shuffle does consume it but reseeds to 0 first.
2. **Double chat templating** (client template + `--chat` user message). Output lengths, latency and cost depend on it. Changing it is a
   system change that needs a control arm.
3. **Arrival digest quirk**: `(1e6+t)-1e6` before formatting; digest of raw times never matches the v3 records.
4. Arrivals are normalised so the last one lands at exactly N/λ; the first is > 0. Gamma needs `shape=b, scale=1/(λ·b)`; CV² = 1/b.
5. Nothing between `np.random.seed` and the gamma draws may consume the global numpy stream (tokenizer/pandas today do not; the ready
   check does not). A new library call there silently changes every schedule.
6. **TTFT is server-side** (excludes scheduler queueing); **TPOT mixes client e2el with server TTFT**; ITL contains a ~0 ms first gap.
   Report definitions explicitly or keep them identical.
7. **Off-instance residual** = client e2el − server_latency; server_latency starts before the per-request TCP connect. Switching the
   backend client to a pooled session moves time between the two and lowers backend latency — a code-level difference that needs its
   own control run (global FIRST RULE).
8. Per-request `ClientSession` + unbounded open-loop tasks + sq/po2 stats fan-out = fd churn that killed the scheduler at λ≥18
   ("Bad file descriptor"). The 24 INVALID BR cells are a finding; a hardened rebuild that "fixes" it must still run the era arm.
9. `num_predicted_output_tokens` from the bench is the cap; the scheduler overwrites it; `VI:109` falls back to `max_tokens` on a 0 prediction.
10. **Driver count bug**: `s2_driver.sh:945-946` counts `*.json` including `.provenance.json` (λ30 "produced 70 cell(s)" with 31 valid) so
    rates with INVALID cells were marked done. Count via the manifest.
11. Grid RULE-0 pre runs **before** `post_config` (`lib_serve_v3.sh:280-282`), so a grid cell's `provenance.resolved_effective_config`
    shows the previous POST's state; the burst runner does it after.
12. `/v1/config` merges; `static_tpot_ms_per_tok` and `stats_blind_absorb` persist across cells until restart; send explicit nulls.
13. Two quality scales (§3.3); the aggregator zero bug still exists on `master` and in `route-balance` HEAD `reproduce/compute_bootstrap_ci.py:58`.
    Never port those copies.
14. Corpus gaps: 4 pairs (not 1); coverage < 1 is expected; `check_quality_determinism` would fail such cells.
15. Cost is per 1k **successful** requests; quality is over matched only; 72B input price 0.38 vs 0.40 in the older aggregators.
16. Metadata values are strings (`cap:"1024"`, `seed:"2"`); `burstiness` is a float from args. Readers must cast.
17. `dominance` sensitivity block uses first-of-ties; `dominance` cannot see missing knobs (BR own-best at λ≥18 is over rr only);
    a system with zero cells at a λ is silently omitted from verdicts.
18. `expected_cells` has no notion of accounted INVALID or baseline cells — it always returns rc 2 for λ18/24/30 and the burst dir.
19. `burst_analysis` CLI needs `--cells` (`REPRODUCE.md:81` shows a positional arg); `tcrit` for df not in the table returns the
    next-larger df's (smaller) value.
20. RULE-0 post uses `assert` (stripped by `python -O`); `a_cell.sh` accepts 99% completion — exploratory only.
21. `gate_selftest` suite `static` is bound to `/home/wd312/Code/llm/Block`; the recorded log (153/84) predates the current source (162/89).
22. Request ids are random per run; join cells by unwrapped prompt text, never by id.
23. The ready-check request and 5 warmups go through the scheduler and appear in decision logs and counters.
24. `Authorization: Bearer None` is sent when `OPENAI_API_KEY` is unset; `failed_requests_<ts>.log` lands in the repo root; `set_ulimit`
    always prints a warning.
25. The bench client, the scheduler and a 14B vLLM instance share the head node (`env/ENVIRONMENT.md:51`) — CPU contention is part of the
    measured latency; moving the client changes numbers.
26. `vllm_params.port=7000` in `model_deployment.json` is dead; the backend listens on 8000.
27. `date` is the head's local time string; the 12 h span gate depends on all cells using one clock.
