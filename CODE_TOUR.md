# Code tour

RouteBalance is a scheduler for a fleet of model instances of different sizes on different GPUs. For each
request it picks the (model, instance) pair jointly, trading predicted quality, cost and load-aware latency.
Every baseline of the framework (a router that picks a model, then a dispatcher that picks an instance) runs in the
same server, on the same telemetry and the same dispatch path. That sharing is what makes the comparison fair by
construction. The vLLM Semantic Router runs as its own stack in front of the same engines (`vllm-sr-bench/`).

The core is `routebalance/`. This tour follows the order a request takes through it.
For each file it lists what to check: the invariants the rest of the system relies on. "v3" is the
previous implementation this code was ported from, and "v2" the one before it.

**If you have one hour**, read these about 1,400 lines. They decide the results:
`schedule/` (all four files), `telemetry.py`, `baselines/dispatch.py` + `baselines/pipeline.py`,
`estimate/latency.py`.

---

## 1. Configuration and types: `config.py`, `types.py`, `hw.py`

- `load_fleet`: the fleet (its tiers, each with a model, a GPU, prices and `max_num_seqs`, and its instances) comes
  from `configs/fleet_*.json`, and the addresses from an untracked hosts file.
- `SchedulerConfig` is one complete, strictly parsed document. It has no defaults and rejects unknown keys,
  duplicate JSON keys and NaN/inf. A cross-field validator rejects any combination that would silently do
  nothing, or give a baseline RouteBalance's mechanisms. Examples: weights outside RouteBalance scoring, the
  admission filter in pipeline mode, dead reckoning in a baseline (it is RouteBalance's own, and only the latency
  sweep, which collects RouteBalance's training features, runs with it), and an unsupported (latency source,
  envelope) pair.
- `config_digest` and `fleet_digest` identify what was served. The runner refuses a cell when the live
  digest is not the scope's.
- `hw.py` is the only map from CloudLab node type to GPU.

Check: every experiment parameter is a required field. Nothing is read from the environment.

## 2. Load: `telemetry.py`

- Three ways to read load, set by the config's `telemetry_read`:
  - `per_round`, RouteBalance's own: `fetch_round()` scrapes every instance's Prometheus `/metrics`
    (running, waiting, KV usage, preemptions) at each round's start, and the round is scheduled from that fetch.
  - `poll`: the pipeline systems, which decide per request, read the view from a cache kept by one background
    poller per instance every `poll_ms`, as llm-d's endpoint picker scrapes each model server's `/metrics` every
    50 ms. A slow instance never delays the others. RouteBalance on this cache is the cache control.
  - `per_request`, v3's baselines: each request `fetch()`es the candidates its dispatcher compares (every candidate
    for least outstanding, and power of two's two drawn instances) and reads the engines' own counts. G1 runs it as
    a control.
- Freshness is measured from when a scrape was sent. A failed, slow or malformed scrape leaves the instance's last
  good snapshot in use, as a scraped cache does, and every decision logs the snapshot's age and its last error. Only an
  instance never read fails the round or the request. Nothing falls back to an idle default.
- **Dead reckoning (RouteBalance's own).** Every request an engine serves was sent by this scheduler, so the
  scheduler's own in-flight count is the instance's load. The engine gauge only splits it into running and
  waiting: `running = min(engine running, in_flight)`, `waiting = in_flight − running`. The baselines' views run
  with it off, and their load-reading dispatchers read the engines' gauges.
- **Pending decode tokens.** Each dispatch opens a token with its predicted decode length. Streamed
  progress counts it down. At the prediction it is extended by +10 and counts down again.
- `Load.engine_waiting` is the raw gauge. The round-size rule reads it (`schedule/rounds.py`).

Check: every `record_dispatch` is closed exactly once, by `record_completion` or `record_failure`.
`server.py` guarantees this on every path.

## 3. Engines: `backend.py`

- One streaming request to official vLLM 0.18.1 `/v1/chat/completions`. The raw prompt goes in as a single
  user message, so vLLM applies the chat template exactly once.
- `stream_options.continuous_usage_stats` puts the cumulative `completion_tokens` on every chunk. Progress
  is counted from that, not from chunks, because vLLM merges tokens into one delta under load.
- Success requires the final usage-only chunk. An SSE `error` event is a failure (vLLM's mid-stream failure
  path). `prompt_tokens` comes from vLLM's own usage count. `timed_out` marks timeouts.
- `BackendUnreachable` means the connection was never made, so the request cannot have reached the engine.

## 4. Predictions: `estimate/`

- `prompt.py`: the single chat unwrap. The estimator and every router see the user's text.
- `knn.py`: per-model quality and expected length from the prompt's k nearest neighbours. This is a
  bitwise port of v3 `estimate_batch`. `prior="per_model_mean"` gives the prompt-blind ablation. The index
  is md5-pinned and the encoder revision is pinned.
- `latency.py`: the latency that enters the objective = a TPOT source × an envelope, plus the prompt's prefill
  when the configuration has a prefill table.
  - Sources: `constant` (per tier), `profiled_curve` (TPOT as a function of the running batch, from the lease's
    own calibration), or `xgboost`.
  - Envelopes: `own_length` (queue-blind), `queue_aware` (v3's learned arm), or `batch_aware` (TPOT at batch
    b+1 × length, plus a waiting term when the instance is full).
  - Supported pairs are listed in `config.LATENCY_PAIRS`.
- `features.py` + `xgboost_tpot.py`: the learned heads read only what the scheduler knows at
  serving time. Training rows are built by the same `feature_values`, from the lease-night sweep the
  server logs.

Check: the constant and curve sources never import xgboost, and `predict_calls` stays 0 (the zero-call gate).

## 5. RouteBalance: `schedule/`

- `rounds.py`: v3's round collection. A round closes when it is full or at its timeout.
  - Adaptive limits are 4/50, 16/100 or 32/200 ms, by the share of engines with waiting requests.
  - An adaptive round also takes the requests already waiting in the scheduler when it opens, up to 32.
  - A cancelled collection keeps its requests for the next round.
  - It avoids `asyncio.wait_for`, because on 3.10 that can swallow a cancel.
- `score.py`: the objective `w_q·Q + w_c·C + w_l·L`, summed in v3's order, so scores are bitwise v3's.
  - Q is min-max over the candidates, and C = 1 − cost/max, the cost being the prompt tokens at the input price plus
    the expected length at the output price.
  - L uses `max_divide` (v3) or `fixed_ref`: the request's best unloaded latency over this one.
  - `normalize_over`: `request` (v3) scales each term over the request's own candidates. `round`, RouteBalance's
    main setting, takes the quality range and the cost maximum over every request × instance of the round, and
    `max_divide`'s latency maximum is the round's running maximum.
  - Ties go to the shorter queue, then the least recently assigned instance, then fleet order.
- `assign.py`: one round (Algorithm 1).
  - LPT order by the configured statistic.
  - Every instance is a candidate.
  - Each request is scored on the instances' current state.
  - The winner's state then counts the assignment (one more waiting request, plus its decode tokens), so
    later requests in the round see earlier ones.
  - The optional budget admission filter lives here too: v3's, as its budget sweep ran it (expected cost within
    the budget, else the cheapest by worst-case cost).
- `budget.py`: v3's max_tokens clamp, shared by every system, and the cost arithmetic.

## 6. Baselines: `baselines/`

- `routers.py` picks a model:
  - `passthrough`, which gives every instance to the dispatcher.
  - BEST-Route as its authors ship it: upstream's DeBERTa ranker and routing rule, loaded from a pinned
    checkout.
  - AvengersPro, the upstream cluster router at a pinned commit, its embedder at a pinned revision.
  - `rb_decoupled`, which uses RouteBalance's own predictors and objective on idle-instance latency, so the
    model is fixed before load is visible.
  - A router's model call runs once per request, on its own lane off the event loop, as shipped routers run
    theirs (RouteLLM, NVIDIA's LLM Router). The lane's size is the configuration's router concurrency: 32 calls at
    once in the main grid (RouteBalance's largest round), 1 as shipped. Requests queue at the router, and that wait is
    part of their routing time.
- `dispatch.py` picks an instance among the router's candidates:
  - round robin.
  - least outstanding: the shortest running + waiting queue on the engines' own counts (v3's shortest queue).
  - power of two: of two drawn instances, the one with the shorter engine queue.
  - random.
  - LiteLLM's `lowest_latency` (a 10-sample mean, in which a timeout counts 1000 s).
  - Ties: least outstanding gives them to the least recently assigned instance, as stock balancers rotate among
    ties, power of two to the first instance drawn, and `lowest_latency` to a draw from its seeded generator.
- `pipeline.py`: router, then candidates, then loads from the shared view, then dispatcher. The
  dispatch is recorded with no await in between (its in-flight count and decode tokens are tracked for every
  system), so on the engines' gauges concurrent selections of least outstanding see each other only through its
  least-recently-assigned tie-break.

## 7. The server: `server.py`, `decision_log.py`

- One FastAPI app for every system, served through uvicorn as v3's scheduler and vLLM's own server are:
  - `POST /v1/completions` (the contract is in `bench/client.py`).
  - `GET`/`POST /v1/config`: a swap happens only while idle, is built completely first, and leaves the old
    configuration live on any failure.
  - `/v1/stats` and `/health`.
  - `/v1/manifest`: what the process EXECUTES: the code digests (at start and now), library versions, launch values
    and thread pools, the telemetry and session settings, and the md5 of every artifact actually loaded. The
    campaign's parity gate compares these across the systems of a comparison group.
- Shared by construction:
  - one engine session on vLLM bench's stock connector (no connection limit, keep-alive). A request that waits
    for a connection is counted, and the arm gate invalidates that cell.
  - one telemetry view, with its own scrape pool.
  - one backend client.
  - one 4-thread pool for RouteBalance's batch calls (the estimator and the scheduling round), and a
    lane for the baselines' per-request router calls, sized by the router concurrency.
  - one dispatch path, `Service._serve`: record, budget clamp, stream with progress, close.
- `RouteBalance.run` is the round loop:
  1. collect a round.
  2. estimate (pool).
  3. fetch every instance's load once (the per-batch fetch, or the pollers' cache in the cache control), and adapt
     the round size.
  4. `schedule_round` (pool).
  5. record every dispatch.
  A failed round fails its requests loudly.
- Evidence:
  - `decisions.jsonl`: one line per request, whatever happened, with the config digest.
  - `latency.<digest>.jsonl`: written only for the latency sweep configuration. Its features are exactly the
    learned heads' serving inputs.
  - The event-loop lag probe.

Check: no path leaves a dispatch token open. That covers engine errors, unreachable engines, failed rounds,
cancellation before and after a round places a request, and duplicate request ids.

---

## Outside the core

- `bench/`: the open-loop benchmark.
  - v3's arrival generator.
  - Prompt order.
  - A pooled client that validates every reply strictly.
  - The per-cell record, written once and never overwritten.
- `analysis/`:
  - per-cell metrics, with quality as the DeepEval score of (prompt, served model) in the scored corpus, zero-safe.
  - the cell manifest (expected cells accounted for).
  - frontiers, equal-quality deltas (one treatment configuration on the quality-latency frontier for every
    metric, the range over the seeds, and a request bootstrap within the paired seeds), the SLO (TTFT and TPOT
    within k times the tier's single-request values, after SOLA), its attainment and goodput.
- `tools/`:
  - `campaign.py`: reads the rendered campaign's plan (`configs/render36/plan.json`).
  - `cell_driver.py`: one cell's steps.
  - `engine_plan.py`: fleet × hosts × engine args → one vLLM launch per instance.
  - the TPOT calibration (row W1), the TPOT tables and the XGBoost retrain, and the budget sweep's served-text
    judge loop.
- `scripts/` (shell):
  - `setup.sh`: one node, per role, resumable across the V100 driver reboot, ending with a self-check.
  - `deploy.sh` and the stop scripts.
  - `run_cell.sh`: drain, apply, preflight, read the engines (`tools/engines.py`), warm up, run, read them
    again, gates, publish, mirror. `run_scope.sh` runs a
    scope's cells.
- `configs/`: the fleets, engine args, the scheduler's launch values, the pre-registered rules and the rendered
  campaign (`render36/`). A launch value still `OWNER-MUST-SET` (`scheduler_launch.example.json` is the template)
  is refused at deploy.

`docs/reproduce/` gives the reproduction stages, and `docs/reproduce/gates.md` what each gate refuses.
