# Code tour

RouteBalance is a scheduler for a fleet of model instances of different sizes on different GPUs. For each
request it picks the (model, instance) pair jointly, trading predicted quality, cost and load-aware latency.
Every baseline (a router that picks a model, then a dispatcher that picks an instance) runs in the same
server, on the same telemetry and the same dispatch path. That sharing is what makes the comparison fair
by construction.

The core is `routebalance/` (about 3,600 lines). This tour follows the order a request takes through it.
For each file it lists what to check: the invariants the rest of the system relies on. "v3" is the
previous implementation this code was ported from and tested against (tests/equivalence/).

**If you have one hour**, read these about 1,200 lines. They decide the results:
`schedule/` (all four files), `telemetry.py`, `baselines/dispatch.py` + `baselines/pipeline.py`,
`estimate/latency.py`.

---

## 1. Configuration and types: `config.py` (414), `types.py` (33), `hw.py` (32)

- `load_fleet`: the fleet (tiers: model, GPU, prices, `max_num_seqs`; instances) comes from
  `configs/fleet_*.json`; addresses come from an untracked hosts file.
- `SchedulerConfig` is one complete, strictly parsed document. It has no defaults and rejects unknown keys,
  duplicate JSON keys and NaN/inf. A cross-field validator rejects any combination that would silently do
  nothing, or give a baseline RouteBalance's mechanisms. Examples: weights outside RouteBalance scoring; the
  admission filter in pipeline mode; dead reckoning in a baseline (it is RouteBalance's own; only the latency
  sweep, which collects RouteBalance's training features, runs with it); an unsupported (latency source,
  envelope) pair.
- `config_digest` and `fleet_digest` identify what was served. The runner refuses a cell when the live
  digest is not the scope's.
- `hw.py` is the only map from CloudLab node type to GPU.

Check: every experiment parameter is a required field. Nothing is read from the environment.

## 2. Load: `telemetry.py` (328)

- Three ways to read load, set by the config's `telemetry_read` (D53, D54, D55):
  - `per_round`, RouteBalance's own: `fetch_round()` scrapes every instance's Prometheus `/metrics`
    (running, waiting, KV usage, preemptions) at each round's start, and the round is scheduled from that fetch.
    A failed scrape fails the round; an older snapshot is never used.
  - `poll`: the pipeline systems, which decide per request, read the engines' own counts from a cache kept by
    one background poller per instance every `poll_ms`, as llm-d's endpoint picker scrapes each model server's
    `/metrics` every 50 ms. A slow instance never delays the others. RouteBalance on this cache is the D54 control.
  - `per_request`, v3's baselines: each request `fetch()`es its router's candidates and reads the engines' own
    counts. G1 runs it beside the cache (D55).
- Freshness is measured from when a scrape was sent. A malformed reply invalidates the snapshot. A failed
  fetch lets the last snapshot age out after `max_age_ms`. Nothing falls back to an idle default.
- **Dead reckoning (RouteBalance's own).** Every request an engine serves was sent by this scheduler, so the
  scheduler's own in-flight count is the instance's load. The engine gauge only splits it into running and
  waiting: `running = min(engine running, in_flight)`, `waiting = in_flight − running`. The baselines' views run
  with it off and read the engines' gauges.
- **Pending decode tokens.** Each dispatch opens a token with its predicted decode length. Streamed
  progress counts it down. At the prediction it is extended by +10 and counts down again.
- `Load.engine_waiting` is the raw gauge. The round-size rule reads it (`schedule/rounds.py`).

Check: every `record_dispatch` is closed exactly once, by `record_completion` or `record_failure`.
`server.py` guarantees this on every path.

## 3. Engines: `backend.py` (257)

- One streaming request to official vLLM 0.18.1 `/v1/chat/completions`. The raw prompt goes in as a single
  user message, so vLLM applies the chat template exactly once.
- `stream_options.continuous_usage_stats` puts the cumulative `completion_tokens` on every chunk. Progress
  is counted from that, not from chunks, because vLLM merges tokens into one delta under load.
- Success requires the final usage-only chunk. An SSE `error` event is a failure (vLLM's mid-stream failure
  path). `prompt_tokens` comes from vLLM's own usage count. `timed_out` marks timeouts.
- `BackendUnreachable` means the connection was never made, so the request cannot have reached the engine.

## 4. Predictions: `estimate/`

- `prompt.py` (26): the single chat unwrap. The estimator and every router see the user's text.
- `knn.py` (178): per-model quality and expected length from the prompt's k nearest neighbours. This is a
  bitwise port of v3 `estimate_batch`. `prior="per_model_mean"` gives the prompt-blind ablation. The index
  is md5-pinned and the encoder revision is pinned.
- `latency.py` (245): the latency that enters the objective = a TPOT source × an envelope.
  - Sources: `constant` (per tier), `profiled_curve` (TPOT as a function of the running batch, from the lease's
    own calibration), or `xgboost`.
  - Envelopes: `own_length` (queue-blind), `queue_aware` (v3's learned arm), or `batch_aware` (TPOT at batch
    b+1 × length, plus a waiting term when the instance is full).
  - Supported pairs are listed in `config.LATENCY_PAIRS`.
- `features.py` (22) + `xgboost_tpot.py` (119): the learned heads read only what the scheduler knows at
  serving time. Training rows are built by the same `feature_values`, from the lease-night sweep the
  server logs.

Check: the constant and curve sources never import xgboost, and `predict_calls` stays 0 (the zero-call gate).

## 5. RouteBalance: `schedule/`

- `rounds.py` (104): v3's round collection. A round closes when it is full or at its timeout.
  - Adaptive limits are 4/50, 16/100 or 32/200 ms, by the share of engines with waiting requests.
  - A cancelled collection keeps its requests for the next round.
  - It avoids `asyncio.wait_for`, because on 3.10 that can swallow a cancel.
- `score.py` (87): the objective `w_q·Q + w_c·C + w_l·L`, summed in v3's order, so scores are bitwise v3's.
  - Q is min-max over the candidates.
  - C = 1 − cost/max, with a decode-only cost.
  - L uses `max_divide` (v3) or `fixed_ref`: the request's best unloaded latency over this one.
  - Ties go to the shorter queue, then the least recently assigned instance, then fleet order.
- `assign.py` (148): one round (Algorithm 1).
  - LPT order by the configured statistic.
  - Every instance is a candidate.
  - Each request is scored on the instances' current state.
  - The winner's state then counts the assignment (one more waiting request, plus its decode tokens), so
    later requests in the round see earlier ones.
  - The optional budget admission filter lives here too.
- `budget.py` (63): v3's max_tokens clamp, shared by every system, and the admission rule.

Check: `tests/equivalence/test_rb_decision_replay.py` re-scores every banked v3 decision (λ6 and λ30, 8
presets). The rebuild picks v3's model on all of them.

## 6. Baselines: `baselines/`

- `routers.py` (453) picks a model:
  - `passthrough`, which gives every instance to the dispatcher;
  - BEST-Route, a DeBERTa classifier with an optional confidence floor;
  - AvengersPro, the upstream cluster router at a pinned commit, its embedder at a pinned revision;
  - `rb_decoupled`, which uses RouteBalance's own predictors and objective on idle-instance latency, so the
    model is fixed before load is visible.
  - A router's model call runs once per request, one request at a time, on its own lane off the event loop,
    as shipped routers run theirs (RouteLLM, NVIDIA's LLM Router). Requests queue at the router; that wait is part
    of their routing time.
- `dispatch.py` (173) picks an instance among the router's candidates:
  - round robin;
  - least outstanding: the shortest running + waiting queue on the engines' own counts (v3's shortest queue);
  - power of two;
  - random;
  - LiteLLM's `lowest_latency` (a 10-sample mean; a timeout counts 1000 s).
  - Ties go to the least recently assigned instance, as stock balancers rotate among ties.
- `pipeline.py` (85): router, then candidates, then loads from the shared view, then dispatcher. The
  dispatch is recorded with no await in between (its in-flight count and decode tokens are tracked for every
  system); a baseline's concurrent selections see each other only through the least-recently-assigned tie-break,
  since it reads the engines' own counts.

Check: replays match v3 and the banked cells for BEST-Route and AvengersPro on 3,534/3,534 prompts per
setting. Dispatchers match v3 on a 20,000-event trace.

## 7. The server: `server.py` (717), `decision_log.py` (72)

- One aiohttp app for every system:
  - `POST /v1/completions` (the contract is in `bench/client.py`);
  - `GET`/`POST /v1/config`: a swap happens only while idle, is built completely first, and leaves the old
    configuration live on any failure;
  - `/v1/stats`, and `/health`;
  - `/v1/manifest`: what the process EXECUTES: the code digests (at start and now), library versions, launch values
    and thread pools, the telemetry and session settings, and the md5 of every artifact actually loaded. The
    campaign's parity gate compares these across the systems of a comparison group.
- Shared by construction:
  - one engine session, bounded far above any in-flight count; a request that waits for a connection is
    counted, and the arm gate invalidates that cell (rerun with a larger bound);
  - one telemetry view, with its own scrape pool;
  - one backend client;
  - one 4-thread pool for RouteBalance's batch calls (the estimator and the scheduling round), and a
    one-at-a-time lane for the baselines' per-request router calls;
  - one dispatch path, `Service._serve`: record, budget clamp, stream with progress, close.
- `RouteBalance.run` is the round loop:
  1. collect a round;
  2. estimate (pool);
  3. fetch every instance's load once (the per-batch fetch; the D54 control reads the pollers' cache instead),
     and adapt the round size;
  4. `schedule_round` (pool);
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
  - v3's arrival generator (digests match 73/73).
  - Prompt order.
  - A pooled client that validates every reply strictly.
  - The per-cell record, written once and never overwritten.
- `analysis/`:
  - per-cell metrics, with quality as the DeepEval score of (prompt, served model) in the scored corpus, zero-safe;
  - the cell manifest (expected cells accounted for);
  - frontiers, equal-quality deltas with a paired seed bootstrap, the SLO (TTFT and TPOT within k times the
    tier's single-request values, after SOLA), its attainment and goodput.
  - Re-aggregating the banked v3 cells reproduces v3's tables bit-exactly.
- `tools/` (about 3,000 lines):
  - `campaign.py`: renders the lease campaign (presets × scope) into per-cell configurations.
  - `cell_driver.py`: one cell's steps.
  - `engine_plan.py`: fleet × hosts × engine args → one vLLM launch per instance.
  - W1 calibration; the TPOT tables and the XGBoost retrain.
- `scripts/` (about 1,900 lines of shell):
  - `setup.sh`: one node, per role; resumable across the V100 driver reboot; ends with a self-check.
  - `deploy.sh` and the stop scripts.
  - `run_cell.sh`: drain, apply, preflight, warm up, read the engines (`tools/engines.py`), run, read them
    again, gates, publish, mirror. `run_scope.sh` runs a
    scope's cells.
- `configs/`: the fleets, engine args, presets, the lease scope and the pre-registered rules. Owner values
  are `OWNER-MUST-SET` until set, and a render refuses them.

The campaign gates (`gates/`) and the test suite (`tests/`) are kept on the maintainers' local branch and are
not part of this snapshot. They are what `make check` runs before every commit: unit, contract (a fake vLLM),
the v3 equivalence suite, and gates with MUST-FAIL fixtures.
