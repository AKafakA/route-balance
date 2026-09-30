# vllm-sr-bench: vLLM Semantic Router in the RouteBalance main grid

vLLM-SR runs outside our scheduler, as its authors deploy it: Envoy with vLLM-SR's router behind an ext_proc filter, in
front of the same engines. This module drives it with the same client inputs as every other system and records the
same evidence, so its runs compare apple to apple with the main grid (owner 09-30).

It is separate by design. It imports the shared pieces (`bench/` prompts, arrivals, sampling and records; `tools/`
engine checks) and never modifies them. Nothing under `routebalance/`, `bench/`, `gates/`, `tools/` or `configs/`
changes for it; under `scripts/`, only the node setup gains its role (`setup.sh --role vsr`: Docker, its data root and
classic store on /mydata as v2 ran it (D168), started but never enabled at boot (D165), the CLI's venv, the pinned
images, the classifier weights) and `node_layout.sh` its venv path (`VSR_VENV`) and Docker's data root
(`DOCKER_ROOT`). Nothing here runs from the lease
tree a running grid uses.

## Decisions it implements

Decision log D146, D150, and the owner's 09-30 answers:

**Where it runs:** in the main grid, for Qwen and Gemma, at the six rates (12, 18, 24, 30, 36 and 42 req/s), seed 1 (D166: every row's seeds 2-3 are revised after every
seed-1 row and the paper draft; the block runs any seeds it is given).

**Deployment:**
- Fresh, from upstream v0.4.0 (tag `v0.4.0`, commit `6573123`).
- The images are pinned by digest: the router at the CLI's tag, and Envoy at `envoyproxy/envoy:v1.35.3`.
- Served by `vllm-sr serve --minimal`.

**Policy:** its built-in Mixture-of-Models Balance policy as shipped, in one configuration. Pool assignment by size, as
neighbouring pairs:

| Family | simple | medium | reasoning |
|---|---|---|---|
| Qwen | 3B, 7B | 7B, 14B | 14B, 72B |
| Gemma | 1B, 4B | 4B, 12B | 12B, 27B |

**Quality evidence (owner 09-30: "yes" to our DeepEval score):**
- Each model's mean judged score on the training split (`deepeval-llama3.1-8b-it_reference`), the labels
  RouteBalance's estimator and the other baselines train on (training-split tuning is allowed, as for BEST-Route).
- Supplied through v0.4.0's documented custom evaluation: an operator benchmark, per-model records, an operator index.
- Each pool's `quality.index` points to it: the one edit to the shipped policy, made on upstream's generated text and
  checked to change nothing else (`config.patch_indices`).

**Prices:** the fleet's input and output prices, as for every system. **Replicas:** every fleet instance of a model is
one of its backends, at the experiment-network address our scheduler uses; Envoy picks among them by round robin
(upstream's default). **Cards:** each model's context and output limit are its engine's `--max-model-len`.

**The owner's values (D146, D150), in a copy of upstream's Envoy template (`templates/`, diff in `templates.diff`),
passed as the CLI's `TEMPLATE_ROOT`; its digest is recorded with every run:**
- the listener timeout (a config field) is the client's 6 h, vLLM bench's stock timeout, as for every system;
- the ext_proc `message_timeout` is 1 h, Envoy's maximum (owner: "1 hour is fine");
- the circuit breakers are v2's 200,000 on the router's cluster and every model cluster (stock: Envoy's defaults,
  1,024; owner: "make the large value, to avoid invalid").

**Stop (D151, owner: "a then b and stated in the appendix"):** (a) `stop: ["<|im_end|>"]` was probed
(`probes/stop_token_probe.py`) and does not stop as `stop_token_ids` [151645] does (D102): vLLM's stop strings do not
match special tokens, and v0.4.0 refuses `stop_token_ids`. So (b): the Qwen requests carry no stop, and quality is
judged on the served text.

**Requests:** only the fields v0.4.0 accepts: the same prompt as the single user message, temperature,
repetition and frequency penalties, max tokens, streaming with the final usage chunk. Probed: the router passes them
to vLLM unchanged (the same prompt tokens; replies identical to direct requests to the engine).

**Failures (D160, owner "yes"):** as every system (D134): a run with any failed request is invalid and is rerun.

**Order (D152, D166):** after every RouteBalance-framework row of a family, vLLM-SR's block (seed 1 at the six rates),
then the engine switch. Docker runs on the node only during the block (D163, D165).

## Per run (`python -m vsrbench.run_cell`)

Each run is fresh and isolated, checked by execution (owner 09-30, the P100 rule: a check reads the live system):
0. our scheduler is not running (its port closed, no process of it); no vLLM-SR container or storage volume is left; the
   template tree is the committed one (its digest pinned); a lease run takes only the pinned training rows (never a scored
   evaluation corpus) and starts in the main grid client's clean environment (HOME, PATH, LANG, HF_HOME,
   HF_HUB_OFFLINE=1); the classifier weights' model files are byte for byte `models.sha256`, before serve and after
   startup;
1. the configuration: upstream's generator from the fleet and the means, the index edit, upstream's validator;
2. drain the engines; a new stack (`vllm-sr serve`: new containers, empty volumes created by this run, a new CLI state
   directory); the engines as they report themselves; the live checks, read from the running stack:
   - the containers are upstream's four, on the pinned image digests, none given a device (the CLI's stack runs on CPU),
     none carrying a host variable the CLI forwards (the CLI and the generator run under a clean environment);
   - the router's active configuration has converged, and every setting of the generated configuration is in it with the
     same value (every list the same length): models, prices, replicas, cards, the evaluation records, each named pool's
     models, its orders and margins, the globals; each card's context is its engines' reported context;
   - the stored state is empty: Postgres holds only the replay table, with 0 rows, and Redis no key;
   - Envoy's clusters hold exactly the fleet's replicas, round robin, with the timeouts and breakers above;
3. the warm-up with the proven prefix-cache reset, then drain (as every run: `tools/direct.py`);
4. the run: `bench.runner.run_cell` with this module's streaming client (`vsrbench/client.py`), the same prompts,
   order, arrivals, sampling and bench parameters as the main-grid run of that rate and seed; the raw record is written
   the moment it returns;
5. the engines again; Envoy's access log flushed; the replay store holds exactly the run's requests; the live checks
   again; the evidence copied out (every container's log, stdout and stderr; the access log; the CLI's rendered state);
   the classifier weights' listing unchanged since startup; the stack torn down (a teardown failure is a note). A failure
   here is a problem of the run, never its loss: a run that served is always published;
6. the join (each reply's replica from Envoy's access log; the router's own timings from its log, for every reply or
   none), the record in the main grid's format, the main grid's checks (`analysis.manifest.cell_problems`, the reset
   proof, the traffic check, the engines unchanged), the engines launched as for the paired main-grid run (a lease run's
   `--reference-cells`), `gates/verdict.json`; publish; mirror. `vsr/harness.json` records the code that ran.

**TTFT (D161, owner: option b, "to match rb"):** the SLO's TTFT (`first_token_s`) runs from Envoy's own request start
(vLLM-SR's front door, `%START_TIME%` in its access log, millisecond resolution) to the first streamed message with
choices reaching the client (timed as `routebalance/backend.py` times it), and the client connects to Envoy's container
address directly, not through docker-proxy: the same start as every system on our scheduler, whose TTFT runs from the
request's arrival at our scheduler's handler (its front door, reached directly on the node) to the scheduler's receipt of
the backend's first chunk (D57). Both start after the client-to-system hop and include everything the system does before
the first token (for vLLM-SR, its routing and any wait in the router's queue); a wait before either front door is
outside both. They end one hop apart: ours at the scheduler, vLLM-SR's at the client, after Envoy forwards the first
chunk over the stack's network (Envoy's stock access log has no first-token time). `ttft_s` and `server_latency_s` stay
on the client's clock (send to first token, send to end), so the decode time is the client's. Recorded beside it
(`vsr/requests.jsonl`, `vsr/client_loop_lag.json`): the client-side send-to-first-token time, per request the stream's
end at the client after Envoy's own end (`end_after_envoy_s`, a measure of that last hop and the client's loop; Envoy's
start and duration are both at millisecond resolution, so it reads about 1 ms high on average: recorded, not the hop
itself), and the client's event-loop lag (the scheduler's probe).

**For the analysis:** in vLLM-SR's records `ttft_s` and `server_latency_s` run from the client's send, so the main
grid's off-instance split (`off_instance_ms`, end to end minus `server_latency_s`) reads about 0 by construction and does
not apply; `first_token_s` (D161) and TPOT are comparable. The router's timings are its own (signals, decision,
routing), not our scheduler's keys. Before comparing SLO attainment at 36-42 req/s, set the client's loop-lag p99
(`vsr/client_loop_lag.json`) beside the TTFT limit: vLLM-SR's first token is stamped on that loop.

**Images:** the CLI pulls the router's and Envoy's images by tag at every `vllm-sr serve` (upstream's stock pull policy,
"always"), so every run needs the registries; what runs is pinned by digest through the live check
(`live.container_problems`), which fails a run whose containers are not the pinned digests.

**CPU only (D162):** the stack runs as `vllm-sr serve` does by default, its router's classifiers on the node's CPU, as
every router of our framework runs on that CPU; the stock `--platform nvidia` (a dedicated GPU) is not used. Every run
checks no container is given a device.

## Profiles

- **lease:** the fleet file, the experiment network, the training means; a configuration carrying any test token is
  refused, before the stack starts and again in the live configuration.
- **test (dev-gpu only):** `tests/e2e/` (a two-replica fleet of one small model, or with `--test-pair` two models in the
  pools, behind Docker's bridge gateway; the engines launched by `tests/e2e/engines.sh`, the hosts file written by
  `python -m vsrbench.testprofile hosts`), marked `routebalance-test-profile`, its cells named `test.*`, never under the
  lease's data disk, a short trace (`--test-trace`).
  The fleet file labels the dev-gpu node an A30 because the shared fleet loader knows only the lease's hardware; the
  label only names its tier.

## Layout

| Path | What it is |
|---|---|
| `vsrbench/` | the stack lifecycle, the streaming client, the run driver, the record writer, the checks |
| `probes/` | one-off probes (the stop token) |
| `templates/`, `templates.diff` | upstream's CLI templates with the owner's Envoy values, and the diff |
| `tests/` | unit tests (`PYTHONPATH=..:. pytest tests`); `tests/e2e/`: the dev-gpu test profile |
