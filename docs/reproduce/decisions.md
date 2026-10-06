# The rulings the code refers to

Comments in this tree explain why a thing is the way it is, and several of those reasons are decisions taken once
and then binding. This page restates them, so a reader does not need the development history.

## The comparison

**Every baseline runs in its stock, published form.** BEST-Route as its authors ship it (their
ranker and routing rule from a pinned checkout), AvengersPro's upstream cluster router at a pinned commit,
LiteLLM's lowest-latency rule, and the least-outstanding and power-of-two dispatchers on the engines' own load
gauges. Each runs in the same server as RouteBalance, on the same fleet, the same telemetry, the same HTTP session to the
engines and the same dispatch path. The vLLM Semantic Router runs as its authors deploy it, its own stack in front
of the same engines, driven by the same client. Counting in-flight requests, as NGINX's `least_conn` and Envoy's `LEAST_REQUEST` do, is an
appendix variant.

**RouteBalance's own mechanisms are never given to a baseline.** Scheduling a round of requests together, dead
reckoning (the scheduler's own in-flight count as each instance's load) and the per-round load read belong to
RouteBalance. A baseline places each request as it arrives, and the configuration validator refuses dead reckoning
and the per-round read for it.

**The routers' model calls run as many at once as RouteBalance's largest round** (32) in the main comparison, and
one at a time, as shipped, in the appendix.

**RB-decoupled and RB-single are RouteBalance's own variants.** RB-decoupled uses RouteBalance's predictors and
objective but fixes the model on idle-instance latency before load is visible. RB-single is RouteBalance with one
request per round. Both are compared with RouteBalance pair by pair, at the same weights.

## The serving path

**No value of our own.** Every setting is the upstream system's stock default (vLLM, vLLM's benchmark client,
uvicorn, Envoy), the previous implementation's accepted value, or a value the authors set, and
`configs/provenance.json` records the source of each, which `tests/unit/test_provenance.py` enforces.

**No rule of our own refuses, drops, delays or times out a request.** A stale load snapshot, a late send or a slow
reply is counted and logged, never gated. The benchmark client and the scheduler's sessions use vLLM's benchmark
connector as shipped: no connection limit, keep-alive.

**Every run starts clean.** A fresh scheduler process, every engine's prefix cache emptied by the warm-up and the
emptying proven, the engines' serving counters changed by the run's own requests only, and the same engine
processes before and after. The gates check each of these on what executed (`gates.md`).

**Only the run's own requests reach the engines.** The engines and the scheduler listen on every interface (vLLM's
default, and the launch file's host). Every run's traffic check compares each engine's serving counters with the
run's own requests, and a run that outside requests reached is invalid and rerun: three runs of the campaign were.
From 3 October a firewall rule on every node admitted the two ports only from the experiment network.

## The measurements

**Quality** is the judged score of each answer: the DeepEval score of (prompt, served model) in the scored corpus,
as in the previous implementations. The budget runs, whose replies a budget can cut short, are judged on the text
actually served, and an empty reply scores 0.

**Cost** is the prompt tokens at the model's input price plus the output tokens at its output price.

**The comparison at equal quality** reads RouteBalance's quality-latency frontier at each baseline configuration's
quality, within the tolerance `configs/prereg_rules.json` sets, and reports the range over the seeds and a request
bootstrap within the paired seeds (`analysis/equal_quality.py`).

**The SLO** follows SOLA: a request meets it when its TTFT and TPOT are within k times its tier's single-request
values, with k by model size in `configs/prereg_rules.json`.

**Arrivals are open loop.** The offered rate is the control variable. Arrivals and prompt order come from the
trace and the seed, and the verdict regenerates them to check the record.

## The configuration notes

The free-text notes in `configs/` (`about`, `ref` and similar fields) cite the development decision log by id (D16,
D225 and so on). The log itself is not part of this release, and each note states what was decided. The files are
the ones the runs used, with two exceptions that differ in notes only, every value unchanged:

| file | sha256 as run | sha256 as shipped |
|---|---|---|
| `configs/scope.json` | `8b77ccab9e40bb97d36a4c460bee9a1c39f830c736eddb429adb6b32a6d87583` | `5db088ae7040229ad8beb8c9b1fb46e42fc3c1e873cfa04ee55418423442b7aa` |
| `configs/provenance.json` | `d696d4c99b59a0d87c2c79ec258df91cd7d0bfec5588a3175e7d4444de6731dd` | `1a5a4c56703ba01ba89bcdda961f49ee2280049caec0df7f21417b16bf079f79` |

The rendered campaign records the first column (`render.json`, `scope_sha256`).
