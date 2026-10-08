# Locked defaults — full sweep of gates, silent defaults and dead keys

**Purpose.** Every correct v3 setting was reachable only by an env var or an explicit config key.
That is the same shape as the `quality_norm` incident (config key at the wrong nesting level →
silently served `raw`) and the W1-B incident (ran on `raw` beside minmax tables). A setting that
is correct-but-opt-in is a defect waiting for the next forgotten export.

**Rule applied here.** The working setting becomes the default. Where a wrong value cannot be
detected from the output, disagreement is made **fatal** rather than defaulted. Opt-outs survive
only to reproduce the pre-fix basis for control arms.

Nothing in this document is inferred. Every row names the file:line that proves it.

> ⚠ **Line numbers re-verified 2026-08-01 and many had DRIFTED by 100-250 lines** (e.g.
> `quality_norm` 1718→1826, `lpt_sort_key` 1876→2006, `CARA_MODELS_DIR` 3101→3303,
> `_validate_quality_norm` 2766→2959, `CARA_DUMMY_ESTIMATOR` 1878→2008). Treat every
> `:NNN` below as approximate and grep for the symbol; the promise above did not hold.

---

## A. Env gates — `block/global_scheduler/cara/`, `block/predictor/cara/`

| gate | was | now | why |
|---|---|---|---|
| `CARA_UNWRAP_PROMPT` | opt-in (`!= "1"` → off) | 🔴 **VARIABLE NO LONGER EXISTS — unconditional** | `cara_serve.py:107-111`: *"No switch. …there is no legitimate configuration in which it should not."* `grep -c CARA_UNWRAP_PROMPT block/` = **0**. The wrapped path retrieved chat-templated prompts against a raw-prompt index (agreement `0.000000`) — a defect, so the fix was made unconditional rather than defaulted. **Exporting this variable does nothing.** |
| `CARA_COST_DECODE_ONLY` | opt-in | 🔴 **VARIABLE NO LONGER EXISTS — unconditional** | `cara_serve.py:2944`: `"cost_decode_only": True,  # unconditional; no switch exists`. `grep -c` = **0**. The paper defines cost decode-only (`neurips_2026.tex:154`, reported at `:221`); the code billed input+output. **Exporting this variable does nothing**; use `dominance.py --cost-basis full` to see the other basis. |
| `CARA_INPROC_PREDICTOR` | opt-in, 4 sites | **default ON**, `=0` opts out | `cara_serve.py:1164, 2019, 2225, 3293`. Unset, `sidecar_pred` stays `None` and control falls to `elif predictor:` (`:1212`) — per-call `predict_ttft`/`predict_tpot` instead of the batched in-process path. **The latency mechanism changes and nothing in the cell output records which one ran.** |
| `CARA_MODELS_DIR` | `'/users/asdwb/Block/models/cara'` hardcoded | **derived from the running tree** (`Path(__file__).parents[3]`) | `cara_serve.py:3101`. v3 cells run from `/users/asdwb/Block_v3`, so the predictor audit inspected a *different tree's* artifacts than the ones being served. |
| `CARA_STATIC_TPOT_HEAD` / `CARA_STATIC_LAT` | opt-in env var | 🔴🔴 **NEITHER IS READ ANYWHERE. Arm 4 is CONFIG-driven.** | `cara_serve.py:1709-1719`: *"Arm-4 STATIC latency prior. CONFIG-DRIVEN, not env-driven (2026-07-31). It was an environment variable until today, and that cost a campaign."* Both names survive **only in comments** — measured: 0 `os.environ` reads. Set arm 4 with `slo_defaults.static_tpot_ms_per_tok` (POST `/v1/config`), and verify it with `GET /v1/config` + `tpot_gate.py`. |

> 🚨 **The row above was the single most dangerous line in this handoff set.** It told a
> reader to configure arm 4 by exporting `CARA_STATIC_TPOT_HEAD`. There is no reader for
> that variable, so the export succeeds silently, the scheduler keeps using the learned
> head, and the cells are banked wearing a `static` label. That is precisely the failure
> that invalidated v2, July **and** v3 — three campaigns — and the document that exists to
> prevent it was prescribing it. Corrected 2026-08-01.

| `CARA_DUMMY_SIDECAR` / `CARA_DUMMY_ESTIMATOR` | opt-in | **unchanged, correctly opt-in** | `cara_serve.py:1157, 1878`. Bottleneck-isolation harnesses. Must stay off by default. |

**Also fixed — an audit that passed when it could not audit.** `PREDICTOR_AUDIT` warned and
`continue`d on a missing `fused_config.json`. It now raises when an `EXPECTED_*_ENCODER` hash was
declared: if you assert an expectation, a missing artifact is a failure, not a skip.

---

## B. `k` — the dead key that must NOT be made live

`DefaultModelEstimator` documents `k` in `quality_config` (`model_estimator.py:92`) but never
assigned `self._knn.k` after `KNNEstimator.load`. It served whatever the pickle baked.

**The obvious fix is wrong, and the live configs prove it.** Every Phase-1 config declares
**`k=10`** while pointing at the **`deploy_v3_k50_14919`** artifact:

```
cfg_de_full.json    score_type=deepeval  k=10  dir=deploy_v3_k50_14919
cfg_ju_full.json    score_type=judge     k=10  dir=deploy_v3_k50_14919
cfg_abl_k50.json    score_type=deepeval  k=10  dir=deploy_v3_k50_14919
```

Those cells are k=50 **because the baked value ran and the config key was inert**. Making the
config authoritative would silently reinterpret every one of them as k=10 on the next rerun —
converting a dead key into a live regression, in the exact direction that would look like a
successful reproduction.

**Resolution (author-directed: k=50 always).** `k` defaults to **50** when the config is silent,
and a config/artifact disagreement is **fatal**, naming both values, with an explicit
`k_mismatch_ack` escape. So k=50 is what runs, and no config can silently drop to k=10.

**OWED (author decision):** the v3 configs' `k: 10` must be corrected to match the artifact, or
removed. Until then they will refuse to start under the new check. Requires reading the baked k
out of the artifact — deferred while cells are live to avoid side traffic on the cluster hosts.

> ✅ **NARROWED 2026-08-02 by execution — this does NOT apply to the S2 path.** Checked directly:
> * The config S2 actually serves is `block/config/cara/s2/scheduler_config_s2.json`
>   (`lib_serve_v3.sh:27`, `V3_CFG`), and it declares **`k=50` with
>   `model_dir=models/cara/knn/deploy_v3_k50_14919`** — consistent, no mismatch. It is
>   *generated* at P0_CONFIG, where `s2_driver.sh:286-287` sets `qc["k"]=50` and the k50
>   tree-relative dir, so it cannot drift.
> * `cell_audit_s2.sh` then asserts the live config's `k` equals `EXPECT_K`, which the driver
>   records from the actually-staged config (`s2_driver.sh:583`) — a generated pin, not a
>   hardcoded one.
> * The three files named above (`cfg_de_full.json`, `cfg_ju_full.json`, `cfg_abl_k50.json`)
>   are **not in this tree at all** — they were era artifacts on the cluster.
> * ⚠ **One correction to this section's own framing.** `v3_configs/scheduler_config_v3.json`
>   declares `k: 10` pointing at **`deploy_v3_k10_14919`** — a *k10* artifact, so it is
>   internally consistent, not the "k=10 against the k50 artifact" shape described above. (It
>   also carries an absolute `/mydata/...` path, which the ONE-TREE rule has since replaced with
>   tree-relative paths.) It is a v3-era file that the S2 path never loads.
>
> **Net: the OWED item is real for the era configs and moot for Monday.** No action is required
> before the window opens.

---

## C. Silent defaults in the routing path — verified against the live configs

| key | code default | every live v3 config declares | verdict |
|---|---|---|---|
| `quality_norm` | `minmax` (`cara_serve.py:1718`) | `minmax`, or absent | **already locked** — `_validate_quality_norm` (`:2766`) makes top-level placement, unknown values, and unacknowledged `raw` all fatal |
| `filter_enabled` | `True` (`:1153, 1492`) | **`False` in all 10 configs** | ⚠ **default disagrees with every real config.** A config omitting the key silently enables admission filtering — a different system. Flagged, not flipped: this is an experiment parameter. |
| `lpt_sort_key` | `max` (`:1876, 1979, 2100`) | `max` | consistent |
| `assignment_strategy` | `scoring` (`:1368`) | absent → default applies | consistent |
| `budget_tokens` | `256` (`:434, 647`) | absent → default applies | consistent |
| `score_type` | `judge` (`model_estimator.py:95, 958`) | `deepeval` for Qwen, `judge` for Gemma | ⚠ **default is the D1 defect value.** Flagged, not flipped: Gemma legitimately needs `judge` (its train corpus carries only the Qwen judge), so no single default is right. |

The two ⚠ rows are **reported, not changed** — flipping either alters experiment semantics, which
is the author's call, not mine.

---

## D. What this does not cover

Aggregators (`aggregate_*.py`, and the public `reproduce/aggregate_results.py` `or`-chain that
drops zero scores) are a separate sweep. They affect reported numbers, not served behaviour.

---

## E. Found during the sweep — a surviving copy of the original skew

`cara_serve.py:2131` — `elif model_estimator is not None:` , commented "Fallback: per-request
estimation" — called `model_estimator.estimate(req.prompt_text, ...)` **without**
`_estimator_prompt()`. Every other serving call site unwraps (`:1906`, `:2752`); `:654` is not a
serving path.

So whenever the batched prep path was not taken, the entire batch was scored against the
raw-prompt index using chat-wrapped prompts — the original defect, alive in a branch that the
first fix pass did not reach. **Fixed.**

This is the argument for locking defaults rather than setting env vars per runner: the env var was
correctly exported for every cell, and the skew still survived, because the flag was never the
thing that guaranteed correctness.

## F. CORRECTED — `estimate_batch` is live, and the fallback is reachable

An earlier draft of this document claimed `estimate_batch` was dead in serving. **That was wrong.**
It is invoked by reference — `run_in_executor(None, model_estimator.estimate_batch, prompts,
budget)` at `cara_serve.py:1893` and `:2114` — which a grep for `estimate_batch(` does not match.

The real structure, and the real risk:

```
if model_estimator is not None and hasattr(model_estimator, 'estimate_batch'):
    prompts = [_estimator_prompt(req.prompt_text) for req in batch]   # unwrapped
    ... estimate_batch ...
elif model_estimator is not None:
    ... estimate(req.prompt_text) ...                                 # was NOT unwrapped
```

`estimate_batch` is defined **only on `DefaultModelEstimator`** (`model_estimator.py:611`).
`KNNModelEstimator` (`:1049`) and `PFSModelEstimator` (`:928`) define only `estimate`. So the
branch taken is decided by `model_estimator.type`, and any `type="knn"` or `"pfs"` config lands in
the un-unwrapped fallback.

**Banked cells are unaffected — verified, not assumed.** All ten live configs declare
`type='default'`, so every cell took the batched, unwrapped path. That is luck, not design.

## G. The structural fix — unwrap at the index boundary

Call-site fixes do not compose: the first pass patched every `estimate()` site it found and still
missed `:2131`. Unwrapping is now enforced at the **only place a prompt becomes an embedding**:

* `knn_estimator.unwrap_chat_prompt()` — one canonical definition, in the module that owns the index.
* Applied at `KNNEstimator._find_neighbors` (per-request encode) and at the `estimate_batch` batch
  encode (`model_estimator.py`).
* **Idempotent** — text with no chat markers is returned stripped, so the existing call-site
  unwraps remain correct and harmless.

No caller, present or future, batched or per-request, any estimator subclass, can now reach the
index with a chat-templated prompt. Unit-tested: Qwen and Gemma templates, raw passthrough,
double-application, and empty input all resolve to the same raw text.
