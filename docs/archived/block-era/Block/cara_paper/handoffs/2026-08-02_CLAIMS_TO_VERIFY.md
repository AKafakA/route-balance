# CLAIMS TO VERIFY — what we have told reviewers the code does

> **Status 2026-08-02, first run:** 9 of 12 cleared. **B3 failed** and is resolved by author
> decision as a CR text fix (`2026-08-02_CR_CORRECTION_LIST.md` §1) — the code is marked
> correct. The **C5 findings are now FIXED, verified 2026-08-02** against the source (that
> file, §3): telemetry raises `InstanceStateUnavailable`, `dominance.py` raises
> `MetricUnavailable` instead of zero-filling a comparison axis, and tied optima take all
> co-maximizers. They were left marked OPEN here after the fixes landed and were quoted
> back to the author as outstanding — a stale doc describing code that no longer exists.

**Purpose.** This is the checklist for a claim-verification review. It is not a bug list and
not a code-quality checklist. The single question for every row is:

> **Does the executed code do what we have told reviewers it does?**

**Why this framing.** The most expensive defect in this campaign was found exactly this way: an
experimental arm was *declared* — a config key set, an env var exported, a payload posted — and
never *applied*, because nothing read it. It passed config diffs, routing-mix inspection and
RULE-0, and invalidated the comparison it was built to make. That shape is the target:

> a property asserted in prose, satisfied by a config key or a flag, and not realised in the
> executed path.

**How to answer a row.** Quote the code that implements it, as `file:line`. If nothing
implements it, say so. "It looks right" is not an answer; neither is "the config sets it".

**Provenance of each claim is given** so a disagreement can be settled against the source rather
than against this file.

---

## A. Q1 — the AC's priority-#1 question

*Source: the posted Q1 reply text (verbatim).*

| # | claim | what "verified" means |
|---|---|---|
| A1 | *"the same objective and the same within-tier queue tiebreak, with the learned head's queue-conditioned estimate replaced by a fixed per-tier TPOT constant"* | The within-tier queue tiebreak is genuinely retained in the static arm, and **only** the cross-tier latency term is replaced. |
| A2 | *"Both arms use RB's (0.5, 0.3, 0.2) weights and differ only in the TPOT estimator."* | Diff the two payloads as the runner builds them. Nothing else differs — not weights, not dispatch, not filter. |
| A3 | Constants **3B 10.7 / 7B 19.9 / 14B 16.5 / 72B 42.6 ms/token** | Each constant reaches the scorer **for its own tier**, matched by tier identity rather than first-substring. A hardcoded ladder `("72b","14b","7b","3b")` exists — confirm it cannot mis-assign, and name the model strings it silently fails to match. |
| A4 | *"per-tier medians across the whole sweep rather than at one operating point … 0.22 ms/token mean absolute error"* | The constants are medians over the sweep, not a single-λ fit. |

## B. Latency prediction

*Source: the posted "Latency prediction clarification" paragraph.*

| # | claim | what "verified" means |
|---|---|---|
| B1 | *"Each XGBoost head predicts TPOT only, with one head per (model, GPU) tier, trained from a tier-local QPS sweep."* | One head per tier; the head's output is TPOT, not E2E. |
| B2 | *"Its inputs are the tier's serving-state signals and the request's predicted output length; it does not consume request text."* | **No text or embedding feature reaches the head.** This is a falsifiable negative — check the feature vector construction. |
| B3 | *"End-to-end latency … pending decode tokens divided by the current decode batch size, plus the request's own KNN-predicted output length."* | ❌ **FAILED 2026-08-02, RESOLVED BY AUTHOR DECISION — text fix at CR, code marked correct.** Code divides by `num_running`, not the decode batch size (`cara_serve.py:1268`); and the rebuttal omits the free-slot branch that the paper documents (`neurips_2026.tex:207`). See `2026-08-02_CR_CORRECTION_LIST.md` §1. **Do not re-raise as a code defect.** |

## C. Paper-level claims

| # | claim | source | what "verified" means |
|---|---|---|---|
| C1 | Cost is **decode-only** | `neurips_2026.tex:154`, reported at `:221` | Decode-only everywhere cost is computed — serving path **and** aggregation. A second basis existing in code is fine; being *used* is not. |
| C2 | Quality is the deepeval score `deepeval-llama3.1-8b-it_reference`, via a KNN index | campaign constant | Report the **k the served path actually uses**, and whether it agrees with the config and with the artifact on disk. |
| C3 | RouteBalance performs **joint** model+instance selection; the baselines are router + dispatcher | design | RB has no separate dispatcher axis; the baselines do. |
| C4 | BEST-Route's fallback is the **smallest** model | v1 bug: it was the *first-listed* (14B) | Verify the current fallback, and that a `threshold > 0` payload cannot silently regress to first-listed. |
| C5 | Dominance is **strict on four axes** (quality, cost, mean, p99) against each baseline's **own best knob**, selected by `argmax(quality × throughput)` | plan Phase E′ rules 1–2 | The implementation matches, including tie handling and sign conventions. |

## D. Already known — do not spend effort rediscovering

State only if worse than described.

* `deploy_cara.py` hardcodes `~/Cara` for the monitor — telemetry only; the scheduler reads
  `/instance_stats` from vLLM:8000.
* `llm_judge_scorer.py:581,991` synthesise `is_harmful = [False] * n`, selecting the
  non-safety rubric.
* Era configs declare `k: 10` against a k50 artifact. The S2 config is generated with k=50 and
  asserted per cell, so this is scoped to non-S2 configs.
* **Po2 is a rebuttal addition**: the grid runs BR and AP on `{RR, SQ, Po2}` where the paper
  used `{RR, SQ}`, making the grid a **superset** of `tab:extremes`, not a reproduction.
* `(0.33,0.33,0.33)` sums to 0.99 and `cara_serve` does not normalise — measured
  routing-identical to exact uniform (0 ranking changes in 20,000 draws), so it is cosmetic.
* The 14B constant (16.5) is **faster** than 7B (19.9) because 14B is on V100 and 7B on A30 —
  measured, not a typo.

## E. Report format

Rank findings by whether a reviewer could catch them:

1. **Claims the code does not support** — the highest severity, because they are already posted.
2. **Claims the code supports only under conditions we have not stated.**
3. **Declared parameters that reach a config but not the executed path** — the shape that
   invalidated the previous arm comparison.

---

## Surface to cover

```
block/global_scheduler/cara/          serving, routing, scoring, /v1/config
block/predictor/cara/                 the TPOT heads and the estimator
block/benchmark/cara/                 the load generator and metrics
cara_paper/handoffs/window_setup_work/v3_scripts/   the sweep machinery
```

## Bases

| base | what a diff against it shows |
|---|---|
| `rebuttal-submitted-2026-07-29` | everything that changed since reviewers were given the numbers |
| `v1exact-submitted` | everything that changed since the tree that produced the submitted paper |
