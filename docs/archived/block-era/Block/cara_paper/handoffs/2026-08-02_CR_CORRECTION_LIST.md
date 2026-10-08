# CR CORRECTION LIST — text fixes owed at camera-ready

**Created 2026-08-02.** CR items were scattered across at least four handoffs
(`2026-07-25_EXECUTION_HANDBOOK.md`, `2026-07-25_window_execution_log.md`, and others), which
is how they get lost between a rebuttal and a camera-ready. This is the single list.

**Scope rule.** An item belongs here only if the fix is **text**. Anything requiring a code
change or a rerun is *not* a CR item and must not be parked here — see §3.

---

## 1. NEW — 2026-08-02: the analytical E2E latency description

**Author decision: the CODE IS CORRECT. Fix the paper and the rebuttal at CR.**

Found by the Codex claim-verification review and confirmed by reading the code directly.

### 1a. The divisor

| source | says |
|---|---|
| **code** (`cara_serve.py:1268`) | `queue_iters = pending_dec / max(running_n, 1)` — i.e. **`num_running`** |
| **paper** `neurips_2026.tex:207` | "pending decode tokens divided by **its current decode batch size**" |
| **rebuttal** — Q1 "Latency prediction clarification" | "pending decode tokens divided by **the current decode batch size**" |
| **Jul-25 posted reply** (`2026-07-25_reviewer_actions_table_v2_and_drafts.md:23`) | "queue-aware decode iterations (**pending_decode/running** + own predicted output)" — **accurate** |

`num_running` is **not** the decode batch size: the telemetry schema exposes
`num_active_decode_seqs` separately (`data_structures.py:97`), and both appear in the TPOT
head's own feature list (`xgboost_predictor.py:19-37`) — so the system draws the distinction
elsewhere, which makes the loose wording harder to defend as shorthand.

**Why the code is marked correct rather than changed:**
* It is self-consistent and produced **every published number** — v1, the submitted tables, the
  posted rebuttal, and the −6.7 % burst row. Changing the divisor changes routing and would make
  new cells non-comparable with all of it.
* `num_running` is a defensible proxy for the concurrency competing for decode slots. It is an
  approximation; the defect is describing an approximation in language implying precision.
* Our own Jul-25 posted reply already described it accurately. The drift entered when the
  phrasing was tightened between July and the Q1 text.

**Fix, both places:** "the current decode batch size" → **"its number of running sequences"**.

### 1b. The free-slot branch, missing from the rebuttal only

The code has it (`:1263-1264`, `if slot_open: sidecar_e2e = own_out * tpot`) and **the paper
already documents it** (`:207`: *"If the instance has a free decode slot, only the second term
applies."*). The rebuttal's Q1 clarification paragraph **dropped that clause**, so as written it
claims the queue term applies unconditionally.

**Fix, rebuttal only:** restore a clause equivalent to the paper's.

**Suggested corrected sentence (rebuttal):**

> End-to-end latency is then computed analytically as predicted TPOT multiplied by the request's
> estimated decode iterations: the instance's pending decode tokens divided by **its number of
> running sequences**, plus the request's own KNN-predicted output length. **If the instance has
> a free decode slot and nothing queued, only the second term applies.**

### 1c. Verified clean in the same paragraph — no action

* "Each XGBoost head predicts TPOT only, one head per (model, GPU) tier" — booster set confirms
  `qwen2.5-{3b_a30,7b_a30,14b_v100,72b_a100}.xgb`.
* **"it does not consume request text"** — holds. Full feature list is 17 instance-state signals,
  2 request features (`num_prompt_tokens`, `num_predicted_output_tokens`) and 6 derived. No text,
  no embedding, only token counts.
* "the request's own KNN-predicted output length" — `_expected_length` computes
  `E[output_tokens]` from the KNN's bucket probabilities (`model_estimator.py:428`).

---

## 2. PRE-EXISTING items, transcribed from scattered handoffs

⚠ **Transcribed, not re-verified.** Check each against its source before acting.

| item | source |
|---|---|
| The paper's KNN description "k=50 / 14,919" is a CR correction item; v1 actually served a baked k=10 index of 14,966 examples | `2026-07-25_EXECUTION_HANDBOOK.md:49`; see also the `v1-knn-true-artifact` finding |
| The "14B-preferred by design" relabel is FORECLOSED for v1 — CR correction item, rebuttal never repeats it | `2026-07-25_window_execution_log.md:255` |
| Disclosure decision: silent-in-rebuttal + **corrected-at-CR with rails**; never re-assert the two basis-dependent claims; truthful if asked; CR gets an explicit change note | `2026-07-25_EXECUTION_HANDBOOK.md:177` |
| Text-only CR fix: mechanism description/caption and the `:241` narrative sentence | `2026-07-25_window_execution_log.md:272` |

---

## 3. Live code issues from the same review — ALL THREE FIXED (verified 2026-08-02)

**These were listed here as OPEN. That was stale: the author's instruction "it should raise
exceptions for all" was carried out earlier the same day, and this section was never updated.**
It was then quoted back to the author as outstanding work — the exact failure mode the campaign
guards against, a document describing code that no longer exists. Verified against the source,
not against this file:

| # | issue | fix, with proof |
|---|---|---|
| 3a | telemetry failure swallowed — `_get_instance_stats` returned `{}` on any timeout / non-200 / malformed body / exception, so a blip silently made the *queue-conditioned* learned arm state-blind while the cell banked `OK` | **RAISES.** `InstanceStateUnavailable` (`cara_serve.py:926`) raised at `:956`, `:973`, `:979`, `:981`. **No `return {}` remains** on that path, and the masking sites across RB scoring, `shortest_queue`, both size-grouped variants and length-based routing were removed with it |
| 3b | `dominance.py` turned missing/null `mean_e2el_ms`, `p99_e2el_ms` and throughput into `0.0`, so a partial result could show zero latency and **falsely satisfy strict dominance** | **RAISES.** `_need_metric` (`:135–149`) raises `MetricUnavailable` on missing, non-numeric, NaN and negative; surfaced as a refusal (rc=2), not a traceback. Comment at `:227`: *"NO `or 0.0` ON A COMPARISON AXIS"* |
| 3c | tied baseline optima made the verdict order-dependent — `max(cand, key=q*thru)` took the first co-maximizer, and reversing input order flipped DOMINATED to NOT DOMINATED | **FIXED.** *"ALL co-maximizers, not the first one `max()` happens to return"* (`:412`), with a NaN guard added after the first fix introduced an `IndexError` that the self-test caught |

**Verified by execution, not by reading:** the gate self-test carries MUST-FAIL cases for each —
a gate that cannot fail is not a gate. Current: **149 cases, 81 MUST-FAIL, 0 wrong.**

### 3d. Monitor findings now carry argv (author-approved 2026-08-02)

Added after the contamination gate's first real firing reported only `2 runners, ages [27, 240]`,
refused four cells, and could not be diagnosed because the processes were gone before anyone
looked. Counts and ages cannot separate a genuinely leaked runner from a process that merely
matches the pattern. `DUPLICATE_RUNNER` / `_SCHEDULER` / `_BENCH` findings now carry pid, age and
argv in both `detail` and a structured `procs` field. Self-test pins it: a duplicate finding must
name its pids and argv. **A finding that stops work has to be diagnosable from itself.**

## 4. NEW — 2026-08-04: scheduler-head node / CPU co-location — faithful disclosure + robustness check owed

**Finding (verified from cara_serve logs + cell overhead fields):** the scheduler ran **co-located
on node0 = `c4130-110433`, a V100 (16 cores)**, which *also* ran the `Qwen-2.5-14B_0` serving
instance. Per-request scheduling overhead is system-dependent and non-trivial (λ12): AvengersPro
~55–69 ms, RouteBalance ~161–188 ms, BEST-Route+rr ~180–199 ms, **BEST-Route+po2/sq ~452–498 ms**
(its per-request instance-stats fetch). The BR-po2/sq path is also the one that hit the fail-loud
collapse at λ≥18.

**Risk:** on a CPU-limited, co-located head, BR-po2/sq's CPU-heavy path could be inflated, so the
**"BEST-Route collapses at λ≥18" sub-claim could be partly a scheduler-CPU artifact** rather than
pure BR design. **NOT at risk:** the core RB-vs-AvengersPro frontier — AP has the *lowest* overhead
and RB carries more yet still wins latency, so more scheduler CPU only *helps* RB (the current
setup handicaps RB, it does not flatter it).

**Disclosure (do this regardless):** state faithfully in the paper — *"the scheduler ran on node0,
a V100 provisioned by CloudLab."* Full transparency removes any hidden-confound concern by
construction.

**Action (robustness check):** on a reproduction lease, re-run the vulnerable comparison
(**BR-po2/sq + one RB point + one AP point**, at λ12/18/24, a few seeds on the collapse cells)
with the scheduler **co-located on an A100 (d8545, 48 cores)** instead of the V100 head — same
fleet, only the scheduler node changes; identical for all systems.
  - **If the conclusion holds** (BR still collapses, verdicts unchanged) → report it as a one-line
    robustness ablation; optionally make v4 all-A100 and frame as "scheduler on the best-provisioned
    node." No inconsistency introduced.
  - **If it changes** (BR stops collapsing) → do NOT keep the V100 numbers for consistency; correct
    the BR/collapse claim at CR. The RB-vs-AP frontier is expected to survive either way.

**Rails:** faithful disclosure of the actual node; the A100 run is a separate labeled robustness
line, never substituted into the V100 numbers; no result-flattering; report whichever is true.
