# Full bug inventory and gate coverage — 2026-07-31

Every defect recorded across the project's handoffs, plus the fourteen found on 2026-07-31,
each mapped to **the gate that would catch it if it recurred**. A defect with no gate is
listed as `NONE` and is the honest answer to "could this happen again".

Sources swept: `2026-07-30_V3_DEFECT_INVENTORY.md` (D1–D28, P1–P4, T1–T6, G1–G4, U1–U2),
`2026-07-31_KNN_CHAT_TEMPLATE_SKEW.md`, `2026-07-31_LOCKED_DEFAULTS_AUDIT.md`,
`2026-07-30_v3_execution_log.md`, `2026-07-29_CHECKPOINT_v1_rebuttal_submitted.md`,
and the July 27–28 handoffs.

## Gates that exist

| gate | layer | what it proves |
|---|---|---|
| `preflight_gate.py` | runtime, 2 layers | declared == resolved, **and** resolved == observed behaviour |
| `static_defect_gate.py` | source | the four recurring bug **classes**, tree-wide |
| `gemma_gate14.py` | fleet | every backend up, across **both** config schemas, host:port pairs |
| `assert_effective_config.py` | runtime | asserts the scheduler's resolved state; missing line = FAIL |
| `verify_w1model_rows.py` | reporting | every table row reproduces from its source cell |
| `test_cost_calc.py` | arithmetic | cost conventions, known-answer + independent re-implementation |
| `cell_audit_v3.sh` / RULE-0 | tree | **currently vacuous — both manifests are empty. Must be re-recorded.** |

---

## A. Routing / scoring correctness

| id | defect | status | gate |
|---|---|---|---|
| D1 | routes on `score_type=judge` while every reported figure is deepeval | FIXED — deepeval is now the default | `preflight` L1 (`config score_type == resolved`, `resolved == deepeval`) |
| D2 | aggregator `or`-chain drops deepeval 0.0 — this **is** the coverage column | FIXED in `aggregate_v2_grid.py` | `static_defect_gate` CLASS 1 |
| G1 | `aggregate_overnight_grid.py` still carries the zero-drop | **OPEN** | `static_defect_gate` CLASS 1 |
| D14 | the **public** `reproduce/aggregate_results.py` carries it too | **OPEN — published artifact** | `static_defect_gate` CLASS 1 (scans `reproduce/`) |
| D27 | falsy-zero on EMA rates in `eval_latency_predictors.py` | **OPEN** | `static_defect_gate` CLASS 1 |
| N9 | `predict_by_judge_key` returns **0.0** on a missing judge key — silent all-zero routing | mitigated: deepeval path raises at load and per request | `preflight` L2 ("scores are not all zero") |
| N1 | **the chat-template skew survived the call-site fix** at `cara_serve.py:2131` | FIXED at the index boundary | `preflight` L2 **unwrap test** (wrapped and raw must score identically) |
| U1 | 25% label rung beats the full index (+0.001) | **UNEXPLAINED** | NONE — needs investigation, not a gate |
| N14 | Gemma KNN: 30.8% best-model accuracy vs a 25% floor, identical raw/wrapped | **UNEXPLAINED** | `preflight` L2 ("scores discriminate between models") would catch a fully dead index, not a weak one |

## B. Latency / predictor

| id | defect | status | gate |
|---|---|---|---|
| — | `tpot_lookup` in the scheduler config was **inert** — contained, never read | FIXED (`CARA_STATIC_TPOT_HEAD`) | `preflight` L2 static-head check, **both directions** |
| D3 | A30 tier first-match-P100; large measured effect | FIXED | `preflight` L2 + fleet composition |
| D4 | the zombie P100 tier is **deployed**, not merely configured | **OPEN** | NONE — see "gates still needed" #1 |
| D13 | `cara_serve` tier matching is a hardcoded Qwen ladder; `"3b"` never matches Gemma | partially fixed (family-gated) | `static_defect_gate` CLASS 2 (flags `cara_serve.py:1880`) |
| D26 | node→GPU map duplicated in two files, must stay in sync | **OPEN — confirmed still duplicated** (`cara_serve.py:2978, 3072`) | `static_defect_gate` CLASS 2; equality check still owed |
| D19 | FAISS KNN search never batched, in any version | FIXED and **live** (passed by reference, which my grep missed) | `preflight` L1 `batched_path == true` |
| D20/D21 | E2E formation, Eq. 1 semantics | verified — no defect | — |
| G4 | stale TTFT code to remove | **OPEN** | NONE |

## C. Cost / pricing

| id | defect | status | gate |
|---|---|---|---|
| D11 | rebuttal prices input+output; the submission prices output only | FIXED — decode-only is now default | `test_cost_calc.py`; `preflight` L1 `cost_decode_only` |
| D7 | AvengersPro cost vector unrecoverable | **OPEN** | NONE — see #4 |
| D8 | AP price-proxy trap: default cost term is **model size** | **OPEN** | NONE — see #4 |
| D9 | AP all-72B flood from upstream interface drift | FIXED (shim) | `preflight` L2 mix sanity / degenerate-mix halt |

## D. Configuration / defaults

| id | defect | status | gate |
|---|---|---|---|
| D10 | k: paper says 50/14,919; served artifact baked **k=10** over 14,966 | FIXED — k=50 default, **disagreement is fatal** | `preflight` L1 `config knn_k == resolved` |
| N12 | `k` was a **dead config key** in `DefaultModelEstimator` | FIXED | `static_defect_gate` CLASS 4 |
| N11 | `quality_norm` at config **top level** → silently `raw` | FIXED — fatal validator | `preflight` L1 (explicit top-level check) |
| D5/T6 | cap mismatch: corpus 1024 vs serving 256 | author set **cap 1024** | runner asserts `--max-output-len`; `preflight` records intent |
| D6 | param grids silently ≠ May | **OPEN** | NONE — see #3 |
| D17 | `batch_scoring` is the event-loop offload switch, set by nothing; **keep False** | intentional | `static_defect_gate` CLASS 4 flags it as dead — allowlist with this reason |
| D18 | our Po2 is a **predictive** variant, not standard Po2 | disclosed | NONE — see #5 |
| N10 | `EFFECTIVE_CONFIG` omitted `filter_enabled` — the gate could not check it | FIXED | self-test of the gate |
| — | `filter_enabled` default was `True` while all ten configs set `False` | FIXED — default now False | `preflight` L1 |

## E. Process / infrastructure

| id | defect | status | gate |
|---|---|---|---|
| P1 | scheduling-mode phase split; a bare `scoring_weights` POST raises in pipeline mode | known | NONE — see #6 |
| P2 | native-mode contract mandatory (`CUDA_VISIBLE_DEVICES=""`, inproc predictor) | known | `preflight` L1 `inproc_predictor` |
| P3 | admission filter is a **silent no-op** in pipeline mode | known | `preflight` L1 `filter_enabled` + mode |
| P4 | latency not comparable across sessions (5.9% swing at λ28 from timing alone) | known | NONE — label, don't gate |
| N3 | **my fleet gate read one config schema and passed 8/8 on a fleet missing 6 backends** | FIXED | `gemma_gate14.py` |
| N4 | inline multi-line ssh args **launched nothing** — no process, no log, no error | FIXED | launcher-file pattern + process count after launch |
| N5 | killing a runner locally left the **remote** runner alive; it killed the next scheduler | FIXED (kill remote by name first) | NONE — see #7 |
| N6 | **both RULE-0 manifests vacuous** (`Block` none, `Block_v3` zero lines) | **OPEN** | see #2 — the gate exists but gates nothing |
| N7 | two trees served one campaign, different binaries | recorded | see #2 |
| N8 | `experiment_output` excluded from tree archives — nearly lost the ground-truth table | FIXED (rescued, verified) | preservation count + a reproduce-a-number check |
| N13 | HF revisions captured from node0 only — half the ladder missing | FIXED — all hosts | preservation |
| D25 | `CELL_MAP` is blind to every Gemma cell (`*_n3534` glob only) | **OPEN — confirmed: map says 0, disk has 24** | see #8 |
| G2 | `llm_judge_scorer` silently defaults missing `is_harmful` to `[False]*n` | **OPEN** | `static_defect_gate` CLASS 3 |
| G3 | `p4_offline_bestacc.py` does not work | closed, labelled | — |

## F. Paper text (no cells needed)

T1 TTFT phrasing · T2 Vidur citation · T3 W2-B attributes Qwen figures to Gemma · T4 cap-filter
composition note · T5 Fig 1 stale (RoBERTa box unused, lists P100, omits 3B, "Instatnce" typo) ·
D23 split arithmetic does not close · D15 VOID (mis-anchored) · D12′ · D16 · D24 · U2.
**Gate: NONE possible** — these are prose. They belong on the camera-ready checklist.

---

## Gates still needed, in priority order

1. **No zombie tier in the live fleet (D4).** Assert the deployed instance list contains no
   model/GPU class outside the declared config — P100 was *deployed* while the config said
   otherwise. Cheap: compare `/v1/fleet` against `model_deployment.json`, fail on any extra.
2. **Manifest must be non-empty before it is trusted (N6).** RULE-0 currently passes against
   anything. Gate: `wc -l TREE_MANIFEST*.md5` > 0 **and** it covers every `block/**/*.py`.
   One tree per campaign; fail if two trees are referenced by the runners.
3. **Swept-grid assertion (D6).** Before a sweep, assert the set of parameter tuples the runner
   will emit equals the declared grid — catches "silently ≠ May" without waiting for the numbers.
4. **AvengersPro price provenance (D7, D8).** Assert the artifact carries explicit per-model
   prices and that they match `tab:heterogeneous_cluster`; fail if the size-proxy default was used.
5. **Dispatcher identity (D18).** Assert the live dispatcher type string matches the one the
   paper names, so "Po2" in a table always means the same code path.
6. **Mode/cell-type contract (P1, P3).** Assert `--scheduling` matches the cell class
   (native for RB/budget, pipeline for baselines) before the cell runs, not after it fails.
7. **Remote-orphan check (N5).** Before starting any runner, assert zero matching processes on
   the remote host — a local kill does not reach them.
   **Built** (`runtime_gates.py orphans`, self-match guarded). ⚠ Its default pattern named the
   v1-era `_cells.sh` and **missed both S2 runners** until 2026-08-01; now
   `benchmark_serving|cara_serve|p1_pareto.sh|s2_q1.sh|s2_q1_burst.sh`. It is live — it caught a
   stale smoke-test `cara_serve` on the head node during the 2026-08-01T23:30Z pass.
8. **CELL_MAP reconciliation (D25).** Assert the map's per-directory cell counts equal what is on
   disk. Today the map said 0 for `gemma_cells` (24 present), 23 for `v1exact_recal` (32 present),
   and omitted four Gemma archive directories entirely.

## Live findings from the static gate, this run

> ⚠ **STALE — superseded 2026-08-01. Re-executed, the gate reports 4 occurrences, all
> baselined, `0 new, 0 resolved`, rc=0.** Kept for the audit trail; do not act on the list
> below. Of the four items: `pred.get("quality", 0.0)` **no longer exists** (Phase B fix); the
> duplicated host→GPU map is resolved (the gate now prints *"one canonical host→GPU map at
> `cara_serve.py:180` — D26 satisfied"*); and the "ladder in the serving path" is now `:2010`,
> **inside `if os.environ.get("CARA_DUMMY_ESTIMATOR") == "1"`** — the isolation harness, not the
> serving path. Current state is in `2026-07-31_GATES.md` §G-STATIC and
> `2026-08-01_STATIC_GATE_TRIAGE.md`.

~~16 occurrences across all four classes. The ones that are real defects rather than idiom:~~

- ~~`plot_predictor_comparison.py:83` — genuine falsy-`or`: a best-model accuracy of **0.0** falls
  through to a different source. Same class as D2.~~
- ~~`cara_serve.py:1880` — hardcoded `["Qwen/Qwen2.5-3B", …]` ladder **in the serving path** (D13).~~
- ~~`cara_serve.py:2978` and `:3072` — the host→GPU map defined **twice** (D26, confirmed open).~~
- ~~`cara_serve.py:1720, 1724` — `pred.get("quality", 0.0)` fabricates a score when quality is
  missing; a missing value makes a model look worst and changes routing.~~
- `batch_scoring` confirmed dead tree-wide — consistent with D17. Allowlist it with that reason
  rather than deleting: it is the deliberate BEST-Route offload switch and must stay `False`.
