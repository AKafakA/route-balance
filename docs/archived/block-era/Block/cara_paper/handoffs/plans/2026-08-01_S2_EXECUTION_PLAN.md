# S1½ → S2 — instrument, gate, prove, sweep (deadline: end of Monday Aug 03)

*(Replaces the Jul-30 v3 audit→fix→gate→sweep plan. That plan's Phase 0/1 are done; its Phase 2
gates turned out to be unenforceable. This is the execution plan for the 12-node A30 lease
tonight and the full S2 window, ending at the NeurIPS author-posting close.)*

## Context

Three campaigns (v2 June, v1exact July, v3 Jul-31) each ran a "static TPOT" arm that **never
applied the constant**. Every one passed config diffs, routing-mix inspection and RULE-0, because
every check read a *declaration* — an exported env var, a config key — and never the *effect*.
The v3 arm was proven inert only after the fact: arm 1 and arm 4 agree on 96.2–98.3% of routing
decisions against a 95.5% same-config baseline.

Meanwhile the paper's `tab:bootstrap` prints the evidence of a second defect in its own
`coverage` column: N=2,764 of 3,534 for RouteBalance (78.2%), 2,553 (72.2%) for passthrough. A
DeepEval score of exactly **0.0 is falsy** and was filtered as missing, so every published mean is
an average over non-zero scores only. The public reproduction script already fixes this
(`route-balance` commit `4aeb956`: *"dropping 855 valid 0.0 scores, coverage 0.758→1.0, quality
0.489→0.371"*), so the correction is already in the open — the paper's tables are not.

**Two deliverables by Monday**, both to be defended in the final comment to reviewers and AC:

1. **Static vs learned TPOT, measured, with CIs** — the AC's priority-#1 question, currently
   unanswered by any valid cell in any campaign.
2. **RouteBalance dominates each baseline at that baseline's own best parameter, on the
   zero-safe basis** — and correcting the evaluator bug *helps*, because passthrough loses more
   (72.2% coverage vs 78.2%).

**The blocking discovery, found while planning:** the TPOT gate written tonight reads
`tpot_log.jsonl`, and **nothing in any current tree writes that file**. `CARA_DEBUG_TPOT=1` only
prints a text summary to stdout every 25 requests (`cara_serve.py:1101`). The gate would have
passed vacuously on Monday. Instrumentation must precede gating.

## Phase 0 — ONE TREE (blocking; nothing else matters if this is wrong)

Every defect chased today reduces to **fixed in one tree, run from another**:

- `CARA_STATIC_TPOT_HEAD` had a reader in `Block` (Gemma worked) and none in `Block_v3` (Q1 inert)
- the P100 head was loaded in both, selectable only where the host map lacked the D3 fix
- the author's 50 ms removal (`ddfd506`, Jul-30 23:08) landed in the repo and never reached the
  cluster tree that ran the Gemma cells

So S2 deploys **exactly one tree**, and it is gated:

1. `rsync` this repo to every host as the single tree. No `Block_v3`, no `Block_v1exact`, no
   `Block_v1x_arm4fix`.
2. Record `TREE_MANIFEST.md5` over every `block/**/*.py` **before the first cell**, and prove it
   non-empty and complete with `runtime_gates.py manifest --manifest ... --tree ...`.
3. At every cell, assert the running tree's serving-file md5s still match the manifest.
4. Archive the tree tarball with the cells at close, so no future question is unanswerable the way
   `Block_v1exact` and `Block_v1x_arm4fix` now are.

If step 2 cannot pass, do not run cells.

## Phase A — Instrument (offline, tonight)

Nothing else works until the scheduler records what it did. **Reuse what exists** — the harness is
in better shape than the gates were.

| change | file:line | why |
|---|---|---|
| **Structured TPOT log, ALWAYS ON** | `cara_serve.py:1097-1102`, inside `_batch_predict_tpot_all` | the `out` dict already holds `inst._instance_id → tpot_s`, and `by_tier` already holds `itype` — everything `tpot_gate.py` reads. Emit one JSONL line per prediction round. **No env gate.** |
| Delete `CARA_DEBUG_TPOT` | `cara_serve.py:1101` | it prints prose to stdout every 25th request and carries no `itype`; unparseable and sampled |
| `EFFECTIVE_CONFIG` per cell | `_log_effective_config` + `update_config_endpoint` (`:702`) | re-emit on every `/v1/config` POST. Today one line covers a whole multi-cell session, so a cell cannot be attributed to a config |
| Resolved state **into the result JSON** | `benchmark_serving.py` `--metadata KEY=VALUE` (already merges to top level, `:1389`) | zero code change: the runner passes the resolved config as metadata, so every cell carries its own provenance |

**Then** re-point `tpot_gate.py` at the real emitted format and re-run its four fixtures.

## Phase B — Close the silent-failure paths

Same rule as the env-var purge: a wrong value must fail loudly, never degrade.

| defect | fix |
|---|---|
| `CARA_INPROC_PREDICTOR` reports `true` when the model failed to load (`:3379`) | require all expected tier models; raise on partial load; emit `EFFECTIVE_CONFIG` only after success |
| `CARA_MODELS_DIR` points at a tree | derived from the running tree already; assert the resolved path is inside it |
| `KNNModelEstimator` lacks the deepeval array validation `DefaultModelEstimator` has (`model_estimator.py:1039`) | same startup check; never return 0.0 for a missing judge array |
| `pred.get("quality", 0.0)` fabricates a score (`:1720,1724`) | raise; a missing quality must not silently make a model look worst |
| `slo_defaults.get("quality_gain", 1.0) or 1.0` | a configured 0.0 becomes 1.0 — explicit `is None` |
| `static_defect_gate` CLASS 1 exclusion is unsound; CLASS 2 line-oriented; CLASS 3 misses `[False]*n`; CLASS 4 scope-blind | Codex findings 8–10 + 12; fix or explicitly allowlist with a reason |

## Phase C — Reviewers (parallel with A/B)

Codex adversarial review over `2026-07-31_GATES.md` + every gate + the Phase A/B diffs, with the
specific brief: *find the gate that passes when it cannot obtain its evidence*. That class has now
appeared four times — the env echo, the 8/14 fleet gate, the early-return in `preflight_gate`, and
the TPOT gate reading a file nothing writes.

## Phase D — Prove the gates on 12 A30 nodes (tonight → Sunday)

Fleet: `asdwb-311959`, 12× d7525, **8×3B + 4×7B**, saved at `window_setup_work/s2_cluster/`.
Nodes are reachable (SSH authenticates) but have **no NVIDIA driver** — stock image.

Provision with **`window_setup_work/setup_jul25.sh`** (author-approved, S0–S14; ~31 min for 12
nodes), never `block/exp/cara/setup.sh` and never manual installs. A30 is ampere → `nvidia-open`,
no reboot. Then `prefetch_weights.sh` for 3B+7B (~5.8 + 15 GB per node), then
`deploy_cara.py --models "Qwen-2.5-3B,Qwen-2.5-7B" -d model_instance monitor --parallel-deploy 8`.
A new A30-only `model_deployment.json` is needed: 8 of the 12 hosts appear in `jul25/`, four
(`10311, 10315, 10335, 10337`) are new. **Do not pass `-d predictor`** — `host_configs.json` is
scrubbed and will KeyError.

**Cells here are throwaway.** The deliverable is that every gate demonstrably *fails a real
experiment*:

| gate | how it is proven, on hardware |
|---|---|
| G-TPOT | run one cell with the static prior configured and one without; assert 1 distinct TPOT/tier vs many. Deliberately mis-set one tier's constant and confirm assertion C fires |
| G-TIERS | `--allowed-hw a30` on a fleet that is A30-only — must pass; then declare `a30,v100` and confirm it still passes, and inject a fake tier to confirm it fails |
| G-ARMS | run the *same* config twice → must FAIL (agreement ≥ baseline); run two genuinely different weight tuples → must PASS |
| G-ORPHANS | leave a runner running, confirm the gate refuses to launch |
| G-MANIFEST | record a real manifest over the deployed tree; confirm coverage check catches a deliberately unlisted file |
| G-EFFECTIVE | assert resolved `score_type`/`k`/`quality_norm`/`filter_enabled` match the config, then change one and confirm the mismatch fails |

A gate that cannot be made to fail on demand is not a gate and gets rewritten.

## Phase E — S2 execution (Mon 01:00 → Tue 12:00 BST, 35 h)

Full fleet returns Monday 01:00 (A100 + V100 join the 12 A30s).

**Order is deliberate: Q1 first, rates large→small.** The posting deadline is Aug 03 AOE =
**Tue 12:59 BST**, 59 minutes after the lease ends — nothing landing late can be written up.

| slot | block | cells | est |
|---|---|---:|---|
| 01:00–02:00 | setup, deploy, **all gates green on the real fleet** | — | 1 h |
| 02:00–11:30 | **Q1** (author 2026-08-02): ±static × λ{12,24,30} × {Poisson, γ0.25} × **6 seeds** × **2 presets** (532, uniform) | **144** | **9.5 h** |
| 04:00→ | **Main grid, λ 30→24→18→12→6** (author, 2026-08-02) | **195** | **17.4 h** |
| tail | budget λ16 (9), batching (23), label ladder (8) | 40 | 3 h |

> 🔴 **CORRECTED 2026-08-01: the grid is 273 cells, not 203; 27.6 h, not 20.5 h.** The count
> below (29 configs) omits Po2 as a *third* dispatcher at **every** rate rather than at λ12/λ24
> only. The implemented grid — `p1_pareto.sh`, `s2_driver.sh`'s `GRID_CELLS_PER_RATE:-39`, and
> every banked `INDEX_p1.md` — is **39 configs × 7 rates = 273**: RB 8, BR 4×3, AP 5×3, DISP 4.
> Budget: 27.6 h grid + 1.8 h Q1 + 1 h setup = **30.4 h against a 35 h lease**, so the 3 h tail
> block does NOT fit alongside a full grid. Following the 203/20.5 h figures understates the
> night by 7 h. `2026-07-30_V3_CAMPAIGN_PLAN.md` §0 has carried the corrected number since
> 2026-07-30; this plan did not, and risk ② below inherits the same error.

Main grid = ~~29 configs × 7 rates = 203~~ **39 configs × 7 rates = 273**, reproducing
`tab:extremes` exactly. Reuse
`v3_scripts/p1_pareto.sh`, which already implements this grid with `dlabel()` and an append-only
`INDEX_p1.md`:

| system | knob values | disp. | cells |
|---|---|---|---:|
| RouteBalance | (0.1,0.1,0.8) (0.1,0.8,0.1) (0.2,0.2,0.6) (0.2,0.6,0.2) (0.33,0.33,0.33) (0.5,0.3,0.2) (0.6,0.2,0.2) **(0.8,0.1,0.1)** | joint | 8 × 7 = **56** |
| AvengersPro | pw 0.25 · 0.39 · 0.53 · **0.70** · 0.80 | RR, SQ, **Po2** | 5 × 3 × 7 = **105** |
| BEST-Route | t **0.0** · 0.3 · 0.5 · 0.7 | RR, SQ, **Po2** | 4 × 3 × 7 = **84** |
| Dispatcher-only | — | RR, SQ, Rand, **Po2** | 4 × 7 = **28** |

> ⚠ **Counts corrected 2026-08-02 by enumerating the payloads `p1_pareto.sh` actually emits**
> (not by re-reading the plan). Per rate: **RB 8 + BR 12 + DISP 4 + AP 15 = 39**, matching
> `GRID_CELLS_PER_RATE=39`; × 7 rates = **273**. The old rows said "RR, SQ" for BR and AP and
> gave 70/56/21 — Po2 is a third dispatcher for **both** baselines and a fourth for
> Dispatcher-only, and BR's four thresholds are 0.0/0.3/0.5/0.7 (`t=0.0` is the argmax case,
> passed with an explicit `fallback_model` so the v1 hole cannot reopen).
>
> **All 39 payloads validated offline:** every one parses as JSON; every `checkpoint_path` and
> `artifact_dir` exists on disk; every BEST-Route payload carries `fallback_model`; no two
> configs collide on filename; the 8 RB tuples are distinct.
>
> **One cosmetic non-issue, checked so it is not re-raised:** `(0.33,0.33,0.33)` sums to
> **0.99**, and `cara_serve` does **not** normalise (`score = w_qual·q + w_lat·l + w_cost·c +
> w_bal·b`, weights raw). It is nonetheless **routing-identical** to exact uniform — with
> `w_balance = 0` the score is a positive scalar multiple of `q+l+c`, and argmax is
> scale-invariant. Verified: 0 ranking differences in 20,000 random 4-candidate draws, while a
> genuine 0.20→0.19 change differed in 571. **Do not "fix" it** — it is the paper's tuple and an
> experiment parameter.

`pw=0.70` and `argmax` are **required** — the paper's §5.2 omits them but `tab:extremes` counts
them (D6). Po2 is a rebuttal addition, not in the paper grid. **It runs at EVERY rate** (not λ12/λ24 only), which is where the 203→273 difference comes from.

Per-rate cost from the measured model `duration ≈ N/rate + 10 s`: λ30 141 s → λ6 599 s, N=3,534.
Cap **1024** (`V3_MAX_OUTPUT_LEN`), not the rebuttal's 256.

### Phase E′ — dominance closure (adaptive, runs after each λ completes)

**Rule: if no banked RB tuple dominates a baseline's best point at some λ, sweep more of the
simplex until one does — or report honestly that none can.**

After each rate finishes, before moving to the next:

1. For each baseline, select **its own best knob** at that λ by the paper's rule,
   argmax(quality × throughput) — the same selection `tab:radar_picks` uses.
2. Test every banked RB tuple for **strict domination**: quality ≥, cost ≤, mean e2e ≤, p99 ≤.
3. **If one dominates** → done for that λ, record which.
4. **If none does** → identify the deficit axis and interpolate on the simplex. The knob is
   monotone in the useful direction, measured today at λ12:

   | tuple | q (zero-safe) | cost $/1k | mean ms |
   |---|---:|---:|---:|
   | (0.33,0.33,0.33) | 0.4047 | 0.0127 | 1894 |
   | (0.5,0.3,0.2) | 0.4144 | 0.0173 | 2395 |
   | (0.8,0.1,0.1) | 0.4290 | 0.0321 | 3959 |

   Quality and cost both rise with `w_q`. So: **short on quality → raise `w_q`; short on cost →
   lower it.** The widest unexplored gap is between 0.5 and 0.8, so the first fill points are
   **(0.6,0.2,0.2)** — already a grid member — then **(0.7,0.15,0.15)**, then bisect.
5. **Budget cap: ≤ 3 extra tuples per (baseline, λ), and only at λ ≥ 12** where a cell is ≤ 5 min.

> ⚠ **IMPLEMENTATION STATUS, verified 2026-08-01 — Phase E′ is ADVISORY, not adaptive.**
> * The **λ ≥ 12 half is implemented** (`dominance.py --fill-min-lambda`, default 12; the driver
>   gates on `[ "$lam" -ge 12 ]`). Below λ12 it prints that fill points are forbidden.
> * The **≤ 3-tuple cap does not exist in any file** (`grep` over `s2_driver.sh`,
>   `p1_pareto.sh`, `dominance.py`). It is moot only because —
> * **no fill cell is ever launched.** `dominance.py --suggest` *prints* a tuple and the driver
>   *says* the tuple is in the log. A NOT-DOMINATED rate is therefore never closed by the
>   machinery; **a human must decide to run the fill point.** That is a defensible design, but
>   it is not what rules 4–5 describe, and the difference matters at 04:00 unattended.
> * The **honesty guard is a print statement, not a mechanism.** `dominance.py`'s table has no
>   notion of pre-registered vs interpolated, so a hand-added `(0.7,0.15,0.15)` cell would
>   appear in the headline table indistinguishable from a grid member — which rule 4 forbids.
>   **If a fill point is ever run, its label must be added by hand.**
>
> Rules **1** (argmax q × throughput), **2** (≥ on all four axes, > on ≥1), **3**, and rule 4's
> suggestion *direction* are implemented and verified against all 7 banked rates.
   Below λ12 a fill point costs 6–10 min and the grid matters more.
6. **If domination is still not achieved**, stop searching and report it. Two legitimate outcomes
   that are not failures: (a) RB matches on quality at strictly lower cost *and* latency —
   the iso-quality claim the paper already makes; (b) a baseline genuinely wins one axis at one λ,
   which is stated plainly and offset by the other six rates.

**Honesty guard — this must be disclosed, or it is post-hoc fitting.** The reply says: *"We swept
eight pre-registered simplex tuples. Where a baseline's best operating point fell between two of
them, we added N interpolated tuples to locate the crossing; those points are marked in the table."*
RouteBalance exposes a continuous knob and the paper sweeps a coarse grid — refining it near a
competitor's operating point is legitimate **only if the added points are labelled as such**.
Never present an interpolated tuple as if it had been in the original grid.

**Cut order if late:** λ6 then λ8 — 47% of grid cost, and already agreed as deferrable. Never cut
Q1 or the λ12/λ24 rows that carry the dominance claim.

## Phase F — Analysis and the final comment

- **Q1 with CIs:** 6 seeds per arm per rate → mean ± bootstrap CI **across seeds**, which is a
  different and stronger statistic than the paper's within-run per-prompt bootstrap. Lift
  `bootstrap_ci()` from `smoke_test_apr_13/scripts/bootstrap_ci.py` (B=1000, `default_rng(42)`) —
  but re-point it, since **that file is itself the 4th falsy-zero site** (`:78`, annotated) and is
  what produced `tab:bootstrap`'s 72–78% coverage.
- **Dominance:** for each baseline take **its own best parameter per λ** (`tab:radar_picks`
  selection rule = argmax quality×throughput) and show RB beats it. Lead with the **paired**
  bootstrap (RB−baseline per prompt), as `neurips_2026_v2.tex:337` already does — paired CIs
  exclude zero where the unpaired ones overlap. Report corrected `tab:extremes` and `tab:bootstrap`
  with **N=3,534, coverage 100%** on every row.
- **Disclosure, as decided:** state it plainly — *a DeepEval score of exactly 0.0 is falsy, so
  `per_model.get(m) or per_model.get(stripped)` fell through to a key that is never stored and the
  score was dropped. The `coverage` column in Appendix A is that defect, not a property of the
  data; App-A's definition ("prompts that map to a scored pair") is therefore wrong — the dropped
  pairs are in the lookup, with value 0.0.* Give corrected tables for every quality-bearing claim.
  State which ablations are unaffected: latency, cost, throughput and scheduling-overhead never
  read the quality lookup. Note the public reproduction script has carried the fix since
  `4aeb956` ("dropping 855 valid 0.0 scores, coverage 0.758→1.0").
- **Say that multi-seed closes the paper's own stated gap** (`:318`, "left to future work").

## Verification

- Every gate demonstrated failing on real hardware in Phase D, logged verbatim.
- Every cell carries its own `EFFECTIVE_CONFIG` and TPOT log; no provenance reconstructed after
  the fact.
- G-ARMS run on the Q1 pair **before** any Q1 number is reported.
- Every table row re-derived from its cell by `verify_*_rows.py`, and every *claim sentence*
  checked against what the cell measured — the gap that let Q1 ship.
- Preservation before release: md5 manifest, both trees, runners, environment, ground-truth
  tables, `closegate.sh` + `zstd -t` + offsite.

## Known risks — pre-registered

**① The RB-vs-BEST-Route inversion does NOT reproduce on the corrected basis — measured
2026-07-31.** An earlier draft of this plan pre-registered it as the main risk, from the July v1
re-aggregation (peak RB 0.4042 vs BR-t0.5 0.4056). **That is stale**: those are v1 cells
(pre-unwrap-fix, k=10) with only the *aggregation* corrected. Today's fix+k50 cells at λ12,
zero-safe, say the opposite — and both dominance claims hold **strictly on all four axes**:

*(Quality re-derived 2026-07-31T21:06Z. The earlier copy of this table labelled its column
"zero-safe" while its coverage column read 77-80%, i.e. it was NOT zero-safe. Cost, mean and
p99 were correct. The values below match `2026-07-31_CORRECTED_REBUTTAL_RESULTS.md` cell-exact,
verified by an independent re-derivation. Cost is decode-only, the paper's definition --
author decision 2026-07-31.)*

| system | q (zero-safe, 100% cov) | cost $/1k | mean ms | p99 ms |
|---|---:|---:|---:|---:|
| **RB (0.8,0.1,0.1)** | **0.4290** | 0.0321 | 3959 | 12886 |
| AvengersPro pw0.7 | 0.4276 | 0.0360 | 4362 | 13287 |
| RB (0.5,0.3,0.2) | 0.4144 | 0.0173 | 2395 | 12518 |
| **RB (1/3,1/3,1/3)** | **0.4047** | **0.0127** | **1894** | **5165** |
| BEST-Route Po2 | 0.3982 | 0.0131 | 2727 | 13969 |

- **RB uniform vs BEST-Route:** +0.0065 quality, −3% cost, −833 ms mean, −8804 ms p99.
- **RB (0.8,0.1,0.1) vs AvengersPro:** +0.0014 quality, −10.8% cost, −403 ms mean, −401 ms p99.

**Zero-safe correction HELPS the dominance claim.** BEST-Route has the lowest coverage (77.2%),
so it loses the most under correction — its reported mean was the most inflated.

**Today's numbers are provisional, but NOT contaminated.** An earlier draft of this plan claimed
the P100 zombie tier corrupted them. That was wrong: `Block_v3` — the tree that ran them — has no
`p100` in its host map AND raises `SchedulerConfigError` on an ambiguous instance_type mapping
(the D3 fix, which explicitly refuses to "silently take the first"). The cells ran to completion,
so they resolved to `qwen2.5-3b_a30`; the p100 head was inert baggage in the XGB file. Real p100
*usage* appears only in `arm4_532`, which ran on the unarchived `Block_v1x_arm4fix` tree.
Also checked and did NOT fire in today's cells: in-process predictor fallback, fabricated quality
default, KNN 0.0-for-missing-array.

They remain provisional for the ordinary reason: single-seed, one λ, one knob per baseline.

**What S2 must still establish:** these are λ12 single-seed points against each baseline at one
knob. The grid must confirm the same ordering (a) at every λ and (b) against each baseline's *own
best* knob per λ, not a fixed one. That is the real content of target #2.

**② The bug contaminates all 203 cells, not just Appendix A.** Measured coverage across the
released grid is 0.7221–0.7920 (mean 0.7671), so `fig:pareto`, `fig:radar`, `tab:extremes` and
`tab:radar_picks` are all affected — and each system is scored on its *own* endogenous non-zero
subset, so the exclusion rate depends on the routing under test. Corrected reference already
exists at `rb-exp/v1-exp/results/_reagg_corrected/V1_REAGG_BOTH.csv`.

**③ Two grid members are unnamed in the paper** and must still be run or the cell counts do not
reconcile: **AvengersPro `pw=0.70`** (tab:extremes credits 70 = 5 pw × 2 disp × 7 λ, §5.2 names 4)
and **BEST-Route `argmax`** (56 = 4 knobs × 2 disp × 7 λ; argmax appears in `tab:radar_picks` but
not §5.2). Defect **D6**.

**④ Setup is the schedule risk, not cells** — but it is bounded and measured. Canonical recipe is
`window_setup_work/setup_jul25.sh` (author-approved S0–S14), **not** `block/exp/cara/setup.sh`.
Measured: **31 min for 12 nodes**; longest legs CUDA 5 min, drivers 5 min, vLLM editable 6 min.
A30 = ampere → `nvidia-open`, **no reboot** (only Volta needs one). Pins: vLLM `2894a9872`,
**transformers 4.56.0** (not 4.50.3 — bumped 2026-04-13), torch 2.9.0, numpy 2.2.6.

**⑤ Weights must be downloaded, there is no local mirror.** The VPS HF cache holds no Qwen
weights. Use `prefetch_weights.sh` + `prefetch_one.py` (anonymous, `token=False` — the stored token
401s even on public repos). 3B ≈ 5.8 GB/node, 7B ≈ 15 GB/node. Pin revisions from
`v3_preservation/RERUN_PINS.md`, including `all-MiniLM-L6-v2` — a different revision changes
retrieval and therefore routing.

**⑥ Three concrete blockers on a fresh lease**, all found in exploration:
- `block/config/host_configs.json` is public-scrubbed (`*.cluster.example`); `deploy_cara.py:738`
  indexes it without `.get`, so `-d predictor` KeyErrors. Use `-d model_instance monitor`.
- `jul25/host_configs.json` carries **LAN `10.10.1.x`** addresses — the documented internal-IP
  incident — and the S2 lease has a different node↔IP mapping plus 4 hosts it does not contain.
- `block/exp/cara/setup_env.sh` is stale: `PYTHONPATH` points at `~/Cara` (tree is `~/Block`) and
  an unquoted `${HF_TOKEN:?}` aborts the script when the var is unset.

**⑦ A30-only tonight** means gate behaviour on a 4-tier fleet is *not* proven until Monday. Re-run
every gate proof at 01:00 before trusting it. Use `--allowed-hw a30` alone tonight.

**⑧ `Block_v1exact` and `Block_v1x_arm4fix` were never archived** — three questions about the July
cells are permanently unanswerable. Archive every tree that produces a cell this time.

**⑨ Multi-seed answers the paper's own admission.** `neurips_2026.tex:318` says outright: *"we did
not re-run the cluster sweep under additional Poisson seeds … A multi-seed across-run study is left
to future work."* Q1's 6 seeds close that gap directly — worth saying so in the reply. Note the one
prior multi-seed attempt had to discard its latency half as degraded-window-inflated
(`2026-06-07_v2_rerun_handoff.md:251`); seed the Q1 cells on a verified-healthy window, not
back-to-back after a long grid.

---

## Phase Q1-BURST — static vs learned TPOT under bursty arrivals (pre-registered)

**Why.** Steady-state Q1 separated the arms but barely: static was *faster* than learned by
1.3% (λ12) and 2.7% (λ24), CIs disjoint, on a 2-tier fleet. That is the weakest case for a
learned latency head — one GPU class, almost no heterogeneity to predict. **Burst is where a
latency predictor should earn its place**: if the learned head anticipates queue build-up it
should degrade less than a fixed constant when arrivals clump. The paper already makes a burst
claim (amortized-scoring within ~14% of stationary E2E, serial routers up to +74%), so this
re-measures a claim already in print.

### Design

**λ{12, 24, 30} × {Poisson, gamma 0.25} × 6 seeds × 2 arms = 72 cells, ≈4 h.**

| rate | regime | what it tests |
|---|---|---|
| 12 | bursts absorbed | can the predictor exploit slack? |
| **24** | **discriminating** — queueing matters, headroom remains | **the hypothesis** |
| 30 | overload under burst | does static degrade *worse* when saturated? |

> ✅ **Fleet is now a DECLARED decision (2026-08-02):** `s2_q1_burst.sh` requires
> `BURST_FLEET=2tier|4tier` and refuses to start if it disagrees with `ALLOWED_HW`. ~4 h
> against a window already holding 30.4 h of committed work is not something a script should
> assume. The λ30 re-confirmation cell below is the intended way to settle it with measurement.

**λ30, not λ28.** The old λ30 instability was **scheduler overhead**, and the batched path
removed it — measured tonight on the fixed code vs the July pre-fix cells at near-identical
candidate count (12 vs 13 instances):

| | λ24 → λ30 sched overhead | throughput CV | e2e CV |
|---|---|---|---|
| July, pre-fix | 245 → **1120 ms** (4.6× blow-up) | 8.8% | 25.6% |
| Fixed, batched | 147 → **172 ms** (1.17×, flat) | **3.7%** | **14.5%** |

p99 overhead 2946 → 630 ms. λ30 also matches the paper's existing grid, so the block is
comparable to the main tables; λ28 has no comparison point anywhere in the campaign.

⚠ **Re-confirm on the 4-tier fleet before committing.** More candidates and a wider TPOT spread
mean more scoring work. One λ30 cell at 01:30 settles it.

### Six pre-registered choices

1. **6 seeds, not 3.** Burst variance is structurally high — that is the point of gamma-0.25.
   Three seeds risks an inconclusive answer after paying most of the cost.
2. **Metric = degradation ratio**, `burst p99 / Poisson p99` at matched (λ, seed, arm) — not
   absolute p99. It is the form the paper's existing burst claim already takes, and it cancels
   fleet-level differences that make absolute numbers non-comparable to the submitted ones.
3. **Paired arrival streams.** Same `--seed` must give both arms identical arrival times, making
   this a paired comparison. **Verify empirically** by comparing inter-arrival sequences across
   two same-seed cells; if it does not hold, the pairing argument evaporates silently.
4. **Interleave, do not block.** Alternate patterns within each seed so fleet drift over 4 h hits
   both conditions equally instead of confounding with the variable of interest.
5. **Re-measure the same-config baseline per pattern.** The 98.9–99.2% figure was measured under
   Poisson; burst changes routing-agreement statistics. Importing it would repeat the 95.5%
   four-tier→two-tier mistake exactly.
6. **Gate the burstiness from the cell.** The cell records `burstiness`; assert it equals what
   was requested before banking. Identical defect shape to the static-TPOT arm — a parameter
   declared, plausibly ignored, invisible afterwards.

> ✅ **IMPLEMENTED AND VERIFIED 2026-08-02 (author-approved).** `verify_pairing.py` AST-lifts
> `get_request` verbatim from the shipping file and execs it with `asyncio.sleep` recorded and
> `time.time` frozen, so the REAL arrival generator is exercised offline. Confirms determinism,
> both-arms-identical, seed-varies, and that gamma-0.25 is burstier than Poisson — with the CVs
> matching theory independently (Poisson 0.971 vs 1.0; gamma-0.25 1.976 vs 1/√0.25 = 2.0).
> `s2_q1_burst.sh` now runs it per (seed, rate, pattern) before the sweep (~17 s total) and
> records every schedule digest in `ARRIVAL_DIGESTS.txt`. **A related gate was added that the
> plan did not ask for and should have:** one `--seed` seeds BOTH RNGs, and the python stream
> feeds `_sample_rso` (per-request SLOs) and LoRA — safe today only because `--rso-rate`
> defaults to 0.0, which is now asserted rather than assumed.
>
> ~~NOT IMPLEMENTED — owed before the burst block runs.~~ (superseded) `s2_q1_burst.sh` never compares
> inter-arrival sequences, and `burst_analysis.py` pairs by `(λ, seed)` and simply *assumes*
> the pairing holds. This choice's own text says that if it does not hold "the pairing argument
> evaporates silently" — so the guard against that is exactly what is missing. Neither
> `STATUS §3` nor `GATES.md §G-BURST` mentioned choice 3; both enumerated the design guards and
> listed only choices 4 (interleave) and 5 (per-pattern baseline). Choices 1, 2, 4, 5 and 6 are
> implemented. **Do not run the burst block until choice 3 is either implemented or explicitly
> waived** — the whole analysis is a paired comparison.

### Batched KNN must be PROVEN applied, not assumed

`EFFECTIVE_CONFIG.batched_path` was `hasattr(model_estimator, "estimate_batch")` — a
**capability**, true whenever the object has the method, whether or not the branch ever ran. Both
call sites are `elif`, so a preceding branch takes precedence, and the per-request fallback is
the one that skips `_estimator_prompt()` (the chat-skew defect, locked-defaults §E).

`cara_serve` now **counts** both branches (`_SCORE_BATCHED_CALLS`, `_SCORE_PERREQ_CALLS`),
exposed via `GET /v1/config`. **Gate:** snapshot before and after each cell; if
`score_perreq_calls` moved, some batch took the un-unwrapped fallback and the cell is suspect.
This is the whole reason the batched path can be claimed at all — the λ30 stability argument
above rests on it.

### Open — author decision

**Which fleet.** On the 2-tier rehearsal fleet this measures the MACHINERY; on Monday's four
tiers it measures the CLAIM. If the block is destined for the reply it must run on four tiers.
