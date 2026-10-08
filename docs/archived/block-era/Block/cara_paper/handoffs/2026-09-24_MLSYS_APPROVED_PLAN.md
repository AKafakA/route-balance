# RouteBalance → MLSys 2027: thesis, experiments, paper, minimal repo (plan, 2026-09-24)

## Context

NeurIPS 2026 rejected RouteBalance on 2026-09-24 (sub 28271). It was borderline with no champion. The AC gave
three reasons: novelty vs llm-d/AIBrix; unjustified design choices (XGBoost); weak general-purpose
dispatcher comparisons.

The next target is **MLSys 2027**:
- due **2026-10-30 20:00 UTC**;
- 10 pages, 2-column, `mlsys2025style`;
- the appendix is a separate upload that reviewers need not read;
- arXiv versions are allowed.

Constraints:
- A full paper rewrite under the unchanged title "RouteBalance: A Unified Router and Load Balancer for
  Heterogeneous LLM Serving".
- **One 7-day lease: 2026-09-28 10:00 → 2026-10-05 10:00** (3× c4130 V100, 8× d7525 A30, 2× d8545 A100).
- The repo is rebuilt from scratch as a **minimal, readable** codebase, the only execution repo.

This week's audit (scratchpad `specs/B–F`, claims ledger `specs/G` draft) and a check of the banked v3
cells found that **the v3 comparison was not execution-equal, and RB's latency term does not see load**.
The plan earns the MLSys thesis on a parity-by-construction harness, rather than reusing v3 claims.

## 0. Evidence that reshapes the thesis (verified on banked v3 cells, 2026-09-24)

1. **RB's tier mix is flat across load.** 72B share at (0.8,0.1,0.1): 45.7→45.8% from λ6 to λ30. At
   (0.5,0.3,0.2): 9.6→11.3%. At uniform: 0.7→1.0%.
2. **Why.** In the static cells, the queue term fired in 21.5% of evaluations on 14B/V100 and 13.7% on
   3B/A30, but only 1.5% on 72B/A100 and 1.8% on 7B/A30. Under continuous batching, load shows up as a
   slower TPOT at larger running batches, which a constant TPOT cannot see. Max-divide normalization pins
   the slowest tier's latency score at ~0 whatever its queue.
3. **"Strict dominance at λ18–30" is an artifact of the own-best rule** (argmax quality×throughput
   rewards short tails at fixed N). Against AP pw0.70 at λ30, RB is a trade-off.
4. **RB's scheduler adds 1.0–3.2 s at λ30** (off-instance residual, up to 35% of E2E). RB's
   estimator already runs off the event loop. The suspects are per-request Python scoring, decision-log
   writes, and every in-flight token stream passing through the same event loop
   (`vllm_instance.py:157`). O1 and a loop-lag probe measure this.
5. **The BEST-Route collapse was a harness artifact.** This was already found on 2026-06-10: BEST-Route's
   DeBERTa ran synchronously on the event loop, and AvengersPro had the same flaw. With the fix,
   BEST-Route's λ24/30 latency went from 51/63 s to 6.5/13 s. The v3 tree (v1exact) reverted to inline
   scoring. The per-request stats fetch (p50 2.3 s at λ24) and fd exhaustion remained. With full parity,
   **expect no collapse and close latencies**.
6. The learned TPOT heads were served 5 zero-filled features. Prompts were chat-templated twice. Seeds
   varied only arrivals.
7. **Claims ledger:** 243 design/eval claims across the NeurIPS tex, arXiv tex, rebuttal and outline;
   78 carry inter-source conflicts. Examples: KNN k 50 vs 10; normalization scope; cost term;
   static-arm definitions; telemetry sync vs async; batch-size-1 results; baseline knob counts; hosting
   "architecture-controlled"; quality coverage; decision-compute numbers.

## 1. Decisions locked (owner, 2026-09-24)

| topic | decision |
|---|---|
| venue / paper | MLSys 2027; full rewrite; title and name unchanged; no llm-d baseline |
| repo | route-balance only; rebuilt from scratch, minimal; **clean history: whole gated rebuild squashed into one initial commit** |
| backups | upload the verified bundles (`~/backups/2026-09-24_pre_rebuild/`) to Drive before touching git |
| parity | **share everything**: one `TelemetryView` (background poller + dead-reckoned deltas) read by every system; every router model off the event loop |
| shared fixes | single chat templating; pooled bounded HTTP session; fail loud on silent defaults; atomic validated `/v1/config` with complete weights |
| latency (candidate default) | **profiled TPOT(running batch) per (model,GPU)** from a day-1 calibration (no training) × own predicted length + waiting term when an instance is full; **fixed-reference normalization** (vs the request's best unloaded latency). Adopted only if **G1** passes. |
| ablations | static queue-aware + max-divide (v3-like); **learned XGBoost retrained on served features only**; **prompt-blind = per-model mean prior**; 50% labels; LPT off; round size 1/32; dead reckoning off |
| controls | **RB-decoupled** (same predictors; model chosen without load, then the dispatcher); **least-outstanding-requests** dispatcher; **G1 gate on lease day 2**; **O8 offline replay before the lease** |
| headline rule | **equal-quality deltas (latency, cost, with CIs) + full frontier + SLO goodput above a quality floor**; the own-best argmax(q×tput) rule is retired |
| quality metric | **DeepEval (G-Eval, Llama-3.1-8B judge, reference-grounded), zero-safe over all prompts, both families; scored on the response actually served (store keyed by prompt, model, text hash; D39)** |
| budget | kept: clamp + predicted-length admission filter; one main-text paragraph, appendix details |
| scheduler node | an A100 node for every system (48 cores; co-located with a 72B instance, stated in the paper) |

## 2. Thesis and contributions

**Thesis (if G1 passes):** in a mixed-model, mixed-GPU fleet, the model choice has to see instance load. Router +
load-balancer stacks fix the tier before load is visible. RB decides (model, instance) jointly and
trades quality only where a tier is congested.
- **C1, measured problem:** time queued at the router-chosen tier vs λ; model size ≠ speed across GPU
  types (14B/V100 faster per token than 7B/A30).
- **C2, design:** joint assignment with a profiled, load-reactive latency model. Onboarding needs
  calibration only, no training. Adaptive rounds + dead-reckoned telemetry keep decisions cheap.
- **C3, fair evaluation:** parity-by-construction harness. Two trained routers × four dispatchers,
  plus RB-decoupled, which isolates the joint decision.
- **C4, scale and generality:** offline decision-cost microbenchmark to 1k instances; Gemma-3 family;
  re-provisioned topology.

**Fallback if G1 fails (joint ≈ decoupled):** thesis becomes instance-aware, length-aware routing with a
cheap online loop. The under-load claims are dropped.

**Positioning (Table 1, every cell verified against current docs/papers):**
- routers (RouteLLM, AvengersPro, BEST-Route, vLLM-SR) pick the model but don't see load;
- serving balancers (llm-d EPP, AIBrix, Dynamo, SGLang router, Llumnix) see load but take the model as given;
- offline planners (BOute MLSys'26, RouterWise) set deployment/policy for an expected load;
- RB does all of it online, per request.

RB-decoupled is "router in front of a load balancer" built from our own components.

## 3. Minimal repo rebuild (Sep 24–27)

**Git:**
1. Drive upload + sha check.
2. Re-initialise route-balance's history.
3. Build S1–S11 as local working commits so each review has a diff.
4. Squash into one clean initial commit.
5. Force-push to GitHub `main` only with explicit confirmation at that moment.

**Source of truth:** Block `v3-campaign` 984ea18. Never port June `master` or old route-balance code.

**Layout (~22 modules; `CODE_TOUR.md` walks it in this order):**
```
routebalance/
  config.py, types.py, hw.py                    typed schema, fail-loud, atomic apply; ONE GPU map
  telemetry.py                                  shared poller + dead-reckoned deltas (all systems)
  estimate/knn.py                               quality+length; per-model-mean prior mode
  estimate/latency.py                           TPOT source {profiled_curve, constant, xgboost} × envelope
  schedule/rounds.py, score.py, assign.py       adaptive rounds, objective + normalization {fixed_ref, max_divide}, LPT, tiebreak, vectorized per-round scoring
  schedule/budget.py                            clamp + admission filter
  baselines/routers.py, dispatch.py, pipeline.py   AP, BR, passthrough; RR/SQ/Po2/LOR/random; RB-decoupled mode
  backend.py, server.py, decision_log.py        pooled client, single templating; thin FastAPI; buffered async evidence log + loop-lag probe
bench/     arrivals.py (Poisson, gamma, step traces), dataset.py, runner.py
analysis/  metrics.py (zero-safe DeepEval, decode cost, goodput), equal_quality.py, frontier.py, manifest.py
gates/     preflight, parity (execution-manifest diff across systems), arm (values actually used), zero_call, hygiene, static_defects, selftest
tools/     calibrate_tpot_curve.py, retrain_xgb_served.py, build_prior_index.py, build_ap_artifacts.py, replay_decisions.py (O8), microbench.py (O1)
scripts/   setup.sh (from setup_jul25.sh; pinned wheel URL + sha256), deploy.sh, run_cell.sh, run_scope.sh
configs/   fleet_qwen.json, fleet_gemma.json, presets.json, scope.json, prereg_rules.json
tests/     unit/, contract/ (fake vLLM backend), equivalence/ (v3 fixtures), gates/
```
Dropped (dead for scope): broadcast, ollama, sidecar, legacy strategies, Option-A loops, TIERED/RELAXED, SLO
filter, PFS/RoBERTa/LSTM/roofline, P-D, data collection.

**Per-commit review rule (owner, 2026-09-24): every local commit gets BOTH a local code review and a
Codex review, focused on correctness only, with no nitpicking.**
1. Tests first; port; `make check` green (lint, hygiene, static, unit, contract, equivalence, gates).
2. **Local review:** a fresh subagent runs the `code-review` skill on the staged diff.
3. **Codex review:** `codex:rescue` in read-only review mode on the same diff.
4. Both reviewers get the same brief:
   - Report only correctness defects: wrong behaviour vs the v3 spec or the plan; broken equivalence;
     silent defaults or fallbacks; parity violations between systems; wrong units; race or event-loop
     blocking; data-integrity or gate holes.
   - Every finding comes with file:line and a concrete failure scenario.
   - No style, naming or formatting comments.
5. Fix every confirmed finding. Disputed findings are written down with evidence, never silently dropped.
   Re-run `make check`, then commit. The commit message records the verdict of each review.
6. A commit with an unresolved correctness finding does not land. The final squash happens only after
   every step commit has passed both reviews.

| step | content | acceptance |
|---|---|---|
| S1 | skeleton, pins = v3 head freeze, Makefile, hygiene/static gates | gates fail on seeded violations |
| S2 | config/types/hw | missing or unknown key raises; atomic POST |
| S3 | telemetry + fake backend (16 fields, auto-extend +10) | contract tests; stale/missing raises |
| S4 | knn + prior index + latency modes | v3 fixtures: KNN 1e-9 (stub encoder) / MiniLM 1e-6; banked `all_model_scores` 1e-6; xgb features bitwise |
| S5 | rounds, score (both normalizations), LPT, tiebreak, dead reckoning, budget; vectorized | decision replay == v3 in the v3 modes (constant+own_length+max_divide; xgb+queue+max_divide) |
| S6 | routers, dispatchers incl. LOR, pipeline, RB-decoupled | BR/AP 3534/3534 choices == banked v3; dispatchers identical on a seeded 20k-event trace |
| S7 | server + backend | contract; loop-lag < 5 ms under 64 concurrent routes |
| S8 | bench + analysis | arrival digests 73/73; prompt order; λ30 re-aggregation == `DOMINANCE_l30.csv` |
| S9 | gates | every gate fails its MUST-FAIL fixtures |
| S10 | scripts, CloudLab profile, scope.json, prereg_rules.json | dry-run renders every cell |
| S11 | dev-gpu smoke (3B+7B, Py3.10 venv, TRITON_ATTN, smoke-only); freeze tag + tree md5 | 100-request cells per system, gates green |

S4, S6 and S8 run as parallel subagents after S3.

## 4. Pre-lease data prep (parallel; dev-gpu + CPU)

- DeepEval-score the Gemma **train** split: 8,086 prompts × 4 = 32,344 pairs; the test split already has
  1,978/1,979. Use the same script and judge in a new Py3.10 venv on dev-gpu. Rescore 200 Qwen pairs to
  check drift.
- Rebuild the Gemma KNN (DeepEval, k=50). Build AP-Gemma pw {0.25, 0.39, 0.53, 0.70, 0.80} (only 0.70 exists).
  BEST-Route-Gemma is trained only if dev-gpu time allows; otherwise the paper states BR is Qwen-only.
- Retrain XGBoost on served features; build the prompt-blind prior index.
- **O8:** replay banked v3 decision logs under TPOT(b) + fixed-ref normalization, to size the tier shift
  before lease time is spent.
- **O1:** microbenchmark of decision cost vs instances (13→1024), tiers and round size; also a pre-lease
  overhead gate.
- Write `configs/prereg_rules.json` before the lease: the G1 rule, the equal-quality matching tolerance,
  the goodput SLO and quality floor.

## 5. Lease matrix (7 days; measured v3 min/cell: λ6 10.3, λ12 5.5, λ18 4.1, λ24 3.3, λ30 2.9)

| ID | prio | supports | cells | h |
|---|---|---|---:|---:|
| W0 | must | setup.sh ×13 (in parallel), smoke, parity evidence (loop lag, stats age, router_ms); **reference generation for the served-response score store (D39): every (prompt, model) once on the lease stack, plus a sample twice at different loads to measure divergence; judged on dev-gpu** | — | 3–5 expected (+0.5 for references); 12–16 worst case |
| W1 | must | TPOT(batch) calibration per Qwen tier (held-out prompts, 6 concurrency levels) | 24 | 2 |
| **G1** | must | thesis gate: RB-new / RB-static / RB-decoupled × 2 presets + AP pw0.7/0.8 (LOR) × λ{12,30} | 16 | 1.5 |
| W1b | must | new-stack latency sweep for the XGBoost arm (D38): training-split prompts, random dispatch over all instances, λ 6→36, served features + realized TPOT per request; night 1 after G1 | — | 6–8 |
| W2 | must | main grid, seed 1, 5 λ: RB presets (8–10), RB-decoupled (4), AP pw × LOR, BR 4 t × LOR, dispatcher-only RR/Po2/LOR/random/lowest_latency (SQ = LOR under exact in-flight accounting, D40) | ~170 | 17–21 |
| W2b | must | AC's RR/Po2 rows: AP pw{0.4,0.7,0.8} + BR {argmax,0.3} × {RR,Po2} × λ{12,30} | 20 | 2 |
| W3 | must | seeds 2–3 (per-seed prompt permutation + arrivals): near-frontier configs × λ{12,30} + headline × λ{6,18,24} | ~100 | 12 |
| W4 | must | ablations × λ{12,30} × 3 seeds (paired): LPT off, round 1/32, static+max-divide, XGBoost, four-arm isolation, prompt-blind, 50% labels, dead reckoning off, **normalization arm (curve + max-divide), per-term drops w_q=0 / w_c=0** (D41) | ~78 | 7 |
| W5 | must* | time-varying load (step 6→30→6) + gamma CV²=4 bursts: 5 systems × 2 traces × 3 seeds (*if G1 passes) | 30 | 3 |
| W6 | should | AP top-2 spillover heuristic ("simple heuristic does it") | 20 | 2.5 |
| W7 | must | topology A30 pool 3/5→5/3: 6 systems × λ{12,24} × 3 seeds | 36 | 4 |
| W7b | should | budget sweep (appendix) | ~30 | 2.5 |
| W8 | must | Gemma (redeploy + calibration): RB presets, RB-decoupled, AP 5 pw, passthrough (+BR if trained) × λ{12,24} × 3 seeds | ~80 | 11 |
| W9 | fill | full-grid seeds 2–3, λ30→12→24→18→6 | rest | — |
| W11 | good-to-have (last, only if valuable) | LPT key min / mean × λ{12,30} × 3 seeds (D41) | 12 | 1 |
| W10 | must | two-location + Drive pack before release | — | 4 |

Must work ≈ 80–85 h of 168 h, leaving buffer for reruns and W9.

**Order:** W0 → W1 → G1 → W1b (night 1) → W2 (λ30 + λ12 first) → W2b → W4 → W3 → W5 → W6 → W7 → W7b → W8 → W9 → W10.
**Cut:** A100-head check (every system already runs on an A100 head), vLLM-SR/RouteLLM, λ36.

## 6. Offline analyses (no cluster)

- **O1:** decision-cost microbenchmark.
- **O2:** E2E breakdown per tier/system/λ (scheduler, queue + prefill, decode); backs Fig 1 and the TTFT
  scope argument.
- **O3:** predicted vs realized latency: profiled vs constant vs learned.
- **O4:** router-only frontier from the lookup labels (KNN at 10/25/50/100% labels, prompt-blind, k).
- **O5:** preset sensitivity over the simplex.
- **O6:** check that served responses match the judged ones; re-judge the rest.
- **O7:** steady-state window and end-of-arrival backlog.
- **O8:** replay (pre-lease).

## 7. Paper (10 pages)

| § | pages | content |
|---|---:|---|
| 1 Intro | 1.0 | Fig 1: queued-at-chosen-tier share vs λ |
| 2 Motivation + positioning | 1.0 | Table 1; size ≠ speed table |
| 3 Design | 2.25 | Fig 2: full request flow with return path; Eq 1–2 with notation beside them; Alg 1 (round, LPT, dead reckoning); adaptive rule; profiled latency model; decode-dominated scope |
| 4 Implementation + fairness | 0.5 | what every system shares (the parity manifest) |
| 5.1 Setup | 0.5 | Table 3: fleet, prices, TPOT curves |
| 5.2 Frontier | 1.25 | Fig 3 (λ12, λ30; q–latency, q–cost); Table 4: equal-quality deltas + CIs |
| 5.3 Why joint | 0.75 | Fig 4: tier mix + mean/p99 vs λ for RB, RB-decoupled, AP |
| 5.4 Load change | 0.4 | Fig 5 (W5) |
| 5.5 Ablations | 0.5 | Table 5 (incl. learned XGBoost, prompt-blind, dead reckoning) |
| 5.6 Overhead + scale | 0.4 | Fig 6 (O1 + cluster overhead share) |
| 5.7 Generality | 0.4 | Table 6 (Gemma, topology) |
| 6–7 Related work, limitations, conclusion | 0.55 | TTFT/prefix scope, centralized, fixed deployment, budget ¶ |

The appendix holds datasets, the baseline training/knob table, a request walkthrough, per-λ tables, RR/Po2
tables, fairness evidence, O3–O7, and budget.

**How each likely reviewer attack is answered:**

| attack | answered by |
|---|---|
| llm-d/AIBrix + scorer | Table 1 + RB-decoupled |
| weak/unfair baselines | parity manifest + 4 dispatchers + per-rate knob tuning |
| lookup labels | shared by all systems + O6 |
| three weights | one preset across rates/families/topology + O5 |
| centralized | O1 |
| TTFT/prefix | O2 + scope |
| why not learned | O3 + ablation |
| one cluster | Gemma + topology + O7 |

## 8. Consistency (paper ↔ code ↔ experiments)

- **B1 (after plan approval):** resume the claims agent to write `specs/G_claims.jsonl` + summary from its
  draft (`~/.claude/plans/enter-into-the-plan-quiet-flask-agent-a931c3ae15efcf0a9.md`, 243 claims,
  78 conflicts). Its conflicts become the **do-not-repeat list**.
- **B2:** the new paper's ledger maps every design/setup sentence to a `routebalance/` file:line, a config
  key, or cell IDs. A presubmit gate fails on any unmapped sentence or config mismatch. All numbers are
  traced to cells (G4).

## 9. Timeline and gates

| date | work | gate |
|---|---|---|
| Thu 09-24 | plan approved; Drive upload; git re-init; S1–S2; start Gemma DeepEval on dev-gpu | — |
| Fri 09-25 | S3–S6; XGB retrain; prior index; AP-Gemma artifacts | — |
| Sat 09-26 | S7–S10; O8; O1; prereg_rules | — |
| Sun 09-27 | S11 smoke; squash; freeze tag | **G0 18:00**: gates green, BR/AP 3534/3534 replay, overhead gate, manifest frozen; else run must rows only |
| 09-28 10:00 | W0, W1 | — |
| 09-28 ~18:00–20:00 (lease +8–10 h) | G1 right after W1; then W1b sweep overnight (6–8 h); XGBoost retrain on dev-gpu 09-29 morning | **G1**: preregistered rule picks the latency default or the fallback thesis, computed as soon as the cells finish; latest 09-29 08:00 if W0 needs its worst-case budget |
| 09-29 → 10-01 | W2, W2b, W4, W3 | **G2 09-30 12:00**: λ12/λ30 panels + fairness numbers sane |
| 10-01 → 10-03 | W5–W8 | **G3 10-02 20:00**: all must cells valid |
| 10-03 → 10-05 | W9; results in two locations; release 10:00 | — |
| 10-05 → 10-09 | offline analyses, figures | **G4 10-09**: claims frozen, each tied to cells |
| 10-10 → 10-18 | full draft | — |
| 10-19 → 10-22 | 3 mock reviews (novelty, systems, statistics) | — |
| 10-23 → 10-28 | 10-page fit, presubmit, anonymization, anonymous artifact mirror | ledger gate green |
| 10-29 | submit (a day ahead) | — |

## Acceptance outlook (honest estimate, 2026-09-24)

| scenario | P(scenario) | P(accept) | P(clear accept) |
|---|---:|---:|---:|
| G1 passes: RB-new shifts tier mix with load and beats RB-decoupled on mean/p99 at ≤0.5% quality | ~50–60% | ~45–55% | ~20–30% |
| G1 fails: joint ≈ decoupled; fallback thesis | ~40–50% | ~15–20% | <10% |
| **overall** | | **~30–35%** | **~15%** |

Why the odds are below "NeurIPS borderline":
- The NeurIPS paper's strongest numbers do not survive a fair harness: the BEST-Route collapse, the
  own-best "dominance", and a latency term that never moves tier mix.
- MLSys reviewers probe harness fairness more closely.

Why they can exceed it:
- Better venue fit.
- If G1 passes, the evidence is stronger than NeurIPS'.

What moves it most:
1. The G1 result. **O8 moves to the first task after approval, ahead of the rebuild:** a CPU replay of the
   banked v3 decisions under the new latency model/normalization gives an early read on the tier shift
   within hours. It is only a proxy for G1, since replay cannot capture latency feedback.
2. RB-decoupled + the parity manifest (they close the novelty and fairness objections).
3. Generality + O1 scale.
4. Writing clarity (the 287j issues).

Main execution risk: a 3.5-day rebuild plus a single 7-day lease leaves no second window. G0 and G3 exist to
protect the must-rows.

## Verification

- `make check` green on the frozen tag; equivalence suites green vs v3 fixtures; O8 and O1 results recorded.
- dev-gpu smoke: every system serves 100 requests. The decision log shows the configured latency mode,
  TPOT values and normalization. The parity gate shows identical telemetry/hosting for all systems.
- Cluster: setup.sh on 13 nodes (fork SHA + wheel sha256); calibration writes TPOT curves into configs; one
  cell per system passes all gates before production; G1 is decided by `prereg_rules.json`.
- After the campaign: the manifest accounts for every cell; results exist in two locations before release;
  paper numbers regenerate from cells via `analysis/`.


## Integration follow-ups (from S3b–S6 reviews; owner decisions in DECISION_LOG D30–D40)

- S6 merge: drop shortest_queue from config/dispatch (D40); rb_decoupled calls schedule/score.py; config rejects
  budget.admission_filter=true outside routebalance mode; one unwrap_chat_prompt (S4 imports S6's or vice versa).
- S7b: one shared router/estimator executor (4 threads) and an explicit torch thread count for every system; routers and
  the estimator built off the event loop; on_progress wired identically for every system; last_assigned kept for the
  server's lifetime; a config swap must not drop queued or carried requests; per-request records log the dispatch-time
  feature vector (D38) and the served text (D39); pipeline mode may set pending_decode own_length (sweep only).
- S10: presets carry AP beta 5.0 / top_k (4 Qwen, 3 Gemma) and BR max_length 512; pin the upstream AvengersPro checkout
  (dfb60b1 + local datasets patch) and the embedder revisions; silence upstream tqdm; W0 reference generation + judge loop
  (D39); W1 fixed requests per concurrency level (D37); W1b sweep scope (D38); drain engines before each cell;
  setup.sh installs python3.10-dev and vllm==0.18.1.
- S5b: lpt key min / mean in config and schedule/assign.py (D41), reviewed like any step.
- S11 smoke: confirm continuous_usage_stats and the four metric names on real 0.18.1.
