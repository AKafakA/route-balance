# EuroSys revision — INTEGRATED PLAN v2 (2026-07-10) — window 2026-07-26 → 2026-08-06

**Status:** PLANNED, NOT EXECUTED. Author decision 2026-07-10: document only; sessions after
Jul 26 take over. Cluster RELEASED until 2026-07-26. EuroSys fall deadlines: abstract Sep 17 /
paper Sep 24, 2026 (CFP-verified).

**Supersedes** the v1 spec (2026-06-16, Exp A/B only). v1's two experiments are preserved below as
W3 (Exp A) and W4 (Exp B) inside a larger review-response campaign.

**Triggers (two external reviews of arXiv:2606.17949, tag `arxiv-2026-06-16`):**
1. Claude review (~45–60%): second family + one experiment where the learned predictor earns its keep.
2. **ChatGPT EuroSys-style review 2026-07-10 (~20% accept, weak-reject/borderline)** — RAW text
   verbatim: `2026-07-10_chatgpt_eurosys_review_RAW.md`. Its top-3: (i) reposition vs
   RouterWise/BOute as online live-state scheduling complementary to deployment planning;
   (ii) identical-estimator decoupled baseline + static prior as first-class competitor;
   (iii) second topology/family with long-running steady-state evidence.

**Audit verdict (5 read-only auditors + web, workflow `wf_1c2221cf-e94`; evidence with verbatim
quotes + file:line in `2026-07-10_review_audit/`):** the review is SUBSTANTIALLY CORRECT.
RouterWise (arXiv:2604.10907) and BOute (arXiv:2602.10729, MLSys'26) are REAL, both planners,
both uncited. EuroSys 2027 CFP verbatim requires arXiv-exposed submissions to use a different
title + system name (author accepts the risk and KEEPS RouteBalance — decision 1 below).
Review is WRONG that the PDF de-anonymizes (it is the anonymous build) and did not know the
"missing" Decoupled-SameEstimator baseline already exists as arm-2. Narrowed novelty — *online
per-request request-to-instance scheduling injecting live queue/decode state into model selection
for a FIXED deployment, complementary to deployment-time planners* — was NOT contradicted by
anything found (Llumnix = single-model rescheduling; MuxServe/QLM = offline/swap; planners ≠ online).

**Approved plan of record:** `~/.claude/plans/let-us-plan-everything-robust-snail.md` (2026-07-10,
user-approved). This handoff mirrors it and adds operational detail.

---

## 🔒 Author decisions LOCKED 2026-07-10 (do not re-litigate)
1. **KEEP the RouteBalance name/title.** ⚠️ Recorded risk: CFP says arXiv-exposed submissions
   "should have a substantially different title and use a different system/tool name". Author
   accepts. Do NOT re-raise. All other compliance fixes still land (placeholder header, AI disclosure).
2. **Generality = Gemma primary (W4) + pre-committed topology fallback (W5)**: abort to W5 if
   Gemma bring-up slips >1 day; W5 also runs Day 9 as stretch if W4 succeeds early.
3. **Selection-bias remedy (N6): compute BOTH offline** — (a) request-level val/test split
   (select presets on val half, report test half) and (b) bootstrap-with-selection (B=10k,
   re-select per resample) — author picks the headline after seeing magnitudes.
4. **Predictor framing rule (PRE-REGISTERED, state it in Methods):** if live state clearly wins
   ≥2 of 5 W3 discriminator scenarios → learned predictor stays a core contribution with the
   static prior as a first-class baseline; otherwise the paper leads with the
   "latency-at-model-selection-time is what matters" insight and presents the learned predictor
   as an optional refinement.

## ⛔ Before ANY run (experiment protocol — non-negotiable)
1. Read `~/.claude/shared-experimental-rules/{data_integrity_rules,universal_rules,portable_tools}.md`.
2. Preserve the EXACT deployed models to ≥2 locations + manifest BEFORE touching the cluster.
   Deployed latency predictors: `Block/models/cara/latency/deploy_{tpot,ttft,e2e}/` (Mar-27);
   md5s in `v2-exp/v2_deployed_md5.txt`. (`latency_predictors_apr28/` is a never-deployed trap.)
3. Smoke-test the EXACT config (N=10–100) before any long run.
4. NO silent param changes (cap, k, lr, max_len, gpu-mem, vllm params). On OOM/crash: STOP + report.
5. Quality of record = DeepEval `deepeval-llama3.1-8b-it_reference`; greedy temp=0 / freq_penalty=1.2
   in BOTH lookup generation and serving.
6. ⚠️ TTFT landmine: the deployed scheduler buffers responses (user-facing first token = E2E), the
   paper makes NO user-facing TTFT-advantage claim. W3 SLO metrics: **E2E-deadline goodput primary;
   serving-side TTFT (instance admission → first token) secondary, exactly defined.**

---

## TRACK NOW — CPU-only, no GPU needed (not yet executed; runnable any time, ideally before the window)
Edit target for ALL paper items = `route-balance-paper/eurosys_2027/main.tex` (+ `tables/`,
`references.bib`, `scripts/make_eurosys_figs.py`). Repo local-only; commits local. Line numbers
below verified by the 2026-07-10 audit (`2026-07-10_review_audit/tex_audit.md`).

**A. Positioning (N1) — do first**
1. Rewrite novelty sentence `main.tex:59` ("To our knowledge no LLM serving system jointly
   considers all three…") → narrowed online/live-state/fixed-deployment claim; expand the
   contrast paragraph at `main.tex:84` (planners vs online scheduler; "operate deployments that
   RouterWise/BOute produce").
2. `references.bib`: ADD RouterWise arXiv:2604.10907 + BOute arXiv:2602.10729 (MLSys 2026) —
   read their abstracts/intros first, characterize faithfully. FIX OmniRouter entry
   (`references.bib:526-542`): real = arXiv:2502.20576, KDD-25/SIGKDD Explorations Dec **2025**
   (currently mis-dated Dec 2026). Verify every other year-2026 bib entry (list in tex_audit.md
   P5 finding); never cite anything unverifiable.

**B. Reframe with existing data (N2/N3)**
3. Promote arm-2 (= review's Decoupled-SameEstimator: same KNN estimator, w_lat=0, reactive
   shortest-queue — `cara_serve.py:1624-1631`, `CARA_TIEBREAK`) and arm-4 (static tier prior —
   `cara_serve.py:1476-1497`, `CARA_STATIC_LAT`) to NAMED PRIMARY BASELINES in design+eval prose;
   report existing `_aggregate/P1_fusion_triangle.json` + `arm4_static_prior_result.json` numbers;
   soften "no decoupled combination achieves…" claims.
4. Draft "when does live state matter" subsection from existing `Q3_static_vs_live_probe.json`,
   `P3_nonstationary_3x3.json`, `P3sq_wq6_results.json`, `review_extras/{arm4_sq40,ov_arm1_sq40}.json`,
   `tier_removal/`; placeholders for W2/W3; pre-registered rule stated in Methods.

**C. Quality-bridge integrity fix (N7) — REQUIRED, our own data contradicts a sentence**
5. `main.tex:175` claims "the precomputed score is the score of the text actually served" — but
   `R1_served_vs_lookup.json` shows only ~13% exact served-text match under concurrency
   (RB_wq0.8_l12 match_total 13.02, diverged 86.98). Rewrite around the MEASURED evidence:
   `R2_servedtext_primaryjudge.json` — judge drift ≈ 0 on identical text; served-text re-judge
   preserves the cross-system gap (RB−BR4 +0.0237 CI [0.0102, 0.0371]). Add a compact
   "quality-bridge validity" passage/table. This is the honest AND stronger story.
6. NEW offline join script: `response_details.request_id` → `data/cara/scored/test_scored_filtered.jsonl`
   dataset tags → per-dataset quality table for headline cells (`aggregate_v2_grid.py` lacks this).
   Also: fraction-of-prompts-whose-selected-model-changes (RB vs BR4) from `all_model_scores`,
   to contextualize +0.013.

**D. Offline analyses from stored traces**
7. Steady-state (N5): new `steady_state_offline.py` over `experiment_output/v2_sweep_jun06/*/*.json`
   (fields verified present: num_prompts/completed/failed/duration + per-request e2el/ttft/itl/
   model/instance_id) → offered-vs-completed, per-tier utilization, approximate queue-slope,
   explicit OVERLOADED labels (λ30-class; vLLM-SR collapse cells). Surface EXISTING multiseed
   data (45 cells, seeds 2/3/4 × λ{12,18,24}) as repeated-run CIs. Write warmup/measure/drain
   protocol paragraph. PREPARE (do not deploy) the `cara_serve.py` patch adding wall-clock
   arrival/first-token/completion timestamps to `response_details`.
8. Selection bias (N6): compute BOTH remedies (decision 3). State the mitigating fact that the
   8 weight tuples were a fixed grid, not tuned.
9. Systems cost (N8): GPU-seconds/request per cell (instance_id → GPU count × e2el); surface the
   EXISTING gpuhour row (`cost_sensitivity_e8.json`: RB 1.38 / AP 1.38 / BR4 2.31 / PASS 3.03 —
   RouteBalance wins systems-cost too); state price-table provenance
   (`block/config/cara/scheduler_config_qn_minmax.json:87-112`).
10. Objective sensitivity (N9): FIRST correct the paper's normalization description — quality =
    minmax (`scheduler_config_qn_minmax.json:54`), cost/lat/bal = max-divide
    (`cara_serve.py:1565-1567`) — the review's "max-normalized" is only half right. THEN a
    re-scoring script over stored `all_model_scores`/predicted_* for minmax vs max-divide vs
    z-score + candidate add/remove → selection-agreement table (label as decision-level, not a
    serving counterfactual). Cite existing `cost_sensitivity_e8` + `uniform_mix_vs_lambda`.

**E. Mechanical paper fixes**
11. P1: LPT → "LPT-style heuristic", drop 4/3-guarantee framing (`main.tex:119`; also :61, :88).
12. P2: "3-simplex" → "weight simplex (a 2-simplex)" (`main.tex:61,128,142`).
13. P3: radar (Fig 3) is grayscale-ILLEGIBLE (verified by grayscale render) → replace with
    grayscale-safe small multiples or compact table via `make_eurosys_figs.py` from AGGREGATE.csv.
    Fig 2 is OK (markers survive grayscale).
14. P4: abstract rewrite — mechanism + confident Pareto claim, ≤3 headline numbers, drop the 23×
    defensive parenthetical (results stay in body). Standing rule: concision, never hedging.
15. C1 residuals: remove `Conference'17` running header + dummy ISBN/DOI (set \acmConference/
    \acmISBN/\acmDOI for the review build); ADD the AI-use disclosure section (CFP-mandated);
    verify content ≤ 12pp after all edits.
16. Codename guard: `scripts/make_eurosys_figs.py` contains internal-codename cell keys — scrub
    before ANY artifact/supplementary release; 0-codename scan on everything public.

**F. Public artifact consistency (P6) — route-balance repo, push only with user approval**
17. `REPRODUCE.md:227` `--k 50` → `--k 10` (deployed = k10, `scheduler_config_qn_minmax.json:76`);
    `REPRODUCE.md:106/110` + `reproduce/compute_bootstrap_ci.py:62` B=1000 → 10000 (re-run, confirm
    CI reproduces the paper's values); update single-seed note (:339) to cite multiseed data;
    refresh stale refs; add per-figure command+runtime+output+tolerance map (port from
    `cara_paper/PAPER_REPRODUCTION_MAP.md`).

**G. Window prep**
18. Sweep configs + orchestration for W0–W6 (smoke-gated chain); capacity-ramp script; Gemma
    deployment spec + generation/judging job graph (judge default: llama-3.1-8b primary —
    consistency with Qwen results and avoids Gemma-judges-Gemma self-preference; gemma-3-12b as
    robustness check); timestamp patch reviewed.

---

## TRACK WINDOW — Jul 26 → Aug 6 (~9 productive days + slack)
Capacity evidence: Jun-7 chain = 247 cells < 24h, 0 fails; plan at ~55–100 cells/day.
Must-have ≈ 440 cells — fits. **Priority if compressed: W0 > W1 > W2 > W3 > W4(→W5 fallback) > W5 > W6.**
W1+W2+W3 alone answer the review's three heaviest objections (N2/N3/N5); W4 is the accept-differentiator.
Add NOTHING else to the window.

- **Day 1 — W0 bring-up + capacity calibration [MUST]:** redeploy 13-instance/28-GPU testbed
  (CloudLab profile `route-balance/cloudlab/profile.py`); deploy + validate timestamp patch
  (numeric-equivalence vs a known cell); ~20 ramp cells → measured sustainable λ_max per system →
  defines the 60/80/95/105% capacity-fraction grid. (Prior anchor: RB sustained max ≈ 27.6 req/s.)
- **Days 2–3 — W1 steady-state grid [MUST, N5]:** 4 capacity fractions × 3 presets
  (wq0.8/uniform/wc0.8) × 3 seeds × 4 systems (RB, arm2, arm4, BR4-enhanced) = 144 cells,
  warmup/measure/drain windows, exact queue slopes from timestamps, OVERLOADED labels at 105%.
- **Day 4 — W2 decoupled full weight sweep [MUST, N2]:** 8 tuples × λ{6,12,18,24,30} × {arm2,
  arm4} = 80 cells + 3-seed replicates at each arm's best cell × 3 rates (~18) → arm2/arm4
  frontiers onto the Pareto figures.
- **Day 5 — W3 live-vs-static discriminators [MUST, N3 = old Exp A]:** arms {arm1 live, arm4
  static, arm2 reactive} × scenarios {gamma-burst b=0.3, square-wave 40/6, output-length-quartile
  shift, 72B kill+restore mid-run, SLO admission} × 2 rates × 2–3 seeds ≈ 60–90 cells.
  Metrics: E2E-deadline goodput primary, serving-side TTFT secondary (protocol note 6), recovery
  time, queue growth. Apply the pre-registered ≥2-of-5 rule (decision 4).
- **Days 6–8 — W4 Gemma second family [MUST, N4 = old Exp B]:** Day 6: deploy Gemma tier ladder
  on the same GPU classes; generate eval-workload responses per tier (greedy temp0/freq1.2);
  judge offline (proven ~24 calls/s 4×A100 pipeline) → lookup; train KNN + static priors.
  Days 7–8: reduced grid 3 presets × 4 capacity fractions × {RB, arm2, arm4, BR4} × 2 seeds ≈ 96
  cells + smoke gates. Success criterion: same policy structure wins on the second family.
  **FALLBACK (pre-committed): bring-up slips >1 day → abort to W5.**
- **Day 9 — W5 second topology [STRETCH; MUST if W4 aborted]:** capacity-balanced vs cost-skewed
  replica reallocation (same models/GPU classes → reuses all lookups + predictors), 3 presets ×
  3 fractions × 2 seeds × 3 systems ≈ 54 cells + weight-tuple transfer check (also answers N9-transfer).
- **Days 9–10 — W6 held-out-domain E2E routing [STRETCH, N7]:** ~1000 GSM8K prompts (absent from
  training data): generate on 4 Qwen tiers, judge offline, build lookup, serve 3 presets ×
  λ{12,18} × {RB, BR4} = 12 cells.
- **Days 11–12 — slack + freeze [MUST]:** rerun anomalies (~40-cell budget); rsync ALL outputs
  off-cluster (two locations); regenerate aggregates; record per-figure runtimes for the
  reproduction map; release cluster.

---

## Verification gates (before submission)
- Every new/changed number traces to a result file; re-run
  `cara_paper/smoke_test_apr_13/scripts/check_quality_consistency.py` against the revised tex.
- latexmk clean; content ≤ 12pp; RENDER the PDF to images (pdftoppm) and visually inspect: no
  placeholder header, grayscale-legible figures, no overflow (rendered-PDF rule — text extraction
  is not sufficient).
- 0 internal-codename scan on tex, PDF, figure scripts, any artifact bundle.
- Offline scripts spot-checked: reproduce a known published cell mean before trusting new columns.
- Public-repo fixes: corrected bootstrap command re-run and CI confirmed before commit; push only
  with explicit user approval.
- Final loop: independent reviewer-persona passes (weak-accept bar) on the rendered PDF;
  AI-disclosure present; double-blind check.

## Asset inventory (verified 2026-07-10; supersedes v1 list)
- Paper source of truth: `route-balance-paper/eurosys_2027/main.tex` (tag `arxiv-2026-06-16` =
  arXiv v1; later commits `205c988..df95587` = post-arXiv fixes). NOT `cara_neurips.tex` (stale).
- Arms: arm2/arm3 gates `cara_serve.py:1624-1631` (`CARA_TIEBREAK`), arm4 `cara_serve.py:1476-1497`
  (`CARA_STATIC_LAT`); results `_aggregate/P1_fusion_triangle.json`, `arm4_static_prior_result.json`.
- Served-text bridge: `_aggregate/R1_served_vs_lookup.json`, `R2_servedtext_primaryjudge.json`
  (+ scripts `r1_served_vs_lookup.py`, `r2_aggregate.py`, `r2_offset_diagnostics.py`).
- Dynamics: `P3_bursty_results.json`, `P3sq_wq6_results.json`, `P3_nonstationary_3x3.json`,
  `Q3_static_vs_live_probe.json`, `review_extras/{arm4_sq40,ov_arm1_sq40}.json`, `tier_removal/`.
- Multiseed: `multiseed/` 45 cells. k-sensitivity: `R5_k_sensitivity.json` (deployed k=10).
- Cost: `cost_sensitivity_e8.json` (incl. gpuhour row); prices `scheduler_config_qn_minmax.json:87-112`.
- Deployed predictors: `models/cara/latency/deploy_{tpot,ttft,e2e}/` (Mar-27, md5-manifest match).
- Quality lookups: `data/cara/{train,scored/test}_scored_filtered.jsonl` (14,919/3,634) — also on
  HF `asdwb/route_balance_model_estimator`. Gemma = JUDGE-ONLY (`gemma3_scored_full.jsonl`);
  NO Gemma serving/latency data exists — W4 is a full from-scratch pipeline.
- Reviews: `2026-07-10_chatgpt_eurosys_review_RAW.md` (raw, verbatim) + `2026-07-10_review_audit/`
  (per-claim verification evidence).
