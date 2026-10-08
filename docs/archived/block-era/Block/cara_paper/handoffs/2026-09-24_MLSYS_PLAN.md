# MLSys 2027 PLAN (created 2026-09-24, after the NeurIPS reject)

Owner decision 2026-09-24: **MLSys 2027 is the next target.** This file adds to
`2026-09-01_NEXT_CAMPAIGN_HANDOFF.md` (the fresh-rerun inventory); it does not replace it.
NeurIPS decision text + reviewer map: memory `neurips-2026-rejection`.

## 0. Venue facts (MLSys 2027 CFP, read 2026-09-24)

- Submission opens 2026-10-10 20:00 UTC; **full paper due 2026-10-30 20:00 UTC**. Reviews
  2027-01-18, response due 2027-01-21, notification 2027-02-28.
- **10 pages main text, 2-column**, `mlsys2025style.zip`. References not counted. Appendix is
  unlimited but uploaded separately, and reviewers need not read it, so every load-bearing
  result has to sit in the 10 pages.
- Double-blind (research track). arXiv versions are allowed. Unlike EuroSys, the CFP has no
  rename rule, so RouteBalance can keep its name.
- Starting point: `route-balance-paper/eurosys_2027/main.tex` (2-column, the arXiv build, 14 pp
  PDF). Port it to the MLSys style and cut to 10 pages. Content cuts need owner approval.

## 1. Timeline

| dates | work |
|---|---|
| Sep 24–27 | first-rule execution audit (§2), port any control-arm code, get lease confirmed |
| Sep 28 10:00 → **Oct 5 10:00** | CloudLab window **APPROVED, 7 days** (3× c4130, 8× d7525, 2× d8545): setup.sh + smoke, then the scoped rerun |
| Oct 5–30 | tables/figures on the fresh cells, port to MLSys style, 10-page fit, related work, presubmit checks |

## 2. FIRST RULE: execution-difference audit (must finish before the window)

What RB and the pipeline baselines *execute* differs in these known ways. Each has to be
shared with the baselines or covered by a control arm that is actually run.

| # | difference | where (Block v3 tree) | proposed handling |
|---|---|---|---|
| E1 | **Router-model placement.** RB's estimator and TPOT heads run batched through `run_in_executor`. BEST-Route's DeBERTa forward and AvengersPro's `route_queries_batch` run **inline on the event loop, once per request**. | `cara_serve.py:1311,2136,2361` vs `routers/best_route_4way.py:115-131`, `routers/avengers_pro.py:~266` | **Share:** baselines run their scoring off the loop in the headline (the June "enhanced" variants). The Block v3 tree has no switch for this; the public repo's `batch_scoring` flag (`route-balance/.../best_route_4way.py:32`) must be ported. |
| E2 | **Instance-state access.** RB reads a cached telemetry view with dead-reckoning updates. SQ and Po2 do a synchronous `/instance_stats` fan-out to every candidate on every request. | `_fetch_all_instance_stats`, `_select_via_pipeline` | **Control arm:** SQ/Po2 over the same asynchronously refreshed cache, without the dead-reckoning update. This is also how llm-d's scheduler reads metrics, so the arm doubles as the general-purpose-dispatcher baseline the AC asked for. Verify the RB cache path in Block before building it. |
| E3 | Scheduler on the V100 head, co-located with `14B_0` | handoff 09-01 §3 | already planned: A100-head robustness check |
| E4 | Baselines send `num_predicted_output_tokens = max_output_tokens`; RB sends the KNN prediction | `_select_via_pipeline` | vLLM fork uses it only in `/schedule_trace` (commit `01d19ee1c`). Verify that no stat a baseline dispatcher reads depends on it. |
| E5 | anything not yet listed | — | diff the RB path and the pipeline path end to end, request arrival → backend call, before the window |

Evidence already on file for E1 (`2026-06-07_v2_rerun_handoff.md`, Jun 10 entries): with
off-loop scoring, "λ18 collapse was ENTIRELY artifact" for BEST-Route, and the AvengersPro λ30
SQ upturn was also the event-loop artifact. RB's best latency still beat both enhanced variants
at every λ (2.07–2.39 s vs AP-enh 2.18–2.74 s, BR-enh 3.0–11.4 s). So the win holds under the
control, but at a smaller margin. The serial-deployment ratios (14.3×, 23×) cannot be headline
claims.

## 3. AC reasons → changes

1. **Novelty vs llm-d / AIBrix** (the negative reviewer; our rebuttal gave it one sentence).
   Related work gets a concrete positioning: which systems choose the *model* (routers,
   vLLM-SR) and which choose the *instance* for a named model (llm-d, AIBrix, Dynamo, SGLang
   router). Also cite **BOute (arXiv:2602.10729, MLSys'26)** and **RouterWise
   (arXiv:2604.10907)**. Neither draft cites them yet, and BOute is an MLSys paper. Check the
   current feature sets before writing claims about them.
2. **XGBoost justification.** Proposed: the measured static per-tier TPOT becomes the default
   latency prior, and the learned head becomes an optional extension. The Q1 static/learned
   arms in the 09-01 inventory give the evidence. **Claim change → owner decides.**
3. **General-purpose dispatchers.** Po2 is already in. Add the E2 cached-metrics SQ/Po2 arm
   (llm-d-style).
4. **Clarity (287j) and TTFT/prefix (gTfM: "It needs to be in the paper").** Walk through
   Fig. 1, the adaptive-batch rule, a dead-reckoning definition, LPT pseudocode, a baseline
   table, and a decode-dominated scope sentence. **Added text needs owner approval** (global
   review rule).

## 4. Open decisions for the owner

- D1: headline baselines use off-loop scoring (E1 share). The serial variant is kept as a
  labeled row, or dropped.
- D2: **DECIDED 2026-09-24: no llm-d baseline.** E2 is still an execution difference; handle it
  by sharing the async stats cache with SQ/Po2 (no new baseline), owner to confirm.
- D3: static TPOT as default estimator (claim change).
- D4: **DECIDED: lease approved, Sep 28 10:00 → Oct 5 10:00 (7 days, not 10).** Scope must shrink.

## 5. Owner direction 2026-09-24

Clean up the repo so **route-balance is the only execution repo**; full code audit (all known
 bugs resolved, no new ones); fix the 10-page paper structure; derive the rerun scope from it.

Drift measured 2026-09-24 (Block/block renamed cara→route_balance vs route-balance/route_balance):
81 .py files differ, ~2,130 changed lines (route_balance_serve.py 916, model_estimator.py 256,
best_route_4way.py 133, avengers_pro.py 120, ...); 88 files exist only in route-balance (June
env-gated drivers etc.); 0 only in Block. Block v3 holds the D-series fixes (e.g. raise-only
router fallbacks, D30); route-balance still has the silent `self._fallback or model_pool[0]`
fallback in best_route_4way.py but has the `batch_scoring` switch Block lacks.

## 6. Code location facts (checked 2026-09-24)

- **Block has NO git remote.** 564 commits on 4 branches exist only on openclaw:
  `v1exact-submitted` (0e9410e, NeurIPS-submitted code, 268 files under block/), `master`
  (f9d612e, June v2 incl. `batch_scoring`/enhanced baselines, 365 files), `rebuttal-jul25-window`
  (b5690b6, 384 files), `v3-campaign` (984ea18, HEAD: v1exact + D-series fixes, 262 files, no
  `batch_scoring`). master is an ancestor of v3-campaign, but v3's tree replaced block/, so June
  features live only in history. Bundle all branches to a second location before any cleanup.
- route-balance (GitHub private, `main` 671db9f, 294 files): every file under route_balance/
  exists in Block's tree or history except 2 public templates (`config/host_configs.template.json`,
  `config/route_balance/model_deployment.template.json`).
- Outside both package trees: campaign launchers/gates/analysis in
  `Block/cara_paper/handoffs/window_setup_work/v3_scripts/` (+ `head_scripts/`); vLLM fork
  `AKafakA/vllm@2894a98` on branch `cara_v_11` (on GitHub); AvengersPro upstream checkout
  `~/Code/llm/AvengersPro`; data/models gitignored (Drive packs).
- Executed core ≈ 10k lines: cara_serve.py 3,873; model_estimator 1,142; knn 498; xgboost 231;
  benchmark_serving 1,938 (+ end_point_func 279, dataset 116); instance clients 625;
  routers/dispatch/filters ≈ 1.8k. Block/tests: 15 files.
- Codex CLI 0.156.0 ready and authenticated (checked 2026-09-24).
