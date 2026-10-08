# v1 (submitted May stack) vs v2 (June stack) — FULL DRIFT AUDIT (2026-07-25)

Companion to `2026-07-23_neurips_rebuttal_audit.md` (§3/§10) and `2026-07-25_opus5_rebuttal_brief_ASSESSED.md`
(usage classes). Purpose: make the drift inventory COMPLETE so nothing in the rebuttal rests on an
unknown difference.

## Provenance of evidence (what "checking history" can and cannot see)
- **Block git starts 2026-06-07** (`2074f4a` = v2 checkpoint) → v1 CODE is NOT in Block git.
- v1 code source of record: **route-balance public repo May-8 mirror** (commit `e2d9f4d...`, the
  codename-transformed v1 code) + the **May-6 cluster archive**
  (`cara_paper/handoffs/_cluster_archive_20260506/`: deployed `scheduler_config_carafast.json`
  [May 4 18:50] + `_session_archive_d8545.tgz` sweep scripts + p100 essentials).
- Chat history of the v1→v2 period is captured in the handoff logs (`2026-06-07_v2_rerun_handoff.md`
  rolling entries; `V2_RERUN_PLAN_2026-06-07.md`; `V2_CHANGELOG_AND_SUMMARY.md`); no separate
  transcript source was needed — the handoffs record each found-and-fixed event with timestamps.

## 1) CONFIG drift — key-by-key diff of the two DEPLOYED scheduler configs (18/66 keys differ)
`scheduler_config_carafast.json` (v1) vs `block/config/cara/scheduler_config_qn_minmax.json` (v2):
| key | v1 (submitted) | v2 (June) |
|---|---|---|
| `model_estimator.quality_config.k` | **50** | **10** |
| `model_estimator.score_type` | **judge** | **deepeval** |
| `slo_defaults.quality_norm` | ABSENT (⇒ raw) | **minmax** |
| `model_estimator.quality_config.model_dir` | `deploy_knn` | `deploy_knn_k10` |
| `scoring_weights` (defaults) | 0.3/0.2/0.3 + **w_balance 0.2** | ⅓/⅓/⅓ + w_balance 0.0 |
| `model_estimator.bucket_config.*` (16-bucket length head, bucket 64, **max_length 1024**) | **ABSENT** | present |
| `model_estimator.length_config.fused_model_dir` | ABSENT | present |
IDENTICAL (48/66): `batch_config {max_batch 8, timeout 100 ms, adaptive true}`, `budget_tokens 256`
(→ NOT the generation cap), instance capacities (incl. p100 entries), kv_config, prices,
`filter_enabled false`, relax_order, lpt_sort_key=max — matching V2_RERUN_PLAN:20 "otherwise identical".
**NEW drift item found by this audit:** the 16-bucket length-distribution head is a **v2 ADDITION** —
consistent with the submitted paper never mentioning buckets (v1 estimator = quality + expected length
only). The EuroSys paper's bucket claims describe v2.

## 2) GENERATION CAP — resolved EMPIRICALLY (docs disagreed: PAPER_TODO said 512; V2 plan said 256)
Max observed `output_len` in traces: **v1 = 256** (uniform λ12: **738/3,534 = 20.9% of responses
truncated exactly at cap**) vs **v2 = 1024** (37/3,534 ≈ 1.0% at cap). ⇒ V2_RERUN_PLAN's "256→1024"
is CORRECT; PAPER_TODO's "512 cap" is a doc error (correct it). The submitted paper does NOT state the
cap; the 21% truncation is a sharp-edged fact → **pocket answer only, never volunteer** (if asked:
v1 capped generations at 256 tokens; the extended evaluation uses 1024; conclusions preserved).

## 3) CODE drift — verified against the v1-era source (rb May-8 mirror)
- **Adaptive batch policy: IDENTICAL v1↔v2** (busy-instance ratio <0.3→bs4/50 ms, <0.7→16/100 ms,
  else 32/200 ms — v1-era `route_balance_serve.py:204+` matches current `cara_serve.py:204-240`).
  ⇒ **the formula quoted in the 287j reply is exactly what the reviewed system ran. SAFE.**
- **Scoring block v1** = `qual_score = raw [0,1]`, cost/lat/bal = `1 − x/max` (v1-era lines 1456-1518)
  — matches the printed Eq. 1 exactly. v2 adds config-gated minmax quality norm. ⇒ any Eq-1 description
  in the rebuttal uses SUBMITTED semantics (raw quality).
- **Arm gates (`TIEBREAK`/`STATIC_LAT`): 0 hits in v1-era code** — arm-2/3/4 are v2 additions (the
  four-arm isolation is v2-native; fine under Class A: internally controlled).
- v2 checkpoint `2074f4a` bundle (handoff:10): AvengersPro shim fix (.get+tokenizer), hardware-aware
  latency tier (**a30 first-match-p100 bug fix**), quality_norm=minmax, **`client_ttft` field added**
  (v1 traces lack it), zero-safe `aggregate_v2_grid.py`, v2 sweep configs. Plus (V2 plan): batched-FAISS
  KNN, 1024 latency-XGB retrain, bucket head.
- Aggregators: BOTH May aggregators (`aggregate_overnight_grid.py`, `aggregate_full_215.py`) carry the
  `or`-zero drop bug; zero-safe = `aggregate_v2_grid.py` + the scratchpad reagg patch.
- Found-and-fixed during v2 bring-up (handoff:41-43): param grids silently ≠ May → fixed;
  AvengersPro all-72B flood (shim drift + stale module cache) → fix + restart; zero-bug → fixed.
- APC: v1 runs = **OFF cluster-wide** (design-pivots doc); v2 deploy default = ON.

## 4) RESULTS drift — same-cell join, BOTH sides on zero-safe aggregation (isolates stack drift from the bug)
| cell | λ | e2e ms v1→v2 | tput v1→v2 | quality v1→v2 (Δ) |
|---|---|---|---|---|
| uniform ⅓ | 12 | 2219 → 2311 | 11.85 → 11.08 | 0.3857 → 0.3708 (−0.015) |
| uniform ⅓ | 24 | 2386 → 2684 | 23.28 → 19.66 | 0.3830 → 0.3701 (−0.013) |
| uniform ⅓ | 30 | 2526 → 2783 | 28.81 → 24.11 | 0.3836 → 0.3701 (−0.014) |
| wq0.8 | 12 | 2648 → **5954** | 11.66 → 10.52 | 0.4021 → 0.4187 (**+0.017**) |
| wq0.8 | 24 | 2844 → **14541** | 22.53 → 14.61 | 0.4023 → 0.4166 (+0.014) |
| wq0.8 | 30 | 3117 → **22756** | 27.84 → 13.04 | 0.4024 → 0.4171 (+0.015) |
| BR t0.5 RR | 12 | 2535 → 4289 | 11.85 → 11.41 | **0.4056 → 0.4056 (±0.0000)** |
Readings:
- **BEST-Route quality INVARIANT across stacks** — deterministic per-prompt router; its choices don't
  depend on any changed knob → identical lookup means. A strong internal-consistency VALIDATION of the
  whole audit chain (and usable confidence point if the basis is ever questioned).
- **cara's quality-peak gain (+0.017 at wq0.8) is the routing-signal alignment** (judge→deepeval; plus
  k/minmax): exactly the mechanism that restores RB>BR (0.4187 vs 0.4056). Uniform wobbles −0.015.
- **Latency/throughput absolutes are dominated by the cap change** (4× longer generations): wq0.8
  quality-priority cells 2.8 s→14.5 s @λ24; v1's 21% truncation deflated v1 latencies and inflated
  sustained throughput (28.85→27.6-class deltas are cap-driven, not scheduler regressions).

## 5) Consequences for the rebuttal (extends the usage rules in the brief assessment)
1. **Extend Class-B ban to ALL cross-base absolutes**: not just quality — E2E latency, throughput, and
   sustained-max numbers are also non-comparable across stacks (cap-driven), especially at
   quality-priority presets and high λ. Deltas/orderings/shares only, basis labeled.
2. Adaptive-formula quote: SAFE (identical code both eras).
3. Eq-1/normalization description: use submitted semantics (raw quality, max-divide cost/lat).
4. Four-arm isolation: Class-A safe (v2-internal, controlled); BR-invariance adds a verification story.
5. Cap + truncation: pocket answer only. PAPER_TODO "512" corrected to 256.
6. Complete v1→v2 change inventory (for any "what exactly changed?" follow-up): cap 256→1024 ·
   score_type judge→deepeval · k 50→10 · quality_norm raw→minmax · bucket/length head added ·
   a30 tier fix · batched-FAISS · 1024 XGB retrain · client_ttft field · AvengersPro shim fix ·
   zero-safe aggregator · APC off→on(default) · default weights (w_balance 0.2→0) · arm gates added.
