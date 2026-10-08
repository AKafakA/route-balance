# V3 S2 CAMPAIGN — CLOSEOUT HANDOFF (2026-08-04)

**Read this first if you are resuming the RouteBalance/Cara V3 work cold** (new model, post-
compaction, or next month). It is the single source of truth for where things stand. Repo:
`/home/wd312/Code/llm/Block` (LOCAL git, on openclaw VPS). Cara = RouteBalance (internal
codename, **never public**).

---

## TL;DR — status

- **The S2 production window is CLOSED.** CloudLab lease `asdwb-312083` was released 2026-08-04
  ~10:12Z. **No cluster is live.** A new lease is needed to run anything.
- **The deliverable is DONE and audited**: grid (5 rates) + Q1 (static-vs-learned, 4-tier) +
  BR-collapse fill + λ6 retune. All results two-location (VPS + the backup below).
- **Rebuttal is OVER**: OpenReview posting phase closed (was Aug 04 11:59Z). Corrections can only
  land at **camera-ready, if accepted**. The accept/reject decision now rests on the posted record.

## The result (what the paper claims, corrected + author-confirmed 2026-08-04)

**Dominance vs AvengersPro (AP), decode-cost basis, each baseline at its own-best-by-argmax(q×tput):**

- **λ18/24/30: RB strictly dominates AP** (4-axis) — at high load AP's best knob collapses to a
  low-quality point RB beats outright.
- **λ6/λ12: near-parity** — AP's best stays high-quality (pw0.70); RB (0.6,0.2,0.2) matches its
  quality within ~1.5–1.7% while winning cost (−21%), mean (−13/−15%), p99 (−21/−29%).
- **RB strictly dominates BEST-Route + Dispatcher-only at every rate.**

This is **"3 strict-dominate + 2 near-parity"** — NOT the old loose "4 rates dominated" (λ12 is
near-parity, corrected). Matches the posted frontier framing ("reaches AvengersPro's quality band
at lower cost") → **no posted-claim contradiction.** Details + numbers:
`2026-08-04_BRFILL_AB_RESULT.md`, `window_setup_work/v3-exp/EXPERIMENTS.md`.

Caveat: BEST-Route dominance at λ6/λ12 is **cost-basis-sensitive** (true decode-only, false
full-cost). Decode is the basis of record (`neurips_2026.tex:221`). State it wherever used.

## OPEN DECISIONS / NEXT ACTIONS (for whoever resumes)

1. **CR correction note (the live open item).** Framing DECIDED by author: disclose
   **"evaluator-bug fix + serving-cap enlargement (256→1024)"** — NOT "evaluator only". Verified
   config lineage (see below): rebuttal→new-grid differs ONLY by cap; the evaluator fix + k10→k50
   are already in the *posted* rebuttal. Core frame survives. Rails: never re-assert the two
   basis-dependent claims; truthful-if-asked.
2. **⚠️ UNFINISHED VERIFICATION (do this before finalizing the CR note):** confirm the *posted
   rebuttal* and the *new grid* were **aggregated with the same fixed script** (`aggregate_v2_grid.py`
   fixed vs `aggregate_overnight_grid.py` still-buggy `or`-zero bug, `project_aggregator_zero_score_bug`).
   A serving-config diff can't see this — it's analysis-time. If they used different aggregators,
   the quality numbers may not match and the note needs to say so.
3. **Full paper refresh (later):** re-render all tables/figures on the new grid at cap1024. The
   quality axis is cap-invariant (KNN lookup), so quality numbers = the posted-rebuttal quality;
   only latency/cost move (cap).
4. **Optional:** a cap-256 re-run on a future lease would make "evaluator-only" literally true (if
   the author prefers that over disclosing the cap change). Needs a new lease.
5. **λ6/λ12 retune:** eff-0.70 tested, does NOT dominate (quality ~linear near λ6). No further
   retune worth running — near-parity is the honest ceiling. λ<12 fills are pre-registered off.

## Config lineage / provenance (the load-bearing facts)

| knob | submission (v1) | rebuttal (Jul 29-31) | **new grid (v3, this campaign)** |
|---|---|---|---|
| serving cap | 256 | 256 | **1024** ← the one config change |
| KNN k | 10 (served; paper claimed 50) | 50 | 50 |
| KNN corpus | 14,966 (served) | 14,919 | 14,919 |
| quality | deepeval lookup, minmax; **buggy aggregator** | deepeval, minmax, chat-template fix | deepeval, minmax |
| cost | decode-only | decode-only | decode-only |

- **vLLM = git fork `AKafakA/vllm@2894a98722733c76cdb549c8b5e5068594f12bef`** (NOT PyPI; makes
  transformers 4.56.0 work — the old 4.50.3 pin does NOT apply to V3). Python 3.10.12 system;
  CUDA 12.8; driver 570.86.10; Ubuntu 22.04.5. Full spec: `v3-exp/env/ENVIRONMENT.md`.

## Repo restructure 2026-08-04 (READ — layout changed)

- **`Block/` is now STANDALONE:** `Block/data` and `Block/models` are **real dirs** (were symlinks
  into `rb-exp/v1-exp`). Ownership reversed via `mv`; `v1-exp/{data,models_full}` now symlink back
  to Block. `data/`+`models/` stay gitignored. Block = code + docs + scripts + setup + data + models.
- **V3 results MOVED out of Block → the v3 pack `rb-exp/v3-exp/`:** `cells/` (all S2 results),
  `rebuttal_jul31_v3/` (the corrected v3 re-run, moved out of v1-exp), `docs/`, handoff copies.
- **`v1-exp` DELETED 2026-08-04** (freed ~8.6G). The v1 basis (submitted PDF, 243-config grid,
  provenance) is preserved on the Jul-29 Drive pack `v1_nips_rb_may04_20260729.tar.zst` (6.24GB) —
  traced + verified (closegate zstd-t at creation; live KNN md5 `b9c5c606` still matched recorded).
  To restore: download that pack, `tar --zstd -xf`. Block is unaffected (owns its own data/models).

## Where everything is

- **Results (cells):** `rb-exp/v3-exp/cells/` — grid `s2_p1_l{6,12,18,24,30}` (171 valid + 24
  `.INVALID` = BR collapse record), `s2_q1` (pilot 5), `s2_q1_PROD_zerocall`, `s2_q1burst_PROD`
  (156, Q1 deliverable), `brfill` (13 B + 1 A + 1 λ6 retune), `s2_q1_RETIRED_oldimpl_static_local`.
- **THE V3 BACKUP (Drive):** `v3_pack_2026-08-04.tar.zst` (cells + rebuttal_jul31_v3 + docs),
  SHA in `.sha256`. The v1 basis is separately on Drive as `v1_nips_rb_may04_20260729.tar.zst`.
- **Reproduction apparatus + docs:** `window_setup_work/v3-exp/` (git-tracked): `README.md`,
  `REPRODUCE.md`, `EXPERIMENTS.md`, `env/ENVIRONMENT.md`, `manifest/AUDIT.md`,
  `deployed_config/` (MD5-verified == the live config), `head_scripts/` (16 launchers).
- **THE BACKUP (Drive):** `v3_complete_backup_2026-08-04.tar.zst` — apparatus + ALL results +
  this handoff + repro docs. SHA-256 in `<name>.sha256`. This is the full restart kit.
- **Operating docs:** `2026-08-03_REBUTTAL_CLOSEGATE.md` (§1a current status, §3b endgame),
  `2026-07-30_V3_CAMPAIGN_PLAN.md` (§0), `2026-07-30_v3_execution_log.md` (chronological).
- **Rebuttal posts:** `cara_paper/rb_nips_rebutal/` (POST3, CORRECTED_Q1, CR notes).

## How to reproduce

Follow `v3-exp/REPRODUCE.md` end-to-end: provision 13-host fleet (3×V100, 8×A30, 2×A100) →
install `env/*_pip_freeze.txt` + the vLLM fork → deploy Qwen 3B/7B/14B/72B per
`deployed_config/model_deployment.json` → start `cara_serve` (`--scheduling cara` for RB,
`pipeline` for baselines) → run `head_scripts/_run_p1_l*.sh` (grid) + `_run_qb_prod.sh` (Q1) →
analyze with `v3_scripts/{expected_cells,dominance,burst_analysis,gate_selftest}.py`.
Gate self-test must read **162 cases / 89 MUST-FAIL / 0 wrong**.

## What was purged this closeout (safe)

- `*_REHEARSAL` cell dirs (2.9G, superseded by production) — removed.
- `smoke_20260802_remote_pull` (18M smoke test) — removed.
- v1-exp / v2-exp: not present locally (already backed to Drive elsewhere).
- KEPT: the 24 `.INVALID` (collapse provenance) and `s2_q1_RETIRED_oldimpl` (retired+preserved).

## Hard rules (carry forward)

Never change experiment params without author approval (R3/R7). No self-decided error fallbacks;
no fixed-instance dispatch. Decode-cost basis. Cara is internal codename. Cells preserved ≥2
locations before any disk cleanup. Verify deployed config/code, don't assume.
