# NEXT-CAMPAIGN HANDOFF — full fresh reproduction for CR / MLSys (created 2026-09-01)

**For the session that takes this over.** Trigger to START: **(1) NeurIPS 2026 final decision is
out, AND (2) the CloudLab lease is approved + active.** Until both hold, there is nothing to run —
this is a plan, not a live campaign. Read `2026-08-04_V3_CLOSEOUT_HANDOFF.md` for the prior state;
Cara = RouteBalance (internal codename, never public). Repo: `/home/wd312/Code/llm/Block` (LOCAL git).

---

## 0. Why this campaign exists

The v3 S2 campaign is closed and backed up, but the **final paper version needs a single clean,
fresh end-to-end reproduction under one consistent config** — for camera-ready if NeurIPS accepts,
or for an MLSys 2027 submission if not. Author decision (2026-08/09): **re-run EVERYTHING fresh**
(v3 grid included), not reuse the old cells, so every headline number comes from one uniform run.

## 1. Trigger + booking

- **CloudLab reservation: 2026-09-28 10:00 → 2026-10-08 10:00 (10 days), Hyperion/Wisconsin**,
  13 nodes: **3× c4130 (V100), 8× d7525 (A30), 2× d8545 (A100)**. Submitted 2026-09-01, pending
  approval. ~7 days active compute estimated, 3 days buffer.
- Do NOT launch production until the author gives an explicit GO in chat (standing precondition)
  AND the lease rspec is saved as `s2_cluster/rspec_<new>.xml`.

## 2. THE config to reproduce (v3, verified 2026-08-04)

- **vLLM = git fork `AKafakA/vllm@2894a98722733c76cdb549c8b5e5068594f12bef`** (NOT PyPI; makes
  `transformers==4.56.0` work — the old 4.50.3 pin does NOT apply). Python 3.10.12 (system),
  torch 2.9.0, CUDA 12.8, driver 570.86.10, Ubuntu 22.04.5.
- cap **1024**, KNN **k=50 / corpus 14,919** (`deploy_v3_k50_14919`), cost **decode-only**,
  quality = **deepeval lookup**, `quality_norm=minmax`, seed 1, N=3534.
- Model→tier: **3B + 7B → A30**, **14B → V100**, **72B → A100** (`enforce-eager`). Full spec +
  deployed config MD5s: `window_setup_work/v3-exp/env/ENVIRONMENT.md` + `deployed_config/`.

## 3. ⚠️ THE key change this run — scheduler-head placement (CR item 4)

Last run the scheduler ran **co-located on node0 = a V100 (16 cores)** with the `14B_0` instance;
per-request scheduling overhead was ~180–500 ms (BEST-Route po2/sq heaviest, ~480 ms — the
per-request stats fetch, same path as its high-λ collapse). Open question: **is BR's collapse
partly a scheduler-CPU artifact?** Core RB-vs-AP is NOT threatened (AP has the lowest overhead;
RB carries more and still wins → more CPU only helps RB).

**Plan (do this run):**
1. Reproduce faithfully with the scheduler on the **V100 head** (the paper/rebuttal setting) —
   state it in the paper faithfully: "the scheduler ran on node0, a V100 provisioned by CloudLab."
2. Run the **A100-head (48-core) robustness check** on the vulnerable cells (BR-po2/sq + one RB
   + one AP anchor at λ12/18/24, a few seeds on collapse cells) — same fleet, only the scheduler
   node changes, identical for all systems. (Consider `taskset`-pinning the scheduler to a core
   subset so it isn't starved by the co-located 72B.)
3. **Decision rule:** conclusion holds → report as a robustness line; optionally make the final
   version all-A100 ("scheduler on the best-provisioned node"). Conclusion changes → do NOT keep
   the V100 numbers for consistency; correct the BR/collapse claim. Report whichever is TRUE.

## 4. Experiment inventory (the ~7-day plan) — full fresh rerun

**Qwen (Window A, ~2.5 days):**
- Main quality–throughput Pareto grid: 5 rates {λ6,12,18,24,30} × systems (RB 8-weight sweep,
  AvengersPro pw-sweep×3 disp, BEST-Route t-sweep×3 disp, Dispatcher-only, Passthrough) (~18 h).
- Q1 burst: 3 arms (control / static-zerocall / learned) × 2 presets (532, uniform) × 6 seeds
  × 3 rates (~9.5 h).
- Per-term ablations (no-LPT / no-filter / no-{quality,latency,cost,balance}-term / simple-SQ /
  simple-llumnix) + supporting studies (burstiness, filter-order, RSO density).
- Budget-control sweep (§270) + batching ablation (§292, LPT-off/adaptive-off/fixed-batch).
- Re-seeded latency cells (multiseed n=3–4). Scheduler-head V100/A100 check (§3). Topology W1-A
  (re-provision A30 pool 3/5→5/3, rerun systems).

**Gemma (Window B, ~1.5–2 days):** W1-B family serving grid — **reuse the existing Gemma corpus /
KNN index** (on the Drive pack; regenerating adds ~8–10 h). Deploy Gemma models on the fleet.

**EXCLUDE (author decision — not judged by reviewers): vLLM-SR and RouteLLM-mf baselines.**

Split rationale: one 70 h+ lease is impractical; two windows match the original S2/S3 shape.

## 5. Assets + where things are

- **Backups (Google Drive):** `v3_pack_2026-08-04.tar.zst` (3.8G — v3 cells + rebuttal_jul31_v3
  + docs, sha `0228344d…`); `v1_nips_rb_may04_20260729.tar.zst` (6.24G — v1 basis).
- **Reproduction apparatus (git):** `window_setup_work/v3-exp/` (README, REPRODUCE, EXPERIMENTS,
  ENVIRONMENT, deployed_config, head_scripts), `v3_scripts/` (driver + launchers + analysis).
- **v3 results (local):** `rb-exp/v3-exp/cells/` (kept for reference; the fresh run supersedes).
- **Prior versions public:** arXiv:2606.17949.

## 6. CR correction items (carry into the paper)

`2026-08-02_CR_CORRECTION_LIST.md` + `rb_nips_rebutal/POST_REBUTTAL_CR_NOTES.md`: CR-1 evaluator
zeros / quality basis, CR-2 cost axis, CR-3 KNN k=50 vs served k=10, CR-4 (NEW) scheduler-head
node + the A100 robustness check, plus the E2E-latency description fix. Disclosure rails: faithful,
corrected-at-CR with an explicit change note, never re-assert the two basis-dependent claims.

## 7. Operating rules (standing + a session-specific caution)

- **VERIFY before asserting.** The 2026-08 session made confident wrong claims (e.g. "model-less
  scheduler head" — the head actually ran `14B_0`); check the logs/config before stating any fact.
- No self-decided error fallbacks; no fixed-instance dispatch; never change experiment params
  (R3/R7) without author approval; OOM is reported, never retried with changes.
- Two-location before release; faithful disclosure of the actual hardware; decode-cost basis;
  Cara is an internal codename.
- Result dominance = per baseline's own-best-by-argmax(q×throughput), decode-basis. Current
  finding: **3 strict-dominate (λ18/24/30) + 2 near-parity (λ6/λ12)** vs AvengersPro; strict over
  BEST-Route + Dispatcher everywhere.
