# RouteBalance → MLSys 2027: pre-lease plan (updated 2026-09-25 02:30 UTC)

## Context

- **Target:** MLSys 2027, due 2026-10-30 20:00 UTC.
- **Lease:** 2026-09-28 10:00 → 10-05 10:00. Nodes: 3× c4130 V100, 8× d7525 A30, 2× d8545 A100.
- **Thesis and evaluation:** approved 09-24; details below and in `route-balance-paper/mlsys_2027/DECISION_LOG.md` (D01–D50).
- **Repo:** rebuilt from scratch, reviewed and published.

This plan covers **everything left before the full run on Monday**. The owner now also has:
- a temporary **A30 node** (d7525-10s10309) for about 20 h;
- the **V100 node** (c4130-110133) until the lease.

**Way of working (owner):**
- sequential work in the main thread; no parallel agents;
- per commit: one local reviewer plus a direct Codex call (companion script as a background shell), at most 2 rounds, correctness only;
- CloudLab nodes are touched only by `scripts/setup.sh`, `deploy.sh` and the drivers, never by manual installs;
- ask before any push.

## Where we are (done)

- **Repo:**
  - GitHub `main` has 2 commits: `159f63b` core, `27e610c` bench/analysis/tools/scripts/configs.
  - Local `dev` = main + `d450496` (tests, gates, Makefile).
  - Local `rebuild-steps` holds the 27 reviewed step commits.
- **Checks:** `make check` is green (689 unit, 88 contract, 326 gate tests). Equivalence vs v3 on dev-gpu: 73/74; the one failure is the loop-lag test under host load.
- **V100 validated:**
  - `setup.sh` engine role in 2 phases across the reboot, and the scheduler role: SELF-CHECK OK.
  - vLLM 0.18.1 serves 14B TP4 (TRITON_ATTN).
  - The probe: metrics, per-chunk usage and prompt tokens OK; TPOT 13.0/14.6/17.1/24.1 ms at concurrency 1/8/32/64; 64/64 texts identical solo vs burst.
- **S11 smoke:** 6 cells through the real render, `run_cell.sh` and gates, on 3 engines. All valid; parity green; loop lag p99 ~1.1 ms.
- **Gemma DeepEval judge (dev-gpu):** train split ~40% done, ETA ~12:30 UTC; test split ~17:00 UTC.
- **Campaign dry-run:** 358 configurations. Still missing: **52 owner values**, **19 artifact pins**, and 6 decisions that are made during the lease.

## What's left, in order

### Order and timing (owner answers, 09-25)

- **A: item by item.** Values stay `OWNER-MUST-SET` until you answer each one. The rehearsal uses the
  recommendations as **rehearsal-only** values, in untracked `local/`, as the smoke did.
- **W6, then W7b:** implemented on Saturday.
- **72B fallback:** `max-model-len 4096` is pre-approved (logged as D51). It applies to every system alike and is
  recorded in the engine manifest.
- **The A30 and V100 time:** the full dress rehearsal.

| when (UTC) | work |
|---|---|
| Fri 02:30–05:00 | B1 (disk rule), reviewed; A30 `setup.sh` + serve (D1). Meanwhile, sheet A item by item with you |
| Fri 05:00–12:00 | D2 dress rehearsal on A30 + V100 (the A30 goes ~22:00) |
| Fri 12:00–22:00 | B2–B6 and B8; C once the Gemma judge finishes (train ~12:30, test ~17:00) |
| Sat 26 | B7: W6, then W7b; the rest of C; the runbook |
| Sun 27 | E freeze, G0 18:00 |

### A. Owner decision sheet (blocks the render; recommendations to confirm or edit)

| # | item | recommended | why |
|---|---|---|---|
| 1 | telemetry `poll_ms` / `max_age_ms` / `timeout_s` (D30) | 50 / 500 / 1.0 | llm-d EPP default; ≥3× poll; a hung scrape ages out |
| 2 | RB `pending_decode` (D31) | own_length | a request decodes its own model's length |
| 3 | busy signal (D32) | engine waiting gauge; start at idle limits | v3's signal |
| 4 | budget none-admitted fallback (D33) | keep all | appendix only |
| 5 | S8 analysis choices (D34) | accept | built and tested |
| 6 | server `torch_threads` / `max_connections` / `request_timeout_s` (D44) | 8 / 2048 / 600 | 4 pool threads × 8 of 48 A100-node cores; above peak in-flight |
| 7 | bench `max_connections` / `request_timeout_s` / `max_send_lag_ms` | 4096 / 900 / 100 | a pool wait or late send invalidates a cell |
| 8 | dispatcher seed; fixed-32 round timeout | 1; 200 ms | v3's po2 seed; the adaptive rule pairs 32 with 200 ms |
| 9 | `g1_weights`; `decoupled_weights` | q50l30c20, uniform; q80l10c10, q60l20c20, q50l30c20, uniform | spans the quality-heavy to balanced presets |
| 10 | W0 / W8-ref concurrency; divergence sample | 16 per instance; 200 prompts at 64 per instance | ~20 min for 14,136 Qwen pairs |
| 11 | W1 / W8-cal levels; requests per level | [1, 4, 16, 32, 64, 96]; 128 | covers the λ30 running batch; the A30 rehearsal times it |
| 12 | W1b seeds; `sweep6-36` segments; sweep estimator | [1]; 60 min per λ; **knn-half index + the complementary half of the training prompts** | no self-matches, since the main index contains every training prompt |
| 13 | traces `step6-30-6`; `gamma-cv4` rate | 360 / 1800 / 360; 18 req/s | 3-min cells |
| 14 | Gemma tokenizer | `google/gemma-3-1b-it` at a pinned revision | one template across the family |
| 15 | D45 const+batch_aware control | run (6 cells, 0.5 h) | isolates the TPOT(b) slope |
| 16 | D47 monotone TPOT curves; D48 engine manifest; D50 items | yes; yes; apply counter yes, engine access log off, no CPU pinning | first rule; clean arm gate |
| 17 | held rows: W4-arm3; W6 spillover; W7b budget | drop; **W6 then W7b Sat (owner: both)** | arm3 has no configuration under D35 |
| 18 | `prereg_rules.json`, incl. G1 | tier-mix shift ≥ 5 pp; λ30 mean gain ≥ 10%; p99 ratio ≤ 1.0; quality tolerance 0.005; λ12 ratio ≤ 1.05; 2 presets | as proposed 09-24 |
| 19 | 72B TP4 on 4× A100-40GB (0.95, enforce-eager, 8192) fails at start | **pre-approved: max-model-len 4096** (D51) | untested on 0.18.1 |

### B. Code (Fri, sequential, on `dev`, each reviewed)

1. **`setup.sh` /mydata disk.** Found today: this d7525 has **no nvme0n1** (only sda–sde; v3's recipe assumed nvme0n1).
   - Rule: nvme0n1 if present; otherwise an explicit `--data-disk` from the fleet wrapper's per-node table.
   - It stays fail-loud: it lists the candidate disks and refuses a disk with partitions or mounts.
   - Test on the A30 node.
2. **D47:** `tools/calibrate_tpot_curve.py` writes a running max over b; `routebalance/estimate/latency.py` rejects a decreasing curve at load.
3. **D48:** `deploy.sh` records each engine's argv. `run_cell.sh` captures `engines_before/after.json` (argv hash plus `process_start_time_seconds` from each engine's `/metrics`). `gates/parity.py` compares them; `gates/preflight.py` checks nothing restarted.
4. **D50:**
   - `/v1/stats` gains an `applies` counter; `gates/arm.py` checks exactly one apply per cell.
   - `--disable-uvicorn-access-log` is added to every tier in `configs/engine_args_*.json`.
5. **W1b estimator (A-12):** `tools/prompt_set.py` gains a seeded 50/50 split of the training prompts. `knn-half` is built from one half; the sweep sends the other half.
6. **Prompt files:** regenerate `qwen_train_prompts.jsonl`, `gemma_eval_prompts.jsonl` and `gemma_train_prompts.jsonl` with `tools/prompt_set.py`, and check them against the sha256 pinned in `configs/scope.json`.
7. **Sat (owner: both, W6 first):**
   - W6: an AvengersPro top-2 spillover router (a router option, reusing `routers.py`).
   - W7b: a per-request `budget_cost` in `bench/` (client + dataset) and its scope rows.
8. **Loop-lag test:** re-run `tests/equivalence/test_router_loop_lag.py` alone on a quiet host.

### C. Artifacts (Fri–Sat, dev-gpu; pins for `configs/scope.json`)

- **Gemma judge:** validate the train and test outputs (no missing or invalid scores), then stop the judge server.
- **Gemma indexes:** build `knn.gemma` (DeepEval, k=50, from the repaired train file) and `knn-half.{qwen,gemma}`.
- **AvengersPro-Gemma:** build pw {0.25, 0.39, 0.53, 0.80} with v3's AvengersPro-Gemma builder. First check that it rebuilds the existing pw0.70 bitwise.
- **Lease artifact bundle** (md5-listed), plus a `pins.json` skeleton. The lease adds the TPOT tables (W1/W8-cal) and the XGBoost heads (after W1b).
  - `knn.qwen`, `br.qwen`, `ap.qwen` × 5;
  - the encoders;
  - the W0 reference corpora;
  - the datasets.

### D. Hardware rehearsal (A30 for the next ~20 h, V100 until the lease)

1. **A30 alone:**
   - `setup.sh --role engine` on the Ampere path (nvidia-open, a reboot only if the driver does not load) with the B1 disk rule, until SELF-CHECK OK;
   - `deploy.sh` for 3B, then 7B, with the lease engine args.
2. **Two-node dress rehearsal** (one mini version of lease day 1):
   - Fleet: 7B on the A30 plus 14B TP4 on the V100, the scheduler on the V100 node, real tier ids from `fleet_qwen.json`.
   - Render a mini campaign from the real presets, with the sheet's values and mini traces.
   - Run in order:
     1. W0-mini: 200 references plus a divergence sample;
     2. the judge loop round trip to dev-gpu (format check);
     3. W1-mini: the calibration driver on both tiers, producing a real TPOT table;
     4. G1-mini cells: RB, RB-static, RB-decoupled, and dispatchers;
     5. W1b-mini: a 10-min sweep, then the XGBoost retrain run on its records;
     6. a `run_scope.sh` resume after an interrupt;
     7. the mirror to a second location.
   - Record the durations, and set the A-10/A-11 values from them.
3. Stop every engine and the scheduler afterwards.

### E. Freeze (Sun 27, gate G0 18:00 UTC)

- **Render:** every owner value set, then `tools.campaign render` for W0–W2. Only lease pins and lease decisions may remain pending.
- **Checks and tag:** `make check` and `check-heavy` green on `dev`. Tag `lease-2026-09-28`; record the tree md5.
- **Publish:** carry the non-test changes to `main` and push only after asking.
- **Runbook:** `route-balance-paper/mlsys_2027/LEASE_RUNBOOK.md`, with the exact day-1 commands:
  1. `setup_fleet.sh` on the 13 nodes (72B launch tested first);
  2. deploy;
  3. W0, then W1, then the G1 computation from `prereg_rules.json`;
  4. W1b, then the retrain;
  5. the decision points.
- **Backups:** the artifact bundle and the render go to Drive.

## Day-1 contingencies

- A node fails `setup.sh`: reload it and run `setup.sh` again (R2). Never install by hand.
- 72B does not fit (item A-19): switch to the pre-approved `max-model-len 4096` for the 72B tier, the same for every
  system, and record it in `configs/engine_args_qwen.json` and the engine manifest. If it still fails, stop and ask. No
  other parameter changes (R3/R7).
- V100 is already validated. A30 and A100 get their first lease-hardware run in D and on day 1.

## Unchanged from the approved plan (09-24)

- **Thesis:**
  - if G1 passes: the joint (model, instance) decision beats router + balancer under load;
  - if it fails: two-level shaping (the fallback).
- **Lease order:** W0 → W1 → G1 → W1b → W2 → W2b → W4 → W3 → W5 → W6 → W7 → W7b → W8 → W9 → W11 → W10.
- **Dry-run sizes:** G1 16 cells; W2 130; W2b 20; W4 96; W3 100; W5 30; W7 36; W8 108; W9 ≤ 260.
- **Gates:** G1 on lease day 1 evening; G2 09-30 12:00; G3 10-02 20:00; G4 10-09.
- **Paper:** 10 pages as in the 09-24 plan (§7), claims ledger and B1b sourcing rules, offline analyses O1–O8. O1 moves after the lease.
- **Quality:** the judged score of the text actually served (D39).
- **Baselines:** least-outstanding with the LRU tie-break (D35, D40); LiteLLM-style lowest latency (D25).

## Verification

- **Each code item:** `make check` green, plus local and Codex review (≤ 2 rounds).
- **B1:** proven on the A30 node.
- **B3/B4:** proven by the rehearsal's gates.
- **The rehearsal (D2) passes when:**
  - every mini cell is VALID, with arm, zero_call and parity at exit 0 and the engine manifests unchanged;
  - W1 produces a monotone table;
  - the retrain reads the sweep records;
  - the judge loop round-trips.
- **G0:** the render has no pending owner values; `make check` and `check-heavy` are green; the freeze tag exists; the owner approves the push.
