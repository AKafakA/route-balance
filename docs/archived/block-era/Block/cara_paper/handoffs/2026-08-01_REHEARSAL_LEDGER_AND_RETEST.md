# S2 rehearsal — defect ledger and re-test procedure

**Purpose.** The author reloads the cluster and re-runs from scratch. This is what to run,
what to expect, and what was wrong before, so the re-test can be *checked* rather than
trusted.

**Rehearsal fleet:** 12× A30 (`asdwb-311959`), 8×3B + 4×7B — **not** the paper topology
(3B×3 + 7B×5). Every cell is marked `_REHEARSAL`; `dominance.py` refuses the directory
without `--rehearsal`. **No number below is a result.**

---

> ⚠ **This table is REHEARSAL-shaped.** It records what the phases printed on the 12-host A30
> lease. On Monday's reprovisioned 13-host fleet the same phases print **13/13** hosts and
> `produced 39 cell(s)` per rate, and the self-test line is the count above. A number here that
> does not match Monday's output is not automatically a fault — check the shape first.

## A. Re-test procedure

```bash
cd cara_paper/handoffs/window_setup_work/v3_scripts

# Fresh lease: save the portal rspec to s2_cluster/ first, then
S2_MODE=rehearsal S2_TOPOLOGY="3B=8,7B=4" ALLOWED_HW=a30 ./s2_driver.sh

# Monday, full fleet — no topology override, all four tiers:
S2_MODE=production ALLOWED_HW=a30,v100,a100 ./s2_driver.sh
```

`--status` shows phase state; `--from <PHASE>` resumes; every phase is idempotent.

**State is scoped per mode** (`DRIVER_STATE.production` / `DRIVER_STATE.rehearsal`), so a
completed rehearsal cannot make a production run skip its phases — which it did before
2026-08-01, reporting all eleven "done". No manual `rm` needed; on a genuinely fresh lease
delete the file for that mode only.

**Phase order matters and is now correct:** `P0_CONFIG → P0_HOSTS → P0_REACH → P1_SETUP →
P2_WEIGHTS → P0_TREE → P2B_ARTIFACTS → P3_DEPLOY → P4_GATES → P5_Q1 → P6_GRID`.
`P0_TREE` **must** precede `P2B_ARTIFACTS` — the reverse order wipes the staged artifacts.

### What each phase should print if it is working

| phase | expected proof |
|---|---|
| P0_CONFIG | `COUNT CHANGED — experiment parameter` only when a topology override is given; 0 LAN addresses; `scheduler config: k=50` |
| P0_REACH | `12/12 hosts answer ssh (probed 12)` — the probe count must equal the host count |
| P1_SETUP | `VERIFY-OK` ×12; vLLM SHA `2894a9872` checked **by execution** |
| P2_WEIGHTS | `WEIGHTS-OK` ×12 — then check `du` yourself, not just the counter |
| P0_TREE | `ONE TREE verified: 12/12`, ~148 files |
| P2B_ARTIFACTS | `declared tiers == booster files`, `artifacts staged 12/12`, `RULE-0 pins recorded 12/12` |
| P3_DEPLOY | `N/N declared backends are serving the declared model` — probed at `/v1/models`, not inferred |
| P4_GATES | self-test `67 cases, 40 MUST-FAIL, 0 wrong`; static gate matches its baseline |
| P5_Q1 | `same-config baseline for this fleet: NN%`, then `pilot passed: arms differ` |
| P6_GRID | per rate: `produced 12 cell(s)` → `pulled 12` → Phase E' verdict → `[done] P6_GRID_<λ>` |

### Three things to check yourself, because a green run can still be wrong

1. **`pgrep -cf "[b]enchmark_serving"` on the head node must be 1.** More than one means
   contention, and contended cells *pass* RULE-0 with latency inflated up to **5.7×**.
2. **Cell dates must not straddle an interruption.** `dominance.py` fails on a >12 h span,
   but a 3 h span containing a restart is still mixed provenance.
3. **`SKIP-INCOMPATIBLE` is expected on a partial fleet** — BEST-Route and AvengersPro rank
   all four tiers. On Monday's full fleet it should never appear; if it does, a tier is missing.

---

## B. Defect ledger — everything the rehearsal found

Grouped by class, because the classes matter more than the instances.

### B1. Writer and reader disagreeing about location (7)

The dominant class. Each was fixed as an instance before the rule was named.

| # | defect | would have caused |
|---|---|---|
| 1 | `tpot_gate` read `tpot_log.jsonl`, which nothing writes | gate passes vacuously |
| 2 | `lib_serve_v3.sh` hardcoded `V3TREE=/users/asdwb/Block_v3` | **cells from the unhashed tree while gates verify the other** |
| 3 | `--exclude 'data/'` matched `block/data/` at any depth | 3 `.py` files absent from every node |
| 4 | `rsync --delete` removed staged artifacts — the files were in **neither** the exclude list nor a protect rule (**corrected 2026-08-02**: the old shorthand "exclude ≠ protect" was wrong; `--exclude` *does* protect from `--delete` unless `--delete-excluded` is used) | scheduler refuses to start |
| 5 | same, for `S2_PINS.env` / `TREE_MANIFEST.md5` | RULE-0 refuses all 273 cells |
| 6 | liveness read `cara_serve.log`; the marker is a `print` in `cara_<mode>_<ts>.log` | **grid completes with zero RouteBalance rows** |
| 7 | analysis read `s2_p1_REHEARSAL`; cells land in `s2_p1_l<λ>_REHEARSAL` | Phase E' finds no cells, forever |
| 7b | `rsync` does not expand `$HOME`; `ssh` does | count works, pull fails, same string |
| 7c | cap written top-level as a string; read as `max_output_len` | cap check silently blind |

**Rule:** every path shared by two components must come from ONE definition.

### B2. A step that neither fails nor completes (3)

| # | defect | would have caused |
|---|---|---|
| 8 | `grep -c X \|\| echo 0` → `"0\n0"`, arithmetic in `die` fails, **`die` never runs** | failed phase continues silently |
| 9 | scheduler a direct child of the runner → `do_wait` forever | **phase hangs; window lost** |
| 10 | a rate with zero cells marked complete | resume skips it; grid ends silently short |

`setsid` alone does **not** fix 9 — it forks only if the caller is already a process-group
leader. `setsid --fork` does.

### B3. State leaking across cells (2)

| # | defect | would have caused |
|---|---|---|
| 11 | `/v1/config` **merges**, so a static prior survives into the next "learned" cell | **all 24 Q1 cells silently static** — the v2/July/v3 conclusion from a different cause |
| 12 | orphaned runners after a killed driver | **5.7× latency inflation; cells still pass** |

11 was invisible because `EFFECTIVE_CONFIG` was emitted once at startup, never per POST.

### B4. A gate asserting the wrong thing (3)

| # | defect |
|---|---|
| 13 | P100 **declared** in `xgboost_metadata.json` after its `.xgb` was purged — predictor raises, **no cells at all** |
| 14 | arms-gate baseline 95.5% was calibrated on **four** tiers; measured 98.9–99.2% on two |
| 15 | cap check compared `p99_output_len`, a measured percentile, not the declared cap |

14's fix **tightens** the gate: it would have caught defect 11, which 95.5% did not.

### B5. Fleet/router mismatch (1)

| # | defect |
|---|---|
| 16 | BEST-Route and AvengersPro rank all four tiers; on a partial fleet 706/3534 requests error with an **empty model name** |

---

## C. What is verified working, on hardware

Provisioning · weight prefetch · ONE TREE (12/12, survives re-sync) · artifact staging with
tier assertion · deploy with per-backend liveness probe · fail-closed scheduler (refuses to
start with no estimator) · KNN at **k=50** with distinct per-model DeepEval scores ·
**chat-unwrap identical to 6 dp** · decision-log arm proof (static = one `lat/exp` ratio per
tier at exactly the requested ms/tok) · RULE-0 with generated pins · Q1 with across-seed CIs ·
Phase E' auto-trigger, both DOMINATED and NOT-DOMINATED branches.

## D. Not verified, and cannot be until Monday

Four-tier behaviour: the 14B/72B heads, BEST-Route and AvengersPro cells, and any dominance
comparison against those baselines. The rehearsal fleet cannot produce them.

## E. Author decisions outstanding

1. `/v1/config` — replace rather than merge for arm-defining keys? (left as merge; other
   runners rely on config-file defaults persisting)
2. `deploy_cara.py:336-337` hardcodes `~/Cara` → Monitor fails on every node (telemetry only)
3. `llm_judge_scorer.py:581,991` fabricated `is_harmful` labels — raise or warn?
4. per-phase timeout in the driver, to catch B2's "neither fails nor completes" class
