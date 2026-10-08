# S2 status and machinery — consolidated, 2026-08-02T00:00Z

Single reference for where the campaign stands and what has been built. Supersedes nothing;
`2026-07-30_V3_CAMPAIGN_PLAN.md` §0 remains the running log, `2026-07-31_GATES.md` the gate
detail, `2026-08-01_REHEARSAL_LEDGER_AND_RETEST.md` the re-test procedure.

**Window:** S2 opens **Aug 03 01:00 BST**, runs 35 h to Aug 04 12:00. Author posting closes
**Aug 04 12:59 BST** — 59 minutes after the lease ends.

---

## 1. Fleet — and a correction

**The lease is 12× d7525 (A30) ONLY.** The rspec lists twelve nodes and nothing else.

⚠ **Correction, recorded because I nearly reported the opposite.** A probe appeared to show the
jul25-era V100 (`c4130-*`) and A100 (`d8545-*`) hosts as reachable. They are not:

```
c4130-110433    Permission denied (publickey)
c4130-110233    Connection timed out
c4130-110133    Connection timed out
d8545-10s10505  Permission denied (publickey)
d8545-10s10501  Permission denied (publickey)
```

The check matched `*<hostname>*` against command output, and the string
`asdwb@c4130-110433...: Permission denied` **contains the hostname** — so an error read as a
success. Same self-match class as the orphan-gate defect; fifth occurrence, all in hand-rolled
checks rather than in `runtime_gates.py`, which guards against it correctly.

**Consequence:** the four-tier path cannot be exercised on hardware before Monday. It has been
verified by construction instead — §4.

---

## 2. What was produced

| artefact | count | state |
|---|---|---|
| Q1 cells (6 seeds × 2 rates × 2 arms + 2 controls) | 26 | 0 INVALID, md5-verified node↔local |
| Grid cells (7 rates × 12 configs) | 84 | 0 INVALID, md5-verified node↔local |
| Local mirror | 2.9 G | per-rate `.md5` manifests |

**All rehearsal cells.** 2-tier fleet, 8+4 topology (not the paper's 3+5), marked `_REHEARSAL`;
`dominance.py` refuses the directory without `--rehearsal`. **No number here is a result.**

### Q1 — static vs learned arm, 6 seeds, 0 failed requests

**Metric: `mean_e2el_ms` — mean end-to-end latency.** (Earlier drafts headed this table "TPOT";
that was wrong. Mean TPOT for these same cells is 17.4–19.0 ms, not ~2.7 s. The numbers were
always mean E2E; only the label was wrong. Corrected 2026-08-01 — see [[ttft-metric-is-cara-clocked]]
for why a mislabelled latency metric is a standing hazard in this project.)

| λ | learned | static | Δ |
|---|---:|---:|---:|
| 12 | 2751.6 ms | **2715.3 ms** | −36.3 ms (−1.3%) |
| 24 | 2902.4 ms | **2822.6 ms** | −79.9 ms (−2.7%) |

**The static prior is faster than the learned head, at both rates.** The arms are *paired by
seed* (same seed ⇒ same arrival stream), so the paired test is the one that fits the design:

| λ | mean paired Δ (learned − static) | 95% CI | t(5) | verdict |
|---|---:|---:|---:|---|
| 12 | +36.3 ms | [+17.0, +55.6] | 4.84 | significant (crit 2.571) |
| 24 | +79.9 ms | [+61.9, +97.9] | 11.39 | significant |

All 6 seeds differ in the same direction at both rates (sign test p = 1/32 = 0.031 each).

⚠ **Correction to an earlier claim.** This table previously read "CIs disjoint at both rates".
That used **z = 1.96**, which is wrong for n = 6; the small-sample multiplier is t(0.975,5) =
**2.571**, 31% wider. Under t, the λ12 *independent* CIs **overlap** (learned [2730.1, 2773.1] vs
static [2697.8, 2732.8]); λ24 stays disjoint. The paired test above is significant at both rates,
so the conclusion is unchanged and in fact stronger — but the stated justification was not sound
and would not have survived a reviewer. See §5 open decision 5 on the same multiplier in
`burst_analysis.py`.

Two tiers on one GPU class is the weakest case for a learned predictor — almost no heterogeneity
to predict — so this is *not* an answer to reviewer gTfM. But it is the first time the comparison
has produced a number with error bars, and the direction does not favour the learned head.

### Grid — Phase E′ closed all 7 rates unattended

| λ | verdict | Δq | Δcost | Δmean | Δp99 |
|---|---|---:|---:|---:|---:|
| 30 | DOMINATED | +0.0136 | −4.0% | −116 ms | −687 ms |
| 24 | DOMINATED | +0.0077 | −9.3% | −536 ms | −3687 ms |
| 18 | DOMINATED | +0.0109 | −8.9% | −524 ms | −3664 ms |
| 12 | DOMINATED | +0.0119 | −1.4% | −85 ms | −548 ms |
| 10 | DOMINATED | +0.0110 | −4.4% | −136 ms | −604 ms |
| 8 | **NOT DOMINATED** | — | — | — | short 65.8 ms (0.6%) on p99 alone |
| 6 | DOMINATED | +0.0105 | −6.6% | −350 ms | −2962 ms |

🔴 **The baseline in this table is DISPATCHER-ONLY.** BEST-Route and AvengersPro were
`SKIP-INCOMPATIBLE` at every slot — they rank all four Qwen tiers and this fleet serves two — so
27 of 39 configs per rate never ran. **Not one BR or AP cell exists in this campaign.** The
paper's dominance claim is against BR and AP; this table cannot speak to it. "84 cells = 7 rates
× 12 configs" is 12 *surviving* of 39.

RB `(0.1,0.8,0.1)` wins six of seven and misses the seventh by 0.6% on one axis. At λ8 that is
plan Phase E′ rule 6 **outcome (b)** — *a baseline genuinely wins one axis at one λ*, to be
stated plainly. (An earlier draft cited outcome (a), "matches at strictly lower cost and
latency". That is false on the exact axis of the deficit: RB's p99 is 10871 ms vs 10806 ms,
**worse** by 65.8 ms, and RB's quality is *higher* rather than matching. Corrected 2026-08-01.)

The latency margin peaks in the mid-load band and shrinks at both ends, which is physically
coherent: at λ30 the fleet is queue-bound regardless of policy, at λ12 there is little queue to
optimise.

---

## 3. Machinery built

### Driver — `v3_scripts/s2_driver.sh`

Eleven phases, resumable, **state scoped per mode** (`DRIVER_STATE.$S2_MODE`).
`P0_CONFIG → P0_HOSTS → P0_REACH → P1_SETUP → P2_WEIGHTS → P0_TREE → P2B_ARTIFACTS →
P3_DEPLOY → P4_GATES → P5_Q1 → P6_GRID`. **`P0_TREE` must precede `P2B_ARTIFACTS`**; the reverse
order wipes staged artifacts.

Properties, each added after a specific loss: a failed phase **stops** and never substitutes a
parameter · remote work is always a launcher **file**, never an inline multi-line ssh argument ·
verification is **by execution**, not by grepping a log · `run_on_head` **refuses to start**
beside another runner · a rate must **prove it produced cells** before being marked done · every
remote step runs under a **watchdog**.

### Gates — `2026-07-31_GATES.md` has the detail

| gate | catches |
|---|---|
| G-TPOT | the static prior not applying — reads the decision log, asserts `lat_s/exp_len` per tier |
| G-ARMS | two arms that do not differ, against a **fleet-measured** baseline |
| G-TIERS | a tier typed to hardware outside the deployed fleet |
| G-MANIFEST | a manifest that is empty, digest-only, or does not cover the code |
| G-ORPHANS | a pre-existing runner (self-match guarded) |
| G-FLEETCOMPAT | a router ranking tiers the fleet does not serve |
| G-BATCHPATH | batched KNN **counted**, not inferred from `hasattr` |
| G-COVERAGE | corpus gap distinguished from aggregation failure |
| G-DOMINANCE | Phase E′ closure; refuses mixed provenance, caps, rehearsal dirs |
| G-WATCHDOG | a step that neither fails nor completes |
| G-STATE | rehearsal state leaking into a production run |
| G-BURST | the Q1-BURST assertions (built, not yet run) |
| G-CONFIG | lease shapes: a reprovisioned lease must substitute by GPU class |
| G-GUARDS | the 2026-08-01 review fixes: every one has a MUST-FAIL case |

**Self-test: 85 cases, 52 MUST-FAIL, 0 wrong** (12 suites: tpot, tiers, manifest, arms, orphans,
static, dominance, burst, watchdog, state, config, **guards**). Each of those suites demonstrates its gate
*failing a real defect*, not merely passing on healthy input.

⚠ **Scope, stated precisely.** An earlier draft said "every gate" — it is not every gate in the
table above. **G-FLEETCOMPAT and G-BATCHPATH have no self-test case**, nor do GATES.md's G-FLEET,
G-ROWS, G-COST and G-EFFECTIVE. Those six have never been shown to fail on demand. G-BATCHPATH is
additionally enforced *only* in `s2_q1_burst.sh`, which has never run — so **no banked cell was
gated by it**, and the "batched=8 perreq=0" proof is an execution observation, not a gate result.

### Analysis

`dominance.py` — Phase E′ and the producer for the headline table; cost basis **decode-only**
(author decision), basis sensitivity reported. `burst_analysis.py` — the pre-registered
degradation ratio; recovers a known injected truth to ±0.01.

### Pre-registered, built, not yet run

**Phase Q1-BURST** — λ{12,24,30} × {Poisson, gamma 0.25} × 6 seeds × 2 arms = 72 cells, ≈4 h,
plus a per-pattern baseline and static pilot (~4 extra cells). Per-cell assertions: resolved arm
before the cell, burstiness read back from the cell, scoring counters readable **and** batched
calls advancing **and** `score_perreq_calls` unchanged, and the arm gate on the cell's own slice
of the decision log (now unconditional — it used to be skipped when the log was absent).

🔴 **Pre-registered choice 3 — paired arrival streams — is NOT implemented.** The runner never
compares inter-arrival sequences, and `burst_analysis.py` pairs by `(λ, seed)` and assumes it.
The plan's own words: if pairing does not hold "the pairing argument evaporates silently". The
entire burst analysis is a paired comparison, so this must be implemented or explicitly waived
before the block runs.

**Blocked on two author decisions: which fleet, and choice 3 above.**

---

## 4. Four-tier readiness — verified by construction

### 🔴 The reprovisioning case, and the bug it exposed

**The author confirms the lease will be REPROVISIONED: a new rspec, 13 new hosts, jul25-like.**
That is a different code path from "same hostnames as the reference", and it was broken.

On a synthetic 13-host lease with **zero hostname overlap** (correct hardware: 8 A30, 3 V100,
2 A100), the generator **dropped every tier** and failed:

```
DROPPED TIER Qwen-2.5-14B (v100 x3) — absent from this lease
FAIL: Qwen-2.5-14B: no ['v100'] node in this lease   <- the lease HAS three
```

Substitution only ran when **at least one reference hostname survived** (`if not kept: drop`).
With a fresh lease none do, so `P0_CONFIG` would have failed at 01:00 Monday on a perfectly good
13-node fleet.

**Fixed:** substitution is by **GPU class**, which is what actually determines a tier's identity
— the hostname is incidental. A tier is dropped only when the lease has no node of its class at
all. The substitution is announced, not silent. Verified: all four tiers, 3B×3 / 7B×5 / 14B×3 /
72B×2 = 13 instances, no host assigned to two tiers, and a genuinely A30-only lease still fails
without `--allow-partial`.

Regression-guarded by a new `config` suite in `gate_selftest.py`.

### The same-hostname case, tested first

The A100/V100 hosts are not in the lease (§1), so this was tested against a **synthetic rspec**
carrying the jul25 topology, diffed against the real `jul25/model_deployment.json`:

| check | result |
|---|---|
| tiers emitted | **all four**, no `--allow-partial` needed |
| `node_hosts` per tier | 3B×3, 7B×5, 14B×3, 72B×2 — **IDENTICAL to the reference** |
| non-host fields | verbatim, incl. 72B `enforce-eager` + `tensor-parallel-size: 4` |
| host_configs | 17 hosts, **0 on LAN** |
| host-file derivation | `hosts_ampere` 14, `hosts_volta` 3, `hosts_{3,7,14,72}b` = 3/5/3/2 |
| G-TIERS on a 4-tier loaded set | **PASSED** |
| `P0_CONFIG` production semantics on a 2-tier lease | **correctly REFUSES**, naming both absent tiers |

**What this does and does not establish.** It establishes that config generation, host
derivation and the tier gate handle four tiers correctly, and that the paper topology is
reproduced exactly. It does **not** establish that provisioning, deploy, or serving work on
V100/A100 — those need the hardware. Specific unverified risks: the volta reboot leg in
`setup_jul25.sh` (S7.5, nouveau holds the V100s), 72B tensor-parallel startup, and 14B/72B TPOT
head behaviour under load.

---

## 5. Open — author decisions

1. **Which fleet for Q1-BURST** — 2 tiers measures the machinery, 4 tiers measures the claim.
   Gates all further cluster work.
2. **`/v1/config` merge vs replace** for arm-defining keys. Left as merge; other runners rely on
   config-file defaults persisting.
3. **`deploy_cara.py:336-337` hardcodes `~/Cara`** → Monitor fails on every node. Telemetry only;
   the scheduler reads `/instance_stats` from vLLM:8000.
4. **CI multiplier in `burst_analysis.py`** — it uses **z = 1.96** on **n = 6** seeds. The
   small-sample value is t(0.975,5) = **2.571**, 31% wider, and the verdict line is a
   CI-disjointness test, so 1.96 makes "DISJOINT" materially easier to reach. Left UNCHANGED:
   the burst block is pre-registered and the multiplier is an analysis decision, not an
   infrastructure fix. **Recommendation: switch to t, and to a paired test** — the arms share a
   seed and therefore an arrival stream, so pairing is both correct and more powerful (it is
   what rescued the λ12 Q1 claim; see §2). Not applied without your say-so.
5. **`llm_judge_scorer.py:581,991` fabricated `is_harmful`** → selects the non-safety rubric.
   Now with one concrete data point beside it (§G-COVERAGE: the single unscored pair is on an
   `is_harmful=True` record).

## 6. Owed at Monday 01:00, in order

> 📕 **`2026-08-02_S2_RUNBOOK.md` is now the single ordered page for the window** — what to
> type, what each phase should print, what to do when it stops. This section remains the
> rationale; the runbook is the procedure.


**Launch line — the fleet must be named explicitly. The driver now refuses to start otherwise:**

```bash
ALLOWED_HW=a30,v100,a100 S2_MODE=production ./v3_scripts/s2_driver.sh
```

0. **Save the new rspec** into `s2_cluster/` and **clear stale SSH host keys**. Neither is
   automatable: the reprovisioned lease has 13 new hosts, and `RSPEC` is picked by *newest
   mtime*, so a restored-with-timestamps file can lose to an old one. The rehearsal rspec has
   been moved to `s2_cluster/retired/` so it cannot be selected.
1. `preflight_gate.py` — **run by hand once the scheduler is up; it is NOT a driver phase.**
   Earlier drafts called it "the S2 first action", which was misleading: it has no caller
   anywhere, and it cannot run at P4_GATES because no scheduler exists yet. Its known-hollow
   checks are listed in GATES.md Tier 2 (`hosts[:6]` sampling; `len(raw_scores) > 0` reported as
   "every model"; `/v1/estimate` exercising `estimate()` not `estimate_batch`) — read them
   before trusting a pass. **G-TIERS now runs automatically** at every scheduler start, which
   was the check most worth having and previously had no caller at all.
2. One λ30 cell to re-confirm scheduler-overhead stability on four tiers before committing
   Q1-BURST to λ30.
3. Q1 on four tiers — the measurement promised to gTfM.
