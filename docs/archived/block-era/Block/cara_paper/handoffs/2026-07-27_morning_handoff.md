# MORNING HANDOFF — 2026-07-27 (~06:00Z) · rebuttal window, approved-plan execution

## TL;DR
The approved plan is executing and mostly green. The paper-facing four-arm table is DONE and
the headline SURVIVES on the paper's own binary (γ30 −15.3%, overload −30.8%, quality ties).
Two decisions wait for you (§4); everything not blocked by them is running or banked.

## 1. What completed overnight
- **A0 re-gate: PASS** — v1exact-v2 tree (May-8 + 2 inert ports, manifest 553d76c3) reproduces
  May: ≤0.8pp mix, tput identical, e2e +0.7%. Certified by execution, uncontaminated.
- **R1 bridges 26/26, audit 0 INVALID, synced.** TABLE V1 FILLED in the draft
  (2026-07-27_rebuttal_draft_noGemma.md): stationary parity λ6–24 (±0.8%) · γ30 learned
  −15.3% · overload 40/6 learned −30.8% · quality statistically tied everywhere (|Δ|≤0.003,
  both bases → NO quality-margin claims → no basis-dependence). γ27's current-tree win is
  basis-dependent → reported as parity, NOT claimed. λ30-stationary arm1 +6.7% slower →
  reported honestly (the claim is burst/mismatch, which holds).
- **Cross-era quality equivalence proven**: May wq0.33_l12 vs bridge arm1_l12 Δ<0.001 both bases.
- **perdomain_quality.py** built (EXP2 metric-2 + retroactive baseline slices; 0 lookup misses).
- **PT ×17 running now** (3 banked, 0 fails). Bridge total on disk: 30 files, double-located.

## 2. Incidents — all caught by gates, zero bad cells banked
- v1exact tree lacked models/ → era BR4 raised LOUDLY (v1 validation, as you remembered);
  routerless partial quarantined; fixed with a models/ symlink (infra).
- Era AP silent-fallback caught by the decisive 3-prompt probe (probe-gate hole closed).
- BR4 judged twin FIRED: mix 0.00pp BIT-EXACT, e2e +9.4% (2626 vs 2401ms) > 5% band → family
  halted pending your ruling (§4.1). Forward-speed ruled out (48ms today vs 431ms era bound);
  historical BR-cell reproduction band ≈8%; single-run-vs-single-run.

## 3. Currently running / next in chain
PT ×17 (~2h) → [BR4 ×8 + AP ×36: GATED on your rulings] → Phase-D deletions → P4 ×25 →
EXP2 ×8 → flip column ×12 → GEMMA CHECKPOINT (your go) → Gemma per plan (headline G5 sampled
first). Draft assembly continues in parallel; V2 table fills as router families unblock.

## 4. ONE DECISION WAITING (as of ~08:00Z): "approve C" for BR4
Evidence complete: BR4 v1exact decisions BIT-EXACT vs May (0.00pp, twice); latency NOT stably
May-comparable (runs +9.4%/+23.8%; scheduler-side queue 421→564→773ms — near-saturated serial
scorer amplifies CPU contention superlinearly; = the paper's own documented serial-scoring
fragility). C = claim decisions+quality continuity (exact), BR4 latency within-run only w/
disclosure; drops ONLY the bonus latency-continuity row. BR4-vs-RB latency comparison lives in
the submitted Table 4 already; the new Po2 axis is within-family (tight). AP port: APPROVED +
implemented (June shim, manifest v4); its judged twin runs right after BR4 completes.
## 5. Timeline vs the gate (submission Jul-28 11:59 BST)
PT done ~08:00Z · [your rulings] → BR4+AP ≈ +5h from approval · P4/EXP2/flip ≈ +6.5h →
Qwen complete ~19:00-21:00Z (rulings by ~09:00Z) · draft final (no Gemma) to you ~21:00Z ·
Gemma overnight → sampled headline ~09:00Z Jul-28 (draft complete without it). Slack: thin but
real; pre-committed Gemma fallbacks absorb overruns without touching the gate.

## 6. Standing rails (unchanged)
Presentation gate (nothing posted without your sign-off) · no-mixed-tables (grep-enforced) ·
no cross-era latency outside v1exact cells · dual-basis sign check per claim · deletion only
per approved Phase D (tombstoned) · commits LOCAL only.
