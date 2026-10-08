# FINAL AUDIT + REVIEWER SIMULATION BRIEF — 2026-07-28

Runs **after all 32 cells finish** (topology 16/16 + Gemma 16/16). Author-requested.

---

## 0. THE REVIEWED PAPER — PINNED. DO NOT SUBSTITUTE.

```
/home/wd312/Code/llm/route-balance-paper/neurips_2026.tex
```

Identified by evidence, not filename:

| candidate | modified | carries the headline numbers reviewers quote (0.5141, 2219, 18.73, 28.85) |
|---|---|---|
| **route-balance-paper/neurips_2026.tex** | **2026-05-19** | **all present** ✅ |
| route-balance-paper/neurips_2026_v2.tex | 2026-06-13 | none — post-submission revision |
| Block/cara_paper/cara_neurips.tex | — | different abstract; title "Cara: …" |

**Do NOT open `cara_neurips.tex` or `neurips_2026_v2.tex` for any claim check.** On 2026-07-27 three
"blocking contradictions" and an entire 4-agent reception simulation were produced against
`cara_neurips.tex` and were all void. Tables live in `tables/*.tex` via `\input` — open the
`\input` target, never judge a table from a grep of its caption.

Reviewer mapping (authoritative — `2026-07-27_ACTUAL_REVIEWS_VERBATIM.md`):
**287j** = clarity/batching, rating 3 · **gTfM** = estimator Q1–Q4, rating 4 ·
**faAU** = generality/cold-start/dispatchers/W4, rating 4 · AC **axCN** meta-review.

---

## 1. AUDIT SCOPE — every posted number traced to a banked cell

Deliverable: `2026-07-28_FINAL_AUDIT.md`, one row per claim → source cell → basis → verdict.

1. **Basis purity.** Every Qwen number from the submitted basis
   `fd3e98d956ca15de5b21266d5104ffc2`; every Gemma number from `d34982a7ba017662bb9df6ef6a85efce`.
   Quarantined dirs (`gemma_cells_stale_basis/`, `gemma_cells_pre_port4/`,
   `topology_flip/` = the contaminated current-tree set) must contribute **nothing**.
2. **Cell sanity.** `completed == 1979` (Gemma) / `3534` (Qwen), `failed == 0`, on all 32.
3. **Dual-basis quality.** Report on the paper (buggy `or`-chain) basis; fixed basis as a silent
   gate. Any claim whose sign flips between bases is a STOP-item.
4. **No self-contradiction.** The W1↔W3 BEST-Route discrepancy is what triggered this whole rerun.
   Re-check every system that appears in more than one table.
5. **Normalization + axis stated.** Paper Figure 3 is **min–max** normalized on **mean** latency
   (caption + §5 metric definition). Any table using ratio-to-best or p99 must say so explicitly.
6. **Rails.** No links · no codename · no k=50 claim · no v2/version language · never re-assert the
   two basis-dependent claims · quality margins ≤0.01 only with computed CIs.

## 2. REVISION

Fold into `2026-07-27_replies_FINAL_FORMAT.md`: corrected W1 (author's decision in
`2026-07-28_W1_DECISION.md`), the Gemma family block, the corrected onboarding figure, and the
batch-design rationale. Every table gets a verification line: basis + source cells + gates passed.

## 3. SIMULATION (subagents, after the audit)

Four agents, each given **only** `2026-07-27_POSTED_VIEW.md` (the reviewer-visible text) plus the
pinned paper path above:

- 3 reviewer agents (287j, gTfM, faAU) — weak-accept-bar persona per
  `feedback_neurips_reviewer_rubric`: must surface rejection-grade issues, not be agreeable.
- 1 AC agent (axCN) — weighs the meta-review asks (XGBoost justification, general-purpose
  dispatcher comparison) against what the packet actually delivers.

Each returns: score movement (with reason), unresolved weaknesses, and any claim they would
challenge as unsupported. **Instruct every agent not to read any other paper file in the repo.**

## 4. STOP CONDITIONS

Any of these halts posting and goes to the author: a claim traceable to a quarantined cell · a
sign flip between bases · a self-contradiction across tables · an unstated normalization · a
reviewer agent identifying an unsupported claim that the audit did not already catch.
