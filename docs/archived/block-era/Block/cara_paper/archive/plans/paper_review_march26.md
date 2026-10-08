# Cara Paper Review — March 26, 2026

## Review as EuroSys Reviewer

### Overall: Strong Design, Awaiting Evaluation Results

The paper addresses a real problem (unified quality-aware scheduling for heterogeneous LLM serving) with a clean system design. The MILP formalization, LPT-based greedy solver, two-tier latency prediction, and bucket-based length classification are technically sound. The related work section is excellent (30+ papers). The missing end-to-end evaluation will be filled from April 26 CloudLab experiments.

### What's Strong (keep as-is)
- MILP formalization (§3.2) — clean, complete, well-motivated
- LPT-based greedy algorithm (Algorithm 1) with within-batch state updates
- Two-tier latency: hard TTFT/TPOT SLOs (no output length dependency) + soft E2E scoring
- Safety-aware quality scoring — dual strategy for safe vs harmful prompts
- Bucket classification with probabilistic budget compliance — novel and practical
- Related work — comprehensive, well-positioned against 30+ papers
- Load-aware adaptive batching — nice insight

### Writing Issues to Fix

#### High Priority

1. §3.1 RSO API too verbose — 1.5 pages for a JSON schema. Tighten to 0.5 page.
2. Abstract needs results sentence (currently %% TODO).
3. Notation inconsistency — P_{r,i} (generic) vs hat{P}_{r,i} (latency) in Algorithm 1.
4. "5000-line Python system" — clarify what this counts.
5. "inspired by learned index structures" — connection is weak, strengthen or drop.
6. "forward-compatible" — soften to "designed to be forward-compatible" until demonstrated.

#### Medium Priority

7. Table 1 — clarify "1 instance" means 1 node with 4xA100 TP=4.
8. Bucket size = 64 tokens — add one sentence justification.
9. Merge §3.2 and §3.2.1 (Problem Definition reads at same level).
10. Algorithm 1 line 20 — dispatch immediately, discuss implications for state updates.
11. Appendix A/B — add forward references from main text.

#### Low Priority

12. §3.3 constraint policy — tighten to one paragraph.
13. Double xcolor package (lines 30 and 35).
14. Listing 1 JSON schema breaks column flow.

### Technical Questions Reviewers Will Ask

1. Weight sensitivity (w_lat, w_cost, w_qual, w_u)? → Planned in CloudLab sweep.
2. Batch size dynamics under load? → Planned in CloudLab.
3. Quality predictor on unseen harmful prompts? → Test on held-out set.
4. Scheduling overhead wall-clock time? → Measure during CloudLab.
5. Cold-start for new model? → Discuss limitation.
6. Predictor error → scheduling quality degradation? → Error sensitivity experiment.

### Pre-Submission Checklist

- [ ] Fill all %% TODO with results
- [ ] Abstract results sentence
- [ ] Tighten §3.1
- [ ] Fix notation
- [ ] Bucket size justification
- [ ] Soften forward-compatible
- [ ] Clarify Table 1
- [ ] Remove duplicate xcolor
- [ ] Weight sensitivity results
- [ ] Batch size characterization
- [ ] 8-strategy baseline comparison
- [ ] Ablation study
- [ ] TTFT/TPOT prediction accuracy
- [ ] Scheduling overhead breakdown
