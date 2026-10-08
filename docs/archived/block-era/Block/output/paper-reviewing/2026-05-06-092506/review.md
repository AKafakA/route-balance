# NeurIPS 2026 Review: Cara: Quality-Aware Scheduling for Heterogeneous LLM Serving with Learned Predictors

**Paper**: `cara_neurips.pdf` (13 pages: 9 main + 1 references + 3 appendix)
**Authors**: Anonymous
**Conference Format**: NeurIPS
**Severity**: Standard
**Review Date**: 2026-05-06

---

## Summary

The paper presents Cara, a batch-level scheduler for heterogeneous LLM clusters that fuses model routing and instance dispatch into a single decision, ranking candidate (request, instance) pairs along three axes (quality, latency, cost) using three learned predictors: (i) a KNN quality estimator over MiniLM embeddings, (ii) a fused multi-head ModernBERT bucket length classifier feeding probabilistic budget compliance, and (iii) per-instance XGBoost models for TTFT/TPOT (hard SLO filtering) and E2E (soft ranking). A two-tier filter–then-rank pipeline enforces per-request budgets and SLOs while a user-configurable weight tuple on the simplex exposes a Pareto frontier of named strategies (Balance / Quality / Latency / Cost). Empirical evaluation runs 215 cells (~760K requests) on a 13-instance CloudLab cluster spanning Qwen2.5-3B/7B/14B/72B on A30/V100/A100, sweeping arrival rate λ ∈ [6, 30] req/s under open-loop Poisson load. The headline finding is that Cara strategies span the 3-axis Pareto frontier against three trained-router and dispatcher-only baselines (AvengersPro, BEST-Route 4-way, passthrough+RR/SQ/random); at λ=30, BEST-Route t=0.5 collapses to mean E2E of 41.9s vs Cara-Balance's 2.5s, attributed to BEST-Route routing 98.4% of requests to the 14B tier. Two datasets are released. RouteLLM-mf and vLLM-SR are explicitly deferred.

---

## Strengths

1. **Genuine empirical scale and breadth.** 215 single-seed cells across 8 cara strategy weights, 4 AvengersPro pw values × 2 dispatchers, 3 BEST-Route thresholds × 2 dispatchers, 3 dispatcher-only baselines, and 6 ablation tracks (LPT, adaptive batching, batch-size sweep, SLO filter, budget control) on a real heterogeneous cluster — this is more rigorous baseline coverage than typical NeurIPS ML-systems submissions.

2. **Mechanistic causal story.** The pairing of the 16.6× E2E collapse for BEST-Route at λ=30 with the routing distribution table (Table 3, Figure 3) showing BEST-Route concentrates 98.4% of requests on the 14B tier is a strong, falsifiable explanation for why a quality-greedy router fails under load. This is the paper's most compelling section.

3. **Honest uncertainty quantification.** The paper does not hide that top-three quality systems overlap in within-cell bootstrap CIs (±0.009). The strong claims are confined to cost and latency at iso-quality (gaps 5–50%) and to stability under load (159s vs 5.98s), where inter-system separation is well above any plausible variance. This is exactly the discipline reviewers want.

4. **Releasable artifacts.** Two HuggingFace datasets (`cara_model_estimator` 18.6K prompts, `cara_latency_prediction` 405K records), the AGGREGATE_FULL.csv (all 215 cells), and a clear MILP + greedy algorithm. The reproducibility floor is high.

5. **MAE framing for predictor accuracy.** Appendix C's "MAE vs gap-to-next-tier" framing is more honest than the MAPE numbers it accompanies; the paper successfully argues that a 0.88 ms/tok TPOT MAE is sufficient for tier-level routing because the smallest decision-relevant tier gap is ~10 ms/tok.

---

## Weaknesses

1. **Anonymous placeholder citations for live baselines.** References [3] AvengersPro, [4] BEST-Route, [5] PARS, [6] vLLM-SR, and [26] TimeBill are all `Anonymous, 2025` placeholder bib entries. These are real systems with public artifacts (e.g., vLLM-SR is `github.com/vllm-project/semantic-router`); the bibliography must cite the actual papers/preprints/code. As written, the paper compares against "AvengersPro [3]" without telling the reader what AvengersPro actually is. This is borderline desk-reject material and must be fixed before submission.

2. **Quality leadership claim is statistically marginal.** Cara-Quality at 0.5141 is only 0.4% above BEST-Route t=0.5 (0.5119) and 0.8% above AvengersPro pw=0.8 (0.5101). The paper transparently disclaims this in §6 ("the headline 'highest-quality system' claim concerns the point estimate"), but the abstract and §5.2 still phrase it as a positive headline. A reviewer will read the abstract first and the disclaimer last; the framing should match.

3. **vLLM-SR — the most directly comparable baseline — is deferred.** vLLM-SR is a semantic-classification router that operates above the serving layer in exactly the same regime as Cara. The deferral reason (a transient cluster-NIC issue at the submission window) is plausible but is not a methodological argument. Reviewers will reasonably ask why this could not be re-run before submission.

4. **Single-seed evaluation across the full grid.** The paper acknowledges this but does not provide multi-seed evidence at any λ. Bootstrap CIs from within-cell observations capture *sampling* variance but not Poisson realization variance (which can shift TTFT tails meaningfully under saturation). Three reseeds at λ=12 and λ=30 would substantially strengthen the stability-under-load story.

5. **LPT and SLO-filter ablations show near-null effects.** Both ablations are reported honestly (LPT off ≤0.5%, SLO filter ≤2 ms p99 difference), but the paper retains "Algorithm 1 includes LPT" as part of the contribution. If LPT contributes <1% on this workload, why include it as a design highlight? The paper's argument that "LPT is theoretical guarantee + insurance under stress" is reasonable but should be more prominent — the reader currently has to reconcile the design pitch in §3 with the near-null ablation in §5.4.

6. **Predictor TPOT MAPE 152.9% (14B-V100) is alarming on first read.** The "denominator inflation" explanation in App C is technically correct, but the table format puts MAPE prominently next to MAE. Either drop MAPE from the main table and put it only in a footnote, or order columns to lead with MAE-vs-tier-gap. As laid out, a casual reader will see 152.9% and stop.

7. **Empty Appendix E.** Section E "System extension: dynamic deployment" contains only the placeholder text *"Original §A.1 from the prior draft; ports without modification."* This is unacceptable in a submission. Either port the content or delete the section.

8. **No code release URL.** The paper releases two HuggingFace datasets but does not point to a code repository for the predictor stack, sweep harness, or aggregator. NeurIPS reproducibility checklist will flag this.

9. **Limited scope: single model family, single cluster topology.** Qwen2.5 only, 13 instances on one CloudLab Phase 2 cluster. The "spans the Pareto frontier" claim is on this configuration; cross-family routing (Qwen + Llama + Mistral) and cluster scale (>20 instances) are open questions the paper acknowledges but cannot answer empirically. This is fine for a workshop or systems venue but tight for a NeurIPS bar that emphasizes generality.

10. **Algorithm 1 in appendix.** For a systems paper whose central contribution is a scheduling algorithm, having Algorithm 1 in Appendix A rather than §3 is unusual and may push reviewers to skim. A compressed inline pseudocode block in §3 would help.

---

## Major Issues

1. **Anonymous citations for live systems**: As detailed in Weakness 1, references [3]–[6] and [26] are placeholders. **Suggested fix**: Replace with actual bibtex for AvengersPro (Wang et al. 2024 or whichever paper introduced power-law mixing for LLM routing — verify on Google Scholar), BEST-Route (the actual paper introducing 4-way confidence-thresholded routing), vLLM-SR (cite the GitHub repo + any associated tech report), TimeBill (Anonymous → real cite). Without this fix the paper is not citable.

2. **vLLM-SR comparison gap**: **Suggested fix**: Either run vLLM-SR on the same 7-λ grid (4 hours of cluster time, 1 GPU box for the Envoy router) and add it to Table 2 / Figure 1, or move the explanation of *why* it cannot be compared from §6 into a dedicated subsection in §5 explaining that the deferral is a methodological constraint of the cluster network rather than a fundamental comparison gap. The current treatment buries the gap in the limitations paragraph.

3. **Quality leadership framing inconsistency**: **Suggested fix**: In abstract and §5.2, change "highest quality" to "highest point-estimate quality (within bootstrap CI of BEST-Route t=0.5 and AvengersPro pw=0.8)". This is more accurate and pre-empts reviewer pushback.

4. **Empty Appendix E**: **Suggested fix**: Either port the dynamic-deployment content from the original draft, or delete the appendix entry from the document outline.

5. **Code release**: **Suggested fix**: Add a "Code and Data Availability" paragraph either at the end of §1 or in the Reproducibility Statement, with a (anonymized for review) URL such as `https://anonymous.4open.science/r/cara-XXXXX`.

---

## Minor Issues

1. The first reference `[1] Introducing GPT-5 — openai.com.` — a press release as the first citation is awkward; pair it with a peer-reviewed reference for "heterogeneous LLM clouds" if possible.
2. `Anonymous` author entries (refs [3]–[6], [26]) clearly violate NeurIPS bibliography norms even for a double-blind submission — the cited works are public, only the *current* paper is anonymous.
3. Equation 1: the negative sign in `-w_q Q` is correct for "minimize" but is briefly confusing; a half-sentence noting "we minimize this score; quality is negated to favor higher quality" would help.
4. Figure 2 (4-axis radar): adds little beyond Figure 1 for a NeurIPS audience used to reading Pareto frontiers directly. Consider dropping in favor of an additional λ-axis figure variant or giving the page back to algorithm placement.
5. Figure 3 (model distribution stacked bar): the 1%, 2% labels next to "BEST-Route t=0.5" are hard to read; bold the 98% label or use leader lines.
6. Table 3 title and Figure 3 caption say similar things; consolidate.
7. §5.4 SLO ablation: "we did not run a filter-off cell at λ=12" — an honest acknowledgement, but a 30-minute additional cell would close this gap entirely; flag as recommended for camera-ready.
8. Routing distribution table shows Cara-Latency routes 84.6% to 14B-V100. The narrative says Cara strategies "spread load across at least three model tiers"; Cara-Latency is borderline 2-tier (84.6% + 13.8% to 7B = 98.4%). Tighten the claim to "Balance / Quality / Cost spread; Latency concentrates on V100 because of its TPOT advantage" — currently sounds like all 4 strategies are equally diverse.
9. Section §6 is titled "Related Work" but also contains the Limitations paragraph; consider splitting into "§6 Related Work" + "§6.1 Limitations" or a separate "§7 Limitations" before Conclusion.
10. The MILP in Appendix A omits the makespan term `w_m · max_r (\sum_i x_{r,i} P_{r,i})` from §3's textual description (which mentions LPT as a makespan-reducer but no `w_m`). Reconcile: either the MILP has `w_m`, the text should say so explicitly; or the algorithm doesn't optimize makespan as a weighted term and the MILP should drop `w_m`.

---

## Questions for Authors

1. **Multi-seed**: For the headline λ=12 row (Table 2) and the BEST-Route collapse claim at λ=30, can you provide at least 3-seed mean ± std to confirm that the Poisson-realization variance does not flip the system ordering on cost or latency? The within-cell bootstrap CI does not address this.

2. **vLLM-SR feasibility**: What specifically prevented apple-to-apple comparison? Is the cluster-NIC issue a reproducibility blocker for any reader who wants to extend the study? If so, the limitation should be more prominent.

3. **w_balance ablation**: The paper argues the 4th balance axis is unnecessary because LPT-with-state-update implicitly handles load balance. Did you ablate `w_balance ∈ {0, 0.1, 0.3}` empirically? If not, the design choice is asserted rather than demonstrated.

4. **Predictor latency budget**: §3 says scheduling overhead is ~3 ms/(request, instance) for XGBoost in-process inference. With 13 instances, that is 39 ms/request just for predictions — comparable to the per-instance TTFT itself. Is this consistent with the mean E2E numbers, and how does the per-request scheduling overhead trend with cluster size?

5. **Quality predictor robustness**: KNN k=50 on MiniLM-L6 with a 14,919-prompt training corpus — what happens when the test distribution shifts (e.g., domain-specific prompts not in the training pool)? Is there an OOD detection step?

6. **Fairness across model sizes**: With Cara-Quality routing 17.6% to 72B-A100 and AvengersPro pw=0.8 routing 36% to 72B, there is a tension between user-perceived quality and infrastructure cost-of-ownership. Did you compute total cluster utilization per strategy?

7. **The "spans the Pareto frontier" claim** is illustrated with 4 strategy points. With 8 weight tuples in the sweep, can you show that *every* simplex weight tuple lies on the frontier, or only the four named corners?

---

## Missing Related Work

| Paper | Key Contribution | Relevance | Should Be Cited In Section |
|---|---|---|---|
| Splitwise (Patel et al., 2023) | Phase-split inference (prefill/decode disaggregation) | Cara's Tier-1 latency predictor decomposes TTFT/TPOT for similar reasons | §2 Background or §6 Related Work — heterogeneous serving |
| Mooncake (Qin et al., 2024) | KV-cache-centric disaggregated serving | Cited indirectly via [24] but not by name; Cara's KV-aware filter is closely related | §3 Filter design or §6 Related Work |
| Sarathi-Serve (Agrawal et al., 2024) | Chunked prefill scheduling | Direct alternative to LPT-based batching | §3 LPT discussion or §6 |
| Loongserve (Wu et al., 2024) | Elastic SP for long-context LLM serving | Heterogeneous-replica scheduling | §6 Related Work — heterogeneous serving |
| AlpaServe (Li et al., OSDI 2023) | Statistical multiplexing for serving DNN ensembles | Multi-model heterogeneous serving precursor | §6 Related Work — heterogeneous serving |
| ServerlessLLM (Yang et al., 2024) | Cold-start optimization for LLM serving | Relevant for Appendix E (dynamic deployment) | Appendix E |
| MELT or RouterBench (Hu et al., 2024) | LLM routing benchmark | Cara's evaluation methodology overlaps | §5.1 Setup or §6 |
| Andes (Li et al., 2024) | QoS-aware LLM serving | Direct competitor on the SLO-aware scheduling axis | §6 |
| InferCept (Hu et al., 2024) | Speculative-decoding-aware serving | Tangential but worth one sentence | §6 |
| Triumvirate or RouteLLM-Bandit (various 2024) | Bandit-based exploration over routers | Alternative to learned predictors | §6 Model routers |

The paper's related work section is solid on the major dimensions (serving systems, heterogeneous serving, model routers, length/latency prediction) but misses the disaggregated-serving line (Splitwise, Mooncake, Sarathi-Serve) which is currently the most active research area adjacent to Cara's claims. Adding 3–5 of these citations is a 30-minute fix and visibly strengthens the related work positioning.

---

## NeurIPS Checklist

- **Code & Data Release**: Datasets released on HF; **code release URL missing** (-1)
- **Reproducibility**: Algorithm provided + datasets + hyperparameters described; cluster topology fully specified ✓
- **Computational Resources**: 13-instance CloudLab cluster (4 GPU types), training data: 405K records, ~760K request evaluations ✓ specified
- **Limitations Discussion**: Present, integrated into §6 (one paragraph) ✓
- **Ethical / Broader Impact**: **Missing** — required NeurIPS section ✗
- **Statistical Reporting**: Bootstrap CIs for headline row only; full grid is single-seed (-1)
- **Anonymization**: Authors anonymized ✓; bibliography contains Anonymous placeholder cites for non-anonymous prior work (must fix)

---

## Scores

- **Soundness**: 3/4 — empirical methodology rigorous; claims well-supported by the released CSV; one quality-claim framing inconsistency between abstract and §6 disclaimer
- **Presentation**: 3/4 — clean writing, strong figures (Figure 1 + Figure 3 carry the paper); empty Appendix E and Algorithm-in-appendix placement are flaws
- **Contribution**: 3/4 — joint quality–latency–cost batch scheduling on a real heterogeneous cluster is a genuine contribution; novelty is in the *combination* and the empirical depth, not in any single technique
- **Overall**: **6/10 — Weak Accept** (with mandatory fixes for anonymous citations + empty appendix; would lift to 7 with vLLM-SR baseline added or with multi-seed at headline rows)
- **Confidence**: 4/5 — I have read the paper carefully, verified the numerical claims against the released CSV, and have direct expertise in the LLM serving area. Limited only by not having run the code myself.

### Categorical assessments

- **Novelty**: Medium — the filter-then-rank framing on the 3-axis simplex is novel as an integrated system; component techniques (KNN routing, XGBoost latency prediction, LPT scheduling) are individually known.
- **Technical Soundness**: Medium-High — methodology is rigorous; statistical reporting is honest about overlapping CIs but single-seed across the full grid is a real limitation.
- **Significance**: Medium-High — the BEST-Route 16.6× collapse is the kind of empirical headline that the heterogeneous-serving community will cite; whether the contribution generalizes beyond Qwen2.5 on a 13-instance CloudLab cluster is the open question.
- **Clarity**: Medium-High — well-written, easy to follow; some structural choices (algorithm in appendix, MILP weight discrepancy) cost a fraction of a point.

---

## Recommendation

**Weak Accept (6/10)** with mandatory pre-camera-ready fixes:

1. Replace anonymous placeholder citations with real bib entries (~30 min).
2. Either fill or delete Appendix E (~15 min).
3. Add code release URL + Broader Impact paragraph (~30 min).
4. Reconcile the abstract's "highest quality" claim with the §6 CI-overlap disclaimer.
5. Strongly recommended (lifts to 7): re-run vLLM-SR on the same λ-grid OR run 3 multi-seed cells at λ=12 and λ=30.

This is a paper I would champion in committee discussion if the four mandatory fixes land. The empirical contribution is real, the writing is mostly disciplined, and the failure-mode analysis of BEST-Route is genuinely informative for the heterogeneous LLM serving community. The main risks at the score-6 bar are the anonymous citations (which any careful reviewer flags) and the single-seed evaluation (which a hostile reviewer can use to argue the headline is fragile).

---

*Reviewer notes: I read the full 13-page PDF and verified six key numerical claims (Table 2 headline numbers, λ=30 BEST-Route collapse, BEST-Route routing 98.4% → 14B, bootstrap CI half-widths, p99 separation between Cara-Balance and BEST-Route, AvengersPro pw=0.8 load growth) against `cara_final_e2e/_aggregate/AGGREGATE_FULL.csv` and `p99_lambda_axis.json`. All verified.*
