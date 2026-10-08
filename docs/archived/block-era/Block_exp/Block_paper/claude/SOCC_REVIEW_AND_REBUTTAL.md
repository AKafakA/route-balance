# SoCC 2026 First-Round Review and Rebuttal

**Paper #34**: Astrolabe: Balancing Load in LLM Serving with Context, Knowledge, and Prediction-Guided Scheduling

**Status**: Survived first round (scores: 1, 2, 3 — Reject, Weak Reject, Weak Accept)

**Date**: April 2026

---

## Scores Summary

| Reviewer | Score | Expertise | Verdict |
|----------|-------|-----------|---------|
| R34A | 2 (Weak Reject) | Knowledgeable | Solid eval, but novelty limited; missing KV-aware routing, realistic traces |
| R34B | 1 (Reject) | Knowledgeable | Engineering combination of existing work; concerned about cascaded estimation errors and scalability |
| R34C | 3 (Weak Accept) | Knowledgeable | Well written, interesting approach; wants practical deployment discussion |

---

## Review #34A (Score: 2 — Weak Reject)

### Strengths
- S1: Problem well motivated (commodity clusters without high-speed interconnects)
- S2: Simple, modular design; different length predictors and simulators can be integrated
- S3: Thorough evaluation (multiple setups, baselines, workload scenarios, sensitivity analysis, generality)

### Weaknesses
- **W1 (Novelty)**: Length prediction and performance simulation are both well-studied. Core innovation is combining known techniques — limited novelty.
- **W2 (No KV-cache awareness)**: Paper does not mention prefix caching or KV-cache-aware routing. Not considering reusable KV cache may cause unnecessary recomputation.
- **W3 (Synthetic arrivals only)**: Only Poisson and gamma arrivals tested. Should use real production traces.
- **W4 (Small gains)**: 1.3-13.8% capacity, 1.53%/7.15% latency reduction — may not justify system complexity.

### Questions
1. How to deal with prefix caching of KV cache?
2. How to handle workload drift for the length prediction model?
3. Why is the "+10 token" assumption realistic when a request exceeds predicted length?
4. Llumnix comparison fairness — chunked prefill enabled for Astrolabe but not Llumnix. How much improvement comes from chunked prefill vs. scheduling? Can you prove PCIe/NIC congestion claim?

### Minor
- M1: Blurry figures (5, 8-12). Typos in Fig 5: "Chunked Prefilled" -> "Chunked Prefill"
- M2: Missing TBT/TPOT metrics — does system do tradeoffs?
- M3: Batch size justification needed; is comparison fair across solutions?

---

## Review #34B (Score: 1 — Reject)

### Strengths
- S1: Real systems problem — global load balancing on practical clusters
- S2: Clean, well-motivated, intuitive design

### Weaknesses
- **W1 (Not novel)**: Looks like engineering combination of existing works
- **W2 (Length estimation unreliable)**: Accuracy not guaranteed under extreme cases; experimental setup may not be fair

### Questions
1. **Q1 (Novelty vs. prior work)**: Both length prediction [ELIS, Sequence Scheduling] and simulation [Vidur] are established. What's new about RoBERTa-based model? Can module be trivially replaced? Simulation part is based on existing Vidur — makes design look incremental.
2. **Q2 (Offline tagging)**: Length estimation done offline on dedicated L40 — doesn't demonstrate true online cost of deployed pipeline.
3. **Q3 (Cascaded estimation errors)**: Two layers of estimation (length + latency simulation). Errors can compound. Analysis limited — doesn't cover cases where instances have similar predicted latency or biased errors.
4. **Q4 (Missing latency breakdown)**: CPU utilization shown but not prediction/simulation latency breakdown. Hard to assess overhead.
5. **Q5 (Scalability concern)**: One-shot dispatch benefit may cover scheduling cost at current scale, but as cluster grows, Vidur simulator overhead may become non-negligible.

### Minor
- M1: Font sizes in Figs 3, 8-12 too small

---

## Review #34C (Score: 3 — Weak Accept)

### Strengths
- S1: Important problem — improving LLM serving in constrained environments
- S2: Well written, informative evaluation, convincing practical value
- S3: Avoiding KV-cache migration while improving performance is interesting; viable for reducing operational complexity while delivering SLO

### Weaknesses
- Unclear what specific information needed about user applications for realistic simulation
- Need to discuss practical deployment: obtaining simulation-based latency prediction, user expertise required

### Questions
- Don't you need access to customer code (proprietary) to predict request length/latency?
- If application-specific aspects don't matter, discuss this explicitly

---

## Our Rebuttal (742 words)

### To Reviewer A

**Q1 / W2 (Prefix caching / KV-aware routing)**:
Prefix caching is enabled by default in our backend. Its effect in our setting is limited because we use deduplicated single-turn ShareGPT, where cross-request reusable prefixes are rare. Explicit KV-aware routing is orthogonal and complementary: reusable-KV signals can be exposed to the predictor and per-instance simulator without changing the dispatch policy.

**Q2 (Length model under workload drift)**:
The tagger is an interchangeable lightweight module, not the paper's main contribution. It can be periodically refreshed from logged traffic, and Astrolabe is agnostic to the estimator family.

**Q3 ("+10 token" assumption)**:
This is a bounded online correction for underestimated lengths, applied in 10-token increments during runtime simulation. While it can affect the predicted metric, the same correction is applied uniformly across candidate instances, so its effect on relative ranking is typically smaller than on absolute prediction error.

**Q4 (Llumnix fairness and breakdown)**:
Llumnix does not support chunked prefill, while it materially improves Astrolabe's prediction fidelity (Fig. 5) and reflects modern default serving behavior. Section 6.5 is therefore intended as a best-effort head-to-head under each system's supported configuration, as our claim concerns predictive dispatch under current backend behavior.

**W3 / W4 / M2 / M3 (gains, metrics, and benchmark)**:
The reported gains are against already strong load-aware baselines, and they matter most near capacity, where dispatch quality directly affects SLO attainment. We focus on TTFT/E2E/capacity because they directly reflect global dispatch quality; TPOT/TBT are less directly affected by one-shot dispatch after admission. Our workload follows the vLLM benchmark-suite setting with controlled arrivals, a widely used standardized stress-test for LLM serving. We will also state the uniform per-instance configuration used across schedulers and improve figure readability.

**W1 (Novelty)**: Addressed in Reviewer B Q1 below.

### To Reviewer B

**Q1 (Novelty vs. prior length predictors and Vidur)**:
We agree that length prediction and offline simulation are individually established; the RoBERTa tagger is replaceable and is not our contribution. Our contribution is, to our knowledge, the first online system integrating response-length estimation with per-instance simulation for one-shot, multi-instance LLM dispatch on commodity clusters, together with a power-of-two variant that bounds fanout by a constant. Prior length-prediction work mainly targets local batching/scheduling, while simulators such as Vidur target offline planning. Astrolabe combines them uniquely on the online dispatch path with per-instance runtime state under dynamic batching and memory pressure. This is not a trivial composition of known modules: making these components work effectively online requires concrete system design, backend integration, and implementation that satisfy runtime constraints while preserving useful cross-instance ranking for dispatch. We will sharpen this positioning in sections 1-2.

**Q2 / Q4 (Offline tagging and latency breakdown)**:
Offline tagging was used for experimental control and reproducibility, not because online inference is infeasible. End-to-end overhead is already reported in sections 6.3/6.9: scheduling overhead remains below 3% of E2E latency under capacity and becomes less significant near capacity as queueing dominates. This is consistent with prior online LLM scheduling systems using lightweight predictors/estimators; for example, ELIS reports 11.04 ms average online scheduling overhead including its initial BERT predictor.

**Q3 (Compounded error and robustness)**:
Astrolabe relies on relative ranking, not perfect absolute latency prediction. The same predicted length is fed to every candidate, so common-mode error largely cancels in the cross-instance comparison. Fig. 5 shows that the selected instance tracks the low-latency region and that Rank-1 selection probability remains high even when absolute prediction error is non-trivial. This is why dispatch remains effective despite imperfect prediction.

**Q5 (Scaling overhead)**:
The simulator is per-instance and parallel. The broadcast variant incurs O(N) scheduler-side messaging cost, which is why we also evaluate the power-of-two (Po2) variant that bounds fanout to a constant (k=2) while preserving most of the benefit. As shown in Fig. 9, Po2 remains effective at higher load and helps reduce synchronized fanout, which can mitigate herding and improve latency.

### To Reviewer C

**Q (Deployment assumptions / customer code)**:
Astrolabe does not require customer application code or proprietary application internals. It only needs the prompt and runtime instance states rather than application-specific semantics. Our design is backend-facing: the predictor interface is modular, and in our prototype we adapt Vidur's simulator while exposing the required serving state through a lightweight read-only API in vLLM for online prediction.

---

## Analysis: What Reviewers Want for Camera-Ready

### Must Address (to move scores up)

| Issue | Reviewers | Action Required |
|-------|-----------|-----------------|
| **Novelty positioning** | A (W1), B (W1, Q1) | Sharpen sections 1-2: explicitly state what is new vs. prior work |
| **KV-cache-aware routing** | A (W2) | Add discussion paragraph; explain orthogonality |
| **Realistic traces** | A (W3) | Add production trace experiment or justify current approach |
| **Latency breakdown** | B (Q4) | Add prediction + simulation latency measurements |
| **Cascaded error analysis** | B (Q3) | Analyze edge cases: similar-latency instances, biased errors |
| **Llumnix fairness** | A (Q4) | Decompose chunked-prefill vs. scheduling contribution |
| **Online tagging** | B (Q2) | Run or cite online length estimation overhead |
| **Practical deployment** | C | Add deployment discussion paragraph |
| **Figure quality** | A (M1), B (M1) | Fix blurry figures, enlarge fonts |

### Nice to Have

| Issue | Reviewers | Action |
|-------|-----------|--------|
| TBT/TPOT metrics | A (M2) | Add if data available |
| Batch size justification | A (M3) | Add paragraph |
| Scaling study beyond 12 GPUs | B (Q5) | Po2 partially addresses; could add discussion |
| Workload drift handling | A (Q2) | Brief discussion |

---

*Last updated: April 16, 2026*
*Document location: Block_paper/claude/SOCC_REVIEW_AND_REBUTTAL.md*
