# NeurIPS Review: Astrolabe — Balancing Load in LLM Serving with Context, Knowledge, and Prediction-Guided Scheduling

**Conference:** NeurIPS (ML Systems area — applicable to main track or Datasets & Benchmarks track)
**Severity:** standard
**Review date:** 2026-04-21

---

## Summary

The paper proposes Astrolabe, a stateless global scheduler for multi-instance LLM inference clusters. For each incoming request, Astrolabe (1) estimates the response length with a fine-tuned RoBERTa-base regressor, (2) samples N=2 instances uniformly at random and queries a per-instance Vidur-based simulator to predict end-to-end latency, and (3) dispatches the request to the candidate with the lower predicted latency. The core design choice is *power-of-two-choices* (Po2) over full fanout, motivated by a failure mode the authors call *scheduler herding*: deterministic fanout schedulers compute identical per-instance predictions for concurrent bursty requests and send them all to the same "best" instance. Evaluation on 12×A30 (Llama-2-7B) and 2-node A100 (Llama-2-7B, TP=1; Llama-2-70B TP=4 for prediction-accuracy only) clusters reports Astrolabe-Po2 matches full-fanout capacity within 0.3 QPS, uses 2.8× less per-predictor CPU, and outperforms KV-migration-based Llumnix by up to 2.57× throughput / 123× per-token latency at saturation on the network-constrained testbed.

---

## Strengths

1. **Concrete, named failure mode.** The paper articulates "scheduler herding" — the deterministic-fanout pathology where concurrent requests compute identical predictions and collide on the same instance — as a first-class phenomenon motivating randomized dispatch. This is a small but genuine conceptual contribution that unifies an intuition many systems papers dance around implicitly.

2. **Coherent minimal design.** Combining (i) length prediction, (ii) per-instance simulation, and (iii) Po-N randomized probing into one one-shot dispatch pipeline is a minimal and well-separated design. The Predictor-as-sidecar abstraction (§4.1) cleanly decouples the scheduler from the inference framework and supports swap-in simulators (Vidur, SimAI).

3. **Comprehensive ablations for a systems paper.** Two clusters (A30, A100), two models (Llama-2-7B, Qwen2-7B), three arrival patterns (Poisson, gamma α∈{0.25,0.5,1.0,2.0}, BurstGPT), four backend configurations (bs/chunk sweeps), and a 4×4 prediction-noise sensitivity grid. The N-tunable family study (Po2, Po4, Po8, Fanout) is particularly informative and even exposes that Po4 slightly outperforms Po2 in raw capacity — a self-critical finding the authors still recommend against on principled grounds (herding mitigation).

4. **Realistic head-to-head baseline.** Llumnix is evaluated with KV migration actually enabled (not a strawman), at the intended 8-instance 2-node scale with NVLink intra-node and 100 Gbps inter-node — matching the Llumnix paper's regime as closely as an external reviewer can verify.

5. **Prediction-accuracy and generalization data.** The RoBERTa length tagger achieves MAPE 24.4% on Llama-2-7B and 16.6% on Llama-2-70B (§6.2.2); sensitivity to injected noise (§6.8) shows worst-case +5.1% e2e latency degradation at 100%/100% multiplicative noise. This is a reasonable characterization of the predictor's operating envelope.

---

## Weaknesses

1. **ML novelty is narrow.** The three ingredients (supervised length prediction, deterministic-simulation latency prediction, power-of-N-choices dispatch) are individually well-established — Po-N dates to Mitzenmacher 2001 and Azar et al. 1999 (both cited), length prediction has Sequence Scheduling / ELIS / S³ precedents (cited), and simulation-based serving scheduling was pioneered by Clockwork (OSDI '20; *not* cited). The contribution is the specific adaptation to LLM serving, but the NeurIPS bar for ML-systems papers in the main track usually expects either a theoretical result (e.g., a new Po-N analysis under correlated-arrival workloads) or a new ML component (e.g., a learned scheduler).

2. **No variance / seed reporting.** Headline claims like "Po2 matches Fanout within noise" (31.6 vs 31.7 QPS) and "6-9% lower mean e2e latency under α=0.25" are reported as point estimates without standard deviations, confidence intervals, or multi-seed runs. For a paper whose central argument is *randomization matters*, the absence of run-to-run variance characterization is a meaningful methodological gap.

3. **Herding is argued mechanistically but not measured directly.** §3 and §6.6 describe herding verbally and measure its effect via α=0.25 gamma-arrival e2e latency, but the paper never plots the quantity that would prove it: the per-instance dispatch-count distribution under Fanout vs Po2 within a round-trip window. Without that, "herding" is an interpretation of an effect rather than an observed phenomenon.

4. **Length-prediction error is not trivially small and the downstream implications are under-explored.** MAPE 24.4% / Acc-100 = 77% means ~23% of requests are mispredicted by >100 tokens. §6.8 shows the dispatcher is robust to *multiplicative* Gaussian noise (relative ordering preserved), but real prediction errors are often *skewed* (long-tail underprediction on reasoning traces, code completions). A distributional error analysis (e.g., what does the residual look like on long CoT responses?) would substantiate the robustness claim.

5. **Broader-impact / ethics / limitations discussion is absent.** NeurIPS expects a dedicated Limitations section and a Broader Impact statement. The paper has neither. While LLM serving efficiency is not ethically fraught, reviewers will flag the missing sections mechanically.

---

## Questions for Authors

1. What is the empirical standard deviation of capacity (QPS at SLO P99 TTFT ≤10s) across independent runs at fixed QPS, for at least one scheduler-config pair? Without this, "Po2 matches Fanout within noise" is an unverifiable claim.

2. Can you provide a direct measurement of scheduler herding? For instance, under the α=0.25 gamma-arrival workload, report the Kullback-Leibler divergence between the empirical dispatch distribution and the uniform distribution over instances, for Fanout vs Po2.

3. The Po4-est result (31.9 QPS) is higher than Po2-est (31.6 QPS). You recommend Po2 over Po4 on herding-mitigation grounds, but the N=4 probe set is also randomized. Is the argument that Po4 has 2× higher predictor-query cost for only 1% capacity gain, or is there a principled reason herding mitigation is strictly better at N=2 than N=4?

4. How does your simulator-based `predict` RPC's tail latency (p99) behave as the target instance's in-flight queue grows? If simulation cost scales with queue depth, the scheduler's own overhead becomes load-correlated, which could amplify rather than dampen bursts.

5. Why is the Llumnix end-to-end comparison in §6.5 restricted to Llama-2-7B when §6.2.2 shows prediction accuracy on Llama-2-70B? Does the 70B Llumnix-vs-Astrolabe gap widen, narrow, or invert?

6. The length predictor is trained on ShareGPT. What happens on distribution shift — e.g., a BurstGPT-only or code-generation-heavy workload — without retraining? Table 2's Qwen retraining hints at brittleness here.

---

## Limitations (explicitly addressed in the paper?)

**Partially.** The paper acknowledges:
- Length-prediction error upper-bounds capacity gain (§6.3: "bounded ~3% capacity impact").
- A30 cluster precludes Llumnix with migration (§6.1), forcing the A100 comparison.
- Qwen requires length-tagger retraining (Table 2).
- BurstGPT lacks prompt text, so estimated-mode cannot be evaluated (Table 2 caption).

**The paper does not discuss:**
- Sensitivity to distribution shift over time (model drift, workload shift).
- Cost/latency of the per-instance simulator under very deep queues.
- Behavior at much larger cluster scale (100+ instances) where Po2's advantage over Fanout should theoretically widen but has not been empirically verified.
- Failure modes when the RoBERTa length model misfires pathologically (e.g., adversarial prompts).
- Energy or cost tradeoffs of running N predictors vs using live migration.

A 0.5-page dedicated Limitations subsection before §7 would close this gap for NeurIPS.

---

## Ethics Review Flag

**No ethical concerns flagged.** The paper presents an efficiency/scheduling system for LLM inference with no data-privacy, dual-use, or fairness implications beyond the general concerns that attach to all LLM deployment infrastructure. No flagging action recommended.

Minor note: the RoBERTa length tagger is trained on ShareGPT conversations, which are scraped user interactions; the paper would benefit from a one-sentence acknowledgement of the dataset's provenance and licensing status.

---

## NeurIPS Reproducibility Checklist

| Item | Status | Notes |
|------|--------|-------|
| Code available | ✅ | `anonymous.4open.science/r/Astrolabe-BA60` (§5) |
| Data available | ✅ | ShareGPT, BurstGPT public; A30/A100 CloudLab testbeds reproducible |
| Hyperparameters | ✅ | bs=48, chunk=512, 16 predictors/node, Poisson arrivals specified |
| Training setup for length tagger | ✅ | RoBERTa-base, 125M params, ShareGPT 40k/10k split, L40 GPU |
| Compute requirements | ⚠️ Partial | Per-experiment compute time not given; only GPU counts |
| Claims match evidence | ⚠️ Partial | "Within noise" claims lack variance numbers |
| Broader impact statement | ❌ | Missing |
| Limitations section | ❌ | Not explicit; scattered across §6 |
| Multiple seeds | ❌ | Not reported |
| Error bars on figures | ❌ | Not visible in Figures 6, 8, 10, 11, 12 |

---

## NeurIPS-Specific Fit Assessment

The paper is a *systems* paper with a light ML component (a supervised length regressor). It sits on the boundary between classical NeurIPS material and MLSys/SOSP/SoCC material:

- **Main track fit:** Marginal. The ML novelty is limited to adapting a well-known load-balancing primitive (Po-N) to LLM serving, plus a standard BERT-family fine-tune. NeurIPS main-track reviewers from an ML background may argue the work fits better at MLSys, OSDI, NSDI, or SoCC.
- **ML Systems / Datasets & Benchmarks track:** Better fit. The emphasis on empirical evaluation across realistic serving workloads, open-source release, and reproducibility aligns with that track's expectations.
- **Workshop fit:** A strong fit for MLSys-adjacent NeurIPS workshops (e.g., ML for Systems, Efficient Natural Language and Speech Processing).

The paper as-written is formatted for SoCC '26, which is the most natural venue. If retargeted at NeurIPS, the authors would need to (a) add a theoretical analysis section (e.g., bounding herding probability under correlated arrivals for N=2 vs N=12), (b) expand the length-prediction section to be the ML contribution (e.g., a new training recipe or distribution-shift analysis), or (c) submit to the ML Systems track with softened ML-novelty claims.

---

## Scores

- **Overall Score:** **5 / 10 (Borderline accept)**
  Justification: The work is technically sound and empirically well-executed for its scope, but the ML novelty under NeurIPS's main-track rubric is thin. Under the ML Systems track or at a systems venue, this would score higher (~6.5/10 in the generic-format review). The score discount reflects venue fit, not technical quality.
- **Confidence:** **3 / 5 (fairly confident)**
  Reviewer is familiar with LLM serving, classical load balancing, and the BERT-family regressor training regime; less familiar with NeurIPS ML Systems track norms (hence 3, not 4).
- **Soundness:** **3 / 4** — methodologically sound; variance reporting is the gap.
- **Presentation:** **3 / 4** — well-written; one figure-caption mismatch (Fig. 7) and minor readability issues.
- **Contribution:** **2 / 4** — incremental combination of established primitives; novelty is in the framing (herding) and system integration.

---

## Recommendation

**Borderline accept** for NeurIPS ML Systems / Datasets & Benchmarks track. **Reject** for NeurIPS main track under standard ML-novelty expectations, with clear encouragement to resubmit to SoCC / MLSys / OSDI where the contribution is a natural fit. The authors should:

1. Add a Limitations section and a Broader Impact statement.
2. Report run-to-run variance and seeds on the headline capacity/latency numbers.
3. Add a direct measurement of scheduler herding (dispatch distribution under correlated bursts).
4. Cite and contrast Clockwork (OSDI '20) and Nexus (SOSP '19) as direct ancestors of simulation-based + power-of-choice serving schedulers.
5. If staying at NeurIPS, consider adding a short theoretical section (a Chernoff-style bound on probe-set collision probability under correlated arrivals) to shore up the ML-theory angle.
