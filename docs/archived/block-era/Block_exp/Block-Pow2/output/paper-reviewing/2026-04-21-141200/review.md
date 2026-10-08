# Paper Review: Astrolabe — Balancing Load in LLM Serving with Context, Knowledge, and Prediction-Guided Scheduling

**Conference format:** generic academic (no conference specified)
**Severity:** standard
**Reviewer:** single-reviewer peer review
**Review date:** 2026-04-21

---

## Summary

Astrolabe is a prediction-guided, stateless, global load balancer for multi-instance LLM inference clusters. Each request is routed in one shot: a RoBERTa-base length tagger estimates the response length, the request is then fanned out to a small random subset of instances, each of which runs a local Vidur-based simulator to predict the per-instance end-to-end latency, and the request is dispatched to the lowest predicted-latency candidate. The core design choice is *power-of-N choices* dispatch with N=2 as default (Po2), which the authors argue matches the load-balancing quality of full fanout (N=12) while paying constant per-request predictor cost and — crucially — breaking "scheduler herding" (the tendency of deterministic fanout schedulers to route correlated bursty requests to the same instance). Evaluated on 12-GPU A30 and 2-node A100 clusters across Llama-2-7B, Qwen2-7B, ShareGPT, BurstGPT, and gamma-arrival traces, Astrolabe-Po2 matches Fanout capacity within 0.3 QPS while using 2.8× less per-predictor CPU, beats Llumnix-- by 6–9% mean e2e latency under bursty (α=0.25) arrivals, and outperforms KV-migrating Llumnix on A100 by up to 2.57× throughput and 123× per-token latency at saturation.

---

## Strengths

1. **Well-motivated problem and clean contribution framing.** The three costs of fanout (linear control-plane fanout, linear aggregate CPU, and herding) are named, measured, and each mapped to a specific design choice. The herding phenomenon is a genuinely interesting failure mode that is rarely articulated explicitly in LLM serving work.

2. **Comprehensive empirical evaluation.** Two clusters (A30, A100), two models (Llama-2-7B, Qwen2-7B with TP=4 for Llama-2-70B in prediction-accuracy study), three arrival patterns (Poisson, gamma α∈{0.25,0.5,1.0,2.0}, BurstGPT traces), and four backend configurations (bs∈{24,48}, chunk∈{512,2048}). The head-to-head comparison with Llumnix is particularly strong because migration is actually enabled (`rayrpc` backend with chunked prefill disabled as Llumnix requires), rather than a strawman configuration.

3. **Honest ablation of the N-tunable family.** Figure 9 + §6.6 shows that Po4 is the empirical sweet spot (31.9 QPS vs Po2 31.6 vs Fanout 31.7) — a 1% capacity gain. The authors present this clearly and still recommend Po2 as default on principled grounds (herding mitigation, constant CPU). This is more self-critical than a typical systems paper, which strengthens rather than weakens the position.

4. **Prediction-noise sensitivity study (§6.8) is well-designed.** Injecting independent Gaussian noise on length × latency in a 4×4 grid is a clean way to isolate how prediction fidelity affects dispatch quality. The worst-case finding (+5.1% e2e at 100%/100% noise) is a surprisingly mild degradation that supports the "relative ranking, not absolute prediction" argument.

5. **Reproducibility posture is good.** Code is open-sourced on anonymous.4open.science, 4,000 LoC is a reasonable scope, hyperparameters are specified (bs=48, chunk=512, Poisson arrivals, FlashInfer backend, etc.), and the RoBERTa length-tagger training setup is documented (ShareGPT 40k train / 10k eval, L40 GPU).

6. **Prediction-guided + randomized is a genuine novelty.** Most prior LLM schedulers either (a) do rule-based load-aware dispatch (Llumnix, INFaaS++, MinQPM), (b) do learned routing on output quality (RouteLLM), or (c) do predictive but deterministic dispatch. Combining simulation-based per-instance latency prediction with randomized small-N probing is, to the reviewer's knowledge, novel in this specific form.

---

## Weaknesses

1. **Capacity gap to baselines is small and noise bounds are not quantified.** The headline numbers on the A30 cluster — Po2 31.6 QPS vs Fanout-est 31.7 vs Llumnix-- 31.5 — are within ±0.3 QPS (<1%). The paper claims "matches Fanout within noise" without ever measuring what "noise" is (run-to-run variance, multiple seeds, bootstrap CIs). The larger gaps (Po2 vs RR 23 QPS, vs MinQPM 22 QPS) are against weak baselines that few modern systems would deploy. The question "how much of Astrolabe's win comes from the scheduler vs from having prediction at all?" is only partially answered by the Po2-est vs Po2-oracle pair (31.6 vs 32.4 = 0.8 QPS attributable to prediction accuracy).

2. **Scheduler herding is argued but not directly measured.** The herding claim is central to the paper's narrative but §6.6 only measures its *effect* indirectly (e2e latency gap at α=0.25). A direct measurement — e.g., "instance selection frequency distribution under Fanout vs Po2 at α=0.25" or "fraction of concurrent arriving requests that land on the same instance" — would make the point concretely and turn herding from a plausible story into an observed phenomenon. The third row of Figure 5 hints at rank distributions but covers Astrolabe-only selection positions, not the herding comparison.

3. **Llumnix comparison is at small scale (8 instances).** §6.5 argues Llumnix's migration is "impractical" on the A30 cluster (25 Gbps Ethernet) and therefore moves the comparison to a 2-node A100 cluster with 8 instances. But 8 instances is roughly the minimum scale where KV migration could pay off. At 100-node scale, migration might amortize better, and Po2's constant-probe cost would become more competitive rather than less. The paper's positive framing ("Astrolabe is better than Llumnix") would be sharper if framed as "Astrolabe is better than Llumnix in the common network-constrained regime; migration may catch up at higher interconnect bandwidth."

4. **Length-prediction accuracy is modest and the downstream implication is under-discussed.** MAPE 24.4% and Acc-100 = 77% mean ~23% of requests are substantially mispredicted (>100 tokens off). §6.8 shows dispatch is robust to this via relative-ranking arguments, which is reassuring, but the paper never explicitly asks: *what workloads would break this*? High-variance workloads (e.g., reasoning with very long CoT responses) where underprediction can be arbitrarily large seem most at risk.

5. **Per-instance simulation cost is not rigorously characterized.** The Predictor is described as "simulation-based" using a modified Vidur. §6.9 measures per-predictor mean CPU (20.3% of one core for Po2 at QPS=20), but there is no latency distribution for the simulator itself (p50/p99 per `predict` RPC), no characterization of how it scales with the number of in-flight requests at the target instance, and no discussion of what happens at scale (100+ instances, each running its own simulator). Table 2 shows robustness under bs/chunk variations but this is the scheduling outcome, not the predictor latency.

6. **Generality claim on A100 Llama-2-70B rests on prediction accuracy alone.** §6.2.2 reports MAPE 16.6% for Llama-2-70B on A100 at QPS=3 with 10% sampling. This is good evidence the predictor generalizes but is not an end-to-end serving experiment on the 70B model. The A100 §6.5 Llumnix comparison uses Llama-2-7B, not 70B, so the practical question "does Astrolabe-Po2 beat Llumnix on a production-scale 70B deployment?" remains open.

7. **Figure 7 caption appears to be mismatched with its content.** The Figure 7 caption (page 11) reads "Average/Variance of GPU memory Blocks and Total Number of Preemption Under different QPS," but the panels shown are labeled "TTFT P99 vs QPS" and "Throughput vs QPS." Either (a) the caption and figure are swapped with another figure, or (b) the figure is out-of-order. This should be reconciled before camera-ready.

---

## Major Issues

1. **Missing statistical noise characterization.**
   → Suggested fix: Run at least 3 seeds per configuration at the key QPS points (≈30, 32, 34 on A30) and report mean ± std, or bootstrapped 95% CIs, on capacity and P99 TTFT. Specifically, the "Po2 matches Fanout within noise" claim needs a quantified noise floor; the paper currently relies on a visual "within ±0.3 QPS" statement that could be 1σ or 0.1σ — the reader cannot tell.

2. **Direct measurement of herding.**
   → Suggested fix: Add one plot or mini-table showing *dispatch target distribution* under Fanout vs Po2 at α=0.25 bursty arrivals. For each wall-clock window of 1 round-trip time, report the fraction of requests that land on the same instance. This would move the herding claim from "plausible mechanism" to "observed phenomenon" and would justify the randomization argument end-to-end.

3. **Predictor latency distribution missing.**
   → Suggested fix: Figure 6's "Average Overhead" panel shows total scheduling overhead but not the decomposition into `predict` RPC time vs simulator compute vs RPC serialization. Adding a p50/p99 breakdown — even as a small supplementary table — would let readers evaluate whether Astrolabe is deployable on stricter SLO budgets.

4. **A100 §6.5 scale/model clarification.**
   → Suggested fix: Clearly state in §6.5 that (a) the Llumnix comparison is on Llama-2-7B (not 70B), (b) 8-instance 2-node scale is smaller than Llumnix's original testbed claim of favorable regimes, and (c) the 70B result in §6.2.2 is prediction-accuracy-only. Alternatively, add a small 70B end-to-end §6.5-equivalent experiment if GPU budget permits.

5. **Related-work positioning against DNN-serving load balancers.**
   → Suggested fix: Add a paragraph in §3 contrasting Astrolabe against classical DNN-serving load balancers that use power-of-choice or predictive dispatch: Nexus (SOSP '19), Clockwork (OSDI '20), AlpaServe (OSDI '23), and — for fundamental LB theory — Consistent Hashing with Bounded Loads (Mirrokni et al.). The current §3 focuses on LLM-specific schedulers and treats the classical LB literature as a single Mitzenmacher citation. The contribution would be clearer framed as "we adapt classical Po-N + per-instance prediction to the LLM serving setting with its specific queueing and batching dynamics."

---

## Minor Issues

1. **Abstract, line 13:** "commodity clusters (e.g., Ethernet-connected GPU fleets without high-speed interconnects)" — the A100 testbed uses 100 Gbps NICs which is borderline commodity. Soften or qualify ("without NVLink-class intra-node interconnects").

2. **§6.2.1, line 970:** MAPE 24.4% is presented without a clear statement of whether this is acceptable. Consider "MAPE 24.4% (competitive with prior BERT-based response-length predictors)" with a citation.

3. **Figure 5** (page 9): The three rows use different x-axes (Sample ID, Sample ID, Distribution) making the reading flow awkward. Consider adding an explicit row-label column or splitting into three subfigures with clearer titles.

4. **Table 2 caption:** The "/" notation for BurstGPT with estimated mode ("Po2/Po2-oracle vs Llumnix--") needs parsing effort. Consider a separate BurstGPT sub-table or an explicit "N/A (oracle-only)" marker.

5. **Footnote 1 page 7:** "prohibitive" training cost for the 7B Sequence-Scheduling length model is vague. One concrete number (e.g., "~X A100-hours to reproduce") would be stronger.

6. **Reference [34] (llm-d):** "Accessed 2026-02-03" on a GitHub reference is appropriate but inconsistent with other `@misc` entries that lack the access date. Apply consistently or move to `@software`-style references across the bib.

7. **§6.3, "Latency under load":** the sentence "Both Astrolabe-Po2 matches Fanout..." at line 1150 has a grammar slip. Read: "Po2 matches Fanout at all load levels..."

8. **§4.1 line 610:** "simulator extends the current prediction by assuming the request will complete within the next 10 tokens (by default)" — this is a meaningful design default. Worth naming in the ablation or at least noting whether it ever fires in evaluation.

9. **§6.9 line 1551:** "Mean per-predictor CPU is 20.3% for Po2 vs 56.3% for Fanout at QPS=20" — consider adding a line plot of CPU utilization *over time* during a run, to rule out CPU-pressure spikes that the mean hides.

10. **Figures 6, 7, 8, 10:** All use legend colors that are similar across panels but the Po2-est line is sometimes red, sometimes different shades. A consistent color key (explicit color legend at figure bottom, shared across figures) would help.

11. **Abstract line 31:** "Beating non-predictive load-aware baselines (Llumnix--, INFaaS++), Astrolabe boosts serving capacity by up to 13.8%..." — 13.8% vs what baseline? Clarify (this is vs Llumnix-- under sub-optimal batching, per §6.10).

---

## Questions for Authors

1. **Seeds and variance.** How many independent runs back each QPS data point in Figures 6, 8, 10, 11, 12? What is the empirical run-to-run standard deviation on capacity and on mean e2e latency at a fixed QPS?

2. **Herding, directly.** Can you show the per-instance dispatch count distribution during the α=0.25 bursty experiment, for both Astrolabe-Po2 and Fanout, over a ~30-second window? If Po2 genuinely randomizes probe sets, we should see a flatter distribution.

3. **Predictor SLA.** What is the p99 `predict` RPC latency in §6.6 at QPS=34 on the A30 cluster? Does it stay bounded or does it grow with the target instance's queue depth (since the simulator walks the queue)?

4. **Cluster scale.** Would the herding advantage of Po2 over Fanout grow or shrink as N (number of instances) grows from 12 (your A30 setup) to say 100? Classical Po-N analysis says the probe-overlap probability under Fanout is exactly 1 (identical predictor set) regardless of N, so herding should remain at Fanout; Po2's effective target entropy grows with N, so the gap should widen. Can you add a small analytical argument or a simulated sweep?

5. **Length tagger brittleness.** What happens under distribution shift (e.g., switching from ShareGPT-trained tagger to BurstGPT traces)? You show in Table 2 that Qwen retraining is needed; what is the performance loss of using an untuned Llama-2-7B-trained tagger on Qwen-generated traffic?

6. **70B end-to-end.** Is the Llama-2-70B TP=4 on A100 evaluated only for prediction accuracy, or does the end-to-end Llumnix comparison in §6.5 run 70B anywhere? If not, can the authors comment on why (compute budget? scaffolding gap?) and whether they expect the Po2 vs Llumnix gap to tighten or widen at 70B scale where prefill cost per request grows substantially?

---

## Missing Related Work

The paper already cites Llumnix, Mooncake, DistServe, Splitwise, Sarathi, SGLang, vLLM, LightLLM, ELIS, QLM, S³, TetriServe, DynamoLLM, ServerlessLLM, RouteLLM, and Sequence Scheduling. The following would strengthen the related-work positioning:

| Paper Title | Key Contribution | Relevance | Should Be Cited In Section |
|-------------|------------------|-----------|----------------------------|
| Nexus: A GPU Cluster Engine for Accelerating DNN-Based Video Analysis (SOSP '19) | Power-of-choice routing combined with model co-location and batching for DNN serving | Directly adjacent: uses Po-N for serving, pre-dates LLM-specific work | §3 Related Work; a natural parent of the Po-N-for-serving line |
| Clockwork: Serving Models Like Clockwork (OSDI '20) | Predictable DNN inference via deterministic simulation-guided scheduling | Directly relevant: "simulation-based per-request prediction" is Clockwork's signature idea, applied here to LLMs | §3 and §4.2 (Global Scheduler) |
| AlpaServe: Statistical Multiplexing with Model Parallelism (OSDI '23) | Multi-model serving with queueing-aware dispatch | Relevant for contrast: multi-tenant multiplexing vs single-model instance pool | §3 Related Work |
| FastServe: Fast Distributed Inference Serving for LLMs (2023/2024 — Wu et al., cited in bib as `wu2024fastdistributedinferenceserving`) | Skip-join MLFQ preemptive scheduler for LLMs | Already in bib (`wu2024fastdistributedinferenceserving`) but appears uncited in text — should be discussed | §3 Related Work |
| LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism (SOSP '24) | Elastic sequence parallelism for long-context serving | Complementary: different axis of scheduling (sequence length), contrasting design point | §3 Related Work |
| Consistent Hashing with Bounded Loads (Mirrokni, Thorup, Zadimoghaddam, 2018) | Theoretical LB guarantee with bounded overload | Strengthens the theoretical framing beyond Mitzenmacher 2001 | §1 motivation or §3 |
| Join-Idle-Queue (JIQ) (Lu et al., Performance '11) | Classical dispatcher design: shortlist of idle instances, low-fanout | Pre-Po-N classical LB; orthogonal to but complementary with Po-N | §1 motivation |
| DeepSpeed-FastGen / MII (Holmes et al.) | Dynamic splitfuse and continuous batching for LLMs | Production-scale LLM scheduler contemporaneous with vLLM | §2 background or §3 |
| Preble: Efficient Distributed Prompt Scheduling for LLM Serving (EuroSys '25 or arxiv) | Prompt-aware load balancing exploiting KV-cache sharing | Directly relevant: another prediction-free but cache-aware LB approach | §3 Related Work |

---

## Scores

- **Overall Assessment:** Weak Accept
- **Overall Score:** 6.5 / 10
- **Confidence:** Medium-High (3.5 / 5) — reviewer is familiar with LLM serving and load-balancing literature; has not re-run the experiments but has read the compilation and the referenced figures.
- **Novelty:** Medium — combining simulation-based per-instance latency prediction, length prediction, and randomized Po-N dispatch in one coherent system is novel in the LLM-serving context, though each ingredient has precedent.
- **Technical Soundness:** Medium-High — design is cleanly described, ablations are thoughtful, but run-to-run variance is not reported and one figure caption is mismatched with its content.
- **Significance:** Medium-High — operators running network-constrained LLM clusters (which is the common case outside hyperscalers) have a concrete alternative to migration-based systems. The herding naming is a minor but real contribution.
- **Clarity:** Medium-High — generally well-written, though Figure 7 caption mismatch and Table 2's "/" notation are readability issues.
- **Reproducibility:** High — code is open-sourced, hyperparameters and backend stacks are specified, data traces (ShareGPT, BurstGPT) are public. Missing seeds/variance is the one gap.
- **Experimental Design:** Medium-High — two clusters, three workloads, four configurations, and a prediction-noise sensitivity study show care. The absence of variance reporting and the mismatched-scale A100 comparison are the main weaknesses.

---

## Recommendation

**Weak Accept with revisions.** The paper makes a focused, well-evaluated contribution to an important problem. The two main revisions needed for a camera-ready are: (i) variance/seed reporting for the headline capacity numbers, and (ii) direct measurement of the herding phenomenon that the paper's design pivots on. Weaknesses 3 and 4 (scale of Llumnix comparison, 70B generality) are worth addressing but likely too large for a revision cycle; they can be acknowledged as future work. Minor issues 1–11 are editorial.

The contribution is real — Po2 + simulation + length prediction is a coherent, minimal design for network-constrained LLM clusters — and the empirical case against Llumnix under bursty traffic and without NVLink is convincing enough to recommend acceptance. The ceiling of the paper's impact depends on how much the herding argument holds up as a first-class phenomenon vs as a secondary effect of randomization; a direct measurement would resolve that uncertainty.
