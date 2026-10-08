# Paper Review: Astrolabe: Balancing Load in LLM Serving with Context, Knowledge, and Prediction-Guided Scheduling

**Venue target**: EuroSys (systems-conference format)
**Severity**: standard
**Review date**: 2026-04-22
**Reviewer confidence**: High (familiar with LLM serving, scheduling, and systems evaluation; have skimmed Llumnix, Vidur, ELIS, S³, Sequence Scheduling)

---

## Summary

Astrolabe is a one-shot, prediction-guided request dispatcher for multi-instance LLM serving on commodity, network-constrained clusters where KV-cache migration is infeasible. For each request, Astrolabe (i) estimates the response length with a 125M-parameter RoBERTa regressor, (ii) samples two candidate instances (Power-of-2 choices), (iii) runs a Vidur-based per-instance simulation in each candidate's predictor sidecar to estimate the target metric (default: E2E latency), and (iv) dispatches to the lower-predicted-latency candidate. The central technical claim is that randomized dispatch (Po2) combined with simulation-based per-instance latency prediction delivers load-balancing quality equivalent to full fanout (N=12) at ~2.8× lower per-predictor CPU, while avoiding a newly characterized "scheduler herding" failure mode of deterministic fanout under bursty arrivals. Evaluation spans a 12-GPU A30 cluster (Llama-2-7B on ShareGPT/BurstGPT, with Qwen2-7B generality) and a 2-node A100 cluster where Llumnix's migration-based rebalancing is enabled; Astrolabe wins mean and P99 TTFT/E2E at 15 of 17 QPS points and reports up to 2.6× throughput and 123× per-token latency reduction vs. full Llumnix at saturation on A100.

## Strengths

1. **Clear problem framing that matches a real operational niche.** The paper targets *PCIe-only, Ethernet-connected commodity clusters* — the setting where migration-based schedulers (Llumnix, DistServe) cannot apply. This is underserved in the recent LLM-serving literature, which has largely assumed NVLink/InfiniBand fabrics. Grounding the contribution against this constraint (§1, §2) makes the design choices defensible rather than arbitrary.

2. **"Scheduler herding" is a useful and concrete abstraction.** §1 and §6.5 articulate *why* deterministic fanout schedulers pathologically route many near-simultaneous requests to the same instance (correlated predictions across a burst), and §6.7 stress-tests this with gamma-distributed bursty arrivals (α=0.25, 0.5, 1.0, 2.0). This is a sharper framing than the generic "convoy effect" language used elsewhere in the serving literature.

3. **Comprehensive evaluation design.** The QPS sweep covers 17 operating points from moderate load through saturation, including both mean and P99 metrics for TTFT and E2E. The ablations in §6.5–§6.10 (N∈{2,4,8,12} tuning, oracle-length upper bounds, bursty arrivals, Gaussian noise injection on length/latency estimates, CPU/memory per-predictor, cross-workload generality on Qwen2-7B/BurstGPT/tight-batching) are exactly the right questions a skeptical reviewer would ask, and the results are reported honestly (e.g., conceding the Po4 sweet spot over Po2).

4. **Honest framing of the Po2 vs. Fanout trade-off.** The paper does not oversell Po2 — it shows Po4 is the empirical sweet spot (+0.3 QPS capacity), that oracle lengths lift capacity by only ~3%, and that under noise Po2's E2E degrades at most ~5.1%. This restraint is credible and aligns with classic Po2 theory.

5. **Pluggable, decoupled architecture.** The Predictor-as-sidecar design with a narrow `status`/`predict` interface (§4, §5) is a reasonable software-engineering contribution for a systems paper — the claim that the simulator backend can be swapped (Vidur/SimAI) and the inference backend can be swapped (vLLM/SGLang/LightLLM) is backed by the simulator abstraction. A ~4K-LoC prototype integrated with production vLLM 0.7.2 is a real artifact.

6. **Head-to-head against Llumnix with migration enabled (§6.4).** This is the most important experiment. Showing that a no-migration design can match or beat a migration-based one on its native 100 Gbps testbed is the paper's strongest claim, and the A100 results (QPS≤19 Astrolabe and Llumnix within 10%, QPS≥20 Llumnix collapses to network contention) are consistent with the motivation.

7. **Open artifact with anonymous repo.** Code released, version pins given, public datasets (ShareGPT/BurstGPT). Good signal for reproducibility.

## Weaknesses

1. **Novelty of the individual ingredients is modest.** Length prediction (Sequence Scheduling [69], S³ [28], SSJF/Qiu 2024), Vidur-based simulation [2], and Po2 [37/45] are all pre-existing. The contribution is their *combination* plus the herding analysis — which is fine for a systems paper, but the narrative occasionally overstates novelty ("among the first systems to combine response-length estimation with per-instance simulation for online global request dispatch" — §4). TetriServe already uses length estimation + Po2 for global scheduling; the paper's distinction from it is the simulation layer, which should be stated explicitly.

2. **The "scheduler herding" claim is demonstrated indirectly.** Figure 7 (preemption counts) and §6.7 (burstiness sweep) are suggestive but not a direct micro-benchmark of herding. A controlled experiment — e.g., issuing a burst of K simultaneous identical requests and counting how many Fanout vs. Po2 routed to the same instance — would nail the mechanism. As presented, herding and load-aware overload are conflated in the measurements.

3. **Length-prediction accuracy is weak.** 24.4% MAPE and only 77.15% Acc-100 on ShareGPT means the predictor is often wrong by tens to hundreds of tokens. The paper argues this is fine because (a) Po2 uses *relative* ranking, (b) prediction noise is absorbed by the retry/extend mechanism, (c) §6.8 shows bounded degradation. This argument is defensible but not airtight — §6.8 sweeps *Gaussian* noise, but real predictors have *structured* bias (e.g., systematically underestimating long outputs). A residual-structure analysis of the 24.4% error would strengthen the claim.

4. **Simulator accuracy at scale is concerning.** Vidur reports 16.6% latency-prediction error on the A100 Llama-2-70B setup (§6.2) vs. <9% originally reported by Vidur authors. The paper waves this off as "relative ranking is what matters," but a 16.6% error is large enough that the scheduler can plausibly invert two close candidates' ranking. A calibration plot (predicted vs. actual latency) per instance, with Kendall τ / Spearman ρ against the oracle ranking, would be the right diagnostic.

5. **Llumnix- is a legitimate but under-justified baseline weakening.** The A30 evaluation uses Llumnix-*minus*-migration because the cluster can't support migration. The paper correctly notes this in §6.1 but the reader is left wondering how much of Llumnix's advantage is being stripped. The A100 comparison (§6.4) addresses this only partially — it's a different cluster, different model scale, and single data point. Showing Llumnix-full on the A30 cluster with a *documented* performance collapse (even if it's "migration fails and falls back to local") would make the case airtight.

6. **Single-SLO, single-tenant evaluation.** §5 says "supporting single- and multi-SLO strategies" but all experiments use a single P99 TTFT ≤ 10s SLO. No multi-tenant results, no priority classes, no mixed latency/throughput SLOs. For an EuroSys paper targeting "production multi-instance frameworks," this is a noticeable gap.

7. **Scheduling overhead at high load.** Table 2 shows Astrolabe's TTFT mean at QPS=36 is 18,107 ms (vs. 2,586 ms at QPS=32) — clearly saturating. The per-request scheduling overhead *at capacity* is "~3% of E2E" but Table 2 average-overhead column shows 160ms at QPS=36 for Astrolabe. With 2 predictor queries per request, what fraction of that is RPC vs. simulation? A breakdown would help (FastAPI + REST is known to be heavier than gRPC).

8. **Missing comparison to recent length-prediction schedulers.** FastServe [Wu et al. 2023, MLFQ-based], SSJF [Qiu et al. ATC 2024], and Preble [Srivatsa et al. 2024, prompt-aware distributed scheduling] are not cited and not compared against. ELIS is cited but only mentioned in passing.

9. **The "123× per-token latency" headline number is an overload-regime artifact.** This is measured at saturation where Llumnix is in a network-contention collapse; a 40× ratio at QPS=30 (moderate saturation) would be more honest as the headline. The abstract should be recalibrated so reviewers don't discount the paper on first read.

10. **BurstGPT evaluation is thin (§6.10).** Only oracle-only numbers are shown for BurstGPT, and even those are in Table 3's last column. A full QPS sweep on BurstGPT (which has very different arrival patterns than ShareGPT) would be the strongest generality evidence.

## Major Issues

1. **Herding claim lacks a direct micro-benchmark.**
   The paper introduces "scheduler herding" as a distinct failure mode (§1, contribution 1), but the experimental evidence is indirect: Fig 7's preemption counts, §6.7's burstiness sweep, and §6.5's Po-N comparison. None isolate the herding mechanism cleanly.
   → **Suggested fix**: Add a micro-benchmark in §6.5 or §6.7 with a synthetic burst of K simultaneous identical requests with *known* correct dispatch. Measure the fraction of the burst that lands on the same instance under Po2 vs. Po4 vs. Fanout. This converts the narrative into a quantitative phenomenon.

2. **Simulator calibration not shown where it matters.**
   16.6% A100 latency prediction error is borderline for a scheduler that ranks instances by predicted latency. The reader cannot tell whether the error is common-mode (cancels in ranking) or differential (flips rankings).
   → **Suggested fix**: Per-instance calibration plot (predicted vs. actual latency, with 95% CI) plus rank correlation (Kendall τ) between simulator-predicted and ground-truth instance latency, evaluated at QPS ∈ {20, 28, 32, 36}. Report the fraction of dispatches where the top-predicted instance is also top-2 in ground truth.

3. **Baseline fairness for Llumnix-.**
   Llumnix's migration is its key contribution, and the A30 experiments remove it. The paper must convince the reader that (a) migration genuinely cannot run on A30+25Gbps, and (b) the remaining "Llumnix-" policy is the best adversary available.
   → **Suggested fix**: Either (i) show a real attempt to run Llumnix with migration on A30 and document the failure mode (migration timeouts, client errors, whatever happens), or (ii) reframe the A30 comparison as "against load-aware heuristics" and move the Llumnix-is-the-state-of-the-art argument entirely to §6.4.

4. **Scheduling overhead breakdown is missing.**
   Claimed "~3% overhead within capacity" hides the interaction between RPC (2× predict + 1× status × 2 probes = 6 RPCs per request) and simulation cost.
   → **Suggested fix**: Flame-graph or tabular breakdown of per-request scheduling time into (RPC round-trip × k=2, status-fetch, simulation, other). Report both mean and tail. Particularly at QPS ∈ {28, 32, 36}, show how overhead grows.

5. **Generality beyond ShareGPT is under-reported.**
   Table 3's "generality study" is compact — one column per variation, oracle-only for BurstGPT. Reviewers who care about production applicability will discount single-point results.
   → **Suggested fix**: Run the full 17-point QPS sweep for BurstGPT on Astrolabe-Po2 and at least one baseline (Llumnix-), and add the capacity-at-SLO bar to Figure 6 or a new figure. Include agentic or code workloads if feasible (these have very different length distributions).

6. **Length predictor's structured error is not characterized.**
   §6.8 uses Gaussian noise, but real RoBERTa regressors under-predict long outputs systematically — an error pattern Po2 may or may not tolerate.
   → **Suggested fix**: Plot residuals of the length predictor on the test split (predicted vs. actual, with decile-binned error bars). Then re-run a noise experiment with *bias-matched* structured noise, not Gaussian, to confirm robustness. This is a one-day experiment and directly addresses reviewer skepticism.

## Minor Issues

1. **Abstract wording**: "power-of-two-choices dispatch policy that samples two random instances per request" — emphasize "per request" and "independent" up front; some readers miss that the draw is per-request.
2. **§1 "lifts capacity by up to +6% on Qwen2-7B and +7.1% under tight batching"** — "tight batching" is jargon here; clarify as "bs=24" in the abstract.
3. **Fig 2 caption**: "Original vLLM and Chunked Prefill" — clarify that chunked prefill is the stall-free variant from Sarathi [3,4]; otherwise "chunked prefill" is ambiguous (vLLM has multiple flavors).
4. **Fig 5 is illegible at print size.** The 3×5 grid of small plots with three color layers is visually overwhelming. Consider moving some panels to an appendix, or splitting into two figures.
5. **Figure 8 y-axis labels**: Multiple subplots lack units (e.g., throughput subplot shows `tok/s` only in caption). Please label axes directly.
6. **Table 1**: MAE units are "tokens" — state this in the caption. Also: "Not reported" for prompt-based LLM's MAPE is fine, but worth noting that Acc-50/Acc-100 for the LLM baseline come from [69], not this work.
7. **§4.1**: "extends the current prediction at runtime by assuming the request will complete within the next 10 tokens" — this is a key correctness detail buried in prose. Call it out as a design choice and justify the constant 10.
8. **§5**: "about 4,000 LoC" — specify whether this includes the predictor sidecar, global scheduler, length tagger, or all three.
9. **§6.1 footnote 1**: Mentioning that the 7B length model from Sequence Scheduling was rejected due to cost is useful — consider moving this to the main text as a deployment consideration.
10. **References** — missing DOIs/venues for several entries ([2] Vidur cites arXiv but this appears in MLSys'24; [3] Sarathi cites arXiv but Sarathi-Serve is OSDI'24; [4] same). Please correct before camera-ready.
11. **Anonymized repo** URL in §5 is fine for submission; make sure it survives de-anonymization without breaking links.
12. **Typo §6.7**: "Po2 mean latency (α=0.25) 6–9% lower than Llumnix-" — the gap numbers are reported twice with slightly different framings between abstract and §6.7. Align to a single number.
13. **"Astrolabe-Po2" vs "Astrolabe-Po2-est" vs "Po2-est" vs "Po2-oracle"** — the nomenclature shifts between sections. Standardize a single table of variants up front (§6.1 would be the place).
14. **§7 Conclusion** is extremely terse (6 lines) for a systems paper — expand to include at least one limitation and one future-work item.

## Questions for Authors

1. **On herding:** Can you run the micro-benchmark suggested in Major Issue 1? Specifically, for a burst of K=20 simultaneous requests with identical prompts, what fraction of the burst does Fanout route to the same instance vs. Po2? This would directly validate the mechanism.

2. **On simulator error:** For the A100 Llama-2-70B setup, what is the Kendall-τ rank correlation between Vidur-predicted and ground-truth per-instance latency at QPS ∈ {20, 28, 36}? If τ is high despite 16.6% absolute error, that would explain why the scheduler still works — and would be a nice result to surface.

3. **On Llumnix migration fairness:** Did you attempt Llumnix with migration on the A30 cluster and observe it failing? If yes, what was the failure signature (timeout, client errors, KV-cache corruption)? If no, does the Llumnix authors' published migration-bandwidth lower bound (from [52]) exclude your 25Gbps regime?

4. **On length predictor generalization:** RoBERTa-base is trained on ShareGPT. When you test on Qwen2-7B-ShareGPT (Table 3), does the length tagger need re-training on Qwen2 outputs, or is the ShareGPT-trained tagger sufficient (since ShareGPT labels are model-independent prompts)?

5. **On SLO-aware scheduling:** §5 says the scheduler "supports single- and multi-SLO strategies." Are there any multi-SLO results that didn't make it into the paper? If no, what is the blocker — scheduler logic or experimental setup?

6. **On CPU overhead at scale:** At N=12 with 16 predictor replicas per node, you report 50 cores/node for Fanout (§6.9). How many predictor cores does Astrolabe-Po2 consume at N=2 on the same 64-core node? The "2.8× gap" is relative; the absolute figure matters for operators.

7. **On simulator cache hit rate:** §5 mentions a "per-Predictor cache that memoizes latency predictions per (batch size, token count)." What hit rate does this cache achieve in steady state, and how much of the 2.8× CPU advantage comes from caching vs. Po2 probing fewer instances?

8. **On anonymized artifact:** The repo link at https://anonymous.4open.science/r/Astrolabe-BA60/ — is the end-to-end harness included, or only the scheduler? For AE review, a one-command reproduction of a single capacity-at-SLO bar (even on a single-GPU cloud instance) would be ideal.

## Missing Related Work

| Paper | Key Contribution | Relevance | Where to cite |
|-------|------------------|-----------|---------------|
| **FastServe** (Wu et al., arXiv 2305.05920, 2023) | Preemptive MLFQ scheduler for LLM serving, exploits length as a hint | Alternative scheduling strategy using length info; direct prior art on length-aware LLM scheduling | §3 Response Length Prediction, §6 as a baseline |
| **SSJF** (Qiu et al., ATC 2024) | Speculative shortest-job-first with proxy-model length prediction | Highly similar design (length prediction + scheduling); 30–40% JCT reduction claim is comparable to Astrolabe's | §3 Response Length Prediction; should be compared or at least positioned |
| **Preble** (Srivatsa et al., arXiv 2407.00023, 2024) | Distributed prompt scheduling with prefix-cache-aware routing | Complementary dispatch axis; worth discussing as orthogonal to Astrolabe's length-aware dispatch | §3, §4.2 |
| **DualMap** (Yuan et al., 2026) | Power-of-two with dual hash for cache affinity + load balance | Recent work combining Po2 with LLM serving concerns | §3 or §4 (if known at submission time) |
| **Mitzenmacher et al. (2001/2012 surveys)** | Theory of Power-of-d-choices | [37] is cited; consider adding tighter bound references (the exponentially-better bound vs. random) | §4.2 when invoking Po2 theory |
| **LiveMind / AsyncServe** | Proactive length classification or async dispatch | Tangentially related; check if relevant | §3 |

Note: SSJF and FastServe are particularly important — they both use length prediction + scheduling, published before this paper, and are uncited. A reviewer will ask about them.

## EuroSys-Specific Checklist

- **Systems contribution is real**: ✓ ~4K LoC prototype on vLLM 0.7.2, not simulation-only.
- **Evaluation scale is reasonable**: Partial — 12 A30 GPUs + 2-node A100 is on the lower end for EuroSys but defensible given the "commodity cluster" framing. Larger-scale results would strengthen acceptance odds.
- **Real workloads**: ✓ ShareGPT + BurstGPT are standard; adding an agentic/code trace would help.
- **Head-to-head vs. SOTA**: ✓ Llumnix with migration on its native testbed — this is the right comparison.
- **Artifact available**: ✓ Anonymous repo linked; AE reviewers should be able to evaluate.
- **Production relevance**: ✓ Targets a real deployment constraint (no fast interconnect) that other work ignores.
- **Limitations acknowledged**: Partial — single-SLO and simulator-accuracy limitations are not explicitly discussed.

## Scores

- **Overall Assessment**: **Weak Accept** (borderline, leaning accept). Strong empirical results and a well-motivated problem, held back by claims that outrun evidence (herding mechanism, 123× latency headline) and some under-examined fairness concerns. If the authors address Major Issues 1–3 in revision, this becomes a clear accept.
- **Overall Score**: **6/10** (EuroSys scale: 1=strong reject, 10=strong accept; 5–6 = borderline, 7 = accept)
- **Confidence**: **4/5 (High)** — reviewer is familiar with the LLM serving and load-balancing literature.
- **Novelty**: **Medium** — the combination is novel, each ingredient is not. Herding as a named phenomenon is a genuine contribution.
- **Technical Soundness**: **Medium-High** — methodology is sound; two measurement concerns (simulator error at scale, indirect herding evidence) prevent High.
- **Significance**: **Medium-High** — addresses a real operational niche; results are strong against SOTA.
- **Clarity**: **Medium** — writing is clear but dense; some figures are hard to parse; nomenclature drifts across sections.
- **Reproducibility**: **High** — open artifact, version pins, public datasets, specific hardware.
- **Experimental Design**: **High** — comprehensive sweep, thoughtful ablations, oracle upper bounds, noise sensitivity. One of the better-designed LLM-serving evaluations recently.

## Additional Notes

This is a solid systems paper that correctly identifies an underserved operational regime (commodity clusters without fast interconnect) and delivers a practical scheduler with a real prototype and open artifact. The main weaknesses are rhetorical rather than scientific: the abstract over-claims ("123× lower per-token latency"), the herding mechanism is asserted more than isolated, and key fairness questions about the Llumnix baseline are brushed past. A revision that (1) adds a direct herding micro-benchmark, (2) reports simulator rank-correlation alongside absolute error, (3) clarifies the Llumnix-on-A30 impossibility, and (4) expands the BurstGPT and multi-SLO story would address all my concerns.

Conflict of interest: None declared.
