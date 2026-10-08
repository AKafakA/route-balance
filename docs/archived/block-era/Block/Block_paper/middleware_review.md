Middleware26 Paper #12 Reviews and Comments
===========================================================================
Paper #12 Block: Balancing Load in LLM Serving with Context, Knowledge and
Predictive Scheduling


Review #12A
===========================================================================

Overall merit
-------------
2. Weak reject

Reviewer expertise
------------------
3. Knowledgeable

Paper summary
-------------
Block presents a predictive scheduler for multi-instance LLM serving that combines response length prediction with simulation-based latency prediction (adapted from Vidur's offline simulator) to route requests to optimal instances. The system broadcasts each request to all instance predictors, collects predicted latencies, and selects the best instance. The paper also demonstrates using predicted metrics for preemptive auto-provisioning, showing P99 latency reductions compared to reactive provisioning.

Strengths
---------
- The preemptive auto-provisioning concept effectively demonstrates that provisioning based on predicted metrics avoids the over-provisioning issues of reactive strategies.

- Solid engineering work adapting Vidur's offline simulator for real-time use.

- Demonstrates performance improvements over heuristic baselines.

Weaknesses
----------
- Paper lacks clear focus and coherent narrative. The work tackles multiple problems (scheduling, auto-provisioning, simulation adaptation) without establishing a clear research thesis. Key concepts like "auto-provisioning" and "distributed/stateless" are poorly explained or misleadingly framed, and the relationship between contributions is unclear.

- Critical scope limitation hidden until Section 6.3. The testbed's single-GPU nodes with limited bandwidth makes Llumnix with migration infeasible, but this network-constrained scope is not stated in abstract/introduction. Production clusters with proper interconnects may see different trade-offs.

Comments for authors
--------------------
The core ideas have merit but presentation and evaluation issues prevents better score.

- Scope and framing issues:
State the network-constrained scope explicitly in abstract/introduction. Frame this as "we focus on commodity clusters where high-speed interconnects are unavailable" rather than claiming general applicability.
The "distributed/stateless" framing is misleading: broadcasting to all instances is centralized coordination with distributed computation. Be more precise about what you mean.
Typos: Section 3 "Although highly desirable and *came with*", "Sequence Scheduling [70] *and proposes*"; Section 4.1 "A model instance *is collection*".

- Experimental issues:
Figure 7, row 3: Why does Block* have more preemptions than Block at QPS 32-36? This needs investigation and explanation. 
Figure 7 is generally unreadable due to overlapping curves. Use subplots, different line styles, or data tables.

- Missing discussion:
How do results generalize to production with NVLink/InfiniBand? Would Llumnix with migration outperform Block?
Sensitivity to prediction errors: what happens when length prediction is substantially wrong?
Ablation studies showing value of simulation vs simpler predictive scheduling.

The preemptive provisioning insight is valuable and the engineering is competent, but the paper needs major revisions to: clearly state its scope upfront, explain or resolve the Block* anomaly, articulate the research contribution, and provide more rigorous evaluation of auto-provisioning.



Review #12B
===========================================================================

Overall merit
-------------
2. Weak reject

Reviewer expertise
------------------
3. Knowledgeable

Paper summary
-------------
This paper proposes a scheduling mechanism for concurrent serving of LLM requests. The scheduler employs components that predict the output length of a given request and simulate key target metrics that are then used for optimizing the scheduling of the requests. Evaluations show favorable performance to other schedulers.

Strengths
---------
- Clearly written paper on a relevant topic
- Extensive experiments

Weaknesses
----------
- Contributions above related work and novelty are unclear
- Paper organization: very long background and related work discussion that is not targeted
- Discussion of results unfair (Block* assuming perfect prediction of response length, with weak motivation)

Comments for authors
--------------------
Overall, the submission was interesting to read. The problem is clearly motivated and the building blocks of the solution are introduced. The related work and background section are too lengthy. At the same time, the discussion fails to highlight the key novelties of the proposed scheduler “Block”. The paper does not clearly describe how Block goes beyond Vidur; instead, it is repeatedly mentioned that this and that component / idea of Vidur is used in Block. 

In the technical section, it could be useful to formalize the actual scheduling algorithm, e.g. in a pseudocode listing. This would increase clarity above the pure verbal description that is provided. Also, the authors mention that they “fine-tuned a RoBERTa-base regression model”, but they don’t provide any details how; so this part is not self-contained and reproducible.

In the evaluation section, it is unclear to me why the authors employ the slightly outdated Llama-2 and not a a newer model such as Llama-3. 

Using Block* as a baseline is a bit misleading. Block* assumes perfect prediction of the response lengths, which seems not to be feasible / realistic. The authors argue that “prompt length could be available by prompt cache.” But this is a weak argument: if a prompt is cached and hence, served from the cache, it is not necessarily to schedule this prompt using Block. The interesting prompts are those that are not cached, and for those, we do not know the exact response length. I would propose removing the Block* results from the main discussion; they could be added as a hypothetical scenario into a separate paragraph. Also, other baselines could be equipped with perfect predictors to establish a level playing field.



Detailed remarks:

- Line 122: “at QPS as 32” → What does this mean?
- Typo in Figure 3 “Dynmaic Objects”



Review #12C
===========================================================================

Overall merit
-------------
2. Weak reject

Reviewer expertise
------------------
3. Knowledgeable

Paper summary
-------------
The paper proposes Block, a distributed, stateless, predictive scheduling framework for multi-instance LLM serving (and, in principle, preemptive auto-provisioning). Block aims to reduce serving “unpredictability” by combining response-length estimation with per-instance simulation-based latency prediction: each serving instance runs replicated, stateless Predictor sidecars (adapted from Vidur) that take a snapshot of the local queue and return predicted latency metrics. A global scheduler broadcasts prediction queries to instances and then routes each request to the instance with the lowest predicted latency (in their evaluated policy). On a 12×GPU CloudLab cluster serving LLaMA2-7B, they report improved TTFT/E2E latency and increased throughput/capacity under a TTFT P99 SLO, with engineering optimizations (hot-path reimplementation and caching) to keep prediction overhead low.

Strengths
---------
- Clear architecture & modularity. The system decomposition (length tagger, global scheduler, predictor sidecars, backend) is straightforward, and the predictor is designed to be stateless/replicable for parallelism.
- Grounded in backend reality (not just heuristics). The decision is based on predicted latency metrics from a local snapshot of each instance’s queue, rather than proxy load metrics alone.
- Attempts to quantify prediction accuracy and explain why it still helps scheduling. They report 10–15% absolute prediction error, yet the chosen instance is often near-optimal in rank terms.
- Pragmatic engineering on simulation overhead. They re-implement simulator hot paths and add caching; they also report scheduling overhead being small relative to E2E latency in the evaluated regime.

Weaknesses
----------

1) Novelty is moderate and feels “composition-heavy”
The paper’s core is a combination of (a) length prediction already used in prior systems and (b) simulation (Vidur-like) adapted to online scheduling. The novelty is mainly systems integration + distributed stateless orchestration, plus some cache/parallelism engineering. That may be enough for a systems venue, but the paper should be more explicit about what is innovation compared to previous system design.

2) Evaluation realism and generality are under-supported
- The evaluation uses a single-node-fit model (LLaMA2-7B) because multi-GPU parallelism is infeasible in their testbed.
This is a big limitation: modern production LLM serving often involves larger models, tensor/pipeline parallelism, and different bottlenecks (CPU/network/control-plane), which may alter both the predictability and the cost/benefit of simulation-based scheduling.
- Workload: Poisson arrivals from ShareGPT, greedy decoding, fixed batching params.
That’s fine as a baseline, but a strict reviewer will ask for heterogeneous and burst/mix patterns, and multi-tenant interference scenarios.

3) “Distributed stateless broadcast” scalability is asserted more than proven
The scheduler broadcasts prediction requests to all instances and waits for results; the paper argues latency is bounded by the slowest predictor, not cluster size.
But at real scale, the bottleneck is often:
- control-plane fanout/fanin,
- head-of-line blocking from stragglers,
- network + serialization overhead (they even note JSON/GIL overhead and suggest gRPC).
There is no scaling study (e.g., 12 → 48 → 192 instances) to support the claim that this approach remains low-overhead and stable.

4) Reported wins are sometimes small relative to complexity (and need sharper framing)
Authors report overheads and improvements that vary: e.g., extra overhead ~80ms within capacity (for Block*) and “overhead below 3% of E2E latency” in some conditions.
Capacity gains shown in one cited example are ~1–4% (30.7→31.1→32.0 QPS) , while the abstract claims up to 13.8% capacity improvement.
This mismatch is not necessarily wrong (maybe other configs yield 13.8%), but the paper should foreground when/why gains are large, and when they are marginal.

Comments for authors
--------------------
Questions for the authors

- Scalability: How do scheduling overhead and tail latency change when scaling to 64–512 instances? Since Block relies on broadcast fanout plus per-instance prediction, control-plane overheads and stragglers are a concern at larger scales.
- CPU cost / interference: You run 16 Predictor replicas per node to reduce prediction latency. What is the resulting CPU utilization, and does Predictor compute contend with the serving stack (e.g., tokenization, networking, runtime threads)?
- Robustness to heterogeneity: How stable are predictions and scheduling decisions under heterogeneous GPUs, throttling, or background contention? The evaluation appears largely on homogeneous nodes, so robustness to non-uniform performance remains unclear.
- Snapshot consistency: Predictors simulate based on a snapshot of the local execution queue. How sensitive is Block to drift between snapshot time and actual dispatch (race conditions / queue changes), especially under high load?
- Baselines: Could you include at least one stronger baseline that uses richer but cheap signals (e.g., prefill queue length, decode queue length, KV-cache pressure / memory headroom) without full simulation, to better isolate when simulation is necessary?

Improvement suggestions:
- Add a scaling study (or emulation) demonstrating control-plane stability and tail behavior as instance count grows.
- Evaluate a broader workload suite: bursty arrivals, mixed short/long prompts and outputs, mixed decoding settings, and multi-tenant interference.
- Provide a clean ablation: caching impact, Predictor replication factor, chunked prefill vs prioritized prefill, and sensitivity to length-estimation error.
- Clarify claims boundaries: what Block improves or guarantees without migration/rebalancing, and how it complements (rather than replaces) those mechanisms.
Explain