# OpenReview — AS ACTUALLY POSTED (authoritative record)

**Provenance:** pasted by the author from the live OpenReview page for Submission 28271
on **2026-07-29T07:28:55Z**, recovered from the session transcript on 2026-07-30 and
persisted here. 30,822 characters.

**This file — not `POSTED_2026-07-29.md`, not `Post/`, not `2026-07-27_POSTED_VIEW.md` —
is the record of what reviewers actually read.** Those other files are our local drafts and
reconstructions; they differ from this in at least one material claim (see below). Any check
of the form "does X contradict a posted claim?" MUST be run against this file.

```text
"OpenReview
.net
Search articles, authors and reviews...
Notifications3
Activity
Tasks
Wei Da 
back arrowGo to NeurIPS 2026 Conference homepage
RouteBalance: A Unified Router and Load Balancer for Heterogeneous LLM Serving
Download PDF
Wei Da, Evangelia Kalyvianaki 
04 May 2026 (modified: 28 May 2026)
NeurIPS 2026 Conference Submission
Conference, Senior Area Chairs, Area Chairs, Reviewers, Authors
Revisions
CC BY 4.0
TL;DR: RouteBalance unifies model routing and load balancing in heterogeneous LLM serving, making a single per-request decision directly over model instances to jointly optimize quality, latency, and cost
Abstract:
Heterogeneous LLM serving stacks split scheduling into two layers: model routers pick a model from quality and cost signals while ignoring instance load, and serving load balancers optimize queues while ignoring quality. We present RouteBalance, a serving-aware layer that fuses both into one online assignment over concrete model instances, jointly trading off quality, latency, and cost. A batched in-process predictor stack and a dead-reckoning instance-state cache keep the joint decision cheap on the request hot path. On a 13-instance, 28-GPU heterogeneous CloudLab cluster serving four model sizes, RouteBalance traces the upper region of the three-way Pareto frontier against trained-router baselines, holding per-request scheduling overhead at 149 ms at arrival rate 12 req/s and 220 ms at 30 req/s. Code: https://anonymous.4open.science/r/route-balance-3ECA/README.md.

Checklist Confirmation: I confirm that I have included a paper checklist in the paper PDF.
Responsible Reviewing: We acknowledge the responsible reviewing obligations as authors.
Primary Area: SysML infrastructure (e.g., libraries, improved implementation and scalability, distributed solutions)
Secondary Area: Language and multimodal language models (e.g., text generation, summarization, VQA)
Contribution Type: General: Most submissions will fall into this type.
Academic Integrity: I acknowledge that I have read the NeurIPS Handbook and commit to adhering to all policies in the Handbook (https://neurips.cc/Conferences/2026/MainTrackHandbook), the NeurIPS Code of Conduct and the NeurIPS Academic Integrity Policy.
LLM Usage: Editing (e.g., grammar, spelling, word choice), Implementing standard methods, Visualizing results for submission
LLM Experiment: Opt in to include this paper in the LLM-assisted peer review experiment.
Declaration: I confirm that the above information is accurate.
Reviewer Nomination:  Evangelia Kalyvianaki
Submission Number: 28271
Discussion
Filter by reply type...
Filter by author...
Search keywords...

Sort: Newest First
7 / 7 replies shown
Add:
Meta Review of Submission28271 by Area Chair axCN
Meta Reviewby Area Chair axCN25 Jul 2026, 06:24Senior Area Chairs, Area Chairs, Authors, Reviewers Submitted, Program Chairs, Area Chair axCN
Metareview:
This paper proposes RouteBalance, a serving-layer scheduler that jointly optimizes model routing and instance-level load balancing by considering quality, latency, and cost in a unified assignment framework. The reviewers agree that the paper addresses an important problem in heterogeneous LLM serving and that the proposed joint optimization framework is well motivated. However, the rationale of design choice and and evaluation part of the paper should be improved. Specifically, many design choices such as XGBoost estimator need to be better justified, and comparisons with general-purpose dispatchers can be added to illustrate the effectiveness of the proposed scheduler.

Add:
Official Review of Submission28271 by Reviewer 287j
Official Reviewby Reviewer 287j27 Jun 2026, 18:19 (modified: 28 Jul 2026, 20:01)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 287jRevisions
Summary:
RouteBalance is a system that aims to balance LLM model choice, request routing, and load-balancing. To my understanding, it is similar in philosophy to LLM-D or an AIBrix. The system combines some interesting ideas to enable better tradeoffs management.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths: The paper aims to solve the right problem and probably does a good job in solving it. The problem of routing LLM requests is one of the main points in LLM serving frameworks.

Weaknesses:

The paper is hard to follow in some places and some design decisions seem to be either arbitrary or not well explained. Some parts are also not quite clear to me. For example, you schedule in batches. I suppose that means that you do not use continuous-batching? If not, then what happens when you have an uneven batch? If you actually have Continuous-batching, how does this work with "RouteBalance schedules in batches. Each batch is the set of requests waiting at the scheduler when it fires, and the scheduler assigns every request to an instance before the next batch starts". Does that mean that you intentionally delay a request that arrived, still use continuous batching, but schedule only when you reach the required batch size/time?

Another example is the idea of adaptive size limit for batches. However, besides the vague sentence of " the limit grows when more instances are busy and shrinks when the cluster is mostly idle, so light load produces small batches and saturated load produces larger one", you never discuss how this mechanism works?

There are also several unclear points, e.g., how can the model choice affect the KV-Cache hit-rate? How much does it affects the TTFT (even though the filtration part is left for future work!)?

One suggestion is to walk through the components in Figure1 and discuss them in details in the appendix. Also, What about the datasets used, again something that should be added to the appendix.

For the evaluation, Figure 3 takes large space but in reality, does not tell much.

writing suggestions: -Maybe move Table 2 to be close to equation1 and 2.

Figure 1 can improve. I think the main issue is that the Figure does not really show the complete flow. For example, "estimate" does not return anything anywhere. Also, there are boxes that you never discuss in the paper, e.g., RoBERTa is there but never mentioned.
I think the appendix can include more information on, e.g., LPT, the methods you compare against, dead-reckoning which is AFAIK is not a standard term neither in LLM serving nor in distributed systems.
Quality: 3: good
Clarity: 2: not good
Significance: 4: excellent
Originality: 3: good
Questions:
See the weaknesses.

Limitations:
yes

Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
None

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Add:
Response to Reviewer 287j: Clarifying the Two-Layer Scheduling Design and Revision Commitments
Official Commentby Authors (Wei Da, Evangelia Kalyvianaki)29 Jul 2026, 03:58 (modified: 29 Jul 2026, 04:02)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 287jRevisions
Comment:
We sincerely thank the reviewer for the careful reading. We agree that several mechanisms were under-explained, and we will clarify them in the revision.

Related work clarify LLM-D and AIBrix provide serving infrastructure with routing and load-balancing only. RouteBalance (RB) focuses on a different policy boundary to join quality, latency, and cost when assigning each request to a concrete instance in heterogeneous deployment.

Q1: Batching versus continuous batching. There are two independent scheduling layers.

Global assignment in RB. RB selects the serving instance for each incoming request.
Engine-level continuous batching. After dispatch, each vLLM instance retains its normal continuous-batching loop and groups assigned requests into accelerator execution batches.
Thus, RB does not replace vLLM continuous batching. Once dispatched, a request immediately enters the selected instance’s normal engine-level scheduling loop.

RB additionally forms scheduler-level scheduling rounds. Incoming requests first enter a global queue, and the next (B) requests, where (B) is chosen adaptively, are assigned together. Before each round, RB starts from the latest instance telemetry. It then predicts per-request quality and output length with the KNN estimator and predicts TPOT for each candidate instance with the XGBoost heads.

There is no mandatory minimum batch size. A scheduling round fires when either the adaptive size limit is reached or the timeout expires. Therefore, a lone request waits at most the timeout, and all formation delay is included in the reported client-observed end-to-end latency.

The rationale is threefold:

After each assignment, RB immediately updates its local dead-reckoned view of the selected instance, so later requests in the same round do not repeatedly observe the same stale pre-dispatch state.
Prediction cost is amortized across the round because request embeddings and candidate scoring can be computed in parallel. Measured decision compute falls from 44.7 ms at 
 to 25.7 ms at 
Quality, cost, and latency are normalized over the candidates in the same round before weighted scoring, making the combined objective comparable across heterogeneous tiers.
Q2: Adaptive batch size. The current policy uses the fraction of instances with at least one waiting request:

ratio (<0.3): batch size 4, timeout 50 ms;
ratio (<0.7): batch size 16, timeout 100 ms;
otherwise: batch size 32, timeout 200 ms.
The intuition is that under light load, requests should incur only a short formation delay, so RB uses small rounds and short timeouts. Under high load, requests already experience instance-side queueing, so larger rounds improve prediction amortization and assignment quality while adding relatively little marginal delay.

Disabling adaptivity increases end-to-end latency by 5.1–6.9% as measured. With batch size 1, RB falls back to serial estimation and loses the amortization benefit of scheduler-level batching.

Q3: KV cache, prefix caching, and TTFT. The submitted objective is decode-dominated: predicted per-(model, GPU) TPOT multiplied by queue-aware decode iterations. TTFT and KV occupancy are not separate objective terms, although KV utilization and free-block count are features of the TPOT predictor. Model choice affects KV pressure through model size and sequence length.

The submitted evaluation uses single-turn prompts and prefix caching was disabled, so prefix-hit effects do not confound the comparison. Explicit TTFT/prefill and prefix-affinity terms are future work.

Q4: Figure 1, appendix, datasets, LPT, baselines, and terminology. Thank you for these detailed presentation suggestions. In the revision, we will:

redraw Figure 1 with the complete return path and corrected labels; the unused RoBERTa alternative will be removed or clearly marked as an optional estimator implementation;
add a full request-flow walkthrough and details of the seven workload sources in the appendix;
move Table 2 next to Equations 1–2;
replace the radar chart with compact operating-point tables;
add LPT pseudocode and references to explain the scheduling algorithm more clearly, illustrate the two-layer scheduling design, and analyze the trade-off between scheduling cost and scalability;
add an appendix table describing baseline training, routing policy, dispatcher, state access, batching/concurrency mode, and tuned parameters;
define dead reckoning operationally at first use: starting from the latest asynchronous telemetry, the scheduler locally applies each assignment’s predicted queue effect and reconciles the estimate when the next authoritative update arrives. We will also cite a prior distributed-simulation work using the same state-extrapolation pattern.
The learned-versus-static estimator and the general-purpose dispatcher choices are isolated experimentally in our replies to Reviewers gTfM and faAU.

Add:
Official Review of Submission28271 by Reviewer gTfM
Official Reviewby Reviewer gTfM26 Jun 2026, 16:35 (modified: 28 Jul 2026, 20:01)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer gTfMRevisions
Summary:
The paper proposes RouteBalance, a serving-layer scheduler for heterogeneous LLM clusters that assigns each request to a serving instance. It combines prompt-dependent quality/length prediction, load-aware latency prediction, and cost into a weighted online assignment objective. The evaluation uses a CloudLab cluster with Qwen2.5 models and compares against router-only baselines such as BEST-Route and AvengersPro.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Quality & Clarity: The paper is mostly clearly written, but some design choices are not well described or remain unclear to the reader. For example, why does TPOT require an XGBoost estimator? Can it instead be benchmarked statically? What are its advantages over simpler alternatives, and what exactly is its input?

Significance & Originality: The paper models a router that considers the ⟨cost, latency, quality⟩ of a pool of LLM instances. However, in real-world settings, the cluster (i.e., the model-to-instance assignment) is not fixed as assumed in the paper. In addition, the scheduler appears to be centralized, which limits its practical applicability. That said, this is the first work to demonstrate this capability.

Quality: 3: good
Clarity: 2: not good
Significance: 2: not good
Originality: 4: excellent
Questions:
1/ XGBoost: How much of RouteBalance's gain comes from the XGBoost TPOT predictor versus a simpler calibrated benchmark/lookup-table latency model using per-tier average TPOT and live queue state?

2/ TTFT/KV-cache: Does the latency predictor account for prefill/TTFT effects and prompt-length-dependent KV allocation, or is RouteBalance primarily optimizing decode TPOT? Also, how would prefix-cache hits or cache locality affect the instance-routing decision?

3/ Scheduler: Is the proposed scheduler centralized?

4/ Cluster: Is the cluster fixed (i.e., is the model-to-instance assignment static)?

Limitations:
See questions and weakness above.

Rating: 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
None.

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Add:
Response to Reviewer gTfM (estimator justification · deployment model)
Official Commentby Authors29 Jul 2026, 05:30Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer gTfM
Comment:
We thank the reviewer for these questions and for noting "this is the first work to demonstrate this capability".

Q1. We built exactly the alternative you propose: the same objective and the same within-tier queue tiebreak, with the learned head's queue-conditioned estimate replaced by a fixed per-tier TPOT constant. The constants are per-tier medians across the whole sweep rather than at one operating point — 3B 10.7, 7B 19.9, 14B 16.5, 72B 42.6 ms/token.

These cells and the submitted tables share the same benchmarking and evaluation setting, and both arms use RouteBalance(RB)'s uniform (1/3,1/3,1/3) weights.

λ    quality learned / static    cost $/1k learned / static    cheaper    throughput    mean e2e ms
6    0.5074 / 0.5096    0.02442 / 0.02656    8.1%    6.0 / 6.0    2,080 / 2,079
8    0.5076 / 0.5074    0.02405 / 0.02630    8.6%    7.9 / 7.9    2,118 / 2,137
10    0.5073 / 0.5084    0.02376 / 0.02605    8.8%    9.9 / 9.9    2,175 / 2,198
12    0.5073 / 0.5095    0.02372 / 0.02604    8.9%    11.8 / 11.8    2,243 / 2,224
18    0.5060 / 0.5079    0.02345 / 0.02583    9.2%    17.6 / 17.6    2,298 / 2,363
24    0.5043 / 0.5080    0.02327 / 0.02579    9.8%    23.3 / 23.3    2,458 / 2,487
28    0.5061 / 0.5071    0.02328 / 0.02568    9.3%    26.9 / 27.0    3,143 / 3,079
So, cost is what the learned head buys: 8.1–9.8% per 1,000 requests at every rate, at quality that is statistically indistinguishable — every |Δq| ≤ 0.004, all seven CIs span zero, widest [−0.0063, +0.0014] — throughput within 0.4%, and mean e2e within 2.7% in either direction. The mechanism is the routing mix: at λ=24 the head puts 56% on 14B and 38% on 7B, against 68% / 29% for the constant. A constant is fixed for the whole run and cannot tell a loaded instance from an idle one; 14B decode alone drifts 15.4 → 18.6 ms/token across the sweep. Learned latency prediction is established in LLM simulation stacks such as Vidur. We will cite this in the design section instead of related work.

But we also want to point out that the latency term itself is still necessary. To demonstrate it we designed a further experiment that removes latency and scores on quality and cost alone — unlike the two arms above, which both keep a latency term and differ only in how it is computed. Its score no longer depends on serving state, so routing goes load-blind: the identical assignment at every rate, 69.1% of requests to the smallest tier, and quality falls to 0.4899 against 0.504–0.510 above. So your Q1 is not whether latency belongs in the objective, but whether it has to be learned.

Latency prediction clarification. The head predicts TPOT only, one XGBoost head per (model, GPU) tier, trained from a tier-local QPS sweep. Features are that tier's serving-state signals plus the request's predicted output length; it does not consume request text. End-to-end latency is then formed analytically, not learned: predicted TPOT × the decode iterations the request waits through (pending decode tokens ÷ current decode batch size, plus its own predicted output length, from the KNN estimator).

Q2. TTFT is not an explicit term, as our target is decode-dominated generation. Prefix affinity is likewise not modeled: our corpus is single-turn, so cross-request reuse is minimal and the setting is uniform across compared systems. KV occupancy is not a term either, but KV utilization and free-block count are features the TPOT head consumes, reaching the assignment through predicted TPOT. We will list explicit prefill/TTFT and prefix-affinity terms as future work to improve the generality of the latency model.

Q3. Yes, centralized by design: one assignment decision per request over a global dead-reckoned view, which is what makes the joint comparison across heterogeneous instances possible. Its cost is measured: decision compute per request falls 44.7 → 25.7 ms from λ=6 to λ=24, as one prediction round amortizes over a batch. Scheduler fault tolerance and sharding can also be expanded as future work, e.g. a hierarchical multi-layer balancer with RB running per mini-cluster.

Q4. Yes, fixed per run — RB is an online assignment layer over a given deployment, not an autoscaler. It does respond to a changed deployment without retraining: in the topology experiment in our faAU reply we re-provisioned the small-model pool and re-ran every system with nothing retrained. Joint optimization of placement and assignment is future work.

In summary, we thank the reviewer for the static-TPOT suggestion, which we also applied to the new model-family experiments in our reply to Reviewer faAU, and for pointing to useful future directions.

Add:
Official Review of Submission28271 by Reviewer faAU
Official Reviewby Reviewer faAU26 Jun 2026, 02:41 (modified: 28 Jul 2026, 20:01)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer faAURevisions
Summary:
This paper presents RouteBalance, a serving-aware scheduling layer that fuses model routing (quality/cost-aware model selection) and load balancing (instance-level queue management) into a single online assignment problem. The key insight is that existing LLM serving stacks split these into two disconnected layers—routers ignore instance load, while load balancers ignore quality differences across models—resulting in suboptimal quality–latency–cost trade-offs. RouteBalance formulates scheduling as a joint assignment over concrete instances under a 3-simplex weight space (wq/wl/wc), using a batched in-process predictor stack (MiniLM+KNN quality estimation, XGBoost TPOT prediction) and dead-reckoning instance-state tracking. Evaluated on a 13-instance, 28-GPU heterogeneous CloudLab cluster serving four Qwen2.5 model sizes (3B/7B/14B/72B), RouteBalance traces the upper envelope of the three-way Pareto frontier. Per-request scheduling overhead is 149 ms at 12 req/s and 220 ms at 30 req/s. The evaluation spans 243 configurations and approximately 860K requests.

Contribution Type: Use-inspired: The main contribution is in framing or designing approaches to meet the needs of a specific real-world application. (This often involves, e.g., engaging with domain experts.)
Strengths And Weaknesses:
Strengths
Sharp observation and elegant formulation. The paper identifies a structural flaw in LLM serving stacks: model routers pick models while ignoring instance load, and load balancers optimize queues without reasoning about quality differences across models, and this decoupling prevents joint optimization of quality, latency, and cost. RouteBalance reformulates scheduling as instance-level online assignment under a 3-simplex weight space that captures the full trade-off space—insightful and elegant.

Strong empirical performance. On a heterogeneous cluster, RouteBalance forms the upper convex hull of the quality–throughput Pareto frontier, achieving lower latency and cost at iso-quality. Most notably, its scheduling overhead stays sub-linear as load grows 5× (141→220 ms), whereas BEST-Route inflates to ~33 s under the same architecture—demonstrating the effectiveness of batched amortization.

Pragmatic engineering design. Batch-level scheduling amortizes quality prediction across requests, an in-process XGBoost predictor provides lightweight per-instance latency estimates, and dead-reckoning state caching eliminates synchronous queries while avoiding herding. Pipeline mode hosts all decoupled baselines, ensuring comparisons isolate routing and dispatch policies—precisely targeted, no extraneous complexity.

Weaknesses
Single-cluster topology limits generality. All results come from one 13-instance CloudLab deployment using only the Qwen2.5 model family. While the heterogeneous GPU mix (A30/V100/A100) is representative, generalization to model families beyond Qwen or substantially different cluster topologies remains unverified.

Cold-start dependency of the quality estimator. The KNN-based quality estimator requires labeled training data per model; when deploying to a new model pool, the hottest routing paths lose signal until sufficient labeled data is collected. The paper acknowledges this limitation but does not quantify routing degradation under partial or missing quality signals, nor does it report the training-time cost of collecting the labeled dataset.

Dispatcher baselines can be strengthened. While the model routers (AvengersPro, BEST-Route) are appropriate competitors on the quality/cost dimension, the dispatcher baselines (round-robin, shortest-queue, random) are all basic strategies. Augmenting the comparison with more advanced general-purpose dispatchers such as least-connections or power-of-two-choices would more convincingly isolate the unique gain attributable to joint optimization, thereby making the core contribution more defensible.

Added tuning burden, and the budget gain does not come from the core mechanism. Fusing routing and load balancing exposes three simplex weights (wq, wl, wc) that operators must continually re-tune as the workload drifts, adding complexity over a quality-only router (acknowledged in §7). More tellingly, §5.4 shows the budget-control benefit comes mainly from the admission filter: without the filter, RouteBalance and BEST-Route are within 1.5 pp on budget-exhaustion at every mix, and the paper itself states the budget contribution "lives in the filter, not in the simplex weighting"—weakening the unique value of the "joint optimization" framing on the budget axis.

Quality: 3: good
Clarity: 3: good
Significance: 2: not good
Originality: 3: good
Questions:
See Weaknesses

Limitations:
The cold-start dependency of the quality estimator is the main limitation. We encourage the authors to add: generalization evidence across model families/topologies, the labeling scale and training time required for the quality estimator to become usable when a new model is onboarded, and the routing-degradation curve under missing quality signals.

Rating: 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
See above.

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Add:
Response to Reviewer faAU: Dispatchers, Generality, and Cold Start
Official Commentby Authors29 Jul 2026, 08:26Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer faAU
Comment:
We thank the reviewer for the thorough review.

W3: General-purpose dispatchers. We added standard power-of-two choices (Po2): sample two replicas uniformly and choose fewer inflight requests. We compare it with shortest-queue (SQ), trained routers, and RouteBalance (RB).

system    dispatcher    λ12    λ24 mean e2e ms
Passthrough    SQ    2051    2155
Passthrough    Po2    2257    2272
BEST-Route    Po2    2636    48182
AvengersPro (pw 0.7)    Po2    2665    2818
RB (uniform)    joint    2230    2408
SQ outperforms Po2 for passthrough. BEST-Route reproduces its 18.73 req/s serial ceiling under Po2, as under SQ/RR: at λ24 it reaches 48182 ms versus RB's 2408 ms (20.0×). Thus the collapse is upstream of dispatch: a dispatcher cannot revisit the selected model tier or remove routing-layer scoring overhead.

W1-A: New topology. We re-provisioned the A30 pool from 3B/7B replica counts 3/5 to 5/3 and reran all systems without retraining.

system    disp.    quality    tput    cost $/1k    mean    p99 ms
RB (0.5,0.3,0.2)    —    0.5138    23.25    0.0262    2636    6406
RB (1/3,1/3,1/3)    —    0.5057    23.24    0.0234    2470    5980
BEST-Route    Po2    0.5089    15.79    0.0220    43289    127754
AvengersPro    Po2    0.5037    22.49    0.0380    2914    13813
Passthrough    SQ    0.4899    22.66    0.0222    1887    11046
SQ leads mean latency and cost. RB offers the strongest quality/throughput-tail trade-off (p99 5980–6406 vs 11046 ms for SQ) while sustaining the offered rate. SQ reaches its lower mean by sending 55% to the enlarged 3B tier.

W1-B: New model family. We onboarded Gemma-3 (1B/4B/12B/27B), using measured static per-(model,GPU) tpot, the alternative discussed with Reviewer gTfM. Each cell uses 1979 held-out prompts from a disjoint 8086-prompt training split. Its input-price ladder is non-monotone: per 1M tokens, 1B costs 
0.04/4B and $0.08/27B reported by PricePerToken.

system    disp.    quality    tput    cost $/1k    mean    p99 ms
AvengersPro    Po2    0.6300    22.15    0.0259    3719    9582
RB (0.8,0.1,0.1)    —    0.6160    22.24    0.0220    2673    7952
RB (0.5,0.3,0.2)    —    0.6057    22.25    0.0208    2355    7758
RB (1/3,1/3,1/3)    —    0.6010    22.39    0.0194    2108    7393
Passthrough    SQ    0.5732    22.44    0.0227    1622    6714
RB is cheaper than every passthrough variant. Under the submitted quality procedure, it exceeds SQ/Po2 for every preset; against RR, two presets separate and uniform ties. SQ wins latency, serving 1B for 47% of requests. AvengersPro has the highest quality; versus RB (0.8,0.1,0.1), its +0.014 costs 18% more, with 39% higher mean latency and a 21% wider tail. BEST-Route is absent because extending it to Gemma-3 requires fine-tuning a fresh per-family supervised router, which did not fit within the rebuttal window.

W2-A: Partial/missing labels. At λ12 and preset (0.5,0.3,0.2), we rebuilt the KNN index with half the labels or without prompt conditioning, comparing both with the full index and dispatcher-only baselines.

system    quality    tput    cost $/1k    mix 3B/7B/14B/72B
RB, full index    0.5136    11.85    0.0264    3/28/65/4
RB, 50% labels    0.5144    11.82    0.0270    2/28/66/5
RB, prompt-blind    0.5040    11.85    0.0249    2/34/64/0
Passthrough, RR    0.5043    11.69    0.0322    23/38/23/15
Passthrough, SQ    0.4931    11.74    0.0246    36/27/31/6
Halving labels has negligible effect. Without prompt conditioning, 72B use falls to 0.1% and quality drops to RR's level, showing that prompt-dependent labels identify requests that benefit from the premium tier.

W2-B: Onboarding cost. Gemma onboarding took 6 h to generate 18549 prompts across four models and ~2 h to judge 10065×4=40260 pairs on 2 A30 nodes; index construction took 52.3 s. Original-family latency profiling used 405k requests and 11.1 h across rate sweeps. This motivated static priors for Gemma within the rebuttal window.

W4a: Tuning. Under fixed workload/objective, presets need no retuning solely as rate changes. Workload, topology, or objective changes may require re-sweeping; an SLO controller is future work.

W4b: Budget. We agree the budget benefit comes from the admission filter, not simplex weighting alone. Contribution is the predicted-length admission mechanism.

Add:
About OpenReview
Contact
FAQ
Hosting a Venue
Sponsors
Terms of Use / Privacy Policy
All Venues
Donate
News
OpenReview is a long-term project to advance science through improved peer review with legal nonprofit status. We gratefully acknowledge the support of the OpenReview Sponsors. © 2026 OpenReview

" here is the final submitted version, beside the ac comments```
