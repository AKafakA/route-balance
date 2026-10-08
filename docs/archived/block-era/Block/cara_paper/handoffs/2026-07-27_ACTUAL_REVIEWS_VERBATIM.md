# Submission28271 — ACTUAL REVIEWS (verbatim from OpenReview, pasted by author 2026-07-27)
# AUTHORITATIVE reviewer→topic mapping. Supersedes all paraphrases for drafting purposes.
# Ratings: 287j = 3 (borderline reject, conf 4) · gTfM = 4 (borderline accept, conf 4) ·
# faAU = 4 (borderline accept, conf 4, use-inspired). Meta-review by AC axCN (Jul-25).

## Meta Review (Area Chair axCN, 25 Jul 2026)
"This paper proposes RouteBalance, a serving-layer scheduler that jointly optimizes model
routing and instance-level load balancing by considering quality, latency, and cost in a
unified assignment framework. The reviewers agree that the paper addresses an important
problem in heterogeneous LLM serving and that the proposed joint optimization framework is
well motivated. However, the rationale of design choice and and evaluation part of the paper
should be improved. Specifically, many design choices such as XGBoost estimator need to be
better justified, and comparisons with general-purpose dispatchers can be added to
illustrate the effectiveness of the proposed scheduler."

## Reviewer 287j — Rating 3 (borderline reject), Quality 3 / Clarity 2 / Significance 4 / Originality 3
Summary: "RouteBalance is a system that aims to balance LLM model choice, request routing,
and load-balancing. To my understanding, it is similar in philosophy to LLM-D or an AIBrix.
The system combines some interesting ideas to enable better tradeoffs management."
Strengths: "The paper aims to solve the right problem and probably does a good job in
solving it. The problem of routing LLM requests is one of the main points in LLM serving
frameworks."
Weaknesses (verbatim):
1. "The paper is hard to follow in some places and some design decisions seem to be either
   arbitrary or not well explained. Some parts are also not quite clear to me. For example,
   you schedule in batches. I suppose that means that you do not use continuous-batching?
   If not, then what happens when you have an uneven batch? If you actually have
   Continuous-batching, how does this work with 'RouteBalance schedules in batches. Each
   batch is the set of requests waiting at the scheduler when it fires, and the scheduler
   assigns every request to an instance before the next batch starts'. Does that mean that
   you intentionally delay a request that arrived, still use continuous batching, but
   schedule only when you reach the required batch size/time?"
2. "Another example is the idea of adaptive size limit for batches. However, besides the
   vague sentence of 'the limit grows when more instances are busy and shrinks when the
   cluster is mostly idle, so light load produces small batches and saturated load produces
   larger one', you never discuss how this mechanism works?"
3. "There are also several unclear points, e.g., how can the model choice affect the
   KV-Cache hit-rate? How much does it affects the TTFT (even though the filtration part is
   left for future work!)?"
4. "One suggestion is to walk through the components in Figure1 and discuss them in details
   in the appendix. Also, What about the datasets used, again something that should be added
   to the appendix."
5. "For the evaluation, Figure 3 takes large space but in reality, does not tell much."
6. writing suggestions: "Maybe move Table 2 to be close to equation1 and 2." · "Figure 1 can
   improve... 'estimate' does not return anything anywhere. Also, there are boxes that you
   never discuss in the paper, e.g., RoBERTa is there but never mentioned." · "I think the
   appendix can include more information on, e.g., LPT, the methods you compare against,
   dead-reckoning which is AFAIK is not a standard term neither in LLM serving nor in
   distributed systems."
Questions: "See the weaknesses."

## Reviewer gTfM — Rating 4 (borderline accept), Quality 3 / Clarity 2 / Significance 2 / Originality 4
Summary: system description; "compares against router-only baselines such as BEST-Route and
AvengersPro."
Strengths/Weaknesses (verbatim): "Quality & Clarity: The paper is mostly clearly written,
but some design choices are not well described or remain unclear to the reader. For example,
why does TPOT require an XGBoost estimator? Can it instead be benchmarked statically? What
are its advantages over simpler alternatives, and what exactly is its input?"
"Significance & Originality: The paper models a router that considers the
⟨cost, latency, quality⟩ of a pool of LLM instances. However, in real-world settings, the
cluster (i.e., the model-to-instance assignment) is not fixed as assumed in the paper. In
addition, the scheduler appears to be centralized, which limits its practical applicability.
That said, this is the first work to demonstrate this capability."
Questions (verbatim):
Q1: "XGBoost: How much of RouteBalance's gain comes from the XGBoost TPOT predictor versus
a simpler calibrated benchmark/lookup-table latency model using per-tier average TPOT and
live queue state?"
Q2: "TTFT/KV-cache: Does the latency predictor account for prefill/TTFT effects and
prompt-length-dependent KV allocation, or is RouteBalance primarily optimizing decode TPOT?
Also, how would prefix-cache hits or cache locality affect the instance-routing decision?"
Q3: "Scheduler: Is the proposed scheduler centralized?"
Q4: "Cluster: Is the cluster fixed (i.e., is the model-to-instance assignment static)?"

## Reviewer faAU — Rating 4 (borderline accept), Quality 3 / Clarity 3 / Significance 2 / Originality 3, Use-inspired
Strengths (verbatim, three paragraphs):
- "Sharp observation and elegant formulation. ... RouteBalance reformulates scheduling as
  instance-level online assignment under a 3-simplex weight space that captures the full
  trade-off space—insightful and elegant."
- "Strong empirical performance. On a heterogeneous cluster, RouteBalance forms the upper
  convex hull of the quality–throughput Pareto frontier, achieving lower latency and cost at
  iso-quality. Most notably, its scheduling overhead stays sub-linear as load grows 5×
  (141→220 ms), whereas BEST-Route inflates to ~33 s under the same
  architecture—demonstrating the effectiveness of batched amortization."
- "Pragmatic engineering design. Batch-level scheduling amortizes quality prediction across
  requests, an in-process XGBoost predictor provides lightweight per-instance latency
  estimates, and dead-reckoning state caching eliminates synchronous queries while avoiding
  herding. Pipeline mode hosts all decoupled baselines, ensuring comparisons isolate routing
  and dispatch policies—precisely targeted, no extraneous complexity."
Weaknesses (verbatim):
W1: "Single-cluster topology limits generality. All results come from one 13-instance
CloudLab deployment using only the Qwen2.5 model family. While the heterogeneous GPU mix
(A30/V100/A100) is representative, generalization to model families beyond Qwen or
substantially different cluster topologies remains unverified."
W2: "Cold-start dependency of the quality estimator. The KNN-based quality estimator
requires labeled training data per model; when deploying to a new model pool, the hottest
routing paths lose signal until sufficient labeled data is collected. The paper acknowledges
this limitation but does not quantify routing degradation under partial or missing quality
signals, nor does it report the training-time cost of collecting the labeled dataset."
W3: "Dispatcher baselines can be strengthened. While the model routers (AvengersPro,
BEST-Route) are appropriate competitors on the quality/cost dimension, the dispatcher
baselines (round-robin, shortest-queue, random) are all basic strategies. Augmenting the
comparison with more advanced general-purpose dispatchers such as least-connections or
power-of-two-choices would more convincingly isolate the unique gain attributable to joint
optimization, thereby making the core contribution more defensible."
W4: "Added tuning burden, and the budget gain does not come from the core mechanism. Fusing
routing and load balancing exposes three simplex weights (wq, wl, wc) that operators must
continually re-tune as the workload drifts, adding complexity over a quality-only router
(acknowledged in §7). More tellingly, §5.4 shows the budget-control benefit comes mainly
from the admission filter: without the filter, RouteBalance and BEST-Route are within 1.5 pp
on budget-exhaustion at every mix, and the paper itself states the budget contribution
'lives in the filter, not in the simplex weighting'—weakening the unique value of the
'joint optimization' framing on the budget axis."
Limitations (verbatim): "The cold-start dependency of the quality estimator is the main
limitation. We encourage the authors to add: generalization evidence across model
families/topologies, the labeling scale and training time required for the quality
estimator to become usable when a new model is onboarded, and the routing-degradation curve
under missing quality signals."
