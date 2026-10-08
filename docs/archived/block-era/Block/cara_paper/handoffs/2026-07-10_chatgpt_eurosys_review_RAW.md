# RAW external review — ChatGPT (new model), EuroSys-style — received 2026-07-10

> Provenance: pasted by the author on 2026-07-10, reviewing the **arXiv v1** build
> (arXiv:2606.17949, uploaded 2026-06-16, tag `arxiv-2026-06-16` in `route-balance-paper`).
> Verbatim below — do not edit. Our audited response plan: `2026-07-26_eurosys_revision_experiments.md`;
> per-claim verification evidence: `2026-07-10_review_audit/`.

---

EuroSys-style verdict
Recommendation on the current scientific content: Weak Reject / Borderline, confidence 4/5.
RouteBalance addresses an important EuroSys problem, is clearly written, runs on a meaningful 28-GPU testbed, and contains unusually candid ablations. The main obstacle is not implementation quality. It is that the paper currently presents a relatively simple policy insight as a broader systems novelty than the closest 2026 work will allow, while the evaluation does not yet establish generality or isolate the policy advantage as cleanly as reviewers will demand.
The uploaded paper is still the only arXiv version—v1, submitted on June 16, 2026.

Predicted outcome
These are judgmental probabilities rather than statistically calibrated estimates:

| Outcome | Current scientific version | After the priority revisions below |
|---|---|---|
| Direct accept | 15–25% | 35–50% |
| Revise-and-resubmit | 25–35% | 25–35% |
| Reject | 45–60% | 20–35% |
| Eventual acceptance, including successful R&R | 30–45% | 50–65% |

My central estimate for the current version is approximately 20% accept, 30% R&R, 50% reject. That is around or slightly above the recent EuroSys baseline—the EuroSys 2026 fall cycle accepted 59 of 319 submissions, approximately 18.5%—but the paper is vulnerable specifically to an expert LLM-serving reviewer.
The most likely discussion outcome is: "Good problem and strong implementation, but novelty and generality are not yet sufficient," followed by either R&R or rejection.

Simulated reviewer panel

| Dimension | Assessment | Likely reviewer reaction |
|---|---|---|
| EuroSys fit | 5/5 | Excellent fit: scheduling, heterogeneous accelerators, online serving, and resource management |
| Problem significance | 4.5/5 | The separation between model routing and load balancing is real and important |
| Systems contribution | 3.5/5 | Substantial implementation, but reviewers may see the main mechanism as policy composition rather than a new systems architecture |
| Novelty | 2.5/5 | Most exposed dimension because of RouterWise, BOute, OmniRouter, and the simple static-prior ablation |
| Technical soundness | 3/5 | Generally sound, with concerns about steady state, model/GPU confounding, selection methodology, and one theoretical claim |
| Evaluation | 3.5/5 | Broad within one stack, but only one model family and essentially one deployment topology |
| Clarity | 4/5 | Polished and honest, although the abstract and contribution framing are too defensive and crowded |
| Reproducibility | 3.5/5 | Public code and data are a strength; paper–artifact inconsistencies need resolution |
| Overall | 3/5 | Borderline / weak reject |

A plausible three-reviewer split would be:
LLM-serving expert: Weak Reject—closest-work positioning and static-prior result undermine novelty.
Systems generalist: Weak Accept—strong problem, implementation, fairness effort, and large experimental sweep.
Methodology-focused reviewer: Weak Reject—single topology, proxy cost, finite open-loop experiments, and policy selection on the test workload.
That split is exactly the kind that often produces R&R discussion at EuroSys.

What is already strong

1. The problem formulation is compelling
The paper identifies a genuine architectural mismatch:
model routers generally choose a model based on expected quality or cost but ignore instantaneous serving conditions;
load balancers generally choose an instance after the model has already been selected and cannot revisit the quality–cost decision.
The request-to-instance formulation is intuitively stronger than the conventional two-stage decomposition. It is easy to explain and directly relevant to production multi-model serving.

2. The implementation and experimental scale are credible
The evaluation covers four Qwen2.5 tiers, 13 instances, 28 heterogeneous GPUs, 442 configurations, and approximately 1.5 million requests. That is materially better than a simulator-only routing paper.
The engineering effort also appears real rather than merely wrapping a score function around vLLM: batching, embedding, KNN prediction, latency heads, dead-reckoned instance state, scheduling, concurrent baseline execution, and failure handling all matter.

3. The baseline-fairness effort is unusually good
The distinction between original serial baselines and "enhanced" concurrent-scoring variants is valuable. Many papers would report only the dramatic 23× comparison. This paper attempts to separate:
policy differences;
avoidable implementation serialization;
router inference overhead;
serving-side contention.
That is good scientific practice.

4. The ablations reveal a useful systems insight
The most interesting result may not be that the learned predictor wins. It is that:
Putting a latency/load term into model selection matters much more than using a sophisticated learned latency predictor.
The static per-tier latency prior appears to retain nearly all of the headline result, while LPT ordering and within-tier prediction contribute little. That is an honest and potentially important negative result.
The paper should lean into this instead of treating the result mainly as a defense of the full design.

5. The evaluation contains several good robustness checks
The alternative judge, paired bootstrap analysis, seed experiment, burst and tier-loss tests, tail measurements, budget experiment, and out-of-domain predictor study demonstrate reviewer awareness. The paper does not hide its limitations and generally avoids inflated statistical language.

The acceptance blockers

1. The novelty claim is currently too broad
This is the most serious issue.
The statement that no LLM-serving system jointly accounts for quality, latency, and cost is no longer defensible without very careful qualification. Two especially close papers appeared before RouteBalance:
RouterWise jointly studies resource allocation and routing for latency-aware multi-model serving. Its distinction from RouteBalance is that it is principally a deployment/planning optimization and explicitly does not provide online adaptation to workload or load drift.
BOute, published at MLSys 2026, co-optimizes quality-aware request routing and heterogeneous model/GPU deployment under latency considerations.
Neither appears to be adequately addressed in the current manuscript. This will look worse than an ordinary missing citation because these works overlap with the headline formulation.
A defensible novelty position
The paper's real distinction is narrower but still valuable:
RouteBalance is an online, per-request request-to-instance scheduler that injects live queue and decoding state into model selection for a fixed heterogeneous deployment. It complements deployment-time planners by adapting to instantaneous serving conditions.
That creates a clean division:
RouterWise/BOute: determine or optimize a deployment and routing plan from workload assumptions.
RouteBalance: operate the deployed system request by request as queues, active decodes, and arrivals evolve.
OmniRouter: optimize under global budget/capacity constraints rather than use live per-instance state as the main control signal.
Delete or sharply qualify every "first system to jointly consider all three" sentence. Reviewers can accept a narrower contribution; they are much less forgiving of a broad claim contradicted by contemporaneous work.

2. The cleanest policy-controlled baseline is missing
The paper compares against several routers, but it still does not fully answer the decisive causal question:
What does fused request-to-instance selection add over an otherwise identical model router followed by a strong load balancer?
The required baseline should use exactly the same:
embeddings;
KNN quality estimates;
output-length estimates;
cost definition;
weight tuples;
batching and concurrency;
candidate models and instances.
The only difference should be:
choose a model tier using quality/cost and perhaps a tier-level latency prior;
dispatch to that tier's least-loaded or shortest-predicted-delay instance.
Sweep this decoupled baseline across the complete weight space and all principal request rates—not just one representative cell.
At present, a skeptical reviewer can say that RouteBalance wins because its estimator and policy search are stronger, rather than because fusion is intrinsically necessary.
The static-prior ablation should become a primary baseline
Your own ablation suggests that a static per-tier latency prior plus reactive instance dispatch performs almost as well as the learned system. That is not a minor appendix result; it is potentially the strongest competing design.
Either:
show workloads where live prediction yields a clear benefit—bursts, changing output-length distributions, tier failures, latency-SLO admission, or abrupt load transitions; or
simplify the claimed contribution and present the predictor as an optional refinement rather than a core requirement.
Right now reviewers may conclude that the paper builds a substantial prediction stack to rediscover a strong static weighted router.

3. One model family and one deployment are insufficient for the general claims
The evaluation is broad in configuration count but narrow in independent environments.
The current topology substantially confounds:
model size;
model capability;
GPU class;
replica count;
service capacity.
That makes it difficult to determine whether the scheduler handles heterogeneous hardware or simply learns the operating characteristics of four fixed service tiers.
The highest-value additional experiment is not another large sweep on the same cluster. It is an orthogonal environment.
At minimum, add one of:
a second model family or mixed-family deployment;
a materially different replica/GPU allocation for the same models;
the same model served on two or more GPU types;
different models served on the same GPU class.
The strongest compact addition would be two topologies:
capacity-balanced: similar aggregate capacity per model tier;
cost-skewed or bottlenecked: cheap tiers overloaded, expensive tiers underloaded.
Then demonstrate that the same policy structure—not a topology-specific tuning—improves the frontier in both.
A second model family would be especially persuasive because the paper currently lists this as future work. A single modest Llama or Gemma experiment may add more acceptance probability than dozens of extra Qwen configurations.

4. The load experiments need stronger steady-state evidence
At high offered rates, a finite open-loop run can obscure instability. A system can report acceptable finite-run latency while its queue is growing and would diverge under sustained traffic.
This matters particularly where offered throughput and completed throughput differ—for example, results around 30 requests/s with lower completed throughput.
For the main operating points, report:
offered arrival rate;
completed throughput;
queue length over time;
outstanding-request slope;
per-tier utilization;
warm-up, measurement, and drain windows;
whether the system reaches stationary behavior;
repeated-run confidence intervals.
Run a smaller number of decisive cells for longer durations, rather than reproducing all 442 settings. For example, choose approximately:
60%, 80%, 95%, and 105% of measured sustainable capacity;
quality-oriented, balanced, and cost-oriented policies;
three seeds;
two topologies.
If queue length has a positive sustained slope, label the point overloaded rather than describing its finite-run latency as sustainable.

5. Policy selection appears to reuse the held-out test workload
The paper sweeps weights and baseline thresholds on the held-out requests, then highlights peak or frontier cells from those sweeps. This can create selection bias: the reported maximum benefits are selected after observing the same evaluation sample used to measure them.
Use a three-way division:
training: estimator fitting;
validation: KNN hyperparameters, thresholds, normalization choices, and representative weight presets;
test: final, untouched comparison.
There are two legitimate reporting modes:
report the complete Pareto frontier without claiming that a post-selected point is a generalizable preset; or
choose operating points on validation and report only their test-set outcomes.
For confidence intervals on the "best" configuration, either repeat selection within each bootstrap sample or select on validation before bootstrapping the test result. A bootstrap around a post-selected maximum otherwise understates uncertainty.

6. The quality evaluation needs a stronger bridge to actual served outputs
The main quality metric appears to rely heavily on precomputed response-quality values associated with prompt/model pairs. That is efficient, but it weakens the end-to-end claim unless the served output is shown to match the precomputed output.
Add:
exact-match or normalized-hash agreement between served and precomputed outputs;
re-judging of all mismatches or a statistically meaningful random sample;
per-dataset results, not only the aggregate;
task-appropriate metrics where available;
an end-to-end held-out-domain routing experiment, not merely OOD predictor error.
The reported improvement from 0.406 to 0.419 may be statistically significant, but reviewers will ask whether an absolute difference of 0.013 is practically meaningful and consistent across tasks. Show the distribution of per-dataset effects and the fraction of prompts whose selected model changes.
A modest human preference study on the most decision-sensitive examples would further strengthen the claim, though it is lower priority than the baseline and topology experiments.

7. "Cost" should be disambiguated
The paper uses public per-token prices while executing on a self-hosted GPU cluster. That is better described as:
provider-equivalent API price;
billed-token price;
user-facing inference price;
rather than the actual cost of operating this deployment.
Report a second systems-cost metric such as:
GPU-seconds per request;
cloud-equivalent GPU hourly cost;
energy where measurable;
normalized accelerator resource cost.
The two metrics answer different questions. A model can have a high public API price while running efficiently on the available hardware, and vice versa.

8. The objective has sensitivity and interpretability issues
The max-normalized weighted score is candidate-set dependent. Adding or removing a tier can change the normalized values—and therefore the meaning of the same user weights—even when the original candidates are unchanged.
Add sensitivity tests for:
min-max versus max-only normalization;
fixed calibration constants;
z-score or percentile normalization;
addition or removal of an extreme candidate;
transfer of a weight tuple between topologies and request rates.
This matters to the "single knob" usability claim. Three weights summing to one naturally trace trade-offs; that by itself is not a novel control interface unless the knob is stable and portable.

Recommended revision plan

Priority 1: Fix novelty and comparison
Before adding broad evaluation volume:
Add RouterWise and BOute prominently to the introduction and related work.
Replace the "first joint quality-latency-cost system" claim with the online/live-state distinction.
Add the identical-estimator, decoupled routing-plus-load-balancing baseline.
Elevate the static-prior system to a primary comparison.
Explain how RouteBalance can operate on deployments produced by RouterWise or BOute.
A concise contribution statement could be:
RouteBalance is an online scheduler for fixed heterogeneous LLM deployments that selects a concrete model instance using predicted answer utility, price, and live serving state. Its evaluation isolates a broader result: incorporating cross-tier latency into model selection accounts for most of the quality–latency improvement, while sophisticated within-tier prediction provides smaller incremental gains.
This is more defensible and scientifically more interesting than claiming that every component is necessary.

Priority 2: Add two decisive experiment families
Experiment A: Fusion isolation
Compare:
Full RouteBalance: learned cross-tier and instance-level latency state;
RouteBalance-Static: static tier latency/capacity prior plus live shortest-queue dispatch;
Decoupled-SameEstimator: same quality/cost estimator, select model first, then dispatch;
Quality/Cost Only: no latency at model-selection time;
Oracle or trace-replay upper bound, where feasible.
Use identical software paths and sweep the same weights. Report iso-quality latency, iso-cost quality, tail latency, and frontier hypervolume.
Experiment B: Generality and dynamics
Use a second topology or family and introduce:
sudden 2× bursts;
output-length distribution shifts;
tier removal and restoration;
replica-capacity changes;
transitions between low and high load.
Measure recovery time, queue growth, routing shifts, and regret relative to a hindsight oracle. These scenarios are where live state should clearly outperform a static prior.

Priority 3: Repair experimental methodology
Create a validation split for weights and thresholds.
Report long-run stationarity and queue trajectories.
Add actual resource cost.
Verify correspondence between precomputed and served outputs.
Report per-dataset quality and end-to-end OOD routing.
State exactly how many repetitions underlie every main point.

Priority 4: Simplify the paper
The current paper spends too much space defending component complexity that its own ablations show is not load-bearing.
I would:
make the cross-tier latency term the conceptual center;
move the budget experiment to an appendix unless substantially strengthened;
shorten the discussion of LPT;
reduce emphasis on the 23× serial-baseline comparison;
foreground the comparison with enhanced, engineering-equivalent baselines;
present learned prediction as an incremental mechanism whose benefit is demonstrated specifically under dynamic conditions.

Specific technical and presentation corrections

LPT claim
The reference to LPT's classical 4/3-style approximation guarantee is theoretically unsafe in the present form. That result concerns classical makespan scheduling on identical machines under particular assumptions. RouteBalance has heterogeneous instances, state-dependent service, changing queues, and a multi-objective assignment score.
Call LPT a motivated heuristic, or provide a formal reduction under explicit restricted assumptions. Since the measured effect is negligible, the simplest solution may be to de-emphasize it.

Simplex terminology
Three nonnegative weights constrained to sum to one form a 2-simplex, not a 3-simplex. "Three-way weight simplex" is also safe.

Grayscale figures
The radar chart is difficult to distinguish in grayscale and is not an efficient representation of the central result. EuroSys requires readable figures in grayscale. Replace it with:
small multiples;
a direct trade-off curve;
a compact normalized table;
lines with distinct markers and dash patterns.
Figure 2 is more usable, but some light points also need stronger markers or line patterns.

Abstract
The abstract is dense and somewhat defensive. It contains too many numbers and spends too much attention on implementation-enhanced baselines.
A cleaner abstract should state:
the two-stage architecture problem;
online request-to-instance scheduling;
the fixed-deployment/live-state distinction;
one headline quality result and one latency result;
the causal finding that cross-tier latency awareness matters more than sophisticated prediction.

Bibliography and status checks
Clean future or inconsistent publication dates and verify all venue statuses. In particular, the OmniRouter entry appears to use a future December 2026 date in this July 2026 manuscript. Reviewers notice bibliographic anomalies, especially around missing contemporaneous work.

Artifact consistency
The public artifact is a strength, but its reproduction instructions currently appear inconsistent with the paper in several places, including:
paper k=10 versus a reproduction command using k=50;
10,000 bootstrap resamples in the manuscript versus 1,000 in a command;
different counts or ranges for weight and baseline sweeps;
stale section references;
ambiguity over whether headline results are single-seed or multi-seed.
Resolve these and provide a table mapping every paper figure/table to one command, expected runtime, output filename, and expected numerical tolerance.

EuroSys submission compliance
Do not submit the present PDF as-is.
The EuroSys 2027 CFP says that review is double blind. It permits prior arXiv publication, but says an arXiv-exposed submission should use a substantially different paper title and different system/tool name. The current PDF reveals the authors, institution, arXiv identifier, and public repository, and it still contains the "Conference'17" placeholder header and DOI material. The fall-cycle abstract deadline is September 17, 2026, and the full-paper deadline is September 24, 2026. The 12-page technical body is within the stated length limit.
Before submission:
rename both RouteBalance and the paper title;
anonymize the manuscript and repository;
remove identifying URLs and metadata;
ensure the anonymous artifact does not leak commit authors, paths, usernames, CloudLab project names, or documentation history;
replace the placeholder conference metadata;
disclose AI-tool use as required by the current CFP when material generated with such tools is incorporated into the submission.
The rebuttal period is short and does not permit rescuing the paper with new experiments, so the topology, baseline, and steady-state concerns need to be addressed before submission rather than deferred to author response.

Final assessment
This is not far from being a strong EuroSys paper, but the current version is positioned too broadly and evaluated too narrowly at the exact points where expert reviewers will probe.
The three changes with the largest effect on acceptance probability are:
Reframe against RouterWise and BOute as online live-state scheduling complementary to deployment planning.
Add an identical-estimator decoupled baseline and treat the static-prior variant as a first-class competitor.
Demonstrate the result on a second topology or model family with long-running, steady-state evidence.
Without those changes, I would expect a weak reject or R&R. With them—and with the test-selection and artifact inconsistencies repaired—the paper becomes a credible weak-accept-to-accept submission rather than a borderline one.
