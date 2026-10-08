## POST 2 — Response to Reviewer gTfM (estimator justification · deployment model)

We sincerely thank the reviewer for the substantive questions and for the recognition that the paper is "mostly clearly written" and that "this is the first work to demonstrate this capability".

**Q1: How much of the gain comes from the XGBoost predictor vs a calibrated static lookup + live queue state?**

We built exactly the alternative you propose: the full objective with the learned head's queue-conditioned latency estimate replaced by a fixed per-tier TPOT constant, retaining the same objective and the same within-tier queue tiebreak. The constants are the per-tier medians we measure across the whole sweep, so the comparator is calibrated to this deployment rather than to any single operating point:

| tier / GPU (ms/token) | λ6 | λ8 | λ10 | λ12 | λ18 | λ24 | λ28 | constant used |
|---|---|---|---|---|---|---|---|---|
| 3B / A30 | 10.9 | 10.7 | 10.9 | 10.9 | 10.6 | 10.3 | 10.2 | 10.7 |
| 7B / A30 | 19.7 | 19.8 | 19.8 | 19.9 | 20.1 | 20.5 | 20.8 | 19.9 |
| 14B / V100 | 15.4 | 15.8 | 16.1 | 16.5 | 17.4 | 18.1 | 18.6 | 16.5 |
| 72B / A100 | 40.4 | 43.7 | 42.3 | 43.8 | 43.0 | 40.4 | 42.6 | 42.6 |

Each cell serves 3,534 prompts with zero failures; quality is the paper's DeepEval metric of record, scored through the submission's own retrieval path, so these cells and the submitted tables sit on one basis; cost is input+output tokens at the Table-1 per-model rates, where the submission's cost axis prices output only, so absolute cost figures are not comparable with it; quality CIs are paired per-prompt bootstraps (2,000 resamples) over the prompts scored in both arms; both arms run the uniform (1/3, 1/3, 1/3) weighting.

| λ | quality learned / static | Δ quality 95% CI | cost $/1k learned / static | learned cheaper | throughput |
|---|---|---|---|---|---|
| 6 | 0.5074 / 0.5096 | [−0.0050, +0.0014] | 0.02442 / 0.02656 | **8.1%** | 6.0 / 6.0 |
| 8 | 0.5076 / 0.5074 | [−0.0034, +0.0031] | 0.02405 / 0.02630 | **8.6%** | 7.9 / 7.9 |
| 10 | 0.5073 / 0.5084 | [−0.0042, +0.0026] | 0.02376 / 0.02605 | **8.8%** | 9.9 / 9.9 |
| 12 | 0.5073 / 0.5095 | [−0.0054, +0.0014] | 0.02372 / 0.02604 | **8.9%** | 11.8 / 11.8 |
| 18 | 0.5060 / 0.5079 | [−0.0060, +0.0009] | 0.02345 / 0.02583 | **9.2%** | 17.6 / 17.6 |
| 24 | 0.5043 / 0.5080 | [−0.0063, +0.0014] | 0.02327 / 0.02579 | **9.8%** | 23.3 / 23.3 |
| 28 | 0.5061 / 0.5071 | [−0.0036, +0.0009] | 0.02328 / 0.02568 | **9.3%** | 26.9 / 27.0 |

**Cost is what the learned head buys: 8.1–9.8% per 1,000 requests at every rate**, at delivered quality that is statistically indistinguishable and throughput that matches within 0.4%. The mechanism is in the routing mix: at λ=24 the learned head places 56% of traffic on 14B and 38% on 7B, against 68% / 29% for the calibrated constant — twelve points off the 14B tier, nine of them onto 7B. A constant is fixed for the whole run, so it cannot tell a loaded instance from an idle one; the learned head conditions each estimate on that instance's current serving state, and the per-tier column above shows how much there is to condition on — 14B decode drifts 15.4 to 18.6 ms/token across the sweep. That is what lets it move traffic to a cheaper tier without giving up quality.

**Quality is statistically unchanged.** Every |Δq| is at most 0.004 and every interval spans zero.

**Throughput and latency both match.** Throughput is within 0.4% at every rate. For latency we report only pairs measured in a single session, because either arm is sensitive to when it ran; we ran three such pairs, three seeds each:

| λ | learned (ms) | calibrated static (ms) | difference |
|---|---|---|---|
| 12 | 2,243 | 2,224 | learned 0.8% slower |
| 24 | 2,458 | 2,487 | learned 1.1% faster |
| 28 | 3,143 | 3,079 | learned 2.1% slower |

The gap is within 2.1% at every rate and changes sign, so we read it as no difference. (Because these come from one later session, the learned λ=24 figure differs by about 2% from the corresponding entry in the sweep table in our reply to Reviewer faAU — that sensitivity is exactly why we pair the arms rather than compare across sessions.)

**The latency term itself is necessary.** A third arm that drops the latency term altogether, rather than replacing it with a constant, collapses the routing mix to 69.1% smallest-tier and delivers 0.4899 quality — well below either arm above. So the question your Q1 raises is not whether latency belongs in the objective, but how it should be computed.

**What we claim, and what we do not.** The latency term must be present. Given that it is present, computing it from serving state rather than from a per-tier constant is worth 8.1–9.8% on cost at equal quality, equal latency and equal throughput. We do not claim a latency or quality advantage for the learned head over a calibrated constant, and we will scope the manuscript accordingly.

**Predictor input (your "what exactly is its input").** The learned head predicts **TPOT only** — per-output-token decode latency — with one XGBoost head per (model, GPU) tier, trained from a tier-local QPS sweep on that tier's head node. Its features are that tier's serving-state signals together with the request's predicted output length; it does not consume the request text. End-to-end latency is then formed **analytically**, not learned: predicted TPOT × the decode iterations the request waits through (the instance's pending decode tokens ÷ its current decode batch size, plus the request's own predicted output length, which comes from the KNN estimator rather than from XGBoost). If the instance has a free decode slot, only the second term applies.

**Q2: TTFT / KV-cache / prefix-cache scope**

TTFT is not an explicit term in the objective: the assignment score is decode-dominated (predicted TPOT × queue-aware decode iterations), and the manuscript's phrasing that suggests otherwise will be corrected. Prefix-cache affinity is likewise not modeled — our corpus is assembled from single-turn (prompt, response) pairs, so cross-request prefix reuse is minimal and the setting is uniform across every compared system. KV-cache occupancy is not a term in the objective, though KV utilization and free-block count are among the instance-state features the TPOT head consumes, so they reach the assignment through the predicted TPOT rather than through a term of their own. Adding an explicit prefill/TTFT term and a prefix-affinity term is future work, and we will state that scope explicitly rather than leaving it implied.

**Q3: Is the scheduler centralized?**

Yes, by design: a single assignment decision per request, taken over a global dead-reckoned view of instance state, which is what makes the joint quality–latency–cost comparison across heterogeneous instances possible at all. The cost of that centralization is what the paper measures: decision compute per request *falls* from 44.7 ms at λ=6 to 25.7 ms at λ=24 because one gathered prediction round is amortized over a whole scheduling batch, and scheduler-side residual grows 1.6× (141 → 220 ms) while offered load grows 4×. Fault tolerance and sharding of the scheduler itself are out of scope for this paper and we do not claim them.

**Q4: Is the deployment fixed?**

Yes — the model pool and its placement are fixed for the duration of each run; RouteBalance is an online request-assignment layer over a given deployment, not an autoscaler. It does respond to a changed deployment without retraining: in the topology experiment reported to Reviewer faAU we re-provisioned the small-model pool (3B replicas 3→5, 7B 5→3) and re-ran every system with nothing retrained. Joint optimization of placement and assignment is future work.
