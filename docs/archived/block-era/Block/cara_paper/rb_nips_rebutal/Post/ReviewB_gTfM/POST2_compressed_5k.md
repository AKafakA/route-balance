## POST 2 — Response to Reviewer gTfM (estimator justification · deployment model)

We thank the reviewer for these questions and for noting "this is the first work to demonstrate this capability".

**Q1.** We built exactly the alternative you propose: the same objective and the same within-tier queue tiebreak, with the learned head's queue-conditioned estimate replaced by a fixed per-tier TPOT constant. The constants are per-tier medians across the whole sweep rather than at one operating point — 3B 10.7, 7B 19.9, 14B 16.5, 72B 42.6 ms/token.

These cells and the submitted tables share the same benchmarking and evaluation setting, and both arms use RouteBalance's uniform (1/3,1/3,1/3) weights.

| λ | quality learned / static | cost $/1k learned / static | cheaper | throughput | mean e2e ms |
|---|---|---|---|---|---|
| 6 | 0.5074 / 0.5096 | 0.02442 / 0.02656 | **8.1%** | 6.0 / 6.0 | 2,080 / 2,079 |
| 8 | 0.5076 / 0.5074 | 0.02405 / 0.02630 | **8.6%** | 7.9 / 7.9 | 2,118 / 2,137 |
| 10 | 0.5073 / 0.5084 | 0.02376 / 0.02605 | **8.8%** | 9.9 / 9.9 | 2,175 / 2,198 |
| 12 | 0.5073 / 0.5095 | 0.02372 / 0.02604 | **8.9%** | 11.8 / 11.8 | 2,243 / 2,224 |
| 18 | 0.5060 / 0.5079 | 0.02345 / 0.02583 | **9.2%** | 17.6 / 17.6 | 2,298 / 2,363 |
| 24 | 0.5043 / 0.5080 | 0.02327 / 0.02579 | **9.8%** | 23.3 / 23.3 | 2,458 / 2,487 |
| 28 | 0.5061 / 0.5071 | 0.02328 / 0.02568 | **9.3%** | 26.9 / 27.0 | 3,143 / 3,079 |

So, **cost is what the learned head buys: 8.1–9.8% per 1,000 requests at every rate**, at quality that is statistically indistinguishable — every |Δq| ≤ 0.004, all seven CIs span zero, widest [−0.0063, +0.0014] — throughput within 0.4%, and mean e2e within 2.7% in either direction. The mechanism is the routing mix: at λ=24 the head puts 56% on 14B and 38% on 7B, against 68% / 29% for the constant. A constant is fixed for the whole run and cannot tell a loaded instance from an idle one; 14B decode alone drifts 15.4 → 18.6 ms/token across the sweep. Learned latency prediction is established in LLM simulation stacks: Vidur fits random forests to per-kernel durations and composes them over layer executions, where ours is a single head fitted directly to TPOT. We will cite this in the design section.

But we also want to point out that **the latency term itself is still necessary**. To demonstrate it we designed a further experiment that removes latency and scores on quality and cost alone — unlike the two arms above, which both keep a latency term and differ only in how it is computed. Its score no longer depends on serving state, so routing goes load-blind: the identical assignment at every rate, 69.1% of requests to the smallest tier, and quality falls to 0.4899 against 0.504–0.510 above. So your Q1 is not whether latency belongs in the objective, but whether it has to be learned.

**Latency prediction clarification.** The head predicts **TPOT only**, one XGBoost head per (model, GPU) tier, trained from a tier-local QPS sweep. Features are that tier's serving-state signals plus the request's predicted output length; it does not consume request text. End-to-end latency is then formed **analytically**, not learned: predicted TPOT × the decode iterations the request waits through (pending decode tokens ÷ current decode batch size, plus its own predicted output length, from the KNN estimator).

**Q2.** TTFT is not an explicit term, as our target is decode-dominated generation. Prefix affinity is likewise not modeled: our corpus is single-turn, so cross-request reuse is minimal and the setting is uniform across compared systems. KV occupancy is not a term either, but KV utilization and free-block count are features the TPOT head consumes, reaching the assignment through predicted TPOT. We will list explicit prefill/TTFT and prefix-affinity terms as future work to improve the generality of the latency model.

**Q3.** Yes, centralized by design: one assignment decision per request over a global dead-reckoned view, which is what makes the joint comparison across heterogeneous instances possible. Its cost is measured: decision compute per request *falls* 44.7 → 25.7 ms from λ=6 to λ=24, as one prediction round amortizes over a batch. Scheduler fault tolerance and sharding can also be expanded as future work, e.g. a hierarchical multi-layer balancer with RouteBalance running per mini-cluster.

**Q4.** Yes, fixed per run — RouteBalance is an online assignment layer over a given deployment, not an autoscaler. It does respond to a changed deployment without retraining: in the topology experiment in our faAU reply we re-provisioned the small-model pool (3B replicas 3→5, 7B 5→3) and re-ran every system with nothing retrained. Joint optimization of placement and assignment is future work.

In summary, we thank the reviewer for the static-TPOT suggestion, which we also applied to the new model-family experiments in our reply to Reviewer faAU, and for pointing to useful future directions.
