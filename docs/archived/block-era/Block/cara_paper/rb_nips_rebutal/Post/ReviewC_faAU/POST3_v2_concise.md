We sincerely thank the reviewer for the exceptionally thorough review. We take W3 first, as it is also the meta-review's second key, then W2, W1 and W4. 

**W3: General-purpose dispatchers.** We added standard power-of-two choices (Po2) and compared with other dispatchers and RouteBalance(RB).

| system | dispatcher | λ12 | λ24 mean e2e ms |
|---|---|---:|---:|
| passthrough | SQ | 2051 | 2155 |
| passthrough | Po2 | 2257 | 2272 |
| BEST-Route | Po2 | 2636 | 48182 |
| AvengersPro (pw 0.7) | Po2 | 2665 | 2818 |
| RouteBalance (uniform) | joint | 2230 | 2408 |

SQ outperforms Po2 for passthrough. BEST-Route reproduces its 18.73 req/s serial ceiling under Po2, as under SQ/RR: at λ24 it reaches 48182 ms versus RB's 2408 ms (20.0×). Thus the collapse is upstream of dispatch: a dispatcher cannot revisit the model tier already selected or reduce the overhead the routing layer has already added.

**W1-A New Topology.** We re-provisioned the A30 pool from 3B/7B counts 3/5 to 5/3 to change the topology of our deployment and reran experiments without retraining.

| system | disp. | quality | tput | cost $/1k | mean | p99 ms |
|---|---|---:|---:|---:|---:|---:|
| RB (0.5,0.3,0.2) | — | 0.5138 | 23.25 | 0.0262 | 2636 | 6406 |
| RB (1/3,1/3,1/3) | — | 0.5057 | 23.24 | 0.0234 | 2470 | 5980 |
| BEST-Route | Po2 | 0.5089 | 15.79 | 0.0220 | 43289 | 127754 |
| AvengersPro | Po2 | 0.5037 | 22.49 | 0.0380 | 2914 | 13813 |
| dispatcher-only | SQ | 0.4899 | 22.66 | 0.0222 | 1887 | 11046 |

SQ leads mean latency and cost, while RB leads quality, throughput and tail latency (5980 vs 11046 ms p99). SQ reaches the lower mean by sending 55% of traffic to the 3B tier that re-provisioning enlarged, but all three queue-only variants land near 11 s at p99. 

**W1-B: New Model.** 
We also onboarded Gemma-3 (1B/4B/12B/27B), using measured static per-(model,GPU) latency priors rather than trained heads, an alternative that avoids latency sweeping as discussed with reviewer gTfM. It uses 1979 held-out prompts, disjoint from the 8086 used for training. The family's price ladder is also non-monotone: per 1M input tokens, 1B $0.10 against 4B $0.04 and 27B $0.08.

| system | disp. | quality | tput | cost $/1k | mean | p99 ms |
|---|---|---:|---:|---:|---:|---:|
| AvengersPro | Po2 | 0.6300 | 22.15 | 0.0259 | 3719 | 9582 |
| RB (0.8,0.1,0.1) | — | 0.6160 | 22.24 | 0.0220 | 2673 | 7952 |
| RB (0.5,0.3,0.2) | — | 0.6057 | 22.25 | 0.0208 | 2355 | 7758 |
| RB (1/3,1/3,1/3) | — | 0.6010 | 22.39 | 0.0194 | 2108 | 7393 |
| dispatcher-only | SQ | 0.5732 | 22.44 | 0.0227 | 1622 | 6714 |

RB is cheaper than every dispatcher-only variant. Under the submitted quality procedure it exceeds SQ/Po2 for all presets; against RR, two presets separate and uniform is a tie. SQ wins latency, serving 1B for 47% of requests. AvengersPro has the highest quality; versus RB (0.8,0.1,0.1), its +0.014 costs 18% more, 39% more mean latency and a 21% wider tail. The weighting selects the operating point. BEST-Route is absent because extending it to a new family requires fine-tuning a fresh per-family router.

**W2-A: Partial/missing labels**  We rebuilt KNN index at half the labels and with prompt conditioning removed, against the dispatcher-only baselines at λ12. RB runs the (0.5,0.3,0.2) preset.

| system | quality | tput | cost $/1k | mix 3B/7B/14B/72B |
|---|---:|---:|---:|---:|
| RB, full index | 0.5136 | 11.85 | 0.0264 | 3/28/65/4 |
| RB, 50% of labels | 0.5144 | 11.82 | 0.0270 | 2/28/66/5 |
| RB, prompt-blind | 0.5040 | 11.85 | 0.0249 | 2/34/64/0 |
| dispatcher-only, RR | 0.5043 | 11.69 | 0.0322 | 23/38/23/15 |
| dispatcher-only, SQ | 0.4931 | 11.74 | 0.0246 | 36/27/31/6 |

Halving the label corpus changes nothing. Without prompt conditioning, 72B use collapses to 0.1% and quality falls to round-robin's level. The prompt-conditioned estimate identifies the requests that need the premium tier.

**W2-B: Onboarding Training Cost**  We use the Gemma onboarding process as the example. It took ~6 h to generate 18549 prompts across four gemma models, and ~2 h to judge 10065×4=40260 pairs on 2 of the A30 nodes in our testbed, while the KNN index took only 52.3 s to build. By contrast, latency profiling for the original family took 404929 requests and 11.1 h of sweeping to train the XGBoost heads, which is why we used static TPOT priors here

**W4a: Tuning.** Under fixed workload/objective, presets need no retuning solely as rate changes. Workload, topology, or objective changes may require re-sweeping; an SLO controller is future work.

**W4b: Budget.** We agree the budget benefit comes from the admission filter, not simplex weighting alone. The contribution is the predicted-length admission mechanism; the runtime exhaustion-rate result is evaluator-independent.

We thank the reviewer for these suggestions, which have made our evaluation more rigorous, and will adopt them in the revision.
