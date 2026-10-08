## POST 3b — Response to Reviewer faAU (2/3): generality — topology

**Topology.** We re-provisioned the serving cluster inside this window: the small-model pool was re-balanced across the A30 tier, 3B replicas going from 3 to 5 and 7B from 5 to 3. We then re-ran the comparison with no retraining of any component. Every system runs on that re-provisioned cluster, in the same window, through the same scheduling path, on the submitted configuration. Each cell is 3,534 prompts with zero failures. We ran the full dispatcher sweep — shortest-queue, power-of-two and round-robin — for both the trained per-request router and the dispatcher-only baseline, because your question is precisely whether a result survives a change of dispatcher. Best value per column is in bold. Quality is the paper's DeepEval metric of record; cost prices input+output tokens at the Table-1 rates, where the submission's cost axis prices output only, so these absolute costs are not comparable with the submitted tables.

| λ | system | dispatcher | quality | throughput | cost $/1k | mean e2e (ms) | p99 e2e (ms) | tier mix |
|---|---|---|---|---|---|---|---|---|
| 12 | RouteBalance (0.5,0.3,0.2) | — | **0.5137** | **11.85** | 0.0266 | 2,302 | **5,529** | 14B66/7B27/72B4/3B3 |
| 12 | RouteBalance (1/3,1/3,1/3) | — | 0.5074 | 11.84 | 0.0239 | 2,291 | 5,640 | 14B60/7B35/3B4/72B1 |
| 12 | BEST-Route (argmax) | SQ | 0.5089 | **11.85** | **0.0220** | 2,695 | 8,202 | 14B45/3B40/7B10/72B6 |
| 12 | BEST-Route (argmax) | Po2 | 0.5089 | **11.85** | 0.0221 | 2,624 | 8,068 | 14B45/3B40/7B10/72B6 |
| 12 | BEST-Route (argmax) | RR | 0.5089 | **11.85** | 0.0221 | 2,596 | 8,005 | 14B45/3B40/7B10/72B6 |
| 12 | AvengersPro (pw 0.7) | SQ | 0.5037 | 11.66 | 0.0380 | 2,744 | 13,371 | 3B50/14B28/72B20/7B1 |
| 12 | AvengersPro (pw 0.7) | Po2 | 0.5037 | 11.68 | 0.0379 | 2,722 | 13,377 | 3B50/14B28/72B20/7B1 |
| 12 | AvengersPro (pw 0.7) | RR | 0.5037 | 11.68 | 0.0379 | 2,705 | 13,384 | 3B50/14B28/72B20/7B1 |
| 12 | dispatcher-only | RR | 0.5028 | 11.68 | 0.0302 | 2,347 | 11,235 | 3B38/7B23/14B23/72B15 |
| 12 | dispatcher-only | Po2 | 0.4972 | 11.80 | 0.0251 | 2,021 | 11,157 | 3B48/14B25/7B20/72B8 |
| 12 | dispatcher-only | SQ | 0.4913 | 11.70 | 0.0231 | **1,812** | 10,745 | 3B53/14B29/7B13/72B5 |
| 24 | RouteBalance (0.5,0.3,0.2) | — | **0.5138** | **23.25** | 0.0262 | 2,636 | 6,406 | 14B62/7B29/72B5/3B4 |
| 24 | RouteBalance (1/3,1/3,1/3) | — | 0.5057 | 23.24 | 0.0234 | 2,470 | **5,980** | 14B57/7B38/3B5/72B1 |
| 24 | BEST-Route (argmax) | SQ | 0.5089 | 15.79 | **0.0220** | 43,877 | 126,566 | 14B45/3B40/7B10/72B6 |
| 24 | BEST-Route (argmax) | Po2 | 0.5089 | 15.79 | 0.0220 | 43,289 | 127,754 | 14B45/3B40/7B10/72B6 |
| 24 | BEST-Route (argmax) | RR | 0.5089 | 13.78 | 0.0220 | 61,197 | 139,953 | 14B45/3B40/7B10/72B6 |
| 24 | AvengersPro (pw 0.7) | SQ | 0.5037 | 22.51 | 0.0380 | 2,966 | 13,783 | 3B50/14B28/72B20/7B1 |
| 24 | AvengersPro (pw 0.7) | Po2 | 0.5037 | 22.49 | 0.0380 | 2,914 | 13,813 | 3B50/14B28/72B20/7B1 |
| 24 | AvengersPro (pw 0.7) | RR | 0.5037 | 22.52 | 0.0379 | 2,858 | 13,684 | 3B50/14B28/72B20/7B1 |
| 24 | dispatcher-only | RR | 0.5029 | 22.58 | 0.0302 | 2,392 | 11,254 | 3B38/7B23/14B23/72B15 |
| 24 | dispatcher-only | Po2 | 0.4938 | 22.56 | 0.0240 | 1,999 | 11,160 | 3B50/14B24/7B20/72B7 |
| 24 | dispatcher-only | SQ | 0.4899 | 22.66 | 0.0222 | **1,887** | 11,046 | 3B55/14B25/7B15/72B5 |

**What the re-provisioning does to a queue-only dispatcher.** The new topology has more 3B replicas, so shortest-queue sends more traffic there and wins the mean (1,887 ms). Its 11.0 s p99 is not explained by 72B share alone — round-robin carries three times as much 72B traffic for a 2% higher p99 — but every queue-only variant lands near 11 s while ours stays under 6.5 s: a queue length says which replica is idle now, not what the request will cost there. RouteBalance's score prices the request on the instance before placing it, and its p99 is 5,980–6,406 ms. Taking one configuration at a time: the uniform (1/3,1/3,1/3) preset delivers 0.5057 quality at 23.24 req/s with a 5,980 ms p99 and 0.0234 per 1k — higher throughput than every dispatcher with roughly half their tail, cheaper than Po2 (0.0240) and round-robin (0.0302), and ahead of every dispatcher on quality, though its +0.003 over round-robin is inside the paired-bootstrap width we report for the Gemma family. The (0.5,0.3,0.2) preset trades that cost advantage away: 0.5138 quality at 23.25 req/s, but 0.0262 per 1k, dearer than shortest-queue and Po2 though still cheaper than round-robin.

**What we do not claim.** RouteBalance is not best on every axis here and we will not present it as such. Shortest-queue is cheaper (0.0222) and faster on the mean (1,887 vs 2,470 ms); the trained router is the cheapest per 1k requests, though it cannot sustain the offered rate. On this topology RouteBalance buys quality, throughput and tail, and does not win the mean-latency or cheapest-cost axis.

Both presets were selected on the original cluster and carried over unchanged, so the mean-latency column is not a tuned comparison; a full weight re-sweep did not fit this window. This experiment shows re-provisioning preserves quality, throughput, tail and weight control — not that these presets remain optimal.

**The trained router's collapse does not depend on the dispatcher.** Under every dispatcher it pins at 13.8–15.8 req/s against an offered 24, at 43.3–61.2 s end-to-end — 17.5–24.8× RouteBalance. The ceiling is routing-side, not dispatcher-side, and the power-of-two-choices dispatcher you suggested is what let us separate the two.
