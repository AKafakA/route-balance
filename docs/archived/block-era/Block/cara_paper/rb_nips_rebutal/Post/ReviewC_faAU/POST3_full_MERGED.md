## POST 3 — Response to Reviewer faAU (dispatchers · generality · cold-start · tuning/budget)

We sincerely thank the reviewer for the exceptionally thorough review. We take W3 first, as it is also the meta-review's second key, then W2, W1 and W4. W2–W4 are measured on the submitted configuration; W1 necessarily is not, and we state both scoping notes in our general response.

**W3: General-purpose dispatchers (answered first — it is the meta-review's second key)**

Following your suggestion, we added standard power-of-two-choices: two candidate replicas sampled uniformly without replacement, dispatching to the one with fewer outstanding requests — the same load signal as shortest-queue, so telemetry staleness is identical; with fewer than two replicas it degenerates to full comparison. (The submitted shortest-queue baseline dispatches to the instance with the fewest outstanding requests, running + waiting — the closest reactive LLM-serving analogue of the least-connections policy you name, though it does not weight heterogeneous remaining decode work.) All rows below were re-measured in a single evaluation window on the submitted configuration:

| system | dispatcher | λ6 | λ8 | λ10 | λ12 | λ18 | λ24 (mean e2e ms) |
|---|---|---|---|---|---|---|---|
| passthrough | SQ | 1947 | 1966 | 2017 | 2051 | 2105 | 2155 |
| passthrough | Po2 | 2254 | 2238 | 2261 | 2257 | 2258 | 2272 |
| BEST-Route (argmax) | Po2 | 1911 | 1991 | 2175 | 2636 | 18273 | 48182 |
| AvengersPro (pw 0.7) | Po2 | 2553 | 2592 | 2624 | 2665 | 2737 | 2818 |
| RouteBalance (uniform) | joint assignment | 2080 | 2118 | 2175 | 2230 | 2298 | 2408 |

Shortest-queue beats Po2 at every rate for the passthrough family. The trained per-request router (BEST-Route, argmax) reproduces the paper's documented serial ceiling (18.73 req/s sustained maximum on this topology) — the collapse knee between λ12 and λ18 (2,636 → 18,273 ms) — under Po2 just as it does under shortest-queue, and the topology sweep below shows the same collapse under all three dispatchers: the ceiling is a routing-side property, independent of the dispatcher. RouteBalance remains stable across the whole sweep, 20.0× lower end-to-end latency than BEST-Route+Po2 at λ=24.

Two scoping notes on this table. The AvengersPro rows are run at pw = 0.7, an operating point between the values swept in the submission, so they are a new measurement rather than a reproduction. And a dispatcher — however strong — places requests within the replica pool of a tier that has already been selected; it cannot revisit the model choice itself, which is the structural point the comparison is making.

**W2 + your Limitations request: robustness under partial/missing quality signals, and onboarding cost**

We quantify exactly the degradation you name, on the deployed estimator: (a) a labeled-fraction ladder — the estimator rebuilt at 0/25/50/75% of quality labels and evaluated end-to-end online; and (b) a missing-domain ablation — entire domains (math [GSM8K] and code) held out of the estimator's corpus. Degradation is reported as deltas against the 75%-coverage estimator built and evaluated in the same window, plus the quality slice on prompts from the excluded domain:

| estimator corpus | Δ global quality vs 75%-coverage reference | Δ on excluded-domain prompts |
|---|---|---|
| 50% of labels | −0.002 | — |
| 25% of labels | +0.003 | — |
| 0% of labels (prompt-blind) | −0.013 | — |
| GSM8K domain held out | −0.002 (tie) | +0.002 |
| code domain held out | −0.005 | +0.001 |

The ladder is non-monotone — 25% of the labels scores marginally above 50% — which is itself the finding: across 25–75% coverage the differences are within the run-to-run spread, so the honest reading is that the estimator is insensitive to label coverage over that range rather than that it degrades smoothly through it. Only the prompt-blind arm separates, and it costs 0.013. Holding an entire domain out moves global quality by at most 0.005, while quality on the excluded domain's own prompts is statistically unchanged (paired per-prompt CIs on those slices span zero). End-to-end latency is flat across the entire ladder (all cells within ±2% of the reference), so the quality deltas above are the whole story.

**Onboarding cost, measured.** We onboarded a second model family end to end inside this window, so we can report the phases as durations rather than estimates, for the whole corpus rather than the evaluation slice. Generation broadcast 18,549 candidate prompts across the four-model ladder on 2 A30 nodes in parallel and took approximately 6 hours. After filtering, judging the surviving corpus — 10,065 prompts × 4 models = 40,260 (prompt, model) pairs — took approximately 2 hours on the same two nodes. That corpus splits 8,086 for the estimator and 1,979 held out for the cells below. Building the similarity index takes 52.3 s (measured), and per-tier estimator training takes under a minute each. The one step that does not fit a discussion window is the latency-profiling corpus: for the original family that was 292,239 profiled requests across the four deployed (model, GPU) pairings, which the corpus timestamps put at 11.1 hours of wall-clock with all pairings profiled concurrently.

That is why the isolation experiment reported to Reviewer gTfM matters for your question too: it licenses a practical two-stage onboarding path. A day-1 deployment needs only static priors — per-tier decode medians, which cost a short profiling pass rather than a judged corpus and a trained head — which that isolation shows match the learned estimator on quality and throughput across the whole stationary range, and on latency at the three rates where we measured both arms in one session, at a cost penalty of 8.8–10.8%; the learned heads then train from accumulated serving telemetry as the system runs. We took exactly that path for the second model family below, and we flag it so the two families are compared on a stated basis rather than an assumed one.


**W1: Generality — (1) topology.** We re-provisioned the serving cluster inside this window: the small-model pool was re-balanced across the A30 tier, 3B replicas going from 3 to 5 and 7B from 5 to 3. We then re-ran the comparison with no retraining of any component. Every system runs on that re-provisioned cluster, in the same window, through the same scheduling path, on the submitted configuration. Each cell is 3,534 prompts with zero failures. We ran the full dispatcher sweep — shortest-queue, power-of-two and round-robin — for both the trained per-request router and the dispatcher-only baseline, because your question is precisely whether a result survives a change of dispatcher. Best value per column is in bold. Quality is the paper's DeepEval metric of record.

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


**W1: Generality — (2) a second model family.** We onboarded a second, architecturally distinct family end to end inside this discussion window — the Gemma-3 instruction-tuned ladder (1B/4B/12B/27B) — and re-ran the comparison with no change to the scheduler. Each cell below is 1,979 held-out prompts at λ=24 with zero failures, with quality measured by the same DeepEval judge and the same lookup procedure as the submitted evaluation; the family's quality estimator is built on a disjoint 8,086-prompt training split. Best value per column is in bold.

One scoping note belongs beside this table rather than only in our general response: this family was served on measured static per-(model, GPU) latency priors rather than newly trained latency heads, so what is tested here is the formulation on a new family, not the learned estimator on a new family.

This family is a genuine generality test rather than a re-run, because its published price ladder is **non-monotone**. Prices per 1M tokens, input / output: 1B $0.10 / $0.10, 4B $0.04 / $0.08, 12B $0.05 / $0.15, 27B $0.08 / $0.16. On input the smallest model is the *dearest of the four* — above even the 27B — because few providers serve it at scale. On Qwen price tracks size, so a scheduler that simply prefers smaller models gets cost right by accident; here it cannot.

| system | dispatcher | quality | throughput | cost $/1k | mean e2e (ms) | p99 e2e (ms) | tier mix |
|---|---|---|---|---|---|---|---|
| AvengersPro | Po2 | **0.6300** | 22.15 | 0.0259 | 3,719 | 9,582 | 27b56/4b31/12b13 |
| RouteBalance (0.8,0.1,0.1) | — | 0.6160 | 22.24 | 0.0220 | 2,673 | 7,952 | 4b40/27b27/12b24/1b9 |
| RouteBalance (0.5,0.3,0.2) | — | 0.6057 | 22.25 | 0.0208 | 2,355 | 7,758 | 4b46/27b21/12b18/1b15 |
| RouteBalance (1/3,1/3,1/3) | — | 0.6010 | 22.39 | **0.0194** | 2,108 | 7,393 | 4b56/1b17/27b15/12b12 |
| dispatcher-only | RR | 0.5951 | 22.28 | 0.0226 | 2,172 | 7,066 | 4b36/12b29/1b21/27b14 |
| dispatcher-only | Po2 | 0.5798 | 22.27 | 0.0222 | 1,880 | 6,871 | 4b36/1b32/12b22/27b9 |
| dispatcher-only | SQ | 0.5732 | **22.44** | 0.0227 | **1,622** | **6,714** | 1b47/4b32/12b15/27b6 |

**RouteBalance is cheaper than every dispatcher-only variant, and better on quality in all nine preset/dispatcher comparisons — significantly so in eight.** The uniform (1/3,1/3,1/3) preset is 12.6–14.5% cheaper than shortest-queue, Po2 and round-robin alike; the two quality-leaning presets are cheaper than all three by smaller margins (down to under 1% for the most quality-leaning preset). On quality, the paired per-prompt differences at λ=24 (1,977 prompts carrying a judged score in both cells, bootstrap 95% CIs, 2,000 resamples) are:

| preset | vs shortest-queue | vs Po2 | vs round-robin |
|---|---|---|---|
| (1/3,1/3,1/3) | +0.0262 [+0.0160, +0.0364] | +0.0197 [+0.0097, +0.0304] | +0.0058 [−0.0042, +0.0166] |
| (0.5,0.3,0.2) | +0.0326 [+0.0219, +0.0432] | +0.0246 [+0.0150, +0.0353] | +0.0112 [+0.0012, +0.0214] |
| (0.8,0.1,0.1) | +0.0423 [+0.0308, +0.0536] | +0.0346 [+0.0233, +0.0461] | +0.0224 [+0.0119, +0.0324] |

Eight of the nine intervals exclude zero. The one that does not is the uniform preset against round-robin (+0.0058, CI [−0.0042, +0.0166]) — that comparison is a tie and we report it as one. Both other presets separate from round-robin, (0.5,0.3,0.2) only narrowly (+0.0112, CI lower bound +0.0012).

The mechanism is visible in the tier mix and is the family-generality point: shortest-queue chases the fastest tier, which here is the one that is dearest to prompt (1B, 47% of its traffic), while RouteBalance's cost term reads the actual price table and settles on 4B. That is evidence the formulation generalizes, not the ladder.

**The cluster-based router buys quality with money, and the weighting is what exposes the trade.** AvengersPro reaches the highest delivered quality in the table (0.6300) by sending 56% of traffic to 27B. We do not beat it on quality, and we say so. What the presets show is that this is a *position on a frontier*: raising the quality weight, with cost and latency giving way in step — (1/3,1/3,1/3), (0.5,0.3,0.2), (0.8,0.1,0.1) — moves delivered quality monotonically — 0.6010, 0.6057, 0.6160 — with the 27B share rising 15% → 21% → 27% and cost rising with it, on a fixed deployment and with nothing retrained. AvengersPro's remaining +0.014 quality over our most quality-leaning preset, (0.8,0.1,0.1), costs 18% more per 1k requests, 39% more mean latency and a 21% wider tail. We ran AvengersPro at a single operating point on this family rather than sweeping it, so the honest reading is that both systems expose a frontier and we measured ours more densely — not that theirs is fixed at build time.

**Honest negative.** Shortest-queue attains the best mean latency *and* the best tail on this family (1,622 ms / 6,714 ms at λ=24). On Qwen it also had the best tail of the three dispatchers, though still roughly twice ours; here it takes the tail outright. The Gemma tiers sit much closer together in speed, so the mis-routing that costs it on Qwen has far less room to hurt. It buys that by serving 1B for roughly half of all requests, at the lowest delivered quality of any non-degenerate system in the table (0.5732).

**Why the trained per-request router is not in this table.** Extending it to a new family is not a configuration change — it requires fine-tuning a fresh supervised router (a DeBERTa-v3-small classifier, ~141M parameters) on a freshly judged corpus for that family's pool. That is the same category of per-family cost as the latency-profiling sweep, and this experiment was deliberately run without either. Reporting a baseline that needs a training artifact we chose not to build would compare two different onboarding budgets. One further scoping note: the AvengersPro artifact was built with this family's real published per-token prices rather than the library's default size-based cost proxy, which would have ranked 1B cheapest and is precisely the error this ladder exposes.

**W4a: Tuning burden**

Under a fixed workload and objective, the presets need no retuning as the arrival rate changes: the tier mix is stable across the rates we measured and replica-level state absorbs load shifts. Changes to workload composition, to the objective itself, or to the cluster topology may move the best operating point on the simplex and require re-sweeping the weights; an SLO-driven controller that adapts them automatically is future work. We would rather state that scope than claim a robustness we have not measured.

**W4b: Budget mechanism**

We agree with your reading, and will state the scoping plainly in the revision: on the budget axis, the contribution is the composable admission mechanism, which uses the predicted output length to admit requests at a tier they can complete within budget rather than truncating them at a dearer one. The simplex weighting alone does not enforce budgets. The runtime exhaustion-rate evidence in the paper is retained (it is evaluator-independent).
