## POST 3a — Response to Reviewer faAU (1/3): general-purpose dispatchers

We sincerely thank the reviewer for the exceptionally thorough review. Because the response is long, we split it across three comments: (1) dispatchers, (2) topology, (3) model family and tuning. W2–W4 are measured on the submitted configuration; W1 necessarily is not, and we state both scoping notes in our general response.

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

That is why the isolation experiment reported to Reviewer gTfM matters for your question too: it licenses a practical two-stage onboarding path. A day-1 deployment needs only static priors — per-tier decode medians, which cost a short profiling pass rather than a judged corpus and a trained head — which that isolation shows match the learned estimator on quality and throughput across the whole stationary range, and on latency at the two rates where we measured both arms in one session, at a cost penalty of 8.8–10.8%; the learned heads then train from accumulated serving telemetry as the system runs. We took exactly that path for the second model family below, and we flag it so the two families are compared on a stated basis rather than an assumed one.
