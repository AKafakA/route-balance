
---

## POST 0 — "Thank All Reviewers and General Response" (top-level Official Comment; also
## addresses the Meta Review)

We sincerely thank all reviewers and the Area Chair for their time and constructive
feedback! Following the meta-review's two named priorities — justifying the learned
(XGBoost) estimator and comparing against general-purpose dispatchers — we have run new
experiments for both, on the submitted configuration, with the same scheduler, the same
predictor stack and the same evaluation harness throughout. Two scoping notes we state up front rather than in a footnote: the topology
experiment deliberately re-provisions the cluster (that is what it tests), and the second
model family was served on measured static per-(model, GPU) latency priors rather than newly
trained latency heads. Every other new measurement is on the submitted cluster and
configuration. Responses are posted separately for each reviewer; we would be glad to
discuss further.

To help recall the submission, the strengths highlighted by the reviewers:

[Problem & formulation]
- Reviewer 287j: "The paper aims to solve the right problem and probably does a good job in
  solving it."
- Reviewer faAU: "Sharp observation and elegant formulation ... insightful and elegant."
- Reviewer gTfM: "this is the first work to demonstrate this capability."

[Empirical performance]
- Reviewer faAU: "Strong empirical performance ... its scheduling overhead stays sub-linear
  as load grows 5× ... demonstrating the effectiveness of batched amortization."

[Design]
- Reviewer faAU: "Pragmatic engineering design ... precisely targeted, no extraneous
  complexity."

---

## POST 1 — Response to Reviewer 287j (writing · clarity · batching)

We sincerely thank the reviewer for their time and careful reading! We appreciate the
recognition that "the paper aims to solve the right problem and probably does a good job in
solving it" — the presentation issues you raise are real, and we resolve each one below with
precise wording that we commit to in the revision.

**Relation to LLM-D / AIBrix (your summary).** LLM-D and AIBrix provide serving
infrastructure with routing and load-balancing components; to our knowledge neither fuses
per-request quality–latency–cost scoring into a single assignment over heterogeneous
concrete instances, which — together with the controlled isolation of that fusion boundary
(reported in our reply to Reviewer gTfM) — is our contribution.

**Q1: Batching vs continuous batching**

> "you schedule in batches. I suppose that means that you do not use continuous-batching?
> ... Does that mean that you intentionally delay a request that arrived, still use
> continuous batching, but schedule only when you reach the required batch size/time?"

Your last reading is exactly right, with one important correction: there is no *required*
batch size. There are two batching layers, and we will reserve distinct terms for them in
the revision: (i) engine-level continuous batching — each vLLM instance composes decode
batches per iteration (the serving substrate, unchanged by our system; once dispatched, a
request immediately enters its instance's continuous-batching loop); (ii) scheduler-level
request batching ("scheduling round") — our adaptive batch-formation window groups
concurrently-arriving requests so one gathered prediction round scores the full
request×instance candidate matrix. Requests are never held for a mandatory minimum batch
size: the scheduler fires when either the adaptive size limit is reached or a timeout
expires, with the timer starting at the first arrival — a lone request waits at most the
timeout, and an "uneven" (small) batch simply fires at the timeout. All formation delay is
charged in the client-observed end-to-end latency we report.

**Why batch at all — the design rationale behind Q1.** Routers in the compared systems decide
one request at a time; forming a scheduling round buys three things they cannot have.
(i) It avoids herding: within a round each dispatch immediately updates a local view of the
chosen instance's state (dead-reckoning), so later requests do not all pile onto whichever
instance looked idle before any of them were placed. (ii) It amortizes the fixed cost of a
quality/latency-aware decision over the whole batch, which is why reported decision compute
*falls* as load rises (44.7 ms at λ=6 to 25.7 ms at λ=24) rather than growing. (iii) It gives
the competing signals a common range — quality, cost and latency are min–max normalized across
the candidates present in that round, which is what makes one weighted score meaningful across
heterogeneous tiers. This also motivates making the size limit adaptive rather than fixed (Q2):
the batch grows and the timeout lengthens in the busy regime where those benefits are largest,
and collapses to a small batch and short timeout at low load. The formation wait is charged in
full in every reported end-to-end number.

**Q2: The adaptive size-limit mechanism**

> "besides the vague sentence ... you never discuss how this mechanism works?"

Agreed — the mechanism is concrete and we will state it: the adaptive limit is a
busy-instance-ratio step schedule, where the busy-instance ratio is the fraction of
instances with at least one waiting request: ratio < 0.3 → (batch 4, 50 ms timeout);
< 0.7 → (16, 100 ms); otherwise (32, 200 ms). Our ablations bound each piece: disabling
adaptivity costs 5.1–6.9% end-to-end latency, and batch-size-1 forfeits amortization
entirely and collapses under load on the submitted implementation.

**Q3: Model choice vs KV-cache hit-rate and TTFT**

> "how can the model choice affect the KV-Cache hit-rate? How much does it affects the
> TTFT ...?"

On prefix caching: each request is a single turn — where a source corpus is multi-turn we take
the first turn only, the standard preparation in the vLLM benchmark harness — so consecutive
requests share almost no prefix, and prefix caching
is configured identically for every system since all run the same serving path — so it cannot
advantage RouteBalance. We did not run a with/without ablation. Our
target here is decode-dominated end-to-end latency; workloads where prefix reuse is central
(retrieval-augmented or genuinely multi-turn serving, where a conversation's history is
replayed) are a different regime that we leave to future work. Model choice affects KV
pressure through model size and sequence lengths. KV occupancy is not a term in the
assignment objective, though KV utilization and free-block count are among the instance-state
features the latency predictor reads, so they reach the decision through the predicted TPOT. On TTFT: the online assignment objective is decode-dominated
(learned per-(model, GPU) TPOT × queue-aware decode iterations) and carries no TTFT term —
the current phrasing will be corrected accordingly. Extending the objective to TTFT,
together with a prefix-affinity term, is future work (see also our reply to Reviewer gTfM,
Q2).

**Q4: Figure 1, appendix walkthrough, datasets, and writing items**

> "walk through the components in Figure1 ... What about the datasets used ... Figure 3
> takes large space but in reality, does not tell much ... Maybe move Table 2 ... RoBERTa is
> there but never mentioned ... LPT, the methods you compare against, dead-reckoning ..."

All accepted, and committed for the revision: (a) Figure 1 redrawn with the response return
path and correct labeling — the RoBERTa box is an alternate implementation behind the
pluggable estimator interface, not used in any reported run, and will be removed or
labeled as such; (b) an appendix walkthrough of the full request flow with dataset details;
(c) Table 2 relocated next to Equations 1–2; (d) the radar chart (Figure 3) cut in favor of
compact tables; (e) LPT presented as a motivated heuristic, with the compared methods
detailed in the appendix; (f) dead-reckoning defined where first used, with its design
rationale: it prevents later same-batch assignments from seeing stale pre-dispatch state.

**On the substantive design-justification concern.** For the "design decisions seem
arbitrary" concern, the two central choices are now isolated experimentally on the
submitted configuration: the learned latency estimator (a three-arm isolation, in our reply
to Reviewer gTfM — an experiment matching the alternative that reviewer proposed) and the
dispatcher axis (an updated Table 4 with power-of-two-choices, suggested by Reviewer faAU,
in our reply there). We would be glad to elaborate on any specific design decision.

---

## POST 2 — Response to Reviewer gTfM (estimator justification · deployment model)

We sincerely thank the reviewer for the substantive questions and for the recognition that
the paper is "mostly clearly written" and that "this is the first work to demonstrate this
capability".

**Q1: How much of the gain comes from the XGBoost predictor vs a calibrated static
lookup + live queue state?**

We built exactly the alternative you propose: the full objective with the learned head's
queue-conditioned latency estimate replaced by a fixed per-tier TPOT constant, retaining the
same objective and the same within-tier queue tiebreak. The constants are the per-tier medians
we measure across the whole sweep, so the comparator is calibrated to this deployment rather
than to any single operating point:

| tier / GPU | λ6 | λ8 | λ10 | λ12 | λ18 | λ24 | λ28 | constant used |
|---|---|---|---|---|---|---|---|---|
| 3B / A30 | 10.9 | 10.7 | 10.9 | 10.9 | 10.6 | 10.3 | 10.2 | 10.7 |
| 7B / A30 | 19.7 | 19.8 | 19.8 | 19.9 | 20.1 | 20.5 | 20.8 | 19.9 |
| 14B / V100 | 15.4 | 15.8 | 16.1 | 16.5 | 17.4 | 18.1 | 18.6 | 16.5 |
| 72B / A100 | 40.4 | 43.7 | 42.3 | 43.8 | 43.0 | 40.4 | 42.6 | 42.6 |

Each cell serves 3,534 prompts with zero failures; quality is the paper's DeepEval metric of
record under the paper's own retrieval, which scores the ~76% of served prompts that map to a
judged (prompt, model) pair, the same coverage as the submitted tables; cost is input+output
tokens at the Table-1 per-model rates, where the submission's cost axis prices output only, so
absolute cost figures are not comparable with it; quality CIs are paired per-prompt bootstraps
(2,000 resamples) over the prompts scored in both arms.

| λ | quality learned / static | Δ quality 95% CI | cost $/1k learned / static | Δ cost | mean e2e learned / static | throughput |
|---|---|---|---|---|---|---|
| 6 | 0.5074 / 0.5096 | [−0.0050, +0.0014] | 0.02442 / 0.02656 | **−8.1%** | 2,080 / 2,079 | 6.0 / 6.0 |
| 8 | 0.5076 / 0.5074 | [−0.0034, +0.0031] | 0.02405 / 0.02630 | **−8.6%** | 2,118 / 2,137 | 7.9 / 7.9 |
| 10 | 0.5073 / 0.5084 | [−0.0042, +0.0026] | 0.02376 / 0.02605 | **−8.8%** | 2,175 / 2,198 | 9.9 / 9.9 |
| 12 | 0.5073 / 0.5095 | [−0.0054, +0.0014] | 0.02372 / 0.02604 | **−8.9%** | 2,230 / 2,242 | 11.8 / 11.8 |
| 18 | 0.5060 / 0.5079 | [−0.0060, +0.0009] | 0.02345 / 0.02583 | **−9.2%** | 2,298 / 2,363 | 17.6 / 17.6 |
| 24 | 0.5043 / 0.5080 | [−0.0063, +0.0014] | 0.02327 / 0.02579 | **−9.8%** | 2,408 / 2,626 | 23.3 / 23.3 |
| 28 | 0.5064 / 0.5072 | [−0.0029, +0.0015] | 0.02335 / 0.02570 | **−9.1%** | 2,701 / 2,939 | 27.0 / 26.9 |

The λ=28 row is the mean of three paired runs; the others are single runs, as in the
submission. Three readings.

**Cost: the learned head is cheaper at every rate, by 8.1–9.8%**, at delivered quality that is
statistically unchanged and throughput that is identical to the first decimal. The mechanism is
in the routing mix: at λ=12 the learned head places 58.4% of traffic on 14B and 36.7% on 7B,
against 70.3% / 27.4% for the calibrated constant. A per-tier constant assigns one latency to
every replica of a tier; the learned head estimates each instance separately, and that finer
resolution is what lets it move roughly eleven points of traffic onto a cheaper tier without
giving up quality.

**Quality: statistically unchanged.** Every |Δq| is at most 0.004 and every interval spans
zero.

**Latency and throughput: the learned head is never slower, and faster as load rises.** The two
are within 1% up to λ=12; from λ=18 the learned head pulls ahead, by 2.7% at λ=18 and about 8%
at λ=24 and λ=28. Delivered throughput is identical throughout. So a calibrated constant is a
serviceable latency model while the cluster is lightly loaded, and stops being one as queues
begin to form — which is the regime the estimator exists for.

**What we claim, and what we do not.** The latency term must be present: removing it entirely
(below) freezes the routing mix and costs quality. Given that it is present, computing it from
serving state rather than from a per-tier constant is worth 8.1–9.8% on cost at equal quality
and equal throughput, and a latency advantage that grows with load. We do not claim the
constant is unusable — it is a reasonable model at low load, and we report it as such.

**Predictor input (your "what exactly is its input").** The learned head predicts **TPOT only** — per-output-token
decode latency — with one XGBoost head per (model, GPU) tier, trained from a tier-local QPS
sweep on that tier's head node. Its features are that tier's serving-state signals together
with the request's predicted output length; it does not consume the request text.
End-to-end latency is then formed **analytically**, not learned: predicted TPOT × the decode
iterations the request waits through (the instance's pending decode tokens ÷ its current
decode batch size, plus the request's own predicted output length, which comes from the KNN
estimator rather than from XGBoost). If the instance has a free decode slot, only the second
term applies.

**Q2: TTFT / KV-cache / prefix-cache scope**

TTFT is not an explicit term in the objective: the assignment score is decode-dominated
(predicted TPOT × queue-aware decode iterations), and the manuscript's phrasing that
suggests otherwise will be corrected. Prefix-cache affinity is likewise not modeled — our
corpus is assembled from single-turn (prompt, response) pairs, so cross-request prefix reuse
is minimal and the setting is uniform across every compared system. KV-cache occupancy is
not a term in the objective, though KV utilization and free-block count are among the
instance-state features the TPOT head consumes, so they reach the assignment through the
predicted TPOT rather than through a term of their own. Adding an explicit prefill/TTFT term and a prefix-affinity
term is future work, and we will state that scope explicitly rather than leaving it implied.

**Q3: Is the scheduler centralized?**

Yes, by design: a single assignment decision per request, taken over a global dead-reckoned
view of instance state, which is what makes the joint quality–latency–cost comparison across
heterogeneous instances possible at all. The cost of that centralization is what the paper
measures: decision compute per request *falls* from 44.7 ms at λ=6 to 25.7 ms at λ=24
because one gathered prediction round is amortized over a whole scheduling batch, and
scheduler-side residual grows 1.6× (141 → 220 ms) while offered load grows 5×. Fault
tolerance and sharding of the scheduler itself are out of scope for this paper and we do not
claim them.

**Q4: Is the deployment fixed?**

Yes — the model pool and its placement are fixed for the duration of each run; RouteBalance
is an online request-assignment layer over a given deployment, not an autoscaler. It does
respond to a changed deployment without retraining: in the topology experiment reported to
Reviewer faAU we re-provisioned the small-model pool (3B replicas 3→5, 7B 5→3) and re-ran
every system with nothing retrained. Joint optimization of placement and assignment is
future work.

---

## POST 3 — Response to Reviewer faAU (dispatchers · generality · cold-start · tuning/budget)

We sincerely thank the reviewer for the exceptionally thorough review. We take W3 first, then W2, W1 and W4. W2–W4 are measured on the submitted
configuration; W1 necessarily is not, and we state both scoping notes in our general
response.

**W3: General-purpose dispatchers (answered first — it is the meta-review's second key)**

Following your suggestion, we added standard power-of-two-choices: two candidate replicas
sampled uniformly without replacement, dispatching to the one with fewer outstanding
requests — the same load signal as shortest-queue, so telemetry staleness is identical; with
fewer than two replicas it degenerates to full comparison. (The submitted shortest-queue
baseline dispatches to the instance with the fewest outstanding requests, running + waiting
— the closest reactive LLM-serving analogue of the least-connections policy you name, though
it does not weight heterogeneous remaining decode work.) All rows below were re-measured in
a single evaluation window on the submitted configuration:

| system | dispatcher | λ6 | λ8 | λ10 | λ12 | λ18 | λ24 (mean e2e ms) |
|---|---|---|---|---|---|---|---|
| passthrough | SQ | 1947 | 1966 | 2017 | 2051 | 2105 | 2155 |
| passthrough | Po2 | 2254 | 2238 | 2261 | 2257 | 2258 | 2272 |
| BEST-Route (argmax) | Po2 | 1911 | 1991 | 2175 | 2636 | 18273 | 48182 |
| AvengersPro (pw 0.7) | Po2 | 2553 | 2592 | 2624 | 2665 | 2737 | 2818 |
| RouteBalance (uniform) | joint assignment | 2080 | 2118 | 2175 | 2230 | 2298 | 2408 |

Findings, including the one that does not favour us. Shortest-queue beats Po2 at every rate
for the passthrough family. The
trained per-request router (BEST-Route, argmax) reproduces the paper's documented serial
ceiling (18.73 req/s sustained maximum) — the collapse knee between λ12 and λ18 (2,636 → 18,273 ms) — under Po2 just
as it does under shortest-queue, and the topology sweep below shows the same collapse under
all three dispatchers: the ceiling is a routing-side property, independent of the
dispatcher. RouteBalance remains stable across the whole sweep, 20.0× lower end-to-end
latency than BEST-Route+Po2 at λ=24.

**Stated plainly: passthrough with shortest-queue has a lower mean end-to-end latency than
RouteBalance at every rate in this table** (2,155 vs 2,408 ms at λ=24). It achieves that by
not making a model choice at all — it dispatches within a fixed tier assignment and delivers
correspondingly lower quality, which is the axis this table does not carry. We flag it here
rather than leaving it for a reader to find, and the quality and cost columns for the same
comparison appear in the topology and second-family tables below. We will also correct both
places where the submission puts this too broadly — the appendix sentence claiming
RouteBalance wins on end-to-end latency "at every load we tested", and the abstract's
statement that end-to-end latency "stays below the router-only baselines" — since passthrough
is one of those baselines and this measurement contradicts both.

Two scoping notes on this table. The AvengersPro rows are run at pw = 0.7, an operating
point between the values swept in the submission, so they are a new measurement rather than
a reproduction. And a dispatcher — however strong — places requests within the replica pool
of a tier that has already been selected; it cannot revisit the model choice itself, which
is the structural point the comparison is making.

**W2 + your Limitations request: robustness under partial/missing quality signals, and
onboarding cost**

We quantify exactly the degradation you name, on the deployed estimator: (a) a
labeled-fraction ladder — the estimator rebuilt at 0/25/50/75% of quality labels and
evaluated end-to-end online; and (b) a missing-domain ablation — entire domains (math
[GSM8K] and code) held out of the estimator's corpus. Degradation is reported as deltas
against the 75%-coverage estimator built and evaluated in the same window, plus the
quality slice on prompts from the excluded domain:

| estimator corpus | Δ global quality vs 75%-coverage reference | Δ on excluded-domain prompts |
|---|---|---|
| 50% of labels | −0.002 | — |
| 25% of labels | +0.003 | — |
| 0% of labels (prompt-blind) | −0.013 | — |
| GSM8K domain held out | −0.002 (tie) | +0.002 |
| code domain held out | −0.005 | +0.001 |

The ladder is non-monotone — 25% of the labels scores marginally above 50% — which is itself
the finding: across 25–75% coverage the differences are within the run-to-run spread, so the
honest reading is that the estimator is insensitive to label coverage over that range rather
than that it degrades smoothly through it. Only the prompt-blind arm separates, and it costs
0.013. Holding an entire domain out moves global quality by at most 0.005, while quality on
the excluded domain's own prompts is statistically unchanged (paired per-prompt CIs on those
slices span zero). End-to-end latency is flat across the entire ladder (all cells within ±2%
of the reference), so the quality deltas above are the whole story.

**Onboarding cost, measured.** We onboarded a second model family end to end inside this
window, so we can report the phases as durations rather than estimates, for the whole corpus
rather than the evaluation slice. Generation broadcast 18,549 candidate prompts across the
four-model ladder on 2 A30 nodes in parallel and took approximately 6 hours. After filtering,
judging the surviving corpus — 10,065 prompts × 4 models = 40,260 (prompt, model) pairs —
took approximately 2 hours on the same two nodes. That corpus splits 8,086 for the estimator
and 1,979 held out for the cells below. Building the
similarity index takes 52.3 s (measured), and per-tier estimator training takes under a
minute each. The one step that does not fit a discussion window is the latency-profiling
corpus: for the original family that was 292,239 profiled requests across the four deployed
(model, GPU) pairings, which the corpus timestamps put at 11.1 hours of wall-clock with all
pairings profiled concurrently.

That is why the isolation experiment reported to Reviewer gTfM matters for your question
too: it licenses a practical two-stage onboarding path. A day-1 deployment needs only static
priors — brief median probes at a moderate operating rate, minutes of work — which that
isolation shows match the learned estimator on latency, throughput and quality at stationary
rates below capacity, at a cost penalty of about 7%; the learned heads then train from
accumulated serving telemetry as the system runs. We took exactly that path for the second
model family below, and we flag it so the two families are compared on a stated basis rather
than an assumed one.

**W1: Generality (model family, topology)**

**Topology.** We re-provisioned the serving cluster inside this window: the small-model pool
was re-balanced across the A30 tier, 3B replicas going from 3 to 5 and 7B from 5 to 3. We
then re-ran the comparison with no retraining of any component. Every system runs on that
re-provisioned cluster, in the same window, through the same scheduling path, on the
submitted configuration. Each cell is 3,534 prompts with zero failures. We ran the full
dispatcher sweep — shortest-queue, power-of-two and round-robin — for both the trained
per-request router and the dispatcher-only baseline, because your question is precisely
whether a result survives a change of dispatcher. Best value per column is in bold.

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

**What the re-provisioning does to a queue-only dispatcher.** The new topology has more 3B
replicas, so shortest-queue sends more traffic there and wins the mean (1,887 ms). It also
sends 4.7% of requests to the two 72B replicas, and that is where its 11.0 s p99 comes from:
a queue length says which replica is idle now, not what the request will cost there.
RouteBalance's score prices the request on the instance before placing it, and its p99 is
5,980–6,406 ms. Taking one configuration at a time: the
balanced preset delivers 0.5057 quality at 23.24 req/s with a 5,980 ms p99 and 0.0234 per
1k — higher throughput than every dispatcher with roughly half their tail, cheaper than Po2
(0.0240) and round-robin (0.0302), and ahead of every dispatcher on quality, though its
+0.003 over round-robin is inside the paired-bootstrap width we report elsewhere. The quality-leaning preset
trades that cost advantage away: 0.5138 quality at 23.25 req/s, but 0.0262 per 1k, dearer
than shortest-queue and Po2 though still cheaper than round-robin.

**What we do not claim.** RouteBalance is not best on every axis here and we will not
present it as such. Shortest-queue is cheaper (0.0222) and faster on the mean (1,887 vs
2,470 ms); the trained router is the cheapest per 1k requests, though it cannot sustain the
offered rate. On this topology RouteBalance buys quality, throughput and tail, and does not
win the mean-latency or cheapest-cost axis.

Both presets were selected on the original cluster and carried over unchanged, so the
mean-latency column is not a tuned comparison; a full weight re-sweep did not fit this
window. This experiment shows re-provisioning preserves quality, throughput, tail and weight
control — not that these presets remain optimal.

**The trained router's collapse does not depend on the dispatcher.** Under every dispatcher
it pins at 13.8–15.8 req/s against an offered 24, at 43.3–61.2 s end-to-end — 17.5–24.8×
RouteBalance. The ceiling is routing-side, not dispatcher-side, and the power-of-two-choices
dispatcher you suggested is what let us separate the two.

**W1 (continued): a second model family.** We onboarded a second, architecturally distinct
family end to end inside this discussion window — the Gemma-3 instruction-tuned ladder
(1B/4B/12B/27B) — and re-ran the comparison with no change to the scheduler. Each cell below
is 1,979 held-out prompts at λ=24 with zero failures, with quality measured by the same
DeepEval judge and the same lookup procedure as the submitted evaluation; the family's
quality estimator is built on a disjoint 8,086-prompt training split. Best value per column
is in bold.

This family is a genuine generality test rather than a re-run, because its published price
ladder is **non-monotone**. Prices per 1M tokens, input / output: 1B $0.10 / $0.10, 4B $0.04 /
$0.08, 12B $0.05 / $0.15, 27B $0.08 / $0.16. On input the smallest model is the *dearest of
the four* — above even the 27B — because few providers serve it at scale. On Qwen price tracks
size, so a scheduler that simply prefers smaller models gets cost right by accident; here it
cannot.

| system | dispatcher | quality | throughput | cost $/1k | mean e2e (ms) | p99 e2e (ms) | tier mix |
|---|---|---|---|---|---|---|---|
| AvengersPro | Po2 | **0.6300** | 22.15 | 0.0259 | 3,719 | 9,582 | 27b56/4b31/12b13 |
| RouteBalance (0.8,0.1,0.1) | — | 0.6160 | 22.24 | 0.0220 | 2,673 | 7,952 | 4b40/27b27/12b24/1b9 |
| RouteBalance (0.5,0.3,0.2) | — | 0.6057 | 22.25 | 0.0208 | 2,355 | 7,758 | 4b46/27b21/12b18/1b15 |
| RouteBalance (1/3,1/3,1/3) | — | 0.6010 | 22.39 | **0.0194** | 2,108 | 7,393 | 4b56/1b17/27b15/12b12 |
| dispatcher-only | RR | 0.5951 | 22.28 | 0.0226 | 2,172 | 7,066 | 4b36/12b29/1b21/27b14 |
| dispatcher-only | Po2 | 0.5798 | 22.27 | 0.0222 | 1,880 | 6,871 | 4b36/1b32/12b22/27b9 |
| dispatcher-only | SQ | 0.5732 | **22.44** | 0.0227 | **1,622** | **6,714** | 1b47/4b32/12b15/27b6 |

**RouteBalance is cheaper than every dispatcher-only variant, and better on quality in 8 of
the 9 preset/dispatcher comparisons.** The balanced preset is 12.6–14.5% cheaper than
shortest-queue, Po2 and round-robin alike; the two quality-leaning presets are cheaper than
all three by smaller margins (down to under 1% for the most quality-leaning preset). On quality, the paired per-prompt differences at λ=24 (1,977 prompts carrying a judged
score in both cells, bootstrap 95% CIs, 2,000 resamples) are:

| preset | vs shortest-queue | vs Po2 | vs round-robin |
|---|---|---|---|
| (1/3,1/3,1/3) | +0.0262 [+0.0160, +0.0364] | +0.0197 [+0.0097, +0.0304] | +0.0058 [−0.0042, +0.0166] |
| (0.5,0.3,0.2) | +0.0326 [+0.0219, +0.0432] | +0.0246 [+0.0150, +0.0353] | +0.0112 [+0.0012, +0.0214] |
| (0.8,0.1,0.1) | +0.0423 [+0.0308, +0.0536] | +0.0346 [+0.0233, +0.0461] | +0.0224 [+0.0119, +0.0324] |

Eight of the nine intervals exclude zero. The one that does not is the balanced preset
against round-robin (+0.0058, CI [−0.0042, +0.0166]) — that comparison is a tie and we
report it as one. Both other presets separate from round-robin, the balanced-vs-round-robin
margin only narrowly (+0.0112, CI lower bound +0.0012).

The mechanism is visible in the tier mix and is the family-generality point: a queue-aware
dispatcher chases the fastest tier, which on this family is also the *dearest* (1B), while
RouteBalance's cost term reads the actual price table and settles on 4B. That is evidence the formulation generalizes, not the ladder.

**The cluster-based router buys quality with money, and the weighting is what exposes the
trade.** AvengersPro reaches the highest delivered quality in the table (0.6300) by sending
56% of traffic to 27B. We do not beat it on quality, and we say so. What the presets show is
that this is a *position on a frontier*: sweeping the quality weight alone moves delivered
quality monotonically — 0.6010, 0.6057, 0.6160 — with the 27B share rising 15% → 21% → 27%
and cost rising with it, on a fixed deployment and with nothing retrained. AvengersPro's
remaining +0.014 quality over our quality-leaning preset costs 18% more per 1k requests, 39%
more mean latency and a 21% wider tail. We ran AvengersPro at a single operating point on this family rather than
sweeping it, so the honest reading is that both systems expose a frontier and we measured
ours more densely — not that theirs is fixed at build time.

**Honest negative.** Shortest-queue attains the best mean latency *and* the best tail on this
family (1,622 ms / 6,714 ms at λ=24), unlike on the Qwen topology above where its tail was
the worst. The Gemma tiers sit much closer together in speed, so the mis-routing that
produced the 72B tail on Qwen has far less room to hurt here. It buys that by serving 1B for
roughly half of all requests, at the lowest delivered quality of any non-degenerate system in
the table (0.5732).

**Why the trained per-request router is not in this table.** Extending it to a new family is
not a configuration change — it requires fine-tuning a fresh supervised router (a
DeBERTa-v3-small classifier, ~141M parameters) on a freshly judged corpus for that family's
pool. That is the same category of per-family cost as the latency-profiling sweep, and this
experiment was deliberately run without either. Reporting a baseline that needs a training
artifact we chose not to build would compare two different onboarding budgets. One further
scoping note: the AvengersPro artifact was built with this family's real published per-token
prices rather than the library's default size-based cost proxy, which would have ranked 1B
cheapest and is precisely the error this ladder exposes.

**W4a: Tuning burden**

Under a fixed workload and objective, the presets need no retuning as the arrival rate
changes: the tier mix is stable across the rates we measured and replica-level state absorbs
load shifts. Changes to workload composition, to the objective itself, or to the cluster
topology may move the best operating point on the simplex and require re-sweeping the
weights; an SLO-driven controller that adapts them automatically is future work. We would
rather state that scope than claim a robustness we have not measured.

**W4b: Budget mechanism**

We agree with your reading, and will state the scoping plainly in the revision: on the
budget axis, the contribution is the composable admission mechanism, which uses the
predicted output length to admit requests at a tier they can complete within budget rather
than truncating them at a dearer one. The simplex weighting alone does not enforce budgets.
The runtime exhaustion-rate evidence in the paper is retained (it is evaluator-independent).

---

## POST 4 — Official Comment to the Area Chair (short; AC-first ordering)

We thank the Area Chair for the meta-review naming the two priorities, and summarize how
each is now addressed with new experiments on the submitted configuration.

**(1) Estimator justification.** We implemented exactly the calibrated static alternative
Reviewer gTfM proposed and swept it against the learned head across the stationary range.
Below capacity the two tie on latency (within 1.5%), on throughput, and on quality
(every paired CI spans zero) — we report that plainly. What the learned head buys is cost:
8.1–9.8% cheaper per 1,000 requests at every rate, at equal quality and identical throughput,
via a routing mix that moves about eleven points off the 14B tier onto a cheaper one. It is
also never slower, and pulls ahead as load rises — 2.7% at λ=18 and about 8% at λ=24 and
λ=28. The static comparator is calibrated to the per-tier decode latencies we measure across
the sweep, so this is not an artefact of how the constant was chosen. Removing the latency
term altogether freezes the routing mix at 69.1% smallest-tier and drops quality to 0.4899.
So the latency *term* is necessary, and computing it from serving state rather than a per-tier
constant is worth cost at equal quality plus a latency margin that grows with load.

**(2) General-purpose dispatchers.** Following Reviewer faAU's suggestion we added standard
power-of-two-choices. The trained per-request router reproduces its documented serial ceiling
under Po2 exactly as under shortest-queue — and under round-robin — so the ceiling is
dispatcher-independent, while RouteBalance remains stable across the same sweep (20.0× lower
e2e at λ=24). On a re-provisioned topology the quality-leaning preset leads every dispatcher on quality
(0.5138 vs 0.4899–0.5029) and RouteBalance holds the highest throughput with roughly half
the tail of every dispatcher; on a second model family (the
Gemma-3 ladder, onboarded end to end within this window) it is cheaper than every
dispatcher-only variant — by 12.6–14.5% for the balanced preset, by smaller margins for the
quality-leaning ones — and better on quality in 8 of 9 paired comparisons. We also state
the negatives to you directly: a plain shortest-queue dispatcher achieves a lower mean
end-to-end latency than RouteBalance on both the submitted configuration and the second
family, and a cluster-ranking router reaches higher quality on that family — at 18% more
cost, 39% more mean latency and a 21% wider tail. Full tables and per-question answers are
in the individual replies.
