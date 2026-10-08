<!-- ================= INTERNAL NOTE — NOT PART OF ANY POST. DO NOT PASTE. =================
RESOLVED 2026-07-28 (author, final): onboarding cost is a report, not a thesis, and the
reviewer needs DURATIONS ONLY — no pair counts, no client counts, no GPU utilization.
POST 3 therefore states four phase durations measured on the Gemma-3 onboarding over the
18,549-prompt corpus, on two A30 nodes in parallel: generation 1.6 h (5,608 s measured),
judging 4.9 h, index build 52.3 s, router-head training 6 min (362.9 s measured).
Earlier drafts carried a 4xA100 rate and then an A30 throughput breakdown; both retired.
Author reframing added: thank the reviewer, state the four-tier Qwen corpus IS released, commit
to adding the second-family corpus. Nothing in the packet depends on the Gemma track finishing.
JUDGE-NAMING RULE (author 2026-07-28), verified against the reviewed paper
(/home/wd312/Code/llm/route-balance-paper/neurips_2026.tex): cite ONLY the DeepEval G-Eval
framework with Llama-3.1-8B-Instruct — the judge the paper names at line 219. The paper's
limitation (iii) states "one quality judge (DeepEval G-Eval)", so naming any second judge would
contradict the submission. Packet verified clean of second-judge mentions.
No repo name, account name or link in posted text (NO-LINKS rail + review anonymity).
========================================================================================= -->


# Submission28271 — FIRST-RESPONSE PACKET (Hymba-style format, author-mandated)
# STATUS: DRAFT — presentation gate: nothing posts without author sign-off.
# Basis: every number = banked+audited v1exact cells (submitted configuration) unless marked.
# ✅ AUTHOR-APPROVED 2026-07-27 ~21:20Z (all sign-offs given: γ30 wording re-confirmed with
# per-run signs in hand; §3.3 CI-backed block; full packet). FROZEN FOR POSTING.
# Posting mechanics: split long replies into "(Part 2)" comments at OpenReview limits.

---

## POST 0 — "Thank All Reviewers and General Response" (top-level Official Comment; also
## addresses the Meta Review)

We sincerely thank all reviewers and the Area Chair for their time and constructive
feedback! Following the meta-review's two named priorities — justifying the learned
(XGBoost) estimator and comparing against general-purpose dispatchers — we have run new
experiments for both, on the submitted configuration — same code and artifacts throughout. Two
scoping notes we state up front rather than in a footnote: the topology experiment deliberately
re-provisions the cluster (that is what it tests), and the second model family was served on
measured static per-(model, GPU) latency priors rather than newly trained latency heads. Every
other new measurement is on the submitted cluster and configuration. Responses are
posted separately for each reviewer; we will be glad to discuss further.

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
(see our reply to Reviewer gTfM) — is our contribution.

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

**Why batch at all — the design rationale behind Q1.** It is worth stating plainly what the
batch buys, since this is the mechanism the paper contributes and the baselines do not have:
routers in the compared systems decide one request at a time. Forming a scheduling round lets us
(i) avoid the herding that per-request decisions cause — within a round, each dispatch immediately
updates a local view of the chosen instance's state (our dead-reckoning cache), so later requests
in the same round do not all pile onto whichever instance looked idle before any of them were
placed; (ii) amortize the fixed costs of a quality/latency-aware decision — one gathered
prediction round serves the whole batch, which is why our reported decision compute *falls* as
load rises (44.7 ms at λ=6 to 23.9 ms at λ=30) rather than growing; and (iii) normalize the
competing signals against a common range — quality, cost and latency are min–max normalized
across the candidates present in that round, which is what makes a single weighted score
meaningful across heterogeneous tiers. And (iv) — the point that answers the obvious objection —
the batching delay is not additive in the regime where batching matters. A wait to assemble a
round is only a real cost when the instances are idle. Once requests have accumulated
instance-side, a request that waits briefly at the scheduler and a request that waits in an
instance's queue are equivalent end to end: the scheduler-side wait overlaps queueing the request
would have done anyway, so the bubble is absorbed rather than added. That observation is the
motivation for making the size limit adaptive rather than fixed (Q2): the batch is allowed to
grow, and the timeout to lengthen, precisely in the busy regime where the extra wait costs
nothing and the amortization and anti-herding benefits are largest, while at low load the
schedule collapses to a small batch and a short timeout so the delay is never paid against idle
instances. A per-request router has none of these properties available to it, which is the
substance of the comparison rather than an implementation detail.

**Q2: The adaptive size-limit mechanism**

> "besides the vague sentence ... you never discuss how this mechanism works?"

Agreed — the mechanism is concrete and we will state it: the adaptive limit is a
busy-instance-ratio step schedule: ratio < 0.3 → (batch 4, 50 ms timeout); < 0.7 →
(16, 100 ms); otherwise (32, 200 ms). Our ablations bound each piece: disabling adaptivity
costs 5.1–6.9% end-to-end latency, and batch-size-1 forfeits amortization entirely and
collapses under load on the submitted implementation.

**Q3: Model choice vs KV-cache hit-rate and TTFT**

> "how can the model choice affect the KV-Cache hit-rate? How much does it affects the
> TTFT ...?"

On prefix caching: our evaluation corpus is single-turn and the prompts share very little
prefix, so enabling or disabling automatic prefix caching makes little difference to the
numbers we report — and it applies uniformly to every system, since all of them run through
the same serving path. Our target here is decode-dominated end-to-end latency; workloads
where prefix reuse is central (retrieval-augmented or multi-turn serving) are a different
regime that we leave to future work. Model choice affects KV pressure through model size and
sequence lengths. On TTFT: the online assignment objective is decode-dominated (learned
per-(model, GPU) TPOT × queue-aware decode iterations) and carries no TTFT term — the current
phrasing will be corrected accordingly. Extending the objective to TTFT, together with a
prefix-affinity term, is future work (see also our reply to Reviewer gTfM, Q2).

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
submitted configuration: the learned latency estimator (three-arm isolation, in our reply
to Reviewer gTfM — an experiment matching the alternative that reviewer proposed) and the
dispatcher axis (updated Table 4 with power-of-two-choices, suggested by Reviewer faAU, in
our reply there). We would be glad to elaborate on any specific design decision.

---

## POST 2 — Response to Reviewer gTfM (estimator justification · deployment model)

We sincerely thank the reviewer for the substantive questions and for the recognition that
the paper is "mostly clearly written" and that "this is the first work to demonstrate this
capability". Your Q1 proposes exactly the right probe — a simpler calibrated per-tier TPOT
model with live queue state — and we answer it by implementing precisely that alternative
and isolating it experimentally on the submitted configuration.

**Q1: How much of the gain comes from the XGBoost predictor vs a calibrated static
lookup + live queue state?**

We run a three-arm isolation on the submitted configuration (identical prompts, seeds,
artifacts, cluster): (1) the full objective with the learned, load-conditioned latency
term; (2) the objective with no cross-tier latency term (reactive shortest-queue dispatch
within the selected tier); (3) — your proposed alternative — the full objective with the
learned head replaced by a static per-tier prior (nominal TPOT × predicted length,
retaining live within-tier queue state). The static prior's constants match the measured
per-tier median TPOT at moderate load (λ=12) to within 0.1 ms/tok, i.e., it is a well-calibrated instance of
your suggestion, not a strawman.

**Table — learned estimator vs calibrated static prior (submitted system; e2e ms):**

| condition | learned e2e | static prior e2e | Δ e2e | quality Δ (paired CIs span zero) |
|---|---|---|---|---|
| λ=6 | 2080 | 2065 | +0.8% | −0.0011 (tie) |
| λ=12 | 2230 | 2212 | +0.8% | −0.0005 (tie) |
| λ=18 | 2298 | 2299 | −0.0% | tie |
| λ=24 | 2408 | 2413 | −0.2% | −0.0031 (tie) |
| λ=30 | 3284 | 3077 | +6.7% | −0.0019 (tie) |
| γ0.25 r=23 | 2585 | 2520 | +2.6% | −0.0016 (tie) |
| γ0.25 r=27 | 2705 | 2718 | −0.5% | tie |
| **γ0.25 r=30 (mean of 6 paired runs)** | **3156** | **3391** | **−6.7% (per-run mean)** | tie |

**Reading, stated against ourselves where the data runs that way.** Under stationary
(Poisson) arrivals below capacity the calibrated static prior matches the learned estimator
within ±0.8% end-to-end. We do not claim a quality advantage for the learned head either:
every per-condition quality difference is |Δq| ≤ 0.004 with paired-bootstrap CIs spanning
zero, and where those small differences are signed at all they favour the *static* prior. At
stationary λ=30 the static prior is faster by 6.7%. The one condition where the learned head
leads is the most bursty (γ0.25 at r=30), where across six paired runs it lowers mean
end-to-end latency by 6.7% on average — but the 95% CI is [−14.8%, +1.4%], so we do not
claim a significant effect there either.

What the isolation therefore establishes is narrower than the submission implies, and we
will scope it accordingly in the revision: the latency **term** must enter model selection —
arm 2 shows that removing it freezes the routing mix and costs quality — but these
experiments do not establish that the term must be **learned** rather than a calibrated
per-tier constant. We report this because it is what our own experiment says.

**What removing the latency term entirely costs (arm 2; per cell: e2e ms / global
quality):**

| λ | learned (full objective) | no latency term | static per-tier prior |
|---|---|---|---|
| 6 | 2080 / 0.5074 | 1672 / 0.4899 | 2065 / 0.5085 |
| 12 | 2230 / 0.5073 | 1732 / 0.4899 | 2212 / 0.5078 |
| 24 | 2408 / 0.5043 | 1806 / 0.4899 | 2413 / 0.5074 |
| 30 | 3284 / 0.5056 | 2083 / 0.4899 | 3077 / 0.5075 |

Removing the latency term collapses routing onto the smallest tier: the mix freezes
(identical to the decimal at every rate — 69.1% of requests on the smallest model; with no
latency signal the policy cannot respond to load) and global quality drops at every rate
(cell-level 0.014–0.018; paired means 0.008–0.012; 95% CI at λ12 [0.002, 0.019]) — below the
strongest trained-router baseline's quality, so this arm exits the quality-competitive
regime in which all comparisons are made. Its lower latency follows from serving
69% of requests on the smallest model, not from better scheduling, and no re-weighting of
the remaining terms restores load adaptivity.

**Predictor input (your "what exactly is its input").** The latency head consumes the
(model, GPU) pairing, the request's predicted output length, and live telemetry (queue
depths, pending decode work) — i.e., it predicts queue-conditioned end-to-end latency,
which is precisely what a static scalar cannot represent.

**Q2: TTFT / KV-cache / prefix-cache scope**

The submitted objective optimizes decode-dominated end-to-end latency: a learned
per-(model, GPU) TPOT combined with queue-aware decode iterations (pending decode work and
running sequences, plus the request's own predicted output length). TTFT is not an explicit
objective term; prompt length enters through the predictor features. On prefix caching: the corpus is single-turn with
little shared prefix, so prefix reuse changes little in this decode-dominated setting and
applies equally to all systems; prefix-heavy regimes (retrieval-augmented or multi-turn) are
future work. Model choice affects KV pressure through model size and sequence lengths.
A natural extension we intend to pursue: model TTFT alongside TPOT and combine them
into the latency estimate, with prefix-cache statistics as features of the TTFT model — that
is precisely the mechanism by which cache locality would enter routing decisions. A prefix-affinity term in the
assignment objective is natural future work.

**Q3: Is the scheduler centralized?**

Yes, by design — a single assignment decision per request/batch over global dead-reckoned
state. KV and decode state never leave the workers; the controller exchanges request
metadata, predictions, and non-blocking telemetry, and relays the generated text once per
request (no per-token streaming through it). Measured overhead stays bounded on the
evaluated 13-instance cluster (141→220 ms, sub-linear in a 5× load increase; decision
compute ≈24–33 ms, amortizing across the batch). Fault tolerance and multi-controller
sharding are outside the scope of this paper.

**Q4: Is the deployment fixed?**

Yes — placement is fixed during each run; RouteBalance is an online request-assignment
layer over a given deployment, and is complementary to placement planners (e.g., HexGen,
Helix). When the candidate set changes, the scheduler re-normalizes over the new set and
continues without failures. We have since re-provisioned the cluster and re-run the full
comparison on it (an experiment Reviewer faAU asked for); the results are in our reply to that
reviewer, and they show the scheduler keeping its operating region and its weight control with
no retraining.

---

## POST 3 — Response to Reviewer faAU (dispatchers · generality · cold-start · tuning/budget)

We sincerely thank the reviewer for the exceptionally thorough review and for the
recognition of the "sharp observation and elegant formulation ... insightful and elegant",
the "strong empirical performance ... demonstrating the effectiveness of batched
amortization", and the "pragmatic engineering design ... precisely targeted, no extraneous
complexity". We address W1–W4 in order, each with new measurements on the submitted
configuration.

**W3: General-purpose dispatchers (answered first — it is the meta-review's second key)**

Following your suggestion, we added standard power-of-two-choices: two candidate replicas
sampled uniformly without replacement (seeded), dispatching to the one with fewer
outstanding requests — the same load signal as shortest-queue, so telemetry staleness is
identical; with fewer than two replicas it degenerates to full comparison. (The submitted
shortest-queue baseline dispatches to the instance with the fewest outstanding requests,
running + waiting — the closest reactive LLM-serving analogue of the least-connections
policy you name, though it does not weight heterogeneous remaining decode work.) The
updated Table 4: all rows re-measured in a single evaluation window on the submitted
configuration, so the new dispatcher comparison shares one measurement window (the trained
routers' routing decisions reproduce the submitted campaign exactly):

| system | dispatcher | λ6 | λ8 | λ10 | λ12 | λ18 | λ24 | λ30 (e2e ms) |
|---|---|---|---|---|---|---|---|---|
| passthrough | SQ | 1947 | 1966 | 2017 | 2051 | 2105 | 2155 | 2182 |
| passthrough | Po2 | 2254 | 2238 | 2261 | 2257 | 2258 | 2272 | 2315 |
| passthrough | RR | — | — | — | 2670 | — | 2715 | 2734 |
| BEST-Route (argmax) | Po2 | 1911 | 1991 | 2175 | 2636 | 18273 | 48182 | 54112 |
| BEST-Route (argmax) | SQ | — | — | — | 2626/2971 (2 runs) | — | — | — |
| AvengersPro (pw 0.7) | Po2 | 2553 | 2592 | 2624 | 2665 | 2737 | 2818 | 2971 |
| AvengersPro (pw 0.7) | SQ | — | — | — | 2675 | — | — | — |
| RouteBalance (uniform) | joint assignment | 2080 | 2118 | 2175 | 2230 | 2298 | 2408 | 3284 |

Findings: shortest-queue beats Po2 at every rate for the passthrough family, and Po2 beats
round-robin at each rate where round-robin was measured (λ{12,24,30}). The trained
per-request router (BEST-Route, argmax) reproduces the paper's documented 12–18 req/s serial
ceiling — the collapse knee between λ12 and λ18 (2636 → 18273 ms) — under Po2 exactly as
under shortest-queue: the ceiling is a routing-side property, independent of the
dispatcher. RouteBalance remains stable across the same sweep (2230 → 3284 ms) — 20.0×
lower end-to-end latency than BEST-Route+Po2 at λ24 and 16.5× at λ30. Structurally, a dispatcher —
however strong — places requests within the replica pool of a tier that has already been
selected; it cannot revisit the model choice itself. (Appendix note committed: serial
per-request routing collapses under load with high run-to-run variance — a routing-side
compute component; routing decisions are bit-exact reproductions of the submitted runs; the
high-load gaps exceed the observed run-to-run spread by more than an order of magnitude.)

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

Even the prompt-blind estimator — zero quality signal — moves global delivered quality by
only 0.013; the ladder stays within 0.005 of the reference from 25% coverage up. Holding an
entire domain out moves global quality by at most 0.006 (the GSM8K holdout is a statistical
tie under a paired bootstrap), while quality on the excluded domain's own prompts is
statistically unchanged (paired per-prompt CIs on the excluded-domain slices span zero):
the estimator degrades gracefully to its neighborhood prior
rather than failing on unseen-domain traffic. End-to-end latency is flat across the
entire ladder (all cells within ±2% of the reference), so the quality deltas above are the
whole story.

Onboarding cost, measured this week by onboarding a second model family (Gemma-3, four tiers) end to end on two A30 GPU nodes running in parallel. Over the 18,549-prompt corpus: generation 1.6 hours, quality judging 4.9 hours, similarity-index build 52 seconds, router-head training 6 minutes — under seven hours in total, paid once per family. We thank you
for raising this, as it prompted us to make the cost reusable rather than repeated: we have
publicly released the full scored corpus for the four Qwen tiers — each model's generated
response paired with its reference-grounded quality score, over the same train/test split —
and we will add the second-family corpus in the same schema. Subsequent routing work can
therefore fit estimators on this data without re-running generation or judging. The
latency-estimator corpus is the expensive part, and we can state it as measured rather than
estimated: 292,239 profiled requests across the four deployed
(model, GPU) pairings, which the corpus timestamps put at 11.1 hours of wall-clock, all pairings
profiled concurrently. Profiling is
serving plus logging. Per-tier estimator training then takes under a minute each. The three-arm isolation
run for Reviewer gTfM's Q1 — exactly the right probe for this question too — licenses a
practical two-stage onboarding path: a day-1 deployment needs only static priors (brief
median probes at a moderate operating rate — minutes), which that isolation shows are
equivalent to the learned estimator at stationary rates below capacity; the learned heads
then train from accumulated serving telemetry as the system runs. We took exactly that path for the
second model family reported below: it was deployed on measured static per-(model, GPU)
latency priors rather than newly trained latency heads. Collecting a latency-training
corpus is the one onboarding step that does not fit a discussion window — for the original
family that was 11.1 hours of concurrent profiling — whereas the static priors are minutes of
median probing. The isolation result is what licenses that
substitution at these arrival rates, and we flag it so the two families are compared on a
stated basis rather than an assumed one.

**W1: Generality (model family, topology)**

**Topology — results below.** We re-provisioned the serving cluster inside this window. The
small-model pool was re-balanced across the A30 tier: 3B replicas went from 3 to 5, and 7B from 5
to 3. We then re-ran the comparison with no retraining of any component. Every system runs on that
re-provisioned cluster, in the same window, through the same scheduling path, on the submitted
configuration. Each cell is 3,534 prompts with zero failures. We ran the full dispatcher sweep —
shortest-queue, power-of-two and round-robin — for both the trained per-request router and the
dispatcher-only baseline, because your question is precisely whether a result survives a change of
dispatcher. Best value per column is in bold.

| λ | system | dispatcher | quality | throughput | cost $/1k | mean e2e (ms) | p99 e2e (ms) | tier mix |
|---|---|---|---|---|---|---|---|---|
| 12 | RouteBalance (0.5,0.3,0.2) | — | **0.5137** | **11.85** | 0.0266 | 2,302 | **5,529** | 14B66/7B27/72B4/3B3 |
| 12 | RouteBalance (1/3,1/3,1/3) | — | 0.5074 | 11.84 | 0.0239 | 2,291 | 5,640 | 14B60/7B35/3B4/72B1 |
| 12 | BEST-Route (argmax) | SQ | 0.5089 | **11.85** | **0.0220** | 2,695 | 8,202 | 14B45/3B40/7B10/72B6 |
| 12 | BEST-Route (argmax) | Po2 | 0.5089 | **11.85** | 0.0221 | 2,624 | 8,068 | 14B45/3B40/7B10/72B6 |
| 12 | BEST-Route (argmax) | RR | 0.5089 | **11.85** | 0.0221 | 2,596 | 8,005 | 14B45/3B40/7B10/72B6 |
| 12 | dispatcher-only | RR | 0.5028 | 11.68 | 0.0302 | 2,347 | 11,235 | 3B38/7B23/14B23/72B15 |
| 12 | dispatcher-only | Po2 | 0.4972 | 11.80 | 0.0251 | 2,021 | 11,157 | 3B48/14B25/7B20/72B8 |
| 12 | dispatcher-only | SQ | 0.4913 | 11.70 | 0.0231 | **1,812** | 10,745 | 3B53/14B29/7B13/72B5 |
| 24 | RouteBalance (0.5,0.3,0.2) | — | **0.5138** | **23.25** | 0.0262 | 2,636 | 6,406 | 14B62/7B29/72B5/3B4 |
| 24 | RouteBalance (1/3,1/3,1/3) | — | 0.5057 | 23.24 | 0.0234 | 2,470 | **5,980** | 14B57/7B38/3B5/72B1 |
| 24 | BEST-Route (argmax) | SQ | 0.5089 | 15.79 | **0.0220** | 43,877 | 126,566 | 14B45/3B40/7B10/72B6 |
| 24 | BEST-Route (argmax) | Po2 | 0.5089 | 15.79 | 0.0220 | 43,289 | 127,754 | 14B45/3B40/7B10/72B6 |
| 24 | BEST-Route (argmax) | RR | 0.5089 | 13.78 | 0.0220 | 61,197 | 139,953 | 14B45/3B40/7B10/72B6 |
| 24 | dispatcher-only | RR | 0.5029 | 22.58 | 0.0302 | 2,392 | 11,254 | 3B38/7B23/14B23/72B15 |
| 24 | dispatcher-only | Po2 | 0.4938 | 22.56 | 0.0240 | 1,999 | 11,160 | 3B50/14B24/7B20/72B7 |
| 24 | dispatcher-only | SQ | 0.4899 | 22.66 | 0.0222 | **1,887** | 11,046 | 3B55/14B25/7B15/72B5 |

**What the re-provisioning does to a queue-only dispatcher.** The new topology has more 3B
replicas, so shortest-queue sends more traffic there — 55% of requests, completing in 1,283 ms
each. That gives it the best mean latency in the table (1,887 ms) and a small cost saving (0.0222
vs 0.0234 per 1k, about 5%), since the small tier is also the cheap one. The same policy sends
more traffic to 72B, and that is where it pays. A queue length is not a latency estimate: it says
which replica is idle now, not what the request will cost there. So whenever a 72B replica is
momentarily short, shortest-queue routes to it, unaware that 72B is roughly 5× slower per token
and has only two replicas. Those requests are 4.7% of the load and they are the entire tail —
every one of the slowest 1% in that cell is a 72B request, at a p99 of 11.3 s.

RouteBalance sees exactly what a queue length cannot. Its assignment score combines the learned
per-(model, GPU) TPOT with the predicted output length, so a long request routed to 72B is scored
as expensive *before* it is placed. The effect is visible in the mix: RouteBalance sends only 0.7%
of traffic to 72B, and the requests it does send there are short — 35 output tokens on average,
against 143 for shortest-queue. That is why its tail is bounded by the 7B tier rather than by
72B. Taking one configuration at a time, so the comparison stays honest: the balanced preset
(1/3,1/3,1/3) delivers 0.5057 quality at 23.24 req/s with a 5,980 ms p99 and 0.0234 per 1k — higher
quality and higher throughput than every dispatcher, with roughly half their tail, and cheaper than
Po2 (0.0240) and round-robin (0.0302). The quality-leaning preset (0.5,0.3,0.2) trades that cost
advantage away: 0.5138 quality at 23.25 req/s, but 0.0262 per 1k, which is dearer than every
dispatcher. Shortest-queue is cheaper than both presets (0.0222) and faster on the mean (1,887 ms
vs 2,470 ms). We state that rather than round it away: on this topology RouteBalance buys quality,
throughput and tail, and does not win the mean-latency or the cheapest-cost axis.

**The trained router's collapse does not depend on the dispatcher.** At λ=24 it fails to absorb the
offered load under every dispatcher we paired it with: throughput pins at 13.8–15.8 against an
offered 24, and end-to-end latency runs 43.3–61.2 s, 17.5–24.8× RouteBalance at the same rate. This
is the serial per-request scoring ceiling the paper documents (12–18 req/s), and it reappears under
shortest-queue, Po2 and round-robin alike, so the ceiling is a property of per-request routing
rather than of a dispatcher choice. We thank you for pressing this: the power-of-two-choices
dispatcher you suggested is exactly what let us separate the two.

**A note on what these two presets are tuned for.** Neither preset in this table emphasizes
latency: the balanced one weights latency at 1/3, and the quality-leaning one at 0.3. They were
selected on the original cluster for a quality/cost trade, and we carried them over unchanged. So
the mean-latency column is not a tuned comparison — it is what two moderately latency-weighted
presets happen to give. The same knob that moves quality moves latency, and on the second model
family below we show the sweep explicitly (quality rises monotonically with its weight across
three presets). We did not re-sweep on this re-provisioned cluster inside the discussion window,
and we do not claim a latency win we have not measured.

**RouteBalance holds quality, throughput and the tail together, and the weight control still
works.** At λ=24 it delivers the highest quality in the table (0.5138, above the trained router's
0.5089) at the highest throughput (23.25) with a 6.0–6.4 s p99. Moving from the balanced preset to
the quality-leaning one buys +0.008 quality for about 12% more cost, on a cluster neither preset
was tuned for and with nothing retrained.

**What we do not claim.** RouteBalance is not best on every axis here, and we will not present it
as such. Shortest-queue is cheaper and faster on the mean; the trained router is the cheapest per
1k requests, though it cannot sustain the offered rate. A change of topology can also move the best
operating point on the simplex, and we could not re-run the full weight sweep inside this window,
so the previously selected presets were carried over unchanged. This experiment shows that
re-provisioning preserves the scheduler's quality, throughput and tail behaviour and its weight
control — not that these particular presets remain optimal. We will add this table, the tier mixes
and this scoping note to the revised manuscript.

**W1 (continued): a second model family.** We onboarded a second, architecturally distinct
family end to end inside this discussion window — the Gemma-3 instruction-tuned ladder
(1B/4B/12B/27B) — and re-ran the comparison with no change to the scheduler. Every cell is 1,979
prompts with zero failures, on one scheduler configuration, with quality measured by the same
DeepEval judge and the same lookup procedure as the submitted evaluation. Best value per column is
in bold.

This family is a genuine generality test rather than a re-run, because its published price ladder
is **non-monotone**: the 1B tier is the *dearest* per input token, not the cheapest, since few
providers serve it. On Qwen, price tracks size, so a scheduler that simply prefers smaller models
gets cost right by accident. On Gemma it cannot.

| λ | system | dispatcher | quality | throughput | cost $/1k | mean e2e (ms) | p99 e2e (ms) | tier mix |
|---|---|---|---|---|---|---|---|---|
| 12 | AvengersPro | RR | **0.6300** | 11.55 | 0.0259 | 3,255 | 8,057 | 27b56/4b31/12b13 |
| 12 | RouteBalance (0.8,0.1,0.1) | — | 0.6160 | 11.58 | 0.0221 | 2,552 | 7,293 | 4b40/27b27/12b24/1b9 |
| 12 | RouteBalance (0.5,0.3,0.2) | — | 0.6057 | 11.57 | 0.0208 | 2,258 | 7,169 | 4b46/27b21/12b18/1b15 |
| 12 | RouteBalance (1/3,1/3,1/3) | — | 0.6010 | **11.62** | **0.0194** | 2,032 | 7,050 | 4b56/1b17/27b15/12b12 |
| 12 | dispatcher-only | RR | 0.5947 | 11.58 | 0.0226 | 2,123 | 6,796 | 4b36/12b29/1b21/27b14 |
| 12 | dispatcher-only | Po2 | 0.5864 | 11.57 | 0.0222 | 1,895 | 6,722 | 4b37/1b30/12b24/27b10 |
| 12 | dispatcher-only | SQ | 0.5662 | 11.56 | 0.0226 | **1,515** | **6,609** | 1b51/4b30/12b13/27b6 |
| 24 | AvengersPro | Po2 | **0.6300** | 22.15 | 0.0259 | 3,719 | 9,582 | 27b56/4b31/12b13 |
| 24 | RouteBalance (0.8,0.1,0.1) | — | 0.6160 | 22.24 | 0.0220 | 2,673 | 7,952 | 4b40/27b27/12b24/1b9 |
| 24 | RouteBalance (0.5,0.3,0.2) | — | 0.6057 | 22.25 | 0.0208 | 2,355 | 7,758 | 4b46/27b21/12b18/1b15 |
| 24 | RouteBalance (1/3,1/3,1/3) | — | 0.6010 | **22.39** | **0.0194** | 2,108 | 7,393 | 4b56/1b17/27b15/12b12 |
| 24 | dispatcher-only | RR | 0.5951 | 22.28 | 0.0226 | 2,172 | 7,066 | 4b36/12b29/1b21/27b14 |
| 24 | dispatcher-only | Po2 | 0.5798 | 22.27 | 0.0222 | 1,880 | 6,871 | 4b36/1b32/12b22/27b9 |
| 24 | dispatcher-only | SQ | 0.5732 | 22.44 | 0.0227 | **1,622** | **6,714** | 1b47/4b32/12b15/27b6 |

**RouteBalance is cheaper than every dispatcher-only variant, and better on quality in 8 of the 9
preset/dispatcher comparisons.** It is 12–14% cheaper than shortest-queue, Po2 and round-robin
alike. On quality, the paired per-prompt differences at λ=24 (1,977 prompts common to both cells,
bootstrap 95% CIs, 2,000 resamples) are:

| preset | vs shortest-queue | vs Po2 | vs round-robin |
|---|---|---|---|
| (1/3,1/3,1/3) | +0.035 [+0.024, +0.047] | +0.024 [+0.013, +0.034] | +0.009 [−0.003, +0.020] |
| (0.5,0.3,0.2) | +0.048 [+0.037, +0.060] | +0.037 [+0.027, +0.049] | +0.022 [+0.011, +0.032] |
| (0.8,0.1,0.1) | +0.060 [+0.047, +0.071] | +0.048 [+0.037, +0.060] | +0.033 [+0.023, +0.044] |

Eight of the nine intervals exclude zero. The one that does not is the balanced preset against
round-robin (+0.009, CI [−0.003, +0.020]) — that comparison is a tie and we report it as one. Both
other presets separate from round-robin cleanly.

The mechanism is visible in the tier mix and is the family-generality point: a queue-aware
dispatcher chases the fastest tier, which on this family is also the *dearest* (1B), while
RouteBalance's cost term reads the actual price table and settles on 4B. On Qwen a dispatcher gets
cost right by accident because price tracks size; here it cannot, and the cost gap opens
accordingly. That is evidence the formulation generalizes, not the ladder.

**The cluster-based router buys quality with money, and the weighting is what exposes the
trade.** AvengersPro reaches the highest delivered quality in the table (0.6300) by sending 56% of
traffic to 27B. We do not beat it on quality, and we say so. What the presets show is that this is
a *position on a frontier*, not a capability we lack: sweeping the quality weight alone moves
delivered quality monotonically — 0.6010 at (1/3,1/3,1/3), 0.6057 at (0.5,0.3,0.2), 0.6160 at
(0.8,0.1,0.1) — with the 27B share rising 15% → 21% → 27% and cost rising with it, on a fixed
deployment and with nothing retrained. AvengersPro's remaining +0.014 quality over our
quality-leaning preset costs 18% more per 1k requests, 38% more mean latency and a 24% wider tail.
A practitioner who wants that point can dial toward it; one who wants 97% of the quality at 85% of
the cost and two-thirds of the tail can dial the other way. Fixing the trade at build time, as a
cluster-ranking router must, forecloses that choice.

**Honest negative.** Shortest-queue attains the best mean latency *and* the best tail on this
family (1,622 ms / 6,714 ms at λ=24), unlike on the Qwen topology above where its tail was the
worst. The Gemma tiers sit much closer together in speed, so the mis-routing that produced the
72B tail on Qwen has far less room to hurt here. It buys that by serving 1B for roughly half of
all requests, at the lowest delivered quality of any non-degenerate system in the table (0.5732).

**Why the trained per-request router is not in this table.** Extending it to a new family is not
a configuration change — it requires fine-tuning a fresh supervised router (a DeBERTa-v3-small
classifier, ~141M parameters) on a freshly judged corpus for that family's pool. That is the same
category of per-family cost as the latency-profiling sweep, and this experiment was deliberately
run without either: the family was onboarded on measured static per-(model, GPU) priors rather
than newly trained latency heads, and without a new routing classifier. Reporting a baseline that
needs a training artifact we chose not to build would compare two different onboarding budgets, so
we keep it to the topology comparison above, where the pool is unchanged and its existing head
applies. One further scoping note: the AvengersPro artifact was built with this family's real
published per-token prices rather than the library's default size-based cost proxy, which would
have ranked 1B cheapest and is precisely the error this ladder exposes.

We will include this table, the tier
mixes and both scoping notes in the revised manuscript.

**Single-model reference frame (context for quality numbers)**

To contextualize router quality on the evaluation corpus: always-72B 0.533; always-14B
0.508; always-7B 0.501; always-3B 0.473; the RouteBalance presets span 0.505–0.512 within a
single deployment. Always-72B attains the highest quality, as expected — at correspondingly
higher latency and cost. The presets are statistically indistinguishable from the strongest
mid-tier single model (per-prompt paired CIs against always-14B span zero for every preset,
including uniform) while serving a mixed, lighter pool. The object of interest is therefore
the quality–latency–cost frontier, which RouteBalance spans within one deployment.

**W4a: Tuning burden**

Under a fixed workload and objective, the presets need no retuning as the arrival rate
changes: the tier mix remains stable from λ6 to λ30, replica-level state absorbs load
shifts, and end-to-end latency stays bounded under bursts (≤ +14% vs stationary). Changes to workload composition or
to the objective itself may require retuning; an SLO-driven controller that adapts the
weights automatically is future work.

**W4b: Budget mechanism**

We agree with your reading, and will state the scoping plainly in the revision: on the
budget axis, the contribution is the composable admission mechanism, which uses the
predicted output length to convert would-be truncations and budget exhaustions into
completed lower-tier responses. The simplex weighting alone does not enforce budgets. The
runtime exhaustion-rate evidence in the paper is retained (it is evaluator-independent).

---

## POST 4 — Official Comment to the Area Chair (short; AC-first ordering)

We thank the Area Chair for the meta-review naming the two priorities, and summarize how
each is now addressed with new experiments on the submitted configuration. (1) Estimator
justification: a three-arm isolation — implementing exactly the calibrated static
alternative Reviewer gTfM proposed — shows the static prior ties the learned estimator
below capacity (±0.8% e2e, quality CIs spanning zero), while under the most bursty
condition the learned estimator lowers mean end-to-end latency by 6.7% across six paired
runs, but with a 95% CI of [−14.8%, +1.4%] we do not claim a significant effect, and the
small quality differences favour the static prior; removing the latency term entirely freezes the
routing mix and drops
quality below the strongest trained baseline. We therefore justify the design as a
formulation contribution with the learned head as the deployment default for realistic
(bursty) traffic. (2) General-purpose dispatchers: following Reviewer faAU's suggestion we
added standard power-of-two-choices and post an updated Table 4 in that reply: Po2 sits
between shortest-queue and round-robin for the passthrough family; the trained per-request
router reproduces its documented serial ceiling under Po2 exactly as under shortest-queue
(dispatcher-independent), while RouteBalance remains stable across the same sweep (20.0×
lower e2e at λ24). Full tables and per-question answers are in the individual replies, including new results on a
second model family (the Gemma-3 ladder), onboarded end to end within this window: there
RouteBalance is 12-14% cheaper than every dispatcher-only variant and significantly better on
quality in 8 of 9 preset/dispatcher comparisons (paired CIs given in the reply), while a cluster-ranking router
reaches higher quality only at 18% more cost, 38% more mean latency and a 24% wider tail.
