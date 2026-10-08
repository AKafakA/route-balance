# Submission28271 — ASSEMBLED reply drafts (2026-07-27)

**STATUS: DRAFT — pending author sign-off per table (presentation gate).** Nothing posts
without explicit author approval against the presentation inventory.

**Assembly provenance.** Prose: survivor sections of `2026-07-25_reviewer_actions_table_v2_and_drafts.md`
(everything in the VOID register / withdrawn t-variant blocks purged). Numbers: ONLY from
`2026-07-27_rebuttal_draft_noGemma.md` (filled Tables V1/V2 + supporting blocks). Rails: the
noGemma frame header + BINDING RULINGS 2026-07-27 in `2026-07-25_EXECUTION_HANDBOOK.md`.

**AUTHOR REVIEW ITEMS (updated Jul-27 revision pass):**
1. RESOLVED — arm-2 v1exact cells banked 14:32Z; mix-shift sentence RESTORED (AC + 2.1) on measured data.
2. RESOLVED — k-sensitivity stays dropped (author ruling: not reviewer-asked).
3. RESOLVED — W4a/W4b are faAU's own weaknesses (per the actual reviews): slotted as §3.7/§3.8 with the locked survivor texts.
4. SUPERSEDED — square-wave 40/6 row WITHDRAWN from all reviewer-facing surfaces (gamma-only burst ruling); cells internal.

**Outstanding placeholders in this draft:** [TOPOLOGY-FLIP ROWS] (Table V3; argmax column
pending topoB redeploy) · [V4 RELATIVE ROWS] (Table V4; compute from banked v1exact 6-cell
set) · [BOTH-BASES CHECK] (any quality-ordering sentence in the dispatcher section).
pw-sweep rows: REMOVED (good-to-have per author ~14:55Z; pw0.7 rows filled from banked core).

**Rails checked against every sentence below:** no cross-era latency outside v1exact cells ·
no mixed-basis tables · no k statements · no re-assertion of the withdrawn frontier
characterization or any t05 framing · Pareto/frontier-shaped claims only vs single-model
references (never "beats always-72B") · ONE RB preset per table (default uniform; unshown
preset banked, truthful-if-asked) · NO links/URLs · internal codename appears nowhere ·
no quality margin ≤0.01 claimed without a paired CI (ties reported as "paired CI spans zero") ·
serial-router variance uses the handbook appendix-note framing exactly.

---

## 1) Reply to Reviewer 287j (writing · clarity · batching)

Thank you for the careful reading — the presentation issues you raise are real, and we resolve
each one here with precise wording that we commit to in the revision.

**Relation to LLM-D / AIBrix (your summary).** LLM-D and AIBrix provide serving
infrastructure with routing and load-balancing components; to our knowledge neither fuses
per-request quality–latency–cost scoring into a single assignment over heterogeneous
concrete instances, which — together with the controlled fusion-boundary isolation below —
is our contribution.

**Batching (the main source of confusion).** We will clarify the two batching layers
explicitly: (i) engine-level continuous batching — each vLLM instance composes decode batches
per iteration (the serving substrate we build on, unchanged by our system); (ii)
scheduler-level request batching — our adaptive batch-formation window groups
concurrently-arriving requests so a single predictor invocation scores the full
request×instance candidate matrix. In the revision we reserve "continuous batching" for the
engine layer and use "scheduling round" for our layer throughout. Requests are never held for
a mandatory minimum batch size: the scheduler fires when either the adaptive size limit is
reached or a timeout expires, with the timer starting at the first arrival — a lone request
waits at most the timeout. The adaptive limit is a busy-instance-ratio step schedule:
ratio < 0.3 → (batch 4, 50 ms); < 0.7 → (16, 100 ms); otherwise (32, 200 ms). All formation
delay is charged in the client-observed end-to-end latency we report. Our ablations bound each
piece: disabling adaptivity costs 5.1–6.9% end-to-end latency, and batch-size-1 forfeits
amortization entirely and collapses under load on the submitted implementation.

**Figure and appendix commitments.** On Figure 1: the RoBERTa box is an alternate
implementation behind the pluggable estimator interface, not used in any reported run — we
will redraw the figure with the response return path and correct labeling. We further commit
to: relocating Table 2 as suggested; cutting the radar chart; an appendix walkthrough with
dataset details; presenting LPT as a motivated heuristic; and defining dead-reckoning where
first used (its design rationale: it prevents later same-batch assignments from seeing stale
pre-dispatch state).

**Terminology corrections (committed).** "Shortest queue" will be described precisely: it
dispatches to the instance with the fewest outstanding requests (running + waiting) — the
closest reactive LLM-serving analogue of least-connections, though it does not weight
heterogeneous remaining decode work. On TTFT: the online assignment objective is
decode-dominated (learned per-(model, GPU) TPOT × queue-aware decode iterations); a separate
TTFT head exists for profiling and analysis but is not an explicit objective term — the
current phrasing will be corrected accordingly.

For the substantive estimator question, please see the three-arm isolation experiment and its
tables in our reply to Reviewer gTfM (Table V1 there), run entirely on the submitted
configuration.

---

## 2) Reply to Reviewer gTfM (estimator justification · deployment model)

Thank you for the substantive questions on the latency estimator and the deployment model.
We address Q1 with new measurements on the submitted configuration, and Q2–Q4 directly.

### 2.1 What the learned estimator buys: a three-arm isolation (Q1)

We isolate the learned latency estimator's contribution directly with a three-arm experiment on
the submitted configuration (identical prompts and seeds): (1) the full objective with the
learned, load-conditioned latency term; (2) the objective with no cross-tier latency term
(reactive shortest-queue dispatch within the selected tier); (3) the full objective with the
learned head replaced by a static per-tier prior (nominal TPOT × predicted length), retaining
live within-tier queue state.

The reasoning the data supports: (i) per-token decode latency is load-dependent — it is not a
constant of the (model, GPU) pairing, so any fixed per-tier estimate degrades as the system
approaches saturation; (ii) a deployed system cannot know its future operating rate, so a
static prior commits to one calibration point in advance; (iii) we measure what that
commitment costs when conditions mismatch; (iv) the learned predictor removes the assumption —
it tracks the operating point rather than betting on it.

**Table V1 — learned estimator vs static per-tier prior (submitted system; e2e in ms):**

| condition | learned e2e | static prior e2e | Δ e2e | quality Δ (paper basis; paired CIs span zero) |
|---|---|---|---|---|
| λ=6 | 2080 | 2065 | +0.8% | −0.0011 (tie) |
| λ=12 | 2230 | 2212 | +0.8% | −0.0005 (tie) |
| λ=18 | 2298 | 2299 | −0.0% | tie |
| λ=24 | 2408 | 2413 | −0.2% | −0.0031 (tie) |
| λ=30 | 3284 | 3077 | +6.7% | −0.0019 (tie) |
| γ0.25 r=23 | 2585 | 2520 | +2.6% | −0.0016 (tie) |
| γ0.25 r=27 | 2705 | 2718 | −0.5% | tie |
| **γ0.25 r=30** | **3085** | **3642** | **−15.3%** | +0.0005 (tie) |

**Removing the latency term entirely (three-arm view; per cell: e2e ms / global quality, paper basis):**

| λ | learned (full objective) | no latency term | static per-tier prior |
|---|---|---|---|
| 6 | 2080 / 0.5074 | 1672 / 0.4899 | 2065 / 0.5085 |
| 12 | 2230 / 0.5073 | 1732 / 0.4899 | 2212 / 0.5078 |
| 24 | 2408 / 0.5043 | 1806 / 0.4899 | 2413 / 0.5074 |
| 30 | 3284 / 0.5056 | 2083 / 0.4899 | 3077 / 0.5075 |

Removing the latency term collapses routing onto the smallest tier: the mix freezes (identical
to the decimal at every rate — 69.1% of requests on the smallest model; with no latency signal
the policy cannot respond to load) and global quality drops by roughly 0.01–0.02 at every
rate (sign-stable on both scoring bases; paired-bootstrap 95% CI at λ12: [0.003, 0.019]),
while the full objective holds quality and adapts its mix
as rate rises (dominant-tier share 62.9% → 55.4% from λ6 → λ30).

> [INTERNAL — DO NOT POST] sq40/6 cells WITHDRAWN from reviewer-facing surfaces per the
> gamma-only burst ruling (Jul-27); cells remain banked internally.

*Verification (not posted): basis v1exact-v2, the paper's binary (A0 re-gate PASS 03:06Z:
≤0.8pp mix, +0.7% e2e vs May). All cells: submitted v1 configuration and code (tree 553d76c3 =
May-8 + 2 inert env-gated additions, re-certified against the May campaign at ≤0.8pp mix /
≤1% e2e), Mar-15 estimator artifacts, seed 1, n=3534/cell, 0 failures, per-cell exactness
audits. RB preset: default uniform (arm-1's preset); ss532 cells banked, truthful-if-asked.
Source cells: v1exact_bridge/v1x_arm{1,4}_uniform_l{6,8,10,12,18,24,30} +
v1x_arm{1,4}_g025_r{6,12,23,27,30} + v1x_arm2_uniform_l{6,12,24,30}. Dual-basis rule applied: every Δ sign-checked on
both bases; gamma-quality wording fixed-basis-scoped; γ27 reported as parity (its current-tree
win is basis-dependent, not claimed).*

**Reading of the table.** Under stationary arrivals below capacity (λ6–24) the static prior
matches the learned estimator within ±0.8% end-to-end at statistically indistinguishable
quality (all |Δq| ≤ 0.004; paired-bootstrap CIs span zero in every condition) — the headline results do not depend on the
learned head. Under bursty, capacity-adjacent arrivals (gamma inter-arrivals, burstiness 0.25,
rate 30) the learned predictor delivers 15.3% lower end-to-end latency at statistically
unchanged quality; at moderate bursty rates the two are comparable (differences small and
mixed in sign). At stationary λ=30 the
static prior is nominally faster (learned +6.7%), which we report plainly: at steady
saturation a fixed calibration holds; the learned estimator's edge is under burst and
mismatch — the regime a deployment cannot rule out. We therefore defend the design as a
formulation contribution — the latency term must enter model selection — with the learned head
as the deployment default for calibration under drift and SLO headroom, explicitly not
load-bearing for the headline results.

(The dispatcher-level comparison referenced alongside — updated Table 4 with the added Po2
dispatcher — appears in our reply to Reviewer faAU.)

### 2.2 TTFT / KV-cache / prefix-cache scope (Q2)

The submitted objective optimizes decode-dominated end-to-end latency: a learned
per-(model, GPU) TPOT combined with queue-aware decode iterations (pending decode work and
running sequences, plus the request's own predicted output length). TTFT is separately
profiled by a dedicated head but is not an explicit objective term; prompt length enters
through the predictor features. Prefix caching (APC) was disabled in all reported runs, so
prefix-hit effects are not a confound in any reported number; model choice affects KV
pressure through model size and sequence lengths only. A prefix-affinity term in the
assignment objective is natural future work.

### 2.3 Is the scheduler centralized? (Q3)

Yes, by design — a single assignment decision per request/batch over global dead-reckoned
state. KV and decode state never leave the workers; the controller exchanges request
metadata, predictions, and non-blocking telemetry, and relays the generated text once per
request (no per-token streaming through it). Measured overhead stays bounded on the
evaluated 13-instance cluster (141→220 ms, sub-linear in a 5× load increase; decision
compute ≈24–33 ms, amortizing across the batch). Fault tolerance and multi-controller
sharding are outside the scope of this paper.

### 2.4 Is the deployment fixed? (Q4)

Yes — placement is fixed during each run; RouteBalance is an online request-assignment
layer over a given deployment, and is complementary to placement planners (e.g.,
HexGen, Helix). When the candidate set changes, the scheduler re-normalizes over the new
set and continues without failures.


---

## 3) Reply to Reviewer faAU (dispatchers · topology · generality · cold-start · tuning/budget)

### 3.1 Updated Table 4 after adding the Po2 dispatcher following your suggestion

The submitted shortest-queue baseline dispatches to the instance with the fewest outstanding
requests (running + waiting) — the closest reactive LLM-serving analogue of least-connections,
though it does not weight heterogeneous remaining decode work. Following your suggestion, we
added standard power-of-two-choices: two candidate replicas sampled uniformly without
replacement (seeded), dispatching to the one with fewer outstanding requests — the same load
signal as shortest-queue, so telemetry staleness is identical; with fewer than two replicas it
degenerates to full comparison. The updated Table 4, all cells on the submitted configuration:

**Table V2 — dispatcher comparison incl. Po2 (submitted system; e2e in ms):**

| system | dispatcher | λ6 | λ8 | λ10 | λ12 | λ18 | λ24 | λ30 |
|---|---|---|---|---|---|---|---|---|
| passthrough | SQ | 1947 | 1966 | 2017 | 2051 | 2105 | 2155 | 2182 |
| passthrough | Po2 | 2254 | 2238 | 2261 | 2257 | 2258 | 2272 | 2315 |
| passthrough | RR | — | — | — | 2670 | — | 2715 | 2734 |
| BR4 (argmax) | Po2 | 1911 | 1991 | 2175 | 2636 | 18273 | 48182 | 54112 |
| BR4 (argmax) | SQ | — | — | — | 2626/2971 (2 runs) | — | — | — |
| AP pw0.7 | Po2 | 2553 | 2592 | 2624 | 2665 | 2737 | 2818 | 2971 |
| AP pw0.7 | SQ | — | — | — | 2675 | — | — | — |
| RouteBalance (uniform) | joint assignment | 2080 | 2118 | 2175 | 2230 | 2298 | 2408 | 3284 |

*Verification (not posted): basis v1exact-v2 throughout (no mixed bases). Source cells:
v1x_{pt_sq,pt_rr,pt_po2}_l* + v1x_br4_{sq,po2}_l* + v1x_ap_{sq,po2}_l*; RouteBalance row = the
Table-V1 arm-1 uniform cells (same basis, same window); RB preset: default uniform. Gates:
routing-correctness mix-twins ≤2pp (hard gate) + degenerate halts; e2e recorded; both br4_sq
λ12 runs recorded; BR4 + AP routing decisions bit-exact vs the submitted campaign (0.00pp
both); AP calm-regime e2e within 1.3% of the submitted-campaign λ6 references (continuity evidence);
RR exists at λ{12,24,30} only (v1 coverage). Collapse-regime cells are high-variance (banked
means λ18/24/30 = 18.3/48.2/54.1 s; run-to-run spread evidenced by the two br4_sq λ12 runs,
2626/2971) — covered by the appendix note below.*

**Findings.** For the passthrough family, shortest-queue beats Po2 at every rate, and Po2
beats round-robin at each rate where round-robin was measured (λ{12,24,30}). The trained per-request router (BR4, argmax) reproduces the paper's
documented 12–18 req/s serial ceiling — the collapse knee between λ12 and λ18
(2636 → 18273 ms) — under Po2 exactly as under shortest-queue: the ceiling is a routing-side
property, independent of the dispatcher. RouteBalance remains stable across the same sweep
(2230 → 3284 ms) — 20.0x lower end-to-end latency than BR4+Po2 at λ24 and 16.5x at λ30. Structurally, a
dispatcher — however strong — places requests within the replica pool of a tier that has
already been selected; it cannot revisit the model choice itself.

> [INTERNAL — DO NOT POST] [BOTH-BASES CHECK]: no quality-ordering sentence versus
> dispatcher-only configurations is included until a fresh both-bases check against the
> v1exact ARGMAX numbers passes (per the dual-basis rule, which survives the void register).

**Appendix note (committed to the revision):** serial per-request routing collapses under load
with high run-to-run variance — a routing-side compute component (~2x under CPU co-tenancy;
~4% of end-to-end latency at low rate, amplified near saturation). Routing decisions are
bit-exact reproductions of the submitted runs (0.00 pp mix difference, both routers); the
effect is independent of the dispatcher, and it does not affect the key conclusions — the
high-load gaps exceed the observed run-to-run spread by more than an order of magnitude. It is largely absent for RouteBalance, whose
stability at high rate comes from amortized batch scoring.

### 3.2 Topology change (re-provisioning adaptation)

To test sensitivity to the physical deployment, we are re-provisioning the serving topology
within the same evaluation window (independent re-provisioning and replication of instances)
and re-running the comparison with zero retraining: the system re-normalizes over the new
candidate set and continues. We will post the comparison table as a follow-up in this
discussion window.

**Table V3 — topology flip:**

[TOPOLOGY-FLIP ROWS — banked RB+PT cells (25); BR4-argmax column lands ~21:00Z; fill before
posting]

*Verification (not posted): both topologies measured in this discussion window on the same
system; no May-adjacent numbers in this table; no version/build language (rail #8); RB preset:
default uniform. Claim wording rail: independent re-provisioning/replication — NEVER "always
new hosts".*

### 3.3 Single-model reference frame

To contextualize router quality, the fixed-basis single-model references on the test corpus
are: always-72B 0.4275; always-14B 0.3979; always-3B 0.3461; always-7B 0.3327; the
RouteBalance presets span 0.3967–0.4007. Always-72B attains the highest quality, as expected —
it serves every request on the largest tier, at correspondingly higher latency and cost. The
object of interest is therefore the quality–latency–cost frontier: RouteBalance is the policy
family that spans it within a single deployment, with presets trading quality against latency
and cost.

### 3.4 Generality

We are additionally validating on a second model family (Gemma-3 dense ladder) in this
discussion window and will post those results as a follow-up comment.

### 3.5 Robustness to partial or missing quality signals (your W2 and Limitations request)

We quantify exactly the gap you name, on the deployed estimator with the serving-exact replay
recipe: (a) a labeled-fraction ladder — the estimator rebuilt at 0/25/50/75% of quality labels
and evaluated end-to-end online; and (b) a missing-domain ablation — entire domains (math
[GSM8K] and code) held out of the estimator's corpus. We report degradation as deltas relative
to the 75%-coverage estimator built and evaluated in the same window, plus the quality slice
on prompts from the excluded domain.

**Table V4 — degradation under partial/missing quality signal (deltas vs the 75%-coverage
reference):**

| estimator corpus | Δ global quality vs 75%-coverage reference | Δ on excluded-domain prompts |
|---|---|---|
| 50% of labels | −0.002 | — |
| 25% of labels | +0.003 | — |
| 0% of labels (prompt-blind) | −0.013 | — |
| GSM8K domain held out | −0.002 (tie) | +0.002 |
| code domain held out | −0.005 | +0.001 |

Even the prompt-blind estimator — zero quality signal — moves global delivered quality by only
0.013; the labeled-fraction ladder stays within 0.005 of the reference from 25% coverage
up. Holding an entire domain out of the estimator's corpus moves global quality by at most
0.006 (the GSM8K holdout is a statistical tie under a paired bootstrap), while quality on the
excluded domain's own prompts stays within 0.002 of the reference: the estimator degrades
gracefully to its neighborhood prior rather than failing on unseen-domain traffic.

*Verification (not posted): internal source = the v1exact λ12 6-cell set (banked+audited);
display = relatives only per rail #7; no full-index anchor rows, no baseline rows; no
version/build language anywhere (rail #8).*

### 3.6 Cold-start / onboarding cost accounting (your W2; measured vs derived, labeled)

Onboarding a new model has two cost sides, and we account for both with measured quantities,
labeling anything derived.

*Quality-estimator side.* Onboarding one model requires scoring 18,608 (prompt, model) pairs.
Measured sustained judge throughput is ~24 calls/s (Llama-3.1-8B judge on 4×A100,
data-parallel), which implies ≈13 minutes per model column — a throughput-derived estimate for
judging only, excluding candidate generation, setup, I/O, and retries, which we report
separately. The similarity-index build itself takes 52.3 s (measured).

*Latency-estimator side.* Collecting the latency-training corpus (~405k profiled requests
across five hardware pairs) takes approximately four hours of parallel serving at the
cluster's measured throughput (a throughput-derived estimate; collection parallelizes across
instance types); per-tier model training then takes under a minute each.

*Two-stage bootstrap.* The four-arm result above licenses a practical onboarding path: a day-1
deployment needs only static priors (brief median probes at a moderate operating rate —
minutes), which our ablation shows are equivalent to the learned estimator at stationary rates
below capacity; the learned heads then train from accumulated serving telemetry (profiling is
serving plus logging) and add value under load and burst. Cold-start is thus a measured
two-stage cost, not a monolithic prerequisite. We are additionally measuring the exact
end-to-end onboarding wall-clock (generation → judging → index build) during a second-family
onboarding and will report it in this discussion.

### 3.7 Tuning burden (W4a)

Under a fixed workload and objective, the presets need no retuning as the arrival rate
changes: the tier mix remains stable from λ6 to λ30, replica-level state absorbs load
shifts, and the mix stays bounded under bursts (≤ +14%). Changes to workload composition or
to the objective itself may require retuning; an SLO-driven controller that adapts the
weights automatically is future work.

### 3.8 Budget mechanism (W4b)

We agree with your reading, and will state the scoping plainly in the revision: on the
budget axis, the contribution is the composable admission mechanism, which uses the
predicted output length to convert would-be truncations and budget exhaustions into
completed lower-tier responses. The simplex weighting alone does not enforce budgets. The
runtime exhaustion-rate evidence in the paper is retained (it is evaluator-independent).

---

## 4) AC comment (two paragraphs, AC-first ordering)

On the estimator justification (your first point): we isolate the XGBoost latency estimator's
contribution with a three-arm experiment on the submitted configuration (identical prompts and
seeds): the full objective; the objective without a cross-tier latency term (reactive
shortest-queue dispatch within the selected tier); and the full objective with the learned
head replaced by a static per-tier prior (nominal TPOT × predicted length, live within-tier
queue state retained). Under stationary arrivals below capacity the static
prior matches the learned estimator within ±0.8% end-to-end at statistically indistinguishable
quality — so we justify the design as a formulation contribution: the latency term must enter
model selection, and the learned head is explicitly not load-bearing for the headline results.
The learned head remains the deployment default because it removes the calibration bet a
static prior must make: per-token decode latency is load-dependent, and under bursty
capacity-adjacent arrivals the learned predictor delivers 15.3% lower end-to-end latency at
statistically unchanged quality. The isolation also shows the dominant effect is the
tier-selection mix shift itself: with the latency term removed the routing mix freezes
(identical at every rate, 69.1% of requests on the smallest tier) and global quality drops by
0.014–0.018, while the full objective adapts its mix with load (dominant-tier share
62.9% → 55.4% from λ6 to λ30) at stable quality (full tables in our reply to
Reviewer gTfM).

On general-purpose dispatchers (your second point): the submitted shortest-queue baseline
dispatches to the instance with the fewest outstanding requests (running + waiting) — the
closest reactive LLM-serving analogue of least-connections, though it does not weight
heterogeneous remaining decode work. Following the reviewer's suggestion we added standard
power-of-two-choices (two candidates sampled uniformly without replacement, seeded — the same
load signal as shortest-queue) and post an updated Table 4 in our reply to Reviewer faAU:
Po2 sits between shortest-queue and round-robin for the passthrough family at every rate; the
trained per-request router reproduces its documented serial ceiling under Po2 exactly as under
shortest-queue (the ceiling is dispatcher-independent), while RouteBalance remains stable
across the same sweep. The structural context for interpreting dispatcher results: a
dispatcher, however strong, operates within the replica pool of a tier the router has already
selected. Results on an additional model family (Gemma-3) will follow in this discussion
window.

---

> [INTERNAL — DO NOT POST] Assembly notes:
> 1. RESOLVED (Jul-27 revision): the mix-shift sentence IS carried (AC + 2.1) — arm-2
>    v1exact cells banked 14:32Z ground it (mix frozen 69.1/22.3/8.1/0.5 at all rates vs
>    arm-1 62.9→55.4 dominant-tier shift; quality gap 0.014–0.018 sign-stable both bases).
> 2. The k-sensitivity clause from the older faAU draft was dropped per the no-k-statements
>    rail; the prompt-blind lower bound is likewise not in the 07-27 frame and is omitted.
> 3. Budget-mechanism and tuning-burden survivor paragraphs (old faAU-W4a/W4b) exist in the
>    actions table but were not slotted — the task's section spec does not assign them; slot
>    on author instruction once the reviewer mapping is confirmed.
> 4. Silence rails in force for follow-ups: routing-score vintage (generic "LLM-judge quality
>    labels" only, truthful-if-asked) · APC setting (silent; scoped truthful-if-asked answer
>    banked) · Gemma-4 engine-consistency answer (prepared, never volunteered) · cal-test
>    recalibration cells (internal if-asked backup only).
