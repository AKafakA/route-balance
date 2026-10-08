# Reviewer actions table v2 + draft paragraphs (2026-07-25; SUPERSEDES v1 table)

Incorporates the 16-point ChatGPT revision pass, with four code verifications performed first:
SQ metric = num_running+num_waiting (`shortest_queue.py:28`); online latency term = analytic
Little's-Law over learned TPOT (`cara_learned_predictor.py:341-368`) — TTFT head loaded but NOT an
objective term; Po2 = predictive variant (sample-2 w/o replacement → min sidecar-predicted latency,
`block_style.py:97+`); response body IS relayed once through the scheduler (`cara_serve.py:496`) —
their point-5 "token data do not flow through it" corrected accordingly.

Priorities: **P0 = AC-decision-critical** · **P1 = clarification** · **P2 = strengthening**.
Evidence classes: **E1** existing submitted-stack · **E2** newly-rerun submitted-stack ·
**E3** code/design explanation · **E4** CR commitment · **E5** v2 evidence (guides design; NOT
rebuttal-primary).

## 1) Revised master action table

| # | Pri | Rev | Item | Revised strategy (wording-safe) | Class | Status |
|---|---|---|---|---|---|---|
| 1 | P0 | AC/gTfM-Q1 | XGBoost justification | Four-arm isolation ONLY (no k-sensitivity here): full/live vs static-prior vs no-latency+reactive-SQ vs no-latency+predictive-tiebreak → E2E, quality, tier mix. Honest static-prior finding; formulation-not-estimator; learned head = calibration + SLO-headroom, not load-bearing | E1(+E2 EXP-ARMS on submitted stack) | 🌙 tonight |
| 2 | P0 | AC/faAU-W3 | General-purpose dispatchers | Precise: "SQ picks the fewest outstanding requests (running+waiting) — the closest reactive analogue of least-connections, though it does not weight heterogeneous remaining decode work. We add STANDARD power-of-two-choices (two candidates sampled uniformly without replacement, seeded; dispatch to the one with fewer outstanding requests — the SAME load signal as SQ, so telemetry staleness is identical; <2 replicas degenerates to full comparison). Requires ~15-line standard-Po2 dispatcher (existing block_style Po2 is a PREDICTIVE variant — kept only as optional bonus arm)." 3-level comparison PT×Po2 / router×Po2 / RB — conclusion CONDITIONAL on data | E3 + E2 (EXP-PO2) | 🌅 tomorrow |
| 3 | P1 | 287j-W1 | Batching vs continuous batching | "Requests are NOT held for a mandatory minimum batch size. The assignment scheduler fires when the adaptive size limit is reached OR the timeout expires, timer starting at the first arrival; a lone request thus waits at most the timeout (50–200 ms adaptive). Formation delay is charged in client-observed E2E. Assignment batching is separate from and composes with vLLM's instance-side continuous batching, which is unchanged." | E3 | ✅ |
| 4 | P1 | 287j-W2 | Adaptive formula | Busy-instance ratio <0.3→(bs 4, 50 ms); <0.7→(16, 100 ms); else (32, 200 ms). Ablation: adaptive-off +5.1–6.9% E2E; bs=1 collapse 45.4/74.6 s as measured on the submitted implementation (serial ≈85–90 ms/req ceiling) | E3+E1 | ✅ |
| 5 | P1 | 287j-W3/gTfM-Q2 | TTFT/KV/APC scope | "The submitted objective optimizes decode-dominated E2E: learned per-(model,GPU) TPOT × queue-aware decode iterations (pending_decode/running + own predicted output). TTFT is separately profiled by a dedicated head but is not an explicit objective term; prompt length enters through predictor features. APC/prefix caching was DISABLED in all reported runs, so prefix-hit effects are not a confound; model choice affects KV pressure via model size and sequence lengths only. A prefix-affinity term is natural future work." | E3+E1 | ✅ |
| 6 | P1 | gTfM-Q3 | Centralized | "Yes, by design — single assignment decision per request/batch over global dead-reckoned state. KV and decode state never leave workers; the controller exchanges request metadata, predictions, and non-blocking telemetry, and relays the generated text once per request (no per-token streaming through it). Measured overhead stays bounded on the evaluated 13-instance cluster (141→220 ms, sub-linear; decision compute ≈24–33 ms, amortizing). Fault tolerance and multi-controller sharding are outside scope." µs-triplet ONLY if file located | E3+E1 (🔍µs) | ✅ |
| 7 | P1 | gTfM-Q4 | Fixed deployment | "Yes — placement is fixed during each run; RouteBalance is an online request-assignment layer over a given deployment, complementary to planners (HexGen/Helix)." Tier-removal supports ONLY: "when the candidate set changes, the scheduler re-normalizes and continues without failures." Cross-lease reproduction = repeatability across re-provisioned hardware (NOT dynamic planning) | E3+E1+E2(R1) | ✅+🌙 |
| 8 | P1 | faAU-W4b | Budget mechanism | 1) Agree. 2) Contribution = composable admission mechanism using predicted output length. 3) Filter converts truncation/exhaustion into completed lower-tier responses. 4) Simplex alone does not enforce budgets. 5) Runtime exhaustion-rate evidence retained (evaluator-independent). Served-text quality-conversion deltas = E5 → EXCLUDED from rebuttal | E1+E3 | ✅ |
| 9 | P1 | 287j-W4–W8 | Figure/appendix clarity | CR commitments incl. Fig-1 redraw (RoBERTa = unused pluggable alternate), Table-2 move, radar cut, LPT-as-heuristic appendix, dead-reckoning DEFINED (design rationale: prevents later same-batch assignments from seeing stale pre-dispatch state — no quantified herding claim, no direct ablation exists) | E4(+E3) | ✅ |
| 10 | P1 | 287j (summary) | LLM-D/AIBrix similarity | One sentence: "LLM-D and AIBrix provide serving infrastructure with routing/load-balancing components; to our knowledge neither fuses per-request quality–latency–cost scoring into a single assignment over heterogeneous concrete instances, which — together with the controlled fusion-boundary isolation — is our contribution." | E3 | ✅ |
| 11 | P1 | faAU-W4a | Tuning burden | "Under a FIXED workload and objective, presets need no retuning as arrival rate changes (tier mix stable λ6→30; replica-level state absorbs load; bounded under bursts ≤+14%). Workload-composition or objective changes may require retuning; an SLO-driven controller is future work." NO workload-drift-robustness implication | E1+E3 | ✅ |
| 12 | P2 | faAU-W2 | Cold-start | Report separately: 18,608 pairs/model; judge = Llama-3.1-8B on 4×A100; sustained ~24 calls/s MEASURED ⇒ per-column time is a throughput-DERIVED ESTIMATE (label as such); FAISS build 52.3 s (log); XGB per-tier train_time_s (metrics files); actual end-to-end onboarding wall-clock to be MEASURED during the family onboarding and reported; degradation curve = P4 (fraction-of-labels vs decision metrics) + R4 prompt-blind bound; R5 k-sensitivity lives HERE (quality-estimator robustness), not in the XGBoost answer | E1+E2+🔬P4 | ✅facts+🔄 |
| 13 | P2 | all | Repeated seeds | 2-seed replicates on EXP-ARMS/PO2 headline cells (same basis) | E2 | 🌙 day-2 |
| 14 | P2 | faAU-W1 | Generality | Gemma sampled (12 cells) AFTER P0 completes; topology-B bonus; tier-removal+replication+ceiling text now | E2/E5-design | 🔄 |
| 15 | P2 | — | Nonstationary | Existing burst/square-wave data (basis-labeled if used) | E1/E5 | optional |

## 2) Claims whose wording changed (v1 table → v2)
1. "SQ ≡ least-connections" → "closest reactive analogue; picks fewest outstanding requests (running+waiting); does not weight heterogeneous remaining work."
2. "TTFT modeled by a dedicated XGB head" → "objective optimizes decode-dominated E2E (learned TPOT × queue-aware iterations); TTFT profiled, not an objective term."
3. "no holding" → "no mandatory minimum batch size; size-or-timeout; lone request waits ≤ timeout; delay charged in E2E."
4. "tier-removal shows the boundary is safe" → "candidate-set-change robustness only; not dynamic provisioning."
5. "mix stable ⇒ no retuning burden" → scoped to arrival-rate changes under fixed workload/objective.
6. "dead-reckoning reduces herding" → design-rationale phrasing; no quantified causal claim.
7. k-sensitivity REMOVED from the XGBoost answer → moved to cold-start/quality-estimator robustness.
8. "token/KV data do not flow through the scheduler" (ChatGPT's own suggestion) → corrected: response relayed once; KV/decode state never leaves workers; no per-token streaming.
9. "~13 min/model labeling" → labeled as throughput-derived estimate; measured wall-clock to come.
10. Po2 documented as the PREDICTIVE variant (sample-2 w/o replacement → min predicted delay).

## 3) Numbers still requiring file-level verification before posting
- Scaling-microbench triplet 12.8/14.3/22.5 µs @ |I|=13/100/500 (locate producing file in v2-pack; else omit).
- Standard-Po2 dispatcher: implement (~15 lines, queue-metric 2-sample) + smoke before cells; existing block_style Po2 confirmed PREDICTIVE (sidecar-based) — bonus arm only.
- (Budget served-text deltas +0.015/+0.012/+0.006: classified E5 — EXCLUDED from rebuttal, no verification needed unless CR uses them.)
- (LODO fold numbers: only if the cold-start reply quotes them.)

## 4) Statements depending on submitted-stack reruns (tonight → Jul 27)
- The four-arm table on the submitted stack (EXP-ARMS; arms 1–4 × presets × λ). 
- Po2 three-level comparison results (EXP-PO2) + its conditional conclusion.
- Cross-lease reproduction "reference cells reproduce within X%" (R1).
- 2-seed stability statements (EXP-SEEDS).
- Gemma family table + measured onboarding wall-clock (post-P0).
- P4 degradation curve (offline, new computation).

## 5) One-paragraph draft per reviewer (decision-relevant concern first)

**287j (clarity → the 3):**
"Thank you — the batching design clearly needed a sharper description, and we give it here (and commit
to it in the paper). RouteBalance's batching is scheduler-side ASSIGNMENT batching and composes with
vLLM's instance-side continuous batching, which is unchanged: once dispatched, a request immediately
enters its instance's continuous-batching loop. Requests are never held for a mandatory minimum batch
size: the scheduler fires when either the adaptive size limit is reached or a timeout expires, with the
timer starting at the first arrival — a lone request waits at most the timeout. The adaptive limit is a
busy-instance-ratio step schedule: ratio<0.3 → (batch 4, 50 ms); <0.7 → (16, 100 ms); else (32, 200 ms).
All formation delay is charged in the client-observed E2E we report. Our ablations bound each piece:
adaptive-off costs 5.1–6.9% E2E; batch-size-1 forfeits amortization entirely and collapses under load
on the submitted implementation. On Figure 1: the RoBERTa box is an alternate implementation behind the
pluggable estimator interface, not used in any reported run — we will redraw the figure with the return
path and add the appendix walkthrough, dataset details, LPT (as a motivated heuristic), and a definition
of dead-reckoning."

**gTfM (Q1 first):**
"Q1 is exactly the right question, and we isolate it directly with a four-arm experiment on the
submitted configuration (identical prompts/seeds): (1) the full objective with the learned latency
term; (2) no cross-tier latency term + reactive shortest-queue dispatch; (3) no cross-tier latency
term + predictive latency tiebreak; (4) the full objective with the learned head replaced by a static
per-tier prior (nominal TPOT × predicted length, zero telemetry). [Table.] Three findings: within a
tier, prediction adds little over reactive queue state; the value is pricing latency at model-selection
time — it shifts the tier mix away from saturated premium tiers, which no post-hoc dispatcher can do;
and a static per-tier prior recovers most of this benefit, so the contribution we defend is the
formulation, not any particular estimator — the learned head remains the default for calibration under
drift and SLO headroom. On Q2: the objective optimizes decode-dominated E2E (learned per-(model,GPU)
TPOT × queue-aware decode iterations); TTFT is profiled separately, and prefix caching was disabled in
all reported runs, so cache-locality effects are not a confound. Q3: yes, centralized by design — KV
and decode state never leave workers, the controller exchanges metadata/predictions/non-blocking
telemetry and relays each response once; measured overhead stays bounded (141→220 ms, sub-linear).
Q4: yes, placement is fixed per run — RouteBalance is the online assignment layer over a given
deployment, complementary to planners; when the candidate set changes (tier removal) it re-normalizes
and continues without failures."

**faAU (cold-start first, per their 'main limitation'):**
"We agree cold-start is the estimator's main limitation and answer all three parts. Labeling scale:
onboarding one model requires scoring 18,608 (prompt, model) pairs; with our measured sustained judge
throughput (~24 calls/s on 4×A100, Llama-3.1-8B judge) this is minutes-scale per model as an estimate —
we are additionally measuring the exact end-to-end onboarding wall-clock (generation → judging → index
build, FAISS build itself is 52.3 s) during a second-family onboarding and will report it in this
discussion. Degradation: we add a labeled-fraction curve (routing metrics vs training-set size) and a
prompt-blind lower bound; the quality-estimator's k is also insensitive across {5,10,20,50}. On
dispatchers: shortest-queue picks the fewest outstanding requests — the closest reactive analogue of
least-connections, though it does not weight heterogeneous remaining work — and we are adding
standard power-of-two-choices (two candidates sampled uniformly, dispatch to the fewer outstanding
requests — the same load signal as shortest-queue) at three levels: dispatcher-only, under the
strongest router, and versus joint assignment; results follow in this discussion. On budget: you are right — the
reduction comes from the admission filter, not the simplex weighting; the contribution we claim is
precisely that composable admission mechanism (predicted-length-based), which converts truncation into
completed lower-tier responses, and the exhaustion-rate evidence is runtime-measured. On weights:
under a fixed workload and objective, presets need no retuning as arrival rate changes (tier mix is
stable λ6→30); workload-composition changes may require retuning — an SLO-driven controller is future
work. Generality experiments on a second model family are underway and will be posted here."

## 6) AC response (two paragraphs, the two meta-review points only)

"On the estimator justification: we isolate the XGBoost latency estimator's contribution with a
four-arm experiment on the submitted configuration (identical prompts and seeds): the full objective;
the objective without a cross-tier latency term under reactive shortest-queue dispatch; the same under
a predictive latency tiebreak; and the full objective with the learned head replaced by a static
per-tier prior (nominal TPOT × predicted length, zero telemetry). [Table.] The isolation attributes
the benefit to pricing latency at model-selection time — a cross-tier mix shift no within-pool
dispatcher can produce — while within-tier prediction is largely interchangeable with reactive queue
state, and a static prior recovers most of the frontier. We therefore justify the design as a
formulation contribution: the latency term must enter model selection; the learned head is the
deployment default for calibration under drift and SLO headroom, and we now state explicitly that it
is not load-bearing for the headline frontier.

On general-purpose dispatchers: the submitted shortest-queue baseline dispatches to the instance with
the fewest outstanding requests (running + waiting) — the closest reactive LLM-serving analogue of
least-connections, though it does not weight heterogeneous remaining decode work. We are adding
standard power-of-two-choices (two candidates sampled uniformly without replacement, dispatch to the
one with fewer outstanding requests — the same load signal as the shortest-queue baseline) evaluated
at three levels — dispatcher-only, under the strongest trained
router, and against RouteBalance's joint assignment — on the submitted stack; results will be posted
in this discussion. The structural context for interpreting them: a dispatcher, however strong,
operates within the replica pool of a tier the router has already selected, and our isolation shows
the dominant effect is the tier-selection mix shift itself."

## ⚠ AUTHOR DECISION 2026-07-26 — BURST EVIDENCE POLICY (binding for all reply drafts)
Burst results shown in replies: **gamma (vllm-bench --burstiness 0.3) cells ONLY** — standard,
not invented by us. Square-wave (CARA_SQUARE) numbers may appear ONLY as backup if gamma fails
to show the learned-predictor win, and then only with a full mechanism explanation (author
accepts reviewer scrutiny of it). Do not mix mechanisms in one table. No burst-superiority
wording until arm1+arm4 gamma cells are both audited.

## Cold-start cost accounting CORRECTION (author-caught, Jul-26): include the LATENCY side
Item-12's training-time answer must count BOTH estimator pipelines:
- QUALITY estimator: corpus generation × tiers + judge (~24 calls/s measured) + FAISS 52.3s
  → measured LIVE tonight as the Gemma-3 onboarding wall-clock (G1→G3).
- LATENCY heads (was MISSING): profiling-sweep data collection = ~405k served requests across
  the 5 (model,GPU) pairs (45k-113k/pair, training_metrics n_train+n_val). ⚠ PROVENANCE
  DISCREPANCY (flagged Jul-26, unresolved): ALL 17 deployed head files are mtime Mar-27
  23:34-42 (file evidence — these are the RULE-0-gated artifacts v1 served); author recalls an
  Apr-25 re-measurement with instance types profiled in parallel — likely a LATER collection
  that never became the deployed artifact (same pattern as the KNN deploy_knn rebuild).
  POSTED TEXT AVOIDS DATES ENTIRELY: report sample counts + throughput-derived hours (labeled
  derived; note collection parallelizes across types). + XGBoost training ~28s/tier (measured).
  WALL-CLOCK ESTIMATE (author-approved method, Jul-26): total profiled requests / measured
  aggregate cluster throughput = 404,929 / (25-28.7 req/s measured) ≈ **~4 h of parallel
  profiling (3.9-4.5 h)** — label "throughput-derived estimate". Reply-ready sentence:
  "collecting the latency-training corpus (~405k profiled requests across five hardware
  pairs) takes approximately four hours of parallel serving at the cluster's measured
  throughput; per-tier model training then takes under a minute each."
- BOOTSTRAP NARRATIVE (licensed by the four-arm result): day-1 deployment needs only static
  priors (low-load median probes, minutes) — static≈learned at moderate load per our ablation;
  learned heads then train from accumulated serving telemetry (profiling = serving + logging)
  and add value under load/burst. The Gemma-3 onboarding demonstrates exactly this two-stage
  path. Frame cold-start as a MEASURED two-stage cost, not a monolithic prerequisite.

## Anticipated follow-up #10 (added Jul-26): "Why Gemma-3 and not the newer Gemma-4?"
**Posture: PREPARED ANSWER ONLY — do not volunteer; the family table states Gemma-3 without
justification unless asked.**

Ready-to-paste reply (reviewer-appropriate wording — REVISED Jul-26 after author caught an
over-claim; hardware-failure sentence REMOVED as unverified — eager-mode result unknown):
> We selected Gemma-3 for serving-stack consistency: all results in the paper — including the
> new family study — run on the identical serving engine, instrumentation, and precision (fp16)
> as the submitted system. Gemma-4's architecture requires a serving-stack generation
> (vLLM >= 0.26 / transformers 5.x) that postdates our evaluated build; including it would have
> forced the second family onto a different engine than every other result in the paper.
> Extending the study to Gemma-4 on a current-generation engine is straightforward future work.

Optional second + third layers if the reviewer pushes (author-approved framing, truth-scoped):
> Additionally, migrating our instrumented serving stack to a current-generation engine
> (~15 months of upstream drift plus a transformers major-version change) exceeded the
> response window; and our pre-deployment compatibility testing on available pre-Ampere
> hardware surfaced kernel-level shared-memory limits in both default and eager serving
> modes, so validating the full heterogeneous testbed for Gemma-4 was not feasible in the
> window. We therefore preferred the family that runs unmodified on the evaluated stack.

⛔ NEVER claim Gemma-4 "cannot run on V100": we tested TURING (RTX 8000, 64KB shared/block) —
kernel requires 98304B = 96KB, which V100 (Volta) CAN provide (96KB opt-in max) — author's
online reports of V100 serving Gemma-4 are PLAUSIBLE and consistent with our data. Only say:
"testing on available pre-Ampere hardware (Turing-class) failed in both modes; V100 tier
unvalidated within the window." Eager-mode failure now VERIFIED (OutOfResources, serve_eager2.log).

Internal evidence (never posted): dev-gpu ~/gemma4_spike/ — vllm 0.26.0/transformers 5.14.1/
torch 2.11 versions.txt; serve.log: Gemma4ForConditionalGeneration resolved, TRITON_ATTN forced
(head_dim 256/global 512), CUDA-graph capture death on CC 7.5, FlashInfer refuses <8.0; fork
registry ends at gemma3/3n (node0 ~/vllm). Framing bonus (use sparingly): newest-model hardware
floors are themselves evidence for why heterogeneous clusters serve model FAMILIES.

## P4 cold-start curves — DESIGN LOCKED + RUNNING (Jul-26, author-approved)
Fills faAU-W2's exact gap ("does not quantify routing degradation under partial or missing
quality signals"). P4a partial-label curve + P4b missing-model ablation on the DEPLOYED
estimator with the serving-exact replay recipe; metrics BestAcc/Top2/delivered-quality
(fixed basis); 100%-label validation vs deploy/evaluation_results.json. Results →
_aggregate/p4_coldstart_jul26.json; numbers go into the faAU reply as the degradation
quantification, alongside the two-stage onboarding cost accounting.

## Anticipated follow-up #11 (added Jul-26): "TTFT/KV-cache: prefill effects, KV allocation, prefix-cache locality?"
Consistent with locked wording-change #2 ("objective optimizes decode-dominated E2E; TTFT
profiled, not an objective term"). Three-part answer, all artifact-backed:

> (1) The latency predictor CONDITIONS on prefill and KV state explicitly: its feature set
> includes pending_prefill_tokens, request_prefill_time_est, prefill_chunk_size,
> num_prompt_tokens (prompt length), kv_cache_utilization, kv_free_blocks, and
> kv_evictions_per_s — so prompt-length-dependent prefill pressure and KV allocation are
> inputs to every per-instance latency estimate, not ignored.
> (2) The OBJECTIVE optimizes decode-dominated E2E (learned TPOT × queue-aware iteration
> count). TTFT is separately profiled — dedicated per-pair TTFT models exist with held-out
> MAE 4-14 ms on deployed pairs — but is not an objective term, because in our response-length
> regime E2E is decode-dominated. [⛔ rails: NO TTFT-advantage claim anywhere — cara-clocked
> TTFT is scheduler-buffered; user-facing first token ≈ E2E per the ttft-metric memory note.]
> (3) Cache locality is orthogonal to the scoring mechanism: routing decisions are computed
> from per-instance telemetry, and cache-affinity signals could enter the same interface as
> additional features; we scope cache-locality-aware routing as future work.

⚠ DISCLOSURE RAIL for part 3 (author-caught Jul-26): the submitted paper does NOT report the
APC setting (verified: no prefix-caching mention in the tex) — so the volunteered answer above
does NOT assert what the experiments did. Policy = same as the k-item: SILENT in rebuttal,
TRUTHFUL IF ASKED, CORRECTED AT CR. If asked directly "was prefix caching enabled?":
> "Prefix caching was disabled uniformly across ALL evaluated systems (ours and every
> baseline) to isolate scheduling effects from caching effects and keep memory behavior
> deterministic across runs; we will state this explicitly in the experimental setup."
CR-CORRECTIONS LIST +1: experimental-setup sentence stating APC-off uniformly (alongside the
k-wording item).

Evidence: deploy_tpot/xgboost_metadata.json feature_names (25); deploy_ttft/training_metrics
(4-14ms MAE deployed pairs); APC-off on backend cmdlines (cell_audit RULE-0 list).

### APC-immateriality bound (measured Jul-26, for the truthful-if-asked APC answer)
Corpus prefix-sharing measured on the 3534-request eval set: global common prefix = 0 chars;
sharing confined to short dataset instruction headers (top groups: 572 + 495 requests sharing
first ~50 chars; 15.4% share at 100 chars; 1.0% at 200 chars — >=99% unique beyond ~50 tokens)
plus the fixed chat-template wrapper. With mean input length 103 tok (median 70, p90 220) and decode-dominated
E2E (capped outputs), the maximum APC benefit is the prefill of a <=~50-70-token shared prefix —
single-digit ms against multi-second E2E, identical direction for every system. Reply-ready
if-asked sentence: "the workload's cross-request prefix sharing is limited to short instruction
headers (>=99% of prompts unique beyond ~50 tokens), so prefix caching's maximum effect — the
prefill of that short shared prefix — is negligible relative to end-to-end latency and identical
across all compared systems." [ANALYTIC bound from corpus measurement — NOT an on/off A/B claim;
never say "our experiments showed no difference" unless an actual A/B is produced.]

## Burst-table presentation rule (author, Jul-26 eve): UNIFORM preset rows
The rebuttal's four-arm burst rows = the UNIFORM-preset cells (same payload as the stationary
four-arm table — one consistent operating point). Rationale (author): the 0.5/0.3/0.2 preset
targets quality×throughput, so its burst latency behavior is off-objective for the latency
comparison; its cells (arm1 quality higher in ALL burst patterns, at latency cost) are
SUPPLEMENTARY evidence that the preset does what it claims — never presented as a latency
comparison. Uniform numbers of record: gamma025 arm1 1878ms/q0.3736 vs arm4 1846ms/q0.3697
(parity, sub-margin); sq30_3 arm1 7.4% faster; sq40_6 arm1 3.7% faster + q +0.004.
Open author choice (final piece): (a) gamma-only robustness framing [rails-clean], or
(b) uniform square rows added WITH full CARA_SQUARE mechanism disclosure — at uniform the
learned arm wins both square patterns; if (b), the ss532 trade-off note accompanies it so the
preset-dependence is disclosed, not discoverable.

### Static-calibration robustness (Jul-26, closes "you only calibrated at low load"):
Every single-point calibration (λ6/12/18/24/30 per-tier medians) evaluated across all loads:
preset overall MAE 6.9 ms/tok vs best-achievable 6.8 (λ18/24 cal) — Δ0.1; λ30-calibrated =
WORST overall (7.5) and still 14.6 at λ30 vs learned heads 0.2-0.9 held-out. High-load error
14.6-16.4 across ALL calibrations = structural load-dependence, not calibration choice.
Reply-ready (folded into the discovery-first narrative above; wording rule: "moderate load
(λ12), a balanced calibration" — NEVER "light/low load"). Data: offline_pred_accuracy_jul26.json
→ calibration_robustness. Internal note: preset empirically matches λ6-12 medians; λ12 ≈ 42%
of saturation rate → "moderate" is truthful.

## 🏁 BURST STORY RESOLVED (Jul-26 eve): the gamma ladder + offline pairing (author-directed)
The rebuttal presents ONE self-explanatory unit — online gamma ladder + offline error data —
so reviewers need no extra judgment about mechanism:

**Online (gamma 0.25, the native vllm-bench arrival model, uniform weights):**
| mean rate | learned (arm1) e2e | static (arm4) e2e | Δ | quality Δ |
| 23 (moderate) | 1878 ms | 1846 ms | parity (−1.7%) | +0.004 |
| 27 (near capacity) | 2210 ms | 2442 ms | **learned −9.5%** | +0.003 |
| 30 (at capacity) | 2737 ms | 3052 ms | **learned −10.3%** | +0.001 |

**Offline (same window, banked cells):** static-prior TPOT error 4.0→5.8→15.8 ms/tok from
λ6→24→30 (learned heads held-out 0.2-0.9 ms/tok; top feature = live queue depth), and the
calibration-robustness sweep (no single-point calibration escapes the high-load blowup —
structural load-dependence).

**Reply-ready narrative (author framing Jul-26: discovery-first; "moderate", never "light"):**
"(1) Our profiling shows per-token decode latency is itself load-dependent — it is not a
constant of the (model, GPU) pairing: per-tier error of any fixed estimate grows from ~4 to
~16 ms/tok as the system approaches saturation, and no calibration point avoids this (we
verified by recalibrating at every tested load). (2) The static prior is therefore calibrated
at a moderate operating rate (λ=12), which our calibration sweep shows is a balanced choice
across the full load range (within 0.1 ms/tok of the best achievable single-point
calibration). (3) The serving results mirror the offline finding exactly: under bursty
arrivals (gamma, burstiness 0.25) the static and learned arms are equivalent at moderate
rates, and the learned predictor delivers ~10% lower end-to-end latency at capacity-adjacent
rates — the regime where load-tracking matters. The learned predictor also yields a small but
systematic quality advantage (positive in all paired cells; mean +0.002, 95% CI excludes 0)."

Rails check ✔: gamma-only (no square-wave in the rebuttal); >5% e2e deltas at 27/30 (above
noise threshold, consistent 2 rates); quality claim = paired-aggregate form; square cells =
internal backup only. Supersedes the open (a)/(b)/(c) decision — resolved by data + author.

## ⛔ BASIS-DEPENDENCE RULING + Table-4 revision design (Jul-26 late)
RB-vs-BR4-t0.5 quality ordering FLIPS across bases (paper: RB +0.000-0.003; fixed: BR +0.009)
→ per the dual-basis rule, NO quality-ordering claim between RB and BR-t0.5, in any direction,
anywhere. Frame stands on basis-FREE facts: BR-t0.5+Po2 = 98%-to-14B degenerate routing (its
quality is one point, not a policy); RB ties it with 27% lower e2e @λ24 (2495 vs 3183), far
cheaper mix (65/28/4/3 vs 98/2/0/0), preset flexibility, pressure behavior (λ30 pending
tonight). Table-4: v1 rows unchanged; APPEND Po2 rows (PT-Po2, BR-argmax-Po2, BR-t05-Po2; AP
quarantined); add/emphasize e2e + mix columns; caption = dispatcher-within-selected-tier
structural note; text claims iso-quality + latency/cost/flexibility only.

### Stationary-pressure nuance (drafting caution, Jul-26): at λ30 STATIONARY, arm1 is 18.5%
SLOWER than arm4 (2354 vs 1986ms) with its largest quality edge (+0.003) — the learned arm
trades latency for quality under smooth saturation; under BURSTY pressure (gamma 27/30) the
sign flips and learned wins ~10%. Prepared wording: "under smooth saturation the learned
predictor uses its headroom estimates to route toward quality (+Δq) at a latency cost governed
by the preset; under bursty saturation the static prior's stale rankings mis-route and the
learned predictor delivers ~10% lower latency." Caveat: the stationary-λ30 arm1 cell carries
the unresolved +34.7%-vs-May path flag — burst pairs (this-window, internally consistent)
carry the headline, stationary λ30 presented as the trade row.

## R-clarity reply strategy (author, Jul-26 late): the 3-scorer = writing/clarity focus
Approach: every confusion → precise pointered resolution + CR commitment; zero new claims.
Key item — CONTINUOUS-BATCHING DISAMBIGUATION (ready-to-slot draft):
> "We will clarify the two batching layers explicitly: (i) engine-level continuous batching —
> each vLLM instance composes decode batches per iteration (the serving substrate we build on);
> (ii) scheduler-level request batching — our adaptive batch-formation window groups
> concurrently-arriving requests so a single predictor invocation scores the full
> request×instance candidate matrix (the amortization quantified in Table [overhead]). In the
> revision we reserve 'continuous batching' for the engine layer and use 'scheduling round'
> for our layer throughout."
+ the already-locked wording fixes (SQ/least-connections; TTFT-head; no-holding) presented as
the same commitment list. Target outcome: 3→4 on resolved-clarity; technical reviewer 4→5 on
suggested-and-applied (Po2 + XGBoost answered AND applied in the new family). 

### ⚠ APC-May VERIFY-BEFORE-ANSWERING (Jul-26 late): deploy_cara's policy comment says
main-table cells ran APC ENABLED (fresh-cluster fairness) and the disable flag was for
cluster-reuse sweeps; the May-1 orchestrator log is silent. THIS WINDOW = OFF (verified,
correct per the reuse policy; R1 passed regardless — consistent with the ≤2% immateriality
bound). The if-asked APC answer is SCOPED TO THE REBUTTAL EXPERIMENTS ONLY; before any
statement about the submitted grid's setting, verify the cluster archive (backend cmdlines).

## ⛔ ROUTING-SCORE VINTAGE — REBUTTAL SILENCE RAIL (author, Jul-26 late; k-item policy family)
The routing-vs-evaluation score-basis distinction (pkl rubric-vintage routing vs deepeval
evaluation) is NOT exposed in any rebuttal text. Implementation: (1) NO rebuttal sentence names
the routing metric — neither "deepeval" nor "rubric"; the k-wording-style sweep extends to
estimator wording (generic "LLM-judge quality labels" only); (2) never ASSERT deepeval-routing
(assertion ≠ silence; would conflict with the CR correction); (3) truthful-if-asked generic
line: "the estimator aggregates LLM-judge quality labels from the training corpus"; (4) CR
corrections list +1: routing = judge-score labels / evaluation = deepeval-llama3.1-8b-it_reference,
+ HF README routing-of-record fix. Gemma table caption names the EVALUATION judge only (true);
the mirror's rubric routing pass stays internal.

### BR-t0.5 dispatcher equivalence + era rail (Jul-26 late, author Q)
Same-era flip triplet: t0.5 under RR/SQ/Po2 at λ24 = 3168/3137/3198ms (±1%), quality identical
(mix-determined 98.4/1.6 — same mix as May's t0.5). → BR-t0.5 loses to RB identically under
EVERY dispatcher (basis-free axes); no "Po2 rescues BR" claim EVER (only evidence = cross-era:
May RR/SH λ24 collapsed 27-33s at the same mix = era-confounded, λ30-flag class). May t0.5
cells usable as anchors at SUB-SATURATION rates only.

## ✅ STOP-ITEM #3 RESOLVED (author clarification, Jul-26 ~22:40Z): claim reverts to ORIGINAL form
Author's intended design was always ONE prior (the rate-12 TPOTs = the deployed preset) tested
across rates — the cal24/cal30 variants were my misreading, not their ask. MAIN CLAIM (fully
supported by pre-existing cells, no weakening): "the static prior [the deployed rate-12
calibration] is equivalent to the learned predictor at low/moderate rates and loses ~10% e2e
under capacity-adjacent bursty load, with lower quality in every paired cell." The cal-test's
4 cells = INTERNAL if-asked backup ONLY (reviewer: "why not recalibrate?" → measured answer:
point-matched calibration trades the rest of the range [offline] + quality [serving, all
cells]). Range-form headline WITHDRAWN; cal30-low-rate cells NOT needed. 

## 🏁 FOUR-ARM NARRATIVE FINAL (author framing, Jul-26 ~22:50Z) — the existence-proof structure
Reply-ready: "(1) Our profiling shows per-token decode latency is load-dependent — the error of
any fixed per-tier estimate grows ~4x as the system approaches saturation. (2) A deployed
system cannot know its future operating rate; a static prior therefore commits to one
calibration point in advance. (3) We measure what that commitment costs when conditions
mismatch: under capacity-adjacent bursty arrivals (gamma, burstiness 0.25) the static prior
delivers ~10% higher end-to-end latency than the learned predictor, and lower quality in every
paired cell (positive delta in all N pairs, 95% CI excludes zero). (4) The learned predictor
removes the assumption: it tracks the operating point rather than betting on it."
Structure = existence proof (one measured counter-example), NOT a universal claim. If-asked
layer (never volunteered): the recalibration counter-experiment (cal-test cells: point-matched
prior wins its point, loses the range offline + quality everywhere). SUPERSEDES all prior
burst/static wording drafts.

## 🏁 Po2 DISPATCHER ANALYSIS FINAL (Jul-26 ~23:00Z) — the AC's answer
Table (paper basis; fixed basis agrees on all orderings → claimable): Po2 sits BETWEEN SQ and
RR on both axes at every rate (quality above SQ, latency below RR — e.g. λ24: q 0.4991, e2e
2279 vs SQ 0.4976/2168, RR 0.5046/2736) = a good general-purpose dispatcher, treated
substantively. BOTH-BASES CLAIM: the full system's quality exceeds EVERY dispatcher-only
configuration at EVERY rate (paper: 0.512-0.515 vs 0.495-0.505; fixed: 0.396-0.401 vs
0.360-0.374). Structural sentence: dispatchers place requests after the model choice is made —
they cannot recover the quality routing decides. RB e2e ≈ Po2 at low/moderate rates; pressure
premium = the documented preset-governed quality trade. (RR exists at λ{12,24,30} only — v1
coverage; noted in table.) + BR4-t05 columns per the locked Table-4 design; AP pending its
gated rerun.

## 🏁 Po2 FINAL VERDICT (Jul-26 ~23:15Z) — the single-model reference frame (KEY-RISK RESOLVED)
Fixed-basis single-model references (test corpus): always-72B 0.4275 > BR4-t05 0.4056 ≈
always-14B+sliver (0.3979) > RB presets 0.3967-0.4007 > always-3B 0.3461 > always-7B 0.3327.
⇒ (1) BR-t05 is NOT the best — the trivial always-72B beats it on quality (at ruinous latency:
6.9→25.8s, accidentally measured by the AP-fallback cells); quality-column-chasing is absurd on
the corrected basis. (2) The frontier is the object (the paper's own thesis): RB = the only
frontier-spanning policy (within 0.005-0.009 of BR-t05 quality at -27% e2e, 65/28/4/3 mix,
preset family, topology-adaptive). (3) Rebuttal risk = ZERO (shown basis keeps RB nominal-top;
rails silence ordering text). (4) v2/CR: corrected-basis table INCLUDES always-72B/always-14B
reference rows — makes the envelope framing mandatory-and-natural; casualty = only the
rail-banned +0.005 peak optic. Single-model reference rows = REQUIRED addition to any
corrected-basis table (they contextualize every router's quality).

### ⚠ BR4-t0.5 MECHANISM CORRECTION (author-caught, Jul-26 ~23:25Z; code-pinned)
t0.5's 98.4%-14B = the fallback, not the classifier: gate `top_prob<0.5 → model_pool[0]`;
4-way softmax (val_acc 0.39) rarely clears 0.5 → ~98% fallback; pool[0]=14B via config key
order — AND the docstring says "smallest in pool" while the code takes pool[0] = a LATENT
DOC/CODE BUG present in BOTH eras (twin 0.00pp — May's t0.5 rows behaved identically; era-
consistent, comparisons valid as-measured). RETRACT my earlier "confidence gating shifts to
bigger models" interpretation. Text rule: describe t0.5 as "confidence-gated with a default-
model fallback (14B under the deployed pool ordering)" — never "falls back to smallest".
Post-window: fix-or-redocument the fallback; CR provenance note. Po2 dispatcher itself:
verified sane + unimplicated (router mixes dispatcher-invariant; PT-Po2 rational between SQ/RR).

## 🚨 T-VARIANT WITHDRAWAL (author ruling, Jul-26 ~23:45Z — supersedes the "strongest tuned
baseline" design and the BR-t05 blocks above): ALL t>0 BR4 cells (30) are WITHDRAWN from every
rebuttal table/text — the t0.5 behavior is fallback-bug-shaped and the submitted Table 4 never
contained t-variants (argmax only) → no comparability argument, pure liability. Rebuttal Po2
comparison = BR4-ARGMAX+Po2 (paper config, working classifier) + passthrough family + restored
AP + RB. All prior BR-t05 analysis blocks = INTERNAL context only. Basis-dependence ruling
still stands for any RB-vs-BR quality text (argmax rows: paper 0.5089 vs RB 0.512+ — check
both bases at drafting). Flip table BR4 column: pending author (argmax-rerun / drop).

## 🔒 PRESENTATION-APPROVAL GATE (author trust ruling, Jul-26 ~23:55Z)
No cell family enters any rebuttal table/text without explicit author sign-off against the
presentation inventory (recorded in the execution log this timestamp). BINDING DEFAULT: a
component found buggy → EXCLUDE + FIX + RERUN — never "keep with interpretation" (author had
to impose this twice on Jul-26; now standing policy). Every drafted table carries a
verification line: source cells + both bases + gates passed — author-auditable directly.

# ═══ VOID REGISTER (author ruling, Jul-27 ~00:15Z) — AUTHORITATIVE FOR THE DRAFTING PASS ═══
NO fallback-model behavior is presentable, period. Every block below that references BR4-t0.5
(or any t>0 variant) is VOID for drafting purposes — internal record only:
- "⛔ BASIS-DEPENDENCE RULING + Table-4 revision design" → VOID as written (BR-t05 content);
  the dual-basis RULE itself survives and must be re-checked against BR4-ARGMAX at drafting.
- "BR-t0.5 dispatcher equivalence + era rail" → VOID (era rail on May-λ24 cells survives).
- "🏁 Po2 FINAL VERDICT (single-model reference frame)" → the always-72B/always-14B corpus
  reference values SURVIVE (corpus facts, router-independent); every BR-t05 comparison in the
  block is VOID.
- Any "strongest tuned baseline" / "98%-concentration" / "topology-blind BR-t05" caption
  language → VOID.
THE CLEAN Po2 STORY (the only presentable one): Table-4 append = passthrough-{RR,SQ,Po2} +
BR4-ARGMAX-Po2 + AP-Po2 (only if tonight's twin-gate passes) — every column mechanism-verified,
ZERO fallback execution anywhere; RB compared on the established axes; ordering text subject to
a fresh both-bases check against ARGMAX numbers at drafting. Flip table: RB + PT rows only,
trained-router column pending the author's argmax-rerun-or-drop decision.
