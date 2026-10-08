# Third-Party Reviewer Notes — Astrolabe (SoCC-class revision)

**Reviewer role:** external systems-conference reviewer (SYSTOR/SoCC/EuroSys class)
**Paper:** *Astrolabe: Balancing Load in LLM Serving with Context, Knowledge, and Prediction-Guided Scheduling*
**Length:** 15 pp (12 body + refs)
**Revision focus flagged by the author:** Abstract, §1 final eval-preview paragraph, §6.3 (Request Latency Performance + GPU memory dynamics)

---

## 1. TL;DR for the author

The revision is a clear improvement over the previous "capacity-tied" framing. The new "15 of 17 QPS wins + 6× fewer preemptions + ~2.8× cheaper than fanout" triad is a defensible and venue-appropriate story. However, the paper still has several self-inflicted wounds that will draw reviewer fire:

1. The **capacity-tie** sentence is still placed *before* the capacity-dominant numbers in several spots, so a skim reader still finishes the abstract thinking "31.6 vs 31.5, why bother." The story ordering inside the abstract needs one more pass.
2. The **preemption-storm narrative** is backed by Fig. 7, but the prose in §6.3 reads as if the causal claim ("dispatch quality drives preemption storm") is proven by the same plot that just measures the correlation. Needs either one sentence of causal argument or a hedge.
3. The paper still carries **residual "Fanout default / N=12 default" phrasing and symbol drift** (k vs N vs Po2 vs Astrolabe-Po2 vs Block-Po2) from earlier drafts. Several of these are in §4.2, §6.5, and the intro.
4. The **A100 result (2.57× throughput, 36× mean latency, 123× per-token latency)** is the single most eye-catching number in the whole paper and is buried in §6.4. The abstract mentions 2.6× and 123×, but the intro bullet for A100 is weaker than it could be.
5. The **GPU-memory-variance plot (Fig. 7, middle row)** is a strong "tells a story" figure but is currently referenced as a throwaway ("cumulative preemptions"). The variance collapse under Po2 is more striking than the preemption count and deserves prose.

Detailed critique follows.

---

## 2. Evaluation of the Revised Positioning

### 2.1. Does the "15/17 + 6× preemption" framing land?

**Largely yes, with caveats.** This is a much better hook than the old capacity-centric story. "Lowest TTFT and E2E at 15 of 17 QPS points" is a concrete, falsifiable, reviewer-friendly claim that places Po2 in the Pareto frontier rather than the capacity frontier. The 30–38 % moderate-load gap narrowing to 7–12 % at overload is an honest trajectory that pre-empts the reviewer question "does the gap collapse at saturation?"

**Caveats / risks:**

- The phrasing *"15 of 17 tested QPS points"* implicitly reveals that there are **2 points where Po2 does not win**. The paper never tells the reviewer *which* 2 points Po2 loses, or by how much. A sceptical reviewer will assume the worst (e.g., "Po2 loses at the most interesting QPS = capacity"). Fig. 6 suggests the losses are at the extreme tail (QPS≈34–36) but the text should be explicit: **"the two points where Po2 does not hold the lowest mean are QPS=X and QPS=Y, where it is within Z% of the best."** Without that, the framing invites suspicion.
- "15 / 17" is an unusual way to summarise a continuous sweep. Reviewers may ask "what if you had tested 34 QPS points instead of 17?" — is the win-rate denominator a function of measurement granularity? Either cite the step size (0.1 QPS? paper says "integer QPS sweep" in §6.3 — clarify here) or report the fraction of the QPS *range* where Po2 leads, not the fraction of sampled points.
- "TTFT and E2E" lumps two metrics. It's slightly stronger to say "lowest mean TTFT *and* lowest mean E2E", making the joint claim explicit. Right now the abstract says "lowest mean TTFT and E2E latency among all six schedulers for 15 of 17 tested QPS levels". Is the 15 the intersection of the two metrics or the union? Clarify.

### 2.2. Is "capacity tied but dominant elsewhere" convincing?

**Improved, but still the weakest part of the pitch.** A SoCC/OSDI reviewer skimming the abstract reads:

> "tied with Llumnix- at the standard Llama-2-7B/ShareGPT setting (31.6 vs 31.5 QPS)"

… and their gut reaction is still "so it's not better on the headline benchmark." The revision tries to soften this with "but up to +12.6 % on Qwen2-7B (oracle lengths) and +9.4 % on tight batching," but:

1. The Qwen2-7B gain is partly oracle-length (not what you'd deploy). The reader has to parse "(oracle)" carefully. Concede or lead with the non-oracle Qwen2-7B result (Po2 = 73.9 QPS vs Llumnix 69.7 = +6.0 %, Table 2).
2. "Tight batching bs=24" is a configuration choice most readers won't have a mental model for. It looks like a cherry-picked knob. Re-phrase as "when the cluster is configured for low-latency (small batch) service, capacity increases +9.4 %."
3. The real story the paper wants to tell is: **"capacity is Pareto-equivalent; Po2 trades 0.1 QPS of peak capacity for 6× fewer preemptions, 30–38 % lower moderate-load latency, and ~2.8× less CPU."** That framing turns the tie from a weakness into a *design choice*. At present, the revision still subtly concedes ground.

### 2.3. Is the preemption-storm narrative supported?

**Mostly, but the causal story outruns the evidence.** §6.3 says:

> "Llumnix-'s memory-load heuristic cannot anticipate KV-cache pressure from long-running decodes; so overloaded instances preempt in storms that further imbalance the cluster; Po2's per-instance simulation of each request's effect on the target queue proactively routes around such nodes."

This is a *mechanistic* claim (the heuristic is *unable* to anticipate, and Po2's simulation is *because* it accounts for future decode). The evidence shown (Fig. 7) is a *correlational* plot: Llumnix has ~1800 preemptions at QPS=32, Po2 has ~300. That supports "Po2 preempts less" but not "Po2 preempts less *because* of simulation-based anticipation." Plausible alternative explanations a reviewer will ask about:

- Is the reduction simply because Po2 under-admits requests at QPS=32 (implied by the slight capacity tie — Po2 runs at 31.6 vs Llumnix 31.5, very close, but at QPS=32 *both are overloaded*)? If so, both see the same offered load, so this is not the explanation, but the paper should explicitly rule it out.
- Does the preemption gap persist at matched goodput, or only at matched offered QPS? The paper shows the latter.
- Could it be randomization (power-of-2 vs deterministic) rather than simulation? The §6.5 ablation (Fanout-est = 31.7 QPS) suggests randomization alone doesn't fully explain capacity, but the paper doesn't show a preemption-count-vs-N curve. **Add Fanout-est preemption count to Fig. 7, or note it in the caption.** If Fanout also has ~300 preemptions, the story is "simulation drives this"; if Fanout has ~1800 preemptions, the story is "randomization drives this" (more likely actually).

### 2.4. Un-exploited wins the author should pull forward

1. **GPU-memory variance collapse (Fig. 7, middle row).** Under QPS=28, Po2's `Free GPU Blocks Var` is an order of magnitude lower than Random/MinQPM/Llumnix-. This is a *direct* visualisation of load-balance quality and is arguably more persuasive to a scheduling audience than raw preemption counts. Currently it is mentioned only as "cross-instance variance" in §6.3. Pull it into the abstract ("~10× lower cross-instance memory variance") or at least the §6.3 lede.
2. **A100 throughput 2.57× / mean latency 40× / P99 per-token 135×** (§6.4, Fig. 8). The abstract says "2.6× throughput and 123× per-token latency" — but the more damning numbers are in §6.4 ("36× lower mean request latency, 123× lower per-token latency" for no-CP; **2.58×/40×/135× with CP enabled**). The 40× mean-latency number in particular is stunning and is nowhere in the abstract. Intro paragraph for A100 should name the specific QPS (QPS=36) and cluster.
3. **Burstiness (§6.7):** "6–9 % lower mean e2e under α=0.25 bursty" is a real and underused win. A production audience cares *a lot* about bursty traffic. The abstract mentions it ("6–9 % lower mean e2e under bursty arrivals — all at ~2.8× less per-predictor CPU than full fanout") but only as a subclause. Consider a dedicated bullet in the intro.
4. **Noise robustness (§6.8):** worst-case +5.1 % e2e at 100 %/100 % error is a *very* strong robustness result that de-risks the length-predictor dependency for reviewers. Currently buried in §6.8. Pull into the abstract as "robust to prediction noise (max +5.1 % e2e degradation under 100 % length and latency error)" — this directly addresses the reviewer concern "what happens if your predictor is wrong?"

---

## 3. Inconsistencies and Confusion Flagged

### 3.1. Numeric mismatches / unclear provenance

| Claim | Abstract | §6 / Figure |
|---|---|---|
| Capacity, Llumnix- | 31.5 QPS | §6.3: "31.6 QPS vs 31.5 QPS for Llumnix-". §6.5 Fig. 9: "Po2-est 31.6, Po4-est 31.9, Po8-est 31.7, Fanout-est 31.7". §6.6: "Po2 31.6→32.4" with oracle (+0.8). Abstract says "Po2 matches full-fanout capacity (31.6 vs 31.7 QPS at 10s TTFT-P99 SLO)". **Inconsistency:** §6.5 Fig. 9 says *Fanout-est 31.7* (matches abstract), but §1 says "Po2 matches Fanout … at our measurements" without a number. Fine, but make sure the 31.6 vs 31.7 in the abstract matches Fig. 9's bars. |
| Preemptions Llumnix @ QPS=32 | "~1,800" | §6.3: "~1,800 preemptions" ✔. §6.3 also says "saturates ~2,100" which is for QPS 33–36, not QPS=32. Both are in the abstract and they're hard to keep straight. **The abstract reads as if 1,800 and 2,100 are both at QPS=32; the paper body says 1,800 @ QPS=32, saturating at ~2,100 @ QPS 33–36.** Fix the abstract. |
| A100 wins | "up to 2.6× in throughput and 123× in per-token latency at saturation" | §6.4: with-CP "2.58×/40×/135×"; no-CP "2.57× throughput, 36× mean latency, 123× per-token". The abstract's "2.6× / 123×" picks the higher of each from *different* CP settings. Pick one (with-CP or no-CP) and report both numbers from the same configuration, or report both configs transparently. |
| Qwen2-7B gain | "+12.6 % (oracle)" | Table 2: Qwen2-7B column says Po2-oracle = 78.5 vs Llumnix 69.7 → +12.6 %. Non-oracle Po2 = 73.9 vs 69.7 → +6.0 %. Both are legitimate, but the abstract only cites 12.6 % (oracle). This is fine *if* also reported in the abstract: "**+12.6 % with oracle lengths, +6.0 % with predicted lengths**". |
| Tight batching gain | "+9.4 %" | Table 2 bs=24 column: Po2-oracle 29.1 vs Llumnix 26.6 → +9.4 %. Non-oracle entry for Po2 at bs=24 shown as "28.5" → +7.1 %. Same abstract selective-reporting concern. |

### 3.2. Symbol/term drift

- `k` vs `N`: §4.2 defines "power-of-k choices; k=2 by default, and k=N recovers full fanout", then §4.1 / §4.2 (after Algorithm 1) say "Astrolabe-Po2", and §6.5 switches to "power-of-N choices". **Unify to one symbol (recommend N)**. Algorithm 1 still uses "n (default 2)" — lowercase n, yet another symbol.
- `Llumnix-` vs `Llumnix`: §6 reserves `Llumnix-` for "improved heuristic dispatcher component without migration". On the A100 (§6.4) the comparator is called "Llumnix" (with migration). Good distinction but make sure every occurrence in the abstract/intro is explicit which one: abstract currently says "outperforms Llumnix by up to 2.6×" (A100) vs "30–38 % below Llumnix-" (A30). That is correct but reviewers will miss the hyphen difference — make the first sentence mention "full Llumnix" and the second "Llumnix- (the dispatcher-only heuristic)".
- `Astrolabe-Po2` vs `Po2` vs `Astrolabe-Po2-est` vs `Po2-est` vs `Po2-oracle`: five variant names. Fig. 6 uses "Astrolabe-Po2-est". Figs. 9, 10 use "Po2-est". Fig. 11 uses "Po2-est / Llumnix-". Abstract uses "Po2". Converge. Recommend: `Astrolabe` (system), `Astrolabe-Po2` (default config), `Astrolabe-PoN` (family), with `-est` / `-oracle` as suffix for length source. Drop "Block-Po2" if it still appears (I didn't catch it in the PDF, but it's in your project notes so worth a grep).
- `N=12 default` / `Fanout default`: the revision says "Po2 is the default", but the intro still contains phrasing like *"A naive way to exploit such predictions is to query every instance's predictor in parallel and pick the lowest-latency candidate (full fanout). This gives the best load-balancing quality..."* — the reviewer walks away thinking Fanout is your gold standard. Consider: "...the best *per-dispatch* load-balancing quality, but at prohibitive cost and with a herding pathology. Astrolabe therefore defaults to **N=2** (Po2)." **Make the default call out bold in §4.2.** Right now it's buried.

### 3.3. Broken / weak references, figure-caption mismatches

- **§3 references "[17, 61?]"** — literal question-mark placeholder at line "deploying multiple framework instances is typically necessary". Fix.
- **Figure 4 caption** says "Astrolabe Architecture" but the figure's *outer frame* still reads "Block System". Rename in the figure.
- **Figure 8 caption**: "Astrolabe-Po2 vs Llumnix on the A100 cluster" — but the legend shows three series ("Astrolabe-Po2 (CP)", "Astrolabe-Po2 (no CP)", "Llumnix"). Caption should reflect the CP vs no-CP distinction.
- **Figure 5** caption says "Latency Prediction Metrics" but the text in §6.2 calls it the simulator-accuracy / rank-1 selection figure, not prediction. Rename e.g. "Simulator accuracy and rank-1 selection rate".
- **Figure 6 bottom-right bar chart** ("Capacity @ SLO Po2 TTFT = 10s"): the bar labels show "29.3, 30.7, 29.2, 21.4, 31.5, 31.6" (visually from the PDF) but §6.3 prose says Po2 31.6 / Llumnix- 31.5 / Random/RR/MinQPM/INFaaS++ 21–23 QPS. The first three bars at 29–30 don't match the prose "21–23". Either the prose or the figure is wrong — check.
- **§6.3**: "INFaaS++ degrades rapidly past QPS=24, and Llumnix-'s prefill-length correction still lags the predictive variant" — "prefill-length correction" is the term §6.1 introduced for INFaaS++'s `prefillMemory` fix. You're applying it to Llumnix- here. Either rename the Llumnix- variant's load function here, or clarify "Llumnix-'s best-of-heuristics load function".
- **Table 1** reports Roberta regressor MAE=78.755 / MAPE=24.4 % vs Prompt-based LLM MAE=62, MAPE "not reported". **Acc-50=69.93 % vs 59 %**, Acc-100=77.15 % vs 81 %. The Roberta model does *worse* on MAE (78 vs 62) and on Acc-100 (77 vs 81) but better on Acc-50 (70 vs 59). The paper presents this as "comparable" (§6.2) but a careful reviewer will note it's slightly worse on 2 of 3 reported metrics. Either say "comparable accuracy at ~1000× lower inference cost" (quantify cost advantage) or acknowledge the MAE gap and explain why it doesn't matter for dispatch (Acc-X is what matters, which is already defensible).

### 3.4. Claims without a figure/table/§ reference

- **Abstract:** "reduces mean e2e by 6–9 % under bursty arrivals" — ✔ backed by §6.7 / Fig. 11.
- **Abstract:** "robust to injected prediction noise (worst-case +5.1 % e2e at 100 % length/latency error)" — ✔ §6.8 Fig. 12.
- **Abstract:** "A100 outperforms full Llumnix by up to 2.6× in throughput and 123× in per-token latency at saturation, confirming dispatch quality — *not migration* — drives the tail-latency reductions under load." — the "*not migration*" claim is the interesting one. §6.4 shows CP enabled vs disabled (isolates CP), but *not* Llumnix-with-migration-disabled vs Llumnix-with-migration-enabled. Technically the claim is that *your no-migration Po2 beats their with-migration Llumnix*, which is narrower than "migration is not what matters." Tighten the claim: "*dispatch quality is sufficient* — migration is not required."
- **§6.3:** "Astrolabe delivers the *lowest* mean TTFT and E2E latency among all six schedulers at 15 of 17 tested QPS points" — the 15/17 count is in abstract and §6.3 lede but not justified in the figure. Add a small inset table or one-sentence summary: "of the 17 integer QPS points swept (20–36), Po2 is strictly lowest at 15; at QPS=X Llumnix- is within Y % and at QPS=Z Random ties." This concreteness pays dividends.

---

## 4. Six Standard Review Dimensions

**Novelty (3/5).** The core idea — power-of-two randomized probing with per-instance simulation — is an incremental but *well-motivated* combination of two known primitives (Mitzenmacher [37] + Vidur [2]). The novelty claim "first to combine response-length estimation with per-instance simulation for online global dispatch" is defensible but narrow. The *herding* pathology (§4.2) is a genuine contribution — I have not seen it named in prior LLM-serving work — and is worth highlighting more. Overall, novelty is sufficient for SoCC/SYSTOR but would be borderline for OSDI/SOSP.

**Technical soundness (4/5).** Experimental design is careful: dual-cluster validation, oracle-vs-predicted ablation, noise injection, burstiness sweep, N-sweep, model/dataset generalisation. Simulator accuracy is validated at both 7B (A30) and 70B (A100, 16.6 % MAPE). One gap: the preemption-storm *causal* claim is correlational (see §2.3 above). Another: the "Llumnix-" heuristic is *your* reconstruction of Llumnix's dispatcher, not the original Llumnix-paper code; reviewers will ask whether the reconstruction is faithful, and you only cite it as "the improved heuristic dispatcher component of Llumnix [52]". One more sentence on the reconstruction fidelity (e.g., "we validated our Llumnix- implementation against [52]'s reported numbers on matched configs") would help.

**Clarity (3/5).** The paper is readable but has persistent symbol drift (N/k/n), inconsistent naming of the system (Astrolabe vs Po2 vs Astrolabe-Po2 vs Block System in Fig. 4), and some figures whose captions/prose don't match (Fig. 5, Fig. 6 bar chart). The Related Work section is thorough but sprawls. §4 and §5 overlap (both describe the Predictor service). §6 is well-structured with clear subsections.

**Significance (4/5).** The problem is real and timely (multi-instance LLM serving is the bottleneck for production deployments). The A100 result — a 2.6×/123× gap versus the current state-of-the-art (Llumnix) on a realistic setup — is genuinely significant *if* the reviewer accepts the "network-constrained commodity cluster" framing. Some reviewers will push back with "but our production cluster has NVLink/InfiniBand; Llumnix works there." The paper pre-empts this well in §1 by scoping to "commodity, network-constrained" deployments, but should cite one or two production systems that fit that profile (e.g., common cloud VMs, Kubernetes nodes with standard Ethernet).

**Reproducibility (4/5).** Code is open-sourced (anonymous.4open.science link in §5). Configs, dataset (ShareGPT), models (Llama-2-7B, Llama-2-70B, Qwen2-7B) are all standard. Cluster is CloudLab d7525/d8545 — reproducible by anyone with CloudLab access. Seed/random-draw documentation would help; I don't see mention of confidence intervals or repeated runs.

**Experimental design (4/5).** Strong: multi-model, multi-dataset, multi-cluster, multi-N ablation, noise ablation, burstiness ablation, CPU-overhead ablation. Weak: no direct head-to-head on *full Llumnix* on the A30 cluster (migration impractical there, understood); no confidence intervals on any figure; no cost model / dollar-per-request comparison; no goodput-vs-QPS plot (only latency-vs-QPS and throughput-vs-QPS). The 2/17 QPS points where Po2 loses are not identified.

---

## 5. Punch List — What to Change for Next Iteration (Prioritised)

**#1 (must-fix) — Fix A100 number consistency in abstract.** Currently abstract says "up to 2.6× throughput and 123× per-token latency"; §6.4 says the CP config gives 2.58×/40×/135× and the no-CP config gives 2.57×/36×/123×. Pick one config (I recommend CP, since CP is the standard vLLM setting) and report three consistent numbers: **"2.58× throughput, 40× mean request latency, 135× per-token P99 latency at saturation (A100, full Llumnix comparator)."** The "40×" is currently invisible in the abstract and it's the strongest user-facing number.

**#2 (must-fix) — Identify the 2/17 points where Po2 does not win.** Add one sentence to §6.3: "At QPS=X and QPS=Y, Po2 is within Z % of the best scheduler; both points are in the overload regime." Without this, the "15 of 17" framing invites suspicion.

**#3 (must-fix) — Re-order the abstract so the capacity-tie is *after* the dominance claim, not before.** Current order: "capacity tie … but 12.6 % on Qwen2-7B …". Proposed order: (i) lowest TTFT/E2E at 15/17 QPS points → (ii) 6× fewer preemptions → (iii) 2.8× less CPU than fanout → (iv) capacity is Pareto-equivalent with Llumnix- on Llama-2-7B/ShareGPT *and* +6–12.6 % on alternative models/configs. The revised abstract already moves in this direction but still buries the capacity-tie as a negative-sounding clause. Re-word as "**Peak capacity is Pareto-equivalent** with migration-based Llumnix on Llama-2-7B/ShareGPT (31.6 vs 31.5 QPS) and **strictly higher** on Qwen2-7B (+6.0 %, or +12.6 % with oracle lengths) and tight-batching configs (+9.4 %)."

**#4 (must-fix) — Kill symbol drift.** Global pass: `Po2` (not `Astrolabe-Po2`, not `Block-Po2`), `PoN` (not `power-of-k`, not `k-choices`), `N=2` (not `k=2`, not `n=2`). Algorithm 1 uses lowercase `n` — promote to `N`. Fig. 4 frame still says "Block System" — rename to "Astrolabe". Grep the .tex for "Block" and remove all non-paper-title occurrences.

**#5 (should-fix) — Pull the GPU-memory variance collapse into §6.3 prose and abstract.** Fig. 7 middle row shows ~10× lower cross-instance variance for Po2 under QPS=28. Add to abstract: "and reduces cross-instance GPU-memory variance by ~10×, a direct measure of load-balance quality." Add one sentence to §6.3 under "GPU memory dynamics" explicitly calling out the variance (not just "low and preemptions near zero"). This is a figure that *proves* load balance, not just implies it.

### Lower-priority fixes (still worth doing)

- §3 has a literal "[17, 61?]" placeholder. Fix.
- Figure 4 outer frame labelled "Block System". Rename.
- Figure 5 caption "Latency Prediction Metrics" does not match the prose (simulator accuracy, rank-1 selection). Rename.
- Figure 6 bar chart "Capacity @ SLO" appears to show values 29.3/30.7/29.2 for Random/RR/MinQPM, but §6.3 prose says 21–23 QPS. Reconcile.
- Figure 8 caption: note CP vs no-CP series.
- Tighten abstract's "not migration" claim to "*dispatch quality is sufficient; migration is not required*" — the data shows sufficiency, not necessity.
- Add one sentence on Llumnix- reconstruction fidelity (validation against Llumnix paper numbers).
- Add burstiness and noise-robustness as explicit bullets in §1's contribution list.
- Clarify the "15/17" denominator: is it a 0.1-QPS sweep binned to integer? Abstract / §6.3 should explicitly state "17 integer QPS points from 20 to 36".
- Consider a goodput-vs-QPS plot alongside Fig. 6 — reviewers trained on Llumnix-paper conventions will expect it.
- Table 1: either acknowledge the MAE gap (78 vs 62) or quantify the cost-per-prediction advantage (Roberta: 4.8s on 10k requests in PyTorch vs 7B LLM prompt-based: ?). The "~1000× cheaper" argument, if you can make it, turns Table 1 from "worse on 2/3 metrics" into "comparable at 1000× less cost."

---

## 6. Single-Line Verdict

**Borderline / Weak Accept.** The revised framing (15/17 + 6× preemption + 2.8× cheaper) is defensible and the A100 result is genuinely strong; with the prioritised abstract and §6.3 edits above, this becomes a solid SoCC paper — without them, it will draw a "capacity-tied" critique that overshadows the real wins.
