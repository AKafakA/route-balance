## POST 3c — Response to Reviewer faAU (3/3): generality — model family; tuning and budget

**W1, second half: a second model family.** We onboarded a second, architecturally distinct family end to end inside this discussion window — the Gemma-3 instruction-tuned ladder (1B/4B/12B/27B) — and re-ran the comparison with no change to the scheduler. Each cell below is 1,979 held-out prompts at λ=24 with zero failures, with quality measured by the same DeepEval judge and the same lookup procedure as the submitted evaluation; the family's quality estimator is built on a disjoint 8,086-prompt training split. Best value per column is in bold.

Two scoping notes belong beside this table rather than only in our general response: this family was served on measured static per-(model, GPU) latency priors rather than newly trained latency heads, so what is tested here is the formulation on a new family, not the learned estimator on a new family; and cost again prices input+output tokens, not the submission's output-only axis.

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
