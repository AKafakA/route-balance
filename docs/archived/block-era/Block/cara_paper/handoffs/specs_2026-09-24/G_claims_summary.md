# G_claims summary: RouteBalance claims ledger

Sources: `neurips_2026.tex` plus `tables/` (checklist.tex is empty and has no claims), `eurosys_2027/main.tex` plus `eurosys_2027/tables/`, the posted rebuttal (A1–A5), and `mlsys_2027/OUTLINE.md`. Source-reference formats: `neurips_2026.tex:N`, `neurips:tables/X.tex:N`, `eurosys:N`, `eurosys:tables/X.tex:N`, `rebuttal:Ax/LN` (line in posted_rebuttal_final.md), `outline:LN`.

Total claims: **243** (C001–C243). Claims with a recorded conflict: **78**. Every quote was checked as a verbatim, whitespace-normalized substring of its cited lines.

## Counts by area and kind

| area | design | eval_setup | mechanism | total |
|---|---:|---:|---:|---:|
| objective | 7 | 0 | 0 | 7 |
| normalization | 4 | 0 | 1 | 5 |
| quality_est | 12 | 0 | 2 | 14 |
| latency_est | 14 | 6 | 1 | 21 |
| cost | 1 | 0 | 0 | 1 |
| batching | 7 | 0 | 1 | 8 |
| lpt | 2 | 0 | 0 | 2 |
| state | 5 | 1 | 0 | 6 |
| tiebreak | 2 | 0 | 0 | 2 |
| filter_budget | 3 | 6 | 4 | 13 |
| filter_slo | 1 | 0 | 0 | 1 |
| overhead | 0 | 7 | 9 | 16 |
| centralization | 6 | 0 | 0 | 6 |
| cluster | 0 | 5 | 0 | 5 |
| engine | 2 | 6 | 0 | 8 |
| workload | 0 | 10 | 0 | 10 |
| quality_metric | 0 | 7 | 1 | 8 |
| cost_metric | 0 | 3 | 1 | 4 |
| latency_metric | 0 | 6 | 0 | 6 |
| baselines | 0 | 7 | 1 | 8 |
| hosting | 4 | 6 | 3 | 13 |
| sweeps | 0 | 8 | 0 | 8 |
| selection_rule | 0 | 7 | 0 | 7 |
| ablations | 0 | 11 | 6 | 17 |
| stats | 0 | 5 | 1 | 6 |
| counts | 0 | 5 | 0 | 5 |
| positioning | 0 | 1 | 35 | 36 |
| **total** | 70 | 107 | 66 | 243 |

Checks by `check_against`: code 85, cell_data 62, config 44, scripts 40, doc 12.

## Inter-source (and intra-source) conflicts: full list

Each entry is: ID, area, source, the claim, then the disagreement. "Internal" means the disagreement is inside one source.

- **C009** [normalization] `neurips_2026.tex:175`: Qhat enters the score unnormalized, with no per-request or per-round rescaling, and lies in [0,1].
  - *Conflict:* rebuttal:A1/L15 says quality, cost AND latency are 'normalized over the candidates in the same round'. neurips_2026.tex:134 says the formula normalizes 'latency, cost, and quality against the batch itself'. eurosys:117 normalizes only latency and cost.
- **C011** [normalization] `eurosys:128`: Each normalizer's max is taken per request over that request's candidate instances (max_j Chat_{r,j}, r fixed), not over all request-instance pairs in the round.
  - *Conflict:* Scope of the normalizer. The formula (neurips_2026.tex:167-168, eurosys:123-124) is per request, but the prose says batch- or round-wide: neurips_2026.tex:175 'batch-wide maxima', neurips_2026.tex:134 'against the batch itself', eurosys:117 'within the batch', eurosys:128 'The batch supplies the candidate set', rebuttal:A1/L15 'over the candidates in the same round', outline:L72 'normalization within a round'.
- **C012** [normalization] `neurips_2026.tex:167`: The normalizer max ranges over all deployed instances I, not only budget-feasible ones. The latency max is recomputed from dead-reckoned state after every in-round dispatch.
  - *Conflict:* Internal to eurosys: Alg. 1 (eurosys:167-168) takes the argmax over the budget-filtered set C, while Eq. 1 writes max_j and leaves its range unstated. neurips uses max over j in I. Whether the latency max is refreshed after dead-reckoning updates is not stated.
- **C014** [positioning] `eurosys:128`: The round supplies the normalization reference set, which a point-at-a-time router lacks.
  - *Conflict:* Internal to eurosys: if normalization is per request over candidate instances ('per-request normalized by their candidate maxima', same line), a point-at-a-time router has the same reference, and the claim holds only if the maxima are batch-wide.
- **C017** [quality_est] `neurips_2026.tex:204`: The deployed KNN uses k=50 neighbours.
  - *Conflict:* eurosys:61, eurosys:175 and eurosys:330 give k=10 for the deployed estimator. eurosys:330 also reports a k-sweep over {5,10,20,50}. The two paper versions state different deployed k.
- **C020** [quality_est] `neurips_2026.tex:84`: Quality labels for harmful or safety prompts use a safety-aware rule (e.g. refusal scored high), not the plain reference-grounded G-Eval score.
  - *Conflict:* Only neurips_2026.tex:84 says this. neurips_2026.tex:204, neurips_2026.tex:219 and eurosys:175/207 describe the labels only as DeepEval G-Eval against dataset references. eurosys:338 says safety 'follows from the weight vector, not a separate mechanism'.
- **C032** [latency_est] `eurosys:177`: Each round makes one TPOT call per tier. Per-instance and per-request variation enter only through the analytic formula.
  - *Conflict:* neurips_2026.tex:152 indexes TPOT per (request, instance) as T^tpot_{r,i}. rebuttal:A1/L10 predicts TPOT 'for each candidate instance'. rebuttal:A2/L22 lists the request's predicted output length (per request) and serving-state signals (per instance) as head inputs. Alg. 1 (eurosys:164) calls TierHeads() with no arguments.
- **C037** [latency_est] `rebuttal:A2/L23`: Neither the score nor That has a TTFT or prefill component.
  - *Conflict:* eurosys:tables/tab_predictor.tex:2 reports TTFT MAE and E2E MAPE for the 'deployed' per-(model,GPU) XGBoost, which implies TTFT and E2E heads exist. neurips_2026.tex:74 says accurate scheduling needs real-time prediction from current serving state.
- **C040** [latency_est] `eurosys:164`: Algorithm 1 evaluates the tier heads (line 2) before reading telemetry (line 3).
  - *Conflict:* neurips_2026.tex:193 and rebuttal:A1/L10 put the telemetry read before estimation, and rebuttal:A2/L22 says the heads consume serving-state signals, which needs telemetry first.
- **C041** [latency_est] `eurosys:tables/tab_predictor.tex:2`: Deployed per-tier XGBoost artifacts cover TPOT, TTFT and E2E, with held-out n_val 13,434/22,465/13,502/9,045 (3B/7B/14B/72B).
  - *Conflict:* The design text (neurips_2026.tex:207, eurosys:177, rebuttal:A2/L22) deploys only a TPOT head and computes E2E analytically. The E2E MAPE row may describe a model that is not on the scoring path.
- **C042** [latency_est] `eurosys:248`: The eurosys static-prior arm uses That = nominal per-tier TPOT * Lhat, with no queue term and no telemetry.
  - *Conflict:* Three different static-latency definitions. rebuttal:A2/L21: the static arm keeps 'the same within-tier queue tiebreak' and replaces only the learned head's queue-conditioned estimate with a per-tier constant. outline:L11-12: 'static per-(model,GPU) TPOT × live queue state' (queue term kept). eurosys:248: 'zero telemetry', under which same-tier instances tie in S and an undocumented tiebreak decides placement.
- **C043** [latency_est] `outline:L11-12`: New-paper design: That = measured static per-(model,GPU) TPOT * live queue-aware decode iterations. The learned head is an ablation only.
  - *Conflict:* eurosys:179 and eurosys:248 keep the learned head as the deployment default. neurips uses only the learned head.
- **C044** [latency_est] `rebuttal:A2/L21`: Static-prior constants are 3B 10.7, 7B 19.9, 14B 16.5, 72B 42.6 ms/token: per-tier medians over the whole sweep.
  - *Conflict:* Table 1 TPOT (neurips_2026.tex:115-118, eurosys:103-106) is 10.2/19.6/13.9/41.6 ms 'under moderate load'. eurosys:248 says the prior 'is distilled from its [learned] traces'. outline:L112 re-calibrates 'at cap 1024 (not from test cells)', which implies the rebuttal medians 'across the whole sweep' may include test cells.
- **C046** [latency_est] `rebuttal:A3/L28`: A correct static-TPOT gate replaces That in both the per-candidate term and the normalizer max, and never invokes the learned predictor. The rebuttal says the first static arm failed both.
  - *Conflict:* eurosys:248 reports that arm 4 (static prior) reproduces arm 1 with 'mix and quality unchanged'. That is the signature of the misconfigured arm in rebuttal:A3/L28. The provenance of the arm-4 cells (before or after the fix) must be shown.
- **C048** [ablations] `rebuttal:A4/L32`: Static and learned arms differ on 4.4% of placements; the learned arm sends 2-3 pp more traffic to 72B.
  - *Conflict:* eurosys:248 says the static prior leaves 'mix and quality unchanged'.
- **C049** [positioning] `eurosys:179`: A static per-tier TPOT prior reproduces the headline frontier; the learned head is not needed.
  - *Conflict:* neurips_2026.tex:74 says 'accurate scheduling needs a real-time prediction from the current serving state', and neurips makes the learned XGBoost head a core contribution (neurips_2026.tex:83-86). The rebuttal:A2 version of this claim was retracted in rebuttal:A3.
- **C050** [latency_est] `eurosys:179`: The learned TPOT head is the default latency estimator in the evaluated configs.
  - *Conflict:* outline:L11-12 makes the static prior the main design and demotes the learned head to 'only an ablation row'.
- **C052** [positioning] `eurosys:179`: Learned per-tier heads are chosen over roofline models, which the paper criticizes for assuming a static per-hardware TPOT constant.
  - *Conflict:* The static per-tier TPOT prior championed at eurosys:248, rebuttal:A4/L34 and outline:L11 is exactly such a static constant, so eurosys argues against the design it and the outline later adopt.
- **C055** [cost] `neurips_2026.tex:154`: The scoring cost term is predicted output tokens times one per-model price: decode-only, input tokens excluded.
  - *Conflict:* neurips_2026.tex:179 (Eq. 2), eurosys:141 and eurosys:150 define Chat = l_in*c_in + Lhat*c_out (input + output). outline:L81 says 'decode-cost basis'. Which form enters the score and which the budget filter must be checked.
- **C059** [cost_metric] `neurips_2026.tex:221`: Reported cost = realized output tokens * output price; input tokens not charged.
  - *Conflict:* eurosys:207 says 'realized tokens at public per-token prices' without saying whether input counts. Table 1 lists separate input prices, and Eq. 2 charges input. outline:L81 says 'decode-cost basis'.
- **C065** [batching] `rebuttal:A1/L16`: busy_ratio = fraction of instances with at least one waiting request. <0.3 gives (size 4, 50 ms), <0.7 gives (16, 100 ms), otherwise (32, 200 ms).
  - *Conflict:* The papers say 'busy' instances (neurips_2026.tex:136, eurosys:117) and the rebuttal says 'waiting request'. Queued-vs-running semantics and whether the ratio uses raw telemetry or dead-reckoned state must be checked.
- **C072** [lpt] `eurosys:119`: The LPT sort key is the max over models of Lhat_{r,m}, not the mean or min.
  - *Conflict:* neurips (neurips_2026.tex:84, :134, :193) says only 'predicted output length' and does not give the cross-model aggregation.
- **C074** [ablations] `neurips_2026.tex:294`: Turning LPT off (arrival order within the round) changes E2E by at most ±0.5% over λ 6-30.
  - *Conflict:* eurosys:301 gives ±2.3% for the same ablation.
- **C075** [state] `neurips_2026.tex:193`: At the start of each round the scheduler fetches instance telemetry (RPC) and seeds the dead-reckoned state from it.
  - *Conflict:* Synchronous per-round fetch or asynchronous cache? neurips_2026.tex:339 and both breakdown tables charge a per-round 'stats fetch' RPC (6.6-14.7 ms) to each request. rebuttal:A1/L18 starts from 'the latest asynchronous telemetry' and reconciles 'when the next authoritative update arrives'. outline:L13 has an 'asynchronously refreshed stats cache'.
- **C078** [state] `rebuttal:A1/L18`: Dead-reckoned deltas persist across rounds until the next authoritative telemetry reconciles them. Whether that update overwrites or merges in-flight deltas must be checked.
  - *Conflict:* The paper text (neurips_2026.tex:193, eurosys:165) implies state is re-seeded from fresh telemetry every round, so deltas would live only within a round.
- **C082** [tiebreak] `eurosys:248`: The scheduler has a built-in within-tier tiebreak: among equal-score instances, pick the shortest queue. With w_lat=0 this decides replica placement.
  - *Conflict:* The queue signal the tiebreak reads (raw telemetry or dead-reckoned) is not stated. eurosys:248 arm 4 ('zero telemetry') would leave the tiebreak with no queue data, yet rebuttal:A2/L21 says the static arm keeps 'the same within-tier queue tiebreak'.
- **C092** [filter_budget] `neurips_2026.tex:272`: Three configs: RB+filter at uniform weights, RB without the filter, BEST-Route 'argmax' without the filter.
  - *Conflict:* eurosys:268 does not give RB's weights. Neither paper defines the BEST-Route 'argmax' setting (t=0?).
- **C095** [filter_budget] `neurips_2026.tex:290`: The filter cuts exhaustion by 8.8-10.3 pp at the tight mix and 4.2-4.3 pp at the loose mix.
  - *Conflict:* eurosys:268 gives 6.3 pp (tight) and 2.9 pp (loose). The no-filter rates also differ: neurips_2026.tex:284 has 42.7/26.1/16.9, eurosys:282 has 39.1/23.5/15.5.
- **C096** [filter_budget] `neurips_2026.tex:290`: Without the filter, RB and BEST-Route exhaustion differ by at most 1.5 pp.
  - *Conflict:* The eurosys:281-283 numbers give 39.1 vs 41.5 (2.4 pp) at the tight mix.
- **C099** [filter_slo] `neurips_2026.tex:183`: No SLO filter is active in any evaluated config; any global or per-request SLO path in the scheduler must be off.
  - *Conflict:* neurips_2026.tex:123 frames the missing layer as jointly considering 'quality, SLOs, cost, and live serving state'. eurosys:342 names 'SLO-safe presets' without any SLO mechanism.
- **C105** [overhead] `neurips_2026.tex:339`: net+runtime is a computed remainder, not a measurement.
  - *Conflict:* eurosys:tables/tab_sched_breakdown.tex drops this column. Its spans (e.g. 32.4+47.9+12.6 = 92.9 of 130.0 ms at λ=12) leave about 37 ms unlabeled.
- **C106** [overhead] `neurips_2026.tex:66`: Decision compute per request is about 33 ms at λ=12 and about 24 ms at λ=30 (uniform weights).
  - *Conflict:* The values and trend differ across versions. eurosys:43 gives ≈32 ms at 12 req/s, eurosys:197 '≈28–32 ms/request', eurosys:237 '≈27 ms' and 32.3→28.2 ms over λ 6→30 (−13%). neurips tables and rebuttal:A1/L14 give 44.7→23.9/25.7 ms (−47%).
- **C108** [overhead] `neurips_2026.tex:263`: RB's residual grows sub-linearly with λ.
  - *Conflict:* eurosys:237 gives 129→231 ms (1.8×).
- **C113** [hosting] `neurips_2026.tex:265`: In pipeline mode BEST-Route's DeBERTa scorer runs one prompt at a time on the scheduler path, while RB's estimator is batched per round.
  - *Conflict:* neurips_2026.tex:210 calls the pipeline-mode comparison 'architecture-controlled', yet baseline scoring runs serially per request while RB's runs batched per round — a difference in what executes (first rule). neurips_2026.tex:265 also says Fig. 2(c) 'runs every baseline inside RouteBalance's pipeline mode so the routing-policy gap is measured separately from this architectural one', but pipeline mode contains that serial scorer.
- **C114** [hosting] `eurosys:344`: The as-published Avengers-Pro path scores one request at a time, so its k-means residual climbs from 258 ms to 2.79 s.
  - *Conflict:* Internal to eurosys:344: 'both' seems to include RB (the sentence pairs Avengers-Pro with RB), which contradicts RB's batched estimator.
- **C115** [overhead] `outline:L87-88`: The new paper splits overhead into estimate / score / dispatch spans.
  - *Conflict:* The neurips and eurosys breakdowns use compute / batch wait / stats fetch (/ net+runtime). The outline split has no batch-wait or stats-fetch span.
- **C123** [engine] `neurips_2026.tex:199`: Every vLLM engine arg is default except TP (and eager for 72B). The vLLM version/fork is not stated, though the telemetry endpoint implies a patched vLLM.
  - *Conflict:* Internal to both papers: neurips_2026.tex:219 and eurosys:207 also run the 14B with TP=4, so 'only the 72B has non-default flags' cannot be exactly right. eurosys:195 omits the 72B eager flag of neurips_2026.tex:219.
- **C125** [engine] `neurips_2026.tex:219`: The 72B instances run with enforce_eager=True; the other tiers use CUDA graphs.
  - *Conflict:* eurosys (eurosys:195, eurosys:207) does not mention eager mode.
- **C128** [engine] `outline:L112`: The new run calibrates static TPOT with a cap of 1024 (probably max output tokens).
  - *Conflict:* Neither neurips nor eurosys states a max_tokens cap for the no-budget cells. The cap actually used must be recovered from config and shown to be identical for all systems.
- **C134** [cluster] `rebuttal:A5/L39`: Topology variant: 3B×5 and 7B×3 on A30, every system at λ=24, nothing retrained.
  - *Conflict:* outline:L117 plans λ∈{12,24}.
- **C135** [cluster] `rebuttal:A5/L40`: Gemma cluster: 1B×3 and 4B×5 on A30, 12B×4 and 27B×2 on A100 (14 instances). Per-instance GPU/TP counts are not stated.
  - *Conflict:* The rebuttal ran everything at rate 12 (same line); outline:L116 plans λ∈{12,24} × 3 seeds.
- **C139** [workload] `eurosys:207`: Every serving cell replays the same 3,534-prompt subset of the test split.
  - *Conflict:* neurips never explains 3,634→3,534 (neurips_2026.tex:219 against :272/:318). 'Complete coverage' contradicts neurips_2026.tex:318 and :327-330, where only 72.2-78.2% of served prompts map to a scored (prompt, model) pair.
- **C141** [workload] `neurips_2026.tex:229`: Main-grid rates: λ ∈ {6, 8, 10, 12, 18, 24, 30} req/s.
  - *Conflict:* eurosys:209 writes 'λ∈{6..30}', and its tables show {6,12,18,24,30} (tab_vllm_sr adds 6-10). The eurosys hull counts (16 tuples × 7 = 112 RB cells) imply 7 rates. outline:L113 plans 5.
- **C144** [workload] `eurosys:207`: No evaluation prompt is in the training data of the KNN index, BEST-Route's DeBERTa, Avengers-Pro's clusters, or the XGBoost/static-TPOT calibration traces.
  - *Conflict:* The static-TPOT constants in rebuttal:A2/L21 are medians 'across the whole sweep', and the prompts in the latency-profiling sweep are not stated. outline:L112 requires 'not from test cells'.
- **C145** [workload] `rebuttal:A1/L17`: Every served prompt is a single-turn request.
  - *Conflict:* neurips_2026.tex:219 and eurosys:207 describe the seven datasets as covering 'multi-turn chat' (LMSYS-Chat-1M). rebuttal:A4/L35 softens to 'predominantly single-turn'.
- **C152** [quality_metric] `neurips_2026.tex:318`: Unscored (prompt, model) pairs and failed requests are left out of the quality mean. Coverage is about 72-78%; how zero scores are treated is not stated.
  - *Conflict:* eurosys:207 claims the serving subset has 'complete coverage'. The same w_q=0.8, λ=12 cell scores 0.5141 in neurips_2026.tex:327 but 0.4187 in eurosys:314 (~0.10 apart), which fits a change in how missing or zero scores are aggregated, or relabelling.
- **C155** [quality_metric] `eurosys:272`: Budget-cell quality judges the served text (possibly truncated), not the lookup.
  - *Conflict:* Main-grid quality is a lookup (eurosys:219), so eurosys uses two different quality metrics.
- **C160** [latency_metric] `neurips:tables/tab_sched_breakdown.tex:2`: The breakdown's TTFT is clocked by the scheduler from dispatch to the first streamed token.
  - *Conflict:* eurosys:tables/tab_sched_breakdown.tex:2 reports 'client TTFT' instead: client-observed, including the residual (160-270 ms against 47-53 ms).
- **C162** [sweeps] `neurips_2026.tex:227`: Avengers-Pro pw sweep: {0.25, 0.39, 0.53, 0.8}.
  - *Conflict:* eurosys:209 adds 0.70 (5 values), and outline:L113 has 5. Internal to neurips: neurips:tables/tab_extremes.tex:9 counts 70 AvengersPro cells = 5 pw × 2 dispatchers × 7 λ, which does not match the 4 values listed.
- **C163** [sweeps] `neurips_2026.tex:227`: BEST-Route threshold sweep: {0.3, 0.5, 0.7}.
  - *Conflict:* eurosys:209 lists t∈{0,0.3,0.5,0.6,0.7,0.8}, and outline:L113 has '4 t'. neurips tables use an 'argmax' setting absent from this list (tab_radar_picks, neurips_2026.tex:329, tab_residence, budget). The 56 neurips BEST-Route cells = 4 settings × 2 × 7 λ, not 3.
- **C164** [baselines] `neurips_2026.tex:227`: In the paper grids each trained router is paired with RR and SQ only (no random, no Po2).
  - *Conflict:* outline:L113 plans 3 dispatchers per router, and rebuttal:A5/L38 adds Po2.
- **C165** [baselines] `neurips_2026.tex:227`: Passthrough = model-blind dispatch over all 13 instances with RR, SQ or random: 3 dispatchers × 7 λ = 21 cells.
  - *Conflict:* The tables call the same 21-cell system 'Dispatcher-only' (tab_extremes). outline:L113 lists 'dispatcher-only' and 'passthrough' as separate systems.
- **C166** [baselines] `eurosys:209`: vLLM-SR is an unmodified external process, not a RouterBase plug-in, so it is not hosted in pipeline mode.
  - *Conflict:* neurips_2026.tex:305 excludes vLLM-SR as 'static-category and subsumed by AvengersPro's pw-mix'.
- **C168** [baselines] `eurosys:207`: BEST-Route's DeBERTa-v3 and Avengers-Pro's clusters were trained on the same 14,919-prompt train split with the same DeepEval labels.
  - *Conflict:* neurips does not say how the baselines were trained.
- **C170** [hosting] `eurosys:344`: The enhanced scoring path micro-batches in-flight scoring requests.
  - *Conflict:* Internal to eurosys: eurosys:243 describes the enhanced rung as per-prompt forwards on the default thread-pool executor (32 threads, ~13 concurrent forwards), contrasted with the padded micro-batch collector of rung (ii).
- **C173** [hosting] `eurosys:243`: One serial BEST-Route DeBERTa forward takes 431 ms per prompt.
  - *Conflict:* At 431 ms/prompt a serial scorer sustains at most ~2.3 req/s, yet neurips:tables/tab_sched_combined.tex:11 shows a 131 ms BEST-Route residual at λ=12 and eurosys:tables/tab_residence_vs_e2e.tex:13 shows 697 ms. Either scoring was not strictly serial or the 431 ms figure comes from a different setting.
- **C178** [hosting] `neurips_2026.tex:210`: Baselines go through the same rounds (same adaptive rule), telemetry fetch and dispatch client as RB. Every remaining execution difference must be listed: estimator batching, LPT, dead reckoning, the state SQ reads, and the scoring thread.
  - *Conflict:* outline:L13-14 withholds dead reckoning from baselines ('dead reckoning stays RB-only') and moves router scoring 'off the event loop'. neurips_2026.tex:265 shows pipeline-mode BEST-Route scoring serially. So 'only the routing logic and dispatcher differ' is not true at execution level (first rule).
- **C181** [hosting] `outline:L13-14`: SQ/Po2 read the same asynchronously refreshed stats cache as RB, receive no dead-reckoning updates, and score routers off the event loop.
  - *Conflict:* In neurips/eurosys, RB's state is a per-round synchronous fetch plus dead reckoning (neurips_2026.tex:339, eurosys:165). The state SQ read in the paper runs is not stated.
- **C182** [positioning] `eurosys:43`: The serial-deployed collapse is a deployment effect; the policy comparison uses the enhanced variants.
  - *Conflict:* neurips has no enhanced variants and presents the serial pipeline-mode baselines as the 'architecture-controlled' comparison (neurips_2026.tex:210, neurips_2026.tex:265).
- **C183** [sweeps] `eurosys:209`: RB's no-budget sweep uses 16 weight tuples at each λ.
  - *Conflict:* neurips gives no count, but its 56 RB cells over 7 λ imply 8 tuples (neurips:tables/tab_extremes.tex:8). outline:L113 plans 8 presets. eurosys's 112 cells = 16×7.
- **C187** [ablations] `rebuttal:A2/L21`: The static-vs-learned comparison uses the (0.5, 0.3, 0.2) preset, the one the radar picks select.
  - *Conflict:* eurosys:248 runs static vs learned (arm 4) only at the uniform cell; rebuttal:A4/L32 uses both presets.
- **C190** [selection_rule] `outline:L82`: Each system's representative cell at each λ is its argmax of quality × throughput over its own knobs and dispatchers.
  - *Conflict:* eurosys's headline comparisons use each baseline's 'peak-quality cell' (eurosys:tables/tab_residence_vs_e2e.tex:2, eurosys:308), not quality × throughput. rebuttal:A5/L38 says 'best setting from a full sweep' without giving the criterion.
- **C192** [selection_rule] `neurips_2026.tex:329`: neurips uses 'BEST-Route argmax, RR' as the BEST-Route comparison cell at λ=12.
  - *Conflict:* eurosys:315 (tab_bootstrap) takes BEST-Route t=0.5 as the peak cell, with SQ in tab_multiseed, tab_judge_alt and tab_tail and RR in tab_residence. The BEST-Route headline cell differs across versions, and across tables within eurosys.
- **C193** [selection_rule] `neurips_2026.tex:248`: neurips radar axes are min-max normalized per λ panel, with one picked cell per system.
  - *Conflict:* eurosys:228: 'each axis is the fraction of the best plotted cell at that λ', with three RB presets and λ∈{6,12,18,30}.
- **C195** [selection_rule] `neurips_2026.tex:259`: Overhead cells: RB uniform, BEST-Route t=0.5/RR, Avengers-Pro pw=0.8/RR, passthrough/RR.
  - *Conflict:* Internal to neurips: tab_residence_vs_e2e.tex:12 labels the row 'BEST-Route argmax' (148.8 ms at λ12, 24,793 ms at λ24), but tab_sched_combined gives 131 ms and 19.6 s at the same λ, so two different BEST-Route cells are shown as one operating point. eurosys picks each baseline's 'peak-quality cell' (BEST-Route t=0.5, RR) and adds enhanced SQ rows.
- **C198** [ablations] `eurosys:248`: The static-prior arm matches the full objective's mix and quality and is 8% faster at λ=18.
  - *Conflict:* rebuttal:A3/L28 retracts an earlier static-arm result as misconfigured. rebuttal:A4/L32 says the corrected arms route differently (4.4% of placements, 2-3 pp from 14B to 72B).
- **C200** [ablations] `neurips_2026.tex:294`: Adaptive-off costs 5.1-6.9% E2E at every λ. The fixed size and timeout it uses are not stated.
  - *Conflict:* eurosys:301 gives 0.4-6.0%, 'growing with λ'.
- **C202** [ablations] `neurips_2026.tex:294`: bs=1 collapses at λ≥16.
  - *Conflict:* Direct contradiction: eurosys:301 says 'the batched-KNN estimator keeps bs=1 from collapsing (2.4/4.0 s at λ=16/24)'. rebuttal:A1/L16 agrees with neurips ('falls back to serial estimation and loses the amortization benefit').
- **C212** [stats] `neurips_2026.tex:318`: neurips CIs: unpaired per-system percentile bootstrap, 1,000 resamples, seed 42, over covered served prompts at λ=12.
  - *Conflict:* eurosys:304 and eurosys:308 use a paired per-prompt bootstrap with 10,000 replicates on per-prompt differences.
- **C213** [stats] `neurips_2026.tex:318`: Each neurips main-grid cell has one arrival seed.
  - *Conflict:* eurosys:304 and tab_multiseed use four Poisson seeds for the headline cells (latency n=3). rebuttal:A4/L32 uses 6 seeds. outline:L114 plans +2 seeds.
- **C218** [counts] `neurips_2026.tex:89`: neurips: 243 cells ≈ 860K requests (243×3,534 = 858.8K). The 203 no-budget cells = 56 RB + 70 Avengers-Pro + 56 BEST-Route + 21 dispatcher-only. The other 40 are budget (9) plus batching.
  - *Conflict:* eurosys:68, eurosys:209 and eurosys:348 give 442 configurations, ≈1.5M requests and 287 no-budget cells.
- **C219** [counts] `eurosys:209`: eurosys: 442 cells ≈ 1.56M requests. The 287 no-budget cells = 112 RB + 84 BEST-Route + 70 Avengers-Pro + 21 dispatcher-only, so 175 baseline cells.
  - *Conflict:* neurips_2026.tex:89 gives 243 / 860K / 203.
- **C227** [positioning] `rebuttal:A5/L43`: One preset keeps its operating point as λ changes.
  - *Conflict:* neurips_2026.tex:309 says the three weights 'must be re-tuned as the workload drifts'. The trigger differs (workload drift vs rate), but the two versions frame tuning differently.
- **C233** [positioning] `neurips_2026.tex:243`: On the training set, always-14B beats always-3B by 0.015.
  - *Conflict:* eurosys:234 puts the always-3B to always-14B step at 0.052 (0.346 vs 0.398).
- **C234** [positioning] `neurips_2026.tex:243`: RB's headline quality margins over BEST-Route and Avengers-Pro.
  - *Conflict:* eurosys:219 and eurosys:234 give +0.013 over BEST-Route and +0.043 over Avengers-Pro.
- **C235** [positioning] `neurips_2026.tex:229`: Max sustained throughput: RB 28.85, BEST-Route 18.73 req/s.
  - *Conflict:* eurosys:221 and eurosys:tables/tab_extremes.tex:10-11 give RB 27.6 and BEST-Route 21.8 req/s.
- **C236** [positioning] `eurosys:338`: There is no safety-specific routing rule.
  - *Conflict:* neurips_2026.tex:84 calls the quality labels 'safety-aware', so safety does enter through the labels.
- **C242** [positioning] `neurips_2026.tex:74`: Latency must be predicted from live serving state.
  - *Conflict:* eurosys:248 finding (iii): 'The latency signal need not be learned or live'. eurosys:179 and outline:L39-40 say a static prior is enough.

## The 25 most load-bearing claims

These are the claims the argument rests on. If any of them fails against the code, config or cell data, a headline result or the fairness of a comparison fails with it.

1. **C001** [objective/design, check: code] `neurips_2026.tex:156`: The score has exactly three weights (quality, cost, latency), constrained to sum to 1. No fourth term (e.g. load-balance or KV-memory) and no nonzero balance weight in any evaluated config.
2. **C002** [objective/design, check: code] `neurips_2026.tex:166-168`: Per-candidate score S = w_q*Qhat + w_c*(1 - Chat/max_j Chat) + w_l*(1 - That/max_j That).
3. **C011** [normalization/design, check: code] `eurosys:128`: Each normalizer's max is taken per request over that request's candidate instances (max_j Chat_{r,j}, r fixed), not over all request-instance pairs in the round. **(conflict recorded)**
4. **C009** [normalization/design, check: code] `neurips_2026.tex:175`: Qhat enters the score unnormalized, with no per-request or per-round rescaling, and lies in [0,1]. **(conflict recorded)**
5. **C017** [quality_est/design, check: config] `neurips_2026.tex:204`: The deployed KNN uses k=50 neighbours. **(conflict recorded)**
6. **C018** [quality_est/design, check: config] `neurips_2026.tex:204`: The KNN index holds exactly the 14,919 training-split prompts and none of the 3,634 test prompts.
7. **C035** [latency_est/design, check: code] `eurosys:177`: E2E estimate = predicted TPOT * (pending decode tokens / current decode batch size + Lhat). No TTFT, prefill or admission-queue term.
8. **C030** [latency_est/design, check: code] `neurips_2026.tex:207`: One XGBoost TPOT regressor per (model, GPU) tier (4 for Qwen), each trained only on that tier's QPS-sweep traces from its head node and applied to every instance of the tier.
9. **C032** [latency_est/design, check: code] `eurosys:177`: Each round makes one TPOT call per tier. Per-instance and per-request variation enter only through the analytic formula. **(conflict recorded)**
10. **C046** [latency_est/mechanism, check: code] `rebuttal:A3/L28`: A correct static-TPOT gate replaces That in both the per-candidate term and the normalizer max, and never invokes the learned predictor. The rebuttal says the first static arm failed both. **(conflict recorded)**
11. **C049** [positioning/mechanism, check: cell_data] `eurosys:179`: A static per-tier TPOT prior reproduces the headline frontier; the learned head is not needed. **(conflict recorded)**
12. **C075** [state/design, check: code] `neurips_2026.tex:193`: At the start of each round the scheduler fetches instance telemetry (RPC) and seeds the dead-reckoned state from it. **(conflict recorded)**
13. **C076** [state/design, check: code] `eurosys:181`: After each dispatch the chosen instance's pending decode tokens d_i increase by Lhat_{r,m(i)}. Check whether queue depth, b_i, busy_ratio or KV state (inputs to the SQ tiebreak, the adaptive rule and the TPOT features) also change.
14. **C065** [batching/design, check: code] `rebuttal:A1/L16`: busy_ratio = fraction of instances with at least one waiting request. <0.3 gives (size 4, 50 ms), <0.7 gives (16, 100 ms), otherwise (32, 200 ms). **(conflict recorded)**
15. **C072** [lpt/design, check: code] `eurosys:119`: The LPT sort key is the max over models of Lhat_{r,m}, not the mean or min. **(conflict recorded)**
16. **C082** [tiebreak/design, check: code] `eurosys:248`: The scheduler has a built-in within-tier tiebreak: among equal-score instances, pick the shortest queue. With w_lat=0 this decides replica placement. **(conflict recorded)**
17. **C086** [filter_budget/design, check: code] `neurips_2026.tex:183`: At dispatch, max_tokens = (budget - input cost) / output price, and streaming aborts once the running cost exceeds the budget.
18. **C055** [cost/design, check: code] `neurips_2026.tex:154`: The scoring cost term is predicted output tokens times one per-model price: decode-only, input tokens excluded. **(conflict recorded)**
19. **C100** [overhead/eval_setup, check: scripts] `neurips_2026.tex:259`: Residual = client E2E - instance-reported E2E per request, computed by the same script for every system.
20. **C102** [overhead/eval_setup, check: code] `neurips_2026.tex:339`: The 'compute' span = embedding + KNN + XGBoost TPOT + predict round + scoring.
21. **C178** [hosting/eval_setup, check: code] `neurips_2026.tex:210`: Baselines go through the same rounds (same adaptive rule), telemetry fetch and dispatch client as RB. Every remaining execution difference must be listed: estimator batching, LPT, dead reckoning, the state SQ reads, and the scoring thread. **(conflict recorded)**
22. **C113** [hosting/mechanism, check: code] `neurips_2026.tex:265`: In pipeline mode BEST-Route's DeBERTa scorer runs one prompt at a time on the scheduler path, while RB's estimator is batched per round. **(conflict recorded)**
23. **C168** [baselines/eval_setup, check: scripts] `eurosys:207`: BEST-Route's DeBERTa-v3 and Avengers-Pro's clusters were trained on the same 14,919-prompt train split with the same DeepEval labels. **(conflict recorded)**
24. **C152** [quality_metric/eval_setup, check: scripts] `neurips_2026.tex:318`: Unscored (prompt, model) pairs and failed requests are left out of the quality mean. Coverage is about 72-78%; how zero scores are treated is not stated. **(conflict recorded)**
25. **C190** [selection_rule/eval_setup, check: scripts] `outline:L82`: Each system's representative cell at each λ is its argmax of quality × throughput over its own knobs and dispatchers. **(conflict recorded)**
