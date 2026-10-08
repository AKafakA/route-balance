# CARA Objective Function Design: Normalization, Constraints, and Robust Scoring

## Problem

Scoring instances per request requires combining heterogeneous signals: predicted latency (seconds), predicted quality ([0,1]), and cost proxies (tokens, $). Raw weighted sums are brittle because scales and variances differ across models, GPU types, and workloads.

## Goals

- Respect per-request constraints (RSOs) first (latency, quality, budget).
- Normalize signals so they are commensurate and stable across hardware/models.
- Be robust to prediction errors and estimator timeouts.
- Keep the scheduling path fast (greedy mini-batches), with clear fallbacks.

## Signals

- Latency (P̂_ri): predicted server-side E2E for request r on instance i.
- Quality (Q̂_r,model(i)): best-route quality score for the prompt/model on instance i.
- Cost (Ĉ_ri): token-based proxy: (prompt + predicted output tokens) × per-GPU cost factor.
- Uncertainty: σP, σQ, σC from rolling errors (MAE/percentiles) or model APIs.

## Feasibility Layer (Constraint-First)

Compute conservative bounds:

- latency_u = P̂_ri + kσP (upper bound)
- quality_l = Q̂_r,model(i) − kσQ (lower bound)
- cost_u = Ĉ_ri + kσC (upper bound)

Filter infeasible candidates when RSOs exist:

- latency_u ≤ latency_target
- quality_l ≥ quality_min
- cost_u ≤ budget

If none feasible, relax constraints by RSO.relax_order; represent violations with normalized magnitudes and penalize heavily (see Scalarization).

## Normalization (pick one, keep consistent)

All produce ∈ [0,1]-ish, lower is better (for latency/cost) and higher is better for quality.

1) Per-request min–max (simple, per-batch stable):

- lat_norm = (latency_u − min_latency_u) / (max_latency_u − min_latency_u + ε)
- cost_norm = (cost_u − min_cost_u) / (max_cost_u − min_cost_u + ε)
- qual_norm = 1 − (quality_l − min_quality_l) / (max_quality_l − min_quality_l + ε)

2) Percentile (CDF) per (model or node-type):

- lat_pct = F_lat_i(latency_u), cost_pct = F_cost_i(cost_u), qual_pct = 1 − F_qual_i(quality_l)
- Pros: robust across heterogeneous hardware; requires rolling histograms.

3) Regret vs. best candidate:

- lat_regret = (latency_u − best_latency_u) / (best_latency_u + ε)
- cost_regret = (cost_u − best_cost_u) / (best_cost_u + ε)
- qual_regret = (best_quality_l − quality_l) / (best_quality_l + ε)

## Scalarization (Score)

Base score (lower is better):

- score = α·lat_term + β·cost_term + γ·qual_term  (qual_term is defined so higher true quality reduces score)

Constraint penalties (Lagrangian style):

- viol_lat = max(0, (latency_u − latency_target)/latency_target)
- viol_qual = max(0, (quality_min − quality_l)/quality_min)
- viol_cost = max(0, (cost_u − budget)/budget)
- score += M_lat·viol_lat + M_qual·viol_qual + M_cost·viol_cost with large M_* to dominate base terms.

Lexicographic alternative (aligns with relax_order):

- Sort by tuple (viol_first, viol_second, viol_third, base_score), where violations are normalized magnitudes. Avoids weight tuning when RSOs dictate priority.

## Uncertainty Handling

- Conservative bounds via kσ (UCB/LCB) as above.
- Risk-aware weights: increase α when σP is high; increase γ when σQ is high.
- Adapt k using online calibration (rolling MAE/percentiles).

## Candidate Pruning

- Pareto prune on (lat_term, qual_term, cost_term) per request; discard dominated candidates.
- Top‑K preselection (e.g., smallest latency_u) to bound best-route calls.

## Quality Model Integration (Best‑Route)

- Calibrate outputs to [0,1] (probabilities via temperature scaling or per-request softmax over candidates).
- Timeouts → per-model priors from config.
- Cache quality by (model, truncated_prompt_hash) with TTL to amortize calls.

## Config (scheduler_config.json)

```
{
  "normalization": "minmax",            // or "percentile", "regret"
  "alpha": 1.0, "beta": 0.1, "gamma": 0.5,
  "penalties": {"lat": 10.0, "qual": 10.0, "cost": 10.0},
  "k_sigma": 1.0,
  "lexicographic": true,
  "top_k_models": 3
}
```

## Integration Points

- `cara_serve.py` decision path:
  - Gather P̂_ri (predictor sidecar), Q̂_r,model (best‑route), Ĉ_ri (proxy).
  - Normalize + penalize via a `ScoreNormalizer` and `Scorer` module.
  - Feed to `scheduler.select_assignment` for greedy mini‑batch placement.

## Reference Pseudocode

```
def score_candidates(request, instances, preds, quality, cost, rso, cfg):
    # Step 1: conservative bounds
    lat_u, qual_l, cost_u = bound_with_sigma(preds, quality, cost, cfg.k_sigma)

    # Step 2: feasibility filter
    feas = [i for i in instances if satisfies_rso(lat_u[i], qual_l[i], cost_u[i], rso)]
    if not feas:
        viol = compute_violations(lat_u, qual_l, cost_u, rso)
    else:
        viol = {i: (0,0,0) for i in instances}

    # Step 3: normalization
    lat_term, qual_term, cost_term = normalize(lat_u, qual_l, cost_u, cfg.normalization)

    # Step 4: scalarization
    base = cfg.alpha*lat_term + cfg.beta*cost_term + cfg.gamma*qual_term
    if cfg.lexicographic and rso.relax_order:
        score = tuple_from_relax_order(viol, base, rso.relax_order)
    else:
        score = base + penalties_dot(viol, cfg.penalties)
    return argmin(score)
```

## Recommended Defaults

- Normalization: per-request min–max.
- Penalties M_* = 10.0; k_sigma = 1.0.
- Weights: α=1.0, β=0.1, γ=0.5 (calibrate later).
- Lexicographic: enabled when RSO.relax_order present.

## Evaluation Checklist

- Sensitivity: α/β/γ, normalization modes, k_sigma, penalties.
- Robustness: inject noise into P̂ and Q̂; constraint satisfaction rates.
- Efficiency: estimator Top‑K and caching hit rates; scheduling latency per mini‑batch.
- End‑to‑end: p50/p95 latency, throughput, quality‑weighted throughput, scheduling overhead under mixed RSOs.

## Bucket-Based Budget Filtering (Cost Feasibility)

**Problem:** Pointwise output length prediction has ~100% MAPE for some models, making direct cost filtering unreliable. A predicted length of 347 tokens that's actually 700 tokens would wrongly pass a budget check.

**Solution:** Use the bucket classifier's probability distribution over output length buckets instead of a single point estimate. The bucket classifier (ModernBERT with 16 × 64-token buckets) outputs P(bucket_k | prompt, model) for k ∈ {0..15}.

**Algorithm:**

1. Given request budget B and per-model token pricing (input_price, output_price):
   ```
   max_affordable_output = (B - input_tokens * input_price) / output_price
   max_affordable_bucket = floor(max_affordable_output / bucket_size)
   ```

2. Compute cumulative probability of staying within budget:
   ```
   P(affordable | prompt, model) = sum(P(bucket_k) for k <= max_affordable_bucket)
   ```

3. Filter: if P(affordable) < confidence_threshold (e.g., 0.70), mark model as infeasible for this request's budget constraint.

**Why this works:** Even with 42% exact bucket accuracy, the probability mass in adjacent buckets is meaningful. The CDF estimate is much more reliable than a point prediction for binary "over/under budget" decisions. A 70% confidence threshold tolerates prediction noise while catching clearly unaffordable pairings.

**Integration:** This replaces the simple `cost_u ≤ budget` check in the feasibility layer when bucket predictions are available. Falls back to point-estimate cost check otherwise.

## Latency SLO Decomposition: TTFT + TPOT

**Problem:** E2E latency = TTFT + (output_tokens × TPOT). Predicting E2E directly requires knowing output length, which has high MAPE. This makes E2E latency SLO checking unreliable.

**Solution:** Decompose latency SLO into separate TTFT and TPOT constraints:

- **TTFT SLO:** Depends on prompt length and queue state — predictable without output length.
- **TPOT SLO:** Depends on concurrent load and model/GPU — also predictable without output length.

The XGBoost predictor trains separate models for each target (E2E, TTFT, TPOT) using the same instance state features. TTFT and TPOT predictions avoid the output length dependency that makes E2E prediction noisy.

**RSO extension:**
```
rso.latency_target    → E2E bound (optional, used when length prediction is good)
rso.ttft_target       → TTFT bound (ms)
rso.tpot_target       → TPOT bound (ms/token)
```

**Feasibility check:** A candidate is feasible if predicted TTFT ≤ ttft_target AND predicted TPOT ≤ tpot_target. When E2E latency_target is also set, check both (conservative).

## LPT Sort Key in Heterogeneous Clusters (Design Decision, March 2026)

In a heterogeneous cluster, different models produce different output lengths for the same prompt. The LPT sort key must be defined before the routing decision (which model to use) is made.

**Decision:** Use `max_m E[len_m(r)]` — the maximum predicted output length across all candidate models.

**Rationale:**
- **Conservative:** Ensures the request with the highest *potential* processing cost is scheduled first, preserving the LPT invariant regardless of which instance it ultimately lands on.
- **Model-independent:** The sort key doesn't depend on the scoring function or constraint filtering, so it can be computed in O(|M|) per request before the O(|I|) scoring loop.
- **Practical:** In our 4-model cluster (3B/7B/14B/72B), larger models typically produce longer outputs. Using max captures the worst-case load contribution.

**Alternatives considered:**
- *Weighted average by model likelihood:* More accurate but couples LPT ordering with the scoring function. The "likelihood" of each model isn't known until after scoring — circular dependency.
- *Target model only:* Requires a pre-filter step to determine the most likely model. Adds complexity and the pre-filter may be wrong.
- *Mean across models:* Underestimates for prompts with high variance across models. Max is safer.

**Implementation:** `ModelEstimator.estimate()` returns per-model `length_expected`. The batch loop computes `req.predicted_tokens = max(est.length_expected for est in req.model_estimates.values())`.

