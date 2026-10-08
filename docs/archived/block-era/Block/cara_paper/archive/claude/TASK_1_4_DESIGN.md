# Task 1.4: Multi-Objective Scoring with RSO/SLO - Design Document

**Date**: January 26, 2026
**Status**: ✅ IMPLEMENTATION COMPLETE

> **Implementation Summary** (January 26, 2026):
> - RSO and SLO data structures implemented in `rso.py`
> - MultiObjectiveScorer with tiered constraint handling in `multi_objective_scorer.py`
> - Placeholder estimators for cost and quality
> - TTFT/TPOT derivation from service rates
> - Full integration with CARABatchPolicy via ScoringConfig
> - 23 unit tests for multi-objective scoring
> - All 166 CARA predictor tests passing

---

## Overview

This task implements the full multi-objective scoring system for CARA, including:
- Request-Specific Objectives (RSO) support
- Service-Level Objectives (SLO) support
- Tiered constraint handling (not simple penalties)
- Output length and quality estimators (KNN-based, placeholder for now)
- Cost estimation based on token prices
- Derived TTFT/TPOT from service rates
- RSO assignment for benchmarking

---

## 1. RSO (Request-Specific Objectives)

### 1.1 Schema

RSO fields are **optional** - if not provided, no constraint checked for that dimension.

```python
@dataclass
class RequestSpecificObjective:
    """Per-request constraints and objectives."""
    latency_ms: Optional[float] = None      # Max acceptable E2E latency (ms)
    ttft_ms: Optional[float] = None         # Max acceptable TTFT (ms)
    tpot_ms: Optional[float] = None         # Max acceptable TPOT (ms)
    quality_min: Optional[float] = None     # Min quality score [0-1]
    token_budget: Optional[int] = None      # Max cost in tokens
```

### 1.2 Integration with PredictRequest

Add RSO to existing `PredictRequest`:

```python
@dataclass
class PredictRequest:
    request_id: str
    num_prompt_tokens: int
    num_predicted_output_tokens: int
    # NEW: RSO (optional)
    rso: Optional[RequestSpecificObjective] = None
```

---

## 2. SLO (Service-Level Objectives)

### 2.1 Configuration

SLOs are cluster-wide constraints, configured via `SchedulerConfig`:

```python
@dataclass
class SLOConfig:
    """Cluster-wide service level objectives."""
    latency_ms: Optional[float] = None      # Max E2E latency SLO
    ttft_ms: Optional[float] = None         # Max TTFT SLO
    tpot_ms: Optional[float] = None         # Max TPOT SLO

    # Tier weight: SLO violation counts as weight * 1 tier
    # Default 1.0 = same as RSO, set to 2.0 to make SLO violations worse
    tier_weight: float = 1.0
```

### 2.2 RSO vs SLO Tier Weighting

```python
# Example: RSO tier weight = 1, SLO tier weight = 2
# Instance A: violates 1 RSO → effective_tier = 1 * 1 = 1
# Instance B: violates 1 SLO → effective_tier = 1 * 2 = 2
# Instance A is preferred (lower tier)
```

---

## 3. Estimators

### 3.1 Pipeline Order

```
Request arrives
    │
    ▼
┌─────────────────────────────┐
│ Output Length Estimator     │  (request, model) → predicted_output_tokens
│ (KNN-based, placeholder)    │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Quality Estimator           │  (request, model) → quality_score [0-1]
│ (KNN-based, placeholder)    │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Cost Estimator              │  (request, model) → cost_tokens
│ (Token-based pricing)       │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Latency Predictor           │  (request, state) → E2E latency
│ (Existing XGBoost/etc)      │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ TTFT/TPOT Derivation        │  Derived from service rates
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Multi-Objective Scorer      │  Tiered constraint handling + scoring
└─────────────────────────────┘
```

### 3.2 Output Length Estimator (Placeholder)

```python
class BaseOutputLengthEstimator(ABC):
    """Abstract base class for output length estimation.

    Estimates expected output length for a request on a given model.
    This is called BEFORE the latency predictor to fill the
    `num_predicted_output_tokens` field.

    Implementation Options (Future Work):
    1. KNN-based: Embed prompt, find k nearest neighbors in training data,
       return weighted average of their output lengths.
    2. Regression: Frozen embeddings + MLP/GBDT regressor.
    3. Quantile regression: For conservative bounds (Q90/Q95).

    See: cara_paper/codex/cara-data-and-experiments.md
    """

    @abstractmethod
    def predict(
        self,
        prompt: str,
        model_id: str
    ) -> Tuple[float, float]:
        """Predict output length.

        Args:
            prompt: Input prompt text
            model_id: Model identifier

        Returns:
            Tuple of (mean_output_tokens, p95_output_tokens)
        """
        pass


class PlaceholderOutputLengthEstimator(BaseOutputLengthEstimator):
    """Placeholder that returns configurable defaults.

    TODO: Implement KNN-based estimator using prompt embeddings.
    Training data: cara-best-route dataset with actual output lengths.
    """

    def __init__(self, default_mean: int = 256, default_p95: int = 512):
        self.default_mean = default_mean
        self.default_p95 = default_p95

    def predict(self, prompt: str, model_id: str) -> Tuple[float, float]:
        return (self.default_mean, self.default_p95)
```

### 3.3 Quality Estimator (Placeholder)

```python
class BaseQualityEstimator(ABC):
    """Abstract base class for quality estimation.

    Estimates expected quality score for a request on a given model.
    Quality is defined as similarity to the best available model's response.

    Implementation Options (Future Work):
    1. KNN-based: Embed prompt, find k nearest neighbors, return
       distance-weighted average of their per-model quality scores.
    2. Lightweight encoder head: Frozen embeddings + small MLP per model.

    See: cara_paper/codex/cara-data-and-experiments.md
    """

    @abstractmethod
    def predict(
        self,
        prompt: str,
        model_id: str
    ) -> float:
        """Predict quality score.

        Args:
            prompt: Input prompt text
            model_id: Model identifier

        Returns:
            Quality score in [0, 1]
        """
        pass


class PlaceholderQualityEstimator(BaseQualityEstimator):
    """Placeholder that returns model-size-based quality estimates.

    TODO: Implement KNN-based estimator using prompt embeddings.
    Training data: cara-best-route dataset with LLM-judge quality scores.
    """

    # Default quality by model size (larger = higher quality)
    DEFAULT_QUALITY = {
        "72b": 1.0,
        "32b": 0.95,
        "14b": 0.85,
        "7b": 0.75,
        "3b": 0.65,
    }

    def predict(self, prompt: str, model_id: str) -> float:
        model_lower = model_id.lower()
        for size, quality in self.DEFAULT_QUALITY.items():
            if size in model_lower:
                return quality
        return 0.7  # Default
```

### 3.4 Cost Estimator

```python
class CostEstimator:
    """Token-based cost estimation.

    Cost = input_tokens * input_price + output_tokens * output_price

    Prices loaded from model_deployment.json:
    {
        "Qwen-2.5-3B": {
            "input_token_price": 0.0001,   // $/1K tokens
            "output_token_price": 0.0002,  // $/1K tokens
            ...
        }
    }
    """

    def __init__(self, model_prices: Dict[str, Dict[str, float]]):
        self.model_prices = model_prices

    def predict(
        self,
        num_input_tokens: int,
        num_output_tokens: int,
        model_id: str
    ) -> float:
        """Estimate cost in token units.

        Returns cost normalized to "equivalent tokens" for fair comparison:
        cost_tokens = input_tokens + output_tokens * (output_price / input_price)
        """
        prices = self.model_prices.get(model_id, {})
        input_price = prices.get("input_token_price", 1.0)
        output_price = prices.get("output_token_price", 1.0)

        # Normalize to input token equivalents
        output_ratio = output_price / input_price if input_price > 0 else 1.0
        return num_input_tokens + num_output_tokens * output_ratio
```

### 3.5 TTFT/TPOT Derivation

```python
def derive_ttft_tpot(
    num_prompt_tokens: int,
    num_output_tokens: int,
    e2e_latency_ms: float,
    prefill_rate: float,  # tokens/sec from ServiceRateTracker
    decode_rate: float,   # tokens/sec from ServiceRateTracker
) -> Tuple[float, float]:
    """Derive TTFT and TPOT from service rates.

    TTFT ≈ prompt_tokens / prefill_rate
    TPOT ≈ (E2E - TTFT) / output_tokens

    Future Work: Train multi-output model that directly predicts
    (E2E_latency, TTFT, TPOT) for better accuracy.
    See: cara_paper/codex/cara-data-and-experiments.md

    Args:
        num_prompt_tokens: Number of input tokens
        num_output_tokens: Number of output tokens
        e2e_latency_ms: Predicted E2E latency in ms
        prefill_rate: Prefill throughput (tokens/sec)
        decode_rate: Decode throughput (tokens/sec)

    Returns:
        Tuple of (ttft_ms, tpot_ms)
    """
    # TTFT from prefill rate
    ttft_ms = (num_prompt_tokens / prefill_rate) * 1000 if prefill_rate > 0 else 0

    # TPOT from remaining time
    decode_time_ms = e2e_latency_ms - ttft_ms
    tpot_ms = decode_time_ms / num_output_tokens if num_output_tokens > 0 else 0

    return (max(0, ttft_ms), max(0, tpot_ms))
```

---

## 4. Tiered Constraint Handling

### 4.1 Constraint Checking

```python
@dataclass
class ConstraintViolation:
    """A single constraint violation."""
    constraint_type: str    # "rso_latency", "rso_quality", "slo_latency", etc.
    violation_amount: float # How much the constraint is violated by
    tier_weight: float      # Weight for tier calculation (RSO=1, SLO=configurable)


def check_constraints(
    request: PredictRequest,
    prediction: Dict,
    quality: float,
    cost: float,
    ttft_ms: float,
    tpot_ms: float,
    slo_config: SLOConfig,
) -> List[ConstraintViolation]:
    """Check all RSO and SLO constraints.

    Returns list of violations (empty = all satisfied).
    """
    violations = []
    rso = request.rso

    # RSO constraints (tier_weight = 1.0)
    if rso:
        if rso.latency_ms and prediction['latency'] > rso.latency_ms:
            violations.append(ConstraintViolation(
                "rso_latency",
                prediction['latency'] - rso.latency_ms,
                tier_weight=1.0
            ))
        if rso.ttft_ms and ttft_ms > rso.ttft_ms:
            violations.append(ConstraintViolation(
                "rso_ttft",
                ttft_ms - rso.ttft_ms,
                tier_weight=1.0
            ))
        if rso.tpot_ms and tpot_ms > rso.tpot_ms:
            violations.append(ConstraintViolation(
                "rso_tpot",
                tpot_ms - rso.tpot_ms,
                tier_weight=1.0
            ))
        if rso.quality_min and quality < rso.quality_min:
            violations.append(ConstraintViolation(
                "rso_quality",
                rso.quality_min - quality,
                tier_weight=1.0
            ))
        if rso.token_budget and cost > rso.token_budget:
            violations.append(ConstraintViolation(
                "rso_budget",
                cost - rso.token_budget,
                tier_weight=1.0
            ))

    # SLO constraints (tier_weight = configurable, default 1.0)
    slo_weight = slo_config.tier_weight
    if slo_config.latency_ms and prediction['latency'] > slo_config.latency_ms:
        violations.append(ConstraintViolation(
            "slo_latency",
            prediction['latency'] - slo_config.latency_ms,
            tier_weight=slo_weight
        ))
    if slo_config.ttft_ms and ttft_ms > slo_config.ttft_ms:
        violations.append(ConstraintViolation(
            "slo_ttft",
            ttft_ms - slo_config.ttft_ms,
            tier_weight=slo_weight
        ))
    if slo_config.tpot_ms and tpot_ms > slo_config.tpot_ms:
        violations.append(ConstraintViolation(
            "slo_tpot",
            tpot_ms - slo_config.tpot_ms,
            tier_weight=slo_weight
        ))

    return violations
```

### 4.2 Tier Calculation

```python
def compute_effective_tier(violations: List[ConstraintViolation]) -> float:
    """Compute effective tier from violations.

    Tier = sum of (tier_weight) for each violation.
    This allows SLO violations to count more than RSO violations.

    Examples:
    - No violations → tier = 0
    - 1 RSO violation (weight=1) → tier = 1
    - 1 SLO violation (weight=2) → tier = 2
    - 1 RSO + 1 SLO (weights 1+2) → tier = 3
    """
    return sum(v.tier_weight for v in violations)
```

### 4.3 Scoring Function

```python
@dataclass
class CandidateScore:
    """Score for a candidate instance."""
    instance_id: str
    model_id: str

    # Predictions
    latency_ms: float
    ttft_ms: float
    tpot_ms: float
    quality: float
    cost_tokens: float
    memory_util: float

    # Constraint handling
    violations: List[ConstraintViolation]
    effective_tier: float

    # Normalized scores (for weighted sum)
    lat_norm: float = 0.0
    cost_norm: float = 0.0
    qual_norm: float = 0.0  # Note: higher quality = lower normalized score
    mem_norm: float = 0.0

    # Final score (lower is better)
    base_score: float = 0.0
    final_score: float = 0.0


def compute_scores(
    candidates: List[CandidateScore],
    config: ScoringConfig
) -> None:
    """Compute normalized scores and final scores for all candidates.

    Modifies candidates in-place.
    """
    if not candidates:
        return

    # Min-max normalization (per-batch)
    eps = 1e-9
    lat_min = min(c.latency_ms for c in candidates)
    lat_max = max(c.latency_ms for c in candidates)
    cost_min = min(c.cost_tokens for c in candidates)
    cost_max = max(c.cost_tokens for c in candidates)
    qual_min = min(c.quality for c in candidates)
    qual_max = max(c.quality for c in candidates)
    mem_min = min(c.memory_util for c in candidates)
    mem_max = max(c.memory_util for c in candidates)

    for c in candidates:
        # Normalize (lower is better for all)
        c.lat_norm = (c.latency_ms - lat_min) / (lat_max - lat_min + eps)
        c.cost_norm = (c.cost_tokens - cost_min) / (cost_max - cost_min + eps)
        c.qual_norm = 1.0 - (c.quality - qual_min) / (qual_max - qual_min + eps)  # Invert
        c.mem_norm = (c.memory_util - mem_min) / (mem_max - mem_min + eps)

        # Weighted sum
        c.base_score = (
            config.w_latency * c.lat_norm +
            config.w_cost * c.cost_norm +
            config.w_quality * c.qual_norm +
            config.w_memory * c.mem_norm
        )

        # Final score includes tier for sorting
        c.final_score = c.base_score
```

### 4.4 Selection Modes

```python
class ConstraintMode(Enum):
    STRICT = "strict"    # Reject if no tier-0 candidate
    RELAXED = "relaxed"  # Accept best available
    TIERED = "tiered"    # Try tier-0, then tier-1, etc.


def select_best_instance(
    candidates: List[CandidateScore],
    mode: ConstraintMode
) -> Optional[str]:
    """Select best instance using tiered constraint handling.

    Sorting: (effective_tier, final_score) - lower is better
    """
    if not candidates:
        return None

    # Sort by tier first, then by score
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (c.effective_tier, c.final_score)
    )

    if mode == ConstraintMode.STRICT:
        # Only accept tier 0
        tier0 = [c for c in sorted_candidates if c.effective_tier == 0]
        return tier0[0].instance_id if tier0 else None

    elif mode == ConstraintMode.RELAXED:
        # Accept best available
        return sorted_candidates[0].instance_id

    elif mode == ConstraintMode.TIERED:
        # Try each tier in order
        max_tier = max(c.effective_tier for c in sorted_candidates)
        for target_tier in range(int(max_tier) + 1):
            tier_candidates = [c for c in sorted_candidates
                             if int(c.effective_tier) == target_tier]
            if tier_candidates:
                return tier_candidates[0].instance_id
        return None
```

---

## 5. Configuration Updates

### 5.1 model_deployment.json

Add pricing fields:

```json
{
  "Qwen-2.5-3B": {
    "hf_model_name": "Qwen/Qwen2.5-3B",
    "precision": "fp16",
    "backend": "vllm",
    "node_hosts": [...],
    "input_token_price": 0.0001,
    "output_token_price": 0.0002
  }
}
```

### 5.2 Instance Metadata

Add `model_id` field:

```python
# In cara_serve.py, when building instance_type_map
instance_metadata[instance_id] = {
    "instance_type": inst_type,
    "model_id": hf_model_name,  # NEW
}
```

### 5.3 ScoringConfig (in cara_predictor_config.py)

```python
@dataclass
class ScoringConfig:
    """Configuration for multi-objective scoring."""

    # Objective weights (lower score is better)
    w_latency: float = 1.0
    w_cost: float = 0.1
    w_quality: float = 0.0   # 0 = disabled
    w_memory: float = 0.1

    # Normalization method
    normalization: str = "minmax"  # "minmax", "regret"

    # Constraint handling mode
    constraint_mode: str = "relaxed"  # "strict", "relaxed", "tiered"

    # SLO configuration
    slo_latency_ms: Optional[float] = None
    slo_ttft_ms: Optional[float] = None
    slo_tpot_ms: Optional[float] = None
    slo_tier_weight: float = 1.0  # SLO violation tier weight
```

---

## 6. RSO Assignment for Benchmarking

### 6.1 RSO Assignment Config

```python
@dataclass
class RSOAssignmentConfig:
    """Configuration for random RSO assignment in benchmarks."""

    # Fraction of requests that get each RSO type
    latency_fraction: float = 0.0     # 0 = no latency RSO
    ttft_fraction: float = 0.0
    tpot_fraction: float = 0.0
    quality_fraction: float = 0.0
    budget_fraction: float = 0.0

    # Allow overlapping RSOs on same request
    allow_overlap: bool = True

    # Budget ranges for random assignment
    latency_range_ms: Tuple[float, float] = (200, 2000)
    ttft_range_ms: Tuple[float, float] = (50, 500)
    tpot_range_ms: Tuple[float, float] = (10, 100)
    quality_range: Tuple[float, float] = (0.5, 0.95)
    token_budget_range: Tuple[int, int] = (500, 5000)
```

### 6.2 RSO Assigner Class

```python
class RSOAssigner:
    """Assigns random RSOs to requests for benchmarking.

    Usage:
        assigner = RSOAssigner(config)
        for request in requests:
            request.rso = assigner.assign(request)
    """

    def __init__(self, config: Optional[RSOAssignmentConfig] = None):
        self.config = config

    def assign(self, request: dict) -> Optional[RequestSpecificObjective]:
        """Assign RSO to request based on config.

        Returns None if config is None or no RSO assigned by random chance.
        """
        if self.config is None:
            return None

        rso = RequestSpecificObjective()
        any_assigned = False

        cfg = self.config

        if random.random() < cfg.latency_fraction:
            rso.latency_ms = random.uniform(*cfg.latency_range_ms)
            any_assigned = True
            if not cfg.allow_overlap:
                return rso

        if random.random() < cfg.ttft_fraction:
            rso.ttft_ms = random.uniform(*cfg.ttft_range_ms)
            any_assigned = True
            if not cfg.allow_overlap:
                return rso

        # ... similar for other RSO types

        return rso if any_assigned else None
```

---

## 7. File Organization

```
block/predictor/cara/
├── multi_objective_scorer.py      # Tiered scoring, CandidateScore, selection
├── estimators/
│   ├── __init__.py
│   ├── base_estimator.py          # ABC for estimators
│   ├── output_length_estimator.py # Placeholder + future KNN
│   ├── quality_estimator.py       # Placeholder + future KNN
│   └── cost_estimator.py          # Token-based pricing
├── rso.py                         # RSO dataclass, constraint checking
├── rso_assignment.py              # RSO injection for benchmarks
└── ttft_tpot_derivation.py        # Derive TTFT/TPOT from rates
```

---

## 8. Integration with CARABatchPolicy

```python
class CARABatchPolicy(BaseSchedulingPolicy):
    def __init__(self, ...):
        # ... existing init ...

        # NEW: Initialize estimators and scorer
        self.output_length_estimator = PlaceholderOutputLengthEstimator()
        self.quality_estimator = PlaceholderQualityEstimator()
        self.cost_estimator = CostEstimator(model_prices)
        self.scorer = MultiObjectiveScorer(scoring_config)

    async def _select_best_instance(
        self,
        request: PredictRequest,
        instance_ids: List[str]
    ) -> str:
        """Select best instance using multi-objective scoring."""

        candidates = []

        for inst_id in instance_ids:
            state = self.state_manager.get_state(inst_id)
            model_id = self.instance_model_map.get(inst_id)

            # 1. Estimate output length (if not provided)
            if request.num_predicted_output_tokens == 0:
                mean_len, _ = self.output_length_estimator.predict(
                    request.prompt, model_id
                )
                request.num_predicted_output_tokens = int(mean_len)

            # 2. Estimate quality
            quality = self.quality_estimator.predict(request.prompt, model_id)

            # 3. Estimate cost
            cost = self.cost_estimator.predict(
                request.num_prompt_tokens,
                request.num_predicted_output_tokens,
                model_id
            )

            # 4. Predict latency
            prediction = await self.predictor_registry.predict(
                instance_type=self.instance_type_map[inst_id],
                state=state,
                request=request
            )

            # 5. Derive TTFT/TPOT
            ttft, tpot = derive_ttft_tpot(
                request.num_prompt_tokens,
                request.num_predicted_output_tokens,
                prediction['latency'],
                state.ema_prefill_rate,
                state.ema_decode_rate
            )

            # 6. Build candidate
            candidates.append(CandidateScore(
                instance_id=inst_id,
                model_id=model_id,
                latency_ms=prediction['latency'],
                ttft_ms=ttft,
                tpot_ms=tpot,
                quality=quality,
                cost_tokens=cost,
                memory_util=state.kv_cache_utilization,
                violations=[],
                effective_tier=0
            ))

        # 7. Check constraints and compute scores
        self.scorer.score_candidates(candidates, request)

        # 8. Select best
        return self.scorer.select_best(candidates)
```

---

## 9. Acceptance Criteria

- [ ] RSO dataclass with optional latency/ttft/tpot/quality/budget fields
- [ ] SLO config in SchedulerConfig
- [ ] Tiered constraint handling with configurable tier weights
- [ ] Placeholder output length estimator with clear documentation
- [ ] Placeholder quality estimator with clear documentation
- [ ] Token-based cost estimator using model_deployment.json prices
- [ ] TTFT/TPOT derivation from service rates
- [ ] Multi-objective scorer with min-max normalization
- [ ] Selection modes: STRICT, RELAXED, TIERED
- [ ] RSO assigner class for benchmarking
- [ ] Integration with CARABatchPolicy
- [ ] Unit tests for all components
- [ ] model_deployment.json updated with price fields
- [ ] Instance metadata includes model_id
