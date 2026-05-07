"""Ground truth extraction for ROUTE_BALANCE predictor evaluation."""

from typing import Dict, Optional


def _get_judge_score(m_data: dict, is_harmful: bool = False,
                     judge_key: str = None) -> float:
    """Extract a specific judge score.

    Args:
        m_data: Model data dict with llm_judge_scores.
        is_harmful: If True, use protectai safety score.
        judge_key: Specific judge key to use (e.g., "Qwen_Qwen2.5-7B-Instruct").
            If None, uses first Qwen judge found (backward compat).
    """
    judge_scores = m_data.get("llm_judge_scores", {})
    if is_harmful:
        return float(judge_scores.get("protectai_distilroberta-base-rejection-v1", 0.0) or 0.0)

    # Use specific key if provided
    if judge_key and judge_key in judge_scores:
        return float(judge_scores[judge_key] or 0.0)

    # Default: Qwen blind judge (first Qwen key, not reference/deepeval)
    for k, v in judge_scores.items():
        if "Qwen" in k and v is not None and "protectai" not in k:
            return float(v)
    return 0.0


def get_ground_truth(req: dict, model: str, target: str) -> Optional[float]:
    """Extract ground truth value for a request-model-target triple.

    Args:
        req: Full request dict (with 'models', 'is_harmful', etc.)
        model: Model name (e.g., 'Qwen/Qwen2.5-7B')
        target: One of 'length', 'length_bucket', 'similarity', 'judge',
                'reference_score', 'deepeval', 'prometheus'

    Returns:
        Ground truth float, or None if not available.
    """
    m_data = req.get("models", {}).get(model, {})
    if not m_data:
        return None

    if target in ("length", "length_bucket"):
        val = m_data.get("output_length", 0)
        return float(val) if val else None
    elif target == "similarity":
        val = m_data.get("similarity_score")
        return float(val) if val is not None else None
    elif target == "judge":
        is_harmful = req.get("is_harmful", False)
        return _get_judge_score(m_data, is_harmful)
    elif target == "reference_score":
        val = m_data.get("reference_score")
        return float(val) if val is not None else None
    elif target == "deepeval":
        judge_scores = m_data.get("llm_judge_scores", {})
        val = judge_scores.get("deepeval-llama3.1-8b-it_reference")
        return float(val) if val is not None else None
    elif target == "prometheus":
        judge_scores = m_data.get("llm_judge_scores", {})
        # Try standard reference-grounded keys
        for key in ["prometheus-7b-v2_reference", "qwen2.5-7b-it_reference"]:
            if key in judge_scores:
                return float(judge_scores[key])
        # Legacy: standalone prometheus_score field
        val = m_data.get("prometheus_score")
        return float(val) if val is not None else None
    else:
        return None
