"""
ROUTE_BALANCE Offline Training - Data Preparation and Model Training

Tools for preparing training data and training ML models for ROUTE_BALANCE predictor.

Apr 27: eager imports removed from __init__ — they pulled in torch
(via similarity_scorer + llm_judge_scorer) at package import time, which
breaks bench-only entry points on nodes without GPU torch (e.g. P100 nodes
that lack libcudnn.so.9). Importers that need the scorers should import
them directly: `from route_balance.predictor.route_balance.offline_training.similarity_scorer
import SimilarityScorer`. Lazy __getattr__ below preserves
`from route_balance.predictor.route_balance.offline_training import SimilarityScorer` style.
"""

__all__ = [
    "ModelScorer",
    "SimilarityScorer",
    "LLMJudgeScorer",
    "ResponseFilter",
    "ModelResponse",
]


def __getattr__(name):
    """Lazy-load scorer classes only when requested as attributes."""
    if name == "ModelScorer":
        from route_balance.predictor.route_balance.offline_training.model_scorer import ModelScorer
        return ModelScorer
    if name == "SimilarityScorer":
        from route_balance.predictor.route_balance.offline_training.similarity_scorer import SimilarityScorer
        return SimilarityScorer
    if name == "LLMJudgeScorer":
        from route_balance.predictor.route_balance.offline_training.llm_judge_scorer import LLMJudgeScorer
        return LLMJudgeScorer
    if name in ("ResponseFilter", "ModelResponse"):
        from route_balance.predictor.route_balance.offline_training.response_filter import ResponseFilter, ModelResponse
        return {"ResponseFilter": ResponseFilter, "ModelResponse": ModelResponse}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
