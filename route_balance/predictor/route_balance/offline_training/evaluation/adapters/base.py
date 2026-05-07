"""Base adapter interface for predictor evaluation."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class BaseAdapter(ABC):
    """Base class for predictor adapters.

    All adapters implement predict() for point predictions.
    Bucket-based adapters also implement predict_probs() for distributions.
    """

    @abstractmethod
    def predict(self, prompt: str, target_models: list) -> Dict[str, float]:
        """Point prediction for each target model.

        Returns: {model_name: predicted_value}
        """

    def predict_probs(self, prompt: str, target_models: list) -> Optional[Dict[str, np.ndarray]]:
        """Probability distribution prediction (bucket adapters only).

        Returns: {model_name: (num_buckets,) array} or None if not supported.
        """
        return None

    @property
    def supports_probs(self) -> bool:
        """Whether this adapter supports probability distribution predictions."""
        return False
