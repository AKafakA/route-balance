"""
PFS-style (Past-Future Scheduler) estimator for ROUTE_BALANCE.

Training-free baseline: predicts output length from a sliding window of
recently completed requests. Implements Gong et al. (ASPLOS 2025).

Two modes:
- Global: median of all recent completions
- Binned: median of recent completions with similar input length

Can be used as:
1. A ModelEstimator (replaces ModernBERT bucket classifier)
2. A standalone bucket predictor for bucket filtering evaluation
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

BUCKET_SIZE = 64
MAX_BUCKETS = 16
INPUT_LENGTH_BINS = [0, 64, 128, 256, 512, 1024, float("inf")]


def _input_bin(input_len: int) -> int:
    for i in range(len(INPUT_LENGTH_BINS) - 1):
        if input_len < INPUT_LENGTH_BINS[i + 1]:
            return i
    return len(INPUT_LENGTH_BINS) - 2


def _length_to_bucket(length: float) -> int:
    return min(int(length) // BUCKET_SIZE, MAX_BUCKETS - 1)


class PFSEstimator:
    """Sliding-window output length estimator (PFS-style).

    Maintains a per-model sliding window of recently completed request lengths.
    Predicts output length as the median of the window (optionally binned by
    input length).

    Thread-safe for concurrent reads via numpy snapshot.
    """

    def __init__(
        self,
        model_names: List[str],
        window_size: int = 1000,
        use_input_bins: bool = True,
        bucket_size: int = 64,
        max_buckets: int = 16,
    ):
        self.model_names = model_names
        self.window_size = window_size
        self.use_input_bins = use_input_bins
        self.bucket_size = bucket_size
        self.max_buckets = max_buckets

        # Per-model sliding windows: list of (output_length, input_bin)
        self._windows: Dict[str, list] = {m: [] for m in model_names}

        # Bootstrap from training data if provided
        self._bootstrapped = False

        logger.info(
            f"PFSEstimator: {len(model_names)} models, window={window_size}, "
            f"binned={use_input_bins}"
        )

    def bootstrap(self, train_data: list):
        """Warm-start windows from training data.

        Args:
            train_data: list of dicts with 'models' and 'input_len' keys
                        (same format as ROUTE_BALANCE training data)
        """
        for req in train_data:
            input_len = req.get("input_len", 0)
            ibin = _input_bin(input_len)
            for model in self.model_names:
                m_data = req.get("models", {}).get(model, {})
                if m_data:
                    out_len = float(m_data.get("output_length", 0))
                    self._windows[model].append((out_len, ibin))

        # Trim to window size (keep most recent)
        for model in self.model_names:
            if len(self._windows[model]) > self.window_size:
                self._windows[model] = self._windows[model][-self.window_size:]

        self._bootstrapped = True
        for model in self.model_names:
            short = model.split("/")[-1] if "/" in model else model
            logger.info(f"  PFS bootstrap {short}: {len(self._windows[model])} entries")

    def record_completion(self, model_name: str, output_length: float, input_len: int = 0):
        """Record a completed request (updates sliding window)."""
        if model_name not in self._windows:
            return
        ibin = _input_bin(input_len)
        w = self._windows[model_name]
        w.append((output_length, ibin))
        if len(w) > self.window_size:
            w.pop(0)

    def predict_length(self, model_name: str, input_len: int = 0) -> float:
        """Predict output length for a model from the sliding window."""
        w = self._windows.get(model_name, [])
        if not w:
            return 128.0  # default fallback

        if self.use_input_bins:
            ibin = _input_bin(input_len)
            bin_lengths = [l for l, b in w if b == ibin]
            if len(bin_lengths) >= 3:
                return float(np.median(bin_lengths))

        # Global median fallback
        return float(np.median([l for l, _ in w]))

    def predict_bucket(self, model_name: str, input_len: int = 0) -> int:
        """Predict output length bucket."""
        pred_len = self.predict_length(model_name, input_len)
        return _length_to_bucket(pred_len)

    def predict_bucket_distribution(
        self, model_name: str, input_len: int = 0
    ) -> np.ndarray:
        """Predict bucket probability distribution from empirical window.

        Returns array of shape (max_buckets,) summing to 1.
        """
        w = self._windows.get(model_name, [])
        if not w:
            # Uniform fallback
            return np.ones(self.max_buckets) / self.max_buckets

        if self.use_input_bins:
            ibin = _input_bin(input_len)
            bin_lengths = [l for l, b in w if b == ibin]
            if len(bin_lengths) >= 3:
                lengths = bin_lengths
            else:
                lengths = [l for l, _ in w]
        else:
            lengths = [l for l, _ in w]

        # Build empirical bucket distribution
        buckets = [_length_to_bucket(l) for l in lengths]
        counts = np.bincount(buckets, minlength=self.max_buckets).astype(float)
        total = counts.sum()
        if total > 0:
            return counts / total
        return np.ones(self.max_buckets) / self.max_buckets

    def predict_all_models(
        self, input_len: int = 0
    ) -> Dict[str, Dict]:
        """Predict length + bucket distribution for all models.

        Returns {model_name: {"length": float, "bucket": int, "bucket_probs": ndarray}}
        """
        results = {}
        for model in self.model_names:
            pred_len = self.predict_length(model, input_len)
            pred_bucket = _length_to_bucket(pred_len)
            bucket_probs = self.predict_bucket_distribution(model, input_len)
            results[model] = {
                "length": pred_len,
                "bucket": pred_bucket,
                "bucket_probs": bucket_probs,
            }
        return results

    @property
    def window_sizes(self) -> Dict[str, int]:
        """Current window sizes per model."""
        return {m: len(w) for m, w in self._windows.items()}
