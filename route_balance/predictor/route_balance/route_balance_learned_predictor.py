"""
Learned predictor for ROUTE_BALANCE scheduler.

Wraps multiple trained models:
- Bucket classifier (ModernBERT): output length distribution per model
- XGBoost TTFT: time-to-first-token per instance type
- TPOT lookup table: per-instance-type average TPOT
- KNN quality: quality score per model

Used by the multi-objective scheduler to make routing decisions.
"""
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from route_balance.predictor.route_balance.base_predictor import ROUTE_BALANCE_BasePredictor
from route_balance.predictor.route_balance.data_structures import PredictRequest

logger = logging.getLogger(__name__)


class RouteBalanceLearnedPredictor(ROUTE_BALANCE_BasePredictor):
    """Learned predictor combining bucket classifier, XGBoost TTFT,
    TPOT lookup, and KNN quality models.

    Provides predictions needed by the multi-objective scheduler:
    - Output length distribution (bucket probabilities)
    - Expected output length
    - P(tokens <= budget) for budget compliance
    - TTFT prediction
    - TPOT estimate
    - Quality score
    """

    def __init__(self, config, port: int, hostname: str = "localhost",
                 instance_type: str = "unknown", device: str = "cpu"):
        super().__init__(config, port)
        self._hostname = hostname
        self._instance_type = instance_type
        self._device = device

        # Sub-models (loaded lazily)
        self._bucket_model = None
        self._bucket_tokenizer = None
        self._xgboost_predictor = None
        self._knn_estimator = None

        # Config sections
        self._bucket_config = config.bucket_config
        self._xgboost_config = config.xgboost_config
        self._tpot_lookup = config.tpot_lookup
        self._quality_config = config.quality_config
        self._instance_metadata = config.instance_metadata
        self._scoring_weights = config.scoring_weights
        self._slo_defaults = config.slo_defaults

        # Bucket parameters
        self._bucket_size = self._bucket_config.get("bucket_size", 64)
        self._max_buckets = self._bucket_config.get("max_buckets", 16)
        self._max_length = self._bucket_config.get("max_length", 1024)

        self._load_models()

    def _load_models(self):
        """Load all sub-models from checkpoints."""
        t0 = time.time()

        # 1. Load bucket classifier
        self._load_bucket_classifier()

        # 2. Load XGBoost TTFT
        self._load_xgboost()

        # 3. Load KNN quality
        self._load_knn_quality()

        elapsed = time.time() - t0
        logger.info(f"LearnedPredictor loaded all models in {elapsed:.1f}s")

    def _load_bucket_classifier(self):
        """Load ModernBERT bucket classifier for the instance's model."""
        bucket_dir = self._bucket_config.get("model_dir", "")
        model_map = self._bucket_config.get("model_map", {})

        # Find the right model subdirectory for this instance's target LLM
        meta = self._instance_metadata.get(self._instance_type, {})
        model_name = meta.get("model_name", "")

        subdir = model_map.get(model_name)
        if not subdir or not bucket_dir:
            logger.warning(
                f"No bucket classifier configured for instance_type={self._instance_type}, "
                f"model={model_name}"
            )
            return

        model_path = Path(bucket_dir) / subdir
        if not model_path.exists():
            logger.warning(f"Bucket classifier not found at {model_path}")
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            self._bucket_tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            self._bucket_model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            self._bucket_model.eval()

            # Move to configured device (default CPU to avoid GPU conflict with vLLM)
            self._bucket_model = self._bucket_model.to(self._device)

            logger.info(f"Bucket classifier loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load bucket classifier: {e}")

    def _load_xgboost(self):
        """Load XGBoost latency predictors (TTFT, TPOT, E2E).

        Supports separate model dirs for each target:
            xgboost_ttft.model_dir → TTFT predictor (no output-length dependency)
            xgboost_tpot.model_dir → TPOT predictor (no output-length dependency)
            xgboost_e2e.model_dir → E2E predictor (uses predicted output tokens)
        Falls back to single model_dir for backward compatibility.
        """
        from route_balance.predictor.route_balance.estimators.xgboost_predictor import (
            XGBoostLatencyPredictor,
        )

        # `self._config` is a LearnedPredictorConfig dataclass; the actual dict
        # of options lives at `.raw_config`. Use a local alias so all .get()
        # calls below work (previously these were silently failing with
        # AttributeError because LearnedPredictorConfig has no .get method).
        cfg = self._config.raw_config if hasattr(self._config, "raw_config") else (
            self._config if isinstance(self._config, dict) else {}
        )

        # Try separate TTFT model
        ttft_dir = cfg.get("xgboost_ttft", {}).get("model_dir", "")
        if ttft_dir and Path(ttft_dir).exists():
            try:
                self._xgboost_ttft = XGBoostLatencyPredictor.load(ttft_dir)
                logger.info(f"XGBoost TTFT predictor loaded from {ttft_dir}")
            except Exception as e:
                logger.error(f"Failed to load XGBoost TTFT: {e}")

        # Try separate TPOT model
        tpot_dir = cfg.get("xgboost_tpot", {}).get("model_dir", "")
        if tpot_dir and Path(tpot_dir).exists():
            try:
                self._xgboost_tpot = XGBoostLatencyPredictor.load(tpot_dir)
                logger.info(f"XGBoost TPOT predictor loaded from {tpot_dir}")
            except Exception as e:
                logger.error(f"Failed to load XGBoost TPOT: {e}")

        # Try E2E model (for soft scoring)
        e2e_dir = cfg.get("xgboost_e2e", {}).get("model_dir", "")
        if e2e_dir and Path(e2e_dir).exists():
            try:
                self._xgboost_e2e = XGBoostLatencyPredictor.load(e2e_dir)
                logger.info(f"XGBoost E2E predictor loaded from {e2e_dir}")
            except Exception as e:
                logger.error(f"Failed to load XGBoost E2E: {e}")

        # Backward compat: single model_dir (treated as E2E)
        if not hasattr(self, '_xgboost_ttft'):
            self._xgboost_ttft = None
        if not hasattr(self, '_xgboost_tpot'):
            self._xgboost_tpot = None
        if not hasattr(self, '_xgboost_e2e'):
            self._xgboost_e2e = None

        # New schema: xgboost_3model.model_paths_by_instance.<inst_type>.{ttft,tpot,e2e}
        # Each path is a single .xgb file. Load each into a XGBoostLatencyPredictor
        # (one instance_type per predictor) so predict_ttft/tpot/e2e work via
        # _xgboost_*.predict(instance_type=..., ...) → {"e2e_latency": ...} (single key,
        # the predict methods at lines 240-316 fall through to the e2e_latency key).
        xgb3 = cfg.get("xgboost_3model", {})
        paths_map = xgb3.get("model_paths_by_instance", {})
        inst_paths = paths_map.get(self._instance_type, {}) if isinstance(paths_map, dict) else {}
        if inst_paths:
            try:
                import xgboost as xgb
                for target_key, attr_name in [
                    ("ttft", "_xgboost_ttft"),
                    ("tpot", "_xgboost_tpot"),
                    ("e2e", "_xgboost_e2e"),
                ]:
                    if getattr(self, attr_name, None) is not None:
                        continue  # already loaded via legacy schema
                    path = inst_paths.get(target_key)
                    if not path or not Path(path).exists():
                        continue
                    try:
                        wrap = XGBoostLatencyPredictor()
                        booster = xgb.XGBRegressor()
                        booster.load_model(path)
                        wrap.models[self._instance_type] = booster
                        # Honor log_target if metrics file present
                        metrics_path = Path(str(path).replace(".xgb", ".metrics.json"))
                        if metrics_path.exists():
                            try:
                                wrap._log_target = bool(json.load(open(metrics_path)).get("log_target", False))
                            except Exception:
                                pass
                        setattr(self, attr_name, wrap)
                        logger.info(
                            f"XGBoost {target_key} loaded for {self._instance_type} from {path} "
                            f"(per-instance schema)"
                        )
                    except Exception as e:
                        logger.error(f"Failed to load {target_key} for {self._instance_type}: {e}")
            except ImportError:
                logger.error("xgboost not installed; cannot load per-instance models")

        xgb_dir = self._xgboost_config.get("model_dir", "")
        if xgb_dir and Path(xgb_dir).exists() and self._xgboost_ttft is None:
            try:
                self._xgboost_predictor = XGBoostLatencyPredictor.load(xgb_dir)
                logger.info(f"XGBoost predictor loaded from {xgb_dir} (legacy E2E)")
            except Exception as e:
                logger.error(f"Failed to load XGBoost predictor: {e}")

    def _load_knn_quality(self):
        """Load KNN quality estimator."""
        knn_dir = self._quality_config.get("model_dir", "")
        if not knn_dir or not Path(knn_dir).exists():
            logger.warning(f"KNN model dir not found: {knn_dir}")
            return

        try:
            from route_balance.predictor.route_balance.estimators.knn_estimator import KNNEstimator
            device = self._quality_config.get("device", "cpu")
            self._knn_estimator = KNNEstimator.load(knn_dir, device=device)
            logger.info(f"KNN quality estimator loaded from {knn_dir}")
        except Exception as e:
            logger.error(f"Failed to load KNN quality estimator: {e}")

    def predict_bucket_distribution(self, prompt: str) -> np.ndarray:
        """Predict output length bucket probabilities.

        Returns:
            Array of shape (num_buckets,) with probabilities.
        """
        if self._bucket_model is None or self._bucket_tokenizer is None:
            # Uniform fallback
            return np.ones(self._max_buckets) / self._max_buckets

        import torch

        inputs = self._bucket_tokenizer(
            prompt,
            truncation=True,
            max_length=self._bucket_config.get("max_length", 1024),
            return_tensors="pt",
        )

        device = next(self._bucket_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._bucket_model(**inputs).logits.squeeze()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        return probs

    def predict_expected_length(self, bucket_probs: np.ndarray) -> float:
        """Compute expected output length from bucket probabilities."""
        midpoints = np.array([
            (i * self._bucket_size + self._bucket_size / 2)
            for i in range(len(bucket_probs))
        ])
        return float(np.sum(bucket_probs * midpoints))

    def predict_budget_compliance(
        self, bucket_probs: np.ndarray, budget_tokens: int
    ) -> float:
        """Compute P(output_tokens <= budget) from bucket distribution."""
        budget_bucket = min(budget_tokens // self._bucket_size, len(bucket_probs) - 1)
        return float(np.sum(bucket_probs[: budget_bucket + 1]))

    def predict_ttft(
        self, schedule_state: Dict, num_prompt_tokens: int,
        num_predicted_output_tokens: int
    ) -> float:
        """Predict TTFT using dedicated TTFT XGBoost model.

        TTFT does not depend on output length (only queue state + prompt).
        Returns TTFT in seconds.
        """
        predictor = getattr(self, '_xgboost_ttft', None) or self._xgboost_predictor
        if predictor is None:
            return 1.0  # fallback

        try:
            result = predictor.predict(
                instance_type=self._instance_type,
                schedule_state=schedule_state,
                num_prompt_tokens=num_prompt_tokens,
                num_predicted_output_tokens=num_predicted_output_tokens,
            )
            # Use ttft field if available, else e2e_latency (legacy model)
            return result.get("ttft", result.get("e2e_latency", 1.0))
        except (ValueError, KeyError):
            return 1.0

    def predict_tpot(
        self, schedule_state: Dict = None, num_prompt_tokens: int = 0,
        num_predicted_output_tokens: int = 0
    ) -> float:
        """Predict TPOT using dedicated TPOT XGBoost model.

        TPOT does not depend on output length (only batch composition + hardware).
        Falls back to static lookup if no learned model available.
        Returns TPOT in seconds.
        """
        predictor = getattr(self, '_xgboost_tpot', None)
        if predictor is not None and schedule_state:
            try:
                result = predictor.predict(
                    instance_type=self._instance_type,
                    schedule_state=schedule_state,
                    num_prompt_tokens=num_prompt_tokens,
                    num_predicted_output_tokens=num_predicted_output_tokens,
                )
                return result.get("tpot", result.get("e2e_latency", 0.05))
            except (ValueError, KeyError) as e:
                logger.warning(f"XGBoost TPOT prediction failed for {self._instance_type}: {e}")
        # Fallback: static lookup
        return self._tpot_lookup.get(self._instance_type, 0.05)

    def predict_e2e(
        self, schedule_state: Dict, num_prompt_tokens: int,
        num_predicted_output_tokens: int
    ) -> float:
        """Predict E2E latency via analytical Little's-Law formula on top of
        learned tpot. The tpot signal is the per-(instance, queue) XGBoost
        prediction (so hardware differences flow through naturally), and the
        analytical envelope adds queue-aware decode time:

            slot_open = (running < max_seqs) and (waiting == 0)
            if slot_open: e2e = own_predicted_output × tpot
            else:         e2e = (pending_decode/running + own_output) × tpot

        Real XGBoost tpot for the per-instance hardware mix + analytical queue
        accounting → captures both hardware difference (via tpot) AND queue
        contention (via running/pending state).
        """
        # Get real tpot from XGBoost (per-instance, queue-aware features).
        tpot = self.predict_tpot(
            schedule_state, num_prompt_tokens, num_predicted_output_tokens
        )
        tpot = float(tpot or 0.030)
        ss = schedule_state or {}
        running = int(ss.get("num_running", 0) or 0)
        waiting = int(ss.get("num_waiting", 0) or 0)
        max_seqs = int(ss.get("max_num_seqs", 256) or 256)
        own = max(1, int(num_predicted_output_tokens or 1))
        slot_open = (running < max_seqs) and (waiting == 0)
        if slot_open:
            return own * tpot
        pending = float(ss.get("pending_decode_tokens", 0) or 0)
        queue_iters = pending / max(running, 1)
        return (queue_iters + own) * tpot

    def predict_quality(self, prompt: str, model_name: str) -> float:
        """Predict quality score using KNN estimator.

        Returns quality score in [0, 1].
        """
        if self._knn_estimator is None:
            return 0.5  # fallback: neutral quality

        try:
            return self._knn_estimator.predict_quality(prompt, model_name)
        except (ValueError, KeyError):
            return 0.5

    async def predict(self, target_request: PredictRequest) -> Dict:
        """Full prediction for scheduling decision.

        Returns dict with all metrics needed by multi-objective scheduler.
        """
        # For the predictor API, we don't have the prompt text.
        # The scheduler will call predict_full() with the prompt directly.
        # This method provides a simple metric for backward compatibility.
        return {
            "target_metric": 0.0,
            "gpu_blocks": 0,
            "num_requests": 0,
            "num_preempted": 0,
            "predictor_type": "learned",
            "tpot": self.predict_tpot(),
        }

    def predict_full(
        self, prompt: str, num_prompt_tokens: int,
        num_predicted_output_tokens: int,
        schedule_state: Optional[Dict] = None,
        budget_tokens: int = 256,
    ) -> Dict:
        """Full prediction with all metrics for multi-objective scheduling.

        Args:
            prompt: Request prompt text
            num_prompt_tokens: Number of prompt tokens
            num_predicted_output_tokens: Predicted output tokens
            schedule_state: Current instance state (for TTFT prediction)
            budget_tokens: Token budget for compliance check

        Returns:
            Dict with:
                - bucket_probs: array of bucket probabilities
                - expected_length: expected output tokens
                - p_under_budget: P(output <= budget)
                - ttft: predicted TTFT in seconds
                - tpot: predicted TPOT in seconds
                - quality: predicted quality score [0, 1]
                - model_name: target LLM model name
                - instance_type: instance type string
                - cost_per_token: cost per output token
        """
        meta = self._instance_metadata.get(self._instance_type, {})
        model_name = meta.get("model_name", "unknown")
        cost_per_token = meta.get("cost_per_token", 0.01)

        # 1. Bucket distribution
        bucket_probs = self.predict_bucket_distribution(prompt)
        expected_length = self.predict_expected_length(bucket_probs)
        p_under_budget = self.predict_budget_compliance(bucket_probs, budget_tokens)

        # 2. TTFT
        state = schedule_state or {}
        ttft = self.predict_ttft(state, num_prompt_tokens, num_predicted_output_tokens)

        # 3. TPOT
        tpot = self.predict_tpot()

        # 4. Quality
        quality = self.predict_quality(prompt, model_name)

        return {
            "bucket_probs": bucket_probs.tolist(),
            "expected_length": expected_length,
            "p_under_budget": p_under_budget,
            "ttft": ttft,
            "tpot": tpot,
            "quality": quality,
            "model_name": model_name,
            "instance_type": self._instance_type,
            "cost_per_token": cost_per_token,
        }


class LSTMSidecarPredictor:
    """Adapts LSTMLatencyPredictor to the sidecar interface.

    The LSTM predicts E2E latency from a window of schedule states.
    TTFT and TPOT are decomposed heuristically from E2E:
      TTFT ≈ e2e × (prompt_tokens / total_tokens)
      TPOT ≈ (e2e - TTFT) / output_tokens
    This is an approximation; the LSTM ablation measures E2E directly.
    """

    def __init__(self, config, port: int, hostname: str = "localhost",
                 instance_type: str = "unknown"):
        from route_balance.predictor.route_balance.estimators.lstm_predictor import LSTMLatencyPredictor
        from route_balance.predictor.route_balance.schedule_trace_client import ScheduleTraceClient

        self._instance_type = instance_type
        self._hostname = hostname
        self._config = config.raw_config if hasattr(config, 'raw_config') else {}
        self._tpot_lookup = self._config.get("tpot_lookup", {})
        self._state_window = []  # Rolling window of schedule states
        self._window_size = self._config.get("lstm", {}).get("window_size", 10)

        # Load LSTM model
        lstm_dir = self._config.get("lstm", {}).get("model_dir", "")
        device = "cuda" if self._config.get("lstm", {}).get("use_gpu", False) else "cpu"
        self._lstm = LSTMLatencyPredictor.load(lstm_dir, device=device)
        logger.info(f"LSTMSidecarPredictor: loaded from {lstm_dir}")

        # Schedule trace client for fetching instance state
        self._schedule_trace_client = ScheduleTraceClient(
            hostname=hostname, port=port,
            timeout=self._config.get("schedule_trace_timeout", 5),
        )

    def _update_window(self, state: Dict):
        """Append state to rolling window."""
        self._state_window.append(state)
        if len(self._state_window) > self._window_size:
            self._state_window = self._state_window[-self._window_size:]

    def predict_ttft(self, schedule_state: Dict, num_prompt_tokens: int,
                     num_predicted_output_tokens: int) -> float:
        self._update_window(schedule_state)
        try:
            result = self._lstm.predict(
                self._instance_type, self._state_window,
                num_prompt_tokens, num_predicted_output_tokens,
            )
            e2e = result["e2e_latency"]
            total_tokens = num_prompt_tokens + num_predicted_output_tokens
            if total_tokens > 0:
                return e2e * (num_prompt_tokens / total_tokens)
            return e2e * 0.3
        except (ValueError, KeyError) as e:
            logger.warning(f"LSTM predict_ttft failed for {self._instance_type}: {e}")
            return 1.0

    def predict_tpot(self, schedule_state: Dict = None,
                     num_prompt_tokens: int = 0,
                     num_predicted_output_tokens: int = 0) -> float:
        if schedule_state:
            self._update_window(schedule_state)
        try:
            result = self._lstm.predict(
                self._instance_type, self._state_window,
                num_prompt_tokens, num_predicted_output_tokens,
            )
            e2e = result["e2e_latency"]
            total_tokens = num_prompt_tokens + num_predicted_output_tokens
            ttft_est = e2e * (num_prompt_tokens / max(total_tokens, 1))
            decode_time = e2e - ttft_est
            if num_predicted_output_tokens > 0:
                return decode_time / num_predicted_output_tokens
            return self._tpot_lookup.get(self._instance_type, 0.05)
        except (ValueError, KeyError) as e:
            logger.warning(f"LSTM predict_tpot failed for {self._instance_type}: {e}")
            return self._tpot_lookup.get(self._instance_type, 0.05)

    def predict_e2e(self, schedule_state: Dict, num_prompt_tokens: int,
                    num_predicted_output_tokens: int) -> float:
        self._update_window(schedule_state)
        try:
            result = self._lstm.predict(
                self._instance_type, self._state_window,
                num_prompt_tokens, num_predicted_output_tokens,
            )
            return result["e2e_latency"]
        except (ValueError, KeyError) as e:
            logger.warning(f"LSTM predict_e2e failed for {self._instance_type}: {e}")
            return 1.0


class RooflineSidecarPredictor:
    """Adapts RooflineLatencyPredictor to the sidecar interface.

    Analytical model: latency = queue_wait + prefill_time + decode_time.
    TTFT ≈ queue_wait + prefill_time, TPOT ≈ 1/decode_rate.
    """

    def __init__(self, config, port: int, hostname: str = "localhost",
                 instance_type: str = "unknown"):
        from route_balance.predictor.route_balance.estimators.roofline_predictor import RooflineLatencyPredictor
        from route_balance.predictor.route_balance.schedule_trace_client import ScheduleTraceClient

        self._instance_type = instance_type
        self._hostname = hostname
        self._config = config.raw_config if hasattr(config, 'raw_config') else {}

        # Load roofline model (calibrated rates or defaults)
        roofline_dir = self._config.get("roofline", {}).get("model_dir", "")
        if roofline_dir and Path(roofline_dir).exists():
            self._roofline = RooflineLatencyPredictor.load(roofline_dir)
            logger.info(f"RooflineSidecarPredictor: loaded calibrated rates from {roofline_dir}")
        else:
            self._roofline = RooflineLatencyPredictor()
            logger.info("RooflineSidecarPredictor: using default rates")

        self._schedule_trace_client = ScheduleTraceClient(
            hostname=hostname, port=port,
            timeout=self._config.get("schedule_trace_timeout", 5),
        )

    def predict_ttft(self, schedule_state: Dict, num_prompt_tokens: int,
                     num_predicted_output_tokens: int) -> float:
        result = self._roofline.predict(
            self._instance_type, schedule_state,
            num_prompt_tokens, num_predicted_output_tokens,
        )
        # TTFT = queue_wait + prefill_time
        return result.get("queue_wait", 0) + result.get("prefill_time", 0)

    def predict_tpot(self, schedule_state: Dict = None,
                     num_prompt_tokens: int = 0,
                     num_predicted_output_tokens: int = 0) -> float:
        rates = self._roofline.rates.get(self._instance_type)
        if rates is not None:
            _, decode_rate, _ = rates
            return 1.0 / max(decode_rate, 1.0)
        # Fallback: use EMA from schedule state
        if schedule_state:
            decode_rate = schedule_state.get("ema_decode_tok_per_s", 50)
            return 1.0 / max(decode_rate, 1.0)
        return 0.05

    def predict_e2e(self, schedule_state: Dict, num_prompt_tokens: int,
                    num_predicted_output_tokens: int) -> float:
        try:
            result = self._roofline.predict(
                self._instance_type, schedule_state,
                num_prompt_tokens, num_predicted_output_tokens,
            )
            return result["e2e_latency"]
        except (ValueError, KeyError) as e:
            logger.warning(f"Roofline predict_e2e failed for {self._instance_type}: {e}")
            return 1.0


class _OpportunisticBatcher:
    """Single-model batcher: drains the queue with no wait window.

    First request wakes the worker; the worker takes whatever else is
    already pending (get_nowait), runs one model.predict on the stacked
    batch, then resolves all futures. No fixed delay → no idle bubble.
    """

    def __init__(self, model, log_target: bool, max_batch: int = 128, name: str = ""):
        self._model = model
        self._log_target = log_target
        self._max_batch = max_batch
        self._name = name
        self._queue: Optional[asyncio.Queue] = None
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self):
        self._queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self):
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _worker(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                feature, future = await self._queue.get()
            except asyncio.CancelledError:
                return
            features = [feature]
            futures = [future]
            while not self._queue.empty() and len(features) < self._max_batch:
                try:
                    f, fut = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                features.append(f)
                futures.append(fut)
            try:
                X = np.stack(features).astype(np.float32, copy=False)
                preds = await loop.run_in_executor(None, self._model.predict, X)
                if self._log_target:
                    preds = np.expm1(preds)
                for fut, p in zip(futures, preds):
                    if not fut.done():
                        fut.set_result(float(p))
            except Exception as e:
                logger.warning(f"OpportunisticBatcher[{self._name}] predict failed: {e}")
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)

    async def predict(self, feature_vector: np.ndarray) -> float:
        if self._queue is None:
            await self.start()
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await self._queue.put((feature_vector, future))
        return await future


class XGBoost3ModelSidecarPredictor:
    """3 XGBoost sub-predictors (TTFT, TPOT, E2E) with opportunistic batching.

    Each sub-model has its own batcher. predict_ttft/predict_tpot/predict_e2e
    are async — the api_server calls them in parallel via asyncio.gather, so
    a single /predict_latency hits all 3 models concurrently and each model
    pools concurrent callers into a single matrix predict.

    Config (from raw_config["xgboost_3model"]):
        ttft_model: path to TTFT XGBoost JSON
        tpot_model: path to TPOT XGBoost JSON
        e2e_model:  path to E2E XGBoost JSON
        ttft_log_target / tpot_log_target / e2e_log_target: bool, defaults
            from sibling .metrics.json next to the model file
        max_batch: int, default 128
    """

    def __init__(self, config, port: int, hostname: str = "localhost",
                 instance_type: str = "unknown"):
        from route_balance.predictor.route_balance.schedule_trace_client import ScheduleTraceClient
        from route_balance.predictor.route_balance.estimators.xgboost_predictor import (
            build_feature_vector,
        )
        import xgboost as xgb

        self._instance_type = instance_type
        self._hostname = hostname
        raw = config.raw_config if hasattr(config, "raw_config") else {}
        self._config = raw
        self._build_feature_vector = build_feature_vector
        self._tpot_lookup = raw.get("tpot_lookup", {})

        m_cfg = raw.get("xgboost_3model", {})
        max_batch = int(m_cfg.get("max_batch", 128))
        # New: per-instance path map. Allows one config to serve all instance types.
        # Schema: m_cfg["model_paths_by_instance"][instance_type] = {"ttft": ..., "tpot": ..., "e2e": ...}
        # Falls back to the legacy m_cfg["ttft_model"] / ["tpot_model"] / ["e2e_model"] when no map.
        _paths_map = m_cfg.get("model_paths_by_instance", {})
        _per_inst = _paths_map.get(instance_type, {}) if isinstance(_paths_map, dict) else {}
        _key_short = {"ttft_model": "ttft", "tpot_model": "tpot", "e2e_model": "e2e"}

        def _load(path_key: str, log_key: str):
            short = _key_short.get(path_key, path_key)
            path = _per_inst.get(short) or m_cfg.get(path_key, "")
            if not path or not Path(path).exists():
                logger.warning(f"xgboost_3model {path_key} (inst={instance_type}) not found: {path}")
                return None, False
            model = xgb.XGBRegressor()
            model.load_model(path)
            log_t = m_cfg.get(log_key)
            if log_t is None:
                metrics_path = Path(str(path).replace(".json", ".metrics.json"))
                if metrics_path.exists():
                    try:
                        log_t = json.load(open(metrics_path)).get("log_target", False)
                    except Exception:
                        log_t = False
            return model, bool(log_t)

        self._ttft_model, self._ttft_log = _load("ttft_model", "ttft_log_target")
        self._tpot_model, self._tpot_log = _load("tpot_model", "tpot_log_target")
        self._e2e_model, self._e2e_log = _load("e2e_model", "e2e_log_target")

        self._ttft_batcher = _OpportunisticBatcher(
            self._ttft_model, self._ttft_log, max_batch, name=f"ttft@{instance_type}"
        ) if self._ttft_model is not None else None
        self._tpot_batcher = _OpportunisticBatcher(
            self._tpot_model, self._tpot_log, max_batch, name=f"tpot@{instance_type}"
        ) if self._tpot_model is not None else None
        self._e2e_batcher = _OpportunisticBatcher(
            self._e2e_model, self._e2e_log, max_batch, name=f"e2e@{instance_type}"
        ) if self._e2e_model is not None else None

        self._schedule_trace_client = ScheduleTraceClient(
            backend_host=hostname, backend_port=port,
            timeout=raw.get("schedule_trace_timeout", 5),
        )
        logger.info(
            f"XGBoost3ModelSidecarPredictor[{instance_type}] loaded: "
            f"ttft={self._ttft_model is not None} "
            f"tpot={self._tpot_model is not None} "
            f"e2e={self._e2e_model is not None} max_batch={max_batch}"
        )

    async def start(self):
        if self._ttft_batcher: await self._ttft_batcher.start()
        if self._tpot_batcher: await self._tpot_batcher.start()
        if self._e2e_batcher: await self._e2e_batcher.start()

    async def shutdown(self):
        for b in (self._ttft_batcher, self._tpot_batcher, self._e2e_batcher):
            if b is not None:
                await b.stop()

    async def predict_ttft(self, schedule_state: Dict, num_prompt_tokens: int,
                           num_predicted_output_tokens: int) -> float:
        if self._ttft_batcher is None:
            return 1.0
        try:
            fv = self._build_feature_vector(
                schedule_state, num_prompt_tokens, num_predicted_output_tokens
            )
            return await self._ttft_batcher.predict(fv)
        except Exception as e:
            logger.warning(f"predict_ttft failed: {e}")
            return 1.0

    async def predict_tpot(self, schedule_state: Dict = None,
                           num_prompt_tokens: int = 0,
                           num_predicted_output_tokens: int = 0) -> float:
        if self._tpot_batcher is None or not schedule_state:
            return self._tpot_lookup.get(self._instance_type, 0.05)
        try:
            fv = self._build_feature_vector(
                schedule_state, num_prompt_tokens, num_predicted_output_tokens
            )
            return await self._tpot_batcher.predict(fv)
        except Exception as e:
            logger.warning(f"predict_tpot failed: {e}")
            return self._tpot_lookup.get(self._instance_type, 0.05)

    async def predict_e2e(self, schedule_state: Dict, num_prompt_tokens: int,
                          num_predicted_output_tokens: int) -> float:
        """Analytical e2e under continuous batching with slot-availability check.

        Two regimes:
          (a) slot open: num_running < max_num_seqs AND num_waiting == 0
              → request joins the active batch immediately, no queue wait.
              e2e = own_predicted_output * tpot
          (b) batch full or queued: must wait for slot to free.
              e2e = (pending_decode_tokens / num_running + own_predicted_output) * tpot

        The slot-availability check avoids penalizing idle/light-load
        instances. In regime (a), pending_decode of CURRENTLY in-flight
        requests does NOT extend YOUR latency — they decode in parallel
        with you, sharing iterations. Only when the batch is full
        (num_running == max_num_seqs) or there's already a waiting queue
        does pending decode work delay your start.

        TPOT comes from per-instance XGBoost (queue-aware via training
        data, heterogeneous per (model, GPU)). Replaces the e2e XGBoost
        whose EMA-proxy queue signal was too sluggish for within-tier
        differentiation. Preemption-induced slowdown at very high kv_util
        is partially captured by TPOT XGBoost (training-time inflated
        TPOT under load) but not modeled separately here.
        """
        tpot = await self.predict_tpot(
            schedule_state, num_prompt_tokens, num_predicted_output_tokens
        )
        try:
            tpot = float(tpot or 0.030)
        except (TypeError, ValueError):
            tpot = 0.030

        ss = schedule_state or {}
        running = int(ss.get("num_running", 0) or 0)
        waiting = int(ss.get("num_waiting", 0) or 0)
        max_seqs = int(ss.get("max_num_seqs", 256) or 256)
        own = max(1, int(num_predicted_output_tokens or 1))

        # Regime (a): slot available — join immediately.
        slot_open = (running < max_seqs) and (waiting == 0)
        if slot_open:
            return own * tpot

        # Regime (b): batch full or already-queued requests ahead.
        pending = float(ss.get("pending_decode_tokens", 0) or 0)
        queue_iters = pending / max(running, 1)
        return (queue_iters + own) * tpot
