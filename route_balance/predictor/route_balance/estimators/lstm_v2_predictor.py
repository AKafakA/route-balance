#!/usr/bin/env python3
"""
LSTM v2 latency predictor — per-request queue shape architecture.

Unlike v1 which uses sliding windows of aggregate stats, v2 encodes
each request in the queue as one LSTM timestep, producing a queue
embedding that captures the congestion pattern.

Used at serving time by the ROUTE_BALANCE scheduler.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Import model class from training (avoid circular)
QUEUE_FEAT_DIM = 6   # (prompt_tokens, computed_tokens, total_tokens, output_tokens, is_running, actual_output_tokens)
INST_FEAT_DIM = 10   # ema_decode_tok/s, ema_prefill_tok/s, etc.
REQUEST_FEAT_DIM = 2  # (prompt_tokens, predicted_output_tokens)


class QueueShapeLSTM(nn.Module):
    """LSTM that encodes per-request queue shapes."""

    def __init__(
        self,
        queue_feat_dim: int = QUEUE_FEAT_DIM,
        inst_feat_dim: int = INST_FEAT_DIM,
        req_feat_dim: int = REQUEST_FEAT_DIM,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.queue_lstm = nn.LSTM(
            input_size=queue_feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        head_input_dim = hidden_dim + inst_feat_dim + req_feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, queue_seqs, queue_lengths, inst_features, req_features):
        packed = nn.utils.rnn.pack_padded_sequence(
            queue_seqs, queue_lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False,
        )
        _, (h_n, _) = self.queue_lstm(packed)
        queue_embedding = h_n[-1]
        combined = torch.cat([queue_embedding, inst_features, req_features], dim=1)
        return self.head(combined).squeeze(-1)


# Instance context feature names (must match training)
INSTANCE_CONTEXT_FEATURES = [
    "ema_decode_tok_per_s",
    "ema_prefill_tok_per_s",
    "ema_decode_iter_ms",
    "kv_cache_utilization",
    "kv_free_blocks",
    "num_running",
    "num_waiting",
    "pending_prefill_tokens",
    "pending_decode_tokens",
    "token_budget_per_iter",
]


class LSTMv2LatencyPredictor:
    """LSTM v2 predictor for serving-time latency prediction.

    Uses per-request queue shapes instead of aggregate stats.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.models: Dict[str, QueueShapeLSTM] = {}
        self.norm_stats: Dict[str, Dict] = {}
        self.max_queue_len: int = 32
        self.hidden_dim: int = 64
        self.num_layers: int = 1

    def predict(
        self,
        instance_type: str,
        schedule_state: Dict,
        num_prompt_tokens: int,
        num_predicted_output_tokens: int,
    ) -> Dict[str, float]:
        """Predict latency from current queue state.

        Args:
            instance_type: E.g. "qwen2.5-3b_p100"
            schedule_state: Dict with running_requests[], waiting_requests[],
                           and aggregate stats.
            num_prompt_tokens: Input tokens for the request.
            num_predicted_output_tokens: Predicted output tokens.

        Returns:
            {"e2e_latency": predicted_seconds}
        """
        if instance_type not in self.models:
            raise ValueError(f"No model for {instance_type}")

        model = self.models[instance_type]
        stats = self.norm_stats.get(instance_type, {})

        # Extract queue features
        queue_feats, queue_len = self._extract_queue_features(schedule_state)
        inst_feats = self._extract_instance_features(schedule_state)
        req_feats = np.array([float(num_prompt_tokens), float(num_predicted_output_tokens)],
                             dtype=np.float32)

        # Normalize
        queue_norm = queue_feats.copy()
        mask = queue_feats.sum(axis=-1) != 0
        for i in range(QUEUE_FEAT_DIM):
            mean = stats.get(f"queue_{i}_mean", 0.0)
            std = stats.get(f"queue_{i}_std", 1.0)
            queue_norm[:, i] = (queue_feats[:, i] - mean) / std
            queue_norm[:, i] *= mask

        inst_norm = inst_feats.copy()
        for i in range(INST_FEAT_DIM):
            mean = stats.get(f"inst_{i}_mean", 0.0)
            std = stats.get(f"inst_{i}_std", 1.0)
            inst_norm[i] = (inst_feats[i] - mean) / std

        req_norm = req_feats.copy()
        for i in range(REQUEST_FEAT_DIM):
            mean = stats.get(f"req_{i}_mean", 0.0)
            std = stats.get(f"req_{i}_std", 1.0)
            req_norm[i] = (req_feats[i] - mean) / std

        # Convert to tensors (batch size 1)
        q_t = torch.tensor(queue_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        l_t = torch.tensor([queue_len], dtype=torch.long).to(self.device)
        i_t = torch.tensor(inst_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        r_t = torch.tensor(req_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

        model.eval()
        with torch.no_grad():
            pred = model(q_t, l_t, i_t, r_t)

        return {"e2e_latency": max(float(pred.item()), 0.0)}

    def _extract_queue_features(self, schedule_state: Dict) -> tuple:
        """Extract per-request queue features."""
        features = []

        running = schedule_state.get("running_requests", [])
        running_sorted = sorted(running, key=lambda r: r.get("num_output_tokens", 0), reverse=True)
        for req in running_sorted:
            features.append([
                float(req.get("num_prompt_tokens", 0)),
                float(req.get("num_computed_tokens", 0)),
                float(req.get("total_num_tokens", 0)),
                float(req.get("num_output_tokens", 0)),
                1.0,
                float(req.get("actual_output_tokens", 0)),
            ])

        waiting = schedule_state.get("waiting_requests", [])
        for req in waiting:
            features.append([
                float(req.get("num_prompt_tokens", 0)),
                float(req.get("num_computed_tokens", 0)),
                float(req.get("total_num_tokens", 0)),
                float(req.get("num_output_tokens", 0)),
                0.0,
                float(req.get("actual_output_tokens", 0)),
            ])

        actual_length = len(features)
        if actual_length == 0:
            return np.zeros((self.max_queue_len, QUEUE_FEAT_DIM), dtype=np.float32), 0

        if actual_length > self.max_queue_len:
            features = features[:self.max_queue_len]
            actual_length = self.max_queue_len

        arr = np.array(features, dtype=np.float32)
        padded = np.zeros((self.max_queue_len, QUEUE_FEAT_DIM), dtype=np.float32)
        padded[:len(arr)] = arr
        return padded, actual_length

    def _extract_instance_features(self, schedule_state: Dict) -> np.ndarray:
        """Extract instance-level context features."""
        feats = []
        for key in INSTANCE_CONTEXT_FEATURES:
            feats.append(float(schedule_state.get(key, 0)))
        return np.array(feats, dtype=np.float32)

    def save(self, output_dir: str):
        """Save predictor to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        state = {
            "version": 2,
            "max_queue_len": self.max_queue_len,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "queue_feat_dim": QUEUE_FEAT_DIM,
            "inst_feat_dim": INST_FEAT_DIM,
            "req_feat_dim": REQUEST_FEAT_DIM,
            "instance_types": list(self.models.keys()),
            "model_states": {
                inst: model.state_dict() for inst, model in self.models.items()
            },
            "norm_stats": self.norm_stats,
        }
        torch.save(state, output_path / "lstm_v2_predictor.pt")
        logger.info(f"LSTM v2 predictor saved to {output_path}")

    @classmethod
    def load(cls, model_dir: str, device: str = "cpu") -> "LSTMv2LatencyPredictor":
        """Load predictor from disk."""
        model_path = Path(model_dir) / "lstm_v2_predictor.pt"
        state = torch.load(model_path, map_location=device, weights_only=False)

        predictor = cls(device=device)
        predictor.max_queue_len = state["max_queue_len"]
        predictor.hidden_dim = state["hidden_dim"]
        predictor.num_layers = state["num_layers"]
        predictor.norm_stats = state.get("norm_stats", {})

        for inst_type in state["instance_types"]:
            model = QueueShapeLSTM(
                queue_feat_dim=state.get("queue_feat_dim", QUEUE_FEAT_DIM),
                inst_feat_dim=state.get("inst_feat_dim", INST_FEAT_DIM),
                req_feat_dim=state.get("req_feat_dim", REQUEST_FEAT_DIM),
                hidden_dim=predictor.hidden_dim,
                num_layers=predictor.num_layers,
            ).to(device)
            model.load_state_dict(state["model_states"][inst_type])
            predictor.models[inst_type] = model

        logger.info(f"LSTM v2 predictor loaded: {len(predictor.models)} models")
        return predictor
