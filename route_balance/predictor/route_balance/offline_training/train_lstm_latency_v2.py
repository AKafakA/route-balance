#!/usr/bin/env python3
"""
Train LSTM latency predictor v2 — uses per-request queue shapes.

Unlike v1 which used a sliding window of aggregate stats (same features as
XGBoost), v2 feeds each request in the queue as one LSTM timestep:

    running_requests[0..N] → waiting_requests[0..M] → LSTM → queue embedding

Per-request features (5 per timestep):
    (num_prompt_tokens, num_computed_tokens, total_num_tokens,
     num_output_tokens, is_running)

The queue embedding is concatenated with:
    - Request features (prompt_tokens, predicted_output_tokens)
    - Instance EMA features (decode_tok/s, prefill_tok/s, kv_util, etc.)
and passed through an MLP head to predict latency.

This is the correct architecture for LSTM: sequence over queue shape,
not sliding window over aggregate stats.

Usage:
    python -m route_balance.predictor.route_balance.offline_training.train_lstm_latency_v2 \
        --data-dir data/route_balance/latency_data/all/ \
        --output-dir models/route_balance/lstm_latency_v2/ \
        --target e2el --epochs 50 --device cuda

    # Quick smoke test
    python -m route_balance.predictor.route_balance.offline_training.train_lstm_latency_v2 \
        --data-dir data/route_balance/latency_data/all/ \
        --output-dir /tmp/lstm_smoke/ \
        --target e2el --epochs 5 --max-samples 1000 --device cpu
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from route_balance.predictor.route_balance.offline_training.train_xgboost import load_latency_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Per-request features (one per queue entry)
QUEUE_REQUEST_FEATURES = [
    "num_prompt_tokens",
    "num_computed_tokens",
    "total_num_tokens",
    "num_output_tokens",
    "is_running",  # 1.0 for running, 0.0 for waiting
    "actual_output_tokens",  # final output length (enriched via cross-reference)
]
QUEUE_FEAT_DIM = len(QUEUE_REQUEST_FEATURES)

# Instance-level context features (scalar per record)
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
INST_FEAT_DIM = len(INSTANCE_CONTEXT_FEATURES)

# Request features (the request we're predicting for)
REQUEST_FEAT_DIM = 2  # (num_prompt_tokens, num_predicted_output_tokens)


class QueueShapeLSTM(nn.Module):
    """LSTM that encodes per-request queue shapes into a queue embedding.

    Architecture:
        queue_requests → LSTM → queue_embedding
        [queue_embedding; instance_features; request_features] → MLP → latency
    """

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

        # Queue encoder: processes per-request features
        self.queue_lstm = nn.LSTM(
            input_size=queue_feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Prediction head: queue_embedding + instance + request → latency
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

    def forward(
        self,
        queue_seqs: torch.Tensor,    # (batch, max_queue_len, queue_feat_dim)
        queue_lengths: torch.Tensor,  # (batch,) actual sequence lengths
        inst_features: torch.Tensor,  # (batch, inst_feat_dim)
        req_features: torch.Tensor,   # (batch, req_feat_dim)
    ) -> torch.Tensor:
        """Forward pass.

        Returns:
            (batch,) predicted latency
        """
        # Pack padded sequences for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            queue_seqs, queue_lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False,
        )
        _, (h_n, _) = self.queue_lstm(packed)
        # h_n: (num_layers, batch, hidden_dim) — take last layer
        queue_embedding = h_n[-1]  # (batch, hidden_dim)

        # Concatenate all features
        combined = torch.cat([queue_embedding, inst_features, req_features], dim=1)
        return self.head(combined).squeeze(-1)


def extract_queue_features(
    schedule_state: Dict,
    max_queue_len: int = 32,
) -> Tuple[np.ndarray, int]:
    """Extract per-request queue features from schedule_state.

    Returns:
        queue_features: (max_queue_len, QUEUE_FEAT_DIM) padded
        actual_length: number of requests in queue
    """
    features = []

    # Running requests first (sorted by output tokens desc — most progressed first)
    # Apr 28: replaced "actual_output_tokens" with "predicted_decode_tokens".
    # /schedule_trace from route_balance_v_11 emits predicted_decode_tokens (the oracle
    # output prediction passed at request submission), NOT actual_output_tokens
    # (which would require post-completion enrichment). Predicted is what the
    # scheduler uses for queue planning anyway, so this is the correct feature.
    running = schedule_state.get("running_requests", [])
    running_sorted = sorted(running, key=lambda r: r.get("num_output_tokens", 0), reverse=True)
    for req in running_sorted:
        features.append([
            float(req.get("num_prompt_tokens", 0)),
            float(req.get("num_computed_tokens", 0)),
            float(req.get("total_num_tokens", 0)),
            float(req.get("num_output_tokens", 0)),
            1.0,  # is_running
            float(req.get("predicted_decode_tokens") or req.get("actual_output_tokens", 0)),
        ])

    # Waiting requests (FIFO order)
    waiting = schedule_state.get("waiting_requests", [])
    for req in waiting:
        features.append([
            float(req.get("num_prompt_tokens", 0)),
            float(req.get("num_computed_tokens", 0)),
            float(req.get("total_num_tokens", 0)),
            float(req.get("num_output_tokens", 0)),
            0.0,  # is_waiting
            float(req.get("predicted_decode_tokens") or req.get("actual_output_tokens", 0)),
        ])

    actual_length = len(features)

    # Pad or truncate to max_queue_len
    if actual_length == 0:
        return np.zeros((max_queue_len, QUEUE_FEAT_DIM), dtype=np.float32), 0

    if actual_length > max_queue_len:
        features = features[:max_queue_len]
        actual_length = max_queue_len

    arr = np.array(features, dtype=np.float32)
    padded = np.zeros((max_queue_len, QUEUE_FEAT_DIM), dtype=np.float32)
    padded[:len(arr)] = arr

    return padded, actual_length


def extract_instance_features(schedule_state: Dict) -> np.ndarray:
    """Extract instance-level context features."""
    feats = []
    for key in INSTANCE_CONTEXT_FEATURES:
        feats.append(float(schedule_state.get(key, 0)))
    return np.array(feats, dtype=np.float32)


def build_dataset(
    records: List[Dict],
    target: str,
    max_queue_len: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build training dataset from latency records.

    Returns:
        queue_seqs: (N, max_queue_len, QUEUE_FEAT_DIM)
        queue_lengths: (N,) actual queue lengths
        inst_features: (N, INST_FEAT_DIM)
        req_features: (N, REQUEST_FEAT_DIM)
        targets: (N,) latency values
    """
    queue_list = []
    length_list = []
    inst_list = []
    req_list = []
    target_list = []

    for rec in records:
        target_val = rec.get(target) or rec.get("actual_e2e_latency")
        if target_val is None or target_val <= 0:
            continue

        schedule_state = rec.get("schedule_state", {})
        if not schedule_state:
            continue

        # Check if per-request data exists
        running = schedule_state.get("running_requests", [])
        waiting = schedule_state.get("waiting_requests", [])
        if not running and not waiting:
            continue

        queue_feats, queue_len = extract_queue_features(schedule_state, max_queue_len)
        inst_feats = extract_instance_features(schedule_state)

        num_prompt = float(rec.get("num_prompt_tokens") or rec.get("input_len", 0))
        num_output = float(
            rec.get("num_predicted_output_tokens")
            or rec.get("max_tokens")
            or rec.get("output_len", 0)
        )

        queue_list.append(queue_feats)
        length_list.append(queue_len)
        inst_list.append(inst_feats)
        req_list.append([num_prompt, num_output])
        target_list.append(float(target_val))

    if not queue_list:
        return (
            np.empty((0, max_queue_len, QUEUE_FEAT_DIM)),
            np.empty(0, dtype=np.int64),
            np.empty((0, INST_FEAT_DIM)),
            np.empty((0, REQUEST_FEAT_DIM)),
            np.empty(0),
        )

    return (
        np.stack(queue_list),
        np.array(length_list, dtype=np.int64),
        np.stack(inst_list),
        np.array(req_list, dtype=np.float32),
        np.array(target_list, dtype=np.float32),
    )


def normalize_features(
    queue_seqs: np.ndarray,
    inst_features: np.ndarray,
    req_features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Normalize features using training set statistics.

    Returns normalized arrays and stats dict for inference.
    """
    # Queue features: normalize per feature across all non-padded entries
    # Flatten non-zero entries for computing stats
    mask = queue_seqs.sum(axis=-1) != 0  # (N, max_queue_len) True if non-padded
    stats = {}

    queue_norm = queue_seqs.copy()
    for i in range(queue_seqs.shape[-1]):
        vals = queue_seqs[:, :, i][mask]
        if len(vals) > 0:
            mean = float(vals.mean())
            std = float(vals.std()) + 1e-8
        else:
            mean, std = 0.0, 1.0
        queue_norm[:, :, i] = (queue_seqs[:, :, i] - mean) / std
        # Zero out padding again
        queue_norm[:, :, i] *= mask
        stats[f"queue_{i}_mean"] = mean
        stats[f"queue_{i}_std"] = std

    # Instance features
    inst_norm = inst_features.copy()
    for i in range(inst_features.shape[-1]):
        mean = float(inst_features[:, i].mean())
        std = float(inst_features[:, i].std()) + 1e-8
        inst_norm[:, i] = (inst_features[:, i] - mean) / std
        stats[f"inst_{i}_mean"] = mean
        stats[f"inst_{i}_std"] = std

    # Request features
    req_norm = req_features.copy()
    for i in range(req_features.shape[-1]):
        mean = float(req_features[:, i].mean())
        std = float(req_features[:, i].std()) + 1e-8
        req_norm[:, i] = (req_features[:, i] - mean) / std
        stats[f"req_{i}_mean"] = mean
        stats[f"req_{i}_std"] = std

    return queue_norm, inst_norm, req_norm, stats


def train_model(
    queue_train: np.ndarray,
    lengths_train: np.ndarray,
    inst_train: np.ndarray,
    req_train: np.ndarray,
    y_train: np.ndarray,
    queue_val: np.ndarray,
    lengths_val: np.ndarray,
    inst_val: np.ndarray,
    req_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dim: int = 64,
    num_layers: int = 1,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    device: str = "cpu",
) -> Tuple[QueueShapeLSTM, Dict]:
    """Train queue-shape LSTM model."""
    torch.manual_seed(42)
    np.random.seed(42)

    model = QueueShapeLSTM(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    # Convert to tensors
    q_train_t = torch.tensor(queue_train, dtype=torch.float32).to(device)
    l_train_t = torch.tensor(lengths_train, dtype=torch.long).to(device)
    i_train_t = torch.tensor(inst_train, dtype=torch.float32).to(device)
    r_train_t = torch.tensor(req_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    q_val_t = torch.tensor(queue_val, dtype=torch.float32).to(device)
    l_val_t = torch.tensor(lengths_val, dtype=torch.long).to(device)
    i_val_t = torch.tensor(inst_val, dtype=torch.float32).to(device)
    r_val_t = torch.tensor(req_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(q_train_t), device=q_train_t.device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(q_train_t), batch_size):
            idx = perm[i : i + batch_size]
            pred = model(q_train_t[idx], l_train_t[idx], i_train_t[idx], r_train_t[idx])
            loss = loss_fn(pred, y_train_t[idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation (in batches to avoid OOM)
        model.eval()
        val_preds = []
        with torch.no_grad():
            for i in range(0, len(q_val_t), batch_size):
                end = min(i + batch_size, len(q_val_t))
                pred = model(q_val_t[i:end], l_val_t[i:end], i_val_t[i:end], r_val_t[i:end])
                val_preds.append(pred)
            val_pred = torch.cat(val_preds)
            val_loss = loss_fn(val_pred, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Compute MAE for logging
            val_mae = float(torch.abs(val_pred - y_val_t).mean())
            logger.info(
                f"  Epoch {epoch+1}/{epochs}: train_loss={epoch_loss/n_batches:.6f}, "
                f"val_loss={val_loss:.6f}, val_MAE={val_mae:.4f}s"
            )
            sys.stdout.flush()

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    val_preds = []
    with torch.no_grad():
        for i in range(0, len(q_val_t), batch_size):
            end = min(i + batch_size, len(q_val_t))
            pred = model(q_val_t[i:end], l_val_t[i:end], i_val_t[i:end], r_val_t[i:end])
            val_preds.append(pred.cpu())
    preds = torch.cat(val_preds).numpy()

    errors = np.abs(y_val - preds)
    metrics = {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mape": float(np.mean(errors / np.maximum(y_val, 1e-6)) * 100),
        "p50_error": float(np.percentile(errors, 50)),
        "p95_error": float(np.percentile(errors, 95)),
        "p99_error": float(np.percentile(errors, 99)),
        "n_train": len(y_train),
        "n_val": len(y_val),
    }

    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM v2 latency predictor (per-request queue shapes)"
    )
    parser.add_argument("--data-dir", required=True, help="Latency data directory")
    parser.add_argument("--instance-types", nargs="+", default=None)
    parser.add_argument(
        "--target", default="e2el",
        choices=["e2el", "actual_e2e_latency", "actual_ttft", "actual_tpot"],
    )
    parser.add_argument("--max-queue-len", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", default="models/route_balance/lstm_latency_v2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    records_by_type = load_latency_data(args.data_dir, args.instance_types)
    if not records_by_type:
        logger.error("No latency data found!")
        return

    all_metrics = {}
    all_models = {}
    all_norm_stats = {}

    for inst_type, records in sorted(records_by_type.items()):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training LSTM v2 for {inst_type} ({len(records)} records)")

        if args.max_samples > 0:
            records = records[:args.max_samples]

        # Build dataset
        queue_seqs, queue_lengths, inst_features, req_features, targets = build_dataset(
            records, args.target, args.max_queue_len
        )

        # Filter out records with empty queues
        valid = queue_lengths > 0
        queue_seqs = queue_seqs[valid]
        queue_lengths = queue_lengths[valid]
        inst_features = inst_features[valid]
        req_features = req_features[valid]
        targets = targets[valid]

        logger.info(f"  Valid records: {len(targets)} (with queue data)")
        if len(targets) < 100:
            logger.warning(f"  Too few records for {inst_type}, skipping")
            continue

        # Queue length stats
        logger.info(
            f"  Queue lengths: mean={queue_lengths.mean():.1f}, "
            f"max={queue_lengths.max()}, p95={np.percentile(queue_lengths, 95):.0f}"
        )

        # Time-based split (last 20% as val)
        n_val = int(len(targets) * args.val_split)
        idx_train = slice(None, -n_val)
        idx_val = slice(-n_val, None)

        # Normalize using training set stats
        q_norm, i_norm, r_norm, norm_stats = normalize_features(
            queue_seqs[idx_train], inst_features[idx_train], req_features[idx_train]
        )

        # Apply same normalization to validation
        q_val_norm = queue_seqs[idx_val].copy()
        i_val_norm = inst_features[idx_val].copy()
        r_val_norm = req_features[idx_val].copy()

        mask_val = queue_seqs[idx_val].sum(axis=-1) != 0
        for i in range(QUEUE_FEAT_DIM):
            mean = norm_stats[f"queue_{i}_mean"]
            std = norm_stats[f"queue_{i}_std"]
            q_val_norm[:, :, i] = (queue_seqs[idx_val, :, i] - mean) / std
            q_val_norm[:, :, i] *= mask_val

        for i in range(INST_FEAT_DIM):
            mean = norm_stats[f"inst_{i}_mean"]
            std = norm_stats[f"inst_{i}_std"]
            i_val_norm[:, i] = (inst_features[idx_val, i] - mean) / std

        for i in range(REQUEST_FEAT_DIM):
            mean = norm_stats[f"req_{i}_mean"]
            std = norm_stats[f"req_{i}_std"]
            r_val_norm[:, i] = (req_features[idx_val, i] - mean) / std

        t0 = time.time()
        model, metrics = train_model(
            q_norm, queue_lengths[idx_train], i_norm, r_norm, targets[idx_train],
            q_val_norm, queue_lengths[idx_val], i_val_norm, r_val_norm, targets[idx_val],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
        )
        metrics["train_time_s"] = time.time() - t0
        metrics["n_valid_records"] = int(len(targets))

        all_models[inst_type] = model
        all_metrics[inst_type] = metrics
        all_norm_stats[inst_type] = norm_stats

        logger.info(
            f"  {inst_type}: MAE={metrics['mae']:.4f}s, MAPE={metrics['mape']:.1f}%, "
            f"P95={metrics['p95_error']:.4f}s ({metrics['train_time_s']:.0f}s)"
        )

    # Save
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model states
    state = {
        "version": 2,
        "max_queue_len": args.max_queue_len,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "queue_feat_dim": QUEUE_FEAT_DIM,
        "inst_feat_dim": INST_FEAT_DIM,
        "req_feat_dim": REQUEST_FEAT_DIM,
        "instance_types": list(all_models.keys()),
        "model_states": {
            inst: model.state_dict() for inst, model in all_models.items()
        },
        "norm_stats": all_norm_stats,
    }
    torch.save(state, output_path / "lstm_v2_predictor.pt")

    # Save metrics
    with open(output_path / "training_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save config for reference
    config = {
        "version": 2,
        "architecture": "QueueShapeLSTM",
        "queue_features": QUEUE_REQUEST_FEATURES,
        "instance_features": INSTANCE_CONTEXT_FEATURES,
        "max_queue_len": args.max_queue_len,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "target": args.target,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Summary table
    print(f"\n{'='*70}")
    print(f"LSTM v2 LATENCY PREDICTOR RESULTS (target: {args.target})")
    print(f"{'='*70}")
    header = f"{'Instance Type':<35} {'MAE':>8} {'MAPE':>8} {'P50':>8} {'P95':>8} {'N':>8}"
    print(header)
    print("-" * 70)
    for inst_type, m in sorted(all_metrics.items()):
        print(
            f"{inst_type:<35} {m['mae']:>7.4f}s {m['mape']:>7.1f}% "
            f"{m['p50_error']:>7.4f}s {m['p95_error']:>7.4f}s {m['n_val']:>7d}"
        )
    print(f"\nModels saved to {output_path}")


if __name__ == "__main__":
    main()
