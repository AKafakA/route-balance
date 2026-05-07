#!/usr/bin/env python3
"""
Train LSTM-based length and quality predictor for ROUTE_BALANCE (ablation).

Uses sentence-transformer embeddings as input sequence to LSTM,
evaluating whether sequential prompt structure helps vs bag-of-embeddings.

Usage:
    python -m route_balance.predictor.route_balance.offline_training.train_lstm_predictor \
        --input data/route_balance/training_data/route_balance_v3_all_train.json \
        --test-input data/route_balance/training_data/route_balance_v3_all_test.json \
        --output-dir models/route_balance/lstm_quality/
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LSTMLengthQualityModel(nn.Module):
    """LSTM model for per-model length + quality prediction from token embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_models: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        # Per-model output heads
        self.length_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(num_models)
        ])
        self.quality_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(num_models)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 1, embedding_dim) — single embedding per prompt
        Returns:
            lengths: (batch, num_models)
            qualities: (batch, num_models)
        """
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]  # (batch, hidden)

        lengths = torch.cat([head(h) for head in self.length_heads], dim=-1)
        qualities = torch.cat([head(h) for head in self.quality_heads], dim=-1)
        return lengths, qualities


def _extract_quality(m_data: dict) -> float:
    if "quality_score" in m_data:
        return float(m_data["quality_score"])
    sim = m_data.get("similarity_score")
    judge_scores = m_data.get("llm_judge_scores", {})
    valid_judges = [v for v in judge_scores.values() if v is not None]
    judge_mean = sum(valid_judges) / len(valid_judges) if valid_judges else None
    if sim is not None and judge_mean is not None:
        return 0.5 * float(sim) + 0.5 * float(judge_mean)
    elif sim is not None:
        return float(sim)
    elif judge_mean is not None:
        return float(judge_mean)
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM length/quality predictor for ROUTE_BALANCE"
    )
    parser.add_argument("--input", required=True, help="Training data JSON")
    parser.add_argument("--test-input", default=None, help="Test data JSON")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default="models/route_balance/lstm_quality")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Load data
    with open(args.input) as f:
        if args.input.endswith(".jsonl"):
            train_data = [json.loads(line) for line in f]
        else:
            raw = json.load(f)
            train_data = raw["requests"] if "requests" in raw else raw
    logger.info(f"Training data: {len(train_data)} requests")

    # Encode prompts
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(args.embedding_model, device=args.device)
    prompts = [d["prompt"] for d in train_data]
    embeddings = encoder.encode(prompts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb_dim = embeddings.shape[1]

    # Prepare labels
    model_names = sorted(train_data[0]["models"].keys())
    num_models = len(model_names)

    lengths = np.zeros((len(train_data), num_models), dtype=np.float32)
    qualities = np.zeros((len(train_data), num_models), dtype=np.float32)

    for i, d in enumerate(train_data):
        for j, model in enumerate(model_names):
            m_data = d["models"].get(model, {})
            lengths[i, j] = m_data.get("output_length", 0)
            is_harmful = d.get("is_harmful", False)
            # Safety-aware quality: protectai for harmful, Qwen judge for safe
            js = m_data.get("llm_judge_scores", {})
            if is_harmful:
                qualities[i, j] = float(js.get("protectai_distilroberta-base-rejection-v1", 0.0) or 0.0)
            else:
                qwen_js = [v for k, v in js.items() if v is not None and "protectai" not in k]
                qualities[i, j] = sum(qwen_js) / len(qwen_js) if qwen_js else 0.0

    # Train/val split — use test_input if provided, otherwise random 10%
    if args.test_input:
        with open(args.test_input) as f:
            if args.test_input.endswith(".jsonl"):
                test_data = [json.loads(line) for line in f]
            else:
                raw = json.load(f)
                test_data = raw["requests"] if "requests" in raw else raw
        logger.info(f"Test data: {len(test_data)} requests")

        test_prompts = [d["prompt"] for d in test_data]
        test_embeddings = encoder.encode(test_prompts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

        test_lengths = np.zeros((len(test_data), num_models), dtype=np.float32)
        test_qualities = np.zeros((len(test_data), num_models), dtype=np.float32)
        for i, d in enumerate(test_data):
            for j, model in enumerate(model_names):
                m_data = d["models"].get(model, {})
                test_lengths[i, j] = m_data.get("output_length", 0)
                is_harmful = d.get("is_harmful", False)
                js = m_data.get("llm_judge_scores", {})
                if is_harmful:
                    test_qualities[i, j] = float(js.get("protectai_distilroberta-base-rejection-v1", 0.0) or 0.0)
                else:
                    qwen_js = [v for k, v in js.items() if v is not None and "protectai" not in k]
                    test_qualities[i, j] = sum(qwen_js) / len(qwen_js) if qwen_js else 0.0

        X_train = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(1).to(args.device)
        X_val = torch.tensor(test_embeddings, dtype=torch.float32).unsqueeze(1).to(args.device)
        yl_train = torch.tensor(lengths, dtype=torch.float32).to(args.device)
        yl_val = torch.tensor(test_lengths, dtype=torch.float32).to(args.device)
        yq_train = torch.tensor(qualities, dtype=torch.float32).to(args.device)
        yq_val = torch.tensor(test_qualities, dtype=torch.float32).to(args.device)
    else:
        n = len(embeddings)
        n_val = int(n * 0.1)
        indices = np.random.RandomState(42).permutation(n)
        train_idx, val_idx = indices[:-n_val], indices[-n_val:]

        X = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(1)
        y_len = torch.tensor(lengths, dtype=torch.float32)
        y_qual = torch.tensor(qualities, dtype=torch.float32)

        X_train, X_val = X[train_idx].to(args.device), X[val_idx].to(args.device)
        yl_train, yl_val = y_len[train_idx].to(args.device), y_len[val_idx].to(args.device)
        yq_train, yq_val = y_qual[train_idx].to(args.device), y_qual[val_idx].to(args.device)

    # Train
    model = LSTMLengthQualityModel(
        input_dim=emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_models=num_models,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X_train), device=X_train.device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train), args.batch_size):
            idx = perm[i : i + args.batch_size]
            pred_len, pred_qual = model(X_train[idx])
            loss = mse(pred_len, yl_train[idx]) + mse(pred_qual, yq_train[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            vl, vq = model(X_val)
            val_loss = mse(vl, yl_val).item() + mse(vq, yq_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{args.epochs}: "
                f"train={epoch_loss/n_batches:.4f}, val={val_loss:.4f}"
            )

    if best_state:
        model.load_state_dict(best_state)

    # Save
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_names": model_names,
        "embedding_model": args.embedding_model,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "emb_dim": emb_dim,
    }, output_path / "lstm_quality.pt")

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_len, pred_qual = model(X_val)
        pred_len = pred_len.cpu().numpy()
        pred_qual = pred_qual.cpu().numpy()
    true_len = yl_val.cpu().numpy()
    true_qual = yq_val.cpu().numpy()

    print("\n" + "=" * 70)
    print("LSTM LENGTH/QUALITY PREDICTOR RESULTS")
    print("=" * 70)
    for j, model_name in enumerate(model_names):
        len_mae = float(np.mean(np.abs(true_len[:, j] - pred_len[:, j])))
        qual_mae = float(np.mean(np.abs(true_qual[:, j] - pred_qual[:, j])))
        print(f"  {model_name}: Length MAE={len_mae:.1f}, Quality MAE={qual_mae:.4f}")

    logger.info(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
