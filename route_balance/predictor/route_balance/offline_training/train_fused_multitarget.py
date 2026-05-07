"""
Fused multi-target multi-head encoder for ROUTE_BALANCE.

Extension of MultiHeadEncoder to support multiple prediction targets
(e.g., length_bucket + deepeval) in a single encoder with separate heads
per (model, target) pair.

Architecture:
    1 shared encoder → N models × T targets → N*T output heads
    One forward pass produces all predictions.

Example:
    - Target A: length_bucket (16-class classification) for 4 models = 4 heads
    - Target B: deepeval (1-value regression) for 4 models = 4 heads
    - Total: 1 encoder + 8 heads

Usage:
    python -m route_balance.predictor.route_balance.offline_training.train_fused_multitarget \
        --input data/scored/train_scored_filtered.jsonl \
        --test-input data/scored/test_scored_filtered.jsonl \
        --encoder-name roberta-base \
        --targets length_bucket:classification:16:64 deepeval:regression:1 \
        --epochs 5 --lr 1e-5 --device cuda \
        --output-dir models/route_balance/roberta_fused_multitarget_B
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target specification
# ---------------------------------------------------------------------------

class TargetSpec:
    """Specification for one prediction target."""
    def __init__(self, name: str, problem_type: str, num_labels: int,
                 bucket_size: int = 64):
        self.name = name
        self.problem_type = problem_type  # "regression" or "single_label_classification"
        self.num_labels = num_labels
        self.bucket_size = bucket_size  # only for length_bucket
        self.is_classification = problem_type == "single_label_classification"

    def __repr__(self):
        return f"TargetSpec({self.name}, {self.problem_type}, labels={self.num_labels})"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiTargetEncoder(nn.Module):
    """Shared encoder with per-(model, target) output heads.

    One forward pass → predictions for all target LLM models × all targets.
    """

    def __init__(
        self,
        encoder_name: str,
        model_names: List[str],
        target_specs: List[TargetSpec],
    ):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        config = AutoConfig.from_pretrained(encoder_name, trust_remote_code=True)
        self.hidden_size = config.hidden_size

        try:
            self.encoder = AutoModel.from_pretrained(
                encoder_name, attn_implementation="eager", trust_remote_code=True,
            )
        except (ValueError, TypeError):
            self.encoder = AutoModel.from_pretrained(
                encoder_name, trust_remote_code=True,
            )

        self.model_names = model_names
        self.target_specs = target_specs

        # One head per (model, target) pair
        self.heads = nn.ModuleDict()
        for spec in target_specs:
            for name in model_names:
                key = self._key(name, spec.name)
                self.heads[key] = nn.Linear(self.hidden_size, spec.num_labels)

        drop_rate = getattr(config, "classifier_dropout", None) or getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(drop_rate)

        logger.info(
            f"MultiTargetEncoder: {encoder_name}, {len(model_names)} models, "
            f"{len(target_specs)} targets, {len(self.heads)} heads"
        )

    @staticmethod
    def _key(model_name: str, target_name: str) -> str:
        """Module key for a (model, target) head."""
        m = model_name.replace("/", "_").replace("-", "_").replace(".", "_")
        t = target_name.replace("/", "_").replace("-", "_").replace(".", "_")
        return f"{m}__{t}"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            labels: dict of {target_name: (batch, num_models)} tensors.
                NaN entries are masked.

        Returns:
            dict with 'loss' and 'logits' = {target_name: (batch, num_models, num_labels)}
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])

        batch_size = pooled.size(0)
        num_models = len(self.model_names)

        all_logits = {}
        total_loss = torch.tensor(0.0, device=pooled.device)
        n_valid = 0

        for spec in self.target_specs:
            logits_t = torch.zeros(
                batch_size, num_models, spec.num_labels, device=pooled.device
            )
            target_labels = labels.get(spec.name) if labels else None

            for i, name in enumerate(self.model_names):
                key = self._key(name, spec.name)
                head_logits = self.heads[key](pooled)  # (batch, num_labels)
                logits_t[:, i, :] = head_logits

                if target_labels is not None:
                    model_labels = target_labels[:, i]
                    mask = ~torch.isnan(model_labels)
                    if mask.any():
                        if spec.is_classification:
                            loss_fn = nn.CrossEntropyLoss()
                            loss = loss_fn(head_logits[mask], model_labels[mask].long())
                        else:
                            loss_fn = nn.MSELoss()
                            loss = loss_fn(head_logits[mask].squeeze(-1), model_labels[mask])
                        total_loss = total_loss + loss
                        n_valid += 1

            all_logits[spec.name] = logits_t

        if n_valid > 0:
            total_loss = total_loss / n_valid

        return {"loss": total_loss, "logits": all_logits}

    def predict(self, input_ids, attention_mask, **kwargs):
        """Inference: return per-target, per-model predictions."""
        with torch.no_grad():
            result = self.forward(input_ids, attention_mask)
        return result["logits"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _extract_label(m_data: dict, target: str, bucket_size: int = 64,
                   is_harmful: bool = False) -> Optional[float]:
    """Extract label — same as train_fused_bert_predictor._extract_label + deepeval."""
    if target == "length":
        return float(m_data.get("output_length", 0))
    elif target == "length_bucket":
        length = float(m_data.get("output_length", 0))
        bucket = int(length) // bucket_size
        max_bucket = 1024 // bucket_size - 1
        return float(min(bucket, max_bucket))
    elif target == "length_log":
        val = m_data.get("output_length", 0)
        return float(np.log1p(val)) if val else None
    elif target == "deepeval":
        scores = m_data.get("llm_judge_scores", {})
        val = scores.get("deepeval-llama3.1-8b-it_reference")
        return float(val) if val is not None else None
    elif target == "reference_score":
        val = m_data.get("reference_score")
        return float(val) if val is not None else None
    elif target == "similarity":
        val = m_data.get("similarity_score")
        return float(val) if val is not None else None
    elif target == "judge":
        scores = m_data.get("llm_judge_scores", {})
        for k, v in scores.items():
            if "Qwen" in k and v is not None:
                return float(v)
        return 0.0
    return None


class MultiTargetDataset(Dataset):
    """Dataset returning labels for multiple targets per sample."""

    def __init__(self, data: list, model_names: List[str],
                 target_specs: List[TargetSpec], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_names = model_names
        self.target_specs = target_specs

        self.texts = []
        # {target_name: list of (num_models,) arrays}
        self.labels = {spec.name: [] for spec in target_specs}

        for req in data:
            prompt = req["prompt"]
            is_harmful = req.get("is_harmful", False)
            has_any = False

            per_target_labels = {}
            for spec in target_specs:
                row = []
                for model_name in model_names:
                    m_data = req["models"].get(model_name, {})
                    label = _extract_label(m_data, spec.name, spec.bucket_size, is_harmful)
                    if label is not None:
                        has_any = True
                    row.append(label if label is not None else float("nan"))
                per_target_labels[spec.name] = row

            if has_any:
                self.texts.append(prompt)
                for spec in target_specs:
                    self.labels[spec.name].append(per_target_labels[spec.name])

        logger.info(
            f"MultiTargetDataset: {len(self.texts)} samples, "
            f"{len(model_names)} models, {len(target_specs)} targets"
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # Labels as dict of tensors
        for spec in self.target_specs:
            item[f"labels_{spec.name}"] = torch.tensor(
                self.labels[spec.name][idx], dtype=torch.float32
            )
        return item


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_multitarget(
    train_data: list,
    test_data: list,
    model_names: List[str],
    target_specs: List[TargetSpec],
    encoder_name: str = "roberta-base",
    epochs: int = 5,
    lr: float = 1e-5,
    batch_size: int = 16,
    max_length: int = 512,
    device: str = "cuda",
    output_dir: str = "models/route_balance/fused_multitarget",
):
    """Train a multi-target fused model."""
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import PolynomialLR

    tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)

    train_ds = MultiTargetDataset(train_data, model_names, target_specs, tokenizer, max_length)
    test_ds = MultiTargetDataset(test_data, model_names, target_specs, tokenizer, max_length)

    def collate_fn(features):
        # Separate label keys from input keys
        batch_inputs = {}
        batch_labels = {spec.name: [] for spec in target_specs}

        for f in features:
            for spec in target_specs:
                batch_labels[spec.name].append(f.pop(f"labels_{spec.name}"))

        # Pad inputs
        from transformers import DataCollatorWithPadding
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        batch_inputs = collator(features)

        # Stack labels
        for spec in target_specs:
            batch_inputs[f"labels_{spec.name}"] = torch.stack(batch_labels[spec.name])

        return batch_inputs

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultiTargetEncoder(encoder_name, model_names, target_specs).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = PolynomialLR(optimizer, total_iters=total_steps, power=2)

    logger.info(f"Training: {len(train_ds)} samples, {total_steps} steps, {epochs} epochs")
    t0 = time.monotonic()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Extract labels dict
            labels_dict = {}
            for spec in target_specs:
                key = f"labels_{spec.name}"
                labels_dict[spec.name] = batch.pop(key)

            outputs = model(labels=labels_dict, **batch)
            loss = outputs["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

            if (step + 1) % 100 == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs}, step {step+1}/{len(train_loader)}, loss={loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}")

    train_time = time.monotonic() - t0
    logger.info(f"Training done in {train_time:.0f}s")

    # Evaluate
    model.eval()
    results = evaluate_multitarget(model, test_loader, model_names, target_specs, device)
    results["train_time_s"] = train_time
    results["encoder_name"] = encoder_name
    results["targets"] = [str(s) for s in target_specs]

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save({
        "encoder_name": encoder_name,
        "model_names": model_names,
        "target_specs": [(s.name, s.problem_type, s.num_labels, s.bucket_size) for s in target_specs],
        "state_dict": model.state_dict(),
    }, output_path / "fused_multitarget_model.pt")

    tokenizer.save_pretrained(str(output_path))

    config = {
        "encoder_name": encoder_name,
        "model_names": model_names,
        "targets": {s.name: {"problem_type": s.problem_type, "num_labels": s.num_labels, "bucket_size": s.bucket_size} for s in target_specs},
    }
    with open(output_path / "fused_multitarget_config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(output_path / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Model saved to {output_path}")
    return results


def evaluate_multitarget(model, test_loader, model_names, target_specs, device):
    """Evaluate multi-target model."""
    from scipy.stats import spearmanr

    all_logits = {s.name: [] for s in target_specs}
    all_labels = {s.name: [] for s in target_specs}

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels_dict = {}
            for spec in target_specs:
                key = f"labels_{spec.name}"
                labels_dict[spec.name] = batch.pop(key)

            outputs = model(labels=labels_dict, **batch)

            for spec in target_specs:
                all_logits[spec.name].append(outputs["logits"][spec.name].cpu().numpy())
                all_labels[spec.name].append(labels_dict[spec.name].cpu().numpy())

    results = {}
    for spec in target_specs:
        logits = np.concatenate(all_logits[spec.name], axis=0)
        labels = np.concatenate(all_labels[spec.name], axis=0)

        target_results = {}
        for i, name in enumerate(model_names):
            m_labels = labels[:, i]
            mask = ~np.isnan(m_labels)
            if not mask.any():
                continue

            m_logits = logits[:, i, :][mask]
            m_labels_valid = m_labels[mask]
            short_name = name.split("/")[-1]

            if spec.is_classification:
                preds = np.argmax(m_logits, axis=-1)
                acc = float(np.mean(preds == m_labels_valid.astype(int)))
                adj_acc = float(np.mean(np.abs(preds - m_labels_valid.astype(int)) <= 1))
                target_results[short_name] = {
                    "accuracy": acc,
                    "adjacent_accuracy": adj_acc,
                    "n": int(mask.sum()),
                }
                # Bucket MAE
                if spec.name == "length_bucket":
                    probs = np.exp(m_logits - m_logits.max(axis=-1, keepdims=True))
                    probs = probs / probs.sum(axis=-1, keepdims=True)
                    midpoints = np.array([(j * spec.bucket_size + spec.bucket_size / 2) for j in range(spec.num_labels)])
                    expected = np.sum(probs * midpoints, axis=-1)
                    bucket_mid = m_labels_valid * spec.bucket_size + spec.bucket_size / 2
                    target_results[short_name]["mae_tokens_bucket"] = float(np.mean(np.abs(bucket_mid - expected)))
            else:
                preds = m_logits.squeeze(-1)
                errors = np.abs(preds - m_labels_valid)
                rho, _ = spearmanr(preds, m_labels_valid)
                target_results[short_name] = {
                    "mae": float(np.mean(errors)),
                    "spearman_r": float(rho) if not np.isnan(rho) else 0.0,
                    "n": int(mask.sum()),
                }

        results[spec.name] = target_results

    return results


def load_multitarget_model(model_dir: str, device: str = "cpu"):
    """Load a saved multi-target model."""
    from transformers import AutoTokenizer

    model_path = Path(model_dir)
    checkpoint = torch.load(model_path / "fused_multitarget_model.pt", map_location=device, weights_only=False)

    target_specs = [
        TargetSpec(name=t[0], problem_type=t[1], num_labels=t[2],
                   bucket_size=t[3] if len(t) > 3 else 64)
        for t in checkpoint["target_specs"]
    ]

    model = MultiTargetEncoder(
        encoder_name=checkpoint["encoder_name"],
        model_names=checkpoint["model_names"],
        target_specs=target_specs,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model = model.to(device)

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    except (ValueError, OSError):
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint["encoder_name"], trust_remote_code=True
        )

    return model, tokenizer, target_specs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_target_spec(s: str) -> TargetSpec:
    """Parse target spec string: 'name:type:num_labels[:bucket_size]'

    Examples:
        'length_bucket:classification:16:64'
        'deepeval:regression:1'
        'length_log:regression:1'
    """
    parts = s.split(":")
    name = parts[0]
    ptype = parts[1] if len(parts) > 1 else "regression"
    if ptype == "classification":
        ptype = "single_label_classification"
    num_labels = int(parts[2]) if len(parts) > 2 else 1
    bucket_size = int(parts[3]) if len(parts) > 3 else 64
    return TargetSpec(name, ptype, num_labels, bucket_size)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Train fused multi-target encoder")
    parser.add_argument("--input", required=True, help="Training data JSONL")
    parser.add_argument("--test-input", default=None, help="Test data JSONL")
    parser.add_argument("--encoder-name", default="roberta-base")
    parser.add_argument("--targets", nargs="+", required=True,
                        help="Target specs: 'name:type:num_labels[:bucket_size]' "
                             "(e.g., 'length_bucket:classification:16:64 deepeval:regression:1')")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    target_specs = [parse_target_spec(t) for t in args.targets]
    logger.info(f"Targets: {target_specs}")

    with open(args.input) as f:
        train_data = [json.loads(l) for l in f]

    test_data = None
    if args.test_input:
        with open(args.test_input) as f:
            test_data = [json.loads(l) for l in f]

    model_names = sorted(train_data[0]["models"].keys())
    logger.info(f"Models: {[m.split('/')[-1] for m in model_names]}")

    results = train_multitarget(
        train_data=train_data,
        test_data=test_data or train_data[:100],
        model_names=model_names,
        target_specs=target_specs,
        encoder_name=args.encoder_name,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        output_dir=args.output_dir,
    )

    logger.info("=== Results ===")
    for target_name, target_results in results.items():
        if isinstance(target_results, dict) and any(isinstance(v, dict) for v in target_results.values()):
            logger.info(f"\n  {target_name}:")
            for model, metrics in target_results.items():
                if isinstance(metrics, dict):
                    logger.info(f"    {model}: {metrics}")


if __name__ == "__main__":
    main()
