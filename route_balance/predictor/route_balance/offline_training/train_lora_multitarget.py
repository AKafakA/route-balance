"""
LoRA multi-target encoder for ROUTE_BALANCE (SR-style concurrent adapters).

Frozen encoder + single LoRA adapter set + per-(model, target) output heads.
One forward pass → all models × all targets.

Comparison with SR: they use N separate LoRA adapters (N forward passes).
We use 1 shared LoRA adapter + multi-head (1 forward pass).

Usage:
    python -m route_balance.predictor.route_balance.offline_training.train_lora_multitarget \
        --input data/scored/train_scored_filtered.jsonl \
        --test-input data/scored/test_scored_filtered.jsonl \
        --encoder-name roberta-base \
        --targets 'length_bucket:classification:16:64' 'deepeval:regression:1' \
        --epochs 5 --lr 1e-4 --device cuda \
        --output-dir models/route_balance/lora_multitarget_roberta_B
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from route_balance.predictor.route_balance.offline_training.train_fused_multitarget import (
    TargetSpec, MultiTargetDataset, evaluate_multitarget, parse_target_spec,
    _extract_label,
)


class LoRAMultiTargetEncoder(nn.Module):
    """Frozen encoder + LoRA + per-(model, target) heads.

    One forward pass produces predictions for all target LLMs × all targets.
    LoRA adapts the frozen encoder with minimal trainable parameters.
    """

    def __init__(
        self,
        encoder_name: str,
        model_names: List[str],
        target_specs: List[TargetSpec],
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        from peft import LoraConfig, get_peft_model

        config = AutoConfig.from_pretrained(encoder_name, trust_remote_code=True)
        self.hidden_size = config.hidden_size
        self.model_names = model_names
        self.target_specs = target_specs

        # Load base encoder
        try:
            base_model = AutoModel.from_pretrained(
                encoder_name, attn_implementation="eager", trust_remote_code=True,
            )
        except (ValueError, TypeError):
            base_model = AutoModel.from_pretrained(
                encoder_name, trust_remote_code=True,
            )

        # Apply LoRA
        if "roberta" in encoder_name.lower():
            target_modules = ["query", "value"]
        elif "modernbert" in encoder_name.lower() or "bert" in encoder_name.lower():
            target_modules = ["Wqkv", "query", "value", "key"]
        else:
            target_modules = ["query", "value"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.encoder = get_peft_model(base_model, lora_config)

        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.encoder.parameters())
        logger.info(
            f"LoRAMultiTarget: {encoder_name}, {len(model_names)} models, "
            f"{len(target_specs)} targets, LoRA trainable={trainable:,}/{total:,} "
            f"({100*trainable/total:.2f}%)"
        )

        # Per-(model, target) heads
        self.heads = nn.ModuleDict()
        for spec in target_specs:
            for name in model_names:
                key = f"{name.replace('/', '_').replace('-', '_').replace('.', '_')}__{spec.name}"
                self.heads[key] = nn.Linear(self.hidden_size, spec.num_labels)

        drop_rate = getattr(config, "classifier_dropout", None) or getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(drop_rate)

        n_head_params = sum(p.numel() for p in self.heads.parameters())
        logger.info(f"  Heads: {len(self.heads)} heads, {n_head_params:,} params")
        logger.info(f"  Total trainable: {trainable + n_head_params:,}")

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
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
                key = f"{name.replace('/', '_').replace('-', '_').replace('.', '_')}__{spec.name}"
                head_logits = self.heads[key](pooled)
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


def train_lora_multitarget(
    train_data, test_data, model_names, target_specs, encoder_name,
    epochs=5, lr=1e-4, batch_size=16, max_length=512, device="cuda",
    lora_r=16, lora_alpha=32, output_dir="models/lora_multitarget",
):
    from transformers import AutoTokenizer, DataCollatorWithPadding
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import PolynomialLR

    tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)

    train_ds = MultiTargetDataset(train_data, model_names, target_specs, tokenizer, max_length)
    test_ds = MultiTargetDataset(test_data, model_names, target_specs, tokenizer, max_length)

    def collate_fn(features):
        batch_labels = {spec.name: [] for spec in target_specs}
        for f in features:
            for spec in target_specs:
                batch_labels[spec.name].append(f.pop(f"labels_{spec.name}"))
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        batch_inputs = collator(features)
        for spec in target_specs:
            batch_inputs[f"labels_{spec.name}"] = torch.stack(batch_labels[spec.name])
        return batch_inputs

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = LoRAMultiTargetEncoder(
        encoder_name, model_names, target_specs,
        lora_r=lora_r, lora_alpha=lora_alpha,
    ).to(device)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )
    total_steps = len(train_loader) * epochs
    scheduler = PolynomialLR(optimizer, total_iters=total_steps, power=2)

    logger.info(f"Training: {len(train_ds)} samples, {total_steps} steps, {epochs} epochs")
    t0 = time.monotonic()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels_dict = {}
            for spec in target_specs:
                labels_dict[spec.name] = batch.pop(f"labels_{spec.name}")

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
    results["lora_r"] = lora_r
    results["lora_alpha"] = lora_alpha
    results["targets"] = [str(s) for s in target_specs]

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save({
        "encoder_name": encoder_name,
        "model_names": model_names,
        "target_specs": [(s.name, s.problem_type, s.num_labels, s.bucket_size) for s in target_specs],
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "state_dict": model.state_dict(),
    }, output_path / "lora_multitarget_model.pt")

    tokenizer.save_pretrained(str(output_path))

    with open(output_path / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Model saved to {output_path}")
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Train LoRA multi-target encoder")
    parser.add_argument("--input", required=True)
    parser.add_argument("--test-input", default=None)
    parser.add_argument("--encoder-name", default="roberta-base")
    parser.add_argument("--targets", nargs="+", required=True,
                        help="Target specs: 'name:type:num_labels[:bucket_size]'")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
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

    results = train_lora_multitarget(
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
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
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
