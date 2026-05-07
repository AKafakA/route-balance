"""
LoRA-based encoder predictor for ROUTE_BALANCE (SR-style baseline).

Uses PEFT LoRA adapters on frozen RoBERTa/ModernBERT encoder for
classification/regression. Comparable to vLLM Semantic Router's approach
of using LoRA adapters on ModernBERT for per-task classification.

Architecture: frozen encoder + LoRA adapters + per-model linear heads
(same as MultiHeadEncoder but with LoRA instead of full fine-tuning)

Usage:
    python -m route_balance.predictor.route_balance.offline_training.train_lora_bert_predictor \
        --input data/scored/train_scored_filtered.jsonl \
        --test-input data/scored/test_scored_filtered.jsonl \
        --encoder-name roberta-base \
        --target deepeval --epochs 5 --lr 1e-4 --device cuda \
        --output-dir models/route_balance/lora_encoder_roberta/deepeval
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

# Reuse label extraction from fused encoder trainer
from route_balance.predictor.route_balance.offline_training.train_fused_bert_predictor import (
    _extract_label, FusedDataset, make_fused_metrics,
)


class LoRAMultiHeadEncoder(nn.Module):
    """Frozen encoder + LoRA adapters + per-model output heads.

    Same architecture as MultiHeadEncoder but with PEFT LoRA
    instead of full fine-tuning. Comparable to SR's per-task LoRA approach.
    """

    def __init__(
        self,
        encoder_name: str,
        model_names: List[str],
        num_labels: int = 1,
        problem_type: str = "regression",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        from transformers import AutoModel, AutoConfig
        from peft import LoraConfig, get_peft_model, TaskType

        config = AutoConfig.from_pretrained(encoder_name, trust_remote_code=True)
        self.hidden_size = config.hidden_size
        self.model_names = model_names
        self.num_labels = num_labels
        self.problem_type = problem_type

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
        # Target modules differ by architecture
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
        logger.info(f"LoRA: trainable={trainable:,} / total={total:,} ({100*trainable/total:.2f}%)")

        # Per-model output heads (always trainable)
        self.heads = nn.ModuleDict({
            self._key(name): nn.Linear(self.hidden_size, num_labels)
            for name in model_names
        })

        drop_rate = getattr(config, "classifier_dropout", None) or getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(drop_rate)

    @staticmethod
    def _key(model_name: str) -> str:
        return model_name.replace("/", "_").replace("-", "_").replace(".", "_")

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])

        batch_size = pooled.size(0)
        num_models = len(self.model_names)
        all_logits = torch.zeros(batch_size, num_models, self.num_labels, device=pooled.device)

        total_loss = torch.tensor(0.0, device=pooled.device)
        n_valid = 0

        for i, name in enumerate(self.model_names):
            key = self._key(name)
            head_logits = self.heads[key](pooled)
            all_logits[:, i, :] = head_logits

            if labels is not None:
                model_labels = labels[:, i]
                mask = ~torch.isnan(model_labels)
                if mask.any():
                    if self.problem_type == "single_label_classification":
                        loss_fn = nn.CrossEntropyLoss()
                        loss = loss_fn(head_logits[mask], model_labels[mask].long())
                    else:
                        loss_fn = nn.MSELoss()
                        loss = loss_fn(head_logits[mask].squeeze(-1), model_labels[mask])
                    total_loss = total_loss + loss
                    n_valid += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid

        return {"loss": total_loss, "logits": all_logits}

    def predict(self, input_ids, attention_mask, **kwargs):
        with torch.no_grad():
            result = self.forward(input_ids, attention_mask)
        return {
            name: result["logits"][:, i, :]
            for i, name in enumerate(self.model_names)
        }


def train_lora_encoder(
    train_data, test_data, model_names, target, encoder_name,
    epochs=5, lr=1e-4, batch_size=16, max_length=512, device="cuda",
    bucket_size=64, lora_r=16, lora_alpha=32, output_dir="models/lora_encoder",
):
    from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding

    is_classification = target in ("length_bucket", "judge_class")
    num_labels = (1024 // bucket_size) if target == "length_bucket" else (10 if target == "judge_class" else 1)
    problem_type = "single_label_classification" if is_classification else "regression"

    tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)

    train_ds = FusedDataset(train_data, model_names, target, tokenizer, max_length, bucket_size)
    test_ds = FusedDataset(test_data, model_names, target, tokenizer, max_length, bucket_size)

    model = LoRAMultiHeadEncoder(
        encoder_name, model_names, num_labels, problem_type,
        lora_r=lora_r, lora_alpha=lora_alpha,
    ).to(device)

    class LoRACollator:
        def __init__(self, tokenizer):
            self.inner = DataCollatorWithPadding(tokenizer=tokenizer)
        def __call__(self, features):
            labels = [f.pop("labels") for f in features]
            batch = self.inner(features)
            batch["labels"] = torch.stack(labels)
            return batch

    class LoRATrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(labels=labels, **inputs)
            loss = outputs["loss"]
            class _Out:
                def __init__(self, logits, loss):
                    self.logits = logits
                    self.loss = loss
            out = _Out(outputs["logits"].detach(), loss)
            return (loss, out) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs = self._prepare_inputs(inputs)
            labels = inputs.pop("labels")
            with torch.no_grad():
                outputs = model(labels=labels, **inputs)
            return (outputs["loss"], outputs["logits"], labels)

    metrics_fn = make_fused_metrics(model_names, target, bucket_size, log_transform=False)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=LoRACollator(tokenizer),
        compute_metrics=metrics_fn,
    )

    t0 = time.monotonic()
    trainer.train()
    train_time = time.monotonic() - t0

    # Evaluate
    eval_results = trainer.evaluate()

    # Save model
    torch.save({
        "encoder_name": encoder_name,
        "model_names": model_names,
        "num_labels": num_labels,
        "problem_type": problem_type,
        "target": target,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "state_dict": model.state_dict(),
    }, output_path / "lora_encoder_model.pt")

    tokenizer.save_pretrained(str(output_path))

    # Save config
    config = {
        "encoder_name": encoder_name,
        "model_names": model_names,
        "num_labels": num_labels,
        "problem_type": problem_type,
        "target": target,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "bucket_size": bucket_size,
        "train_time_s": train_time,
    }
    with open(output_path / "lora_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save results
    results = {"train_time_s": train_time, "eval": eval_results}
    with open(output_path / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"LoRA encoder saved to {output_path}, train_time={train_time:.0f}s")
    return results


def load_lora_encoder(model_dir: str, device: str = "cpu"):
    """Load a saved LoRA encoder model."""
    from transformers import AutoTokenizer

    model_path = Path(model_dir)
    checkpoint = torch.load(model_path / "lora_encoder_model.pt", map_location=device, weights_only=False)

    model = LoRAMultiHeadEncoder(
        encoder_name=checkpoint["encoder_name"],
        model_names=checkpoint["model_names"],
        num_labels=checkpoint["num_labels"],
        problem_type=checkpoint["problem_type"],
        lora_r=checkpoint.get("lora_r", 16),
        lora_alpha=checkpoint.get("lora_alpha", 32),
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

    return model, tokenizer


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Train LoRA encoder predictor (SR-style)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--test-input", default=None)
    parser.add_argument("--encoder-name", default="roberta-base")
    parser.add_argument("--target",
                        choices=["length", "length_bucket", "similarity",
                                 "judge_class", "reference_score", "deepeval", "quality"],
                        required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--bucket-size", type=int, default=64)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        train_data = [json.loads(l) for l in f]

    test_data = None
    if args.test_input:
        with open(args.test_input) as f:
            test_data = [json.loads(l) for l in f]

    model_names = sorted(train_data[0]["models"].keys())
    logger.info(f"Models: {[m.split('/')[-1] for m in model_names]}, target: {args.target}")

    results = train_lora_encoder(
        train_data=train_data,
        test_data=test_data or train_data[:100],
        model_names=model_names,
        target=args.target,
        encoder_name=args.encoder_name,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        bucket_size=args.bucket_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        output_dir=args.output_dir,
    )

    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()
