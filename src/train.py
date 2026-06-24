#!/usr/bin/env python3
"""
train.py — Fine-tune distilbert-base-uncased on the TakeMeter r/nba dataset.

Targets transformers 5.x (uses `eval_strategy` and `processing_class`).
Run `python src/prepare_split.py` first.

Usage:
    python src/train.py            # defaults below
    python src/train.py --epochs 8 --lr 2e-5 --batch 16

KEY HYPERPARAMETER DECISION (documented in README):
  The Colab default is a FIXED 3 epochs. With only ~196 training examples on a
  subjective task, a fixed epoch count risks under- or over-fitting, and the right
  number is noisy run-to-run. So we train up to --epochs (default 8) with
  EARLY STOPPING on validation macro-F1 (patience 2) and load_best_model_at_end,
  which trains until validation macro-F1 plateaus and reverts to the best
  checkpoint. We also add warmup (10%) + weight decay (0.01) for small-data
  stability. The actual best epoch chosen is recorded in outputs/train_log.json.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)

from taxonomy import LABELS, LABEL2ID, ID2LABEL


class WeightedTrainer(Trainer):
    """Trainer with optional inverse-frequency class weights (for the imbalanced set)."""

    def __init__(self, class_weights=None, **kw):
        super().__init__(**kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = torch.nn.CrossEntropyLoss(weight=w)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "distilbert-base-uncased"
SEED = 42


def load_split(name, tokenizer):
    df = pd.read_csv(ROOT / "data" / f"{name}.csv")
    df = df[df["label"].isin(LABELS)].copy()
    df["labels"] = df["label"].map(LABEL2ID)
    ds = Dataset.from_pandas(df[["text", "labels"]], preserve_index=False)
    ds = ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=128), batched=True)
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--class_weights", action="store_true",
                    help="use inverse-frequency class weights (for imbalance)")
    args = ap.parse_args()

    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_ds, val_ds = load_split("train", tokenizer), load_split("val", tokenizer)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    class_weights = None
    if args.class_weights:
        counts = Counter(train_ds["labels"])
        n, k = len(train_ds), len(LABELS)
        class_weights = torch.tensor([n / (k * counts[i]) for i in range(k)], dtype=torch.float)
        print("class weights:", {ID2LABEL[i]: round(float(class_weights[i]), 3) for i in range(k)})

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    targs = TrainingArguments(
        output_dir=str(ROOT / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_total_limit=1,
        seed=SEED,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()
    final_val = trainer.evaluate()
    print("best val metrics:", final_val)

    out = ROOT / "model"
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    # record training history + the epoch that won (for the README hyperparam writeup)
    hist = trainer.state.log_history
    epoch_evals = [h for h in hist if "eval_macro_f1" in h]
    best = max(epoch_evals, key=lambda h: h["eval_macro_f1"]) if epoch_evals else {}
    (ROOT / "outputs").mkdir(exist_ok=True)
    with open(ROOT / "outputs" / "train_log.json", "w") as f:
        json.dump({
            "base_model": BASE_MODEL,
            "hyperparameters": {"max_epochs": args.epochs, "lr": args.lr,
                                "batch": args.batch, "early_stopping_patience": args.patience,
                                "warmup_ratio": 0.1, "weight_decay": 0.01, "seed": SEED,
                                "class_weights": args.class_weights},
            "best_epoch": best.get("epoch"),
            "best_val_macro_f1": best.get("eval_macro_f1"),
            "final_val_metrics": final_val,
            "epoch_evals": epoch_evals,
        }, f, indent=2)
    print(f"saved model -> {out} ; best epoch={best.get('epoch')} macro_f1={best.get('eval_macro_f1')}")


if __name__ == "__main__":
    main()
