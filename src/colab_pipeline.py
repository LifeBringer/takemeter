#!/usr/bin/env python3
"""
colab_pipeline.py — Self-contained fine-tune + evaluation to run on a Colab VM via the
`colab` CLI. Reproduces the rubric's intended Colab workflow on real GPU/CPU hardware and
writes Colab-native artifacts.

Expects these uploaded to /content (the Colab default dir):
  train.csv, val.csv, test.csv         (from src/prepare_split.py)
  baseline_results.json                (from src/baseline_groq.py — merged for comparison)

Writes to /content:
  colab_evaluation_results.json, confusion_matrix.png, colab_train_log.json

Version-tolerant: handles transformers 4.x (`evaluation_strategy`, `tokenizer=`) and
5.x (`eval_strategy`, `processing_class=`).
"""
import inspect
import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)

BASE = "/content"
BASE_MODEL = "distilbert-base-uncased"
LABELS = ["analysis", "hot_take", "reaction", "banter"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
SEED = 42


def load_split(name, tok):
    df = pd.read_csv(f"{BASE}/{name}.csv")
    df = df[df["label"].isin(LABELS)].copy()
    df["labels"] = df["label"].map(LABEL2ID)
    ds = Dataset.from_pandas(df[["text", "labels"]], preserve_index=False)
    return df, ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)


def compute_metrics(p):
    preds = np.argmax(p.predictions if hasattr(p, "predictions") else p[0], axis=-1)
    labels = p.label_ids if hasattr(p, "label_ids") else p[1]
    return {"accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
            "weighted_f1": f1_score(labels, preds, average="weighted")}


def make_targs(**kw):
    params = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    return TrainingArguments(**kw)


def make_trainer(**kw):
    params = set(inspect.signature(Trainer.__init__).parameters)
    tok = kw.pop("tokenizer")
    kw["processing_class" if "processing_class" in params else "tokenizer"] = tok
    return Trainer(**kw)


def main():
    print("device:", "cuda" if torch.cuda.is_available() else "cpu",
          "| torch", torch.__version__)
    set_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    _, train_ds = load_split("train", tok)
    _, val_ds = load_split("val", tok)
    test_df, _ = load_split("test", tok)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_df)}")

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=4, id2label=ID2LABEL, label2id=LABEL2ID)
    targs = make_targs(
        output_dir=f"{BASE}/ckpt", eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
        num_train_epochs=16, learning_rate=4e-5, per_device_train_batch_size=16,
        per_device_eval_batch_size=16, warmup_ratio=0.1, weight_decay=0.01,
        logging_steps=10, save_total_limit=1, seed=SEED, report_to="none")
    trainer = make_trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)])
    trainer.train()
    val_metrics = trainer.evaluate()
    print("best val:", {k: round(v, 4) for k, v in val_metrics.items() if isinstance(v, float)})

    # evaluate on test
    enc = tok(list(test_df["text"]), truncation=True, max_length=128, padding=True, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
    y_true = test_df["label"].tolist()
    y_pred = [ID2LABEL[int(p.argmax())] for p in probs]

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("TakeMeter (Colab) — confusion matrix")
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > thr else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(f"{BASE}/confusion_matrix.png", dpi=150)

    baseline = {}
    if os.path.exists(f"{BASE}/baseline_results.json"):
        baseline = json.load(open(f"{BASE}/baseline_results.json")).get("metrics", {})

    out = {
        "ran_on": "google-colab",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "test_size": len(test_df),
        "fine_tuned": {
            "model": "distilbert-base-uncased (fine-tuned on Colab)",
            "accuracy": acc, "macro_f1": macro,
            "per_class": {l: {"precision": rep[l]["precision"], "recall": rep[l]["recall"],
                              "f1": rep[l]["f1-score"], "support": int(rep[l]["support"])} for l in LABELS},
        },
        "baseline": baseline,
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
    }
    json.dump(out, open(f"{BASE}/colab_evaluation_results.json", "w"), indent=2)
    json.dump({"best_val": val_metrics, "config": {"lr": 4e-5, "epochs": 16, "batch": 16}},
              open(f"{BASE}/colab_train_log.json", "w"), indent=2, default=float)

    print(f"\nCOLAB FINE-TUNED  acc={acc:.3f}  macro_f1={macro:.3f}")
    for l in LABELS:
        print(f"  {l:9s} F1={rep[l]['f1-score']:.2f}")
    if baseline:
        print(f"BASELINE          acc={baseline.get('accuracy',0):.3f}  macro_f1={baseline.get('macro_f1',0):.3f}")
    print("wrote /content/colab_evaluation_results.json + confusion_matrix.png")


if __name__ == "__main__":
    main()
