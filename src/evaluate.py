#!/usr/bin/env python3
"""
evaluate.py — Evaluate the fine-tuned model on the locked test set, compare to the
Groq baseline, and emit the full report artifacts.

Usage:
    python src/evaluate.py
Inputs:  model/ (from train.py), data/test.csv, outputs/baseline_results.json
Outputs: outputs/evaluation_results.json, outputs/confusion_matrix.png

Includes the confidence-calibration stretch (reliability bins + ECE).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from predict import predict
from taxonomy import LABELS

ROOT = Path(__file__).resolve().parent.parent


def per_class(y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {l: {"precision": rep[l]["precision"], "recall": rep[l]["recall"],
                "f1": rep[l]["f1-score"], "support": int(rep[l]["support"])} for l in LABELS}


def md_confusion(cm):
    head = "| true \\ pred | " + " | ".join(LABELS) + " |\n"
    head += "|" + "---|" * (len(LABELS) + 1) + "\n"
    rows = ""
    for i, l in enumerate(LABELS):
        rows += f"| **{l}** | " + " | ".join(str(int(cm[i][j])) for j in range(len(LABELS))) + " |\n"
    return head + rows


def calibration(y_true, y_pred, confs, bins=(0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)):
    out, ece, n = [], 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = [k for k, c in enumerate(confs) if lo <= c < hi]
        if not idx:
            out.append({"range": f"{lo:.2f}-{hi:.2f}", "n": 0, "accuracy": None, "avg_conf": None})
            continue
        acc = np.mean([y_true[k] == y_pred[k] for k in idx])
        avg = np.mean([confs[k] for k in idx])
        out.append({"range": f"{lo:.2f}-{min(hi,1.0):.2f}", "n": len(idx),
                    "accuracy": float(acc), "avg_conf": float(avg)})
        ece += (len(idx) / n) * abs(acc - avg)
    return out, float(ece)


def main():
    test = pd.read_csv(ROOT / "data" / "test.csv")
    test = test[test["label"].isin(LABELS)].reset_index(drop=True)
    y_true = test["label"].tolist()

    preds = predict(test["text"].tolist())
    y_pred = [p["label"] for p in preds]
    confs = [p["confidence"] for p in preds]

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    # confusion matrix PNG
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("TakeMeter fine-tuned — confusion matrix (test set)")
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > thr else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "confusion_matrix.png", dpi=150)
    print("wrote outputs/confusion_matrix.png")

    # wrong predictions (most-confident errors first — most informative)
    wrong = [{"text": test["text"][k], "true": y_true[k], "pred": y_pred[k],
              "confidence": round(confs[k], 3)}
             for k in range(len(y_true)) if y_true[k] != y_pred[k]]
    wrong.sort(key=lambda w: -w["confidence"])

    # sample classifications (5: spread across confidence)
    order = np.argsort(confs)[::-1]
    pick = list(order[:3]) + list(order[-2:])
    samples = [{"text": test["text"][k], "true": y_true[k], "pred": y_pred[k],
                "confidence": round(confs[k], 3), "correct": y_true[k] == y_pred[k],
                "probs": {l: round(preds[k]["probs"][l], 3) for l in LABELS}} for k in pick]

    cal, ece = calibration(y_true, y_pred, confs)

    baseline = {}
    bpath = ROOT / "outputs" / "baseline_results.json"
    if bpath.exists():
        baseline = json.load(open(bpath))["metrics"]

    results = {
        "test_size": len(test),
        "fine_tuned": {
            "model": "distilbert-base-uncased (fine-tuned)",
            "accuracy": acc, "macro_f1": macro, "weighted_f1": weighted,
            "per_class": per_class(y_true, y_pred),
        },
        "baseline": baseline,
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist(),
                             "markdown": md_confusion(cm)},
        "wrong_predictions": wrong,
        "sample_classifications": samples,
        "calibration": {"bins": cal, "ece": ece},
    }
    with open(ROOT / "outputs" / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nFINE-TUNED  acc={acc:.3f}  macro_f1={macro:.3f}  weighted_f1={weighted:.3f}")
    for l in LABELS:
        c = results["fine_tuned"]["per_class"][l]
        print(f"  {l:9s} P={c['precision']:.2f} R={c['recall']:.2f} F1={c['f1']:.2f} (n={c['support']})")
    if baseline:
        print(f"BASELINE    acc={baseline['accuracy']:.3f}  macro_f1={baseline['macro_f1']:.3f}")
    print(f"\nconfusion matrix:\n{md_confusion(cm)}")
    print(f"errors={len(wrong)}  ECE={ece:.3f}")
    print("wrote outputs/evaluation_results.json")


if __name__ == "__main__":
    main()
