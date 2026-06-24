#!/usr/bin/env python3
"""
prepare_split.py — Stratified 70/15/15 train/val/test split with a fixed seed.

Locks the test set BEFORE training/baseline so the fine-tuned model and the Groq
baseline are evaluated on exactly the same held-out examples (and so test data can
never leak into training). Mirrors the Colab notebook's 70/15/15 split.

Usage:
    python src/prepare_split.py
Inputs:  data/takemeter_nba.csv  (columns: text,label,...)
Outputs: data/train.csv, data/val.csv, data/test.csv ; outputs/label_map.json
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from taxonomy import LABELS, LABEL2ID, ID2LABEL

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def main():
    df = pd.read_csv(ROOT / "data" / "takemeter_nba.csv")
    df = df[df["label"].isin(LABELS)].dropna(subset=["text"]).reset_index(drop=True)
    print(f"loaded {len(df)} labeled rows")
    print("label distribution:\n", df["label"].value_counts().to_string())

    # 70 / 15 / 15, stratified by label
    train, temp = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["label"])
    val, test = train_test_split(temp, test_size=0.50, random_state=SEED, stratify=temp["label"])

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(ROOT / "data" / f"{name}.csv", index=False)
        print(f"\n{name}: {len(part)}")
        print(part["label"].value_counts().to_string())

    (ROOT / "outputs").mkdir(exist_ok=True)
    with open(ROOT / "outputs" / "label_map.json", "w") as f:
        json.dump({"labels": LABELS, "label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)
    print("\nwrote splits + outputs/label_map.json")


if __name__ == "__main__":
    main()
