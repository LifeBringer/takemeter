#!/usr/bin/env python3
"""
baseline_groq.py — Zero-shot baseline: prompt Groq llama-3.3-70b-versatile to
classify each TEST example with no task-specific training. Establishes what the
fine-tuned model has to beat.

Usage:
    python src/baseline_groq.py
Requires: GROQ_API_KEY in .env (gitignored). Run prepare_split.py first.
Output:   outputs/baseline_results.json
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import accuracy_score, classification_report, f1_score

from taxonomy import LABELS, RULES

ROOT = Path(__file__).resolve().parent.parent
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = f"""You are an expert annotator of r/nba discourse. Classify the user's comment into EXACTLY ONE of these four labels.

{RULES}

Respond with ONLY the single label word — one of: analysis, hot_take, reaction, banter. No punctuation, no explanation."""


def parse_label(text: str) -> str | None:
    t = (text or "").strip().lower().replace("-", "_").replace(" ", "_")
    for l in LABELS:                       # exact / startswith first
        if t == l or t.startswith(l):
            return l
    for l in LABELS:                       # then substring
        if l in t:
            return l
    return None


def classify(client, comment: str) -> tuple[str | None, str]:
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=0.0, max_tokens=8,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": comment}],
            )
            raw = resp.choices[0].message.content
            return parse_label(raw), raw
        except Exception as e:
            if attempt == 4:
                print(f"  ! giving up on one example: {e}")
                return None, f"ERROR: {e}"
            time.sleep(2 * (attempt + 1))
    return None, "ERROR"


def main():
    load_dotenv(ROOT / ".env")
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY not found in .env")
    client = Groq(api_key=key)

    test = pd.read_csv(ROOT / "data" / "test.csv")
    test = test[test["label"].isin(LABELS)].reset_index(drop=True)
    print(f"baseline on {len(test)} test examples with {MODEL}")

    preds, raws, unparseable = [], [], 0
    for i, row in test.iterrows():
        pred, raw = classify(client, row["text"])
        if pred is None:
            unparseable += 1
            pred = "reaction"  # fallback so metrics are computable; flagged below
        preds.append(pred)
        raws.append(raw)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(test)}")
        time.sleep(0.25)

    y_true, y_pred = test["label"].tolist(), preds
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    metrics = {
        "model": MODEL,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0),
        "per_class": {l: {"precision": report[l]["precision"], "recall": report[l]["recall"],
                          "f1": report[l]["f1-score"], "support": report[l]["support"]} for l in LABELS},
        "unparseable": unparseable,
        "n_test": len(test),
    }
    predictions = [{"text": t, "true": yt, "pred": yp, "raw": r}
                   for t, yt, yp, r in zip(test["text"], y_true, y_pred, raws)]

    (ROOT / "outputs").mkdir(exist_ok=True)
    with open(ROOT / "outputs" / "baseline_results.json", "w") as f:
        json.dump({"prompt_system": SYSTEM_PROMPT, "metrics": metrics, "predictions": predictions},
                  f, indent=2, ensure_ascii=False)

    print(f"\nBASELINE  acc={metrics['accuracy']:.3f}  macro_f1={metrics['macro_f1']:.3f}"
          f"  unparseable={unparseable}")
    for l in LABELS:
        c = metrics["per_class"][l]
        print(f"  {l:9s} P={c['precision']:.2f} R={c['recall']:.2f} F1={c['f1']:.2f} (n={int(c['support'])})")


if __name__ == "__main__":
    main()
