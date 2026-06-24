#!/usr/bin/env python3
"""
build_dataset.py — Assemble the final labeled CSV from the labeling-workflow output.

Joins the workflow's selected labels (source_id -> label/note) back to the comment
texts in candidates.csv, and writes:
  data/takemeter_nba.csv   — the balanced 280-example dataset (the deliverable)
  data/all_labeled.csv     — all 570 consensus labels (supplementary)
  data/difficult_examples.json — adjudicated hard cases (for planning.md §3.4)

Usage:
    python src/build_dataset.py <workflow_output.json>
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Use ALL consensus labels per class (no cap) for maximum training signal. This is
# mildly imbalanced (reaction is the limiting class at ~16%), but every class is far
# under the 70% ceiling. A large TARGET => takes all available in each class.
TARGET = 10_000
SEED = 42


def main():
    wf_path = sys.argv[1]
    result = json.load(open(wf_path))["result"]

    # source_id -> text (+ meta) from the candidate pool
    cand = {r["source_id"]: r for r in csv.DictReader(open(ROOT / "data" / "candidates.csv", encoding="utf-8"))}

    def rows_for(items):
        out = []
        for it in items:
            c = cand.get(it["source_id"])
            if not c:
                continue
            out.append({
                "text": c["text"],
                "label": it["label"],
                "notes": it.get("note", ""),
                "pre_labeled": True,
                "agreed": it.get("agreed"),
                "difficult": it.get("difficult"),
                "confidence": it.get("confidence"),
                "source_id": it["source_id"],
                "created_utc": c.get("created_utc"),
            })
        return out

    cols = ["text", "label", "notes", "pre_labeled", "agreed", "difficult",
            "confidence", "source_id", "created_utc"]

    def write(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    # Representative balanced selection: a reproducible random TARGET/class from each
    # class's full pool, so adjudicated hard cases appear at their NATURAL rate
    # (avoids cherry-picking only easy examples, which would inflate accuracy).
    rng = random.Random(SEED)
    by_class = defaultdict(list)
    for f in result["allFinals"]:
        if cand.get(f["source_id"]):
            by_class[f["label"]].append(f)
    chosen = []
    for label in ["analysis", "hot_take", "reaction", "banter"]:
        pool = sorted(by_class[label], key=lambda x: x["source_id"])  # determinism
        chosen += rng.sample(pool, min(TARGET, len(pool)))

    selected = rows_for(chosen)
    write(ROOT / "data" / "takemeter_nba.csv", selected)
    all_rows = rows_for(result["allFinals"])
    write(ROOT / "data" / "all_labeled.csv", all_rows)

    # difficult examples with text (for §3.4)
    diff = rows_for(result["difficultExamples"])
    json.dump(diff, open(ROOT / "data" / "difficult_examples.json", "w"),
              indent=2, ensure_ascii=False)

    dist = Counter(r["label"] for r in selected)
    print(f"takemeter_nba.csv: {len(selected)} rows")
    for l, n in sorted(dist.items()):
        print(f"  {l:9s} {n}  ({n/len(selected)*100:.1f}%)")
    print(f"all_labeled.csv: {len(all_rows)} rows")
    print(f"difficult_examples.json: {len(diff)} adjudicated hard cases")
    k = result.get("kappa", {})
    print(f"kappa={k.get('kappa'):.3f} (po={k.get('po'):.3f}, n={k.get('n')})")


if __name__ == "__main__":
    main()
