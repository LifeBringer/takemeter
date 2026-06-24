#!/usr/bin/env python3
"""
calibration_plot.py — Reliability diagram for the confidence-calibration stretch.
Reads outputs/evaluation_results.json (calibration bins) -> outputs/calibration.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
cal = json.load(open(ROOT / "outputs" / "evaluation_results.json"))["calibration"]
bins = [b for b in cal["bins"] if b["n"] > 0]

xs = [b["avg_conf"] for b in bins]
ys = [b["accuracy"] for b in bins]
ns = [b["n"] for b in bins]

fig, ax = plt.subplots(figsize=(5.2, 5))
ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
ax.plot(xs, ys, "o-", color="#1f77b4", label="fine-tuned model")
for x, y, n in zip(xs, ys, ns):
    ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8)
ax.set_xlabel("Mean predicted confidence")
ax.set_ylabel("Actual accuracy")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title(f"Reliability diagram (ECE = {cal['ece']:.3f})")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(ROOT / "outputs" / "calibration.png", dpi=150)
print(f"wrote outputs/calibration.png (ECE={cal['ece']:.3f})")
for b in bins:
    print(f"  conf {b['range']}: acc={b['accuracy']:.2f} (n={b['n']}, avg_conf={b['avg_conf']:.2f})")
