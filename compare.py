# compare.py
# Compares the machine learning model and the deep learning model.
# Run it after train_ml.py and train_dl.py.

import json

import matplotlib

matplotlib.use("Agg")  # draw the chart into a file (no window needed)
import matplotlib.pyplot as plt

from data import FOLDER

results = []
for name in ("results_ml.json", "results_dl.json"):
    path = FOLDER / name
    if not path.exists():
        raise SystemExit(f"Missing {name}. Run train_ml.py and train_dl.py first.")
    with open(path) as f:
        results.append(json.load(f))

# Text table
print(f"\n{'Model':<34}{'Accuracy':>10}{'F1 (spam)':>12}{'Time (s)':>10}")
print("-" * 66)
for r in results:
    print(
        f"{r['model']:<34}{r['accuracy']:>10.4f}{r['f1_spam']:>12.4f}"
        f"{r['training_seconds']:>10}"
    )

winner = max(results, key=lambda r: r["f1_spam"])
print(f"\nBest model on spam detection (F1-score): {winner['model']}")

# Bar chart
labels = ["Accuracy", "F1 (spam)"]
width = 0.35
positions = range(len(labels))

fig, ax = plt.subplots(figsize=(7, 4))
for i, r in enumerate(results):
    values = [r["accuracy"], r["f1_spam"]]
    bars = ax.bar([p + i * width for p in positions], values, width, label=r["model"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.3f}",
                ha="center", fontsize=9)

ax.set_xticks([p + width / 2 for p in positions])
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1)
ax.set_title("Machine learning vs deep learning (SMS spam)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(FOLDER / "comparison.png", dpi=150)
print("Saved: comparison.png")
