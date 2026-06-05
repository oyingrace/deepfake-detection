"""Rebuild data/metadata.csv from all .npz files in data/processed/."""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
processed = ROOT / "data" / "processed"
out_csv = ROOT / "data" / "metadata.csv"

rows = []
for npz in sorted(processed.glob("*.npz")):
    try:
        d = np.load(npz, allow_pickle=True)
        label = int(d["label"])
        video_id = str(d["video_id"])
        rows.append({"npz_path": str(npz.resolve()), "label": label, "video_id": video_id})
    except Exception as e:
        print(f"skipping {npz.name}: {e}", file=sys.stderr)

# Identity-disjoint split
rng = np.random.default_rng(42)
unique_ids = sorted({r["video_id"] for r in rows})
rng.shuffle(unique_ids)
n = len(unique_ids)
train_ids = set(unique_ids[: int(0.7 * n)])
val_ids = set(unique_ids[int(0.7 * n) : int(0.85 * n)])

for r in rows:
    if r["video_id"] in train_ids:
        r["split"] = "train"
    elif r["video_id"] in val_ids:
        r["split"] = "val"
    else:
        r["split"] = "test"

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["npz_path", "label", "video_id", "split"])
    writer.writeheader()
    writer.writerows(rows)

from collections import Counter
label_counts = Counter(r["label"] for r in rows)
split_counts = Counter(r["split"] for r in rows)
print(f"Rebuilt {len(rows)} sequences")
print(f"  real (0): {label_counts[0]}, fake (1): {label_counts[1]}")
print(f"  train: {split_counts['train']}, val: {split_counts['val']}, test: {split_counts['test']}")
print(f"  written to: {out_csv}")
