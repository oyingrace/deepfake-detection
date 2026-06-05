from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def discover_dataset(root: Path, dataset_name: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        label = 1 if "fake" in str(path).lower() else 0
        identity = path.parent.name
        rows.append(
            {
                "dataset": dataset_name,
                "video_path": str(path.resolve()),
                "label": label,
                "identity": identity,
            }
        )
    return rows


def identity_disjoint_split(df: pd.DataFrame, train: float, val: float) -> pd.DataFrame:
    identities = sorted(df["identity"].unique())
    n = len(identities)
    n_train = int(n * train)
    n_val = int(n * val)
    train_ids = set(identities[:n_train])
    val_ids = set(identities[n_train : n_train + n_val])

    def split_for_identity(identity: str) -> str:
        if identity in train_ids:
            return "train"
        if identity in val_ids:
            return "val"
        return "test"

    df["split"] = df["identity"].map(split_for_identity)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    rows: List[Dict[str, str]] = []
    ffpp = args.data_root / "FaceForensics++"
    celeb = args.data_root / "Celeb-DF"
    if ffpp.exists():
        rows.extend(discover_dataset(ffpp, "FaceForensics++"))
    if celeb.exists():
        rows.extend(discover_dataset(celeb, "Celeb-DF"))
    if not rows:
        raise FileNotFoundError("No supported videos found under data root.")

    df = pd.DataFrame(rows)
    df = identity_disjoint_split(df, args.train_ratio, args.val_ratio)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved metadata: {args.out} ({len(df)} videos)")


if __name__ == "__main__":
    main()
