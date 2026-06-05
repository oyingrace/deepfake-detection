from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EyeSequenceDataset(Dataset):
    def __init__(self, metadata_csv: str, split: str) -> None:
        self.samples: List[Dict[str, str]] = []
        df = pd.read_csv(metadata_csv)
        df = df[df["split"] == split]
        for row in df.to_dict(orient="records"):
            if "npz_path" in row:
                self.samples.append(
                    {"path": str(row["npz_path"]), "label": int(row["label"])}
                )
                continue
            # Legacy layout from extract_eye_sequences.py
            seq_dir = Path(row["sequence_dir"])
            for npz in sorted(seq_dir.glob("*.npz")):
                self.samples.append({"path": str(npz), "label": int(row["label"])})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        obj = np.load(sample["path"])
        frames = obj["frames"].astype(np.float32) / 255.0
        ear_key = "ear" if "ear" in obj else "blink"
        ear = obj[ear_key].astype(np.float32)
        # T,H,W,C -> T,C,H,W
        frames = np.transpose(frames, (0, 3, 1, 2))
        return {
            "frames": torch.tensor(frames),
            "ear": torch.tensor(ear),
            "blink": torch.tensor(ear),
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }
