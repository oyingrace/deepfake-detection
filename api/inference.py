from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.eye_extract import extract_sequences_from_video_bytes
from src.models.lrcn_vit import LRCNViT
from src.utils import load_yaml

_BLINK_THRESHOLD = 0.2
_BLINK_MIN_FRAMES = 2


def _count_blink_rate(ear_sequences: list[np.ndarray], fps: float = 25.0) -> float:
    """Estimate blinks per second from EAR signal across all sequences."""
    if not ear_sequences:
        return 0.0
    ear = np.concatenate(ear_sequences)
    blink = ear < _BLINK_THRESHOLD
    transitions = np.diff(blink.astype(int))
    blink_starts = int((transitions == 1).sum())
    total_seconds = len(ear) / fps
    return round(blink_starts / max(total_seconds, 1e-6), 2)


def load_model(checkpoint_path: str, config_path: str, device: str = "cpu") -> LRCNViT:
    cfg = load_yaml(config_path)
    model = LRCNViT(
        backbone_name=cfg["model"]["backbone"],
        backbone_pretrained=False,
        lstm_hidden=cfg["model"]["lstm_hidden"],
        lstm_layers=cfg["model"]["lstm_layers"],
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
        use_blink_head=cfg["model"].get("use_blink_head", True),
        image_size=cfg["data"]["image_size"],
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


@torch.no_grad()
def predict_video(
    video_bytes: bytes,
    model: LRCNViT,
    seq_len: int = 8,
    device: str = "cpu",
) -> dict:
    sequences = extract_sequences_from_video_bytes(video_bytes, label=0, video_id="upload", seq_len=seq_len)

    if not sequences:
        return {
            "label": "UNKNOWN",
            "confidence": 0.0,
            "blink_rate": 0.0,
            "frame_scores": [],
            "attention_map_url": None,
        }

    frame_scores: list[float] = []
    ear_arrays: list[np.ndarray] = []

    for seq in sequences:
        frames = torch.tensor(
            seq["frames"].astype(np.float32) / 255.0
        ).permute(0, 3, 1, 2).unsqueeze(0).to(device)   # 1,T,C,H,W
        ear = torch.tensor(seq["ear"], dtype=torch.float32).unsqueeze(0).to(device)

        logits, _ = model(frames, ear)
        prob_fake = float(torch.softmax(logits, dim=1)[0, 1].item())
        frame_scores.append(prob_fake)
        ear_arrays.append(seq["ear"])

    confidence = float(np.mean(frame_scores))
    label = "FAKE" if confidence >= 0.5 else "REAL"
    blink_rate = _count_blink_rate(ear_arrays)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "blink_rate": blink_rate,
        "frame_scores": [round(s, 4) for s in frame_scores],
        "attention_map_url": None,
    }
