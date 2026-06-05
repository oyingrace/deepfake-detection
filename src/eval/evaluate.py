from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data.dataset import EyeSequenceDataset
from src.models.lrcn_vit import LRCNViT
from src.train.train import merge_config


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for batch in loader:
        frames = batch["frames"].to(device)
        blink = batch["blink"].to(device)
        labels = batch["label"].cpu().numpy()
        logits, _ = model(frames, blink)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(labels.tolist())
        y_pred.extend(pred.tolist())
        y_prob.extend(probs.tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = merge_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata_csv = cfg["data"].get("metadata_csv", "data/metadata.csv")
    ds = EyeSequenceDataset(metadata_csv, split="test")
    loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    model = LRCNViT(
        backbone_name=cfg["model"]["backbone"],
        backbone_pretrained=False,
        lstm_hidden=cfg["model"]["lstm_hidden"],
        lstm_layers=cfg["model"]["lstm_layers"],
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
        use_blink_head=cfg["model"].get("use_blink_head", True),
        image_size=cfg["data"]["image_size"],
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    y_true, y_pred, y_prob = run_eval(model, loader, device)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
