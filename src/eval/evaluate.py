from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import EyeSequenceDataset
from src.eval.plots import save_evaluation_plots
from src.models.lrcn_vit import LRCNViT
from src.train.train import merge_config


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for batch in tqdm(loader, desc="Evaluating", unit="batch"):
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
    parser.add_argument("--out-dir", default="outputs/eval", help="Where to save metrics and label arrays")
    parser.add_argument("--plots", action="store_true", help="Generate confusion matrix and ROC curve PNGs")
    args = parser.parse_args()

    cfg = merge_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    metadata_csv = cfg["data"].get("metadata_csv", "data/metadata.csv")
    print(f"Loading test split from {metadata_csv}...", flush=True)
    ds = EyeSequenceDataset(metadata_csv, split="test")
    if len(ds) == 0:
        raise SystemExit(f"No test samples found in {metadata_csv}. Train/eval needs a test split.")
    print(f"Test samples: {len(ds)}", flush=True)

    # DataLoader workers often hang silently on macOS; keep eval single-process.
    loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=0)

    print("Building model and loading checkpoint...", flush=True)
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
    print(f"Loaded {args.checkpoint}. Running inference...", flush=True)

    y_true, y_pred, y_prob = run_eval(model, loader, device)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
    }
    print(json.dumps(metrics, indent=2))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "y_true.npy", y_true)
    np.save(out_dir / "y_pred.npy", y_pred)
    np.save(out_dir / "y_prob.npy", y_prob)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics and arrays to {out_dir}/")

    if args.plots:
        save_evaluation_plots(y_true, y_pred, y_prob, out_dir)
        print(f"Saved plots to {out_dir}/confusion_matrix.png and {out_dir}/roc_curve.png")


if __name__ == "__main__":
    main()
