from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import EyeSequenceDataset
from src.models.lrcn_vit import LRCNViT
from src.train.adversarial import (
    attention_consistency_loss,
    blink_timing_regularizer,
    fgsm_attack,
    pgd_attack,
)
from src.utils import ensure_dir, load_yaml, set_seed


def build_model(cfg: Dict) -> LRCNViT:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    return LRCNViT(
        backbone_name=model_cfg["backbone"],
        backbone_pretrained=model_cfg["backbone_pretrained"],
        lstm_hidden=model_cfg["lstm_hidden"],
        lstm_layers=model_cfg["lstm_layers"],
        dropout=model_cfg["dropout"],
        num_classes=model_cfg["num_classes"],
        use_blink_head=model_cfg.get("use_blink_head", True),
        image_size=data_cfg["image_size"],
    )


def merge_config(config_path: str) -> Dict:
    cfg = load_yaml(config_path)
    if "inherits" not in cfg:
        return cfg
    merged: Dict = {}
    for p in cfg["inherits"]:
        parent = load_yaml(p)
        for k, v in parent.items():
            if isinstance(v, dict):
                merged.setdefault(k, {}).update(v)
            else:
                merged[k] = v
    for k, v in cfg.items():
        if k == "inherits":
            continue
        if isinstance(v, dict):
            merged.setdefault(k, {}).update(v)
        else:
            merged[k] = v
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = merge_config(args.config)

    set_seed(cfg["project"]["seed"])
    out_dir = ensure_dir(cfg["project"]["output_dir"])

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    metadata_csv = cfg["data"].get("metadata_csv", "data/metadata.csv")
    train_ds = EyeSequenceDataset(metadata_csv, split="train")
    val_ds = EyeSequenceDataset(metadata_csv, split="val")
    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"])

    model = build_model(cfg).to(device)
    optim = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    best_val = 0.0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}")
        for batch in pbar:
            frames = batch["frames"].to(device)
            blink = batch["blink"].to(device)
            labels = batch["label"].to(device)

            logits_clean, aux_clean = model(frames, blink)
            loss_clean = F.cross_entropy(logits_clean, labels)
            loss = cfg["adv"]["clean_weight"] * loss_clean

            if cfg["adv"]["enabled"]:
                if cfg["adv"]["attack"].lower() == "fgsm":
                    adv_frames = fgsm_attack(model, frames, blink, labels, eps=cfg["adv"]["fgsm_eps"])
                else:
                    adv_frames = pgd_attack(
                        model,
                        frames,
                        blink,
                        labels,
                        eps=cfg["adv"]["eps"],
                        alpha=cfg["adv"]["alpha"],
                        steps=cfg["adv"]["steps"],
                    )
                logits_adv, aux_adv = model(adv_frames, blink)
                loss_adv = F.cross_entropy(logits_adv, labels)
                loss = loss + cfg["adv"]["adv_weight"] * loss_adv

                if cfg["aat"]["enabled"]:
                    attn_loss = attention_consistency_loss(aux_clean["temporal_feat"], aux_adv["temporal_feat"])
                    blink_loss = blink_timing_regularizer(
                        blink,
                        fps=cfg["aat"]["fps"],
                        min_seconds=cfg["aat"]["blink_min_seconds"],
                        max_seconds=cfg["aat"]["blink_max_seconds"],
                    )
                    loss = (
                        loss
                        + cfg["aat"]["attention_consistency_weight"] * attn_loss
                        + cfg["aat"]["blink_timing_weight"] * blink_loss
                    )

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optim.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_acc = evaluate_simple(model, val_loader, device)
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), out_dir / "best.pt")
        print(f"Epoch {epoch + 1}: val_acc={val_acc:.4f}, best={best_val:.4f}")


@torch.no_grad()
def evaluate_simple(model, loader, device: str) -> float:
    model.eval()
    total = 0
    correct = 0
    for batch in loader:
        frames = batch["frames"].to(device)
        blink = batch["blink"].to(device)
        labels = batch["label"].to(device)
        logits, _ = model(frames, blink)
        pred = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return correct / max(total, 1)


if __name__ == "__main__":
    main()
