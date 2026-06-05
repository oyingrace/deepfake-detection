from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbones import build_backbone, infer_feature_dim


class BlinkHead(nn.Module):
    def __init__(self, in_dim: int = 1, hidden: int = 32, out_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,T
        x = x.unsqueeze(-1)
        x = self.net(x)
        return x.mean(dim=1)


class LRCNViT(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_tiny_patch16_224",
        backbone_pretrained: bool = True,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2,
        use_blink_head: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(backbone_name, pretrained=backbone_pretrained)
        feat_dim = infer_feature_dim(self.backbone, image_size=image_size)
        self.temporal = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.use_blink_head = use_blink_head
        if use_blink_head:
            self.blink_head = BlinkHead(in_dim=1, hidden=32, out_dim=4)
            cls_in = lstm_hidden + 4
        else:
            cls_in = lstm_hidden
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(cls_in, num_classes)

    def extract_frame_features(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: B,T,C,H,W
        b, t, c, h, w = frames.shape
        x = frames.reshape(b * t, c, h, w)
        f = self.backbone(x)
        return f.reshape(b, t, -1)

    def forward(self, frames: torch.Tensor, blink_seq: torch.Tensor):
        feats = self.extract_frame_features(frames)
        out, _ = self.temporal(feats)
        temporal_feat = out[:, -1]
        if self.use_blink_head:
            blink_feat = self.blink_head(blink_seq)
            fused = torch.cat([temporal_feat, blink_feat], dim=-1)
        else:
            fused = temporal_feat
        logits = self.classifier(self.dropout(fused))
        return logits, {"temporal_feat": temporal_feat}
