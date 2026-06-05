from __future__ import annotations

import timm
import torch
import torch.nn as nn


def build_backbone(name: str, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
    return model


@torch.no_grad()
def infer_feature_dim(backbone: nn.Module, image_size: int = 224) -> int:
    x = torch.randn(1, 3, image_size, image_size)
    y = backbone(x)
    return int(y.shape[-1])
