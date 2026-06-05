from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

from src.models.lrcn_vit import LRCNViT


class CamWrapper(torch.nn.Module):
    def __init__(self, model: LRCNViT) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blink = torch.zeros((x.shape[0], x.shape[1]), device=x.device)
        logits, _ = self.model(x, blink)
        return logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LRCNViT(backbone_pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    image = np.array(Image.open(args.image).convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
    tensor = transforms.ToTensor()(image).unsqueeze(0).unsqueeze(1).to(device)
    blink = torch.zeros((1, 1), device=device)

    target_layer = [model.backbone.blocks[-1].norm1] if hasattr(model.backbone, "blocks") else [model.backbone]
    cam = GradCAM(model=CamWrapper(model), target_layers=target_layer)
    # For sequence models we use single-frame sequence.
    grayscale_cam = cam(input_tensor=tensor, targets=None)[0]
    cam_img = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))
    print(f"Saved attention map to {args.out}")


if __name__ == "__main__":
    main()
