from __future__ import annotations

import torch
import torch.nn.functional as F


def fgsm_attack(model, frames, blink, labels, eps: float):
    frames_adv = frames.detach().clone().requires_grad_(True)
    logits, _ = model(frames_adv, blink)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    perturbed = frames_adv + eps * frames_adv.grad.sign()
    return perturbed.clamp(0.0, 1.0).detach()


def pgd_attack(model, frames, blink, labels, eps: float, alpha: float, steps: int):
    ori = frames.detach()
    adv = ori.clone()
    for _ in range(steps):
        adv.requires_grad_(True)
        logits, _ = model(adv, blink)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        adv = adv + alpha * adv.grad.sign()
        delta = torch.clamp(adv - ori, min=-eps, max=eps)
        adv = torch.clamp(ori + delta, 0.0, 1.0).detach()
    return adv


def attention_consistency_loss(clean_feat: torch.Tensor, adv_feat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(clean_feat, adv_feat)


def blink_timing_regularizer(
    blink_seq: torch.Tensor, fps: float, min_seconds: float, max_seconds: float
) -> torch.Tensor:
    # Penalize blink durations outside physiologic range.
    min_frames = min_seconds * fps
    max_frames = max_seconds * fps
    blink_binary = (blink_seq < blink_seq.mean(dim=1, keepdim=True)).float()
    durations = blink_binary.sum(dim=1)
    low_pen = torch.relu(min_frames - durations)
    high_pen = torch.relu(durations - max_frames)
    return (low_pen + high_pen).mean()
