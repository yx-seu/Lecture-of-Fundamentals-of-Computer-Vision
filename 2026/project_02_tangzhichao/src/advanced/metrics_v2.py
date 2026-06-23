"""
Evaluation metrics and loss functions for semantic segmentation.

- mIoU (Mean Intersection over Union)
- Per-class IoU
- Pixel Accuracy (auxiliary)
- BoundaryAwareLoss (edge-weighted cross-entropy with background re-weighting)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> dict:
    """
    GPU-accelerated mIoU via confusion matrix (single bincount, no Python loop).

    Args:
        pred:        (N, C, H, W) logits or (N, H, W) class indices
        target:      (N, H, W) ground-truth class indices
        num_classes: total number of classes (including background)
        ignore_index: pixels with this label are excluded

    Returns:
        {'miou': float, 'iou_per_class': list[float], 'pixel_acc': float}
    """
    if pred.dim() == 4 and pred.size(1) > 1:
        pred = pred.argmax(dim=1)

    # Keep on GPU — filter ignored pixels
    mask = target != ignore_index
    pred = pred[mask].long()
    target = target[mask].long()

    if pred.numel() == 0:
        return {"miou": 0.0, "iou_per_class": [0.0] * num_classes, "pixel_acc": 0.0}

    # Confusion matrix: row=pred, col=target
    cm = torch.bincount(
        pred * num_classes + target,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes).float()

    pixel_acc = cm.diag().sum() / cm.sum()

    intersection = cm.diag()
    union = cm.sum(dim=1) + cm.sum(dim=0) - intersection
    iou = intersection / union.clamp(min=1)
    iou_list = iou.cpu().tolist()

    # mIoU over classes that actually appear (union > 0)
    valid = union > 0
    miou = iou[valid].mean().item() if valid.any() else 0.0

    return {
        "miou": miou,
        "iou_per_class": iou_list,
        "pixel_acc": pixel_acc.item(),
    }


class AverageMeter:
    """Track a series of values and report their average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


def compute_miou_over_dataset(
    model,
    dataloader,
    num_classes: int,
    device: torch.device,
    ignore_index: int = 255,
) -> dict:
    """
    Evaluate mIoU over an entire dataset.

    Returns same dict as compute_iou().
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            out = model(images)
            logits = out["logits"]  # (B, C, H, W)

            all_preds.append(logits.cpu())
            all_targets.append(masks.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return compute_iou(preds, targets, num_classes, ignore_index=ignore_index)


class BoundaryAwareLoss(nn.Module):
    """
    Edge-aware cross-entropy with background re-weighting.

    - Background (class 0) gets reduced weight (``bg_weight``) to prevent
      large-area background from dominating gradients, but is still supervised
      (no blind ignore_index).
    - Truly unlabeled pixels (255) are ignored via ``ignore_index=255``.
    - Object-boundary pixels receive an amplified loss (``edge_beta``) via
      Laplacian edge detection on the ground-truth mask, forcing the model
      to produce sharp, accurate borders.

    Args:
        num_classes:  total classes including background (default 51)
        bg_weight:    weight multiplier for background class 0  (default 0.4)
        edge_beta:    amplification factor for boundary pixels (default 2.0)
    """

    def __init__(self, num_classes: int = 51, bg_weight: float = 0.4,
                 edge_beta: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.bg_weight = bg_weight
        self.edge_beta = edge_beta
        self.ignore_index = ignore_index

        # Per-class weight: class 0 (bg) gets reduced weight
        weight = torch.ones(num_classes)
        weight[0] = bg_weight
        self.register_buffer("class_weight", weight)

        # Laplacian kernel for edge detection on GT masks
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=torch.float32)
        kernel = kernel.reshape(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", kernel)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, num_classes, H, W)
            targets: (B, H, W)  integer class labels
        Returns:
            scalar loss
        """
        # 1. Per-pixel cross-entropy (ignored pixels get zero loss)
        # Ensure class_weight is on the same device as logits
        weight = self.class_weight.to(logits.device)
        base_loss = F.cross_entropy(
            logits, targets,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )  # (B, H, W)

        # 2. Edge mask from ground-truth via Laplacian
        tgt_float = targets.unsqueeze(1).float()        # (B, 1, H, W)
        tgt_float = torch.where(targets.unsqueeze(1) == self.ignore_index,
                                torch.zeros_like(tgt_float), tgt_float)
        edge = F.conv2d(tgt_float, self.laplacian_kernel, padding=1)
        edge_mask = (edge.abs() > 0.1).float().squeeze(1)  # (B, H, W)

        # 3. Weight matrix: boundary pixels get amplified
        weight_matrix = 1.0 + self.edge_beta * edge_mask  # (B, H, W)

        # 4. Weighted mean over valid pixels
        valid_mask = targets != self.ignore_index
        n_valid = valid_mask.sum().clamp(min=1)
        loss = (base_loss * weight_matrix).sum() / n_valid
        return loss


if __name__ == "__main__":
    # Quick smoke test — metrics
    pred = torch.randint(0, 5, (2, 128, 128))
    target = torch.randint(0, 5, (2, 128, 128))
    result = compute_iou(pred, target, num_classes=5)
    print(f"mIoU: {result['miou']:.4f}")
    print(f"Pixel Acc: {result['pixel_acc']:.4f}")

    # Quick smoke test — BoundaryAwareLoss
    print("\nBoundaryAwareLoss test:")
    criterion = BoundaryAwareLoss(num_classes=51)
    logits = torch.randn(2, 51, 128, 128)
    targets = torch.randint(0, 51, (2, 128, 128))
    targets[:, :10, :10] = 255  # some ignored pixels
    loss_val = criterion(logits, targets)
    print(f"  loss = {loss_val.item():.4f}")
    print(f"  class_weight[:3] = {criterion.class_weight[:3].tolist()}")
