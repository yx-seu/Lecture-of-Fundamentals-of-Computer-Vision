"""
ImageNet-S50 Semantic Segmentation — Training Script.

Supports two experiment modes for ablation study:
    baseline — DINOv2 backbone + simple bilinear-upsample head
    advanced — DINOv2 backbone + ADBA-Head (Attention Diffusion Bridge Attention)

Usage:
    python src/train.py --mode advanced    # Full ADBA-Head model
    python src/train.py --mode baseline    # Ablation baseline

Key training features:
    - Discriminative LR (head 2e-4, backbone 2e-5) with AdamW
    - Linear warmup (3 epochs) + Cosine annealing
    - Label smoothing (0.1) + CrossEntropyLoss (ignore_index=255)
    - DropPath (0.1) for structural regularisation
    - Best-model checkpointing by val mIoU
"""

import os
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from advanced.dataset_v2 import ImageNetSDataset, collate_fn, IMAGENET_MEAN, IMAGENET_STD
from advanced.dinov2_seg import DINOv2Seg, MultiHeadADBAHead, EMBED_DIM, NUM_HEADS, NUM_PREFIX
from advanced.metrics_v2 import compute_iou, BoundaryAwareLoss


# =========================================================================
# Simple Baseline Head (bilinear upsample — no attention diffusion)
# =========================================================================

class BaselineHead(nn.Module):
    """
    Minimal segmentation head: 1×1 conv projection + bilinear upsample.

    This serves as the ablation baseline to isolate the contribution of
    the attention-diffusion and shallow-bridge components in ADBA-Head.
    """

    def __init__(self, embed_dim: int = 768, num_classes: int = 50):
        super().__init__()
        self.proj = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        """
        Args:
            features: (B, N_total, embed_dim)  — backbone output
        Returns:
            logits:   (B, num_classes, H, W)    — full-resolution logits
        """
        B = features.shape[0]
        # Remove prefix tokens
        x = features[:, NUM_PREFIX:, :]                  # (B, patches, C)
        x = x.transpose(1, 2).reshape(B, -1, patch_h, patch_w)  # (B, C, Hp, Wp)
        x = self.proj(x)                                 # (B, num_classes, Hp, Wp)
        x = F.interpolate(x, scale_factor=14.0, mode="bilinear", align_corners=False)
        return x


# =========================================================================
# Baseline wrapper model
# =========================================================================

class BaselineModel(nn.Module):
    """DINOv2 backbone + BaselineHead."""

    def __init__(self, num_classes: int = 50, img_size: int = 448,
                 pretrained: bool = True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "vit_base_patch14_reg4_dinov2",
            pretrained=pretrained, img_size=img_size, num_classes=0,
        )
        self.head = BaselineHead(embed_dim=EMBED_DIM, num_classes=num_classes)
        self.img_size = img_size
        self.patch_h = img_size // 14
        self.patch_w = img_size // 14

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.backbone.forward_features(x)
        logits = self.head(feat, self.patch_h, self.patch_w)
        return {"logits": logits}


# =========================================================================
# Warmup + Cosine scheduler
# =========================================================================

def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """
    Linear warmup → Cosine annealing.

    Returns a LambdaLR scheduler.
    """

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Linear warmup from 0 → 1
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing from 1 → 0
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# =========================================================================
# One epoch
# =========================================================================

def train_one_epoch(
    model, dataloader, optimizer, criterion, device, epoch: int, scaler=None
) -> dict:
    """Single training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Train E{epoch:02d}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                out = model(images)
                loss = criterion(out["logits"], masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out["logits"], masks)
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return {"loss": total_loss / total_samples}


@torch.no_grad()
def validate(model, dataloader, criterion, num_classes, device) -> dict:
    """Validation: loss + mIoU."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds, all_targets = [], []

    pbar = tqdm(dataloader, desc="Val", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        out = model(images)
        logits = out["logits"]

        loss = criterion(logits, masks)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        all_preds.append(logits.cpu())
        all_targets.append(masks.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    iou_result = compute_iou(preds, targets, num_classes, ignore_index=255)

    return {
        "loss": total_loss / total_samples,
        "miou": iou_result["miou"],
        "pixel_acc": iou_result["pixel_acc"],
    }


# =========================================================================
# Main entry point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ImageNet-S50 Segmentation")
    parser.add_argument("--mode", type=str, default="advanced",
                        choices=["advanced", "baseline"],
                        help="advanced=ADBA-Head, baseline=bilinear head")
    parser.add_argument("--data-root", type=str,
                        default="data/imagenet-s/ImageNetS50")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=22)
    parser.add_argument("--img-size", type=int, default=448,
                        help="input size (multiple of 14)")
    parser.add_argument("--lr-head", type=float, default=2e-4)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True,
                        help="use automatic mixed precision")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="directory for checkpoints and logs")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="load DINOv2 pretrained weights")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--resume", type=str, default=None,
                        help="path to checkpoint to resume training from")
    parser.add_argument("--bg-weight", type=float, default=1.0,
                        help="background class weight in BoundaryAwareLoss (0=ignore bg)")
    parser.add_argument("--edge-beta", type=float, default=2.0,
                        help="boundary pixel amplification (0=no edge boost)")
    args = parser.parse_args()

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Output directory ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{args.mode}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    print(f"Output: {out_dir}")

    # ---- Datasets ----
    print(f"\nLoading datasets (size={args.img_size})...")
    ds_train = ImageNetSDataset(args.data_root, mode="train-semi-plus" if os.path.isdir(
        os.path.join(args.data_root, "sam_pseudo")) else "train-semi",
                                size=args.img_size, augment=True)
    ds_val = ImageNetSDataset(args.data_root, mode="val",
                              size=args.img_size, augment=False)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers,
                          pin_memory=True, collate_fn=collate_fn,
                          drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, collate_fn=collate_fn)

    print(f"Train-semi: {len(ds_train)}  |  Val: {len(ds_val)}")
    print(f"Batches/epoch: train={len(dl_train)}, val={len(dl_val)}")

    # ---- Model ----
    print(f"\nBuilding model ({args.mode})...")
    if args.mode == "advanced":
        model = DINOv2Seg(num_classes=ds_train.num_classes,
                          img_size=args.img_size,
                          freeze_backbone=True,
                          pretrained=args.pretrained)
    else:
        model = BaselineModel(num_classes=ds_train.num_classes,
                              img_size=args.img_size,
                              pretrained=args.pretrained)

    model = model.to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    # ---- Differential learning rates + selective weight decay ----
    # Rule: NO weight decay on LayerNorm / BatchNorm weights, nor on any bias.
    # This is standard practice for transformer+CNN hybrid architectures and
    # prevents the optimizer from needlessly shrinking normalisation scales.
    def _exclude_from_decay(name: str, param: torch.Tensor) -> bool:
        return param.ndim < 2 or "norm" in name.lower() or "bias" in name.lower()

    head_decay, head_no_decay = [], []
    backbone_decay, backbone_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        nod = _exclude_from_decay(name, param)
        if "seg_head" in name or "head" in name:
            (head_no_decay if nod else head_decay).append(param)
        else:
            (backbone_no_decay if nod else backbone_decay).append(param)

    optimizer = AdamW([
        {"params": head_decay,       "lr": args.lr_head,     "weight_decay": args.weight_decay},
        {"params": head_no_decay,    "lr": args.lr_head,     "weight_decay": 0.0},
        {"params": backbone_decay,   "lr": args.lr_backbone, "weight_decay": args.weight_decay},
        {"params": backbone_no_decay,"lr": args.lr_backbone, "weight_decay": 0.0},
    ], weight_decay=0.0)  # weight_decay set per-group above

    n_decay = len(head_decay) + len(backbone_decay)
    n_nodecay = len(head_no_decay) + len(backbone_no_decay)
    print(f"Optimizer: AdamW (head_lr={args.lr_head}, backbone_lr={args.lr_backbone})")
    print(f"  w/  weight_decay: {n_decay} params  (conv/linear weights)")
    print(f"  w/o weight_decay: {n_nodecay} params  (norm weights, biases)")

    # ---- Scheduler ----
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)
    print(f"Scheduler: Linear warmup ({args.warmup_epochs}ep) + Cosine ({args.epochs}ep)")

    # ---- Loss (boundary-aware with background re-weighting) ----
    criterion = BoundaryAwareLoss(
        num_classes=ds_train.num_classes,
        bg_weight=args.bg_weight,
        edge_beta=args.edge_beta,
        ignore_index=255,
    ).to(device)
    print(f"Loss: BoundaryAwareLoss (bg_weight={args.bg_weight}, edge_beta={args.edge_beta}, ignore=255)")

    # ---- AMP ----
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None
    print(f"AMP: {'on' if scaler else 'off'}")

    # ---- Resume ----
    start_epoch = 1
    best_miou = 0.0
    best_epoch = 0
    history = []

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        # Rebuild scheduler with new total_epochs, then fast-forward to start_epoch.
        # Loading old scheduler state doesn't work because total_epochs may differ.
        scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)
        for _ in range(start_epoch - 1):
            scheduler.step()  # advance last_epoch to start_epoch - 1
        # Override LR with command-line args
        lr_values = [args.lr_head, args.lr_head, args.lr_backbone, args.lr_backbone]
        for pg, lr in zip(optimizer.param_groups, lr_values):
            pg["lr"] = lr
        if hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs = lr_values[:len(scheduler.base_lrs)]
        best_miou = 0.0   # track best in THIS resumed run
        best_epoch = 0
        # Create a NEW output folder for resumed training
        out_dir = Path(args.output_dir) / f"{args.mode}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        print(f"  Resuming from epoch {start_epoch}/{args.epochs}")
        print(f"  Source ckpt mIoU={ckpt.get('best_miou', 0):.4f} @ ep {ckpt.get('epoch', 0)}")
        print(f"  LR reset: head={args.lr_head:.1e}, backbone={args.lr_backbone:.1e}")
        print(f"  Scheduler rebuilt for {args.epochs} epochs, warmed to step {start_epoch-1}")
        print(f"  New output: {out_dir}")

    # ---- Training loop ----
    print(f"\n{'='*60}")
    print(f"Training — epochs {start_epoch}→{args.epochs}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        # --- Train ---
        train_metrics = train_one_epoch(
            model, dl_train, optimizer, criterion, device, epoch, scaler=scaler
        )

        # --- Validate ---
        val_metrics = validate(model, dl_val, criterion,
                               num_classes=ds_train.num_classes, device=device)

        # --- Scheduler step ---
        scheduler.step()
        current_lr_head = optimizer.param_groups[0]["lr"]
        current_lr_backbone = optimizer.param_groups[2]["lr"] if len(optimizer.param_groups) > 2 else "N/A"

        # --- Log ---
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_miou": val_metrics["miou"],
            "val_pixel_acc": val_metrics["pixel_acc"],
        })

        print(
            f"E{epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_mIoU={val_metrics['miou']:.4f}  "
            f"val_pAcc={val_metrics['pixel_acc']:.4f}  "
            f"lr_head={current_lr_head:.2e}"
        )

        # --- Checkpoint ---
        is_best = val_metrics["miou"] > best_miou
        if is_best:
            best_miou = val_metrics["miou"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": best_miou,
                "args": vars(args),
            }, ckpt_dir / "best_model.pth")
            print(f"  ✅ Best model saved (mIoU={best_miou:.4f})")

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, ckpt_dir / "last_model.pth")

    # ---- Final summary ----
    print(f"\n{'='*60}")
    print(f"Training finished!")
    print(f"Best val mIoU: {best_miou:.4f}  @ epoch {best_epoch}")
    print(f"Checkpoint: {ckpt_dir / 'best_model.pth'}")
    print(f"{'='*60}")

    # Save history as simple text log
    log_path = out_dir / "training_log.csv"
    with open(log_path, "w") as f:
        keys = history[0].keys()
        f.write(",".join(keys) + "\n")
        for entry in history:
            f.write(",".join(str(entry[k]) for k in keys) + "\n")
    print(f"Log saved: {log_path}")

    # Clean up hooks for ADBA model
    if args.mode == "advanced" and hasattr(model, "remove_hooks"):
        model.remove_hooks()


if __name__ == "__main__":
    main()
