"""
Comprehensive visualization suite for DINOv2 + MultiHead ADBA-Head.

1. Training curves (loss, mIoU, pAcc) from training_log.csv
2. Prediction overlays on validation samples (multi-sample grid)
3. Failure case analysis (worst IoU predictions)
4. Attention map heatmaps from the model's attention capture hook

Usage:
    # Training curves
    python -m src.visualize_v2 --mode curves --log results/training_log.csv

    # Prediction grid
    python -m src.visualize_v2 --mode predictions --checkpoint outputs/best_model.pth

    # Failure cases
    python -m src.visualize_v2 --mode failures --checkpoint outputs/best_model.pth

    # All
    python -m src.visualize_v2 --mode all --checkpoint outputs/best_model.pth
"""

import os, sys, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from advanced.dinov2_seg import DINOv2Seg
from advanced.dataset_v2 import ImageNetSDataset, IMAGENET_MEAN, IMAGENET_STD
from advanced.metrics_v2 import compute_iou

IMG_SIZE = 448
NUM_CLASSES = 51


def plot_curves(log_path, output_dir):
    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(df["epoch"], df["train_loss"], label="Train", lw=2)
    axes[0].plot(df["epoch"], df["val_loss"], label="Val", lw=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(df["epoch"], df["val_miou"], lw=2, color="darkgreen")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mIoU")
    axes[1].set_title("Validation mIoU"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(df["epoch"], df["val_pixel_acc"], lw=2, color="darkblue")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Pixel Accuracy")
    axes[2].set_title("Validation Pixel Accuracy"); axes[2].grid(True, alpha=0.3)
    plt.suptitle("Training Curves — DINOv2 + Multi-Head ADBA-Head", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out / 'training_curves.png'}")


def plot_predictions(model, device, output_dir, n=8):
    ds = ImageNetSDataset("data/imagenet-s/ImageNetS50", mode="val", size=IMG_SIZE, augment=False)
    scored = []
    for idx in tqdm(range(len(ds)), desc="Scoring"):
        s = ds[idx]
        img_t = s["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_t)["logits"].argmax(dim=1).squeeze(0).cpu().numpy()
        r = compute_iou(torch.from_numpy(pred), s["mask"], NUM_CLASSES, 255)
        scored.append({"idx": idx, "miou": r["miou"], "pred": pred, "class": s["class_name"]})
    scored.sort(key=lambda x: -x["miou"])
    top, seen = [], set()
    for s in scored:
        if s["class"] in seen: continue
        seen.add(s["class"]); top.append(s)
        if len(top) >= n: break

    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 11))
    for col, s in enumerate(top):
        item = ds[s["idx"]]
        img_np = (item["image"].permute(1, 2, 0).cpu().numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)).clip(0, 1)
        gt = item["mask"].numpy(); pred = s["pred"]
        gt_bin, pred_bin = gt > 0, pred > 0
        axes[0, col].imshow(img_np)
        axes[0, col].set_title(s["class"], fontsize=9, fontweight="bold"); axes[0, col].axis("off")
        axes[1, col].imshow(img_np)
        axes[1, col].imshow(np.dstack([gt_bin * 0.15, gt_bin * 0.75, gt_bin * 0.15]), alpha=0.5); axes[1, col].axis("off")
        axes[2, col].imshow(img_np)
        ov = np.zeros_like(img_np)
        ov[gt_bin & pred_bin] = [0.2, 0.7, 0.2]
        ov[pred_bin & ~gt_bin] = [0.95, 0.5, 0.1]
        axes[2, col].imshow(ov, alpha=0.5)
        axes[2, col].set_title(f"IoU={s['miou']:.3f}", fontsize=8); axes[2, col].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("GT", fontsize=13, fontweight="bold")
    axes[2, 0].set_ylabel("Prediction", fontsize=13, fontweight="bold")
    plt.suptitle("Top IoU Predictions — DINOv2 + ADBA-Head", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "prediction_grid.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out / 'prediction_grid.png'}")


def plot_failures(model, device, output_dir, n=8):
    ds = ImageNetSDataset("data/imagenet-s/ImageNetS50", mode="val", size=IMG_SIZE, augment=False)
    scored = []
    for idx in tqdm(range(len(ds)), desc="Scoring"):
        s = ds[idx]
        img_t = s["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_t)["logits"].argmax(dim=1).squeeze(0).cpu().numpy()
        r = compute_iou(torch.from_numpy(pred), s["mask"], NUM_CLASSES, 255)
        scored.append({"idx": idx, "miou": r["miou"], "pacc": r["pixel_acc"],
                       "pred": pred, "class": s["class_name"]})
    scored.sort(key=lambda x: x["miou"])
    worst = scored[:n]
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7.5))
    for col, s in enumerate(worst):
        item = ds[s["idx"]]
        img_np = (item["image"].permute(1, 2, 0).cpu().numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)).clip(0, 1)
        gt, pred = item["mask"].numpy(), s["pred"]
        gt_bin, pred_bin = gt > 0, pred > 0
        axes[0, col].imshow(img_np)
        gt_ov = np.zeros_like(img_np); gt_ov[gt_bin] = [0.15, 0.75, 0.15]
        axes[0, col].imshow(gt_ov, alpha=0.5)
        axes[0, col].set_title(f"{s['class']}", fontsize=8); axes[0, col].axis("off")
        axes[1, col].imshow(img_np)
        ov = np.zeros_like(img_np)
        ov[gt_bin & pred_bin] = [0.2, 0.7, 0.2]
        ov[gt_bin & ~pred_bin] = [1.0, 0.2, 0.2]
        ov[pred_bin & ~gt_bin] = [0.95, 0.5, 0.1]
        axes[1, col].imshow(ov, alpha=0.55)
        axes[1, col].set_title(f"IoU={s['miou']:.3f} pAcc={s['pacc']:.3f}\ngreen=ok red=miss orange=FP", fontsize=7)
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("GT", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Prediction", fontsize=13, fontweight="bold")
    plt.suptitle("Failure Cases — DINOv2 + ADBA-Head (worst IoU)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "failure_cases.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {out / 'failure_cases.png'}")
    print("\nFailure analysis:")
    for s in worst:
        print(f"  {s['class']}: IoU={s['miou']:.4f} pAcc={s['pacc']:.4f}")


def plot_attention(model, device, output_dir, indices=(0, 50, 100, 200)):
    ds = ImageNetSDataset("data/imagenet-s/ImageNetS50", mode="val", size=IMG_SIZE, augment=False)
    for idx in indices:
        item = ds[idx]
        img_t = item["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img_t)
        attn = out["attn"][0].mean(dim=0).mean(dim=0).cpu().numpy()
        patch_attn = attn[5:]
        p = int(np.sqrt(len(patch_attn)))
        attn_map = patch_attn.reshape(p, p)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        img_np = (item["image"].permute(1, 2, 0).cpu().numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)).clip(0, 1)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(attn_map, cmap="hot"); axes[1].set_title("Layer-11 Attention Map"); axes[1].axis("off")
        attn_big = np.array(Image.fromarray((attn_map * 255).astype(np.uint8)).resize((448, 448), Image.BILINEAR)) / 255.0
        axes[2].imshow(img_np); axes[2].imshow(attn_big, cmap="hot", alpha=0.55)
        axes[2].set_title("Overlay"); axes[2].axis("off")
        plt.suptitle(f"Attention Analysis — {item['class_name']} (idx={idx})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        plt.savefig(out / f"attention_{idx:04d}.png", dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved attention maps → {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["curves","predictions","failures","attention","all"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--log", default="results/training_log.csv")
    parser.add_argument("--output", default="results/figures")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if args.mode in ("predictions", "failures", "attention", "all"):
        if args.checkpoint is None:
            print("ERROR: --checkpoint required"); return
        model = DINOv2Seg(num_classes=NUM_CLASSES, img_size=IMG_SIZE, freeze_backbone=True, pretrained=False).to(device).eval()
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Model: epoch={ckpt.get('epoch','?')}, mIoU={ckpt.get('best_miou',0):.4f}")

    if args.mode in ("curves", "all"):
        if Path(args.log).exists():
            plot_curves(args.log, f"{args.output}/curves")
    if args.mode in ("predictions", "all") and model:
        plot_predictions(model, device, f"{args.output}/predictions")
    if args.mode in ("failures", "all") and model:
        plot_failures(model, device, f"{args.output}/failure_cases")
    if args.mode in ("attention", "all") and model:
        plot_attention(model, device, f"{args.output}/attention")

    if model and hasattr(model, "remove_hooks"):
        model.remove_hooks()


if __name__ == "__main__":
    main()
