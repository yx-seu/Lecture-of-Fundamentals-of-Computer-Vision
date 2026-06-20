#!/usr/bin/env python3
"""
Demo: run DINOv2 + ADBA-Head inference on test images and visualize results.

Usage:
    python demo.py
    python demo.py --checkpoint outputs/best_model.pth
"""

import os, sys, argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE = 448
NUM_CLASSES = 51
PALETTE = np.random.RandomState(42).randint(64, 256, (NUM_CLASSES, 3), dtype=np.uint8)
PALETTE[0] = [0, 0, 0]

TEST_DIR = Path("data/test_examples/ours")
OUTPUT_DIR = Path("results/demo")
CHECKPOINT_CANDIDATES = [
    "outputs/imagenet_s50_dinov2_adba/best_model/pytorch_model.bin",
    "outputs/best_model.pth",
    # user can also pass --checkpoint
]


def ensure_size_multiple_of_14(image):
    w, h = image.size
    new_w = ((w + 13) // 14) * 14
    new_h = ((h + 13) // 14) * 14
    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.BILINEAR)
    return image


def mask_to_rgb(mask):
    rgb = PALETTE[mask % len(PALETTE)]
    rgb[mask == 0] = [0, 0, 0]
    return rgb


def main():
    parser = argparse.ArgumentParser(description="DINOv2 ADBA-Head Demo")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    # ---- Find checkpoint ----
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        for cand in CHECKPOINT_CANDIDATES:
            if os.path.exists(cand):
                ckpt_path = cand
                break

    if ckpt_path is None or not os.path.exists(ckpt_path):
        print("=" * 60)
        print("  ERROR: No model checkpoint found!")
        print("=" * 60)
        print("\n  Please provide the path to best_model.pth:")
        print("    python demo.py --checkpoint <path/to/best_model.pth>")
        print("\n  Or place it at one of:")
        for c in CHECKPOINT_CANDIDATES:
            print(f"    {c}")
        sys.exit(1)

    print(f"Checkpoint: {ckpt_path}")

    # ---- Load model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from advanced.dinov2_seg import DINOv2Seg

    model = DINOv2Seg(num_classes=NUM_CLASSES, img_size=IMG_SIZE,
                      freeze_backbone=True, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  epoch={ckpt.get('epoch','?')}, mIoU={ckpt.get('best_miou',0):.4f}")

    # ---- Find test images ----
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted([p for p in TEST_DIR.iterdir() if p.suffix.lower() in exts])
    if not images:
        print(f"ERROR: no images in {TEST_DIR}")
        return

    print(f"Images: {len(images)}")

    # ---- Preprocessing ----
    tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # ---- Run inference on all images ----
    results = []
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        image_padded = ensure_size_multiple_of_14(image)
        img_t = tf(image_padded).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_t)["logits"]
        logits = F.interpolate(logits, size=(orig_h, orig_w),
                               mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        results.append({
            "name": img_path.stem,
            "image": np.asarray(image),
            "pred": pred,
            "unique": np.unique(pred).tolist(),
        })

    if hasattr(model, "remove_hooks"):
        model.remove_hooks()

    # ---- Visualize: multi-panel grid ----
    n = len(results)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows * 2, cols, figsize=(3.5 * cols, 3.5 * rows * 2))
    if rows == 1 and cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif rows == 1:
        axes = axes.reshape(2, -1)

    for i, r in enumerate(results):
        row, col = i // cols, i % cols
        img_rgb = r["image"]
        pred_rgb = mask_to_rgb(r["pred"])

        # Original
        ax = axes[row * 2, col]
        ax.imshow(img_rgb)
        ax.set_title(r["name"].replace("_", " "), fontsize=9, fontweight="bold")
        ax.axis("off")

        # Prediction overlay
        ax = axes[row * 2 + 1, col]
        ax.imshow(img_rgb)
        mask_alpha = (r["pred"] > 0).astype(np.float32)[:, :, None] * 0.45
        overlay = img_rgb.astype(np.float32) * (1 - mask_alpha) + pred_rgb.astype(np.float32) * mask_alpha
        ax.imshow(overlay.astype(np.uint8))
        ax.set_title(f"predicted ({len(r['unique'])} classes)", fontsize=8)
        ax.axis("off")

    # Hide unused subplots
    for i in range(n, rows * cols):
        row, col = i // cols, i % cols
        axes[row * 2, col].axis("off")
        axes[row * 2 + 1, col].axis("off")

    plt.suptitle(f"DINOv2 + ADBA-Head Demo  |  {len(images)} test images  |  mIoU={ckpt.get('best_miou',0):.4f}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "demo_overview.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Save individual results ----
    for r in results:
        out_dir = OUTPUT_DIR / r["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(r["pred"].astype(np.uint8)).save(out_dir / "pred_mask.png")
        Image.fromarray(mask_to_rgb(r["pred"])).save(out_dir / "pred_color.png")

        # Overlay
        img = r["image"].astype(np.float32)
        pred_c = mask_to_rgb(r["pred"]).astype(np.float32)
        alpha = (r["pred"] > 0).astype(np.float32)[:, :, None] * 0.45
        ov = (img * (1 - alpha) + pred_c * alpha).astype(np.uint8)
        Image.fromarray(ov).save(out_dir / "overlay.png")

    print(f"\nSaved:")
    print(f"  Overview: {out_path}")
    print(f"  Per-image: {OUTPUT_DIR}/<name>/")
    print("Done!")


if __name__ == "__main__":
    main()
