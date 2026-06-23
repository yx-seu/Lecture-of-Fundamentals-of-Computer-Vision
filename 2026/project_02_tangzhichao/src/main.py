"""
Sample inference entry point for DINOv2 + Multi-Head ADBA-Head.

Usage:
    python -m src.main --checkpoint outputs/best_model.pth --image_path test.jpg --output results/demo
    python -m src.main --checkpoint outputs/best_model.pth --input data/test_examples --output results/demo
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Constants (must match training)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE = 448
NUM_CLASSES = 51  # 1 background + 50 categories

# 50-class colour palette
PALETTE = np.random.RandomState(42).randint(64, 256, (NUM_CLASSES, 3), dtype=np.uint8)
PALETTE[0] = [0, 0, 0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_image_paths(input_path: Path) -> list:
    if input_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted([p for p in input_path.iterdir() if p.suffix.lower() in exts])
    return [input_path]


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = PALETTE[mask % len(PALETTE)]
    rgb[mask == 0] = [0, 0, 0]
    return rgb


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    mask_rgb = mask_to_rgb(mask)
    overlay = image.astype(np.float32) * (1 - alpha) + mask_rgb.astype(np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def ensure_size_multiple_of_14(image: Image.Image) -> Image.Image:
    """Pad image so H and W are multiples of 14 (DINOv2 requirement)."""
    w, h = image.size
    new_w = ((w + 13) // 14) * 14
    new_h = ((h + 13) // 14) * 14
    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.BILINEAR)
    return image


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: torch.device):
    """Load DINOv2 + ADBA-Head from checkpoint."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from advanced.dinov2_seg import DINOv2Seg

    model = DINOv2Seg(num_classes=NUM_CLASSES, img_size=IMG_SIZE,
                      freeze_backbone=True, pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, image: Image.Image, device: torch.device) -> np.ndarray:
    """Run inference, return predicted mask (H, W) at original image size."""
    orig_w, orig_h = image.size

    # Ensure image size is multiple of 14 (DINOv2 requirement)
    image = ensure_size_multiple_of_14(image)

    # Preprocess
    tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    img_t = tf(image).unsqueeze(0).to(device)
    logits = model(img_t)["logits"]  # (1, 51, 448, 448)

    # Upsample back to original image size
    logits = F.interpolate(logits, size=(orig_h, orig_w),
                           mode="bilinear", align_corners=False)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 + Multi-Head ADBA-Head — Sample Inference"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pth")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Single image to segment")
    parser.add_argument("--input", type=str, default=None,
                        help="Directory of images (alternative to --image_path)")
    parser.add_argument("--output", type=str, default="results/demo_inference",
                        help="Output directory")
    args = parser.parse_args()

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # ---- Load model ----
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"  epoch={ckpt.get('epoch','?')}, mIoU={ckpt.get('best_miou',0):.4f}")
    model = load_model(args.checkpoint, device)
    print("  Model loaded OK")

    # ---- Input ----
    if args.image_path:
        input_paths = [Path(args.image_path)]
    elif args.input:
        input_paths = collect_image_paths(Path(args.input))
    else:
        print("ERROR: provide --image_path or --input")
        return

    if not input_paths:
        print("ERROR: no images found")
        return

    # ---- Output ----
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    # ---- Run ----
    for image_path in input_paths:
        print(f"Processing: {image_path.name}")
        image = Image.open(image_path).convert("RGB")
        pred_mask = run_inference(model, image, device)

        image_np = np.asarray(image)
        overlay = overlay_mask(image_np, pred_mask)

        # Save per-image results
        out_dir = output_root / image_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(pred_mask.astype(np.uint8)).save(out_dir / "pred_mask.png")
        Image.fromarray(mask_to_rgb(pred_mask)).save(out_dir / "pred_color.png")
        Image.fromarray(overlay).save(out_dir / "overlay.png")

        labels = np.unique(pred_mask).astype(int)
        np.savetxt(out_dir / "pred_labels.txt", labels, fmt="%d")
        print(f"  → {out_dir}")

    if hasattr(model, "remove_hooks"):
        model.remove_hooks()
    print("Done!")


if __name__ == "__main__":
    main()
