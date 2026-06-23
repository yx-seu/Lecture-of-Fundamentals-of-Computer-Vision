"""
Model-guided SAM labeling: Our model → bbox/points → SAM → refined mask.

Key difference from v1: replaces Grounding DINO with our own trained model.
Our model knows the 50 ImageNet-S classes → provides the correct bbox →
SAM produces a precise mask for the correct object.

Usage:
    conda activate vit_seg
    python src/sam_label_v2.py --topk 20
"""

import os, sys, argparse, random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from advanced.dinov2_seg import DINOv2Seg
from advanced.dataset_v2 import ImageNetSDataset, IMAGENET_MEAN, IMAGENET_STD


def load_sam_predictor(ckpt: str, device: torch.device):
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_h"](checkpoint=ckpt).to(device)
    sam.eval()
    return SamPredictor(sam)


def load_our_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DINOv2Seg(num_classes=51, img_size=448, pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def mask_to_bbox(mask: np.ndarray) -> np.ndarray | None:
    """Binary mask → bounding box (x1, y1, x2, y2), or None if empty."""
    ys, xs = np.where(mask)
    if len(ys) < 50:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()])


def model_predict_mask(model, img_pil: Image.Image, cls_id: int,
                       device: torch.device, img_size: int = 448) -> np.ndarray | None:
    """
    Run our model, return binary mask for cls_id at original image resolution.
    """
    orig_w, orig_h = img_pil.size

    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img_t = tf(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)["logits"]  # (1, 51, 448, 448)
        pred_448 = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    # Binary mask for target class at 448
    bin_448 = (pred_448 == cls_id).astype(np.uint8)
    if bin_448.sum() < 200:
        return None

    # Resize back to original resolution
    bin_orig = np.array(Image.fromarray(bin_448 * 255).resize((orig_w, orig_h), Image.NEAREST)) > 0
    return bin_orig.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str,
                        default="results/ours_20260619_205153/checkpoints/best_model.pth")
    parser.add_argument("--sam-checkpoint", type=str,
                        default="/root/autodl-tmp/sam_vit_h.pth")
    parser.add_argument("--data-root", type=str, default="data/imagenet-s/ImageNetS50")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf-thresh", type=float, default=0.3,
                        help="min fraction of image that model must predict as target class")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    out_root = Path(args.output or os.path.join(args.data_root, "sam_pseudo_v2"))

    # ---- Load models ----
    print(f"Loading our model: {args.model_checkpoint}")
    our_model, ckpt = load_our_model(args.model_checkpoint, device)
    print(f"  epoch={ckpt['epoch']}, mIoU={ckpt['best_miou']:.4f}")

    print("Loading SAM ViT-H...")
    sam = load_sam_predictor(args.sam_checkpoint, device)

    # ---- Datasets ----
    ds_semi  = ImageNetSDataset(args.data_root, mode="train-semi", size=448, augment=False)
    ds_train = ImageNetSDataset(args.data_root, mode="train", size=448, augment=False)
    class_names = sorted(ds_semi.class_to_id.keys())
    print(f"{len(class_names)} classes")

    # ---- Output ----
    out_img_dir  = out_root / "train-pseudo"
    out_mask_dir = out_root / "train-pseudo-segmentation"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    for cls_name in tqdm(class_names, desc="Classes"):
        cls_id = ds_semi.class_to_id[cls_name]

        cls_img_dir  = out_img_dir / cls_name
        cls_mask_dir = out_mask_dir / cls_name
        cls_img_dir.mkdir(parents=True, exist_ok=True)
        cls_mask_dir.mkdir(parents=True, exist_ok=True)

        # Skip already-existing (don't overwrite)
        existing = set(os.listdir(cls_mask_dir)) if cls_mask_dir.exists() else set()

        semi_fnames = {os.path.basename(s["image"]) for s in ds_semi.samples
                       if s["class_name"] == cls_name}
        candidates = [s for s in ds_train.samples
                      if s["class_name"] == cls_name
                      and os.path.basename(s["image"]) not in semi_fnames]
        random.shuffle(candidates)

        kept = 0
        for cand in tqdm(candidates, desc=f"  {cls_name}", leave=False):
            if kept >= args.topk:
                break

            fname = os.path.basename(cand["image"])
            mask_name = os.path.splitext(fname)[0] + ".png"
            if mask_name in existing:
                continue

            try:
                img = Image.open(cand["image"]).convert("RGB")
            except Exception:
                continue

            # Step 1: Our model predicts a mask for this class
            model_mask = model_predict_mask(our_model, img, cls_id, device)
            if model_mask is None or model_mask.sum() < 200:
                continue

            # Confidence: fraction of image predicted as target class
            conf = model_mask.sum() / (img.size[0] * img.size[1])
            if conf < args.conf_thresh or conf > 0.85:
                continue

            # Step 2: Extract bbox from model prediction
            bbox = mask_to_bbox(model_mask)
            if bbox is None:
                continue

            # Step 3: SAM refines with bbox prompt
            img_np = np.array(img)
            sam.set_image(img_np)
            masks, scores, _ = sam.predict(
                box=bbox[None, :], multimask_output=False
            )
            if masks is None or len(masks) == 0:
                continue

            sam_mask = masks[0].astype(np.uint8)
            if sam_mask.sum() < 300 or sam_mask.sum() / sam_mask.size > 0.85:
                continue

            # Step 4: Save
            h, w = sam_mask.shape
            mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            mask_rgb[:, :, 0] = sam_mask * (cls_id % 256)
            mask_rgb[:, :, 1] = sam_mask * (cls_id // 256)
            Image.fromarray(mask_rgb).save(cls_mask_dir / mask_name)

            dst_img = cls_img_dir / fname
            if not dst_img.exists():
                train_path = os.path.join("data/imagenet-s/ImageNetS50/train", cls_name, fname)
                os.symlink(os.path.abspath(train_path), dst_img)

            kept += 1

        print(f"  {cls_name}: {kept}/{args.topk}")
        total += kept

    print(f"\nSaved {total} pseudo-labels → {out_root}")
    our_model.remove_hooks()


if __name__ == "__main__":
    main()
