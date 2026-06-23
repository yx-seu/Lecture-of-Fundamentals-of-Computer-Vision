"""
Compute overlap (IoU) between trained model predictions and SAM pseudo-labels.

Usage (run when GPU is free):
    conda activate vit_seg
    python src/sam_overlap.py

Output:
    results/sam_overlap.csv      — per-sample IoU
    results/sam_overlap_summary.txt  — per-class + overall stats
"""

import os, sys, numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ---- Constants matching the checkpoint ----
EMBED_DIM = 768
NUM_HEADS = 12
NUM_PREFIX = 5  # CLS + 4 registers
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Exact old architecture (matching checkpoint ours_20260618_222210)
# ---------------------------------------------------------------------------

class OldMultiHeadADBAHead(nn.Module):
    """
    Matches the checkpoint architecture:
    - coarse_conv: Conv2d(768→384, groups=12) → BN → ReLU
    - shallow_conv: Conv2d(768→120, groups=12) → BN → ReLU
    - fusion: 2× Conv (504→504, 504→256) with BN+ReLU
    - cls_proj: Conv2d(256→51)
    - refine: Conv2d(51→51) → BN → ReLU
    - Upsample: bilinear 14×
    """
    def __init__(self, num_classes=51):
        super().__init__()
        coarse_dim, shallow_dim = 384, 120
        fusion_dim = coarse_dim + shallow_dim
        self.num_heads = NUM_HEADS
        self.head_dim = coarse_dim // NUM_HEADS

        self.coarse_conv = nn.Sequential(
            nn.Conv2d(EMBED_DIM, coarse_dim, 3, padding=1, groups=NUM_HEADS),
            nn.BatchNorm2d(coarse_dim), nn.ReLU(True))
        self.shallow_conv = nn.Sequential(
            nn.Conv2d(EMBED_DIM, shallow_dim, 1, groups=NUM_HEADS),
            nn.BatchNorm2d(shallow_dim), nn.ReLU(True))
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1),
            nn.BatchNorm2d(fusion_dim), nn.ReLU(True),
            nn.Conv2d(fusion_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(True))
        self.cls_proj = nn.Conv2d(256, num_classes, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, 3, padding=1),
            nn.BatchNorm2d(num_classes), nn.ReLU(True))

    def forward(self, coarse, attn, shallow, patch_h, patch_w):
        B, P = coarse.shape[0], patch_h * patch_w

        # Attention: remove prefix, keep heads
        A = attn[:, :, NUM_PREFIX:, NUM_PREFIX:]  # (B, 12, P, P)

        # Coarse features → spatial → conv
        xc = coarse[:, NUM_PREFIX:, :].transpose(1, 2).reshape(B, EMBED_DIM, patch_h, patch_w)
        M = self.coarse_conv(xc)  # (B, 384, Hp, Wp)

        # Multi-head attention diffusion
        M_flat = M.reshape(B, 384, P)
        M_heads = M_flat.reshape(B, NUM_HEADS, self.head_dim, P).transpose(2, 3)  # (B, 12, P, 32)
        M_diff = (A @ M_heads).transpose(2, 3).reshape(B, 384, patch_h, patch_w)

        # Shallow bridge
        xs = shallow[:, NUM_PREFIX:, :].transpose(1, 2).reshape(B, EMBED_DIM, patch_h, patch_w)
        shallow_map = self.shallow_conv(xs)

        # Fusion + upsample
        fused = self.fusion(torch.cat([M_diff, shallow_map], dim=1))
        out = self.cls_proj(fused)
        out = F.interpolate(out, scale_factor=14, mode='bilinear', align_corners=False)
        return self.refine(out)


class AttentionCapture:
    """Hook on attn_drop to capture attention matrix."""
    def __init__(self, attn_module):
        self.attn_matrix = None
        self._fused = getattr(attn_module, 'fused_attn', True)
        attn_module.fused_attn = False
        self.hook = attn_module.attn_drop.register_forward_hook(self._fn)

    def _fn(self, m, args, output):
        self.attn_matrix = args[0]

    def remove(self):
        self.hook.remove()


class FeatureCapture:
    """Hook on block output."""
    def __init__(self, block):
        self.features = None
        self.hook = block.register_forward_hook(self._fn)

    def _fn(self, m, args, output):
        self.features = output

    def remove(self):
        self.hook.remove()


class OldSegModel(nn.Module):
    """Exact model matching checkpoint ours_20260618_222210."""
    def __init__(self, num_classes=51):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_base_patch14_reg4_dinov2', pretrained=False,
            img_size=448, num_classes=0)
        self.attn_capture = AttentionCapture(self.backbone.blocks[10].attn)
        self.feat_capture = FeatureCapture(self.backbone.blocks[2])
        self.seg_head = OldMultiHeadADBAHead(num_classes)
        self.patch_h = self.patch_w = 32

    def forward(self, x):
        coarse = self.backbone.forward_features(x)
        attn = self.attn_capture.attn_matrix
        shallow = self.feat_capture.features
        logits = self.seg_head(coarse, attn, shallow, self.patch_h, self.patch_w)
        return {'logits': logits}

    def remove_hooks(self):
        self.attn_capture.remove()
        self.feat_capture.remove()


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def compute_sample_iou(pred_mask, sam_mask):
    """
    IoU between model prediction (foreground) and SAM mask.
    Returns: iou, precision, recall
    """
    pred_fg = pred_mask > 0
    sam_fg = sam_mask > 0
    inter = (pred_fg & sam_fg).sum()
    union = (pred_fg | sam_fg).sum()
    iou = inter / union if union > 0 else 0.0
    precision = inter / pred_fg.sum() if pred_fg.sum() > 0 else 0.0
    recall = inter / sam_fg.sum() if sam_fg.sum() > 0 else 0.0
    return iou, precision, recall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='results/ours_20260618_222210/checkpoints/best_model.pth')
    parser.add_argument('--data-root', type=str, default='data/imagenet-s/ImageNetS50')
    parser.add_argument('--output', type=str, default='results/sam_overlap')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    print(f"  epoch={ckpt['epoch']}, mIoU={ckpt['best_miou']:.4f}")

    # ---- Build model ----
    model = OldSegModel(num_classes=51).to(device).eval()
    model.load_state_dict(ckpt['model_state_dict'])
    print("  Model loaded OK")

    # ---- Transforms ----
    tf = transforms.Compose([
        transforms.Resize(448), transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # ---- Scan SAM pseudo-labels ----
    img_dir = os.path.join(args.data_root, 'sam_pseudo', 'train-pseudo')
    mask_dir = os.path.join(args.data_root, 'sam_pseudo', 'train-pseudo-segmentation')
    if not os.path.isdir(img_dir):
        print(f"ERROR: {img_dir} not found. Run sam_label.py first.")
        return

    samples = []
    for cls in sorted(os.listdir(img_dir)):
        cls_img = os.path.join(img_dir, cls)
        cls_mask = os.path.join(mask_dir, cls)
        if not os.path.isdir(cls_mask):
            continue
        for f in os.listdir(cls_mask):
            name = os.path.splitext(f)[0]
            for ext in ['.JPEG', '.jpg', '.png']:
                ip = os.path.join(cls_img, name + ext)
                if os.path.exists(ip): break
            else:
                continue
            samples.append({'class': cls, 'image': ip,
                          'mask': os.path.join(cls_mask, f)})

    print(f"SAM samples: {len(samples)}")

    # ---- Compute overlap ----
    results = []
    per_class = defaultdict(list)

    for s in tqdm(samples, desc='Computing overlap'):
        try:
            img = Image.open(s['image']).convert('RGB')
        except Exception:
            continue

        # Model inference
        img_t = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_t)['logits'].argmax(dim=1).squeeze(0).cpu().numpy()

        # SAM mask → resize to 448
        sam = np.array(Image.open(s['mask']))
        sam_cls = sam[:, :, 1].astype(np.int32) * 256 + sam[:, :, 0].astype(np.int32)
        sam_bin = (sam_cls > 0).astype(np.uint8)
        sam_bin_448 = np.array(Image.fromarray(sam_bin * 255)
                               .resize((448, 448), Image.NEAREST)) > 0

        iou, prec, rec = compute_sample_iou(pred, sam_bin_448)
        results.append({**s, 'iou': iou, 'precision': prec, 'recall': rec})
        per_class[s['class']].append(iou)

    # ---- Summary ----
    all_iou = np.array([r['iou'] for r in results])
    print(f"\n{'='*50}")
    print(f"Overall (n={len(results)})")
    print(f"  Mean IoU:       {all_iou.mean():.3f}")
    print(f"  Median IoU:     {np.median(all_iou):.3f}")
    for th, label in [(0.5, 'Good (>0.5)'), (0.3, 'OK (0.3-0.5)'), (0.1, 'Poor (0.1-0.3)'), (0.0, 'Bad (<0.1)')]:
        n = (all_iou >= th).sum() if th > 0 else (all_iou < 0.1).sum()
        if th > 0:
            n = n - (all_iou >= (th + 0.2 if th == 0.3 else 0.5)).sum()
        if th == 0.5:
            n = (all_iou > 0.5).sum()
        if th == 0.3:
            n = ((all_iou > 0.3) & (all_iou <= 0.5)).sum()
        if th == 0.1:
            n = ((all_iou >= 0.1) & (all_iou <= 0.3)).sum()
        if th == 0.0:
            n = (all_iou < 0.1).sum()
        print(f"  {label:16s}: {n:4d} ({n/len(results)*100:.0f}%)")

    print(f"\nPer-class mean IoU:")
    for cls in sorted(per_class):
        vals = per_class[cls]
        print(f"  {cls}: {np.mean(vals):.3f}  (n={len(vals)}, min={np.min(vals):.3f}, max={np.max(vals):.3f})")

    # ---- Save ----
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output + '.csv', 'w') as f:
        f.write('class,image,iou,precision,recall\n')
        for r in results:
            f.write(f"{r['class']},{os.path.basename(r['image'])},{r['iou']:.4f},{r['precision']:.4f},{r['recall']:.4f}\n")
    print(f"\nSaved: {args.output}.csv")

    model.remove_hooks()


if __name__ == '__main__':
    main()
