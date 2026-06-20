# ViT Semantic Segmentation on ImageNet-S50

Project 02 — DINOv2 + Multi-Head ADBA-Head for full 50-class ImageNet-S semantic segmentation.

This repository merges a **SegFormer-B0 baseline** (10-class, CPU) with a **DINOv2-based full model** (50-class, GPU, RTX 4090) and includes SAM-augmented pseudo-labeling, boundary-aware training, and progressive multi-stage upsampling.

## 1. Project Objective

Build a ViT-based semantic segmentation model on ImageNet-S50 (50 foreground classes + background = 51 labels). The core innovation is a **Multi-Head Attention Diffusion Bridge Attention (ADBA) Head** that exploits DINOv2's self-attention matrices for precise object boundary recovery.

## 2. Solution Approach

### 2.1 Architecture

```
Input (3 × 448 × 448)
  │
  ▼
┌─ DINOv2 ViT-B/14 + 4 Registers ────────────┐
│  Hook @ Layer 3  → shallow features         │
│  Hook @ Layer 11 → self-attention matrix    │
│  Output          → coarse features          │
└─────────────────────────────────────────────┘
  │                    │              │
  ▼                    ▼              ▼
┌─ Multi-Head ADBA Head ─────────────────────┐
│  1. Coarse features → Conv → multi-head M  │
│  2. Attention A (B,12,P,P) × M (B,12,P,32) │
│     → per-head independent diffusion       │
│  3. Shallow bridge (Layer 3) → Conv        │
│  4. Deep fusion (2-layer conv, 504ch)      │
│  5. Progressive upsample:                  │
│     2× PixelShuffle → 2× PixelShuffle      │
│     → 1.75× bilinear → refine conv         │
│     → logits (B, 51, 448, 448)             │
└─────────────────────────────────────────────┘
```

Key design choices:
- **Multi-head attention diffusion**: each of 12 heads operates independently (no averaging), preserving per-head subspace semantics.
- **Progressive upsampling**: 3-stage learned upsampling (not single 14× bilinear) for sharper boundaries.
- **Boundary-aware loss**: Laplacian edge detection on GT masks upweights boundary pixels 3×, background supervised at full weight.

### 2.2 Dataset

**ImageNet-S50** — 50 ImageNet categories with pixel-level segmentation masks.

| Split | Images | Masks | Usage |
|-------|--------|-------|-------|
| train | 64,431 | none | unsupervised / pseudo-label source |
| train-semi | 500 | yes (10/class) | supervised training |
| train-semi-plus | 1,482 | yes (~29/class) | original + SAM-augmented |
| validation | 752 | yes | evaluation |
| test | 1,682 | none | online benchmark |

**SAM Pseudo-Labeling**: Our trained model predicts → bbox → SAM refines → high-quality pseudo-masks added to training set.

### 2.3 Training Strategy

Two-stage training:

| Stage | Dataset | Images | Loss | LR | Epochs |
|-------|---------|--------|------|-----|--------|
| 1 (warmup) | train-semi | 500 | CrossEntropy | head=2e-3, bb=3e-4 | 30 |
| 2 (refine) | train-semi-plus | 1,482 | BoundaryAware (edge_beta=2.5) | head=5e-4, bb=5e-5 | 30 |

Optimizer: AdamW with differential LR (head/backbone) + selective weight decay (no decay on norm/bias).

### 2.4 Baseline (SegFormer-B0)

For comparison, the repository also includes a **SegFormer-B0 baseline** trained on a 10-class animal subset (256×256, CPU). See configs folder for details.

## 3. Instructions

### 3.1 Environment

```bash
conda create -n vit_seg python=3.10 -y
conda activate vit_seg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3.2 Dataset

Extract ImageNet-S50 to `data/imagenet-s/ImageNetS50/`. Expected structure:

```
data/imagenet-s/ImageNetS50/
├── train/          # full training images (symlinks OK)
├── train-semi/
├── train-semi-segmentation/
├── validation/
├── validation-segmentation/
└── test/
```

Download DINOv2 weights (auto-downloaded by timm on first run, or manually place at ~/.cache/).

### 3.3 Training

```bash
# Stage 1 — warmup on original data
python -m src.train_v2 --mode ours --epochs 30 --img-size 448 \
    --lr-head 2e-3 --lr-backbone 3e-4 --batch-size 20 --num-workers 2 \
    --bg-weight 1.0 --edge-beta 0.0

# Stage 2 — continued on SAM-augmented data with boundary loss
python -m src.train_v2 --mode ours --epochs 60 --img-size 448 \
    --lr-head 5e-4 --lr-backbone 5e-5 --batch-size 20 --num-workers 2 \
    --bg-weight 1.0 --edge-beta 2.5 \
    --resume results/ours_<timestamp>/checkpoints/best_model.pth
```

### 3.4 Evaluation

```bash
python -m src.visualize_v2 --checkpoint results/ours_<timestamp>/checkpoints/best_model.pth
```

### 3.5 Sample Inference

```bash
python -m src.main --model_dir results/ours_<timestamp>/checkpoints \
    --input data/test_examples --output results/demo
```

### 3.6 Demo Notebook

See `demos/inference_demo.ipynb` for interactive visualization.

### 3.7 SAM Pseudo-Labeling

```bash
# Option A: Grounding DINO + SAM (text-prompted)
python src/sam_label.py --topk 20

# Option B: Model-guided SAM (our model → bbox → SAM)
python src/sam_label_v2.py --topk 20 \
    --model-checkpoint results/ours_<timestamp>/checkpoints/best_model.pth
```

## 4. Results

### 4.1 DINOv2 + ADBA-Head (50 classes, 448×448)

Latest run (epoch 99):

| Metric | Value |
|--------|-------|
| mIoU | 0.6947 |
| Pixel Accuracy | ~0.89 |
| Training samples | 1,482 |
| Parameters | 96.4M |
| Trainable params | 32.1M |

### 4.2 Baseline (SegFormer-B0, 10 classes, 256×256)

| Metric | Value |
|--------|-------|
| mIoU | 0.7898 |
| Parameters | 3.7M |
| Training time | 580s (CPU) |

### 4.3 Ablation Studies

| Experiment | mIoU | Notes |
|------------|------|-------|
| DINOv2 + bilinear head | 0.737 (5ep) | Simple baseline |
| DINOv2 + ADBA (original) | 0.8085 (23ep) | Old architecture, 500 images |
| DINOv2 + MultiHead ADBA | 0.6947 (99ep) | Current, converges slower but more capacity |
| SegFormer-B0 (10cls) | 0.7898 | Small model, subset of classes |

### 4.4 Visualizations

Generated figures in `results/`:
- `top_iou_8_final.png` — Top-8 IoU validation samples (original / GT / prediction)
- `pred_overlay_8samples.png` — Multi-class prediction overlay
- `grounded_sam_quality.png` — SAM pseudo-label quality check
- `training_metrics.csv` — Full training log (mIoU, pAcc, loss)

## 5. Repository Structure

```
project_02_tangzhichao/
├── README.md
├── requirements.txt
├── references.md
├── configs/
│   ├── imagenet_s_animals_10cls_trainval_holdout_ade.yaml  # SegFormer baseline
│   └── imagenet_s50_dinov2_adba.yaml                       # DINOv2 full model
├── data/
│   ├── dataset_info.txt
│   └── test_examples/
├── demos/
│   └── inference_demo.ipynb
├── models/
│   └── segformer-b0-ade/              # ADE20K pretrained weights
├── outputs/                           # Training outputs per experiment
├── results/                           # Visualizations, tables, CSVs
│   ├── figures/
│   ├── tables/
│   └── run1_fresh.csv
├── scripts/                           # Shell scripts for train/eval/infer
└── src/
    ├── main.py                        # Sample inference entry
    ├── model.py                       # SegFormer-B0 model (baseline)
    ├── own_segformer.py               # SegFormer implementation
    ├── dinov2_seg.py                  # DINOv2 + MultiHead ADBA-Head (ours)
    ├── dataset.py                     # Dataset (10-class, baseline)
    ├── dataset_v2.py                  # Dataset (50-class + SAM pseudo-labels)
    ├── train.py                       # Training script (baseline)
    ├── train_v2.py                    # Training script (DINOv2, ours)
    ├── evaluate.py                    # Evaluation
    ├── infer.py                       # Inference utilities
    ├── visualize.py                   # Visualization (baseline)
    ├── visualize_v2.py                # Visualization (DINOv2 model)
    ├── metrics.py                     # Metrics (baseline)
    ├── metrics_v2.py                  # Metrics (GPU-accelerated + BoundaryAwareLoss)
    ├── sam_label.py                   # Grounding DINO + SAM labeling
    ├── sam_label_v2.py                # Model-guided SAM labeling
    ├── sam_overlap.py                 # Model-SAM overlap computation
    ├── transforms.py                  # Transforms
    ├── config.py                      # Config loader
    └── utils.py                       # Utilities
```

## 6. Conclusion

The DINOv2 + Multi-Head ADBA-Head model demonstrates that attention diffusion from self-supervised ViT features can drive semantic segmentation. The multi-head design preserves per-head subspace semantics, and progressive upsampling + boundary-aware loss improve edge quality. SAM-augmented pseudo-labeling expands the effective training set from 500 to ~1,500 images.

Current limitations:
- Convergence is slower than the simpler baseline (mIoU plateaus ~0.7 vs old architecture's 0.81)
- Some classes with thin/elongated objects (bobsled, hook) remain challenging
- SAM pseudo-label quality depends on the guiding model's accuracy

Future work: LoRA fine-tuning of the backbone, larger DINOv2 variants (ViT-L), and iterative self-training to bootstrap SAM labeling quality.

## 7. References

- DINOv2: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", arXiv:2304.07193
- ImageNet-S: Gao et al., "Large-scale Unsupervised Semantic Segmentation", TPAMI 2022
- SAM: Kirillov et al., "Segment Anything", ICCV 2023
- Grounding DINO: Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training", ECCV 2024
- SegFormer: Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers", NeurIPS 2021

See `references.md` for detailed attribution.
