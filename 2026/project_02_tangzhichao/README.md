# ViT Semantic Segmentation on ImageNet-S

Project 02 — Two ViT-based semantic segmentation approaches on ImageNet-S.

| | Baseline | Advanced |
|---|---|---|
| **Model** | SegFormer-B0 (ADE20K pretrained) | DINOv2 ViT-B/14 + Multi-Head ADBA-Head |
| **Classes** | **11** (10 foreground + bg) | **51** (50 foreground + bg) |
| **Resolution** | 256×256 | 448×448 |
| **Params** | 3.7M | 96.4M |
| **Code** | `src/baseline/` | `src/advanced/` |
| **Checkpoint** | Not interchangeable | Not interchangeable |

> ⚠️ The two models use **different class counts** (11 vs 51). Checkpoints and configs are NOT cross-compatible.

## 1. Project Objective

Build ViT-based semantic segmentation models on ImageNet-S50:
- **Baseline**: SegFormer-B0 on a 10-class animal subset — a lightweight, reproducible reference.
- **Advanced**: DINOv2 + Multi-Head Attention Diffusion Bridge Attention (ADBA) Head on the full 50-class dataset — the core innovation.

## 2. Solution Approach

### 2.1 Advanced — DINOv2 + Multi-Head ADBA-Head

```
Input (3 × 448 × 448)
  → DINOv2 ViT-B/14 + 4 Registers
    ├─ Hook @ Layer 3  → shallow features
    ├─ Hook @ Layer 11 → self-attention matrix (B, 12, N, N)
    └─ Output          → coarse features
  → Multi-Head ADBA Head
    ├─ Coarse features → Conv → multi-head M (B, 12, P, 32)
    ├─ Attention diffusion: A @ M (per-head independent)
    ├─ Shallow bridge (Layer 3) → Conv
    ├─ Deep fusion (2-layer conv)
    └─ Progressive upsample → logits (B, 51, 448, 448)
```

Key innovations:
- **Multi-head attention diffusion**: 12 heads operate independently (no averaging), preserving per-head subspace semantics.
- **Progressive 3-stage upsampling**: PixelShuffle(2)→PixelShuffle(2)→bilinear→refine conv.
- **Boundary-aware loss**: Laplacian edge detection upweights boundary pixels 3×.
- **SAM-augmented dataset**: Model-guided SAM pseudo-labeling expands training from 500 → 1,482 images.

### 2.2 Baseline — SegFormer-B0

Hierarchical Transformer encoder + MLP decoder on a 10-class animal subset. ADE20K pretrained initialization. Weighted Cross Entropy + Dice Loss.

## 3. Instructions

### 3.1 Environment

```bash
conda create -n vit_seg python=3.10 -y
conda activate vit_seg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3.2 Dataset

Extract ImageNet-S50 to `data/imagenet-s/ImageNetS50/`. See `data/dataset_info.txt`.

### 3.3 Training (Advanced)

```bash
# Stage 1 — warmup on original 500 images
python -m src.advanced.train_v2 --mode advanced --epochs 30 --img-size 448 \
    --lr-head 2e-3 --lr-backbone 3e-4 --batch-size 20 --num-workers 2 \
    --bg-weight 1.0 --edge-beta 0.0

# Stage 2 — resume on SAM-augmented 1,482 images
python -m src.advanced.train_v2 --mode advanced --epochs 60 --img-size 448 \
    --lr-head 5e-4 --lr-backbone 5e-5 --batch-size 20 --num-workers 2 \
    --bg-weight 1.0 --edge-beta 2.5 \
    --resume results/advanced_<ts>/checkpoints/best_model.pth
```

### 3.4 Training (Baseline)

```bash
python -m src.baseline.train --config configs/imagenet_s_animals_10cls_trainval_holdout_ade.yaml
```

### 3.5 Sample Inference (Advanced)

```bash
python -m src.main --checkpoint <path/to/best_model.pth> \
    --input data/test_examples/advanced --output results/demo_advanced
```

Output per image: `pred_mask.png`, `pred_color.png`, `overlay.png`, `pred_labels.txt`.

### 3.6 Sample Inference (Baseline)

```bash
python -m src.main --model_dir outputs/.../best_model \
    --input data/test_examples/baseline --output results/demo_baseline
```

### 3.7 Evaluation & Visualization (Advanced)

```bash
# Full suite: curves + predictions + failure cases + attention heatmaps
python -m src.advanced.visualize_v2 --mode all \
    --checkpoint <best_model.pth> --log <training_log.csv>
```

### 3.8 SAM Pseudo-Labeling

```bash
# Model-guided SAM (recommended)
python src/advanced/sam_label_v2.py --topk 20 \
    --model-checkpoint <best_model.pth>
```

## 4. Results

### 4.1 Advanced — DINOv2 + Multi-Head ADBA (50 classes)

| Metric | Value |
|--------|-------|
| mIoU (val) | 0.6947 |
| Pixel Accuracy | ~0.89 |
| Training samples | 1,482 |
| Parameters | 96.4M |

### 4.2 Baseline — SegFormer-B0 (10 classes)

| Metric | Value |
|--------|-------|
| mIoU (test) | 0.7898 |
| Parameters | 3.7M |
| Training time | 580s (CPU) |

## 5. Repository Structure

```
project_02_tangzhichao/
├── README.md
├── requirements.txt
├── references.md
├── configs/
│   ├── imagenet_s_animals_10cls_trainval_holdout_ade.yaml  # baseline
│   └── imagenet_s50_dinov2_adba.yaml                       # advanced
├── data/
│   ├── dataset_info.txt
│   └── test_examples/
│       ├── baseline/          # 4 images for SegFormer demo
│       └── advanced/          # 10 images for DINOv2 demo
├── demos/
│   └── inference_demo.ipynb
├── outputs/                   # Training outputs
├── results/                   # Visualizations & logs
├── scripts/
└── src/
    ├── main.py                # Unified inference entry
    ├── baseline/              # SegFormer-B0 10-class
    │   ├── model.py
    │   ├── own_segformer.py
    │   ├── dataset.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── visualize.py
    │   └── ...
    └── advanced/          # DINOv2 + ADBA-Head 50-class
        ├── dinov2_seg.py
        ├── dataset_v2.py
        ├── train_v2.py
        ├── metrics_v2.py
        ├── visualize_v2.py
        ├── sam_label.py
        ├── sam_label_v2.py
        └── sam_overlap.py
```

## 6. Conclusion

The DINOv2 + Multi-Head ADBA-Head demonstrates attention diffusion from self-supervised ViT features for semantic segmentation. The multi-head design preserves per-head semantics, and SAM-augmented pseudo-labeling expands the training set 3×. The SegFormer-B0 baseline provides a lightweight reference on a 10-class subset.

Future work: LoRA backbone fine-tuning, iterative self-training, larger DINOv2 variants.

## 7. References

See `references.md`.
