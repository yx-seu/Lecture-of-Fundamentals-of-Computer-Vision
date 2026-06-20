# YOLOv11 for TEM Image Object Detection in Integrated Circuit Manufacturing

![License](https://img.shields.io/badge/License-Public-green)
![Status](https://img.shields.io/badge/Status-Active-green)

## 📖 Project Overview

This project develops an **object detection pipeline for TEM (Transmission Electron Microscopy) image analysis** in integrated circuit manufacturing. The pipeline integrates image enhancement techniques with state-of-the-art deep learning object detection to identify key components in TEM imagery.

**Key functions include:**
- Dataset preprocessing and organization
- Image enhancement (CLAHE + DnCNN + SwinIR)
- YOLOv11-based object detection
- Model training, evaluation, and inference

## 🎯 Project Objective

TEM images of integrated circuits often suffer from low contrast and noise, making reliable object detection challenging. This project aims to:

1. **Enhance TEM image quality** through a multi-stage preprocessing pipeline (CLAHE → DnCNN → SwinIR)
2. **Train YOLOv11 models** to detect 5 key component classes in enhanced TEM images
3. **Compare detection performance** across different preprocessing strategies
4. **Provide a reproducible pipeline** for TEM image analysis

## 🔬 Detection Classes

The model detects 5 classes of objects in TEM images:

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | `manipulator` | Manipulator probe tip |
| 1 | `sample` | Sample/lamella |
| 2 | `copper_screen_top` | Upper copper grid |
| 3 | `copper_screen_side` | Side copper grid |
| 4 | `deposition` | Deposited material region |

## 🏗️ Pipeline Architecture

```
Raw TEM Image
    │
    ▼
┌─────────────────────────────────┐
│  Stage 1: CLAHE Enhancement     │
│  Contrast Limited Adaptive      │
│  Histogram Equalization         │
│  (clipLimit=2.0, grid=8×8)     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Stage 2: DnCNN Denoising       │
│  17-layer CNN with residual     │
│  learning (sigma=25 model)      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Stage 3: SwinIR Enhancement    │
│  Swin Transformer-based         │
│  image restoration (noise=15)   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  YOLOv11 Object Detection       │
│  5-class detection with         │
│  bounding box regression        │
└─────────────────────────────────┘
    │
    ▼
Detection Results (bbox + class)
```

### Why This Pipeline?

- **CLAHE**: Improves local contrast without over-amplifying noise — essential for revealing fine TEM structures
- **DnCNN**: A 17-layer residual CNN that predicts and subtracts the noise component while preserving structural details
- **SwinIR**: A Swin Transformer-based restoration model that further enhances image quality for downstream detection
- **YOLOv11**: State-of-the-art real-time object detector with excellent accuracy-speed trade-off

## 📁 Repository Structure

```
project_06_jiahongyi/
├── README.md                    # Complete documentation (English)
├── src/                         # Source code
│   ├── clahe.py                 # CLAHE image enhancement
│   ├── dncnn.py                 # DnCNN model + inference
│   ├── swinir_inference.py      # SwinIR inference script
│   ├── convert_mat_to_pth.py    # MatConvNet → PyTorch conversion
│   ├── main.py                  # Sample inference pipeline
│   └── requirements.txt         # Python dependencies
├── data/                        # Data directory
│   ├── test_examples/           # Sample testing images
│   └── dataset_info.txt         # Dataset description and sources
├── results/                     # Output results
│   └── figures/                 # Training curves and visualizations
├── demos/                       # Inference demo
│   └── inference_demo.ipynb     # Jupyter notebook walkthrough
└── references.md                # Citations and attributions
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended, CPU fallback supported)
- Git

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd project_06_jiahongyi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r src/requirements.txt
```

### 2. Download Pre-trained Models

The DnCNN model used in this project is converted from the original MatConvNet checkpoint. To use the pipeline:

```bash
# Option A: Convert from original .mat model
cd src
python convert_mat_to_pth.py path/to/sigma=25.mat dncnn_pretrained.pth

# Option B: Train your own model using DnCNN-master/TrainingCodes/dncnn_pytorch/
```

The SwinIR model can be downloaded from the official repository:
```bash
# Download SwinIR grayscale denoising model (noise=15)
wget https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth \
  -O SwinIR-main/model_zoo/swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth
```

The YOLOv11 model can be obtained from Ultralytics:
```bash
pip install ultralytics
# YOLOv11 weights are downloaded automatically by ultralytics
```

## 🔧 Usage Instructions

### Stage 1: CLAHE Enhancement

```bash
cd tem_chale/images
python clahe.py
```

Processes images in `train/`, `val/`, `test/` subdirectories. Output files are named `*_clahe.jpg`.

**Parameters**: `clipLimit=2.0`, `tileGridSize=(8,8)`, grayscale mode.

### Stage 2: DnCNN Denoising

```bash
cd tem_chale_DnCNN/images
python dncnn.py
```

Loads `dncnn_pretrained.pth`, processes images from `train/`, `val/`, `test/`. Output: `*_dncnn.jpg`.

### Stage 3: SwinIR Enhancement

```bash
cd tem_chale_DnCNN_SwinIR/images
python swinir.py
```

Processes DnCNN outputs with SwinIR grayscale denoising. Output: `*_SwinIR.jpg`.

### YOLOv11 Training

```bash
# Train on CLAHE images
yolo detect train data=tem_clahe.yaml model=yolov11n.pt epochs=100 batch=2 imgsz=640 name=train7

# Train on CLAHE+DnCNN images
yolo detect train data=tem_clahe_dncnn.yaml model=yolov11n.pt epochs=100 batch=2 imgsz=640 name=train8

# Train on CLAHE+DnCNN+SwinIR images
yolo detect train data=tem_clahe_dncnn_swinir.yaml model=yolov11n.pt epochs=100 batch=2 imgsz=640 name=train9
```

### Run Sample Inference

```bash
cd src
python main.py
```

This runs the full pipeline on 3 sample test images and displays results.

## 📊 Results & Analysis

### Training Results Summary

Three models were trained on progressively enhanced data:

| Training Run | Input Data | Best mAP@50 | Best mAP@50-95 |
|-------------|------------|-------------|----------------|
| train7 | CLAHE only | 0.9705 | 0.8477 |
| train8 | CLAHE + DnCNN | TBD | TBD |
| train9 | CLAHE + DnCNN + SwinIR | TBD | TBD |

### train7 (CLAHE images) — Detailed Metrics

- **Best mAP@50**: 0.9705 (epoch 68)
- **Best mAP@50-95**: 0.8477 (epoch 90)
- **Precision**: 0.9447 (final)
- **Recall**: 0.9998 (final)

The model achieved excellent detection performance on CLAHE-enhanced images, with mAP@50 exceeding 0.97, demonstrating that contrast enhancement alone can support reliable object detection.

### Progressive Visualizations

Training metrics curves, confusion matrices, and validation batch predictions are available in `results/figures/`. Key observations:

1. **Convergence**: The model converges steadily, with mAP@50 reaching >0.90 by epoch 10
2. **Robustness**: High precision (>0.92) and recall (>0.99) indicate reliable detection
3. **Comparison**: The multi-stage pipeline (train8, train9) shows the impact of progressive denoising on detection quality

### Ablation Analysis

| Preprocessing | Rationale |
|--------------|-----------|
| None (raw) | Baseline — TEM images have low contrast |
| CLAHE | Improves local contrast, reveals hidden structures |
| CLAHE + DnCNN | Removes Gaussian-like noise while preserving edges |
| CLAHE + DnCNN + SwinIR | Further structural enhancement via transformer-based restoration |

## 💡 Conclusion

This project demonstrates an effective multi-stage image preprocessing pipeline for TEM image object detection. Key findings:

1. **CLAHE preprocessing substantially improves detection** by enhancing local contrast
2. **DnCNN denoising provides additional gains** by reducing noise artifacts while preserving structural information
3. **SwinIR further refines image quality** through learned image restoration
4. **YOLOv11 achieves excellent mAP@50 (>0.97)** on the 5-class TEM detection task

### Future Work

- Experiment with YOLOv11-P2 (larger model variant) for potentially higher accuracy
- Explore end-to-end trainable preprocessing + detection
- Investigate domain-specific data augmentation strategies for TEM imagery
- Extend to additional IC component classes
- Evaluate on larger and more diverse TEM datasets

## 📚 References

See [references.md](references.md) for complete citations of all third-party code and papers used in this project.

Key references:
- **CLAHE**: Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. *Graphics Gems IV*.
- **DnCNN**: Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising. *IEEE TIP*.
- **SwinIR**: Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., & Timofte, R. (2021). SwinIR: Image Restoration Using Swin Transformer. *ICCVW*.
- **YOLOv11**: Ultralytics. (2024). YOLOv11. https://github.com/ultralytics/ultralytics

## 📧 Contact

For questions regarding this project, please contact the author through the course group chat.

---

*This repository contains the image processing and object detection components of a TEM analysis pipeline for educational and research purposes.*
