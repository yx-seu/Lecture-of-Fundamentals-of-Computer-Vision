# References & Citations

This document acknowledges all third-party code, models, and research papers
used in this project.

---

## Image Processing

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

- **Original Paper**: Zuiderveld, K. (1994). "Contrast Limited Adaptive Histogram Equalization." In *Graphics Gems IV*, Academic Press, pp. 474–485.
- **Implementation**: OpenCV `cv2.createCLAHE()`
- **License**: OpenCV is released under the Apache 2.0 License.

### DnCNN (Denoising Convolutional Neural Network)

- **Original Paper**: Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising." *IEEE Transactions on Image Processing*, 26(7), 3142–3155.
- **Official Repository**: https://github.com/cszn/DnCNN
- **Code Location**: `DnCNN-master/` (included in the parent repository)
- The DnCNN model architecture is defined in `src/dncnn.py` (adapted from the official PyTorch training code at `DnCNN-master/TrainingCodes/dncnn_pytorch/main_train.py`).
- Pre-trained weights were converted from the official MatConvNet `.mat` checkpoint (sigma=25) using `src/convert_mat_to_pth.py`.
- **License**: The DnCNN repository is publicly available for research purposes. See the original repository for license details.

### SwinIR (Swin Transformer for Image Restoration)

- **Original Paper**: Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., & Timofte, R. (2021). "SwinIR: Image Restoration Using Swin Transformer." In *Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)*, pp. 1833–1844.
- **Official Repository**: https://github.com/JingyunLiang/SwinIR
- **Code Location**: `SwinIR-main/` (included in the parent repository)
- The SwinIR model architecture (`models/network_swinir.py`) and inference logic were adapted from the official implementation.
- Pre-trained model: `004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth` (grayscale denoising, noise level 15)
- **License**: The SwinIR repository is released under the Apache 2.0 License. See `SwinIR-main/LICENSE`.

---

## Object Detection

### YOLOv11 (Ultralytics)

- **Framework**: Ultralytics YOLOv11
- **Official Repository**: https://github.com/ultralytics/ultralytics
- **Documentation**: https://docs.ultralytics.com/
- **License**: Ultralytics YOLO is released under the AGPL-3.0 License.
- YOLOv11 was used for training and inference of the 5-class TEM object detector.
- Training commands and configurations are recorded in the `train7/args.yaml`, `train8/args.yaml`, and `train9/args.yaml` files.

### P2 Detection Head

- The YOLOv11-P2 variant incorporates a larger detection head architecture adapted from:
  - Glenn Jocher, Ayush Chaurasia, & Jing Qiu. "YOLO by Ultralytics" (2023).
  - Related concept: "Extended detection head for improved small object detection" as implemented in the Ultralytics framework.

---

## Training Methodology

### YOLOv11 Training Configuration

The following hyperparameters were used for all training runs:
- **Epochs**: 100
- **Batch size**: 2
- **Image size**: 640×640
- **Optimizer**: AdamW (auto)
- **Learning rate**: 0.01 (initial), cosine scheduling to 0.0001
- **Momentum**: 0.937
- **Weight decay**: 0.0005
- **Data augmentation**: Mosaic, random affine (scale ±50%, translate ±10%), HSV jitter, horizontal flip, random erasing (0.4)

---

## Development Tools

- **Python**: Primary programming language
- **PyTorch**: Deep learning framework (https://pytorch.org/)
- **OpenCV**: Image processing library (https://opencv.org/)
- **NumPy**: Numerical computing (https://numpy.org/)
- **h5py**: MATLAB .mat file reading for model conversion (https://www.h5py.org/)

---

## Academic Integrity Statement

All third-party code and models used in this project are properly cited above.
The original authors retain all rights to their respective works.

Contributions of this project:
1. Integration of CLAHE + DnCNN + SwinIR preprocessing pipeline for TEM images
2. Comparative analysis of YOLOv11 detection performance across preprocessing stages
3. Custom dataset labeling and annotation for TEM integrated circuit imagery
4. Training configuration optimization for the specific detection task

---
*Last updated: 2026-06-14*
