# Model Weights

Do not commit large checkpoint files to GitHub.

Required models:

1. YOLOv11 detection model
   - Default: `yolo11n.pt`
   - The Ultralytics package downloads it automatically when first used.

2. SAM 2.1 segmentation model
   - Recommended for an 8 GB GPU: `sam2.1_hiera_small.pt`
   - Put the checkpoint here:

```text
weights/sam2.1_hiera_small.pt
```

If the SAM 2 checkpoint is too large for GitHub submission, submit the code and documentation in GitHub and send the checkpoint separately according to the course instruction.
