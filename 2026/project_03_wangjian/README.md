# SAM 2 Automatic Mask Generation for Multi-Class Dataset Annotation

## Project Objective

This project implements an automatic annotation pipeline for five common object categories: **cup**, **bottle**, **bowl**, **book**, and **cell phone**. The goal is to generate instance-level segmentation labels with less manual pixel annotation work.

The pipeline first detects target objects with YOLOv11, then uses each detection box as a prompt for SAM 2. The final output is a COCO-style annotation file with categories, bounding boxes, segmentation polygons, confidence scores, and visualization images.

## Solution Approach

SAM 2 produces high-quality masks from prompts, but the masks themselves do not contain category names. YOLOv11 is used to provide semantic labels and bounding boxes:

1. **YOLOv11 detection**: locate objects and predict the class label among the five target categories.
2. **SAM 2 box-prompt segmentation**: use each YOLO bounding box as a prompt and generate a pixel-level instance mask.
3. **Annotation export**: save bounding boxes, classes, confidence scores, and segmentation polygons in COCO-style JSON.
4. **Visualization**: draw class labels, boxes, and translucent masks on the original images.
5. **Evaluation**: compare automatic masks with COCO-format ground truth using IoU, Dice/F1, Precision, and Recall.

This design separates recognition and segmentation: YOLOv11 predicts object categories and prompts, while SAM 2 refines the object boundary.

## Repository Structure

```text
project_03_wangjian/
  README.md
  src/
    main.py
    utils.py
    requirements.txt
  data/
    test_examples/
    manual_labelme/
    dataset_info.txt
  results/
    annotations/
    figures/
    tables/
  demos/
    demo_inference.ipynb
  weights/
    README.md
    references.md
```

## Installation

Python 3.10 or newer is recommended. A CUDA-enabled GPU is recommended for running SAM 2 efficiently.

On Windows, the following script creates a virtual environment and installs the packages used in the experiment:

```powershell
.\setup_windows_core.ps1
```

After installation, activate the environment and check the configuration:

```powershell
C:\cv03_env\Scripts\Activate.ps1
python src\check_env.py
```

For a manual setup, use:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\activate
pip install -r src/requirements.txt
```

Check GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

The SAM 2.1 small checkpoint should be placed at:

```text
weights/sam2.1_hiera_small.pt
```

The submitted experiment was run with YOLOv11n and `sam2.1_hiera_small.pt`.

## Running Instructions

Run inference on the provided examples:

```bash
python src/main.py --input data/test_examples --output results
```

Prepare the 100-image Desktop100 dataset:

```bash
python src/prepare_desktop100_from_coco128.py --source data/coco128-seg --output data/desktop100 --num-images 100
```

Run YOLOv11 + SAM 2 inference:

```bash
python src/main.py --input data/desktop100/images --output results --conf 0.20 --imgsz 640 --device cuda
```

Evaluate against the COCO-style ground truth:

```bash
python src/evaluate_coco.py --pred results/annotations/auto_annotations_coco.json --gt data/desktop100/ground_truth_coco.json --output results/tables
```

Main outputs from the completed experiment:

- `results/figures/`: visualized masks, boxes, class labels, and metric plots.
- `results/annotations/auto_annotations_coco.json`: automatic COCO-style annotation file.
- `results/tables/detection_summary.csv`: detected objects, confidence scores, boxes, and mask areas.
- `results/tables/metrics_per_class.csv`: per-class IoU, Dice/F1, Precision, and Recall.

## Demo

The notebook `demos/demo_inference.ipynb` demonstrates the inference pipeline on test images and displays the generated visualization and detection table.

## Results and Analysis

The experiment uses **Desktop100**, a 100-image dataset derived from Ultralytics COCO128-seg by filtering the five target categories and applying deterministic augmentations. The dataset keeps COCO-style segmentation polygons, so the automatic masks can be compared with reference annotations.

Quantitative results on Desktop100:

| Class | IoU | Dice/F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| book | 0.321 | 0.383 | 0.722 | 0.374 |
| bottle | 0.348 | 0.445 | 0.511 | 0.655 |
| bowl | 0.382 | 0.450 | 0.644 | 0.517 |
| cell phone | 0.000 | 0.000 | 0.000 | 0.000 |
| cup | 0.455 | 0.520 | 0.731 | 0.524 |
| overall | 0.340 | 0.403 | 0.644 | 0.447 |

The best results are obtained on cup and bowl, where the detector provides more stable prompts. The cell phone category is the weakest because YOLOv11n misses the small phone instances in this subset. This shows an important limitation of the approach: SAM 2 can refine a prompt into a mask, but it cannot recover an object that is not detected.

Generated outputs include:

- `results/figures/sample_montage.jpg`
- `results/figures/metrics_per_class.png`
- `results/tables/metrics_per_class.csv`
- `results/annotations/auto_annotations_coco.json`

## Conclusion

YOLOv11 + SAM 2 is a practical pipeline for automatic dataset annotation. YOLOv11 supplies semantic class labels and object proposals, while SAM 2 converts bounding box prompts into instance masks. The experiment shows that the quality of the final masks depends strongly on the recall and localization quality of the detector.

Future improvements include using a stronger detector, fine-tuning YOLO on the target classes, and adding a human review step for missed or uncertain detections.

