"""Evaluate automatic masks against COCO-style ground-truth segmentations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


TARGET_CLASSES = ("cup", "bottle", "bowl", "book", "cell phone")


def polygon_mask(size: Tuple[int, int], segmentations: Sequence[Sequence[float]]) -> np.ndarray:
    width, height = size
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for segmentation in segmentations:
        if not isinstance(segmentation, list) or len(segmentation) < 6:
            continue
        points = [(float(segmentation[i]), float(segmentation[i + 1])) for i in range(0, len(segmentation), 2)]
        draw.polygon(points, outline=1, fill=1)
    return np.array(image).astype(bool)


def ann_to_mask(size: Tuple[int, int], ann: dict) -> np.ndarray:
    segmentation = ann.get("segmentation", [])
    if isinstance(segmentation, list):
        return polygon_mask(size, segmentation)
    # RLE annotations are uncommon for the selected non-crowd subset. Keep a
    # bbox fallback to avoid optional pycocotools.
    x, y, w, h = [int(round(v)) for v in ann.get("bbox", [0, 0, 0, 0])]
    mask = np.zeros((size[1], size[0]), dtype=bool)
    mask[max(0, y) : max(0, y + h), max(0, x) : max(0, x + w)] = True
    return mask


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = float(tp / union) if union else np.nan
    dice = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else np.nan
    precision = float(tp / (tp + fp)) if (tp + fp) else np.nan
    recall = float(tp / (tp + fn)) if (tp + fn) else np.nan
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate automatic COCO masks against ground truth.")
    parser.add_argument("--pred", type=Path, default=Path("results/annotations/auto_annotations_coco.json"))
    parser.add_argument("--gt", type=Path, default=Path("data/coco_desktop_100/ground_truth_coco.json"))
    parser.add_argument("--output", type=Path, default=Path("results/tables"))
    args = parser.parse_args()

    pred = json.loads(args.pred.read_text(encoding="utf-8"))
    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    pred_categories = {cat["id"]: cat["name"] for cat in pred["categories"]}
    gt_categories = {cat["id"]: cat["name"] for cat in gt["categories"]}

    pred_images = {Path(img["file_name"]).name: img for img in pred["images"]}
    gt_images = {img["file_name"]: img for img in gt["images"]}

    pred_anns: Dict[Tuple[str, str], List[dict]] = {}
    pred_image_name_by_id = {img["id"]: Path(img["file_name"]).name for img in pred["images"]}
    for ann in pred["annotations"]:
        class_name = pred_categories.get(ann["category_id"])
        image_name = pred_image_name_by_id.get(ann["image_id"])
        if class_name in TARGET_CLASSES and image_name:
            pred_anns.setdefault((image_name, class_name), []).append(ann)

    gt_anns: Dict[Tuple[str, str], List[dict]] = {}
    gt_image_name_by_id = {img["id"]: img["file_name"] for img in gt["images"]}
    for ann in gt["annotations"]:
        class_name = gt_categories.get(ann["category_id"])
        image_name = gt_image_name_by_id.get(ann["image_id"])
        if class_name in TARGET_CLASSES and image_name:
            gt_anns.setdefault((image_name, class_name), []).append(ann)

    rows = []
    for image_name, image in gt_images.items():
        if image_name not in pred_images:
            continue
        size = (int(image["width"]), int(image["height"]))
        for class_name in TARGET_CLASSES:
            pred_mask = np.zeros((size[1], size[0]), dtype=bool)
            gt_mask = np.zeros((size[1], size[0]), dtype=bool)
            for ann in pred_anns.get((image_name, class_name), []):
                pred_mask |= ann_to_mask(size, ann)
            for ann in gt_anns.get((image_name, class_name), []):
                gt_mask |= ann_to_mask(size, ann)
            metrics = binary_metrics(pred_mask, gt_mask)
            if not np.isnan(metrics["iou"]):
                rows.append({"image": image_name, "class": class_name, **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(args.output / "metrics_per_image_class.csv", index=False)
    per_class = df.groupby("class")[["iou", "dice", "precision", "recall"]].mean().reset_index()
    overall = pd.DataFrame([{"class": "overall", **df[["iou", "dice", "precision", "recall"]].mean().to_dict()}])
    pd.concat([per_class, overall], ignore_index=True).to_csv(args.output / "metrics_per_class.csv", index=False)
    print(per_class)
    print("Saved metrics to", args.output)


if __name__ == "__main__":
    main()
