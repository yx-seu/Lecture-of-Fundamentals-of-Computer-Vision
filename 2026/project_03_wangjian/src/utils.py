"""Utilities for YOLOv11 + SAM 2 automatic dataset annotation.

The project uses YOLO to provide semantic classes and bounding boxes, then uses
SAM 2 to convert each box prompt into an instance mask. A lightweight fallback is
included only for local smoke tests when model weights are not installed yet.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except Exception:
    cv2 = None


TARGET_CLASSES: Tuple[str, ...] = ("cup", "bottle", "bowl", "book", "cell phone")


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    mask: Optional[np.ndarray] = None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(input_path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if input_path.is_file() and input_path.suffix.lower() in exts:
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in exts)


def load_image_rgb(path: Path) -> np.ndarray:
    if cv2 is not None:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Cannot read image: {path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return np.array(Image.open(path).convert("RGB"))


def save_image_rgb(path: Path, image_rgb: np.ndarray) -> None:
    ensure_dir(path.parent)
    if cv2 is not None:
        cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    else:
        Image.fromarray(image_rgb.astype(np.uint8), "RGB").save(path)


def load_yolo_model(model_name: str = "yolo11n.pt") -> Optional[Any]:
    try:
        from ultralytics import YOLO

        return YOLO(model_name)
    except Exception as exc:
        print(f"[WARN] YOLO model is unavailable: {exc}")
        return None


def load_sam2_predictor(
    config: str,
    checkpoint: Optional[Path],
    device: str = "cuda",
) -> Optional[Any]:
    """Load SAM 2 image predictor if the package and checkpoint are available."""
    if checkpoint is None or not checkpoint.exists():
        print("[WARN] SAM 2 checkpoint is missing; using fallback mask generation.")
        return None
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(config, str(checkpoint), device=device)
        return SAM2ImagePredictor(model)
    except Exception as exc:
        print(f"[WARN] SAM 2 predictor is unavailable: {exc}")
        return None


def detect_with_yolo(
    model: Optional[Any],
    image_rgb: np.ndarray,
    target_classes: Sequence[str] = TARGET_CLASSES,
    confidence_threshold: float = 0.25,
    image_size: int = 960,
) -> List[Detection]:
    if model is None:
        return fallback_detect_colored_regions(image_rgb, target_classes)

    results = model.predict(image_rgb, conf=confidence_threshold, imgsz=image_size, verbose=False)
    detections: List[Detection] = []
    if not results:
        return detections

    names = results[0].names
    boxes = results[0].boxes
    if boxes is None:
        return detections

    for box in boxes:
        class_id = int(box.cls[0].item())
        class_name = str(names[class_id])
        if class_name not in target_classes:
            continue
        confidence = float(box.conf[0].item())
        xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
        detections.append(Detection(class_name=class_name, confidence=confidence, bbox_xyxy=xyxy))
    return detections


def fallback_detect_colored_regions(
    image_rgb: np.ndarray,
    target_classes: Sequence[str] = TARGET_CLASSES,
) -> List[Detection]:
    """Smoke-test detector for synthetic examples when YOLO weights are absent."""
    if cv2 is None:
        return fallback_detect_colored_regions_numpy(image_rgb, target_classes)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    color_range = np.ptp(image_rgb.astype(np.int16), axis=2)
    mask = np.logical_or(color_range > 35, gray < 95).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections: List[Detection] = []
    for idx, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)):
        area = cv2.contourArea(contour)
        if area < 400:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        class_name = target_classes[idx % len(target_classes)]
        detections.append(
            Detection(
                class_name=class_name,
                confidence=0.50,
                bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
            )
        )
    return detections


def fallback_detect_colored_regions_numpy(
    image_rgb: np.ndarray,
    target_classes: Sequence[str] = TARGET_CLASSES,
) -> List[Detection]:
    gray = image_rgb.mean(axis=2)
    color_range = np.ptp(image_rgb.astype(np.int16), axis=2)
    foreground = np.logical_or(color_range > 35, gray < 95)
    h, w = foreground.shape
    visited = np.zeros_like(foreground, dtype=bool)
    detections: List[Detection] = []

    for start_y in range(h):
        for start_x in range(w):
            if not foreground[start_y, start_x] or visited[start_y, start_x]:
                continue
            q: deque[Tuple[int, int]] = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            xs: List[int] = []
            ys: List[int] = []

            while q:
                y, x = q.popleft()
                xs.append(x)
                ys.append(y)
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and foreground[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))

            if len(xs) < 400:
                continue
            x1, x2 = min(xs), max(xs) + 1
            y1, y2 = min(ys), max(ys) + 1
            # Ignore the large synthetic background panels.
            if (x2 - x1) > 0.85 * w or (y2 - y1) > 0.85 * h:
                continue
            class_name = target_classes[len(detections) % len(target_classes)]
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=0.50,
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
    return sorted(detections, key=lambda d: (d.bbox_xyxy[1], d.bbox_xyxy[0]))


def segment_with_sam2(
    predictor: Optional[Any],
    image_rgb: np.ndarray,
    detections: List[Detection],
) -> List[Detection]:
    if predictor is not None:
        predictor.set_image(image_rgb)

    for det in detections:
        box = np.array(det.bbox_xyxy, dtype=np.float32)
        if predictor is None:
            det.mask = fallback_mask_from_box(image_rgb, det.bbox_xyxy)
            continue

        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        det.mask = masks[best_idx].astype(bool)
    return detections


def fallback_mask_from_box(image_rgb: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]) -> np.ndarray:
    """Use GrabCut with a box prompt as a weak fallback for demo smoke tests."""
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), dtype=bool)
    if cv2 is None:
        simple_mask = np.zeros((h, w), dtype=bool)
        simple_mask[y1:y2, x1:x2] = True
        return simple_mask

    rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        return np.logical_or(mask == cv2.GC_FGD, mask == cv2.GC_PR_FGD)
    except Exception:
        simple_mask = np.zeros((h, w), dtype=bool)
        simple_mask[y1:y2, x1:x2] = True
        return simple_mask


def visualize_annotations(
    image_rgb: np.ndarray,
    detections: Sequence[Detection],
    alpha: float = 0.45,
) -> np.ndarray:
    if cv2 is None:
        return visualize_annotations_pil(image_rgb, detections, alpha)

    output = image_rgb.copy()
    overlay = image_rgb.copy()
    colors = {
        "cup": (231, 76, 60),
        "bottle": (46, 204, 113),
        "bowl": (52, 152, 219),
        "book": (241, 196, 15),
        "cell phone": (155, 89, 182),
    }

    for det in detections:
        color = colors.get(det.class_name, (255, 127, 80))
        if det.mask is not None:
            overlay[det.mask.astype(bool)] = color

    output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)
    for det in detections:
        color = colors.get(det.class_name, (255, 127, 80))
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        draw_label(output, label, (x1, max(18, y1 - 8)), color)
    return output


def visualize_annotations_pil(
    image_rgb: np.ndarray,
    detections: Sequence[Detection],
    alpha: float = 0.45,
) -> np.ndarray:
    colors = {
        "cup": (231, 76, 60),
        "bottle": (46, 204, 113),
        "bowl": (52, 152, 219),
        "book": (241, 196, 15),
        "cell phone": (155, 89, 182),
    }
    base = Image.fromarray(image_rgb.astype(np.uint8), "RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for det in detections:
        color = colors.get(det.class_name, (255, 127, 80))
        if det.mask is not None:
            mask_img = Image.fromarray((det.mask.astype(np.uint8) * int(255 * alpha)), "L")
            color_img = Image.new("RGBA", base.size, (*color, int(255 * alpha)))
            overlay = Image.composite(color_img, overlay, mask_img)

    composed = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(composed)
    for det in detections:
        color = colors.get(det.class_name, (255, 127, 80))
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{det.class_name} {det.confidence:.2f}"
        text_bbox = draw.textbbox((x1, max(0, y1 - 20)), label)
        draw.rectangle((text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 4, text_bbox[3] + 3), fill=color)
        draw.text((x1 + 1, max(0, y1 - 20)), label, fill=(255, 255, 255))
    return np.array(composed.convert("RGB"))


def draw_label(image_rgb: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    if cv2 is None:
        return
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image_rgb, (x, y - th - baseline - 4), (x + tw + 6, y + 3), color, -1)
    cv2.putText(image_rgb, text, (x + 3, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def detections_to_coco(
    image_records: Sequence[Dict[str, Any]],
    annotations_by_image: Dict[int, Sequence[Detection]],
    category_names: Sequence[str] = TARGET_CLASSES,
) -> Dict[str, Any]:
    categories = [{"id": i + 1, "name": name, "supercategory": "desktop object"} for i, name in enumerate(category_names)]
    category_id = {cat["name"]: int(cat["id"]) for cat in categories}

    annotations: List[Dict[str, Any]] = []
    ann_id = 1
    for image in image_records:
        image_id = int(image["id"])
        for det in annotations_by_image.get(image_id, []):
            x1, y1, x2, y2 = det.bbox_xyxy
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            mask = det.mask.astype(np.uint8) if det.mask is not None else np.zeros((image["height"], image["width"]), np.uint8)
            segmentation, area = mask_to_polygons(mask)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id[det.class_name],
                    "bbox": bbox,
                    "area": float(area),
                    "segmentation": segmentation,
                    "iscrowd": 0,
                    "score": float(det.confidence),
                }
            )
            ann_id += 1

    return {
        "images": list(image_records),
        "annotations": annotations,
        "categories": categories,
    }


def mask_to_polygons(mask: np.ndarray) -> Tuple[List[List[float]], float]:
    mask_u8 = (mask > 0).astype(np.uint8)
    if cv2 is None:
        ys, xs = np.where(mask_u8 > 0)
        area = float(len(xs))
        if len(xs) == 0:
            return [], area
        x1, x2 = float(xs.min()), float(xs.max() + 1)
        y1, y2 = float(ys.min()), float(ys.max() + 1)
        return [[x1, y1, x2, y1, x2, y2, x1, y2]], area

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[float]] = []
    area = float(mask_u8.sum())
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
        contour = contour.reshape(-1, 2)
        if len(contour) < 3:
            continue
        polygons.append(contour.astype(float).flatten().tolist())
    return polygons, area


def save_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_detection_summary(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def load_labelme_masks(
    labelme_dir: Path,
    image_shape: Tuple[int, int],
    image_stem: str,
    category_names: Sequence[str] = TARGET_CLASSES,
) -> Dict[str, np.ndarray]:
    json_path = labelme_dir / f"{image_stem}.json"
    h, w = image_shape
    masks = {name: np.zeros((h, w), dtype=bool) for name in category_names}
    if not json_path.exists():
        return masks

    data = json.loads(json_path.read_text(encoding="utf-8"))
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in masks:
            continue
        points = np.array(shape.get("points", []), dtype=np.int32)
        if len(points) >= 3:
            if cv2 is not None:
                cv2.fillPoly(masks[label], [points], True)
            else:
                img = Image.new("1", (w, h), 0)
                draw = ImageDraw.Draw(img)
                draw.polygon([tuple(p) for p in points.tolist()], outline=1, fill=1)
                masks[label] |= np.array(img).astype(bool)
    return masks


def evaluate_predictions(
    image_rows: Sequence[Dict[str, Any]],
    annotations_by_image: Dict[int, Sequence[Detection]],
    labelme_dir: Path,
    category_names: Sequence[str] = TARGET_CLASSES,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for image in image_rows:
        image_path = Path(image["file_name"])
        h, w = int(image["height"]), int(image["width"])
        gt_masks = load_labelme_masks(labelme_dir, (h, w), Path(image_path).stem, category_names)
        pred_masks = {name: np.zeros((h, w), dtype=bool) for name in category_names}

        for det in annotations_by_image.get(int(image["id"]), []):
            if det.mask is not None:
                pred_masks[det.class_name] |= det.mask.astype(bool)

        for class_name in category_names:
            metrics = binary_mask_metrics(pred_masks[class_name], gt_masks[class_name])
            rows.append({"image": image_path.name, "class": class_name, **metrics})
    return pd.DataFrame(rows)


def binary_mask_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    iou = float(tp / union) if union else math.nan
    precision = float(tp / (tp + fp)) if (tp + fp) else math.nan
    recall = float(tp / (tp + fn)) if (tp + fn) else math.nan
    dice = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else math.nan
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}
