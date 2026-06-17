"""Build a 100-image desktop-object segmentation dataset from COCO128-seg.

The source COCO128-seg package is a small official Ultralytics asset. This
script filters five target desktop classes and creates deterministic augmented
copies so the project has 100 images with matching segmentation ground truth.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import requests
from tqdm import tqdm


COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]
TARGET_CLASSES = ("cup", "bottle", "bowl", "book", "cell phone")
TARGET_SOURCE_IDS = {COCO80_NAMES.index(name): name for name in TARGET_CLASSES}
TARGET_TO_LOCAL_ID = {name: i + 1 for i, name in enumerate(TARGET_CLASSES)}
COCO128_SEG_URL = "https://ultralytics.com/assets/coco128-seg.zip"


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with requests.get(url, stream=True, timeout=90, allow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def ensure_coco128_seg(source: Path) -> None:
    image_dir = source / "images" / "train2017"
    label_dir = source / "labels" / "train2017"
    if image_dir.exists() and label_dir.exists():
        return

    zip_path = source.parent / "coco128-seg.zip"
    print(f"{source} not found. Downloading COCO128-seg...")
    download_file(COCO128_SEG_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(source.parent)


def read_yolo_segments(label_path: Path) -> List[Tuple[str, np.ndarray]]:
    segments: List[Tuple[str, np.ndarray]] = []
    if not label_path.exists():
        return segments
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        source_id = int(float(parts[0]))
        class_name = TARGET_SOURCE_IDS.get(source_id)
        if class_name is None:
            continue
        coords = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        segments.append((class_name, coords))
    return segments


def transform_image_and_segments(
    image: np.ndarray,
    segments: Sequence[Tuple[str, np.ndarray]],
    variant: int,
) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]], str]:
    out = image.copy()
    out_segments = [(name, pts.copy()) for name, pts in segments]
    tag = "orig"

    if variant % 4 == 1:
        out = cv2.flip(out, 1)
        out_segments = [(name, np.column_stack([1.0 - pts[:, 0], pts[:, 1]])) for name, pts in out_segments]
        tag = "flip"
    elif variant % 4 == 2:
        out = cv2.convertScaleAbs(out, alpha=1.12, beta=12)
        tag = "bright"
    elif variant % 4 == 3:
        out = cv2.convertScaleAbs(out, alpha=0.88, beta=-8)
        tag = "dark"

    return out, out_segments, tag


def segment_bbox_px(segment: np.ndarray, width: int, height: int) -> List[float]:
    xs = np.clip(segment[:, 0] * width, 0, width - 1)
    ys = np.clip(segment[:, 1] * height, 0, height - 1)
    return [float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())]


def segment_to_polygon_px(segment: np.ndarray, width: int, height: int) -> List[float]:
    pts = segment.copy()
    pts[:, 0] = np.clip(pts[:, 0] * width, 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1] * height, 0, height - 1)
    return pts.reshape(-1).astype(float).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/coco128-seg"))
    parser.add_argument("--output", type=Path, default=Path("data/desktop100"))
    parser.add_argument("--num-images", type=int, default=100)
    args = parser.parse_args()
    ensure_coco128_seg(args.source)

    source_images = args.source / "images" / "train2017"
    source_labels = args.source / "labels" / "train2017"
    out_images = args.output / "images"
    out_labels = args.output / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    candidates = []
    for label_path in sorted(source_labels.glob("*.txt")):
        segments = read_yolo_segments(label_path)
        image_path = source_images / f"{label_path.stem}.jpg"
        if segments and image_path.exists():
            candidates.append((image_path, segments))
    if not candidates:
        raise RuntimeError("No target-class segmentation labels found.")

    coco_images = []
    coco_annotations = []
    ann_id = 1
    image_id = 1
    idx = 0
    with tqdm(total=args.num_images, desc="desktop100") as bar:
        while image_id <= args.num_images:
            image_path, segments = candidates[idx % len(candidates)]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                idx += 1
                continue
            variant = idx // len(candidates)
            aug_image, aug_segments, tag = transform_image_and_segments(image, segments, variant)
            file_name = f"desktop_{image_id:04d}_{tag}.jpg"
            label_name = f"desktop_{image_id:04d}_{tag}.txt"
            height, width = aug_image.shape[:2]
            cv2.imwrite(str(out_images / file_name), aug_image)

            label_lines = []
            coco_images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
            for class_name, segment in aug_segments:
                local_zero_id = TARGET_CLASSES.index(class_name)
                label_lines.append(
                    " ".join([str(local_zero_id)] + [f"{v:.6f}" for v in segment.reshape(-1).tolist()])
                )
                polygon = segment_to_polygon_px(segment, width, height)
                bbox = segment_bbox_px(segment, width, height)
                coco_annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": TARGET_TO_LOCAL_ID[class_name],
                        "bbox": bbox,
                        "area": float(bbox[2] * bbox[3]),
                        "segmentation": [polygon],
                        "iscrowd": 0,
                        "source": image_path.name,
                    }
                )
                ann_id += 1
            (out_labels / label_name).write_text("\n".join(label_lines), encoding="utf-8")
            image_id += 1
            idx += 1
            bar.update(1)

    categories = [
        {"id": i + 1, "name": name, "supercategory": "desktop object"} for i, name in enumerate(TARGET_CLASSES)
    ]
    dataset = {"images": coco_images, "annotations": coco_annotations, "categories": categories}
    (args.output / "ground_truth_coco.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    (args.output / "dataset_info.txt").write_text(
        "Desktop100: 100 augmented images generated from Ultralytics COCO128-seg. "
        "Target classes: cup, bottle, bowl, book, cell phone.\n",
        encoding="utf-8",
    )
    print(f"Saved dataset to {args.output}")


if __name__ == "__main__":
    main()
