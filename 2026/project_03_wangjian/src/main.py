"""Run sample inference for YOLOv11 + SAM 2 automatic annotation.

Example:
    python src/main.py --input data/test_examples --output results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import (
    TARGET_CLASSES,
    detect_with_yolo,
    detections_to_coco,
    ensure_dir,
    evaluate_predictions,
    list_images,
    load_image_rgb,
    load_sam2_predictor,
    load_yolo_model,
    save_image_rgb,
    save_json,
    segment_with_sam2,
    visualize_annotations,
    write_detection_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatic multi-class mask annotation with YOLOv11 and SAM 2.")
    parser.add_argument("--input", type=Path, default=Path("data/test_examples"), help="Image file or directory.")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory.")
    parser.add_argument("--yolo-model", default="yolo11n.pt", help="Ultralytics YOLO model path/name.")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_s.yaml", help="SAM 2 model config.")
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("weights/sam2.1_hiera_small.pt"), help="SAM 2 checkpoint.")
    parser.add_argument("--device", default="cuda", help="Device for SAM 2, usually cuda or cpu.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size.")
    parser.add_argument("--manual-labelme", type=Path, default=None, help="Optional LabelMe JSON directory for evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = list_images(args.input)
    if not images:
        raise FileNotFoundError(f"No images found in {args.input}. Put 2-3 examples in data/test_examples first.")

    figure_dir = ensure_dir(args.output / "figures")
    annotation_dir = ensure_dir(args.output / "annotations")
    table_dir = ensure_dir(args.output / "tables")

    yolo_model = load_yolo_model(args.yolo_model)
    sam2_predictor = load_sam2_predictor(args.sam2_config, args.sam2_checkpoint, args.device)

    image_records = []
    annotations_by_image = {}
    summary_rows = []

    for image_id, image_path in enumerate(images, start=1):
        image_rgb = load_image_rgb(image_path)
        height, width = image_rgb.shape[:2]
        image_records.append(
            {
                "id": image_id,
                "file_name": str(image_path.as_posix()),
                "width": width,
                "height": height,
            }
        )

        detections = detect_with_yolo(
            yolo_model,
            image_rgb,
            target_classes=TARGET_CLASSES,
            confidence_threshold=args.conf,
            image_size=args.imgsz,
        )
        detections = segment_with_sam2(sam2_predictor, image_rgb, detections)
        annotations_by_image[image_id] = detections

        vis = visualize_annotations(image_rgb, detections)
        save_image_rgb(figure_dir / f"{image_path.stem}_annotated.jpg", vis)

        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            summary_rows.append(
                {
                    "image": image_path.name,
                    "class": det.class_name,
                    "confidence": round(det.confidence, 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "mask_area": int(det.mask.sum()) if det.mask is not None else 0,
                }
            )

    coco = detections_to_coco(image_records, annotations_by_image, TARGET_CLASSES)
    save_json(annotation_dir / "auto_annotations_coco.json", coco)
    write_detection_summary(table_dir / "detection_summary.csv", summary_rows)

    if args.manual_labelme is not None and args.manual_labelme.exists():
        metrics = evaluate_predictions(image_records, annotations_by_image, args.manual_labelme, TARGET_CLASSES)
        metrics.to_csv(table_dir / "metrics_per_image_class.csv", index=False, encoding="utf-8")
        metrics.groupby("class")[["iou", "dice", "precision", "recall"]].mean().reset_index().to_csv(
            table_dir / "metrics_per_class.csv", index=False, encoding="utf-8"
        )
    else:
        pd.DataFrame(
            [
                {
                    "note": "Add LabelMe JSON files to data/manual_labelme and rerun with "
                    "--manual-labelme data/manual_labelme to compute IoU/Dice/Precision/Recall."
                }
            ]
        ).to_csv(table_dir / "metrics_todo.csv", index=False, encoding="utf-8")

    print(f"Processed {len(images)} image(s).")
    print(f"Visualizations: {figure_dir}")
    print(f"COCO annotations: {annotation_dir / 'auto_annotations_coco.json'}")
    print(f"Detection summary: {table_dir / 'detection_summary.csv'}")


if __name__ == "__main__":
    main()
