from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yolov8_realtime.config import DetectorConfig, VideoRunConfig
from yolov8_realtime.detector import YoloV8Detector
from yolov8_realtime.video import parse_video_source, run_video_detection


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv8 real-time video object detection")
    parser.add_argument("--source", default="0", help="Webcam index or video file path")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model path, e.g. yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default=None, help="Device, e.g. cpu or 0")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="Optional class ids to keep")
    parser.add_argument("--half", action="store_true", help="Use FP16 when supported")
    parser.add_argument("--track", action="store_true", help="Use YOLOv8 tracker mode")
    parser.add_argument("--output", type=Path, default=None, help="Optional annotated video path")
    parser.add_argument("--no-show", action="store_true", help="Do not open a display window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Display window scaling")
    args = parser.parse_args()

    detector_config = DetectorConfig(
        model_path=args.model,
        confidence=args.conf,
        iou=args.iou,
        image_size=args.imgsz,
        device=args.device,
        classes=args.classes,
        half=args.half,
        use_tracker=args.track,
    )
    video_config = VideoRunConfig(
        source=parse_video_source(args.source),
        output_path=args.output,
        show=not args.no_show,
        max_frames=args.max_frames,
        display_scale=args.display_scale,
    )

    detector = YoloV8Detector(detector_config)
    metrics = run_video_detection(
        detector=detector,
        source=video_config.source,
        output_path=video_config.output_path,
        show=video_config.show,
        max_frames=video_config.max_frames,
        window_name=video_config.window_name,
        display_scale=video_config.display_scale,
        fourcc=video_config.fourcc,
        model_name=detector.model_name,
    )
    print(json.dumps(metrics.summary(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

