from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yolov8_realtime.benchmark import run_frame_benchmark, run_synthetic_benchmark, save_benchmark_report
from yolov8_realtime.config import BenchmarkConfig, DetectorConfig
from yolov8_realtime.detector import Detection, YoloV8Detector


class FakeDetector:
    model_name = "fake-detector"

    def predict(self, frame: Any) -> list[Detection]:
        index = int(frame.get("index", 0)) if isinstance(frame, dict) else 0
        return [
            Detection(label="demo", confidence=0.9, box_xyxy=(10.0, 12.0, 110.0, 140.0), class_id=0)
            for _ in range(index % 4)
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLOv8 benchmark and save JSON/CSV reports")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model path")
    parser.add_argument("--frames", type=int, default=120, help="Measured frame count")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup frame count")
    parser.add_argument("--width", type=int, default=1280, help="Synthetic frame width")
    parser.add_argument("--height", type=int, default=720, help="Synthetic frame height")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default=None, help="Device, e.g. cpu or 0")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="Optional class ids to keep")
    parser.add_argument("--half", action="store_true", help="Use FP16 when supported")
    parser.add_argument("--fake", action="store_true", help="Run a dependency-free smoke benchmark")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "benchmarks", help="Report directory")
    parser.add_argument("--report-name", default="benchmark_latest", help="Report filename without extension")
    args = parser.parse_args()

    benchmark_config = BenchmarkConfig(
        frames=args.frames,
        warmup=args.warmup,
        width=args.width,
        height=args.height,
        output_dir=args.output_dir,
        report_name=args.report_name,
    )

    if args.fake:
        detector = FakeDetector()
        frames = ({"index": index} for index in range(args.frames + args.warmup))
        metrics = run_frame_benchmark(
            detector=detector,
            frames=frames,
            warmup=args.warmup,
            source="fake_frames",
            model_name=detector.model_name,
        )
    else:
        detector_config = DetectorConfig(
            model_path=args.model,
            confidence=args.conf,
            iou=args.iou,
            image_size=args.imgsz,
            device=args.device,
            classes=args.classes,
            half=args.half,
        )
        detector = YoloV8Detector(detector_config)
        metrics = run_synthetic_benchmark(detector, benchmark_config, model_name=detector.model_name)

    json_path, csv_path = save_benchmark_report(metrics, benchmark_config.output_dir, benchmark_config.report_name)
    print(json.dumps(metrics.summary(), ensure_ascii=False, indent=2))
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

