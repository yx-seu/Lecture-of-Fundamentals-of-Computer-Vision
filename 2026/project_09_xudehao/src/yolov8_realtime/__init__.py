"""YOLOv8 real-time video object detection utilities."""

from .config import BenchmarkConfig, DetectorConfig, VideoRunConfig
from .detector import Detection, YoloV8Detector
from .metrics import RuntimeMetrics

__all__ = [
    "BenchmarkConfig",
    "Detection",
    "DetectorConfig",
    "RuntimeMetrics",
    "VideoRunConfig",
    "YoloV8Detector",
]

