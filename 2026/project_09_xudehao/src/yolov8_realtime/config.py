from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DetectorConfig:
    """YOLOv8 inference settings."""

    model_path: str = "yolov8n.pt"
    confidence: float = 0.25
    iou: float = 0.45
    image_size: int = 640
    device: str | None = None
    classes: list[int] | None = None
    half: bool = False
    use_tracker: bool = False


@dataclass(slots=True)
class VideoRunConfig:
    """Video source, display, and output settings."""

    source: str | int = 0
    output_path: Path | None = None
    show: bool = True
    max_frames: int | None = None
    window_name: str = "YOLOv8 Real-Time Detection"
    display_scale: float = 1.0
    fourcc: str = "mp4v"


@dataclass(slots=True)
class BenchmarkConfig:
    """Synthetic benchmark settings."""

    frames: int = 120
    warmup: int = 10
    width: int = 1280
    height: int = 720
    output_dir: Path = field(default_factory=lambda: Path("outputs") / "benchmarks")
    report_name: str = "benchmark_latest"

