from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Iterable

from .config import BenchmarkConfig
from .detector import Detector
from .metrics import RuntimeMetrics


def run_frame_benchmark(
    detector: Detector,
    frames: Iterable[Any],
    warmup: int = 0,
    source: str = "synthetic",
    model_name: str = "unknown",
) -> RuntimeMetrics:
    """Benchmark a detector over an iterable of frames."""

    metrics = RuntimeMetrics(source=source, model=model_name)
    measured_started: float | None = None
    measured_finished: float | None = None
    for index, frame in enumerate(frames):
        if index == warmup:
            measured_started = time.perf_counter()
        frame_started = time.perf_counter()
        detections = detector.predict(frame)
        latency_ms = (time.perf_counter() - frame_started) * 1000.0
        if index >= warmup:
            metrics.add_frame(latency_ms=latency_ms, detection_count=len(detections))
            measured_finished = time.perf_counter()
    elapsed_s = 0.0 if measured_started is None or measured_finished is None else measured_finished - measured_started
    metrics.finish(elapsed_s)
    return metrics


def run_synthetic_benchmark(
    detector: Detector,
    config: BenchmarkConfig,
    model_name: str = "unknown",
) -> RuntimeMetrics:
    """Run a benchmark on generated frames."""

    total_frames = config.frames + config.warmup
    frames = synthetic_frames(total_frames, width=config.width, height=config.height)
    return run_frame_benchmark(
        detector=detector,
        frames=frames,
        warmup=config.warmup,
        source=f"synthetic_{config.width}x{config.height}",
        model_name=model_name,
    )


def synthetic_frames(count: int, width: int = 1280, height: int = 720) -> Iterable[Any]:
    """Generate simple moving-shape frames for detector benchmarking."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Synthetic image frames require numpy. Run: "
            "pip install -e \".[runtime]\" or use scripts/run_benchmark.py --fake."
        ) from exc

    for index in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        gradient = (index * 3) % 255
        frame[:, :, 0] = gradient
        frame[:, :, 1] = 28
        frame[:, :, 2] = 42

        box_w = max(width // 8, 40)
        box_h = max(height // 6, 40)
        x1 = (index * 17) % max(1, width - box_w)
        y1 = (index * 9) % max(1, height - box_h)
        frame[y1 : y1 + box_h, x1 : x1 + box_w, :] = (60, 180, 230)

        x2 = width - x1 - box_w
        y2 = height - y1 - box_h
        frame[y2 : y2 + box_h, x2 : x2 + box_w, :] = (220, 80, 90)
        yield frame


def save_benchmark_report(metrics: RuntimeMetrics, output_dir: Path, report_name: str = "benchmark_latest") -> tuple[Path, Path]:
    """Save benchmark metrics as JSON and CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = metrics.summary()
    json_path = output_dir / f"{report_name}.json"
    csv_path = output_dir / f"{report_name}.csv"

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    scalar_fields = {key: value for key, value in summary.items() if not isinstance(value, list)}
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_fields))
        writer.writeheader()
        writer.writerow(scalar_fields)

    return json_path, csv_path
