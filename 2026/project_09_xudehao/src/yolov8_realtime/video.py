from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from .detector import Detection, Detector
from .metrics import RuntimeMetrics


def parse_video_source(source: str) -> str | int:
    """Turn numeric CLI source strings into webcam indexes."""

    value = str(source).strip()
    if value.isdigit():
        return int(value)
    return value


def run_video_detection(
    detector: Detector,
    source: str | int = 0,
    output_path: Path | None = None,
    show: bool = True,
    max_frames: int | None = None,
    window_name: str = "YOLOv8 Real-Time Detection",
    display_scale: float = 1.0,
    fourcc: str = "mp4v",
    model_name: str = "unknown",
) -> RuntimeMetrics:
    """Run real-time detection on a webcam index or video file."""

    cv2 = _import_cv2()
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")

    writer = None
    metrics = RuntimeMetrics(source=str(source), model=model_name)
    started = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if max_frames is not None and metrics.frame_count >= max_frames:
                break

            frame_started = time.perf_counter()
            detections = detector.predict(frame)
            latency_ms = (time.perf_counter() - frame_started) * 1000.0
            metrics.add_frame(latency_ms=latency_ms, detection_count=len(detections))

            annotated = draw_detections(frame, detections)
            _draw_status_bar(cv2, annotated, metrics, latency_ms)

            if output_path is not None:
                writer = writer or _make_writer(cv2, capture, annotated, output_path, fourcc)
                writer.write(annotated)

            if show:
                display = _resize_for_display(cv2, annotated, display_scale)
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        metrics.finish(time.perf_counter() - started)
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    return metrics


def draw_detections(frame: Any, detections: list[Detection]) -> Any:
    """Draw boxes and labels on an OpenCV frame."""

    cv2 = _import_cv2()
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.box_xyxy)
        color = _color_for_label(detection.label)
        label = f"{detection.label} {detection.confidence:.2f}"
        if detection.track_id is not None:
            label = f"#{detection.track_id} {label}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_width, text_height = text_size
        top = max(y1 - text_height - baseline - 6, 0)
        cv2.rectangle(annotated, (x1, top), (x1 + text_width + 8, top + text_height + baseline + 6), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, top + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _draw_status_bar(cv2: Any, frame: Any, metrics: RuntimeMetrics, latency_ms: float) -> None:
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    text = f"Frame {metrics.frame_count} | {latency_ms:.1f} ms | {fps:.1f} FPS | detections {metrics.detections_per_frame[-1]}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(frame, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def _make_writer(cv2: Any, capture: Any, frame: Any, output_path: Path, fourcc: str) -> Any:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    height, width = frame.shape[:2]
    codec = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(output_path), codec, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {output_path}")
    return writer


def _resize_for_display(cv2: Any, frame: Any, scale: float) -> Any:
    if scale == 1.0:
        return frame
    if scale <= 0:
        raise ValueError("display_scale must be greater than 0")
    width = max(1, int(frame.shape[1] * scale))
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _color_for_label(label: str) -> tuple[int, int, int]:
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return (80 + digest[0] % 176, 80 + digest[1] % 176, 80 + digest[2] % 176)


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Missing OpenCV. Run: "
            "pip install -e \".[runtime]\" or pip install -r requirements.txt"
        ) from exc
    return cv2
