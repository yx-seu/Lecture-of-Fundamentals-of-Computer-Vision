from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .config import DetectorConfig


@dataclass(frozen=True, slots=True)
class Detection:
    """One object detection in xyxy image coordinates."""

    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    class_id: int | None = None
    track_id: int | None = None


class Detector(Protocol):
    """Protocol used by video and benchmark pipelines."""

    def predict(self, frame: Any) -> list[Detection]:
        """Return detections for one frame."""


class YoloV8Detector:
    """Small wrapper around ultralytics.YOLO with a stable project interface."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Missing YOLOv8 runtime dependencies. Run: "
                "pip install -e \".[runtime]\" or pip install -r requirements.txt"
            ) from exc

        self.model = YOLO(config.model_path)

    @property
    def model_name(self) -> str:
        return self.config.model_path

    def predict(self, frame: Any) -> list[Detection]:
        runner = self.model.track if self.config.use_tracker else self.model.predict
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self.config.confidence,
            "iou": self.config.iou,
            "imgsz": self.config.image_size,
            "half": self.config.half,
            "classes": self.config.classes,
            "verbose": False,
        }
        if self.config.device is not None:
            kwargs["device"] = self.config.device
        if self.config.use_tracker:
            kwargs["persist"] = True

        results = runner(**kwargs)
        if not results:
            return []
        return _parse_ultralytics_result(results[0])


def _parse_ultralytics_result(result: Any) -> list[Detection]:
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = _tensor_to_list(boxes.xyxy)
    confidences = _tensor_to_list(boxes.conf)
    class_ids = [int(value) for value in _tensor_to_list(boxes.cls)]
    track_ids = None
    if getattr(boxes, "id", None) is not None:
        track_ids = [int(value) for value in _tensor_to_list(boxes.id)]

    detections: list[Detection] = []
    for index, box in enumerate(xyxy):
        class_id = class_ids[index]
        label = str(names.get(class_id, class_id))
        track_id = track_ids[index] if track_ids is not None else None
        detections.append(
            Detection(
                label=label,
                confidence=float(confidences[index]),
                box_xyxy=tuple(float(value) for value in box),
                class_id=class_id,
                track_id=track_id,
            )
        )
    return detections


def _tensor_to_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)
