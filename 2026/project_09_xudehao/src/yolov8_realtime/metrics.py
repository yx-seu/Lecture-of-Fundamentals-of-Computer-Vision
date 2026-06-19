from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean


@dataclass(slots=True)
class RuntimeMetrics:
    """Collect and summarize per-frame runtime measurements."""

    source: str = "unknown"
    model: str = "unknown"
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    frame_count: int = 0
    total_detections: int = 0
    elapsed_s: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    detections_per_frame: list[int] = field(default_factory=list)

    def add_frame(self, latency_ms: float, detection_count: int) -> None:
        self.frame_count += 1
        self.total_detections += detection_count
        self.latencies_ms.append(float(latency_ms))
        self.detections_per_frame.append(int(detection_count))

    def finish(self, elapsed_s: float) -> None:
        self.elapsed_s = max(float(elapsed_s), 0.0)

    @property
    def fps_wall(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return self.frame_count / self.elapsed_s

    @property
    def fps_inference_mean(self) -> float:
        if not self.latencies_ms:
            return 0.0
        avg_latency = mean(self.latencies_ms)
        if avg_latency <= 0:
            return 0.0
        return 1000.0 / avg_latency

    def summary(self) -> dict[str, float | int | str | list[float] | list[int]]:
        return {
            "source": self.source,
            "model": self.model,
            "started_at": self.started_at,
            "frames": self.frame_count,
            "elapsed_s": round(self.elapsed_s, 4),
            "fps_wall": round(self.fps_wall, 3),
            "fps_inference_mean": round(self.fps_inference_mean, 3),
            "detections_total": self.total_detections,
            "detections_per_frame_mean": round(_safe_mean(self.detections_per_frame), 3),
            "latency_ms_mean": round(_safe_mean(self.latencies_ms), 3),
            "latency_ms_p50": round(percentile(self.latencies_ms, 50), 3),
            "latency_ms_p90": round(percentile(self.latencies_ms, 90), 3),
            "latency_ms_p95": round(percentile(self.latencies_ms, 95), 3),
            "latency_ms_max": round(max(self.latencies_ms), 3) if self.latencies_ms else 0.0,
            "latencies_ms": [round(value, 4) for value in self.latencies_ms],
            "detections_per_frame": list(self.detections_per_frame),
        }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))

    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * (q / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _safe_mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return float(mean(values))

