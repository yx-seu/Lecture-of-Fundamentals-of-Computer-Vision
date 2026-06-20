from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yolov8_realtime.metrics import RuntimeMetrics, percentile


class MetricsTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([], 95), 0.0)
        self.assertEqual(percentile([10.0], 95), 10.0)
        self.assertAlmostEqual(percentile([10.0, 20.0, 30.0, 40.0], 50), 25.0)
        self.assertAlmostEqual(percentile([10.0, 20.0, 30.0, 40.0], 90), 37.0)

    def test_runtime_metrics_summary(self) -> None:
        metrics = RuntimeMetrics(source="unit", model="fake")
        metrics.add_frame(latency_ms=10.0, detection_count=2)
        metrics.add_frame(latency_ms=20.0, detection_count=0)
        metrics.finish(elapsed_s=0.1)

        summary = metrics.summary()
        self.assertEqual(summary["frames"], 2)
        self.assertEqual(summary["detections_total"], 2)
        self.assertEqual(summary["fps_wall"], 20.0)
        self.assertEqual(summary["latency_ms_mean"], 15.0)


if __name__ == "__main__":
    unittest.main()

