from __future__ import annotations

import sys
from pathlib import Path
import tempfile
from typing import Any
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yolov8_realtime.benchmark import run_frame_benchmark, save_benchmark_report
from yolov8_realtime.detector import Detection


class FakeDetector:
    def predict(self, frame: Any) -> list[Detection]:
        return [
            Detection(label="unit", confidence=0.99, box_xyxy=(0.0, 0.0, 10.0, 10.0))
            for _ in range(frame["detections"])
        ]


class BenchmarkTests(unittest.TestCase):
    def test_run_frame_benchmark_skips_warmup(self) -> None:
        frames = [{"detections": value} for value in [9, 1, 2, 3]]
        metrics = run_frame_benchmark(FakeDetector(), frames, warmup=1, source="unit", model_name="fake")
        summary = metrics.summary()

        self.assertEqual(summary["frames"], 3)
        self.assertEqual(summary["detections_total"], 6)
        self.assertEqual(summary["detections_per_frame"], [1, 2, 3])

    def test_save_benchmark_report(self) -> None:
        metrics = run_frame_benchmark(FakeDetector(), [{"detections": 1}], warmup=0)
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, csv_path = save_benchmark_report(metrics, Path(temp_dir), "unit_report")
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertIn('"frames": 1', json_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

