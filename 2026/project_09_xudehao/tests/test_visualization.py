from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yolov8_realtime.visualization import render_benchmark_html, save_benchmark_html


class VisualizationTests(unittest.TestCase):
    def test_render_benchmark_html_contains_charts(self) -> None:
        html = render_benchmark_html(
            {
                "model": "fake",
                "source": "unit",
                "frames": 3,
                "fps_wall": 30.0,
                "fps_inference_mean": 60.0,
                "latency_ms_mean": 16.6,
                "latency_ms_p95": 20.0,
                "detections_total": 4,
                "latencies_ms": [10.0, 20.0, 15.0],
                "detections_per_frame": [1, 2, 1],
            }
        )
        self.assertIn("YOLOv8 Benchmark Report", html)
        self.assertIn("<svg", html)
        self.assertIn("fake", html)

    def test_save_benchmark_html(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "report.html"
            save_benchmark_html({"latencies_ms": [], "detections_per_frame": []}, output)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()

