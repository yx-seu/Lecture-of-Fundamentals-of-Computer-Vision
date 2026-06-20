from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yolov8_realtime.visualization import load_report, save_benchmark_html


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an HTML visualization from a benchmark JSON report")
    parser.add_argument("report", type=Path, help="Benchmark JSON report path")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path")
    args = parser.parse_args()

    report = load_report(args.report)
    output_path = args.output or args.report.with_suffix(".html")
    save_benchmark_html(report, output_path)
    print(f"HTML report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

