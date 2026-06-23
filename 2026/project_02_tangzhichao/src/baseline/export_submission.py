import argparse
import json
import shutil
from pathlib import Path

from .utils import ensure_dir, log


def copy_if_exists(src: Path, dst: Path):
    if not src.exists():
        return
    ensure_dir(dst.parent)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Export report-friendly artifacts into results/.")
    parser.add_argument("--output_dir", required=True, help="Experiment directory under outputs/.")
    parser.add_argument("--results_dir", default="results", help="Submission-friendly results directory.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    figures_dir = ensure_dir(results_dir / "figures")
    tables_dir = ensure_dir(results_dir / "tables")

    copy_if_exists(output_dir / "figures", figures_dir / output_dir.name)
    copy_if_exists(output_dir / "summary.json", tables_dir / f"{output_dir.name}_summary.json")
    copy_if_exists(output_dir / "report_assets.md", tables_dir / f"{output_dir.name}_report_assets.md")
    copy_if_exists(output_dir / "eval" / "test_metrics.json", tables_dir / f"{output_dir.name}_test_metrics.json")
    copy_if_exists(output_dir / "eval" / "test_metrics.csv", tables_dir / f"{output_dir.name}_test_metrics.csv")
    log(f"Exported submission artifacts from {output_dir} to {results_dir}")


if __name__ == "__main__":
    main()
