import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config_path = output_dir / "config_used.yaml"
    summary_path = output_dir / "summary.json"
    log_path = output_dir / "logs" / "train_log.csv"
    report_path = output_dir / "report_assets.md"

    config = {}
    summary = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    epoch_count = 0
    if log_path.exists():
        epoch_count = len(pd.read_csv(log_path))

    figure_paths = sorted(
        [
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*.png")
        ]
    )

    lines = [
        "# Report Assets",
        "",
        "## Experiment Summary",
        f"- Project: {summary.get('project_name', config.get('project_name', 'unknown'))}",
        f"- Model: {summary.get('model_name', config.get('model', {}).get('name', 'unknown'))}",
        f"- Dataset: {summary.get('dataset', 'Oxford-IIIT Pet')}",
        f"- Image size: {summary.get('image_size', config.get('data', {}).get('image_size', 'unknown'))}",
        f"- Epochs logged: {epoch_count}",
        f"- Train / Val / Test: {summary.get('num_train_samples', 0)} / {summary.get('num_val_samples', 0)} / {summary.get('num_test_samples', 0)}",
        "",
        "## Best Metrics",
        f"- Best epoch: {summary.get('best_epoch', 'unknown')}",
        f"- Best val mIoU: {summary.get('best_val_miou', 0.0):.4f}" if "best_val_miou" in summary else "- Best val mIoU: unknown",
        f"- Best val foreground IoU: {summary.get('best_val_foreground_iou', 0.0):.4f}" if "best_val_foreground_iou" in summary else "- Best val foreground IoU: unknown",
        f"- Best val foreground Dice: {summary.get('best_val_foreground_dice', 0.0):.4f}" if "best_val_foreground_dice" in summary else "- Best val foreground Dice: unknown",
        f"- Test mIoU: {summary.get('test_miou', 0.0):.4f}" if "test_miou" in summary else "- Test mIoU: unknown",
        f"- Test foreground IoU: {summary.get('test_foreground_iou', 0.0):.4f}" if "test_foreground_iou" in summary else "- Test foreground IoU: unknown",
        f"- Test foreground Dice: {summary.get('test_foreground_dice', 0.0):.4f}" if "test_foreground_dice" in summary else "- Test foreground Dice: unknown",
        f"- Test pixel accuracy: {summary.get('test_pixel_accuracy', 0.0):.4f}" if "test_pixel_accuracy" in summary else "- Test pixel accuracy: unknown",
        "",
        "## Runtime",
        f"- Total training time (sec): {summary.get('total_training_time_sec', 'unknown')}",
        f"- Average epoch time (sec): {summary.get('average_epoch_time_sec', 'unknown')}",
        f"- Inference FPS: {summary.get('inference_fps', 'unknown')}",
        "",
        "## Figure Paths",
    ]
    lines.extend([f"- {path}" for path in figure_paths] or ["- No figures found"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
