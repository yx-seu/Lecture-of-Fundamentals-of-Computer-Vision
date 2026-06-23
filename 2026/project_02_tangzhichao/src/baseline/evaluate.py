import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import load_config
from .dataset import build_dataloaders
from .metrics import SegmentationMetric
from .model import load_model
from .utils import get_device, log, save_csv, save_json
from .visualize import plot_confusion_matrix


def evaluate_loader(model, loader, device, num_classes, ignore_index, class_weights=None):
    metric = SegmentationMetric(num_classes=num_classes, ignore_index=ignore_index)
    total_loss = 0.0
    total_samples = 0
    total_time = 0.0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            start = time.perf_counter()
            outputs = model(pixel_values=pixel_values, labels=labels)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - start

            logits = F.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = outputs.loss
            if class_weights is not None:
                loss = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=ignore_index)
            preds = logits.argmax(dim=1)
            metric.update(preds.cpu().numpy(), labels.cpu().numpy())

            batch_size = pixel_values.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    metrics = metric.compute()
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["inference_fps"] = total_samples / total_time if total_time > 0 else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["train"]["device"])
    model, image_processor = load_model(args.model_dir, device)
    train_loader, val_loader, test_loader = build_dataloaders(config, image_processor)
    split_to_loader = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = split_to_loader[args.split]

    metrics = evaluate_loader(
        model=model,
        loader=loader,
        device=device,
        num_classes=config["model"]["num_labels"],
        ignore_index=config["data"]["ignore_index"],
    )

    output_root = Path(config["output"]["root"])
    eval_dir = output_root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_path = eval_dir / f"{args.split}_metrics.json"
    metrics_csv_path = eval_dir / f"{args.split}_metrics.csv"
    save_json(metrics, metrics_json_path)
    csv_row = [{k: v for k, v in metrics.items() if k != "confusion_matrix"}]
    save_csv(csv_row, metrics_csv_path)

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=[config["model"]["id2label"][idx] for idx in range(config["model"]["num_labels"])],
        save_path=output_root / "figures" / "confusion_matrix.png",
    )
    log(f"Saved evaluation results to {eval_dir}")


if __name__ == "__main__":
    main()
