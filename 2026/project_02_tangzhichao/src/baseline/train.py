import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from tqdm import tqdm

from .config import load_config
from .dataset import build_dataloaders, build_visualization_dataset
from .evaluate import evaluate_loader
from .model import build_image_processor, build_model, load_model, save_model
from .utils import (
    count_parameters,
    ensure_dir,
    format_seconds,
    get_device,
    get_gpu_memory,
    log,
    save_csv,
    save_json,
    set_seed,
    to_serializable,
)
from .visualize import (
    plot_confusion_matrix,
    plot_training_curves,
    visualize_dataset_samples,
    visualize_failure_cases,
    visualize_predictions,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None


class NoOpSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


def save_checkpoint(state, path):
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)


def build_class_weights(config, device):
    loss_cfg = config.get("loss", {})
    if not loss_cfg.get("use_class_weights", False):
        return None
    weights = loss_cfg.get("class_weights")
    if weights is None:
        background_weight = float(loss_cfg.get("background_weight", 0.25))
        foreground_weight = float(loss_cfg.get("foreground_weight", 2.0))
        weights = [background_weight] + [foreground_weight] * (config["model"]["num_labels"] - 1)
    if len(weights) != config["model"]["num_labels"]:
        raise ValueError(
            f"class_weights length {len(weights)} does not match num_labels={config['model']['num_labels']}"
        )
    return torch.tensor(weights, dtype=torch.float32, device=device)


def segmentation_loss(outputs, labels, class_weights, ignore_index):
    logits = F.interpolate(
        outputs.logits,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    ce_loss = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=ignore_index)
    return ce_loss


def multiclass_dice_loss(logits, labels, num_classes, ignore_index, smooth=1.0):
    probs = F.softmax(logits, dim=1)
    valid = labels != ignore_index
    labels = labels.clamp(min=0)
    one_hot = F.one_hot(labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1).float()
    probs = probs * valid
    one_hot = one_hot * valid

    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    cardinality = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


def combined_segmentation_loss(outputs, labels, class_weights, ignore_index, config):
    logits = F.interpolate(
        outputs.logits,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    loss_cfg = config.get("loss", {})
    ce_weight = float(loss_cfg.get("ce_weight", 1.0))
    dice_weight = float(loss_cfg.get("dice_weight", 0.0))
    ce = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=ignore_index)
    if dice_weight <= 0:
        return ce
    dice = multiclass_dice_loss(
        logits=logits,
        labels=labels,
        num_classes=config["model"]["num_labels"],
        ignore_index=ignore_index,
    )
    return ce_weight * ce + dice_weight * dice


def train_one_epoch(model, loader, optimizer, device, scaler, amp_enabled, grad_clip_norm, class_weights, ignore_index, config):
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(loader, desc="Training", leave=False)
    for batch in progress:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = combined_segmentation_loss(outputs, labels, class_weights, ignore_index, config)

        if torch.isnan(loss):
            raise RuntimeError("NaN loss encountered during training.")

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        batch_size = pixel_values.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_samples, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = get_device(config["train"]["device"])
    amp_enabled = bool(config["train"]["amp"] and device.type == "cuda")

    output_root = Path(config["output"]["root"])
    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    logs_dir = ensure_dir(output_root / "logs")
    figures_dir = ensure_dir(output_root / "figures")
    tensorboard_dir = ensure_dir(logs_dir / "tensorboard")
    eval_dir = ensure_dir(output_root / "eval")

    with (output_root / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    image_processor = build_image_processor(config)
    train_loader, val_loader, test_loader = build_dataloaders(config, image_processor)
    model = build_model(config).to(device)
    class_weights = build_class_weights(config, device)
    if class_weights is not None:
        log(f"Using class weights: {[round(x, 4) for x in class_weights.detach().cpu().tolist()]}")
    optimizer = AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    writer = SummaryWriter(log_dir=str(tensorboard_dir)) if SummaryWriter is not None else NoOpSummaryWriter()

    param_stats = count_parameters(model)
    log(f"Model parameters: {param_stats}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    viz_train_dataset = build_visualization_dataset(config, image_processor, split=config["data"]["train_split"])
    visualize_dataset_samples(viz_train_dataset, figures_dir / "dataset_samples", num_samples=8)

    history = []
    best_metric_name = config["output"]["save_best_metric"]
    best_metric_value = float("-inf")
    best_epoch = 0
    total_training_time = 0.0

    for epoch in range(1, config["train"]["epochs"] + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            grad_clip_norm=config["train"]["grad_clip_norm"],
            class_weights=class_weights,
            ignore_index=config["data"]["ignore_index"],
            config=config,
        )

        val_metrics = evaluate_loader(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=config["model"]["num_labels"],
            ignore_index=config["data"]["ignore_index"],
            class_weights=class_weights,
        )
        epoch_time = time.perf_counter() - epoch_start
        total_training_time += epoch_time
        memory_stats = get_gpu_memory()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_miou": val_metrics["miou"],
            "val_foreground_iou": val_metrics["foreground_iou"],
            "val_foreground_dice": val_metrics["foreground_dice"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
            "gpu_memory_allocated": memory_stats["gpu_memory_allocated_mb"],
            "gpu_memory_reserved": memory_stats["gpu_memory_reserved_mb"],
            "max_gpu_memory_allocated": memory_stats["max_gpu_memory_allocated_mb"],
        }
        history.append(row)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/miou", val_metrics["miou"], epoch)
        writer.add_scalar("val/foreground_dice", val_metrics["foreground_dice"], epoch)
        writer.add_scalar("val/pixel_accuracy", val_metrics["pixel_accuracy"], epoch)
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)

        current_metric = val_metrics[best_metric_name]
        if current_metric > best_metric_value:
            best_metric_value = current_metric
            best_epoch = epoch
            save_model(model, image_processor, output_root / "best_model")
            save_json(val_metrics, eval_dir / "val_metrics.json")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "history": history,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "best_epoch": best_epoch,
        }
        save_checkpoint(checkpoint, checkpoints_dir / "last_checkpoint.pth")
        if epoch % config["train"]["save_every"] == 0:
            save_checkpoint(checkpoint, checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pth")

        save_csv(history, logs_dir / "train_log.csv")
        save_json({"history": to_serializable(history)}, logs_dir / "train_log.json")

        log(
            "Epoch "
            f"{epoch}/{config['train']['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_miou={val_metrics['miou']:.4f} | "
            f"val_fg_dice={val_metrics['foreground_dice']:.4f} | "
            f"time={format_seconds(epoch_time)}"
        )

    writer.close()
    plot_training_curves(logs_dir / "train_log.csv", figures_dir / "curves")

    best_model_dir = output_root / "best_model"
    best_model, best_processor = load_model(best_model_dir, device)
    test_metrics = evaluate_loader(
        model=best_model,
        loader=test_loader,
        device=device,
        num_classes=config["model"]["num_labels"],
        ignore_index=config["data"]["ignore_index"],
        class_weights=class_weights,
    )
    save_json(test_metrics, eval_dir / "test_metrics.json")
    save_csv([{k: v for k, v in test_metrics.items() if k != "confusion_matrix"}], eval_dir / "test_metrics.csv")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names=[config["model"]["id2label"][idx] for idx in range(config["model"]["num_labels"])],
        save_path=figures_dir / "confusion_matrix.png",
    )

    test_viz_dataset = build_visualization_dataset(config, best_processor, split=config["data"]["test_split"])
    predictions_info = visualize_predictions(
        best_model,
        best_processor,
        test_viz_dataset,
        device,
        figures_dir / "predictions",
        num_samples=12,
    )
    visualize_failure_cases(predictions_info, figures_dir / "failure_cases", top_k=8)

    summary = {
        "project_name": config["project_name"],
        "model_name": config["model"]["name"],
        "dataset": config["data"]["dataset"],
        "task": "ImageNet-S semantic segmentation" if config["data"]["dataset"] == "imagenet_s" else "binary pet segmentation",
        "image_size": config["data"]["image_size"],
        "mask_mode": config["data"].get("mask_mode", "imagenet_s_multiclass"),
        "num_labels": config["model"]["num_labels"],
        "selected_classes": config["data"].get("selected_classes", []),
        "ignore_index": config["data"]["ignore_index"],
        "num_train_samples": len(train_loader.dataset),
        "num_val_samples": len(val_loader.dataset),
        "num_test_samples": len(test_loader.dataset),
        **param_stats,
        "best_epoch": best_epoch,
        "best_val_miou": best_metric_value if best_metric_name == "miou" else max(row["val_miou"] for row in history),
        "best_val_foreground_iou": max(row["val_foreground_iou"] for row in history),
        "best_val_foreground_dice": max(row["val_foreground_dice"] for row in history),
        "test_mean_foreground_iou": test_metrics.get("mean_foreground_iou", 0.0),
        "test_mean_foreground_dice": test_metrics.get("mean_foreground_dice", 0.0),
        "test_miou": test_metrics["miou"],
        "test_foreground_iou": test_metrics["foreground_iou"],
        "test_foreground_dice": test_metrics["foreground_dice"],
        "test_pixel_accuracy": test_metrics["pixel_accuracy"],
        "total_training_time_sec": total_training_time,
        "average_epoch_time_sec": total_training_time / max(len(history), 1),
        "inference_fps": test_metrics["inference_fps"],
        "max_gpu_memory_allocated_mb": max((row["max_gpu_memory_allocated"] for row in history), default=0.0),
    }
    save_json(summary, output_root / "summary.json")

    log("Generating report_assets.md")
    from .report_assets import main as report_assets_main
    import sys

    old_argv = sys.argv[:]
    sys.argv = ["report_assets", "--output_dir", str(output_root)]
    try:
        report_assets_main()
    finally:
        sys.argv = old_argv

    log(f"Training completed. Outputs saved to {output_root}")


if __name__ == "__main__":
    main()
