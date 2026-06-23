from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .utils import ensure_dir


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [20, 20, 20],
            [255, 140, 0],
            [0, 170, 255],
            [46, 204, 113],
            [155, 89, 182],
            [231, 76, 60],
            [241, 196, 15],
            [26, 188, 156],
            [52, 152, 219],
            [230, 126, 34],
            [149, 165, 166],
        ],
        dtype=np.uint8,
    )
    safe_mask = np.asarray(mask, dtype=np.int64)
    rgb = palette[safe_mask % len(palette)]
    rgb[mask == 255] = [180, 180, 180]
    return rgb


def _overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    mask_rgb = _mask_to_rgb(mask)
    overlay = image.astype(np.float32) * (1 - alpha) + mask_rgb.astype(np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def visualize_dataset_samples(dataset, save_dir, num_samples=8):
    save_dir = ensure_dir(save_dir)
    count = min(num_samples, len(dataset))
    fig, axes = plt.subplots(count, 2, figsize=(8, 4 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)
    for row in range(count):
        sample = dataset[row]
        axes[row, 0].imshow(sample["original_image"])
        axes[row, 0].set_title(f"Image {sample['image_id']}")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(_mask_to_rgb(sample["original_mask"]))
        axes[row, 1].set_title("Converted Mask")
        axes[row, 1].axis("off")
    fig.tight_layout()
    fig.savefig(Path(save_dir) / "sample_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(log_csv, save_dir):
    save_dir = ensure_dir(save_dir)
    df = pd.read_csv(log_csv)
    curve_specs = [
        ("train_loss", "Train Loss", "loss_curve.png"),
        ("val_miou", "Validation mIoU", "miou_curve.png"),
        ("val_foreground_dice", "Validation Foreground Dice", "dice_curve.png"),
        ("val_pixel_accuracy", "Validation Pixel Accuracy", "pixel_acc_curve.png"),
        ("learning_rate", "Learning Rate", "learning_rate_curve.png"),
    ]
    for column, title, filename in curve_specs:
        if column not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["epoch"], df[column], marker="o")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(column)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(Path(save_dir) / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def visualize_predictions(model, image_processor, dataset, device, save_dir, num_samples=12):
    save_dir = ensure_dir(save_dir)
    prediction_infos = []
    chosen = min(num_samples, len(dataset))
    grid_rows = []

    model.eval()
    for idx in range(chosen):
        sample = dataset[idx]
        pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(pixel_values=pixel_values).logits
            logits = F.interpolate(
                logits,
                size=sample["labels"].shape,
                mode="bilinear",
                align_corners=False,
            )
            pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        image = sample["original_image"]
        gt_mask = sample["original_mask"]
        overlay = _overlay_mask(image, pred_mask)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        titles = ["Original", "Ground Truth", "Prediction", "Overlay"]
        views = [image, _mask_to_rgb(gt_mask), _mask_to_rgb(pred_mask), overlay]
        for ax, title, view in zip(axes, titles, views):
            ax.imshow(view)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(Path(save_dir) / f"pred_{idx:03d}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        valid = gt_mask != 255
        intersection = np.logical_and(pred_mask == 1, gt_mask == 1)
        intersection = np.logical_and(intersection, valid).sum()
        union = np.logical_or(pred_mask == 1, gt_mask == 1)
        union = np.logical_and(union, valid).sum()
        fg_iou = float(intersection / union) if union > 0 else 0.0

        prediction_infos.append(
            {
                "image_id": sample["image_id"],
                "original_image": image,
                "ground_truth": gt_mask,
                "prediction": pred_mask,
                "foreground_iou": fg_iou,
            }
        )
        grid_rows.append((image, _mask_to_rgb(gt_mask), _mask_to_rgb(pred_mask), overlay))

    if grid_rows:
        rows = len(grid_rows)
        fig, axes = plt.subplots(rows, 4, figsize=(14, 4 * rows))
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)
        for row_idx, row in enumerate(grid_rows):
            for col_idx, view in enumerate(row):
                axes[row_idx, col_idx].imshow(view)
                axes[row_idx, col_idx].axis("off")
        fig.tight_layout()
        fig.savefig(Path(save_dir) / "pred_grid.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    return prediction_infos


def visualize_failure_cases(predictions_info, save_dir, top_k=8):
    save_dir = ensure_dir(save_dir)
    ranked = sorted(predictions_info, key=lambda item: item["foreground_iou"])[:top_k]
    grid_rows = []
    for idx, item in enumerate(ranked):
        gt = item["ground_truth"]
        pred = item["prediction"]
        image = item["original_image"]
        error_map = np.zeros((*gt.shape, 3), dtype=np.uint8)
        valid = gt != 255
        true_positive = np.logical_and(pred == 1, gt == 1) & valid
        false_positive = np.logical_and(pred == 1, gt == 0) & valid
        false_negative = np.logical_and(pred == 0, gt == 1) & valid
        error_map[true_positive] = [0, 200, 0]
        error_map[false_positive] = [255, 0, 0]
        error_map[false_negative] = [255, 255, 0]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        views = [image, _mask_to_rgb(gt), _mask_to_rgb(pred), error_map]
        titles = ["Original", "Ground Truth", "Prediction", "Error Map"]
        for ax, title, view in zip(axes, titles, views):
            ax.imshow(view)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(Path(save_dir) / f"failure_{idx:03d}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        grid_rows.append(views)

    if grid_rows:
        rows = len(grid_rows)
        fig, axes = plt.subplots(rows, 4, figsize=(14, 4 * rows))
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)
        for row_idx, row in enumerate(grid_rows):
            for col_idx, view in enumerate(row):
                axes[row_idx, col_idx].imshow(view)
                axes[row_idx, col_idx].axis("off")
        fig.tight_layout()
        fig.savefig(Path(save_dir) / "failure_grid.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_confusion_matrix(confusion_matrix, class_names, save_path):
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    cm = np.asarray(confusion_matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
