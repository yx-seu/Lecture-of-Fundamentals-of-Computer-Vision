import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from infer import IMAGENETTE_LABELS, load_model_from_checkpoint
from vit import create_imagefolder_loaders


def compute_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for label, prediction in zip(labels.view(-1), predictions.view(-1)):
        matrix[int(label), int(prediction)] += 1
    return matrix


def per_class_accuracy(confusion_matrix: torch.Tensor) -> List[float]:
    totals = confusion_matrix.sum(dim=1)
    correct = confusion_matrix.diag()
    accuracies = []
    for class_correct, class_total in zip(correct.tolist(), totals.tolist()):
        accuracies.append(0.0 if class_total == 0 else class_correct / class_total)
    return accuracies


def read_metrics_csv(csv_path: str) -> List[Dict[str, float]]:
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "lr": float(row["lr"]),
                    "train_loss": float(row["train_loss"]),
                    "train_acc": float(row["train_acc"]),
                    "val_loss": float(row["val_loss"]),
                    "val_acc": float(row["val_acc"]),
                }
            )
    return rows


def save_training_curves(metrics_rows: List[Dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in metrics_rows]
    train_loss = [row["train_loss"] for row in metrics_rows]
    val_loss = [row["val_loss"] for row in metrics_rows]
    train_acc = [row["train_acc"] for row in metrics_rows]
    val_acc = [row["val_acc"] for row in metrics_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_acc, marker="o", label="Train Acc")
    axes[1].plot(epochs, val_acc, marker="o", label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_confusion_matrix_plot(
    matrix: torch.Tensor,
    class_labels: List[str],
    output_path: Path,
) -> None:
    matrix_np = matrix.numpy()
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix_np, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Validation Confusion Matrix")

    threshold = matrix_np.max() / 2 if matrix_np.max() else 0
    for row in range(matrix_np.shape[0]):
        for col in range(matrix_np.shape[1]):
            value = matrix_np[row, col]
            color = "white" if value > threshold else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_per_class_accuracy(
    class_names: List[str],
    accuracies: List[float],
    csv_path: Path,
    plot_path: Path,
) -> None:
    rows = []
    for class_id, accuracy in zip(class_names, accuracies):
        rows.append(
            {
                "class_id": class_id,
                "label": IMAGENETTE_LABELS.get(class_id, class_id),
                "accuracy": accuracy,
                "accuracy_percent": accuracy * 100.0,
            }
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_id", "label", "accuracy", "accuracy_percent"],
        )
        writer.writeheader()
        writer.writerows(rows)

    labels = [f"{row['class_id']}\n{row['label']}" for row in rows]
    values = [row["accuracy_percent"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color="#4f81bd")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Validation Accuracy")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str,
    data_root: str,
    output_dir: str,
    metrics_csv: str = "",
    batch_size: int = 64,
    num_workers: int = 2,
    device_name: str = "",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))

    model, class_names, model_args = load_model_from_checkpoint(checkpoint_path, device)
    _, val_loader, loader_class_names = create_imagefolder_loaders(
        data_root=data_root,
        img_size=model_args["img_size"],
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if class_names != loader_class_names:
        raise ValueError("checkpoint 类别顺序和验证集类别顺序不一致")

    all_predictions = []
    all_labels = []
    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_predictions.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    matrix = compute_confusion_matrix(predictions, labels, len(class_names))
    accuracies = per_class_accuracy(matrix)

    class_labels = [IMAGENETTE_LABELS.get(class_id, class_id) for class_id in class_names]
    torch.save(matrix, output_path / "confusion_matrix.pt")
    np.savetxt(output_path / "confusion_matrix.csv", matrix.numpy(), delimiter=",", fmt="%d")
    save_confusion_matrix_plot(matrix, class_labels, output_path / "confusion_matrix.png")
    save_per_class_accuracy(
        class_names,
        accuracies,
        output_path / "per_class_accuracy.csv",
        output_path / "per_class_accuracy.png",
    )

    if metrics_csv:
        metrics_rows = read_metrics_csv(metrics_csv)
        save_training_curves(metrics_rows, output_path / "training_curves.png")

    overall_acc = (predictions == labels).float().mean().item() * 100.0
    with open(output_path / "evaluation_summary.txt", "w", encoding="utf-8") as handle:
        handle.write(f"checkpoint={checkpoint_path}\n")
        handle.write(f"data_root={data_root}\n")
        handle.write(f"samples={len(labels)}\n")
        handle.write(f"overall_accuracy={overall_acc:.2f}\n")
        handle.write(f"mean_per_class_accuracy={sum(accuracies) / len(accuracies) * 100.0:.2f}\n")

    print(f"Overall accuracy: {overall_acc:.2f}%")
    print(f"Saved report files to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 ViT 验证集评估图表")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/imagenette2-160")
    parser.add_argument("--output-dir", default="reports/final_eval")
    parser.add_argument("--metrics-csv", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        metrics_csv=args.metrics_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
