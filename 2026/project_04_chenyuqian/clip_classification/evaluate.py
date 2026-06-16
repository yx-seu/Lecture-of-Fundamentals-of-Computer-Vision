"""Evaluation: accuracy, per-class metrics, confusion matrix, misclassified examples."""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate(predictions, labels, class_names, output_dir="."):
    """Run full evaluation and save artifacts.

    Args:
        predictions: (N,) array-like of predicted class indices.
        labels: (N,) array-like of ground-truth class indices.
        class_names: list of class name strings.
        output_dir: directory for saved artifacts.
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    # 1. Overall accuracy
    acc = accuracy_score(labels, predictions)
    print(f"\n{'='*50}")
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"{'='*50}\n")

    # Safety check
    if acc < 0.50:
        print(
            "[WARNING] Accuracy is below 50%. Possible issues: preprocessing "
            "normalization applied twice, L2 normalization missing, or prompt "
            "templates not matching the dataset domain."
        )

    # 2. Per-class accuracy
    print("Per-class Accuracy:")
    print("-" * 30)
    for i, name in enumerate(class_names):
        mask = labels == i
        if mask.sum() > 0:
            cls_acc = accuracy_score(labels[mask], predictions[mask])
            print(f"  {name:>12s}: {cls_acc:.4f} ({cls_acc*100:.1f}%)")
    print()

    # 3. Classification report (precision, recall, F1)
    print("Classification Report:")
    print(classification_report(
        labels, predictions, target_names=class_names, digits=4, zero_division=0
    ))

    # 4. Confusion matrix
    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("CIFAR-10 Zero-Shot Classification Confusion Matrix")
    plt.colorbar(im, ax=ax)

    # Annotate with counts
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                fontsize=8,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to: {cm_path}")

    # 5. Save misclassified examples
    misclass_dir = os.path.join(output_dir, "misclassified_examples")
    os.makedirs(misclass_dir, exist_ok=True)
    misclass_mask = predictions != labels
    misclass_indices = np.where(misclass_mask)[0]
    print(f"\nTotal misclassified: {len(misclass_indices)} / {len(labels)}")

    # Save up to 20 misclassified indices for reference
    # (actual images are saved from main.py which has access to the dataset)
    np.savez(
        os.path.join(misclass_dir, "misclassified_indices.npz"),
        indices=misclass_indices,
        true_labels=labels[misclass_mask],
        pred_labels=predictions[misclass_mask],
    )
    print(f"Misclassified indices saved to: {misclass_dir}")

    return acc, cm
