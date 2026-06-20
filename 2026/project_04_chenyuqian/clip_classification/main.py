"""CLIP Zero-Shot CIFAR-10 Classification Experiment.

Pipeline (no training involved):
  CIFAR-10 image → CLIP image encoder → image feature vector (L2 norm)
                                                                      → cosine similarity → argmax → prediction
  class prompt text → CLIP text encoder → text feature vector (L2 norm)

Key facts:
  - No training / fine-tuning
  - CIFAR-10 training set is never used
  - Only test set is used for evaluation
  - Classification is based on semantic similarity in shared latent space
"""

import os
import torch
from tqdm import tqdm

from data_loader import get_test_loader, CIFAR10_CLASSES
from prompt_templates import (
    SINGLE_TEMPLATE,
    MULTI_TEMPLATES,
    generate_prompts,
)
from model import CLIPZeroShotClassifier
from evaluate import evaluate


def save_misclassified_images(data_loader, predictions, labels, class_names, output_dir):
    """Save misclassified example images to disk."""
    misclass_dir = os.path.join(output_dir, "misclassified_examples")
    os.makedirs(misclass_dir, exist_ok=True)

    # Map sample index to image via second pass through the loader
    all_images = []
    for images, _ in data_loader:
        all_images.extend(images)

    saved = 0
    max_save = 20
    for i in range(len(labels)):
        if predictions[i] != labels[i] and saved < max_save:
            img = all_images[i]
            true_name = class_names[labels[i]]
            pred_name = class_names[predictions[i]]
            fname = f"idx={i}_true={true_name}_pred={pred_name}.png"
            img.save(os.path.join(misclass_dir, fname))
            saved += 1

    print(f"Saved {saved} misclassified examples to: {misclass_dir}")


def run_experiment(classifier, data_loader, class_names, templates, use_synonyms, desc):
    """Run a single zero-shot classification experiment.

    Args:
        classifier: CLIPZeroShotClassifier instance.
        data_loader: DataLoader yielding (list[PIL.Image], tensor_labels) batches.
        class_names: list of class name strings.
        templates: list of template strings.
        use_synonyms: bool.
        desc: experiment label.

    Returns:
        accuracy (float).
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {desc}")
    print(f"{'='*60}")

    # Build text features from prompts
    prompts = generate_prompts(class_names, templates, use_synonyms=use_synonyms)
    ensembling = len(templates) > 1
    text_features = classifier.build_text_features(class_names, prompts, ensembling=ensembling)

    # Sanity check: text features shape
    assert text_features.shape == (len(class_names), classifier.feature_dim), \
        f"Expected ({len(class_names)}, {classifier.feature_dim}), got {text_features.shape}"
    print(f"Text features shape: {text_features.shape}")

    # Predict over the entire dataset
    all_preds = []
    all_labels = []
    for images, labels in tqdm(data_loader, desc="Predicting"):
        preds = classifier.predict(images, text_features)
        all_preds.append(preds.cpu())
        all_labels.append(labels)

        # Sanity check per batch
        assert preds.shape == (len(images),), \
            f"Expected preds shape ({len(images)},), got {preds.shape}"

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Sanity check: total samples
    assert len(all_labels) == 10000, \
        f"Expected 10000 test samples, got {len(all_labels)}"

    # Evaluate
    acc, _ = evaluate(all_preds, all_labels, class_names, output_dir=".")

    return acc


def main():
    print("=" * 60)
    print("CLIP Zero-Shot CIFAR-10 Classification")
    print("=" * 60)
    print()
    print("Core pipeline:")
    print("  CIFAR-10 image → CLIP image encoder → image feature (L2 norm)")
    print("  class prompt text → CLIP text encoder → text feature (L2 norm)")
    print("  → cosine similarity → argmax → zero-shot prediction")
    print()
    print("Note: No training. No CIFAR-10 train set. Only test set for eval.")
    print()

    # Load model
    classifier = CLIPZeroShotClassifier()

    # Load CIFAR-10 test set
    data_loader, class_names = get_test_loader(batch_size=64)
    print(f"Test set size: {len(data_loader.dataset)}")

    results = {}

    # Experiment 1: Single Prompt
    results["Single Prompt"] = run_experiment(
        classifier, data_loader, class_names,
        templates=SINGLE_TEMPLATE,
        use_synonyms=False,
        desc="Single Prompt (a photo of a {})",
    )

    # Experiment 2: Multi-template Ensemble
    results["Multi-template Ensemble"] = run_experiment(
        classifier, data_loader, class_names,
        templates=MULTI_TEMPLATES,
        use_synonyms=False,
        desc="Multi-template Ensemble (8 prompts)",
    )

    # Comparison table
    print(f"\n{'='*60}")
    print("Comparison")
    print(f"{'='*60}")
    print(f"{'Experiment':<30s} {'Accuracy':>10s}")
    print("-" * 42)
    for name, acc in results.items():
        print(f"{name:<30s} {acc*100:>9.2f}%")
    print(f"{'='*60}")

    # Save misclassified examples from the better experiment
    best_exp = max(results, key=results.get)
    print(f"\nBest experiment: {best_exp} ({results[best_exp]*100:.2f}%)")

    print("Saving misclassified examples from best experiment...")
    templates = MULTI_TEMPLATES if best_exp == "Multi-template Ensemble" else SINGLE_TEMPLATE
    prompts = generate_prompts(class_names, templates, use_synonyms=False)
    text_features = classifier.build_text_features(
        class_names, prompts, ensembling=len(templates) > 1
    )
    all_preds = []
    all_labels = []
    for images, labels in tqdm(data_loader, desc="Final prediction"):
        preds = classifier.predict(images, text_features)
        all_preds.append(preds.cpu())
        all_labels.append(labels)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    save_misclassified_images(
        data_loader, all_preds, all_labels, class_names, output_dir="."
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
