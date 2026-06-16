"""Cat-centered CLIP latent-space analysis.

This script extends the CIFAR-10 zero-shot experiment from "classification"
to "semantic alignment" analysis. It keeps CLIP frozen and uses only forward
passes through the image/text encoders.
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from tqdm import tqdm

from data_loader import CIFAR10_CLASSES, get_test_loader
from model import CLIPZeroShotClassifier
from prompt_templates import (
    CAT_FINE_GRAINED_PROMPTS,
    CAT_MISLEADING_PROMPTS,
    MULTI_TEMPLATES,
    generate_prompts,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_image_paths(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_pil_images(paths):
    images = []
    valid_paths = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
        except OSError:
            continue
        images.append(img)
        valid_paths.append(path)
    return images, valid_paths


def encode_cifar10(classifier, data_loader):
    image_features = []
    labels = []
    images = []

    for batch_images, batch_labels in tqdm(data_loader, desc="Encoding CIFAR-10 test"):
        feats = classifier.encode_images(batch_images)
        image_features.append(feats.cpu())
        labels.append(batch_labels.cpu())
        images.extend(batch_images)

    return torch.cat(image_features), torch.cat(labels), images


def make_text_features(classifier, texts):
    features = classifier.encode_texts(texts).cpu()
    norms = features.norm(dim=-1).numpy()
    return features, norms


def make_cifar_text_features(classifier, class_names):
    prompts = generate_prompts(class_names, MULTI_TEMPLATES, use_synonyms=False)
    return classifier.build_text_features(class_names, prompts, ensembling=True).cpu()


def save_montage(rows, images, class_names, out_path, max_images=30, thumb_size=96):
    selected = rows[:max_images]
    if not selected:
        return

    cols = 5
    rows_count = int(np.ceil(len(selected) / cols))
    label_h = 34
    canvas = Image.new(
        "RGB",
        (cols * thumb_size, rows_count * (thumb_size + label_h)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for k, row in enumerate(selected):
        img = images[row["dataset_index"]].resize((thumb_size, thumb_size))
        x = (k % cols) * thumb_size
        y = (k // cols) * (thumb_size + label_h)
        canvas.paste(img, (x, y))
        label = f"#{row['rank']} {row['true_label']}->{row['predicted_label']}"
        sim = f"cat={float(row['cat_similarity']):.3f}"
        draw.text((x + 3, y + thumb_size + 2), label[:18], fill=(0, 0, 0))
        draw.text((x + 3, y + thumb_size + 17), sim, fill=(0, 0, 0))

    canvas.save(out_path)


def semantic_neighborhood(
    output_dir,
    image_features,
    labels,
    images,
    class_names,
    cifar_text_features,
    cat_text_feature,
    top_k,
):
    cat_idx = class_names.index("cat")
    dog_idx = class_names.index("dog")

    similarities = image_features @ cifar_text_features.T
    predictions = similarities.argmax(dim=-1)
    cat_sim = image_features @ cat_text_feature[0]
    dog_sim = similarities[:, dog_idx]

    order = torch.argsort(cat_sim, descending=True).tolist()
    rows = []
    for rank, idx in enumerate(order, start=1):
        rows.append({
            "rank": rank,
            "dataset_index": idx,
            "true_label": class_names[int(labels[idx])],
            "predicted_label": class_names[int(predictions[idx])],
            "cat_similarity": f"{float(cat_sim[idx]):.6f}",
            "dog_similarity": f"{float(dog_sim[idx]):.6f}",
            "cat_dog_margin": f"{float(cat_sim[idx] - dog_sim[idx]):.6f}",
            "is_true_cat": bool(int(labels[idx]) == cat_idx),
        })

    write_csv(
        output_dir / "cat_similarity_ranking.csv",
        rows,
        [
            "rank", "dataset_index", "true_label", "predicted_label",
            "cat_similarity", "dog_similarity", "cat_dog_margin", "is_true_cat",
        ],
    )

    non_cat_rows = [r for r in rows if not r["is_true_cat"]]
    write_csv(
        output_dir / "cat_nearest_non_cat.csv",
        non_cat_rows[:top_k],
        [
            "rank", "dataset_index", "true_label", "predicted_label",
            "cat_similarity", "dog_similarity", "cat_dog_margin", "is_true_cat",
        ],
    )
    save_montage(non_cat_rows, images, class_names, output_dir / "cat_nearest_non_cat_montage.png")

    top_non_cat_counter = Counter(r["true_label"] for r in non_cat_rows[:top_k])
    class_summary = []
    for class_idx, name in enumerate(class_names):
        mask = labels == class_idx
        vals = cat_sim[mask].numpy()
        class_summary.append({
            "class": name,
            "mean_cat_similarity": float(vals.mean()),
            "std_cat_similarity": float(vals.std()),
            "max_cat_similarity": float(vals.max()),
            "top_non_cat_count": int(top_non_cat_counter.get(name, 0)),
        })

    write_csv(
        output_dir / "cat_similarity_by_true_class.csv",
        class_summary,
        ["class", "mean_cat_similarity", "std_cat_similarity", "max_cat_similarity", "top_non_cat_count"],
    )

    return {
        "top_k_non_cat": top_k,
        "top_non_cat_counts": dict(top_non_cat_counter),
        "true_cat_recall_under_cifar_prompts": float(((predictions == cat_idx) & (labels == cat_idx)).sum() / (labels == cat_idx).sum()),
        "cat_precision_under_cifar_prompts": float(((predictions == cat_idx) & (labels == cat_idx)).sum() / (predictions == cat_idx).sum()),
        "total_cifar_test_images": int(len(labels)),
    }


def latent_space_plot(output_dir, image_features, labels, class_names, text_features):
    target_names = ["cat", "dog", "deer", "horse"]
    target_indices = [class_names.index(name) for name in target_names]
    max_per_class = 250

    selected_indices = []
    point_labels = []
    for class_idx, name in zip(target_indices, target_names):
        idxs = torch.where(labels == class_idx)[0][:max_per_class].tolist()
        selected_indices.extend(idxs)
        point_labels.extend([name] * len(idxs))

    image_points = image_features[selected_indices].numpy()
    text_points = text_features[target_indices].numpy()
    combined = np.vstack([image_points, text_points])
    projected = PCA(n_components=2, random_state=0).fit_transform(combined)
    image_xy = projected[:len(image_points)]
    text_xy = projected[len(image_points):]

    colors = {
        "cat": "#d62728",
        "dog": "#1f77b4",
        "deer": "#2ca02c",
        "horse": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    for name in target_names:
        mask = np.array(point_labels) == name
        ax.scatter(
            image_xy[mask, 0],
            image_xy[mask, 1],
            s=12,
            alpha=0.45,
            c=colors[name],
            label=f"{name} images",
        )

    for i, name in enumerate(target_names):
        ax.scatter(
            text_xy[i, 0],
            text_xy[i, 1],
            s=220,
            marker="*",
            c=colors[name],
            edgecolor="black",
            linewidth=0.8,
            label=f"{name} text",
        )
        ax.text(text_xy[i, 0], text_xy[i, 1], f"  {name}", fontsize=11, weight="bold")

    ax.set_title("PCA projection of CLIP latent space: cat/dog/deer/horse")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "cat_dog_deer_horse_pca.png", dpi=180)
    plt.close(fig)


def fine_grained_cat_analysis(output_dir, image_features, labels, class_names, fine_features):
    cat_idx = class_names.index("cat")
    cat_mask = labels == cat_idx
    cat_images = image_features[cat_mask]
    sims = cat_images @ fine_features.T
    winners = sims.argmax(dim=-1).numpy()

    rows = []
    for i, prompt in enumerate(CAT_FINE_GRAINED_PROMPTS):
        vals = sims[:, i].numpy()
        rows.append({
            "prompt": prompt,
            "mean_similarity_on_cifar_cat": float(vals.mean()),
            "std_similarity_on_cifar_cat": float(vals.std()),
            "top1_count_on_cifar_cat": int((winners == i).sum()),
            "top1_rate_on_cifar_cat": float((winners == i).mean()),
        })

    write_csv(
        output_dir / "fine_grained_cat_prompts_on_cifar_cat.csv",
        rows,
        [
            "prompt", "mean_similarity_on_cifar_cat", "std_similarity_on_cifar_cat",
            "top1_count_on_cifar_cat", "top1_rate_on_cifar_cat",
        ],
    )
    return rows


def analyze_external_images(
    output_dir,
    group_name,
    image_dir,
    classifier,
    cifar_text_features,
    fine_features,
    class_names,
):
    paths = load_image_paths(image_dir)
    images, valid_paths = load_pil_images(paths)
    group_dir = output_dir / group_name
    ensure_dir(group_dir)

    if not images:
        summary = {
            "group": group_name,
            "image_dir": str(image_dir),
            "status": "skipped_no_images",
            "image_count": 0,
        }
        write_json(group_dir / "summary.json", summary)
        return summary

    feature_batches = []
    batch_size = 32
    for start in tqdm(range(0, len(images), batch_size), desc=f"Encoding {group_name}"):
        feature_batches.append(classifier.encode_images(images[start:start + batch_size]).cpu())
    features = torch.cat(feature_batches)

    cifar_sims = features @ cifar_text_features.T
    cifar_preds = cifar_sims.argmax(dim=-1)
    fine_sims = features @ fine_features.T
    fine_preds = fine_sims.argmax(dim=-1)

    cat_idx = class_names.index("cat")
    dog_idx = class_names.index("dog")
    rows = []
    for i, path in enumerate(valid_paths):
        sorted_cifar = torch.argsort(cifar_sims[i], descending=True).tolist()
        cat_rank = sorted_cifar.index(cat_idx) + 1
        rows.append({
            "path": str(path),
            "cifar_prediction": class_names[int(cifar_preds[i])],
            "cat_rank_among_cifar10": cat_rank,
            "cat_similarity": f"{float(cifar_sims[i, cat_idx]):.6f}",
            "dog_similarity": f"{float(cifar_sims[i, dog_idx]):.6f}",
            "cat_dog_margin": f"{float(cifar_sims[i, cat_idx] - cifar_sims[i, dog_idx]):.6f}",
            "fine_grained_prediction": CAT_FINE_GRAINED_PROMPTS[int(fine_preds[i])],
            "fine_grained_similarity": f"{float(fine_sims[i, fine_preds[i]]):.6f}",
        })

    write_csv(
        group_dir / "per_image_results.csv",
        rows,
        [
            "path", "cifar_prediction", "cat_rank_among_cifar10", "cat_similarity",
            "dog_similarity", "cat_dog_margin", "fine_grained_prediction",
            "fine_grained_similarity",
        ],
    )

    cat_top1 = sum(r["cifar_prediction"] == "cat" for r in rows)
    summary = {
        "group": group_name,
        "image_dir": str(image_dir),
        "status": "ok",
        "image_count": len(rows),
        "cifar_cat_top1_count": cat_top1,
        "cifar_cat_top1_rate": cat_top1 / len(rows),
        "fine_grained_prediction_counts": dict(Counter(r["fine_grained_prediction"] for r in rows)),
    }
    write_json(group_dir / "summary.json", summary)
    return summary


def misleading_prompt_analysis(output_dir, image_features, labels, class_names, misleading_features, cat_text_feature):
    cat_idx = class_names.index("cat")
    cat_images = image_features[labels == cat_idx]
    base_sim = cat_images @ cat_text_feature[0]
    misleading_sims = cat_images @ misleading_features.T

    rows = []
    for i, prompt in enumerate(CAT_MISLEADING_PROMPTS):
        vals = misleading_sims[:, i]
        delta = vals - base_sim
        rows.append({
            "prompt": prompt,
            "mean_similarity_on_cifar_cat": float(vals.mean()),
            "mean_delta_vs_plain_cat_prompt": float(delta.mean()),
            "fraction_higher_than_plain_cat_prompt": float((delta > 0).float().mean()),
        })

    write_csv(
        output_dir / "misleading_prompts_on_cifar_cat.csv",
        rows,
        [
            "prompt", "mean_similarity_on_cifar_cat",
            "mean_delta_vs_plain_cat_prompt", "fraction_higher_than_plain_cat_prompt",
        ],
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run cat-centered CLIP latent-space analysis.")
    parser.add_argument("--output-dir", default="outputs/cat_alignment")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--custom-cat-dir", default="data/custom_cats")
    parser.add_argument("--imagenet-cat-dir", default="data/imagenet_cats")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    log_lines = []

    def log(message):
        print(message)
        log_lines.append(message)

    log("CLIP cat latent-space semantic alignment analysis")
    log(f"model_name={args.model_name}")
    log("No training, no fine-tuning, CIFAR-10 train split is not used.")

    classifier = CLIPZeroShotClassifier(model_name=args.model_name)
    data_loader, class_names = get_test_loader(data_root=args.data_root, batch_size=args.batch_size)
    if class_names != CIFAR10_CLASSES:
        raise ValueError("Unexpected CIFAR-10 class order.")

    image_features, labels, images = encode_cifar10(classifier, data_loader)
    image_norms = image_features.norm(dim=-1).numpy()
    log(f"encoded_cifar_test_images={len(labels)}")
    log(f"image_feature_norm_min={image_norms.min():.6f}, max={image_norms.max():.6f}")

    cifar_text_features = make_cifar_text_features(classifier, class_names)
    cat_text_feature, cat_text_norms = make_text_features(classifier, ["a photo of a cat."])
    fine_features, fine_norms = make_text_features(classifier, CAT_FINE_GRAINED_PROMPTS)
    misleading_features, misleading_norms = make_text_features(classifier, CAT_MISLEADING_PROMPTS)
    log(f"cat_text_feature_norm={cat_text_norms[0]:.6f}")
    log(f"fine_prompt_norm_min={fine_norms.min():.6f}, max={fine_norms.max():.6f}")
    log(f"misleading_prompt_norm_min={misleading_norms.min():.6f}, max={misleading_norms.max():.6f}")

    semantic_summary = semantic_neighborhood(
        output_dir,
        image_features,
        labels,
        images,
        class_names,
        cifar_text_features,
        cat_text_feature,
        args.top_k,
    )
    latent_space_plot(output_dir, image_features, labels, class_names, cifar_text_features)
    fine_summary = fine_grained_cat_analysis(output_dir, image_features, labels, class_names, fine_features)
    misleading_summary = misleading_prompt_analysis(
        output_dir,
        image_features,
        labels,
        class_names,
        misleading_features,
        cat_text_feature,
    )
    custom_summary = analyze_external_images(
        output_dir,
        "custom_cats",
        args.custom_cat_dir,
        classifier,
        cifar_text_features,
        fine_features,
        class_names,
    )
    imagenet_summary = analyze_external_images(
        output_dir,
        "imagenet_cats",
        args.imagenet_cat_dir,
        classifier,
        cifar_text_features,
        fine_features,
        class_names,
    )

    summary = {
        "model_name": args.model_name,
        "strict_zero_shot_controls": [
            "model.eval() is set inside CLIPZeroShotClassifier",
            "all encoder calls use torch.no_grad()",
            "CIFAR-10 train split is not loaded",
            "external cat directories are evaluation-only inputs",
        ],
        "feature_norm_checks": {
            "image_min": float(image_norms.min()),
            "image_max": float(image_norms.max()),
            "cat_text": float(cat_text_norms[0]),
            "fine_prompt_min": float(fine_norms.min()),
            "fine_prompt_max": float(fine_norms.max()),
            "misleading_prompt_min": float(misleading_norms.min()),
            "misleading_prompt_max": float(misleading_norms.max()),
        },
        "semantic_neighborhood": semantic_summary,
        "fine_grained_cat_prompts": fine_summary,
        "misleading_prompts": misleading_summary,
        "custom_cats": custom_summary,
        "imagenet_cats": imagenet_summary,
    }
    write_json(output_dir / "summary.json", summary)

    with open(output_dir / "run_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    log(f"Saved cat alignment artifacts to {output_dir}")


if __name__ == "__main__":
    main()
