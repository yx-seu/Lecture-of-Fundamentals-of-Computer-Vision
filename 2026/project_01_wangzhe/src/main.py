"""Vision Transformer inference demo — entry point for the coursework submission.

Usage:
    python main.py                                                # run on all test images
    python main.py --image data/test_examples/example_1_tench.JPEG  # single image
    python main.py --image <path> --checkpoint <path> --topk 3     # custom
    python main.py --list-classes                                  # print class names
"""

import argparse
import sys
from pathlib import Path

import torch

# Allow running from the project root: python src/main.py
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils import IMAGENETTE_LABELS, load_model_from_checkpoint, predict_image  # noqa: E402

# ---------------------------------------------------------------------------
# Default paths (relative to project root)
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = str(_SCRIPT_DIR.parent / "outputs_v3" / "vit_imagenette10_best.pt")
DEFAULT_TEST_DIR = str(_SCRIPT_DIR.parent / "data" / "test_examples")
DEFAULT_TOP_K = 5

# ---------------------------------------------------------------------------
# Pre-selected test images (expected labels verified against the final model)
# ---------------------------------------------------------------------------
TEST_IMAGES = [
    {
        "path": "example_1_tench.JPEG",
        "expected_class": "n01440764",
        "expected_label": "tench",
    },
    {
        "path": "example_2_springer.JPEG",
        "expected_class": "n02102040",
        "expected_label": "English springer",
    },
    {
        "path": "example_3_cassette.JPEG",
        "expected_class": "n02979186",
        "expected_label": "cassette player",
    },
]


def _resolve_checkpoint(path: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def run_single_inference(
    image_path: str,
    checkpoint_path: str,
    device: torch.device,
    topk: int = DEFAULT_TOP_K,
) -> None:
    """Load model and print top-k predictions for one image."""
    model, class_names, model_args = load_model_from_checkpoint(checkpoint_path, device)
    predictions = predict_image(
        model, image_path, class_names, model_args["img_size"], device, topk=topk,
    )

    print(f"Image:    {image_path}")
    print(f"Model:    {checkpoint_path}")
    print(f"Top-{topk} predictions:")
    for rank, pred in enumerate(predictions, start=1):
        print(f"  {rank}. {pred['class_id']} ({pred['label']}): {pred['probability'] * 100:.2f}%")
    print()


def run_demo(
    checkpoint_path: str,
    test_dir: str,
    device: torch.device,
    topk: int = DEFAULT_TOP_K,
) -> None:
    """Run inference on the three bundled test images and print a summary."""
    print("=" * 68)
    print("Vision Transformer — Imagenette-10 Inference Demo")
    print("=" * 68)
    print()

    model, class_names, model_args = load_model_from_checkpoint(checkpoint_path, device)
    print(f"Model loaded: {Path(checkpoint_path).name}")
    print(f"Classes: {len(class_names)}")
    print(f"Device: {device}")
    print()

    correct = 0
    total = 0

    for item in TEST_IMAGES:
        image_path = str(Path(test_dir) / item["path"])
        if not Path(image_path).is_file():
            print(f"[SKIP] {item['path']} — file not found")
            continue

        predictions = predict_image(
            model, image_path, class_names, model_args["img_size"], device, topk=topk,
        )
        top1 = predictions[0]
        is_correct = top1["class_id"] == item["expected_class"]
        if is_correct:
            correct += 1
        total += 1

        print(f"--- {item['path']} ---")
        print(f"  Expected: {item['expected_class']} ({item['expected_label']})")
        print(f"  Predicted (top-{topk}):")
        for rank, pred in enumerate(predictions, start=1):
            marker = " <== correct" if pred["class_id"] == item["expected_class"] else ""
            print(f"    {rank}. {pred['class_id']} ({pred['label']}): {pred['probability'] * 100:.2f}%{marker}")
        print()

    print(f"Summary: {correct}/{total} test images correctly classified")
    print()


def list_classes() -> None:
    print("Imagenette-10 classes (WordNet ID → label):")
    for wnid, label in sorted(IMAGENETTE_LABELS.items()):
        print(f"  {wnid}  {label}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ViT inference demo for Imagenette-10 classification",
    )
    parser.add_argument(
        "--image", default=None,
        help="Path to a single image. If omitted, runs on all bundled test images.",
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help=f"Path to trained checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--device", default=None,
        help="e.g. 'cuda' or 'cpu'; auto-select if omitted",
    )
    parser.add_argument(
        "--list-classes", action="store_true",
        help="Print the 10 Imagenette class names and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_classes:
        list_classes()
        return

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    _resolve_checkpoint(args.checkpoint)

    if args.image:
        run_single_inference(args.image, args.checkpoint, device, topk=args.topk)
    else:
        run_demo(args.checkpoint, DEFAULT_TEST_DIR, device, topk=args.topk)


if __name__ == "__main__":
    main()
