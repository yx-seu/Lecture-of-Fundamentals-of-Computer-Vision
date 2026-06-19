"""Single-image inference CLI for the trained ViT model.

Usage:
    python infer.py --image path/to/image.jpg
    python infer.py --image path/to/image.jpg --checkpoint path/to/checkpoint.pt --topk 3
"""

import argparse
import sys
from pathlib import Path

import torch

# Allow running from the project root: python src/infer.py
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils import load_model_from_checkpoint, predict_image  # noqa: E402

DEFAULT_CHECKPOINT = str(Path(_SCRIPT_DIR).parent / "outputs_v3" / "vit_imagenette10_best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a ViT checkpoint and run top-k prediction on a single image.",
    )
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", default=None, help="e.g. 'cuda' or 'cpu'; auto-select if omitted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, class_names, model_args = load_model_from_checkpoint(str(checkpoint_path), device)
    predictions = predict_image(
        model, str(image_path), class_names, model_args["img_size"], device, topk=args.topk,
    )

    print(f"Image:      {image_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print("Top predictions:")
    for rank, pred in enumerate(predictions, start=1):
        print(
            f"{rank}. {pred['class_id']} ({pred['label']}): "
            f"{pred['probability'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
