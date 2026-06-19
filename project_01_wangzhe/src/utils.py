"""Shared utilities for ViT inference: checkpoint loading, image preprocessing, prediction.

All functions are reusable by main.py, infer.py, and the Jupyter demo.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from vit import VisionTransformer

# --- Imagenette-10 class labels -------------------------------------------------
IMAGENETTE_LABELS: Dict[str, str] = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def _checkpoint_args(checkpoint: dict) -> dict:
    """Extract model hyperparameters stored inside a training checkpoint."""
    args = checkpoint.get("args", {})
    return {
        "img_size": int(args.get("img_size", 224)),
        "patch_size": int(args.get("patch_size", 16)),
        "embed_dim": int(args.get("embed_dim", 256)),
        "depth": int(args.get("depth", 6)),
        "num_heads": int(args.get("num_heads", 8)),
        "mlp_ratio": int(args.get("mlp_ratio", 4)),
        "dropout": float(args.get("dropout", 0.0)),
        "drop_path": float(args.get("drop_path", 0.0)),
    }


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[VisionTransformer, List[str], dict]:
    """Load a trained ViT checkpoint and return (model, class_names, model_args)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = list(checkpoint["class_names"])
    model_args = _checkpoint_args(checkpoint)

    model = VisionTransformer(
        img_size=model_args["img_size"],
        patch_size=model_args["patch_size"],
        num_classes=len(class_names),
        embed_dim=model_args["embed_dim"],
        depth=model_args["depth"],
        num_heads=model_args["num_heads"],
        mlp_ratio=model_args["mlp_ratio"],
        dropout=model_args["dropout"],
        drop_path=model_args["drop_path"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, class_names, model_args


def build_inference_transform(img_size: int) -> transforms.Compose:
    """Validation-style deterministic transform used at inference time."""
    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


@torch.no_grad()
def predict_image(
    model: VisionTransformer,
    image_path: str,
    class_names: List[str],
    img_size: int,
    device: torch.device,
    topk: int = 5,
) -> List[dict]:
    """Run top-k prediction on a single image.

    Returns a list of dicts with keys: class_id, label, probability.
    """
    image = Image.open(image_path).convert("RGB")
    transform = build_inference_transform(img_size)
    batch = transform(image).unsqueeze(0).to(device)

    logits = model(batch)
    probabilities = logits.softmax(dim=1).squeeze(0)
    k = min(topk, len(class_names))
    top_probs, top_indices = probabilities.topk(k)

    predictions = []
    for probability, index in zip(top_probs.tolist(), top_indices.tolist()):
        class_id = class_names[index]
        predictions.append(
            {
                "class_id": class_id,
                "label": IMAGENETTE_LABELS.get(class_id, class_id),
                "probability": probability,
            }
        )
    return predictions
