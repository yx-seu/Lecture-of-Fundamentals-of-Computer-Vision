import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import load_model
from .utils import ensure_dir


def collect_image_paths(input_path: Path):
    if input_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted([path for path in input_path.iterdir() if path.suffix.lower() in exts])
    return [input_path]


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0, 0, 0],
            [255, 140, 0],
            [0, 168, 150],
            [70, 130, 180],
            [220, 80, 90],
            [155, 89, 182],
            [46, 204, 113],
            [241, 196, 15],
            [230, 126, 34],
            [52, 73, 94],
            [231, 76, 60],
            [26, 188, 156],
            [149, 165, 166],
            [39, 174, 96],
            [142, 68, 173],
            [243, 156, 18],
        ],
        dtype=np.uint8,
    )
    rgb = palette[mask % len(palette)]
    rgb[mask == 0] = [0, 0, 0]
    return rgb


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    mask_rgb = mask_to_rgb(mask)
    overlay = image.astype(np.float32) * (1 - alpha) + mask_rgb.astype(np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def run_inference(model, image_processor, image: Image.Image, device):
    inputs = image_processor(images=image.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
    logits = F.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_processor = load_model(args.model_dir, device)

    input_path = Path(args.input)
    output_root = ensure_dir(args.output)

    for image_path in collect_image_paths(input_path):
        image = Image.open(image_path).convert("RGB")
        pred_mask = run_inference(model, image_processor, image, device)
        image_np = np.asarray(image)
        overlay = overlay_mask(image_np, pred_mask)

        stem_dir = ensure_dir(output_root / image_path.stem)
        Image.fromarray(pred_mask.astype(np.uint8)).save(stem_dir / "pred_mask.png")
        Image.fromarray(mask_to_rgb(pred_mask)).save(stem_dir / "pred_color.png")
        Image.fromarray(overlay).save(stem_dir / "overlay.png")
        np.savetxt(stem_dir / "pred_labels.txt", np.unique(pred_mask).astype(int), fmt="%d")


if __name__ == "__main__":
    main()
