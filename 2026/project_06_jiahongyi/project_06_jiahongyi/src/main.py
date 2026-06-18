"""
main.py — Sample Inference Pipeline for TEM Image Object Detection

This script demonstrates the full preprocessing + detection pipeline on
sample test images. It runs CLAHE enhancement, DnCNN denoising,
SwinIR restoration, and YOLOv11 object detection.

Usage:
    python main.py

Prerequisites:
    - Install dependencies: pip install -r requirements.txt
    - Place dncnn_pretrained.pth in src/
    - Place SwinIR model in SwinIR-main/model_zoo/swinir/
    - Place sample images in data/test_examples/
"""

import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import argparse

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Paths
TEST_EXAMPLES_DIR = os.path.join(PROJECT_DIR, "data", "test_examples")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results", "inference_output")
DnCNN_MODEL_PATH = os.path.join(SCRIPT_DIR, "dncnn_pretrained.pth")

# CLAHE parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dncnn_model(model_path):
    """Load DnCNN model from checkpoint."""
    from dncnn import DnCNN

    model = DnCNN(channels=1)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[DnCNN] Loaded from {model_path}")
    else:
        print(f"[DnCNN] WARNING: Model not found at {model_path}")
        print("[DnCNN] Running in demo mode — will skip DnCNN step.")
        return None

    model.to(DEVICE)
    model.eval()
    return model


def apply_clahe(image_path, output_path=None):
    """Apply CLAHE enhancement to a grayscale image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    img_clahe = clahe.apply(img)

    if output_path:
        cv2.imwrite(output_path, img_clahe)

    return img_clahe


def apply_dncnn(model, image, output_path=None):
    """Apply DnCNN denoising to a grayscale image (numpy array)."""
    if model is None:
        return image

    h, w = image.shape
    img_np = image.astype(np.float32) / 255.0
    input_tensor = (
        torch.tensor(img_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    )

    with torch.no_grad():
        output_tensor = model(input_tensor)

    output_np = output_tensor.squeeze().cpu().numpy()
    output_np = np.clip(output_np, 0, 1)
    output_img = (output_np * 255).astype(np.uint8)

    if output_path:
        cv2.imwrite(output_path, output_img)

    return output_img


def run_yolo_inference(image_path):
    """Run YOLOv11 inference on an image and return detection results."""
    try:
        from ultralytics import YOLO

        # Use a pretrained YOLOv11 model
        # Replace with your trained weights path if available
        model = YOLO("yolov11n.pt")

        results = model(image_path, verbose=False)
        return results
    except ImportError:
        print("[YOLO] ultralytics not installed. Skipping YOLO inference.")
        print("[YOLO] Install with: pip install ultralytics")
        return None
    except Exception as e:
        print(f"[YOLO] Inference error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="TEM Image Object Detection Pipeline — Sample Inference"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image for inference. If not specified, "
             "processes all raw images in test_examples/.",
    )
    parser.add_argument(
        "--skip-dncnn",
        action="store_true",
        help="Skip DnCNN denoising step.",
    )
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip YOLO detection step.",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("TEM Image Processing & Detection Pipeline")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load DnCNN model
    dncnn_model = None
    if not args.skip_dncnn:
        dncnn_model = load_dncnn_model(DnCNN_MODEL_PATH)

    # Determine which images to process
    if args.image:
        image_paths = [args.image]
    else:
        # Find all raw images in test_examples
        image_paths = sorted([
            os.path.join(TEST_EXAMPLES_DIR, f)
            for f in os.listdir(TEST_EXAMPLES_DIR)
            if f.endswith("_raw.jpg")
        ])

    if not image_paths:
        print("No images found to process.")
        print(f"Please add images to {TEST_EXAMPLES_DIR}")
        return

    print(f"\nProcessing {len(image_paths)} image(s)...\n")

    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n--- Processing: {img_name} ---")

        # Step 1: CLAHE
        print("[1/3] Applying CLAHE...")
        clahe_path = os.path.join(OUTPUT_DIR, f"{img_name}_clahe.jpg")
        clahe_img = apply_clahe(img_path, clahe_path)
        print(f"      Saved: {clahe_path}")

        # Step 2: DnCNN
        print("[2/3] Applying DnCNN...")
        dncnn_path = os.path.join(OUTPUT_DIR, f"{img_name}_dncnn.jpg")
        dncnn_img = apply_dncnn(dncnn_model, clahe_img, dncnn_path)
        print(f"      Saved: {dncnn_path}")

        # Step 3: YOLO Detection
        if not args.skip_yolo:
            print("[3/3] Running YOLOv11 detection...")
            results = run_yolo_inference(dncnn_path)
            if results is not None:
                # Save annotated result
                annotated_path = os.path.join(
                    OUTPUT_DIR, f"{img_name}_detected.jpg"
                )
                results[0].save(annotated_path)
                print(f"      Saved: {annotated_path}")

                # Print detections
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    class_names = results[0].names
                    print(f"      Detections: {len(boxes)} objects")
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        print(
                            f"        - {class_names[cls_id]}: "
                            f"confidence={conf:.3f}"
                        )
                else:
                    print("      No objects detected.")
            else:
                print("[3/3] YOLO detection skipped (model not available).")

    print("\n" + "=" * 60)
    print(f"Done! Output saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
