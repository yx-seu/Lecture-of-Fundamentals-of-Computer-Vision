"""
SwinIR Processing Script — Grayscale Image Denoising (gray_dn, noise=15)

Applies SwinIR to CLAHE+DnCNN processed images in train/val/test folders.
Input:  *_clahe_dncnn.jpg
Output: *_clahe_dncnn_SwinIR.jpg  (does NOT overwrite originals)
"""

import os
import sys
import time
import cv2
import torch
import numpy as np

# Resolve paths reliably
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWINIR_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "SwinIR-main"))
sys.path.insert(0, SWINIR_ROOT)

from models.network_swinir import SwinIR

# =====================================================
# Configuration
# =====================================================

MODEL_PATH = os.path.join(SWINIR_ROOT, "model_zoo", "swinir",
                          "004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 8

# =====================================================
# Build SwinIR model for grayscale denoising
# =====================================================

print(f"Device: {DEVICE}", flush=True)

model = SwinIR(
    upscale=1,
    in_chans=1,
    img_size=128,
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler="",
    resi_connection="1conv",
)

# Load weights
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}", flush=True)
    print("Please download the pretrained model first.", flush=True)
    sys.exit(1)

print(f"Loading model from {MODEL_PATH}...", flush=True)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(
    checkpoint["params"] if "params" in checkpoint else checkpoint, strict=True
)
model.eval()
model = model.to(DEVICE)
print(f"Model loaded.", flush=True)

# =====================================================
# Process images
# =====================================================

BASE_DIR = SCRIPT_DIR
SUB_DIRS = ["train", "val", "test"]

total_count = 0

for sub_dir in SUB_DIRS:
    folder_path = os.path.join(BASE_DIR, sub_dir)

    if not os.path.exists(folder_path):
        print(f"  [SKIP] Folder not found: {folder_path}", flush=True)
        continue

    # Count files first
    jpg_files = sorted([f for f in os.listdir(folder_path)
                        if f.lower().endswith(".jpg")
                        and "_dncnn" in f
                        and "_SwinIR" not in f
                        and "_swinir" not in f.lower()])
    n_files = len(jpg_files)
    if n_files == 0:
        print(f"\n{sub_dir}/: no files to process", flush=True)
        continue

    print(f"\n{'='*60}", flush=True)
    print(f"Processing: {sub_dir}/  ({n_files} images)", flush=True)
    print(f"{'='*60}", flush=True)

    count = 0
    t_start = time.time()

    for idx, file_name in enumerate(jpg_files):
        img_path = os.path.join(folder_path, file_name)

        try:
            t_img = time.time()

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [{idx+1}/{n_files}] ERROR Cannot read: {file_name}", flush=True)
                continue

            img_np = img.astype(np.float32) / 255.0
            img_tensor = (
                torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            )

            with torch.no_grad():
                _, _, h_old, w_old = img_tensor.size()
                h_pad = (h_old // WINDOW_SIZE + 1) * WINDOW_SIZE - h_old
                w_pad = (w_old // WINDOW_SIZE + 1) * WINDOW_SIZE - w_old

                img_padded = torch.cat(
                    [img_tensor, torch.flip(img_tensor, [2])], 2
                )[:, :, : h_old + h_pad, :]
                img_padded = torch.cat(
                    [img_padded, torch.flip(img_padded, [3])], 3
                )[:, :, :, : w_old + w_pad]

                output = model(img_padded)
                output = output[..., :h_old, :w_old]

            output_np = (
                output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            )
            output_img = (output_np * 255.0).round().astype(np.uint8)

            base_name = os.path.splitext(file_name)[0]
            save_name = f"{base_name}_SwinIR.jpg"
            save_path = os.path.join(folder_path, save_name)

            cv2.imwrite(save_path, output_img)
            count += 1
            total_count += 1
            elapsed = time.time() - t_img

            # Estimate remaining time
            done = idx + 1
            avg_time = (time.time() - t_start) / done
            eta_sec = avg_time * (n_files - done)
            print(f"  [{done}/{n_files}] {file_name} -> {save_name}  "
                  f"({elapsed:.1f}s, ETA: {eta_sec/60:.1f}min)", flush=True)

        except Exception as e:
            print(f"  [{idx+1}/{n_files}] ERROR {file_name}: {e}", flush=True)

    elapsed_total = time.time() - t_start
    print(f"Done {sub_dir}: {count} images in {elapsed_total/60:.1f} min", flush=True)

print(f"\n{'='*60}", flush=True)
print(f"All SwinIR processing complete!  Total: {total_count} images", flush=True)
print(f"{'='*60}", flush=True)
