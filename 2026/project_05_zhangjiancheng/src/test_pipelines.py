"""
Test script for all 4 scenario pipelines.
Run: python test_pipelines.py
"""
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
from src.pipeline.inference import ControlNetInference
from src.pipeline.scenarios import ScenarioPipeline

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
SD_PATH = os.path.join(MODELS_DIR, "stable-diffusion-v1-5")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a test image with some visual structure
img = Image.new('RGB', (512, 512), color=(200, 180, 160))
arr = np.array(img)
arr[50:200, 100:400] = [150, 130, 110]
arr[250:400, 150:350] = [100, 80, 60]
arr[50:200, 100:120] = [50, 30, 20]
# Add a circular pattern
yy, xx = np.ogrid[:512, :512]
circle = ((xx - 350)**2 + (yy - 150)**2) < 60**2
arr[circle] = [220, 200, 180]
img = Image.fromarray(arr)
img.save(os.path.join(OUTPUT_DIR, "test_input.png"))
print("Test image saved to outputs/test_input.png")

# Initialize engine
print("\n[1/5] Initializing engine...")
engine = ControlNetInference(
    sd_model_path=SD_PATH,
    device="cuda",
    torch_dtype=torch.float16,
    local_files_only=True,
)
print(f"  GPU: {torch.cuda.get_device_name(0)}")

sp = ScenarioPipeline(engine=engine, sd_models_dir=MODELS_DIR)

# Test scenarios one by one
tests = [
    {
        "name": "lineart_coloring",
        "method": sp.run_lineart_coloring,
        "kwargs": {"image": img, "style": "vivid_anime", "steps": 10},
        "output": "test_lineart_coloring.png",
    },
    {
        "name": "sketch_to_realistic",
        "method": sp.run_sketch_to_realistic,
        "kwargs": {"image": img, "style": "architecture", "steps": 10},
        "output": "test_sketch_to_realistic.png",
    },
    {
        "name": "photo_to_anime",
        "method": sp.run_photo_to_anime,
        "kwargs": {"image": img, "style": "anime_film", "steps": 10},
        "output": "test_photo_to_anime.png",
    },
    {
        "name": "old_photo_restore",
        "method": sp.run_old_photo_restore,
        "kwargs": {"image": img, "style": "restore_bw", "steps": 10},
        "output": "test_old_photo_restore.png",
    },
]

for i, test in enumerate(tests):
    print(f"\n[{i+2}/5] Testing: {test['name']}...")
    try:
        result = test["method"](**test["kwargs"])
        out_path = os.path.join(OUTPUT_DIR, test["output"])
        result["output_image"].save(out_path)
        print(f"  SUCCESS! Saved to {test['output']}")
        print(f"  Seed: {result['seed']}, Time: {result['time']:.1f}s")
        print(f"  Prompt: {result.get('prompt_used', 'N/A')[:80]}...")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)

# Show GPU memory
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU memory reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
