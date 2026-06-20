"""
Generate test images, run all 4 scenarios, and produce progressive visualization results.
Outputs go to data/test_examples/ and results/figures/.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.pipeline.inference import ControlNetInference
from src.pipeline.scenarios import ScenarioPipeline
from src.pipeline.preprocessors import PreprocessorRegistry

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
SD_PATH = os.path.join(MODELS_DIR, "stable-diffusion-v1-5")
DATA_DIR = "data/test_examples"
RESULTS_DIR = "results/figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 1. Create 3 diverse test images ──────────────────────────
print("[1/5] Creating test images...")

def create_test_image_1(output_path):
    """Hand-drawn style lineart of a simple character (simulated)"""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    # Head circle
    cv2.circle(img, (256, 160), 80, (0, 0, 0), 2)
    # Body
    cv2.line(img, (256, 240), (256, 380), (0, 0, 0), 2)
    # Arms
    cv2.line(img, (256, 280), (160, 320), (0, 0, 0), 2)
    cv2.line(img, (256, 280), (350, 320), (0, 0, 0), 2)
    # Legs
    cv2.line(img, (256, 380), (190, 470), (0, 0, 0), 2)
    cv2.line(img, (256, 380), (320, 470), (0, 0, 0), 2)
    # Face
    cv2.circle(img, (236, 145), 5, (0, 0, 0), -1)
    cv2.circle(img, (276, 145), 5, (0, 0, 0), -1)
    cv2.ellipse(img, (256, 170), (12, 6), 0, 0, 180, (0, 0, 0), 1)
    # Hair
    cv2.ellipse(img, (256, 110), (82, 30), 0, 180, 360, (0, 0, 0), 2)
    Image.fromarray(img).save(output_path)
    print(f"  Created: {output_path}")

def create_test_image_2(output_path):
    """Sketch of a house/building (simulated doodle)"""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    # House body
    cv2.rectangle(img, (120, 220), (380, 460), (0, 0, 0), 3)
    # Roof triangle
    pts = np.array([[100, 220], [250, 80], [400, 220]], np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 0), 3)
    # Door
    cv2.rectangle(img, (220, 330), (280, 460), (0, 0, 0), 2)
    cv2.circle(img, (270, 400), 3, (0, 0, 0), -1)
    # Windows
    cv2.rectangle(img, (150, 260), (200, 310), (0, 0, 0), 2)
    cv2.line(img, (175, 260), (175, 310), (0, 0, 0), 1)
    cv2.line(img, (150, 285), (200, 285), (0, 0, 0), 1)
    cv2.rectangle(img, (300, 260), (350, 310), (0, 0, 0), 2)
    cv2.line(img, (325, 260), (325, 310), (0, 0, 0), 1)
    cv2.line(img, (300, 285), (350, 285), (0, 0, 0), 1)
    # Ground
    cv2.line(img, (60, 460), (450, 450), (0, 0, 0), 3)
    # Sun
    cv2.circle(img, (380, 110), 35, (0, 0, 0), 2)
    Image.fromarray(img).save(output_path)
    print(f"  Created: {output_path}")

def create_test_image_3(output_path):
    """Pattern with geometric shapes (simulating an old photo of structures)"""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 220
    # Add some texture/noise (old photo look)
    noise = np.random.randint(0, 30, (512, 512, 3), dtype=np.uint8)
    img = img - noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    # Large buildings
    cv2.rectangle(img, (80, 150), (200, 400), (120, 120, 120), -1)
    cv2.rectangle(img, (220, 200), (340, 400), (90, 90, 90), -1)
    cv2.rectangle(img, (360, 180), (440, 400), (140, 140, 140), -1)
    # Windows pattern
    for y in range(170, 380, 30):
        for x in range(100, 180, 25):
            cv2.rectangle(img, (x, y), (x + 15, y + 20), (50, 50, 50), -1)
    # Sky gradient
    for y in range(150):
        color = int(180 + y * 0.5)
        img[y, :] = [color + 30, color + 50, color + 70]
    Image.fromarray(img).save(output_path)
    print(f"  Created: {output_path}")

import cv2

test_images = {
    "test_lineart_input.png": create_test_image_1,
    "test_sketch_input.png": create_test_image_2,
    "test_oldphoto_input.png": create_test_image_3,
}

for fname, func in test_images.items():
    func(os.path.join(DATA_DIR, fname))

# Also copy test_input.png as a generic input
Image.open("outputs/test_input.png").save(os.path.join(DATA_DIR, "test_generic_input.png"))
print("  Created: test_generic_input.png (copy)")

# ── 2. Initialize engine ─────────────────────────────────────
print("\n[2/5] Initializing inference engine...")
engine = ControlNetInference(
    sd_model_path=SD_PATH, device="cuda",
    torch_dtype=torch.float16, local_files_only=True,
)
sp = ScenarioPipeline(engine=engine, sd_models_dir=MODELS_DIR)
print("  Engine initialized.")

# ── 3. Run all 4 scenarios with quality settings ─────────────
scenarios = [
    {
        "name": "lineart_coloring",
        "method": sp.run_lineart_coloring,
        "input_file": "test_lineart_input.png",
        "style": "vivid_anime",
        "steps": 25,
    },
    {
        "name": "sketch_to_realistic",
        "method": sp.run_sketch_to_realistic,
        "input_file": "test_sketch_input.png",
        "style": "architecture",
        "steps": 25,
    },
    {
        "name": "photo_to_anime",
        "method": sp.run_photo_to_anime,
        "input_file": "test_generic_input.png",
        "style": "anime_film",
        "steps": 25,
    },
    {
        "name": "old_photo_restore",
        "method": sp.run_old_photo_restore,
        "input_file": "test_oldphoto_input.png",
        "style": "restore_bw",
        "steps": 25,
    },
]

print("\n[3/5] Running scenario pipelines (25 steps each)...")
for i, sc in enumerate(scenarios):
    print(f"\n  Scenario {i+1}/4: {sc['name']}...")
    input_path = os.path.join(DATA_DIR, sc["input_file"])
    input_img = Image.open(input_path).convert("RGB")

    # Save input to results
    input_img.save(os.path.join(RESULTS_DIR, f"{sc['name']}_01_input.png"))

    # Preprocessing: get control images
    config = sp.get_config(sc["name"])
    control_images = []
    for preproc_name in config.preprocessor_names:
        preprocessor = PreprocessorRegistry.get(preproc_name)
        result = preprocessor(input_img, output_size=512)
        if isinstance(result, list):
            control_images.extend(result)
        else:
            control_images.append(result)

    # Save control images
    for j, cimg in enumerate(control_images[:len(config.controlnet_names)]):
        cn_name = config.controlnet_names[j]
        cimg.save(os.path.join(RESULTS_DIR, f"{sc['name']}_02_control_{cn_name}.png"))

    # Run inference
    result = sc["method"](image=input_img, style=sc["style"], steps=sc["steps"], seed=42)

    # Save output
    result["output_image"].save(os.path.join(RESULTS_DIR, f"{sc['name']}_03_output.png"))
    print(f"    Seed: {result['seed']}, Time: {result['time']:.1f}s")
    print(f"    Prompt: {result.get('prompt_used', 'N/A')[:100]}...")

# ── 4. Create composite visualization ────────────────────────
print("\n[4/5] Creating composite visualizations...")

def create_composite(scenario_name, title_text):
    """Create a horizontal composite: input | control | output"""
    input_img = Image.open(os.path.join(RESULTS_DIR, f"{scenario_name}_01_input.png"))
    output_img = Image.open(os.path.join(RESULTS_DIR, f"{scenario_name}_03_output.png"))

    # Find control images
    control_files = sorted([
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith(scenario_name) and "_02_control_" in f
    ])

    # Create composite
    panels = [input_img]
    for cf in control_files:
        panels.append(Image.open(os.path.join(RESULTS_DIR, cf)))
    panels.append(output_img)

    total_w = sum(p.size[0] for p in panels) + 20 * (len(panels) - 1)
    max_h = max(p.size[1] for p in panels)
    composite = Image.new("RGB", (total_w, max_h + 40), (255, 255, 255))

    x_offset = 0
    labels = ["Input"] + [cf.replace(f"{scenario_name}_02_control_", "").replace(".png", "")
                           for cf in control_files] + ["Output"]
    for j, (panel, label) in enumerate(zip(panels, labels)):
        composite.paste(panel, (x_offset, 40))
        # Add label (simple text using PIL)
        draw = ImageDraw.Draw(composite)
        draw.text((x_offset + 5, 10), label, fill=(0, 0, 0))
        x_offset += panel.size[0] + 20

    composite.save(os.path.join(RESULTS_DIR, f"{scenario_name}_composite.png"))
    print(f"  Created: {scenario_name}_composite.png")

for sc in scenarios:
    create_composite(sc["name"], sc["name"])

# ── 5. Save metadata ─────────────────────────────────────────
print("\n[5/5] Saving metadata...")
with open(os.path.join(RESULTS_DIR, "run_info.txt"), "w") as f:
    f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
    f.write(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB\n")
    f.write(f"Base Model: Stable Diffusion 1.5\n")
    f.write(f"ControlNets: Lineart, Canny, Scribble, AnimeLineart, OpenPose, Depth\n")
    f.write(f"Framework: diffusers {torch.__version__}\n")
    f.write(f"Date: 2026-06-19\n")

print("\nDone! All results saved to results/figures/")
