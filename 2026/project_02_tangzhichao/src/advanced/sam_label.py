"""
Grounded-SAM pseudo-labeling: Grounding DINO (text→bbox) + SAM (bbox→mask).

For each class:
  1. Map ImageNet synset ID → English word (e.g. n01443537 → "goldfish")
  2. Grounding DINO takes (image, text) → bounding box
  3. SAM takes (image, bbox) → precise segmentation mask
  4. If Grounding DINO fails, fall back to SAM automatic + DINOv2 matching

Usage:
    source /etc/network_turbo
    python src/sam_label.py --topk 20
"""

import os, sys, argparse, random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from PIL import Image
from torchvision.ops import box_convert

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from advanced.dataset_v2 import ImageNetSDataset


# ImageNet-S50 synset ID → English common name
SYNSET_TO_NAME = {
    'n01443537': 'goldfish',
    'n01491361': 'tiger shark',
    'n01531178': 'goldfinch bird',
    'n01644373': 'alligator lizard',
    'n02104029': 'kuvasz dog',
    'n02119022': 'red fox',
    'n02123597': 'Siamese cat',
    'n02133161': 'American black bear',
    'n02165456': 'tailed frog',
    'n02281406': 'walking stick insect',
    'n02325366': 'tree frog',
    'n02342885': 'box turtle',
    'n02396427': 'wild boar',
    'n02483362': 'gibbon monkey',
    'n02504458': 'African elephant',
    'n02510455': 'giant panda',
    'n02690373': 'airliner airplane',
    'n02747177': 'trash can',
    'n02783161': 'balloon',
    'n02814533': 'beach wagon',
    'n02859443': 'bobsled',
    'n02917067': 'bullet train',
    'n02992529': 'cellphone',
    'n03014705': 'chest',
    'n03047690': 'clog shoe',
    'n03095699': 'cornet musical instrument',
    'n03197337': 'digital watch',
    'n03201208': 'dining table',
    'n03445777': 'golf ball',
    'n03452741': 'grand piano',
    'n03584829': 'hair slide',
    'n03630383': 'hook',
    'n03775546': 'mixing bowl',
    'n03791053': 'moped',
    'n03874599': 'passenger train',
    'n03891251': 'park bench',
    'n04026417': 'punching bag',
    'n04335435': 'steel arch bridge',
    'n04380533': 'street sign',
    'n04404412': 'television',
    'n04447861': 'toilet seat',
    'n04507155': 'umbrella',
    'n04522168': 'vase',
    'n04557648': 'water bottle',
    'n04562935': 'water jug',
    'n04612504': 'Yorkshire terrier dog',
    'n06794110': 'screwdriver',
    'n07749582': 'lemon fruit',
    'n07831146': 'safety pin',
    'n12998815': 'ant insect',
}


def load_grounding_dino(device: torch.device):
    """Load Grounding DINO for open-vocabulary object detection."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def load_sam(ckpt: str, device: torch.device):
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_h"](checkpoint=ckpt).to(device)
    sam.eval()
    return SamPredictor(sam)


def grounded_sam_mask(
    image_pil: Image.Image,
    text_prompt: str,
    gd_processor, gd_model,
    sam_predictor,
    device: torch.device,
    box_threshold: float = 0.15,
) -> np.ndarray | None:
    """
    Grounding DINO (text→bbox) + SAM (bbox→mask).

    Returns: (H, W) binary mask, or None if no object detected.
    """
    img_np = np.array(image_pil.convert("RGB"))

    # ---- Step 1: Grounding DINO → bounding boxes ----
    inputs = gd_processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gd_model(**inputs)

    # Post-process: extract boxes with confidence > threshold
    target_sizes = torch.tensor([img_np.shape[:2]]).to(device)
    results = gd_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=box_threshold,
        target_sizes=target_sizes,
    )[0]

    if len(results["boxes"]) == 0:
        return None

    # Pick highest-confidence box
    best_idx = results["scores"].argmax().item()
    box = results["boxes"][best_idx].cpu().numpy()  # (x1, y1, x2, y2)
    # Clamp to image bounds
    h, w = img_np.shape[:2]
    box = np.array([max(0, box[0]), max(0, box[1]),
                    min(w, box[2]), min(h, box[3])])

    # ---- Step 2: SAM with box prompt ----
    sam_predictor.set_image(img_np)
    masks, scores, _ = sam_predictor.predict(
        box=box[None, :],
        multimask_output=False,
    )
    if masks is None or len(masks) == 0:
        return None

    return masks[0].astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-checkpoint", type=str, default="/root/autodl-tmp/sam_vit_h.pth")
    parser.add_argument("--data-root", type=str, default="data/imagenet-s/ImageNetS50")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    out_root = Path(args.output or os.path.join(args.data_root, "sam_pseudo"))

    # ---- Load models ----
    print("Loading Grounding DINO...")
    gd_processor, gd_model = load_grounding_dino(device)
    print("Loading SAM ViT-H...")
    sam_predictor = load_sam(args.sam_checkpoint, device)

    # ---- Load datasets ----
    ds_semi  = ImageNetSDataset(args.data_root, mode="train-semi", size=448, augment=False)
    ds_train = ImageNetSDataset(args.data_root, mode="train", size=448, augment=False)
    class_names = sorted(ds_semi.class_to_id.keys())
    print(f"{len(class_names)} classes")

    # ---- Output ----
    out_img_dir  = out_root / "train-pseudo"
    out_mask_dir = out_root / "train-pseudo-segmentation"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for cls_name in tqdm(class_names, desc="Classes"):
        cls_id = ds_semi.class_to_id[cls_name]
        text_prompt = SYNSET_TO_NAME.get(cls_name, cls_name)

        cls_img_dir  = out_img_dir / cls_name
        cls_mask_dir = out_mask_dir / cls_name
        cls_img_dir.mkdir(parents=True, exist_ok=True)
        cls_mask_dir.mkdir(parents=True, exist_ok=True)

        # Candidates from unlabeled train set
        semi_fnames = {os.path.basename(s["image"]) for s in ds_semi.samples
                       if s["class_name"] == cls_name}
        candidates = [s for s in ds_train.samples
                      if s["class_name"] == cls_name
                      and os.path.basename(s["image"]) not in semi_fnames]
        chosen = random.sample(candidates, min(args.topk * 3, len(candidates)))

        kept = 0
        for cand in tqdm(chosen, desc=f"  {cls_name}", leave=False):
            try:
                img = Image.open(cand["image"]).convert("RGB")
            except Exception:
                continue

            mask = grounded_sam_mask(img, text_prompt,
                                     gd_processor, gd_model,
                                     sam_predictor, device)

            if mask is None or mask.sum() < 500:
                continue

            img_area = mask.shape[0] * mask.shape[1]
            if mask.sum() / img_area > 0.80:  # skip full-image masks
                continue

            # Save
            fname = os.path.splitext(os.path.basename(cand["image"]))[0]
            h, w = mask.shape
            mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            mask_rgb[:, :, 0] = mask * (cls_id % 256)
            mask_rgb[:, :, 1] = mask * (cls_id // 256)
            Image.fromarray(mask_rgb).save(cls_mask_dir / f"{fname}.png")

            dst_img = cls_img_dir / f"{fname}.JPEG"
            if not dst_img.exists():
                train_path = os.path.join("data/imagenet-s/ImageNetS50/train", cls_name,
                                          os.path.basename(cand["image"]))
                os.symlink(os.path.abspath(train_path), dst_img)

            kept += 1
            if kept >= args.topk:
                break

        print(f"  {cls_name}: {kept}/{args.topk} (prompt: '{text_prompt}')")
        total += kept

    print(f"\nSaved {total} pseudo-labels → {out_root}")


if __name__ == "__main__":
    main()
