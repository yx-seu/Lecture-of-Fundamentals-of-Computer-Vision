"""
ImageNet-S50 Dataset and DataLoader utilities.

ImageNet-S class IDs are 1-indexed based on sorted ImageNet tag IDs.
Segmentation masks encode class IDs as: class_id = R + G * 256
  - class_id = 0:    other category
  - class_id = 1000: ignored region
"""

import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Class mapping is built at init time by scanning the data directory,
# which matches the sorted order of ImageNetS_categories_im50.txt


class ImageNetSDataset(Dataset):
    """
    Dataset for ImageNet-S50 semantic segmentation.

    Modes:
        'train'        — training images, no mask
        'train-semi'   — training images + pixel-level masks (~10 per class)
        'val'          — validation images + masks
        'test'         — test images, no mask
    """

    def __init__(
        self,
        root: str,
        mode: str = "train",
        size: int = 448,
        augment: bool = False,
    ):
        """
        Args:
            root:   path to ImageNetS50/ (contains train/, validation/, etc.)
            mode:   'train' | 'train-semi' | 'val' | 'test'
            size:   target image size (must be multiple of 14 for DINOv2)
            augment: apply data augmentation (only for training modes)
        """
        assert mode in ("train", "train-semi", "train-semi-plus", "val", "test"), \
            f"Unknown mode: {mode}"
        assert size % 14 == 0, f"size must be multiple of 14, got {size}"

        self.root = os.path.expanduser(root)
        self.mode = mode
        self.size = size
        self.augment = augment

        # ---- determine image / mask directories ----
        if mode == "train":
            self.img_dir = os.path.join(self.root, "train")
            self.mask_dir = None
        elif mode == "train-semi":
            self.img_dir = os.path.join(self.root, "train-semi")
            self.mask_dir = os.path.join(self.root, "train-semi-segmentation")
        elif mode == "train-semi-plus":
            # Combined: original 500 + SAM pseudo-labels
            self.img_dir = [
                os.path.join(self.root, "train-semi"),
                os.path.join(self.root, "sam_pseudo", "train-pseudo"),
            ]
            self.mask_dir = [
                os.path.join(self.root, "train-semi-segmentation"),
                os.path.join(self.root, "sam_pseudo", "train-pseudo-segmentation"),
            ]
        elif mode == "val":
            self.img_dir = os.path.join(self.root, "validation")
            self.mask_dir = os.path.join(self.root, "validation-segmentation")
        else:  # test
            self.img_dir = os.path.join(self.root, "test")
            self.mask_dir = None

        # ---- build class mapping (locked to canonical train directory) ----
        # Always use root/train as the single source of truth for class ordering.
        # This prevents label shifts when train-semi / val / test might differ
        # (e.g. due to missing class directories, partial downloads, etc.).
        canonical_dir = os.path.join(self.root, "train")
        if not os.path.isdir(canonical_dir):
            # Fallback: if train/ is absent, use first image dir
            canonical_dir = self.img_dir if isinstance(self.img_dir, str) else self.img_dir[0]
        class_names = sorted(os.listdir(canonical_dir))
        self.class_to_id = {name: i + 1 for i, name in enumerate(class_names)}
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}
        # +1 for background class (class_id=0 in masks)
        self.num_classes = len(class_names) + 1

        # ---- build sample list ----
        self.samples = []

        # Support multi-directory sources (e.g. train-semi + sam_pseudo)
        img_dirs = self.img_dir if isinstance(self.img_dir, list) else [self.img_dir]
        mask_dirs = self.mask_dir if isinstance(self.mask_dir, list) else (
            [self.mask_dir] if self.mask_dir else [None])

        for img_d, mask_d in zip(img_dirs, mask_dirs):
            for cls_name in class_names:
                cls_img_dir = os.path.join(img_d, cls_name)
                if not os.path.isdir(cls_img_dir):
                    continue
                cls_mask_dir = os.path.join(mask_d, cls_name) if mask_d else None
                for fname in sorted(os.listdir(cls_img_dir)):
                    if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
                        continue
                    img_path = os.path.join(cls_img_dir, fname)
                    if cls_mask_dir is not None:
                        mask_name = os.path.splitext(fname)[0] + ".png"
                        mask_path = os.path.join(cls_mask_dir, mask_name)
                        if not os.path.exists(mask_path):
                            continue
                    else:
                        mask_path = None
                    self.samples.append({
                        "image": img_path,
                        "mask": mask_path,
                        "class_id": self.class_to_id[cls_name],
                        "class_name": cls_name,
                    })

        # ---- transforms ----
        self._build_transforms()

    def _build_transforms(self):
        """Build image and mask transform pipelines."""
        if self.mode in ("train", "train-semi", "train-semi-plus") and self.augment:
            # Post-spatial image-only transforms (colour jitter, normalise)
            self.img_post = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            self._use_augment = True
        else:
            self.img_post = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            self._use_augment = False

    @staticmethod
    def _paired_spatial(img: Image.Image, mask: Image.Image | None, size: int, augment: bool):
        """
        Apply the SAME spatial transforms to image and mask.
        Returns (img, mask) as PIL Images.
        """
        if augment:
            # --- RandomResizedCrop: sample crop params once ---
            # scale ∈ [0.5, 1.0] — upper bound ≤ 1.0 avoids out-of-bounds
            # padding that would inject undefined black borders into
            # segmentation masks (undefined behaviour for this task).
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=(0.5, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
            )
            img = TF.resized_crop(img, i, j, h, w, (size, size),
                                  interpolation=transforms.InterpolationMode.BILINEAR,
                                  antialias=True)
            if mask is not None:
                mask = TF.resized_crop(mask, i, j, h, w, (size, size),
                                       interpolation=transforms.InterpolationMode.NEAREST)

            # --- RandomHorizontalFlip: shared coin flip ---
            if random.random() < 0.5:
                img = TF.hflip(img)
                if mask is not None:
                    mask = TF.hflip(mask)
        else:
            img = TF.resize(img, size, antialias=True)
            img = TF.center_crop(img, size)
            if mask is not None:
                mask = TF.resize(mask, size, interpolation=transforms.InterpolationMode.NEAREST)
                mask = TF.center_crop(mask, size)

        return img, mask

    @staticmethod
    def _parse_mask(mask_pil: Image.Image) -> torch.Tensor:
        """
        Parse a segmentation mask PIL Image (RGB) into a LongTensor (H, W).
        class_id = R + G * 256
        0   = other category
        1000 = ignored (mapped to 255 = CrossEntropy ignore_index)
        """
        mask_np = np.array(mask_pil).astype(np.int32)
        cls = mask_np[:, :, 1] * 256 + mask_np[:, :, 0]  # R + G * 256
        cls[cls == 1000] = 255
        return torch.from_numpy(cls).long()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # ---- load image and mask as PIL Images ----
        img = Image.open(sample["image"]).convert("RGB")
        mask = None
        if sample["mask"] is not None:
            mask = Image.open(sample["mask"]).convert("RGB")

        # ---- paired spatial transforms (same crop & flip for both) ----
        img, mask_pil = self._paired_spatial(img, mask, self.size, self._use_augment)

        # ---- image-specific post-processing (colour jitter, normalise) ----
        img = self.img_post(img)

        # ---- parse mask to class-id tensor ----
        if mask_pil is not None:
            mask = self._parse_mask(mask_pil)
        # mask stays None otherwise

        out = {
            "image": img,            # (3, H, W) float32, normalized
            "class_id": sample["class_id"],
            "class_name": sample["class_name"],
            "image_path": sample["image"],
        }
        if mask is not None:
            out["mask"] = mask      # (H, W) int64
        return out


def collate_fn(batch: list) -> dict:
    """
    Custom collate function for ImageNetSDataset.
    Stacks images and optionally masks into batched tensors.
    """
    images = torch.stack([item["image"] for item in batch], dim=0)

    out = {
        "image": images,                    # (B, 3, H, W)
        "class_id": torch.tensor([item["class_id"] for item in batch]),
        "class_name": [item["class_name"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }

    if "mask" in batch[0] and batch[0]["mask"] is not None:
        masks = torch.stack([item["mask"] for item in batch], dim=0)
        out["mask"] = masks                 # (B, H, W)

    return out


def build_dataloaders(
    root: str,
    size: int = 448,
    batch_size: int = 16,
    num_workers: int = 4,
):
    """
    Convenience function to build train, val, and test dataloaders.

    Returns:
        dict with keys: 'train', 'train-semi', 'val', 'test'
        Each value is a DataLoader or None if not available.
    """
    from torch.utils.data import DataLoader

    loaders = {}

    # -- training set (full, no masks) --
    ds_train = ImageNetSDataset(root, mode="train", size=size, augment=True)
    loaders["train"] = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # -- train-semi set (with masks, for supervised fine-tuning) --
    ds_semi = ImageNetSDataset(root, mode="train-semi", size=size, augment=True)
    loaders["train-semi"] = DataLoader(
        ds_semi,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # -- validation set --
    ds_val = ImageNetSDataset(root, mode="val", size=size, augment=False)
    loaders["val"] = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # -- test set --
    ds_test = ImageNetSDataset(root, mode="test", size=size, augment=False)
    loaders["test"] = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return loaders


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/imagenet-s/ImageNetS50"

    for mode in ("train", "train-semi", "val", "test"):
        ds = ImageNetSDataset(root, mode=mode, size=448)
        print(f"[{mode}] samples: {len(ds)},  classes: {ds.num_classes}")
        if len(ds) > 0:
            item = ds[0]
            print(f"        image shape: {item['image'].shape}, has_mask: {'mask' in item}")
            if "mask" in item:
                unique = torch.unique(item["mask"])
                print(f"        mask unique ids: {unique.tolist()}")
