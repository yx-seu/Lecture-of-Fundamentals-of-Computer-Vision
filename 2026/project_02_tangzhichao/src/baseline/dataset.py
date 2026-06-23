from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import SegmentationTransform
from .utils import log


_MASK_VALUES_LOGGED = False


class ImageNetSSegmentationDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        image_processor,
        image_size,
        ignore_index,
        class_names,
        mask_mode="multiclass",
        samples_per_class=None,
        use_validation_for_train=False,
        validation_holdout_per_class=0,
        validation_holdout_stems=None,
        validation_holdout_only=False,
        return_original=False,
    ):
        self.root = Path(root)
        self.split = split
        self.image_processor = image_processor
        self.image_size = image_size
        self.ignore_index = ignore_index
        self.class_names = list(class_names)
        self.mask_mode = mask_mode
        self.use_validation_for_train = use_validation_for_train
        self.validation_holdout_per_class = int(validation_holdout_per_class or 0)
        self.validation_holdout_stems = validation_holdout_stems or {}
        self.validation_holdout_only = validation_holdout_only
        self.class_to_label = {class_name: idx + 1 for idx, class_name in enumerate(self.class_names)}
        self.return_original = return_original
        self.transform = SegmentationTransform(image_size=image_size, is_train=split == "train")
        self.samples = self._collect_samples(samples_per_class=samples_per_class)

        if not self.samples:
            raise RuntimeError(f"No ImageNet-S samples found for split={split} under {self.root}")

    def _split_dirs(self):
        if self.split == "train":
            return self.root / "train-semi", self.root / "train-semi-segmentation"
        if self.split in {"val", "validation", "test"}:
            return self.root / "validation", self.root / "validation-segmentation"
        raise ValueError(f"Unsupported ImageNet-S split: {self.split}")

    def _collect_from_dirs(self, image_root, mask_root, class_name, image_paths):
        samples = []
        for image_path in image_paths:
            mask_path = mask_root / class_name / f"{image_path.stem}.png"
            if mask_path.exists():
                samples.append((image_path, mask_path, class_name))
        return samples

    def _validation_paths_for_class(self, class_name):
        validation_image_dir = self.root / "validation" / class_name
        return sorted(validation_image_dir.glob("*.JPEG"))

    def _holdout_paths_for_class(self, class_name, val_paths):
        requested_stem = self.validation_holdout_stems.get(class_name)
        if requested_stem:
            requested = [path for path in val_paths if path.stem == requested_stem]
            if requested:
                return requested
            log(f"Requested holdout stem {requested_stem} not found for {class_name}; using default holdout.")
        return val_paths[: self.validation_holdout_per_class]

    def _collect_samples(self, samples_per_class=None):
        samples = []
        for class_name in self.class_names:
            if self.split == "train":
                train_image_root, train_mask_root = self.root / "train-semi", self.root / "train-semi-segmentation"
                train_paths = sorted((train_image_root / class_name).glob("*.JPEG"))
                if samples_per_class is not None:
                    train_paths = train_paths[:samples_per_class]
                samples.extend(self._collect_from_dirs(train_image_root, train_mask_root, class_name, train_paths))

                if self.use_validation_for_train:
                    val_paths = self._validation_paths_for_class(class_name)
                    holdout_stems = {path.stem for path in self._holdout_paths_for_class(class_name, val_paths)}
                    if holdout_stems:
                        val_paths = [path for path in val_paths if path.stem not in holdout_stems]
                    samples.extend(
                        self._collect_from_dirs(
                            self.root / "validation",
                            self.root / "validation-segmentation",
                            class_name,
                            val_paths,
                        )
                    )
                continue

            image_root, mask_root = self._split_dirs()
            image_paths = sorted((image_root / class_name).glob("*.JPEG"))
            if self.validation_holdout_only and self.validation_holdout_per_class > 0:
                image_paths = self._holdout_paths_for_class(class_name, image_paths)
            elif samples_per_class is not None:
                image_paths = image_paths[:samples_per_class]
            samples.extend(self._collect_from_dirs(image_root, mask_root, class_name, image_paths))
        return samples

    def __len__(self):
        return len(self.samples)

    def _convert_mask(self, mask, class_name):
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        converted = np.zeros(mask.shape, dtype=np.int64)
        if self.mask_mode == "binary_foreground":
            converted[mask > 0] = 1
        elif self.mask_mode == "multiclass":
            converted[mask > 0] = self.class_to_label[class_name]
        else:
            raise ValueError(f"Unsupported ImageNet-S mask mode: {self.mask_mode}")
        return converted

    def __getitem__(self, idx):
        global _MASK_VALUES_LOGGED

        image_path, mask_path, class_name = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        raw_mask = Image.open(mask_path)

        if not _MASK_VALUES_LOGGED:
            unique_values = np.unique(np.asarray(raw_mask)).tolist()
            log(f"Raw ImageNet-S mask unique values: {unique_values[:20]}")
            _MASK_VALUES_LOGGED = True

        image, resized_raw_mask = self.transform(image, raw_mask)
        converted_mask = self._convert_mask(resized_raw_mask, class_name=class_name)
        processor_outputs = self.image_processor(images=image, return_tensors="pt")

        item = {
            "pixel_values": processor_outputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(converted_mask, dtype=torch.long),
            "image_id": image_path.stem,
        }
        if self.return_original:
            item["original_image"] = np.asarray(image)
            item["original_mask"] = converted_mask
        return item


def _collate_fn(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "image_id": [item["image_id"] for item in batch],
    }


def build_dataloaders(config, image_processor):
    data_cfg = config["data"]
    batch_size = config["train"]["batch_size"]

    if data_cfg["dataset"] != "imagenet_s":
        raise ValueError("This project is simplified for ImageNet-S only. Set data.dataset: imagenet_s.")

    root = Path(data_cfg["root"])
    all_classes = sorted([path.name for path in (root / "train-semi").iterdir() if path.is_dir()])
    selected_classes = data_cfg.get("class_names") or all_classes[: data_cfg.get("num_classes", 5)]
    config["data"]["selected_classes"] = selected_classes
    mask_mode = data_cfg.get("mask_mode", "multiclass")
    if mask_mode == "binary_foreground":
        config["model"]["num_labels"] = 2
        config["model"]["id2label"] = {0: "background", 1: "object"}
    else:
        config["model"]["num_labels"] = len(selected_classes) + 1
        config["model"]["id2label"] = {
            0: "background",
            **{idx + 1: name for idx, name in enumerate(selected_classes)},
        }
    config["model"]["label2id"] = {v: k for k, v in config["model"]["id2label"].items()}

    train_dataset = ImageNetSSegmentationDataset(
        root=root,
        split="train",
        image_processor=image_processor,
        image_size=data_cfg["image_size"],
        ignore_index=data_cfg["ignore_index"],
        class_names=selected_classes,
        mask_mode=mask_mode,
        samples_per_class=data_cfg.get("samples_per_class_train"),
        use_validation_for_train=data_cfg.get("use_validation_for_train", False),
        validation_holdout_per_class=data_cfg.get("validation_holdout_per_class", 0),
        validation_holdout_stems=data_cfg.get("validation_holdout_stems", {}),
    )
    val_dataset = ImageNetSSegmentationDataset(
        root=root,
        split="validation",
        image_processor=image_processor,
        image_size=data_cfg["image_size"],
        ignore_index=data_cfg["ignore_index"],
        class_names=selected_classes,
        mask_mode=mask_mode,
        samples_per_class=data_cfg.get("samples_per_class_val"),
        validation_holdout_per_class=data_cfg.get("validation_holdout_per_class", 0),
        validation_holdout_stems=data_cfg.get("validation_holdout_stems", {}),
        validation_holdout_only=data_cfg.get("use_validation_for_train", False),
    )
    test_dataset = val_dataset

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": data_cfg["num_workers"],
        "pin_memory": data_cfg["pin_memory"],
        "collate_fn": _collate_fn,
    }

    log(f"ImageNet-S classes: {selected_classes}")
    log(f"ImageNet-S train samples={len(train_dataset)}, val samples={len(val_dataset)}")
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    )


def build_visualization_dataset(config, image_processor, split, indices=None):
    data_cfg = config["data"]
    if data_cfg["dataset"] != "imagenet_s":
        raise ValueError("This project is simplified for ImageNet-S only. Set data.dataset: imagenet_s.")

    selected_classes = data_cfg.get("selected_classes") or data_cfg.get("class_names")
    if selected_classes is None:
        root = Path(data_cfg["root"])
        all_classes = sorted([path.name for path in (root / "train-semi").iterdir() if path.is_dir()])
        selected_classes = all_classes[: data_cfg.get("num_classes", 5)]
    split_name = "train" if split in {"train", data_cfg.get("train_split")} else "validation"
    return ImageNetSSegmentationDataset(
        root=data_cfg["root"],
        split=split_name,
        image_processor=image_processor,
        image_size=data_cfg["image_size"],
        ignore_index=data_cfg["ignore_index"],
        class_names=selected_classes,
        mask_mode=data_cfg.get("mask_mode", "multiclass"),
        samples_per_class=data_cfg.get("samples_per_class_val"),
        use_validation_for_train=data_cfg.get("use_validation_for_train", False) and split_name == "train",
        validation_holdout_per_class=data_cfg.get("validation_holdout_per_class", 0),
        validation_holdout_stems=data_cfg.get("validation_holdout_stems", {}),
        validation_holdout_only=data_cfg.get("use_validation_for_train", False) and split_name != "train",
        return_original=True,
    )
