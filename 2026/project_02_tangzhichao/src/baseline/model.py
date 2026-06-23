from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from .own_segformer import BasicImageProcessor, OwnSegformerForImageNetS


def build_image_processor(config):
    if config["model"].get("use_own_segformer", False):
        return BasicImageProcessor()
    return SegformerImageProcessor.from_pretrained(
        config["model"]["name"],
        do_reduce_labels=False,
        do_resize=False,
        do_rescale=True,
        do_normalize=True,
    )


def build_model(config):
    if config["model"].get("use_own_segformer", False):
        return OwnSegformerForImageNetS.from_ade_checkpoint(config["model"]["name"], config)

    model = SegformerForSemanticSegmentation.from_pretrained(
        config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        id2label={int(k): v for k, v in config["model"]["id2label"].items()},
        label2id=config["model"]["label2id"],
        ignore_mismatched_sizes=True,
    )
    model.config.semantic_loss_ignore_index = config["data"]["ignore_index"]
    model.config.do_reduce_labels = False
    return model


def save_model(model, image_processor, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    image_processor.save_pretrained(save_dir)


def load_model(model_dir, device):
    model_dir = Path(model_dir)
    if (model_dir / "own_segformer_config.json").exists():
        model = OwnSegformerForImageNetS.from_pretrained(model_dir, device=device)
        image_processor = BasicImageProcessor.from_pretrained(model_dir)
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
        image_processor = SegformerImageProcessor.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, image_processor
