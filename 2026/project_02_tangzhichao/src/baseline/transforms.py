import random

import numpy as np
from PIL import Image, ImageEnhance


class SegmentationTransform:
    def __init__(self, image_size: int, is_train: bool):
        self.image_size = image_size
        self.is_train = is_train

    def _color_jitter(self, image: Image.Image) -> Image.Image:
        enhancers = [
            (ImageEnhance.Brightness, (0.9, 1.1)),
            (ImageEnhance.Contrast, (0.9, 1.1)),
            (ImageEnhance.Color, (0.9, 1.1)),
        ]
        for enhancer_cls, scale_range in enhancers:
            factor = random.uniform(*scale_range)
            image = enhancer_cls(image).enhance(factor)
        return image

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = image.convert("RGB")
        mask = mask.copy()

        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)

        if self.is_train and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if self.is_train:
            image = self._color_jitter(image)

        mask_array = np.asarray(mask, dtype=np.int64)
        return image, mask_array
