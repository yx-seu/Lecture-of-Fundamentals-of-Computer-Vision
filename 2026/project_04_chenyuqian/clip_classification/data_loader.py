"""Load CIFAR-10 test set. Returns raw PIL images — no CLIP preprocessing."""

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def pil_collate(batch):
    """Collate function that keeps images as PIL, labels as tensor."""
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


def get_test_loader(data_root="./data", batch_size=64, num_workers=0):
    dataset = CIFAR10(
        root=data_root,
        train=False,
        download=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pil_collate,
    )
    return loader, CIFAR10_CLASSES
