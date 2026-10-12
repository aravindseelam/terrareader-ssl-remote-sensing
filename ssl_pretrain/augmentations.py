"""
Augmentation pipelines for self-supervised pretraining on remote sensing imagery.

Satellite tiles have no canonical "up" direction, so we use free rotation
(0/90/180/270) in addition to the usual SimCLR-style photometric jitter.
Color jitter strength is toned down relative to natural-image SimCLR because
satellite RGB composites are less colorful and over-aggressive jitter can
destroy the land-cover signal we want the model to learn.
"""

import random

import torch
from torchvision import transforms


class RandomDiscreteRotation:
    """Rotate by a random multiple of 90 degrees (safe for top-down imagery)."""

    def __init__(self, angles=(0, 90, 180, 270)):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        if angle == 0:
            return img
        return transforms.functional.rotate(img, angle)


def get_simclr_augmentations(image_size: int = 64):
    """Two-view augmentation pipeline used for contrastive (SimCLR) pretraining."""
    color_jitter = transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05
    )
    pipeline = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.5, 1.0), antialias=True
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomDiscreteRotation(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3444, 0.3803, 0.4078], std=[0.2037, 0.1366, 0.1148]
            ),
        ]
    )
    return pipeline


def get_eval_transform(image_size: int = 64):
    """Deterministic transform used for linear probing / fine-tuning / inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3444, 0.3803, 0.4078], std=[0.2037, 0.1366, 0.1148]
            ),
        ]
    )


class TwoCropsTransform:
    """Wraps a base transform to return two independently augmented views."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, img):
        return self.base_transform(img), self.base_transform(img)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
