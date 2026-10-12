"""
Dataset utilities built on top of torchvision's built-in EuroSAT dataset.

EuroSAT (Helber et al., 2019) is a Sentinel-2 based land-use/land-cover
classification dataset: 27,000 RGB image patches (64x64), 10 classes
(AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture,
PermanentCrop, Residential, River, SeaLake). torchvision downloads and
extracts it automatically the first time it's used, which makes it a
convenient, laptop-friendly stand-in for a larger Sentinel-2 corpus.

Swap `EuroSAT` for your own `ImageFolder`-style directory of unlabeled tiles
once you're ready to scale up to real Sentinel-2/Landsat imagery (see
data/download_sentinel2.py for a Google-Earth-Engine based downloader).
"""

import random

import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import EuroSAT

from ssl_pretrain.augmentations import TwoCropsTransform, get_simclr_augmentations


CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def get_raw_eurosat(root: str = "./data/eurosat", download: bool = True):
    """Returns the raw EuroSAT dataset (PIL images, integer labels)."""
    return EuroSAT(root=root, download=download, transform=None)


class SSLDataset(Dataset):
    """Wraps EuroSAT to return two augmented views of each image (labels discarded).

    This is what the SSL pretraining stage consumes -- pretend the labels
    don't exist, because in the real low-label setting they mostly don't.
    """

    def __init__(self, root: str = "./data/eurosat", image_size: int = 64, download: bool = True):
        self.base = EuroSAT(root=root, download=download, transform=None)
        self.transform = TwoCropsTransform(get_simclr_augmentations(image_size))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        view1, view2 = self.transform(img.convert("RGB"))
        return view1, view2


class LabeledSubsetDataset(Dataset):
    """Wraps EuroSAT with a deterministic transform and integer labels, for
    linear probing / fine-tuning / evaluation."""

    def __init__(self, root: str, transform, indices=None, download: bool = True):
        self.base = EuroSAT(root=root, download=download, transform=None)
        self.transform = transform
        self.indices = indices if indices is not None else list(range(len(self.base)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.base[real_idx]
        img = self.transform(img.convert("RGB"))
        return img, label


def make_splits(root: str = "./data/eurosat", seed: int = 42, download: bool = True):
    """Creates a fixed, reproducible train/val/test split (70/10/20) at the
    index level, so every script in this project shares the same split.
    """
    base = EuroSAT(root=root, download=download, transform=None)
    n = len(base)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(0.70 * n)
    n_val = int(0.10 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def subsample_labels(indices, fraction: float, seed: int = 42):
    """Randomly keeps only `fraction` of the given indices -- used to simulate
    the low-label regime during fine-tuning (e.g. fraction=0.05 for 5% labels).
    """
    if fraction >= 1.0:
        return indices
    rng = random.Random(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    n_keep = max(1, int(len(shuffled) * fraction))
    return shuffled[:n_keep]
