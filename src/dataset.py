"""Dataset and data split utilities for Microsoft Cats vs Dogs.

Expected raw layout (data/raw/PetImages/):
    data/raw/PetImages/Cat/0.jpg
    data/raw/PetImages/Dog/0.jpg
    ...
Class label is inferred from the parent folder name (Cat=0, Dog=1).

`build_splits()` produces deterministic train/val/test index lists saved to
`data/processed/splits.json` so training and evaluation see the same partitions.
Corrupt images are silently skipped during split construction.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    PROCESSED_DIR,
    RAW_DIR,
    SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

SPLITS_PATH = PROCESSED_DIR / "splits.json"


def _label_from_folder(folder_name: str) -> int:
    return CLASS_NAMES.index(folder_name.lower())


def _list_images(root: Path) -> list[Path]:
    """Collect all valid images from root/Cat/ and root/Dog/, skipping corrupt files."""
    images: list[Path] = []
    for cls in CLASS_NAMES:
        cls_dir = root / cls.capitalize()
        if not cls_dir.is_dir():
            raise FileNotFoundError(
                f"Expected folder {cls_dir}. "
                f"Extract the dataset so data/raw/PetImages/Cat/ and .../Dog/ exist."
            )
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                # Full decode (not just .verify()) catches truncated files
                # that PIL can open headers for but fails on pixel load.
                with Image.open(p) as im:
                    im.convert("RGB").load()
                images.append(p)
            except Exception:
                pass  # skip corrupt files
    return images


def build_splits(raw_dir: Path = RAW_DIR, out_path: Path = SPLITS_PATH) -> dict:
    """Create and persist deterministic train/val/test splits."""
    images = _list_images(raw_dir)
    rng = random.Random(SEED)
    indices = list(range(len(images)))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": [str(images[i].relative_to(raw_dir)) for i in indices[:n_train]],
        "val": [str(images[i].relative_to(raw_dir)) for i in indices[n_train : n_train + n_val]],
        "test": [str(images[i].relative_to(raw_dir)) for i in indices[n_train + n_val :]],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(splits, indent=2))
    return splits


def load_splits(path: Path = SPLITS_PATH) -> dict:
    if not path.exists():
        return build_splits()
    return json.loads(path.read_text())


def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CatDogDataset(Dataset):
    def __init__(self, files: Sequence[str], root: Path = RAW_DIR, transform=None):
        self.root = Path(root)
        self.files = list(files)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        rel = self.files[idx]
        img = Image.open(self.root / rel).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = _label_from_folder(Path(rel).parent.name)
        return img, label


def get_datasets(root: Path = RAW_DIR) -> dict[str, CatDogDataset]:
    splits = load_splits()
    return {
        "train": CatDogDataset(splits["train"], root=root, transform=train_transform()),
        "val": CatDogDataset(splits["val"], root=root, transform=eval_transform()),
        "test": CatDogDataset(splits["test"], root=root, transform=eval_transform()),
    }
