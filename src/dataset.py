"""
Dataset and patient-level split for LGG MRI segmentation.
"""

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import config


def _patient_dirs() -> List[Path]:
    """Return all patient folders under DATA_ROOT (folders only)."""
    return sorted([p for p in config.DATA_ROOT.iterdir() if p.is_dir()])


def _slice_pairs(patient_dir: Path) -> List[Tuple[Path, Path]]:
    """Return (image_path, mask_path) pairs for one patient folder."""
    pairs = []
    for img_path in sorted(patient_dir.glob(f"*{config.SLICE_SUFFIX}")):
        if img_path.name.endswith(config.MASK_SUFFIX):
            continue
        mask_path = img_path.with_name(img_path.stem + config.MASK_SUFFIX)
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    return pairs


def split_patients() -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Patient-level split so slices from one patient never appear in two sets.
    Seeded for reproducibility.
    """
    patients = _patient_dirs()
    rng = random.Random(config.SPLIT_SEED)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * config.TRAIN_RATIO)
    n_val   = int(n * config.VAL_RATIO)

    train = patients[:n_train]
    val   = patients[n_train:n_train + n_val]
    test  = patients[n_train + n_val:]
    return train, val, test


class LGGDataset(Dataset):
    """Returns (image float32 CHW in [0,1], mask float32 1HW in {0,1})."""

    def __init__(self, patient_dirs: List[Path], augment: bool = False):
        self.augment = augment
        self.items: List[Tuple[Path, Path]] = []
        for p in patient_dirs:
            self.items.extend(_slice_pairs(p))

    def __len__(self) -> int:
        return len(self.items)

    def _load_image(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        if config.IMAGE_SIZE is not None:
            img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    def _load_mask(self, path: Path) -> np.ndarray:
        m = Image.open(path).convert("L")
        if config.IMAGE_SIZE is not None:
            m = m.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.NEAREST)
        m = (np.asarray(m, dtype=np.float32) > 127).astype(np.float32)
        return m

    def __getitem__(self, idx: int):
        img_path, mask_path = self.items[idx]
        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        if self.augment:
            # Simple flips only: safe for brain MRI in axial view.
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if random.random() < 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

        image = np.transpose(image, (2, 0, 1))          # HWC -> CHW
        mask = np.expand_dims(mask, axis=0)             # HW -> 1HW

        return (
            torch.from_numpy(image),
            torch.from_numpy(mask),
            str(img_path),
        )