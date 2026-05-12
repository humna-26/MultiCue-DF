"""
MultiCue-DF Dataset Loader.
Returns (stream1_tensor, stream2_tensor, stream3_tensor, label, manipulation_type).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import CONFIG


class MultiCueDataset(Dataset):

    # Train augmentation for stream 1 & 2 (spatial streams)
    _TRAIN_S12 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Eval transform for stream 1 & 2
    _EVAL_S12 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Stream 3 (FFT) — no augmentation in any split
    _TRANSFORM_S3 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    def __init__(self, split: str = "train"):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split

        manifest_path = PROJECT_ROOT / "processed" / "manifest.csv"
        rows = []
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)

        labels      = np.array([int(r["label"])            for r in rows])
        manip_types = [r["manipulation_type"]               for r in rows]
        s1_paths    = [r["stream1_path"]                    for r in rows]
        s2_paths    = [r["stream2_path"]                    for r in rows]
        s3_paths    = [r["stream3_path"]                    for r in rows]
        indices     = np.arange(len(rows))

        seed = CONFIG["seed"]

        # Stratified 70 / 30 split, then 30 → 15 / 15
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.30, stratify=labels, random_state=seed
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50,
            stratify=labels[temp_idx], random_state=seed
        )

        sel = {"train": train_idx, "val": val_idx, "test": test_idx}[split]

        self.labels      = labels[sel].tolist()
        self.manip_types = [manip_types[i] for i in sel]
        self.s1_paths    = [s1_paths[i]    for i in sel]
        self.s2_paths    = [s2_paths[i]    for i in sel]
        self.s3_paths    = [s3_paths[i]    for i in sel]

        self._t_s12 = self._TRAIN_S12 if split == "train" else self._EVAL_S12
        self._t_s3  = self._TRANSFORM_S3

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        load = lambda p: Image.open(PROJECT_ROOT / p).convert("RGB")
        s1 = self._t_s12(load(self.s1_paths[idx]))
        s2 = self._t_s12(load(self.s2_paths[idx]))
        s3 = self._t_s3( load(self.s3_paths[idx]))
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return s1, s2, s3, label, self.manip_types[idx]

    def get_labels(self) -> np.ndarray:
        return np.array(self.labels)


def get_dataloaders(num_workers: int = 4):
    """
    Returns (train_loader, val_loader, test_loader, class_weights_tensor).
    class_weights_tensor: shape (2,) for use with nn.CrossEntropyLoss(weight=...).
    """
    train_ds = MultiCueDataset("train")
    val_ds   = MultiCueDataset("val")
    test_ds  = MultiCueDataset("test")

    # Class weights from training labels (handles real/fake imbalance)
    train_labels  = train_ds.get_labels()
    class_counts  = np.bincount(train_labels, minlength=2).astype(np.float64)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 2   # normalise to sum=2
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    bs = CONFIG["batch_size"]

    # prefetch_factor only valid when num_workers > 0
    _extra = {"prefetch_factor": 2} if num_workers > 0 else {}

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), **_extra,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), **_extra,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), **_extra,
    )

    return train_loader, val_loader, test_loader, class_weights_tensor
