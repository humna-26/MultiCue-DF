"""
MultiCue-DF Utility Functions.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(model: nn.Module, optimizer_name: str,
                  lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = filter(lambda p: p.requires_grad, model.parameters())
    name = optimizer_name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9,
                               weight_decay=weight_decay, nesterov=True)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name!r}")


def get_scheduler(optimizer: torch.optim.Optimizer,
                  scheduler_name: str,
                  epochs: int) -> torch.optim.lr_scheduler._LRScheduler:
    name = scheduler_name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 4), gamma=0.1
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, val_acc: float, path: str | Path) -> None:
    torch.save({
        "epoch":      epoch,
        "val_acc":    val_acc,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }, str(path))


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    path: str | Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return {"epoch": ckpt["epoch"], "val_acc": ckpt["val_acc"]}


def compute_class_weights(labels) -> torch.Tensor:
    """
    Inverse-frequency class weights, normalised so they sum to n_classes.
    Suitable for nn.CrossEntropyLoss(weight=...).
    """
    labels = np.asarray(labels)
    n_classes = len(np.unique(labels))
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32)
