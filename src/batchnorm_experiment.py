"""
BatchNorm Ablation Experiment.
Trains Stream2CNN with BN enabled (default) vs disabled.
All other hyperparameters identical (Adam, lr=1e-4, epochs=15, seed=42).
Results saved to results/batchnorm_comparison.json.

Usage: python src/batchnorm_experiment.py
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn

from config import CONFIG
from utils import set_seed
from experiment_runner import run_experiment


# ── Stream2 variant with togglable BatchNorm ──────────────────────────────────

def _he_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Stream2WithBN(nn.Module):
    """
    Stream2CNN architecture with togglable BatchNorm.
    Adds a classification head for standalone training.
    input_stream=2 tells Trainer to feed s2 (eye+mouth) to this model.
    """

    input_stream = 2

    def __init__(self, use_bn: bool = True):
        super().__init__()
        self.use_bn_flag = use_bn

        act  = lambda: nn.ReLU(inplace=True)
        norm = lambda ch: nn.BatchNorm2d(ch) if use_bn else nn.Identity()

        def conv_block(in_ch, out_ch, pool):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1), norm(out_ch), act(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), norm(out_ch), act(),
                nn.MaxPool2d(2) if pool == "max" else nn.AdaptiveAvgPool2d(1),
            ]
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3,   32,  "max"),
            conv_block(32,  64,  "max"),
            conv_block(64,  128, "max"),
            conv_block(128, 256, "avg"),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), act(), nn.Dropout(0.3),
            nn.Linear(256, 2),
        )
        self.apply(_he_init)

    def forward(self, x):
        return self.head(self.features(x))


# ── Run experiments ───────────────────────────────────────────────────────────

EPOCHS = 15
LR     = CONFIG["learning_rate"]


def main():
    set_seed(CONFIG["seed"])

    comparison = {}

    for use_bn, label in [(True, "with_bn"), (False, "without_bn")]:
        exp_name = f"bn_{label}"
        print(f"\n{'='*60}")
        print(f"  BatchNorm experiment: {label}")
        print(f"{'='*60}")

        model = Stream2WithBN(use_bn=use_bn)
        results = run_experiment(
            experiment_name=exp_name,
            model=model,
            optimizer_name="adam",
            epochs=EPOCHS,
            lr=LR,
            num_workers=0,
        )
        comparison[label] = results

    # Summary
    print("\n" + "="*65)
    print(f"  {'Config':<15} {'Test Acc':>9} {'AUC':>7} {'F1':>7}")
    print("="*65)
    for label, r in comparison.items():
        print(
            f"  {label:<15} "
            f"{r['test_accuracy']*100:>8.1f}% "
            f"{r['test_auc']:>7.3f} "
            f"{r['test_f1']:>7.3f}"
        )
    print("="*65)

    out_path = PROJECT_ROOT / CONFIG["results_dir"] / "batchnorm_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nBatchNorm comparison saved -> {out_path}")


if __name__ == "__main__":
    main()
