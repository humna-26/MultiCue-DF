"""
Full-Model BatchNorm Ablation Experiment.
Trains the full MultiCueDFModel (Stream1 ResNet18 + Stream2 CNN + Stream3 FFT + fusion)
twice, varying BatchNorm in Stream2CNN only (with_bn / without_bn).
Stream1, Stream3, attention, and the fusion head are identical to exp09.
All other hypers: Adam, lr=1e-4, dropout=0.5, epochs=15, seed=42.
Results saved to results/batchnorm_fullmodel_comparison.json.

Usage: python src/batchnorm_experiment_fullmodel.py
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch.nn as nn

from config import CONFIG
from utils import set_seed
from models import MultiCueDFModel
from experiment_runner import run_experiment


def _he_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Stream2WithBN(nn.Module):
    """
    Stream2CNN body with togglable BatchNorm.
    Outputs 256-d feature vector — no classification head.
    Plug-in replacement for MultiCueDFModel.stream2.
    """

    def __init__(self, use_bn: bool = True):
        super().__init__()
        act  = lambda: nn.ReLU(inplace=True)
        norm = lambda ch: nn.BatchNorm2d(ch) if use_bn else nn.Identity()

        def conv_block(in_ch, out_ch, pool):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1), norm(out_ch), act(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), norm(out_ch), act(),
                nn.MaxPool2d(2) if pool == "max" else nn.AdaptiveAvgPool2d(1),
            ]
            return nn.Sequential(*layers)

        self.block1 = conv_block(3,   32,  "max")
        self.block2 = conv_block(32,  64,  "max")
        self.block3 = conv_block(64,  128, "max")
        self.block4 = conv_block(128, 256, "avg")
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), act(), nn.Dropout(0.3),
        )
        self.apply(_he_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x)  # (B, 256)


class MultiCueDFWithBN(MultiCueDFModel):
    """
    Full MultiCueDFModel with Stream2CNN BatchNorm toggled.
    Passes isinstance(model, MultiCueDFModel) so the Trainer feeds all 3 streams.
    """

    def __init__(self, use_bn: bool = True):
        super().__init__(dropout=0.5, use_attention=True)
        self.stream2 = Stream2WithBN(use_bn=use_bn)


# ── Run experiments ───────────────────────────────────────────────────────────

EPOCHS = 15
LR     = CONFIG["learning_rate"]


def main():
    set_seed(CONFIG["seed"])

    comparison = {}

    for use_bn, label in [(True, "with_bn"), (False, "without_bn")]:
        exp_name = f"bn_fullmodel_{label}"
        print(f"\n{'='*60}")
        print(f"  Full-Model BatchNorm experiment: {label}")
        print(f"{'='*60}")

        model = MultiCueDFWithBN(use_bn=use_bn)
        results = run_experiment(
            experiment_name=exp_name,
            model=model,
            optimizer_name="adam",
            epochs=EPOCHS,
            lr=LR,
            num_workers=0,
        )
        comparison[label] = results

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

    out_path = PROJECT_ROOT / CONFIG["results_dir"] / "batchnorm_fullmodel_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nBatchNorm full-model comparison saved -> {out_path}")


if __name__ == "__main__":
    main()
