"""
Full-Model Activation Function Comparison Experiment.
Trains the full MultiCueDFModel (Stream1 ResNet18 + Stream2 CNN + Stream3 FFT + fusion)
three times, varying the activation in Stream2CNN only (relu / leaky_relu / elu).
Stream1, Stream3, attention, and the fusion head are identical to exp09.
All other hypers: Adam, lr=1e-4, dropout=0.5, epochs=15, seed=42.
Results saved to results/activation_fullmodel_comparison.json.

Usage: python src/activation_experiment_fullmodel.py
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


class Stream2WithAct(nn.Module):
    """
    Stream2CNN body with swappable activation function.
    Outputs 256-d feature vector — no classification head.
    Plug-in replacement for MultiCueDFModel.stream2.
    """

    def __init__(self, activation: str = "relu"):
        super().__init__()
        _ACT = {
            "relu":       lambda: nn.ReLU(inplace=True),
            "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.1, inplace=True),
            "elu":        lambda: nn.ELU(inplace=True),
        }
        if activation not in _ACT:
            raise ValueError(f"Unknown activation: {activation!r}")
        act = _ACT[activation]

        def conv_block(in_ch, out_ch, pool):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), act(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), act(),
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


class MultiCueDFWithActivation(MultiCueDFModel):
    """
    Full MultiCueDFModel with Stream2CNN activation swapped.
    Passes isinstance(model, MultiCueDFModel) so the Trainer feeds all 3 streams.
    """

    def __init__(self, activation: str = "relu"):
        super().__init__(dropout=0.5, use_attention=True)
        self.stream2 = Stream2WithAct(activation)


# ── Run experiments ───────────────────────────────────────────────────────────

ACTIVATIONS = ["relu", "leaky_relu", "elu"]

EPOCHS = 15
LR     = CONFIG["learning_rate"]


def main():
    set_seed(CONFIG["seed"])

    comparison = {}

    for act_name in ACTIVATIONS:
        exp_name = f"act_fullmodel_{act_name}"
        print(f"\n{'='*60}")
        print(f"  Full-Model Activation experiment: {act_name.upper()}")
        print(f"{'='*60}")

        model = MultiCueDFWithActivation(activation=act_name)
        results = run_experiment(
            experiment_name=exp_name,
            model=model,
            optimizer_name="adam",
            epochs=EPOCHS,
            lr=LR,
            num_workers=0,
        )
        comparison[act_name] = results

    print("\n" + "="*65)
    print(f"  {'Activation':<15} {'Test Acc':>9} {'AUC':>7} {'F1':>7}")
    print("="*65)
    for act_name, r in comparison.items():
        print(
            f"  {act_name:<15} "
            f"{r['test_accuracy']*100:>8.1f}% "
            f"{r['test_auc']:>7.3f} "
            f"{r['test_f1']:>7.3f}"
        )
    print("="*65)

    out_path = PROJECT_ROOT / CONFIG["results_dir"] / "activation_fullmodel_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nActivation full-model comparison saved -> {out_path}")


if __name__ == "__main__":
    main()
