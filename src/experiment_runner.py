"""
MultiCue-DF Experiment Runner.
Runs all 12 ablation experiments sequentially and prints a master summary table.

Usage (from multicue-df/ root):
    python src/experiment_runner.py
"""

import sys
import csv
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn

from config import CONFIG
from dataset import get_dataloaders
from models import (MLPBaseline, SingleCNNBaseline, MultiCueDFModel)
from utils import get_optimizer, get_scheduler, count_parameters, load_checkpoint
from trainer import Trainer


# ── Core experiment function ──────────────────────────────────────────────────

def run_experiment(
    experiment_name: str,
    model: nn.Module,
    optimizer_name: str = "adam",
    epochs: int = None,
    lr: float = None,
    scheduler_name: str = "cosine",
    num_workers: int = 4,
) -> dict:
    """
    Train model, evaluate on test set, save results.
    Returns results dict.
    """
    if epochs is None:
        epochs = CONFIG["epochs"]
    if lr is None:
        lr = CONFIG["learning_rate"]

    start_time = time.time()
    device = "cuda"

    print(f"\n{'#'*90}")
    print(f"  STARTING: {experiment_name}")
    print(f"  optimizer={optimizer_name}  epochs={epochs}  lr={lr}  device={device}")
    print(f"{'#'*90}")

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        num_workers=num_workers
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = get_optimizer(model, optimizer_name, lr, CONFIG["weight_decay"])
    scheduler = get_scheduler(optimizer, scheduler_name, epochs)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        experiment_name=experiment_name,
        device=device,
        config=CONFIG,
        epochs=epochs,
    )

    history, best_val_acc, epochs_trained = trainer.fit()

    # Load best checkpoint weights back into model for test evaluation
    ckpt_path = PROJECT_ROOT / CONFIG["checkpoints_dir"] / f"{experiment_name}_best.pth"
    if ckpt_path.exists():
        load_checkpoint(trainer.model, None, ckpt_path)

    test_loss, test_acc, test_auc, test_f1 = trainer.validate(test_loader)
    elapsed_min = (time.time() - start_time) / 60.0

    results = {
        "experiment_name":       experiment_name,
        "test_accuracy":         round(test_acc,    6),
        "test_auc":              round(test_auc,    6),
        "test_f1":               round(test_f1,     6),
        "total_params":          count_parameters(model),
        "best_val_accuracy":     round(best_val_acc, 6),
        "epochs_trained":        epochs_trained,
        "training_time_minutes": round(elapsed_min,  2),
    }

    results_dir = PROJECT_ROOT / CONFIG["results_dir"]
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"{experiment_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"\nTest -> Acc: {test_acc*100:.1f}%  AUC: {test_auc:.3f}  "
        f"F1: {test_f1:.3f}  Time: {elapsed_min:.1f} min"
    )
    print(f"Results saved -> {results_path}")

    return results


# ── Experiment definitions (lazy model instantiation via lambdas) ─────────────

EXPERIMENTS = [
    # 1. MLP Baseline — flattened spatial features (capped at 10: won't improve further)
    {
        "name":           "exp01_mlp_baseline",
        "model_fn":       lambda: MLPBaseline(),
        "optimizer_name": "adam",
        "epochs":         10,
    },
    # 2. Single CNN Baseline
    {
        "name":           "exp02_single_cnn",
        "model_fn":       lambda: SingleCNNBaseline(),
        "optimizer_name": "adam",
        "epochs":         20,
    },
    # 3. Stream 1 alone (ResNet-18 full face)
    {
        "name":           "exp03_stream1_only",
        "model_fn":       lambda: MultiCueDFModel(use_stream1=True,  use_stream2=False, use_stream3=False),
        "optimizer_name": "adam",
        "epochs":         20,
    },
    # 4. Stream 2 alone (eye+mouth CNN)
    {
        "name":           "exp04_stream2_only",
        "model_fn":       lambda: MultiCueDFModel(use_stream1=False, use_stream2=True,  use_stream3=False),
        "optimizer_name": "adam",
        "epochs":         20,
    },
    # 5. Stream 3 alone (FFT CNN)
    {
        "name":           "exp05_stream3_only",
        "model_fn":       lambda: MultiCueDFModel(use_stream1=False, use_stream2=False, use_stream3=True),
        "optimizer_name": "adam",
        "epochs":         20,
    },
    # 6. Stream 1+2 (spatial only, no frequency)
    {
        "name":           "exp06_stream1_stream2",
        "model_fn":       lambda: MultiCueDFModel(use_stream1=True,  use_stream2=True,  use_stream3=False),
        "optimizer_name": "adam",
        "epochs":         20,
    },
    # 7. Stream 1+3 (full face + frequency, no region crops)
    {
        "name":           "exp07_stream1_stream3",
        "model_fn":       lambda: MultiCueDFModel(use_stream1=True,  use_stream2=False, use_stream3=True),
        "optimizer_name": "adam",
        "epochs":         20,
    },
    # 8. Stream 2+3 (region crops + frequency, no full face)
    {
        "name":           "exp08_stream2_stream3",
        "model_fn":       lambda: MultiCueDFModel(use_stream1=False, use_stream2=True,  use_stream3=True),
        "optimizer_name": "adam",
        "epochs":         20,
    },
    # 9. Full model — Adam, dropout=0.5 (main model)
    {
        "name":           "exp09_full_model_adam",
        "model_fn":       lambda: MultiCueDFModel(dropout=0.5),
        "optimizer_name": "adam",
        "epochs":         20,
    },
    # 10. Full model — SGD+momentum (optimizer comparison)
    {
        "name":           "exp10_full_model_sgd",
        "model_fn":       lambda: MultiCueDFModel(dropout=0.5),
        "optimizer_name": "sgd",
        "epochs":         20,
    },
    # 11. Full model — dropout=0.3 (dropout ablation)
    {
        "name":           "exp11_full_model_dropout03",
        "model_fn":       lambda: MultiCueDFModel(dropout=0.3),
        "optimizer_name": "adam",
        "epochs":         20,
    },
    # 12. Full model — no attention (attention ablation)
    {
        "name":           "exp12_full_model_no_attention",
        "model_fn":       lambda: MultiCueDFModel(dropout=0.5, use_attention=False),
        "optimizer_name": "adam",
        "epochs":         20,
    },
]


# ── Summary helpers ───────────────────────────────────────────────────────────

def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def print_and_save_summary(all_results: list[dict]) -> None:
    print("\n" + "="*95)
    print(f"  {'Experiment':<35} {'Test Acc':>9} {'AUC':>7} {'F1':>7} {'Params':>9} {'Time(min)':>10}")
    print("="*95)
    for r in all_results:
        print(
            f"  {r['experiment_name']:<35} "
            f"{r['test_accuracy']*100:>8.1f}% "
            f"{r['test_auc']:>7.3f} "
            f"{r['test_f1']:>7.3f} "
            f"{_fmt_params(r['total_params']):>9} "
            f"{r['training_time_minutes']:>10.1f}"
        )
    print("="*95)

    # Save CSV
    results_dir = PROJECT_ROOT / CONFIG["results_dir"]
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "ablation_summary.csv"
    fields = ["experiment_name", "test_accuracy", "test_auc", "test_f1",
              "total_params", "best_val_accuracy", "epochs_trained",
              "training_time_minutes"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in fields})
    print(f"\nAblation summary saved -> {csv_path}")


# ── CSV append helper (crash-safe: writes each result immediately) ─────────────

CSV_FIELDS = ["experiment_name", "test_accuracy", "test_auc", "test_f1",
              "total_params", "best_val_accuracy", "epochs_trained",
              "training_time_minutes"]


def append_result_to_csv(results: dict) -> None:
    csv_path = PROJECT_ROOT / CONFIG["results_dir"] / "ablation_summary.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: results[k] for k in CSV_FIELDS})


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils import set_seed
    set_seed(CONFIG["seed"])

    n_total    = len(EXPERIMENTS)
    all_results = []

    for idx, exp in enumerate(EXPERIMENTS, start=1):
        # ── Progress banner ───────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Starting Experiment {idx}/{n_total}: {exp['name']}")
        print(f"{'='*60}")

        model   = exp["model_fn"]()
        results = run_experiment(
            experiment_name=exp["name"],
            model=model,
            optimizer_name=exp.get("optimizer_name", "adam"),
            epochs=exp.get("epochs", CONFIG["epochs"]),
            num_workers=4,
        )
        all_results.append(results)

        # Append immediately — protects completed results if run crashes later
        append_result_to_csv(results)
        print(f"  -> Result appended to ablation_summary.csv  ({idx}/{n_total} done)")

    print_and_save_summary(all_results)
    print("\nAll experiments complete.")
