"""
2-epoch smoke test for exp01_mlp_baseline with num_workers=4.
Must be run as a top-level script (not -c) so Windows multiprocessing spawn works.

Usage: python src/smoke_test.py
"""

import sys
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    from utils import set_seed
    from models import MLPBaseline
    from experiment_runner import run_experiment

    set_seed(42)

    print("=== SMOKE TEST: exp01_mlp_baseline, 2 epochs, num_workers=4 ===\n")
    t0 = time.time()

    model = MLPBaseline()
    results = run_experiment(
        experiment_name="exp01_mlp_baseline",
        model=model,
        optimizer_name="adam",
        epochs=2,
        num_workers=4,
    )

    elapsed = (time.time() - t0) / 60.0

    print("\n--- Verifying output files ---")
    root = PROJECT_ROOT
    files = {
        "Checkpoint": root / "checkpoints" / "exp01_mlp_baseline_best.pth",
        "History":    root / "logs"        / "exp01_mlp_baseline_history.json",
        "Results":    root / "results"     / "exp01_mlp_baseline_results.json",
    }
    for label, path in files.items():
        print(f"  {label:<12}: {'EXISTS' if path.exists() else 'MISSING'}  ({path.name})")

    print("\n--- History (2 epochs) ---")
    with open(files["History"]) as f:
        h = json.load(f)
    for k, v in h.items():
        print(f"  {k}: {v}")

    print("\n--- Results JSON ---")
    for k, v in results.items():
        print(f"  {k}: {v}")

    print(f"\nTotal wall-clock time: {elapsed:.1f} min")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
