"""
MultiCue-DF Trainer.
Handles training for MLPBaseline, SingleCNNBaseline, and all MultiCueDFModel variants.
Model type is detected automatically via isinstance() and the `input_stream` attribute.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

from models import MultiCueDFModel
from utils import save_checkpoint


class Trainer:

    EARLY_STOP_PATIENCE = 5

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        experiment_name: str,
        device: str,
        config: dict,
        epochs: int = None,
    ):
        self.model           = model.to(device)
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.optimizer       = optimizer
        self.scheduler       = scheduler
        self.criterion       = criterion
        self.experiment_name = experiment_name
        self.device          = device
        self.config          = config
        self.epochs          = epochs if epochs is not None else config["epochs"]

        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "val_auc":    [], "val_f1":    [],
        }
        self.best_val_acc = 0.0
        self.best_epoch   = 0
        self._no_improve  = 0

    # ── Forward dispatch ──────────────────────────────────────────────────────
    # MultiCueDFModel → all 3 streams
    # Models with input_stream=2 → s2 (eye+mouth experiments)
    # Models with input_stream=3 → s3 (FFT experiments)
    # Everything else            → s1

    def _forward(self, s1, s2, s3):
        if isinstance(self.model, MultiCueDFModel):
            return self.model(s1, s2, s3)
        stream = getattr(self.model, "input_stream", 1)
        if stream == 2:
            return self.model(s2)
        if stream == 3:
            return self.model(s3)
        return self.model(s1)

    # ── Train one epoch ───────────────────────────────────────────────────────

    def train_one_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for s1, s2, s3, labels, _ in self.train_loader:
            s1     = s1.to(self.device, non_blocking=True)
            s2     = s2.to(self.device, non_blocking=True)
            s3     = s3.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self._forward(s1, s2, s3)
            loss   = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            bs          = labels.size(0)
            total_loss += loss.item() * bs
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += bs

        return total_loss / total, correct / total

    # ── Validate / test ───────────────────────────────────────────────────────

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        # Accumulate on GPU; single .cpu() transfer after the loop
        gpu_labels, gpu_probs, gpu_preds = [], [], []

        with torch.no_grad():
            for s1, s2, s3, labels, _ in loader:
                s1     = s1.to(self.device, non_blocking=True)
                s2     = s2.to(self.device, non_blocking=True)
                s3     = s3.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self._forward(s1, s2, s3)
                loss   = self.criterion(logits, labels)

                bs = labels.size(0)
                total_loss += loss.item() * bs
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total   += bs

                gpu_labels.append(labels)
                gpu_probs.append(probs)
                gpu_preds.append(preds)

        # One transfer per epoch instead of one per batch
        all_labels = torch.cat(gpu_labels).cpu().numpy().tolist()
        all_probs  = torch.cat(gpu_probs).cpu().numpy().tolist()
        all_preds  = torch.cat(gpu_preds).cpu().numpy().tolist()

        avg_loss = total_loss / total
        accuracy = correct / total

        try:
            auc = float(roc_auc_score(all_labels, all_probs))
        except Exception:
            auc = 0.5
        try:
            f1 = float(f1_score(all_labels, all_preds, zero_division=0))
        except Exception:
            f1 = 0.0

        return avg_loss, accuracy, auc, f1

    # ── Full training loop ────────────────────────────────────────────────────

    def fit(self):
        logs_dir = PROJECT_ROOT / self.config["logs_dir"]
        ckpt_dir = PROJECT_ROOT / self.config["checkpoints_dir"]
        logs_dir.mkdir(exist_ok=True)
        ckpt_dir.mkdir(exist_ok=True)

        print(f"\n{'='*90}")
        print(f"  {self.experiment_name}  |  epochs={self.epochs}  |  device={self.device}")
        print(f"{'='*90}")

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc, val_auc, val_f1 = self.validate()

            if isinstance(self.scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()

            self.history["train_loss"].append(round(train_loss, 6))
            self.history["train_acc"].append(round(train_acc,  6))
            self.history["val_loss"].append(round(val_loss,   6))
            self.history["val_acc"].append(round(val_acc,    6))
            self.history["val_auc"].append(round(val_auc,    6))
            self.history["val_f1"].append(round(val_f1,     6))

            print(
                f"Epoch {epoch:02d}/{self.epochs:02d} | "
                f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.1f}% | "
                f"AUC: {val_auc:.3f} | F1: {val_f1:.3f}"
            )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch   = epoch
                self._no_improve  = 0
                ckpt_path = ckpt_dir / f"{self.experiment_name}_best.pth"
                save_checkpoint(self.model, self.optimizer, epoch, val_acc, ckpt_path)
                print(f"           -> checkpoint saved  val_acc={val_acc*100:.2f}%")
            else:
                self._no_improve += 1
                if self._no_improve >= self.EARLY_STOP_PATIENCE:
                    print(f"           -> early stop ({self.EARLY_STOP_PATIENCE} epochs without improvement)")
                    break

        history_path = logs_dir / f"{self.experiment_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        epochs_trained = len(self.history["train_loss"])
        print(
            f"\nFinished: best val_acc={self.best_val_acc*100:.2f}% "
            f"(epoch {self.best_epoch}) | history -> {history_path}"
        )
        return self.history, self.best_val_acc, epochs_trained
