"""
train.py
--------
Full training loop with:
    - MLflow experiment tracking
    - Early stopping
    - Cosine LR scheduler
    - Gradient clipping
    - Per-epoch metrics (RMSE, MAE, class accuracy, per-class recall)
    - Model checkpointing
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, Dict, Optional
import yaml
import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score
)

from dataset import preprocess
from model import build_model, build_loss, CMAPSS_CNN_LSTM, CMAPSSLoss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rul_metrics(
    preds: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    RMSE, MAE, and NASA score function.
    NASA score penalises late predictions more than early ones.
    """
    errors = preds - targets

    rmse = np.sqrt(np.mean(errors ** 2))
    mae  = np.mean(np.abs(errors))

    # NASA score function
    score = np.sum(
        np.where(errors < 0,
                 np.exp(-errors / 13) - 1,
                 np.exp(errors / 10) - 1)
    )

    return {"rmse": rmse, "mae": mae, "nasa_score": score}


def compute_class_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 4
) -> Dict[str, float]:
    """Accuracy, per-class recall, macro F1."""
    accuracy = (preds == targets).mean()

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))

    per_class_recall = {}
    class_names = ["healthy", "degrading", "warning", "critical"]
    for i, name in enumerate(class_names):
        if cm[i].sum() > 0:
            per_class_recall[f"recall_{name}"] = cm[i, i] / cm[i].sum()
        else:
            per_class_recall[f"recall_{name}"] = 0.0

    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)

    metrics = {
        "accuracy":  float(accuracy),
        "macro_f1":  float(macro_f1),
        **{k: float(v) for k, v in per_class_recall.items()}
    }
    return metrics


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model:       CMAPSS_CNN_LSTM,
    loader:      DataLoader,
    criterion:   CMAPSSLoss,
    optimizer:   Optional[torch.optim.Optimizer],
    device:      torch.device,
    training:    bool
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one epoch (train or eval).

    Returns:
        metrics:        dict of loss components
        rul_preds:      predicted RUL values
        rul_targets:    true RUL values
        class_preds:    predicted class indices
        class_targets:  true class indices
    """
    model.train() if training else model.eval()

    total_loss_sum = 0.0
    mse_sum        = 0.0
    focal_sum      = 0.0

    all_rul_preds    = []
    all_rul_targets  = []
    all_class_preds  = []
    all_class_targets = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for X, static, y_rul, y_class in loader:
            X        = X.to(device)
            static   = static.to(device)
            y_rul    = y_rul.to(device)
            y_class  = y_class.to(device)

            rul_pred, class_logits, _, _ = model(X, static)

            total, mse, focal = criterion(
                rul_pred, y_rul, class_logits, y_class
            )

            if training:
                optimizer.zero_grad()
                total.backward()
                # Gradient clipping — prevents LSTM exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss_sum += total.item()
            mse_sum        += mse.item()
            focal_sum      += focal.item()

            # Collect predictions
            all_rul_preds.append(
                rul_pred.squeeze(-1).detach().cpu().numpy()
            )
            all_rul_targets.append(y_rul.detach().cpu().numpy())
            all_class_preds.append(
                class_logits.argmax(dim=-1).detach().cpu().numpy()
            )
            all_class_targets.append(y_class.detach().cpu().numpy())

    n_batches = len(loader)
    loss_metrics = {
        "total_loss": total_loss_sum / n_batches,
        "mse_loss":   mse_sum        / n_batches,
        "focal_loss": focal_sum      / n_batches
    }

    rul_preds     = np.concatenate(all_rul_preds)
    rul_targets   = np.concatenate(all_rul_targets)
    class_preds   = np.concatenate(all_class_preds)
    class_targets = np.concatenate(all_class_targets)

    return loss_metrics, rul_preds, rul_targets, class_preds, class_targets


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.stop       = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter    = 0
        return self.stop


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config_path: str = "configs/config.yaml"):
    cfg  = load_config(config_path)
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    mlcfg = cfg["mlflow"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n[INFO] Preprocessing data...")
    train_loader, val_loader, test_loader, class_weights, artifacts = preprocess(
        config_path=config_path
    )
    class_weights = class_weights.to(device)

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(config_path).to(device)
    print(f"\n[INFO] Model parameters: {model.count_parameters():,}")

    criterion = build_loss(class_weights, config_path).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"]
    )

    if tcfg["lr_scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=tcfg["epochs"],
            eta_min=tcfg["lr_min"]
        )
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    early_stopping = EarlyStopping(
        patience=tcfg["patience"],
        min_delta=tcfg["min_delta"]
    )

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(mlcfg["tracking_uri"])
    mlflow.set_experiment(mlcfg["experiment_name"])

    with mlflow.start_run():

        # Log config
        mlflow.log_params({
            "epochs":         tcfg["epochs"],
            "batch_size":     tcfg["batch_size"],
            "learning_rate":  tcfg["learning_rate"],
            "alpha":          tcfg["alpha"],
            "focal_gamma":    tcfg["focal_gamma"],
            "hidden_dim":     mcfg["hidden_dim"],
            "num_lstm_layers":mcfg["num_lstm_layers"],
            "dropout":        mcfg["dropout"],
            "cnn_kernels":    str(mcfg["cnn_kernels"]),
            "window_size":    cfg["data"]["window_size"],
            "rul_cap":        cfg["data"]["rul_cap"],
            "device":         str(device)
        })

        best_val_loss  = float("inf")
        best_model_path = "artifacts/best_model.pt"
        os.makedirs("artifacts", exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Starting training for {tcfg['epochs']} epochs")
        print(f"{'='*60}\n")

        for epoch in range(1, tcfg["epochs"] + 1):
            t_start = time.time()

            # ── Train ──────────────────────────────────────────────────────
            train_losses, tr_rul_p, tr_rul_t, tr_cls_p, tr_cls_t = run_epoch(
                model, train_loader, criterion, optimizer, device, training=True
            )
            scheduler.step()

            # ── Validate ───────────────────────────────────────────────────
            val_losses, val_rul_p, val_rul_t, val_cls_p, val_cls_t = run_epoch(
                model, val_loader, criterion, None, device, training=False
            )

            # ── Compute metrics ────────────────────────────────────────────
            train_rul_metrics  = compute_rul_metrics(tr_rul_p,  tr_rul_t)
            val_rul_metrics    = compute_rul_metrics(val_rul_p, val_rul_t)
            train_cls_metrics  = compute_class_metrics(tr_cls_p,  tr_cls_t)
            val_cls_metrics    = compute_class_metrics(val_cls_p, val_cls_t)

            elapsed = time.time() - t_start

            # ── Log to MLflow ──────────────────────────────────────────────
            mlflow.log_metrics({
                # Loss
                "train/total_loss":  train_losses["total_loss"],
                "train/mse_loss":    train_losses["mse_loss"],
                "train/focal_loss":  train_losses["focal_loss"],
                "val/total_loss":    val_losses["total_loss"],
                "val/mse_loss":      val_losses["mse_loss"],
                "val/focal_loss":    val_losses["focal_loss"],
                # RUL
                "train/rmse":        train_rul_metrics["rmse"],
                "train/mae":         train_rul_metrics["mae"],
                "val/rmse":          val_rul_metrics["rmse"],
                "val/mae":           val_rul_metrics["mae"],
                "val/nasa_score":    val_rul_metrics["nasa_score"],
                # Classification
                "train/accuracy":    train_cls_metrics["accuracy"],
                "train/macro_f1":    train_cls_metrics["macro_f1"],
                "val/accuracy":      val_cls_metrics["accuracy"],
                "val/macro_f1":      val_cls_metrics["macro_f1"],
                "val/recall_critical": val_cls_metrics["recall_critical"],
                # LR
                "learning_rate":     optimizer.param_groups[0]["lr"]
            }, step=epoch)

            # ── Print progress ─────────────────────────────────────────────
            print(
                f"Epoch {epoch:03d}/{tcfg['epochs']} | "
                f"Time: {elapsed:.1f}s | "
                f"Train Loss: {train_losses['total_loss']:.4f} | "
                f"Val Loss: {val_losses['total_loss']:.4f} | "
                f"Val RMSE: {val_rul_metrics['rmse']:.2f} | "
                f"Val Acc: {val_cls_metrics['accuracy']:.3f} | "
                f"Critical Recall: {val_cls_metrics['recall_critical']:.3f}"
            )

            # ── Checkpoint best model ──────────────────────────────────────
            if val_losses["total_loss"] < best_val_loss:
                best_val_loss = val_losses["total_loss"]
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "val_loss":    best_val_loss,
                    "val_rmse":    val_rul_metrics["rmse"],
                    "val_acc":     val_cls_metrics["accuracy"]
                }, best_model_path)
                print(f"  → New best model saved (val_loss={best_val_loss:.4f})")

            # ── Early stopping ─────────────────────────────────────────────
            if early_stopping(val_losses["total_loss"]):
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                break

        # ── Final evaluation on test set ───────────────────────────────────
        print(f"\n{'='*60}")
        print("Final evaluation on test set")
        print(f"{'='*60}")

        # Load best model
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

        test_losses, te_rul_p, te_rul_t, te_cls_p, te_cls_t = run_epoch(
            model, test_loader, criterion, None, device, training=False
        )

        test_rul_metrics  = compute_rul_metrics(te_rul_p, te_rul_t)
        test_cls_metrics  = compute_class_metrics(te_cls_p, te_cls_t)

        print(f"\nTest RUL metrics:")
        print(f"  RMSE:       {test_rul_metrics['rmse']:.4f}")
        print(f"  MAE:        {test_rul_metrics['mae']:.4f}")
        print(f"  NASA Score: {test_rul_metrics['nasa_score']:.2f}")

        print(f"\nTest classification metrics:")
        print(f"  Accuracy:   {test_cls_metrics['accuracy']:.4f}")
        print(f"  Macro F1:   {test_cls_metrics['macro_f1']:.4f}")
        print(f"  Critical recall: {test_cls_metrics['recall_critical']:.4f}")

        print(f"\nClassification report:")
        class_names = ["Healthy", "Degrading", "Warning", "Critical"]
        print(classification_report(te_cls_t, te_cls_p,
                                    target_names=class_names))

        # Log final test metrics
        mlflow.log_metrics({
            "test/rmse":            test_rul_metrics["rmse"],
            "test/mae":             test_rul_metrics["mae"],
            "test/nasa_score":      test_rul_metrics["nasa_score"],
            "test/accuracy":        test_cls_metrics["accuracy"],
            "test/macro_f1":        test_cls_metrics["macro_f1"],
            "test/recall_critical": test_cls_metrics["recall_critical"]
        })

        # Log model to MLflow registry
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="cmapss_cnn_lstm"
        )

        # Log preprocessing artifacts
        mlflow.log_artifact("artifacts/preprocessing.pkl")
        mlflow.log_artifact("artifacts/best_model.pt")
        mlflow.log_artifact(config_path)

        run_id = mlflow.active_run().info.run_id
        print(f"\n[✓] Training complete.")
        print(f"    MLflow run ID: {run_id}")
        print(f"    View results:  mlflow ui --port 5000")

    return model, artifacts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
