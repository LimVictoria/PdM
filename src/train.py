"""
train.py
--------
Full training loop with:
    - DagsHub / MLflow experiment tracking (works from Colab/Kaggle/anywhere)
    - Early stopping
    - ReduceLROnPlateau scheduler
    - Gradient clipping
    - Per-epoch metrics (RMSE, MAE, class accuracy, per-class recall)
    - Model checkpointing
    - Auto Google Drive saving (when running in Colab)
    - Auto Hugging Face model saving (optional)
"""

import os
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
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
# Detect runtime environment
# ---------------------------------------------------------------------------

def detect_environment() -> str:
    try:
        import google.colab
        return "colab"
    except ImportError:
        pass
    if os.path.exists("/kaggle/working"):
        return "kaggle"
    return "local"


# ---------------------------------------------------------------------------
# Google Drive saving
# ---------------------------------------------------------------------------

def mount_google_drive() -> Optional[str]:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        drive_path = "/content/drive/MyDrive/cmapss-rul"
        os.makedirs(drive_path, exist_ok=True)
        print(f"[INFO] Google Drive mounted at {drive_path}")
        return drive_path
    except Exception as e:
        print(f"[WARNING] Could not mount Google Drive: {e}")
        return None


def save_to_google_drive(artifacts_dir: str, drive_path: str) -> None:
    print(f"\n[INFO] Saving artifacts to Google Drive...")
    os.makedirs(drive_path, exist_ok=True)
    files_to_save = [
        "artifacts/best_model.pt",
        "artifacts/preprocessing.pkl",
        "configs/config.yaml"
    ]
    for f in files_to_save:
        src = Path(f)
        if src.exists():
            dst = Path(drive_path) / src.name
            shutil.copy(src, dst)
            print(f"  → Saved {src.name} to {drive_path}")
    print(f"[✓] Artifacts saved to Google Drive: {drive_path}")


def load_from_google_drive(drive_path: str, artifacts_dir: str = "artifacts") -> bool:
    model_path = Path(drive_path) / "best_model.pt"
    if not model_path.exists():
        return False
    os.makedirs(artifacts_dir, exist_ok=True)
    shutil.copy(model_path, Path(artifacts_dir) / "best_model.pt")
    pkl_path = Path(drive_path) / "preprocessing.pkl"
    if pkl_path.exists():
        shutil.copy(pkl_path, Path(artifacts_dir) / "preprocessing.pkl")
    print(f"[INFO] Loaded previous model from Google Drive: {drive_path}")
    return True


# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------

def setup_mlflow(mlcfg: dict, env: str) -> None:
    dagshub_user = mlcfg.get("dagshub_user", None)
    dagshub_repo = mlcfg.get("dagshub_repo", None)

    if dagshub_user and dagshub_repo:
        try:
            import dagshub
            dagshub.init(
                repo_owner=dagshub_user,
                repo_name=dagshub_repo,
                mlflow=True
            )
            print(f"[INFO] MLflow → DagsHub: dagshub.com/{dagshub_user}/{dagshub_repo}")
            return
        except ImportError:
            print("[WARNING] dagshub not installed. Run: pip install dagshub")
        except Exception as e:
            print(f"[WARNING] DagsHub init failed: {e}")

    mlflow.set_tracking_uri(mlcfg.get("tracking_uri", "mlruns"))
    if env in ("colab", "kaggle"):
        print("[INFO] MLflow → local mlruns/ (add dagshub config to view remotely)")
    else:
        print("[INFO] MLflow → local mlruns/  View with: mlflow ui --port 5000")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rul_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    errors = preds - targets
    rmse   = np.sqrt(np.mean(errors ** 2))
    mae    = np.mean(np.abs(errors))
    score  = np.sum(
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
    accuracy    = (preds == targets).mean()
    cm          = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    class_names = ["healthy", "degrading", "warning", "critical"]

    per_class_recall = {}
    for i, name in enumerate(class_names):
        if cm[i].sum() > 0:
            per_class_recall[f"recall_{name}"] = cm[i, i] / cm[i].sum()
        else:
            per_class_recall[f"recall_{name}"] = 0.0

    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        **{k: float(v) for k, v in per_class_recall.items()}
    }


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model:     CMAPSS_CNN_LSTM,
    loader:    DataLoader,
    criterion: CMAPSSLoss,
    optimizer: Optional[torch.optim.Optimizer],
    device:    torch.device,
    training:  bool
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    model.train() if training else model.eval()

    total_loss_sum = mse_sum = focal_sum = 0.0
    all_rul_preds    = []
    all_rul_targets  = []
    all_class_preds  = []
    all_class_targets = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for X, static, y_rul, y_class in loader:
            X       = X.to(device)
            static  = static.to(device)
            y_rul   = y_rul.to(device)
            y_class = y_class.to(device)

            rul_pred, class_logits, _, _ = model(X, static)
            total, mse, focal = criterion(rul_pred, y_rul, class_logits, y_class)

            if training:
                optimizer.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss_sum += total.item()
            mse_sum        += mse.item()
            focal_sum      += focal.item()

            all_rul_preds.append(rul_pred.squeeze(-1).detach().cpu().numpy())
            all_rul_targets.append(y_rul.detach().cpu().numpy())
            all_class_preds.append(class_logits.argmax(dim=-1).detach().cpu().numpy())
            all_class_targets.append(y_class.detach().cpu().numpy())

    n = len(loader)
    loss_metrics = {
        "total_loss": total_loss_sum / n,
        "mse_loss":   mse_sum / n,
        "focal_loss": focal_sum / n
    }

    return (
        loss_metrics,
        np.concatenate(all_rul_preds),
        np.concatenate(all_rul_targets),
        np.concatenate(all_class_preds),
        np.concatenate(all_class_targets)
    )


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
    cfg   = load_config(config_path)
    tcfg  = cfg["training"]
    mcfg  = cfg["model"]
    mlcfg = cfg["mlflow"]

    # ── Detect environment ────────────────────────────────────────────────
    env    = detect_environment()
    print(f"[INFO] Running on: {env.upper()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ── Google Drive (Colab only) ─────────────────────────────────────────
    drive_path = None
    if env == "colab":
        drive_path = mount_google_drive()

    # ── MLflow setup ──────────────────────────────────────────────────────
    setup_mlflow(mlcfg, env)
    mlflow.set_experiment(mlcfg["experiment_name"])

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n[INFO] Preprocessing data...")
    train_loader, val_loader, test_loader, class_weights, artifacts = preprocess(
        config_path=config_path
    )
    class_weights = class_weights.to(device)

    # ── Model ─────────────────────────────────────────────────────────────
    model     = build_model(config_path).to(device)
    criterion = build_loss(class_weights, config_path).to(device)
    print(f"[INFO] Model parameters: {model.count_parameters():,}")

    optimizer = Adam(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"]
    )

    lr_scheduler_name = tcfg.get("lr_scheduler", "plateau")
    if lr_scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=tcfg["epochs"],
            eta_min=tcfg["lr_min"]
        )
    elif lr_scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=tcfg.get("lr_factor", 0.5),
            patience=tcfg.get("lr_patience", 5),
            min_lr=tcfg["lr_min"]
        )
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    early_stopping  = EarlyStopping(
        patience=tcfg["patience"],
        min_delta=tcfg["min_delta"]
    )

    best_val_loss   = float("inf")
    best_model_path = "artifacts/best_model.pt"
    os.makedirs("artifacts", exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────
    with mlflow.start_run():

        mlflow.log_params({
            "epochs":          tcfg["epochs"],
            "batch_size":      tcfg["batch_size"],
            "learning_rate":   tcfg["learning_rate"],
            "alpha":           tcfg["alpha"],
            "focal_gamma":     tcfg["focal_gamma"],
            "hidden_dim":      mcfg["hidden_dim"],
            "num_lstm_layers": mcfg["num_lstm_layers"],
            "dropout":         mcfg["dropout"],
            "cnn_kernels":     str(mcfg["cnn_kernels"]),
            "window_size":     cfg["data"]["window_size"],
            "rul_cap":         cfg["data"]["rul_cap"],
            "device":          str(device),
            "environment":     env
        })

        print(f"\n{'='*60}")
        print(f"Training for {tcfg['epochs']} epochs")
        print(f"{'='*60}\n")

        for epoch in range(1, tcfg["epochs"] + 1):
            t_start = time.time()

            # ── Train ─────────────────────────────────────────────────────
            train_losses, tr_rul_p, tr_rul_t, tr_cls_p, tr_cls_t = run_epoch(
                model, train_loader, criterion, optimizer, device, training=True
            )

            # ── Validate ──────────────────────────────────────────────────
            val_losses, val_rul_p, val_rul_t, val_cls_p, val_cls_t = run_epoch(
                model, val_loader, criterion, None, device, training=False
            )

            # ── Step scheduler AFTER validation ───────────────────────────
            if lr_scheduler_name == "plateau":
                scheduler.step(val_losses["total_loss"])
            else:
                scheduler.step()

            # ── Compute metrics ───────────────────────────────────────────
            train_rul = compute_rul_metrics(tr_rul_p,  tr_rul_t)
            val_rul   = compute_rul_metrics(val_rul_p, val_rul_t)
            train_cls = compute_class_metrics(tr_cls_p,  tr_cls_t)
            val_cls   = compute_class_metrics(val_cls_p, val_cls_t)

            elapsed = time.time() - t_start

            mlflow.log_metrics({
                "train/total_loss":    train_losses["total_loss"],
                "train/mse_loss":      train_losses["mse_loss"],
                "val/total_loss":      val_losses["total_loss"],
                "val/mse_loss":        val_losses["mse_loss"],
                "train/rmse":          train_rul["rmse"],
                "val/rmse":            val_rul["rmse"],
                "val/mae":             val_rul["mae"],
                "val/nasa_score":      val_rul["nasa_score"],
                "train/accuracy":      train_cls["accuracy"],
                "val/accuracy":        val_cls["accuracy"],
                "val/macro_f1":        val_cls["macro_f1"],
                "val/recall_critical": val_cls["recall_critical"],
                "learning_rate":       optimizer.param_groups[0]["lr"]
            }, step=epoch)

            print(
                f"Epoch {epoch:03d}/{tcfg['epochs']} | "
                f"{elapsed:.1f}s | "
                f"Loss: {val_losses['total_loss']:.4f} | "
                f"RMSE: {val_rul['rmse']:.2f} | "
                f"Acc: {val_cls['accuracy']:.3f} | "
                f"Critical: {val_cls['recall_critical']:.3f}"
            )

            # ── Checkpoint ────────────────────────────────────────────────
            if val_losses["total_loss"] < best_val_loss:
                best_val_loss = val_losses["total_loss"]
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "val_loss":    best_val_loss,
                    "val_rmse":    val_rul["rmse"],
                    "val_acc":     val_cls["accuracy"]
                }, best_model_path)
                print(f"  → Best model saved (val_loss={best_val_loss:.4f})")

            if early_stopping(val_losses["total_loss"]):
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                break

        # ── Test evaluation ────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("Test set evaluation")
        print(f"{'='*60}")

        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

        _, te_rul_p, te_rul_t, te_cls_p, te_cls_t = run_epoch(
            model, test_loader, criterion, None, device, training=False
        )

        test_rul = compute_rul_metrics(te_rul_p, te_rul_t)
        test_cls = compute_class_metrics(te_cls_p, te_cls_t)

        print(f"\n  RMSE:            {test_rul['rmse']:.4f}")
        print(f"  MAE:             {test_rul['mae']:.4f}")
        print(f"  NASA Score:      {test_rul['nasa_score']:.2f}")
        print(f"  Accuracy:        {test_cls['accuracy']:.4f}")
        print(f"  Macro F1:        {test_cls['macro_f1']:.4f}")
        print(f"  Critical Recall: {test_cls['recall_critical']:.4f}")
        print(f"\n{classification_report(te_cls_t, te_cls_p, target_names=['Healthy','Degrading','Warning','Critical'])}")

        mlflow.log_metrics({
            "test/rmse":            test_rul["rmse"],
            "test/mae":             test_rul["mae"],
            "test/nasa_score":      test_rul["nasa_score"],
            "test/accuracy":        test_cls["accuracy"],
            "test/macro_f1":        test_cls["macro_f1"],
            "test/recall_critical": test_cls["recall_critical"]
        })

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="cmapss_cnn_lstm"
        )
        mlflow.log_artifact("artifacts/preprocessing.pkl")
        mlflow.log_artifact("artifacts/best_model.pt")
        mlflow.log_artifact(config_path)

        run_id = mlflow.active_run().info.run_id
        print(f"\n[✓] Training complete. MLflow run ID: {run_id}")

    # ── Save to Google Drive (Colab only) ──────────────────────────────────
    if env == "colab" and drive_path:
        save_to_google_drive("artifacts", drive_path)
    elif env == "kaggle":
        print(f"[INFO] Model saved at: artifacts/best_model.pt")
        print(f"       Download from Kaggle output tab.")
    else:
        print(f"[INFO] Model saved at: artifacts/best_model.pt")

    return model, artifacts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
