"""
train.py
--------
Pure RUL regression training loop for CMAPSS dataset.

Metrics reported:
    RMSE       — root mean squared error in cycles
    MAE        — mean absolute error in cycles
    NASA Score — asymmetric score penalising late predictions more
"""

import os
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, Dict, Optional
import yaml
import mlflow
import mlflow.pytorch
import tempfile

from dataset import preprocess
from model import build_model, build_loss, CMAPSS_CNN_LSTM, RULLoss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def detect_environment() -> str:
    try:
        import google.colab
        return "colab"
    except ImportError:
        pass
    return "kaggle" if os.path.exists("/kaggle/working") else "local"


def mount_google_drive() -> Optional[str]:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        drive_path = "/content/drive/MyDrive/cmapss-rul"
        os.makedirs(drive_path, exist_ok=True)
        return drive_path
    except Exception as e:
        print(f"[WARNING] Could not mount Google Drive: {e}")
        return None


def save_to_google_drive(drive_path: str) -> None:
    for f in ["artifacts/best_model.pt", "artifacts/preprocessing.pkl", "configs/config.yaml"]:
        src = Path(f)
        if src.exists():
            shutil.copy(src, Path(drive_path) / src.name)
            print(f"  -> Saved {src.name} to {drive_path}")


def setup_mlflow(mlcfg: dict) -> None:
    user = mlcfg.get("dagshub_user")
    repo = mlcfg.get("dagshub_repo")
    if user and repo:
        try:
            import dagshub
            dagshub.init(repo_owner=user, repo_name=repo, mlflow=True)
            print(f"[INFO] MLflow -> DagsHub: dagshub.com/{user}/{repo}")
            return
        except Exception as e:
            print(f"[WARNING] DagsHub failed: {e}")
    mlflow.set_tracking_uri(mlcfg.get("tracking_uri", "mlruns"))
    print("[INFO] MLflow -> local mlruns/")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    errors = preds - targets
    rmse   = float(np.sqrt(np.mean(errors ** 2)))
    mae    = float(np.mean(np.abs(errors)))
    score  = float(np.sum(np.where(
        errors < 0,
        np.exp(-errors / 13) - 1,
        np.exp(errors / 10) - 1
    )))
    return {"rmse": rmse, "mae": mae, "nasa_score": score}


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model:     CMAPSS_CNN_LSTM,
    loader:    DataLoader,
    criterion: RULLoss,
    optimizer: Optional[torch.optim.Optimizer],
    device:    torch.device,
    training:  bool
) -> Tuple[float, np.ndarray, np.ndarray]:

    model.train() if training else model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X, static, y_rul in loader:
            X, static, y_rul = X.to(device), static.to(device), y_rul.to(device)

            rul_pred, _ = model(X, static)
            loss, _     = criterion(rul_pred, y_rul)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            all_preds.append(rul_pred.squeeze(-1).detach().cpu().numpy())
            all_targets.append(y_rul.detach().cpu().numpy())

    return (
        total_loss / len(loader),
        np.concatenate(all_preds),
        np.concatenate(all_targets)
    )


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.1):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = None
        self.counter    = 0

    def __call__(self, val_rmse: float) -> bool:
        score = -val_rmse
        if self.best is None or score > self.best + self.min_delta:
            self.best   = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config_path: str = "configs/config.yaml"):
    cfg   = load_config(config_path)
    tcfg  = cfg["training"]
    mlcfg = cfg["mlflow"]

    env    = detect_environment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on: {env.upper()}  |  Device: {device}")

    drive_path = mount_google_drive() if env == "colab" else None

    setup_mlflow(mlcfg)
    mlflow.set_experiment(mlcfg["experiment_name"])

    print("\n[INFO] Preprocessing data...")
    train_loader, val_loader, test_loader, artifacts = preprocess(config_path=config_path)

    # Auto-detect input_dim from data
    sample_X, _, _ = next(iter(train_loader))
    actual_input_dim = sample_X.shape[-1]
    print(f"[INFO] Auto-detected input_dim: {actual_input_dim}")

    # Build model with correct input_dim
    cfg["model"]["input_dim"] = actual_input_dim
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(cfg, tmp); tmp.close()
    model     = build_model(tmp.name).to(device)
    criterion = build_loss(config_path).to(device)
    os.unlink(tmp.name)
    print(f"[INFO] Model parameters: {model.count_parameters():,}")

    optimizer = Adam(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min",
        factor=tcfg.get("lr_factor", 0.5),
        patience=tcfg.get("lr_patience", 5),
        min_lr=tcfg["lr_min"]
    )

    early_stop      = EarlyStopping(patience=tcfg["patience"], min_delta=tcfg["min_delta"])
    best_val_rmse   = float("inf")
    best_model_path = "artifacts/best_model.pt"
    os.makedirs("artifacts", exist_ok=True)

    with mlflow.start_run():

        mlflow.log_params({
            "epochs":        tcfg["epochs"],
            "batch_size":    tcfg["batch_size"],
            "lr":            tcfg["learning_rate"],
            "weight_decay":  tcfg["weight_decay"],
            "hidden_dim":    cfg["model"]["hidden_dim"],
            "lstm_layers":   cfg["model"]["num_lstm_layers"],
            "dropout":       cfg["model"]["dropout"],
            "window_size":   cfg["data"]["window_size"],
            "stride":        cfg["data"].get("stride", 10),
            "rul_cap":       cfg["data"]["rul_cap"],
            "input_dim":     actual_input_dim,
            "device":        str(device),
        })

        print(f"\n{'='*60}")
        print(f"Training for {tcfg['epochs']} epochs  [pure RUL regression]")
        print(f"{'='*60}\n")

        for epoch in range(1, tcfg["epochs"] + 1):
            t0 = time.time()

            tr_loss, tr_p, tr_t = run_epoch(model, train_loader, criterion, optimizer, device, True)
            va_loss, va_p, va_t = run_epoch(model, val_loader,   criterion, None,      device, False)

            scheduler.step(va_loss)

            tr_m = compute_metrics(tr_p, tr_t)
            va_m = compute_metrics(va_p, va_t)

            mlflow.log_metrics({
                "train/loss":  tr_loss,   "val/loss":  va_loss,
                "train/rmse":  tr_m["rmse"], "val/rmse":  va_m["rmse"],
                "train/mae":   tr_m["mae"],  "val/mae":   va_m["mae"],
                "lr": optimizer.param_groups[0]["lr"]
            }, step=epoch)

            saved = ""
            if va_m["rmse"] < best_val_rmse:
                best_val_rmse = va_m["rmse"]
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_rmse": best_val_rmse,
                    "val_mae":  va_m["mae"],
                }, best_model_path)
                saved = f"  -> Best saved (val_rmse={best_val_rmse:.4f})"

            print(f"Epoch {epoch:03d}/{tcfg['epochs']} | "
                  f"{time.time()-t0:.1f}s | "
                  f"Train RMSE: {tr_m['rmse']:.2f} | "
                  f"Val RMSE: {va_m['rmse']:.2f} | "
                  f"Val MAE: {va_m['mae']:.2f}"
                  f"{saved}")

            if early_stop(va_m["rmse"]):
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                break

        # ── Test evaluation ────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("Test set evaluation")
        print(f"{'='*60}")

        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

        _, te_p, te_t = run_epoch(model, test_loader, criterion, None, device, False)
        te_m = compute_metrics(te_p, te_t)

        print(f"\n  RMSE:       {te_m['rmse']:.4f} cycles")
        print(f"  MAE:        {te_m['mae']:.4f} cycles")
        print(f"  NASA Score: {te_m['nasa_score']:.2f}")

        mlflow.log_metrics({
            "test/rmse":       te_m["rmse"],
            "test/mae":        te_m["mae"],
            "test/nasa_score": te_m["nasa_score"],
        })

        mlflow.pytorch.log_model(model, artifact_path="model",
                                 registered_model_name="cmapss_cnn_lstm")
        mlflow.log_artifact("artifacts/preprocessing.pkl")
        mlflow.log_artifact("artifacts/best_model.pt")

        print(f"\n[OK] Training complete. MLflow run: {mlflow.active_run().info.run_id}")

    if env == "colab" and drive_path:
        save_to_google_drive(drive_path)

    return model, artifacts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
