"""
tune.py
-------
Optuna hyperparameter search for CNN-LSTM RUL model.

Searches over:
  - learning_rate:         1e-4  to 1e-2  (log scale)
  - alpha:                 0.3   to 0.7
  - dropout:               0.1   to 0.5
  - hidden_dim:            64    to 256
  - class_weight_critical: 1.0   to 4.0

Each trial trains for 10 epochs (fast evaluation).
Best params then used for full 50-epoch training.

Run with:
    python src/tune.py

Results saved to:
    artifacts/optuna_best_params.yaml
    artifacts/optuna_study.pkl
"""

import os
import sys
import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from dataset import preprocess
from model import CMAPSS_CNN_LSTM, CMAPSSLoss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_CONFIG  = "configs/config.yaml"
TUNING_EPOCHS = 10       # fast trials
N_TRIALS      = 30       # number of Optuna trials
ARTIFACTS_DIR = "artifacts"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_config(path: str = BASE_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    """
    Objective for Optuna.
    Returns: negative Macro F1 (Optuna minimises by default)
    We optimise for Macro F1 because it captures all 4 classes.
    """
    cfg = load_config()

    # ── Sample hyperparameters ────────────────────────────────────────────
    lr           = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    alpha        = trial.suggest_float("alpha", 0.3, 0.7)
    dropout      = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_dim   = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    w_critical   = trial.suggest_float("class_weight_critical", 1.0, 4.0)
    w_degrading  = trial.suggest_float("class_weight_degrading", 1.0, 4.0)
    w_warning    = trial.suggest_float("class_weight_warning", 1.0, 3.0)

    # Class weights: [Healthy, Degrading, Warning, Critical]
    class_weight_override = [0.5, w_degrading, w_warning, w_critical]

    # ── Override config with trial params ─────────────────────────────────
    cfg["model"]["hidden_dim"]    = hidden_dim
    cfg["model"]["dropout"]       = dropout
    cfg["training"]["learning_rate"] = lr
    cfg["training"]["alpha"]         = alpha
    cfg["data"]["class_weight_override"] = class_weight_override

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # ── Data ──────────────────────────────────────────────────────────
        train_loader, val_loader, _, class_weights, _ = preprocess(
            config_path=BASE_CONFIG,
            config_override=cfg
        )
        class_weights = class_weights.to(device)

        # ── Model ─────────────────────────────────────────────────────────
        mcfg = cfg["model"]
        model = CMAPSS_CNN_LSTM(
            input_dim=mcfg["input_dim"],
            static_dim=mcfg["static_dim"],
            hidden_dim=hidden_dim,
            num_layers=mcfg["num_lstm_layers"],
            dropout=dropout,
            cnn_kernels=mcfg["cnn_kernels"],
            cnn_channels=mcfg["cnn_out_channels"],
            attention_dim=mcfg["attention_dim"],
            num_classes=mcfg["num_classes"]
        ).to(device)

        criterion = CMAPSSLoss(
            class_weights=class_weights,
            alpha=alpha,
            gamma=cfg["training"]["focal_gamma"]
        ).to(device)

        optimizer = Adam(model.parameters(), lr=lr,
                         weight_decay=cfg["training"]["weight_decay"])
        scheduler = CosineAnnealingLR(optimizer, T_max=TUNING_EPOCHS,
                                      eta_min=lr * 0.01)

        # ── Train for TUNING_EPOCHS ────────────────────────────────────────
        best_macro_f1 = 0.0

        for epoch in range(TUNING_EPOCHS):
            # Train
            model.train()
            for X, static, y_rul, y_class in train_loader:
                X, static = X.to(device), static.to(device)
                y_rul, y_class = y_rul.to(device), y_class.to(device)

                rul_pred, class_logits, _, _ = model(X, static)
                total, _, _ = criterion(rul_pred, y_rul, class_logits, y_class)

                optimizer.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            # Validate
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for X, static, y_rul, y_class in val_loader:
                    X, static = X.to(device), static.to(device)
                    _, class_logits, _, _ = model(X, static)
                    preds = class_logits.argmax(dim=-1).cpu().numpy()
                    all_preds.append(preds)
                    all_targets.append(y_class.numpy())

            preds   = np.concatenate(all_preds)
            targets = np.concatenate(all_targets)
            macro_f1 = f1_score(targets, preds, average="macro",
                                zero_division=0)

            best_macro_f1 = max(best_macro_f1, macro_f1)

            # Optuna pruning — stop bad trials early
            trial.report(macro_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        print(f"  Trial {trial.number}: "
              f"lr={lr:.5f} alpha={alpha:.2f} "
              f"hidden={hidden_dim} dropout={dropout:.2f} "
              f"w_crit={w_critical:.2f} → F1={best_macro_f1:.4f}")

        return -best_macro_f1   # negative because Optuna minimises

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        return 0.0   # worst possible score


# ---------------------------------------------------------------------------
# Run tuning
# ---------------------------------------------------------------------------

def tune():
    print("=" * 60)
    print(f"Optuna Hyperparameter Search")
    print(f"  Trials:        {N_TRIALS}")
    print(f"  Epochs/trial:  {TUNING_EPOCHS}")
    print(f"  Estimated time: ~{N_TRIALS * TUNING_EPOCHS * 15 / 60:.0f} minutes")
    print("=" * 60)

    sampler = TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=3
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="cmapss_cnn_lstm"
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # ── Results ───────────────────────────────────────────────────────────
    best = study.best_trial
    best_params = best.params
    best_f1 = -best.value

    print(f"\n{'='*60}")
    print(f"Best Macro F1: {best_f1:.4f}")
    print(f"Best params:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # ── Save best params to yaml ──────────────────────────────────────────
    best_params["best_macro_f1"] = best_f1
    out_path = f"{ARTIFACTS_DIR}/optuna_best_params.yaml"
    with open(out_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
    print(f"\nBest params saved to: {out_path}")

    # ── Save full study ───────────────────────────────────────────────────
    study_path = f"{ARTIFACTS_DIR}/optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"Full study saved to:  {study_path}")

    # ── Update config.yaml with best params ──────────────────────────────
    cfg = load_config()
    cfg["training"]["learning_rate"] = best_params["learning_rate"]
    cfg["training"]["alpha"]         = best_params["alpha"]
    cfg["model"]["dropout"]          = best_params["dropout"]
    cfg["model"]["hidden_dim"]       = best_params["hidden_dim"]
    cfg["data"]["class_weight_override"] = [
        0.5,
        best_params["class_weight_degrading"],
        best_params["class_weight_warning"],
        best_params["class_weight_critical"]
    ]

    with open(BASE_CONFIG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\nconfig.yaml updated with best params.")
    print(f"Now run: python src/train.py")
    print(f"="*60)

    return study, best_params


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tune()
