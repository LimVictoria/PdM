"""
predict.py
----------
Recursive inference with MC Dropout uncertainty estimation.

Recursive loop:
    [Window t-30:t] → Model → RUL(t+1), Class(t+1), uncertainty
                                  ↓
    [Window t-29:t+1] → Model → RUL(t+2) ...  ← LSTM state carried forward
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yaml

from model import build_model, CMAPSS_CNN_LSTM
from dataset import load_artifacts


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Single-window prediction
# ---------------------------------------------------------------------------

def predict_single(
    model:      CMAPSS_CNN_LSTM,
    x:          torch.Tensor,
    static:     torch.Tensor,
    hidden:     Optional[Tuple[torch.Tensor, torch.Tensor]],
    mc_passes:  int,
    device:     torch.device
) -> Tuple[float, float, int, np.ndarray, Tuple]:
    """
    Single window prediction with MC Dropout.

    Args:
        x:          [1, window_size, input_dim]
        static:     [1, static_dim]
        hidden:     LSTM hidden state from previous step
        mc_passes:  Number of MC Dropout forward passes
        device:     Torch device

    Returns:
        rul_mean:      Mean predicted RUL across MC passes
        rul_std:       Std of predicted RUL (uncertainty)
        health_class:  Most likely health class
        class_probs:   Mean class probabilities [n_classes]
        hidden:        Updated LSTM hidden state
    """
    model.enable_dropout()   # Keep dropout active for MC uncertainty

    rul_samples    = []
    class_samples  = []
    hidden_final   = None

    with torch.no_grad():
        for _ in range(mc_passes):
            rul_pred, class_logits, _, (h_n, c_n) = model(
                x, static, hidden
            )
            rul_samples.append(rul_pred.squeeze().item())
            class_samples.append(
                torch.softmax(class_logits, dim=-1).cpu().numpy()
            )
            hidden_final = (h_n, c_n)

    rul_samples   = np.array(rul_samples)
    class_samples = np.array(class_samples).squeeze(1)   # [mc_passes, n_classes]

    rul_mean     = float(np.mean(rul_samples))
    rul_std      = float(np.std(rul_samples))
    class_probs  = class_samples.mean(axis=0)             # [n_classes]
    health_class = int(np.argmax(class_probs))

    return rul_mean, rul_std, health_class, class_probs, hidden_final


# ---------------------------------------------------------------------------
# Recursive inference over full engine life
# ---------------------------------------------------------------------------

def recursive_predict(
    model:      CMAPSS_CNN_LSTM,
    sensor_sequence: np.ndarray,
    static_vec: np.ndarray,
    window_size: int = 30,
    mc_passes:   int = 50,
    device:      torch.device = torch.device("cpu")
) -> Dict:
    """
    Run recursive RUL prediction over a complete sensor sequence.
    LSTM hidden state is carried forward between windows.

    Args:
        model:            Trained CMAPSS_CNN_LSTM
        sensor_sequence:  [n_cycles, n_features] normalised sensor data
        static_vec:       [static_dim] one-hot static features
        window_size:      Sliding window size (must match training)
        mc_passes:        MC Dropout passes per step
        device:           Torch device

    Returns:
        Dict with:
            rul_mean:     [n_cycles] mean predicted RUL per cycle
            rul_std:      [n_cycles] uncertainty (std) per cycle
            health_class: [n_cycles] predicted health class per cycle
            class_probs:  [n_cycles, 4] class probabilities per cycle
            class_names:  List of class name strings
    """
    n_cycles    = len(sensor_sequence)
    class_names = ["Healthy", "Degrading", "Warning", "Critical"]

    rul_means    = []
    rul_stds     = []
    health_classes = []
    all_probs    = []

    # Prepare static tensor
    static_t = torch.FloatTensor(static_vec).unsqueeze(0).to(device)

    hidden = None   # Will be initialised from static on first pass

    for t in range(1, n_cycles + 1):
        # Build window with left-padding for early cycles
        if t < window_size:
            pad_len = window_size - t
            window  = sensor_sequence[0:t]
            pad     = np.repeat(sensor_sequence[0:1], pad_len, axis=0)
            window  = np.concatenate([pad, window], axis=0)
        else:
            window = sensor_sequence[t - window_size:t]

        x_t = torch.FloatTensor(window).unsqueeze(0).to(device)
        # x_t: [1, window_size, n_features]

        # Only use hidden state from previous step after first window
        # (first window gets h₀ from static encoder)
        prev_hidden = hidden if t > window_size else None

        rul_mean, rul_std, h_class, probs, hidden = predict_single(
            model, x_t, static_t, prev_hidden, mc_passes, device
        )

        rul_means.append(rul_mean)
        rul_stds.append(rul_std)
        health_classes.append(h_class)
        all_probs.append(probs)

    return {
        "rul_mean":     np.array(rul_means),
        "rul_std":      np.array(rul_stds),
        "health_class": np.array(health_classes),
        "class_probs":  np.array(all_probs),
        "class_names":  class_names,
        "n_cycles":     n_cycles
    }


# ---------------------------------------------------------------------------
# Batch prediction (for test set evaluation)
# ---------------------------------------------------------------------------

def batch_predict(
    model:      CMAPSS_CNN_LSTM,
    loader:     torch.utils.data.DataLoader,
    mc_passes:  int = 50,
    device:     torch.device = torch.device("cpu")
) -> Dict:
    """
    Batch prediction with MC Dropout on a DataLoader.
    Used for test set evaluation.

    Returns dict with predictions and uncertainties.
    """
    model.enable_dropout()

    all_rul_means    = []
    all_rul_stds     = []
    all_class_preds  = []
    all_class_probs  = []
    all_rul_targets  = []
    all_class_targets = []

    with torch.no_grad():
        for X, static, y_rul, y_class in loader:
            X       = X.to(device)
            static  = static.to(device)
            batch   = X.size(0)

            mc_ruls    = []
            mc_classes = []

            for _ in range(mc_passes):
                rul_pred, class_logits, _, _ = model(X, static)
                mc_ruls.append(rul_pred.squeeze(-1).cpu().numpy())
                mc_classes.append(
                    torch.softmax(class_logits, dim=-1).cpu().numpy()
                )

            mc_ruls    = np.stack(mc_ruls)     # [mc_passes, batch]
            mc_classes = np.stack(mc_classes)  # [mc_passes, batch, n_classes]

            rul_mean  = mc_ruls.mean(axis=0)   # [batch]
            rul_std   = mc_ruls.std(axis=0)    # [batch]
            class_probs = mc_classes.mean(axis=0)   # [batch, n_classes]
            class_pred  = class_probs.argmax(axis=1) # [batch]

            all_rul_means.append(rul_mean)
            all_rul_stds.append(rul_std)
            all_class_preds.append(class_pred)
            all_class_probs.append(class_probs)
            all_rul_targets.append(y_rul.numpy())
            all_class_targets.append(y_class.numpy())

    return {
        "rul_mean":      np.concatenate(all_rul_means),
        "rul_std":       np.concatenate(all_rul_stds),
        "class_pred":    np.concatenate(all_class_preds),
        "class_probs":   np.concatenate(all_class_probs),
        "rul_target":    np.concatenate(all_rul_targets),
        "class_target":  np.concatenate(all_class_targets),
        "class_names":   ["Healthy", "Degrading", "Warning", "Critical"]
    }


# ---------------------------------------------------------------------------
# Preprocess raw sensor data for inference
# ---------------------------------------------------------------------------

def preprocess_for_inference(
    raw_sensor_data: np.ndarray,
    op_settings:     np.ndarray,
    artifacts:       dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess raw sensor readings for inference using saved artifacts.

    Args:
        raw_sensor_data: [n_cycles, 21] raw sensor readings
        op_settings:     [n_cycles, 3] op_setting_1,2,3
        artifacts:       Loaded preprocessing artifacts dict

    Returns:
        normalised_seq:  [n_cycles, 17] normalised features
        static_vec:      [8] one-hot static features
    """
    import pandas as pd

    sensor_cols  = artifacts["sensor_cols"]
    feature_cols = artifacts["feature_cols"]
    flat_sensors = artifacts["flat_sensors"]
    kmeans       = artifacts["kmeans"]
    scalers      = artifacts["scalers"]

    # Build DataFrame
    all_sensor_names = [f"s{i}" for i in range(1, 22)]
    df = pd.DataFrame(raw_sensor_data, columns=all_sensor_names)
    df["op_setting_1"] = op_settings[:, 0]
    df["op_setting_2"] = op_settings[:, 1]
    df["op_setting_3"] = op_settings[:, 2]

    # Drop flat sensors
    for s in flat_sensors:
        if s in df.columns:
            df = df.drop(columns=[s])

    # Assign cluster
    df["op_cluster"] = kmeans.predict(op_settings)

    # Dominant cluster for this engine
    engine_cluster = int(
        pd.Series(df["op_cluster"]).mode().iloc[0]
    )

    # Normalise per cluster
    normalised = df[feature_cols].copy()
    for cluster_id, scaler in scalers.items():
        mask = df["op_cluster"] == cluster_id
        if mask.sum() > 0:
            normalised.loc[mask] = scaler.transform(
                df.loc[mask, feature_cols]
            )

    # Static vector: cluster one-hot + fault_mode one-hot
    # fault_mode must be provided externally at deployment
    # Default to 0 (FD001/FD002 type) if unknown
    cluster_onehot = np.eye(6, dtype=np.float32)[engine_cluster]
    fault_onehot   = np.eye(2, dtype=np.float32)[0]
    static_vec     = np.concatenate([cluster_onehot, fault_onehot])

    return normalised.values.astype(np.float32), static_vec


# ---------------------------------------------------------------------------
# Entry point for testing recursive inference
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    cfg    = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model().to(device)
    checkpoint = torch.load("artifacts/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print(f"[INFO] Loaded model from epoch {checkpoint['epoch']}")

    # Simulate a sensor sequence (30 cycles, 17 features)
    # In real use, this would come from actual sensor readings
    np.random.seed(42)
    fake_sequence = np.random.randn(200, 17).astype(np.float32)
    fake_static   = np.zeros(8, dtype=np.float32)
    fake_static[0] = 1.0   # cluster 0
    fake_static[6] = 1.0   # fault mode 0

    print("\n[INFO] Running recursive prediction on simulated sequence...")
    results = recursive_predict(
        model=model,
        sensor_sequence=fake_sequence,
        static_vec=fake_static,
        window_size=cfg["data"]["window_size"],
        mc_passes=cfg["inference"]["mc_passes"],
        device=device
    )

    print(f"\nPredictions over {results['n_cycles']} cycles:")
    print(f"{'Cycle':>6} | {'RUL Mean':>9} | {'RUL Std':>8} | {'Class':>12}")
    print("-" * 45)
    class_names = results["class_names"]
    for t in range(0, results["n_cycles"], 20):
        print(
            f"{t+1:>6} | "
            f"{results['rul_mean'][t]:>9.1f} | "
            f"{results['rul_std'][t]:>8.2f} | "
            f"{class_names[results['health_class'][t]]:>12}"
        )
