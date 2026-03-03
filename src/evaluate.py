"""
evaluate.py
-----------
Evaluation metrics and visualisation for CMAPSS predictions.

Generates:
    - RUL prediction vs actual plots
    - Confusion matrix
    - Per-class uncertainty plots
    - Health state timeline
    - NASA score breakdown per subset
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, Optional
import torch
import yaml


CLASS_NAMES   = ["Healthy", "Degrading", "Warning", "Critical"]
CLASS_COLOURS = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Plot 1 — RUL Prediction vs Actual
# ---------------------------------------------------------------------------

def plot_rul_predictions(
    rul_pred:   np.ndarray,
    rul_target: np.ndarray,
    rul_std:    Optional[np.ndarray] = None,
    save_path:  Optional[str] = None,
    title:      str = "RUL Prediction vs Actual"
):
    """Scatter plot of predicted vs actual RUL with uncertainty bands."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Scatter ──────────────────────────────────────────────────────────
    ax = axes[0]
    if rul_std is not None:
        scatter = ax.scatter(
            rul_target, rul_pred,
            c=rul_std, cmap="YlOrRd",
            alpha=0.5, s=8
        )
        plt.colorbar(scatter, ax=ax, label="Uncertainty (std)")
    else:
        ax.scatter(rul_target, rul_pred, alpha=0.4, s=8, color="#3498db")

    lim = max(rul_target.max(), rul_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Perfect")
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("Predicted vs Actual RUL")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Error distribution ────────────────────────────────────────────────
    ax = axes[1]
    errors = rul_pred - rul_target
    ax.hist(errors, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
    ax.axvline(0,   color="black", linestyle="--", linewidth=1)
    ax.axvline(errors.mean(), color="red",
               linestyle="--", linewidth=1, label=f"Mean={errors.mean():.1f}")
    ax.set_xlabel("Prediction Error (Pred - Actual)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    rmse = np.sqrt(np.mean(errors ** 2))
    mae  = np.mean(np.abs(errors))
    fig.suptitle(f"{title} | RMSE={rmse:.2f} | MAE={mae:.2f}", fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 2 — Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    class_pred:   np.ndarray,
    class_target: np.ndarray,
    save_path:    Optional[str] = None,
    normalise:    bool = True
):
    """Confusion matrix with optional normalisation."""
    cm = confusion_matrix(class_target, class_pred,
                          labels=list(range(4)))

    if normalise:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
        cbar_label = "Recall"
    else:
        cm_plot = cm
        fmt = "d"
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_plot,
        annot=True, fmt=fmt,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": cbar_label},
        linewidths=0.5
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Health State Confusion Matrix")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 3 — Health state timeline for one engine
# ---------------------------------------------------------------------------

def plot_engine_timeline(
    rul_mean:     np.ndarray,
    rul_std:      np.ndarray,
    health_class: np.ndarray,
    true_rul:     Optional[np.ndarray] = None,
    engine_id:    int = 0,
    save_path:    Optional[str] = None
):
    """
    Full lifecycle visualisation for one engine:
    - RUL prediction with uncertainty band
    - Health state colour-coded background
    """
    n = len(rul_mean)
    cycles = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Health state background bands
    prev_class = health_class[0]
    start = 0
    for t in range(1, n + 1):
        curr_class = health_class[t - 1] if t < n else health_class[-1]
        if curr_class != prev_class or t == n:
            ax.axvspan(
                start + 1, t,
                alpha=0.15,
                color=CLASS_COLOURS[prev_class],
                label=CLASS_NAMES[prev_class]
            )
            prev_class = curr_class
            start = t - 1

    # Uncertainty band
    ax.fill_between(
        cycles,
        rul_mean - 2 * rul_std,
        rul_mean + 2 * rul_std,
        alpha=0.25, color="#3498db", label="±2σ uncertainty"
    )

    # Predicted RUL
    ax.plot(cycles, rul_mean, color="#2980b9",
            linewidth=2, label="Predicted RUL")

    # True RUL if available
    if true_rul is not None:
        ax.plot(cycles, true_rul, color="black",
                linewidth=1.5, linestyle="--", label="True RUL")

    # Class boundary lines
    for threshold, label in [(125, "Healthy→Degrading"),
                              (75,  "Degrading→Warning"),
                              (25,  "Warning→Critical")]:
        ax.axhline(threshold, color="gray",
                   linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(n * 0.01, threshold + 2, label,
                fontsize=7, color="gray")

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels)
              if l not in seen and not seen.add(l)]
    ax.legend(*zip(*unique), loc="upper right", fontsize=8)

    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"Engine {engine_id} — RUL Prediction Timeline")
    ax.set_xlim(1, n)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 4 — Uncertainty vs Error
# ---------------------------------------------------------------------------

def plot_uncertainty_calibration(
    rul_pred:   np.ndarray,
    rul_target: np.ndarray,
    rul_std:    np.ndarray,
    save_path:  Optional[str] = None
):
    """
    Check if uncertainty correlates with actual error.
    Well-calibrated model: high std → high error.
    """
    errors = np.abs(rul_pred - rul_target)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(rul_std, errors, alpha=0.3, s=8, color="#8e44ad")

    # Trend line
    z = np.polyfit(rul_std, errors, 1)
    p = np.poly1d(z)
    x_line = np.linspace(rul_std.min(), rul_std.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=1.5, label="Trend")

    correlation = np.corrcoef(rul_std, errors)[0, 1]
    ax.set_xlabel("Predicted Uncertainty (std)")
    ax.set_ylabel("Absolute Error |Pred - Actual|")
    ax.set_title(f"Uncertainty Calibration | Correlation={correlation:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Print full evaluation summary
# ---------------------------------------------------------------------------

def print_evaluation_summary(results: Dict):
    """Print a clean evaluation summary to console."""
    rul_pred    = results["rul_mean"]
    rul_target  = results["rul_target"]
    class_pred  = results["class_pred"]
    class_target = results["class_target"]
    rul_std     = results.get("rul_std", None)

    errors = rul_pred - rul_target
    rmse   = np.sqrt(np.mean(errors ** 2))
    mae    = np.mean(np.abs(errors))
    nasa   = np.sum(
        np.where(errors < 0,
                 np.exp(-errors / 13) - 1,
                 np.exp(errors / 10) - 1)
    )
    accuracy = (class_pred == class_target).mean()

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"\nRUL Regression:")
    print(f"  RMSE:         {rmse:.4f}")
    print(f"  MAE:          {mae:.4f}")
    print(f"  NASA Score:   {nasa:.2f}  (lower is better)")

    if rul_std is not None:
        print(f"\nUncertainty:")
        print(f"  Mean std:     {rul_std.mean():.4f}")
        print(f"  Std-Error correlation: "
              f"{np.corrcoef(rul_std, np.abs(errors))[0,1]:.4f}")

    print(f"\nHealth Classification:")
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"\n{classification_report(class_target, class_pred, target_names=CLASS_NAMES)}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Run all plots
# ---------------------------------------------------------------------------

def run_full_evaluation(
    results:   Dict,
    save_dir:  Optional[str] = "artifacts/plots"
):
    """Run and save all evaluation plots."""
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    print_evaluation_summary(results)

    plot_rul_predictions(
        results["rul_mean"],
        results["rul_target"],
        results.get("rul_std"),
        save_path=f"{save_dir}/rul_predictions.png" if save_dir else None
    )

    plot_confusion_matrix(
        results["class_pred"],
        results["class_target"],
        save_path=f"{save_dir}/confusion_matrix.png" if save_dir else None
    )

    if "rul_std" in results:
        plot_uncertainty_calibration(
            results["rul_mean"],
            results["rul_target"],
            results["rul_std"],
            save_path=f"{save_dir}/uncertainty_calibration.png" if save_dir else None
        )
