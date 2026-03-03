"""
serve.py
--------
FastAPI serving endpoint for CMAPSS RUL prediction.

Endpoints:
    POST /predict          → Single window prediction
    POST /predict/recursive → Full lifecycle recursive prediction
    GET  /health           → API health check
    GET  /model/info       → Model metadata

Run with:
    uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict
import yaml
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import build_model, CMAPSS_CNN_LSTM
from predict import predict_single, recursive_predict, preprocess_for_inference
from dataset import load_artifacts


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

class ModelState:
    model:     Optional[CMAPSS_CNN_LSTM] = None
    artifacts: Optional[dict] = None
    device:    torch.device = torch.device("cpu")
    config:    dict = {}
    model_info: dict = {}

state = ModelState()


# ---------------------------------------------------------------------------
# Lifespan — load model on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and artifacts on startup, cleanup on shutdown."""
    print("[INFO] Loading model and artifacts...")

    cfg = load_config()
    state.config = cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state.device = device

    # Load preprocessing artifacts
    try:
        state.artifacts = load_artifacts("artifacts")
        print("[INFO] Preprocessing artifacts loaded.")
    except FileNotFoundError:
        print("[WARNING] No preprocessing artifacts found. "
              "Run train.py first.")

    # Load model
    model_path = Path("artifacts/best_model.pt")
    if model_path.exists():
        model = build_model().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        state.model = model

        state.model_info = {
            "epoch":      checkpoint.get("epoch", "unknown"),
            "val_loss":   checkpoint.get("val_loss", "unknown"),
            "val_rmse":   checkpoint.get("val_rmse", "unknown"),
            "val_acc":    checkpoint.get("val_acc", "unknown"),
            "parameters": model.count_parameters(),
            "device":     str(device)
        }
        print(f"[INFO] Model loaded from epoch {state.model_info['epoch']}.")
    else:
        print("[WARNING] No trained model found at artifacts/best_model.pt. "
              "Run train.py first.")

    yield

    print("[INFO] Shutting down API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CMAPSS RUL Prediction API",
    description=(
        "Predicts Remaining Useful Life (RUL) and health state "
        "for turbofan engines using a CNN-LSTM model trained on NASA CMAPSS."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class SensorWindow(BaseModel):
    """Single 30-cycle sensor window for one-shot prediction."""

    sensors: List[List[float]] = Field(
        ...,
        description="30 × 21 sensor readings (all 21 sensors, raw values)",
        min_items=1
    )
    op_settings: List[List[float]] = Field(
        ...,
        description="30 × 3 operating settings [altitude, mach, TRA]",
        min_items=1
    )
    fault_mode: int = Field(
        default=0,
        description="Fault mode: 0=HPC degradation, 1=Fan+HPC degradation",
        ge=0, le=1
    )
    mc_passes: int = Field(
        default=50,
        description="Number of MC Dropout passes for uncertainty estimation",
        ge=1, le=200
    )


class RecursiveInput(BaseModel):
    """Full sensor sequence for recursive lifecycle prediction."""

    sensors: List[List[float]] = Field(
        ...,
        description="N × 21 sensor readings for entire observed life",
        min_items=30
    )
    op_settings: List[List[float]] = Field(
        ...,
        description="N × 3 operating settings",
        min_items=30
    )
    fault_mode: int = Field(default=0, ge=0, le=1)
    mc_passes:  int = Field(default=50, ge=1, le=200)


class PredictionResponse(BaseModel):
    rul_mean:     float
    rul_std:      float
    health_class: int
    health_label: str
    class_probs:  Dict[str, float]
    confidence:   float
    alert:        Optional[str]


class RecursivePredictionResponse(BaseModel):
    n_cycles:     int
    rul_mean:     List[float]
    rul_std:      List[float]
    health_class: List[int]
    health_label: List[str]
    current_rul:  float
    current_std:  float
    current_class: int
    current_label: str
    alert:        Optional[str]


# ---------------------------------------------------------------------------
# Helper: generate alert message
# ---------------------------------------------------------------------------

def generate_alert(health_class: int, rul_mean: float, rul_std: float) -> Optional[str]:
    """Generate maintenance alert based on health state and uncertainty."""
    if health_class == 3:
        if rul_std > 10:
            return (f"CRITICAL: Estimated RUL = {rul_mean:.0f} ± {rul_std:.0f} cycles. "
                    f"High uncertainty — immediate inspection recommended.")
        return (f"CRITICAL: Estimated RUL = {rul_mean:.0f} cycles. "
                f"Schedule immediate maintenance.")
    elif health_class == 2:
        return (f"WARNING: Estimated RUL = {rul_mean:.0f} ± {rul_std:.0f} cycles. "
                f"Schedule maintenance within next service window.")
    elif health_class == 1 and rul_mean < 100:
        return (f"MONITOR: Degradation detected. "
                f"Estimated RUL = {rul_mean:.0f} cycles.")
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """API health check."""
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "artifacts_loaded": state.artifacts is not None
    }


@app.get("/model/info")
async def model_info():
    """Model metadata and training statistics."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return state.model_info


@app.post("/predict", response_model=PredictionResponse)
async def predict(body: SensorWindow):
    """
    Single window prediction.

    Accepts the most recent 30 cycles of sensor data.
    Returns RUL estimate with uncertainty and health class.
    """
    if state.model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Run train.py first.")
    if state.artifacts is None:
        raise HTTPException(status_code=503,
                            detail="Artifacts not loaded. Run train.py first.")

    sensors     = np.array(body.sensors,     dtype=np.float32)
    op_settings = np.array(body.op_settings, dtype=np.float32)

    if sensors.shape[0] < 1:
        raise HTTPException(status_code=422,
                            detail="At least 1 cycle of sensor data required.")
    if sensors.shape[1] != 21:
        raise HTTPException(status_code=422,
                            detail=f"Expected 21 sensors, got {sensors.shape[1]}.")

    # Preprocess
    try:
        norm_seq, static_vec = preprocess_for_inference(
            sensors, op_settings, state.artifacts
        )
        # Override fault mode from request
        fault_onehot       = np.eye(2, dtype=np.float32)[body.fault_mode]
        static_vec[6:8]    = fault_onehot
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Preprocessing error: {str(e)}")

    # Build window (last 30 cycles or pad)
    window_size = state.artifacts["window_size"]
    n = len(norm_seq)
    if n >= window_size:
        window = norm_seq[-window_size:]
    else:
        pad    = np.repeat(norm_seq[0:1], window_size - n, axis=0)
        window = np.concatenate([pad, norm_seq], axis=0)

    x_t = torch.FloatTensor(window).unsqueeze(0).to(state.device)
    s_t = torch.FloatTensor(static_vec).unsqueeze(0).to(state.device)

    # Predict
    rul_mean, rul_std, health_class, class_probs, _ = predict_single(
        state.model, x_t, s_t,
        hidden=None,
        mc_passes=body.mc_passes,
        device=state.device
    )

    class_names = ["Healthy", "Degrading", "Warning", "Critical"]
    confidence  = float(class_probs[health_class])
    alert       = generate_alert(health_class, rul_mean, rul_std)

    return PredictionResponse(
        rul_mean=round(rul_mean, 2),
        rul_std=round(rul_std, 2),
        health_class=health_class,
        health_label=class_names[health_class],
        class_probs={
            name: round(float(p), 4)
            for name, p in zip(class_names, class_probs)
        },
        confidence=round(confidence, 4),
        alert=alert
    )


@app.post("/predict/recursive", response_model=RecursivePredictionResponse)
async def predict_recursive(body: RecursiveInput):
    """
    Full lifecycle recursive prediction.

    Accepts the complete observed sensor history for one engine.
    Returns RUL and health class for every cycle.
    LSTM hidden state is carried forward between cycles.
    """
    if state.model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Run train.py first.")

    sensors     = np.array(body.sensors,     dtype=np.float32)
    op_settings = np.array(body.op_settings, dtype=np.float32)

    if sensors.shape[1] != 21:
        raise HTTPException(status_code=422,
                            detail=f"Expected 21 sensors, got {sensors.shape[1]}.")

    # Preprocess
    try:
        norm_seq, static_vec = preprocess_for_inference(
            sensors, op_settings, state.artifacts
        )
        fault_onehot    = np.eye(2, dtype=np.float32)[body.fault_mode]
        static_vec[6:8] = fault_onehot
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Preprocessing error: {str(e)}")

    # Recursive prediction
    results = recursive_predict(
        model=state.model,
        sensor_sequence=norm_seq,
        static_vec=static_vec,
        window_size=state.artifacts["window_size"],
        mc_passes=body.mc_passes,
        device=state.device
    )

    class_names   = ["Healthy", "Degrading", "Warning", "Critical"]
    health_labels = [class_names[c] for c in results["health_class"]]

    current_class = int(results["health_class"][-1])
    current_rul   = float(results["rul_mean"][-1])
    current_std   = float(results["rul_std"][-1])
    alert         = generate_alert(current_class, current_rul, current_std)

    return RecursivePredictionResponse(
        n_cycles=results["n_cycles"],
        rul_mean=[round(float(v), 2) for v in results["rul_mean"]],
        rul_std=[round(float(v), 2)  for v in results["rul_std"]],
        health_class=results["health_class"].tolist(),
        health_label=health_labels,
        current_rul=round(current_rul, 2),
        current_std=round(current_std, 2),
        current_class=current_class,
        current_label=class_names[current_class],
        alert=alert
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    uvicorn.run(
        "serve:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=False
    )
