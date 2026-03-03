# CMAPSS RUL Prediction — CNN-LSTM

Predicts **Remaining Useful Life (RUL)** and **health state** of turbofan engines using a multi-scale 1D CNN → stacked LSTM → attention → dual head architecture trained on NASA CMAPSS.

---

## Architecture

```
Static features (op_cluster + fault_mode)
        ↓
Linear → tanh → h₀, c₀
        ↓
Sensor sequence → Multi-scale 1D CNN (kernels 3,5,7)
        ↓
LSTM Layer 1 (h₀, c₀) + MC Dropout
        ↓
LSTM Layer 2 + MC Dropout
        ↓
Attention Layer
        ↓
   ┌────┴────┐
RUL Head  Class Head
(MSE)     (Focal CE)
```

**Health classes:**
| Class | Label | RUL Range |
|---|---|---|
| 0 | Healthy | > 125 cycles |
| 1 | Degrading | 75–125 cycles |
| 2 | Warning | 25–75 cycles |
| 3 | Critical | ≤ 25 cycles |

---

## Project Structure

```
cmapss-rul/
├── data/raw/               ← Place CMAPSS .txt files here
├── src/
│   ├── dataset.py          ← Preprocessing pipeline
│   ├── model.py            ← CNN-LSTM architecture
│   ├── train.py            ← Training loop + MLflow
│   ├── evaluate.py         ← Metrics + visualisation
│   ├── predict.py          ← Recursive inference + MC Dropout
│   └── serve.py            ← FastAPI endpoint
├── configs/
│   └── config.yaml         ← All hyperparameters
├── notebooks/
│   └── results.ipynb       ← Visualise results
├── artifacts/              ← Saved model + scalers (auto-created)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download CMAPSS data
Download from [NASA Prognostics Center](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

Place in `data/raw/`:
```
data/raw/train_FD001.txt
data/raw/train_FD002.txt
data/raw/train_FD003.txt
data/raw/train_FD004.txt
data/raw/test_FD001.txt
data/raw/test_FD002.txt
data/raw/test_FD003.txt
data/raw/test_FD004.txt
data/raw/RUL_FD001.txt
data/raw/RUL_FD002.txt
data/raw/RUL_FD003.txt
data/raw/RUL_FD004.txt
```

### 3. Train
```bash
cd src
python train.py
```

### 4. View results in MLflow
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 5. Serve API
```bash
cd src
uvicorn serve:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API health check |
| GET | `/model/info` | Model metadata |
| POST | `/predict` | Single window prediction |
| POST | `/predict/recursive` | Full lifecycle prediction |

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensors": [[...30 cycles of 21 sensor values...]],
    "op_settings": [[...30 cycles of 3 op settings...]],
    "fault_mode": 0,
    "mc_passes": 50
  }'
```

### Example Response
```json
{
  "rul_mean": 42.3,
  "rul_std": 5.1,
  "health_class": 2,
  "health_label": "Warning",
  "class_probs": {
    "Healthy": 0.02,
    "Degrading": 0.15,
    "Warning": 0.71,
    "Critical": 0.12
  },
  "confidence": 0.71,
  "alert": "WARNING: Estimated RUL = 42 ± 5 cycles. Schedule maintenance within next service window."
}
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Framework | PyTorch | Best for recursive LSTM, custom loss, ONNX export |
| RUL cap | 125 cycles | Focuses model on detectable degradation |
| Static injection | Hidden state init (h₀, c₀) | Context shapes entire temporal reasoning |
| Loss | α·MSE + (1-α)·FocalCE | Jointly optimises RUL regression + classification |
| Uncertainty | MC Dropout (T=50) | Free uncertainty quantification |
| Inference | Recursive | Arbitrary horizon, no fixed N required |
| Normalisation | Cluster-wise Z-score | Removes condition-induced variance |
| Experiment tracking | MLflow | Free, local, production standard |
