# CAREOS Module 1: Dynamic Outpatient Queue & Wait-Time Predictor

## 🏥 Overview

**CAREOS** (Clinical AI Real-time Evaluation & Operations System) Module 1 is a production-grade Dynamic Outpatient Queue & Wait-Time Predictor that uses advanced machine learning and simulation techniques to accurately predict patient waiting times in outpatient departments.

### Key Features

- **Calibrated SimPy Digital Twin**: Non-stationary, multi-class priority queueing network (M_t/G_i/c) calibrated to real clinical datasets (Hangu, PriMock57, Brazil No-Show)
- **Two-Stage ML Pipeline**: 
  - Stage 1: XGBoost consultation duration predictor
  - Stage 2: LightGBM wait-time predictor with Asymmetric Quad-Quad Loss (α=0.8)
- **Conformalized Quantile Regression (CQR) + Adaptive Conformal Inference (ACI)**: 90% calibrated prediction intervals that adapt online to clinic surges
- **TreeSHAP Patient Explanations**: Transparent, patient-friendly wait-time breakdowns
- **JSEW Doctor Routing**: Join-the-Shortest-Expected-Workload for intelligent multi-doctor assignment
- **SOAP Telemetry Integration**: Cross-module feedback from Module 2 (Ambient Scribe) to track consultation progress in real-time
- **LSTM Sequence Encoder**: Captures sequential congestion patterns
- **Virtual Queue Time-to-Leave Pacer**: Notifies remote patients when to head to the clinic

## 📁 Project Structure

```
CAREOS-Module-1/
├── configs/
│   └── clinic_config.py          # Empirical parameters (Hangu/PriMock57/Brazil)
├── src/
│   ├── simulation/
│   │   └── clinic_simpy_twin.py  # Calibrated SimPy Digital Twin
│   ├── features/
│   │   └── queue_feature_engine.py # 16-group, 28-variable feature engineering
│   ├── models/
│   │   └── train_pipeline.py     # Two-Stage ML + CQR/ACI + LSTM
│   ├── api/
│   │   └── main.py               # FastAPI event-driven backend
│   └── dashboard/
│       └── streamlit_app.py      # Three-persona Streamlit UI
├── data/                          # Generated simulation data
├── models/                        # Trained model artifacts
├── notebooks/                     # Analysis notebooks
├── requirements.txt
├── run_pipeline.py               # End-to-end pipeline runner
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (Simulation → Training → Evaluation)

```bash
python run_pipeline.py
```

This will:
- Run 100 clinic sessions simulating ~6,000+ patient journeys
- Compute 28 queue-mining features across 16 feature groups
- Train Stage-1 Duration Predictor (XGBoost)
- Train Stage-2 Wait-Time Predictor (LightGBM + Asymmetric Loss + CQR + ACI)
- Train LSTM Sequence Encoder
- Output evaluation metrics

### 3. Start the API Server

```bash
uvicorn src.api.main:app --reload
```

API docs available at: http://localhost:8000/docs

### 4. Launch the Dashboard

```bash
streamlit run src/dashboard/streamlit_app.py
```

## 📊 Evaluation Results

| Metric | Rolling Avg Baseline | Basic GBDT (6 features) | **CAREOS Module 1** |
|--------|---------------------|------------------------|---------------------|
| MAE | 18.05 min | 8.39 min | **5.92 min** |
| RMSE | 27.04 min | 13.69 min | **10.72 min** |
| PICP (90% target) | 70.7% | 77.9% | **90.3%** |
| SUR > 5min | 63.8% | 35.2% | **21.1%** |

## 🔬 Six Novel Contributions

1. **Two-Stage Workload Decomposition**: Predicts individual consultation durations → sums expected workload ahead
2. **Cross-Module SOAP Telemetry**: Live consultation progress from Module 2 (Ambient Scribe)
3. **CQR + ACI Intervals**: Distribution-free, online-adaptive 90% prediction intervals
4. **Asymmetric Quad-Quad Loss + DAES**: Penalizes underestimation 4× more than overestimation
5. **TreeSHAP Patient Explanations**: Transparent wait-time breakdowns with Virtual Queue Pacer
6. **JSEW Prescriptive Routing**: Workload-aware doctor assignment (51.3% wait reduction vs. Round-Robin)

## 🏗️ Technical Architecture

### Empirical Data Grounding
- **Hangu Dataset**: 6,637 outpatient consultations with Log-Normal calibration
- **PriMock57**: 57 primary care consultations (12 complaint categories)
- **MIMIC-IV-ED**: 5-level ESI acuity triage
- **Brazil No-Shows**: 110,527 appointments with lead-time dependent no-show rates

### Queue-Mining Features (28 Variables, 16 Groups)
1. Temporal & Diurnal Cyclical (sin/cos time, day_of_week, arrival_rate)
2. Queue-Length & Q-Lasso Fluid Estimators
3. Delay-History & Elapsed State (rolling waits, HOL elapsed, conditional remaining time)
4. Two-Stage Workload & Clinical (pred_own_duration, expected_queued_workload, physician PACT/PNP)

### API Endpoints
- `POST /api/v1/queue/check-in` — Patient check-in with JSEW routing
- `POST /api/v1/queue/start-consultation` — Doctor calls patient, ACI update
- `POST /api/v1/queue/soap-telemetry` — Module 2 → Module 1 SOAP webhook
- `POST /api/v1/queue/emergency-preempt` — Emergency insertion
- `POST /api/v1/queue/end-consultation` — Consultation complete, update PACT

## 👥 Team

CAREOS - Bennett University CSET343 Project

## 📄 License

Academic use only — Bennett University
