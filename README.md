<div align="center">

# 🏥 CARE'OS — Clinical AI Real-time Evaluation & Operations System
### Complete Three-Module Clinical Architecture

*Dynamic Queueing • Ambient Clinical AI Scribe • Precision Pharmacogenomics*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green.svg)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)](https://fastapi.tiangolo.com/)
[![CPIC](https://img.shields.io/badge/CPIC-Level%20A-purple.svg)](https://cpicpgx.org/)
[![FHIR R4](https://img.shields.io/badge/HL7-FHIR%20R4-firebrick.svg)](https://hl7.org/fhir/R4/)

</div>

---

> **⚕️ CLINICAL USE BOUNDARY**: CARE'OS is clinical decision support software designed for clinician review only. It is **NEVER** an automatic prescription system, diagnostic tool, or autonomous agent. All predictions, pharmacogenomic alerts, and draft notes are suggestions requiring independent clinician sign-off.

---

## 🏛️ End-to-End System Architecture

CARE'OS unifies three interconnected clinical modules through an event-driven telemetry and inter-module bus:

```
                            CLINICAL AUDIO / ENCOUNTER
                                        │
                                        ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                 MODULE 2: AMBIENT CLINICAL AI SCRIBE                   │
    │  • Audio Ingestion & Diarization (WhisperX / pyannote)                 │
    │  • Medical ASR Post-Correction (Phonetic + RxNorm grounding)           │
    │  • Clinical NLP & SOAP Extractor (Med7 / Posology)                     │
    │  • Dual-Pass Hallucination Verifier (MedNLI / DeBERTa-v3)              │
    └───────────────────────┬────────────────────────┬───────────────────────┘
                            │                        │
       Live Consultation    │                        │ Prescribed Medications
       SOAP Stage Telemetry │                        │ & Patient Clinical Data
       (S ➔ O ➔ A ➔ P)      ▼                        ▼ (CPIC Level 1A Triggers)
    ┌───────────────────────────────────┐    ┌──────────────────────────────────┐
    │    MODULE 1: QUEUE & WAIT-TIME    │    │      MODULE 3: HELIX GENOMICS    │
    │             PREDICTOR             │    │         RECOMMENDATION ENGINE    │
    │  • Calibrated SimPy Digital Twin  │    │  • 200+ SNP PyTorch Attention    │
    │  • Two-Stage ML (XGBoost/LightGBM)│    │    Variational Autoencoder (VAE) │
    │  • 90% CQR + Online ACI Intervals │    │  • Multi-Modal Cross-Attention   │
    │  • JSEW Workload-Aware Routing    │    │    Fusion (Genomics + Labs)      │
    │  • TreeSHAP Patient Explanations  │    │  • Conformal Dose CI Regression  │
    │  • Time-to-Leave Virtual Pacer    │    │  • HL7 FHIR R4 DetectedIssues    │
    └───────────────────────────────────┘    └──────────────────────────────────┘
```

---

## 📦 System Modules Overview

### 1. Module 1: Dynamic Outpatient Queue & Wait-Time Predictor
* **Directory**: `src/` (API in `src/api/`, UI in `src/dashboard/`, ML in `src/features/` & `src/models/`, Simulation in `src/simulation/`)
* **Core ML**: Two-Stage Workload Decomposition (XGBoost consultation duration $\hat{S}_j$ $\rightarrow$ LightGBM wait time with Asymmetric Quad-Quad Loss $\alpha=0.80$).
* **Uncertainty Quantification**: Conformalized Quantile Regression (CQR) + Adaptive Conformal Inference (ACI) guaranteeing 90% coverage during sudden walk-in surges.
* **Triage Routing**: Join-the-Shortest-Expected-Workload (JSEW) multi-doctor dynamic balancing.
* **Telemetry**: Ingests live S-O-A-P stage transitions from Module 2 to refine ongoing consult times.

### 2. Module 2: Ambient AI Scribe & 2nd-Pass Verifier
* **Directory**: `module2_careos/`
* **Core NLP**: Diarized audio transcription $\rightarrow$ ASR post-correction grounded in RxNorm $\rightarrow$ Med7 Posology entity extraction $\rightarrow$ Structured SOAP note synthesis.
* **Safety Verification**: 2nd-pass MedNLI and DeBERTa-v3 natural language inference to flag and eliminate medical hallucinations.
* **Inter-Module Bus**: Emits live consultation telemetry to Module 1 and triggers Module 3 upon detecting CPIC Level A medications.

### 3. Module 3: Helix Genomics-Informed Treatment Recommendation Engine
* **Directory**: `careos_genomics/`
* **Deep Learning**: 16-dimensional Latent Self-Attention Variational Autoencoder (VAE) compressing 200+ single nucleotide polymorphisms (SNPs).
* **Multi-Modal Fusion**: Cross-attends genomic latent embeddings against clinical laboratory values (Age, Weight, CrCl, ALT, AST, INR).
* **Precision Guidance**: Predicts drug response categories (Responder, Intermediate, Adverse Risk) and optimal starting doses with 90% conformal safety intervals.

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Module 1 (Queue Predictor API & UI)
```bash
# Terminal 1: Start FastAPI backend
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Start Streamlit Dashboard
streamlit run src/dashboard/streamlit_app.py
```

### 3. Run Integrated Module 2 + Module 3 Pipeline
```bash
# Run end-to-end CLI workflow
python run_integrated_careos.py

# Launch unified clinical workstation
streamlit run integrated_dashboard.py
```

### 4. Run Test Suites
```bash
python -m unittest discover tests -v
```

---

## 👥 Authors & Academic Affiliation
CARE'OS Clinical AI Real-Time Evaluation & Operations System — Bennett University CSET343 Project.
Academic and clinical research prototype.
