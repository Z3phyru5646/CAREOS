<div align="center">

# 🏥 CARE'OS — Clinical AI Real-time Evaluation & Operations System
### Unified Three-Module Autonomous Outpatient Intelligence Platform

*Dynamic Outpatient Queue Management • Ambient Clinical AI Scribe • Precision Pharmacogenomics*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-12.0%20%7C%20RTX%204060-success.svg)](https://developer.nvidia.com/cuda-zone)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![CPIC Level A](https://img.shields.io/badge/CPIC-Level%20A%20%26%20B-purple.svg)](https://cpicpgx.org/)
[![HL7 FHIR R4](https://img.shields.io/badge/HL7-FHIR%20R4%20Compliant-firebrick.svg)](https://hl7.org/fhir/R4/)
[![Tests Passing](https://img.shields.io/badge/Tests-69%20Passed%20(100%25)-success.svg)](tests/)

</div>

---

> **⚕️ CLINICAL DECISION SUPPORT BOUNDARY**: CARE'OS is designed strictly as assistive decision support for credentialed clinicians. It is **NEVER** an autonomous diagnostic tool or automatic prescription generator. All triage scores, queue adjustments, transcribed SOAP notes, and pharmacogenomic recommendations are non-blocking suggestions requiring independent clinical verification and physician sign-off.

---

## 🌟 System Overview & Three-Module Architecture

CARE'OS unifies three interconnected, high-performance clinical intelligence engines into a single outpatient operations platform:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               MODULE 1: OUTPATIENT QUEUE & HEALTH-STATUS TRIAGE PREDICTOR              │
│  • Patient Arrival & Check-In (Vitals: SpO2, SBP/DBP, HR, RR, Temp, Pain 0-10)         │
│  • GPU Clinical Health-Status Triage Engine (NEWS2/qSOFA + GPU Semantic Complaint NLP) │
│  • Dynamic Starvation-Free Priority Queue Position Adjustment (HAI + Aging Factor)     │
│  • Multi-Doctor Join-the-Shortest-Expected-Workload (JSEW) Dynamic Routing             │
│  • Conformalized Quantile Regression (CQR) Wait-Time Predictions (90% Confidence CI)  │
└──────────────────────────────────────────────┬─────────────────────────────────────────┘
                                               │
                                               │ 1. Patient Called to Consultation Room
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MODULE 2: AMBIENT AI SCRIBE & VERIFIER                          │
│  • Stage 1: Audio Preprocessing & Diarization + Whisper on CUDA (NVIDIA RTX 4060)      │
│  • Stage 2: Medical ASR Post-Correction with RxNorm Phonetic Grounding                 │
│  • Stage 3: Clinical NLP Posology Extraction & Constrained Section-by-Section SOAP     │
│  • Stage 4: Dual-Tier 2nd-Pass Verification (Literal vs. Inference Hallucination)      │
│  • Live SOAP Stage Telemetry Dispatcher: S ➔ O ➔ A ➔ P updates Module 1 in real time   │
└──────────────────────────────────────────────┬─────────────────────────────────────────┘
                                               │
                                               │ 2. Extracted Prescriptions (CPIC Level A Trigger)
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      MODULE 3: PRECISION PHARMACOGENOMICS ML ENGINE                    │
│  • High-Dimensional SNP Genotypes (230+ Loci) ➔ PyTorch Self-Attention VAE (Latent z)  │
│  • Gated Cross-Attention Multi-Modal Fusion (Genomics + Transcriptomics + Lab Vitals)  │
│  • Bayesian-Tuned XGBoost Drug Response Classifier (Responder / Adverse Risk)          │
│  • Gradient Boosting Starting Dose Regressor + 90% Conformal Prediction Intervals      │
│  • TreeSHAP Top-10 Biomarker Feature Attributions & Curated 9-Panel CPIC Knowledge Base│
└──────────────────────────────────────────────┬─────────────────────────────────────────┘
                                               │
                                               │ 3. Closed-Loop Clinical Synthesis
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       UNIFIED CLINICAL ARTIFACTS & EXPORTS                             │
│  1. Enriched Verified SOAP Note: Injects `[PGx Decision Support]` directly into Plan   │
│  2. Dynamic Waiting Room Sync: Live consultation progress recalculates queue wait times│
│  3. HL7 FHIR R4 Bundle: Exports Patient, Encounter, Observation, & DetectedIssue alerts│
│  4. E/M Reimbursement Coding: Computes suggested medical complexity level (e.g., 99214)│
│  5. Bilingual Patient Summaries: English & Hinglish personalized after-visit summaries │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Detailed Module Capabilities

### Module 1: Outpatient Queue & Health Triage Predictor
* **Clinical Health-Status Triage Engine** (`src/features/health_triage_engine.py`):
  * Runs on dedicated GPU (`cuda:0`, NVIDIA GeForce RTX 4060).
  * Evaluates physiological derangements across blood pressure, heart rate, oxygen saturation ($SpO_2$), respiratory rate, temperature, and pain score.
  * Evaluates narrative clinical presentations against emergency prototype manifolds (STEMI/ACS, Stroke, Severe Asthma, Sepsis, Hemorrhage) to compute a continuous **Health Acuity Index (HAI)** $\in [0.0, 1.0]$ and map ESI levels (1–5).
* **Dynamic Anti-Starvation Queue Adjustment**:
  * Prioritizes patients dynamically using:
    $$\text{PriorityScore}_i(t) = 0.65 \cdot \text{HAI}_i + 0.35 \cdot \left(\frac{t - t_{\text{checkin}, i}}{T_{\text{max\_target}}(\text{ESI}_i)}\right)$$
  * High-acuity patients advance forward in queue ($\uparrow +2$ or $+3$ spots), while linear aging prevents starvation of routine follow-up patients.
* **JSEW Workload Balancing & Conformal Wait Times**:
  * Balances patient loads across multi-doctor clinic rooms using Join-the-Shortest-Expected-Workload (JSEW).
  * Computes distribution-free 90% prediction intervals $[t_{\text{lower}}, t_{\text{upper}}]$.

### Module 2: Ambient AI Scribe & 2nd-Pass Verifier
* **GPU-Accelerated Speech-to-Text**:
  * Transcribes real consultation audio with OpenAI Whisper on CUDA with automatic float32 formatting and silence detection.
* **Stage 2 RxNorm Phonetic Post-Correction**:
  * Resolves accented/ASR misspellings into normalized RxNorm entities:
    * `cloppedogril` / `clopadogral` $\rightarrow$ `clopidogrel` (RxCUI C0070166)
    * `ompresol` / `pantoprasal` $\rightarrow$ `omeprazole` / `pantoprazole`
    * `adorvastatin` $\rightarrow$ `atorvastatin`
    * `ecosprin` $\rightarrow$ `aspirin`
    * `tick a griller` $\rightarrow$ `ticagrelor`
* **Stage 3 Clinical NLP & Section-by-Section SOAP**:
  * Deterministically extracts symptoms, pertinent negatives, vitals, and medications.
  * Generates schema-constrained atomic claims across Subjective, Objective, Assessment, and Plan.
* **Stage 4 Dual-Tier 2nd-Pass Verification**:
  * Evaluates atomic claims against transcript source utterances across 5 tiers (T1 Verbatim, T2 Terminology, T3 Inference, T4 Unsupported, T5 Contradiction).
  * Distinguishes between literal and clinically acceptable inference-aware hallucination rates.
* **Real-time SOAP Telemetry Webhook**:
  * Emits live consultation phase heartbeats (`S` $\rightarrow$ `O` $\rightarrow$ `A` $\rightarrow$ `P`) directly to Module 1, ensuring accurate remaining time estimates for waiting room patients.

### Module 3: Precision Pharmacogenomics ML Engine
* **Self-Attentive Variational Autoencoder (VAE)**:
  * Epistatic dimensionality reduction compressing 230+ discrete SNP alleles ($0, 1, 2$) into a smooth 16-dimensional latent manifold $z \sim \mathcal{N}(\mu, \sigma^2)$ using multi-head self-attention and cyclical KL annealing.
* **Gated Cross-Attention Multi-Modal Fusion**:
  * Dynamically weights latent genomics, transcriptomic gene expression, and tabular physiological indicators (CrCl, eGFR, AST, ALT, total bilirubin, baseline INR, age, weight).
* **Bayesian-Optimized Drug Response Classification**:
  * Optuna-tuned XGBoost classifier predicting multi-class patient outcomes (Responder, Non-Responder, Adverse Event Risk) with calibrated posterior probabilities.
* **Conformal Dose Regression**:
  * Gradient boosting continuous starting dose prediction bounded by non-parametric distribution-free 90% conformal prediction intervals.
* **Curated 9-Panel CPIC Guideline Knowledge Base**:
  * Warfarin (`CYP2C9`, `VKORC1`, `CYP4F2`)
  * Clopidogrel (`CYP2C19`)
  * Codeine / Tramadol (`CYP2D6`)
  * Simvastatin (`SLCO1B1`, `ABCG2`)
  * Fluoropyrimidines / 5-FU (`DPYD`)
  * Azathioprine / 6-MP (`TPMT`, `NUDT15`)
  * Tacrolimus (`CYP3A5`)
  * SSRIs / Paroxetine (`CYP2D6`, `CYP2C19`)
  * PPIs / Omeprazole (`CYP2C19`)

---

## 🚀 Quick Start & Launch Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Z3phyru5646/CAREOS.git
cd CAREOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Unified Catppuccin Workstation UI

```powershell
python -m streamlit run integrated_dashboard.py
```

The unified dashboard features 5 cohesive navigation tabs:
1. **🏥 Outpatient Queue & Triage (Module 1)**: Live doctor workstations, patient check-in with vitals and quick presets (🚨 Acute Cardiac, ⚡ Stroke Alert, 🩺 Asthma, 📋 Routine), and the dynamic health-adjusted queue.
2. **🎙️ Ambient AI Scribe (Module 2)**: Audio consultation upload with CUDA Whisper, phonetic drug corrections, and verified SOAP notes.
3. **🧬 Precision Pharmacogenomics (Module 3)**: Standalone CPIC Level A trigger engine, VAE latent epistasis, and TreeSHAP attribution.
4. **🔄 Unified 3-Module Journey**: One-click complete clinical lifecycle (Triage $\rightarrow$ Scribe $\rightarrow$ Telemetry $\rightarrow$ Genomic Alert $\rightarrow$ Final Care Package).
5. **📤 Clinical & FHIR Exports**: HL7 FHIR R4 bundles, E/M coding, and bilingual English/Hinglish summaries.

### 3. (Optional) Run the Module 1 FastAPI REST API

```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🧪 Verification & Automated Test Suite

Run all **69 unit and integration tests** verifying cross-module interoperability:

```powershell
python -m unittest discover tests -v
```

```text
Ran 69 tests in 59.196s
OK
  [Module 1 OK] JSEW Doctor: 0 | Wait: 0.0m [0, 8.0]
  [Module 2 -> Module 1 OK] Telemetry emitted successfully for Room 0 -> PLAN phase
  [Module 2 -> Module 3 OK] CPIC Trigger fired for Clopidogrel -> Gene ['CYP2C19']
```

---

## 📦 Repository Structure

```text
CAREOS/
├── configs/
│   └── clinic_config.py             # Clinic parameters, doctor pacing, and simulation configs
├── careos_genomics/                 # MODULE 3: Precision Pharmacogenomics ML Engine
│   ├── autoencoder.py               # Self-Attentive PyTorch VAE for SNP epistasis
│   ├── dashboard.py                 # Standalone Catppuccin Streamlit dashboard for Module 3
│   ├── data_models.py               # Pydantic data schemas & HL7 FHIR serializers
│   ├── engine.py                    # Unified GenomicsEngine: VAE + Fusion + XGBoost + Conformal
│   ├── fusion.py                    # Cross-attention gated multi-modal fusion network
│   ├── knowledge_base.py            # CPIC Level A/B clinical practice rules
│   └── synthetic_data.py            # Biological cohort generator (Hardy-Weinberg equilibrium)
├── module2_careos/                  # MODULE 2: Ambient AI Scribe & 2nd-Pass Verifier
│   ├── stage1_audio/                # Whisper on CUDA GPU & Pyannote diarization
│   ├── stage2_correction/           # RxNorm phonetic drug name post-correction
│   ├── stage3_clinical_nlp/         # Clinical NLP fact extraction & constrained SOAP drafting
│   ├── stage4_verifier/             # Dual-tier 2nd-pass MedNLI verification engine
│   ├── stage5_export/               # HL7 FHIR R4 bundle & bilingual patient summaries
│   └── inter_module_bus/            # Low-latency in-process event bus & HTTP telemetry bridge
├── src/                             # MODULE 1: Outpatient Queue & Wait-Time Predictor
│   ├── api/main.py                  # FastAPI queue management & telemetry webhook server
│   ├── features/
│   │   ├── health_triage_engine.py  # GPU clinical health triage & dynamic queue position adjuster
│   │   └── queue_feature_engine.py  # 16-group queue-mining feature engineering
│   ├── models/train_pipeline.py     # Stage-1 duration & Stage-2 CQR wait-time regressors
│   └── simulation/clinic_simpy_twin.py # Calibrated SimPy digital twin outpatient simulation
├── tests/                           # 69 automated unit and integration tests
│   ├── test_health_triage.py        # GPU triage & queue adjustment unit tests
│   ├── test_three_modules_integration.py # End-to-end 3-module integration tests
│   ├── test_pipeline.py             # Module 2 pipeline tests
│   └── test_genomics_engine.py      # Module 3 ML engine tests
├── integrated_dashboard.py          # Unified Catppuccin Mocha Streamlit Application (Modules 1+2+3)
├── requirements.txt                 # Unified project dependencies
└── README.md                        # Platform documentation
```

---

## 📜 Compliance, Guidelines & Standards

* **Clinical Guidelines**: CPIC Levels A & B, PharmGKB, Dutch Pharmacogenetics Working Group (DPWG).
* **Interoperability**: HL7 FHIR Release 4 (JSON representation with `DetectedIssue` and `Observation` extensions).
* **Regulatory Stance**: Clinical Decision Support Software per FDA Guidance Section 520(o)(1)(E) and EU MDR Annex VIII Rule 11 (assistive, non-autonomous, full clinician audit trail).
* **Design System**: Catppuccin Mocha Palette with ergonomic dark-mode visual hierarchy.
