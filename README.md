<div align="center">

# 🧬 CARE'OS — Integrated Clinical Decision Support System

### Module 2: Ambient AI Scribe & 2nd-Pass Verifier
### Module 3: Genomics-Informed Treatment Recommendation Engine

*Speech-to-Text • Clinical NLP • NLI Hallucination Verification • Deep Attention-VAE • Precision Pharmacogenomics*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![CPIC](https://img.shields.io/badge/CPIC-Level%20A-purple.svg)](https://cpicpgx.org/)
[![FHIR R4](https://img.shields.io/badge/HL7-FHIR%20R4-firebrick.svg)](https://hl7.org/fhir/R4/)
[![Tests Passing](https://img.shields.io/badge/Tests-62%20Passed%20(100%25)-success.svg)](tests/)

</div>

---

> **⚕️ CLINICAL USE BOUNDARY**: This software is clinical decision support for clinician review only. It is **NEVER** an automatic prescription system, diagnostic tool, or autonomous agent. All pharmacogenomic alerts and draft notes are suggestions requiring independent clinician sign-off.

---

## 🏗️ Integrated System Architecture

CARE'OS connects Module 2 (AI Scribe) and Module 3 (Genomics Engine) via a high-throughput, low-latency in-process event bus (`InterModuleBus`):

```
                                  CLINICAL AUDIO (WAV / MP3)
                                              │
                                              ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MODULE 2: AMBIENT AI SCRIBE & VERIFIER                         │
│                                                                                        │
│  Stage 1: Audio Pipeline        Stage 2: Medical ASR Post-Correction                   │
│  WhisperX + pyannote diarization ──► Phonetic + Levenshtein + RxNorm Grounding         │
│                                              │                                         │
│                                              ▼                                         │
│                                 Stage 3: Clinical NLP & SOAP Note                     │
│                                 Med7 Posology Extractor + Constrained Pydantic Schema  │
│                                 (Chief Complaint, Diagnoses, Medications, Vitals)      │
└──────────────────────────────────────────────┬─────────────────────────────────────────┘
                                               │
                                               │ Trigger Keys (ICD-10 Codes + Drug Names)
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      INTER-MODULE BUS (CPIC Level A Event Bridge)                       │
│  • Matches prescribed medications against CPIC Level 1A watchlist                     │
│  • Intercepts clinical lab telemetry & patient genomic profile                         │
│  • Fires real ML analysis to Module 3 in <10ms                                         │
└──────────────────────────────────────────────┬─────────────────────────────────────────┘
                                               │
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      MODULE 3: PHARMACOGENOMICS ML ENGINE                              │
│                                                                                        │
│  ┌─────────────────────────┐  ┌───────────────────────────┐  ┌──────────────────────┐  │
│  │  High-Dimensional SNPs  │  │  Clinical Lab Features    │  │  Gene Expression     │  │
│  │  (0, 1, or 2 alleles)   │  │  (Age, CrCl, ALT/AST, INR)│  │  (CYP2C9, CYP2C19...)│  │
│  └────────────┬────────────┘  └─────────────┬─────────────┘  └──────────┬───────────┘  │
│               │                             │                           │              │
│      ┌────────▼────────┐                    │                           │              │
│      │ PyTorch Attention│                    │                           │              │
│      │ VAE (Latent z)  │                    │                           │              │
│      └────────┬────────┘                    │                           │              │
│               │                             │                           │              │
│               └───────────────┬─────────────┴───────────────────────────┘              │
│                               ▼                                                        │
│                  Cross-Attention Multi-Modal Fusion                                    │
│                               │                                                        │
│                ┌──────────────┴──────────────┐                                         │
│                ▼                             ▼                                         │
│      Optuna-Tuned XGBoost         Gradient Boosting Regressor                          │
│      (Responder / Adverse Risk)   (Optimal Dose + 90% Conformal CI)                    │
│                │                             │                                         │
│                └──────────────┬──────────────┘                                         │
│                               ▼                                                        │
│                 SHAP Explainer + CPIC Knowledge Base                                   │
└──────────────────────────────────────────────┬─────────────────────────────────────────┘
                                               │
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       ENRICHED CLINICAL ARTIFACTS & EXPORTS                            │
│                                                                                        │
│  1. Draft SOAP Note: Appends atomic claims `[PGx Decision Support]` to Plan            │
│  2. PGx SOAP Appendix: Comprehensive variant, phenotype, confidence, and CI report     │
│  3. Stage 4 Verifier: Dual-tier 2nd-pass MedNLI + DeBERTa-v3 verification              │
│  4. FHIR R4 Bundle: Generates `DetectedIssue` resources for active PGx alerts          │
│  5. Patient Summary: Bilingual (EN / Hinglish) personalized medication warnings        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start & Execution

### 1. Installation

```bash
# Clone the repository and checkout the integrated branch
git clone https://github.com/Z3phyru5646/CAREOS.git
cd CAREOS
git checkout Genomic-Treatment-Recommendation

# Install dependencies (Python 3.10+ recommended)
pip install torch xgboost optuna shap scikit-learn pydantic streamlit fastapi soundfile pyyaml
```

### 2. Run Integrated Pipeline Demo (CLI)

Run the full end-to-end clinical workflow from simulated audio through transcript, verifier, genomics engine, and FHIR export:

```bash
python run_integrated_careos.py
```

Options:
```bash
python run_integrated_careos.py --audio path/to/consultation.wav   # Custom audio
python run_integrated_careos.py --patient-id PT-10492              # Specific patient
python run_integrated_careos.py --language hinglish                # Hinglish code-switching
python run_integrated_careos.py --dashboard                        # Launch Streamlit GUI
```

### 3. Launch Unified Catppuccin Clinical Workstation

Launch the unified Streamlit clinical workstation featuring the **Catppuccin Mocha** palette:

```bash
streamlit run integrated_dashboard.py
```

Access the UI at `http://localhost:8501`:
- **Tab 1: 🎙️ AI Scribe Pipeline** — Real-time audio upload, diarization, SOAP note, and claim verification
- **Tab 2: 🧬 Genomics Engine** — Standalone trigger console, SNP sliders, SHAP attribution plots, and dose CI
- **Tab 3: 🚨 PGx Alerts** — Clinician decision-support console with severity badges and full SOAP appendices
- **Tab 4: 📤 Exports** — HL7 FHIR R4 document bundle (`DetectedIssue`), E/M coding readiness, and After-Visit Summary

### 4. Standalone Module 3 Engine Runner

```bash
python run_demo.py          # Full synthetic cohort demo
python run_demo.py --fast   # Fast demo (fewer iterations)
```

---

## 🧪 Comprehensive Test Suite (62/62 Passing)

The repository features 100% passing test coverage across all subsystems:

```bash
python -m unittest discover tests -v
```

| Test Suite | File | Tests | Status |
|------------|------|-------|--------|
| **Module 3 Unit Tests** | `tests/test_careos_advanced.py` | 27 | ✅ 27 Passed |
| **Module 3 Integration** | `tests/test_genomics_engine.py` | 3 | ✅ 3 Passed |
| **Module 2 Unit Tests** | `tests/test_pipeline_unittest.py` | 5 | ✅ 5 Passed |
| **Module 2 Verification** | `tests/test_pipeline.py` | 14 | ✅ 14 Passed |
| **Integrated E2E Pipeline** | `tests/test_integrated_pipeline.py` | 13 | ✅ 13 Passed |
| **TOTAL** | — | **62** | **✅ 100% PASSING** |

---

## 🧬 Key Innovations

### 1. Multi-Head Attention Variational Autoencoder (VAE)
- Compresses 200+ single nucleotide polymorphisms (SNPs) into a 16-dimensional dense latent vector.
- Self-attention blocks model epistatic non-linear gene interactions ($CYP2C9 \times VKORC1$).
- Cyclical KL annealing prevents posterior collapse.

### 2. Multi-Modal Cross-Attention Fusion
- Cross-attends genomic latent representations against clinical lab markers ($Age, Weight, CrCl, ALT, AST, INR$).
- Gated residual bypass prevents genomic feature dilution in patients with normal organ function.

### 3. Conformal Prediction Dose Regression
- Predicts optimal initial starting doses with distribution-free 90% confidence intervals ($[lower, upper]$).
- Conformal guarantees provide safety bounds for critical anticoagulants (warfarin) and immunosuppressants.

### 4. Dual-Pass Hallucination Verification
- Tier 1: Lexical verbatim string matching.
- Tier 2: Ontology grounding (RxNorm / UMLS / SNOMED CT).
- Tier 3: DeBERTa-v3 NLI inference verification.
- Distinguishes permissible clinical inferences from literal hallucinations.

### 5. HL7 FHIR R4 Interoperability
- Maps active PGx warnings into FHIR R4 `DetectedIssue` resources linked to the patient's `Encounter`.
- Includes evidence detail with gene symbol, metabolizer phenotype, predicted response category, and conformal dose interval.

---

## 🎨 Catppuccin Mocha UI Palette

The unified interface is crafted in the soothing, high-contrast Catppuccin Mocha theme:

| Role | Color | Hex Code | Purpose |
|------|-------|----------|---------|
| **Primary Accent** | Mauve | `#cba6f7` | Active triggers, ML badges, DNA header |
| **Secondary Accent** | Sapphire | `#74c7ec` | Clinical scribe, audio timestamps |
| **Success** | Green | `#a6e3a1` | Responder phenotype, Tier 1 verbatim claims |
| **Warning** | Yellow / Peach | `#f9e2af` / `#fab387` | Moderate risk, Intermediate metabolizers |
| **Critical Alert** | Red | `#f38ba8` | Contraindications, adverse risk, Poor metabolizers |
| **Dark Base** | Base / Mantle | `#1e1e2e` / `#181825` | Ergonomic dark-mode workspace background |

---

## 📦 Directory Structure

```
├── careos_genomics/               # Module 3 Core ML Package
│   ├── autoencoder.py             # PyTorch Self-Attention VAE
│   ├── fusion.py                  # Multi-Modal Cross-Attention Fusion
│   ├── engine.py                  # GenomicsEngine Orchestrator
│   ├── knowledge_base.py          # CPIC Level A Rules & Guidelines
│   ├── data_models.py             # Pydantic Schemas & DTOs
│   ├── synthetic_data.py          # Synthetic PGx Cohort Generator
│   └── dashboard.py               # Standalone Module 3 Streamlit App
├── module2_careos/                # Module 2 Scribe & Verifier Package
│   ├── stage1_audio/              # Audio ingestion & diarization
│   ├── stage2_correction/         # Medical ASR post-correction
│   ├── stage3_clinical_nlp/       # Med7 clinical entity extraction & SOAP
│   ├── stage4_verifier/           # Dual-tier 2nd-pass verification
│   ├── stage5_export/             # FHIR R4, E/M coding, patient summary
│   ├── inter_module_bus/          # Bidirectional CPIC & telemetry bus
│   ├── copilot/                   # Clinical Co-Pilot Whisperer
│   └── main.py                    # Module 2 Pipeline Orchestrator
├── tests/                         # Full Unified Test Suite
│   ├── test_careos_advanced.py    # Module 3 advanced tests
│   ├── test_genomics_engine.py    # Module 3 integration tests
│   ├── test_pipeline_unittest.py  # Module 2 unit tests
│   ├── test_pipeline.py           # Module 2 integration tests
│   └── test_integrated_pipeline.py# Integrated Module 2+3 E2E tests
├── data/                          # Sample Audio & Clinical Datasets
│   ├── sample_audio/demo.wav      # Sample clinical consultation
│   └── output/                    # Pipeline JSON results
├── integrated_dashboard.py        # Unified Streamlit Workstation
├── run_integrated_careos.py       # Integrated Pipeline CLI Runner
├── run_demo.py                    # Standalone Module 3 Demo Runner
└── README.md                      # Complete System Documentation
```

---

## 📜 Clinical Disclaimer & Licensing

CARE'OS is an open-source clinical research prototype. Pharmacogenomic guidance is strictly based on published [CPIC Guidelines](https://cpicpgx.org/) and [PharmGKB](https://www.pharmgkb.org/). All clinical recommendations must be reviewed and verified by a licensed healthcare professional before prescribing.
