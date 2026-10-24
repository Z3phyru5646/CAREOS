<div align="center">

# 🧬 CARE'OS Helix — Module 3

### Genomics-Informed Treatment Recommendation Engine

*Advanced Pharmacogenomic Decision Support for Clinician Review*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![CPIC](https://img.shields.io/badge/CPIC-Guidelines-purple.svg)](https://cpicpgx.org/)

</div>

---

> **⚕️ CLINICAL USE BOUNDARY**: This module is decision-support software for
> clinician review only. It is **NOT** a validated medical device, diagnosis tool,
> or prescription system. All outputs are suggestions requiring independent
> clinical verification. Never deploy on synthetic or unvalidated training data.

---

## 🏗️ Architecture

```
Module 2 (AI Scribe) ──────────────────────────────────────► Module 3
   │ ICD-10 diagnosis codes                                    │
   │ Medication names                                          │
   │ (Trigger Keys)                                            │
   ▼                                                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    CARE'OS Genomics Engine                        │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ CPIC         │  │ SNP Profile  │  │ Clinical Data           │ │
│  │ Knowledge    │  │ (0, 1, 2)    │  │ (age, sex, weight,      │ │
│  │ Base         │  │              │  │  CrCl, liver markers)   │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬──────────────┘ │
│         │                 │                      │                │
│         │        ┌────────▼────────┐             │                │
│         │        │ Variational     │             │                │
│         │        │ Autoencoder     │             │                │
│         │        │ (Self-Attention)│             │                │
│         │        └────────┬────────┘             │                │
│         │                 │ latent z              │                │
│         │        ┌────────▼──────────────────────▼──────────┐    │
│         │        │        Multi-Modal Data Fusion            │    │
│         │        │   (Cross-Attention + Gated Fusion)        │    │
│         │        └────────────────────┬──────────────────────┘    │
│         │                             │                           │
│         │        ┌────────────────────▼──────────────────────┐    │
│         │        │  XGBoost Classifier    │  GBR Dose        │    │
│         │        │  (Optuna-optimized)    │  Regressor       │    │
│         │        │  + Calibration         │  + Conformal CI  │    │
│         │        └───────────┬────────────┴────────┬─────────┘    │
│         │                    │                     │              │
│         │        ┌───────────▼─────────────────────▼─────────┐    │
│         │        │         SHAP Explainability                │    │
│         │        └───────────┬────────────────────────────────┘    │
│         │                    │                                    │
│  ┌──────▼────────────────────▼────────────────────────────────┐   │
│  │              Pharmacogenomic Alert Generator                │   │
│  │  • Gene variant + metabolizer phenotype                     │   │
│  │  • Drug response prediction + probabilities                 │   │
│  │  • Dose estimate + conformal prediction interval            │   │
│  │  • CPIC guideline recommendation + monitoring advice        │   │
│  │  • Drug interaction warnings                                │   │
│  │  • SHAP feature importance explanation                      │   │
│  └───────────────────────────┬────────────────────────────────┘   │
│                              │                                    │
└──────────────────────────────┼────────────────────────────────────┘
                               │
                               ▼
                    Pharmacogenomic Alert (JSON)
                    → Appended to SOAP Note as
                      clinician suggestion
```

## 🚀 Quick Start

### Installation

```bash
cd "Module 3"
pip install -r requirements.txt
```

### Run Demo Pipeline

```bash
# Full demo with synthetic data
python run_demo.py

# Quick demo (fewer samples, faster training)
python run_demo.py --fast

# Launch interactive Catppuccin-themed dashboard
python run_demo.py --dashboard
```

## 📦 Package Structure

```
Module 3/
├── careos_genomics/
│   ├── __init__.py            # Package initialization
│   ├── engine.py              # Core orchestrator pipeline
│   ├── autoencoder.py         # PyTorch VAE with self-attention
│   ├── fusion.py              # Multi-modal cross-attention fusion
│   ├── knowledge_base.py      # CPIC pharmacogenomic rules
│   ├── data_models.py         # Pydantic schemas
│   ├── synthetic_data.py      # Realistic data generator
│   └── dashboard.py           # Catppuccin Streamlit UI
├── tests/
│   ├── test_genomics_engine.py   # Legacy tests
│   └── test_careos_advanced.py   # Comprehensive test suite
├── genomics_engine.py         # Legacy engine (Module 3 v1.0)
├── run_demo.py                # Demo runner script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🧬 Key Features

### 1. Variational Autoencoder with Self-Attention
- Compresses high-dimensional SNP profiles (200+ variants) to compact latent vectors
- Multi-head self-attention captures epistatic SNP interactions
- KL annealing prevents posterior collapse
- Residual connections for training stability

### 2. CPIC Knowledge Base
- **9 curated gene-drug rules** covering major pharmacogenomic interactions:
  - CYP2C9/VKORC1 + Warfarin
  - CYP2C19 + Clopidogrel
  - CYP2D6 + Codeine
  - SLCO1B1 + Simvastatin
  - DPYD + Fluoropyrimidines
  - TPMT/NUDT15 + Thiopurines
  - CYP3A5 + Tacrolimus
  - CYP2D6 + SSRIs
  - CYP2C19 + PPIs

### 3. XGBoost with Bayesian Optimization
- Optuna-powered hyperparameter search
- Calibrated probability estimates (isotonic regression)
- AUC-ROC evaluation with macro one-vs-rest for multiclass

### 4. Conformal Prediction Intervals
- Distribution-free prediction intervals for dose estimates
- Guaranteed coverage probability (default 90%)
- No distributional assumptions required

### 5. SHAP Explainability
- TreeExplainer for every prediction
- Top-10 feature importance ranking
- Direction and magnitude of each feature's contribution

### 6. Drug Interaction Detection
- Cross-references concurrent medications against CPIC guidelines
- Severity-graded interaction warnings
- Evidence-level annotations

## 📋 Output: Pharmacogenomic Alert

```json
{
  "alert_type": "pharmacogenomic_alert",
  "severity": "high",
  "detected_variants": [
    {"gene": "CYP2C9", "variant_id": "rs1799853", "copies_of_variant": 1}
  ],
  "metabolizer_phenotype": "intermediate_metabolizer",
  "drug_response_prediction": {
    "predicted_category": "responder",
    "confidence": 0.72,
    "probabilities": {"responder": 0.72, "non_responder": 0.18, "adverse_event_risk": 0.10}
  },
  "dose_estimate": {
    "predicted_dose": 3.5,
    "unit": "mg/day",
    "confidence_interval_lower": 2.1,
    "confidence_interval_upper": 4.9,
    "confidence_level": 0.9
  },
  "cpic_guidance": {
    "recommendation_text": "Consider 20-40% dose reduction...",
    "monitoring_advice": "Enhanced INR monitoring recommended..."
  },
  "safety_notice": "Decision support only. Not a prescription."
}
```

## 🎨 Catppuccin Theme

The dashboard uses the **Catppuccin Mocha** color palette:

| Element | Color | Hex |
|---------|-------|-----|
| Primary | Mauve | `#cba6f7` |
| Secondary | Blue | `#89b4fa` |
| Success | Green | `#a6e3a1` |
| Warning | Yellow | `#f9e2af` |
| Error | Red | `#f38ba8` |
| Background | Base | `#1e1e2e` |
| Surface | Surface0 | `#313244` |
| Text | Text | `#cdd6f4` |

## ⚕️ Safety & Ethics

- **Never** used as an automatic prescription system
- **All** outputs are labelled as "suggestions for clinician review"
- **No** built-in genotype-to-phenotype rules without explicit CPIC citation
- **Synthetic data only** for demonstration — never train production models on synthetic data
- **Audit trail** via consultation IDs and alert IDs
- **Drug interaction warnings** cross-referenced with CPIC guidelines

## 📚 References

- [CPIC Guidelines](https://cpicpgx.org/guidelines/)
- [PharmGKB](https://www.pharmgkb.org/)
- [FDA Pharmacogenomic Biomarkers](https://www.fda.gov/drugs/science-and-research-drugs/table-pharmacogenomic-biomarkers-drug-labeling)
