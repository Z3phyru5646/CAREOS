"""
╔══════════════════════════════════════════════════════════════════════════╗
║  CARE'OS Module 3 — Genomics-Informed Treatment Recommendation Engine  ║
║  Advanced Pharmacogenomic Decision Support for Clinician Review        ║
╚══════════════════════════════════════════════════════════════════════════╝

This module provides a production-grade genomics pipeline featuring:
  • PyTorch Variational Autoencoder with self-attention for SNP compression
  • Multi-modal data fusion with cross-attention mechanisms
  • XGBoost classifier with Bayesian hyperparameter optimization (Optuna)
  • Gradient Boosting regressor with conformal prediction intervals
  • SHAP-based explainability for every prediction
  • Comprehensive CPIC pharmacogenomics knowledge base
  • Catppuccin-themed Streamlit dashboard

CLINICAL USE BOUNDARY: This is decision-support software for clinician
review only. It must NEVER be used as an automatic prescription system.
All outputs are suggestions requiring independent clinical verification.
"""

from careos_genomics.engine import GenomicsEngine
from careos_genomics.knowledge_base import CPICKnowledgeBase, PharmacogenomicRule
from careos_genomics.data_models import (
    TriggerPayload,
    PatientGenomicProfile,
    ClinicalFeatures,
    PharmacogenomicAlert,
    PredictionResult,
    DoseEstimate,
    ExplainabilityReport,
)
from careos_genomics.autoencoder import VariationalSNPAutoencoder, AttentionBlock
from careos_genomics.fusion import MultiModalFusionNetwork
from careos_genomics.synthetic_data import SyntheticPharmacogenomicDataGenerator

__version__ = "3.0.0"
__codename__ = "CARE'OS Helix"

__all__ = [
    "GenomicsEngine",
    "CPICKnowledgeBase",
    "PharmacogenomicRule",
    "TriggerPayload",
    "PatientGenomicProfile",
    "ClinicalFeatures",
    "PharmacogenomicAlert",
    "PredictionResult",
    "DoseEstimate",
    "ExplainabilityReport",
    "VariationalSNPAutoencoder",
    "AttentionBlock",
    "MultiModalFusionNetwork",
    "SyntheticPharmacogenomicDataGenerator",
]
