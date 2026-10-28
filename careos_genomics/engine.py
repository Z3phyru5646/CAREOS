"""
CARE'OS Module 3 — Genomics-Informed Treatment Engine (Core Pipeline).

This is the central orchestrator that ties together:
  • SNP compression via Variational Autoencoder
  • Multi-modal data fusion
  • XGBoost drug response classifier with Bayesian optimization
  • Gradient Boosting dose regressor with conformal prediction
  • SHAP explainability
  • CPIC knowledge base integration
  • Pharmacogenomic alert generation

The engine activates when triggered by Module 2's diagnosis/medication keys
and produces structured alerts for SOAP note integration.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
import warnings
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger("careos.genomics")


# Try importing optional heavy dependencies gracefully
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import optuna
    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    HAS_OPTUNA = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


from careos_genomics.data_models import (
    AlertSeverity,
    CPICGuidanceDetail,
    DoseEstimate,
    DrugInteractionWarning,
    DrugResponseCategory,
    ExplainabilityReport,
    FeatureImportance,
    GeneVariantDetail,
    MetabolizerPhenotype,
    PharmacogenomicAlert,
    PredictionResult,
    RiskLevel,
    TriggerPayload,
    ClinicalFeatures,
    PatientGenomicProfile,
)
from careos_genomics.knowledge_base import CPICKnowledgeBase, PharmacogenomicRule


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

class EngineConfig:
    """Configuration for the GenomicsEngine."""

    def __init__(
        self,
        # VAE settings
        vae_latent_dim: int = 32,
        vae_hidden_dims: Optional[list[int]] = None,
        vae_epochs: int = 100,
        vae_batch_size: int = 64,
        vae_learning_rate: float = 1e-3,
        # Classifier settings
        use_xgboost: bool = True,
        xgb_n_trials: int = 50,  # Optuna trials for hyperparameter search
        xgb_n_estimators: int = 500,
        # Regressor settings
        gbr_n_estimators: int = 300,
        gbr_learning_rate: float = 0.05,
        gbr_max_depth: int = 5,
        # Conformal prediction
        conformal_alpha: float = 0.1,  # 90% prediction intervals
        # General
        random_state: int = 42,
        n_jobs: int = -1,
        enable_shap: bool = True,
        enable_vae: bool = True,
        pca_components: int = 32,  # Fallback if VAE not available
    ) -> None:
        self.vae_latent_dim = vae_latent_dim
        self.vae_hidden_dims = vae_hidden_dims
        self.vae_epochs = vae_epochs
        self.vae_batch_size = vae_batch_size
        self.vae_learning_rate = vae_learning_rate
        self.use_xgboost = use_xgboost and HAS_XGBOOST
        self.xgb_n_trials = xgb_n_trials
        self.xgb_n_estimators = xgb_n_estimators
        self.gbr_n_estimators = gbr_n_estimators
        self.gbr_learning_rate = gbr_learning_rate
        self.gbr_max_depth = gbr_max_depth
        self.conformal_alpha = conformal_alpha
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.enable_shap = enable_shap and HAS_SHAP
        self.enable_vae = enable_vae and HAS_TORCH
        self.pca_components = pca_components


# ═══════════════════════════════════════════════════════════════════════════
# Main Engine
# ═══════════════════════════════════════════════════════════════════════════

class GenomicsEngine:
    """
    Production-grade Genomics-Informed Treatment Engine.

    Pipeline:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Module 2 Trigger (ICD-10 codes + medication names)                │
    └──────────────────────────┬──────────────────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────────────────┐
    │  1. Knowledge Base Lookup → matched pharmacogenomic rules          │
    └──────────────────────────┬──────────────────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────────────────┐
    │  2. SNP Compression (VAE / PCA) → latent genomic features          │
    └──────────────────────────┬──────────────────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────────────────┐
    │  3. Data Fusion → concatenate [latent, clinical, expression]       │
    └──────────────────────────┬──────────────────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────────────────┐
    │  4. Classification (XGBoost) → response prediction + probabilities │
    │  5. Regression (GBR) → dose estimate + conformal interval          │
    │  6. Explainability (SHAP) → feature importance report              │
    └──────────────────────────┬──────────────────────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────────────────┐
    │  7. Alert Generation → PharmacogenomicAlert → SOAP appendix        │
    └────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        self.knowledge_base = CPICKnowledgeBase()

        # ML components (initialized during training)
        self._vae_trainer = None
        self._pca = None
        self._classifier = None
        self._calibrated_classifier = None
        self._label_encoder = None
        self._dose_regressor = None
        self._shap_explainer = None

        # Feature schema tracking
        self._snp_columns: Optional[list[str]] = None
        self._clinical_columns: Optional[list[str]] = None
        self._expression_columns: Optional[list[str]] = None
        self._feature_names: Optional[list[str]] = None
        self._conformal_residuals: Optional[np.ndarray] = None

        # Training metrics
        self.training_metrics: dict[str, Any] = {}
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Return True if the engine has been trained."""
        return self._is_trained

        logger.info(
            "GenomicsEngine initialized | XGBoost=%s | VAE=%s | SHAP=%s | Optuna=%s",
            self.config.use_xgboost, self.config.enable_vae,
            self.config.enable_shap, HAS_OPTUNA,
        )

    # ═══════════════════════════════════════════════════════════════════
    # Training Pipeline
    # ═══════════════════════════════════════════════════════════════════

    def train(
        self,
        snp_profiles: pd.DataFrame,
        clinical_data: pd.DataFrame,
        gene_expression: pd.DataFrame,
        response_labels: np.ndarray,
        optimal_doses: np.ndarray,
        validation_split: float = 0.2,
    ) -> dict[str, Any]:
        """
        Full training pipeline: VAE → feature fusion → classifier → regressor.

        Args:
            snp_profiles: (n_samples, n_snps) with values 0, 1, 2
            clinical_data: (n_samples, n_clinical) tabular features
            gene_expression: (n_samples, n_genes) expression data
            response_labels: (n_samples,) drug response categories
            optimal_doses: (n_samples,) target doses
            validation_split: fraction to hold out

        Returns:
            Dictionary of training metrics and evaluation results
        """
        start_time = time.time()
        n_samples = len(snp_profiles)
        logger.info("Starting training pipeline with %d samples", n_samples)

        # ── Validate inputs ──────────────────────────────────────────
        self._validate_training_inputs(
            snp_profiles, clinical_data, gene_expression,
            response_labels, optimal_doses,
        )

        # Store feature schemas
        self._snp_columns = list(snp_profiles.columns)
        self._clinical_columns = list(clinical_data.columns)
        self._expression_columns = list(gene_expression.columns)

        # ── Train/Validation split ───────────────────────────────────
        rng = np.random.default_rng(self.config.random_state)
        n_val = max(2, int(n_samples * validation_split))
        indices = rng.permutation(n_samples)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        # ── Step 1: SNP Compression ──────────────────────────────────
        logger.info("Step 1: SNP dimensionality reduction")
        snp_matrix = snp_profiles.values.astype(float)

        if self.config.enable_vae:
            latent_train, latent_val = self._train_vae(
                snp_matrix, train_idx, val_idx
            )
            compression_method = "variational_autoencoder"
        else:
            latent_train, latent_val = self._train_pca(
                snp_matrix, train_idx, val_idx
            )
            compression_method = "pca"

        # ── Step 2: Feature Fusion ───────────────────────────────────
        logger.info("Step 2: Multi-modal data fusion")
        clinical_matrix = self._prepare_clinical_features(clinical_data)
        expression_matrix = gene_expression.values.astype(float)

        train_features = np.concatenate([
            latent_train,
            clinical_matrix[train_idx],
            expression_matrix[train_idx],
        ], axis=1)

        val_features = np.concatenate([
            latent_val,
            clinical_matrix[val_idx],
            expression_matrix[val_idx],
        ], axis=1)

        # Feature names for SHAP
        n_latent = latent_train.shape[1]
        latent_names = [f"latent_snp_{i}" for i in range(n_latent)]
        clinical_names = [f"clin_{c}" for c in self._get_clinical_feature_names(clinical_data)]
        expression_names = [f"expr_{c}" for c in gene_expression.columns]
        self._feature_names = latent_names + clinical_names + expression_names

        # ── Step 3: Train Classifier ─────────────────────────────────
        logger.info("Step 3: Training drug response classifier")
        train_labels = response_labels[train_idx]
        val_labels = response_labels[val_idx]

        classifier_metrics = self._train_classifier(
            train_features, train_labels, val_features, val_labels
        )

        # ── Step 4: Train Dose Regressor ─────────────────────────────
        logger.info("Step 4: Training dose regressor")
        train_doses = optimal_doses[train_idx]
        val_doses = optimal_doses[val_idx]

        regressor_metrics = self._train_dose_regressor(
            train_features, train_doses, val_features, val_doses
        )

        # ── Step 5: SHAP Explainability ──────────────────────────────
        if self.config.enable_shap:
            logger.info("Step 5: Building SHAP explainer")
            self._build_shap_explainer(train_features)

        # ── Compile Metrics ──────────────────────────────────────────
        elapsed = time.time() - start_time
        self.training_metrics = {
            "training_time_seconds": round(elapsed, 2),
            "n_training_samples": len(train_idx),
            "n_validation_samples": len(val_idx),
            "snp_compression_method": compression_method,
            "snp_input_dim": snp_matrix.shape[1],
            "snp_latent_dim": n_latent,
            "total_fused_features": train_features.shape[1],
            "classifier": classifier_metrics,
            "regressor": regressor_metrics,
        }

        if self.config.enable_vae and self._vae_trainer is not None:
            self.training_metrics["vae_reconstruction_accuracy"] = round(
                self._vae_trainer.reconstruction_accuracy(snp_matrix[val_idx]), 4
            )

        self._is_trained = True
        logger.info(
            "Training complete in %.1fs | AUC=%.4f | MSE=%.4f | R²=%.4f",
            elapsed,
            classifier_metrics.get("val_auc_roc", 0),
            regressor_metrics.get("val_mse", 0),
            regressor_metrics.get("val_r_squared", 0),
        )

        return self.training_metrics

    # ─── SNP Compression Methods ────────────────────────────────────

    def _train_vae(
        self, snp_matrix: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train Variational Autoencoder for SNP compression."""
        from careos_genomics.autoencoder import SNPAutoencoderTrainer

        input_dim = snp_matrix.shape[1]
        self._vae_trainer = SNPAutoencoderTrainer(
            input_dim=input_dim,
            latent_dim=self.config.vae_latent_dim,
            hidden_dims=self.config.vae_hidden_dims,
            n_epochs=self.config.vae_epochs,
            batch_size=self.config.vae_batch_size,
            learning_rate=self.config.vae_learning_rate,
            device="cpu",  # Use CPU for reproducibility in production
        )

        logger.info("Training VAE: input_dim=%d, latent_dim=%d", input_dim, self.config.vae_latent_dim)
        self._vae_trainer.fit(snp_matrix[train_idx])

        latent_train = self._vae_trainer.transform(snp_matrix[train_idx])
        latent_val = self._vae_trainer.transform(snp_matrix[val_idx])
        return latent_train, latent_val

    def _train_pca(
        self, snp_matrix: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """PCA fallback for SNP compression."""
        from sklearn.decomposition import PCA

        n_components = min(
            self.config.pca_components,
            snp_matrix.shape[1],
            len(train_idx),
        )
        self._pca = PCA(n_components=n_components, random_state=self.config.random_state)
        latent_train = self._pca.fit_transform(snp_matrix[train_idx])
        latent_val = self._pca.transform(snp_matrix[val_idx])

        explained = sum(self._pca.explained_variance_ratio_)
        logger.info("PCA: %d components explain %.1f%% variance", n_components, explained * 100)
        return latent_train, latent_val

    # ─── Classifier Training ────────────────────────────────────────

    def _train_classifier(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
    ) -> dict[str, Any]:
        """Train the drug response classifier (XGBoost with optional Optuna)."""
        self._label_encoder = LabelEncoder()
        train_encoded = self._label_encoder.fit_transform(train_labels)
        val_encoded = self._label_encoder.transform(val_labels)
        n_classes = len(self._label_encoder.classes_)

        if self.config.use_xgboost:
            if HAS_OPTUNA and self.config.xgb_n_trials > 0:
                best_params = self._optimize_xgb_hyperparams(
                    train_features, train_encoded, n_classes
                )
            else:
                best_params = {
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "min_child_weight": 3,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "gamma": 0.1,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                }

            self._classifier = xgb.XGBClassifier(
                n_estimators=self.config.xgb_n_estimators,
                objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                num_class=n_classes if n_classes > 2 else None,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                early_stopping_rounds=30,
                **best_params,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._classifier.fit(
                    train_features, train_encoded,
                    eval_set=[(val_features, val_encoded)],
                    verbose=False,
                )
        else:
            from sklearn.ensemble import RandomForestClassifier
            self._classifier = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                class_weight="balanced_subsample",
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )
            self._classifier.fit(train_features, train_encoded)

        # Calibrate probabilities
        try:
            self._calibrated_classifier = CalibratedClassifierCV(
                self._classifier, cv=3, method="isotonic"
            )
            self._calibrated_classifier.fit(train_features, train_encoded)
        except Exception:
            self._calibrated_classifier = None
            logger.warning("Probability calibration failed; using raw probabilities")

        # Evaluate
        val_probs = self._get_probabilities(val_features)
        if n_classes == 2:
            auc = roc_auc_score(val_encoded, val_probs[:, 1])
        else:
            auc = roc_auc_score(
                val_encoded, val_probs, multi_class="ovr", average="macro"
            )

        return {
            "val_auc_roc": round(float(auc), 4),
            "n_classes": n_classes,
            "classes": list(self._label_encoder.classes_),
            "model_type": "XGBoost" if self.config.use_xgboost else "RandomForest",
            "calibrated": self._calibrated_classifier is not None,
        }

    def _optimize_xgb_hyperparams(
        self, features: np.ndarray, labels: np.ndarray, n_classes: int
    ) -> dict[str, Any]:
        """Bayesian hyperparameter optimization using Optuna."""
        logger.info("Running Optuna hyperparameter optimization (%d trials)", self.config.xgb_n_trials)

        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_state)
            scores = []

            for train_i, val_i in cv.split(features, labels):
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                    num_class=n_classes if n_classes > 2 else None,
                    random_state=self.config.random_state,
                    n_jobs=1,
                    **params,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(features[train_i], labels[train_i])

                probs = model.predict_proba(features[val_i])
                if n_classes == 2:
                    auc = roc_auc_score(labels[val_i], probs[:, 1])
                else:
                    auc = roc_auc_score(
                        labels[val_i], probs, multi_class="ovr", average="macro"
                    )
                scores.append(auc)

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.xgb_n_trials, show_progress_bar=False)

        logger.info("Best Optuna AUC: %.4f", study.best_value)
        return study.best_params

    def _get_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Get calibrated probabilities if available."""
        if self._calibrated_classifier is not None:
            return self._calibrated_classifier.predict_proba(features)
        return self._classifier.predict_proba(features)

    # ─── Dose Regressor Training ────────────────────────────────────

    def _train_dose_regressor(
        self,
        train_features: np.ndarray,
        train_doses: np.ndarray,
        val_features: np.ndarray,
        val_doses: np.ndarray,
    ) -> dict[str, float]:
        """Train Gradient Boosting dose regressor with conformal prediction."""
        self._dose_regressor = GradientBoostingRegressor(
            n_estimators=self.config.gbr_n_estimators,
            learning_rate=self.config.gbr_learning_rate,
            max_depth=self.config.gbr_max_depth,
            subsample=0.8,
            random_state=self.config.random_state,
        )
        self._dose_regressor.fit(train_features, train_doses)

        # Conformal prediction: compute calibration residuals
        cal_predictions = self._dose_regressor.predict(val_features)
        self._conformal_residuals = np.abs(val_doses - cal_predictions)
        self._conformal_residuals.sort()

        val_predictions = self._dose_regressor.predict(val_features)
        mse = mean_squared_error(val_doses, val_predictions)
        r2 = r2_score(val_doses, val_predictions)

        return {
            "val_mse": round(float(mse), 4),
            "val_r_squared": round(float(r2), 4),
            "val_rmse": round(float(np.sqrt(mse)), 4),
            "conformal_alpha": self.config.conformal_alpha,
        }

    def _get_conformal_interval(self, point_estimate: float) -> tuple[float, float]:
        """Compute conformal prediction interval."""
        if self._conformal_residuals is None or len(self._conformal_residuals) == 0:
            return (point_estimate * 0.7, point_estimate * 1.3)

        n = len(self._conformal_residuals)
        q_level = np.ceil((n + 1) * (1 - self.config.conformal_alpha)) / n
        q_level = min(q_level, 1.0)
        idx = int(q_level * n) - 1
        idx = max(0, min(idx, n - 1))
        width = self._conformal_residuals[idx]

        return (max(0, point_estimate - width), point_estimate + width)

    # ─── SHAP Explainability ────────────────────────────────────────

    def _build_shap_explainer(self, train_features: np.ndarray) -> None:
        """Build a SHAP TreeExplainer for the classifier with fallback."""
        try:
            n_background = min(50, len(train_features))
            background = train_features[:n_background]
            self._shap_explainer = shap.TreeExplainer(
                self._classifier,
                data=background,
            )
            logger.info("SHAP explainer built successfully")
        except Exception as e:
            try:
                # Fallback to general Explainer
                n_background = min(30, len(train_features))
                background = train_features[:n_background]
                self._shap_explainer = shap.Explainer(
                    self._classifier.predict_proba,
                    background,
                )
                logger.info("SHAP Explainer (general) built successfully")
            except Exception as e2:
                logger.warning("SHAP explainer creation failed: %s; using tree importance fallback", e2)
                self._shap_explainer = None

    def _explain_prediction(
        self, features: np.ndarray, predicted_class_idx: int
    ) -> Optional[ExplainabilityReport]:
        """Generate explanation using SHAP or tree feature importance."""
        if self._feature_names is None:
            return None

        # 1. Try SHAP first
        if self._shap_explainer is not None:
            try:
                shap_values = self._shap_explainer.shap_values(features)

                if isinstance(shap_values, list):
                    values = shap_values[predicted_class_idx][0]
                    base = float(self._shap_explainer.expected_value[predicted_class_idx])
                elif hasattr(shap_values, "values"):
                    # SHAP Explanation object
                    vals = shap_values.values
                    if vals.ndim == 3:
                        values = vals[0, :, predicted_class_idx]
                    else:
                        values = vals[0]
                    base = float(shap_values.base_values[0, predicted_class_idx] if vals.ndim == 3 else shap_values.base_values[0])
                else:
                    values = shap_values[0]
                    base = float(self._shap_explainer.expected_value)

                abs_values = np.abs(values)
                top_indices = np.argsort(abs_values)[::-1][:10]

                top_features = []
                for idx in top_indices:
                    sv = float(values[idx])
                    abs_sv = abs(sv)
                    if abs_sv < 1e-6:
                        continue
                    top_features.append(FeatureImportance(
                        feature_name=self._feature_names[idx] if idx < len(self._feature_names) else f"feature_{idx}",
                        shap_value=round(sv, 4),
                        direction="increases_risk" if sv > 0 else "decreases_risk",
                        magnitude="high" if abs_sv > np.percentile(abs_values, 90) else (
                            "moderate" if abs_sv > np.percentile(abs_values, 50) else "low"
                        ),
                    ))

                prediction_value = base + float(np.sum(values))
                explanation_parts = [
                    f"{f.feature_name} ({f.direction}, {f.magnitude} impact)"
                    for f in top_features[:5]
                ]
                return ExplainabilityReport(
                    top_features=top_features,
                    base_value=round(base, 4),
                    prediction_value=round(prediction_value, 4),
                    explanation_text=(
                        "Key factors influencing this prediction (SHAP): "
                        + "; ".join(explanation_parts)
                        + ". This explanation is model-derived and should be interpreted in clinical context."
                    ),
                )
            except Exception as e:
                logger.warning("SHAP explanation failed, falling back to tree importances: %s", e)

        # 2. Tree feature importance fallback
        if self._classifier is not None and hasattr(self._classifier, "feature_importances_"):
            try:
                importances = self._classifier.feature_importances_
                top_indices = np.argsort(importances)[::-1][:10]

                top_features = []
                for idx in top_indices:
                    imp = float(importances[idx])
                    if imp < 1e-5:
                        continue
                    feat_val = float(features[0, idx]) if features.ndim == 2 else float(features[idx])
                    direction = "increases_risk" if feat_val > 0.5 else "decreases_risk"
                    top_features.append(FeatureImportance(
                        feature_name=self._feature_names[idx] if idx < len(self._feature_names) else f"feature_{idx}",
                        shap_value=round(imp, 4),
                        direction=direction,
                        magnitude="high" if imp > np.percentile(importances, 90) else (
                            "moderate" if imp > np.percentile(importances, 50) else "low"
                        ),
                    ))

                explanation_parts = [
                    f"{f.feature_name} (weight: {f.shap_value:.3f}, {f.magnitude})"
                    for f in top_features[:5]
                ]
                return ExplainabilityReport(
                    top_features=top_features,
                    base_value=0.5,
                    prediction_value=round(float(np.max(self._classifier.predict_proba(features)[0])), 4),
                    explanation_text=(
                        "Key factors influencing this prediction (Tree Attribution): "
                        + "; ".join(explanation_parts)
                        + ". This explanation is model-derived and should be interpreted in clinical context."
                    ),
                )
            except Exception as e:
                logger.warning("Feature importance fallback failed: %s", e)

        return None

    # ─── Clinical Feature Processing ────────────────────────────────

    def _prepare_clinical_features(self, clinical_data: pd.DataFrame) -> np.ndarray:
        """Convert clinical DataFrame to numeric matrix."""
        df = clinical_data.copy()

        # Encode categorical columns
        for col in df.select_dtypes(include=["object", "category", "bool"]).columns:
            if col == "sex":
                df[col] = df[col].map({"M": 0, "F": 1, "male": 0, "female": 1, "other": 0.5}).fillna(0.5)
            elif df[col].dtype == bool or col.startswith("is_"):
                df[col] = df[col].astype(float)
            else:
                df[col] = pd.Categorical(df[col]).codes.astype(float)

        return df.values.astype(float)

    def _get_clinical_feature_names(self, clinical_data: pd.DataFrame) -> list[str]:
        """Get feature names after encoding."""
        return list(clinical_data.columns)

    # ─── Alert Generation ────────────────────────────────────────────

    def process_trigger(
        self,
        trigger: TriggerPayload,
        genomic_profile: PatientGenomicProfile,
        clinical: ClinicalFeatures,
    ) -> list[PharmacogenomicAlert]:
        """
        Main entry point: process a trigger from Module 2.

        Args:
            trigger: Diagnosis codes + medication names from Module 2
            genomic_profile: Patient's SNP profile + optional expression data
            clinical: Patient's clinical features

        Returns:
            List of pharmacogenomic alerts (may be empty if no rules match)
        """
        if not self._is_trained:
            raise RuntimeError("Engine must be trained before processing triggers. Call train() first.")

        # Find matching rules
        rules = self.knowledge_base.find_matching_rules(
            trigger.diagnosis_codes,
            trigger.medication_names,
        )

        if not rules:
            logger.info("No pharmacogenomic rules matched for trigger %s", trigger.consultation_id)
            return []

        logger.info(
            "Found %d matching rules for patient %s",
            len(rules), trigger.patient_id,
        )

        # Generate alert for each matching rule
        alerts = []
        for rule in rules:
            try:
                alert = self._generate_alert(trigger, genomic_profile, clinical, rule)
                if alert is not None:
                    alerts.append(alert)
            except Exception as e:
                logger.error("Alert generation failed for rule %s: %s", rule.rule_id, e)

        return alerts

    def _generate_alert(
        self,
        trigger: TriggerPayload,
        genomic_profile: PatientGenomicProfile,
        clinical: ClinicalFeatures,
        rule: PharmacogenomicRule,
    ) -> Optional[PharmacogenomicAlert]:
        """Generate a single pharmacogenomic alert for a matched rule."""

        # ── Determine genotype and metabolizer status ────────────────
        variant_details = []
        max_genotype = 0
        for variant_id in rule.variant_ids:
            genotype = genomic_profile.snp_profile.get(variant_id, 0)
            max_genotype = max(max_genotype, genotype)

            detail = GeneVariantDetail(
                gene=rule.gene_name.split("/")[0],
                variant_id=variant_id,
                copies_of_variant=genotype,
            )
            # Add star allele info if available
            if genomic_profile.star_alleles:
                gene_key = rule.gene_name.split("/")[0]
                if gene_key in genomic_profile.star_alleles:
                    detail.star_allele_diplotype = "/".join(
                        genomic_profile.star_alleles[gene_key]
                    )
            variant_details.append(detail)

        metabolizer = rule.genotype_to_metabolizer.get(
            max_genotype, MetabolizerPhenotype.INDETERMINATE
        )

        risk_level = rule.metabolizer_to_risk.get(metabolizer, RiskLevel.LOW)
        severity = rule.metabolizer_to_severity.get(metabolizer, AlertSeverity.INFORMATIONAL)

        # ── Run ML predictions ───────────────────────────────────────
        features = self._prepare_single_patient_features(genomic_profile, clinical)

        # Classification
        probs = self._get_probabilities(features)
        predicted_idx = int(np.argmax(probs[0]))
        predicted_label = self._label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(probs[0][predicted_idx])

        response_category = DrugResponseCategory(predicted_label)
        prob_dict = {
            self._label_encoder.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(probs[0])
        }

        prediction_result = PredictionResult(
            predicted_category=response_category,
            probabilities=prob_dict,
            confidence=round(confidence, 4),
            calibrated=self._calibrated_classifier is not None,
        )

        # Dose regression
        dose_estimate = None
        if self._dose_regressor is not None:
            raw_dose = float(self._dose_regressor.predict(features)[0])
            if np.isfinite(raw_dose) and raw_dose > 0:
                # Apply CPIC dose adjustment factor
                adjustment = rule.dose_adjustment_factors.get(metabolizer, 1.0)
                adjusted_dose = raw_dose * adjustment
                ci_lower, ci_upper = self._get_conformal_interval(adjusted_dose)

                dose_estimate = DoseEstimate(
                    predicted_dose=round(adjusted_dose, 2),
                    unit=rule.dose_unit,
                    confidence_interval_lower=round(max(0, ci_lower), 2),
                    confidence_interval_upper=round(ci_upper, 2),
                    confidence_level=1 - self.config.conformal_alpha,
                )

        # Explainability
        explainability = None
        if self.config.enable_shap:
            explainability = self._explain_prediction(features, predicted_idx)

        # CPIC guidance
        recommendation = rule.cpic_recommendation.get(
            metabolizer,
            "No specific CPIC recommendation available for this metabolizer phenotype. "
            "Consult the current CPIC guidelines and clinical pharmacology resources."
        )
        monitoring = rule.monitoring_advice.get(
            metabolizer,
            "Standard monitoring per drug prescribing information. Verify current "
            "CPIC guidelines for genotype-specific monitoring recommendations."
        )

        cpic_guidance = CPICGuidanceDetail(
            guideline_id=rule.cpic_guideline_id or None,
            guideline_version=rule.cpic_guideline_version or None,
            evidence_level=rule.cpic_evidence_level or None,
            recommendation_text=recommendation,
            monitoring_advice=monitoring,
            source_url=rule.source_url or None,
        )

        # Drug interaction warnings
        interaction_warnings = []
        if clinical.concurrent_medications and rule.drug_interactions:
            for interaction in rule.drug_interactions:
                for med in clinical.concurrent_medications:
                    if interaction["drug"].lower() in med.lower():
                        interaction_warnings.append(DrugInteractionWarning(
                            interacting_entity=interaction["drug"],
                            interaction_type="drug-drug interaction",
                            severity=AlertSeverity(interaction.get("severity", "moderate")),
                            description=interaction["effect"],
                            evidence_level="CPIC guideline",
                        ))

        # Matched codes/medications
        matched_codes = [
            c for c in trigger.diagnosis_codes
            if c.upper() in {d.upper() for d in rule.diagnosis_codes}
        ]
        matched_meds = [
            m for m in trigger.medication_names
            if m.lower() in {d.lower() for d in rule.drug_names}
        ]

        # SOAP suggestion text
        soap_lines = [
            f"⚕️ PHARMACOGENOMIC ALERT [{severity.value.upper()}]",
            f"Gene: {rule.gene_name}",
            f"Metabolizer: {metabolizer.value}",
            f"Predicted response: {predicted_label} (confidence: {confidence:.1%})",
        ]
        if dose_estimate:
            soap_lines.append(
                f"Suggested dose: {dose_estimate.predicted_dose} {dose_estimate.unit} "
                f"(90% CI: {dose_estimate.confidence_interval_lower}-{dose_estimate.confidence_interval_upper})"
            )
        soap_lines.append(f"CPIC: {recommendation[:200]}")
        soap_lines.append("This is a decision-support suggestion only — NOT an automatic prescription.")
        soap_suggestion = " | ".join(soap_lines)

        # Clinical action items
        action_items = [
            f"Review {rule.gene_name} pharmacogenomic implications for current therapy",
            f"Verify metabolizer phenotype ({metabolizer.value}) with confirmatory testing if not previously confirmed",
            monitoring[:150],
        ]
        if risk_level in (RiskLevel.HIGH, RiskLevel.VERY_HIGH):
            action_items.insert(0, f"⚠️ HIGH PRIORITY: {rule.gene_name} genotype suggests elevated risk — immediate pharmacist consultation recommended")
        if interaction_warnings:
            action_items.append(f"Review {len(interaction_warnings)} potential drug interaction(s)")

        return PharmacogenomicAlert(
            severity=severity,
            patient_id=trigger.patient_id,
            consultation_id=trigger.consultation_id,
            matched_diagnosis_codes=matched_codes,
            matched_medication_names=matched_meds,
            detected_variants=variant_details,
            metabolizer_phenotype=metabolizer,
            risk_level=risk_level,
            drug_response_prediction=prediction_result,
            dose_estimate=dose_estimate,
            cpic_guidance=cpic_guidance,
            drug_interaction_warnings=interaction_warnings,
            explainability=explainability,
            soap_note_suggestion=soap_suggestion,
            clinical_action_items=action_items,
        )

    def _prepare_single_patient_features(
        self,
        genomic_profile: PatientGenomicProfile,
        clinical: ClinicalFeatures,
    ) -> np.ndarray:
        """Prepare a single patient's features for prediction."""
        # SNP profile → latent vector
        snp_vector = np.zeros((1, len(self._snp_columns)))
        for i, col in enumerate(self._snp_columns):
            snp_vector[0, i] = genomic_profile.snp_profile.get(col, 0)

        if self._vae_trainer is not None:
            latent = self._vae_trainer.transform(snp_vector)
        elif self._pca is not None:
            latent = self._pca.transform(snp_vector)
        else:
            raise RuntimeError("No SNP compression model available")

        # Clinical features
        clinical_dict = clinical.model_dump()
        clinical_df = pd.DataFrame([{
            col: clinical_dict.get(col, 0)
            for col in self._clinical_columns
        }])
        clinical_matrix = self._prepare_clinical_features(clinical_df)

        # Expression features
        expression_vector = np.zeros((1, len(self._expression_columns)))
        if genomic_profile.gene_expression:
            for i, col in enumerate(self._expression_columns):
                expression_vector[0, i] = genomic_profile.gene_expression.get(col, 0.0)

        # Fuse
        return np.concatenate([latent, clinical_matrix, expression_vector], axis=1)

    # ─── Validation ─────────────────────────────────────────────────

    @staticmethod
    def _validate_training_inputs(
        snp_profiles: pd.DataFrame,
        clinical_data: pd.DataFrame,
        gene_expression: pd.DataFrame,
        response_labels: np.ndarray,
        optimal_doses: np.ndarray,
    ) -> None:
        """Validate all training inputs."""
        n = len(snp_profiles)
        if n < 10:
            raise ValueError(f"Need at least 10 training samples, got {n}")

        for name, df in [("snp_profiles", snp_profiles), ("clinical_data", clinical_data), ("gene_expression", gene_expression)]:
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame")
            if df.empty:
                raise ValueError(f"{name} must not be empty")
            if len(df) != n:
                raise ValueError(f"{name} must have {n} rows, got {len(df)}")

        if len(response_labels) != n:
            raise ValueError(f"response_labels must have {n} entries")
        if len(optimal_doses) != n:
            raise ValueError(f"optimal_doses must have {n} entries")

        valid_labels = {"responder", "non_responder", "adverse_event_risk"}
        invalid = set(response_labels) - valid_labels
        if invalid:
            raise ValueError(f"Invalid response labels: {invalid}")

        if not np.isfinite(optimal_doses).all() or (optimal_doses <= 0).any():
            raise ValueError("optimal_doses must be finite positive values")

        # Validate SNP values
        snp_values = snp_profiles.values
        if not np.isin(snp_values, [0, 1, 2]).all():
            raise ValueError("SNP profiles must contain only 0, 1, or 2 values")

    # ─── Serialization ──────────────────────────────────────────────

    def get_training_summary(self) -> str:
        """Get a human-readable training summary."""
        if not self._is_trained:
            return "Engine has not been trained yet."

        m = self.training_metrics
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║    CARE'OS Module 3 — Genomics Engine Training Summary      ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║  Training Time:       {m['training_time_seconds']:.1f}s",
            f"║  Training Samples:    {m['n_training_samples']}",
            f"║  Validation Samples:  {m['n_validation_samples']}",
            f"║  SNP Compression:     {m['snp_compression_method']}",
            f"║  SNP Input → Latent:  {m['snp_input_dim']} → {m['snp_latent_dim']}",
            f"║  Total Fused Features: {m['total_fused_features']}",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║  Classifier:          {m['classifier']['model_type']}",
            f"║  Validation AUC-ROC:  {m['classifier']['val_auc_roc']:.4f}",
            f"║  Calibrated:          {m['classifier']['calibrated']}",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║  Dose Regressor MSE:  {m['regressor']['val_mse']:.4f}",
            f"║  Dose Regressor R²:   {m['regressor']['val_r_squared']:.4f}",
            f"║  Dose Regressor RMSE: {m['regressor']['val_rmse']:.4f}",
            f"║  Conformal α:         {m['regressor']['conformal_alpha']}",
            "╚══════════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)
