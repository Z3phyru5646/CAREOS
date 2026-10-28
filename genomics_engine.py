"""Genomics-informed treatment support for clinician review.

This module supplies a configurable modeling pipeline. It is not a clinical
guideline implementation and must not be used to issue or change prescriptions.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score


RESPONSE_CATEGORIES = frozenset(
    {"responder", "non_responder", "adverse_event_risk"}
)
REQUIRED_CLINICAL_COLUMNS = frozenset(
    {"age", "sex", "weight_kg", "creatinine_clearance"}
)
KNOWN_LIVER_MARKER_COLUMNS = frozenset(
    {"alt", "ast", "bilirubin", "alp", "albumin", "inr"}
)


@dataclass(frozen=True)
class TriggerRule:
    """An explicitly configured diagnosis/medication trigger and annotation."""

    diagnosis_codes: frozenset[str]
    medication_names: frozenset[str]
    gene_name: str
    variant_id: str
    genotype_to_metabolizer: Mapping[int, str]
    dose_unit: str
    cpic_guideline_source: str = ""
    cpic_monitoring_advice: str = ""


@dataclass
class _FeatureState:
    snp_columns: list[str]
    clinical_columns: list[str]
    expression_columns: list[str]
    tabular_columns: list[str]


def _normalize_medication(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _normalize_diagnosis(value: str) -> str:
    return value.strip().upper()


def _require_frame(name: str, value: pd.DataFrame) -> None:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if value.empty:
        raise ValueError(f"{name} must contain at least one row and one column.")
    if value.columns.has_duplicates:
        raise ValueError(f"{name} must not contain duplicate column names.")
    if value.index.has_duplicates:
        raise ValueError(f"{name} must not contain duplicate sample indexes.")


class GenomicsTreatmentEngine:
    """Fit and run a PCA + tabular feature-fusion treatment model.

    The engine does not contain built-in CPIC recommendations. An alert is
    emitted only for a configured trigger rule; guideline text is included only
    when supplied by the application from a verified, current source.
    """

    def __init__(
        self,
        target_medication: str,
        trigger_rules: Sequence[TriggerRule] = (),
        snp_components: int = 16,
        random_state: int = 42,
    ) -> None:
        if snp_components < 1:
            raise ValueError("snp_components must be at least 1.")
        if not target_medication.strip():
            raise ValueError("target_medication must be provided.")
        normalized_target = _normalize_medication(target_medication)
        self.trigger_rules = tuple(trigger_rules)
        self.target_medication = target_medication.strip()
        self.snp_components = snp_components
        self.random_state = random_state
        self._pca: PCA | None = None
        self._feature_state: _FeatureState | None = None
        self._classifier: RandomForestClassifier | None = None
        self._dose_regressor: GradientBoostingRegressor | None = None

        for rule in self.trigger_rules:
            if not rule.gene_name.strip() or not rule.variant_id.strip():
                raise ValueError("Trigger rules need a gene name and variant ID.")
            if not rule.dose_unit.strip():
                raise ValueError("Trigger rules need an explicit dose unit.")
            if normalized_target not in {
                _normalize_medication(name) for name in rule.medication_names
            }:
                raise ValueError(
                    "Each trigger rule must include the engine's target medication "
                    "in medication_names."
                )

    def fit_features(
        self,
        snp_profiles: pd.DataFrame,
        clinical_data: pd.DataFrame,
        gene_expression: pd.DataFrame,
    ) -> np.ndarray:
        """Fit SNP PCA and return fused features for model training."""
        self._validate_inputs(snp_profiles, clinical_data, gene_expression)
        if len(snp_profiles) < 2 or snp_profiles.shape[1] < 1:
            raise ValueError("PCA training needs at least two samples and one SNP.")

        snps = self._snp_values(snp_profiles)
        component_count = min(
            self.snp_components, snp_profiles.shape[0], snp_profiles.shape[1]
        )
        self._pca = PCA(n_components=component_count, random_state=self.random_state)
        latent = self._pca.fit_transform(snps)

        tabular = self._make_tabular_features(clinical_data, gene_expression)
        self._feature_state = _FeatureState(
            snp_columns=list(snp_profiles.columns),
            clinical_columns=list(clinical_data.columns),
            expression_columns=list(gene_expression.columns),
            tabular_columns=list(tabular.columns),
        )
        return np.concatenate((latent, tabular.to_numpy(dtype=float)), axis=1)

    def transform_features(
        self,
        snp_profiles: pd.DataFrame,
        clinical_data: pd.DataFrame,
        gene_expression: pd.DataFrame,
    ) -> np.ndarray:
        """Transform aligned samples using the training feature schema."""
        if self._pca is None or self._feature_state is None:
            raise RuntimeError("Call fit_features before transform_features.")
        self._validate_inputs(snp_profiles, clinical_data, gene_expression)

        expected_snps = self._feature_state.snp_columns
        if set(snp_profiles.columns) != set(expected_snps):
            raise ValueError("SNP columns must match the fitted SNP schema.")
        snps = self._snp_values(snp_profiles[expected_snps])

        for name, frame, expected in (
            ("clinical_data", clinical_data, self._feature_state.clinical_columns),
            (
                "gene_expression",
                gene_expression,
                self._feature_state.expression_columns,
            ),
        ):
            missing = set(expected) - set(frame.columns)
            if missing:
                raise ValueError(f"{name} is missing fitted columns: {sorted(missing)}")

        tabular = self._make_tabular_features(
            clinical_data[self._feature_state.clinical_columns],
            gene_expression[self._feature_state.expression_columns],
        ).reindex(columns=self._feature_state.tabular_columns, fill_value=0.0)
        latent = self._pca.transform(snps)
        return np.concatenate((latent, tabular.to_numpy(dtype=float)), axis=1)

    def fit_classifier(
        self, features: np.ndarray, response_labels: Sequence[str]
    ) -> GenomicsTreatmentEngine:
        labels = np.asarray(response_labels, dtype=str)
        self._validate_training_targets(features, labels, "response_labels")
        unknown = set(labels) - RESPONSE_CATEGORIES
        if unknown:
            raise ValueError(
                "response_labels must use responder, non_responder, or "
                f"adverse_event_risk; got {sorted(unknown)}."
            )
        if len(set(labels)) < 2:
            raise ValueError("Classification training requires at least two classes.")
        self._classifier = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._classifier.fit(features, labels)
        return self

    def fit_dose_regressor(
        self, features: np.ndarray, doses: Sequence[float]
    ) -> GenomicsTreatmentEngine:
        targets = np.asarray(doses, dtype=float)
        self._validate_training_targets(features, targets, "doses")
        if not np.isfinite(targets).all() or (targets <= 0).any():
            raise ValueError("Training doses must be finite positive values.")
        self._dose_regressor = GradientBoostingRegressor(random_state=self.random_state)
        self._dose_regressor.fit(features, targets)
        return self

    def evaluate_classifier(
        self, features: np.ndarray, response_labels: Sequence[str]
    ) -> dict[str, float]:
        """Return ROC AUC (binary or macro one-vs-rest multiclass)."""
        if self._classifier is None:
            raise RuntimeError("Call fit_classifier before evaluation.")
        labels = np.asarray(response_labels, dtype=str)
        self._validate_training_targets(features, labels, "response_labels")
        classes = self._classifier.classes_
        probabilities = self._classifier.predict_proba(features)
        if len(classes) == 2:
            if set(labels) != set(classes):
                raise ValueError("Binary AUC evaluation data must contain both classes.")
            auc = roc_auc_score(labels, probabilities[:, 1], labels=classes)
        else:
            if set(labels) != set(classes):
                raise ValueError(
                    "Multiclass AUC evaluation data must contain every fitted class."
                )
            auc = roc_auc_score(
                labels,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
        return {"auc_roc": float(auc)}

    def evaluate_dose_regressor(
        self, features: np.ndarray, doses: Sequence[float]
    ) -> dict[str, float]:
        """Return mean squared error and R-squared for held-out samples."""
        if self._dose_regressor is None:
            raise RuntimeError("Call fit_dose_regressor before evaluation.")
        targets = np.asarray(doses, dtype=float)
        self._validate_training_targets(features, targets, "doses")
        if len(targets) < 2:
            raise ValueError("R-squared evaluation needs at least two samples.")
        predictions = self._dose_regressor.predict(features)
        return {
            "mse": float(mean_squared_error(targets, predictions)),
            "r_squared": float(r2_score(targets, predictions)),
        }

    def create_alert(
        self,
        diagnosis_codes: Sequence[str],
        medication_names: Sequence[str],
        snp_profile: pd.DataFrame,
        clinical_data: pd.DataFrame,
        gene_expression: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """Return an alert for one registered trigger, or None if not triggered."""
        rule = self._find_trigger(diagnosis_codes, medication_names)
        if rule is None:
            return None
        if len(snp_profile) != 1 or len(clinical_data) != 1 or len(gene_expression) != 1:
            raise ValueError("Alert generation requires exactly one patient row.")
        if self._pca is None or self._feature_state is None:
            raise RuntimeError("Call fit_features before creating an alert.")
        if self._classifier is None:
            raise RuntimeError("Call fit_classifier before creating an alert.")
        if rule.variant_id not in snp_profile.columns:
            raise ValueError(
                f"Configured variant {rule.variant_id!r} is absent from the SNP profile."
            )

        genotype_value = self._snp_values(snp_profile[[rule.variant_id]])[0, 0]
        genotype = int(genotype_value)
        metabolizer = rule.genotype_to_metabolizer.get(genotype, "unknown")
        features = self.transform_features(snp_profile, clinical_data, gene_expression)
        probabilities = self._classifier.predict_proba(features)[0]
        predicted_response = str(self._classifier.classes_[int(np.argmax(probabilities))])

        suggested_dose: dict[str, Any] | None = None
        warnings: list[str] = []
        if self._dose_regressor is not None:
            dose = float(self._dose_regressor.predict(features)[0])
            if np.isfinite(dose) and dose > 0:
                suggested_dose = {
                    "value": round(dose, 2),
                    "unit": rule.dose_unit,
                    "status": "model_estimate_requires_clinician_review",
                }
            else:
                warnings.append(
                    "Dose model returned a non-positive or invalid estimate; "
                    "no dose suggestion is provided."
                )
        else:
            warnings.append("No fitted dose model is available; no dose is suggested.")

        advice = rule.cpic_monitoring_advice.strip() or (
            "No CPIC monitoring advice is configured. Verify the current CPIC "
            "guideline and patient-specific factors before clinical use."
        )
        source = rule.cpic_guideline_source.strip() or None
        probabilities_by_class = {
            str(label): float(probability)
            for label, probability in zip(self._classifier.classes_, probabilities)
        }
        soap_suggestion = (
            f"Pharmacogenomic decision-support suggestion for clinician review: "
            f"{rule.gene_name} {rule.variant_id} genotype {genotype} "
            f"(metabolizer type: {metabolizer}); model-predicted response: "
            f"{predicted_response}. "
            + (
                f"Model-estimated starting dose: {suggested_dose['value']} "
                f"{suggested_dose['unit']}. "
                if suggested_dose
                else "No starting dose estimate is available. "
            )
            + f"Monitoring: {advice} This is not an automatic prescription; "
            "the treating clinician must verify the evidence and decide care."
        )

        return {
            "alert_type": "pharmacogenomic_alert",
            "trigger": {
                "matched_diagnosis_codes": sorted(
                    code
                    for code in diagnosis_codes
                    if self._rule_matches_diagnosis(rule, code)
                ),
                "matched_medication_names": sorted(
                    name
                    for name in medication_names
                    if self._rule_matches_medication(rule, name)
                ),
            },
            "model_medication": self.target_medication,
            "detected_gene_variant": {
                "gene": rule.gene_name,
                "variant_id": rule.variant_id,
                "copies_of_variant": genotype,
            },
            "metabolizer_type": metabolizer,
            "predicted_response": {
                "category": predicted_response,
                "probabilities": probabilities_by_class,
            },
            "suggested_dose": suggested_dose,
            "cpic_guideline_monitoring_advice": {
                "advice": advice,
                "source": source,
            },
            "soap_note_suggestion": soap_suggestion,
            "warnings": warnings,
            "safety_notice": (
                "Decision support only. Not a diagnosis or prescription. "
                "Requires clinician review and independent verification."
            ),
        }

    @staticmethod
    def alert_to_json(alert: Mapping[str, Any]) -> str:
        """Serialize an alert as standards-compliant JSON."""
        return json.dumps(alert, indent=2, allow_nan=False)

    def _find_trigger(
        self, diagnosis_codes: Sequence[str], medication_names: Sequence[str]
    ) -> TriggerRule | None:
        matches = [
            rule
            for rule in self.trigger_rules
            if any(self._rule_matches_diagnosis(rule, code) for code in diagnosis_codes)
            or any(
                self._rule_matches_medication(rule, name)
                for name in medication_names
            )
        ]
        if len(matches) > 1:
            raise ValueError(
                "Input matches multiple trigger rules; resolve the trigger "
                "configuration before generating an alert."
            )
        return matches[0] if matches else None

    @staticmethod
    def _rule_matches_diagnosis(rule: TriggerRule, code: str) -> bool:
        normalized = {_normalize_diagnosis(item) for item in rule.diagnosis_codes}
        return _normalize_diagnosis(code) in normalized

    @staticmethod
    def _rule_matches_medication(rule: TriggerRule, name: str) -> bool:
        normalized = {_normalize_medication(item) for item in rule.medication_names}
        return _normalize_medication(name) in normalized

    @staticmethod
    def _validate_inputs(
        snp_profiles: pd.DataFrame,
        clinical_data: pd.DataFrame,
        gene_expression: pd.DataFrame,
    ) -> None:
        _require_frame("snp_profiles", snp_profiles)
        _require_frame("clinical_data", clinical_data)
        _require_frame("gene_expression", gene_expression)
        if not (
            snp_profiles.index.equals(clinical_data.index)
            and snp_profiles.index.equals(gene_expression.index)
        ):
            raise ValueError(
                "SNP, clinical, and expression rows must have identical aligned indexes."
            )
        missing = REQUIRED_CLINICAL_COLUMNS - set(clinical_data.columns)
        if missing:
            raise ValueError(f"Clinical data is missing required columns: {sorted(missing)}")
        if not set(clinical_data.columns) & KNOWN_LIVER_MARKER_COLUMNS:
            raise ValueError(
                "Clinical data must include at least one recognized liver function "
                f"marker: {sorted(KNOWN_LIVER_MARKER_COLUMNS)}."
            )

    @staticmethod
    def _snp_values(snp_profiles: pd.DataFrame) -> np.ndarray:
        try:
            values = snp_profiles.to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("SNP profiles must contain numeric 0, 1, or 2 values.") from error
        if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0, 2.0)).all():
            raise ValueError("SNP profiles must contain only finite 0, 1, or 2 values.")
        return values

    @staticmethod
    def _make_tabular_features(
        clinical_data: pd.DataFrame, gene_expression: pd.DataFrame
    ) -> pd.DataFrame:
        clinical = clinical_data.add_prefix("clinical__")
        expression = gene_expression.add_prefix("expression__")
        features = pd.concat((clinical, expression), axis=1)
        if features.isna().any().any():
            raise ValueError("Clinical and expression features must not contain missing values.")
        categorical = [
            column
            for column, dtype in features.dtypes.items()
            if pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(dtype)
            or isinstance(dtype, pd.StringDtype)
        ]
        features = pd.get_dummies(features, columns=categorical, dtype=float)
        try:
            numeric = features.to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Clinical and expression features must be numeric or categorical."
            ) from error
        if not np.isfinite(numeric).all():
            raise ValueError("Clinical and expression features must be finite.")
        return features

    @staticmethod
    def _validate_training_targets(
        features: np.ndarray, targets: np.ndarray, target_name: str
    ) -> None:
        matrix = np.asarray(features)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("features must be a non-empty two-dimensional matrix.")
        if len(targets) != matrix.shape[0]:
            raise ValueError(
                f"{target_name} count must match the feature row count."
            )
        if not np.isfinite(matrix.astype(float)).all():
            raise ValueError("features must contain only finite values.")
        if pd.isna(targets).any():
            raise ValueError(f"{target_name} must not contain missing values.")
