"""
Pydantic data models for type-safe data flow through the CARE'OS genomics pipeline.
All inter-module communication uses these validated schemas.
"""

from __future__ import annotations

import datetime
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class DrugResponseCategory(str, Enum):
    """Predicted drug response classification."""
    RESPONDER = "responder"
    NON_RESPONDER = "non_responder"
    ADVERSE_EVENT_RISK = "adverse_event_risk"


class MetabolizerPhenotype(str, Enum):
    """CYP enzyme metabolizer phenotype classification."""
    ULTRARAPID = "ultrarapid_metabolizer"
    RAPID = "rapid_metabolizer"
    NORMAL = "normal_metabolizer"
    INTERMEDIATE = "intermediate_metabolizer"
    POOR = "poor_metabolizer"
    INDETERMINATE = "indeterminate"


class AlertSeverity(str, Enum):
    """Clinical alert severity level."""
    INFORMATIONAL = "informational"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLevel(str, Enum):
    """Adverse event risk stratification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# ═══════════════════════════════════════════════════════════════════════════
# Input Models (from Module 2 → Module 3)
# ═══════════════════════════════════════════════════════════════════════════

class TriggerPayload(BaseModel):
    """Trigger keys passed from Module 2 (AI Scribe & Verifier)."""
    patient_id: str = Field(..., description="Unique patient identifier from Module 1")
    diagnosis_codes: list[str] = Field(..., min_length=1, description="ICD-10 codes extracted from SOAP note")
    medication_names: list[str] = Field(..., min_length=1, description="Medication names from SOAP note")
    prescribing_physician: str = Field(default="CAREOS-CLINICIAN", description="Physician ID for audit trail")
    consultation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique consultation session ID")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)

    @field_validator("diagnosis_codes")
    @classmethod
    def normalize_icd10(cls, codes: list[str]) -> list[str]:
        return [c.strip().upper() for c in codes]

    @field_validator("medication_names")
    @classmethod
    def normalize_medications(cls, names: list[str]) -> list[str]:
        return [n.strip().lower() for n in names]


class ClinicalFeatures(BaseModel):
    """Tabular clinical data for a single patient."""
    age: int = Field(..., ge=0, le=150)
    sex: str = Field(..., pattern=r"^(M|F|male|female|other)$")
    weight_kg: float = Field(..., gt=0, le=500)
    height_cm: float = Field(default=170.0, gt=0, le=300)
    creatinine_clearance: float = Field(..., ge=0, description="mL/min (Cockcroft-Gault)")
    egfr: Optional[float] = Field(default=None, ge=0, description="Estimated GFR mL/min/1.73m²")
    # Liver function markers
    alt: Optional[float] = Field(default=None, ge=0, description="Alanine aminotransferase U/L")
    ast: Optional[float] = Field(default=None, ge=0, description="Aspartate aminotransferase U/L")
    bilirubin: Optional[float] = Field(default=None, ge=0, description="Total bilirubin mg/dL")
    albumin: Optional[float] = Field(default=None, ge=0, description="Serum albumin g/dL")
    inr: Optional[float] = Field(default=None, ge=0, description="International normalized ratio")
    alp: Optional[float] = Field(default=None, ge=0, description="Alkaline phosphatase U/L")
    # Additional clinical context
    is_smoker: bool = False
    is_pregnant: bool = False
    concurrent_medications: list[str] = Field(default_factory=list)
    comorbidities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_liver_marker(self) -> "ClinicalFeatures":
        markers = [self.alt, self.ast, self.bilirubin, self.albumin, self.inr, self.alp]
        if all(m is None for m in markers):
            raise ValueError("At least one liver function marker (ALT, AST, bilirubin, albumin, INR, ALP) is required")
        return self


class PatientGenomicProfile(BaseModel):
    """Patient's genomic data: SNP calls + optional gene expression."""
    snp_profile: dict[str, int] = Field(..., description="rsID → allele count (0, 1, 2)")
    gene_expression: Optional[dict[str, float]] = Field(default=None, description="Gene symbol → normalized expression level")
    star_alleles: Optional[dict[str, list[str]]] = Field(default=None, description="Gene → star allele diplotype, e.g. CYP2D6: ['*1', '*4']")
    cnv_data: Optional[dict[str, int]] = Field(default=None, description="Gene → copy number")

    @field_validator("snp_profile")
    @classmethod
    def validate_snp_values(cls, profile: dict[str, int]) -> dict[str, int]:
        for rsid, count in profile.items():
            if count not in (0, 1, 2):
                raise ValueError(f"SNP {rsid} must have allele count 0, 1, or 2, got {count}")
        return profile


# ═══════════════════════════════════════════════════════════════════════════
# Output Models (Module 3 → SOAP Note)
# ═══════════════════════════════════════════════════════════════════════════

class FeatureImportance(BaseModel):
    """Single feature's contribution to the prediction."""
    feature_name: str
    shap_value: float
    direction: str = Field(..., description="'increases_risk' or 'decreases_risk'")
    magnitude: str = Field(..., description="'low', 'moderate', 'high'")


class ExplainabilityReport(BaseModel):
    """SHAP-based explanation of the model's prediction."""
    top_features: list[FeatureImportance]
    base_value: float
    prediction_value: float
    explanation_text: str


class DoseEstimate(BaseModel):
    """Predicted dose with uncertainty quantification."""
    predicted_dose: float = Field(..., ge=0)
    unit: str
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_level: float = Field(default=0.9, ge=0, le=1)
    status: str = "model_estimate_requires_clinician_review"


class DrugInteractionWarning(BaseModel):
    """Detected potential drug-gene or drug-drug interaction."""
    interacting_entity: str
    interaction_type: str
    severity: AlertSeverity
    description: str
    evidence_level: str


class PredictionResult(BaseModel):
    """Classification prediction with probability distribution."""
    predicted_category: DrugResponseCategory
    probabilities: dict[str, float]
    confidence: float = Field(..., ge=0, le=1)
    calibrated: bool = False


class GeneVariantDetail(BaseModel):
    """Detected pharmacogenomic variant details."""
    gene: str
    variant_id: str
    copies_of_variant: int = Field(..., ge=0, le=2)
    star_allele_diplotype: Optional[str] = None
    functional_status: Optional[str] = None
    clinical_significance: Optional[str] = None


class CPICGuidanceDetail(BaseModel):
    """CPIC guideline information for the alert."""
    guideline_id: Optional[str] = None
    guideline_version: Optional[str] = None
    evidence_level: Optional[str] = None
    recommendation_text: str
    monitoring_advice: str
    source_url: Optional[str] = None
    last_updated: Optional[str] = None


class PharmacogenomicAlert(BaseModel):
    """
    The primary output of Module 3: a structured pharmacogenomic alert
    to be appended to the SOAP note as a clinician suggestion.
    """
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: str = "pharmacogenomic_alert"
    severity: AlertSeverity
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)

    # Trigger context
    patient_id: str
    consultation_id: str
    matched_diagnosis_codes: list[str]
    matched_medication_names: list[str]

    # Genomic findings
    detected_variants: list[GeneVariantDetail]
    metabolizer_phenotype: MetabolizerPhenotype
    risk_level: RiskLevel

    # Model predictions
    drug_response_prediction: PredictionResult
    dose_estimate: Optional[DoseEstimate] = None

    # Clinical guidance
    cpic_guidance: CPICGuidanceDetail
    drug_interaction_warnings: list[DrugInteractionWarning] = Field(default_factory=list)

    # Explainability
    explainability: Optional[ExplainabilityReport] = None

    # SOAP integration
    soap_note_suggestion: str
    clinical_action_items: list[str] = Field(default_factory=list)

    # Safety
    warnings: list[str] = Field(default_factory=list)
    safety_notice: str = (
        "⚕️ DECISION SUPPORT ONLY — This pharmacogenomic alert is a suggestion "
        "for clinician review. It is NOT a diagnosis, prescription, or clinical "
        "order. The treating physician must independently verify all findings "
        "and make care decisions based on the complete clinical picture."
    )

    model_config = {"json_schema_extra": {"examples": []}}

    def to_soap_appendix(self) -> str:
        """Format as a SOAP note appendix section."""
        lines = [
            "═" * 72,
            "  PHARMACOGENOMIC DECISION SUPPORT — FOR CLINICIAN REVIEW ONLY",
            "═" * 72,
            "",
            f"  Alert ID:    {self.alert_id}",
            f"  Severity:    {self.severity.value.upper()}",
            f"  Patient:     {self.patient_id}",
            f"  Generated:   {self.timestamp.isoformat()}",
            "",
            "  ── Genomic Findings ─────────────────────────────────────────────",
        ]
        for v in self.detected_variants:
            lines.append(f"    Gene:     {v.gene}")
            lines.append(f"    Variant:  {v.variant_id} (copies: {v.copies_of_variant})")
            if v.star_allele_diplotype:
                lines.append(f"    Diplotype: {v.star_allele_diplotype}")
            lines.append("")
        lines.append(f"  Metabolizer Phenotype: {self.metabolizer_phenotype.value}")
        lines.append(f"  Risk Level:            {self.risk_level.value}")
        lines.append("")
        lines.append("  ── Predicted Drug Response ───────────────────────────────────────")
        lines.append(f"    Category:   {self.drug_response_prediction.predicted_category.value}")
        lines.append(f"    Confidence: {self.drug_response_prediction.confidence:.1%}")
        for cat, prob in self.drug_response_prediction.probabilities.items():
            lines.append(f"      P({cat}): {prob:.3f}")
        lines.append("")
        if self.dose_estimate:
            lines.append("  ── Dose Estimate ────────────────────────────────────────────────")
            lines.append(f"    Predicted:  {self.dose_estimate.predicted_dose:.2f} {self.dose_estimate.unit}")
            lines.append(f"    {self.dose_estimate.confidence_level:.0%} CI:     [{self.dose_estimate.confidence_interval_lower:.2f}, {self.dose_estimate.confidence_interval_upper:.2f}] {self.dose_estimate.unit}")
            lines.append("")
        lines.append("  ── CPIC Guidance ────────────────────────────────────────────────")
        lines.append(f"    {self.cpic_guidance.recommendation_text}")
        lines.append(f"    Monitoring: {self.cpic_guidance.monitoring_advice}")
        if self.cpic_guidance.source_url:
            lines.append(f"    Source: {self.cpic_guidance.source_url}")
        lines.append("")
        if self.drug_interaction_warnings:
            lines.append("  ── Drug Interaction Warnings ────────────────────────────────────")
            for w in self.drug_interaction_warnings:
                lines.append(f"    ⚠ {w.interacting_entity}: {w.description} [{w.severity.value}]")
            lines.append("")
        if self.clinical_action_items:
            lines.append("  ── Suggested Clinical Actions ───────────────────────────────────")
            for i, item in enumerate(self.clinical_action_items, 1):
                lines.append(f"    {i}. {item}")
            lines.append("")
        lines.append(f"  ⚕️  {self.safety_notice}")
        lines.append("═" * 72)
        return "\n".join(lines)
