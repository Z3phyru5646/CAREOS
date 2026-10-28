"""
CARE'OS Module 2 — Pydantic Data Models (Schemas)
===================================================
All structured data types used across the 5-stage pipeline.
Enforced via Instructor constrained JSON decoding.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class SpeakerRole(str, Enum):
    DOCTOR = "Doctor"
    PATIENT = "Patient"
    MEDICAL_ASSISTANT = "Medical Assistant"
    OTHER = "Other"


class SOAPSection(str, Enum):
    SUBJECTIVE = "S"
    OBJECTIVE = "O"
    ASSESSMENT = "A"
    PLAN = "P"


class VerificationTier(str, Enum):
    TIER_1_VERBATIM = "tier_1_verbatim"
    TIER_2A_TERMINOLOGY_UPGRADE = "tier_2a_terminology_upgrade"
    TIER_2B_ONTOLOGY_SYNONYM = "tier_2b_ontology_synonym"
    TIER_3_CLINICAL_INFERENCE = "tier_3_clinical_inference"
    TIER_4_UNSUPPORTED = "tier_4_unsupported"
    TIER_5_CONTRADICTION = "tier_5_contradiction"


class VerifierVerdict(str, Enum):
    SUPPORTED = "supported"
    SUPPORTED_TERMINOLOGY_UPGRADE = "supported_terminology_upgrade"
    SUPPORTED_SYNONYM = "supported_synonym"
    SUPPORTED_INFERENCE = "supported_inference"
    HALLUCINATED_UNSUPPORTED = "hallucinated_unsupported"
    HALLUCINATED_CONTRADICTION = "hallucinated_contradiction"


class RemediationAction(str, Enum):
    RETAIN = "retain"
    RETAIN_WITH_TOOLTIP = "retain_with_tooltip"
    RETAIN_WITH_BADGE = "retain_with_badge"
    AUTO_TRIM = "auto_trim"
    AUTO_REPAIR = "auto_repair"
    WITHHOLD = "withhold"


class NegationStatus(str, Enum):
    AFFIRMED = "affirmed"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"


class MEDECErrorCategory(str, Enum):
    DIAGNOSIS = "diagnosis"
    MANAGEMENT = "management"
    TREATMENT = "treatment"
    PHARMACOTHERAPY = "pharmacotherapy"
    CAUSAL_ORGANISM = "causal_organism"


# ─── Stage 1: Audio & Diarization Models ─────────────────────────────────────

class WordTimestamp(BaseModel):
    """A single word with phoneme-aligned timestamp from WhisperX."""
    word: str
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    speaker_id: Optional[str] = None


class DiarizedUtterance(BaseModel):
    """A speaker-attributed utterance with word-level timestamps."""
    utterance_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    speaker_id: str
    speaker_role: Optional[SpeakerRole] = None
    text: str
    words: list[WordTimestamp] = []
    start_time: float
    end_time: float
    language: str = "en"
    is_overlap: bool = False


class DiarizedTranscript(BaseModel):
    """Complete speaker-diarized transcript from Stage 1."""
    audio_file: str
    duration_seconds: float
    sample_rate: int = 16000
    num_speakers: int
    utterances: list[DiarizedUtterance]
    language: str = "en"
    hinglish_mode: bool = False
    created_at: datetime = Field(default_factory=datetime.now)


# ─── Stage 2: ASR Correction Models ──────────────────────────────────────────

class DrugCorrectionResult(BaseModel):
    """Result of pharmaceutical post-correction."""
    original_text: str
    corrected_text: str
    rxnorm_cui: Optional[str] = None
    rxnorm_name: Optional[str] = None
    correction_type: str = ""  # "phonetic", "substitution", "fragmentation", "none"
    confidence: float = 0.0
    formulary_match: bool = False


class CorrectedTranscript(BaseModel):
    """Transcript after Stage 2 medical post-correction."""
    original_transcript: DiarizedTranscript
    corrections: list[DrugCorrectionResult] = []
    corrected_utterances: list[DiarizedUtterance]
    medical_terms_corrected: int = 0
    drug_names_corrected: int = 0


# ─── Stage 3: Clinical NLP & SOAP Models ─────────────────────────────────────

class ClinicalEntity(BaseModel):
    """An extracted clinical entity with ontology linking."""
    text: str
    label: str  # DRUG, DISEASE, SYMPTOM, CHEMICAL, etc.
    start_char: int = 0
    end_char: int = 0
    negation: NegationStatus = NegationStatus.AFFIRMED
    umls_cui: Optional[str] = None
    umls_name: Optional[str] = None
    rxnorm_cui: Optional[str] = None
    snomed_code: Optional[str] = None
    icd10_code: Optional[str] = None
    source_utterance_id: Optional[str] = None
    confidence: float = 0.0


class MedicationEntity(BaseModel):
    """A Med7-extracted medication with posology attributes."""
    drug_name: str
    strength: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    form: Optional[str] = None
    duration: Optional[str] = None
    rxnorm_cui: Optional[str] = None
    negation: NegationStatus = NegationStatus.AFFIRMED
    source_utterance_id: Optional[str] = None


class EncounterFactSheet(BaseModel):
    """
    Structured pre-extraction from Stage 3 clinical NLP pipeline.
    This JSON object grounds the SOAP generation LLM.
    """
    patient_id: str = "anonymous"
    encounter_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    encounter_date: datetime = Field(default_factory=datetime.now)
    
    # Extracted entities
    symptoms: list[ClinicalEntity] = []
    diagnoses: list[ClinicalEntity] = []
    medications: list[MedicationEntity] = []
    procedures: list[ClinicalEntity] = []
    lab_results: list[ClinicalEntity] = []
    vitals: list[ClinicalEntity] = []
    allergies: list[ClinicalEntity] = []
    social_history: list[ClinicalEntity] = []
    family_history: list[ClinicalEntity] = []
    
    # Pertinent negatives (explicitly denied symptoms)
    pertinent_negatives: list[ClinicalEntity] = []
    
    # Speaker-attributed chief complaint
    chief_complaint: Optional[str] = None
    chief_complaint_utterance_id: Optional[str] = None


class AtomicClinicalClaim(BaseModel):
    """
    A single atomic assertion in the SOAP note.
    Schema-enforced via Instructor + Pydantic constrained decoding.
    """
    claim_id: str = Field(default_factory=lambda: f"c_{uuid.uuid4().hex[:8]}")
    soap_section: SOAPSection
    claim_text: str
    source_utterance_ids: list[str] = []
    audio_start_sec: Optional[float] = None
    audio_end_sec: Optional[float] = None
    extracted_cuis: list[str] = []
    original_lay_text: Optional[str] = None  # For Tier 2a tooltip


class SOAPNote(BaseModel):
    """Complete structured SOAP note with atomic claims."""
    encounter_id: str
    patient_id: str = "anonymous"
    
    # Section-by-section claims
    subjective: list[AtomicClinicalClaim] = []
    objective_exam: list[AtomicClinicalClaim] = []
    objective_results: list[AtomicClinicalClaim] = []
    assessment_plan: list[AtomicClinicalClaim] = []
    
    # Pharmacogenomic Decision Support (Module 3 Integration)
    pharmacogenomic_appendix: Optional[str] = None
    pharmacogenomic_alerts: list[dict] = Field(default_factory=list)
    
    # Metadata
    generated_by: str = "MedGemma-27B"
    generation_temperature: float = 0.1
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def all_claims(self) -> list[AtomicClinicalClaim]:
        return self.subjective + self.objective_exam + self.objective_results + self.assessment_plan


# ─── Stage 4: Verifier Models ────────────────────────────────────────────────

class ClaimVerificationResult(BaseModel):
    """Verification result for a single atomic claim."""
    claim_id: str
    claim_text: str
    soap_section: SOAPSection
    
    # Tier-1 Fast-Path NLI results
    minicheck_score: float = 0.0
    minicheck_label: int = 0  # 1=supported, 0=unsupported
    mednli_label: str = "neutral"  # entailment, neutral, contradiction
    mednli_confidence: float = 0.0
    
    # Bi-directional entailment score
    bidirectional_entailment: float = 0.0
    
    # Tier classification
    verification_tier: VerificationTier
    verdict: VerifierVerdict
    remediation: RemediationAction
    
    # Evidence
    supporting_evidence: list[str] = []
    matched_transcript_utterance_ids: list[str] = []
    ontology_matches: list[str] = []
    
    # Slow-Path reasoning (if applicable)
    slow_path_reasoning: Optional[str] = None
    
    # Negation polarity flip detection
    polarity_flip_detected: bool = False
    
    # Audio provenance
    audio_start_sec: Optional[float] = None
    audio_end_sec: Optional[float] = None


class MEDECCheckResult(BaseModel):
    """MEDEC clinical error detection result."""
    has_error: bool = False
    error_category: Optional[MEDECErrorCategory] = None
    erroneous_sentence: Optional[str] = None
    corrected_sentence: Optional[str] = None
    confidence: float = 0.0


class VerificationReport(BaseModel):
    """Complete 2nd-Pass Verification Report for the SOAP note."""
    encounter_id: str
    
    # Per-claim results
    claim_results: list[ClaimVerificationResult] = []
    
    # MEDEC results
    medec_results: list[MEDECCheckResult] = []
    
    # Aggregate statistics
    total_claims: int = 0
    tier1_count: int = 0
    tier2a_count: int = 0
    tier2b_count: int = 0
    tier3_count: int = 0
    tier4_count: int = 0
    tier5_count: int = 0
    
    # Hallucination rates
    literal_hallucination_rate: float = 0.0   # Stage 1 (lexical) rate
    inference_aware_hallucination_rate: float = 0.0  # Stage 2 (ontology-aware) rate
    
    created_at: datetime = Field(default_factory=datetime.now)
    
    def compute_stats(self):
        self.total_claims = len(self.claim_results)
        self.tier1_count = sum(1 for c in self.claim_results if c.verification_tier == VerificationTier.TIER_1_VERBATIM)
        self.tier2a_count = sum(1 for c in self.claim_results if c.verification_tier == VerificationTier.TIER_2A_TERMINOLOGY_UPGRADE)
        self.tier2b_count = sum(1 for c in self.claim_results if c.verification_tier == VerificationTier.TIER_2B_ONTOLOGY_SYNONYM)
        self.tier3_count = sum(1 for c in self.claim_results if c.verification_tier == VerificationTier.TIER_3_CLINICAL_INFERENCE)
        self.tier4_count = sum(1 for c in self.claim_results if c.verification_tier == VerificationTier.TIER_4_UNSUPPORTED)
        self.tier5_count = sum(1 for c in self.claim_results if c.verification_tier == VerificationTier.TIER_5_CONTRADICTION)
        
        if self.total_claims > 0:
            # Literal: anything not Tier 1 is "hallucinated"
            self.literal_hallucination_rate = (
                (self.tier2a_count + self.tier2b_count + self.tier3_count + self.tier4_count + self.tier5_count)
                / self.total_claims * 100
            )
            # Inference-aware: only Tier 4 and 5 are true hallucinations
            self.inference_aware_hallucination_rate = (
                (self.tier4_count + self.tier5_count) / self.total_claims * 100
            )


# ─── Co-Pilot Models ─────────────────────────────────────────────────────────

class DiagnosticHypothesis(BaseModel):
    """A working diagnostic hypothesis from the Co-Pilot."""
    hypothesis: str
    confidence: float = 0.0
    supporting_symptoms: list[str] = []
    icd10_code: Optional[str] = None


class RedFlagQuestion(BaseModel):
    """A red-flag question suggested by the Co-Pilot."""
    question: str
    rationale: str
    priority: str = "medium"  # high, medium, low
    related_hypothesis: Optional[str] = None


class CoPilotUpdate(BaseModel):
    """Periodic update from the Clinical Co-Pilot."""
    timestamp: datetime = Field(default_factory=datetime.now)
    current_soap_phase: str = "subjective"  # subjective, objective, assessment_plan
    hypotheses: list[DiagnosticHypothesis] = []
    red_flag_questions: list[RedFlagQuestion] = []
    pertinent_negatives_to_ask: list[str] = []
    elapsed_duration_sec: float = 0.0


# ─── Inter-Module Bus Models ─────────────────────────────────────────────────

class CPICTriggerRequest(BaseModel):
    """Module 2 → Module 3: CPIC Level A pharmacogenomic trigger."""
    patient_id: str
    encounter_id: str
    rxnorm_cui: str
    drug_name: str
    prescribed_dose: Optional[str] = None
    target_genes: list[str]
    therapeutic_class: str
    timestamp: datetime = Field(default_factory=datetime.now)


class CPICAlertResponse(BaseModel):
    """Module 3 → Module 2: Pharmacogenomic alert."""
    patient_id: str
    drug_name: str
    gene: str
    phenotype: str
    risk_level: str  # "critical", "high", "moderate"
    clinical_action: str
    alternative_drugs: list[str] = []


class EncounterTelemetry(BaseModel):
    """Module 2 → Module 1: Live consultation progress telemetry."""
    encounter_id: str
    patient_id: str
    elapsed_duration_sec: float
    current_soap_phase: str
    active_concept_count: int
    speaker_overlap_ratio: float = 0.0
    estimated_remaining_sec: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# ─── Stage 5: Export Models ───────────────────────────────────────────────────

class FHIRExportBundle(BaseModel):
    """HL7 FHIR R4 export metadata."""
    bundle_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    encounter_id: str
    patient_id: str
    bundle_type: str = "document"
    fhir_version: str = "R4"
    resources: list[dict] = []
    created_at: datetime = Field(default_factory=datetime.now)


class PatientSummary(BaseModel):
    """Plain-language patient after-visit summary."""
    encounter_id: str
    patient_id: str
    language: str = "en"
    assessment_summary: str = ""
    plan_summary: str = ""
    medications_list: list[str] = []
    red_flag_warnings: list[str] = []
    follow_up_instructions: str = ""
    teach_back_questions: list[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
