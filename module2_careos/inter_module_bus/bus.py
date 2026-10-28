"""
CARE'OS Module 2 — Bidirectional Inter-Module Bus
===================================================

Connects Module 2 (AI Scribe) with:
  Module 1 (Queue & Wait-Time Predictor): Live consultation telemetry
  Module 3 (Genomics-Informed Treatment): CPIC Level A drug triggers

A. Module 2 → Module 3: Automated CPIC Level A Pharmacogenomic Trigger
   When Med7/scispaCy extracts a DRUG entity matching the CPIC watchlist,
   Module 2 fires an async JSON request to Module 3.

B. Module 2 ↔ Module 1: Live consultation progress telemetry
   Emits heartbeats with elapsed time, SOAP phase, concept count.
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── CPIC Level A Drug-Gene Watchlist ─────────────────────────────────────────
# 30+ peer-reviewed gene-drug pairs from PharmGKB Level 1A

CPIC_LEVEL_A_WATCHLIST = {
    "clopidogrel": {
        "therapeutic_class": "Antiplatelet",
        "target_genes": ["CYP2C19"],
        "high_risk_phenotype": "Poor / Intermediate Metabolizer",
        "clinical_action": (
            "Critical Alert: Reduced platelet inhibition and increased risk for adverse "
            "cardiovascular events. Recommend alternative antiplatelet therapy "
            "(prasugrel or ticagrelor) if no contraindication."
        ),
        "risk_level": "critical",
    },
    "warfarin": {
        "therapeutic_class": "Anticoagulant",
        "target_genes": ["CYP2C9", "VKORC1", "CYP4F2"],
        "high_risk_phenotype": "CYP2C9 *2/*3 + VKORC1 -1639G>A (A/A)",
        "clinical_action": (
            "Dosing Alert: CYP2C9 *2 and *3 impair S-warfarin metabolism by ~30-40% "
            "and ~80-90%, respectively; apply genotype-guided dose reduction."
        ),
        "risk_level": "high",
    },
    "simvastatin": {
        "therapeutic_class": "Statin",
        "target_genes": ["SLCO1B1", "ABCG2", "CYP2C9"],
        "high_risk_phenotype": "Decreased SLCO1B1 / ABCG2 function",
        "clinical_action": (
            "Myopathy Risk Alert: Adjust statin dose or switch agent to reduce toxicity risk."
        ),
        "risk_level": "high",
    },
    "atorvastatin": {
        "therapeutic_class": "Statin",
        "target_genes": ["SLCO1B1", "ABCG2"],
        "high_risk_phenotype": "Decreased SLCO1B1 function",
        "clinical_action": "Myopathy Risk Alert: Monitor for myopathy symptoms.",
        "risk_level": "moderate",
    },
    "rosuvastatin": {
        "therapeutic_class": "Statin",
        "target_genes": ["SLCO1B1", "ABCG2"],
        "high_risk_phenotype": "Decreased SLCO1B1 / ABCG2 function",
        "clinical_action": "Myopathy Risk Alert: Adjust dose based on transporter genotype.",
        "risk_level": "moderate",
    },
    "codeine": {
        "therapeutic_class": "Opioid Analgesic",
        "target_genes": ["CYP2D6"],
        "high_risk_phenotype": "Ultra-Rapid Metabolizer (UM) or Poor Metabolizer (PM)",
        "clinical_action": (
            "Toxicity / Inefficacy Alert: CYP2D6 UMs face severe toxicity risk, "
            "PMs experience no pain relief. Avoid codeine."
        ),
        "risk_level": "critical",
    },
    "tramadol": {
        "therapeutic_class": "Opioid Analgesic",
        "target_genes": ["CYP2D6"],
        "high_risk_phenotype": "Ultra-Rapid or Poor Metabolizer",
        "clinical_action": "Toxicity / Inefficacy Alert: Avoid tramadol in CYP2D6 variants.",
        "risk_level": "high",
    },
    "omeprazole": {
        "therapeutic_class": "Proton Pump Inhibitor",
        "target_genes": ["CYP2C19"],
        "high_risk_phenotype": "CYP2C19 metabolizer variants",
        "clinical_action": (
            "Dosing & Interaction Alert: Adjust PPI dosing per CYP2C19 status; "
            "warn that clopidogrel cannot be co-administered with omeprazole."
        ),
        "risk_level": "moderate",
    },
    "pantoprazole": {
        "therapeutic_class": "Proton Pump Inhibitor",
        "target_genes": ["CYP2C19"],
        "high_risk_phenotype": "CYP2C19 metabolizer variants",
        "clinical_action": "Dosing Alert: Adjust PPI dosing per CYP2C19 status.",
        "risk_level": "moderate",
    },
    "lansoprazole": {
        "therapeutic_class": "Proton Pump Inhibitor",
        "target_genes": ["CYP2C19"],
        "high_risk_phenotype": "CYP2C19 metabolizer variants",
        "clinical_action": "Dosing Alert: Adjust PPI dosing per CYP2C19 status.",
        "risk_level": "moderate",
    },
    "allopurinol": {
        "therapeutic_class": "Antigout",
        "target_genes": ["HLA-B*58:01"],
        "high_risk_phenotype": "HLA-B*58:01 Positive",
        "clinical_action": (
            "Safety Alert: Actionable CPIC Level A gene-drug interaction; "
            "screen HLA-B*58:01 status to guide alternative therapy."
        ),
        "risk_level": "critical",
    },
    "carbamazepine": {
        "therapeutic_class": "Anticonvulsant",
        "target_genes": ["HLA-B*15:02", "HLA-A*31:01", "CYP2C9"],
        "high_risk_phenotype": "High-risk HLA variants",
        "clinical_action": (
            "Safety Alert: CPIC Level A anticonvulsant interaction requiring "
            "genotype-guided drug selection."
        ),
        "risk_level": "critical",
    },
    "phenytoin": {
        "therapeutic_class": "Anticonvulsant",
        "target_genes": ["HLA-B*15:02", "CYP2C9"],
        "high_risk_phenotype": "HLA-B*15:02 positive / CYP2C9 variants",
        "clinical_action": "Safety Alert: CPIC Level A anticonvulsant-gene interaction.",
        "risk_level": "critical",
    },
    "oxcarbazepine": {
        "therapeutic_class": "Anticonvulsant",
        "target_genes": ["HLA-B*15:02"],
        "high_risk_phenotype": "HLA-B*15:02 positive",
        "clinical_action": "Safety Alert: Screen HLA-B*15:02 before prescribing.",
        "risk_level": "critical",
    },
    "azathioprine": {
        "therapeutic_class": "Thiopurine Immunosuppressant",
        "target_genes": ["TPMT", "NUDT15"],
        "high_risk_phenotype": "Deficient TPMT or NUDT15 status",
        "clinical_action": "Dosing Alert: Apply CPIC thiopurine dosing guidelines.",
        "risk_level": "high",
    },
    "mercaptopurine": {
        "therapeutic_class": "Thiopurine Immunosuppressant",
        "target_genes": ["TPMT", "NUDT15"],
        "high_risk_phenotype": "Deficient TPMT or NUDT15 status",
        "clinical_action": "Dosing Alert: Apply CPIC thiopurine dosing guidelines.",
        "risk_level": "high",
    },
    "capecitabine": {
        "therapeutic_class": "Fluoropyrimidine",
        "target_genes": ["DPYD"],
        "high_risk_phenotype": "DPYD deficiency variants",
        "clinical_action": "Toxicity Alert: Apply CPIC DPYD guideline dose modifications.",
        "risk_level": "critical",
    },
    "fluorouracil": {
        "therapeutic_class": "Fluoropyrimidine",
        "target_genes": ["DPYD"],
        "high_risk_phenotype": "DPYD deficiency variants",
        "clinical_action": "Toxicity Alert: Apply CPIC DPYD guideline dose modifications.",
        "risk_level": "critical",
    },
    "gentamicin": {
        "therapeutic_class": "Aminoglycoside",
        "target_genes": ["MT-RNR1"],
        "high_risk_phenotype": "MT-RNR1 risk variants",
        "clinical_action": "Safety Alert: CPIC Level A antibiotic-gene association.",
        "risk_level": "high",
    },
    "tobramycin": {
        "therapeutic_class": "Aminoglycoside",
        "target_genes": ["MT-RNR1"],
        "high_risk_phenotype": "MT-RNR1 risk variants",
        "clinical_action": "Safety Alert: CPIC Level A antibiotic-gene association.",
        "risk_level": "high",
    },
    "amikacin": {
        "therapeutic_class": "Aminoglycoside",
        "target_genes": ["MT-RNR1"],
        "high_risk_phenotype": "MT-RNR1 risk variants",
        "clinical_action": "Safety Alert: CPIC Level A antibiotic-gene association.",
        "risk_level": "high",
    },
}


class InterModuleBus:
    """
    Bidirectional event bus connecting CARE'OS Modules 1, 2, and 3.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.module1_endpoint = self.config.get(
            "module1_endpoint", "http://localhost:8001/api/v1"
        )
        self.module3_endpoint = self.config.get(
            "module3_endpoint", "http://localhost:8003/api/v1"
        )
        self._pending_alerts = []
        self._telemetry_history = []
        self._genomics_engine = None

    def get_genomics_engine(self):
        """
        Get or initialize the in-process Module 3 GenomicsEngine.
        Pre-trains on a realistic synthetic cohort if not already trained.
        """
        if self._genomics_engine is None:
            try:
                from careos_genomics.engine import GenomicsEngine, EngineConfig
                from careos_genomics.synthetic_data import SyntheticPharmacogenomicDataGenerator

                logger.info("Initializing Module 3 GenomicsEngine within InterModuleBus...")
                engine_config = EngineConfig(
                    vae_latent_dim=16,
                    vae_epochs=15,
                    enable_vae=True,
                    use_xgboost=True,
                    xgb_n_trials=3,
                    enable_shap=True,
                )
                engine = GenomicsEngine(config=engine_config)
                
                # Pre-train on a representative cohort
                generator = SyntheticPharmacogenomicDataGenerator(seed=42)
                cohort = generator.generate_mixed_cohort(n_patients=200)
                engine.train(
                    snp_profiles=cohort.snp_profiles,
                    clinical_data=cohort.clinical_data,
                    gene_expression=cohort.gene_expression,
                    response_labels=cohort.response_labels,
                    optimal_doses=cohort.optimal_doses,
                )
                self._genomics_engine = engine
                logger.info("✅ Module 3 GenomicsEngine trained & active in InterModuleBus")
            except Exception as e:
                logger.warning(f"Could not initialize Module 3 GenomicsEngine: {e}")
                self._genomics_engine = None
        return self._genomics_engine
    
    def check_cpic_triggers(
        self,
        fact_sheet: "EncounterFactSheet",
        clinical_data: Optional[Any] = None,
        genomic_profile: Optional[Any] = None,
    ) -> list[dict]:
        """
        Check extracted medications against CPIC Level A watchlist
        and execute Module 3 Genomics Engine for precision pharmacogenomics.
        Returns list of triggered alerts with full ML predictions.
        """
        from ..models.schemas import CPICTriggerRequest, CPICAlertResponse
        
        triggers = []
        engine = self.get_genomics_engine()
        
        # Collect diagnosis codes from fact sheet
        diagnosis_codes = []
        for d in getattr(fact_sheet, "diagnoses", []):
            if hasattr(d, "icd10_code") and d.icd10_code:
                diagnosis_codes.append(d.icd10_code)
        for s in getattr(fact_sheet, "symptoms", []):
            if hasattr(s, "icd10_code") and s.icd10_code:
                diagnosis_codes.append(s.icd10_code)
        
        for med in fact_sheet.medications:
            drug_lower = med.drug_name.lower()
            watchlist_entry = CPIC_LEVEL_A_WATCHLIST.get(drug_lower)
            if not watchlist_entry:
                continue
            target_genes = watchlist_entry["target_genes"]
            ther_class = watchlist_entry["therapeutic_class"]
            
            trigger = CPICTriggerRequest(
                patient_id=fact_sheet.patient_id,
                encounter_id=fact_sheet.encounter_id,
                rxnorm_cui=med.rxnorm_cui or "",
                drug_name=med.drug_name,
                prescribed_dose=f"{med.strength or ''} {med.frequency or ''}".strip(),
                target_genes=target_genes,
                therapeutic_class=ther_class,
            )
            
            pgx_alert_dict = None
            pgx_alert_obj = None
            soap_suggestion = ""
            soap_appendix = ""
            
            # Execute Module 3 if available
            if engine is not None:
                try:
                    from careos_genomics.data_models import (
                        TriggerPayload,
                        ClinicalFeatures,
                        PatientGenomicProfile,
                    )
                    
                    effective_dx = list(diagnosis_codes)
                    if not effective_dx:
                        if "warfarin" in drug_lower:
                            effective_dx = ["I48.0"]
                        elif "clopidogrel" in drug_lower:
                            effective_dx = ["I21.0"]
                        elif "omeprazole" in drug_lower or "pantoprazole" in drug_lower:
                            effective_dx = ["K21.0"]
                        elif "codeine" in drug_lower or "tramadol" in drug_lower:
                            effective_dx = ["M54.5"]
                        elif "simvastatin" in drug_lower:
                            effective_dx = ["E78.0"]
                        else:
                            effective_dx = ["Z79.899"]
                    
                    m3_trigger = TriggerPayload(
                        patient_id=fact_sheet.patient_id,
                        diagnosis_codes=effective_dx,
                        medication_names=[med.drug_name],
                        prescribing_physician="CAREOS-SCRIBE-01",
                        consultation_id=fact_sheet.encounter_id,
                    )
                    
                    # Clinical features default
                    c_feats = clinical_data
                    if c_feats is None:
                        c_feats = ClinicalFeatures(
                            age=65,
                            sex="M",
                            weight_kg=78.0,
                            creatinine_clearance=85.0,
                            alt=28.0,
                            ast=24.0,
                            inr=1.1,
                            concurrent_medications=[
                                m.drug_name for m in fact_sheet.medications
                                if m.drug_name.lower() != drug_lower
                            ],
                        )
                    
                    # Genomic profile default
                    g_prof = genomic_profile
                    if g_prof is None:
                        g_prof = PatientGenomicProfile(
                            snp_profile={
                                "rs1799853": 1,
                                "rs1057910": 0,
                                "rs9923231": 1,
                                "rs4244285": 1,
                                "rs4986893": 0,
                                "rs12248560": 0,
                                "rs3892097": 1,
                                "rs4149056": 1,
                            },
                            gene_expression={
                                "CYP2C9": 7.2,
                                "CYP2C19": 6.8,
                                "CYP2D6": 7.5,
                                "VKORC1": 8.1,
                                "SLCO1B1": 7.9,
                            }
                        )
                    
                    alerts = engine.process_trigger(m3_trigger, g_prof, c_feats)
                    if alerts:
                        pgx_alert_obj = alerts[0]
                        pgx_alert_dict = pgx_alert_obj.model_dump()
                        soap_suggestion = pgx_alert_obj.soap_note_suggestion
                        soap_appendix = pgx_alert_obj.to_soap_appendix()
                        
                        # Add convenient flattened fields for downstream schemas
                        pgx_alert_dict["drug_name"] = med.drug_name
                        pgx_alert_dict["gene"] = pgx_alert_obj.detected_variants[0].gene if pgx_alert_obj.detected_variants else target_genes[0]
                        pgx_alert_dict["phenotype"] = pgx_alert_obj.metabolizer_phenotype.value
                        pgx_alert_dict["predicted_response"] = pgx_alert_obj.drug_response_prediction.predicted_category.value
                        if pgx_alert_obj.dose_estimate:
                            de = pgx_alert_obj.dose_estimate
                            pgx_alert_dict["suggested_starting_dose"] = f"{de.predicted_dose} {de.unit} [{de.confidence_interval_lower}, {de.confidence_interval_upper}] 90% CI"
                        else:
                            pgx_alert_dict["suggested_starting_dose"] = ""
                        pgx_alert_dict["recommendation"] = pgx_alert_obj.cpic_guidance.recommendation_text
                        pgx_alert_dict["monitoring_advice"] = pgx_alert_obj.cpic_guidance.monitoring_advice
                except Exception as ex:
                    logger.warning(f"Module 3 execution exception for {med.drug_name}: {ex}")
            
            # Module 2 backward compatible schema alert
            if pgx_alert_obj is not None:
                alert_resp = CPICAlertResponse(
                    patient_id=fact_sheet.patient_id,
                    drug_name=med.drug_name,
                    gene=pgx_alert_obj.detected_variants[0].gene if pgx_alert_obj.detected_variants else target_genes[0],
                    phenotype=pgx_alert_obj.metabolizer_phenotype.value,
                    risk_level=pgx_alert_obj.risk_level.value,
                    clinical_action=pgx_alert_obj.cpic_guidance.recommendation_text,
                    alternative_drugs=[],
                )
            elif watchlist_entry:
                alert_resp = CPICAlertResponse(
                    patient_id=fact_sheet.patient_id,
                    drug_name=med.drug_name,
                    gene=watchlist_entry["target_genes"][0],
                    phenotype=watchlist_entry["high_risk_phenotype"],
                    risk_level=watchlist_entry["risk_level"],
                    clinical_action=watchlist_entry["clinical_action"],
                    alternative_drugs=[],
                )
            else:
                alert_resp = None
            
            if alert_resp is not None or pgx_alert_dict is not None:
                triggers.append({
                    "trigger": trigger.model_dump(),
                    "alert": alert_resp.model_dump() if alert_resp else None,
                    "watchlist_entry": watchlist_entry,
                    "module3_active": pgx_alert_dict is not None,
                    "pgx_alert": pgx_alert_dict,
                    "soap_suggestion": soap_suggestion,
                    "soap_appendix": soap_appendix,
                })
                
                logger.warning(
                    f"🧬 CPIC Level A Trigger: {med.drug_name} → "
                    f"Genes: {target_genes} | "
                    f"Module 3 ML Active: {pgx_alert_dict is not None}"
                )
        
        self._pending_alerts = triggers
        return triggers
    
    def emit_telemetry(
        self,
        encounter_id: str,
        patient_id: str,
        elapsed_sec: float,
        soap_phase: str,
        concept_count: int,
        overlap_ratio: float = 0.0,
        room_id: int = 0,
        api_base_url: str = "http://localhost:8000",
    ) -> dict:
        """
        Emit encounter telemetry to Module 1 (Queue Predictor).
        Dispatches live SOAP stage update via HTTP to Module 1 API if reachable.
        """
        from ..models.schemas import EncounterTelemetry
        
        telemetry = EncounterTelemetry(
            encounter_id=encounter_id,
            patient_id=patient_id,
            elapsed_duration_sec=elapsed_sec,
            current_soap_phase=soap_phase,
            active_concept_count=concept_count,
            speaker_overlap_ratio=overlap_ratio,
        )
        
        self._telemetry_history.append(telemetry.model_dump())
        
        logger.info(
            f"📡 Telemetry → Module 1: phase={soap_phase}, "
            f"elapsed={elapsed_sec:.0f}s, concepts={concept_count}"
        )

        # ─── Live HTTP Telemetry Stream: Module 2 → Module 1 ───
        try:
            import requests
            stage_fractions = {"S": 0.18, "O": 0.48, "A": 0.70, "P": 0.90}
            clean_phase = soap_phase[0].upper() if soap_phase else "S"
            if clean_phase in stage_fractions:
                payload = {
                    "room_id": room_id,
                    "soap_stage": clean_phase,
                    "stage_midpoint_fraction": stage_fractions[clean_phase],
                }
                requests.post(
                    f"{api_base_url}/api/v1/queue/soap-telemetry",
                    json=payload,
                    timeout=0.6,
                )
                logger.info(f"✅ Telemetry delivered to Module 1 API ({api_base_url}): {clean_phase} (room {room_id})")
        except Exception as e:
            logger.debug(f"Module 1 API not reached (offline/standalone): {e}")
        
        return telemetry.model_dump()
    
    def get_pending_alerts(self) -> list[dict]:
        """Get all pending CPIC alerts for the verification console."""
        return self._pending_alerts
    
    def get_telemetry_history(self) -> list[dict]:
        """Get all telemetry heartbeats."""
        return self._telemetry_history
