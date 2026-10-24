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
from typing import Optional

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
    
    def check_cpic_triggers(
        self,
        fact_sheet: "EncounterFactSheet",
    ) -> list[dict]:
        """
        Check extracted medications against CPIC Level A watchlist.
        Returns list of triggered alerts.
        """
        from ..models.schemas import CPICTriggerRequest, CPICAlertResponse
        
        triggers = []
        
        for med in fact_sheet.medications:
            drug_lower = med.drug_name.lower()
            
            if drug_lower in CPIC_LEVEL_A_WATCHLIST:
                watchlist_entry = CPIC_LEVEL_A_WATCHLIST[drug_lower]
                
                trigger = CPICTriggerRequest(
                    patient_id=fact_sheet.patient_id,
                    encounter_id=fact_sheet.encounter_id,
                    rxnorm_cui=med.rxnorm_cui or "",
                    drug_name=med.drug_name,
                    prescribed_dose=f"{med.strength or ''} {med.frequency or ''}".strip(),
                    target_genes=watchlist_entry["target_genes"],
                    therapeutic_class=watchlist_entry["therapeutic_class"],
                )
                
                # Simulate Module 3 response
                alert = CPICAlertResponse(
                    patient_id=fact_sheet.patient_id,
                    drug_name=med.drug_name,
                    gene=watchlist_entry["target_genes"][0],
                    phenotype=watchlist_entry["high_risk_phenotype"],
                    risk_level=watchlist_entry["risk_level"],
                    clinical_action=watchlist_entry["clinical_action"],
                )
                
                triggers.append({
                    "trigger": trigger.model_dump(),
                    "alert": alert.model_dump(),
                    "watchlist_entry": watchlist_entry,
                })
                
                logger.warning(
                    f"🧬 CPIC Level A Trigger: {med.drug_name} → "
                    f"Genes: {watchlist_entry['target_genes']} | "
                    f"Risk: {watchlist_entry['risk_level']}"
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
    ) -> dict:
        """
        Emit encounter telemetry to Module 1 (Queue Predictor).
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
        
        return telemetry.model_dump()
    
    def get_pending_alerts(self) -> list[dict]:
        """Get all pending CPIC alerts for the verification console."""
        return self._pending_alerts
    
    def get_telemetry_history(self) -> list[dict]:
        """Get all telemetry heartbeats."""
        return self._telemetry_history
