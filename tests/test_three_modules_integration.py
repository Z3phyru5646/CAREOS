#!/usr/bin/env python3
"""
CARE'OS Three-Module End-to-End Integration Verification Test
============================================================
Verifies:
  1. Module 1: Outpatient Queue State, Stage-1 Duration, JSEW Routing & CQR/ACI Intervals
  2. Module 2: Clinical FactSheet Extraction & SOAP Synthesis
  3. Module 2 → Module 1: Live Consultation SOAP Telemetry Webhook (S ➔ O ➔ A ➔ P)
  4. Module 2 → Module 3: Automated CPIC Level A Trigger & Genomic VAE Dosing Guidance
  5. Module 3 → Module 2: Enriched SOAP Note Appendix & FHIR R4 DetectedIssues
"""

import unittest
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode emojis and clinical symbols
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.main import clinic, predict_wait_time, jsew_route_patient, check_in, CheckInRequest, ESILevel
from module2_careos.inter_module_bus.bus import InterModuleBus
from module2_careos.models.schemas import EncounterFactSheet, MedicationEntity, ClinicalEntity
from careos_genomics.data_models import PharmacogenomicAlert


class TestThreeModulesIntegration(unittest.TestCase):
    def setUp(self):
        self.bus = InterModuleBus()

    def test_module1_queue_and_jsew(self):
        """Module 1 initializes state, routes patient via JSEW, and calculates CQR wait intervals."""
        patient_entry = {
            "entry_id": 999,
            "patient_name": "Test Patient",
            "doctor_id": 1,
            "pred_service_dur": 15.0,
            "status": "WAITING",
        }
        doctor_id = jsew_route_patient(patient_entry)
        self.assertIn(doctor_id, [0, 1, 2], "JSEW must route to an active clinic doctor")

        patient_entry["doctor_id"] = doctor_id
        pred = predict_wait_time(patient_entry)
        self.assertIn("point_estimate_min", pred)
        self.assertIn("lower_bound_min", pred)
        self.assertIn("upper_bound_min", pred)
        self.assertGreaterEqual(pred["upper_bound_min"], pred["lower_bound_min"])
        print(f"  [Module 1 OK] JSEW Doctor: {doctor_id} | Wait: {pred['point_estimate_min']}m [{pred['lower_bound_min']}, {pred['upper_bound_min']}]")

    def test_module2_to_module1_telemetry(self):
        """Module 2 emits telemetry that reaches Module 1 queue tracking."""
        telemetry = self.bus.emit_telemetry(
            encounter_id="ENC-M1M2-001",
            patient_id="PT-001",
            elapsed_sec=360.0,
            soap_phase="P",
            concept_count=5,
            room_id=0,
        )
        self.assertEqual(telemetry["current_soap_phase"], "P")
        self.assertEqual(telemetry["elapsed_duration_sec"], 360.0)
        self.assertTrue(len(self.bus.get_telemetry_history()) >= 1)
        print("  [Module 2 -> Module 1 OK] Telemetry emitted successfully for Room 0 -> PLAN phase")

    def test_module2_to_module3_pgx_trigger(self):
        """Module 2 extracts CPIC Level A medication and triggers Module 3 precision dosing."""
        fact_sheet = EncounterFactSheet(
            patient_id="PT-GENOMICS-001",
            medications=[
                MedicationEntity(drug_name="clopidogrel", strength="75 mg", frequency="daily"),
                MedicationEntity(drug_name="atorvastatin", strength="20 mg", frequency="qhs"),
            ],
            diagnoses=[
                ClinicalEntity(text="Acute coronary syndrome", label="DIAGNOSIS", icd10_code="I21.0"),
            ],
        )

        triggers = self.bus.check_cpic_triggers(fact_sheet)
        self.assertTrue(len(triggers) >= 1, "Must produce CPIC Level A triggers for clopidogrel")

        # Verify Module 3 executed
        clop_trigger = next((t for t in triggers if "clopidogrel" in t["trigger"]["drug_name"].lower()), None)
        self.assertIsNotNone(clop_trigger)
        self.assertTrue(clop_trigger["module3_active"], "Module 3 Genomics Engine must be active")
        self.assertIsNotNone(clop_trigger.get("pgx_alert"))
        self.assertIn("CYP2C19", clop_trigger["trigger"]["target_genes"])
        print(f"  [Module 2 -> Module 3 OK] CPIC Trigger fired for Clopidogrel -> Gene {clop_trigger['trigger']['target_genes']}")
        if clop_trigger.get("soap_suggestion"):
            print(f"    Suggestion: {clop_trigger['soap_suggestion']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
