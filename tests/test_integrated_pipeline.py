#!/usr/bin/env python3
"""
CARE'OS — Integrated Module 2 + Module 3 End-to-End Pipeline Tests
====================================================================

Validates the seamless integration between:
  - Module 2 (AI Scribe & Verifier): Audio → SOAP note pipeline
  - Module 3 (Genomics Engine): SNP analysis → Pharmacogenomic alerts

Tests verify that:
  1. Module 2 transcription and verification complete successfully
  2. CPIC triggers activate real Module 3 GenomicsEngine (not stubs)
  3. PGx alerts are injected into SOAP note assessment_plan
  4. Pharmacogenomic appendix is generated
  5. FHIR export contains DetectedIssue resources for PGx alerts
  6. Patient summary includes personalized genetic safety warnings
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reconfigure stdout for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class TestModuleBusIntegration(unittest.TestCase):
    """Test the InterModuleBus integration with real Module 3 GenomicsEngine."""

    def setUp(self):
        """Set up a fresh InterModuleBus with Module 3 engine."""
        from module2_careos.inter_module_bus.bus import InterModuleBus
        self.bus = InterModuleBus({})

    def _make_fact_sheet(self, drug_names: list[str]):
        """Create a mock EncounterFactSheet with given medications."""
        from module2_careos.models.schemas import (
            EncounterFactSheet,
            MedicationEntity,
            ClinicalEntity,
        )
        medications = [
            MedicationEntity(
                drug_name=name,
                strength="20 mg" if "omeprazole" in name.lower() else "5 mg",
                frequency="once daily",
                route="oral",
            )
            for name in drug_names
        ]
        symptoms = [
            ClinicalEntity(
                text="epigastric pain",
                label="SYMPTOM",
                icd10_code="K21.0",
                confidence=0.95,
            ),
        ]
        return EncounterFactSheet(
            patient_id="PGX-TEST-001",
            medications=medications,
            symptoms=symptoms,
            chief_complaint="stomach pain after eating",
        )

    def test_genomics_engine_initializes(self):
        """Module 3 GenomicsEngine should load and train within the bus."""
        engine = self.bus.get_genomics_engine()
        self.assertIsNotNone(engine, "GenomicsEngine should initialize successfully")
        self.assertTrue(engine.is_trained, "Engine should be trained on synthetic cohort")

    def test_cpic_trigger_with_watchlist_drug(self):
        """CPIC trigger fires for omeprazole (CYP2C19 watchlist drug) with Module 3 active."""
        fact_sheet = self._make_fact_sheet(["omeprazole"])
        triggers = self.bus.check_cpic_triggers(fact_sheet)

        self.assertTrue(len(triggers) >= 1, "At least 1 CPIC trigger expected for omeprazole")

        t = triggers[0]
        self.assertTrue(t["module3_active"], "Module 3 should be active")
        self.assertIsNotNone(t["pgx_alert"], "PGx alert dict should be populated")
        self.assertIn("soap_suggestion", t, "Should have SOAP suggestion")
        self.assertTrue(len(t["soap_suggestion"]) > 0, "SOAP suggestion should not be empty")
        self.assertIn("soap_appendix", t, "Should have SOAP appendix")

    def test_cpic_trigger_for_warfarin(self):
        """CPIC trigger fires for warfarin with dose prediction."""
        fact_sheet = self._make_fact_sheet(["warfarin"])
        triggers = self.bus.check_cpic_triggers(fact_sheet)

        self.assertTrue(len(triggers) >= 1)
        t = triggers[0]
        self.assertTrue(t["module3_active"])
        pgx = t["pgx_alert"]
        self.assertIn("suggested_starting_dose", pgx)
        self.assertIn("predicted_response", pgx)

    def test_multiple_drugs_trigger(self):
        """Multiple CPIC drugs produce separate triggers."""
        fact_sheet = self._make_fact_sheet(["omeprazole", "warfarin", "codeine"])
        triggers = self.bus.check_cpic_triggers(fact_sheet)
        self.assertTrue(len(triggers) >= 3, f"Expected >=3 triggers, got {len(triggers)}")

    def test_non_cpic_drug_skipped(self):
        """Non-CPIC drugs should not generate triggers."""
        fact_sheet = self._make_fact_sheet(["acetaminophen"])
        triggers = self.bus.check_cpic_triggers(fact_sheet)
        self.assertEqual(len(triggers), 0, "acetaminophen is not on the CPIC watchlist")


class TestSOAPNoteEnrichment(unittest.TestCase):
    """Test that PGx alerts are correctly injected into SOAP note structures."""

    def setUp(self):
        from module2_careos.inter_module_bus.bus import InterModuleBus
        from module2_careos.models.schemas import (
            SOAPNote, AtomicClinicalClaim, SOAPSection,
            EncounterFactSheet, MedicationEntity, ClinicalEntity,
        )
        self.bus = InterModuleBus({})
        self.SOAPNote = SOAPNote
        self.AtomicClinicalClaim = AtomicClinicalClaim
        self.SOAPSection = SOAPSection
        self.EncounterFactSheet = EncounterFactSheet
        self.MedicationEntity = MedicationEntity
        self.ClinicalEntity = ClinicalEntity

    def test_pgx_claim_appended_to_assessment_plan(self):
        """PGx Decision Support claims are appended to SOAP assessment_plan."""
        fact_sheet = self.EncounterFactSheet(
            patient_id="TEST-002",
            medications=[
                self.MedicationEntity(drug_name="clopidogrel", strength="75 mg"),
            ],
            symptoms=[
                self.ClinicalEntity(text="chest pain", label="SYMPTOM", icd10_code="I21.0"),
            ],
        )

        soap_note = self.SOAPNote(
            encounter_id=fact_sheet.encounter_id,
            patient_id="TEST-002",
            assessment_plan=[
                self.AtomicClinicalClaim(
                    soap_section=self.SOAPSection.ASSESSMENT,
                    claim_text="Acute MI, NSTEMI",
                    source_utterance_ids=[],
                ),
            ],
        )

        triggers = self.bus.check_cpic_triggers(fact_sheet)

        # Simulate what main.py does
        for t in triggers:
            if t.get("soap_suggestion"):
                soap_note.assessment_plan.append(self.AtomicClinicalClaim(
                    soap_section=self.SOAPSection.PLAN,
                    claim_text=f"[PGx Decision Support] {t['soap_suggestion']}",
                    source_utterance_ids=[],
                ))
            if t.get("pgx_alert"):
                soap_note.pharmacogenomic_alerts.append(t["pgx_alert"])

        # Assert PGx claim was added
        pgx_claims = [c for c in soap_note.assessment_plan if "[PGx Decision Support]" in c.claim_text]
        self.assertTrue(len(pgx_claims) >= 1, "PGx Decision Support claim should be in assessment_plan")

        # Assert PGx alerts populated
        self.assertTrue(len(soap_note.pharmacogenomic_alerts) >= 1, "pharmacogenomic_alerts should be populated")

    def test_pharmacogenomic_appendix_generated(self):
        """Pharmacogenomic appendix text is generated from Module 3 output."""
        fact_sheet = self.EncounterFactSheet(
            patient_id="TEST-003",
            medications=[
                self.MedicationEntity(drug_name="simvastatin", strength="40 mg"),
            ],
        )

        triggers = self.bus.check_cpic_triggers(fact_sheet)
        appendices = [t["soap_appendix"] for t in triggers if t.get("soap_appendix")]

        self.assertTrue(len(appendices) >= 1, "SOAP appendix should be generated")
        self.assertTrue(len(appendices[0]) > 50, "Appendix should contain substantial content")


class TestFHIRExportEnrichment(unittest.TestCase):
    """Test that FHIR R4 exports include DetectedIssue for PGx alerts."""

    def setUp(self):
        from module2_careos.stage5_export.export_pipeline import FHIRExporter
        from module2_careos.models.schemas import (
            SOAPNote, EncounterFactSheet, VerificationReport,
            MedicationEntity, ClinicalEntity,
        )
        self.exporter = FHIRExporter()
        self.SOAPNote = SOAPNote
        self.EncounterFactSheet = EncounterFactSheet
        self.VerificationReport = VerificationReport
        self.MedicationEntity = MedicationEntity
        self.ClinicalEntity = ClinicalEntity

    def test_fhir_bundle_contains_detected_issue(self):
        """FHIR bundle has DetectedIssue resources when PGx alerts are present."""
        fact_sheet = self.EncounterFactSheet(
            patient_id="FHIR-001",
            medications=[
                self.MedicationEntity(drug_name="warfarin", strength="5 mg"),
            ],
        )
        soap_note = self.SOAPNote(
            encounter_id=fact_sheet.encounter_id,
            patient_id="FHIR-001",
            pharmacogenomic_alerts=[
                {
                    "gene": "CYP2C9",
                    "drug_name": "warfarin",
                    "diplotype": "*1/*2",
                    "phenotype": "Intermediate Metabolizer",
                    "predicted_response": "responder",
                    "risk_category": "high",
                    "suggested_starting_dose": "3.5 mg/day [2.8, 4.2] 90% CI",
                    "recommendation": "Reduce warfarin dose per CPIC guideline",
                    "monitoring_advice": "Monitor INR weekly for 4 weeks",
                },
            ],
        )
        verification = self.VerificationReport(encounter_id=fact_sheet.encounter_id)

        bundle = self.exporter.export(soap_note, fact_sheet, verification)

        detected_issues = [
            e for e in bundle["entry"]
            if e.get("resource", {}).get("resourceType") == "DetectedIssue"
        ]
        self.assertTrue(len(detected_issues) >= 1, "Should have at least 1 DetectedIssue")

        di = detected_issues[0]["resource"]
        self.assertEqual(di["status"], "preliminary")
        self.assertIn("PGX", di["code"]["coding"][0]["code"].upper())
        self.assertIn("CYP2C9", di["evidence"][0]["detail"][0]["display"])

    def test_fhir_bundle_without_pgx_alerts(self):
        """FHIR bundle has no DetectedIssue when no PGx alerts."""
        fact_sheet = self.EncounterFactSheet(
            patient_id="FHIR-002",
            medications=[],
        )
        soap_note = self.SOAPNote(
            encounter_id=fact_sheet.encounter_id,
            patient_id="FHIR-002",
        )
        verification = self.VerificationReport(encounter_id=fact_sheet.encounter_id)

        bundle = self.exporter.export(soap_note, fact_sheet, verification)
        detected_issues = [
            e for e in bundle["entry"]
            if e.get("resource", {}).get("resourceType") == "DetectedIssue"
        ]
        self.assertEqual(len(detected_issues), 0)


class TestPatientSummaryPGxEnrichment(unittest.TestCase):
    """Test that patient summaries include PGx safety warnings."""

    def setUp(self):
        from module2_careos.stage5_export.export_pipeline import PatientSummaryGenerator
        from module2_careos.models.schemas import SOAPNote, EncounterFactSheet
        self.generator = PatientSummaryGenerator()
        self.SOAPNote = SOAPNote
        self.EncounterFactSheet = EncounterFactSheet

    def test_english_summary_has_pgx_warning(self):
        """English patient summary includes genetic alert in red flags."""
        fact_sheet = self.EncounterFactSheet(patient_id="SUM-001")
        soap_note = self.SOAPNote(
            encounter_id=fact_sheet.encounter_id,
            patient_id="SUM-001",
            pharmacogenomic_alerts=[
                {
                    "drug_name": "warfarin",
                    "gene": "CYP2C9",
                    "phenotype": "Intermediate Metabolizer",
                    "predicted_response": "responder",
                    "suggested_starting_dose": "3.5 mg/day",
                    "monitoring_advice": "Weekly INR checks",
                },
            ],
        )
        summary = self.generator.generate(soap_note, fact_sheet, language="en")

        pgx_warnings = [w for w in summary.red_flag_warnings if "Genetic Alert" in w]
        self.assertTrue(len(pgx_warnings) >= 1, "Should have genetic alert in red flags")
        self.assertIn("CYP2C9", pgx_warnings[0])
        self.assertIn("warfarin", pgx_warnings[0])

    def test_hinglish_summary_has_pgx_warning(self):
        """Hinglish patient summary includes genetic alert."""
        fact_sheet = self.EncounterFactSheet(patient_id="SUM-002")
        soap_note = self.SOAPNote(
            encounter_id=fact_sheet.encounter_id,
            patient_id="SUM-002",
            pharmacogenomic_alerts=[
                {
                    "drug_name": "codeine",
                    "gene": "CYP2D6",
                    "phenotype": "Ultra-Rapid Metabolizer",
                    "predicted_response": "adverse_event_risk",
                    "suggested_starting_dose": "",
                    "monitoring_advice": "Avoid codeine",
                },
            ],
        )
        summary = self.generator.generate(soap_note, fact_sheet, language="hinglish")

        pgx_warnings = [w for w in summary.red_flag_warnings if "Genetic Alert" in w]
        self.assertTrue(len(pgx_warnings) >= 1)
        self.assertIn("CYP2D6", pgx_warnings[0])


class TestEMCodingPGxEnrichment(unittest.TestCase):
    """Test that E/M coding recognizes pharmacogenomic review."""

    def setUp(self):
        from module2_careos.stage5_export.export_pipeline import EMCodingChecker
        from module2_careos.models.schemas import (
            SOAPNote, EncounterFactSheet, AtomicClinicalClaim, SOAPSection,
        )
        self.checker = EMCodingChecker()
        self.SOAPNote = SOAPNote
        self.EncounterFactSheet = EncounterFactSheet
        self.AtomicClinicalClaim = AtomicClinicalClaim
        self.SOAPSection = SOAPSection

    def test_pharmacogenomic_review_flag(self):
        """E/M coding check includes pharmacogenomic_review when alerts present."""
        fact_sheet = self.EncounterFactSheet(patient_id="EM-001")
        soap_note = self.SOAPNote(
            encounter_id=fact_sheet.encounter_id,
            patient_id="EM-001",
            assessment_plan=[
                self.AtomicClinicalClaim(
                    soap_section=self.SOAPSection.ASSESSMENT,
                    claim_text="Suspected peptic ulcer",
                ),
            ],
            pharmacogenomic_alerts=[{"gene": "CYP2C19", "drug_name": "omeprazole"}],
        )
        result = self.checker.check(soap_note, fact_sheet)
        self.assertTrue(result.get("pharmacogenomic_review"), "Should flag PGx review")


class TestEndToEndPipeline(unittest.TestCase):
    """
    Full end-to-end test: CareOSModule2Pipeline.process_encounter()
    with Module 3 Genomics Engine integration.
    """

    def test_full_pipeline_demo_mode(self):
        """Run complete pipeline in demo mode and verify PGx integration."""
        from module2_careos.main import CareOSModule2Pipeline

        pipeline = CareOSModule2Pipeline()
        demo_audio = Path(__file__).resolve().parent.parent / "data" / "sample_audio" / "demo.wav"

        results = pipeline.process_encounter(
            audio_path=str(demo_audio),
            patient_id="E2E-TEST-001",
        )

        # Core pipeline sanity
        self.assertIn("encounter_id", results)
        self.assertIn("transcript", results)
        self.assertIn("soap_note", results)
        self.assertIn("verification", results)
        self.assertIn("fhir_bundle", results)

        # CPIC triggers should fire (demo encounter includes CPIC drugs)
        self.assertIn("cpic_triggers", results)
        triggers = results["cpic_triggers"]
        self.assertTrue(len(triggers) >= 1, "Demo encounter should produce CPIC triggers")

        # Module 3 should be active
        active_triggers = [t for t in triggers if t.get("module3_active")]
        self.assertTrue(len(active_triggers) >= 1, "At least 1 trigger should have Module 3 active")

        # PGx alerts should be in results
        self.assertIn("pharmacogenomic_alerts", results)
        if active_triggers:
            self.assertTrue(len(results["pharmacogenomic_alerts"]) >= 1)

        # SOAP note should contain PGx claims
        soap = results["soap_note"]
        pgx_claims = [
            c for c in soap.get("assessment_plan", [])
            if "[PGx Decision Support]" in c.get("claim_text", "")
        ]
        if active_triggers:
            self.assertTrue(len(pgx_claims) >= 1, "SOAP should contain PGx Decision Support claims")


if __name__ == "__main__":
    unittest.main(verbosity=2)
