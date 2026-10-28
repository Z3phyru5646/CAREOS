"""
CARE'OS Module 2 — Unit & Integration Test Suite
=================================================
"""

import unittest
from module2_careos.main import CareOSModule2Pipeline


class TestCareOSModule2Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = CareOSModule2Pipeline()

    def test_pipeline_initialization(self):
        self.assertIsNotNone(self.pipeline.stage1)
        self.assertIsNotNone(self.pipeline.stage2)
        self.assertIsNotNone(self.pipeline.stage3)
        self.assertIsNotNone(self.pipeline.stage4)
        self.assertIsNotNone(self.pipeline.stage5)
        self.assertIsNotNone(self.pipeline.copilot)
        self.assertIsNotNone(self.pipeline.bus)

    def test_demo_encounter_execution(self):
        results = self.pipeline.process_encounter(
            audio_path="data/sample_audio/demo.wav",
            patient_id="TEST-PT-001",
            language="en",
        )
        self.assertIn("encounter_id", results)
        self.assertIn("verification", results)
        self.assertIn("soap_note", results)
        self.assertIn("fhir_bundle", results)
        self.assertIn("copilot", results)
        self.assertIn("cpic_triggers", results)
        self.assertIn("telemetry", results)

    def test_verification_taxonomy_tiers(self):
        results = self.pipeline.process_encounter(
            audio_path="data/sample_audio/demo.wav",
            patient_id="TEST-PT-002",
        )
        ver = results["verification"]
        total = ver["total_claims"]
        self.assertGreater(total, 0)
        sum_tiers = (
            ver["tier1_count"]
            + ver["tier2a_count"]
            + ver["tier2b_count"]
            + ver["tier3_count"]
            + ver["tier4_count"]
            + ver["tier5_count"]
        )
        self.assertEqual(sum_tiers, total)

    def test_inter_module_cpic_triggers(self):
        results = self.pipeline.process_encounter(
            audio_path="data/sample_audio/demo.wav",
            patient_id="TEST-PT-003",
        )
        triggers = results["cpic_triggers"]
        self.assertGreaterEqual(len(triggers), 1)
        omeprazole_trigger = any(t["alert"]["drug_name"] == "omeprazole" for t in triggers)
        self.assertTrue(omeprazole_trigger)

    def test_fhir_r4_bundle_export(self):
        results = self.pipeline.process_encounter(
            audio_path="data/sample_audio/demo.wav",
            patient_id="TEST-PT-004",
        )
        bundle = results["fhir_bundle"]
        self.assertEqual(bundle["resourceType"], "Bundle")
        self.assertEqual(bundle["type"], "document")
        self.assertGreater(len(bundle["entry"]), 0)


if __name__ == "__main__":
    unittest.main()
