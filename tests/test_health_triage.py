"""
Unit Tests for Module 1 Health-Status Triage & Dynamic Queue Position Adjustment
"""

import unittest
from datetime import datetime, timedelta
from src.features.health_triage_engine import (
    DynamicHealthAcuityTriageEngine,
    PatientVitals,
    TriageAssessment,
)


class TestHealthTriageEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DynamicHealthAcuityTriageEngine()

    def test_emergency_cardiac_presentation(self):
        """Acute chest pain with severe vitals must be flagged Level 1 or 2."""
        vitals = PatientVitals(
            systolic_bp=188.0,
            heart_rate=122.0,
            spo2=91.0,
            resp_rate=26.0,
            pain_score=9.0,
        )
        assessment = self.engine.assess_patient(
            patient_id="EMERG-001",
            patient_name="Devendra Sharma",
            chief_complaint="Crushing substernal chest pain radiating to left jaw with diaphoresis",
            vitals=vitals,
            comorbidities=["Coronary Artery Disease"],
        )
        self.assertIn(assessment.esi_level, [1, 2])
        self.assertGreater(assessment.health_acuity_index, 0.65)
        self.assertTrue(any("Acute Coronary" in rf for rf in assessment.red_flags))
        self.assertTrue(any("Hypoxemia" in rf for rf in assessment.red_flags))

    def test_routine_followup_presentation(self):
        """Routine medication refill with normal vitals must be Level 5."""
        vitals = PatientVitals(
            systolic_bp=122.0,
            heart_rate=70.0,
            spo2=99.0,
            resp_rate=16.0,
            pain_score=0.0,
        )
        assessment = self.engine.assess_patient(
            patient_id="ROUTINE-001",
            patient_name="Pooja Patel",
            chief_complaint="Routine prescription refill for blood pressure, feeling fine",
            vitals=vitals,
            comorbidities=["Hypertension"],
        )
        self.assertIn(assessment.esi_level, [4, 5])
        self.assertLess(assessment.health_acuity_index, 0.35)
        self.assertEqual(len(assessment.red_flags), 0)

    def test_queue_position_adjustment_advances_urgent_patient(self):
        """Acute patient arriving later must advance ahead of non-urgent patient in queue."""
        now = datetime.now()
        routine_patient = {
            "entry_id": 1,
            "patient_name": "Routine Patient",
            "check_in_time": now - timedelta(minutes=15),
            "health_acuity_index": 0.08,
            "esi_acuity": 5,
            "pred_service_dur": 10.0,
        }
        acute_patient = {
            "entry_id": 2,
            "patient_name": "Acute Patient",
            "check_in_time": now - timedelta(minutes=2),
            "health_acuity_index": 0.78,
            "esi_acuity": 2,
            "pred_service_dur": 22.0,
        }

        adjustments = self.engine.adjust_queue_positions(
            waiting_patients=[routine_patient, acute_patient],
            current_time=now,
        )

        self.assertEqual(len(adjustments), 2)
        # Acute patient should be ranked #1
        self.assertEqual(adjustments[0].entry_id, 2)
        self.assertEqual(adjustments[0].adjusted_position, 1)
        self.assertEqual(adjustments[0].position_shift, 1)  # Advanced +1 spot

        # Routine patient should be ranked #2
        self.assertEqual(adjustments[1].entry_id, 1)
        self.assertEqual(adjustments[1].adjusted_position, 2)
        self.assertEqual(adjustments[1].position_shift, -1) # Yielded spot

    def test_anti_starvation_aging_prevents_permanent_delay(self):
        """A routine patient waiting a very long time should have their priority elevated."""
        now = datetime.now()
        # Patient waiting for 150 minutes (way beyond target wait)
        starving_routine = {
            "entry_id": 10,
            "patient_name": "Long Waiting Patient",
            "check_in_time": now - timedelta(minutes=150),
            "health_acuity_index": 0.15,
            "esi_acuity": 4,
            "pred_service_dur": 10.0,
        }
        # Mildly acute patient who just arrived 30 seconds ago
        new_mild_patient = {
            "entry_id": 20,
            "patient_name": "Fresh Mild Patient",
            "check_in_time": now - timedelta(seconds=30),
            "health_acuity_index": 0.35,
            "esi_acuity": 4,
            "pred_service_dur": 12.0,
        }

        adjustments = self.engine.adjust_queue_positions(
            waiting_patients=[starving_routine, new_mild_patient],
            current_time=now,
        )
        # The starving routine patient should be served first due to aging!
        self.assertEqual(adjustments[0].entry_id, 10)


if __name__ == "__main__":
    unittest.main()
