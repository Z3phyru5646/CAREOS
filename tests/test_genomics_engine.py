import json
import unittest

import numpy as np
import pandas as pd

from genomics_engine import GenomicsTreatmentEngine, TriggerRule


class GenomicsTreatmentEngineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.index = pd.Index([f"patient-{i}" for i in range(48)])
        self.snps = pd.DataFrame(
            rng.integers(0, 3, size=(48, 8)),
            index=self.index,
            columns=[f"rs{i}" for i in range(8)],
        )
        self.clinical = pd.DataFrame(
            {
                "age": rng.integers(18, 85, size=48),
                "sex": rng.choice(["F", "M"], size=48),
                "weight_kg": rng.uniform(45, 120, size=48),
                "creatinine_clearance": rng.uniform(30, 130, size=48),
                "alt": rng.uniform(5, 80, size=48),
            },
            index=self.index,
        )
        self.expression = pd.DataFrame(
            rng.normal(size=(48, 4)),
            index=self.index,
            columns=[f"gene_{i}" for i in range(4)],
        )
        self.labels = np.array(
            ["responder", "non_responder", "adverse_event_risk"] * 16
        )
        self.doses = rng.uniform(1, 8, size=48)
        self.rule = TriggerRule(
            diagnosis_codes=frozenset({"I48.0"}),
            medication_names=frozenset({"example medication"}),
            gene_name="GENE",
            variant_id="rs0",
            genotype_to_metabolizer={
                0: "normal",
                1: "intermediate",
                2: "poor",
            },
            dose_unit="mg/day",
            cpic_guideline_source="Verified guideline",
            cpic_monitoring_advice="Follow approved monitoring guidance.",
        )
        self.engine = GenomicsTreatmentEngine(
            target_medication="example medication",
            trigger_rules=[self.rule],
            snp_components=3,
        )
        self.features = self.engine.fit_features(
            self.snps, self.clinical, self.expression
        )
        self.engine.fit_classifier(self.features, self.labels)
        self.engine.fit_dose_regressor(self.features, self.doses)

    def test_fusion_classifier_regressor_and_alert(self):
        self.assertEqual(self.features.shape, (48, 3 + 4 + 2 + 4))
        self.assertGreaterEqual(
            self.engine.evaluate_classifier(self.features, self.labels)["auc_roc"],
            0.0,
        )
        metrics = self.engine.evaluate_dose_regressor(self.features, self.doses)
        self.assertIn("mse", metrics)
        self.assertIn("r_squared", metrics)

        alert = self.engine.create_alert(
            ["i48.0"],
            ["Example   Medication"],
            self.snps.iloc[[0]],
            self.clinical.iloc[[0]],
            self.expression.iloc[[0]],
        )
        self.assertIsNotNone(alert)
        assert alert is not None
        self.assertEqual(alert["alert_type"], "pharmacogenomic_alert")
        self.assertEqual(alert["detected_gene_variant"]["copies_of_variant"], int(self.snps.iloc[0]["rs0"]))
        self.assertIn("not an automatic prescription", alert["soap_note_suggestion"])
        self.assertEqual(json.loads(self.engine.alert_to_json(alert)), alert)

    def test_unregistered_trigger_does_not_generate_alert(self):
        alert = self.engine.create_alert(
            ["Z99.9"],
            ["unconfigured medication"],
            self.snps.iloc[[0]],
            self.clinical.iloc[[0]],
            self.expression.iloc[[0]],
        )
        self.assertIsNone(alert)

    def test_rejects_invalid_snp_dosage(self):
        invalid = self.snps.copy()
        invalid.iloc[0, 0] = 3
        with self.assertRaisesRegex(ValueError, "only finite 0, 1, or 2"):
            self.engine.transform_features(
                invalid, self.clinical, self.expression
            )


if __name__ == "__main__":
    unittest.main()
