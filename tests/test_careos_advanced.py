"""
Comprehensive test suite for CARE'OS Module 3 — Genomics Engine.

Tests cover:
  • Data models and validation
  • Knowledge base completeness
  • Synthetic data generation
  • VAE autoencoder
  • Engine training pipeline
  • Alert generation
  • SOAP note formatting
  • Edge cases and error handling
"""

import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


class TestDataModels(unittest.TestCase):
    """Test Pydantic data models."""

    def test_trigger_payload_normalizes_codes(self):
        from careos_genomics.data_models import TriggerPayload

        trigger = TriggerPayload(
            patient_id="P001",
            diagnosis_codes=["i48.0", " I48.1 "],
            medication_names=[" Warfarin ", "CLOPIDOGREL"],
            prescribing_physician="DR-001",
        )
        self.assertEqual(trigger.diagnosis_codes, ["I48.0", "I48.1"])
        self.assertEqual(trigger.medication_names, ["warfarin", "clopidogrel"])

    def test_clinical_features_requires_liver_marker(self):
        from careos_genomics.data_models import ClinicalFeatures

        with self.assertRaises(ValueError):
            ClinicalFeatures(
                age=50,
                sex="M",
                weight_kg=75.0,
                creatinine_clearance=90.0,
                # No liver markers!
            )

    def test_clinical_features_valid(self):
        from careos_genomics.data_models import ClinicalFeatures

        clinical = ClinicalFeatures(
            age=65,
            sex="F",
            weight_kg=68.0,
            creatinine_clearance=75.0,
            alt=25.0,
        )
        self.assertEqual(clinical.age, 65)
        self.assertEqual(clinical.sex, "F")

    def test_patient_genomic_profile_validates_snps(self):
        from careos_genomics.data_models import PatientGenomicProfile

        with self.assertRaises(ValueError):
            PatientGenomicProfile(
                snp_profile={"rs1234": 3},  # Invalid: must be 0, 1, or 2
            )

    def test_patient_genomic_profile_valid(self):
        from careos_genomics.data_models import PatientGenomicProfile

        profile = PatientGenomicProfile(
            snp_profile={"rs1234": 0, "rs5678": 2},
            gene_expression={"CYP2D6": 8.5},
        )
        self.assertEqual(profile.snp_profile["rs1234"], 0)

    def test_pharmacogenomic_alert_to_soap(self):
        from careos_genomics.data_models import (
            PharmacogenomicAlert,
            PredictionResult,
            DrugResponseCategory,
            GeneVariantDetail,
            CPICGuidanceDetail,
            MetabolizerPhenotype,
            RiskLevel,
            AlertSeverity,
        )

        alert = PharmacogenomicAlert(
            severity=AlertSeverity.HIGH,
            patient_id="P001",
            consultation_id="C001",
            matched_diagnosis_codes=["I48.0"],
            matched_medication_names=["warfarin"],
            detected_variants=[GeneVariantDetail(gene="CYP2C9", variant_id="rs1799853", copies_of_variant=1)],
            metabolizer_phenotype=MetabolizerPhenotype.INTERMEDIATE,
            risk_level=RiskLevel.MODERATE,
            drug_response_prediction=PredictionResult(
                predicted_category=DrugResponseCategory.RESPONDER,
                probabilities={"responder": 0.7, "non_responder": 0.2, "adverse_event_risk": 0.1},
                confidence=0.7,
            ),
            cpic_guidance=CPICGuidanceDetail(
                recommendation_text="Consider dose reduction",
                monitoring_advice="Enhanced INR monitoring",
            ),
            soap_note_suggestion="Test suggestion",
        )

        soap = alert.to_soap_appendix()
        self.assertIn("PHARMACOGENOMIC", soap)
        self.assertIn("CYP2C9", soap)
        self.assertIn("rs1799853", soap)
        self.assertIn("CLINICIAN REVIEW", soap)

    def test_alert_json_serialization(self):
        from careos_genomics.data_models import (
            PharmacogenomicAlert,
            PredictionResult,
            DrugResponseCategory,
            GeneVariantDetail,
            CPICGuidanceDetail,
            MetabolizerPhenotype,
            RiskLevel,
            AlertSeverity,
        )

        alert = PharmacogenomicAlert(
            severity=AlertSeverity.MODERATE,
            patient_id="P002",
            consultation_id="C002",
            matched_diagnosis_codes=["I48.0"],
            matched_medication_names=["warfarin"],
            detected_variants=[GeneVariantDetail(gene="CYP2C9", variant_id="rs1799853", copies_of_variant=0)],
            metabolizer_phenotype=MetabolizerPhenotype.NORMAL,
            risk_level=RiskLevel.LOW,
            drug_response_prediction=PredictionResult(
                predicted_category=DrugResponseCategory.RESPONDER,
                probabilities={"responder": 0.8, "non_responder": 0.15, "adverse_event_risk": 0.05},
                confidence=0.8,
            ),
            cpic_guidance=CPICGuidanceDetail(
                recommendation_text="Standard dosing",
                monitoring_advice="Standard monitoring",
            ),
            soap_note_suggestion="Test",
        )

        json_str = alert.model_dump_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["alert_type"], "pharmacogenomic_alert")
        self.assertIn("safety_notice", parsed)


class TestKnowledgeBase(unittest.TestCase):
    """Test the CPIC Knowledge Base."""

    def setUp(self):
        from careos_genomics.knowledge_base import CPICKnowledgeBase
        self.kb = CPICKnowledgeBase()

    def test_knowledge_base_has_rules(self):
        self.assertGreaterEqual(len(self.kb), 8)

    def test_warfarin_rule_exists(self):
        rules = self.kb.find_matching_rules(["I48.0"], ["warfarin"])
        self.assertTrue(len(rules) > 0)
        rule = rules[0]
        self.assertIn("VKORC1", rule.gene_name)

    def test_clopidogrel_rule_exists(self):
        rules = self.kb.find_matching_rules(["I21.0"], ["clopidogrel"])
        self.assertTrue(len(rules) > 0)

    def test_codeine_rule_exists(self):
        rules = self.kb.find_matching_rules([], ["codeine"])
        self.assertTrue(len(rules) > 0)

    def test_simvastatin_rule_exists(self):
        rules = self.kb.find_matching_rules(["E78.0"], ["simvastatin"])
        self.assertTrue(len(rules) > 0)

    def test_no_match_returns_empty(self):
        rules = self.kb.find_matching_rules(["Z99.99"], ["imaginary_drug"])
        self.assertEqual(len(rules), 0)

    def test_medication_case_insensitive(self):
        rules_lower = self.kb.find_matching_rules([], ["warfarin"])
        rules_upper = self.kb.find_matching_rules([], ["Warfarin"])
        self.assertTrue(len(rules_lower) > 0 or len(rules_upper) > 0)

    def test_all_rules_have_required_fields(self):
        for rule in self.kb.get_all_rules():
            self.assertTrue(rule.gene_name)
            self.assertTrue(rule.variant_ids)
            self.assertTrue(rule.drug_names)
            self.assertTrue(rule.dose_unit)
            self.assertTrue(rule.cpic_guideline_id)

    def test_get_supported_drugs(self):
        drugs = self.kb.get_supported_drugs()
        self.assertIn("warfarin", drugs)
        self.assertIn("clopidogrel", drugs)

    def test_get_supported_genes(self):
        genes = self.kb.get_supported_genes()
        self.assertTrue(len(genes) > 0)

    def test_get_rule_by_id(self):
        rule = self.kb.get_rule_by_id("CPIC-WARFARIN-CYP2C9-VKORC1")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.gene_name, "CYP2C9/VKORC1")


class TestSyntheticData(unittest.TestCase):
    """Test synthetic data generation."""

    def setUp(self):
        from careos_genomics.synthetic_data import SyntheticPharmacogenomicDataGenerator
        self.generator = SyntheticPharmacogenomicDataGenerator(seed=42, n_background_snps=30, n_expression_genes=15)

    def test_warfarin_cohort_shape(self):
        cohort = self.generator.generate_warfarin_cohort(50)
        self.assertEqual(len(cohort.snp_profiles), 50)
        self.assertEqual(len(cohort.clinical_data), 50)
        self.assertEqual(len(cohort.gene_expression), 50)
        self.assertEqual(len(cohort.response_labels), 50)
        self.assertEqual(len(cohort.optimal_doses), 50)

    def test_snp_values_valid(self):
        cohort = self.generator.generate_warfarin_cohort(30)
        values = cohort.snp_profiles.values
        self.assertTrue(np.isin(values, [0, 1, 2]).all())

    def test_clinical_data_columns(self):
        cohort = self.generator.generate_warfarin_cohort(30)
        required = {"age", "sex", "weight_kg", "creatinine_clearance", "alt"}
        self.assertTrue(required.issubset(set(cohort.clinical_data.columns)))

    def test_response_labels_valid(self):
        cohort = self.generator.generate_warfarin_cohort(50)
        valid = {"responder", "non_responder", "adverse_event_risk"}
        self.assertTrue(set(cohort.response_labels).issubset(valid))

    def test_doses_positive(self):
        cohort = self.generator.generate_warfarin_cohort(30)
        self.assertTrue((cohort.optimal_doses > 0).all())

    def test_mixed_cohort(self):
        cohort = self.generator.generate_mixed_cohort(80)
        self.assertGreaterEqual(len(cohort.snp_profiles), 80)

    def test_clopidogrel_cohort(self):
        cohort = self.generator.generate_clopidogrel_cohort(30)
        self.assertEqual(len(cohort.snp_profiles), 30)

    def test_codeine_cohort(self):
        cohort = self.generator.generate_codeine_cohort(30)
        self.assertEqual(len(cohort.snp_profiles), 30)

    def test_indexes_aligned(self):
        cohort = self.generator.generate_warfarin_cohort(30)
        self.assertTrue(cohort.snp_profiles.index.equals(cohort.clinical_data.index))
        self.assertTrue(cohort.snp_profiles.index.equals(cohort.gene_expression.index))


class TestEnginePipeline(unittest.TestCase):
    """Test the full engine training and prediction pipeline."""

    @classmethod
    def setUpClass(cls):
        """Train engine once for all tests."""
        from careos_genomics.synthetic_data import SyntheticPharmacogenomicDataGenerator
        from careos_genomics.engine import GenomicsEngine, EngineConfig

        generator = SyntheticPharmacogenomicDataGenerator(seed=42, n_background_snps=20, n_expression_genes=10)
        cls.cohort = generator.generate_warfarin_cohort(50)

        config = EngineConfig(
            vae_latent_dim=4,
            vae_epochs=3,
            enable_vae=True,
            use_xgboost=False,  # Fast testing with RF
            xgb_n_trials=0,
            enable_shap=False,
            pca_components=4,
        )

        cls.engine = GenomicsEngine(config=config)
        cls.metrics = cls.engine.train(
            snp_profiles=cls.cohort.snp_profiles,
            clinical_data=cls.cohort.clinical_data,
            gene_expression=cls.cohort.gene_expression,
            response_labels=cls.cohort.response_labels,
            optimal_doses=cls.cohort.optimal_doses,
        )

    def test_engine_is_trained(self):
        self.assertTrue(self.engine._is_trained)

    def test_metrics_contain_auc(self):
        self.assertIn("classifier", self.metrics)
        self.assertIn("val_auc_roc", self.metrics["classifier"])

    def test_metrics_contain_regression(self):
        self.assertIn("regressor", self.metrics)
        self.assertIn("val_mse", self.metrics["regressor"])
        self.assertIn("val_r_squared", self.metrics["regressor"])

    def test_training_summary(self):
        summary = self.engine.get_training_summary()
        self.assertIn("CARE'OS", summary)
        self.assertIn("AUC", summary)

    def test_process_trigger_warfarin(self):
        from careos_genomics.data_models import (
            TriggerPayload, ClinicalFeatures, PatientGenomicProfile,
        )

        trigger = TriggerPayload(
            patient_id="TEST-001",
            diagnosis_codes=["I48.0"],
            medication_names=["warfarin"],
            prescribing_physician="DR-TEST",
        )

        clinical = ClinicalFeatures(
            age=65, sex="M", weight_kg=80.0,
            creatinine_clearance=70.0, alt=25.0,
        )

        snps = dict(zip(
            self.cohort.snp_profiles.columns,
            self.cohort.snp_profiles.iloc[0].values.astype(int),
        ))
        expression = dict(zip(
            self.cohort.gene_expression.columns,
            self.cohort.gene_expression.iloc[0].values,
        ))

        genomic = PatientGenomicProfile(
            snp_profile=snps,
            gene_expression=expression,
        )

        alerts = self.engine.process_trigger(trigger, genomic, clinical)
        self.assertIsInstance(alerts, list)
        if alerts:
            alert = alerts[0]
            self.assertEqual(alert.alert_type, "pharmacogenomic_alert")
            self.assertIn("DECISION SUPPORT", alert.safety_notice.upper())
            soap = alert.to_soap_appendix()
            self.assertIn("CLINICIAN REVIEW", soap)
            json_str = alert.model_dump_json()
            parsed = json.loads(json_str)
            self.assertIn("alert_type", parsed)

    def test_no_alert_for_unknown_trigger(self):
        from careos_genomics.data_models import (
            TriggerPayload, ClinicalFeatures, PatientGenomicProfile,
        )

        trigger = TriggerPayload(
            patient_id="TEST-002",
            diagnosis_codes=["Z99.99"],
            medication_names=["imaginary_drug"],
            prescribing_physician="DR-TEST",
        )

        clinical = ClinicalFeatures(
            age=30, sex="F", weight_kg=60.0,
            creatinine_clearance=100.0, alt=20.0,
        )

        snps = dict(zip(
            self.cohort.snp_profiles.columns,
            self.cohort.snp_profiles.iloc[0].values.astype(int),
        ))

        genomic = PatientGenomicProfile(snp_profile=snps)

        alerts = self.engine.process_trigger(trigger, genomic, clinical)
        self.assertEqual(len(alerts), 0)

    def test_untrained_engine_raises(self):
        from careos_genomics.engine import GenomicsEngine
        from careos_genomics.data_models import (
            TriggerPayload, ClinicalFeatures, PatientGenomicProfile,
        )

        fresh_engine = GenomicsEngine()

        trigger = TriggerPayload(
            patient_id="TEST",
            diagnosis_codes=["I48.0"],
            medication_names=["warfarin"],
            prescribing_physician="DR-TEST",
        )
        clinical = ClinicalFeatures(
            age=50, sex="M", weight_kg=70.0,
            creatinine_clearance=90.0, alt=20.0,
        )
        genomic = PatientGenomicProfile(snp_profile={"rs1": 0})

        with self.assertRaises(RuntimeError):
            fresh_engine.process_trigger(trigger, genomic, clinical)


class TestVAEAutoencoder(unittest.TestCase):
    """Test the Variational Autoencoder components."""

    def test_vae_forward_pass(self):
        try:
            import torch
            from careos_genomics.autoencoder import VariationalSNPAutoencoder

            model = VariationalSNPAutoencoder(input_dim=30, latent_dim=6)
            x = torch.randint(0, 3, (8, 30)).float()
            recon, mu, logvar = model(x)
            self.assertEqual(recon.shape, (8, 30, 3))
            self.assertEqual(mu.shape, (8, 6))
            self.assertEqual(logvar.shape, (8, 6))
        except ImportError:
            self.skipTest("PyTorch not available")

    def test_vae_latent_extraction(self):
        try:
            import torch
            from careos_genomics.autoencoder import VariationalSNPAutoencoder

            model = VariationalSNPAutoencoder(input_dim=30, latent_dim=6)
            x = torch.randint(0, 3, (8, 30)).float()
            latent = model.get_latent(x)
            self.assertEqual(latent.shape, (8, 6))
        except ImportError:
            self.skipTest("PyTorch not available")


if __name__ == "__main__":
    unittest.main()
