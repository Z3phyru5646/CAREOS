"""
Synthetic Pharmacogenomic Data Generator for CARE'OS Module 3.

Generates realistic synthetic patient data for training, validation, and
demonstration of the genomics engine. The generator creates biologically
plausible correlations between:
  • SNP genotypes and metabolizer phenotypes
  • Metabolizer status and drug response categories
  • Genotype, clinical features, and optimal dosing
  • Gene expression levels correlated with SNP profiles

This data is for DEVELOPMENT AND TESTING ONLY. Never train a production
clinical model on synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Allele Frequency Distributions (based on published population genetics)
# ═══════════════════════════════════════════════════════════════════════════

# Representative minor allele frequencies for pharmacogenomic variants
PHARMACOGENOMIC_VARIANTS = {
    # CYP2C9 variants (warfarin metabolism)
    "rs1799853": {"maf": 0.12, "gene": "CYP2C9", "effect": "reduced_function"},    # *2
    "rs1057910": {"maf": 0.07, "gene": "CYP2C9", "effect": "reduced_function"},    # *3
    # VKORC1 (warfarin target sensitivity)
    "rs9923231": {"maf": 0.38, "gene": "VKORC1", "effect": "increased_sensitivity"},
    # CYP2C19 variants (clopidogrel activation)
    "rs4244285": {"maf": 0.15, "gene": "CYP2C19", "effect": "loss_of_function"},   # *2
    "rs4986893": {"maf": 0.03, "gene": "CYP2C19", "effect": "loss_of_function"},   # *3
    "rs12248560": {"maf": 0.20, "gene": "CYP2C19", "effect": "gain_of_function"},  # *17
    # CYP2D6 variants (codeine, SSRIs)
    "rs3892097": {"maf": 0.20, "gene": "CYP2D6", "effect": "loss_of_function"},    # *4
    "rs5030655": {"maf": 0.02, "gene": "CYP2D6", "effect": "loss_of_function"},    # *6
    "rs1065852": {"maf": 0.18, "gene": "CYP2D6", "effect": "reduced_function"},    # *10
    "rs16947":   {"maf": 0.30, "gene": "CYP2D6", "effect": "reduced_function"},    # *2
    # SLCO1B1 (simvastatin transport)
    "rs4149056": {"maf": 0.15, "gene": "SLCO1B1", "effect": "reduced_transport"},
    # DPYD (fluoropyrimidine metabolism)
    "rs3918290":  {"maf": 0.01, "gene": "DPYD", "effect": "loss_of_function"},
    "rs55886062": {"maf": 0.005, "gene": "DPYD", "effect": "loss_of_function"},
    "rs67376798": {"maf": 0.01, "gene": "DPYD", "effect": "reduced_function"},
    "rs75017182": {"maf": 0.04, "gene": "DPYD", "effect": "reduced_function"},
    # TPMT (thiopurine metabolism)
    "rs1800462": {"maf": 0.01, "gene": "TPMT", "effect": "loss_of_function"},
    "rs1800460": {"maf": 0.05, "gene": "TPMT", "effect": "loss_of_function"},
    "rs1142345": {"maf": 0.04, "gene": "TPMT", "effect": "loss_of_function"},
    # NUDT15 (thiopurine metabolism)
    "rs116855232": {"maf": 0.03, "gene": "NUDT15", "effect": "loss_of_function"},
    # CYP3A5 (tacrolimus)
    "rs776746": {"maf": 0.30, "gene": "CYP3A5", "effect": "gain_of_function"},
    # Additional background SNPs for dimensionality
    "rs1045642": {"maf": 0.45, "gene": "ABCB1", "effect": "altered_transport"},
    "rs2032582": {"maf": 0.40, "gene": "ABCB1", "effect": "altered_transport"},
    "rs1128503": {"maf": 0.42, "gene": "ABCB1", "effect": "altered_transport"},
    "rs2231142": {"maf": 0.10, "gene": "ABCG2", "effect": "reduced_transport"},
    "rs717620":  {"maf": 0.20, "gene": "ABCC2", "effect": "altered_transport"},
    "rs4148323": {"maf": 0.08, "gene": "UGT1A1", "effect": "reduced_function"},
    "rs8175347": {"maf": 0.35, "gene": "UGT1A1", "effect": "reduced_function"},
    "rs1801133": {"maf": 0.30, "gene": "MTHFR", "effect": "reduced_function"},
    "rs1801131": {"maf": 0.25, "gene": "MTHFR", "effect": "reduced_function"},
    "rs1799945": {"maf": 0.12, "gene": "HFE", "effect": "iron_overload_risk"},
}


@dataclass
class SyntheticCohort:
    """Container for a generated synthetic patient cohort."""
    snp_profiles: pd.DataFrame
    clinical_data: pd.DataFrame
    gene_expression: pd.DataFrame
    response_labels: np.ndarray
    optimal_doses: np.ndarray
    metabolizer_phenotypes: np.ndarray
    patient_metadata: pd.DataFrame


class SyntheticPharmacogenomicDataGenerator:
    """
    Generates biologically plausible synthetic pharmacogenomic data.

    The generator creates realistic correlations between genotype, phenotype,
    and drug response to enable meaningful model training and evaluation.

    Usage:
        generator = SyntheticPharmacogenomicDataGenerator(seed=42)
        cohort = generator.generate_warfarin_cohort(n_patients=500)
    """

    def __init__(
        self,
        seed: int = 42,
        n_background_snps: int = 200,
        n_expression_genes: int = 50,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.n_background_snps = n_background_snps
        self.n_expression_genes = n_expression_genes

    def generate_warfarin_cohort(self, n_patients: int = 500) -> SyntheticCohort:
        """Generate a synthetic warfarin pharmacogenomics cohort.

        Creates patients with realistic CYP2C9/VKORC1 genotypes correlated
        with warfarin dose requirements and response categories.
        """
        return self._generate_cohort(
            n_patients=n_patients,
            drug_name="warfarin",
            key_variants=["rs1799853", "rs1057910", "rs9923231"],
            dose_range=(0.5, 15.0),
            dose_formula=self._warfarin_dose_formula,
        )

    def generate_clopidogrel_cohort(self, n_patients: int = 500) -> SyntheticCohort:
        """Generate a synthetic clopidogrel pharmacogenomics cohort."""
        return self._generate_cohort(
            n_patients=n_patients,
            drug_name="clopidogrel",
            key_variants=["rs4244285", "rs4986893", "rs12248560"],
            dose_range=(75.0, 75.0),
            dose_formula=self._clopidogrel_dose_formula,
        )

    def generate_codeine_cohort(self, n_patients: int = 500) -> SyntheticCohort:
        """Generate a synthetic codeine pharmacogenomics cohort."""
        return self._generate_cohort(
            n_patients=n_patients,
            drug_name="codeine",
            key_variants=["rs3892097", "rs5030655", "rs1065852", "rs16947"],
            dose_range=(15.0, 60.0),
            dose_formula=self._codeine_dose_formula,
        )

    def generate_simvastatin_cohort(self, n_patients: int = 500) -> SyntheticCohort:
        """Generate a synthetic simvastatin pharmacogenomics cohort."""
        return self._generate_cohort(
            n_patients=n_patients,
            drug_name="simvastatin",
            key_variants=["rs4149056"],
            dose_range=(5.0, 40.0),
            dose_formula=self._simvastatin_dose_formula,
        )

    def generate_mixed_cohort(self, n_patients: int = 1000) -> SyntheticCohort:
        """Generate a large mixed cohort spanning all supported drug-gene pairs."""
        per_drug = max(n_patients // 4, 50)
        cohorts = [
            self.generate_warfarin_cohort(per_drug),
            self.generate_clopidogrel_cohort(per_drug),
            self.generate_codeine_cohort(per_drug),
            self.generate_simvastatin_cohort(per_drug),
        ]
        return self._merge_cohorts(cohorts)

    # ─── Internal Generation ────────────────────────────────────────────

    def _generate_cohort(
        self,
        n_patients: int,
        drug_name: str,
        key_variants: list[str],
        dose_range: tuple[float, float],
        dose_formula: callable,
    ) -> SyntheticCohort:
        patient_ids = [f"PAT-{drug_name[:3].upper()}-{i:05d}" for i in range(n_patients)]
        index = pd.Index(patient_ids, name="patient_id")

        # Generate SNP profiles
        snp_profiles = self._generate_snp_profiles(n_patients, index)

        # Generate clinical data
        clinical_data = self._generate_clinical_data(n_patients, index)

        # Generate gene expression (correlated with SNPs)
        gene_expression = self._generate_gene_expression(n_patients, snp_profiles, index)

        # Generate metabolizer phenotypes based on key variant genotypes
        metabolizer_scores = np.zeros(n_patients)
        for variant in key_variants:
            if variant in snp_profiles.columns:
                info = PHARMACOGENOMIC_VARIANTS.get(variant, {})
                effect = info.get("effect", "neutral")
                if effect in ("loss_of_function", "reduced_function"):
                    metabolizer_scores += snp_profiles[variant].values
                elif effect == "gain_of_function":
                    metabolizer_scores -= snp_profiles[variant].values * 0.5
                elif effect == "increased_sensitivity":
                    metabolizer_scores += snp_profiles[variant].values * 0.7

        metabolizer_phenotypes = np.array([
            self._score_to_phenotype(s) for s in metabolizer_scores
        ])

        # Generate response labels
        response_labels = self._generate_response_labels(
            metabolizer_phenotypes, clinical_data, n_patients
        )

        # Generate optimal doses
        optimal_doses = dose_formula(
            metabolizer_scores, clinical_data, snp_profiles, n_patients, dose_range
        )

        # Patient metadata
        metadata = pd.DataFrame({
            "drug": drug_name,
            "metabolizer_score": metabolizer_scores,
            "metabolizer_phenotype": metabolizer_phenotypes,
        }, index=index)

        return SyntheticCohort(
            snp_profiles=snp_profiles,
            clinical_data=clinical_data,
            gene_expression=gene_expression,
            response_labels=response_labels,
            optimal_doses=optimal_doses,
            metabolizer_phenotypes=metabolizer_phenotypes,
            patient_metadata=metadata,
        )

    def _generate_snp_profiles(self, n: int, index: pd.Index) -> pd.DataFrame:
        """Generate SNP genotype profiles using Hardy-Weinberg equilibrium."""
        all_variants = {}

        # Known pharmacogenomic variants
        for rsid, info in PHARMACOGENOMIC_VARIANTS.items():
            maf = info["maf"]
            p = 1 - maf
            q = maf
            # HWE genotype frequencies: p², 2pq, q²
            probs = [p**2, 2*p*q, q**2]
            all_variants[rsid] = self.rng.choice([0, 1, 2], size=n, p=probs)

        # Background SNPs with random MAFs
        for i in range(self.n_background_snps):
            maf = self.rng.uniform(0.01, 0.45)
            p = 1 - maf
            probs = [p**2, 2*p*(1-p), (1-p)**2]
            all_variants[f"rs_bg_{i:04d}"] = self.rng.choice([0, 1, 2], size=n, p=probs)

        return pd.DataFrame(all_variants, index=index)

    def _generate_clinical_data(self, n: int, index: pd.Index) -> pd.DataFrame:
        """Generate realistic clinical measurements."""
        ages = self.rng.integers(18, 90, size=n)
        sexes = self.rng.choice(["M", "F"], size=n)
        heights = np.where(
            sexes == "M",
            self.rng.normal(175, 8, n),
            self.rng.normal(162, 7, n),
        ).clip(140, 210)
        bmis = self.rng.normal(26, 5, n).clip(16, 50)
        weights = (bmis * (heights / 100) ** 2).clip(35, 200)

        # Renal function (age-correlated decline)
        base_crcl = self.rng.normal(100, 20, n)
        age_effect = np.maximum(0, (ages - 40) * 0.5)
        creatinine_clearance = (base_crcl - age_effect).clip(15, 150)

        egfr = creatinine_clearance * self.rng.normal(1.0, 0.1, n)
        egfr = egfr.clip(10, 150)

        # Liver function
        alt = self.rng.lognormal(3.0, 0.5, n).clip(5, 200)
        ast = (alt * self.rng.normal(0.85, 0.15, n)).clip(5, 200)
        bilirubin = self.rng.lognormal(-0.5, 0.5, n).clip(0.1, 5.0)
        albumin = self.rng.normal(4.0, 0.4, n).clip(2.0, 5.5)
        inr = self.rng.normal(1.0, 0.15, n).clip(0.8, 2.0)
        alp = self.rng.lognormal(4.2, 0.3, n).clip(30, 300)

        is_smoker = self.rng.choice([True, False], size=n, p=[0.18, 0.82])

        return pd.DataFrame({
            "age": ages,
            "sex": sexes,
            "weight_kg": np.round(weights, 1),
            "height_cm": np.round(heights, 1),
            "creatinine_clearance": np.round(creatinine_clearance, 1),
            "egfr": np.round(egfr, 1),
            "alt": np.round(alt, 1),
            "ast": np.round(ast, 1),
            "bilirubin": np.round(bilirubin, 2),
            "albumin": np.round(albumin, 2),
            "inr": np.round(inr, 2),
            "alp": np.round(alp, 1),
            "is_smoker": is_smoker,
        }, index=index)

    def _generate_gene_expression(
        self, n: int, snp_profiles: pd.DataFrame, index: pd.Index
    ) -> pd.DataFrame:
        """Generate gene expression data with SNP-correlated variation."""
        expression_data = {}

        # Pharmacogenes with SNP-correlated expression
        pharma_genes = [
            "CYP2D6", "CYP2C9", "CYP2C19", "CYP3A4", "CYP3A5",
            "VKORC1", "DPYD", "TPMT", "NUDT15", "SLCO1B1",
            "ABCB1", "UGT1A1", "CYP1A2", "NAT2", "GSTT1",
        ]

        gene_to_snps = {}
        for rsid, info in PHARMACOGENOMIC_VARIANTS.items():
            gene = info["gene"]
            gene_to_snps.setdefault(gene, []).append(rsid)

        for gene in pharma_genes:
            base_expr = self.rng.normal(8.0, 1.5, n)
            relevant_snps = gene_to_snps.get(gene, [])
            for rsid in relevant_snps:
                if rsid in snp_profiles.columns:
                    effect_size = self.rng.uniform(-0.8, 0.3)
                    base_expr += snp_profiles[rsid].values * effect_size
            base_expr += self.rng.normal(0, 0.3, n)
            expression_data[gene] = np.round(base_expr, 3)

        remaining = max(0, self.n_expression_genes - len(pharma_genes))
        for i in range(remaining):
            expression_data[f"GENE_{i:03d}"] = np.round(
                self.rng.normal(7.5, 2.0, n), 3
            )

        return pd.DataFrame(expression_data, index=index)

    def _generate_response_labels(
        self, metabolizer_phenotypes: np.ndarray, clinical: pd.DataFrame, n: int
    ) -> np.ndarray:
        """Generate drug response labels correlated with metabolizer status."""
        labels = []
        for i in range(n):
            pheno = metabolizer_phenotypes[i]
            age = clinical.iloc[i]["age"]

            if pheno == "poor_metabolizer":
                probs = [0.15, 0.35, 0.50]
            elif pheno == "intermediate_metabolizer":
                probs = [0.40, 0.35, 0.25]
            elif pheno == "ultrarapid_metabolizer":
                probs = [0.30, 0.40, 0.30]
            else:
                probs = [0.70, 0.20, 0.10]

            if age > 70:
                probs[2] = min(probs[2] + 0.10, 0.60)
                total = sum(probs)
                probs = [p / total for p in probs]

            label = self.rng.choice(
                ["responder", "non_responder", "adverse_event_risk"],
                p=probs,
            )
            labels.append(label)

        return np.array(labels)

    def _score_to_phenotype(self, score: float) -> str:
        if score >= 3.0:
            return "poor_metabolizer"
        elif score >= 1.5:
            return "intermediate_metabolizer"
        elif score <= -1.0:
            return "ultrarapid_metabolizer"
        else:
            return "normal_metabolizer"

    # ─── Dose Formulas ──────────────────────────────────────────────────

    def _warfarin_dose_formula(
        self, met_scores, clinical, snps, n, dose_range
    ) -> np.ndarray:
        """Warfarin dose based on IWPC algorithm (simplified synthetic version)."""
        base = 5.0
        age_effect = -0.03 * (clinical["age"].values - 50)
        weight_effect = 0.01 * (clinical["weight_kg"].values - 70)
        genotype_effect = -0.8 * met_scores
        noise = self.rng.normal(0, 0.3, n)
        doses = base + age_effect + weight_effect + genotype_effect + noise
        return doses.clip(dose_range[0], dose_range[1])

    def _clopidogrel_dose_formula(
        self, met_scores, clinical, snps, n, dose_range
    ) -> np.ndarray:
        """Clopidogrel is fixed-dose; score represents efficacy."""
        base = 75.0
        return np.full(n, base)

    def _codeine_dose_formula(
        self, met_scores, clinical, snps, n, dose_range
    ) -> np.ndarray:
        base = 30.0
        weight_effect = 0.1 * (clinical["weight_kg"].values - 70)
        genotype_effect = -3.0 * met_scores
        noise = self.rng.normal(0, 2.0, n)
        doses = base + weight_effect + genotype_effect + noise
        return doses.clip(dose_range[0], dose_range[1])

    def _simvastatin_dose_formula(
        self, met_scores, clinical, snps, n, dose_range
    ) -> np.ndarray:
        base = 20.0
        genotype_effect = -5.0 * met_scores
        age_effect = -0.1 * (clinical["age"].values - 50)
        noise = self.rng.normal(0, 2.0, n)
        doses = base + genotype_effect + age_effect + noise
        return doses.clip(dose_range[0], dose_range[1])

    def _merge_cohorts(self, cohorts: list[SyntheticCohort]) -> SyntheticCohort:
        """Merge multiple cohorts into a single dataset."""
        all_snp_cols = set()
        all_clinical_cols = set()
        all_expression_cols = set()
        for c in cohorts:
            all_snp_cols.update(c.snp_profiles.columns)
            all_clinical_cols.update(c.clinical_data.columns)
            all_expression_cols.update(c.gene_expression.columns)

        snps = pd.concat(
            [c.snp_profiles.reindex(columns=sorted(all_snp_cols), fill_value=0) for c in cohorts],
            axis=0,
        )
        clinical = pd.concat(
            [c.clinical_data.reindex(columns=sorted(all_clinical_cols)) for c in cohorts],
            axis=0,
        )
        expression = pd.concat(
            [c.gene_expression.reindex(columns=sorted(all_expression_cols), fill_value=0.0) for c in cohorts],
            axis=0,
        )
        responses = np.concatenate([c.response_labels for c in cohorts])
        doses = np.concatenate([c.optimal_doses for c in cohorts])
        metabolizers = np.concatenate([c.metabolizer_phenotypes for c in cohorts])
        metadata = pd.concat([c.patient_metadata for c in cohorts], axis=0)

        return SyntheticCohort(
            snp_profiles=snps,
            clinical_data=clinical,
            gene_expression=expression,
            response_labels=responses,
            optimal_doses=doses,
            metabolizer_phenotypes=metabolizers,
            patient_metadata=metadata,
        )
