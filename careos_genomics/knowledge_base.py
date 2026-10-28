"""
CPIC Pharmacogenomics Knowledge Base for CARE'OS Module 3.

Contains curated pharmacogenomic rules based on Clinical Pharmacogenetics
Implementation Consortium (CPIC) guidelines. This knowledge base maps
gene-drug pairs to clinical recommendations.

IMPORTANT: This is a reference implementation with representative rules.
In production, this must be populated from the official CPIC database
and kept current with guideline updates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from careos_genomics.data_models import (
    AlertSeverity,
    MetabolizerPhenotype,
    RiskLevel,
)


@dataclass(frozen=True)
class PharmacogenomicRule:
    """A single gene-drug pharmacogenomic interaction rule."""
    rule_id: str
    gene_name: str
    variant_ids: frozenset[str]
    drug_names: frozenset[str]
    diagnosis_codes: frozenset[str]
    star_allele_mappings: dict[str, MetabolizerPhenotype] = field(default_factory=dict)
    genotype_to_metabolizer: dict[int, MetabolizerPhenotype] = field(default_factory=dict)
    metabolizer_to_risk: dict[MetabolizerPhenotype, RiskLevel] = field(default_factory=dict)
    metabolizer_to_severity: dict[MetabolizerPhenotype, AlertSeverity] = field(default_factory=dict)
    dose_unit: str = "mg/day"
    dose_adjustment_factors: dict[MetabolizerPhenotype, float] = field(default_factory=dict)
    standard_dose_range: tuple[float, float] = (0.0, 100.0)
    cpic_guideline_id: str = ""
    cpic_guideline_version: str = ""
    cpic_evidence_level: str = ""
    cpic_recommendation: dict[MetabolizerPhenotype, str] = field(default_factory=dict)
    monitoring_advice: dict[MetabolizerPhenotype, str] = field(default_factory=dict)
    drug_interactions: list[dict[str, str]] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    source_url: str = ""


class CPICKnowledgeBase:
    """
    Curated pharmacogenomic knowledge base with CPIC guidelines.

    Contains rules for major gene-drug pairs including:
    - CYP2D6 + Codeine/Tramadol (pain management)
    - CYP2C19 + Clopidogrel (antiplatelet)
    - CYP2C9/VKORC1 + Warfarin (anticoagulation)
    - CYP2C19 + Omeprazole/Lansoprazole (PPI therapy)
    - HLA-B*57:01 + Abacavir (HIV)
    - DPYD + Fluoropyrimidines (oncology)
    - TPMT/NUDT15 + Thiopurines (immunosuppression)
    - SLCO1B1 + Simvastatin (statin therapy)
    - CYP2D6 + SSRIs (psychiatry)
    - CYP3A5 + Tacrolimus (transplant)
    """

    def __init__(self) -> None:
        self._rules: list[PharmacogenomicRule] = []
        self._drug_index: dict[str, list[PharmacogenomicRule]] = {}
        self._diagnosis_index: dict[str, list[PharmacogenomicRule]] = {}
        self._gene_index: dict[str, list[PharmacogenomicRule]] = {}
        self._load_cpic_rules()

    def _load_cpic_rules(self) -> None:
        """Load the curated CPIC pharmacogenomic rules."""

        # ── CYP2C9/VKORC1 + Warfarin ────────────────────────────────────
        self._add_rule(PharmacogenomicRule(
            rule_id="CPIC-WARFARIN-CYP2C9-VKORC1",
            gene_name="CYP2C9/VKORC1",
            variant_ids=frozenset({"rs1799853", "rs1057910", "rs9923231"}),
            drug_names=frozenset({"warfarin", "coumadin", "jantoven"}),
            diagnosis_codes=frozenset({
                "I48.0", "I48.1", "I48.2", "I48.91",  # Atrial fibrillation
                "I26.9", "I26.99",                      # Pulmonary embolism
                "I82.40", "I82.401", "I82.409",         # Deep vein thrombosis
                "Z79.01",                               # Long-term anticoagulant use
            }),
            genotype_to_metabolizer={
                0: MetabolizerPhenotype.NORMAL,
                1: MetabolizerPhenotype.INTERMEDIATE,
                2: MetabolizerPhenotype.POOR,
            },
            metabolizer_to_risk={
                MetabolizerPhenotype.NORMAL: RiskLevel.LOW,
                MetabolizerPhenotype.INTERMEDIATE: RiskLevel.MODERATE,
                MetabolizerPhenotype.POOR: RiskLevel.HIGH,
                MetabolizerPhenotype.ULTRARAPID: RiskLevel.MODERATE,
            },
            metabolizer_to_severity={
                MetabolizerPhenotype.NORMAL: AlertSeverity.INFORMATIONAL,
                MetabolizerPhenotype.INTERMEDIATE: AlertSeverity.MODERATE,
                MetabolizerPhenotype.POOR: AlertSeverity.HIGH,
                MetabolizerPhenotype.ULTRARAPID: AlertSeverity.MODERATE,
            },
            dose_unit="mg/day",
            dose_adjustment_factors={
                MetabolizerPhenotype.NORMAL: 1.0,
                MetabolizerPhenotype.INTERMEDIATE: 0.7,
                MetabolizerPhenotype.POOR: 0.4,
                MetabolizerPhenotype.ULTRARAPID: 1.3,
            },
            standard_dose_range=(0.5, 15.0),
            cpic_guideline_id="CPIC-2017-warfarin",
            cpic_guideline_version="3.0 (2017 update)",
            cpic_evidence_level="A (strong)",
            cpic_recommendation={
                MetabolizerPhenotype.NORMAL: "Use standard warfarin dosing algorithm. Target INR 2.0-3.0 for most indications.",
                MetabolizerPhenotype.INTERMEDIATE: "Consider 20-40% dose reduction from standard algorithm. CYP2C9 intermediate metabolizer status associated with reduced warfarin clearance. More frequent INR monitoring recommended during initiation.",
                MetabolizerPhenotype.POOR: "Consider 40-60% dose reduction. CYP2C9 poor metabolizer — significantly reduced warfarin metabolism. Risk of supratherapeutic INR and bleeding. Initiate at lower dose with frequent INR monitoring. Consider alternative anticoagulant if available.",
                MetabolizerPhenotype.ULTRARAPID: "May require higher than standard doses. Monitor INR closely for subtherapeutic levels.",
            },
            monitoring_advice={
                MetabolizerPhenotype.NORMAL: "Standard INR monitoring. Check INR 2-3 times/week during initiation, then monthly when stable.",
                MetabolizerPhenotype.INTERMEDIATE: "Enhanced INR monitoring recommended. Check INR 3-4 times/week during initiation. Watch for signs of over-anticoagulation (bruising, bleeding gums, dark stools).",
                MetabolizerPhenotype.POOR: "INTENSIVE INR monitoring required. Daily INR for first week, then 2-3 times/week for 2-4 weeks. Educate patient on bleeding precautions. Consider pharmacist-managed anticoagulation clinic referral.",
                MetabolizerPhenotype.ULTRARAPID: "Monitor for subtherapeutic INR. More frequent dose adjustments may be needed. Assess for thromboembolic risk if INR consistently low.",
            },
            drug_interactions=[
                {"drug": "amiodarone", "effect": "Increases warfarin effect by inhibiting CYP2C9/CYP3A4. Reduce warfarin dose by 30-50%.", "severity": "high"},
                {"drug": "fluconazole", "effect": "Potent CYP2C9 inhibitor. May dramatically increase INR. Monitor closely.", "severity": "high"},
                {"drug": "rifampin", "effect": "Potent CYP2C9 inducer. May decrease warfarin effect significantly.", "severity": "high"},
                {"drug": "aspirin", "effect": "Additive bleeding risk. Use lowest effective aspirin dose if combined.", "severity": "moderate"},
                {"drug": "ibuprofen", "effect": "NSAIDs increase bleeding risk and may affect warfarin protein binding.", "severity": "moderate"},
            ],
            source_url="https://cpicpgx.org/guidelines/guideline-for-warfarin-and-cyp2c9-and-vkorc1/",
        ))

        # ── CYP2C19 + Clopidogrel ───────────────────────────────────────
        self._add_rule(PharmacogenomicRule(
            rule_id="CPIC-CLOPIDOGREL-CYP2C19",
            gene_name="CYP2C19",
            variant_ids=frozenset({"rs4244285", "rs4986893", "rs12248560"}),
            drug_names=frozenset({"clopidogrel", "plavix"}),
            diagnosis_codes=frozenset({
                "I21.0", "I21.1", "I21.2", "I21.3", "I21.4", "I21.9",  # MI
                "I25.1", "I25.10", "I25.110",                           # CAD
                "Z95.1", "Z95.5",                                        # Cardiac stent
                "I63.9",                                                  # Stroke
            }),
            genotype_to_metabolizer={
                0: MetabolizerPhenotype.NORMAL,
                1: MetabolizerPhenotype.INTERMEDIATE,
                2: MetabolizerPhenotype.POOR,
            },
            metabolizer_to_risk={
                MetabolizerPhenotype.NORMAL: RiskLevel.LOW,
                MetabolizerPhenotype.INTERMEDIATE: RiskLevel.MODERATE,
                MetabolizerPhenotype.POOR: RiskLevel.HIGH,
                MetabolizerPhenotype.ULTRARAPID: RiskLevel.LOW,
            },
            metabolizer_to_severity={
                MetabolizerPhenotype.NORMAL: AlertSeverity.INFORMATIONAL,
                MetabolizerPhenotype.INTERMEDIATE: AlertSeverity.MODERATE,
                MetabolizerPhenotype.POOR: AlertSeverity.CRITICAL,
                MetabolizerPhenotype.ULTRARAPID: AlertSeverity.INFORMATIONAL,
            },
            dose_unit="mg/day",
            dose_adjustment_factors={
                MetabolizerPhenotype.NORMAL: 1.0,
                MetabolizerPhenotype.INTERMEDIATE: 1.0,  # Alternative drug recommended
                MetabolizerPhenotype.POOR: 1.0,          # Alternative drug recommended
                MetabolizerPhenotype.ULTRARAPID: 1.0,
            },
            standard_dose_range=(75.0, 75.0),
            cpic_guideline_id="CPIC-2022-clopidogrel",
            cpic_guideline_version="4.0 (2022 update)",
            cpic_evidence_level="A (strong)",
            cpic_recommendation={
                MetabolizerPhenotype.NORMAL: "Standard clopidogrel 75mg daily is expected to be effective. Normal CYP2C19 metabolism ensures adequate active metabolite formation.",
                MetabolizerPhenotype.INTERMEDIATE: "Reduced clopidogrel activation expected. Consider alternative antiplatelet agent (prasugrel or ticagrelor) if no contraindications. If clopidogrel is continued, consider platelet function testing.",
                MetabolizerPhenotype.POOR: "CRITICAL: Clopidogrel is expected to be INEFFECTIVE. CYP2C19 poor metabolizer cannot adequately convert clopidogrel to its active form. STRONGLY recommend alternative antiplatelet: prasugrel 10mg daily or ticagrelor 90mg twice daily (if no contraindications).",
                MetabolizerPhenotype.ULTRARAPID: "Enhanced clopidogrel activation. Standard dosing appropriate. Ultrarapid metabolism may provide enhanced antiplatelet effect.",
            },
            monitoring_advice={
                MetabolizerPhenotype.NORMAL: "Standard antiplatelet monitoring. Annual review of indication.",
                MetabolizerPhenotype.INTERMEDIATE: "Consider platelet function testing (e.g., VerifyNow P2Y12 assay). Monitor for signs of treatment failure (recurrent ischemic events).",
                MetabolizerPhenotype.POOR: "Do NOT rely on clopidogrel for platelet inhibition. If alternative agent is started, confirm adequate platelet inhibition with function testing. Monitor closely for thrombotic events.",
                MetabolizerPhenotype.ULTRARAPID: "Standard monitoring. Watch for excessive bleeding.",
            },
            source_url="https://cpicpgx.org/guidelines/guideline-for-clopidogrel-and-cyp2c19/",
        ))

        # ── CYP2D6 + Codeine ────────────────────────────────────────────
        self._add_rule(PharmacogenomicRule(
            rule_id="CPIC-CODEINE-CYP2D6",
            gene_name="CYP2D6",
            variant_ids=frozenset({"rs3892097", "rs5030655", "rs1065852", "rs16947"}),
            drug_names=frozenset({"codeine", "codeine phosphate", "co-codamol", "tylenol 3"}),
            diagnosis_codes=frozenset({
                "R05", "R05.9",        # Cough
                "M54.5",              # Low back pain
                "G89.29",             # Chronic pain
                "R52",                # Pain, unspecified
            }),
            genotype_to_metabolizer={
                0: MetabolizerPhenotype.NORMAL,
                1: MetabolizerPhenotype.INTERMEDIATE,
                2: MetabolizerPhenotype.POOR,
            },
            metabolizer_to_risk={
                MetabolizerPhenotype.ULTRARAPID: RiskLevel.VERY_HIGH,
                MetabolizerPhenotype.NORMAL: RiskLevel.LOW,
                MetabolizerPhenotype.INTERMEDIATE: RiskLevel.MODERATE,
                MetabolizerPhenotype.POOR: RiskLevel.MODERATE,
            },
            metabolizer_to_severity={
                MetabolizerPhenotype.ULTRARAPID: AlertSeverity.CRITICAL,
                MetabolizerPhenotype.NORMAL: AlertSeverity.INFORMATIONAL,
                MetabolizerPhenotype.INTERMEDIATE: AlertSeverity.MODERATE,
                MetabolizerPhenotype.POOR: AlertSeverity.MODERATE,
            },
            dose_unit="mg/dose",
            standard_dose_range=(15.0, 60.0),
            cpic_guideline_id="CPIC-2020-codeine",
            cpic_guideline_version="4.0 (2020 update)",
            cpic_evidence_level="A (strong)",
            cpic_recommendation={
                MetabolizerPhenotype.ULTRARAPID: "AVOID CODEINE. CYP2D6 ultrarapid metabolizer converts codeine to morphine at dangerously elevated rates. Risk of FATAL respiratory depression, especially in children and postoperative patients. Use non-opioid analgesic alternatives.",
                MetabolizerPhenotype.NORMAL: "Standard codeine dosing expected to provide adequate analgesia. Normal CYP2D6 metabolism.",
                MetabolizerPhenotype.INTERMEDIATE: "Reduced codeine efficacy expected due to decreased morphine formation. Consider alternative analgesic if inadequate pain relief. Avoid dose escalation — use non-codeine alternative instead.",
                MetabolizerPhenotype.POOR: "Codeine will be INEFFECTIVE. CYP2D6 poor metabolizer cannot convert codeine to morphine. Use alternative analgesic (e.g., acetaminophen, NSAIDs, or non-CYP2D6-dependent opioid if needed).",
            },
            monitoring_advice={
                MetabolizerPhenotype.ULTRARAPID: "DO NOT PRESCRIBE CODEINE. If already administered: monitor for respiratory depression, excessive sedation, confusion, shallow breathing. Have naloxone available. This is a BLACK BOX WARNING scenario.",
                MetabolizerPhenotype.NORMAL: "Standard pain assessment. Monitor for common opioid side effects (constipation, nausea, sedation).",
                MetabolizerPhenotype.INTERMEDIATE: "Monitor pain relief adequacy. If analgesia insufficient, switch to alternative — do NOT increase codeine dose.",
                MetabolizerPhenotype.POOR: "Codeine will not provide analgesia. Switch to alternative pain management strategy.",
            },
            source_url="https://cpicpgx.org/guidelines/guideline-for-codeine-and-cyp2d6/",
        ))

        # ── SLCO1B1 + Simvastatin ───────────────────────────────────────
        self._add_rule(PharmacogenomicRule(
            rule_id="CPIC-SIMVASTATIN-SLCO1B1",
            gene_name="SLCO1B1",
            variant_ids=frozenset({"rs4149056"}),
            drug_names=frozenset({"simvastatin", "zocor", "vytorin"}),
            diagnosis_codes=frozenset({
                "E78.0", "E78.00", "E78.01",  # Hypercholesterolemia
                "E78.5",                        # Dyslipidemia
                "I25.10",                       # CAD
                "Z79.899",                      # Long-term drug therapy
            }),
            genotype_to_metabolizer={
                0: MetabolizerPhenotype.NORMAL,
                1: MetabolizerPhenotype.INTERMEDIATE,
                2: MetabolizerPhenotype.POOR,
            },
            metabolizer_to_risk={
                MetabolizerPhenotype.NORMAL: RiskLevel.LOW,
                MetabolizerPhenotype.INTERMEDIATE: RiskLevel.MODERATE,
                MetabolizerPhenotype.POOR: RiskLevel.HIGH,
            },
            metabolizer_to_severity={
                MetabolizerPhenotype.NORMAL: AlertSeverity.INFORMATIONAL,
                MetabolizerPhenotype.INTERMEDIATE: AlertSeverity.MODERATE,
                MetabolizerPhenotype.POOR: AlertSeverity.HIGH,
            },
            dose_unit="mg/day",
            standard_dose_range=(5.0, 40.0),
            cpic_guideline_id="CPIC-2014-simvastatin",
            cpic_guideline_version="2.0 (2014)",
            cpic_evidence_level="A (strong)",
            cpic_recommendation={
                MetabolizerPhenotype.NORMAL: "Standard simvastatin dosing. No increased myopathy risk from SLCO1B1 genotype.",
                MetabolizerPhenotype.INTERMEDIATE: "Increased risk of simvastatin-induced myopathy. Avoid simvastatin doses >20mg/day. Consider alternative statin (pravastatin, rosuvastatin) at equivalent LDL-lowering dose.",
                MetabolizerPhenotype.POOR: "HIGH risk of simvastatin-induced myopathy and rhabdomyolysis. AVOID simvastatin. Switch to pravastatin or rosuvastatin (which are not SLCO1B1-dependent). If muscle symptoms present, check CK level urgently.",
            },
            monitoring_advice={
                MetabolizerPhenotype.NORMAL: "Standard statin monitoring: lipid panel, LFTs at baseline and as indicated.",
                MetabolizerPhenotype.INTERMEDIATE: "Monitor for muscle symptoms (pain, tenderness, weakness). Check CK if symptomatic. Consider dose reduction or switch. Lipid panel at 6-12 weeks.",
                MetabolizerPhenotype.POOR: "Discontinue simvastatin if not already done. Check baseline CK. Start alternative statin at low dose. Monitor for muscle symptoms closely for 3-6 months.",
            },
            source_url="https://cpicpgx.org/guidelines/guideline-for-simvastatin-and-slco1b1/",
        ))

        # ── DPYD + Fluoropyrimidines ─────────────────────────────────────
        self._add_rule(PharmacogenomicRule(
            rule_id="CPIC-FLUOROPYRIMIDINES-DPYD",
            gene_name="DPYD",
            variant_ids=frozenset({"rs3918290", "rs55886062", "rs67376798", "rs75017182"}),
            drug_names=frozenset({"fluorouracil", "5-fu", "capecitabine", "xeloda", "tegafur"}),
            diagnosis_codes=frozenset({
                "C18.9", "C19", "C20",     # Colorectal cancer
                "C50.9", "C50.91",          # Breast cancer
                "C16.9",                    # Gastric cancer
                "C25.9",                    # Pancreatic cancer
            }),
            genotype_to_metabolizer={
                0: MetabolizerPhenotype.NORMAL,
                1: MetabolizerPhenotype.INTERMEDIATE,
                2: MetabolizerPhenotype.POOR,
            },
            metabolizer_to_risk={
                MetabolizerPhenotype.NORMAL: RiskLevel.LOW,
                MetabolizerPhenotype.INTERMEDIATE: RiskLevel.HIGH,
                MetabolizerPhenotype.POOR: RiskLevel.VERY_HIGH,
            },
            metabolizer_to_severity={
                MetabolizerPhenotype.NORMAL: AlertSeverity.INFORMATIONAL,
                MetabolizerPhenotype.INTERMEDIATE: AlertSeverity.HIGH,
                MetabolizerPhenotype.POOR: AlertSeverity.CRITICAL,
            },
            dose_unit="mg/m²",
            dose_adjustment_factors={
                MetabolizerPhenotype.NORMAL: 1.0,
                MetabolizerPhenotype.INTERMEDIATE: 0.5,
                MetabolizerPhenotype.POOR: 0.0,
            },
            cpic_guideline_id="CPIC-2018-fluoropyrimidines",
            cpic_guideline_version="2.0 (2018 update)",
            cpic_evidence_level="A (strong)",
            cpic_recommendation={
                MetabolizerPhenotype.NORMAL: "Standard fluoropyrimidine dosing. Normal DPD enzyme activity expected.",
                MetabolizerPhenotype.INTERMEDIATE: "REDUCE fluoropyrimidine dose by at least 50%. DPD intermediate metabolizer — reduced drug clearance with elevated risk of severe/fatal toxicity (grade 3-4 mucositis, myelosuppression, hand-foot syndrome).",
                MetabolizerPhenotype.POOR: "DO NOT USE fluoropyrimidines. DPD poor metabolizer — complete or near-complete DPD deficiency. Risk of FATAL toxicity. Use alternative chemotherapy regimen.",
            },
            monitoring_advice={
                MetabolizerPhenotype.NORMAL: "Standard oncology monitoring: CBC with differential, LFTs before each cycle. Monitor for mucositis, diarrhea, hand-foot syndrome.",
                MetabolizerPhenotype.INTERMEDIATE: "INTENSIVE monitoring: CBC 2x/week for first 2 cycles. Daily assessment for mucositis and diarrhea. Consider therapeutic drug monitoring (TDM) of 5-FU levels. Dose-escalate cautiously if tolerated.",
                MetabolizerPhenotype.POOR: "CONTRAINDICATED. If inadvertently administered: immediate discontinuation. Supportive care, colony-stimulating factors, close ICU monitoring may be needed. Report to oncology pharmacist.",
            },
            source_url="https://cpicpgx.org/guidelines/guideline-for-fluoropyrimidines-and-dpyd/",
        ))

        # ── CYP2C19 + Proton Pump Inhibitors ────────────────────────────
        self._add_rule(PharmacogenomicRule(
            rule_id="CPIC-PPI-CYP2C19",
            gene_name="CYP2C19",
            variant_ids=frozenset({"rs4244285", "rs4986893", "rs12248560"}),
            drug_names=frozenset({"omeprazole", "lansoprazole", "pantoprazole", "esomeprazole", "dexlansoprazole", "prilosec", "prevacid", "protonix", "nexium"}),
            diagnosis_codes=frozenset({
                "K21.0",    # GERD with esophagitis
                "K21.9",    # GERD without esophagitis
                "K25.9",    # Gastric ulcer
                "K26.9",    # Duodenal ulcer
                "K27.9",    # Peptic ulcer
                "B96.81",   # H. pylori
            }),
            genotype_to_metabolizer={
                0: MetabolizerPhenotype.NORMAL,
                1: MetabolizerPhenotype.INTERMEDIATE,
                2: MetabolizerPhenotype.POOR,
            },
            metabolizer_to_risk={
                MetabolizerPhenotype.ULTRARAPID: RiskLevel.MODERATE,
                MetabolizerPhenotype.NORMAL: RiskLevel.LOW,
                MetabolizerPhenotype.INTERMEDIATE: RiskLevel.LOW,
                MetabolizerPhenotype.POOR: RiskLevel.LOW,
            },
            metabolizer_to_severity={
                MetabolizerPhenotype.ULTRARAPID: AlertSeverity.MODERATE,
                MetabolizerPhenotype.NORMAL: AlertSeverity.INFORMATIONAL,
                MetabolizerPhenotype.INTERMEDIATE: AlertSeverity.INFORMATIONAL,
                MetabolizerPhenotype.POOR: AlertSeverity.INFORMATIONAL,
            },
            dose_unit="mg/day",
            standard_dose_range=(10.0, 40.0),
            cpic_guideline_id="CPIC-2021-PPI",
            cpic_guideline_version="1.0 (2021)",
            cpic_evidence_level="A (strong) for H. pylori; B-C for GERD",
            cpic_recommendation={
                MetabolizerPhenotype.ULTRARAPID: "Increased PPI dose by 50-100% may be needed for H. pylori eradication. Standard GERD dosing may be subtherapeutic. CYP2C19 rapid metabolism decreases PPI exposure. Consider switching to rabeprazole (less CYP2C19-dependent).",
                MetabolizerPhenotype.NORMAL: "Standard PPI dosing appropriate for indication.",
                MetabolizerPhenotype.INTERMEDIATE: "Standard dosing likely adequate. May have slightly increased PPI exposure — generally beneficial for acid suppression.",
                MetabolizerPhenotype.POOR: "Reduced PPI clearance — increased drug exposure. Standard doses provide enhanced acid suppression. For H. pylori, standard dosing expected to be highly effective. For long-term GERD, consider lowest effective dose.",
            },
            monitoring_advice={
                MetabolizerPhenotype.ULTRARAPID: "Monitor H. pylori eradication success. If treatment failure, consider higher PPI dose or non-CYP2C19-dependent PPI. For GERD, assess symptom control at 4-8 weeks.",
                MetabolizerPhenotype.NORMAL: "Standard follow-up. H. pylori: confirm eradication at 4 weeks post-treatment. GERD: symptom assessment.",
                MetabolizerPhenotype.INTERMEDIATE: "Standard monitoring. Assess for PPI-related side effects (headache, diarrhea) if prolonged use.",
                MetabolizerPhenotype.POOR: "Enhanced acid suppression expected. Monitor for PPI side effects with long-term use (hypomagnesemia, bone density, C. difficile risk). May tolerate lower doses.",
            },
            source_url="https://cpicpgx.org/guidelines/guideline-for-proton-pump-inhibitors-and-cyp2c19/",
        ))

        # ── TPMT/NUDT15 + Thiopurines ───────────────────────────────────
        self._add_rule(PharmacogenomicRule(
            rule_id="CPIC-THIOPURINES-TPMT-NUDT15",
            gene_name="TPMT/NUDT15",
            variant_ids=frozenset({"rs1800462", "rs1800460", "rs1142345", "rs116855232"}),
            drug_names=frozenset({"azathioprine", "mercaptopurine", "6-mp", "thioguanine", "imuran", "purinethol"}),
            diagnosis_codes=frozenset({
                "K50.9",    # Crohn's disease
                "K51.9",    # Ulcerative colitis
                "C91.0",    # ALL (acute lymphoblastic leukemia)
                "M32.9",    # SLE
                "L40.9",    # Psoriasis
            }),
            genotype_to_metabolizer={
                0: MetabolizerPhenotype.NORMAL,
                1: MetabolizerPhenotype.INTERMEDIATE,
                2: MetabolizerPhenotype.POOR,
            },
            metabolizer_to_risk={
                MetabolizerPhenotype.NORMAL: RiskLevel.LOW,
                MetabolizerPhenotype.INTERMEDIATE: RiskLevel.HIGH,
                MetabolizerPhenotype.POOR: RiskLevel.VERY_HIGH,
            },
            metabolizer_to_severity={
                MetabolizerPhenotype.NORMAL: AlertSeverity.INFORMATIONAL,
                MetabolizerPhenotype.INTERMEDIATE: AlertSeverity.HIGH,
                MetabolizerPhenotype.POOR: AlertSeverity.CRITICAL,
            },
            dose_unit="mg/kg/day",
            dose_adjustment_factors={
                MetabolizerPhenotype.NORMAL: 1.0,
                MetabolizerPhenotype.INTERMEDIATE: 0.5,
                MetabolizerPhenotype.POOR: 0.1,
            },
            standard_dose_range=(0.5, 3.0),
            cpic_guideline_id="CPIC-2019-thiopurines",
            cpic_guideline_version="3.0 (2019 update)",
            cpic_evidence_level="A (strong)",
            cpic_recommendation={
                MetabolizerPhenotype.NORMAL: "Standard thiopurine dosing appropriate. Normal TPMT/NUDT15 activity.",
                MetabolizerPhenotype.INTERMEDIATE: "REDUCE starting dose by 30-70%. TPMT/NUDT15 intermediate metabolizer — increased risk of myelosuppression. Titrate dose based on clinical response and tolerance. Allow 2-4 weeks to reach steady state before dose adjustments.",
                MetabolizerPhenotype.POOR: "DRASTICALLY REDUCE dose (reduce by ≥90%) or use alternative agent. TPMT/NUDT15 poor metabolizer — very HIGH risk of life-threatening myelosuppression at standard doses. If thiopurine is essential, start at ≤10% of standard dose with thrice-weekly CBC.",
            },
            monitoring_advice={
                MetabolizerPhenotype.NORMAL: "CBC with differential weekly for first month, then biweekly for 2 months, then monthly. Monitor LFTs. Consider thiopurine metabolite levels (6-TGN, 6-MMP).",
                MetabolizerPhenotype.INTERMEDIATE: "CBC with differential TWICE WEEKLY for first month. Then weekly for 2 months. Consider 6-TGN therapeutic drug monitoring. Target 6-TGN 235-450 pmol/8x10⁸ RBC. Hold drug if WBC <3.5 or ANC <1.5.",
                MetabolizerPhenotype.POOR: "CBC with differential THREE TIMES/WEEK for first 4 weeks. Mandatory therapeutic drug monitoring. Immediate discontinuation if WBC <3.0 or ANC <1.0. Have growth factor support plan ready.",
            },
            source_url="https://cpicpgx.org/guidelines/guideline-for-thiopurines-and-tpmt/",
        ))

        # ── CYP3A5 + Tacrolimus ─────────────────────────────────────────
        self._add_rule(PharmacogenomicRule(
            rule_id="CPIC-TACROLIMUS-CYP3A5",
            gene_name="CYP3A5",
            variant_ids=frozenset({"rs776746"}),
            drug_names=frozenset({"tacrolimus", "prograf", "envarsus", "astagraf"}),
            diagnosis_codes=frozenset({
                "Z94.0",    # Kidney transplant
                "Z94.1",    # Heart transplant
                "Z94.4",    # Liver transplant
                "T86.10",   # Kidney transplant rejection
                "Z79.899",  # Long-term immunosuppressant
            }),
            genotype_to_metabolizer={
                0: MetabolizerPhenotype.POOR,       # CYP3A5 non-expresser (*3/*3)
                1: MetabolizerPhenotype.INTERMEDIATE,  # heterozygous (*1/*3)
                2: MetabolizerPhenotype.NORMAL,       # CYP3A5 expresser (*1/*1)
            },
            metabolizer_to_risk={
                MetabolizerPhenotype.NORMAL: RiskLevel.MODERATE,  # Need higher doses
                MetabolizerPhenotype.INTERMEDIATE: RiskLevel.MODERATE,
                MetabolizerPhenotype.POOR: RiskLevel.LOW,
            },
            metabolizer_to_severity={
                MetabolizerPhenotype.NORMAL: AlertSeverity.MODERATE,
                MetabolizerPhenotype.INTERMEDIATE: AlertSeverity.MODERATE,
                MetabolizerPhenotype.POOR: AlertSeverity.INFORMATIONAL,
            },
            dose_unit="mg/day",
            dose_adjustment_factors={
                MetabolizerPhenotype.NORMAL: 1.5,      # CYP3A5 expressers need higher doses
                MetabolizerPhenotype.INTERMEDIATE: 1.25,
                MetabolizerPhenotype.POOR: 1.0,
            },
            standard_dose_range=(0.5, 15.0),
            cpic_guideline_id="CPIC-2015-tacrolimus",
            cpic_guideline_version="2.0 (2015)",
            cpic_evidence_level="A (strong)",
            cpic_recommendation={
                MetabolizerPhenotype.NORMAL: "CYP3A5 expresser — rapid tacrolimus metabolism. Increase starting dose by 1.5-2x to achieve target trough levels faster. Standard starting dose will likely result in subtherapeutic levels, risking rejection.",
                MetabolizerPhenotype.INTERMEDIATE: "CYP3A5 intermediate expresser. Consider 25-50% increase in starting dose. Monitor trough levels closely during initiation.",
                MetabolizerPhenotype.POOR: "CYP3A5 non-expresser (most common genotype). Standard tacrolimus dosing appropriate. Typical target trough 5-15 ng/mL depending on time post-transplant and organ type.",
            },
            monitoring_advice={
                MetabolizerPhenotype.NORMAL: "Trough level monitoring 3x/week during initiation until therapeutic range achieved. Expect to need higher doses. Monitor renal function (BUN, creatinine), magnesium, glucose, potassium. Assess for neurotoxicity (tremor, headache).",
                MetabolizerPhenotype.INTERMEDIATE: "Trough level monitoring 2-3x/week during initiation. Target organ-specific therapeutic range. Standard adverse effect monitoring.",
                MetabolizerPhenotype.POOR: "Standard trough level monitoring. Weekly during first month, then per transplant center protocol. Monitor for tacrolimus toxicity if dose not adjusted for elevated levels.",
            },
            source_url="https://cpicpgx.org/guidelines/guideline-for-tacrolimus-and-cyp3A5/",
        ))

        # ── CYP2D6 + SSRIs (Paroxetine/Fluvoxamine) ────────────────────
        self._add_rule(PharmacogenomicRule(
            rule_id="CPIC-SSRI-CYP2D6",
            gene_name="CYP2D6",
            variant_ids=frozenset({"rs3892097", "rs5030655", "rs1065852", "rs16947"}),
            drug_names=frozenset({"paroxetine", "paxil", "fluvoxamine", "luvox", "fluoxetine", "prozac", "venlafaxine", "effexor"}),
            diagnosis_codes=frozenset({
                "F32.1", "F32.2", "F32.9",  # Major depressive disorder
                "F33.0", "F33.1", "F33.2",  # Recurrent depressive
                "F41.0", "F41.1",            # Anxiety disorders
                "F42.9",                      # OCD
                "F43.10",                     # PTSD
            }),
            genotype_to_metabolizer={
                0: MetabolizerPhenotype.NORMAL,
                1: MetabolizerPhenotype.INTERMEDIATE,
                2: MetabolizerPhenotype.POOR,
            },
            metabolizer_to_risk={
                MetabolizerPhenotype.ULTRARAPID: RiskLevel.MODERATE,
                MetabolizerPhenotype.NORMAL: RiskLevel.LOW,
                MetabolizerPhenotype.INTERMEDIATE: RiskLevel.MODERATE,
                MetabolizerPhenotype.POOR: RiskLevel.HIGH,
            },
            metabolizer_to_severity={
                MetabolizerPhenotype.ULTRARAPID: AlertSeverity.MODERATE,
                MetabolizerPhenotype.NORMAL: AlertSeverity.INFORMATIONAL,
                MetabolizerPhenotype.INTERMEDIATE: AlertSeverity.MODERATE,
                MetabolizerPhenotype.POOR: AlertSeverity.HIGH,
            },
            dose_unit="mg/day",
            dose_adjustment_factors={
                MetabolizerPhenotype.ULTRARAPID: 1.5,
                MetabolizerPhenotype.NORMAL: 1.0,
                MetabolizerPhenotype.INTERMEDIATE: 0.75,
                MetabolizerPhenotype.POOR: 0.5,
            },
            standard_dose_range=(10.0, 60.0),
            cpic_guideline_id="CPIC-2015-SSRIs",
            cpic_guideline_version="2.0 (2015 update)",
            cpic_evidence_level="A (strong) for paroxetine/fluvoxamine; B for others",
            cpic_recommendation={
                MetabolizerPhenotype.ULTRARAPID: "Reduced drug exposure expected. Consider alternative SSRI not primarily metabolized by CYP2D6 (e.g., sertraline, citalopram, escitalopram) or increase dose by 50%. Monitor for therapeutic failure.",
                MetabolizerPhenotype.NORMAL: "Standard SSRI dosing appropriate. Normal CYP2D6 metabolism.",
                MetabolizerPhenotype.INTERMEDIATE: "Consider 25% dose reduction for paroxetine/fluvoxamine. Monitor for increased side effects (nausea, sexual dysfunction, serotonin excess). Alternatively, use CYP2D6-independent SSRI.",
                MetabolizerPhenotype.POOR: "Consider 50% dose reduction or select alternative SSRI not metabolized by CYP2D6 (sertraline, citalopram, escitalopram). CYP2D6 poor metabolizer — elevated drug levels expected with increased adverse effect risk.",
            },
            monitoring_advice={
                MetabolizerPhenotype.ULTRARAPID: "Assess therapeutic response at 2-4 weeks. If inadequate, consider dose increase or switch to non-CYP2D6 SSRI. Use PHQ-9 or GAD-7 for standardized assessment.",
                MetabolizerPhenotype.NORMAL: "Standard psychiatric follow-up. Assess response at 4-6 weeks. Monitor for common SSRI side effects.",
                MetabolizerPhenotype.INTERMEDIATE: "Early follow-up at 1-2 weeks for side effect assessment. Monitor for serotonergic symptoms. Dose titration may need to be slower.",
                MetabolizerPhenotype.POOR: "Close monitoring for adverse effects in first 2 weeks. Watch for serotonin syndrome symptoms (agitation, hyperthermia, clonus, diaphoresis). Consider therapeutic drug monitoring if available.",
            },
            source_url="https://cpicpgx.org/guidelines/guideline-for-selective-serotonin-reuptake-inhibitors-and-cyp2d6-and-cyp2c19/",
        ))

    def _add_rule(self, rule: PharmacogenomicRule) -> None:
        """Index a rule by drugs, diagnoses, and gene."""
        self._rules.append(rule)
        for drug in rule.drug_names:
            key = drug.strip().lower()
            self._drug_index.setdefault(key, []).append(rule)
        for code in rule.diagnosis_codes:
            key = code.strip().upper()
            self._diagnosis_index.setdefault(key, []).append(rule)
        gene_key = rule.gene_name.strip().upper()
        self._gene_index.setdefault(gene_key, []).append(rule)

    def find_matching_rules(
        self,
        diagnosis_codes: list[str],
        medication_names: list[str],
    ) -> list[PharmacogenomicRule]:
        """Find all rules triggered by the given diagnosis codes and medications."""
        matched: set[str] = set()
        results: list[PharmacogenomicRule] = []

        for code in diagnosis_codes:
            normalized = code.strip().upper()
            for rule in self._diagnosis_index.get(normalized, []):
                if rule.rule_id not in matched:
                    matched.add(rule.rule_id)
                    results.append(rule)

        for name in medication_names:
            normalized = name.strip().lower()
            for rule in self._drug_index.get(normalized, []):
                if rule.rule_id not in matched:
                    matched.add(rule.rule_id)
                    results.append(rule)

        return results

    def get_rule_by_id(self, rule_id: str) -> Optional[PharmacogenomicRule]:
        """Retrieve a specific rule by its ID."""
        for rule in self._rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    def get_all_rules(self) -> list[PharmacogenomicRule]:
        """Return all registered rules."""
        return list(self._rules)

    def get_supported_drugs(self) -> set[str]:
        """Return all drug names with pharmacogenomic rules."""
        return set(self._drug_index.keys())

    def get_supported_genes(self) -> set[str]:
        """Return all gene names with pharmacogenomic rules."""
        return set(self._gene_index.keys())

    def __len__(self) -> int:
        return len(self._rules)

    def __repr__(self) -> str:
        return f"CPICKnowledgeBase(rules={len(self._rules)}, drugs={len(self._drug_index)}, genes={len(self._gene_index)})"
