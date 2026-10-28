#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  CARE'OS Module 3 — Genomics Engine Demo Runner                        ║
║  Run this script to see the full pipeline in action                    ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage:
    python run_demo.py              # Run full demo with synthetic data
    python run_demo.py --fast       # Quick demo (fewer samples/epochs)
    python run_demo.py --dashboard  # Launch Streamlit dashboard
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np


def print_banner():
    """Print the CARE'OS banner."""
    banner = """
\033[38;2;203;166;247m╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║     ██████╗ █████╗ ██████╗ ███████╗ ██╗ ██████╗ ███████╗                 ║
║    ██╔════╝██╔══██╗██╔══██╗██╔════╝ ╚██╗██╔══██╗██╔════╝                ║
║    ██║     ███████║██████╔╝█████╗    ║██║██║  ██║███████╗                ║
║    ██║     ██╔══██║██╔══██╗██╔══╝    ║██║██║  ██║╚════██║                ║
║    ╚██████╗██║  ██║██║  ██║███████╗ ██╔╝╚██████╔╝███████║                ║
║     ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═════╝ ╚══════╝                ║
║                                                                          ║
║\033[38;2;137;180;250m         Module 3 — Genomics-Informed Treatment Engine               \033[38;2;203;166;247m║
║\033[38;2;166;227;161m              Pharmacogenomic Decision Support System                \033[38;2;203;166;247m║
║\033[38;2;148;226;213m                     v3.0.0 "Helix"                                 \033[38;2;203;166;247m║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝\033[0m
"""
    print(banner)


def print_section(title: str, color: str = "mauve"):
    """Print a styled section header."""
    colors = {
        "mauve": "\033[38;2;203;166;247m",
        "blue": "\033[38;2;137;180;250m",
        "green": "\033[38;2;166;227;161m",
        "yellow": "\033[38;2;249;226;175m",
        "red": "\033[38;2;243;139;168m",
        "peach": "\033[38;2;250;179;135m",
        "teal": "\033[38;2;148;226;213m",
    }
    c = colors.get(color, colors["mauve"])
    reset = "\033[0m"
    print(f"\n{c}{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}{reset}\n")


def run_demo(fast: bool = False):
    """Run the full CARE'OS Module 3 demo pipeline."""
    print_banner()

    n_patients = 150 if fast else 400
    vae_epochs = 15 if fast else 35
    optuna_trials = 3 if fast else 10

    # ── Step 1: Generate Synthetic Data ──────────────────────────────
    print_section("STEP 1: Generating Synthetic Pharmacogenomic Data", "blue")

    from careos_genomics.synthetic_data import SyntheticPharmacogenomicDataGenerator

    generator = SyntheticPharmacogenomicDataGenerator(seed=42)
    print(f"  Generating warfarin cohort ({n_patients} patients)...")
    cohort = generator.generate_warfarin_cohort(n_patients)

    print(f"  ✓ SNP profiles:      {cohort.snp_profiles.shape}")
    print(f"  ✓ Clinical data:     {cohort.clinical_data.shape}")
    print(f"  ✓ Gene expression:   {cohort.gene_expression.shape}")
    print(f"  ✓ Response labels:   {dict(zip(*np.unique(cohort.response_labels, return_counts=True)))}")
    print(f"  ✓ Dose range:        {cohort.optimal_doses.min():.1f} - {cohort.optimal_doses.max():.1f} mg/day")

    # ── Step 2: Configure and Train Engine ───────────────────────────
    print_section("STEP 2: Training Genomics Engine", "green")

    from careos_genomics.engine import GenomicsEngine, EngineConfig

    config = EngineConfig(
        vae_latent_dim=16,
        vae_epochs=vae_epochs,
        enable_vae=True,
        use_xgboost=True,
        xgb_n_trials=optuna_trials,
        enable_shap=True,
    )

    engine = GenomicsEngine(config=config)

    start = time.time()
    metrics = engine.train(
        snp_profiles=cohort.snp_profiles,
        clinical_data=cohort.clinical_data,
        gene_expression=cohort.gene_expression,
        response_labels=cohort.response_labels,
        optimal_doses=cohort.optimal_doses,
    )
    elapsed = time.time() - start

    print(engine.get_training_summary())

    # ── Step 3: Knowledge Base ───────────────────────────────────────
    print_section("STEP 3: CPIC Knowledge Base", "yellow")

    kb = engine.knowledge_base
    print(f"  Knowledge Base: {kb}")
    print(f"  Supported Drugs: {sorted(kb.get_supported_drugs())[:10]}...")
    print(f"  Supported Genes: {sorted(kb.get_supported_genes())}")

    # ── Step 4: Simulate Module 2 Trigger ────────────────────────────
    print_section("STEP 4: Simulating Module 2 Trigger", "peach")

    from careos_genomics.data_models import (
        TriggerPayload,
        ClinicalFeatures,
        PatientGenomicProfile,
    )

    # Simulate a patient with atrial fibrillation on warfarin
    trigger = TriggerPayload(
        patient_id="PAT-DEMO-42",
        diagnosis_codes=["I48.0"],  # Atrial fibrillation
        medication_names=["warfarin"],
        prescribing_physician="DR-WATSON-001",
    )

    print(f"  Patient ID:   {trigger.patient_id}")
    print(f"  Diagnosis:    {trigger.diagnosis_codes} (Atrial Fibrillation)")
    print(f"  Medication:   {trigger.medication_names}")
    print(f"  Physician:    {trigger.prescribing_physician}")
    print(f"  Consult ID:   {trigger.consultation_id}")

    # Create clinical features for a 70-year-old patient
    clinical = ClinicalFeatures(
        age=70,
        sex="M",
        weight_kg=82.0,
        height_cm=175.0,
        creatinine_clearance=65.0,
        alt=28.0,
        ast=24.0,
        bilirubin=0.8,
        albumin=3.8,
        inr=1.1,
        concurrent_medications=["amiodarone", "aspirin"],
    )

    print(f"\n  Clinical Profile:")
    print(f"    Age: {clinical.age}, Sex: {clinical.sex}")
    print(f"    Weight: {clinical.weight_kg} kg, Height: {clinical.height_cm} cm")
    print(f"    CrCl: {clinical.creatinine_clearance} mL/min")
    print(f"    ALT: {clinical.alt}, AST: {clinical.ast}, INR: {clinical.inr}")
    print(f"    Concurrent meds: {clinical.concurrent_medications}")

    # Use first patient from cohort as genomic profile
    sample_idx = 0
    sample_snps = dict(zip(
        cohort.snp_profiles.columns,
        cohort.snp_profiles.iloc[sample_idx].values.astype(int),
    ))
    sample_expression = dict(zip(
        cohort.gene_expression.columns,
        cohort.gene_expression.iloc[sample_idx].values,
    ))

    genomic = PatientGenomicProfile(
        snp_profile=sample_snps,
        gene_expression=sample_expression,
    )

    # Key pharmacogenomic variants for display
    key_variants = ["rs1799853", "rs1057910", "rs9923231"]
    print(f"\n  Key Pharmacogenomic Variants:")
    for v in key_variants:
        copies = sample_snps.get(v, 0)
        print(f"    {v}: {copies} copies")

    # ── Step 5: Generate Pharmacogenomic Alert ───────────────────────
    print_section("STEP 5: Generating Pharmacogenomic Alert", "red")

    alerts = engine.process_trigger(trigger, genomic, clinical)

    if not alerts:
        print("  ℹ No alerts generated (no matching rules)")
    else:
        for i, alert in enumerate(alerts):
            print(f"\n  Alert {i+1}/{len(alerts)}:")
            print(f"  {'─' * 60}")
            print(alert.to_soap_appendix())

            # Show JSON
            print_section("STEP 6: JSON Alert Output (for SOAP integration)", "teal")
            alert_json = json.loads(alert.model_dump_json())
            print(json.dumps(alert_json, indent=2))

    # ── Summary ──────────────────────────────────────────────────────
    print_section("DEMO COMPLETE", "mauve")
    print(f"  ✓ Trained on {n_patients} synthetic patients")
    print(f"  ✓ AUC-ROC:  {metrics['classifier']['val_auc_roc']:.4f}")
    print(f"  ✓ R² Score: {metrics['regressor']['val_r_squared']:.4f}")
    print(f"  ✓ Generated {len(alerts)} pharmacogenomic alert(s)")
    print(f"  ✓ Total time: {elapsed:.1f}s")
    print(f"\n  \033[38;2;166;227;161m⚕️ All outputs are SUGGESTIONS for clinician review only.\033[0m")
    print(f"  \033[38;2;166;227;161m   They are NOT automatic prescriptions.\033[0m\n")


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    print_banner()
    print("\n  🚀 Launching CARE'OS Helix Dashboard...\n")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "careos_genomics/dashboard.py",
        "--theme.backgroundColor=#1e1e2e",
        "--theme.secondaryBackgroundColor=#181825",
        "--theme.textColor=#cdd6f4",
        "--theme.primaryColor=#cba6f7",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARE'OS Module 3 Demo")
    parser.add_argument("--fast", action="store_true", help="Quick demo with fewer samples")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    args = parser.parse_args()

    if args.dashboard:
        launch_dashboard()
    else:
        run_demo(fast=args.fast)
