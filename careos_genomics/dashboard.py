"""
CARE'OS Module 3 — Catppuccin-Themed Streamlit Dashboard.

A beautiful, interactive pharmacogenomics decision-support dashboard
featuring the Catppuccin Mocha color palette for a premium clinical UI.

Run with: streamlit run careos_genomics/dashboard.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from careos_genomics.data_models import (
    ClinicalFeatures,
    PatientGenomicProfile,
    TriggerPayload,
)
from careos_genomics.engine import GenomicsEngine, EngineConfig
from careos_genomics.knowledge_base import CPICKnowledgeBase
from careos_genomics.synthetic_data import SyntheticPharmacogenomicDataGenerator


# ═══════════════════════════════════════════════════════════════════════════
# Catppuccin Mocha Color Palette
# ═══════════════════════════════════════════════════════════════════════════

CATPPUCCIN = {
    "rosewater": "#f5e0dc",
    "flamingo":  "#f2cdcd",
    "pink":      "#f5c2e7",
    "mauve":     "#cba6f7",
    "red":       "#f38ba8",
    "maroon":    "#eba0ac",
    "peach":     "#fab387",
    "yellow":    "#f9e2af",
    "green":     "#a6e3a1",
    "teal":      "#94e2d5",
    "sky":       "#89dceb",
    "sapphire":  "#74c7ec",
    "blue":      "#89b4fa",
    "lavender":  "#b4befe",
    "text":      "#cdd6f4",
    "subtext1":  "#bac2de",
    "subtext0":  "#a6adc8",
    "overlay2":  "#9399b2",
    "overlay1":  "#7f849c",
    "overlay0":  "#6c7086",
    "surface2":  "#585b70",
    "surface1":  "#45475a",
    "surface0":  "#313244",
    "base":      "#1e1e2e",
    "mantle":    "#181825",
    "crust":     "#11111b",
}


def get_catppuccin_css() -> str:
    """Generate comprehensive Catppuccin Mocha CSS for Streamlit."""
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global Styles ────────────────────────────────────────────── */
    .stApp {{
        background-color: {CATPPUCCIN['base']};
        color: {CATPPUCCIN['text']};
        font-family: 'Inter', sans-serif;
    }}

    .stApp > header {{
        background-color: {CATPPUCCIN['mantle']};
    }}

    /* ── Sidebar ──────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background-color: {CATPPUCCIN['mantle']};
        border-right: 1px solid {CATPPUCCIN['surface0']};
    }}

    [data-testid="stSidebar"] .stMarkdown p {{
        color: {CATPPUCCIN['subtext1']};
    }}

    /* ── Text Elements ────────────────────────────────────────────── */
    h1 {{
        color: {CATPPUCCIN['mauve']} !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
    }}

    h2 {{
        color: {CATPPUCCIN['blue']} !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }}

    h3 {{
        color: {CATPPUCCIN['sapphire']} !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }}

    p, span, label {{
        color: {CATPPUCCIN['text']} !important;
    }}

    /* ── Cards / Containers ───────────────────────────────────────── */
    [data-testid="stExpander"] {{
        background-color: {CATPPUCCIN['surface0']};
        border: 1px solid {CATPPUCCIN['surface1']};
        border-radius: 12px;
    }}

    .stAlert {{
        background-color: {CATPPUCCIN['surface0']};
        border-radius: 8px;
    }}

    /* ── Metrics ──────────────────────────────────────────────────── */
    [data-testid="stMetric"] {{
        background-color: {CATPPUCCIN['surface0']};
        padding: 16px;
        border-radius: 12px;
        border: 1px solid {CATPPUCCIN['surface1']};
    }}

    [data-testid="stMetricLabel"] {{
        color: {CATPPUCCIN['subtext0']} !important;
        font-size: 0.85rem !important;
    }}

    [data-testid="stMetricValue"] {{
        color: {CATPPUCCIN['text']} !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* ── Buttons ──────────────────────────────────────────────────── */
    .stButton > button {{
        background-color: {CATPPUCCIN['mauve']};
        color: {CATPPUCCIN['crust']};
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }}

    .stButton > button:hover {{
        background-color: {CATPPUCCIN['pink']};
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(203, 166, 247, 0.3);
    }}

    /* ── Inputs ───────────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {{
        background-color: {CATPPUCCIN['surface0']} !important;
        color: {CATPPUCCIN['text']} !important;
        border: 1px solid {CATPPUCCIN['surface2']} !important;
        border-radius: 8px !important;
    }}

    .stMultiSelect > div > div {{
        background-color: {CATPPUCCIN['surface0']} !important;
        border: 1px solid {CATPPUCCIN['surface2']} !important;
        border-radius: 8px !important;
    }}

    /* ── Tabs ─────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {CATPPUCCIN['mantle']};
        border-radius: 12px;
        padding: 4px;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {CATPPUCCIN['subtext0']} !important;
        border-radius: 8px;
        font-weight: 500;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {CATPPUCCIN['surface0']} !important;
        color: {CATPPUCCIN['mauve']} !important;
    }}

    /* ── DataFrames / Tables ──────────────────────────────────────── */
    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
    }}

    [data-testid="stDataFrameContainer"] {{
        background-color: {CATPPUCCIN['surface0']};
        border-radius: 12px;
    }}

    /* ── Code Blocks ─────────────────────────────────────────────── */
    .stCodeBlock {{
        background-color: {CATPPUCCIN['mantle']} !important;
        border: 1px solid {CATPPUCCIN['surface0']} !important;
        border-radius: 8px;
    }}

    code {{
        font-family: 'JetBrains Mono', monospace !important;
        color: {CATPPUCCIN['green']} !important;
    }}

    /* ── Progress Bar ────────────────────────────────────────────── */
    .stProgress > div > div > div {{
        background-color: {CATPPUCCIN['mauve']};
    }}

    /* ── Divider ─────────────────────────────────────────────────── */
    hr {{
        border-color: {CATPPUCCIN['surface1']} !important;
    }}

    /* ── Custom Alert Cards ──────────────────────────────────────── */
    .alert-critical {{
        background: linear-gradient(135deg, {CATPPUCCIN['surface0']}, rgba(243,139,168,0.1));
        border-left: 4px solid {CATPPUCCIN['red']};
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    }}

    .alert-high {{
        background: linear-gradient(135deg, {CATPPUCCIN['surface0']}, rgba(250,179,135,0.1));
        border-left: 4px solid {CATPPUCCIN['peach']};
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    }}

    .alert-moderate {{
        background: linear-gradient(135deg, {CATPPUCCIN['surface0']}, rgba(249,226,175,0.1));
        border-left: 4px solid {CATPPUCCIN['yellow']};
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    }}

    .alert-info {{
        background: linear-gradient(135deg, {CATPPUCCIN['surface0']}, rgba(137,180,250,0.1));
        border-left: 4px solid {CATPPUCCIN['blue']};
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    }}

    .genomic-card {{
        background-color: {CATPPUCCIN['surface0']};
        border: 1px solid {CATPPUCCIN['surface1']};
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
    }}

    .metric-highlight {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, {CATPPUCCIN['mauve']}, {CATPPUCCIN['blue']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    /* ── Scrollbar ────────────────────────────────────────────────── */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: {CATPPUCCIN['mantle']};
    }}

    ::-webkit-scrollbar-thumb {{
        background: {CATPPUCCIN['surface2']};
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {CATPPUCCIN['overlay0']};
    }}
    </style>
    """


def render_header():
    """Render the dashboard header."""
    st.markdown(get_catppuccin_css(), unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0; color: {CATPPUCCIN['mauve']} !important;">
            🧬 CARE'OS <span style="color: {CATPPUCCIN['blue']};">Helix</span>
        </h1>
        <p style="font-size: 1.1rem; color: {CATPPUCCIN['subtext0']} !important; margin-top: 8px;">
            Module 3 — Genomics-Informed Treatment Recommendation Engine
        </p>
        <p style="font-size: 0.8rem; color: {CATPPUCCIN['overlay1']} !important; margin-top: 4px;">
            v3.0.0 • Clinical Decision Support • For Clinician Review Only
        </p>
    </div>
    <hr style="border-color: {CATPPUCCIN['surface1']}; margin: 10px 0 20px 0;">
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px 0;">
            <span style="font-size: 2rem;">🧬</span>
            <h3 style="color: {CATPPUCCIN['mauve']} !important; margin: 5px 0;">Configuration</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown(f"<h4 style='color: {CATPPUCCIN['blue']} !important;'>⚙️ Model Settings</h4>", unsafe_allow_html=True)

        use_vae = st.checkbox("Use VAE (vs PCA)", value=True, help="Variational Autoencoder for SNP compression")
        latent_dim = st.slider("Latent Dimension", 8, 64, 16, step=8)
        vae_epochs = st.slider("VAE Epochs", 5, 100, 25, step=5) if use_vae else 0
        use_xgboost = st.checkbox("Use XGBoost (vs RF)", value=True)
        n_optuna_trials = st.slider("Optuna Trials", 0, 50, 10, step=5) if use_xgboost else 0
        enable_shap = st.checkbox("Enable SHAP", value=True)

        st.markdown("---")

        st.markdown(f"<h4 style='color: {CATPPUCCIN['blue']} !important;'>📊 Data Generation</h4>", unsafe_allow_html=True)

        n_patients = st.slider("Training Patients", 100, 1000, 300, step=50)
        drug_scenario = st.selectbox(
            "Drug Scenario",
            ["Warfarin (CYP2C9/VKORC1)", "Clopidogrel (CYP2C19)", "Codeine (CYP2D6)", "Simvastatin (SLCO1B1)", "Mixed Cohort"],
        )

        st.markdown("---")

        st.markdown(f"""
        <div style="background-color: {CATPPUCCIN['surface0']}; border-radius: 8px; padding: 12px; margin-top: 10px;">
            <p style="font-size: 0.75rem; color: {CATPPUCCIN['overlay1']} !important; text-align: center;">
                ⚕️ Decision Support Only<br>
                Not a clinical device or prescription system
            </p>
        </div>
        """, unsafe_allow_html=True)

        return {
            "use_vae": use_vae,
            "latent_dim": latent_dim,
            "vae_epochs": vae_epochs,
            "use_xgboost": use_xgboost,
            "n_optuna_trials": n_optuna_trials,
            "enable_shap": enable_shap,
            "n_patients": n_patients,
            "drug_scenario": drug_scenario,
        }


def render_knowledge_base_tab():
    """Render the CPIC Knowledge Base explorer."""
    kb = CPICKnowledgeBase()

    st.markdown(f"""
    <div class="genomic-card">
        <h3 style="color: {CATPPUCCIN['mauve']} !important; margin-top: 0;">📚 CPIC Pharmacogenomic Knowledge Base</h3>
        <p style="color: {CATPPUCCIN['subtext0']} !important;">
            {len(kb)} curated gene-drug interaction rules based on CPIC guidelines
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rules", len(kb))
    with col2:
        st.metric("Supported Drugs", len(kb.get_supported_drugs()))
    with col3:
        st.metric("Gene Panels", len(kb.get_supported_genes()))

    st.markdown("### Rule Details")

    for rule in kb.get_all_rules():
        severity_colors = {
            "critical": CATPPUCCIN['red'],
            "high": CATPPUCCIN['peach'],
            "moderate": CATPPUCCIN['yellow'],
            "informational": CATPPUCCIN['blue'],
        }

        worst_severity = "informational"
        for sev in rule.metabolizer_to_severity.values():
            if sev.value == "critical":
                worst_severity = "critical"
                break
            elif sev.value == "high" and worst_severity != "critical":
                worst_severity = "high"
            elif sev.value == "moderate" and worst_severity not in ("critical", "high"):
                worst_severity = "moderate"

        color = severity_colors.get(worst_severity, CATPPUCCIN['blue'])

        with st.expander(f"🧬 {rule.gene_name} → {', '.join(sorted(rule.drug_names)[:3])}", expanded=False):
            st.markdown(f"""
            <div style="border-left: 3px solid {color}; padding-left: 15px;">
                <p><strong>Rule ID:</strong> <code>{rule.rule_id}</code></p>
                <p><strong>Gene:</strong> {rule.gene_name}</p>
                <p><strong>Drugs:</strong> {', '.join(sorted(rule.drug_names))}</p>
                <p><strong>Variants:</strong> {', '.join(sorted(rule.variant_ids))}</p>
                <p><strong>CPIC Guideline:</strong> {rule.cpic_guideline_version}</p>
                <p><strong>Evidence Level:</strong> {rule.cpic_evidence_level}</p>
                <p><strong>Source:</strong> <a href="{rule.source_url}" target="_blank" style="color: {CATPPUCCIN['sapphire']};">{rule.source_url}</a></p>
            </div>
            """, unsafe_allow_html=True)

            if rule.cpic_recommendation:
                st.markdown("**CPIC Recommendations by Metabolizer Phenotype:**")
                for pheno, rec in rule.cpic_recommendation.items():
                    pheno_color = CATPPUCCIN['green']
                    if "poor" in pheno.value:
                        pheno_color = CATPPUCCIN['red']
                    elif "intermediate" in pheno.value:
                        pheno_color = CATPPUCCIN['yellow']
                    elif "ultrarapid" in pheno.value:
                        pheno_color = CATPPUCCIN['peach']

                    st.markdown(f"""
                    <div style="background-color: {CATPPUCCIN['surface0']}; border-left: 3px solid {pheno_color};
                                padding: 10px 15px; margin: 5px 0; border-radius: 0 8px 8px 0;">
                        <strong style="color: {pheno_color};">{pheno.value.replace('_', ' ').title()}</strong><br>
                        <span style="color: {CATPPUCCIN['subtext1']}; font-size: 0.9rem;">{rec}</span>
                    </div>
                    """, unsafe_allow_html=True)


def render_training_tab(config: dict):
    """Render the model training interface."""
    st.markdown(f"""
    <div class="genomic-card">
        <h3 style="color: {CATPPUCCIN['mauve']} !important; margin-top: 0;">🔬 Model Training Pipeline</h3>
        <p style="color: {CATPPUCCIN['subtext0']} !important;">
            Train the genomics engine on synthetic pharmacogenomic data
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Train Genomics Engine", use_container_width=True):
        _run_training(config)


def _run_training(config: dict):
    """Execute the training pipeline with live progress."""
    progress = st.progress(0, text="Initializing...")

    # Generate data
    progress.progress(5, text="🧬 Generating synthetic pharmacogenomic data...")
    generator = SyntheticPharmacogenomicDataGenerator(seed=42, n_background_snps=100, n_expression_genes=30)

    scenario = config["drug_scenario"]
    n = config["n_patients"]

    if "Warfarin" in scenario:
        cohort = generator.generate_warfarin_cohort(n)
    elif "Clopidogrel" in scenario:
        cohort = generator.generate_clopidogrel_cohort(n)
    elif "Codeine" in scenario:
        cohort = generator.generate_codeine_cohort(n)
    elif "Simvastatin" in scenario:
        cohort = generator.generate_simvastatin_cohort(n)
    else:
        cohort = generator.generate_mixed_cohort(n)

    progress.progress(20, text="📊 Data generated! Configuring engine...")

    engine_config = EngineConfig(
        vae_latent_dim=config["latent_dim"],
        vae_epochs=config["vae_epochs"],
        enable_vae=config["use_vae"],
        use_xgboost=config["use_xgboost"],
        xgb_n_trials=config["n_optuna_trials"],
        enable_shap=config["enable_shap"],
        pca_components=config["latent_dim"],
    )

    engine = GenomicsEngine(config=engine_config)
    st.session_state["engine"] = engine
    st.session_state["cohort"] = cohort

    progress.progress(35, text="🧠 Training models...")

    metrics = engine.train(
        snp_profiles=cohort.snp_profiles,
        clinical_data=cohort.clinical_data,
        gene_expression=cohort.gene_expression,
        response_labels=cohort.response_labels,
        optimal_doses=cohort.optimal_doses,
    )

    progress.progress(100, text="✅ Training complete!")
    st.session_state["metrics"] = metrics

    time.sleep(0.3)
    progress.empty()

    st.success("🎉 Genomics Engine trained successfully!")
    st.code(engine.get_training_summary(), language="text")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        auc = metrics["classifier"]["val_auc_roc"]
        st.metric("AUC-ROC", f"{auc:.4f}")
    with col2:
        r2 = metrics["regressor"]["val_r_squared"]
        st.metric("R² Score", f"{r2:.4f}")
    with col3:
        st.metric("RMSE", f"{metrics['regressor']['val_rmse']:.4f}")
    with col4:
        st.metric("Training Time", f"{metrics['training_time_seconds']:.1f}s")


def render_prediction_tab():
    """Render the clinical prediction interface."""
    st.markdown(f"""
    <div class="genomic-card">
        <h3 style="color: {CATPPUCCIN['mauve']} !important; margin-top: 0;">⚕️ Patient Pharmacogenomic Analysis</h3>
        <p style="color: {CATPPUCCIN['subtext0']} !important;">
            Simulate a Module 2 trigger and generate pharmacogenomic alerts
        </p>
    </div>
    """, unsafe_allow_html=True)

    engine = st.session_state.get("engine")
    cohort = st.session_state.get("cohort")

    if engine is None or not engine._is_trained:
        st.warning("⚠️ Please train the engine first in the Training tab.")
        return

    st.markdown("### 🏥 Module 2 Trigger Simulation")

    col1, col2 = st.columns(2)
    with col1:
        diagnosis_input = st.text_input(
            "ICD-10 Diagnosis Codes (comma-separated)",
            value="I48.0",
            help="Enter ICD-10 codes, e.g., I48.0 for atrial fibrillation"
        )
        patient_id = st.text_input("Patient ID", value="PAT-DEMO-001")

    with col2:
        medication_input = st.text_input(
            "Medication Names (comma-separated)",
            value="warfarin",
            help="Enter medication names, e.g., warfarin, clopidogrel"
        )
        physician_id = st.text_input("Prescribing Physician", value="DR-SMITH-001")

    st.markdown("### 👤 Clinical Data")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", 18, 100, 65)
        sex = st.selectbox("Sex", ["M", "F"])
    with col2:
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 75.0)
        height = st.number_input("Height (cm)", 140.0, 210.0, 170.0)
    with col3:
        crcl = st.number_input("Creatinine Clearance", 10.0, 150.0, 85.0)
        alt_val = st.number_input("ALT (U/L)", 5.0, 200.0, 25.0)
    with col4:
        ast_val = st.number_input("AST (U/L)", 5.0, 200.0, 22.0)
        inr_val = st.number_input("INR", 0.8, 5.0, 1.0)

    concurrent_meds = st.multiselect(
        "Concurrent Medications",
        ["amiodarone", "fluconazole", "rifampin", "aspirin", "ibuprofen", "omeprazole", "metoprolol"],
        default=["amiodarone"],
    )

    if st.button("🧬 Generate Pharmacogenomic Alert", use_container_width=True):
        try:
            diagnosis_codes = [c.strip() for c in diagnosis_input.split(",") if c.strip()]
            medication_names = [m.strip() for m in medication_input.split(",") if m.strip()]

            trigger = TriggerPayload(
                patient_id=patient_id,
                diagnosis_codes=diagnosis_codes,
                medication_names=medication_names,
                prescribing_physician=physician_id,
            )

            clinical = ClinicalFeatures(
                age=age,
                sex=sex,
                weight_kg=weight,
                height_cm=height,
                creatinine_clearance=crcl,
                alt=alt_val,
                ast=ast_val,
                inr=inr_val,
                concurrent_medications=concurrent_meds,
            )

            sample_snps = dict(zip(
                cohort.snp_profiles.columns,
                cohort.snp_profiles.iloc[0].values.astype(int),
            ))
            sample_expression = dict(zip(
                cohort.gene_expression.columns,
                cohort.gene_expression.iloc[0].values,
            ))

            genomic_profile = PatientGenomicProfile(
                snp_profile=sample_snps,
                gene_expression=sample_expression,
            )

            with st.spinner("Processing pharmacogenomic analysis..."):
                alerts = engine.process_trigger(trigger, genomic_profile, clinical)

            if not alerts:
                st.info("ℹ️ No pharmacogenomic rules matched for the given diagnosis codes and medications.")
            else:
                for alert in alerts:
                    _render_alert(alert)

        except Exception as e:
            st.error(f"Error generating alert: {e}")
            import traceback
            st.code(traceback.format_exc(), language="python")


def _render_alert(alert):
    """Render a pharmacogenomic alert with Catppuccin styling."""
    severity_class = {
        "critical": "alert-critical",
        "high": "alert-high",
        "moderate": "alert-moderate",
        "informational": "alert-info",
    }.get(alert.severity.value, "alert-info")

    severity_emoji = {
        "critical": "🔴",
        "high": "🟠",
        "moderate": "🟡",
        "informational": "🔵",
    }.get(alert.severity.value, "🔵")

    st.markdown(f"""
    <div class="{severity_class}">
        <h3 style="margin-top: 0;">{severity_emoji} Pharmacogenomic Alert — {alert.severity.value.upper()}</h3>
        <p><strong>Alert ID:</strong> <code>{alert.alert_id}</code></p>
        <p><strong>Patient:</strong> {alert.patient_id} | <strong>Consultation:</strong> {alert.consultation_id}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🧬 Genomic Findings")
        for v in alert.detected_variants:
            st.markdown(f"**{v.gene}** `{v.variant_id}` — {v.copies_of_variant} copies")
        st.markdown(f"**Metabolizer:** {alert.metabolizer_phenotype.value.replace('_', ' ').title()}")
        st.markdown(f"**Risk Level:** {alert.risk_level.value.upper()}")

    with col2:
        st.markdown("#### 📊 Prediction")
        pred = alert.drug_response_prediction
        st.markdown(f"**Predicted:** {pred.predicted_category.value}")
        st.markdown(f"**Confidence:** {pred.confidence:.1%}")
        for cat, prob in pred.probabilities.items():
            color = CATPPUCCIN['green'] if cat == "responder" else (
                CATPPUCCIN['red'] if cat == "adverse_event_risk" else CATPPUCCIN['yellow']
            )
            st.markdown(f"<span style='color: {color};'>● {cat}: {prob:.3f}</span>", unsafe_allow_html=True)

    with col3:
        st.markdown("#### 💊 Dose Estimate")
        if alert.dose_estimate:
            de = alert.dose_estimate
            st.markdown(f"**Predicted:** {de.predicted_dose} {de.unit}")
            st.markdown(f"**{de.confidence_level:.0%} CI:** [{de.confidence_interval_lower}, {de.confidence_interval_upper}]")
        else:
            st.markdown("*No dose estimate available*")

    with st.expander("📚 CPIC Guideline Recommendation", expanded=True):
        st.markdown(f"""
        <div style="background-color: {CATPPUCCIN['surface0']}; padding: 15px; border-radius: 8px;">
            <p><strong>Guideline:</strong> {alert.cpic_guidance.guideline_id or 'N/A'} ({alert.cpic_guidance.guideline_version or 'N/A'})</p>
            <p><strong>Evidence:</strong> {alert.cpic_guidance.evidence_level or 'N/A'}</p>
            <p style="margin-top: 10px;">{alert.cpic_guidance.recommendation_text}</p>
            <p style="margin-top: 10px; color: {CATPPUCCIN['yellow']};"><strong>Monitoring:</strong> {alert.cpic_guidance.monitoring_advice}</p>
        </div>
        """, unsafe_allow_html=True)

    if alert.drug_interaction_warnings:
        with st.expander(f"⚠️ Drug Interaction Warnings ({len(alert.drug_interaction_warnings)})", expanded=True):
            for w in alert.drug_interaction_warnings:
                st.markdown(f"""
                <div style="background-color: rgba(243,139,168,0.1); border-left: 3px solid {CATPPUCCIN['red']};
                            padding: 10px 15px; margin: 5px 0; border-radius: 0 8px 8px 0;">
                    <strong>{w.interacting_entity}</strong> [{w.severity.value}]<br>
                    <span style="color: {CATPPUCCIN['subtext1']};">{w.description}</span>
                </div>
                """, unsafe_allow_html=True)

    if alert.explainability:
        with st.expander("🔍 SHAP Explainability Report"):
            st.markdown(f"*{alert.explainability.explanation_text}*")
            for f in alert.explainability.top_features:
                direction_color = CATPPUCCIN['red'] if f.direction == "increases_risk" else CATPPUCCIN['green']
                st.markdown(f"""
                <span style="color: {direction_color};">{'▲' if f.direction == 'increases_risk' else '▼'}</span>
                **{f.feature_name}** — SHAP: {f.shap_value:.4f} ({f.magnitude})
                """, unsafe_allow_html=True)

    if alert.clinical_action_items:
        with st.expander("📋 Suggested Clinical Actions"):
            for i, item in enumerate(alert.clinical_action_items, 1):
                st.markdown(f"{i}. {item}")

    with st.expander("📄 SOAP Note Appendix"):
        st.code(alert.to_soap_appendix(), language="text")

    with st.expander("🔧 Raw JSON Alert"):
        st.json(json.loads(alert.model_dump_json()))

    st.markdown(f"""
    <div style="background-color: {CATPPUCCIN['surface0']}; border: 1px solid {CATPPUCCIN['surface2']};
                border-radius: 8px; padding: 12px; text-align: center; margin-top: 15px;">
        <p style="font-size: 0.8rem; color: {CATPPUCCIN['overlay1']} !important;">
            {alert.safety_notice}
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="CARE'OS Helix — Genomics Engine",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_header()
    sidebar_config = render_sidebar()

    tab1, tab2, tab3 = st.tabs([
        "🔬 Model Training",
        "⚕️ Patient Analysis",
        "📚 Knowledge Base",
    ])

    with tab1:
        render_training_tab(sidebar_config)

    with tab2:
        render_prediction_tab()

    with tab3:
        render_knowledge_base_tab()


if __name__ == "__main__":
    main()
