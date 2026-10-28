#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CARE'OS — Unified Clinical Workstation (Module 2 + Module 3)              ║
║  Catppuccin Mocha Themed Streamlit Dashboard                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

A single-pane clinical decision support interface combining:
  - Module 2: Audio capture → AI Scribe → 2nd-Pass Verification
  - Module 3: Genomics Engine → Pharmacogenomic Alerts → Dose Optimization

Run: streamlit run integrated_dashboard.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ═══════════════════════════════════════════════════════════════════════════
# Catppuccin Mocha Palette
# ═══════════════════════════════════════════════════════════════════════════

CATPPUCCIN = {
    "rosewater": "#f5e0dc", "flamingo": "#f2cdcd", "pink": "#f5c2e7",
    "mauve": "#cba6f7", "red": "#f38ba8", "maroon": "#eba0ac",
    "peach": "#fab387", "yellow": "#f9e2af", "green": "#a6e3a1",
    "teal": "#94e2d5", "sky": "#89dceb", "sapphire": "#74c7ec",
    "blue": "#89b4fa", "lavender": "#b4befe", "text": "#cdd6f4",
    "subtext1": "#bac2de", "subtext0": "#a6adc8", "overlay2": "#9399b2",
    "overlay1": "#7f849c", "overlay0": "#6c7086", "surface2": "#585b70",
    "surface1": "#45475a", "surface0": "#313244", "base": "#1e1e2e",
    "mantle": "#181825", "crust": "#11111b",
}


def inject_css():
    """Inject Catppuccin Mocha CSS theme."""
    st.markdown(f"""
    <style>
        /* ─── Global ─────────────────────────────────────── */
        .stApp {{
            background-color: {CATPPUCCIN['base']};
            color: {CATPPUCCIN['text']};
        }}
        .stApp header {{
            background-color: {CATPPUCCIN['mantle']} !important;
        }}
        .stSidebar > div:first-child {{
            background-color: {CATPPUCCIN['mantle']};
        }}

        /* ─── Cards ──────────────────────────────────────── */
        .careos-card {{
            background: {CATPPUCCIN['surface0']};
            border: 1px solid {CATPPUCCIN['surface1']};
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
        }}
        .careos-card-accent {{
            background: linear-gradient(135deg, {CATPPUCCIN['surface0']}, {CATPPUCCIN['mantle']});
            border: 1px solid {CATPPUCCIN['mauve']}40;
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1rem;
        }}

        /* ─── Metric Badges ──────────────────────────────── */
        .metric-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: {CATPPUCCIN['surface1']};
            border-radius: 8px;
            padding: 6px 14px;
            margin: 4px;
            font-size: 0.85rem;
        }}
        .metric-badge .value {{
            font-weight: 700;
            font-size: 1.1rem;
        }}

        /* ─── Alert Severity ─────────────────────────────── */
        .alert-critical {{
            background: {CATPPUCCIN['red']}20;
            border-left: 4px solid {CATPPUCCIN['red']};
            color: {CATPPUCCIN['red']};
            border-radius: 0 8px 8px 0;
            padding: 0.8rem 1rem;
            margin: 0.5rem 0;
        }}
        .alert-high {{
            background: {CATPPUCCIN['peach']}20;
            border-left: 4px solid {CATPPUCCIN['peach']};
            color: {CATPPUCCIN['peach']};
            border-radius: 0 8px 8px 0;
            padding: 0.8rem 1rem;
            margin: 0.5rem 0;
        }}
        .alert-moderate {{
            background: {CATPPUCCIN['yellow']}20;
            border-left: 4px solid {CATPPUCCIN['yellow']};
            color: {CATPPUCCIN['yellow']};
            border-radius: 0 8px 8px 0;
            padding: 0.8rem 1rem;
            margin: 0.5rem 0;
        }}

        /* ─── Pipeline Stage Indicator ───────────────────── */
        .stage-indicator {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 0;
        }}
        .stage-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }}
        .stage-dot.complete {{ background: {CATPPUCCIN['green']}; }}
        .stage-dot.active {{ background: {CATPPUCCIN['blue']}; animation: pulse 1s infinite; }}
        .stage-dot.pending {{ background: {CATPPUCCIN['surface2']}; }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.4; }}
        }}

        /* ─── DNA Helix Animation ────────────────────────── */
        .dna-header {{
            text-align: center;
            padding: 1rem 0;
        }}
        .dna-header h1 {{
            background: linear-gradient(90deg, {CATPPUCCIN['mauve']}, {CATPPUCCIN['pink']}, {CATPPUCCIN['sapphire']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.2rem;
            font-weight: 800;
        }}

        /* ─── SOAP Section Colors ────────────────────────── */
        .soap-s {{ color: {CATPPUCCIN['blue']}; }}
        .soap-o {{ color: {CATPPUCCIN['green']}; }}
        .soap-a {{ color: {CATPPUCCIN['peach']}; }}
        .soap-p {{ color: {CATPPUCCIN['mauve']}; }}

        /* ─── Tabs ───────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background: {CATPPUCCIN['mantle']};
            border-radius: 10px;
            padding: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {CATPPUCCIN['subtext0']};
            border-radius: 8px;
            padding: 8px 16px;
        }}
        .stTabs [aria-selected="true"] {{
            background: {CATPPUCCIN['surface0']};
            color: {CATPPUCCIN['mauve']};
        }}

        /* ─── Buttons ────────────────────────────────────── */
        .stButton > button {{
            background: linear-gradient(135deg, {CATPPUCCIN['mauve']}, {CATPPUCCIN['blue']});
            color: {CATPPUCCIN['base']};
            border: none;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
        }}
        .stButton > button:hover {{
            background: linear-gradient(135deg, {CATPPUCCIN['pink']}, {CATPPUCCIN['sapphire']});
            transform: translateY(-1px);
            box-shadow: 0 4px 12px {CATPPUCCIN['mauve']}40;
        }}
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Engine Caching
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_module2_pipeline():
    """Initialize and cache Module 2 pipeline."""
    from module2_careos.main import CareOSModule2Pipeline
    return CareOSModule2Pipeline()


@st.cache_resource
def get_module3_engine():
    """Initialize and cache Module 3 engine with pre-trained models."""
    from careos_genomics.engine import GenomicsEngine, EngineConfig
    from careos_genomics.synthetic_data import SyntheticPharmacogenomicDataGenerator

    config = EngineConfig(
        vae_latent_dim=16, vae_epochs=15, enable_vae=True,
        use_xgboost=True, xgb_n_trials=3, enable_shap=True,
    )
    engine = GenomicsEngine(config=config)
    gen = SyntheticPharmacogenomicDataGenerator(seed=42)
    cohort = gen.generate_mixed_cohort(n_patients=200)
    engine.train(
        snp_profiles=cohort.snp_profiles,
        clinical_data=cohort.clinical_data,
        gene_expression=cohort.gene_expression,
        response_labels=cohort.response_labels,
        optimal_doses=cohort.optimal_doses,
    )
    return engine


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render the sidebar with patient configuration."""
    with st.sidebar:
        st.markdown(f"""
        <div class="dna-header">
            <h1>CARE'OS</h1>
            <p style="color: {CATPPUCCIN['subtext1']}; font-size: 0.85rem;">
                Clinical Decision Support System
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown(f"### Patient Configuration")
        patient_id = st.text_input("Patient ID", value="PGX-DEMO-001")
        age = st.slider("Age", 18, 100, 65)
        sex = st.selectbox("Sex", ["M", "F"])
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 78.0)

        st.markdown("---")
        st.markdown(f"### Renal & Hepatic")
        creatinine = st.number_input("Creatinine Clearance (mL/min)", 10.0, 200.0, 85.0)
        alt = st.number_input("ALT (U/L)", 0.0, 500.0, 28.0)
        ast = st.number_input("AST (U/L)", 0.0, 500.0, 24.0)
        inr = st.number_input("INR", 0.5, 10.0, 1.1, step=0.1)

        st.markdown("---")
        st.markdown(f"### Genomic Profile")
        snp_preset = st.selectbox(
            "SNP Profile Preset",
            ["CYP2C19 Intermediate Metabolizer", "CYP2D6 Ultra-Rapid",
             "CYP2C9 Poor Metabolizer", "Normal / Wild-Type", "Custom"],
        )

        st.markdown("---")
        st.markdown(f"### Audio Configuration")
        language = st.selectbox("Language", ["en", "hinglish", "hi"])

        return {
            "patient_id": patient_id, "age": age, "sex": sex, "weight": weight,
            "creatinine": creatinine, "alt": alt, "ast": ast, "inr": inr,
            "snp_preset": snp_preset, "language": language,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Main Tabs
# ═══════════════════════════════════════════════════════════════════════════

def render_scribe_tab(config: dict):
    """Module 2: AI Scribe & Verifier tab."""
    st.markdown(f"""
    <div class="careos-card-accent">
        <h3 style="color: {CATPPUCCIN['sapphire']};">🎙️ Module 2 — AI Scribe & 2nd-Pass Verifier</h3>
        <p style="color: {CATPPUCCIN['subtext1']};">
            Upload clinical audio or use the demo encounter to generate a verified SOAP note
            with real-time CPIC pharmacogenomic trigger detection.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_audio = st.file_uploader("Upload Clinical Audio", type=["wav", "mp3", "m4a", "ogg"])
    with col2:
        use_demo = st.checkbox("Use Demo Baseline Audio", value=(uploaded_audio is None))

    if uploaded_audio and not use_demo:
        st.audio(uploaded_audio)
        st.caption(f"📁 Selected audio: **{uploaded_audio.name}** ({uploaded_audio.size / 1024:.1f} KB)")
    elif use_demo:
        st.info("ℹ️ Using baseline demo consultation audio (`demo.wav`).")

    if st.button("🚀 Run Complete Pipeline", use_container_width=True):
        # Determine audio path
        if uploaded_audio and not use_demo:
            timestamp_id = int(time.time())
            audio_path = Path(__file__).parent / "data" / "uploads" / f"{timestamp_id}_{uploaded_audio.name}"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            with open(audio_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())
        else:
            audio_path = Path(__file__).parent / "data" / "sample_audio" / "demo.wav"

        # Build clinical data for Module 3
        from careos_genomics.data_models import ClinicalFeatures, PatientGenomicProfile
        clinical_data = ClinicalFeatures(
            age=config["age"], sex=config["sex"], weight_kg=config["weight"],
            creatinine_clearance=config["creatinine"],
            alt=config["alt"], ast=config["ast"], inr=config["inr"],
        )

        # SNP presets
        snp_presets = {
            "CYP2C19 Intermediate Metabolizer": {"rs4244285": 1, "rs4986893": 0, "rs12248560": 0},
            "CYP2D6 Ultra-Rapid": {"rs3892097": 0, "rs1065852": 0, "rs16947": 2},
            "CYP2C9 Poor Metabolizer": {"rs1799853": 2, "rs1057910": 1},
            "Normal / Wild-Type": {"rs1799853": 0, "rs1057910": 0, "rs4244285": 0},
        }
        snp_profile = snp_presets.get(config["snp_preset"], {})
        genomic_profile = PatientGenomicProfile(
            snp_profile=snp_profile,
            gene_expression={"CYP2C9": 7.2, "CYP2C19": 6.8, "VKORC1": 8.1},
        )

        # Run pipeline with progress
        progress_bar = st.progress(0, "Initializing pipeline...")
        pipeline = get_module2_pipeline()

        progress_bar.progress(20, "Running 5-stage pipeline...")
        t0 = time.time()
        results = pipeline.process_encounter(
            audio_path=str(audio_path),
            patient_id=config["patient_id"],
            language=config["language"],
            clinical_data=clinical_data,
            genomic_profile=genomic_profile,
        )
        elapsed = time.time() - t0
        progress_bar.progress(100, f"Pipeline complete in {elapsed:.1f}s")

        # Store in session state
        st.session_state["pipeline_results"] = results
        st.session_state["pipeline_time"] = elapsed
        st.rerun()

    # Display results if available
    if "pipeline_results" in st.session_state:
        results = st.session_state["pipeline_results"]
        elapsed = st.session_state.get("pipeline_time", 0)

        # ─── Metrics Row ─────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        verif = results.get("verification", {})
        with m1:
            st.metric("Processing Time", f"{elapsed:.1f}s")
        with m2:
            st.metric("Total Claims", verif.get("total_claims", 0))
        with m3:
            st.metric("Hallucination Rate",
                       f"{verif.get('inference_aware_hallucination_rate', 0):.1f}%")
        with m4:
            st.metric("CPIC Triggers", len(results.get("cpic_triggers", [])))

        # ─── Diarized Transcript ─────────────────────────
        transcript = results.get("transcript", {})
        utterances = transcript.get("utterances", [])
        if utterances:
            with st.expander(f"🎙️ Diarized Transcript ({len(utterances)} utterances, {transcript.get('duration_seconds', 0):.1f}s)", expanded=False):
                for u in utterances:
                    speaker = u.get("speaker_role") or u.get("speaker_id", "Speaker")
                    text = u.get("text", "")
                    st.markdown(f"**[{speaker}]**: {text}")

        # ─── Extracted Fact Sheet ────────────────────────
        fact_sheet = results.get("fact_sheet", {})
        if fact_sheet:
            with st.expander("🩺 Extracted Clinical Fact Sheet (Pre-SOAP Grounding)", expanded=False):
                fc1, fc2, fc3 = st.columns(3)
                with fc1:
                    st.markdown(f"**Chief Complaint:** {fact_sheet.get('chief_complaint', 'N/A')}")
                    meds = fact_sheet.get("medications", [])
                    st.markdown(f"**Medications ({len(meds)}):**")
                    for m in meds:
                        st.markdown(f"- 💊 {m.get('drug_name', '')} {m.get('strength', '')} ({m.get('frequency', '')})")
                with fc2:
                    syms = fact_sheet.get("symptoms", [])
                    st.markdown(f"**Symptoms ({len(syms)}):**")
                    for s in syms:
                        st.markdown(f"- 🩺 {s.get('text', '')}")
                with fc3:
                    negs = fact_sheet.get("pertinent_negatives", [])
                    st.markdown(f"**Pertinent Negatives ({len(negs)}):**")
                    for n in negs:
                        st.markdown(f"- ❌ {n.get('text', '')}")

        # ─── SOAP Note ───────────────────────────────────
        st.markdown(f"### 📋 Verified SOAP Note")
        soap = results.get("soap_note", {})

        for section, label, css_class, icon in [
            ("subjective", "Subjective", "soap-s", "🗣️"),
            ("objective_exam", "Objective (Exam)", "soap-o", "🔍"),
            ("objective_results", "Objective (Results)", "soap-o", "🧪"),
            ("assessment_plan", "Assessment & Plan", "soap-a", "📊"),
        ]:
            claims = soap.get(section, [])
            if claims:
                with st.expander(f"{icon} {label} ({len(claims)} claims)", expanded=section == "assessment_plan"):
                    for claim in claims:
                        text = claim.get("claim_text", "")
                        if "[PGx Decision Support]" in text:
                            st.markdown(f"""
                            <div class="alert-moderate">
                                🧬 {text}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"- {text}")

        # ─── Verification Breakdown ──────────────────────
        st.markdown(f"### 🔬 Verification Breakdown")
        tier_data = {
            "Tier": ["T1 Verbatim", "T2a Terminology", "T2b Ontology",
                     "T3 Inference", "T4 Unsupported", "T5 Contradiction"],
            "Count": [
                verif.get("tier1_count", 0), verif.get("tier2a_count", 0),
                verif.get("tier2b_count", 0), verif.get("tier3_count", 0),
                verif.get("tier4_count", 0), verif.get("tier5_count", 0),
            ],
            "Color": [
                CATPPUCCIN["green"], CATPPUCCIN["teal"], CATPPUCCIN["sapphire"],
                CATPPUCCIN["yellow"], CATPPUCCIN["peach"], CATPPUCCIN["red"],
            ],
        }
        df_tiers = pd.DataFrame(tier_data)
        st.bar_chart(df_tiers.set_index("Tier")["Count"], color=CATPPUCCIN["mauve"])

        # ─── Co-Pilot ────────────────────────────────────
        copilot = results.get("copilot", {})
        if copilot.get("hypotheses"):
            st.markdown(f"### 🧠 Clinical Co-Pilot")
            for h in copilot["hypotheses"]:
                st.markdown(
                    f"- **{h.get('hypothesis', '')}** — "
                    f"Confidence: `{h.get('confidence', 0):.0%}`"
                )


def render_genomics_tab(config: dict):
    """Module 3: Standalone Genomics Engine tab."""
    st.markdown(f"""
    <div class="careos-card-accent">
        <h3 style="color: {CATPPUCCIN['mauve']};">🧬 Module 3 — Genomics-Informed Treatment Recommendation</h3>
        <p style="color: {CATPPUCCIN['subtext1']};">
            Attention-VAE SNP encoding → Cross-Attention Multi-Modal Fusion → 
            Optuna-Tuned XGBoost Classification → Conformal Dose Regression → CPIC Alerts
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Direct Module 3 trigger
    from careos_genomics.data_models import (
        TriggerPayload, ClinicalFeatures, PatientGenomicProfile,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Trigger Keys (from Module 2)")
        diagnosis_codes = st.multiselect(
            "ICD-10 Codes", ["I48.0", "I21.0", "K21.0", "E78.0", "M54.5", "Z79.899"],
            default=["I48.0"],
        )
        medication_names = st.multiselect(
            "Medications", ["warfarin", "clopidogrel", "omeprazole", "simvastatin",
                           "codeine", "carbamazepine", "allopurinol"],
            default=["warfarin"],
        )

    with col2:
        st.markdown("#### SNP Variants")
        snp_data = {}
        for rs_id, label in [
            ("rs1799853", "CYP2C9*2"), ("rs1057910", "CYP2C9*3"),
            ("rs9923231", "VKORC1"), ("rs4244285", "CYP2C19*2"),
            ("rs3892097", "CYP2D6*4"), ("rs4149056", "SLCO1B1*5"),
        ]:
            snp_data[rs_id] = st.select_slider(
                f"{label} ({rs_id})", options=[0, 1, 2], value=1,
                key=f"snp_{rs_id}",
            )

    if st.button("🧬 Run Genomics Engine", use_container_width=True):
        engine = get_module3_engine()

        trigger = TriggerPayload(
            patient_id=config["patient_id"],
            diagnosis_codes=diagnosis_codes,
            medication_names=medication_names,
            prescribing_physician="CAREOS-CLINICIAN",
        )
        clinical = ClinicalFeatures(
            age=config["age"], sex=config["sex"], weight_kg=config["weight"],
            creatinine_clearance=config["creatinine"],
            alt=config["alt"], ast=config["ast"], inr=config["inr"],
        )
        genomic = PatientGenomicProfile(
            snp_profile=snp_data,
            gene_expression={"CYP2C9": 7.2, "CYP2C19": 6.8, "VKORC1": 8.1},
        )

        with st.spinner("Processing pharmacogenomic analysis..."):
            t0 = time.time()
            alerts = engine.process_trigger(trigger, genomic, clinical)
            inference_time = time.time() - t0

        st.session_state["pgx_alerts"] = alerts
        st.session_state["pgx_time"] = inference_time
        st.rerun()

    # Display PGx results
    if "pgx_alerts" in st.session_state:
        alerts = st.session_state["pgx_alerts"]
        inference_time = st.session_state.get("pgx_time", 0)

        st.markdown(f"""
        <div class="careos-card">
            <span class="metric-badge">
                <span>Inference Time</span>
                <span class="value" style="color: {CATPPUCCIN['green']};">{inference_time*1000:.1f}ms</span>
            </span>
            <span class="metric-badge">
                <span>Alerts Generated</span>
                <span class="value" style="color: {CATPPUCCIN['mauve']};">{len(alerts)}</span>
            </span>
        </div>
        """, unsafe_allow_html=True)

        for alert in alerts:
            a = alert.model_dump() if hasattr(alert, "model_dump") else alert

            risk = a.get("risk_level", {})
            if isinstance(risk, dict):
                risk_val = risk.get("value", "moderate")
            elif hasattr(risk, "value"):
                risk_val = risk.value
            else:
                risk_val = str(risk)

            alert_class = "alert-critical" if risk_val in ("critical", "very_high") else \
                         "alert-high" if risk_val == "high" else "alert-moderate"

            st.markdown(f"""
            <div class="{alert_class}">
                <strong>🧬 Pharmacogenomic Alert</strong><br/>
                <strong>Drug:</strong> {a.get('drug_name', 'N/A')}<br/>
                <strong>Predicted Response:</strong> {a.get('predicted_response', 'N/A')}<br/>
                <strong>Metabolizer Type:</strong> {a.get('metabolizer_phenotype', 'N/A')}<br/>
                <strong>Suggested Dose:</strong> {a.get('suggested_starting_dose', 'N/A')}<br/>
                <strong>Confidence:</strong> {a.get('prediction_confidence', 'N/A')}
            </div>
            """, unsafe_allow_html=True)

            # CPIC Guidance
            guidance = a.get("cpic_guidance", {})
            if guidance:
                with st.expander("📖 CPIC Guideline Details"):
                    st.markdown(f"**Recommendation:** {guidance.get('recommendation_text', 'N/A')}")
                    st.markdown(f"**Monitoring:** {guidance.get('monitoring_parameters', 'N/A')}")
                    st.markdown(f"**Evidence Level:** {guidance.get('evidence_level', 'N/A')}")

            # SOAP Note Suggestion
            if hasattr(alert, "soap_note_suggestion"):
                with st.expander("📋 SOAP Note Suggestion"):
                    st.info(alert.soap_note_suggestion)

            # Detected Variants
            variants = a.get("detected_variants", [])
            if variants:
                with st.expander("🔬 Detected Variants"):
                    vdf = pd.DataFrame(variants)
                    st.dataframe(vdf, use_container_width=True)

            # Feature Importance
            feat_imp = a.get("feature_importance", {})
            if feat_imp and isinstance(feat_imp, dict):
                with st.expander("📊 Feature Importance (SHAP)"):
                    sorted_feats = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
                    feat_df = pd.DataFrame(sorted_feats, columns=["Feature", "Importance"])
                    st.bar_chart(feat_df.set_index("Feature"), color=CATPPUCCIN["teal"])


def render_alerts_tab():
    """Combined PGx Alerts view from pipeline results."""
    st.markdown(f"""
    <div class="careos-card-accent">
        <h3 style="color: {CATPPUCCIN['pink']};">🚨 Pharmacogenomic Decision Support Console</h3>
        <p style="color: {CATPPUCCIN['subtext1']};">
            All active alerts from the integrated pipeline. These are <strong>clinician suggestions</strong>,
            never automatic prescriptions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    results = st.session_state.get("pipeline_results", {})
    triggers = results.get("cpic_triggers", [])

    if not triggers:
        st.info("No active CPIC triggers. Run the pipeline from the Scribe tab first.")
        return

    for i, t in enumerate(triggers):
        alert = t.get("alert", {})
        pgx = t.get("pgx_alert", {})
        is_m3 = t.get("module3_active", False)

        risk = alert.get("risk_level", "moderate")
        alert_class = "alert-critical" if risk == "critical" else \
                     "alert-high" if risk == "high" else "alert-moderate"

        st.markdown(f"""
        <div class="{alert_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <strong style="font-size: 1.1rem;">
                    {'🧬' if is_m3 else '📋'} {alert.get('drug_name', 'Unknown')} — {alert.get('gene', '?')}
                </strong>
                <span class="metric-badge">
                    <span>{'ML Engine' if is_m3 else 'Watchlist'}</span>
                </span>
            </div>
            <br/>
            <strong>Phenotype:</strong> {alert.get('phenotype', 'N/A')}<br/>
            <strong>Risk Level:</strong> {risk.upper()}<br/>
            <strong>Action:</strong> {alert.get('clinical_action', 'N/A')}
        </div>
        """, unsafe_allow_html=True)

        if is_m3 and pgx:
            cols = st.columns(3)
            with cols[0]:
                st.metric("Predicted Response", pgx.get("predicted_response", "N/A"))
            with cols[1]:
                st.metric("Suggested Dose", pgx.get("suggested_starting_dose", "N/A"))
            with cols[2]:
                conf = pgx.get("prediction_confidence", "N/A")
                if isinstance(conf, (int, float)):
                    conf = f"{conf:.1%}"
                st.metric("Confidence", conf)

        if t.get("soap_suggestion"):
            with st.expander(f"📋 SOAP Suggestion for {alert.get('drug_name', '')}"):
                st.success(t["soap_suggestion"])

    # PGx Appendix
    appendix = results.get("pharmacogenomic_appendix")
    if appendix:
        st.markdown("---")
        st.markdown(f"### 📄 Full Pharmacogenomic SOAP Appendix")
        st.code(appendix, language="text")


def render_exports_tab():
    """FHIR, E/M Coding, and Patient Summary exports."""
    st.markdown(f"""
    <div class="careos-card-accent">
        <h3 style="color: {CATPPUCCIN['sky']};">📤 Post-Visit Artifact Exports</h3>
        <p style="color: {CATPPUCCIN['subtext1']};">
            FHIR R4 Bundle • E/M Coding Readiness • Patient After-Visit Summary
        </p>
    </div>
    """, unsafe_allow_html=True)

    results = st.session_state.get("pipeline_results", {})
    if not results:
        st.info("Run the pipeline first to generate exports.")
        return

    sub_tabs = st.tabs(["🏥 FHIR R4 Bundle", "💰 E/M Coding", "📝 Patient Summary"])

    with sub_tabs[0]:
        fhir = results.get("fhir_bundle", {})
        resources = fhir.get("entry", [])
        resource_types = {}
        for e in resources:
            rt = e.get("resource", {}).get("resourceType", "Unknown")
            resource_types[rt] = resource_types.get(rt, 0) + 1

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Resource Summary")
            for rt, count in resource_types.items():
                icon = {"Patient": "👤", "Encounter": "🏥", "Observation": "📊",
                       "Condition": "🩺", "MedicationRequest": "💊",
                       "DetectedIssue": "🧬"}.get(rt, "📋")
                color = CATPPUCCIN["mauve"] if rt == "DetectedIssue" else CATPPUCCIN["text"]
                st.markdown(f"{icon} **{rt}**: `{count}`")

        with col2:
            st.metric("Total Resources", len(resources))
            detected_issues = [
                e for e in resources
                if e.get("resource", {}).get("resourceType") == "DetectedIssue"
            ]
            if detected_issues:
                st.metric("PGx DetectedIssues", len(detected_issues))

        with st.expander("View Full FHIR Bundle JSON"):
            st.json(fhir)

    with sub_tabs[1]:
        em = results.get("em_coding", {})
        st.markdown(f"#### Suggested E/M Level: `{em.get('suggested_em_level', 'N/A')}`")
        for key, val in em.items():
            if key != "suggested_em_level":
                icon = "✅" if val else "❌"
                st.markdown(f"{icon} {key.replace('_', ' ').title()}: `{val}`")

    with sub_tabs[2]:
        summary_en = results.get("patient_summary_en", {})
        if summary_en:
            st.markdown("#### English After-Visit Summary")
            st.markdown(f"**Assessment:** {summary_en.get('assessment_summary', '')}")
            st.markdown(f"**Plan:** {summary_en.get('plan_summary', '')}")
            st.markdown("**Medications:**")
            for med in summary_en.get("medications_list", []):
                st.markdown(f"- {med}")
            st.markdown("**Red Flag Warnings:**")
            for rf in summary_en.get("red_flag_warnings", []):
                if "Genetic Alert" in rf:
                    st.warning(rf)
                else:
                    st.markdown(f"- {rf}")


# ═══════════════════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="CARE'OS — Clinical Decision Support",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    # Header
    st.markdown(f"""
    <div class="dna-header">
        <h1>CARE'OS Clinical Workstation</h1>
        <p style="color: {CATPPUCCIN['subtext1']}; margin-top: -0.5rem;">
            Module 2 (AI Scribe & Verifier) + Module 3 (Genomics Engine) — Unified Interface
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    config = render_sidebar()

    # Main tabs
    tabs = st.tabs([
        "🎙️ AI Scribe Pipeline",
        "🧬 Genomics Engine",
        "🚨 PGx Alerts",
        "📤 Exports",
    ])

    with tabs[0]:
        render_scribe_tab(config)
    with tabs[1]:
        render_genomics_tab(config)
    with tabs[2]:
        render_alerts_tab()
    with tabs[3]:
        render_exports_tab()


if __name__ == "__main__":
    main()
