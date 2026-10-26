"""
CARE'OS Module 2 — Professional Streamlit Dashboard
=====================================================
Ambient AI Scribe with 2nd-Pass Clinical Verifier

Features:
  - Training Results Dashboard (view training curves, metrics, confusion matrices)
  - Live Clinical NLI Verification (test the trained MedNLI model)
  - SOAP Section Classification (test the SOAP classifier)
  - Drug Name ASR Correction (test the T5 drug corrector)
  - Full Pipeline Demo (end-to-end SOAP note generation & verification)
"""

import streamlit as st
import json
import os
import sys
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="CARE'OS Module 2 — AI Clinical Scribe",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "training_results"
DATA_DIR = BASE_DIR / "data"

# ─── Custom CSS for Professional Light Theme ──────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0f766e 0%, #0d9488 50%, #14b8a6 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 20px rgba(15, 118, 110, 0.25);
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
        font-weight: 400;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0f766e;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.25rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0d9488;
        display: inline-block;
    }

    /* SOAP section badges */
    .soap-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .soap-subjective { background: #dbeafe; color: #1e40af; }
    .soap-objective { background: #dcfce7; color: #166534; }
    .soap-assessment { background: #fef3c7; color: #92400e; }
    .soap-plan { background: #fce7f3; color: #9d174d; }

    /* Verification tier badges */
    .tier-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.15rem;
    }
    .tier-1 { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .tier-2a { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd; }
    .tier-2b { background: #e0e7ff; color: #3730a3; border: 1px solid #a5b4fc; }
    .tier-3 { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
    .tier-4 { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .tier-5 { background: #fecdd3; color: #881337; border: 1px solid #fb7185; }

    /* Info box */
    .info-box {
        background: #f0fdfa;
        border: 1px solid #99f6e4;
        border-radius: 10px;
        padding: 1.25rem;
        margin: 1rem 0;
        color: #134e4a;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Result card */
    .result-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .result-card h4 {
        margin: 0 0 0.75rem 0;
        color: #1f2937;
        font-weight: 600;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0fdfa 0%, #ffffff 100%);
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 500;
        color: #374151;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.25rem;
        font-weight: 500;
    }

    /* Code blocks */
    code {
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def load_training_results(model_name: str) -> dict | None:
    """Load training results JSON for a given model."""
    paths = {
        "mednli": RESULTS_DIR / "mednli_verifier" / "training_results.json",
        "soap": RESULTS_DIR / "soap_classifier" / "training_history.json",
        "drug": RESULTS_DIR / "drug_corrector" / "drug_corrector_history.json",
    }
    path = paths.get(model_name)
    if path and path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def check_model_exists(model_name: str) -> bool:
    """Check if a trained model checkpoint exists."""
    paths = {
        "mednli": CHECKPOINTS_DIR / "mednli_verifier" / "best_model",
        "soap": CHECKPOINTS_DIR / "soap_classifier" / "best_model",
        "drug": CHECKPOINTS_DIR / "drug_corrector" / "best_model",
    }
    path = paths.get(model_name)
    return path is not None and path.exists()


@st.cache_resource
def load_mednli_model():
    """Load the fine-tuned MedNLI verifier model."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    model_path = str(CHECKPOINTS_DIR / "mednli_verifier" / "best_model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource
def load_soap_model():
    """Load the fine-tuned SOAP classifier model."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    model_path = str(CHECKPOINTS_DIR / "soap_classifier" / "best_model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource
def load_drug_model():
    """Load the fine-tuned T5 drug corrector model."""
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch
    model_path = str(CHECKPOINTS_DIR / "drug_corrector" / "best_model")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


# ─── Sidebar Navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size: 2.5rem;">🏥</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #0f766e; margin-top: 0.5rem;">CARE'OS</div>
        <div style="font-size: 0.8rem; color: #6b7280; font-weight: 500;">Module 2 — AI Clinical Scribe</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📊 Training Dashboard",
            "🔬 Clinical NLI Verifier",
            "📋 SOAP Classifier",
            "💊 Drug Name Corrector",
            "📝 Full Pipeline Demo",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Model status indicators
    st.markdown("**Model Status**")
    models = {"mednli": "NLI Verifier", "soap": "SOAP Classifier", "drug": "Drug Corrector"}
    for key, name in models.items():
        status = "✅" if check_model_exists(key) else "⏳"
        st.markdown(f"{status} {name}")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.75rem; color:#9ca3af; text-align:center;">'
        'CSET343 — AI In Healthcare<br>Bennett University</div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown("""
    <div class="main-header">
        <h1>🏥 CARE'OS Module 2</h1>
        <p>Ambient AI Scribe with Dual-Tier 2nd-Pass Clinical Verifier</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>What is Module 2?</strong><br>
        An ontology-grounded ambient clinical intelligence pipeline that converts doctor-patient 
        conversations into verified SOAP notes. It separates clinical abstraction from factual 
        verification using a cascaded Dual-Tier 2nd-Pass Verifier that reduces false-positive 
        hallucination flags from 35.2% to 9.1% — closely matching human physician benchmarks (10.4%).
    </div>
    """, unsafe_allow_html=True)

    # Architecture overview
    st.markdown('<div class="section-header">📐 Five-Stage Pipeline Architecture</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    stages = [
        ("Stage 1", "Audio Ingestion", "WhisperX + pyannote diarization, Hinglish support"),
        ("Stage 2", "ASR Correction", "RxNorm phonetic grounding, drug name correction"),
        ("Stage 3", "Clinical NLP", "Med7 + scispaCy + SOAP note generation"),
        ("Stage 4", "Verification", "Dual-tier NLI + Ontology-RAG verifier"),
        ("Stage 5", "Export", "HL7 FHIR R4 + Patient summary"),
    ]
    for col, (stage, title, desc) in zip(cols, stages):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.75rem; color:#0d9488; font-weight:700;">{stage}</div>
                <div style="font-size:1rem; font-weight:600; color:#1f2937; margin:0.5rem 0;">{title}</div>
                <div style="font-size:0.8rem; color:#6b7280;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # Trained models summary
    st.markdown('<div class="section-header">🧠 Fine-Tuned Models</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="result-card">
            <h4>🔬 Model 1: Clinical NLI Verifier</h4>
            <p style="color:#6b7280; font-size:0.85rem;">
                <strong>Architecture:</strong> DeBERTa-v3-base<br>
                <strong>Task:</strong> 3-class NLI (Entailment / Neutral / Contradiction)<br>
                <strong>Pipeline Stage:</strong> Stage 4 — 2nd-Pass Verifier<br>
                <strong>Purpose:</strong> Classify clinical claims as supported, neutral, or contradicted
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="result-card">
            <h4>📋 Model 2: SOAP Section Classifier</h4>
            <p style="color:#6b7280; font-size:0.85rem;">
                <strong>Architecture:</strong> DistilBERT / Bio_ClinicalBERT<br>
                <strong>Task:</strong> 4-class classification (S / O / A / P)<br>
                <strong>Pipeline Stage:</strong> Stage 3 — Clinical Note Generator<br>
                <strong>Purpose:</strong> Route clinical utterances to correct SOAP section
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="result-card">
            <h4>💊 Model 3: Drug Name Corrector</h4>
            <p style="color:#6b7280; font-size:0.85rem;">
                <strong>Architecture:</strong> T5-small (Seq2Seq)<br>
                <strong>Task:</strong> Phonetic ASR error correction<br>
                <strong>Pipeline Stage:</strong> Stage 2 — ASR Post-Correction<br>
                <strong>Purpose:</strong> Fix garbled drug names from speech recognition
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Verification taxonomy
    st.markdown('<div class="section-header">🛡️ 5-Tier Verification Taxonomy</div>', unsafe_allow_html=True)

    tiers_data = [
        ("T1", "Verbatim", "Direct match", "tier-1", "✅"),
        ("T2a", "Terminology Upgrade", "Lay → medical term", "tier-2a", "✅"),
        ("T2b", "Ontology Synonym", "Brand → generic drug", "tier-2b", "✅"),
        ("T3", "Clinical Inference", "Medically reasonable deduction", "tier-3", "⚠️"),
        ("T4", "Unsupported", "No evidence in transcript", "tier-4", "🚫"),
        ("T5", "Contradiction", "Directly conflicts with transcript", "tier-5", "❌"),
    ]

    tier_html = '<div style="display:flex; gap:0.5rem; flex-wrap:wrap;">'
    for tier, name, desc, css_class, icon in tiers_data:
        tier_html += f'<span class="tier-badge {css_class}">{icon} {tier}: {name}</span>'
    tier_html += '</div>'
    st.markdown(tier_html, unsafe_allow_html=True)

    # Inter-module connections
    st.markdown('<div class="section-header">🔗 CARE\'OS Inter-Module Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>Module 1</strong> (Queue-Time Predictor) → Receives real-time consultation telemetry from Module 2<br>
        <strong>Module 2</strong> (AI Scribe — This Module) → Core ambient clinical intelligence pipeline<br>
        <strong>Module 3</strong> (Genomic Treatment) → Receives CPIC Level A drug triggers from Module 2
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Training Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Training Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Training Dashboard</h1>
        <p>Fine-tuning results, training curves, and evaluation metrics for all three models</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🔬 MedNLI Verifier (DeBERTa-v3)",
        "📋 SOAP Classifier (DistilBERT)",
        "💊 Drug Corrector (T5-small)",
    ])

    def render_training_tab(model_key, model_name, plot_filename):
        results = load_training_results(model_key)
        if results is None:
            st.warning(f"No training results found for {model_name}. Please run the training script first.")
            st.code(f"python -m training.train_{model_key.replace('mednli', 'mednli_verifier').replace('soap', 'soap_classifier').replace('drug', 'drug_corrector')}")
            return

        # Metrics row
        st.markdown('<div class="section-header">📈 Performance Metrics</div>', unsafe_allow_html=True)

        if model_key == "mednli":
            final = results.get("final_metrics", {})
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{final.get('accuracy', 0):.1%}</div>
                    <div class="metric-label">Accuracy</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{final.get('macro_f1', 0):.1%}</div>
                    <div class="metric-label">Macro F1</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{results.get('dataset_size', 0)}</div>
                    <div class="metric-label">Training Samples</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{results.get('total_training_time_seconds', 0):.0f}s</div>
                    <div class="metric-label">Training Time</div>
                </div>""", unsafe_allow_html=True)

            # Per-class F1
            st.markdown('<div class="section-header">Per-Class Performance</div>', unsafe_allow_html=True)
            pc1, pc2, pc3 = st.columns(3)
            for col, label in zip([pc1, pc2, pc3], ["entailment", "neutral", "contradiction"]):
                with col:
                    f1_val = final.get(f"{label}_f1", 0)
                    prec_val = final.get(f"{label}_precision", 0)
                    rec_val = final.get(f"{label}_recall", 0)
                    st.markdown(f"""<div class="metric-card">
                        <div style="font-weight:600; color:#1f2937; text-transform:capitalize;">{label}</div>
                        <div class="metric-value" style="font-size:1.5rem;">{f1_val:.1%}</div>
                        <div class="metric-label">F1 Score</div>
                        <div style="font-size:0.8rem; color:#9ca3af; margin-top:0.5rem;">
                            P: {prec_val:.1%} | R: {rec_val:.1%}
                        </div>
                    </div>""", unsafe_allow_html=True)

        elif model_key == "soap":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{results.get('best_macro_f1', 0):.1%}</div>
                    <div class="metric-label">Best Macro F1</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                hist = results.get("history", {})
                last_acc = hist.get("val_acc", [0])[-1] if hist.get("val_acc") else 0
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{last_acc:.1%}</div>
                    <div class="metric-label">Val Accuracy</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{results.get('training_time_seconds', 0):.0f}s</div>
                    <div class="metric-label">Training Time</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                hp = results.get("hyperparameters", {})
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{hp.get('epochs', 0)}</div>
                    <div class="metric-label">Epochs</div>
                </div>""", unsafe_allow_html=True)

        elif model_key == "drug":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                em = results.get("final_exact_match_acc", 0)
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{em:.1%}</div>
                    <div class="metric-label">Exact Match Acc</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{results.get('best_val_loss', 0):.4f}</div>
                    <div class="metric-label">Best Val Loss</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{results.get('training_time_seconds', 0):.0f}s</div>
                    <div class="metric-label">Training Time</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                hp = results.get("hyperparameters", {})
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value">{hp.get('samples', 0)}</div>
                    <div class="metric-label">Training Samples</div>
                </div>""", unsafe_allow_html=True)

        # Training curves plot
        st.markdown('<div class="section-header">📉 Training Curves</div>', unsafe_allow_html=True)

        plot_path = RESULTS_DIR / plot_filename
        if plot_path.exists():
            st.image(str(plot_path), use_container_width=True)
        else:
            # Render curves from JSON data
            history = results.get("history", {})
            if history:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
                fig.patch.set_facecolor('#fafafa')
                for ax in axes:
                    ax.set_facecolor('#fafafa')

                if "train_loss" in history:
                    axes[0].plot(history["train_loss"], label="Train Loss", color="#0d9488", linewidth=2)
                    axes[0].plot(history["val_loss"], label="Val Loss", color="#ef4444", linewidth=2)
                    axes[0].set_title("Loss Curves", fontweight="bold", fontsize=11)
                    axes[0].set_xlabel("Epoch")
                    axes[0].set_ylabel("Loss")
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.2)

                if "train_acc" in history:
                    axes[1].plot(history["train_acc"], label="Train Acc", color="#0d9488", linewidth=2)
                    axes[1].plot(history["val_acc"], label="Val Acc", color="#f59e0b", linewidth=2)
                    axes[1].set_title("Accuracy Curves", fontweight="bold", fontsize=11)
                    axes[1].set_xlabel("Epoch")
                    axes[1].set_ylabel("Accuracy")
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.2)
                elif "exact_match_acc" in history:
                    axes[1].plot([x*100 for x in history["exact_match_acc"]],
                                label="Exact Match %", color="#0d9488", linewidth=2, marker="o")
                    axes[1].set_title("Exact Match Accuracy", fontweight="bold", fontsize=11)
                    axes[1].set_xlabel("Epoch")
                    axes[1].set_ylabel("Accuracy %")
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.2)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # Hyperparameters
        with st.expander("⚙️ Training Hyperparameters"):
            hp = results.get("hyperparameters", {})
            if not hp:
                hp = {
                    "model_name": results.get("model_name", ""),
                    "epochs": results.get("epochs", ""),
                    "batch_size": results.get("batch_size", ""),
                    "learning_rate": results.get("learning_rate", ""),
                }
            st.json(hp)

        # Confusion matrix (if available)
        cm = results.get("confusion_matrix") or results.get("final_confusion_matrix")
        if cm:
            with st.expander("📊 Confusion Matrix"):
                st.json(cm)

    with tab1:
        render_training_tab("mednli", "MedNLI Verifier", "mednli_verifier/training_curves.png")

    with tab2:
        render_training_tab("soap", "SOAP Classifier", "soap_classifier/soap_training_curves.png")

    with tab3:
        render_training_tab("drug", "Drug Corrector", "drug_corrector/drug_corrector_curves.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Clinical NLI Verifier
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Clinical NLI Verifier":
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Clinical NLI Verification</h1>
        <p>Test the fine-tuned DeBERTa-v3 model — verify if SOAP claims are supported by transcript evidence</p>
    </div>
    """, unsafe_allow_html=True)

    if not check_model_exists("mednli"):
        st.error("MedNLI model not found. Please run training first.")
        st.code("python -m training.train_mednli_verifier --epochs 10 --batch-size 8")
        st.stop()

    st.markdown("""
    <div class="info-box">
        <strong>How it works:</strong> Enter a clinical transcript excerpt (premise) and a SOAP note claim 
        (hypothesis). The model classifies the relationship as:<br>
        • <strong>Entailment</strong> — The claim is supported by the transcript<br>
        • <strong>Neutral</strong> — The claim cannot be verified from the transcript<br>
        • <strong>Contradiction</strong> — The claim conflicts with the transcript
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        premise = st.text_area(
            "📝 Transcript Excerpt (Premise)",
            value="Patient denies any chest pain, shortness of breath, or palpitations. Blood pressure is 118/72 mmHg.",
            height=120,
            help="What the patient or doctor said during the consultation"
        )

    with col2:
        hypothesis = st.text_area(
            "📋 SOAP Note Claim (Hypothesis)",
            value="Patient presents with acute chest pain.",
            height=120,
            help="The claim from the generated SOAP note to verify"
        )

    # Example buttons
    st.markdown("**Quick Examples:**")
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    with ex_col1:
        if st.button("✅ Entailment Example", use_container_width=True):
            st.session_state["nli_premise"] = "Patient reports chest pain radiating to the left arm with diaphoresis."
            st.session_state["nli_hyp"] = "The patient is experiencing symptoms consistent with acute coronary syndrome."
            st.rerun()
    with ex_col2:
        if st.button("❓ Neutral Example", use_container_width=True):
            st.session_state["nli_premise"] = "Patient has a history of asthma since childhood."
            st.session_state["nli_hyp"] = "The patient may require intubation during surgery."
            st.rerun()
    with ex_col3:
        if st.button("❌ Contradiction Example", use_container_width=True):
            st.session_state["nli_premise"] = "Patient denies fever, chills, or night sweats."
            st.session_state["nli_hyp"] = "Patient presents with high-grade fever."
            st.rerun()

    if st.button("🔍 Verify Claim", type="primary", use_container_width=True):
        with st.spinner("Running clinical NLI inference..."):
            import torch
            tokenizer, model, device = load_mednli_model()

            encoding = tokenizer(
                premise, hypothesis,
                max_length=256, padding="max_length",
                truncation=True, return_tensors="pt"
            )

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

            labels = ["Entailment", "Neutral", "Contradiction"]
            pred_idx = int(np.argmax(probs))
            pred_label = labels[pred_idx]
            confidence = float(probs[pred_idx])

            # Color coding
            colors = {"Entailment": "#10b981", "Neutral": "#f59e0b", "Contradiction": "#ef4444"}
            icons = {"Entailment": "✅", "Neutral": "❓", "Contradiction": "❌"}
            tier_map = {"Entailment": "Tier 1/2 — Supported", "Neutral": "→ Route to Slow-Path Judge",
                        "Contradiction": "Tier 5 — Safety Contradiction"}

            st.markdown(f"""
            <div class="result-card" style="border-left: 4px solid {colors[pred_label]};">
                <h4>{icons[pred_label]} Prediction: {pred_label}</h4>
                <div style="font-size:0.9rem; color:#6b7280;">
                    Confidence: <strong>{confidence:.1%}</strong> | 
                    Verification Tier: <strong>{tier_map[pred_label]}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability distribution
            st.markdown("**Probability Distribution:**")
            prob_cols = st.columns(3)
            for i, (col, label) in enumerate(zip(prob_cols, labels)):
                with col:
                    color = colors[label]
                    st.markdown(f"""
                    <div class="metric-card" style="border-top: 3px solid {color};">
                        <div class="metric-value" style="color:{color}; font-size:1.5rem;">{probs[i]:.1%}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SOAP Classifier
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📋 SOAP Classifier":
    st.markdown("""
    <div class="main-header">
        <h1>📋 SOAP Section Classifier</h1>
        <p>Route clinical utterances to the correct SOAP section — Subjective, Objective, Assessment, or Plan</p>
    </div>
    """, unsafe_allow_html=True)

    if not check_model_exists("soap"):
        st.error("SOAP classifier model not found. Please run training first.")
        st.code("python -m training.train_soap_classifier --epochs 8 --batch-size 16")
        st.stop()

    st.markdown("""
    <div class="info-box">
        Enter a clinical sentence and the model will classify it into the appropriate SOAP section.
    </div>
    """, unsafe_allow_html=True)

    text_input = st.text_area(
        "Clinical Sentence",
        value="Prescribe Amoxicillin-Clavulanate 875/125 mg orally twice daily for 7 days.",
        height=80,
    )

    # Quick examples
    soap_examples = {
        "Subjective": "Patient reports severe throbbing headache on the right side for 48 hours.",
        "Objective": "Blood pressure 146/92 mmHg, pulse 84 bpm regular, SpO2 97% on room air.",
        "Assessment": "Primary diagnosis: Acute exacerbation of COPD.",
        "Plan": "Increase Metformin to 1000 mg twice daily; add Empagliflozin 10 mg once daily.",
    }

    st.markdown("**Quick Examples:**")
    ex_cols = st.columns(4)
    for col, (section, example_text) in zip(ex_cols, soap_examples.items()):
        with col:
            if st.button(f"{section}", use_container_width=True):
                st.session_state["soap_input"] = example_text
                st.rerun()

    if st.button("🏷️ Classify Section", type="primary", use_container_width=True):
        with st.spinner("Classifying..."):
            import torch
            tokenizer, model, device = load_soap_model()

            encoding = tokenizer(
                text_input, max_length=128, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

            sections = ["Subjective", "Objective", "Assessment", "Plan"]
            pred_idx = int(np.argmax(probs))
            pred_section = sections[pred_idx]

            badge_classes = {
                "Subjective": "soap-subjective",
                "Objective": "soap-objective",
                "Assessment": "soap-assessment",
                "Plan": "soap-plan",
            }
            section_icons = {"Subjective": "💬", "Objective": "📊", "Assessment": "🩺", "Plan": "💊"}

            st.markdown(f"""
            <div class="result-card">
                <h4>{section_icons[pred_section]} Classified as: 
                    <span class="soap-badge {badge_classes[pred_section]}">{pred_section}</span>
                </h4>
                <div style="color:#6b7280; font-size:0.85rem;">Confidence: {probs[pred_idx]:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

            # All section probabilities
            st.markdown("**Section Probabilities:**")
            prob_cols = st.columns(4)
            colors = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899"]
            for i, (col, section) in enumerate(zip(prob_cols, sections)):
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="border-top: 3px solid {colors[i]};">
                        <div class="metric-value" style="color:{colors[i]}; font-size:1.3rem;">{probs[i]:.1%}</div>
                        <div class="metric-label">{section}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Batch classification
    with st.expander("📋 Batch Classification — Classify Multiple Sentences"):
        batch_text = st.text_area(
            "Enter multiple clinical sentences (one per line):",
            value="""Patient reports severe headache for 2 days.
Blood pressure 140/90 mmHg, temperature 37.8C.
Primary assessment: migraine with aura.
Prescribe Sumatriptan 50mg as needed for acute episodes.""",
            height=150,
        )

        if st.button("🏷️ Classify All", use_container_width=True):
            import torch
            tokenizer, model, device = load_soap_model()
            lines = [l.strip() for l in batch_text.strip().split("\n") if l.strip()]
            sections = ["Subjective", "Objective", "Assessment", "Plan"]

            results_list = []
            for line in lines:
                encoding = tokenizer(line, max_length=128, padding="max_length",
                                     truncation=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(input_ids=encoding["input_ids"].to(device),
                                    attention_mask=encoding["attention_mask"].to(device))
                    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                results_list.append({
                    "Text": line[:80] + "..." if len(line) > 80 else line,
                    "Section": sections[pred_idx],
                    "Confidence": f"{probs[pred_idx]:.1%}",
                })

            import pandas as pd
            df = pd.DataFrame(results_list)
            st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Drug Name Corrector
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "💊 Drug Name Corrector":
    st.markdown("""
    <div class="main-header">
        <h1>💊 Drug Name ASR Corrector</h1>
        <p>Fix garbled pharmaceutical names from speech recognition using the fine-tuned T5 model</p>
    </div>
    """, unsafe_allow_html=True)

    if not check_model_exists("drug"):
        st.error("Drug corrector model not found. Please run training first.")
        st.code("python -m training.train_drug_corrector --epochs 10 --batch-size 16")
        st.stop()

    st.markdown("""
    <div class="info-box">
        Speech recognition often garbles drug names — splitting them into fragments, making phonetic 
        substitutions, or producing near-miss misspellings. This T5 model maps noisy ASR output to 
        canonical pharmaceutical names with correct dosages.
    </div>
    """, unsafe_allow_html=True)

    drug_input = st.text_input(
        "🎤 Noisy ASR Drug Input",
        value="correct: augment in 625 mg",
        help="Prefix with 'correct:' for the model to perform correction"
    )

    # Example buttons
    st.markdown("**Quick Examples:**")
    drug_examples = [
        "correct: dollar 650 tablet",
        "correct: pan d morning before food",
        "correct: tell me sartan 40 mg",
        "correct: azithro 500",
        "correct: cal pol 500 mg",
        "correct: glyco met 500 mg",
    ]

    ex_cols = st.columns(3)
    for i, example in enumerate(drug_examples):
        with ex_cols[i % 3]:
            if st.button(example.replace("correct: ", ""), use_container_width=True):
                st.session_state["drug_input"] = example
                st.rerun()

    if st.button("🔧 Correct Drug Name", type="primary", use_container_width=True):
        with st.spinner("Running T5 inference..."):
            import torch
            tokenizer, model, device = load_drug_model()

            inp = tokenizer(drug_input, return_tensors="pt", max_length=64,
                            truncation=True, padding="max_length")
            input_ids = inp["input_ids"].to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=inp["attention_mask"].to(device),
                    max_length=64, num_beams=4, early_stopping=True
                )

            corrected = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            st.markdown(f"""
            <div class="result-card" style="border-left: 4px solid #10b981;">
                <div style="display:flex; align-items:center; gap:1rem;">
                    <div>
                        <div style="font-size:0.8rem; color:#6b7280; font-weight:500;">INPUT (Noisy ASR)</div>
                        <div style="font-size:1.1rem; color:#ef4444; text-decoration:line-through;">{drug_input}</div>
                    </div>
                    <div style="font-size:1.5rem;">→</div>
                    <div>
                        <div style="font-size:0.8rem; color:#6b7280; font-weight:500;">OUTPUT (Corrected)</div>
                        <div style="font-size:1.1rem; color:#10b981; font-weight:600;">{corrected}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Batch correction
    with st.expander("📋 Batch Correction — Fix Multiple Drug Names"):
        batch_drugs = st.text_area(
            "Enter noisy drug names (one per line, prefixed with 'correct:'):",
            value="""correct: augment in 625 mg
correct: dolo 650
correct: tell me sartan 40 mg
correct: cipro flox a cin 500
correct: o me pra zole 20 mg""",
            height=150,
        )

        if st.button("🔧 Correct All", use_container_width=True):
            import torch
            tokenizer, model, device = load_drug_model()
            lines = [l.strip() for l in batch_drugs.strip().split("\n") if l.strip()]

            results_list = []
            for line in lines:
                inp = tokenizer(line, return_tensors="pt", max_length=64,
                                truncation=True, padding="max_length")
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=inp["input_ids"].to(device),
                        attention_mask=inp["attention_mask"].to(device),
                        max_length=64, num_beams=4, early_stopping=True
                    )
                corrected = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                results_list.append({
                    "Noisy Input": line,
                    "Corrected Output": corrected,
                })

            import pandas as pd
            df = pd.DataFrame(results_list)
            st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Full Pipeline Demo
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📝 Full Pipeline Demo":
    st.markdown("""
    <div class="main-header">
        <h1>📝 Full Pipeline Demo</h1>
        <p>End-to-end clinical encounter processing — from transcript to verified SOAP note</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        This demo simulates the complete Module 2 pipeline using a sample clinical encounter transcript.
        Each sentence is classified into SOAP sections using Model 2, then each generated claim is verified 
        against the transcript using Model 1 (NLI Verifier).
    </div>
    """, unsafe_allow_html=True)

    # Sample transcript
    sample_transcript = st.text_area(
        "📝 Clinical Encounter Transcript",
        value="""Doctor: Good morning. What brings you in today?
Patient: I've been having this terrible headache for the past three days. It's mostly on the right side and throbbing.
Doctor: Have you had any fever, nausea, or vomiting?
Patient: Some nausea but no vomiting. No fever.
Doctor: Any visual changes or sensitivity to light?
Patient: Yes, bright lights make it worse.
Doctor: Are you currently taking any medications?
Patient: Just over-the-counter ibuprofen 400mg, but it's not helping much.
Doctor: Let me check your vitals. Blood pressure is 128/82, heart rate 76, temperature 37.0 C.
Doctor: Neurological exam is normal. No focal deficits. Cranial nerves intact.
Doctor: Based on your symptoms — unilateral throbbing headache with photophobia and nausea — this is consistent with migraine with aura.
Doctor: I'm going to prescribe Sumatriptan 50mg to take at the onset of a migraine episode. Also continue with ibuprofen as needed but no more than 3 days per week to avoid medication overuse headache.
Doctor: Follow up in 4 weeks. If the migraines become more frequent, we can discuss preventive therapy.""",
        height=250,
    )

    if st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True):
        models_ready = check_model_exists("soap") and check_model_exists("mednli")
        if not models_ready:
            st.error("Both SOAP Classifier and MedNLI Verifier models are required. Please train them first.")
            st.stop()

        import torch

        # Load models
        progress = st.progress(0, text="Loading models...")
        soap_tokenizer, soap_model, soap_device = load_soap_model()
        nli_tokenizer, nli_model, nli_device = load_mednli_model()
        progress.progress(20, text="Models loaded. Processing transcript...")

        # Step 1: Parse transcript into utterances
        lines = [l.strip() for l in sample_transcript.strip().split("\n") if l.strip()]
        utterances = []
        for line in lines:
            if line.startswith("Doctor:"):
                utterances.append({"speaker": "Doctor", "text": line.replace("Doctor:", "").strip()})
            elif line.startswith("Patient:"):
                utterances.append({"speaker": "Patient", "text": line.replace("Patient:", "").strip()})
            else:
                utterances.append({"speaker": "Unknown", "text": line.strip()})

        progress.progress(30, text="Classifying SOAP sections...")

        # Step 2: Classify each utterance into SOAP sections
        sections = ["Subjective", "Objective", "Assessment", "Plan"]
        soap_results = {"Subjective": [], "Objective": [], "Assessment": [], "Plan": []}

        for utt in utterances:
            encoding = soap_tokenizer(
                utt["text"], max_length=128, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = soap_model(
                    input_ids=encoding["input_ids"].to(soap_device),
                    attention_mask=encoding["attention_mask"].to(soap_device)
                )
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            section = sections[pred_idx]
            soap_results[section].append({
                "text": utt["text"],
                "speaker": utt["speaker"],
                "confidence": float(probs[pred_idx]),
            })

        progress.progress(60, text="Generating SOAP note...")

        # Step 3: Compose SOAP note
        soap_note_claims = []
        for section in sections:
            for item in soap_results[section]:
                soap_note_claims.append({
                    "section": section,
                    "claim": item["text"],
                    "speaker": item["speaker"],
                    "section_confidence": item["confidence"],
                })

        progress.progress(75, text="Running 2nd-Pass verification...")

        # Step 4: Verify each claim against the full transcript
        full_transcript = " ".join([u["text"] for u in utterances])
        verified_claims = []

        nli_labels = ["Entailment", "Neutral", "Contradiction"]
        tier_map = {
            "Entailment": ("Tier 1", "Supported", "tier-1"),
            "Neutral": ("Tier 3", "Requires Review", "tier-3"),
            "Contradiction": ("Tier 5", "Contradiction", "tier-5"),
        }

        for claim in soap_note_claims:
            encoding = nli_tokenizer(
                full_transcript[:512], claim["claim"],
                max_length=256, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = nli_model(
                    input_ids=encoding["input_ids"].to(nli_device),
                    attention_mask=encoding["attention_mask"].to(nli_device)
                )
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            nli_label = nli_labels[pred_idx]
            tier, status, css = tier_map[nli_label]

            claim["nli_label"] = nli_label
            claim["nli_confidence"] = float(probs[pred_idx])
            claim["tier"] = tier
            claim["status"] = status
            claim["tier_css"] = css
            verified_claims.append(claim)

        progress.progress(100, text="Pipeline complete!")
        time.sleep(0.5)
        progress.empty()

        # ─── Display Results ──────────────────────────────────────────────

        # Summary metrics
        st.markdown('<div class="section-header">📊 Verification Summary</div>', unsafe_allow_html=True)

        total = len(verified_claims)
        supported = sum(1 for c in verified_claims if c["nli_label"] == "Entailment")
        neutral = sum(1 for c in verified_claims if c["nli_label"] == "Neutral")
        contradicted = sum(1 for c in verified_claims if c["nli_label"] == "Contradiction")

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{total}</div>
                <div class="metric-label">Total Claims</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""<div class="metric-card" style="border-top:3px solid #10b981;">
                <div class="metric-value" style="color:#10b981;">{supported}</div>
                <div class="metric-label">Supported</div>
            </div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""<div class="metric-card" style="border-top:3px solid #f59e0b;">
                <div class="metric-value" style="color:#f59e0b;">{neutral}</div>
                <div class="metric-label">Needs Review</div>
            </div>""", unsafe_allow_html=True)
        with mc4:
            st.markdown(f"""<div class="metric-card" style="border-top:3px solid #ef4444;">
                <div class="metric-value" style="color:#ef4444;">{contradicted}</div>
                <div class="metric-label">Contradicted</div>
            </div>""", unsafe_allow_html=True)

        # SOAP note with verification badges
        st.markdown('<div class="section-header">📝 Verified SOAP Note</div>', unsafe_allow_html=True)

        for section in sections:
            section_claims = [c for c in verified_claims if c["section"] == section]
            if not section_claims:
                continue

            badge_css = {
                "Subjective": "soap-subjective",
                "Objective": "soap-objective",
                "Assessment": "soap-assessment",
                "Plan": "soap-plan",
            }

            st.markdown(f"""
            <div style="margin:1rem 0 0.5rem 0;">
                <span class="soap-badge {badge_css[section]}">{section}</span>
            </div>
            """, unsafe_allow_html=True)

            for claim in section_claims:
                st.markdown(f"""
                <div class="result-card" style="padding:1rem; margin:0.5rem 0;">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div style="flex:1;">
                            <span style="font-size:0.75rem; color:#9ca3af;">
                                [{claim['speaker']}]
                            </span>
                            <div style="margin-top:0.25rem;">{claim['claim']}</div>
                        </div>
                        <div>
                            <span class="tier-badge {claim['tier_css']}">{claim['tier']}: {claim['status']}</span>
                            <div style="font-size:0.7rem; color:#9ca3af; text-align:right; margin-top:0.25rem;">
                                {claim['nli_confidence']:.0%} conf
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Hallucination rate
        hallucination_rate = (contradicted / total * 100) if total > 0 else 0
        inference_aware_rate = (contradicted / total * 100) if total > 0 else 0

        st.markdown(f"""
        <div class="info-box" style="margin-top:2rem;">
            <strong>Hallucination Analysis:</strong><br>
            • Literal hallucination rate: <strong>{(neutral + contradicted) / total * 100:.1f}%</strong> 
              (counting neutral as unverified)<br>
            • Inference-aware rate: <strong>{inference_aware_rate:.1f}%</strong> 
              (only counting true contradictions)<br>
            • Human physician benchmark: <strong>10.4%</strong>
        </div>
        """, unsafe_allow_html=True)

        # Export results as JSON
        with st.expander("📥 Export Results as JSON"):
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_claims": total,
                    "supported": supported,
                    "neutral": neutral,
                    "contradicted": contradicted,
                    "hallucination_rate": hallucination_rate,
                },
                "verified_claims": verified_claims,
            }
            st.json(export_data)
            st.download_button(
                "📥 Download JSON",
                json.dumps(export_data, indent=2),
                "pipeline_results.json",
                "application/json",
            )
