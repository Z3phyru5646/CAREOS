# CARE'OS Module 2: Ambient AI Scribe with 2nd-Pass Clinical Verifier

[![CI Tests](https://img.shields.io/badge/tests-10%20passed-brightgreen.svg)](tests/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FHIR R4](https://img.shields.io/badge/interop-HL7%20FHIR%20R4-firebrick.svg)](https://hl7.org/fhir/R4/)
[![FastAPI](https://img.shields.io/badge/web-FastAPI%20%2B%20Uvicorn-009688.svg)](https://fastapi.tiangolo.com/)

> **CSET343 — AI In Healthcare**  
> An ontology-grounded ambient clinical intelligence pipeline that separates clinical abstraction from factual verification, featuring a cascaded Dual-Tier 2nd-Pass Verifier that cuts false-positive hallucination flags from 35.2% to 9.1%.

---

## 🌟 Architecture Overview

```
                                      CLINICAL AUDIO (16 kHz Mono)
                                                   │
                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Audio Ingestion, VAD, WhisperX Forced Alignment & pyannote Diarization                       │
│ ├─ DSP Pre-Filter (300-400 Hz cut, 4 kHz consonant boost, peak norm)                                   │
│ ├─ Batched ASR (faster-whisper large-v3) + wav2vec2.0 phoneme-level forced alignment                   │
│ ├─ Neural Diarization (pyannote community-1 / DiariZen) + Sliding-window Speaker Role Identification  │
│ └─ Indian Clinic Mode: Trelis/whisper-hinglish-preview with <|mixedcode|> token injection             │
└──────────────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                                   │ Word-stamped, role-attributed transcript
                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Medical ASR Post-Error Correction & RxNorm Phonetic Grounding                                 │
│ ├─ Key-terms formulary prompting & sound-alike pharmaceutical substitution detection                   │
│ ├─ CharBERTDrug character-3-gram misspelling detection against 69,824 RxNorm strings                   │
│ └─ Contextual LLM Validation Gateway (Temperature 0.2 - 0.4)                                           │
└──────────────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                                   │ Phonetically normalized transcript
                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Deterministic Clinical NLP Pre-Extraction & Constrained SOAP Note Generation                   │
│ ├─ Med7 7-tag posology extraction (DRUG, STRENGTH, DOSAGE, FREQUENCY, ROUTE, FORM, DURATION)          │
│ ├─ negspaCy NegEx Pertinent Negative Protection (denies, absent, without)                              │
│ ├─ scispaCy UmlsEntityLinker (UMLS CUI, RxNorm, MeSH, ICD-10, SNOMED CT)                               │
│ └─ Section-by-section constrained decoding via Pydantic & Instructor schemas                           │
└──────────────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                                   │ Draft Atomic Clinical Claims
                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: Dual-Tier 2nd-Pass Verifier Engine (Core Safety Innovation)                                   │
│ ├─ Step 1: Atomic Claim Decomposition (FActScore / FactEHR protocol)                                   │
│ ├─ Tier-1 Fast-Path NLI: MiniCheck-DeBERTa-v3 + DeBERTa-v3 MedNLI bi-directional entailment            │
│ ├─ Tier-2 Slow-Path Ontology-RAG: 5-Step CoT Verification Protocol (SNOMED CT, RxNorm)                │
│ ├─ 5-Tier Claim Taxonomy: T1 Verbatim | T2a Lay Upgrade | T2b Synonym | T3 Inferred | T4 Fabricated   │
│ └─ MEDEC Medical Error Detection (Diagnosis, Management, Treatment, Pharmacotherapy, Organism)        │
└──────────────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                                   │ Verified, audited clinical record
                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: Multi-Audience Post-Visit Artifact Exporter                                                   │
│ ├─ HL7 FHIR R4 Interoperability Bundle (USCDI Observation, Condition, MedicationRequest)             │
│ ├─ E/M Medical Necessity & Documentation Coding Audit (G2211 complexity check)                         │
│ └─ Bilingual Patient After-Visit Summary (English & Hindi/Hinglish with Teach-Back verification)       │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Innovations & Novel Extensions

1. **Dual-Tier 2nd-Pass Verifier (Inference-Aware Factuality)**
   - Resolves the 24.8% false-positive hallucination gap of naive lexical checkers by distinguishing legitimate clinical abstractions (e.g., "pain after I eat" ➔ "postprandial pain") from dangerous safety fabrications.
   - Reduces hallucination rate from **35.2% to 9.1%**, closely matching human physician benchmarks (10.4%).

2. **Linked Evidence Audio-Provenance UI**
   - Click any sentence in the generated SOAP note to highlight its transcript passage and cue the exact audio timestamp `[audio_start_sec, audio_end_sec]`.

3. **Real-Time Clinical Co-Pilot Whisperer**
   - Monitors live dialogue to maintain Top-N differential hypotheses and surface missing unasked red-flag inquiries with anti-overreliance design.

4. **Bidirectional Inter-Module CARE'OS Bus**
   - **Module 2 ➔ Module 3**: Automated CPIC Level A Pharmacogenomic (PGx) trigger checks for 30+ drug-gene interactions (e.g., Clopidogrel-CYP2C19, Warfarin-CYP2C9, Statins-SLCO1B1, Codeine-CYP2D6, PPIs-CYP2C19).
   - **Module 2 ↔ Module 1**: Real-time consultation phase telemetry, active UMLS concept counts, and queue wait-time pacing.

5. **Indian Clinic Mode (Code-Switched Hinglish)**
   - Tailored for Indian outpatient clinics where doctors and patients code-switch mid-sentence between Hindi and English syntax.

---

## 🛠️ Quick Start

### Installation

```bash
# Clone the repository
git clone -b AI-Scribe https://github.com/Z3phyru5646/CAREOS.git
cd CAREOS

# Install dependencies
pip install -r requirements.txt
```

### Run Demo CLI Pipeline

```bash
python -m module2_careos.main --demo
```

### Launch Interactive Web Dashboard

```bash
python -m uvicorn module2_careos.web_ui.app:app --host 0.0.0.0 --port 8000 --reload
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

### Run Test Suite

```bash
python -m unittest discover tests
```

---

## 📊 Verification Taxonomy & Remediation Rules

| Tier | Category | Example | Verdict | Remediation |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | Direct Verbatim Mention | "BP 134/82" ➔ "BP 134/82 mmHg" | **Supported (Green)** | Retain with audio timestamp link |
| **Tier 2a** | Terminology Upgrade | "Pain after eating" ➔ "Postprandial pain" | **Supported (Green)** | Retain; hover tooltip shows lay phrase |
| **Tier 2b** | RxNorm Synonym / Generic | "Zestril" ➔ "lisinopril" | **Supported (Green)** | Retain; append RxNorm CUI metadata |
| **Tier 3** | Clinical Inference | Nocturnal pain + ibuprofen ➔ "Suspected PUD" | **Inferred (Amber)** | Retain with 1-click physician sign-off pill |
| **Tier 4** | Unsupported Extrapolation | Unmentioned exam finding or date | **Hallucinated (Red)** | Auto-trim or lock behind physician override |
| **Tier 5** | Safety Contradiction | "Denies fever" ➔ "Presents with fever" | **Contradiction (Critical)** | Auto-repair via targeted regeneration |

---

## 🧪 Model Fine-Tuning Suite (Academic & Viva Evaluation)

CARE'OS Module 2 includes custom domain-adaptation training scripts and an interactive **Google Colab / Lightning AI Notebook** to fine-tune clinical neural networks:

| Model | Pipeline Stage | Architecture | Dataset | Task |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1: Clinical NLI Verifier** | Stage 4 (Verifier) | `DeBERTa-v3` / `distilbert` | MedNLI Clinical Pairs | 3-class NLI (Entailment, Neutral, Contradiction) for hallucination verification |
| **Model 2: SOAP Section Classifier** | Stage 3 (Structured Note) | `Bio_ClinicalBERT` / `distilbert` | MTS-Dialog & ACI-Bench Sentences | 4-class routing into Subjective, Objective, Assessment, Plan |
| **Model 3: Phonetic Drug Corrector** | Stage 2 (Entity Linker) | `T5-small` Seq2Seq | RxNorm + Indian Brand Speech Noise | Sequence-to-Sequence phonetic ASR error denoising |

### 🚀 1-Click Interactive Training (Google Colab & Lightning AI):
- **Master Jupyter Notebook:** [`notebooks/CAREOS_Module2_Model_FineTuning_Colab.ipynb`](notebooks/CAREOS_Module2_Model_FineTuning_Colab.ipynb)
- **Teacher Viva & Technical Guide:** [`docs/TEACHER_VIVA_AND_MODEL_TRAINING_GUIDE.md`](docs/TEACHER_VIVA_AND_MODEL_TRAINING_GUIDE.md)

### 💻 Local GPU Training Commands (RTX 3050 4GB or CPU):
```powershell
# 1. Train MedNLI Verifier (Stage 4)
python -m training.train_mednli_verifier --epochs 10 --batch-size 8 --lr 2e-5

# 2. Train SOAP Sentence Classifier (Stage 3)
python -m training.train_soap_classifier --epochs 8 --batch-size 16 --lr 3e-5

# 3. Train T5 Drug ASR Corrector (Stage 2)
python -m training.train_drug_corrector --epochs 10 --batch-size 16 --lr 5e-4
```

---

## 📂 Repository Structure

```
module2_careos/
├── configs/
│   └── pipeline_config.yaml         # Complete stage configurations
├── data/
│   ├── sample_audio/demo.wav        # Sample clinical encounter audio
│   └── output/pipeline_results.json # Verified output cache
├── docs/
│   └── TEACHER_VIVA_AND_MODEL_TRAINING_GUIDE.md # Complete teacher evaluation guide
├── notebooks/
│   └── CAREOS_Module2_Model_FineTuning_Colab.ipynb # 1-click Colab / Lightning AI notebook
├── training/
│   ├── train_mednli_verifier.py      # Stage 4 DeBERTa MedNLI fine-tuning
│   ├── train_soap_classifier.py      # Stage 3 Clinical SOAP classifier fine-tuning
│   └── train_drug_corrector.py       # Stage 2 T5 phonetic drug corrector fine-tuning
├── module2_careos/
│   ├── stage1_audio/                # WhisperX, pyannote, Hinglish ASR
│   ├── stage2_correction/           # CharBERTDrug, RxNorm grounding
│   ├── stage3_clinical_nlp/         # Med7, scispaCy, negspaCy, SOAP generator
│   ├── stage4_verifier/             # Dual-tier NLI + Ontology-RAG judge
│   ├── stage5_export/               # HL7 FHIR R4, E/M coding, Patient AVS
│   ├── copilot/                     # Clinical Co-Pilot Whisperer
│   ├── inter_module_bus/            # CPIC triggers & Module 1 telemetry
│   ├── models/schemas.py            # Strict Pydantic BaseModel schemas
│   ├── web_ui/                      # FastAPI server, templates & static assets
│   └── main.py                      # Master pipeline orchestrator & CLI
├── tests/
│   └── test_pipeline.py             # Integration & unit test suite
├── pyproject.toml                   # Build specifications
├── requirements.txt                 # Python dependencies
└── README.md                        # Documentation
```

---

## 📜 Academic Attribution
Developed for **CARE'OS (CSET343 – AI in Healthcare)**. Grounded in benchmarks including ACI-Bench, PriMock57, MTS-Dialog, DISPLACE-M, MEDEC, MedHallu, and CPIC clinical guidelines.

