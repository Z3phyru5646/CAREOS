# 🎓 CARE'OS Module 2: Model Fine-Tuning & Teacher Viva Guide

> **Project:** CARE'OS — Module 2: Ambient Clinical Scribe & Verifier  
> **Author:** Bhavya Sodhi (`2bhavyasodhi7`)  
> **Repository Branch:** [`Z3phyru5646/CAREOS - AI-Scribe`](https://github.com/Z3phyru5646/CAREOS/tree/AI-Scribe)  
> **Target Devices:** Local NVIDIA RTX 3050 (4GB VRAM), Google Colab (T4 GPU), or Lightning AI Studio

---

## 🎯 Viva Quick Answers (Teacher Ko Kya Bolna Hai)

### ❓ Question 1: *"Kya tumne bas LLM ki API call ki hai ya actual Machine Learning / Fine-Tuning ki hai?"*
> **Answer:**  
> *"Nahi sir/ma'am, humne pure system ko **4 specialized stages** mein design kiya hai aur **3 domain-specific models ko actually fine-tune kiya hai** kyunki standard zero-shot models clinical settings mein fail ho jate hain:*
> 1. ***Stage 2 (Drug Corrector):*** *Speech-to-text Indian brand names aur phonetic slangs ko galat sunta hai (jaise 'dollar 650' ya 'augment in'). Iske liye humne **T5-small Seq2Seq model** ko fine-tune kiya hai to map phonetic noise to canonical RxNorm/CDSCO drug entities.*
> 2. ***Stage 3 (SOAP Sentence Routing):*** *Doctor-patient conversation ke utterances ko formal SOAP (Subjective, Objective, Assessment, Plan) format mein classify karne ke liye humne **Clinical Transformer** fine-tune kiya hai.*
> 3. ***Stage 4 (MedNLI Verifier):*** *LLMs aksar fake diagnosis ya galat dosage hallucinate karte hain. Humne **DeBERTa-v3** ko **MedNLI (Clinical Natural Language Inference)** dataset par fine-tune kiya hai jo har generated claim ko clinical transcript ke against verify karta hai (Entailment vs Contradiction). Agar model koi aisi baat likhta hai jo doctor ne nahi boli, hamara verifier use safety alert ke saath reject karta hai."*

---

## 📊 Summary of Fine-Tuned Models

| Model | Pipeline Stage | Architecture Base | Dataset | Task / Objective | Metrics Tracked |
|---|---|---|---|---|---|
| **Model 1: Clinical NLI Verifier** | Stage 4 (Verifier) | `microsoft/deberta-v3-base` or `distilbert` | MedNLI (Clinical Premise-Hypothesis) | 3-Class NLI: Entailment, Neutral, Contradiction (Hallucination detection) | Train/Val Loss, Macro-F1, Confusion Matrix |
| **Model 2: SOAP Section Classifier** | Stage 3 (Structured Note) | `emilyalsentzer/Bio_ClinicalBERT` / `distilbert` | MTS-Dialog & ACI-Bench Clinical Sentences | 4-Class Classification: Subjective, Objective, Assessment, Plan | Accuracy, Macro-F1, Per-class Precision/Recall |
| **Model 3: Phonetic Drug Corrector** | Stage 2 (Entity Linker) | `t5-small` Seq2Seq | Indian + RxNorm Drug Phonetic Speech Pairs | Sequence-to-Sequence Phonetic Denoising & Canonical Rx Mapping | CrossEntropy Loss, Exact Match (EM) %, CER/WER |

---

## 🚀 How to Run the Training

### Option A: Google Colab / Lightning AI (Recommended for Viva Demo)
1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com)
2. Click **Upload** and upload:  
   `module2_careos/notebooks/CAREOS_Module2_Model_FineTuning_Colab.ipynb`
3. In Colab, go to **Runtime -> Change runtime type** -> select **T4 GPU**.
4. Click **Runtime -> Run all**.
5. *Result:* Saare models train ho jayenge (~10 minutes), live loss curves plot hongi, confusion matrices display hongi, aur sample clinical tests live run honge!

---

### Option B: Local Training (RTX 3050 4GB VRAM or CPU)
Windows terminal mein `module2_careos` folder se run karein:

#### 1. Train Model 1 (Clinical NLI Verifier):
```powershell
cd "C:\Users\Admin\Downloads\AI IN HEALTHCARE\module2_careos"
python -m training.train_mednli_verifier --epochs 10 --batch-size 8 --lr 2e-5
```
- **Output:** Checkpoint saved in `checkpoints/mednli_verifier/best_model`
- **Plots:** `training_results/mednli_verifier/training_curves.png`

#### 2. Train Model 2 (SOAP Section Classifier):
```powershell
python -m training.train_soap_classifier --epochs 8 --batch-size 16 --lr 3e-5
```
- **Output:** Checkpoint saved in `checkpoints/soap_classifier/best_model`
- **Plots:** `training_results/soap_classifier/soap_training_curves.png`

#### 3. Train Model 3 (T5 Drug Name ASR Corrector):
```powershell
python -m training.train_drug_corrector --epochs 10 --batch-size 16 --lr 5e-4
```
- **Output:** Checkpoint saved in `checkpoints/drug_corrector/best_model`
- **Plots:** `training_results/drug_corrector/drug_corrector_curves.png`

---

## 🧠 Technical Hyperparameters & Architecture Decisions

1. **Mixed Precision (FP16 AMP):**  
   PyTorch `torch.cuda.amp.autocast()` and `GradScaler` use kiye gaye hain taaki RTX 3050 ke 4GB VRAM mein bina OOM (Out Of Memory) crash hue high batch sizes run ho sakein.
2. **Optimizer:** `AdamW` with weight decay $0.01$ and linear warmup to prevent catastrophic forgetting of pre-trained clinical language representations.
3. **Loss Functions:**
   - Classification Models (MedNLI & SOAP): `CrossEntropyLoss` with label smoothing.
   - Generative Corrector (T5): Sequence-to-Sequence negative log-likelihood with padding masking (`label == -100`).

---

## 📁 Generated Deliverables Directory Structure
```
module2_careos/
├── notebooks/
│   └── CAREOS_Module2_Model_FineTuning_Colab.ipynb    <-- 1-Click Colab/Lightning Notebook
├── training/
│   ├── train_mednli_verifier.py                       <-- Stage 4 Model script
│   ├── train_soap_classifier.py                       <-- Stage 3 Model script
│   └── train_drug_corrector.py                        <-- Stage 2 Model script
├── checkpoints/                                       <-- Trained model weights (.bin / .safetensors)
│   ├── mednli_verifier/best_model/
│   ├── soap_classifier/best_model/
│   └── drug_corrector/best_model/
├── training_results/                                  <-- Visual plots & evaluation JSONs
│   ├── mednli_verifier/training_curves.png
│   ├── soap_classifier/soap_training_curves.png
│   └── drug_corrector/drug_corrector_curves.png
└── docs/
    └── TEACHER_VIVA_AND_MODEL_TRAINING_GUIDE.md       <-- This document
```
