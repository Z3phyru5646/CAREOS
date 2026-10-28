"""
CARE'OS Module 2 — Training Script 2 of 3
==========================================
Fine-Tune Clinical Transformer on SOAP Section Classification
Used in: Stage 3 Clinical Structured Note Generator (SOAP Sentence Routing)

What This Does:
  - Classifies clinical conversation utterances and doctor notes into 4 SOAP sections:
      0: Subjective  (Chief complaint, patient history, reported symptoms)
      1: Objective   (Vitals, physical exam, lab values, diagnostic imaging)
      2: Assessment  (Clinical diagnosis, differential diagnosis, clinical impression)
      3: Plan        (Prescriptions, therapies, follow-up visits, patient education)
  - Backbone: 'emilyalsentzer/Bio_ClinicalBERT' or 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    (defaults to 'distilbert-base-uncased' or 'roberta-base' for fast local training on 4GB VRAM)
  - Saves: Best model checkpoint, classification report, confusion matrix, loss & F1 curves

Hardware: Runs on RTX 3050 4GB VRAM / Google Colab T4 / Lightning AI
Time: ~10-15 minutes for 8-10 epochs

Usage:
  python -m training.train_soap_classifier
  python -m training.train_soap_classifier --epochs 10 --batch-size 16 --lr 3e-5
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CARE'OS.Training.SOAP")

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints" / "soap_classifier"
PLOTS_DIR = BASE_DIR / "training_results" / "soap_classifier"
DATA_DIR = BASE_DIR / "data" / "training" / "soap"

for d in [CHECKPOINTS_DIR, PLOTS_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SOAP_LABELS = ["Subjective", "Objective", "Assessment", "Plan"]
LABEL2ID = {name: i for i, name in enumerate(SOAP_LABELS)}
ID2LABEL = {i: name for i, name in enumerate(SOAP_LABELS)}


# ==============================================================================
# Clinical SOAP Dataset (MTS-Dialog & Clinical Practice Aligned)
# ==============================================================================

SOAP_RAW_SAMPLES = [
    # --- SUBJECTIVE (Patient complaints, history, symptoms) ---
    {"text": "Patient reports severe throbbing headache on the right side for the past 48 hours.", "label": "Subjective"},
    {"text": "I have been feeling dizzy and lightheaded whenever I stand up quickly.", "label": "Subjective"},
    {"text": "Complains of sharp substernal chest discomfort radiating to left shoulder on exertion.", "label": "Subjective"},
    {"text": "Doctor, my stomach burns intensely about thirty minutes after having spicy food.", "label": "Subjective"},
    {"text": "States she has had productive cough with yellowish-green phlegm for nearly a week.", "label": "Subjective"},
    {"text": "Past medical history notable for essential hypertension, type 2 diabetes, and asthma.", "label": "Subjective"},
    {"text": "Patient admits to missing insulin doses occasionally over weekends.", "label": "Subjective"},
    {"text": "I feel short of breath when climbing one flight of stairs or walking fast.", "label": "Subjective"},
    {"text": "Reports nausea and two episodes of non-bloody vomiting since yesterday morning.", "label": "Subjective"},
    {"text": "Patient denies any fevers, chills, night sweats, or recent weight changes.", "label": "Subjective"},
    {"text": "Family history is significant for maternal early onset myocardial infarction at age 48.", "label": "Subjective"},
    {"text": "Patient reports left knee pain that worsens with prolonged standing and kneeling.", "label": "Subjective"},
    {"text": "I noticed swelling in both of my ankles starting around three days ago.", "label": "Subjective"},
    {"text": "Current medications include amlodipine 5mg daily and atorvastatin 20mg at bedtime.", "label": "Subjective"},
    {"text": "Reports difficulty sleeping due to persistent nocturnal dry coughing spells.", "label": "Subjective"},
    {"text": "Patient states that his vision becomes slightly blurry in the late afternoons.", "label": "Subjective"},
    {"text": "Allergic to penicillin; developed hives and facial pruritus in childhood.", "label": "Subjective"},
    {"text": "Patient endorses generalized fatigue, low energy, and poor appetite over 3 weeks.", "label": "Subjective"},

    # --- OBJECTIVE (Physical exam, vitals, labs, diagnostics) ---
    {"text": "Blood pressure 146/92 mmHg, pulse 84 bpm regular, respiratory rate 18/min, SpO2 97% on room air.", "label": "Objective"},
    {"text": "Temperature 38.6 C (101.5 F), oral; patient appears mildly flushed and toxic.", "label": "Objective"},
    {"text": "Chest auscultation reveals diffuse expiratory wheezes bilaterally with prolonged expiratory phase.", "label": "Objective"},
    {"text": "Heart: Regular rate and rhythm, normal S1 and S2, no murmurs, rubs, or gallops heard.", "label": "Objective"},
    {"text": "Abdomen is soft, non-tender to light palpation, positive Murphy's sign on deep right upper quadrant palpation.", "label": "Objective"},
    {"text": "Bilateral 2+ pitting edema noted extending up to mid-calf bilaterally.", "label": "Objective"},
    {"text": "Fasting blood sugar 198 mg/dL, serum creatinine 1.3 mg/dL, eGFR 58 mL/min/1.73m2.", "label": "Objective"},
    {"text": "Hemoglobin A1c is 8.9%, indicating inadequate glycemic control.", "label": "Objective"},
    {"text": "Complete blood count shows WBC 14,200/mcL with 82% neutrophils and mild bandemia.", "label": "Objective"},
    {"text": "Right knee shows mild joint effusion, medial joint line tenderness, no ligamentous laxity.", "label": "Objective"},
    {"text": "ECG reveals normal sinus rhythm with 1.5mm ST elevations in leads V1-V3.", "label": "Objective"},
    {"text": "Chest X-ray demonstrates right lower lobe consolidation consistent with acute pneumonia.", "label": "Objective"},
    {"text": "Pharynx is erythematous with bilateral tonsillar hypertrophy and whitish exudates.", "label": "Objective"},
    {"text": "Neurological: Cranial nerves II-XII intact, power 5/5 in all four extremities, gait steady.", "label": "Objective"},
    {"text": "Urine dipstick positive for 2+ protein, negative for nitrites and leukocyte esterase.", "label": "Objective"},
    {"text": "Skin examination reveals erythematous maculopapular rash across trunk sparing palms and soles.", "label": "Objective"},

    # --- ASSESSMENT (Diagnosis, clinical impression, differential) ---
    {"text": "Primary diagnosis: Acute exacerbation of chronic obstructive pulmonary disease (AECOPD).", "label": "Assessment"},
    {"text": "1. Uncontrolled Type 2 Diabetes Mellitus with secondary diabetic peripheral neuropathy.", "label": "Assessment"},
    {"text": "Clinical impression favors acute bacterial rhinosinusitis over viral upper respiratory infection.", "label": "Assessment"},
    {"text": "Essential hypertension, stage 2, currently suboptimally controlled on monotherapy.", "label": "Assessment"},
    {"text": "Differential diagnosis includes gastroesophageal reflux disease versus biliary colic versus non-cardiac chest pain.", "label": "Assessment"},
    {"text": "Community-acquired pneumonia, CURB-65 score 1, suitable for outpatient therapeutic management.", "label": "Assessment"},
    {"text": "Suspected acute medial meniscal tear of the right knee secondary to mechanical twisting injury.", "label": "Assessment"},
    {"text": "Congestive heart failure exacerbation NYHA Class III triggered by dietary sodium non-compliance.", "label": "Assessment"},
    {"text": "Streptococcal pharyngitis confirmed by positive rapid antigen testing.", "label": "Assessment"},
    {"text": "Chronic kidney disease Stage 3a, secondary to long-standing hypertensive nephrosclerosis.", "label": "Assessment"},
    {"text": "Migraine headache with aura, episodic pattern with increasing frequency.", "label": "Assessment"},
    {"text": "Iron deficiency anemia, microcytic hypochromic picture, etiology under workup.", "label": "Assessment"},
    {"text": "Urinary tract infection, uncomplicated, likely Escherichia coli.", "label": "Assessment"},

    # --- PLAN (Rx, dosage, labs ordered, referrals, follow-up) ---
    {"text": "Prescribe Amoxicillin-Clavulanate 875/125 mg orally twice daily with meals for 7 days.", "label": "Plan"},
    {"text": "Increase Metformin to 1000 mg twice daily; add Empagliflozin 10 mg once daily in morning.", "label": "Plan"},
    {"text": "Order routine fasting lipid panel, comprehensive metabolic panel, and urine microalbumin ratio.", "label": "Plan"},
    {"text": "Administer Albuterol 2.5mg / Ipratropium 0.5mg nebulization in clinic now; repeat if needed in 20 minutes.", "label": "Plan"},
    {"text": "Schedule 2D echocardiogram and bilateral lower extremity venous Doppler ultrasound within one week.", "label": "Plan"},
    {"text": "Discontinue Ibuprofen immediately given elevated serum creatinine and gastrointestinal upset.", "label": "Plan"},
    {"text": "Refer to Cardiology for outpatient nuclear stress testing and evaluation of exertional angina.", "label": "Plan"},
    {"text": "Patient counselled on low-sodium DASH diet (less than 2g sodium/day) and daily morning weight logging.", "label": "Plan"},
    {"text": "Furosemide 40 mg PO daily for 5 days; monitor potassium and renal function in one week.", "label": "Plan"},
    {"text": "Return to emergency department immediately if experiencing chest pressure, dyspnea at rest, or syncope.", "label": "Plan"},
    {"text": "Follow up in outpatient clinic in 2 weeks for blood pressure re-evaluation and lab review.", "label": "Plan"},
    {"text": "Omeprazole 40 mg orally once daily before breakfast for 4 weeks for acid suppression.", "label": "Plan"},
    {"text": "Physical therapy referral for right knee strengthening exercises and quadriceps reconditioning.", "label": "Plan"},
    {"text": "Provide smoking cessation counseling and offer nicotine replacement patch 21mg/day.", "label": "Plan"},
]


def generate_augmented_dataset(base_samples, target_multiplier=5):
    """
    Synthesize diverse clinical variants using domain-specific template transforms
    to expand training data while maintaining high clinical fidelity.
    """
    augmented = list(base_samples)
    
    variations = {
        "Subjective": [
            ("Patient reports", "Patient states"),
            ("Complains of", "Presents with chief complaint of"),
            ("I have been feeling", "Patient remarks feeling"),
            ("Past medical history notable for", "Prior medical background includes"),
            ("Patient endorses", "Patient affirms"),
            ("Doctor, my", "Subjective notes indicate"),
        ],
        "Objective": [
            ("Blood pressure", "BP documented at"),
            ("Chest auscultation reveals", "Lungs on exam show"),
            ("Heart: Regular rate and rhythm", "Cardiovascular exam: Normal heart sounds"),
            ("Abdomen is soft", "Abdominal examination reveals"),
            ("Complete blood count shows", "Laboratory hematology results indicate"),
            ("ECG reveals", "12-lead Electrocardiogram demonstrates"),
        ],
        "Assessment": [
            ("Primary diagnosis:", "Assessment 1:"),
            ("Clinical impression favors", "Primary diagnostic impression:"),
            ("Differential diagnosis includes", "Differentials considered:"),
            ("Suspected", "Clinical presumption of"),
            ("confirmed by", "corroborated by"),
        ],
        "Plan": [
            ("Prescribe", "Initiate"),
            ("Increase", "Uptitrate dosage of"),
            ("Order routine", "Diagnostic workup ordered:"),
            ("Refer to", "Request formal specialist consultation with"),
            ("Follow up in", "Review patient in"),
            ("Return to emergency department immediately if", "Warning signs advised; attend ER if"),
        ]
    }
    
    for sample in base_samples:
        lbl = sample["label"]
        txt = sample["text"]
        for old_phrase, new_phrase in variations.get(lbl, []):
            if old_phrase.lower() in txt.lower():
                # case-insensitive replace
                idx = txt.lower().find(old_phrase.lower())
                new_txt = txt[:idx] + new_phrase + txt[idx + len(old_phrase):]
                augmented.append({"text": new_txt, "label": lbl})

    # Add extra medical domain templates to reach sufficient sample size
    templates = [
        ("Subjective", "Patient notes experiencing intermittent {symptom} for the past {time}."),
        ("Subjective", "Reports onset of {symptom} accompanied by mild dizziness."),
        ("Objective", "Physical examination demonstrates clear signs of {finding} upon inspection."),
        ("Objective", "Laboratory evaluation confirms elevated {lab} at {value}."),
        ("Assessment", "Patient presents with clinical presentation characteristic of {condition}."),
        ("Assessment", "Provisional assessment: Acute {condition} with secondary complications."),
        ("Plan", "Start patient on {medication} {dose} once daily with regular monitoring."),
        ("Plan", "Advise repeat blood tests for {lab} in {time} and dietary modification."),
    ]
    
    symptoms = ["fatigue and night sweats", "sharp lower back pain", "persistent dry cough", "palpitations and tremors"]
    times = ["3 days", "2 weeks", "one month", "48 hours"]
    findings = ["peripheral edema", "mild throat erythema", "abdominal guarding", "bilateral crepitations"]
    labs = ["C-reactive protein", "serum ALT and AST", "troponin-I", "D-dimer"]
    values = ["elevated above reference range", "within borderline limits", "14.5 mg/L", "2.1 ng/mL"]
    conditions = ["bronchial asthma exacerbation", "acute gastroenteritis", "urinary tract infection", "stage 1 hypertension"]
    medications = ["Azithromycin", "Atorvastatin", "Metoprolol", "Pantoprazole"]
    doses = ["500 mg", "20 mg", "50 mg", "40 mg"]
    
    for i in range(len(base_samples) * target_multiplier):
        t_lbl, t_fmt = random.choice(templates)
        filled = t_fmt.format(
            symptom=random.choice(symptoms),
            time=random.choice(times),
            finding=random.choice(findings),
            lab=random.choice(labs),
            value=random.choice(values),
            condition=random.choice(conditions),
            medication=random.choice(medications),
            dose=random.choice(doses)
        )
        augmented.append({"text": filled, "label": t_lbl})
        
    random.seed(42)
    random.shuffle(augmented)
    return augmented


# ==============================================================================
# PyTorch Dataset
# ==============================================================================

class SOAPDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label_id = LABEL2ID[item["label"]]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label_id, dtype=torch.long)
        }


# ==============================================================================
# Evaluation Metrics
# ==============================================================================

def calculate_metrics(y_true, y_pred, num_classes=4):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = np.mean(y_true == y_pred)

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    per_class_f1 = []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1.append(f1)

    macro_f1 = np.mean(per_class_f1)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": [float(f) for f in per_class_f1],
        "confusion_matrix": confusion.tolist()
    }


# ==============================================================================
# Training Pipeline
# ==============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 80)
    logger.info("CARE'OS MODULE 2 — SOAP SECTION CLASSIFIER FINE-TUNING")
    logger.info(f"Target Device:     {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    logger.info(f"Base Model:        {args.model_name}")
    logger.info(f"Batch Size:        {args.batch_size}")
    logger.info(f"Epochs:            {args.epochs}")
    logger.info(f"Learning Rate:     {args.lr}")
    logger.info(f"Max Seq Length:    {args.max_length}")
    logger.info("=" * 80)

    # 1. Load Tokenizer & Model
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    logger.info(f"Loading tokenizer & model: {args.model_name} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=4,
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
    except Exception as e:
        logger.warning(f"Could not load {args.model_name}: {e}. Falling back to 'distilbert-base-uncased'...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=4,
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )

    model.to(device)

    # 2. Build Dataset
    logger.info("Preparing clinical SOAP training dataset with clinical data augmentation...")
    dataset_records = generate_augmented_dataset(SOAP_RAW_SAMPLES, target_multiplier=args.multiplier)
    logger.info(f"Total dataset size: {len(dataset_records)} clinical samples across 4 SOAP categories.")

    # Save dataset to disk
    dataset_path = DATA_DIR / "soap_dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset_records, f, indent=2)
    logger.info(f"Saved dataset snapshot to: {dataset_path}")

    full_ds = SOAPDataset(dataset_records, tokenizer, max_length=args.max_length)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    logger.info(f"Train samples: {train_size} | Validation samples: {val_size}")

    # 3. Optimizer & Criterion
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # 4. Training Loop
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "per_class_f1": []
    }

    best_val_f1 = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        total_train_loss = 0.0
        train_preds, train_targets = [], []

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.detach().cpu().numpy())

        train_loss = total_train_loss / train_size
        train_acc = float(np.mean(np.array(train_preds) == np.array(train_targets)))

        # --- Validation ---
        model.eval()
        total_val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                with autocast(enabled=torch.cuda.is_available()):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                total_val_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        val_loss = total_val_loss / val_size
        metrics = calculate_metrics(val_targets, val_preds, num_classes=4)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(metrics["accuracy"], 4))
        history["val_macro_f1"].append(round(metrics["macro_f1"], 4))
        history["per_class_f1"].append(metrics["per_class_f1"])

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {metrics['accuracy']:.3f} Macro-F1: {metrics['macro_f1']:.3f}"
        )

        # Save Best Model Checkpoint
        if metrics["macro_f1"] > best_val_f1:
            best_val_f1 = metrics["macro_f1"]
            best_model_path = CHECKPOINTS_DIR / "best_model"
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            logger.info(f"  >>> Best model checkpoint saved! Macro-F1: {best_val_f1:.4f}")

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f}s | Best Macro-F1: {best_val_f1:.4f}")

    # 5. Save History & Metrics JSON
    results_json_path = PLOTS_DIR / "training_history.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": args.model_name,
            "hyperparameters": vars(args),
            "best_macro_f1": best_val_f1,
            "training_time_seconds": round(total_time, 2),
            "history": history,
            "final_confusion_matrix": metrics["confusion_matrix"]
        }, f, indent=2)

    # 6. Generate Training Curves & Confusion Matrix Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle("CARE'OS Stage 3: SOAP Section Classifier Fine-Tuning Metrics", fontsize=15, fontweight="bold")

        # Loss Curve
        axes[0, 0].plot(history["train_loss"], label="Train Loss", color="#2563eb", linewidth=2.5)
        axes[0, 0].plot(history["val_loss"], label="Val Loss", color="#dc2626", linewidth=2.5)
        axes[0, 0].set_title("Training vs Validation Loss", fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy Curve
        axes[0, 1].plot(history["train_acc"], label="Train Accuracy", color="#059669", linewidth=2.5)
        axes[0, 1].plot(history["val_acc"], label="Val Accuracy", color="#d97706", linewidth=2.5)
        axes[0, 1].set_title("Training vs Validation Accuracy", fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Macro F1 Curve
        axes[1, 0].plot(history["val_macro_f1"], label="Val Macro-F1", color="#7c3aed", linewidth=2.5, marker="o")
        axes[1, 0].set_title("Validation Macro-F1 Over Epochs", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Macro-F1 Score")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Confusion Matrix Heatmap
        cm = np.array(metrics["confusion_matrix"])
        im = axes[1, 1].imshow(cm, cmap="Blues", aspect="auto")
        axes[1, 1].set_title("Validation Confusion Matrix (SOAP)", fontweight="bold")
        axes[1, 1].set_xticks(range(4))
        axes[1, 1].set_yticks(range(4))
        axes[1, 1].set_xticklabels(SOAP_LABELS, rotation=20)
        axes[1, 1].set_yticklabels(SOAP_LABELS)
        axes[1, 1].set_xlabel("Predicted Section")
        axes[1, 1].set_ylabel("True Section")
        for i in range(4):
            for j in range(4):
                axes[1, 1].text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white", fontweight="bold")
        fig.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plot_file = PLOTS_DIR / "soap_training_curves.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Training charts saved to: {plot_file}")
    except Exception as e:
        logger.warning(f"Could not generate matplotlib plots: {e}")

    logger.info("=" * 80)
    logger.info("SOAP Classifier Training Pipeline Complete!")
    logger.info(f"Checkpoints: {CHECKPOINTS_DIR / 'best_model'}")
    logger.info(f"Metrics & Plots: {PLOTS_DIR}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Transformer on Clinical SOAP classification")
    parser.add_argument("--model-name", default="distilbert-base-uncased",
                        help="HuggingFace model (e.g. emilyalsentzer/Bio_ClinicalBERT, distilbert-base-uncased)")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (fits 4GB VRAM)")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--multiplier", type=int, default=5, help="Augmentation multiplier")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
