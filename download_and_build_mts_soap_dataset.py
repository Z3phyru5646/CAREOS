"""
CARE'OS Module 2 — Official MTS-Dialog & MTSamples Clinical Dataset Ingestion
=============================================================================
Replaces synthetic SOAP dataset with 100% genuine, authentic clinical sentences
extracted directly from:
  1. MTS-Dialog (NIH / ACL BioNLP Official Benchmark Repository by Abacha et al.)
  2. MTSamples Medical Transcription Corpus (Real physician clinical encounters)
"""

import os
import sys
import csv
import io
import re
import json
import random
import urllib.request
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
SOAP_DATA_FILE = BASE_DIR / "data" / "training" / "soap" / "soap_dataset.json"
SOAP_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. FETCH AUTHENTIC MTS-DIALOG DATASET (ACL / BioNLP Benchmark)
# ══════════════════════════════════════════════════════════════════════════════

MTS_DIALOG_TRAIN_URL = "https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/Main-Dataset/MTS-Dialog-TrainingSet.csv"
MTS_DIALOG_VAL_URL = "https://raw.githubusercontent.com/abachaa/MTS-Dialog/main/Main-Dataset/MTS-Dialog-ValidationSet.csv"
MTSAMPLES_URL = "https://raw.githubusercontent.com/socd06/medical-nlp/master/data/mtsamples.csv"

MTS_HEADER_MAP = {
    # Subjective
    "CC": "Subjective",
    "GENHX": "Subjective",
    "PASTMEDICALHX": "Subjective",
    "PASTSURGICAL": "Subjective",
    "FAM/SOCHX": "Subjective",
    "ROS": "Subjective",
    "ALLERGY": "Subjective",
    "MEDICATIONS": "Subjective",
    "GYNHX": "Subjective",
    "OTHER_HISTORY": "Subjective",
    # Objective
    "EXAM": "Objective",
    "LABS": "Objective",
    "IMAGING": "Objective",
    # Assessment
    "ASSESSMENT": "Assessment",
    "DIAGNOSIS": "Assessment",
    # Plan
    "PLAN": "Plan",
    "DISPOSITION": "Plan",
    "PROCEDURES": "Plan",
    "EDCOURSE": "Plan",
    "IMMUNIZATIONS": "Plan"
}

def clean_sentence(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[0-9\.\-\*\#\:\)]+\s*', '', text)
    return text

def fetch_mts_dialog_sentences():
    print("[1/3] Fetching authentic MTS-Dialog clinical benchmark from ACL BioNLP...")
    extracted = {"Subjective": [], "Objective": [], "Assessment": [], "Plan": []}
    
    for url in [MTS_DIALOG_TRAIN_URL, MTS_DIALOG_VAL_URL]:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            content = urllib.request.urlopen(req, timeout=20).read().decode('utf-8', errors='ignore')
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                header = row.get("section_header", "").strip()
                soap_lbl = MTS_HEADER_MAP.get(header)
                if not soap_lbl:
                    continue
                text = row.get("section_text", "").strip()
                if not text:
                    continue
                # Split text into clean sentences
                sents = [clean_sentence(s) for s in re.split(r'(?<=[.?!])\s+', text) if len(s.strip()) > 20]
                for s in sents:
                    if len(s) > 25 and not s.endswith(":") and s not in extracted[soap_lbl]:
                        extracted[soap_lbl].append(s)
        except Exception as e:
            print(f"Warning fetching MTS-Dialog from {url}: {e}")
            
    print(f"   MTS-Dialog extracted: Subjective={len(extracted['Subjective'])}, Objective={len(extracted['Objective'])}, Assessment={len(extracted['Assessment'])}, Plan={len(extracted['Plan'])}")
    return extracted

# ══════════════════════════════════════════════════════════════════════════════
# 2. FETCH AUTHENTIC MTSAMPLES CONSULTATION TRANSCRIPTIONS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_mtsamples_soap_sentences():
    print("[2/3] Fetching authentic MTSamples clinical encounter corpus...")
    extracted = {"Subjective": [], "Objective": [], "Assessment": [], "Plan": []}
    
    try:
        req = urllib.request.Request(MTSAMPLES_URL, headers={'User-Agent': 'Mozilla/5.0'})
        # Read streaming 10MB chunk
        data = urllib.request.urlopen(req, timeout=25).read(10000000).decode('utf-8', errors='ignore')
        reader = csv.DictReader(io.StringIO(data))
        
        for row in reader:
            t = row.get("transcription", "")
            if not t:
                continue
            
            # 1. Subjective Regex
            m_s = re.search(r'(?:SUBJECTIVE|HISTORY OF PRESENT ILLNESS|CHIEF COMPLAINT):(.*?)(?:OBJECTIVE|PHYSICAL EXAMINATION|VITAL SIGNS|ASSESSMENT|IMPRESSION|PLAN|$)', t, re.DOTALL | re.IGNORECASE)
            if m_s:
                s_text = m_s.group(1).strip()
                sents = [clean_sentence(s) for s in re.split(r'(?<=[.?!])\s+', s_text) if 25 < len(s.strip()) < 220]
                for s in sents:
                    if s not in extracted["Subjective"]:
                        extracted["Subjective"].append(s)
                        
            # 2. Objective Regex (Physical exam, vitals, labs)
            m_o = re.search(r'(?:OBJECTIVE|PHYSICAL EXAMINATION|VITAL SIGNS|LABORATORY DATA):(.*?)(?:ASSESSMENT|IMPRESSION|DIAGNOSIS|PLAN|DISPOSITION|$)', t, re.DOTALL | re.IGNORECASE)
            if m_o:
                o_text = m_o.group(1).strip()
                sents = [clean_sentence(s) for s in re.split(r'(?<=[.?!])\s+', o_text) if 25 < len(s.strip()) < 220]
                for s in sents:
                    if s not in extracted["Objective"]:
                        extracted["Objective"].append(s)
                        
            # 3. Assessment Regex (Impression, Diagnosis)
            m_a = re.search(r'(?:ASSESSMENT|IMPRESSION|DIAGNOSIS|PREOPERATIVE DIAGNOSIS):(.*?)(?:PLAN|DISPOSITION|TREATMENT|RECOMMENDATIONS|$)', t, re.DOTALL | re.IGNORECASE)
            if m_a:
                a_text = m_a.group(1).strip()
                sents = [clean_sentence(s) for s in re.split(r'(?<=[.?!;\n])\s+', a_text) if 20 < len(s.strip()) < 200]
                for s in sents:
                    if s not in extracted["Assessment"]:
                        extracted["Assessment"].append(s)
                        
            # 4. Plan Regex (Medications, instructions, follow-up)
            m_p = re.search(r'(?:PLAN|TREATMENT|DISPOSITION|DISCHARGE INSTRUCTIONS|RECOMMENDATIONS):(.*?)(?:DISPOSITION|SIGNATURE|$)', t, re.DOTALL | re.IGNORECASE)
            if m_p:
                p_text = m_p.group(1).strip()
                sents = [clean_sentence(s) for s in re.split(r'(?<=[.?!;\n])\s+', p_text) if 20 < len(s.strip()) < 220]
                for s in sents:
                    if s not in extracted["Plan"]:
                        extracted["Plan"].append(s)
                        
    except Exception as e:
        print(f"Warning fetching MTSamples: {e}")
        
    print(f"   MTSamples extracted: Subjective={len(extracted['Subjective'])}, Objective={len(extracted['Objective'])}, Assessment={len(extracted['Assessment'])}, Plan={len(extracted['Plan'])}")
    return extracted

# ══════════════════════════════════════════════════════════════════════════════
# 3. COMBINE, BALANCE & SAVE AUTHENTIC DATASET
# ══════════════════════════════════════════════════════════════════════════════

def build_final_soap_dataset():
    mts_dialog = fetch_mts_dialog_sentences()
    mtsamples = fetch_mtsamples_soap_sentences()
    
    combined = {"Subjective": [], "Objective": [], "Assessment": [], "Plan": []}
    
    for lbl in combined:
        # Merge both sources
        all_for_label = mts_dialog[lbl] + mtsamples[lbl]
        # Remove duplicates
        seen = set()
        unique = []
        for s in all_for_label:
            # Filter out non-informative fragments
            if s not in seen and len(s) > 20 and not any(skip in s.lower() for skip in ["page 1", "transcribed by", "date of birth", "mrn:"]):
                seen.add(s)
                unique.append(s)
        combined[lbl] = unique

    print("\n[3/3] Merging and balancing authentic clinical categories...")
    for lbl, items in combined.items():
        print(f"   {lbl}: {len(items)} unique authentic clinical sentences")
        
    # Cap each class to a balanced quota (target ~200-250 sentences per section)
    target_per_class = 225
    random.seed(42)
    
    final_dataset = []
    for lbl, items in combined.items():
        selected = items[:target_per_class] if len(items) >= target_per_class else items
        for s in selected:
            final_dataset.append({
                "text": s,
                "label": lbl
            })
            
    random.shuffle(final_dataset)
    
    # Save to disk
    with open(SOAP_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2)
        
    line_count = len(open(SOAP_DATA_FILE, "r", encoding="utf-8").readlines())
    file_size_kb = SOAP_DATA_FILE.stat().st_size / 1024
    
    print("\n" + "=" * 65)
    print(f"✅ SUCCESS: 100% Genuine MTS-Dialog & MTSamples Dataset Built!")
    print(f"   Destination: {SOAP_DATA_FILE.resolve()}")
    print(f"   Total Genuine Sentences: {len(final_dataset)}")
    print(f"   Total Lines:             {line_count:,} lines")
    print(f"   File Size:               {file_size_kb:.1f} KB")
    print(f"   Class Distribution:      {Counter(x['label'] for x in final_dataset)}")
    print("=" * 65)

if __name__ == "__main__":
    build_final_soap_dataset()
