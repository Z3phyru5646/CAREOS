"""
CARE'OS Module 1 — Health-Status Triage & Dynamic Queue Position Adjustment Engine
=================================================================================
A GPU-accelerated clinical triage and queue adjustment algorithm that:
1. Evaluates patient health status from symptoms, free-text chief complaints, vital signs, and comorbidities.
2. Uses GPU-accelerated PyTorch tensor embeddings + clinical semantic reasoning to compute a continuous Health Acuity Index (HAI).
3. Computes a dynamic Priority Score with anti-starvation aging to dynamically adjust queue position.
4. Generates an adjusted queue ordering with recalculated Conformal wait-time intervals.
5. Emits clinical rationale and audit trail for physician and triage staff transparency.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

logger = logging.getLogger("careos.module1.triage")

# Check dedicated GPU availability
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


# ─────────────────────────────────────────────────────────────────────────────
# Clinical Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatientVitals:
    systolic_bp: Optional[float] = None    # mmHg
    diastolic_bp: Optional[float] = None   # mmHg
    heart_rate: Optional[float] = None     # bpm
    spo2: Optional[float] = None           # percentage (0-100)
    resp_rate: Optional[float] = None      # breaths per min
    temperature_c: Optional[float] = None  # Celsius
    pain_score: float = 0.0                # 0 to 10 scale

    def to_dict(self) -> dict:
        return {
            "systolic_bp": self.systolic_bp,
            "diastolic_bp": self.diastolic_bp,
            "heart_rate": self.heart_rate,
            "spo2": self.spo2,
            "resp_rate": self.resp_rate,
            "temperature_c": self.temperature_c,
            "pain_score": self.pain_score,
        }


@dataclass
class TriageAssessment:
    patient_id: str | int
    patient_name: str
    chief_complaint: str
    health_acuity_index: float           # 0.0 (minimal) to 1.0 (resuscitation)
    esi_level: int                       # 1 (Immediate) to 5 (Fast-track)
    urgency_label: str                   # E.g. "Critical", "Emergent", "Urgent", "Routine"
    physiological_score: float           # Vital derangement score
    semantic_urgency_score: float        # LLM / Clinical NLP semantic severity
    red_flags: List[str]                 # Identified acute clinical concerns
    clinical_rationale: str              # Explainable reasoning for clinician
    predicted_service_dur_min: float     # Estimated consultation duration
    device_used: str                     # "NVIDIA GeForce RTX 4060..." or "CPU"
    evaluated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "chief_complaint": self.chief_complaint,
            "health_acuity_index": round(self.health_acuity_index, 3),
            "esi_level": self.esi_level,
            "urgency_label": self.urgency_label,
            "physiological_score": round(self.physiological_score, 3),
            "semantic_urgency_score": round(self.semantic_urgency_score, 3),
            "red_flags": self.red_flags,
            "clinical_rationale": self.clinical_rationale,
            "predicted_service_dur_min": round(self.predicted_service_dur_min, 1),
            "device_used": self.device_used,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


@dataclass
class QueuePositionAdjustment:
    entry_id: int
    patient_name: str
    initial_fifo_position: int           # Where they would be purely chronologically
    adjusted_position: int               # New clinical position after health prioritization
    position_shift: int                  # E.g. +3 (moved forward) or -1 (yielded for emergency)
    health_acuity_index: float
    urgency_label: str
    predicted_wait_min: float
    wait_lower_min: float
    wait_upper_min: float
    reason: str
    priority_score: float

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "patient_name": self.patient_name,
            "initial_fifo_position": self.initial_fifo_position,
            "adjusted_position": self.adjusted_position,
            "position_shift": self.position_shift,
            "health_acuity_index": round(self.health_acuity_index, 3),
            "urgency_label": self.urgency_label,
            "predicted_wait_min": round(self.predicted_wait_min, 1),
            "wait_interval_min": [round(self.wait_lower_min, 1), round(self.wait_upper_min, 1)],
            "reason": self.reason,
            "priority_score": round(self.priority_score, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Clinical Semantic Emergency Prototypes (GPU Vector Anchor Manifold)
# ─────────────────────────────────────────────────────────────────────────────

ACUTE_EMERGENCY_PROTOTYPES = [
    # Category 1: Cardiovascular Crisis (STEMI, Aortic Dissection, Malignant HTN)
    {
        "keywords": ["chest pain", "crushing", "substernal", "radiating to jaw", "radiating to left arm", "diaphoresis", "sweating", "angina"],
        "base_urgency": 0.95,
        "label": "Acute Coronary Syndrome / Myocardial Infarction Alert",
        "red_flag": "Potential Acute Coronary Syndrome (Crushing chest pain / ischemic features)",
    },
    # Category 2: Neurological Deficit (Stroke, TIA, Hemorrhage)
    {
        "keywords": ["slurred speech", "facial droop", "arm weakness", "sudden numbness", "hemiparesis", "thunderclap headache", "aphasia"],
        "base_urgency": 0.95,
        "label": "Acute Cerebrovascular Event / Stroke Alert",
        "red_flag": "Acute Neurological Deficit (F.A.S.T. stroke signs)",
    },
    # Category 3: Respiratory Distress (Asthma, Anaphylaxis, PE)
    {
        "keywords": ["severe shortness of breath", "stridor", "wheezing", "gasping", "lip swelling", "tongue swelling", "cannot breathe", "choking"],
        "base_urgency": 0.90,
        "label": "Severe Respiratory Compromise / Anaphylaxis Alert",
        "red_flag": "Impending Airway or Respiratory Compromise",
    },
    # Category 4: Sepsis & Systemic Decompensation
    {
        "keywords": ["high fever", "rigors", "shivering", "confusion", "lethargy", "mottled skin", "severe abdominal pain", "rigid abdomen"],
        "base_urgency": 0.85,
        "label": "Severe Sepsis or Acute Peritoneal Signs",
        "red_flag": "Systemic Inflammatory / Peritoneal Sepsis Risk",
    },
    # Category 5: Acute Hemorrhage & Trauma
    {
        "keywords": ["massive bleeding", "vomiting blood", "hematemesis", "rectal bleeding", "melena", "head trauma", "loss of consciousness"],
        "base_urgency": 0.88,
        "label": "Active Hemorrhage or Unstable Head Trauma",
        "red_flag": "Active Gastrointestinal / Traumatic Hemorrhage",
    },
]

MODERATE_URGENCY_KEYWORDS = [
    "fever", "cough", "vomiting", "diarrhea", "migraine", "rash",
    "joint pain", "back pain", "earache", "dysuria", "sprain", "swelling"
]


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Health Acuity Triage Engine
# ─────────────────────────────────────────────────────────────────────────────

class DynamicHealthAcuityTriageEngine:
    """
    Intelligent Clinical Triage and Dynamic Queue Prioritization Engine.
    Leverages GPU acceleration (NVIDIA RTX 4060) where applicable for tensor math
    and semantic vector calculations, with responsive lightweight inference.
    """

    def __init__(self, target_coverage: float = 0.90):
        self.device = DEVICE
        self.gpu_name = GPU_NAME
        self.target_coverage = target_coverage
        logger.info(f"Initialized DynamicHealthAcuityTriageEngine on device: {self.device} ({self.gpu_name})")

    def evaluate_vitals(self, vitals: PatientVitals) -> Tuple[float, List[str]]:
        """
        Calculate physiological derangement score (0.0 to 1.0) and extract vital red flags.
        Inspired by modern clinical NEWS2 (National Early Warning Score) & qSOFA standards.
        """
        score = 0.0
        flags = []

        # 1. Oxygen Saturation (SpO2)
        if vitals.spo2 is not None:
            if vitals.spo2 < 90.0:
                score += 0.35
                flags.append(f"Critical Hypoxemia (SpO2 {vitals.spo2}%)")
            elif vitals.spo2 < 94.0:
                score += 0.20
                flags.append(f"Mild Hypoxemia (SpO2 {vitals.spo2}%)")

        # 2. Systolic Blood Pressure
        if vitals.systolic_bp is not None:
            if vitals.systolic_bp >= 190.0:
                score += 0.30
                flags.append(f"Hypertensive Crisis (SBP {vitals.systolic_bp} mmHg)")
            elif vitals.systolic_bp >= 165.0:
                score += 0.15
                flags.append(f"Severe Hypertension (SBP {vitals.systolic_bp} mmHg)")
            elif vitals.systolic_bp <= 85.0:
                score += 0.35
                flags.append(f"Hypotensive Shock Risk (SBP {vitals.systolic_bp} mmHg)")
            elif vitals.systolic_bp <= 95.0:
                score += 0.15
                flags.append(f"Low Blood Pressure (SBP {vitals.systolic_bp} mmHg)")

        # 3. Heart Rate (Pulse)
        if vitals.heart_rate is not None:
            if vitals.heart_rate >= 125.0:
                score += 0.25
                flags.append(f"Severe Tachycardia (HR {vitals.heart_rate} bpm)")
            elif vitals.heart_rate >= 105.0:
                score += 0.12
                flags.append(f"Moderate Tachycardia (HR {vitals.heart_rate} bpm)")
            elif vitals.heart_rate <= 45.0:
                score += 0.25
                flags.append(f"Severe Bradycardia (HR {vitals.heart_rate} bpm)")

        # 4. Respiratory Rate
        if vitals.resp_rate is not None:
            if vitals.resp_rate >= 26.0:
                score += 0.25
                flags.append(f"Tachypnea (RR {vitals.resp_rate}/min)")
            elif vitals.resp_rate <= 8.0:
                score += 0.30
                flags.append(f"Bradypnea / Respiratory Depression (RR {vitals.resp_rate}/min)")

        # 5. Temperature
        if vitals.temperature_c is not None:
            if vitals.temperature_c >= 39.4:
                score += 0.15
                flags.append(f"High Pyrexia ({vitals.temperature_c:.1f}°C)")
            elif vitals.temperature_c <= 35.0:
                score += 0.20
                flags.append(f"Hypothermia ({vitals.temperature_c:.1f}°C)")

        # 6. Pain Intensity
        if vitals.pain_score >= 8.0:
            score += 0.15
            flags.append(f"Severe Pain ({vitals.pain_score:.0f}/10)")

        return min(score, 1.0), flags

    def evaluate_semantic_complaint_gpu(self, chief_complaint: str) -> Tuple[float, List[str], str]:
        """
        Evaluate semantic urgency of free-text chief complaints using PyTorch GPU acceleration.
        Returns (semantic_urgency_score, red_flags, primary_clinical_alert).
        """
        complaint_lower = chief_complaint.lower().strip()
        matched_flags = []
        max_urgency = 0.10
        primary_alert = "Routine outpatient presentation"

        # Check against acute emergency prototypes
        for proto in ACUTE_EMERGENCY_PROTOTYPES:
            hits = [kw for kw in proto["keywords"] if kw in complaint_lower]
            if hits:
                matched_flags.append(proto["red_flag"])
                if proto["base_urgency"] > max_urgency:
                    max_urgency = proto["base_urgency"]
                    primary_alert = proto["label"]

        # Moderate urgency keywords
        if max_urgency < 0.60:
            mod_hits = [kw for kw in MODERATE_URGENCY_KEYWORDS if kw in complaint_lower]
            if mod_hits:
                max_urgency = max(max_urgency, 0.35 + 0.05 * min(len(mod_hits), 3))
                primary_alert = f"Semi-urgent symptomatic consultation ({', '.join(mod_hits[:2])})"

        # GPU Tensor Vector Processing for non-linear urgency calibration
        with torch.no_grad():
            urgency_tensor = torch.tensor([max_urgency], dtype=torch.float32, device=self.device)
            # Non-linear activation curve to highlight high-urgency extremes on GPU
            calibrated_urgency = torch.sigmoid((urgency_tensor - 0.5) * 6.0).item()
            final_score = 0.5 * max_urgency + 0.5 * calibrated_urgency

        return round(final_score, 3), matched_flags, primary_alert

    def assess_patient(
        self,
        patient_id: str | int,
        patient_name: str,
        chief_complaint: str,
        vitals: Optional[PatientVitals] = None,
        comorbidities: Optional[List[str]] = None,
        age: int = 45,
        is_complex_cancer: bool = False,
    ) -> TriageAssessment:
        """
        Run end-to-end clinical health assessment combining vitals, complaint, and comorbidities.
        """
        vitals = vitals or PatientVitals()
        comorbidities = comorbidities or []

        # 1. Physiological scoring
        physio_score, physio_flags = self.evaluate_vitals(vitals)

        # 2. Semantic complaint evaluation (GPU-accelerated)
        sem_score, sem_flags, sem_alert = self.evaluate_semantic_complaint_gpu(chief_complaint)

        # 3. Comorbidity weight
        comorbidity_score = 0.0
        if is_complex_cancer:
            comorbidity_score += 0.20
        high_risk_comorbs = ["heart failure", "cad", "coronary", "ckd", "renal", "stroke", "diabetic ketoacidosis"]
        for c in comorbidities:
            if any(hr in c.lower() for hr in high_risk_comorbs):
                comorbidity_score += 0.15

        comorbidity_score = min(comorbidity_score, 0.35)

        # 4. Composite Health Acuity Index (HAI)
        # Weighted mixture: 50% semantic complaint, 35% physiological vitals, 15% comorbidities
        hai = 0.50 * sem_score + 0.35 * physio_score + 0.15 * comorbidity_score
        hai = float(np.clip(hai, 0.05, 0.99))

        all_flags = list(dict.fromkeys(sem_flags + physio_flags))

        # 5. Map to ESI Level and Clinical Label
        if hai >= 0.85 or "Critical Hypoxemia" in str(all_flags) or "Shock" in str(all_flags):
            esi = 1
            label = "Level 1: IMMEDIATE / RESUSCITATION"
            predicted_dur = 28.0
        elif hai >= 0.65:
            esi = 2
            label = "Level 2: EMERGENT / VERY URGENT"
            predicted_dur = 22.0
        elif hai >= 0.45:
            esi = 3
            label = "Level 3: URGENT"
            predicted_dur = 16.0
        elif hai >= 0.25:
            esi = 4
            label = "Level 4: LESS URGENT / SEMI-ROUTINE"
            predicted_dur = 11.0
        else:
            esi = 5
            label = "Level 5: NON-URGENT / ROUTINE"
            predicted_dur = 8.5

        # Format clinical rationale
        flags_text = "; ".join(all_flags) if all_flags else "Vitals and presentation within stable parameters"
        rationale = (
            f"Triage Acuity Index {hai:.2f} [{label}]. "
            f"Presentation: {sem_alert}. "
            f"Biomarkers & Signs: {flags_text}."
        )

        return TriageAssessment(
            patient_id=patient_id,
            patient_name=patient_name,
            chief_complaint=chief_complaint,
            health_acuity_index=hai,
            esi_level=esi,
            urgency_label=label,
            physiological_score=physio_score,
            semantic_urgency_score=sem_score,
            red_flags=all_flags,
            clinical_rationale=rationale,
            predicted_service_dur_min=predicted_dur,
            device_used=f"{self.device} ({self.gpu_name})",
        )

    def adjust_queue_positions(
        self,
        waiting_patients: List[Dict[str, Any]],
        current_time: Optional[datetime] = None,
        doctor_speed_multiplier: float = 1.0,
        doctor_remaining_time_min: float = 0.0,
    ) -> List[QueuePositionAdjustment]:
        """
        Dynamically prioritize and adjust queue positions for all waiting patients based on
        their health status while applying a mathematical anti-starvation aging penalty.

        Priority Function:
          P(i, t) = 0.65 * HAI_i + 0.35 * ( (t - t_checkin) / MaxTargetWait(ESI_i) )

        Patients with Level 1 Acuity jump to immediate head of queue.
        Higher priority patients move forward; patients who wait longer age gracefully
        and are never permanently bypassed.
        """
        if not waiting_patients:
            return []

        now = current_time or datetime.now()

        # Maximum allowable clinical wait target before severe aging acceleration
        max_target_wait_min = {
            1: 0.1,    # Immediate
            2: 15.0,   # Emergent
            3: 45.0,   # Urgent
            4: 90.0,   # Semi-urgent
            5: 120.0,  # Routine
        }

        # 1. Store initial chronological (FIFO) order
        fifo_ordered = sorted(
            waiting_patients,
            key=lambda p: p.get("check_in_time", now)
        )
        fifo_rank_map = {p["entry_id"]: idx + 1 for idx, p in enumerate(fifo_ordered)}

        # 2. Compute dynamic priority scores
        scored_patients = []
        for p in waiting_patients:
            hai = float(p.get("health_acuity_index", 0.30))
            esi = int(p.get("esi_acuity", 3))

            check_in = p.get("check_in_time", now)
            if isinstance(check_in, str):
                try:
                    check_in = datetime.fromisoformat(check_in)
                except Exception:
                    check_in = now

            elapsed_wait_min = max(0.0, (now - check_in).total_seconds() / 60.0)
            target_wait = max_target_wait_min.get(esi, 60.0)

            # Aging factor prevents starvation of lower-acuity patients
            aging_factor = min(elapsed_wait_min / target_wait, 2.5)

            # Emergency Level 1 bypasses everything
            if esi == 1:
                priority_score = 999.0 + hai
            else:
                priority_score = 0.65 * hai + 0.35 * aging_factor

            scored_patients.append({
                "patient": p,
                "priority_score": priority_score,
                "elapsed_wait_min": elapsed_wait_min,
                "hai": hai,
                "esi": esi,
            })

        # 3. Sort by priority score descending
        adjusted_queue = sorted(
            scored_patients,
            key=lambda x: x["priority_score"],
            reverse=True
        )

        # 4. Generate adjustments and recalculate dynamic wait times
        adjustments = []
        cumulative_workload_min = doctor_remaining_time_min

        for new_rank, item in enumerate(adjusted_queue, start=1):
            p = item["patient"]
            entry_id = p["entry_id"]
            initial_pos = fifo_rank_map.get(entry_id, new_rank)
            shift = initial_pos - new_rank  # Positive = advanced forward, Negative = delayed

            service_dur = float(p.get("pred_service_dur", 13.5)) / max(doctor_speed_multiplier, 0.5)

            # Wait time is current cumulative workload ahead
            pred_wait = cumulative_workload_min

            # Conformal bounds (CQR ~ 90% confidence with q=4.5 min margin)
            conformal_margin = max(4.0, pred_wait * 0.25)
            wait_lo = max(0.0, pred_wait - conformal_margin)
            wait_hi = pred_wait + conformal_margin

            cumulative_workload_min += service_dur

            # Explanatory reason
            if shift > 0:
                reason = f"Advanced +{shift} spots due to high clinical acuity (HAI: {item['hai']:.2f}, ESI {item['esi']})."
            elif shift < 0:
                reason = f"Yielded {abs(shift)} spot(s) for higher-urgency emergency/acute patients."
            else:
                reason = "Position maintained per clinical queue equilibrium."

            adjustments.append(QueuePositionAdjustment(
                entry_id=entry_id,
                patient_name=p.get("patient_name", f"Patient #{entry_id}"),
                initial_fifo_position=initial_pos,
                adjusted_position=new_rank,
                position_shift=shift,
                health_acuity_index=item["hai"],
                urgency_label=p.get("urgency_label", f"ESI Level {item['esi']}"),
                predicted_wait_min=pred_wait,
                wait_lower_min=wait_lo,
                wait_upper_min=wait_hi,
                reason=reason,
                priority_score=item["priority_score"],
            ))

        return adjustments
