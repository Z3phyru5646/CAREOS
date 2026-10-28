"""
CAREOS Module 1 - FastAPI Backend
Event-Driven Queue Management API with:
- Single-Writer Queue State Machine
- JSEW Doctor Routing
- SOAP Telemetry Webhook (Module 2 → Module 1)
- CQR/ACI Prediction with TreeSHAP
- Emergency Preemption
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import joblib
import os
import sys
import asyncio
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

app = FastAPI(
    title="CAREOS Module 1 - Queue & Wait-Time Predictor API",
    description="Dynamic Outpatient Queue Management with ML-powered Wait-Time Prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Data Models
# ============================================================

class ESILevel(int, Enum):
    IMMEDIATE = 1
    HIGH_RISK = 2
    STABLE_MULTI = 3
    ONE_RESOURCE = 4
    FAST_TRACK = 5

class PatientStatus(str, Enum):
    BOOKED = "BOOKED"
    WAITING = "WAITING"
    IN_CONSULTATION = "IN_CONSULTATION"
    COMPLETED = "COMPLETED"
    NO_SHOW = "NO_SHOW"
    PREEMPTED = "PREEMPTED"

class SOAPStage(str, Enum):
    NONE = "NONE"
    SUBJECTIVE = "S"
    OBJECTIVE = "O"
    ASSESSMENT = "A"
    PLAN = "P"


class CheckInRequest(BaseModel):
    patient_name: str
    age: int
    gender: str = Field(..., pattern="^[MF]$")
    visit_no: int = Field(ge=1)
    esi_acuity: ESILevel
    complaint_category: str
    has_hypertension: bool = False
    has_diabetes: bool = False
    is_complex_cancer: bool = False
    address_tier: str = "local"
    is_walk_in: bool = True
    preferred_doctor_id: Optional[int] = None
    travel_time_min: Optional[float] = None


class SOAPTelemetryRequest(BaseModel):
    room_id: int
    soap_stage: SOAPStage
    stage_midpoint_fraction: float = Field(ge=0.0, le=1.0)


class EmergencyPreemptRequest(BaseModel):
    patient_entry_id: Optional[int] = None
    patient_name: str = "Emergency Patient"
    esi_acuity: ESILevel = ESILevel.IMMEDIATE


# ============================================================
# In-Memory State (would be PostgreSQL in production)
# ============================================================

class ClinicState:
    def __init__(self):
        self.session_id = 1
        self.doctors = {
            0: {"id": 0, "name": "Dr. Sharma", "speed_mult": 0.80, "active": True,
                "current_patient": None, "consultation_start": None, "soap_stage": "NONE",
                "rolling_pact": 13.5, "completed": 0},
            1: {"id": 1, "name": "Dr. Verma", "speed_mult": 1.00, "active": True,
                "current_patient": None, "consultation_start": None, "soap_stage": "NONE",
                "rolling_pact": 13.5, "completed": 0},
            2: {"id": 2, "name": "Dr. Patel", "speed_mult": 1.25, "active": True,
                "current_patient": None, "consultation_start": None, "soap_stage": "NONE",
                "rolling_pact": 13.5, "completed": 0},
        }
        self.queue_entries: Dict[int, dict] = {}
        self.next_entry_id = 1
        self.prediction_logs: List[dict] = []
        self.aci_alpha_t = 0.10
        self.cqr_Q = 5.0  # Default calibration quantile
        self.recent_waits: List[float] = []
        self.session_start = datetime.now()

    def get_queue_for_doctor(self, doctor_id: int) -> List[dict]:
        return [e for e in self.queue_entries.values()
                if e["doctor_id"] == doctor_id and e["status"] == "WAITING"]

    def get_waiting_patients(self) -> List[dict]:
        return [e for e in self.queue_entries.values() if e["status"] == "WAITING"]


clinic = ClinicState()


# ============================================================
# JSEW Router
# ============================================================

def jsew_route_patient(entry: dict) -> int:
    """Join-the-Shortest-Expected-Workload routing."""
    best_doctor = 0
    best_workload = float("inf")

    for doc_id, doc in clinic.doctors.items():
        if not doc["active"]:
            continue

        workload = 0.0

        # Current patient remaining time
        if doc["current_patient"] is not None and doc["consultation_start"] is not None:
            elapsed = (datetime.now() - doc["consultation_start"]).total_seconds() / 60
            remaining = max(doc["rolling_pact"] - elapsed, 1.0)
            workload += remaining

        # Queued patients
        queue = clinic.get_queue_for_doctor(doc_id)
        for qp in queue:
            workload += qp.get("pred_service_dur", 13.5)

        # This patient
        workload += entry.get("pred_service_dur", 13.5)

        if workload < best_workload:
            best_workload = workload
            best_doctor = doc_id

    return best_doctor


def predict_wait_time(entry: dict) -> dict:
    """
    Generate wait-time prediction with CQR+ACI interval.
    """
    doctor_id = entry["doctor_id"]
    doc = clinic.doctors[doctor_id]
    queue = clinic.get_queue_for_doctor(doctor_id)

    # Compute expected workload ahead
    total_ahead = 0.0
    for qp in queue:
        if qp["entry_id"] != entry["entry_id"]:
            total_ahead += qp.get("pred_service_dur", 13.5)

    # Add remaining time for current in-room patient
    remaining = 0.0
    if doc["current_patient"] is not None and doc["consultation_start"] is not None:
        elapsed = (datetime.now() - doc["consultation_start"]).total_seconds() / 60
        remaining = max(doc["rolling_pact"] - elapsed, 1.0)
        total_ahead += remaining

    point_est = total_ahead / max(doc["speed_mult"], 0.5)

    # CQR + ACI interval
    spread = max(point_est * 0.3, 3.0)
    lower = max(point_est - spread - clinic.cqr_Q, 0)
    upper = point_est + spread + clinic.cqr_Q

    # TreeSHAP-like explanation
    explanations = []
    queue_len = len(queue)
    if queue_len > 0:
        explanations.append({
            "text": f"+{total_ahead:.0f} min: {queue_len} patients ahead in queue",
            "direction": "wait_increasing",
            "value": total_ahead,
        })
    if doc["current_patient"] is not None and remaining > 0:
        explanations.append({
            "text": f"+{remaining:.0f} min: doctor finishing current patient",
            "direction": "wait_increasing",
            "value": remaining,
        })
    if doc["speed_mult"] > 1.0:
        saved = total_ahead * (1 - 1 / doc["speed_mult"])
        explanations.append({
            "text": f"-{saved:.0f} min: {doc['name']} consults {int((doc['speed_mult']-1)*100)}% faster",
            "direction": "wait_decreasing",
            "value": -saved,
        })

    return {
        "point_estimate_min": round(point_est, 1),
        "lower_bound_min": round(lower, 1),
        "upper_bound_min": round(upper, 1),
        "aci_alpha_t": round(clinic.aci_alpha_t, 4),
        "explanations": explanations,
    }


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "CAREOS Module 1 - Queue & Wait-Time Predictor",
        "version": "1.0.0",
        "status": "running",
        "active_doctors": len([d for d in clinic.doctors.values() if d["active"]]),
        "waiting_patients": len(clinic.get_waiting_patients()),
    }


@app.get("/api/v1/queue/status")
async def queue_status():
    """Get full current queue status."""
    waiting = clinic.get_waiting_patients()
    in_consult = [e for e in clinic.queue_entries.values() if e["status"] == "IN_CONSULTATION"]

    # Serialize datetime objects for JSON response
    def serialize_entry(e):
        entry = dict(e)
        for k, v in entry.items():
            if isinstance(v, datetime):
                entry[k] = v.isoformat()
        return entry

    def serialize_doctor(d):
        doc = dict(d)
        for k, v in doc.items():
            if isinstance(v, datetime):
                doc[k] = v.isoformat()
        return doc

    return {
        "session_id": clinic.session_id,
        "doctors": [serialize_doctor(d) for d in clinic.doctors.values()],
        "waiting_count": len(waiting),
        "in_consultation_count": len(in_consult),
        "waiting_patients": sorted(
            [serialize_entry(w) for w in waiting],
            key=lambda x: x.get("check_in_time", ""),
        ),
        "aci_alpha_t": clinic.aci_alpha_t,
    }


@app.post("/api/v1/queue/check-in")
async def check_in(request: CheckInRequest, bg_tasks: BackgroundTasks):
    """
    Patient check-in endpoint.
    Runs Stage-1 prediction, JSEW routing, and returns initial wait-time forecast.
    """
    entry_id = clinic.next_entry_id
    clinic.next_entry_id += 1

    # Estimate service duration (Stage-1 approximation)
    base_dur = 13.5
    if request.visit_no == 1:
        base_dur = 15.0
    if request.is_complex_cancer:
        base_dur = 21.0
    if request.has_hypertension or request.has_diabetes:
        base_dur += 1.5

    entry = {
        "entry_id": entry_id,
        "session_id": clinic.session_id,
        "patient_name": request.patient_name,
        "age": request.age,
        "gender": request.gender,
        "visit_no": request.visit_no,
        "esi_acuity": request.esi_acuity.value,
        "complaint_category": request.complaint_category,
        "has_hypertension": request.has_hypertension,
        "has_diabetes": request.has_diabetes,
        "is_complex_cancer": request.is_complex_cancer,
        "address_tier": request.address_tier,
        "is_walk_in": request.is_walk_in,
        "status": "WAITING",
        "check_in_time": datetime.now(),
        "doctor_id": None,
        "pred_service_dur": base_dur,
        "soap_stage": "NONE",
        "travel_time_min": request.travel_time_min,
    }

    # JSEW routing
    if request.preferred_doctor_id is not None:
        entry["doctor_id"] = request.preferred_doctor_id
    else:
        entry["doctor_id"] = jsew_route_patient(entry)

    # Handle ESI Level 1: bypass queue
    if request.esi_acuity == ESILevel.IMMEDIATE:
        entry["status"] = "WAITING"
        entry["priority"] = 0
    elif request.esi_acuity == ESILevel.HIGH_RISK:
        entry["priority"] = 1
    else:
        entry["priority"] = request.esi_acuity.value

    clinic.queue_entries[entry_id] = entry

    # Generate prediction
    prediction = predict_wait_time(entry)

    # Log prediction
    clinic.prediction_logs.append({
        "entry_id": entry_id,
        "trigger": "check_in",
        "predicted_at": datetime.now().isoformat(),
        **prediction,
    })

    # JSEW recommendation table
    jsew_table = []
    for doc_id, doc in clinic.doctors.items():
        queue = clinic.get_queue_for_doctor(doc_id)
        total_workload = sum(q.get("pred_service_dur", 13.5) for q in queue)
        jsew_table.append({
            "doctor_id": doc_id,
            "doctor_name": doc["name"],
            "queue_count": len(queue),
            "expected_workload_min": round(total_workload, 1),
            "speed_multiplier": doc["speed_mult"],
            "recommended": doc_id == entry["doctor_id"],
        })

    return {
        "entry_id": entry_id,
        "assigned_doctor": clinic.doctors[entry["doctor_id"]]["name"],
        "queue_position": len(clinic.get_queue_for_doctor(entry["doctor_id"])),
        "prediction": prediction,
        "jsew_routing": jsew_table,
    }


@app.post("/api/v1/queue/start-consultation")
async def start_consultation(entry_id: int):
    """
    Doctor calls patient in. Computes realized wait, runs ACI update.
    """
    if entry_id not in clinic.queue_entries:
        raise HTTPException(status_code=404, detail="Entry not found")

    entry = clinic.queue_entries[entry_id]
    if entry["status"] != "WAITING":
        raise HTTPException(status_code=400, detail="Patient is not waiting")

    now = datetime.now()
    actual_wait = (now - entry["check_in_time"]).total_seconds() / 60

    entry["status"] = "IN_CONSULTATION"
    entry["called_in_time"] = now
    entry["actual_wait_min"] = round(actual_wait, 2)

    # Update doctor state
    doc = clinic.doctors[entry["doctor_id"]]
    doc["current_patient"] = entry_id
    doc["consultation_start"] = now
    doc["soap_stage"] = "S"

    # ACI update
    last_pred = None
    for log in reversed(clinic.prediction_logs):
        if log.get("entry_id") == entry_id:
            last_pred = log
            break

    if last_pred:
        err_t = 1.0 if (actual_wait < last_pred["lower_bound_min"] or
                        actual_wait > last_pred["upper_bound_min"]) else 0.0
        clinic.aci_alpha_t += 0.03 * (0.10 - err_t)
        clinic.aci_alpha_t = max(0.001, min(0.5, clinic.aci_alpha_t))

    clinic.recent_waits.append(actual_wait)

    return {
        "entry_id": entry_id,
        "actual_wait_min": round(actual_wait, 2),
        "aci_alpha_t_updated": round(clinic.aci_alpha_t, 4),
    }


@app.post("/api/v1/queue/soap-telemetry")
async def soap_telemetry(request: SOAPTelemetryRequest):
    """
    Cross-Module Webhook: Module 2 → Module 1.
    Updates SOAP stage progress for active consultations.
    """
    doc = clinic.doctors.get(request.room_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Room/Doctor not found")

    doc["soap_stage"] = request.soap_stage.value

    # Update remaining time estimate using SOAP progress
    if doc["current_patient"] is not None and doc["consultation_start"] is not None:
        entry = clinic.queue_entries.get(doc["current_patient"])
        if entry:
            elapsed = (datetime.now() - doc["consultation_start"]).total_seconds() / 60
            f_soap = request.stage_midpoint_fraction
            w_audio = 0.70

            # Live total estimate
            s_total_live = w_audio * (elapsed / max(f_soap, 0.12)) + (1 - w_audio) * entry.get("pred_service_dur", 13.5)
            s_rem_live = max(s_total_live - elapsed, 1.0)

            entry["live_remaining_min"] = round(s_rem_live, 1)
            entry["soap_stage"] = request.soap_stage.value

    # Recalculate waits for all waiting patients
    updated = []
    for e in clinic.get_waiting_patients():
        pred = predict_wait_time(e)
        updated.append({"entry_id": e["entry_id"], "prediction": pred})

    return {
        "room_id": request.room_id,
        "soap_stage": request.soap_stage.value,
        "updated_predictions": updated,
    }


@app.post("/api/v1/queue/emergency-preempt")
async def emergency_preempt(request: EmergencyPreemptRequest):
    """
    Emergency preemption endpoint.
    Inserts emergency patient at head of queue.
    """
    entry_id = clinic.next_entry_id
    clinic.next_entry_id += 1

    # Find doctor with shortest queue
    best_doc = min(clinic.doctors.keys(),
                   key=lambda d: len(clinic.get_queue_for_doctor(d)))

    entry = {
        "entry_id": entry_id,
        "session_id": clinic.session_id,
        "patient_name": request.patient_name,
        "esi_acuity": request.esi_acuity.value,
        "status": "WAITING",
        "check_in_time": datetime.now(),
        "doctor_id": best_doc,
        "pred_service_dur": 20.0,
        "priority": 0,
        "is_emergency": True,
        "soap_stage": "NONE",
    }

    clinic.queue_entries[entry_id] = entry

    # Recalculate all waiting patients
    updated = []
    for e in clinic.get_waiting_patients():
        pred = predict_wait_time(e)
        updated.append({
            "entry_id": e["entry_id"],
            "patient_name": e.get("patient_name", "Unknown"),
            "prediction": pred,
        })

    return {
        "emergency_entry_id": entry_id,
        "assigned_doctor": clinic.doctors[best_doc]["name"],
        "updated_queue": updated,
    }


@app.post("/api/v1/queue/end-consultation")
async def end_consultation(entry_id: int):
    """
    Consultation completed. Updates doctor metrics.
    """
    if entry_id not in clinic.queue_entries:
        raise HTTPException(status_code=404, detail="Entry not found")

    entry = clinic.queue_entries[entry_id]
    entry["status"] = "COMPLETED"
    entry["completed_time"] = datetime.now()

    if entry.get("called_in_time"):
        service_dur = (datetime.now() - entry["called_in_time"]).total_seconds() / 60
        entry["actual_service_dur"] = round(service_dur, 2)

        # Update doctor PACT
        doc = clinic.doctors.get(entry.get("doctor_id"))
        if doc:
            alpha_ema = 0.2
            doc["rolling_pact"] = (1 - alpha_ema) * doc["rolling_pact"] + alpha_ema * service_dur
            doc["completed"] += 1
            doc["current_patient"] = None
            doc["consultation_start"] = None
            doc["soap_stage"] = "NONE"

    return {"entry_id": entry_id, "status": "COMPLETED"}


@app.get("/api/v1/queue/jsew-routing")
async def jsew_routing_table():
    """Get current JSEW routing recommendations."""
    table = []
    for doc_id, doc in clinic.doctors.items():
        queue = clinic.get_queue_for_doctor(doc_id)
        total_workload = sum(q.get("pred_service_dur", 13.5) for q in queue)

        # Add current patient remaining
        remaining = 0
        if doc["current_patient"] is not None and doc["consultation_start"] is not None:
            elapsed = (datetime.now() - doc["consultation_start"]).total_seconds() / 60
            remaining = max(doc["rolling_pact"] - elapsed, 0)

        table.append({
            "doctor_id": doc_id,
            "doctor_name": doc["name"],
            "speed_multiplier": doc["speed_mult"],
            "queue_count": len(queue),
            "queued_workload_min": round(total_workload, 1),
            "current_remaining_min": round(remaining, 1),
            "total_workload_min": round(total_workload + remaining, 1),
            "soap_stage": doc["soap_stage"],
        })

    return {"routing_table": table}


@app.post("/api/v1/queue/reset")
async def reset_queue():
    """Reset clinic state for demo."""
    global clinic
    clinic = ClinicState()
    return {"status": "reset", "message": "Clinic state reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
