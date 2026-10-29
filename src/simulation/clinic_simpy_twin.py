"""
CAREOS Module 1 - Calibrated SimPy Digital Twin Simulation Engine
Non-Stationary, Multi-Class Priority Queueing Network (M_t/G_i/c)

Features:
- Dual-Stream Arrival Generator (scheduled + NHPP walk-ins)
- Patient-Dependent Stochastic Service Times (Log-Normal calibrated to Hangu/PriMock57)
- Three-Mode Priority & Emergency Preemption (HOL, Preemptive-Resume, Impatience)
- 5-Level ESI Triage
- No-Show modelling from Brazil dataset
- SOAP stage simulation for Module 2 telemetry
"""

import simpy
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import sys
import os
import random
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from configs.clinic_config import (
    LOGNORMAL_PARAMS, ESI_LEVELS, COMPLAINT_CATEGORIES,
    COMPLAINT_DURATION_MULTIPLIERS, NO_SHOW_RATES, SOAP_STAGES,
    SIM_CONFIG, ADDRESS_TIERS,
)


class PatientStatus(Enum):
    BOOKED = "BOOKED"
    WAITING = "WAITING"
    IN_CONSULTATION = "IN_CONSULTATION"
    COMPLETED = "COMPLETED"
    NO_SHOW = "NO_SHOW"
    PREEMPTED = "PREEMPTED"
    BALKED = "BALKED"
    RENEGED = "RENEGED"


class SOAPStage(Enum):
    NONE = "NONE"
    SUBJECTIVE = "SUBJECTIVE"
    OBJECTIVE = "OBJECTIVE"
    ASSESSMENT = "ASSESSMENT"
    PLAN = "PLAN"


@dataclass
class SimulatedPatient:
    patient_id: int
    is_walk_in: bool
    scheduled_time: float  # minutes from session start
    actual_arrival_time: float  # with unpunctuality offset
    visit_no: int  # 1 = first-time, >1 = follow-up
    age: int
    gender: str  # 'M' or 'F'
    is_complex_cancer: bool
    has_hypertension: bool
    has_diabetes: bool
    comorbidity_count: int
    complaint_category: str
    address_tier: str  # 'local', 'out_of_city', 'out_of_province'
    esi_acuity: int  # 1-5
    am_pm: str  # 'morning' or 'afternoon'
    doctor_id: Optional[int] = None
    status: PatientStatus = PatientStatus.BOOKED
    check_in_time: Optional[float] = None
    called_in_time: Optional[float] = None
    consultation_start_time: Optional[float] = None
    consultation_end_time: Optional[float] = None
    actual_service_duration: Optional[float] = None
    predicted_service_duration: Optional[float] = None
    actual_wait_time: Optional[float] = None
    predicted_wait_time: Optional[float] = None
    predicted_wait_lower: Optional[float] = None
    predicted_wait_upper: Optional[float] = None
    soap_stages_times: Dict[str, float] = field(default_factory=dict)
    queue_position_at_checkin: int = 0
    no_show: bool = False


@dataclass
class DoctorState:
    doctor_id: int
    name: str
    speed_multiplier: float
    is_active: bool = True
    current_patient_id: Optional[int] = None
    consultation_start_time: Optional[float] = None
    current_soap_stage: SOAPStage = SOAPStage.NONE
    rolling_pact_min: float = 13.5
    rolling_pnp_rate: float = 4.0
    completed_count: int = 0
    total_service_time: float = 0.0


def get_service_time_params(patient: SimulatedPatient) -> Tuple[float, float]:
    """
    Determine the Log-Normal (mu, sigma) parameters for a patient
    based on their clinical attributes, calibrated to Hangu/PriMock57 data.
    """
    if patient.is_complex_cancer and patient.visit_no == 1:
        params = LOGNORMAL_PARAMS["complex_cancer_first"]
    elif patient.is_complex_cancer and patient.visit_no > 1:
        params = LOGNORMAL_PARAMS["complex_cancer_followup"]
    elif patient.visit_no == 1 and patient.gender == "F":
        params = LOGNORMAL_PARAMS["first_time_female"]
    elif patient.visit_no == 1 and patient.gender == "M":
        params = LOGNORMAL_PARAMS["first_time_male"]
    elif patient.visit_no > 1 and patient.gender == "F":
        params = LOGNORMAL_PARAMS["followup_female"]
    elif patient.visit_no > 1 and patient.gender == "M":
        params = LOGNORMAL_PARAMS["followup_male"]
    else:
        params = LOGNORMAL_PARAMS["primary_care_routine"]

    mu = params["mu"]
    sigma = params["sigma"]

    # Apply complaint category multiplier
    cat_mult = COMPLAINT_DURATION_MULTIPLIERS.get(patient.complaint_category, 1.0)
    mu += np.log(cat_mult)

    # Apply address tier adjustment (from Hangu regression)
    addr_extra = ADDRESS_TIERS.get(patient.address_tier, {}).get("extra_seconds", 0)
    if addr_extra > 0:
        mu += np.log(1 + addr_extra / (np.exp(mu + sigma ** 2 / 2) * 60))

    # AM/PM adjustment
    if patient.am_pm == "afternoon":
        mu += np.log(1 + 49.37 / (np.exp(mu + sigma ** 2 / 2) * 60))

    return mu, sigma


def sample_service_duration(patient: SimulatedPatient, doctor: DoctorState, rng: np.random.Generator) -> float:
    """
    Sample consultation duration from calibrated Log-Normal distribution,
    adjusted by doctor speed multiplier.
    """
    mu, sigma = get_service_time_params(patient)
    # Draw from log-normal (returns minutes)
    duration_min = rng.lognormal(mu, sigma)
    # Apply doctor speed multiplier (faster doctor = shorter duration)
    duration_min /= doctor.speed_multiplier
    # Clamp to [3, 60] minutes
    duration_min = np.clip(duration_min, 3.0, 60.0)
    return float(duration_min)


def nhpp_walk_in_intensity(t_min: float, config: dict) -> float:
    """
    Non-Homogeneous Poisson Process intensity function λ(t) for walk-in arrivals.
    Twin-peak curve: morning rush ~9:00-9:30 and afternoon ~13:30-14:00.
    t_min: minutes from midnight.
    """
    hour = t_min / 60.0
    base = config["base_walk_in_lambda"] / 60.0  # per minute

    # Morning peak: Gaussian centered at 9:15 AM (9.25h)
    morning_peak = config["peak_lambda_multiplier"] * base * np.exp(
        -0.5 * ((hour - config["peak_morning_hour"]) / 0.5) ** 2
    )
    # Afternoon peak: Gaussian centered at 1:45 PM (13.75h)
    afternoon_peak = config["peak_lambda_multiplier"] * base * np.exp(
        -0.5 * ((hour - config["peak_afternoon_hour"]) / 0.5) ** 2
    )

    return base + morning_peak + afternoon_peak


def simulate_soap_stages(duration_min: float, rng: np.random.Generator) -> Dict[str, float]:
    """
    Simulate SOAP stage transition times within a consultation.
    Returns dict of stage -> time offset from consultation start (in minutes).
    """
    stages = {}
    cumulative = 0.0
    for stage_name, stage_info in SOAP_STAGES.items():
        low, high = stage_info["range"]
        # Add some noise to the transition point
        frac = rng.uniform(low, high)
        stages[stage_name] = frac * duration_min
    return stages


def generate_patient_pool(session_id: int, am_pm: str, config: dict, rng: np.random.Generator) -> List[SimulatedPatient]:
    """
    Generate the patient pool for one clinic session.
    Combines scheduled appointments with walk-in arrivals.
    """
    patients = []
    patient_counter = session_id * 1000

    num_doctors = config["num_doctors"]
    session_duration = config["session_duration_hours"] * 60  # minutes
    start_hour = config["morning_start_hour"] if am_pm == "morning" else config["afternoon_start_hour"]
    session_start_min = start_hour * 60  # minutes from midnight

    # 1. Scheduled appointments
    num_scheduled = int(config["patients_per_session"] * (1 - config["walk_in_fraction"]))
    interval = config["appointment_interval_min"]

    for i in range(num_scheduled):
        patient_counter += 1
        slot_time = (i // num_doctors) * interval  # minutes from session start
        doctor_slot = i % num_doctors

        # Determine no-show based on lead time (simulate random lead time)
        lead_days = rng.choice([0, 1, 3, 7, 14, 21, 30], p=[0.10, 0.20, 0.20, 0.20, 0.15, 0.10, 0.05])
        if lead_days == 0:
            no_show_rate = NO_SHOW_RATES["same_day"]
        elif lead_days <= 7:
            no_show_rate = NO_SHOW_RATES["1_7_days"]
        elif lead_days <= 14:
            no_show_rate = NO_SHOW_RATES["8_14_days"]
        elif lead_days <= 30:
            no_show_rate = NO_SHOW_RATES["15_30_days"]
        else:
            no_show_rate = NO_SHOW_RATES["31_60_days"]

        is_no_show = rng.random() < no_show_rate

        # Arrival unpunctuality for attending patients
        if not is_no_show:
            early_offset = rng.normal(config["arrival_early_mean_min"], config["arrival_early_std_min"])
            # 15% chance of late arrival (tardy)
            if rng.random() < 0.15:
                early_offset = rng.exponential(5.0)  # late by ~5 min on average
            actual_arrival = slot_time + early_offset
        else:
            actual_arrival = slot_time

        # Patient demographics
        gender = rng.choice(["M", "F"])
        age = int(np.clip(rng.normal(45, 18), 5, 90))
        visit_no = rng.choice([1, 2, 3, 4, 5], p=[0.35, 0.25, 0.20, 0.12, 0.08])
        is_complex = rng.random() < 0.08
        has_hyp = rng.random() < 0.197
        has_diab = rng.random() < 0.072
        comorbidity_count = int(has_hyp) + int(has_diab) + int(is_complex)
        complaint = rng.choice(COMPLAINT_CATEGORIES)
        address = rng.choice(
            list(ADDRESS_TIERS.keys()),
            p=[v["weight"] for v in ADDRESS_TIERS.values()]
        )
        # ESI acuity: mostly 3-5 for scheduled
        esi = rng.choice([2, 3, 4, 5], p=[0.05, 0.35, 0.40, 0.20])

        p = SimulatedPatient(
            patient_id=patient_counter,
            is_walk_in=False,
            scheduled_time=slot_time,
            actual_arrival_time=max(0, actual_arrival),
            visit_no=visit_no,
            age=age,
            gender=gender,
            is_complex_cancer=is_complex,
            has_hypertension=has_hyp,
            has_diabetes=has_diab,
            comorbidity_count=comorbidity_count,
            complaint_category=complaint,
            address_tier=address,
            esi_acuity=esi,
            am_pm=am_pm,
            doctor_id=doctor_slot,
            no_show=is_no_show,
        )
        if is_no_show:
            p.status = PatientStatus.NO_SHOW
        patients.append(p)

    # 2. Walk-in arrivals via thinned NHPP
    t = 0.0
    max_lambda = config["peak_lambda_multiplier"] * config["base_walk_in_lambda"] / 60.0 * 3
    while t < session_duration:
        # Thinning algorithm for NHPP
        inter = rng.exponential(1.0 / max(max_lambda, 0.01))
        t += inter
        if t >= session_duration:
            break
        # Accept/reject based on actual intensity
        actual_t_min = session_start_min + t
        actual_lambda = nhpp_walk_in_intensity(actual_t_min, config)
        if rng.random() < actual_lambda / max(max_lambda, 0.01):
            patient_counter += 1
            gender = rng.choice(["M", "F"])
            age = int(np.clip(rng.normal(40, 20), 3, 95))
            visit_no = rng.choice([1, 2, 3], p=[0.55, 0.30, 0.15])
            is_complex = rng.random() < 0.05
            has_hyp = rng.random() < 0.15
            has_diab = rng.random() < 0.06
            comorbidity_count = int(has_hyp) + int(has_diab) + int(is_complex)
            complaint = rng.choice(COMPLAINT_CATEGORIES)
            address = rng.choice(
                list(ADDRESS_TIERS.keys()),
                p=[v["weight"] for v in ADDRESS_TIERS.values()]
            )
            # Walk-ins: wider ESI distribution, small chance of emergency
            esi = rng.choice([1, 2, 3, 4, 5], p=[0.01, 0.08, 0.35, 0.36, 0.20])

            p = SimulatedPatient(
                patient_id=patient_counter,
                is_walk_in=True,
                scheduled_time=t,
                actual_arrival_time=t,
                visit_no=visit_no,
                age=age,
                gender=gender,
                is_complex_cancer=is_complex,
                has_hypertension=has_hyp,
                has_diabetes=has_diab,
                comorbidity_count=comorbidity_count,
                complaint_category=complaint,
                address_tier=address,
                esi_acuity=esi,
                am_pm=am_pm,
                doctor_id=None,  # To be assigned by JSEW routing
            )
            patients.append(p)

    # Sort by actual arrival time
    patients.sort(key=lambda p: p.actual_arrival_time)
    return patients


class ClinicDigitalTwin:
    """
    Full SimPy-based Clinical Digital Twin for Module 1.
    Implements M_t/G_i/c queueing with priority preemption.
    """

    def __init__(self, config: dict = None, seed: int = 42, preemption_mode: str = "HOL"):
        self.config = config or SIM_CONFIG
        self.rng = np.random.default_rng(seed)
        self.preemption_mode = preemption_mode  # "HOL", "PREEMPTIVE_RESUME", "IMPATIENCE"

        # Results storage
        self.all_patient_records: List[dict] = []
        self.session_summaries: List[dict] = []

    def _jsew_route(self, patient: SimulatedPatient, doctors: List[DoctorState],
                    queues: Dict[int, List[SimulatedPatient]], env_now: float) -> int:
        """
        Join-the-Shortest-Expected-Workload (JSEW) routing.
        Routes unassigned patient to doctor with minimum total expected workload.
        """
        best_doctor = 0
        best_workload = float("inf")

        for doc in doctors:
            if not doc.is_active:
                continue

            workload = 0.0

            # Remaining service time for current patient
            if doc.current_patient_id is not None and doc.consultation_start_time is not None:
                elapsed = env_now - doc.consultation_start_time
                # Use conditional remaining time (simplified)
                expected_total = doc.rolling_pact_min
                remaining = max(expected_total - elapsed, 1.0)
                workload += remaining

            # Sum expected durations of queued patients
            for qp in queues.get(doc.doctor_id, []):
                mu, sigma = get_service_time_params(qp)
                expected_dur = np.exp(mu + sigma ** 2 / 2) / doc.speed_multiplier
                workload += expected_dur

            # Add expected duration for this patient on this doctor
            mu, sigma = get_service_time_params(patient)
            expected_dur = np.exp(mu + sigma ** 2 / 2) / doc.speed_multiplier
            workload += expected_dur

            if workload < best_workload:
                best_workload = workload
                best_doctor = doc.doctor_id

        return best_doctor

    def run_session(self, session_id: int, am_pm: str = "morning") -> pd.DataFrame:
        """Run a single clinic session simulation."""
        env = simpy.Environment()
        config = self.config
        num_doctors = config["num_doctors"]

        # Initialize doctors
        doctors = []
        for d in range(num_doctors):
            doc = DoctorState(
                doctor_id=d,
                name=config["doctor_names"][d],
                speed_multiplier=config["doctor_speed_multipliers"][d],
            )
            doctors.append(doc)

        # Doctor resources (preemptive or not)
        if self.preemption_mode == "PREEMPTIVE_RESUME":
            resources = {d: simpy.PreemptiveResource(env, capacity=1) for d in range(num_doctors)}
        else:
            resources = {d: simpy.Resource(env, capacity=1) for d in range(num_doctors)}

        # Queues per doctor
        queues: Dict[int, List[SimulatedPatient]] = {d: [] for d in range(num_doctors)}

        # Generate patients
        patients = generate_patient_pool(session_id, am_pm, config, self.rng)

        # Track completed patients
        completed_patients: List[SimulatedPatient] = []
        recent_waits: List[float] = []

        def patient_process(patient: SimulatedPatient):
            """SimPy process for a single patient."""
            # Wait until arrival time
            if patient.actual_arrival_time > env.now:
                yield env.timeout(patient.actual_arrival_time - env.now)

            # Skip no-shows
            if patient.no_show:
                return

            # Check-in
            patient.check_in_time = env.now
            patient.status = PatientStatus.WAITING

            # JSEW routing for walk-ins (unassigned patients)
            if patient.doctor_id is None:
                patient.doctor_id = self._jsew_route(patient, doctors, queues, env.now)

            doc_id = patient.doctor_id
            doc = doctors[doc_id]

            # Count queue position
            patient.queue_position_at_checkin = len(queues[doc_id])
            queues[doc_id].append(patient)

            # Calculate predicted service duration (Stage 1)
            mu, sigma = get_service_time_params(patient)
            patient.predicted_service_duration = np.exp(mu + sigma ** 2 / 2) / doc.speed_multiplier

            # Request doctor resource
            priority = ESI_LEVELS[patient.esi_acuity]["priority"]

            if self.preemption_mode == "PREEMPTIVE_RESUME" and patient.esi_acuity == 1:
                req = resources[doc_id].request(priority=priority, preempt=True)
            elif self.preemption_mode == "PREEMPTIVE_RESUME":
                req = resources[doc_id].request(priority=priority, preempt=False)
            else:
                req = resources[doc_id].request()

            # Wait in queue
            yield req

            # Called in
            patient.called_in_time = env.now
            patient.actual_wait_time = patient.called_in_time - patient.check_in_time
            patient.status = PatientStatus.IN_CONSULTATION
            patient.consultation_start_time = env.now

            # Remove from queue
            if patient in queues[doc_id]:
                queues[doc_id].remove(patient)

            # Update doctor state
            doc.current_patient_id = patient.patient_id
            doc.consultation_start_time = env.now

            # Sample actual service duration
            service_dur = sample_service_duration(patient, doc, self.rng)
            patient.actual_service_duration = service_dur

            # Simulate SOAP stages
            patient.soap_stages_times = simulate_soap_stages(service_dur, self.rng)

            # Simulate SOAP stage transitions
            soap_order = ["SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN"]
            elapsed = 0.0
            for stage in soap_order:
                stage_time = patient.soap_stages_times[stage]
                wait_dur = max(stage_time - elapsed, 0.01)
                doc.current_soap_stage = SOAPStage[stage]
                try:
                    yield env.timeout(wait_dur)
                    elapsed += wait_dur
                except simpy.Interrupt:
                    # Preempted by emergency
                    patient.status = PatientStatus.PREEMPTED
                    remaining = service_dur - elapsed
                    # Wait for emergency to complete, then resume
                    yield env.timeout(0.5)  # brief interruption delay
                    if remaining > 0:
                        yield env.timeout(remaining)
                    break

            # Remaining time after PLAN stage
            remaining_after_soap = service_dur - elapsed
            if remaining_after_soap > 0:
                yield env.timeout(remaining_after_soap)

            # Consultation complete
            patient.consultation_end_time = env.now
            patient.status = PatientStatus.COMPLETED

            # Update doctor metrics
            doc.completed_count += 1
            doc.total_service_time += service_dur
            # Exponential moving average for PACT
            alpha_ema = 0.2
            doc.rolling_pact_min = (1 - alpha_ema) * doc.rolling_pact_min + alpha_ema * service_dur
            doc.rolling_pnp_rate = doc.completed_count / max((env.now / 60.0), 0.01)
            doc.current_patient_id = None
            doc.consultation_start_time = None
            doc.current_soap_stage = SOAPStage.NONE

            # Track recent waits
            recent_waits.append(patient.actual_wait_time)

            # Release resource
            resources[doc_id].release(req)
            completed_patients.append(patient)

        # Start all patient processes
        for patient in patients:
            env.process(patient_process(patient))

        # Run simulation
        session_duration = config["session_duration_hours"] * 60 + 60  # extra buffer
        env.run(until=session_duration)

        # Collect results
        records = []
        for p in patients:
            record = {
                "session_id": session_id,
                "patient_id": p.patient_id,
                "is_walk_in": p.is_walk_in,
                "scheduled_time": p.scheduled_time,
                "actual_arrival_time": p.actual_arrival_time,
                "visit_no": p.visit_no,
                "age": p.age,
                "gender": p.gender,
                "is_complex_cancer": p.is_complex_cancer,
                "has_hypertension": p.has_hypertension,
                "has_diabetes": p.has_diabetes,
                "comorbidity_count": p.comorbidity_count,
                "complaint_category": p.complaint_category,
                "address_tier": p.address_tier,
                "esi_acuity": p.esi_acuity,
                "am_pm": am_pm,
                "doctor_id": p.doctor_id,
                "status": p.status.value,
                "check_in_time": p.check_in_time,
                "called_in_time": p.called_in_time,
                "consultation_start_time": p.consultation_start_time,
                "consultation_end_time": p.consultation_end_time,
                "actual_service_duration": p.actual_service_duration,
                "predicted_service_duration": p.predicted_service_duration,
                "actual_wait_time": p.actual_wait_time,
                "queue_position_at_checkin": p.queue_position_at_checkin,
                "no_show": p.no_show,
                "soap_subjective_time": p.soap_stages_times.get("SUBJECTIVE"),
                "soap_objective_time": p.soap_stages_times.get("OBJECTIVE"),
                "soap_assessment_time": p.soap_stages_times.get("ASSESSMENT"),
                "soap_plan_time": p.soap_stages_times.get("PLAN"),
            }
            records.append(record)

        self.all_patient_records.extend(records)
        return pd.DataFrame(records)

    def run_multi_session(self, num_sessions: int = None) -> pd.DataFrame:
        """Run multiple clinic sessions and collect all data."""
        if num_sessions is None:
            num_sessions = self.config["num_sessions"]

        print(f"Running {num_sessions} clinic sessions...")
        all_dfs = []

        for s in range(num_sessions):
            am_pm = "morning" if s % 2 == 0 else "afternoon"
            df = self.run_session(session_id=s, am_pm=am_pm)
            all_dfs.append(df)

            if (s + 1) % 10 == 0:
                print(f"  Completed {s + 1}/{num_sessions} sessions")

        full_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Total records: {len(full_df)}")
        print(f"Completed consultations: {len(full_df[full_df['status'] == 'COMPLETED'])}")
        print(f"No-shows: {len(full_df[full_df['status'] == 'NO_SHOW'])}")

        return full_df


def run_simulation_and_save(output_path: str = None, num_sessions: int = 100, seed: int = 42):
    """
    Main entry point: run the full simulation and save dataset.
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "simulated_clinic_data.parquet")

    twin = ClinicDigitalTwin(seed=seed, preemption_mode="HOL")
    df = twin.run_multi_session(num_sessions=num_sessions)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")

    # Print summary statistics
    completed = df[df["status"] == "COMPLETED"]
    if len(completed) > 0:
        print(f"\n--- Simulation Summary ---")
        print(f"Total patients generated: {len(df)}")
        print(f"Completed consultations:  {len(completed)}")
        print(f"Mean wait time:           {completed['actual_wait_time'].mean():.2f} min")
        print(f"Median wait time:         {completed['actual_wait_time'].median():.2f} min")
        print(f"P90 wait time:            {completed['actual_wait_time'].quantile(0.9):.2f} min")
        print(f"Mean service duration:    {completed['actual_service_duration'].mean():.2f} min")
        print(f"Median service duration:  {completed['actual_service_duration'].median():.2f} min")

    return df


if __name__ == "__main__":
    run_simulation_and_save(num_sessions=100)
