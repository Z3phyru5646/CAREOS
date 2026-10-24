"""
CAREOS Module 1 - Clinic Configuration
Empirical parameters from Hangu, PriMock57, and Brazil No-Show datasets.
"""

# ============================================================
# Log-Normal Service Time Parameters (from Section 2.1)
# Each subgroup: (mu, sigma) for LogNormal distribution
# ============================================================

LOGNORMAL_PARAMS = {
    # Primary Care Routine (PriMock57, n=57)
    "primary_care_routine": {"mu": 2.137, "sigma": 0.372, "mean_min": 9.08, "sd_min": 3.50},
    # General OPD Female (Hangu, n=6637)
    "general_opd_female": {"mu": 2.495, "sigma": 0.442, "mean_min": 13.36, "sd_min": 6.21},
    # General OPD Male (Hangu)
    "general_opd_male": {"mu": 2.485, "sigma": 0.466, "mean_min": 13.37, "sd_min": 6.58},
    # First-Time Visit Female
    "first_time_female": {"mu": 2.635, "sigma": 0.423, "mean_min": 15.26, "sd_min": 6.76},
    # First-Time Visit Male
    "first_time_male": {"mu": 2.616, "sigma": 0.435, "mean_min": 15.03, "sd_min": 6.86},
    # Follow-Up Visit Female
    "followup_female": {"mu": 2.417, "sigma": 0.423, "mean_min": 12.27, "sd_min": 5.43},
    # Follow-Up Visit Male
    "followup_male": {"mu": 2.416, "sigma": 0.431, "mean_min": 12.29, "sd_min": 5.55},
    # Non-Cancer First-Time
    "non_cancer_first": {"mu": 2.610, "sigma": 0.404, "mean_min": 14.76, "sd_min": 6.21},
    # Non-Cancer Follow-Up
    "non_cancer_followup": {"mu": 2.399, "sigma": 0.435, "mean_min": 12.11, "sd_min": 5.52},
    # Complex/Oncology First-Time
    "complex_cancer_first": {"mu": 3.062, "sigma": 0.035, "mean_min": 21.38, "sd_min": 0.76},
    # Complex/Oncology Follow-Up
    "complex_cancer_followup": {"mu": 2.524, "sigma": 0.412, "mean_min": 13.59, "sd_min": 5.85},
    # Morning Session
    "morning_session": {"mu": 2.481, "sigma": 0.412, "mean_min": 13.01, "sd_min": 5.59},
    # Afternoon Session
    "afternoon_session": {"mu": 2.511, "sigma": 0.468, "mean_min": 13.75, "sd_min": 6.81},
    # Out-of-Province Referral First Visit
    "out_of_province_first": {"mu": 2.826, "sigma": 0.368, "mean_min": 18.05, "sd_min": 6.87},
    # Literature Benchmark: High-Complexity Class A
    "high_complexity_benchmark": {"mu": 2.751, "sigma": 0.472, "mean_min": 17.50, "sd_min": 8.75},
}

# ============================================================
# ESI Acuity Levels (from MIMIC-IV-ED, Section 2.2)
# ============================================================

ESI_LEVELS = {
    1: {"name": "Immediate Resuscitation", "bypasses_queue": True, "priority": 0},
    2: {"name": "High-Risk/Severe Pain", "bypasses_queue": False, "priority": 1},
    3: {"name": "Stable, 2+ Resources", "bypasses_queue": False, "priority": 2},
    4: {"name": "One Resource", "bypasses_queue": False, "priority": 3},
    5: {"name": "Fast-Track Routine", "bypasses_queue": False, "priority": 4},
}

# ============================================================
# PriMock57 Complaint Categories (12 classes)
# ============================================================

COMPLAINT_CATEGORIES = [
    "Cardiovascular", "Gastroenteritis", "Migraine",
    "Urinary Tract Infection", "Upper Respiratory Infection",
    "Dermatitis", "Fever", "Anaphylactic Reaction",
    "Mental Health", "Otitis", "Asthma", "Physical Injury",
]

# Duration multipliers per complaint category (relative to baseline)
COMPLAINT_DURATION_MULTIPLIERS = {
    "Cardiovascular": 1.25,
    "Gastroenteritis": 0.90,
    "Migraine": 0.85,
    "Urinary Tract Infection": 0.80,
    "Upper Respiratory Infection": 0.75,
    "Dermatitis": 0.70,
    "Fever": 0.85,
    "Anaphylactic Reaction": 1.40,
    "Mental Health": 1.35,
    "Otitis": 0.75,
    "Asthma": 0.95,
    "Physical Injury": 1.10,
}

# ============================================================
# No-Show Rates by Lead Time (from Brazil Dataset, Section 2.2)
# ============================================================

NO_SHOW_RATES = {
    "same_day": 0.12,
    "1_7_days": 0.2415,
    "8_14_days": 0.3047,
    "15_30_days": 0.3259,
    "31_60_days": 0.3415,
}

# ============================================================
# SOAP Stage Progress Fractions (from Section 5.2)
# ============================================================

SOAP_STAGES = {
    "SUBJECTIVE": {"range": (0.00, 0.35), "midpoint": 0.18},
    "OBJECTIVE": {"range": (0.35, 0.60), "midpoint": 0.48},
    "ASSESSMENT": {"range": (0.60, 0.80), "midpoint": 0.70},
    "PLAN": {"range": (0.80, 1.00), "midpoint": 0.90},
}

# ============================================================
# Simulation Configuration
# ============================================================

SIM_CONFIG = {
    "num_doctors": 3,
    "doctor_speed_multipliers": [0.80, 1.00, 1.25],
    "doctor_names": ["Dr. Sharma", "Dr. Verma", "Dr. Patel"],
    "doctor_specialties": ["General Medicine", "General Medicine", "General Medicine"],
    "session_duration_hours": 4,
    "morning_start_hour": 9,
    "afternoon_start_hour": 13.5,
    "appointment_interval_min": 15,
    "patients_per_session": 60,
    "num_sessions": 100,
    "walk_in_fraction": 0.30,
    "arrival_early_mean_min": -5.0,
    "arrival_early_std_min": 1.7,
    "peak_morning_hour": 9.25,
    "peak_afternoon_hour": 13.75,
    "peak_lambda_multiplier": 2.5,
    "base_walk_in_lambda": 4.0,   # per hour
    "aci_gamma": 0.03,
    "aci_alpha_target": 0.10,
    "w_audio": 0.70,
    "asymmetric_alpha": 0.80,
}

# ============================================================
# Address Tiers
# ============================================================

ADDRESS_TIERS = {
    "local": {"extra_seconds": 0, "weight": 0.60},
    "out_of_city": {"extra_seconds": 61.47, "weight": 0.25},
    "out_of_province": {"extra_seconds": 152.43, "weight": 0.15},
}
