"""
CAREOS Module 1 - Queue-Mining & Mathematical Feature Engineering
Implements the 16-group, 28-variable feature specification from Section 4.2.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Dict, Optional
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from configs.clinic_config import LOGNORMAL_PARAMS, SOAP_STAGES, SIM_CONFIG


def conditional_remaining_service_time(mu: float, sigma: float, elapsed: float) -> float:
    """
    Exact analytical Conditional Expected Remaining Service Time
    for LogNormal(mu, sigma^2) given elapsed time e.

    E[S - e | S > e] = exp(mu + sigma^2/2) * [1 - Phi((ln(e) - mu - sigma^2)/sigma)]
                       / [1 - Phi((ln(e) - mu)/sigma)] - e

    Handles the Inspection Paradox correctly.
    """
    if elapsed <= 0:
        return np.exp(mu + sigma ** 2 / 2)

    ln_e = np.log(max(elapsed, 0.01))

    numerator_z = (ln_e - mu - sigma ** 2) / sigma
    denominator_z = (ln_e - mu) / sigma

    sf_num = norm.sf(numerator_z)  # 1 - Phi(z)
    sf_den = norm.sf(denominator_z)

    if sf_den < 1e-12:
        # Extremely long elapsed time, hazard rate nearly zero
        return max(elapsed * 0.1, 1.0)

    expected_remaining = np.exp(mu + sigma ** 2 / 2) * sf_num / sf_den - elapsed
    return max(expected_remaining, 0.5)


def compute_queue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full 16-group, 28-variable queue-mining feature set.
    Input: DataFrame with columns from simulation output.
    Output: DataFrame with engineered features added.
    """
    df = df.copy()

    # Filter to completed patients with valid times
    mask = df["status"] == "COMPLETED"

    # ======== Group 1: Temporal & Diurnal Cyclical ========
    df["minutes_from_midnight"] = df["actual_arrival_time"]
    # For morning sessions, add 9*60; for afternoon, add 13.5*60
    df.loc[df["am_pm"] == "morning", "minutes_from_midnight"] += 9 * 60
    df.loc[df["am_pm"] == "afternoon", "minutes_from_midnight"] += 13.5 * 60

    df["sin_time"] = np.sin(2 * np.pi * df["minutes_from_midnight"] / 1440)
    df["cos_time"] = np.cos(2 * np.pi * df["minutes_from_midnight"] / 1440)

    # Day of week (simulate for sessions)
    df["day_of_week"] = df["session_id"] % 7
    df["month"] = (df["session_id"] // 30) % 12 + 1
    df["is_holiday"] = 0  # Could be enhanced

    # Time zone smoothness (1 = smooth, 0 = peak rush)
    hour = df["minutes_from_midnight"] / 60
    df["time_zone_smoothness"] = 1
    df.loc[(hour >= 9.0) & (hour <= 9.5), "time_zone_smoothness"] = 0
    df.loc[(hour >= 13.5) & (hour <= 14.0), "time_zone_smoothness"] = 0

    # ======== Group 2: Queue-Length & Q-Lasso Fluid ========
    df["queue_length"] = df["queue_position_at_checkin"]

    # Compute per-session, per-doctor active and busy counts
    # (approximated from queue position data)
    df["active_doctors"] = SIM_CONFIG["num_doctors"]
    df["busy_doctors"] = df.groupby(["session_id"])["queue_length"].transform(
        lambda x: np.minimum(x, SIM_CONFIG["num_doctors"])
    ).clip(0, SIM_CONFIG["num_doctors"])

    # Q-Lasso fluid estimators
    # Compute mean duration per doctor from completed patients only (avoid NaN)
    completed_means = df[df["actual_service_duration"].notna()].groupby("doctor_id")["actual_service_duration"].mean()
    doc_mean_duration = df["doctor_id"].map(completed_means).fillna(13.5)

    df["fluid_estimator_1"] = df["queue_length"] / df["active_doctors"].clip(1) * doc_mean_duration
    overall_mean_dur = doc_mean_duration.mean() if doc_mean_duration.mean() > 0 else 13.5
    df["fluid_estimator_2"] = df["queue_length"] / (
        df["active_doctors"].clip(1) * doc_mean_duration / overall_mean_dur
    ).clip(0.01)

    # Queue composition ratios
    df["is_new_visit"] = (df["visit_no"] == 1).astype(int)
    # Per-session rolling ratio of new visits in queue
    df["queue_new_ratio"] = df.groupby("session_id")["is_new_visit"].transform(
        lambda x: x.expanding().mean()
    ).fillna(0.35)
    df["queue_return_ratio"] = 1 - df["queue_new_ratio"]

    # High acuity count ahead (ESI <= 2)
    df["is_high_acuity"] = (df["esi_acuity"] <= 2).astype(int)
    df["queued_high_acuity_count"] = df.groupby("session_id")["is_high_acuity"].transform(
        lambda x: x.expanding().sum()
    ).fillna(0)

    # Arrival rate: expanding count divided by elapsed session time in hours
    df["arrival_rate_1h"] = df.groupby("session_id")["actual_arrival_time"].transform(
        lambda x: x.expanding().count() / ((x + 1.0) / 60.0).clip(lower=0.1)
    ).fillna(SIM_CONFIG["base_walk_in_lambda"]).clip(upper=60.0)

    # ======== Group 3: Delay-History & Elapsed State ========
    # Rolling wait of last 5 and 10 patients
    completed_df = df[mask].sort_values(["session_id", "called_in_time"])

    rolling_wait_5 = completed_df.groupby("session_id")["actual_wait_time"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    rolling_wait_10 = completed_df.groupby("session_id")["actual_wait_time"].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )

    df["rolling_wait_last_5"] = 0.0
    df["rolling_wait_last_10"] = 0.0
    df.loc[mask, "rolling_wait_last_5"] = rolling_wait_5
    df.loc[mask, "rolling_wait_last_10"] = rolling_wait_10

    # Head-of-line elapsed wait (approximated)
    df["hol_elapsed_wait"] = df.groupby(["session_id", "doctor_id"])["actual_wait_time"].transform(
        lambda x: x.shift(1).fillna(0)
    )

    # In-room elapsed statistics (approximated from service durations)
    df["in_room_elapsed_mean"] = df.groupby(["session_id", "doctor_id"])["actual_service_duration"].transform(
        lambda x: x.shift(1).expanding().mean().fillna(13.5)
    )
    df["in_room_elapsed_min"] = df.groupby(["session_id", "doctor_id"])["actual_service_duration"].transform(
        lambda x: x.shift(1).expanding().min().fillna(5.0)
    )

    # Conditional remaining active time (using log-normal formula)
    # Approximated as: for each busy doctor, compute E[S-e|S>e]
    df["cond_remaining_active_time"] = df.apply(
        lambda row: _compute_cond_remaining(row), axis=1
    )

    # ======== Group 4: Two-Stage Workload & Clinical ========
    df["pred_own_duration"] = df["predicted_service_duration"].fillna(13.5)

    # Expected queued workload (sum of predicted durations ahead / num doctors)
    df["expected_queued_workload"] = (
        df["queue_length"] * df["pred_own_duration"] / df["active_doctors"].clip(1)
    )

    # Physician efficiency
    df["physician_pact"] = doc_mean_duration.fillna(13.5)
    df["physician_pnp"] = df.groupby(["session_id", "doctor_id"]).cumcount() / (
        df["actual_arrival_time"] / 60 + 0.1
    ).clip(0.1)

    # Patient clinical features (already in raw data)
    df["patient_acuity_esi"] = df["esi_acuity"]

    return df


def _compute_cond_remaining(row) -> float:
    """Compute conditional remaining time for a patient row."""
    if pd.isna(row.get("actual_service_duration")) or pd.isna(row.get("actual_wait_time")):
        return 10.0

    # Use approximate elapsed time for the doctor's current patient
    elapsed = max(row.get("in_room_elapsed_mean", 5.0) * 0.5, 0.1)

    # Get approximate log-normal params
    if row.get("is_complex_cancer", False) and row.get("visit_no", 1) == 1:
        mu, sigma = 2.751, 0.472
    elif row.get("visit_no", 1) == 1:
        mu, sigma = 2.616, 0.435
    else:
        mu, sigma = 2.417, 0.423

    return conditional_remaining_service_time(mu, sigma, elapsed)


def get_feature_columns() -> List[str]:
    """Return the list of all engineered feature column names."""
    return [
        # Group 1: Temporal
        "sin_time", "cos_time", "day_of_week", "month", "is_holiday",
        "time_zone_smoothness", "arrival_rate_1h",
        # Group 2: Queue-Length & Fluid
        "queue_length", "active_doctors", "busy_doctors",
        "fluid_estimator_1", "fluid_estimator_2",
        "queue_new_ratio", "queue_return_ratio", "queued_high_acuity_count",
        # Group 3: Delay-History
        "rolling_wait_last_5", "rolling_wait_last_10",
        "hol_elapsed_wait", "in_room_elapsed_mean", "in_room_elapsed_min",
        "cond_remaining_active_time",
        # Group 4: Two-Stage Workload & Clinical
        "pred_own_duration", "expected_queued_workload",
        "physician_pact", "physician_pnp",
        "patient_acuity_esi", "visit_no", "comorbidity_count",
        "is_walk_in",
    ]


def get_stage1_feature_columns() -> List[str]:
    """Features for Stage-1 Consultation Duration Predictor."""
    return [
        "visit_no", "age", "comorbidity_count", "esi_acuity",
        "is_complex_cancer", "has_hypertension", "has_diabetes",
        "is_walk_in", "sin_time", "cos_time", "day_of_week",
    ]


def get_stage2_feature_columns() -> List[str]:
    """Features for Stage-2 Wait-Time Predictor."""
    return get_feature_columns()
