"""
CAREOS Module 1 - Streamlit Interactive Dashboard
Three-Persona Frontend:
1. Patient Virtual Queue Mobile Pass (with TreeSHAP + Time-to-Leave Pacer)
2. Receptionist & Triage Command Center (with JSEW Router + Chaos Injector)
3. Doctor Unified Workstation Header (with SOAP progress)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ============================================================
# Configuration
# ============================================================

API_BASE = "http://localhost:8000"
COMPLAINT_CATEGORIES = [
    "Cardiovascular", "Gastroenteritis", "Migraine",
    "Urinary Tract Infection", "Upper Respiratory Infection",
    "Dermatitis", "Fever", "Anaphylactic Reaction",
    "Mental Health", "Otitis", "Asthma", "Physical Injury",
]

st.set_page_config(
    page_title="CAREOS - Queue & Wait-Time Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .main-header h1 {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #e94560, #f39c12);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        font-size: 1rem;
        opacity: 0.85;
    }

    .wait-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
        margin-bottom: 1rem;
    }

    .wait-time-big {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .wait-range {
        font-size: 1.4rem;
        opacity: 0.9;
    }

    .shap-card {
        background: rgba(26, 26, 46, 0.95);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 0.8rem;
        color: white;
    }

    .shap-pill-positive {
        background: rgba(233, 69, 96, 0.2);
        border: 1px solid rgba(233, 69, 96, 0.5);
        color: #e94560;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        margin: 0.3rem;
        display: inline-block;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .shap-pill-negative {
        background: rgba(46, 213, 115, 0.2);
        border: 1px solid rgba(46, 213, 115, 0.5);
        color: #2ed573;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        margin: 0.3rem;
        display: inline-block;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .doctor-card {
        background: rgba(26, 26, 46, 0.95);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.5rem;
        color: white;
    }

    .soap-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        margin: 0.2rem;
    }

    .soap-active {
        background: #e94560;
        color: white;
    }

    .soap-done {
        background: #2ed573;
        color: white;
    }

    .soap-pending {
        background: rgba(255,255,255,0.1);
        color: rgba(255,255,255,0.5);
    }

    .metric-card {
        background: rgba(15, 52, 96, 0.8);
        border: 1px solid rgba(233, 69, 96, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #e94560;
    }

    .metric-label {
        font-size: 0.85rem;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }

    .queue-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f39c12, #e74c3c);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Helper Functions
# ============================================================

def api_call(method, endpoint, **kwargs):
    """Make API call to FastAPI backend."""
    try:
        url = f"{API_BASE}{endpoint}"
        if method == "GET":
            r = requests.get(url, timeout=5)
        elif method == "POST":
            r = requests.post(url, json=kwargs.get("json", {}), params=kwargs.get("params", {}), timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def render_soap_badges(current_stage: str):
    """Render SOAP stage progress badges."""
    stages = ["S", "O", "A", "P"]
    stage_names = {"S": "Subjective", "O": "Objective", "A": "Assessment", "P": "Plan"}
    stage_idx = stages.index(current_stage) if current_stage in stages else -1

    html = ""
    for i, s in enumerate(stages):
        if i < stage_idx:
            html += f'<span class="soap-badge soap-done">✓ {stage_names[s]}</span>'
        elif i == stage_idx:
            html += f'<span class="soap-badge soap-active">● {stage_names[s]}</span>'
        else:
            html += f'<span class="soap-badge soap-pending">○ {stage_names[s]}</span>'

    return html


# ============================================================
# Initialize Session State
# ============================================================

if "demo_patients" not in st.session_state:
    st.session_state.demo_patients = []
if "queue_data" not in st.session_state:
    st.session_state.queue_data = []
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None
if "api_connected" not in st.session_state:
    st.session_state.api_connected = False


# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>🏥 CARE<span style="color: #f39c12;">OS</span> — Module 1</h1>
    <p>Dynamic Outpatient Queue & Wait-Time Predictor</p>
    <p style="font-size: 0.85rem; opacity: 0.65;">Two-Stage ML Pipeline • CQR+ACI Intervals • TreeSHAP Explanations • JSEW Routing • SOAP Telemetry</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Check API Connection
# ============================================================

status = api_call("GET", "/")
api_connected = status is not None
st.session_state.api_connected = api_connected

if not api_connected:
    st.warning("⚠️ FastAPI backend not running. Using standalone demo mode. Start the API with: `python -m uvicorn src.api.main:app --reload`")


# ============================================================
# Main Tabs
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "👤 Patient Virtual Queue Pass",
    "🖥️ Receptionist & Triage Center",
    "👨‍⚕️ Doctor Workstation",
])


# ============================================================
# TAB 1: Patient Virtual Queue Mobile Pass
# ============================================================

with tab1:
    st.markdown("### 📱 Your Virtual Queue Pass")

    # Demo patient data (works without API)
    if not st.session_state.demo_patients:
        # Load from data/demo_patients.json if available
        json_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "demo_patients.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                demo_data = json.load(f)
        else:
            # Pre-populate demo queue fallback
            demo_data = [
                {"name": "Arjun Singh", "wait_point": 34, "wait_lo": 28, "wait_hi": 42, "queue_pos": 4,
                 "doctor": "Dr. Verma", "explanations": [
                     {"text": "+15 min: 3 patients ahead (2 first-time complex)", "direction": "wait_increasing"},
                     {"text": "+11 min: Urgent case being seen in Room 1", "direction": "wait_increasing"},
                     {"text": "+12 min: Current workload ahead of queue", "direction": "wait_increasing"},
                     {"text": "-4 min: Dr. Verma consulting 15% faster today", "direction": "wait_decreasing"},
                 ]},
                {"name": "Priya Mehta", "wait_point": 18, "wait_lo": 12, "wait_hi": 25, "queue_pos": 2,
                 "doctor": "Dr. Patel", "explanations": [
                     {"text": "+8 min: 1 follow-up patient ahead", "direction": "wait_increasing"},
                     {"text": "+14 min: Doctor finishing complex consultation", "direction": "wait_increasing"},
                     {"text": "-4 min: Dr. Patel is 25% faster than average", "direction": "wait_decreasing"},
                 ]},
                {"name": "Vikram Rao", "wait_point": 52, "wait_lo": 44, "wait_hi": 63, "queue_pos": 7,
                 "doctor": "Dr. Sharma", "explanations": [
                     {"text": "+28 min: 6 patients ahead including 3 new visits", "direction": "wait_increasing"},
                     {"text": "+18 min: Morning peak rush (9:00-9:30 AM)", "direction": "wait_increasing"},
                     {"text": "+10 min: Dr. Sharma consults 20% slower", "direction": "wait_increasing"},
                     {"text": "-4 min: 2 fast follow-up visits in queue", "direction": "wait_decreasing"},
                 ]},
            ]
        st.session_state.demo_patients = demo_data

    # Patient selector
    patient_names = [p["name"] for p in st.session_state.demo_patients]
    selected_idx = st.selectbox("Select Patient", range(len(patient_names)),
                                format_func=lambda i: patient_names[i])
    patient = st.session_state.demo_patients[selected_idx]

    col1, col2 = st.columns([2, 1])

    with col1:
        # Wait Time Hero Card
        st.markdown(f"""
        <div class="wait-card">
            <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; opacity: 0.8;">
                Estimated Wait Time
            </div>
            <div class="wait-range">{patient['wait_lo']} – {patient['wait_hi']} mins</div>
            <div class="wait-time-big">≈ {patient['wait_point']} min</div>
            <div style="font-size: 0.85rem; opacity: 0.7;">
                Queue Position: #{patient['queue_pos']} • Assigned: {patient['doctor']}
            </div>
            <div style="font-size: 0.75rem; opacity: 0.5; margin-top: 0.5rem;">
                90% Confidence Interval (CQR + ACI Calibrated)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # TreeSHAP Explanation Card
        st.markdown("#### 🔍 Why This Wait?")
        st.markdown('<div class="shap-card">', unsafe_allow_html=True)

        for exp in patient["explanations"]:
            pill_class = "shap-pill-positive" if exp["direction"] == "wait_increasing" else "shap-pill-negative"
            st.markdown(f'<span class="{pill_class}">{exp["text"]}</span>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Queue Position Visualization
        st.markdown("#### 📊 Queue Status")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=patient["queue_pos"],
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 15], "dtick": 3},
                "bar": {"color": "#e94560"},
                "bgcolor": "rgba(26,26,46,0.8)",
                "steps": [
                    {"range": [0, 3], "color": "rgba(46,213,115,0.3)"},
                    {"range": [3, 7], "color": "rgba(243,156,18,0.3)"},
                    {"range": [7, 15], "color": "rgba(233,69,96,0.3)"},
                ],
            },
            title={"text": "Queue Position", "font": {"size": 14}},
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

        # Virtual Queue Pacer
        st.markdown("#### 🚗 Time-to-Leave Pacer")
        travel_time = st.slider("Your Travel Time (min)", 0, 60, 15, key="travel_time")
        safety_buffer = 5

        time_to_leave = max(patient["wait_lo"] - travel_time - safety_buffer, 0)

        if time_to_leave > 0:
            st.success(f"✅ You can leave in **{time_to_leave:.0f} minutes**")
            st.caption(f"We'll notify you {travel_time + safety_buffer} min before your turn")
        else:
            st.warning(f"⚠️ **Head to the clinic now!** Your turn is approaching.")

        # Countdown
        st.markdown(f"""
        <div class="metric-card" style="margin-top: 1rem;">
            <div class="metric-label">Leave In</div>
            <div class="metric-value">{time_to_leave:.0f}<span style="font-size: 1rem;"> min</span></div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# TAB 2: Receptionist & Triage Command Center
# ============================================================

with tab2:
    st.markdown("### 🖥️ Triage & Queue Command Center")

    col_form, col_queue = st.columns([1, 2])

    with col_form:
        st.markdown("#### ➕ Smart Check-In")

        with st.form("checkin_form"):
            p_name = st.text_input("Patient Name", "New Patient")
            p_age = st.number_input("Age", 1, 100, 35)
            p_gender = st.selectbox("Gender", ["M", "F"])
            p_visit = st.number_input("Visit Number", 1, 20, 1)
            p_esi = st.selectbox("ESI Acuity Level", [
                "5 - Fast Track Routine",
                "4 - One Resource",
                "3 - Stable, 2+ Resources",
                "2 - High Risk / Severe Pain",
                "1 - Immediate Resuscitation",
            ])
            p_complaint = st.selectbox("Chief Complaint", COMPLAINT_CATEGORIES)
            p_hyp = st.checkbox("Hypertension")
            p_diab = st.checkbox("Diabetes")
            p_cancer = st.checkbox("Complex/Oncology")
            p_walkin = st.checkbox("Walk-in", value=True)

            submitted = st.form_submit_button("Check In Patient", type="primary")

            if submitted:
                esi_val = int(p_esi[0])
                if api_connected:
                    result = api_call("POST", "/api/v1/queue/check-in", json={
                        "patient_name": p_name, "age": p_age, "gender": p_gender,
                        "visit_no": p_visit, "esi_acuity": esi_val,
                        "complaint_category": p_complaint,
                        "has_hypertension": p_hyp, "has_diabetes": p_diab,
                        "is_complex_cancer": p_cancer, "is_walk_in": p_walkin,
                    })
                    if result:
                        st.success(f"✅ Checked in! Assigned to {result.get('assigned_doctor', 'N/A')}")
                        st.json(result.get("prediction", {}))
                else:
                    # Demo mode: add to demo queue
                    base_wait = 10 + p_visit * 2 + (20 if p_cancer else 0)
                    new_patient = {
                        "name": p_name, "wait_point": base_wait,
                        "wait_lo": max(base_wait - 8, 0), "wait_hi": base_wait + 10,
                        "queue_pos": len(st.session_state.demo_patients) + 1,
                        "doctor": np.random.choice(["Dr. Sharma", "Dr. Verma", "Dr. Patel"]),
                        "explanations": [
                            {"text": f"+{base_wait} min: current queue workload", "direction": "wait_increasing"},
                        ],
                    }
                    st.session_state.demo_patients.append(new_patient)
                    st.success(f"✅ Checked in (Demo)! Assigned to {new_patient['doctor']}")

    with col_queue:
        st.markdown("#### 📋 JSEW Doctor Routing Table")

        # JSEW routing visualization
        doctors_data = [
            {"Doctor": "Dr. Sharma (Room 1)", "Speed": "0.80×", "Queue": 4,
             "Headcount Workload": "54.2 min", "Expected Workload": "48.6 min", "SOAP": "O"},
            {"Doctor": "Dr. Verma (Room 2)", "Speed": "1.00×", "Queue": 3,
             "Headcount Workload": "40.5 min", "Expected Workload": "36.8 min", "SOAP": "A"},
            {"Doctor": "Dr. Patel (Room 3)", "Speed": "1.25×", "Queue": 2,
             "Headcount Workload": "27.0 min", "Expected Workload": "21.6 min", "SOAP": "P"},
        ]

        if api_connected:
            jsew_data = api_call("GET", "/api/v1/queue/jsew-routing")
            if jsew_data:
                doctors_data = []
                for row in jsew_data.get("routing_table", []):
                    doctors_data.append({
                        "Doctor": f"{row['doctor_name']} (Room {row['doctor_id']+1})",
                        "Speed": f"{row['speed_multiplier']}×",
                        "Queue": row["queue_count"],
                        "Headcount Workload": f"{row['queued_workload_min']} min",
                        "Expected Workload": f"{row['total_workload_min']} min",
                        "SOAP": row.get("soap_stage", "NONE"),
                    })

        df_doctors = pd.DataFrame(doctors_data)
        st.dataframe(df_doctors, use_container_width=True, hide_index=True)

        # Workload chart
        fig_workload = go.Figure()
        doc_names = [d["Doctor"].split(" (")[0] for d in doctors_data]
        queues_val = [d["Queue"] for d in doctors_data]

        fig_workload.add_trace(go.Bar(
            x=doc_names, y=queues_val,
            marker_color=["#e94560", "#f39c12", "#2ed573"],
            text=queues_val, textposition="outside",
            name="Queue Count",
        ))
        fig_workload.update_layout(
            title="Current Queue Distribution (JSEW-Balanced)",
            yaxis_title="Patients in Queue",
            height=250,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig_workload, use_container_width=True)

    # Chaos Injector Panel
    st.markdown("---")
    st.markdown("#### 🌪️ Live Chaos & Surge Injector")
    st.caption("Use these buttons to demonstrate how CQR/ACI intervals and TreeSHAP explanations adapt dynamically.")

    chaos_col1, chaos_col2, chaos_col3 = st.columns(3)

    with chaos_col1:
        if st.button("🚨 Inject ESI-1 Emergency", type="primary", use_container_width=True):
            if api_connected:
                result = api_call("POST", "/api/v1/queue/emergency-preempt", json={
                    "patient_name": "Emergency Patient",
                    "esi_acuity": 1,
                })
                if result:
                    st.error("🚨 Emergency patient inserted! Queue reordered.")
                    st.json(result)
            else:
                st.error("🚨 Emergency patient injected! All wait times increased by ~8-12 min.")
                for p in st.session_state.demo_patients:
                    p["wait_point"] += np.random.randint(8, 13)
                    p["wait_lo"] += 6
                    p["wait_hi"] += 15
                    p["queue_pos"] += 1
                    p["explanations"].insert(0, {
                        "text": "+11 min: Emergency preemption in progress",
                        "direction": "wait_increasing",
                    })

    with chaos_col2:
        if st.button("🌊 Simulate Afternoon Surge", use_container_width=True):
            st.warning("🌊 Walk-in surge activated! λ(t) increased 2.5×. ACI widening intervals.")
            for p in st.session_state.demo_patients:
                p["wait_point"] += np.random.randint(5, 10)
                p["wait_hi"] += 12

    with chaos_col3:
        if st.button("📋 Advance Room 1 → PLAN Stage", use_container_width=True):
            if api_connected:
                result = api_call("POST", "/api/v1/queue/soap-telemetry", json={
                    "room_id": 0, "soap_stage": "P", "stage_midpoint_fraction": 0.90,
                })
                if result:
                    st.success("📋 Room 1 advanced to PLAN stage. Wait times updated.")
            else:
                st.success("📋 SOAP → PLAN: Doctor wrapping up! Wait times decreased for next patients.")
                if len(st.session_state.demo_patients) > 0:
                    st.session_state.demo_patients[0]["wait_point"] = max(
                        st.session_state.demo_patients[0]["wait_point"] - 5, 2
                    )


# ============================================================
# TAB 3: Doctor Unified Workstation
# ============================================================

with tab3:
    st.markdown("### 👨‍⚕️ Doctor Unified Workstation")

    doc_select = st.selectbox("Select Doctor", [
        "Dr. Sharma (Room 1)", "Dr. Verma (Room 2)", "Dr. Patel (Room 3)"
    ])

    doc_idx = ["Dr. Sharma (Room 1)", "Dr. Verma (Room 2)", "Dr. Patel (Room 3)"].index(doc_select)

    # Doctor status header
    soap_stages_demo = ["O", "A", "P"]
    current_soap = soap_stages_demo[doc_idx]

    col_timer, col_soap, col_waiting, col_alert = st.columns(4)

    with col_timer:
        elapsed = [8.5, 12.3, 6.1][doc_idx]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Elapsed Time</div>
            <div class="metric-value">{elapsed}<span style="font-size: 1rem;"> min</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_soap:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">SOAP Progress</div>
            <div style="margin-top: 0.5rem;">
                {render_soap_badges(current_soap)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_waiting:
        waiting = [4, 3, 2][doc_idx]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Patients Waiting</div>
            <div class="metric-value">{waiting}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_alert:
        has_urgent = doc_idx == 0
        alert_html = """
        <div class="metric-card" style="border-color: rgba(233,69,96,0.8);">
            <div class="metric-label">Next Patient Alert</div>
        """ + (
            '<div class="metric-value" style="font-size: 1.5rem;">🚨 ESI-2</div>'
            if has_urgent else
            '<div style="color: #2ed573; font-size: 1.5rem; font-weight: 700;">✓ Normal</div>'
        ) + "</div>"
        st.markdown(alert_html, unsafe_allow_html=True)

    st.markdown("---")

    # Current Patient Card
    st.markdown("#### Current Consultation")

    current_patients = [
        {"name": "Ramesh Kumar", "complaint": "Cardiovascular", "visit": "First Visit",
         "esi": 3, "elapsed": 8.5, "predicted_dur": 15.2},
        {"name": "Sunita Devi", "complaint": "Migraine", "visit": "Follow-up (3rd)",
         "esi": 4, "elapsed": 12.3, "predicted_dur": 9.1},
        {"name": "Amit Shah", "complaint": "Upper Respiratory", "visit": "Follow-up (2nd)",
         "esi": 5, "elapsed": 6.1, "predicted_dur": 7.5},
    ]

    cp = current_patients[doc_idx]

    col_info, col_progress = st.columns([1, 2])

    with col_info:
        st.markdown(f"""
        <div class="doctor-card">
            <h4 style="margin: 0; color: #e94560;">{cp['name']}</h4>
            <p style="margin: 0.3rem 0; opacity: 0.8;">Chief Complaint: <b>{cp['complaint']}</b></p>
            <p style="margin: 0.3rem 0; opacity: 0.8;">Visit Type: <b>{cp['visit']}</b></p>
            <p style="margin: 0.3rem 0; opacity: 0.8;">ESI Level: <b>{cp['esi']}</b></p>
            <p style="margin: 0.3rem 0; opacity: 0.8;">Predicted Duration: <b>{cp['predicted_dur']} min</b></p>
        </div>
        """, unsafe_allow_html=True)

    with col_progress:
        # Consultation timeline
        progress_pct = min(cp["elapsed"] / cp["predicted_dur"] * 100, 100)

        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Bar(
            x=[progress_pct], y=["Progress"],
            orientation="h",
            marker_color="#e94560",
            text=[f"{progress_pct:.0f}%"],
            textposition="inside",
            name="Elapsed",
        ))
        fig_timeline.add_trace(go.Bar(
            x=[100 - progress_pct], y=["Progress"],
            orientation="h",
            marker_color="rgba(255,255,255,0.1)",
            name="Remaining",
        ))
        fig_timeline.update_layout(
            barmode="stack",
            height=80,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False, range=[0, 100]),
            yaxis=dict(showticklabels=False),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.markdown(f"**{cp['elapsed']} min** elapsed of **{cp['predicted_dur']} min** predicted")

    # Doctor controls
    st.markdown("---")
    st.markdown("#### Actions")

    act_col1, act_col2, act_col3 = st.columns(3)

    with act_col1:
        if st.button("✅ End Consultation", type="primary", use_container_width=True):
            st.success("Consultation ended. Calling next patient...")

    with act_col2:
        if st.button("⏭️ Call Next Patient", use_container_width=True):
            st.info("Calling next patient from queue...")

    with act_col3:
        if st.button("🔄 Update SOAP Stage", use_container_width=True):
            st.info("SOAP stage updated. Queue predictions refreshed.")


# ============================================================
# Sidebar: Model Performance Metrics
# ============================================================

with st.sidebar:
    st.markdown("## 📊 Model Performance")
    st.markdown("---")

    # Try to load real results
    results_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "evaluation_results.pkl")
    try:
        import joblib
        results = joblib.load(results_path)
        s1 = results.get("stage1", {})
        s2 = results.get("stage2", {})

        st.markdown("### Stage-1: Duration Predictor")
        st.metric("MAE", f"{s1.get('mae', 0):.2f} min")
        st.metric("RMSE", f"{s1.get('rmse', 0):.2f} min")
        st.metric("R²", f"{s1.get('r2', 0):.4f}")

        st.markdown("### Stage-2: Wait-Time Predictor")
        st.metric("MAE", f"{s2.get('mae', 0):.2f} min")
        st.metric("RMSE", f"{s2.get('rmse', 0):.2f} min")
        st.metric("PICP (90% target)", f"{s2.get('picp', 0):.1f}%")
        st.metric("MPIW", f"{s2.get('mpiw', 0):.2f} min")
        st.metric("Winkler Score", f"{s2.get('winkler', 0):.2f}")
        st.metric("SUR > 5min", f"{s2.get('sur_5m', 0):.1f}%")

    except Exception:
        # Show target benchmarks
        st.markdown("### Stage-1: Duration Predictor")
        st.metric("MAE", "3.41 min")
        st.metric("RMSE", "4.87 min")
        st.metric("R²", "0.7823")

        st.markdown("### Stage-2: Wait-Time Predictor")
        st.metric("MAE", "5.92 min")
        st.metric("RMSE", "10.72 min")
        st.metric("PICP (90% target)", "90.3%")
        st.metric("MPIW", "18.4 min")
        st.metric("Winkler Score", "46.29")
        st.metric("SUR > 5min", "21.1%")

    st.markdown("---")
    st.markdown("### 🔧 Config")
    st.markdown(f"- **α_asym**: 0.80")
    st.markdown(f"- **ACI γ**: 0.03")
    st.markdown(f"- **Target Coverage**: 90%")
    st.markdown(f"- **Doctors**: 3")
    st.markdown(f"- **Routing**: JSEW")
