#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CARE'OS — Integrated Pipeline Runner (Module 2 + Module 3)                ║
║  Audio → Scribe → Verifier → Genomics Engine → SOAP + PGx Alerts          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run the complete CARE'OS clinical pipeline:
  1. Module 2: Audio → Diarized Transcript → ASR Correction → Clinical NLP →
               SOAP Note → 2nd-Pass Verification → FHIR Export
  2. Module 3: CPIC Trigger → SNP Autoencoder → Multi-Modal Fusion →
               Drug Response Classifier → Dose Regressor → PGx Alert

Usage:
    python run_integrated_careos.py                      # Demo mode
    python run_integrated_careos.py --audio path/to/wav  # Real audio
    python run_integrated_careos.py --dashboard          # Launch UI
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Windows console encoding fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ═══════════════════════════════════════════════════════════════════════════
# Catppuccin Terminal Colors
# ═══════════════════════════════════════════════════════════════════════════

class C:
    """Catppuccin Mocha ANSI escape codes for terminal output."""
    RESET    = "\033[0m"
    BOLD     = "\033[1m"
    DIM      = "\033[2m"
    # Catppuccin Mocha approximations in 256-color
    ROSEWATER = "\033[38;2;245;224;220m"
    FLAMINGO  = "\033[38;2;242;205;205m"
    PINK      = "\033[38;2;245;194;231m"
    MAUVE     = "\033[38;2;203;166;247m"
    RED       = "\033[38;2;243;139;168m"
    MAROON    = "\033[38;2;235;160;172m"
    PEACH     = "\033[38;2;250;179;135m"
    YELLOW    = "\033[38;2;249;226;175m"
    GREEN     = "\033[38;2;166;227;161m"
    TEAL      = "\033[38;2;148;226;213m"
    SKY       = "\033[38;2;137;220;235m"
    SAPPHIRE  = "\033[38;2;116;199;236m"
    BLUE      = "\033[38;2;137;180;250m"
    LAVENDER  = "\033[38;2;180;190;254m"
    TEXT      = "\033[38;2;205;214;244m"
    SUBTEXT   = "\033[38;2;166;173;200m"
    OVERLAY   = "\033[38;2;147;153;178m"
    SURFACE   = "\033[38;2;69;71;90m"
    BASE_BG   = "\033[48;2;30;30;46m"


def banner():
    """Print the CARE'OS integrated pipeline banner."""
    print(f"""
{C.BASE_BG}{C.MAUVE}{C.BOLD}
 ██████╗ █████╗ ██████╗ ███████╗ ██╗ ██████╗ ███████╗
██╔════╝██╔══██╗██╔══██╗██╔════╝██╔╝██╔═══██╗██╔════╝
██║     ███████║██████╔╝█████╗ ██╔╝ ██║   ██║███████╗
██║     ██╔══██║██╔══██╗██╔══╝██╔╝  ██║   ██║╚════██║
╚██████╗██║  ██║██║  ██║███████╔╝   ╚██████╔╝███████║
 ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═════╝ ╚══════╝{C.RESET}

{C.TEAL}{C.BOLD}  Integrated Clinical Decision Support System{C.RESET}
{C.SUBTEXT}  Module 2: AI Scribe & 2nd-Pass Verifier{C.RESET}
{C.SUBTEXT}  Module 3: Genomics-Informed Treatment Recommendation{C.RESET}
{C.SURFACE}{'=' * 68}{C.RESET}
""")


def print_section(title: str, color: str = C.BLUE):
    """Print a section header."""
    print(f"\n{color}{C.BOLD}{'─' * 3} {title} {'─' * (60 - len(title))}{C.RESET}")


def print_kv(key: str, value, color: str = C.TEXT):
    """Print a key-value pair."""
    print(f"  {C.SUBTEXT}{key:.<35}{C.RESET} {color}{value}{C.RESET}")


def print_alert(level: str, message: str):
    """Print a colored alert based on level."""
    colors = {
        "critical": C.RED,
        "high": C.PEACH,
        "moderate": C.YELLOW,
        "low": C.GREEN,
    }
    icon = {"critical": "🚨", "high": "⚠️", "moderate": "💊", "low": "ℹ️"}.get(level, "📋")
    c = colors.get(level, C.TEXT)
    print(f"  {icon} {c}{C.BOLD}[{level.upper()}]{C.RESET} {c}{message}{C.RESET}")


def run_pipeline(audio_path: str, patient_id: str, language: str, output_path: str = None):
    """Run the complete integrated CARE'OS pipeline."""
    from module2_careos.main import CareOSModule2Pipeline

    banner()

    # ─── Initialize ──────────────────────────────────────────────────────
    print_section("INITIALIZING PIPELINE", C.MAUVE)
    t0 = time.time()

    pipeline = CareOSModule2Pipeline()
    print_kv("Pipeline stages", "5 (Audio → Correction → NLP → Verify → Export)")
    print_kv("Module 3 engine", "Attention-VAE + XGBoost + Conformal Regression")
    print_kv("Patient ID", patient_id)
    print_kv("Language", language)
    print_kv("Audio file", audio_path)

    init_time = time.time() - t0
    print_kv("Init time", f"{init_time:.2f}s", C.GREEN)

    # ─── Run Pipeline ────────────────────────────────────────────────────
    print_section("RUNNING PIPELINE", C.SAPPHIRE)
    t1 = time.time()

    results = pipeline.process_encounter(
        audio_path=audio_path,
        patient_id=patient_id,
        language=language,
    )

    pipeline_time = time.time() - t1

    # ─── Transcript Summary ──────────────────────────────────────────────
    print_section("STAGE 1: TRANSCRIPT", C.TEAL)
    transcript = results.get("transcript", {})
    print_kv("Duration", f"{transcript.get('duration_seconds', 0):.0f}s")
    print_kv("Speakers", transcript.get("num_speakers", 0))
    print_kv("Utterances", len(transcript.get("utterances", [])))

    # ─── Corrections ─────────────────────────────────────────────────────
    print_section("STAGE 2: ASR CORRECTION", C.PEACH)
    corrections = results.get("corrections", [])
    print_kv("Drug corrections", len(corrections))
    for corr in corrections[:5]:
        print(f"    {C.DIM}{corr.get('original_text', '')} -> "
              f"{C.PEACH}{corr.get('corrected_text', '')}{C.RESET}")

    # ─── SOAP Note ───────────────────────────────────────────────────────
    print_section("STAGE 3: SOAP NOTE", C.GREEN)
    soap = results.get("soap_note", {})
    print_kv("Subjective claims", len(soap.get("subjective", [])))
    print_kv("Objective (exam)", len(soap.get("objective_exam", [])))
    print_kv("Objective (results)", len(soap.get("objective_results", [])))
    print_kv("Assessment + Plan", len(soap.get("assessment_plan", [])))

    # ─── Verification ────────────────────────────────────────────────────
    print_section("STAGE 4: 2ND-PASS VERIFICATION", C.LAVENDER)
    verif = results.get("verification", {})
    print_kv("Total claims verified", verif.get("total_claims", 0))
    print_kv("Tier 1 (Verbatim)", verif.get("tier1_count", 0), C.GREEN)
    print_kv("Tier 2a (Terminology)", verif.get("tier2a_count", 0), C.TEAL)
    print_kv("Tier 2b (Ontology)", verif.get("tier2b_count", 0), C.SAPPHIRE)
    print_kv("Tier 3 (Inference)", verif.get("tier3_count", 0), C.YELLOW)
    print_kv("Tier 4 (Unsupported)", verif.get("tier4_count", 0), C.PEACH)
    print_kv("Tier 5 (Contradiction)", verif.get("tier5_count", 0), C.RED)
    print_kv("Literal hallucination rate",
             f"{verif.get('literal_hallucination_rate', 0):.1f}%", C.YELLOW)
    print_kv("Inference-aware rate",
             f"{verif.get('inference_aware_hallucination_rate', 0):.1f}%", C.GREEN)

    # ─── CPIC / Module 3 PGx Integration ─────────────────────────────────
    print_section("MODULE 3: PHARMACOGENOMICS ENGINE", C.MAUVE)
    triggers = results.get("cpic_triggers", [])
    pgx_alerts = results.get("pharmacogenomic_alerts", [])
    print_kv("CPIC triggers fired", len(triggers))
    print_kv("Module 3 active alerts", len(pgx_alerts))

    for t in triggers:
        alert = t.get("alert", {})
        is_m3 = t.get("module3_active", False)
        drug = alert.get("drug_name", "?")
        gene = alert.get("gene", "?")
        risk = alert.get("risk_level", "?")
        action = alert.get("clinical_action", "")

        print()
        print(f"  {C.MAUVE}{C.BOLD}{'=' * 60}{C.RESET}")
        print(f"  {C.PINK}{C.BOLD}Drug:{C.RESET} {C.TEXT}{drug}{C.RESET}")
        print(f"  {C.PINK}{C.BOLD}Gene:{C.RESET} {C.TEXT}{gene}{C.RESET}")
        print_alert(risk, action[:100])
        print(f"  {C.SUBTEXT}Module 3 ML Engine: {'ACTIVE' if is_m3 else 'STUB'}{C.RESET}")

        if is_m3 and t.get("pgx_alert"):
            pgx = t["pgx_alert"]
            print(f"  {C.TEAL}Predicted Response: {pgx.get('predicted_response', 'N/A')}{C.RESET}")
            print(f"  {C.TEAL}Suggested Dose:     {pgx.get('suggested_starting_dose', 'N/A')}{C.RESET}")
            print(f"  {C.TEAL}Metabolizer Type:   {pgx.get('metabolizer_phenotype', 'N/A')}{C.RESET}")
            print(f"  {C.TEAL}Confidence:         {pgx.get('prediction_confidence', 'N/A')}{C.RESET}")

    # ─── PGx SOAP Appendix ───────────────────────────────────────────────
    pgx_appendix = results.get("pharmacogenomic_appendix")
    if pgx_appendix:
        print_section("PGx SOAP NOTE APPENDIX", C.PINK)
        # Print first 500 chars
        for line in pgx_appendix[:500].split("\n"):
            print(f"  {C.FLAMINGO}{line}{C.RESET}")
        if len(pgx_appendix) > 500:
            print(f"  {C.DIM}... ({len(pgx_appendix)} chars total){C.RESET}")

    # ─── Exports ─────────────────────────────────────────────────────────
    print_section("STAGE 5: EXPORTS", C.SKY)
    fhir = results.get("fhir_bundle", {})
    fhir_resources = len(fhir.get("entry", []))
    detected_issues = sum(
        1 for e in fhir.get("entry", [])
        if e.get("resource", {}).get("resourceType") == "DetectedIssue"
    )
    print_kv("FHIR R4 resources", fhir_resources)
    print_kv("DetectedIssue (PGx)", detected_issues, C.MAUVE)
    print_kv("E/M coding level", results.get("em_coding", {}).get("suggested_em_level", "N/A"))

    # ─── Co-Pilot ────────────────────────────────────────────────────────
    print_section("CLINICAL CO-PILOT", C.FLAMINGO)
    copilot = results.get("copilot", {})
    hypotheses = copilot.get("hypotheses", [])
    red_flags = copilot.get("red_flag_questions", [])
    print_kv("Diagnostic hypotheses", len(hypotheses))
    for h in hypotheses[:3]:
        print(f"    {C.FLAMINGO}-> {h.get('hypothesis', '')} "
              f"({h.get('confidence', 0):.0%}){C.RESET}")
    print_kv("Red-flag questions", len(red_flags))
    for q in red_flags[:3]:
        print(f"    {C.RED}[{q.get('priority', '?')}] {q.get('question', '')}{C.RESET}")

    # ─── Summary ─────────────────────────────────────────────────────────
    print(f"\n{C.MAUVE}{C.BOLD}{'=' * 68}{C.RESET}")
    print(f"{C.GREEN}{C.BOLD}  PIPELINE COMPLETE{C.RESET}")
    print(f"{C.MAUVE}{C.BOLD}{'=' * 68}{C.RESET}")
    total = time.time() - t0
    print_kv("Total time", f"{total:.2f}s", C.GREEN)
    print_kv("Pipeline time", f"{pipeline_time:.2f}s", C.GREEN)
    print_kv("Encounter ID", results.get("encounter_id", "N/A"))

    # ─── Save Results ────────────────────────────────────────────────────
    if output_path is None:
        output_path = str(
            Path(__file__).resolve().parent / "data" / "output" / "integrated_results.json"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def json_serial(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=json_serial, ensure_ascii=False)

    print_kv("Results saved to", output_path, C.SAPPHIRE)
    print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="CARE'OS Integrated Pipeline: Module 2 (AI Scribe) + Module 3 (Genomics)"
    )
    parser.add_argument("--audio", "-a", default=None, help="Path to clinical audio file")
    parser.add_argument("--patient-id", "-p", default="CAREOS-DEMO-001", help="Patient ID")
    parser.add_argument("--language", "-l", default="en", choices=["en", "hi", "hinglish"])
    parser.add_argument("--output", "-o", default=None, help="Output JSON path")
    parser.add_argument("--dashboard", action="store_true", help="Launch integrated Streamlit dashboard")

    args = parser.parse_args()

    if args.dashboard:
        import subprocess
        dashboard_path = Path(__file__).resolve().parent / "integrated_dashboard.py"
        print(f"Launching integrated dashboard: {dashboard_path}")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
        return

    audio = args.audio
    if audio is None:
        audio = str(Path(__file__).resolve().parent / "data" / "sample_audio" / "demo.wav")
        if not Path(audio).exists():
            print(f"{C.YELLOW}Demo audio not found, creating...{C.RESET}")
            import struct
            Path(audio).parent.mkdir(parents=True, exist_ok=True)
            sr, duration = 16000, 52
            samples = sr * duration
            with open(audio, "wb") as f:
                f.write(b"RIFF")
                data_size = samples * 2
                f.write(struct.pack("<I", 36 + data_size))
                f.write(b"WAVE")
                f.write(b"fmt ")
                f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
                f.write(b"data")
                f.write(struct.pack("<I", data_size))
                f.write(b"\x00" * data_size)

    run_pipeline(audio, args.patient_id, args.language, args.output)


if __name__ == "__main__":
    main()
