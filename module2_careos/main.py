"""
CARE'OS Module 2 — Main Pipeline Orchestrator
================================================

Orchestrates the complete 5-stage clinical pipeline:
  Stage 1: Audio → Diarized Transcript (WhisperX + pyannote)
  Stage 2: Medical ASR Post-Correction (RxNorm grounding)
  Stage 3: Clinical NLP → Constrained SOAP Note (Med7 + Pydantic)
  Stage 4: Dual-Tier 2nd-Pass Verification (NLI + Ontology-RAG)
  Stage 5: FHIR R4 Export + Patient Summary

Plus novel features:
  - Clinical Co-Pilot Whisperer
  - CPIC Level A Inter-Module Triggers
  - Module 1 Telemetry
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from .stage1_audio.audio_pipeline import Stage1Pipeline
from .stage2_correction.asr_corrector import Stage2Pipeline
from .stage3_clinical_nlp.clinical_extractor import Stage3Pipeline
from .stage4_verifier.verifier_engine import Stage4Pipeline
from .stage5_export.export_pipeline import Stage5Pipeline
from .copilot.copilot_engine import ClinicalCoPilot
from .inter_module_bus.bus import InterModuleBus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CARE'OS.Module2")


class CareOSModule2Pipeline:
    """
    Complete CARE'OS Module 2 Pipeline Orchestrator.
    
    Processes clinical audio through 5 stages to produce:
      - Verified, speaker-attributed SOAP note
      - HL7 FHIR R4 interoperability bundle
      - Patient-friendly after-visit summary
      - CPIC pharmacogenomic alerts
    """
    
    def __init__(self, config_path: str = None):
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        # Initialize all stages
        self.stage1 = Stage1Pipeline(self.config.get("stage1_audio", {}))
        self.stage2 = Stage2Pipeline(self.config.get("stage2_correction", {}))
        self.stage3 = Stage3Pipeline(self.config.get("stage3_clinical_nlp", {}))
        self.stage4 = Stage4Pipeline(self.config.get("stage4_verifier", {}))
        self.stage5 = Stage5Pipeline(self.config.get("stage5_export", {}))
        
        # Novel features
        self.copilot = ClinicalCoPilot(self.config.get("copilot", {}))
        self.bus = InterModuleBus(self.config.get("inter_module_bus", {}))
        
        # Results storage
        self.results = {}
    
    def load_models(self):
        """Load all ML models across all stages."""
        logger.info("=" * 80)
        logger.info("CARE'OS Module 2 — Loading Models")
        logger.info("=" * 80)
        
        self.stage1.load_models()
        self.stage3.load_models()
        self.stage4.load_models()
        
        logger.info("All models loaded successfully")
    
    def process_encounter(
        self,
        audio_path: str,
        patient_id: str = "anonymous",
        language: str = "en",
    ) -> dict:
        """
        Run the complete 5-stage pipeline on a clinical encounter.
        
        Args:
            audio_path: Path to the audio file (WAV, MP3, etc.)
            patient_id: Patient identifier
            language: "en" for English, "hi" for Hindi, "hinglish" for code-switched
            
        Returns:
            Dictionary containing all pipeline outputs
        """
        start_time = time.time()
        
        logger.info("=" * 80)
        logger.info(f"CARE'OS Module 2 — Processing Encounter")
        logger.info(f"Audio: {audio_path}")
        logger.info(f"Patient: {patient_id} | Language: {language}")
        logger.info("=" * 80)
        
        # ─── Stage 1: Audio → Diarized Transcript ────────────────────
        transcript = self.stage1.process(audio_path)
        transcript.utterances[0].speaker_role  # Ensure roles assigned
        
        # ─── Co-Pilot: Real-time monitoring ──────────────────────────
        copilot_update = self.copilot.update(
            transcript.utterances,
            elapsed_seconds=transcript.duration_seconds,
        )
        
        # ─── Stage 2: Medical ASR Post-Correction ────────────────────
        corrected = self.stage2.process(transcript)
        
        # ─── Stage 3: Clinical NLP + SOAP Generation ─────────────────
        fact_sheet, soap_note = self.stage3.process(corrected, transcript)
        fact_sheet.patient_id = patient_id
        soap_note.patient_id = patient_id
        
        # ─── Inter-Module Bus: CPIC Triggers ─────────────────────────
        cpic_triggers = self.bus.check_cpic_triggers(fact_sheet)
        
        # ─── Inter-Module Bus: Module 1 Telemetry ────────────────────
        telemetry = self.bus.emit_telemetry(
            encounter_id=fact_sheet.encounter_id,
            patient_id=patient_id,
            elapsed_sec=transcript.duration_seconds,
            soap_phase="assessment_plan",
            concept_count=len(fact_sheet.symptoms) + len(fact_sheet.medications),
        )
        
        # ─── Stage 4: Dual-Tier 2nd-Pass Verification ────────────────
        verification = self.stage4.verify(soap_note, fact_sheet, transcript)
        
        # ─── Stage 5: Export Artifacts ────────────────────────────────
        exports = self.stage5.process(soap_note, fact_sheet, verification)
        
        elapsed = time.time() - start_time
        
        # ─── Compile Results ──────────────────────────────────────────
        self.results = {
            "encounter_id": fact_sheet.encounter_id,
            "patient_id": patient_id,
            "processing_time_sec": round(elapsed, 2),
            
            # Stage 1
            "transcript": transcript.model_dump(),
            
            # Stage 2
            "corrections": [c.model_dump() for c in corrected.corrections],
            
            # Stage 3
            "fact_sheet": fact_sheet.model_dump(),
            "soap_note": soap_note.model_dump(),
            
            # Stage 4
            "verification": verification.model_dump(),
            
            # Stage 5
            "fhir_bundle": exports["fhir_bundle"],
            "em_coding": exports["em_coding"],
            "patient_summary_en": exports["patient_summary_en"].model_dump(),
            "patient_summary_hi": exports["patient_summary_hi"].model_dump(),
            
            # Novel Features
            "copilot": copilot_update.model_dump(),
            "cpic_triggers": cpic_triggers,
            "telemetry": telemetry,
        }
        
        logger.info("=" * 80)
        logger.info(f"✅ Pipeline complete in {elapsed:.2f}s")
        logger.info(f"   SOAP Claims: {verification.total_claims}")
        logger.info(f"   Verification: T1={verification.tier1_count} T2a={verification.tier2a_count} "
                     f"T2b={verification.tier2b_count} T3={verification.tier3_count} "
                     f"T4={verification.tier4_count} T5={verification.tier5_count}")
        logger.info(f"   Literal Hallucination Rate: {verification.literal_hallucination_rate:.1f}%")
        logger.info(f"   Inference-Aware Rate: {verification.inference_aware_hallucination_rate:.1f}%")
        logger.info(f"   CPIC Triggers: {len(cpic_triggers)}")
        logger.info(f"   MEDEC Errors: {len(verification.medec_results)}")
        logger.info("=" * 80)
        
        return self.results
    
    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            "stage1_audio": {
                "target_sample_rate": 16000,
                "target_channels": 1,
                "dsp": {
                    "highpass_hz": 80,
                    "lowpass_hz": 8000,
                },
                "whisperx": {
                    "model_size": "large-v3",
                    "compute_type": "float16",
                    "device": "cpu",
                    "batch_size": 16,
                    "language": "en",
                    "condition_on_prev_text": False,
                    "hinglish_mode": False,
                },
                "diarization": {
                    "model": "pyannote/speaker-diarization-community-1",
                    "min_speakers": 2,
                    "max_speakers": 4,
                    "sri": {"context_window_size": 5},
                },
            },
            "stage2_correction": {
                "drug_correction": {
                    "misspelling_threshold": 0.75,
                    "phonetic_similarity_threshold": 0.80,
                },
            },
            "stage3_clinical_nlp": {
                "spacy": {},
                "soap_llm": {
                    "model_name": "google/medgemma-27b-text-it",
                    "temperature": 0.1,
                },
            },
            "stage4_verifier": {
                "fast_path": {
                    "minicheck_confidence_threshold": 0.85,
                },
                "slow_path": {},
                "medec": {},
            },
            "stage5_export": {},
            "copilot": {
                "top_n_hypotheses": 5,
                "red_flag_questions": 5,
            },
            "inter_module_bus": {},
        }


def main():
    """CLI entry point for CARE'OS Module 2."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CARE'OS Module 2: Ambient AI Scribe with 2nd-Pass Verifier"
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=None,
        help="Path to clinical audio file"
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to pipeline config YAML"
    )
    parser.add_argument(
        "--patient-id", "-p",
        default="anonymous",
        help="Patient identifier"
    )
    parser.add_argument(
        "--language", "-l",
        default="en",
        choices=["en", "hi", "hinglish"],
        help="Audio language"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON path"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with simulated audio"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CareOSModule2Pipeline(args.config)
    
    if args.demo or not args.audio_path:
        logger.info("Running demo mode with simulated clinical encounter...")
        
        # Create a dummy audio file for demo
        demo_audio = Path(__file__).parent.parent / "data" / "sample_audio" / "demo.wav"
        demo_audio.parent.mkdir(parents=True, exist_ok=True)
        
        if not demo_audio.exists():
            import struct
            # Generate a minimal valid WAV file
            sr = 16000
            duration = 52  # seconds
            samples = sr * duration
            with open(demo_audio, 'wb') as f:
                # WAV header
                f.write(b'RIFF')
                data_size = samples * 2
                f.write(struct.pack('<I', 36 + data_size))
                f.write(b'WAVE')
                f.write(b'fmt ')
                f.write(struct.pack('<I', 16))  # chunk size
                f.write(struct.pack('<H', 1))   # PCM
                f.write(struct.pack('<H', 1))   # mono
                f.write(struct.pack('<I', sr))   # sample rate
                f.write(struct.pack('<I', sr * 2))  # byte rate
                f.write(struct.pack('<H', 2))   # block align
                f.write(struct.pack('<H', 16))  # bits per sample
                f.write(b'data')
                f.write(struct.pack('<I', data_size))
                # Write silence
                for _ in range(samples):
                    f.write(struct.pack('<h', 0))
            
            logger.info(f"Created demo audio: {demo_audio}")
        
        audio_path = str(demo_audio)
    else:
        audio_path = args.audio_path
    
    # Run pipeline
    results = pipeline.process_encounter(
        audio_path=audio_path,
        patient_id=args.patient_id,
        language=args.language,
    )
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(__file__).parent.parent / "data" / "output" / "pipeline_results.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert datetime objects for JSON serialization
    def json_serial(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=json_serial)
    
    logger.info(f"Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("CARE'OS Module 2 — Pipeline Results Summary")
    print("=" * 80)
    print(f"Encounter ID:    {results['encounter_id']}")
    print(f"Processing Time: {results['processing_time_sec']}s")
    print(f"Total Claims:    {results['verification']['total_claims']}")
    print(f"")
    print(f"Verification Breakdown:")
    print(f"  Tier 1 (Verbatim):           {results['verification']['tier1_count']}")
    print(f"  Tier 2a (Terminology):        {results['verification']['tier2a_count']}")
    print(f"  Tier 2b (Ontology Synonym):   {results['verification']['tier2b_count']}")
    print(f"  Tier 3 (Clinical Inference):  {results['verification']['tier3_count']}")
    print(f"  Tier 4 (Unsupported):         {results['verification']['tier4_count']}")
    print(f"  Tier 5 (Contradiction):       {results['verification']['tier5_count']}")
    print(f"")
    print(f"Hallucination Rates:")
    print(f"  Literal (lexical):       {results['verification']['literal_hallucination_rate']:.1f}%")
    print(f"  Inference-Aware:         {results['verification']['inference_aware_hallucination_rate']:.1f}%")
    print(f"  Human Physician (ref):   10.4%")
    print(f"")
    print(f"CPIC Triggers: {len(results['cpic_triggers'])}")
    for t in results['cpic_triggers']:
        alert = t['alert']
        print(f"  [PGx] {alert['drug_name']} -> {alert['gene']} ({alert['risk_level']})")
    print(f"")
    print(f"Co-Pilot Hypotheses: {len(results['copilot']['hypotheses'])}")
    for h in results['copilot']['hypotheses']:
        print(f"  -> {h['hypothesis']} ({h['confidence']:.0%})")
    print(f"")
    print(f"Red-Flag Questions: {len(results['copilot']['red_flag_questions'])}")
    for q in results['copilot']['red_flag_questions']:
        print(f"  [?] [{q['priority']}] {q['question']}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
