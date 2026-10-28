"""
CARE'OS Module 2: Ambient AI Scribe with Dual-Tier 2nd-Pass Clinical Verifier
================================================================================

A five-stage, ontology-grounded clinical pipeline:
  Stage 1: Audio Ingestion, WhisperX ASR, Speaker Diarization
  Stage 2: Medical ASR Post-Correction & RxNorm Phonetic Grounding  
  Stage 3: Clinical NLP Pre-Extraction (Med7, scispaCy, negspaCy) + Constrained SOAP Generation
  Stage 4: Dual-Tier 2nd-Pass Verifier (Fast-Path NLI + Slow-Path Ontology-RAG Judge)
  Stage 5: FHIR R4 Export, E/M Coding, Patient-Friendly AVS

Novel Features:
  1. Hinglish Code-Switched Clinical ASR
  2. Linked Evidence Audio-Provenance UI
  3. Real-Time Clinical Co-Pilot Whisperer
  4. Bidirectional CARE'OS Inter-Module Bus (Module 1 ↔ 2 ↔ 3)
  5. Multi-Audience Post-Visit Artifact Generator
"""

__version__ = "1.0.0"
__project__ = "CARE'OS Module 2"
