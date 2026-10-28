"""
CARE'OS Module 2 — Web Application Server
==========================================

Interactive Clinician Dashboard featuring:
  - Linked Evidence Audio-Provenance Player
  - 5-Tier Verification Badge Visualizer (Solid Green, Amber Inferred, Red Contradiction)
  - Real-Time Clinical Co-Pilot Whisperer & Red-Flag Questions
  - CPIC Level A Pharmacogenomic (PGx) Triggers to Module 3
  - Module 1 Queue Pacing Telemetry
  - HL7 FHIR R4 & Bilingual Patient Summary Exporter
"""

import json
import os
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from module2_careos.main import CareOSModule2Pipeline

app = FastAPI(
    title="CARE'OS Module 2: Ambient AI Scribe with 2nd-Pass Verifier",
    description="Publication-grade clinical documentation pipeline with Dual-Tier inference-aware verification",
    version="2.0.0",
)

# Setup directories
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
DATA_DIR = BASE_DIR.parent.parent / "data"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Pipeline instance
pipeline = CareOSModule2Pipeline()


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health_check():
    return {"status": "online", "module": "CARE'OS Module 2: Ambient AI Scribe"}


@app.post("/api/process-encounter")
async def process_encounter(
    audio: UploadFile = File(None),
    patient_id: str = Form("PT-9042"),
    language: str = Form("en"),
    demo_mode: bool = Form(True),
):
    try:
        sample_audio = DATA_DIR / "sample_audio" / "demo.wav"
        if audio is not None:
            upload_path = DATA_DIR / "uploads" / audio.filename
            upload_path.parent.mkdir(parents=True, exist_ok=True)
            with open(upload_path, "wb") as f:
                f.write(await audio.read())
            audio_path = str(upload_path)
        else:
            sample_audio.parent.mkdir(parents=True, exist_ok=True)
            if not sample_audio.exists():
                from module2_careos.main import main
                # Generate sample audio if not exists
                import struct
                sr = 16000
                duration = 52
                samples = sr * duration
                with open(sample_audio, "wb") as f:
                    f.write(b"RIFF")
                    data_size = samples * 2
                    f.write(struct.pack("<I", 36 + data_size))
                    f.write(b"WAVE")
                    f.write(b"fmt ")
                    f.write(struct.pack("<I", 16))
                    f.write(struct.pack("<H", 1))
                    f.write(struct.pack("<H", 1))
                    f.write(struct.pack("<I", sr))
                    f.write(struct.pack("<I", sr * 2))
                    f.write(struct.pack("<H", 2))
                    f.write(struct.pack("<H", 16))
                    f.write(b"data")
                    f.write(struct.pack("<I", data_size))
                    for _ in range(samples):
                        f.write(struct.pack("<h", 0))
            audio_path = str(sample_audio)

        # Run pipeline
        results = pipeline.process_encounter(
            audio_path=audio_path,
            patient_id=patient_id,
            language=language,
        )
        return JSONResponse(content=results)
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )


@app.get("/api/demo-encounter")
async def get_demo_encounter():
    """Returns pre-computed clinical encounter for instantaneous UI demo."""
    results_path = DATA_DIR / "output" / "pipeline_results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            return JSONResponse(content=json.load(f))
    
    # Fallback: run on demo audio
    sample_audio = DATA_DIR / "sample_audio" / "demo.wav"
    results = pipeline.process_encounter(
        audio_path=str(sample_audio),
        patient_id="PT-CARDIOLOGY-104",
        language="en",
    )
    return JSONResponse(content=results)


def start_server(host="0.0.0.0", port=8000):
    uvicorn.run("module2_careos.web_ui.app:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    start_server()
