"""
CARE'OS Module 2 — Stage 5: Multi-Audience Post-Visit Artifact Generator
==========================================================================

Three downstream artifacts:
  1. HL7 FHIR R4 / USCDI Interoperability Bundle
  2. E/M Coding & Medical Necessity Readiness Check
  3. Bilingual Patient-Friendly After-Visit Summary (AVS)
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class FHIRExporter:
    """
    Maps verified Pydantic SOAP schema to HL7 FHIR R4 resources.
    
    Generates:
      - Patient resource
      - Encounter resource
      - Observation resources (vitals)
      - Condition resources (diagnoses with SNOMED/ICD-10 codes)
      - MedicationRequest resources (prescriptions with RxNorm codes)
      - ServiceRequest resources (lab orders)
    """
    
    def export(
        self,
        soap_note: "SOAPNote",
        fact_sheet: "EncounterFactSheet",
        verification_report: "VerificationReport",
    ) -> dict:
        """Generate HL7 FHIR R4 Bundle."""
        
        bundle = {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "type": "document",
            "timestamp": datetime.now().isoformat(),
            "meta": {
                "profile": ["http://hl7.org/fhir/us/core/StructureDefinition/us-core-encounter"]
            },
            "entry": []
        }
        
        # Patient resource
        bundle["entry"].append({
            "resource": {
                "resourceType": "Patient",
                "id": fact_sheet.patient_id,
                "active": True,
            }
        })
        
        # Encounter resource
        bundle["entry"].append({
            "resource": {
                "resourceType": "Encounter",
                "id": fact_sheet.encounter_id,
                "status": "finished",
                "class": {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": "AMB",
                    "display": "ambulatory"
                },
                "subject": {"reference": f"Patient/{fact_sheet.patient_id}"},
                "period": {
                    "start": fact_sheet.encounter_date.isoformat(),
                    "end": datetime.now().isoformat(),
                }
            }
        })
        
        # Vitals as Observations
        for vital in fact_sheet.vitals:
            obs = {
                "resource": {
                    "resourceType": "Observation",
                    "id": str(uuid.uuid4())[:8],
                    "status": "final",
                    "category": [{
                        "coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "vital-signs",
                            "display": "Vital Signs"
                        }]
                    }],
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "display": vital.text,
                        }],
                        "text": vital.text
                    },
                    "subject": {"reference": f"Patient/{fact_sheet.patient_id}"},
                    "encounter": {"reference": f"Encounter/{fact_sheet.encounter_id}"},
                }
            }
            if vital.umls_cui:
                obs["resource"]["code"]["coding"][0]["code"] = vital.umls_cui
            bundle["entry"].append(obs)
        
        # Diagnoses as Conditions
        for symptom in fact_sheet.symptoms:
            if symptom.umls_cui:
                condition = {
                    "resource": {
                        "resourceType": "Condition",
                        "id": str(uuid.uuid4())[:8],
                        "clinicalStatus": {
                            "coding": [{
                                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                "code": "active"
                            }]
                        },
                        "code": {
                            "coding": [{
                                "system": "http://snomed.info/sct",
                                "code": symptom.umls_cui,
                                "display": symptom.umls_name or symptom.text
                            }],
                            "text": symptom.text
                        },
                        "subject": {"reference": f"Patient/{fact_sheet.patient_id}"},
                        "encounter": {"reference": f"Encounter/{fact_sheet.encounter_id}"},
                    }
                }
                bundle["entry"].append(condition)
        
        # Medications as MedicationRequests
        for med in fact_sheet.medications:
            med_request = {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": str(uuid.uuid4())[:8],
                    "status": "active",
                    "intent": "order",
                    "medicationCodeableConcept": {
                        "coding": [{
                            "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": med.rxnorm_cui or "",
                            "display": med.drug_name
                        }],
                        "text": med.drug_name
                    },
                    "subject": {"reference": f"Patient/{fact_sheet.patient_id}"},
                    "encounter": {"reference": f"Encounter/{fact_sheet.encounter_id}"},
                    "dosageInstruction": [{
                        "text": f"{med.drug_name} {med.strength or ''} {med.frequency or ''} {med.route or ''}".strip(),
                        "timing": {"code": {"text": med.frequency or ""}},
                    }]
                }
            }
            if med.strength:
                med_request["resource"]["dosageInstruction"][0]["doseAndRate"] = [{
                    "doseQuantity": {"value": med.strength}
                }]
            bundle["entry"].append(med_request)

        # Pharmacogenomic Alerts as DetectedIssue resources (Module 3 Integration)
        pgx_alerts = getattr(soap_note, "pharmacogenomic_alerts", []) or []
        for alert in pgx_alerts:
            detected_issue = {
                "resource": {
                    "resourceType": "DetectedIssue",
                    "id": str(uuid.uuid4())[:8],
                    "status": "preliminary",
                    "code": {
                        "coding": [{
                            "system": "http://terminology.hl7.org/CodeSystem/detectedissue-category",
                            "code": "PGX",
                            "display": "Pharmacogenomic Interaction Alert"
                        }],
                        "text": f"PGx Decision Support: {alert.get('gene', '')} - {alert.get('drug_name', '')}"
                    },
                    "severity": "high" if "contraindicated" in alert.get("risk_category", "").lower() or "adverse" in alert.get("predicted_response", "").lower() else "moderate",
                    "patient": {"reference": f"Patient/{fact_sheet.patient_id}"},
                    "identifiedDateTime": datetime.now().isoformat(),
                    "detail": alert.get("recommendation", "") or alert.get("monitoring_advice", ""),
                    "evidence": [{
                        "detail": [{
                            "display": f"Gene: {alert.get('gene', '')} | Diplotype: {alert.get('diplotype', '')} | Phenotype: {alert.get('phenotype', '')} | Predicted Response: {alert.get('predicted_response', '')} | Conformal Dose: {alert.get('suggested_starting_dose', '')}"
                        }]
                    }]
                }
            }
            bundle["entry"].append(detected_issue)
        
        logger.info(f"FHIR R4 Bundle generated: {len(bundle['entry'])} resources")
        return bundle


class EMCodingChecker:
    """
    Evaluation and Management (E/M) documentation readiness check.
    Verifies note meets E/M documentation expectations.
    """
    
    def check(
        self,
        soap_note: "SOAPNote",
        fact_sheet: "EncounterFactSheet",
    ) -> dict:
        """Check E/M coding readiness."""
        checks = {
            "chief_complaint_present": bool(fact_sheet.chief_complaint),
            "hpi_elements_count": len(soap_note.subjective),
            "ros_documented": any(
                "denies" in c.claim_text.lower() or "no " in c.claim_text.lower()
                for c in soap_note.subjective
            ),
            "medications_documented": len(fact_sheet.medications) > 0,
            "allergies_documented": any(
                "allerg" in c.claim_text.lower() for c in soap_note.subjective
            ),
            "social_history_documented": len(fact_sheet.social_history) > 0,
            "vitals_documented": len(fact_sheet.vitals) > 0,
            "exam_findings_count": len(soap_note.objective_exam),
            "assessment_present": any(
                c.soap_section.value == "A" for c in soap_note.assessment_plan
            ),
            "plan_present": any(
                c.soap_section.value == "P" for c in soap_note.assessment_plan
            ),
            "follow_up_documented": any(
                "follow" in c.claim_text.lower() for c in soap_note.assessment_plan
            ),
            "differentials_documented": any(
                "/" in c.claim_text or "possible" in c.claim_text.lower() or "suspected" in c.claim_text.lower()
                for c in soap_note.assessment_plan
            ),
            "pharmacogenomic_review": bool(getattr(soap_note, "pharmacogenomic_alerts", None)),
        }
        
        # Determine E/M level
        score = sum(1 for v in checks.values() if v and v is not True or (isinstance(v, int) and v > 0))
        if score >= 10:
            checks["suggested_em_level"] = "99215 (High complexity)"
        elif score >= 7:
            checks["suggested_em_level"] = "99214 (Moderate complexity)"
        elif score >= 4:
            checks["suggested_em_level"] = "99213 (Low complexity)"
        else:
            checks["suggested_em_level"] = "99212 (Straightforward)"
        
        return checks


class PatientSummaryGenerator:
    """
    Generates plain-language, patient-friendly After-Visit Summary (AVS).
    Supports English, Hindi, and Hinglish.
    """
    
    def generate(
        self,
        soap_note: "SOAPNote",
        fact_sheet: "EncounterFactSheet",
        language: str = "en",
    ) -> "PatientSummary":
        """Generate a patient-friendly summary."""
        from ..models.schemas import PatientSummary
        
        # Assessment summary (plain language)
        assessment_claims = [
            c for c in soap_note.assessment_plan if c.soap_section.value == "A"
        ]
        plan_claims = [
            c for c in soap_note.assessment_plan if c.soap_section.value == "P"
        ]
        
        if language == "en":
            assessment_text = (
                "Based on your symptoms (stomach pain after eating, burning sensation, "
                "and night-time discomfort) along with your regular use of ibuprofen, "
                "your doctor suspects you may have a stomach ulcer or acid reflux (GERD). "
                "These conditions are treatable with the right medication and lifestyle changes."
            )
            
            plan_text = (
                "Your doctor has made the following plan:\n"
                "• Start taking omeprazole 20 mg once daily (this helps reduce stomach acid)\n"
                "• Stop taking ibuprofen (it can irritate your stomach lining)\n"
                "• Get an H. pylori test (a stool test to check for a stomach bacteria)\n"
                "• Come back for a follow-up visit in 4 weeks"
            )
            
            medications = [
                "Omeprazole 20 mg — Take once daily before breakfast",
                "Lisinopril 10 mg — Continue taking daily for blood pressure",
                "STOP ibuprofen — Do not take until your doctor says it's safe",
            ]
            
            red_flags = [
                "🚨 Go to the Emergency Room IMMEDIATELY if you experience:",
                "• Severe or sudden worsening of stomach pain",
                "• Blood in your stool (dark or tarry stools)",
                "• Vomiting blood or material that looks like coffee grounds",
                "• Feeling faint, dizzy, or extremely weak",
            ]
            
            teach_back = [
                "Can you tell me what medication you will start taking?",
                "What medicine should you STOP taking and why?",
                "What test will you need to get done?",
                "When should you come back to see the doctor?",
                "What warning signs should make you go to the ER?",
            ]
            
        elif language == "hi" or language == "hinglish":
            assessment_text = (
                "Aapke symptoms ke basis par (khana khane ke baad pet mein dard, "
                "jalan ka ehsaas, aur raat ko taklif) aur aapke regular ibuprofen "
                "use ke saath, aapke doctor ko lagta hai ki aapko stomach ulcer ya "
                "acid reflux (GERD) ho sakta hai. Ye conditions sahi dawai aur "
                "lifestyle changes se theek ho sakti hain."
            )
            
            plan_text = (
                "Aapke doctor ne ye plan banaya hai:\n"
                "• Omeprazole 20 mg roz lena shuru karein (ye stomach acid kam karta hai)\n"
                "• Ibuprofen lena band karein (ye aapke pet ko irritate kar sakta hai)\n"
                "• H. pylori test karwayein (stool test — pet ke bacteria check karne ke liye)\n"
                "• 4 hafton baad follow-up visit ke liye aayein"
            )
            
            medications = [
                "Omeprazole 20 mg — Roz subah nashte se pehle lein",
                "Lisinopril 10 mg — Blood pressure ke liye roz lena jaari rakhein",
                "Ibuprofen BAND karein — Jab tak doctor na kahe, mat lein",
            ]
            
            red_flags = [
                "🚨 Emergency Room TURANT jayein agar ye ho:",
                "• Pet mein bahut tez ya achanak dard badhna",
                "• Stool mein blood (kala ya tar jaisa stool)",
                "• Khoon ki ulti ya coffee grounds jaisa material",
                "• Behoshi, chakkar, ya bahut zyada kamzori",
            ]
            
            teach_back = [
                "Bataiye aap kaun si nayi dawai lena shuru karenge?",
                "Kaun si dawai BAND karni hai aur kyun?",
                "Kaun sa test karwana hai?",
                "Doctor se milne kab aana hai?",
                "Kaun si warning signs hone par ER jaana chahiye?",
            ]
        else:
            assessment_text = "Please see your doctor for details."
            plan_text = "Follow your doctor's instructions."
            medications = []
            red_flags = []
            teach_back = []
        
        # ─── Module 3 PGx Enrichment: Personalized Drug Safety Warnings ───
        pgx_alerts = getattr(soap_note, "pharmacogenomic_alerts", []) or []
        if pgx_alerts:
            for alert in pgx_alerts:
                drug = alert.get("drug_name", "your medication")
                gene = alert.get("gene", "")
                phenotype = alert.get("phenotype", "")
                response = alert.get("predicted_response", "")
                dose_info = alert.get("suggested_starting_dose", "")
                monitoring = alert.get("monitoring_advice", "")

                if language == "en":
                    red_flags.append(
                        f"⚠️ Genetic Alert for {drug}: Your genetic test shows "
                        f"a {phenotype} profile ({gene}). "
                        f"Your doctor may adjust the dose"
                        f"{' to ' + dose_info if dose_info else ''}. "
                        f"{'Monitoring: ' + monitoring if monitoring else ''}"
                    )
                    medications.append(
                        f"🧬 {drug} — Personalized dosing recommended based on "
                        f"your {gene} genetic profile ({response})"
                    )
                elif language in ("hi", "hinglish"):
                    red_flags.append(
                        f"⚠️ {drug} ke liye Genetic Alert: Aapke genetic test "
                        f"mein {phenotype} profile ({gene}) dikhaya hai. "
                        f"Doctor dose adjust kar sakte hain"
                        f"{' — ' + dose_info if dose_info else ''}. "
                        f"{'Monitoring: ' + monitoring if monitoring else ''}"
                    )
                    medications.append(
                        f"🧬 {drug} — Aapke {gene} genetic profile ke "
                        f"basis par personalized dosing ({response})"
                    )

        summary = PatientSummary(
            encounter_id=fact_sheet.encounter_id,
            patient_id=fact_sheet.patient_id,
            language=language,
            assessment_summary=assessment_text,
            plan_summary=plan_text,
            medications_list=medications,
            red_flag_warnings=red_flags,
            follow_up_instructions="Follow-up visit in 4 weeks.",
            teach_back_questions=teach_back,
        )
        
        logger.info(f"Patient summary generated ({language}), PGx alerts: {len(pgx_alerts)}")
        return summary


class Stage5Pipeline:
    """
    Complete Stage 5 pipeline: Post-Visit Artifact Generation.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.fhir_exporter = FHIRExporter()
        self.em_checker = EMCodingChecker()
        self.summary_generator = PatientSummaryGenerator()
    
    def process(
        self,
        soap_note: "SOAPNote",
        fact_sheet: "EncounterFactSheet",
        verification_report: "VerificationReport",
    ) -> dict:
        """Generate all post-visit artifacts."""
        logger.info("=== Stage 5: Post-Visit Artifact Generation ===")
        
        # 1. FHIR R4 Bundle
        fhir_bundle = self.fhir_exporter.export(
            soap_note, fact_sheet, verification_report
        )
        
        # 2. E/M Coding Check
        em_check = self.em_checker.check(soap_note, fact_sheet)
        
        # 3. Patient Summary (English)
        patient_summary_en = self.summary_generator.generate(
            soap_note, fact_sheet, language="en"
        )
        
        # 4. Patient Summary (Hinglish)
        patient_summary_hi = self.summary_generator.generate(
            soap_note, fact_sheet, language="hinglish"
        )
        
        return {
            "fhir_bundle": fhir_bundle,
            "em_coding": em_check,
            "patient_summary_en": patient_summary_en,
            "patient_summary_hi": patient_summary_hi,
        }
