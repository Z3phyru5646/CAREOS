"""
CARE'OS Module 2 — Clinical Co-Pilot Whisperer
================================================

Real-time clinical assistant that monitors the live consultation and:
  1. Maintains top-N diagnostic hypotheses
  2. Suggests red-flag questions not yet asked
  3. Surfaces pertinent negatives to rule out
  4. Anti-overreliance design (suggestions, not directives)

Modeled on DeepMind's dual-agent Planner+Talker architecture
and the CLARITY clinical dialogue system.
"""

import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Clinical Knowledge Base for Co-Pilot ─────────────────────────────────────

# Red-flag questions by symptom cluster
RED_FLAG_QUESTIONS_DB = {
    "epigastric_pain": {
        "differentials": [
            {"diagnosis": "Peptic Ulcer Disease (PUD)", "icd10": "K27.9", "confidence": 0.75},
            {"diagnosis": "Gastroesophageal Reflux Disease (GERD)", "icd10": "K21.0", "confidence": 0.70},
            {"diagnosis": "Gastritis (NSAID-induced)", "icd10": "K29.6", "confidence": 0.65},
            {"diagnosis": "Cholelithiasis", "icd10": "K80.20", "confidence": 0.30},
            {"diagnosis": "Pancreatitis", "icd10": "K85.9", "confidence": 0.20},
        ],
        "red_flags": [
            {
                "question": "Have you experienced any unintentional weight loss recently?",
                "rationale": "Unintentional weight loss with epigastric pain raises concern for gastric malignancy",
                "priority": "high"
            },
            {
                "question": "Do you have any difficulty swallowing (dysphagia)?",
                "rationale": "Dysphagia with epigastric symptoms may indicate esophageal pathology or stricture",
                "priority": "high"
            },
            {
                "question": "Have you noticed any jaundice (yellowing of eyes or skin)?",
                "rationale": "Jaundice with epigastric pain raises concern for biliary obstruction or pancreatic pathology",
                "priority": "high"
            },
            {
                "question": "Does the pain radiate to your back?",
                "rationale": "Back radiation of epigastric pain is a red flag for pancreatitis or aortic pathology",
                "priority": "medium"
            },
            {
                "question": "Any family history of stomach cancer or H. pylori infection?",
                "rationale": "Family history guides risk stratification for upper GI pathology",
                "priority": "medium"
            },
            {
                "question": "Have you traveled internationally recently?",
                "rationale": "Travel history may indicate infectious gastritis (H. pylori endemic regions)",
                "priority": "low"
            },
        ]
    },
    "chest_pain": {
        "differentials": [
            {"diagnosis": "Acute Coronary Syndrome", "icd10": "I21.9", "confidence": 0.60},
            {"diagnosis": "Pulmonary Embolism", "icd10": "I26.99", "confidence": 0.30},
            {"diagnosis": "Pneumothorax", "icd10": "J93.9", "confidence": 0.20},
            {"diagnosis": "GERD", "icd10": "K21.0", "confidence": 0.40},
            {"diagnosis": "Costochondritis", "icd10": "M94.0", "confidence": 0.35},
        ],
        "red_flags": [
            {
                "question": "Does the pain radiate to your left arm, jaw, or neck?",
                "rationale": "Classic radiation pattern for acute coronary syndrome",
                "priority": "high"
            },
            {
                "question": "Are you experiencing shortness of breath or sweating?",
                "rationale": "Associated symptoms suggestive of cardiac event",
                "priority": "high"
            },
            {
                "question": "Have you had any recent prolonged immobility or long travel?",
                "rationale": "Risk factor for pulmonary embolism (VTE risk assessment)",
                "priority": "high"
            },
        ]
    }
}


class ClinicalCoPilot:
    """
    Real-Time Clinical Co-Pilot that monitors the consultation
    and provides diagnostic suggestions and red-flag prompts.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.top_n = self.config.get("top_n_hypotheses", 5)
        self.num_red_flags = self.config.get("red_flag_questions", 5)
        self.sim_threshold_reuse = self.config.get("similarity_threshold_reuse", 0.965)
        self.sim_threshold_dedup = self.config.get("similarity_threshold_dedup", 0.86)
        
        self._current_hypotheses = []
        self._asked_questions = set()
        self._session_questions = []
    
    def update(
        self,
        transcript_so_far: list["DiarizedUtterance"],
        fact_sheet: "EncounterFactSheet" = None,
        elapsed_seconds: float = 0.0,
    ) -> "CoPilotUpdate":
        """
        Process current transcript state and generate Co-Pilot suggestions.
        Called periodically during the live consultation.
        """
        from ..models.schemas import (
            CoPilotUpdate, DiagnosticHypothesis, RedFlagQuestion
        )
        
        # Analyze current symptoms
        full_text = " ".join(u.text.lower() for u in transcript_so_far)
        
        # Determine SOAP phase
        soap_phase = self._detect_soap_phase(transcript_so_far)
        
        # Match symptom cluster
        cluster = self._match_symptom_cluster(full_text)
        
        hypotheses = []
        red_flags = []
        pertinent_negatives = []
        
        if cluster and cluster in RED_FLAG_QUESTIONS_DB:
            kb = RED_FLAG_QUESTIONS_DB[cluster]
            
            # Generate diagnostic hypotheses
            for diff in kb["differentials"][:self.top_n]:
                hypotheses.append(DiagnosticHypothesis(
                    hypothesis=diff["diagnosis"],
                    confidence=diff["confidence"],
                    icd10_code=diff["icd10"],
                    supporting_symptoms=self._find_supporting_symptoms(
                        full_text, diff["diagnosis"]
                    ),
                ))
            
            # Generate red-flag questions (filter already asked)
            for rf in kb["red_flags"]:
                if not self._is_already_asked(rf["question"], full_text):
                    red_flags.append(RedFlagQuestion(
                        question=rf["question"],
                        rationale=rf["rationale"],
                        priority=rf["priority"],
                        related_hypothesis=kb["differentials"][0]["diagnosis"],
                    ))
                    if len(red_flags) >= self.num_red_flags:
                        break
            
            # Suggest pertinent negatives to ask about
            pertinent_negatives = self._suggest_pertinent_negatives(
                full_text, cluster
            )
        
        update = CoPilotUpdate(
            current_soap_phase=soap_phase,
            hypotheses=hypotheses,
            red_flag_questions=red_flags,
            pertinent_negatives_to_ask=pertinent_negatives,
            elapsed_duration_sec=elapsed_seconds,
        )
        
        return update
    
    def _detect_soap_phase(self, utterances: list) -> str:
        """Detect current SOAP phase from conversation flow."""
        if not utterances:
            return "subjective"
        
        last_texts = " ".join(u.text.lower() for u in utterances[-3:])
        
        if any(word in last_texts for word in [
            "examine", "blood pressure", "heart rate", "abdomen",
            "palpation", "auscultation", "inspection"
        ]):
            return "objective"
        elif any(word in last_texts for word in [
            "assessment", "diagnosis", "concerned about", "think it's",
            "plan", "prescribe", "order", "follow-up", "recommend"
        ]):
            return "assessment_plan"
        
        return "subjective"
    
    def _match_symptom_cluster(self, text: str) -> Optional[str]:
        """Match transcript text to a known symptom cluster."""
        if any(w in text for w in ["epigastric", "stomach", "burning", "after eating", "nausea"]):
            return "epigastric_pain"
        elif any(w in text for w in ["chest pain", "chest tightness", "chest pressure"]):
            return "chest_pain"
        return None
    
    def _is_already_asked(self, question: str, transcript: str) -> bool:
        """Check if a similar question has already been asked in the dialogue."""
        question_keywords = set(question.lower().split())
        important_keywords = {
            w for w in question_keywords 
            if len(w) > 4 and w not in {"have", "been", "your", "does", "recently"}
        }
        
        # Check if key topics were already discussed
        discussed = sum(1 for kw in important_keywords if kw in transcript)
        return discussed >= len(important_keywords) * 0.6
    
    def _find_supporting_symptoms(self, text: str, diagnosis: str) -> list[str]:
        """Find symptoms in transcript that support a diagnosis."""
        symptom_maps = {
            "Peptic Ulcer Disease": ["pain", "burning", "epigastric", "ibuprofen", "nsaid", "nocturnal"],
            "GERD": ["burning", "reflux", "eating", "night", "acid"],
            "Gastritis": ["pain", "stomach", "nausea", "ibuprofen"],
            "Cholelithiasis": ["pain", "fatty food", "right upper"],
            "Pancreatitis": ["pain", "back", "severe", "nausea"],
        }
        
        keywords = symptom_maps.get(diagnosis, [])
        return [kw for kw in keywords if kw in text]
    
    def _suggest_pertinent_negatives(self, text: str, cluster: str) -> list[str]:
        """Suggest pertinent negatives that should be asked."""
        pertinent_neg_db = {
            "epigastric_pain": [
                "Weight loss", "Dysphagia", "Hematemesis",
                "Melena", "Jaundice", "Fever"
            ],
            "chest_pain": [
                "Diaphoresis", "Syncope", "Palpitations",
                "Leg swelling", "Recent surgery"
            ],
        }
        
        suggestions = []
        for neg in pertinent_neg_db.get(cluster, []):
            if neg.lower() not in text:
                suggestions.append(neg)
        
        return suggestions[:5]
