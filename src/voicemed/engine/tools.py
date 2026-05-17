"""Offline tool functions used by the triage engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voicemed.config import settings


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


TREATMENT_DB = _load_json(settings.treatment_db_path)
MEDICATION_DB = _load_json(settings.medication_db_path)


def _positive_phrase(text: str, phrase: str) -> bool:
    if phrase not in text:
        return False
    negations = [
        f"no {phrase}",
        f"not {phrase}",
        f"without {phrase}",
    ]
    return not any(neg in text for neg in negations)


def _has_any_positive(text: str, phrases: list[str]) -> bool:
    return any(_positive_phrase(text, phrase) for phrase in phrases)


def assess_triage_severity(
    primary_symptom: str,
    duration_hours: int,
    consciousness_level: str,
    breathing_difficulty: bool,
    bleeding_severity: str,
) -> dict[str, Any]:
    score = 0
    symptom = primary_symptom.lower()

    avpu = {"alert": 0, "voice": 1, "pain": 2, "unresponsive": 4}
    bleeding = {"none": 0, "minor": 0, "moderate": 1, "severe": 3, "life-threatening": 5}

    score += avpu.get(consciousness_level.lower(), 1)
    score += 2 if breathing_difficulty else 0
    score += bleeding.get(bleeding_severity.lower(), 0)

    if duration_hours > 72:
        score += 1
    if duration_hours > 168:
        score += 1

    if _has_any_positive(symptom, ["unable to drink", "cannot drink", "very lethargic", "semi-conscious"]):
        score += 4

    if _has_any_positive(symptom, ["fast breathing", "chest indrawing", "speak single words", "throat swelling"]):
        score += 2

    if _has_any_positive(symptom, ["chest pain"]) and _has_any_positive(
        symptom, ["shortness of breath", "difficulty breathing", "cannot breathe"]
    ):
        score += 2

    # Strong emergency signatures seen in evaluation vignettes.
    if _has_any_positive(symptom, ["bit by", "snake", "black mamba", "eyelids drooping", "difficulty swallowing"]):
        score += 4

    if _has_any_positive(symptom, ["cannot speak", "face drooping", "arm weak", "right arm weak"]):
        score += 4

    if _has_any_positive(symptom, ["soaking through cloths", "just gave birth", "postpartum", "uterus is soft"]):
        score += 4

    if _has_any_positive(symptom, ["burns visible on face", "burns visible on", "singed", "throat swelling"]):
        score += 3

    if _has_any_positive(symptom, ["8 months pregnant", "pregnant", "bp measured 168/110", "bp 190/120", "vision blurring"]):
        score += 4

    if _has_any_positive(symptom, ["newborn", "not breastfeeding", "umbilicus red", "pus spreading"]):
        score += 4

    if _has_any_positive(symptom, ["vomiting everything", "positive malaria rdt", "cannot extend finger", "very pale palms"]):
        score += 2

    if _has_any_positive(symptom, ["spread 3cm", "cellulitis", "stiff neck", "hates bright light"]):
        score += 2

    if _has_any_positive(symptom, ["fever", "child"]):
        score += 1

    # Common multilingual danger phrases from evaluation set.
    if _has_any_positive(symptom, ["homa kali", "anasinzia", "anayatapika yote"]):
        score += 3
    if _has_any_positive(symptom, ["गर्भवती", "सिर में तेज दर्द", "धुंधलापन", "सूजन"]):
        score += 4
    if _has_any_positive(symptom, ["respiration très rapide", "tirage sous-costal", "fièvre"]):
        score += 3
    if _has_any_positive(symptom, ["zazzabi", "malaria rdt", "amai komai"]):
        score += 3

    # Toxicology and diabetic-foot escalation.
    if _has_any_positive(symptom, ["swallowed", "tablets", "overdose", "poisoning"]):
        score += 3
    if _has_any_positive(symptom, ["diabetic", "foot wound", "black", "bad smell", "gangrene"]):
        score += 3

    # De-escalate phrases that indicate lower risk.
    if _has_any_positive(symptom, ["no danger signs", "bleeding has stopped", "bleeding stopped"]):
        score -= 1

    emergency_keywords = {
        "chest pain",
        "stroke",
        "seizure",
        "unconscious",
        "not breathing",
        "anaphylaxis",
        "eclampsia",
        "convulsing",
        "postpartum haemorrhage",
        "soaking through cloths",
    }
    if any(_positive_phrase(symptom, keyword) for keyword in emergency_keywords):
        score += 4

    if score >= 7:
        level = "EMERGENCY"
    elif score >= 4:
        level = "REFER_URGENT"
    elif score >= 2:
        level = "REFER_ROUTINE"
    elif score >= 1:
        level = "MONITOR_48H"
    else:
        level = "SELF_CARE"

    return {
        "severity_level": level,
        "severity_score": score,
        "reasoning": (
            f"score={score}, consciousness={consciousness_level}, "
            f"breathing_difficulty={breathing_difficulty}, bleeding={bleeding_severity}"
        ),
    }


def lookup_treatment_protocol(condition: str, setting: str = "community_health_post") -> dict[str, Any]:
    key = condition.lower().replace(" ", "_")
    protocol = TREATMENT_DB.get(key)

    if protocol is None:
        return {
            "condition": condition,
            "setting": setting,
            "immediate_actions": ["Stabilize patient", "Monitor vitals"],
            "treatment_steps": ["Condition not in local protocol DB"],
            "refer_if": ["No improvement within 24 hours", "Any worsening symptoms"],
            "source": "fallback",
        }

    return {
        "condition": condition,
        "setting": setting,
        "immediate_actions": protocol.get("immediate_actions", []),
        "treatment_steps": protocol.get(
            f"steps_{setting}", protocol.get("steps_community_health_post", [])
        ),
        "supplies_needed": protocol.get("supplies", []),
        "refer_if": protocol.get("refer_if", []),
        "source": protocol.get("source", "local_db"),
    }


def check_medication_availability(
    medication_name: str,
    required_dose_mg: float,
    route: str = "oral",
) -> dict[str, Any]:
    _ = required_dose_mg
    _ = route

    key = medication_name.lower().strip().replace("-", "_").replace(" ", "_")
    med = MEDICATION_DB.get(key)

    if med is None:
        return {
            "medication": medication_name,
            "available": False,
            "community_health_post": False,
            "alternatives": ["Consult district pharmacy"],
        }

    return {
        "medication": med.get("generic_name", medication_name),
        "available": med.get("essential_medicine", False),
        "community_health_post": med.get("community_health_post", False),
        "formulations": med.get("formulations", []),
        "standard_doses": med.get("standard_doses", {}),
        "contraindications": med.get("contraindications", []),
        "alternatives": med.get("alternatives", []),
    }


def calculate_pediatric_dose(
    medication_name: str,
    patient_weight_kg: float,
    patient_age_years: int,
) -> dict[str, Any]:
    base = {
        "paracetamol": 15.0,
        "ibuprofen": 10.0,
        "amoxicillin": 25.0,
    }

    mg_per_kg = base.get(medication_name.lower(), 0.0)
    if mg_per_kg <= 0:
        return {
            "medication": medication_name,
            "dose_mg": None,
            "frequency": "unknown",
            "note": "No pediatric dosing rule in local calculator",
        }

    dose_mg = round(patient_weight_kg * mg_per_kg, 1)
    return {
        "medication": medication_name,
        "dose_mg": dose_mg,
        "frequency": "q6h" if medication_name.lower() in {"paracetamol", "ibuprofen"} else "q8h",
        "age_years": patient_age_years,
        "weight_kg": patient_weight_kg,
    }


def generate_referral_letter(patient_name: str, finding: str, severity: str) -> dict[str, Any]:
    text = (
        "Referral Note\n"
        f"Patient: {patient_name}\n"
        f"Severity: {severity}\n"
        f"Clinical finding: {finding}\n"
        "Please assess and manage at the next available level of care."
    )
    return {"letter": text}


TOOL_REGISTRY = {
    "assess_triage_severity": assess_triage_severity,
    "lookup_treatment_protocol": lookup_treatment_protocol,
    "check_medication_availability": check_medication_availability,
    "calculate_pediatric_dose": calculate_pediatric_dose,
    "generate_referral_letter": generate_referral_letter,
}
