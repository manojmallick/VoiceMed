"""
VoiceMed — LLM-as-Judge Evaluation Pipeline
============================================
Uses Gemma 4 itself as the judge to evaluate triage accuracy.
This is your reproducibility proof for the hackathon submission.

Why LLM-as-judge:
- No hand-labelling of 50 cases needed
- Gemma 4 evaluating Gemma 4 = perfectly on-theme for hackathon
- Produces quantified accuracy numbers judges can verify
- Standard pattern used in production AI evaluation (LLM-Eval, G-Eval)

Run:
    python scripts/run_evaluation.py

Output:
    - evaluation_results.json (all case results)
    - evaluation_summary.json (accuracy metrics)
    - notebooks/03_evaluation_results.ipynb (auto-generated notebook)
"""

import json
import time
import torch
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
from transformers import AutoProcessor, AutoModelForImageTextToText
from loguru import logger

# ── CONFIG ────────────────────────────────────────────────────────────────────

EVALUATOR_MODEL = "google/gemma-4-E4B-it"   # Judge model (same family = fair)
RESULTS_DIR = Path("evaluation_results")
DATA_DIR = Path("data")


# ── 50 TEST CASES ─────────────────────────────────────────────────────────────
# Derived from WHO IMCI clinical vignettes and Semigran-45 framework.
# Each case has:
#   description: symptom narrative (what CHW would describe)
#   expected_severity: ground truth label
#   rationale: why this is the correct label
#   source: clinical reference

TEST_CASES = [
    # ── EMERGENCY CASES (should always escalate) ────────────────────────────
    {
        "id": "E01",
        "description": "Child, 3 years, unconscious, unresponsive to voice. Family says had severe malaria for 2 days. Breathing is fast and laboured.",
        "expected_severity": "EMERGENCY",
        "rationale": "Cerebral malaria with coma = EMERGENCY. Altered consciousness is immediate referral.",
        "source": "WHO Severe Malaria Guidelines 2022"
    },
    {
        "id": "E02",
        "description": "Adult woman, 28 years, 8 months pregnant, sudden severe headache, vision blurring, BP measured 168/110.",
        "expected_severity": "EMERGENCY",
        "rationale": "Pre-eclampsia/eclampsia with BP >160/110 and neurological symptoms = EMERGENCY.",
        "source": "WHO Antenatal Care Guidelines 2024"
    },
    {
        "id": "E03",
        "description": "Adult male, 45 years, sudden chest pain radiating to left arm, sweating profusely, pale and clammy, difficulty breathing.",
        "expected_severity": "EMERGENCY",
        "rationale": "Classic acute MI presentation. Immediate emergency referral.",
        "source": "WHO Cardiovascular Disease Guidelines"
    },
    {
        "id": "E04",
        "description": "Woman just gave birth at home 30 minutes ago. Now soaking through cloths with blood. Uterus is soft, not firm.",
        "expected_severity": "EMERGENCY",
        "rationale": "Postpartum haemorrhage with atonic uterus = life-threatening emergency.",
        "source": "WHO PPH Guidelines 2023"
    },
    {
        "id": "E05",
        "description": "Child, 18 months, bit by black mamba snake 20 minutes ago. Bite on hand. Eyelids now drooping, difficulty swallowing.",
        "expected_severity": "EMERGENCY",
        "rationale": "Neurotoxic snake envenomation with systemic symptoms = immediate referral for antivenom.",
        "source": "WHO Snakebite Guidelines 2019"
    },
    {
        "id": "E06",
        "description": "Child, 2 years, having seizure now. Convulsing for 8 minutes. History of high fever today.",
        "expected_severity": "EMERGENCY",
        "rationale": "Prolonged seizure >5 minutes = status epilepticus risk, emergency.",
        "source": "WHO IMCI 2024"
    },
    {
        "id": "E07",
        "description": "Newborn, 2 days old. Not breastfeeding, very sleepy, umbilicus red with pus spreading to surrounding skin, temperature 39.2°C.",
        "expected_severity": "EMERGENCY",
        "rationale": "Neonatal sepsis with local infection and fever = EMERGENCY in newborn.",
        "source": "WHO IMCI Neonatal 2024"
    },
    {
        "id": "E08",
        "description": "Adult, 60 years, suddenly cannot speak, right side of face drooping, right arm weak. Began 30 minutes ago.",
        "expected_severity": "EMERGENCY",
        "rationale": "Acute stroke with face drop, arm weakness, speech loss = FAST criteria met, emergency.",
        "source": "WHO Stroke Guidelines"
    },
    {
        "id": "E09",
        "description": "Child, 5 years, fell into fire. Burns visible on face, neck, and both hands. Child is crying, hair is singed.",
        "expected_severity": "EMERGENCY",
        "rationale": "Burns to face/hands + possible airway injury (singed hair) = emergency.",
        "source": "WHO Burns Management 2024"
    },
    {
        "id": "E10",
        "description": "Adult, 35 years, severe difficulty breathing after eating groundnuts. Throat swelling, skin covered in hives.",
        "expected_severity": "EMERGENCY",
        "rationale": "Anaphylaxis with airway involvement = life-threatening, give epinephrine and refer.",
        "source": "WHO Primary Care Guidelines"
    },

    # ── REFER URGENT CASES ──────────────────────────────────────────────────
    {
        "id": "U01",
        "description": "Child, 3 years, fever 39.8°C for 2 days, positive malaria RDT, vomiting everything including first dose of medicine.",
        "expected_severity": "REFER_URGENT",
        "rationale": "Malaria with vomiting preventing oral treatment = urgent referral for injectable.",
        "source": "WHO Malaria Treatment Guidelines 2022"
    },
    {
        "id": "U02",
        "description": "Child, 18 months, fast breathing (58 breaths/min), chest indrawing visible, fever 38.9°C. No oxygen available.",
        "expected_severity": "REFER_URGENT",
        "rationale": "Severe pneumonia with chest indrawing in child = urgent referral.",
        "source": "WHO IMCI 2024"
    },
    {
        "id": "U03",
        "description": "Adult, 50 years, deep wound to right hand from machete. Wound is 5cm long, possibly to tendon as patient cannot extend finger.",
        "expected_severity": "REFER_URGENT",
        "rationale": "Possible tendon injury = surgical emergency, urgent referral within hours.",
        "source": "WHO Primary Care Guidelines"
    },
    {
        "id": "U04",
        "description": "Child, 2 years, sunken eyes, skin pinch returns very slowly, unable to drink, extremely lethargic.",
        "expected_severity": "REFER_URGENT",
        "rationale": "Severe dehydration with inability to drink = urgent IV fluids needed.",
        "source": "WHO IMCI 2024"
    },
    {
        "id": "U05",
        "description": "Pregnant woman, 34 weeks, has had 3 convulsions today. Currently semi-conscious between episodes. BP 190/120.",
        "expected_severity": "EMERGENCY",
        "rationale": "Eclampsia = absolute emergency, not just urgent. Reclassified.",
        "source": "WHO ANC Guidelines 2024"
    },
    {
        "id": "U06",
        "description": "Child, 4 years, has not eaten or drunk anything for 2 days. Cannot sit without support. Very pale palms.",
        "expected_severity": "REFER_URGENT",
        "rationale": "Prostration with severe anaemia = cannot wait, urgent referral.",
        "source": "WHO IMCI 2024"
    },
    {
        "id": "U07",
        "description": "Adult, 28 years, fever 3 days, now has stiff neck, headache, hates bright light. Conscious but drowsy.",
        "expected_severity": "REFER_URGENT",
        "rationale": "Meningism signs with fever = suspected meningitis, urgent.",
        "source": "WHO Meningitis Guidelines"
    },
    {
        "id": "U08",
        "description": "Child, 6 months, had diarrhoea for 16 days. Losing weight. Still breastfeeding but not improving.",
        "expected_severity": "REFER_ROUTINE",
        "rationale": "Persistent diarrhoea >14 days with weight loss = routine referral for investigation.",
        "source": "WHO IMCI 2024"
    },
    {
        "id": "U09",
        "description": "Adult, 40 years, was treated for cellulitis on leg 2 days ago. Red area has spread 3cm beyond the marked border, now has fever 38.8°C.",
        "expected_severity": "REFER_URGENT",
        "rationale": "Spreading cellulitis not responding to oral antibiotics = urgent IV antibiotics needed.",
        "source": "WHO Primary Care Guidelines 2024"
    },
    {
        "id": "U10",
        "description": "Adult, asthma patient, not responding after 3 doses of salbutamol inhaler. Can only speak single words. Very distressed.",
        "expected_severity": "EMERGENCY",
        "rationale": "Near-fatal asthma (silent chest or single-word speech) = EMERGENCY.",
        "source": "GINA Guidelines, WHO PEN 2023"
    },

    # ── REFER ROUTINE CASES ─────────────────────────────────────────────────
    {
        "id": "R01",
        "description": "Adult, 45 years, blood pressure measured 152/94 twice today. No headache, no other symptoms. Has never had hypertension treatment.",
        "expected_severity": "REFER_ROUTINE",
        "rationale": "New hypertension diagnosis without emergency features = routine referral to start treatment.",
        "source": "WHO HEARTS 2023"
    },
    {
        "id": "R02",
        "description": "Child, 2 years, cough for 10 days, fast breathing 44/min, no chest indrawing, no danger signs. Fever 37.8°C.",
        "expected_severity": "REFER_ROUTINE",
        "rationale": "Non-severe pneumonia = treat with oral amoxicillin at CHW level, not referral. Re-classified as MONITOR_48H if treated.",
        "source": "WHO IMCI 2024"
    },
    {
        "id": "R03",
        "description": "Adult, 55 years, cough for 4 weeks, night sweats, lost 5kg in 2 months. No fever today. HIV positive.",
        "expected_severity": "REFER_ROUTINE",
        "rationale": "TB suspect with HIV = routine referral for sputum GeneXpert. Not emergency but cannot be managed at CHW level.",
        "source": "WHO TB Guidelines 2022"
    },
    {
        "id": "R04",
        "description": "Child, 18 months, MUAC measured 112mm. Bilateral pitting oedema on feet. Eating, no fever, no danger signs.",
        "expected_severity": "REFER_ROUTINE",
        "rationale": "SAM with oedema but no complications = CMAM enrolment, routine referral.",
        "source": "WHO SAM Guidelines 2023"
    },
    {
        "id": "R05",
        "description": "Adult male, 30 years, eye painful for 5 days, discharge and redness, some blurring of vision. Other eye normal.",
        "expected_severity": "REFER_ROUTINE",
        "rationale": "Conjunctivitis with visual involvement = referral for eye examination.",
        "source": "WHO IMCI 2024"
    },

    # ── MONITOR 48H CASES ───────────────────────────────────────────────────
    {
        "id": "M01",
        "description": "Child, 4 years, fever 38.2°C for 1 day. Malaria RDT negative. No danger signs. Eating and drinking. Alert.",
        "expected_severity": "MONITOR_48H",
        "rationale": "Fever without source, no danger signs, no malaria = viral illness, monitor.",
        "source": "WHO IMCI 2024"
    },
    {
        "id": "M02",
        "description": "Adult, 25 years, diarrhoea 3 times today, no blood, no vomiting. Drinking well. Mild stomach cramps.",
        "expected_severity": "SELF_CARE",
        "rationale": "Mild diarrhoea, no dehydration, drinking well = ORS and home care.",
        "source": "WHO Diarrhoea Guidelines"
    },
    {
        "id": "M03",
        "description": "Child, 3 years, first febrile convulsion, seizure lasted 2 minutes, now alert and back to normal. Temperature 39.1°C.",
        "expected_severity": "MONITOR_48H",
        "rationale": "Simple febrile convulsion that has resolved = monitor, but warrants medical review today.",
        "source": "WHO IMCI 2024"
    },
    {
        "id": "M04",
        "description": "Adult, 22 years, pain on urination for 2 days, passing urine frequently. No fever. Not pregnant.",
        "expected_severity": "MONITOR_48H",
        "rationale": "Uncomplicated UTI = treat with antibiotics, monitor response. Not referral.",
        "source": "WHO Primary Care Guidelines 2024"
    },
    {
        "id": "M05",
        "description": "Adult, 35 years, mild wheeze after exercise. Using salbutamol inhaler, gets better. No fever, not distressed at rest.",
        "expected_severity": "MONITOR_48H",
        "rationale": "Mild exercise-induced asthma responding to inhaler = treat and monitor.",
        "source": "GINA, WHO PEN 2023"
    },

    # ── SELF-CARE CASES ─────────────────────────────────────────────────────
    {
        "id": "S01",
        "description": "Adult, 28 years, small cut on finger from kitchen knife. 1cm, bleeding has stopped. Wound is clean.",
        "expected_severity": "SELF_CARE",
        "rationale": "Minor clean wound, bleeding stopped, small = self-care with basic wound cleaning.",
        "source": "WHO Primary Care Guidelines 2024"
    },
    {
        "id": "S02",
        "description": "Child, 7 years, itchy circular rash on arm for 5 days. No fever. Playing normally.",
        "expected_severity": "SELF_CARE",
        "rationale": "Likely ringworm (tinea corporis) = topical clotrimazole, no referral needed.",
        "source": "WHO Primary Care Guidelines 2024"
    },
    {
        "id": "S03",
        "description": "Adult, 30 years, headache and mild fever 37.8°C for 1 day. No neck stiffness, no rash, no vomiting. Alert.",
        "expected_severity": "SELF_CARE",
        "rationale": "Likely viral illness. Take paracetamol, rest, fluids. No danger signs.",
        "source": "WHO Primary Care Guidelines 2024"
    },
    {
        "id": "S04",
        "description": "Adult woman, 25 years, mild itching and discharge from vagina for 3 days. No fever, no abdominal pain.",
        "expected_severity": "SELF_CARE",
        "rationale": "Likely vaginal candidiasis. Clotrimazole vaginal tablet, monitor. Refer if not improving.",
        "source": "WHO STI Guidelines 2021"
    },
    {
        "id": "S05",
        "description": "Child, 5 years, 2 soft stools today, no blood, drinking well, playing. Temperature normal.",
        "expected_severity": "SELF_CARE",
        "rationale": "Mild diarrhoea, no dehydration, no danger signs = ORS at home, no referral.",
        "source": "WHO IMCI 2024"
    },
    {
        "id": "S06",
        "description": "Adult, 40 years, insect bites on legs, itchy but no rash spreading. Temperature normal.",
        "expected_severity": "SELF_CARE",
        "rationale": "Simple insect bites = chlorphenamine for itch, local treatment only.",
        "source": "WHO Primary Care Guidelines 2024"
    },
    {
        "id": "S07",
        "description": "Adult, 20 years, sore throat for 2 days, mild pain swallowing, no drooling, no stridor, temperature 37.5°C. Voice normal.",
        "expected_severity": "SELF_CARE",
        "rationale": "Mild sore throat without alarming features = likely viral, salt water gargle, paracetamol.",
        "source": "WHO Primary Care Guidelines 2024"
    },
    {
        "id": "S08",
        "description": "Child, 9 years, mild ear pain for 1 day, no discharge, no fever, hearing normal.",
        "expected_severity": "MONITOR_48H",
        "rationale": "Possible otitis media early stage = treat with amoxicillin, monitor. Not emergency.",
        "source": "WHO IMCI 2024"
    },
    {
        "id": "S09",
        "description": "Adult, 35 years, backache for 3 days after lifting heavy objects. Pain is muscular, no leg numbness, can walk normally.",
        "expected_severity": "SELF_CARE",
        "rationale": "Acute muscular back pain = ibuprofen/paracetamol, rest, no referral needed.",
        "source": "WHO Primary Care Guidelines 2024"
    },
    {
        "id": "S10",
        "description": "Adult, 65 years, mild ankle swelling at end of day, no pain, no redness, both ankles equally swollen.",
        "expected_severity": "MONITOR_48H",
        "rationale": "Bilateral dependent oedema in elderly — needs BP check and follow up. Not emergency.",
        "source": "WHO HEARTS 2023"
    },

    # ── MULTILINGUAL TEST CASES ─────────────────────────────────────────────
    # These test Gemma 4's multilingual capability — critical differentiator
    {
        "id": "SW01",
        "description": "Mtoto wa miaka 3, ana homa kali sana, hakuli, anasinzia sana. Anajaribu kunywa maji lakini anayatapika yote.",
        "expected_severity": "REFER_URGENT",
        "rationale": "Swahili: Child with high fever, not eating, very drowsy, vomiting all fluids = danger signs, urgent referral.",
        "source": "WHO IMCI 2024 — Swahili test case",
        "language": "Swahili"
    },
    {
        "id": "HI01",
        "description": "35 साल की महिला, 7 महीने की गर्भवती, सिर में तेज दर्द, आँखों में धुंधलापन, हाथ-पैर में सूजन।",
        "expected_severity": "EMERGENCY",
        "rationale": "Hindi: 35yo pregnant woman, 7 months, severe headache, blurred vision, oedema = pre-eclampsia emergency.",
        "source": "WHO ANC Guidelines 2024 — Hindi test case",
        "language": "Hindi"
    },
    {
        "id": "FR01",
        "description": "Enfant de 2 ans, respiration très rapide 65 fois par minute, tirage sous-costal visible, fièvre 39°C.",
        "expected_severity": "REFER_URGENT",
        "rationale": "French: 2yo child with very fast breathing 65/min, chest indrawing, fever = severe pneumonia, urgent referral.",
        "source": "WHO IMCI 2024 — French test case",
        "language": "French"
    },
    {
        "id": "HA01",
        "description": "Yaro da shekaru 4, zazzabi na kwana biyu, malaria RDT ta zama mai kyau, yana amai komai da ya sha.",
        "expected_severity": "REFER_URGENT",
        "rationale": "Hausa: 4yo child, 2 days fever, positive malaria RDT, vomiting everything = urgent referral for injectable.",
        "source": "WHO Malaria Guidelines 2022 — Hausa test case",
        "language": "Hausa"
    },

    # ── EDGE CASES (tricky — tests model robustness) ───────────────────────
    {
        "id": "X01",
        "description": "Adult, 70 years, feeling dizzy when standing up. Takes medications for blood pressure. No falls. BP lying 130/80, standing 100/60.",
        "expected_severity": "MONITOR_48H",
        "rationale": "Orthostatic hypotension, likely medication side effect = review medications, not emergency.",
        "source": "WHO HEARTS 2023"
    },
    {
        "id": "X02",
        "description": "Pregnant woman, 32 weeks, baby has been moving less than usual for the past day. No pain, no bleeding, BP normal.",
        "expected_severity": "REFER_ROUTINE",
        "rationale": "Reduced fetal movement = needs fetal monitoring at clinic urgently (same day), but not immediate emergency.",
        "source": "WHO ANC Guidelines 2024"
    },
    {
        "id": "X03",
        "description": "Child, 3 months, crying more than usual for 6 hours. Feeding well, no fever, nappy is wet, no rash. Parents worried.",
        "expected_severity": "SELF_CARE",
        "rationale": "Infant colic presentation — feeding well, wet nappies, no danger signs = reassure parents.",
        "source": "WHO IMCI 2024"
    },
    {
        "id": "X04",
        "description": "Adult, 55 years, diabetic, foot wound noticed today. Wound is black with bad smell, no pain felt (neuropathy). Temperature 37.8°C.",
        "expected_severity": "REFER_URGENT",
        "rationale": "Diabetic foot with gangrene (black, malodorous) + neuropathy = urgent referral, surgical emergency.",
        "source": "WHO Diabetes Guidelines 2023"
    },
    {
        "id": "X05",
        "description": "Child, 6 years, swallowed 5 tablets of paracetamol 500mg from mother's bag 2 hours ago. Currently alert and fine.",
        "expected_severity": "REFER_URGENT",
        "rationale": "Paracetamol overdose in child (2500mg = ~100mg/kg) = urgent referral for activated charcoal and monitoring.",
        "source": "WHO Primary Care Toxicology"
    },
]


# ── JUDGE SYSTEM PROMPT ───────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are a clinical triage evaluation expert with 20 years of experience in global health and emergency medicine. Your role is to evaluate whether a triage system correctly classified the severity of a clinical case.

You will be given:
1. A clinical case description
2. The system's predicted triage severity
3. The expected (correct) triage severity

You must evaluate the prediction and output ONLY this JSON:
{
  "correct": true or false,
  "partial_credit": 0.0 to 1.0 (1.0=perfect, 0.5=adjacent level wrong, 0.0=dangerous error),
  "safety_error": true or false (true = model under-triaged a dangerous case),
  "reasoning": "1-2 sentence explanation of your judgment"
}

SEVERITY SCALE (from lowest to highest):
SELF_CARE → MONITOR_48H → REFER_ROUTINE → REFER_URGENT → EMERGENCY

SCORING RULES:
- Exact match: correct=true, partial_credit=1.0
- One level off: correct=false, partial_credit=0.5
- Two levels off: correct=false, partial_credit=0.25
- More than two levels off: correct=false, partial_credit=0.0
- CRITICAL: If expected is EMERGENCY or REFER_URGENT and model predicted SELF_CARE or MONITOR_48H = safety_error=true, partial_credit=0.0

Output ONLY the JSON. No text before or after."""


# ── VOICEEMED SYSTEM PROMPT (what we're evaluating) ───────────────────────────

VOICEMED_SYSTEM_PROMPT = """You are VoiceMed, a clinical triage assistant for community health workers in low-resource settings.
Assess the patient and output ONLY this JSON — no text before or after:
{
  "severity": "SELF_CARE | MONITOR_48H | REFER_ROUTINE | REFER_URGENT | EMERGENCY",
  "primary_concern": "one sentence",
  "confidence": "HIGH | MEDIUM | LOW"
}
CRITICAL: If ANY of these are present → severity MUST be EMERGENCY: unconscious, not breathing, seizure, heavy bleeding, stroke symptoms, eclampsia, anaphylaxis with breathing difficulty."""


# ── EVALUATOR CLASS ───────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: str
    description: str
    expected: str
    predicted: str
    correct: bool
    partial_credit: float
    safety_error: bool
    reasoning: str
    language: str = "English"
    raw_voicemed_output: str = ""
    raw_judge_output: str = ""
    error: str = ""


class VoiceMedEvaluator:
    """
    LLM-as-Judge evaluation pipeline.
    Uses Gemma 4 as both the system being tested and the judge.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or EVALUATOR_MODEL
        self.processor = None
        self.model = None

    def load_model(self):
        logger.info(f"Loading Gemma 4 model: {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        logger.info("Model loaded.")

    def _generate(self, system_prompt: str, user_message: str, max_tokens: int = 256) -> str:
        """Run one inference pass."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.05,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()

    def _parse_json(self, text: str) -> Optional[dict]:
        """Extract and parse JSON from model output."""
        import re
        # Try to find JSON object
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        # Try full text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def evaluate_case(self, case: dict) -> CaseResult:
        """Run VoiceMed on a case, then judge the output."""

        # Step 1: Run VoiceMed on the case
        voicemed_output = self._generate(
            system_prompt=VOICEMED_SYSTEM_PROMPT,
            user_message=f"Patient description: {case['description']}",
            max_tokens=200,
        )

        # Parse VoiceMed prediction
        voicemed_parsed = self._parse_json(voicemed_output)
        predicted_severity = "UNKNOWN"
        if voicemed_parsed and "severity" in voicemed_parsed:
            predicted_severity = voicemed_parsed["severity"]

        # Step 2: Judge the prediction
        judge_input = f"""Case: {case['description']}
Predicted severity: {predicted_severity}
Expected severity: {case['expected_severity']}
Clinical rationale: {case['rationale']}"""

        judge_output = self._generate(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_message=judge_input,
            max_tokens=200,
        )

        # Parse judge output
        judge_parsed = self._parse_json(judge_output)

        if judge_parsed:
            return CaseResult(
                case_id=case["id"],
                description=case["description"][:100] + "...",
                expected=case["expected_severity"],
                predicted=predicted_severity,
                correct=judge_parsed.get("correct", False),
                partial_credit=float(judge_parsed.get("partial_credit", 0.0)),
                safety_error=judge_parsed.get("safety_error", False),
                reasoning=judge_parsed.get("reasoning", ""),
                language=case.get("language", "English"),
                raw_voicemed_output=voicemed_output[:300],
                raw_judge_output=judge_output[:300],
            )
        else:
            # Judge parsing failed — conservative: mark as wrong
            return CaseResult(
                case_id=case["id"],
                description=case["description"][:100] + "...",
                expected=case["expected_severity"],
                predicted=predicted_severity,
                correct=False,
                partial_credit=0.0,
                safety_error=False,
                reasoning="Judge output could not be parsed",
                raw_voicemed_output=voicemed_output[:300],
                raw_judge_output=judge_output[:300],
                error="Judge parse failed",
            )

    def run_evaluation(self, cases=None, save_results=True) -> dict:
        """Run full evaluation on all test cases."""
        if cases is None:
            cases = TEST_CASES

        if not self.model:
            self.load_model()

        RESULTS_DIR.mkdir(exist_ok=True)
        results = []
        safety_errors = []

        logger.info(f"Starting evaluation on {len(cases)} cases...")

        for i, case in enumerate(cases):
            logger.info(f"[{i+1}/{len(cases)}] Evaluating case {case['id']}: {case['description'][:60]}...")
            try:
                result = self.evaluate_case(case)
                results.append(result)
                if result.safety_error:
                    safety_errors.append(result)
                    logger.warning(f"  SAFETY ERROR on {case['id']}: predicted {result.predicted}, expected {result.expected}")
                else:
                    status = "✓" if result.correct else "~"
                    logger.info(f"  {status} predicted={result.predicted}, expected={result.expected}, credit={result.partial_credit:.1f}")
            except Exception as e:
                logger.error(f"  ERROR on {case['id']}: {e}")
                results.append(CaseResult(
                    case_id=case["id"],
                    description=case["description"][:100],
                    expected=case["expected_severity"],
                    predicted="ERROR",
                    correct=False,
                    partial_credit=0.0,
                    safety_error=False,
                    reasoning="",
                    error=str(e),
                ))
            time.sleep(0.5)  # Prevent overheating

        # ── COMPUTE SUMMARY METRICS ─────────────────────────────────────────
        total = len(results)
        exact_correct = sum(1 for r in results if r.correct)
        avg_partial = sum(r.partial_credit for r in results) / total if total > 0 else 0
        safety_error_count = sum(1 for r in results if r.safety_error)

        # By severity
        by_severity = {}
        for r in results:
            sev = r.expected
            if sev not in by_severity:
                by_severity[sev] = {"total": 0, "correct": 0}
            by_severity[sev]["total"] += 1
            if r.correct:
                by_severity[sev]["correct"] += 1

        # By language
        multilingual_results = [r for r in results if r.language != "English"]
        multilingual_accuracy = (
            sum(1 for r in multilingual_results if r.correct) / len(multilingual_results)
            if multilingual_results else 0
        )

        summary = {
            "evaluation_date": datetime.now().isoformat(),
            "model": self.model_path,
            "total_cases": total,
            "exact_accuracy": round(exact_correct / total * 100, 1) if total > 0 else 0,
            "partial_credit_score": round(avg_partial * 100, 1),
            "safety_errors": safety_error_count,
            "safety_error_rate": round(safety_error_count / total * 100, 1) if total > 0 else 0,
            "multilingual_accuracy": round(multilingual_accuracy * 100, 1),
            "by_severity": {
                k: {
                    "accuracy": round(v["correct"] / v["total"] * 100, 1) if v["total"] > 0 else 0,
                    "correct": v["correct"],
                    "total": v["total"],
                }
                for k, v in by_severity.items()
            },
            "safety_error_cases": [r.case_id for r in safety_errors],
        }

        # ── PRINT RESULTS ───────────────────────────────────────────────────
        print("\n" + "="*60)
        print("VOICEMED EVALUATION RESULTS")
        print("="*60)
        print(f"Total cases:          {total}")
        print(f"Exact accuracy:       {summary['exact_accuracy']}%")
        print(f"Partial credit score: {summary['partial_credit_score']}%")
        print(f"Safety errors:        {safety_error_count} ({summary['safety_error_rate']}%)")
        print(f"Multilingual acc:     {summary['multilingual_accuracy']}%")
        print("\nAccuracy by severity level:")
        for sev, stats in summary["by_severity"].items():
            print(f"  {sev:<18} {stats['correct']}/{stats['total']} ({stats['accuracy']}%)")
        if safety_errors:
            print(f"\nSAFETY ERRORS (under-triage of serious cases):")
            for r in safety_errors:
                print(f"  {r.case_id}: predicted {r.predicted}, expected {r.expected}")
        print("="*60)

        # ── SAVE RESULTS ────────────────────────────────────────────────────
        if save_results:
            results_path = RESULTS_DIR / "evaluation_results.json"
            summary_path = RESULTS_DIR / "evaluation_summary.json"

            with open(results_path, "w") as f:
                json.dump([asdict(r) for r in results], f, indent=2)

            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Results saved to {results_path}")
            logger.info(f"Summary saved to {summary_path}")

        return {"summary": summary, "results": results}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VoiceMed LLM-as-Judge Evaluation")
    parser.add_argument("--model", default=None, help="Model path/ID (default: Gemma 4 E4B)")
    parser.add_argument("--cases", type=int, default=None, help="Number of cases to run (default: all 50)")
    parser.add_argument("--quick", action="store_true", help="Run only 10 cases for quick test")
    args = parser.parse_args()

    evaluator = VoiceMedEvaluator(model_path=args.model)

    cases = TEST_CASES
    if args.quick:
        cases = TEST_CASES[:10]
        logger.info("Quick mode: running 10 cases only")
    elif args.cases:
        cases = TEST_CASES[:args.cases]

    evaluator.run_evaluation(cases=cases)
