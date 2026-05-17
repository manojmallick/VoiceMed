"""Initial VoiceMed triage engine.

This first implementation is intentionally offline and lightweight.
It provides deterministic triage using rule-based extraction + tool calls,
and can be swapped with full Gemma inference in a later milestone.
"""

from __future__ import annotations

import re
import time
import json
import urllib.error
import urllib.request
from typing import Any

from voicemed.config import settings
from voicemed.engine.system_prompt import SYSTEM_PROMPT
from voicemed.engine.tool_executor import ToolExecutor
from voicemed.output.schemas import SeverityLevel, TriageResult


class Gemma4TriageEngine:
    def __init__(self) -> None:
        self.tool_executor = ToolExecutor()
        self._model = None
        self._processor = None
        self._model_ready = False
        self._model_load_attempted = False
        self._model_load_error: str | None = None
        self._last_model_reason: str | None = None
        self._debug_snapshot: dict[str, Any] = {
            "enable_model_inference": settings.enable_model_inference,
            "model_id": settings.model_id,
            "use_ollama": settings.use_ollama,
            "ollama_model": settings.ollama_model,
        }

    def get_debug_snapshot(self) -> dict[str, Any]:
        return dict(self._debug_snapshot)

    def _set_debug(self, **kwargs: Any) -> None:
        self._debug_snapshot.update(kwargs)

    def load(self) -> None:
        """Optionally loads model components when feature-flagged."""
        started = time.perf_counter()
        self._set_debug(load_called=True)
        if not settings.enable_model_inference or self._model_load_attempted:
            self._set_debug(
                model_ready=self._model_ready,
                model_load_attempted=self._model_load_attempted,
                model_load_error=self._model_load_error,
            )
            return

        self._model_load_attempted = True
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(settings.model_id)
            self._model = AutoModelForImageTextToText.from_pretrained(
                settings.model_id,
                dtype=torch.float32,
                device_map="cpu",
            )
            self._model_ready = True
            self._model_load_error = None
            self._set_debug(
                model_ready=True,
                model_load_error=None,
                load_elapsed_sec=round(time.perf_counter() - started, 3),
            )
        except Exception as exc:
            self._model_ready = False
            self._model_load_error = f"{type(exc).__name__}: {exc}"
            self._last_model_reason = self._model_load_error
            self._set_debug(
                model_ready=False,
                model_load_error=self._model_load_error,
                load_elapsed_sec=round(time.perf_counter() - started, 3),
            )

    def triage(
        self,
        text_description: str,
        patient_age: int | None = None,
        patient_weight_kg: float | None = None,
    ) -> TriageResult:
        started = time.perf_counter()
        facts = self._extract_features(text_description)
        severity, used_model = self._resolve_severity(text_description, facts)
        self._set_debug(
            used_model_inference=used_model,
            model_reason=self._last_model_reason,
            selected_severity=severity,
        )

        protocol = self.tool_executor.execute(
            "lookup_treatment_protocol",
            {"condition": facts["condition_guess"], "setting": "community_health_post"},
        )
        protocol_result = protocol.get("result", {}) if protocol.get("ok") else {}

        actions = protocol_result.get("immediate_actions", []) + protocol_result.get("treatment_steps", [])
        if not actions:
            actions = ["Stabilize patient", "Monitor and escalate if worsening"]

        advice = self._build_advice(severity)

        referral_letter = None
        if severity in {"REFER_URGENT", "EMERGENCY", "REFER_ROUTINE"}:
            letter_resp = self.tool_executor.execute(
                "generate_referral_letter",
                {
                    "patient_name": "Unknown",
                    "finding": facts["primary_symptom"],
                    "severity": severity,
                },
            )
            if letter_resp.get("ok"):
                referral_letter = letter_resp["result"].get("letter")

        if patient_age is not None and patient_weight_kg is not None and patient_age < 18:
            dose_resp = self.tool_executor.execute(
                "calculate_pediatric_dose",
                {
                    "medication_name": "paracetamol",
                    "patient_weight_kg": patient_weight_kg,
                    "patient_age_years": patient_age,
                },
            )
            if dose_resp.get("ok"):
                dose = dose_resp["result"]
                actions.append(
                    f"Pediatric dose guide: paracetamol {dose.get('dose_mg')} mg {dose.get('frequency')}"
                )

        result = TriageResult(
            severity=SeverityLevel(severity),
            primary_concern=facts["primary_symptom"],
            recommended_actions=actions[:8],
            red_flags=facts["red_flags"],
            local_advice=advice,
            referral_letter=referral_letter,
            confidence="HIGH" if used_model else "MEDIUM",
        )
        self._set_debug(
            triage_elapsed_sec=round(time.perf_counter() - started, 3),
            confidence=result.confidence,
        )
        return result

    def _resolve_severity(self, text_description: str, facts: dict[str, Any]) -> tuple[str, bool]:
        model_severity = self._try_model_severity(text_description)
        if model_severity is not None:
            self._last_model_reason = "model_label_used"
            return model_severity, True

        severity_eval = self.tool_executor.execute(
            "assess_triage_severity",
            {
                "primary_symptom": facts["primary_symptom"],
                "duration_hours": facts["duration_hours"],
                "consciousness_level": facts["consciousness_level"],
                "breathing_difficulty": facts["breathing_difficulty"],
                "bleeding_severity": facts["bleeding_severity"],
            },
        )
        if not severity_eval.get("ok"):
            self._last_model_reason = self._last_model_reason or "tool_eval_failed"
            return SeverityLevel.REFER_URGENT.value, False
        self._last_model_reason = self._last_model_reason or "fallback_heuristic"
        return severity_eval["result"]["severity_level"], False

    def _try_model_severity(self, text_description: str) -> str | None:
        if not settings.enable_model_inference:
            self._last_model_reason = "model_inference_disabled"
            return None

        if settings.use_ollama:
            return self._try_ollama_severity(text_description)

        self.load()
        if not self._model_ready or self._model is None or self._processor is None:
            self._last_model_reason = self._model_load_error or "model_not_ready"
            return None

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Classify triage severity. Return only one label from: "
                    "SELF_CARE, MONITOR_48H, REFER_ROUTINE, REFER_URGENT, EMERGENCY.\n\n"
                    f"Case: {text_description}"
                ),
            },
        ]

        try:
            import torch
            gen_started = time.perf_counter()

            prompt_text = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self._processor(text=prompt_text, return_tensors="pt")
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=12,
                    do_sample=False,
                    max_time=settings.model_inference_timeout_sec,
                    pad_token_id=self._processor.tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            output_text = self._processor.decode(new_tokens, skip_special_tokens=True).upper()
            self._set_debug(model_generate_elapsed_sec=round(time.perf_counter() - gen_started, 3))
        except Exception:
            self._last_model_reason = "model_generate_failed"
            return None

        for label in [
            SeverityLevel.EMERGENCY.value,
            SeverityLevel.REFER_URGENT.value,
            SeverityLevel.REFER_ROUTINE.value,
            SeverityLevel.MONITOR_48H.value,
            SeverityLevel.SELF_CARE.value,
        ]:
            if re.search(rf"\b{label}\b", output_text):
                return label
        self._last_model_reason = "model_label_parse_failed"
        return None

    def _try_ollama_severity(self, text_description: str) -> str | None:
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "Classify triage severity. Return only one label from: "
            "SELF_CARE, MONITOR_48H, REFER_ROUTINE, REFER_URGENT, EMERGENCY.\n\n"
            f"Case: {text_description}\n"
            "Label:"
        )

        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }
        req = urllib.request.Request(
            url=f"{settings.ollama_base_url.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=settings.ollama_timeout_sec) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            self._last_model_reason = f"ollama_unreachable: {exc}"
            self._set_debug(
                ollama_elapsed_sec=round(time.perf_counter() - started, 3),
                ollama_error=str(exc),
            )
            return None
        except Exception as exc:
            self._last_model_reason = f"ollama_failed: {type(exc).__name__}: {exc}"
            self._set_debug(
                ollama_elapsed_sec=round(time.perf_counter() - started, 3),
                ollama_error=str(exc),
            )
            return None

        self._set_debug(ollama_elapsed_sec=round(time.perf_counter() - started, 3))
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            self._last_model_reason = "ollama_invalid_json"
            return None

        output_text = str(parsed.get("response", "")).upper()
        for label in [
            SeverityLevel.EMERGENCY.value,
            SeverityLevel.REFER_URGENT.value,
            SeverityLevel.REFER_ROUTINE.value,
            SeverityLevel.MONITOR_48H.value,
            SeverityLevel.SELF_CARE.value,
        ]:
            if re.search(rf"\b{label}\b", output_text):
                self._last_model_reason = "ollama_label_used"
                return label

        self._last_model_reason = "ollama_label_parse_failed"
        return None

    def _build_advice(self, severity: str) -> str:
        mapping = {
            "SELF_CARE": "Home care is reasonable now; return if symptoms worsen.",
            "MONITOR_48H": "Monitor for 48 hours and reassess sooner if any red flags appear.",
            "REFER_ROUTINE": "Arrange clinic review within 24 hours.",
            "REFER_URGENT": "Refer urgently to a higher facility today.",
            "EMERGENCY": "Activate emergency referral immediately.",
        }
        return mapping.get(severity, "Seek clinical review.")

    def _extract_features(self, text: str) -> dict[str, Any]:
        lower = text.lower()

        duration_hours = 24
        match = re.search(r"(\d+)\s*(hour|hours|day|days)", lower)
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            duration_hours = value * 24 if unit.startswith("day") else value

        consciousness_level = "alert"
        if any(k in lower for k in ["unconscious", "unresponsive"]):
            consciousness_level = "unresponsive"
        elif "drowsy" in lower:
            consciousness_level = "voice"

        breathing_difficulty = any(
            k in lower
            for k in [
                "difficulty breathing",
                "shortness of breath",
                "laboured breathing",
                "cannot breathe",
                "fast breathing",
                "chest indrawing",
                "speak single words",
            ]
        )

        bleeding_severity = "none"
        if "life-threatening bleeding" in lower or "soaking" in lower:
            bleeding_severity = "life-threatening"
        elif "severe bleeding" in lower:
            bleeding_severity = "severe"
        elif "bleeding stopped" in lower or "has stopped bleeding" in lower:
            bleeding_severity = "minor"
        elif "bleeding" in lower:
            bleeding_severity = "moderate"

        condition_guess = "laceration" if any(k in lower for k in ["cut", "laceration", "wound"]) else "fever_child"
        if "diarrhoea" in lower or "diarrhea" in lower:
            condition_guess = "diarrhoea_acute"
        elif "pneumonia" in lower or "fast breathing" in lower:
            condition_guess = "pneumonia_child"
        elif "malaria" in lower:
            condition_guess = "malaria_uncomplicated"

        red_flags = []
        for token in [
            "seizure",
            "stroke",
            "chest pain",
            "unconscious",
            "pregnant",
            "anaphylaxis",
            "unable to drink",
            "chest indrawing",
            "throat swelling",
            "very sleepy",
        ]:
            if token in lower:
                red_flags.append(token)

        primary_symptom = text.strip()[:500] if text.strip() else "unspecified concern"

        return {
            "primary_symptom": primary_symptom,
            "duration_hours": duration_hours,
            "consciousness_level": consciousness_level,
            "breathing_difficulty": breathing_difficulty,
            "bleeding_severity": bleeding_severity,
            "condition_guess": condition_guess,
            "red_flags": red_flags,
        }
