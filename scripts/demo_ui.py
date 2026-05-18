"""Quick Gradio UI for VoiceMed triage demo."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import speech_recognition as sr
from PIL import Image, ImageStat
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, HRFlowable, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from voicemed.engine.model import Gemma4TriageEngine  # noqa: E402
from voicemed.config import settings  # noqa: E402

# ── Language support ────────────────────────────────────────────────────────
LANGUAGES = {
    "English":                "en",
    "Kiswahili (Swahili)":    "sw",
    "Français (French)":      "fr",
    "हिंदी (Hindi)":           "hi",
    "العربية (Arabic)":        "ar",
}
SPEECH_LANG = {"en": "en-US", "sw": "sw-KE", "fr": "fr-FR", "hi": "hi-IN", "ar": "ar-SA"}

SEVERITY_LABELS: dict[str, dict[str, str]] = {
    "en": {"EMERGENCY": "EMERGENCY", "REFER_URGENT": "URGENT REFERRAL",
           "REFER_ROUTINE": "ROUTINE REFERRAL", "MONITOR_48H": "MONITOR 48H", "SELF_CARE": "SELF-CARE"},
    "sw": {"EMERGENCY": "DHARURA", "REFER_URGENT": "RUFAA YA HARAKA",
           "REFER_ROUTINE": "RUFAA YA KAWAIDA", "MONITOR_48H": "FUATILIA MASAA 48", "SELF_CARE": "HUDUMA YA NYUMBANI"},
    "fr": {"EMERGENCY": "URGENCE", "REFER_URGENT": "RÉFÉRENCE URGENTE",
           "REFER_ROUTINE": "RÉFÉRENCE ROUTINIÈRE", "MONITOR_48H": "SURVEILLER 48H", "SELF_CARE": "SOINS À DOMICILE"},
    "hi": {"EMERGENCY": "आपातकाल", "REFER_URGENT": "तत्काल रेफरल",
           "REFER_ROUTINE": "नियमित रेफरल", "MONITOR_48H": "48 घंटे निगरानी", "SELF_CARE": "घरेलू देखभाल"},
    "ar": {"EMERGENCY": "طارئ", "REFER_URGENT": "إحالة عاجلة",
           "REFER_ROUTINE": "إحالة روتينية", "MONITOR_48H": "مراقبة 48 ساعة", "SELF_CARE": "رعاية منزلية"},
}
UI_LABELS: dict[str, dict[str, str]] = {
    "en": {"primary": "Primary concern", "actions": "Recommended actions",
           "advice": "Patient advice", "referral": "Referral letter generated", "yes": "Yes"},
    "sw": {"primary": "Tatizo kuu", "actions": "Hatua zinazopendekezwa",
           "advice": "Ushauri kwa mgonjwa", "referral": "Barua ya rufaa imetolewa", "yes": "Ndiyo"},
    "fr": {"primary": "Préoccupation principale", "actions": "Actions recommandées",
           "advice": "Conseils au patient", "referral": "Lettre de référence générée", "yes": "Oui"},
    "hi": {"primary": "मुख्य चिंता", "actions": "अनुशंसित कार्रवाई",
           "advice": "रोगी को सलाह", "referral": "रेफरल पत्र तैयार", "yes": "हाँ"},
    "ar": {"primary": "المخاوف الرئيسية", "actions": "الإجراءات الموصى بها",
           "advice": "نصيحة للمريض", "referral": "تم إنشاء خطاب الإحالة", "yes": "نعم"},
}
ADVICE_TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "Home care is reasonable now; return if symptoms worsen.":
            "Home care is reasonable now; return if symptoms worsen.",
        "Monitor for 48 hours and reassess sooner if any red flags appear.":
            "Monitor for 48 hours and reassess sooner if any red flags appear.",
        "Arrange clinic review within 24 hours.": "Arrange clinic review within 24 hours.",
        "Refer urgently to a higher facility today.": "Refer urgently to a higher facility today.",
        "Activate emergency referral immediately.": "Activate emergency referral immediately.",
        "Seek clinical review.": "Seek clinical review.",
    },
    "sw": {
        "Home care is reasonable now; return if symptoms worsen.":
            "Huduma ya nyumbani inafaa; rudi ikiwa dalili zitazidi.",
        "Monitor for 48 hours and reassess sooner if any red flags appear.":
            "Fuatilia masaa 48 na tathmini upya ikiwa dalili za hatari zitatokea.",
        "Arrange clinic review within 24 hours.": "Panga ziara ya kliniki ndani ya masaa 24.",
        "Refer urgently to a higher facility today.": "Peleka kwa haraka kwenye kituo cha juu leo.",
        "Activate emergency referral immediately.": "Anzisha rufaa ya dharura mara moja.",
        "Seek clinical review.": "Tafuta tathmini ya kimatibabu.",
    },
    "fr": {
        "Home care is reasonable now; return if symptoms worsen.":
            "Les soins à domicile sont raisonnables; revenez si les symptômes s'aggravent.",
        "Monitor for 48 hours and reassess sooner if any red flags appear.":
            "Surveiller 48 heures et réévaluer si des signaux d'alarme apparaissent.",
        "Arrange clinic review within 24 hours.": "Organiser une consultation dans les 24 heures.",
        "Refer urgently to a higher facility today.": "Référer d'urgence vers un établissement supérieur aujourd'hui.",
        "Activate emergency referral immediately.": "Activer la référence d'urgence immédiatement.",
        "Seek clinical review.": "Consulter un médecin.",
    },
    "hi": {
        "Home care is reasonable now; return if symptoms worsen.":
            "अभी घरेलू देखभाल उचित है; लक्षण बिगड़ें तो वापस आएं।",
        "Monitor for 48 hours and reassess sooner if any red flags appear.":
            "48 घंटे निगरानी करें; खतरे के संकेत दिखें तो पहले पुनर्मूल्यांकन करें।",
        "Arrange clinic review within 24 hours.": "24 घंटों में क्लिनिक जांच की व्यवस्था करें।",
        "Refer urgently to a higher facility today.": "आज उच्च केंद्र में तत्काल रेफर करें।",
        "Activate emergency referral immediately.": "तुरंत आपातकालीन रेफरल सक्रिय करें।",
        "Seek clinical review.": "चिकित्सीय समीक्षा लें।",
    },
    "ar": {
        "Home care is reasonable now; return if symptoms worsen.":
            "الرعاية المنزلية مناسبة الآن؛ عد إذا تفاقمت الأعراض.",
        "Monitor for 48 hours and reassess sooner if any red flags appear.":
            "راقب 48 ساعة وأعد التقييم مبكراً إذا ظهرت علامات تحذيرية.",
        "Arrange clinic review within 24 hours.": "رتب مراجعة في العيادة خلال 24 ساعة.",
        "Refer urgently to a higher facility today.": "أحل بشكل عاجل إلى مرفق أعلى اليوم.",
        "Activate emergency referral immediately.": "فعّل الإحالة الطارئة على الفور.",
        "Seek clinical review.": "اطلب مراجعة سريرية.",
    },
}


ENGINE = Gemma4TriageEngine()
SESSION_LOG_PATH = PROJECT_ROOT / "evaluation_results" / "demo_session_log.jsonl"
REFERRAL_DIR = PROJECT_ROOT / "evaluation_results" / "referrals"

_LANG_NAMES = {"sw": "Swahili", "fr": "French", "hi": "Hindi", "ar": "Arabic"}


def _translate_via_ollama(items: list[str], lang: str) -> list[str]:
    """Translate a list of strings using Gemma via Ollama. Returns originals on failure."""
    if lang == "en" or not items or not settings.use_ollama:
        return items
    lang_name = _LANG_NAMES.get(lang, "English")
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items))
    prompt = (
        f"Translate the following medical phrases to {lang_name}. "
        "Return ONLY the translations, numbered the same way, nothing else.\n\n"
        + numbered
    )
    payload = json.dumps({
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }).encode("utf-8")
    req = urllib.request.Request(
        url=f"{settings.ollama_base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        raw = body.get("response", "").strip()
        # Parse numbered lines back out
        import re as _re2
        translated = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip leading "1. " / "2. " etc.
            cleaned = _re2.sub(r"^\d+\.\s*", "", line)
            if cleaned:
                translated.append(cleaned)
        # Fall back to originals for any missing lines
        if len(translated) == len(items):
            return translated
        # Partial — pad with originals
        return translated + items[len(translated):]
    except Exception as _exc:
        import sys
        print(f"[VoiceMed] Translation error: {_exc}", file=sys.stderr)
        return items


def backend_status_html() -> str:
    if not settings.enable_model_inference:
        return (
            "<div style='padding:10px 12px;border-radius:10px;background:#fff3bf;color:#5f3f00;'>"
            "<b>Backend:</b> Offline heuristic engine (Gemma model inference disabled). "
            "Set ENABLE_MODEL_INFERENCE=true to attempt model-backed inference."
            "</div>"
        )

    if settings.use_ollama:
        return (
            "<div style='padding:10px 12px;border-radius:10px;background:#d0ebff;color:#0b3d91;'>"
            "<b>Backend:</b> Ollama mode enabled. "
            f"Model: {settings.ollama_model} at {settings.ollama_base_url}."
            "</div>"
        )

    ENGINE.load()
    model_ready = getattr(ENGINE, "_model_ready", False)
    if model_ready:
        return (
            "<div style='padding:10px 12px;border-radius:10px;background:#d3f9d8;color:#0b5d1e;'>"
            "<b>Backend:</b> Gemma model inference enabled and loaded."
            "</div>"
        )

    model_error = getattr(ENGINE, "_model_load_error", None)
    details = f"<br><small>{model_error}</small>" if model_error else ""
    return (
        "<div style='padding:10px 12px;border-radius:10px;background:#ffe3e3;color:#7a1a1a;'>"
        "<b>Backend:</b> ENABLE_MODEL_INFERENCE=true but model did not load. "
        "Using safe offline heuristic fallback."
        f"{details}"
        "</div>"
    )


def _severity_badge(severity: str, lang: str = "en") -> str:
    color = {
        "EMERGENCY": "#c92a2a",
        "REFER_URGENT": "#d9480f",
        "REFER_ROUTINE": "#e67700",
        "MONITOR_48H": "#2b8a3e",
        "SELF_CARE": "#1971c2",
    }.get(severity, "#495057")
    label = SEVERITY_LABELS.get(lang, SEVERITY_LABELS["en"]).get(severity, severity)
    return (
        "<div style='padding:10px 14px;border-radius:10px;"
        f"background:{color};color:white;font-weight:700;display:inline-block'>"
        f"⚕ {label}</div>"
    )


def _append_session_log(
    text: str,
    age: int | None,
    weight: float | None,
    payload: dict,
    image_path: str | None,
    audio_path: str | None,
) -> None:
    SESSION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "text": text,
            "age": age,
            "weight": weight,
            "has_image": bool(image_path),
            "has_audio": bool(audio_path),
        },
        "output": payload,
    }
    with SESSION_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


SEVERITY_COLORS = {
    "EMERGENCY":    colors.HexColor("#c92a2a"),
    "REFER_URGENT": colors.HexColor("#d9480f"),
    "REFER_ROUTINE":colors.HexColor("#e67700"),
    "MONITOR_48H":  colors.HexColor("#2b8a3e"),
    "SELF_CARE":    colors.HexColor("#1971c2"),
}


def _build_referral_pdf(
    referral_letter: str | None,
    data: dict,
    image_path: str | None,
    patient_name: str = "Unknown",
) -> str | None:
    if not referral_letter:
        return None
    REFERRAL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    pdf_path = REFERRAL_DIR / f"referral-{ts}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    severity = data.get("severity", "UNKNOWN")
    sev_color = SEVERITY_COLORS.get(severity, colors.grey)

    title_style = ParagraphStyle(
        "title",
        parent=styles["Heading1"],
        textColor=colors.HexColor("#0d3b3e"),
        fontSize=18,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "subtitle",
        parent=styles["Normal"],
        textColor=colors.grey,
        fontSize=9,
        spaceAfter=12,
    )
    body_style = ParagraphStyle(
        "body",
        parent=styles["Normal"],
        fontSize=10,
        leading=16,
        spaceAfter=8,
    )
    label_style = ParagraphStyle(
        "label",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
        spaceAfter=2,
    )

    story = []

    # ── Header ────────────────────────────────────────────────────
    story.append(Paragraph("🩺 VoiceMed — Clinical Referral Letter", title_style))
    story.append(Paragraph(
        f"Patient: <b>{patient_name}</b> &nbsp;&nbsp;·&nbsp;&nbsp;"
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ·  "
        "AI-assisted triage · For clinician review",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0d3b3e")))
    story.append(Spacer(1, 0.4 * cm))

    # ── Severity badge table ───────────────────────────────────────
    badge_data = [[Paragraph(
        f"<b>SEVERITY: {severity}</b>",
        ParagraphStyle("badge", fontSize=13, textColor=colors.white),
    )]]
    badge_table = Table(badge_data, colWidths=[doc.width])
    badge_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), sev_color),
        ("ROUNDEDCORNERS", [8]),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(badge_table)
    story.append(Spacer(1, 0.5 * cm))

    # ── Clinical image (if provided) ───────────────────────────────
    if image_path:
        try:
            img = Image.open(image_path)
            max_w, max_h = 12 * cm, 8 * cm
            ratio = min(max_w / img.width, max_h / img.height)
            rl_img = RLImage(image_path, width=img.width * ratio, height=img.height * ratio)
            story.append(Paragraph("Clinical Image", label_style))
            story.append(rl_img)
            story.append(Spacer(1, 0.4 * cm))
        except Exception:
            pass

    # ── Letter body ────────────────────────────────────────────────
    story.append(Paragraph("Referral Details", label_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 0.2 * cm))
    for line in referral_letter.splitlines():
        if line.strip():
            story.append(Paragraph(line, body_style))

    story.append(Spacer(1, 0.5 * cm))

    # ── Recommended actions ────────────────────────────────────────
    actions = data.get("recommended_actions", [])
    if actions:
        story.append(Paragraph("Recommended Actions", label_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Spacer(1, 0.2 * cm))
        for action in actions[:6]:
            story.append(Paragraph(f"• {action}", body_style))

    # ── Red flags ─────────────────────────────────────────────────
    red_flags = data.get("red_flags", [])
    if red_flags:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("Red Flags", label_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Spacer(1, 0.2 * cm))
        flag_text = ",  ".join(f"⚠ {f}" for f in red_flags)
        story.append(Paragraph(flag_text, ParagraphStyle(
            "flags", parent=body_style, textColor=colors.HexColor("#c92a2a")
        )))

    # ── Footer ────────────────────────────────────────────────────
    story.append(Spacer(1, 0.8 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Paragraph(
        "This referral was generated by VoiceMed AI triage assistant. "
        "It is intended to support — not replace — clinical judgment. "
        "A qualified clinician must review and confirm all decisions.",
        ParagraphStyle("footer", parent=styles["Normal"], fontSize=8,
                       textColor=colors.grey, leading=11),
    ))

    doc.build(story)
    return str(pdf_path)


def _build_summary(data: dict) -> str:
    actions = data.get("recommended_actions", [])
    top_actions = "\n".join(f"- {a}" for a in actions[:3]) if actions else "- No actions generated"
    return (
        f"### Quick Summary\n"
        f"- Severity: **{data.get('severity', 'UNKNOWN')}**\n"
        f"- Confidence: **{data.get('confidence', 'UNKNOWN')}**\n"
        f"- Primary concern: {data.get('primary_concern', '')}\n"
        f"- Top actions:\n{top_actions}"
    )


def _image_context(image_path: str | None) -> str:
    if not image_path:
        return ""
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            brightness = round(stat.mean[0], 1)
            return (
                "\n\n[Image attached] "
                f"size={img.width}x{img.height}, brightness_mean={brightness}. "
                "Use image as supportive context only; prioritize clinical danger signs."
            )
    except Exception:
        return "\n\n[Image attached but unreadable]"


import re as _re


def _parse_voice_fields(transcript: str) -> tuple[str, str, int | None, float | None]:
    """Extract name / age / weight from transcript; return (symptoms, name, age, weight)."""
    t = transcript.strip()
    name: str = ""
    age: int | None = None
    weight: float | None = None

    # ── Name: flexible patterns for natural speech ────────────────
    # Handles: "patient name is Mary", "name is Mary", "my name is Mary",
    #          "patient Mary", "for patient Mary Achieng", "called Mary"
    # Stop words prevent absorbing "age is" / "weight" etc. into the name
    _STOP = r"(?:age[d]?|weight|is\s|has|the|and|with|having|weeks?|years?|kg|percent|patient)"
    name_match = _re.search(
        r"(?:"
        r"(?:(?:the\s+)?patient(?:'s)?\s+name\s+is|name\s+is|my\s+name\s+is|patient\s+name\s*:?)\s+"
        r"|(?:for\s+patient|patient)\s+"
        r"|called\s+"
        r")([A-Za-z]+(?:\s+(?!" + _STOP + r")[A-Za-z]+){0,2})",
        t,
        _re.IGNORECASE,
    )
    if name_match:
        raw_name = name_match.group(1).strip()
        # Reject single stop/filler words that aren't names
        if raw_name.lower() not in {"a", "an", "the", "is", "has", "she", "he", "age", "aged"}:
            name = raw_name.title()

    # ── Age: flexible patterns ────────────────────────────────────
    # Handles: "32 years old", "aged 32", "age 32", "she is 32 years",
    #          "35-year-old", "35 yr old"
    age_match = _re.search(
        r"(\d{1,3})\s*[\-\s]?(?:year[s]?[\-\s]old|yr[s]?[\-\s]old|years?\s+old)|"  # 32 years old
        r"(?:aged?|age\s+is|is\s+aged?)\s+(\d{1,3})|"  # aged 32 / age is 32
        r"(\d{1,3})\s+(?:year[s]?|yr[s]?)\s+(?:old|of\s+age)",  # 32 years of age
        t, _re.IGNORECASE,
    )
    if age_match:
        raw_age = next(g for g in age_match.groups() if g is not None)
        v = int(raw_age)
        if 0 < v <= 120:
            age = v

    # ── Weight: flexible patterns ─────────────────────────────────
    # Handles: "65 kg", "weighs 65", "65 kilos", "weight is 65"
    weight_match = _re.search(
        r"(\d{1,3}(?:\.\d)?)\s*(?:kg|kgs|kilos?|kilograms?)|"  # 65 kg
        r"(?:weigh[s]?|weight\s+(?:is|of))\s+(\d{1,3}(?:\.\d)?)|"  # weighs 65
        r"(?:weighing)\s+(\d{1,3}(?:\.\d)?)",  # weighing 65
        t, _re.IGNORECASE,
    )
    if weight_match:
        raw_w = next(g for g in weight_match.groups() if g is not None)
        w = float(raw_w)
        if 0 < w < 300:
            weight = w

    # ── Strip extracted metadata fragments, keep clinical symptoms ─
    symptoms = t
    # Remove name fragment
    if name:
        symptoms = _re.sub(
            r"(?:(?:the\s+)?patient(?:'s)?\s+name\s+is|name\s+is|my\s+name\s+is|patient\s+name\s*:?|for\s+patient|called)\s+" + _re.escape(name),
            "", symptoms, flags=_re.IGNORECASE,
        )
        # Also try removing bare name if it appears near the start
        symptoms = _re.sub(r"^\s*" + _re.escape(name) + r"\s*[,\.\s]", "", symptoms, flags=_re.IGNORECASE)
    # Remove age fragment
    if age:
        symptoms = _re.sub(
            r"\b" + str(age) + r"\s*[\-\s]?(?:year[s]?[\-\s]old|yr[s]?[\-\s]old|years?\s+old|year[s]?|yr[s]?)",
            "", symptoms, flags=_re.IGNORECASE,
        )
        symptoms = _re.sub(r"(?:aged?|age\s+is)\s+" + str(age), "", symptoms, flags=_re.IGNORECASE)
    # Remove weight fragment
    if weight:
        wstr = str(int(weight)) if weight == int(weight) else str(weight)
        symptoms = _re.sub(
            r"\b" + wstr + r"\s*(?:kg|kgs|kilos?|kilograms?)",
            "", symptoms, flags=_re.IGNORECASE,
        )
        symptoms = _re.sub(r"(?:weigh[s]?|weight\s+(?:is|of)|weighing)\s+" + wstr, "", symptoms, flags=_re.IGNORECASE)
    # Tidy up filler words and punctuation
    symptoms = _re.sub(r"\b(?:she|he)\s+is\b", "", symptoms, flags=_re.IGNORECASE)
    symptoms = _re.sub(r"\s*,\s*,", ",", symptoms)
    symptoms = _re.sub(r"^[\s,\.and]+|[\s,\.]+$", "", symptoms)
    symptoms = _re.sub(r"\s{2,}", " ", symptoms)

    return symptoms, name, age, weight


def transcribe_audio_to_text(
    audio_path: str | None,
    current_text: str | None,
) -> tuple[str, str, int | None, float | None, str]:
    """Transcribe audio and extract name/age/weight into separate fields."""
    base = (current_text or "").strip()
    if not audio_path:
        return base, "", None, None, "No audio file provided."

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        transcript = recognizer.recognize_google(audio_data)
        symptoms, name, age, weight = _parse_voice_fields(transcript)
        merged = f"{base}\n{symptoms}".strip() if base else symptoms
        status_parts = ["✅ Transcription done"]
        if name:
            status_parts.append(f"name: **{name}**")
        if age is not None:
            status_parts.append(f"age: **{age}**")
        if weight is not None:
            status_parts.append(f"weight: **{weight} kg**")
        # Use gr.update() for undetected fields so existing values aren't overwritten
        return (
            merged,
            gr.update(value=name) if name else gr.update(),
            gr.update(value=age) if age is not None else gr.update(),
            gr.update(value=weight) if weight is not None else gr.update(),
            "  ·  ".join(status_parts),
        )
    except sr.UnknownValueError:
        return base, gr.update(), gr.update(), gr.update(), "Could not understand the audio."
    except sr.RequestError as exc:
        return base, gr.update(), gr.update(), gr.update(), f"⚠️ Speech service unavailable (internet required for voice transcription): {exc}"
    except Exception as exc:
        return base, gr.update(), gr.update(), gr.update(), f"Transcription failed: {exc}"


def assess_case(
    text: str,
    patient_name: str | None,
    age: float | None,
    weight: float | None,
    image_path: str | None,
    audio_path: str | None,
    language: str = "English",
) -> tuple[str, dict, str, str | None]:
    lang = LANGUAGES.get(language or "English", "en")
    lbl = UI_LABELS.get(lang, UI_LABELS["en"])
    adv_map = ADVICE_TRANSLATIONS.get(lang, ADVICE_TRANSLATIONS["en"])
    speech_lang_code = SPEECH_LANG.get(lang, "en-US")

    if not text or not text.strip():
        return (
            "<b>Please enter a patient description.</b>",
            {"ok": False, "error": "empty_text"},
            "### Quick Summary\n- No input provided",
            None,
        )

    parsed_age = int(age) if age is not None else None
    parsed_weight = float(weight) if weight is not None else None
    final_text = text.strip() + _image_context(image_path)
    name = (patient_name or "").strip() or "Unknown"

    result = ENGINE.triage(
        text_description=final_text,
        patient_age=parsed_age,
        patient_weight_kg=parsed_weight,
    )
    data = result.to_dict()
    data["_debug"] = ENGINE.get_debug_snapshot()

    # Substitute patient name in the referral letter generated by the engine
    if data.get("referral_letter"):
        data["referral_letter"] = data["referral_letter"].replace(
            "Patient: Unknown", f"Patient: {name}"
        )

    try:
        _append_session_log(final_text, parsed_age, parsed_weight, data, image_path, audio_path)
    except Exception as exc:
        data.setdefault("_debug", {})["log_error"] = str(exc)

    referral_file = _build_referral_pdf(data.get("referral_letter"), data, image_path, name)
    summary_md = _build_summary(data)

    severity = data["severity"]
    local_advice_en = data.get("local_advice", "")
    local_advice = adv_map.get(local_advice_en, local_advice_en)
    actions = data.get("recommended_actions", [])
    primary_concern = data["primary_concern"]

    # ── Translate dynamic content via Gemma/Ollama ────────────────
    if lang != "en":
        to_translate = [primary_concern] + actions
        translated = _translate_via_ollama(to_translate, lang)
        primary_concern = translated[0] if translated else primary_concern
        actions = translated[1:] if len(translated) > 1 else actions

    C = "color:#e8f4f5"  # light text on dark card
    html = [
        _severity_badge(severity, lang),
        f"<p style='{C}'><b>{lbl['primary']}:</b> {primary_concern}</p>",
        f"<p style='{C}'><b>{lbl['actions']}:</b></p><ul>",
    ]
    for action in actions:
        html.append(f"<li style='{C}'>{action}</li>")
    html.append("</ul>")
    html.append(f"<p style='{C}'><b>{lbl['advice']}:</b> {local_advice}</p>")
    if data.get("referral_letter"):
        html.append(f"<p style='{C}'><b>{lbl['referral']}:</b> {lbl['yes']}</p>")

    # ── Embed TTS data as a hidden span (DOMPurify keeps id + data-*) ──
    import html as _html_enc
    speak_parts = [
        SEVERITY_LABELS.get(lang, SEVERITY_LABELS["en"]).get(severity, severity),
        primary_concern,
        local_advice,
    ] + actions[:3]
    speak_text = ". ".join(p for p in speak_parts if p)
    html.append(
        f'<span id="vm-tts-data" '
        f'data-text="{_html_enc.escape(speak_text, quote=True)}" '
        f'data-lang="{speech_lang_code}" '
        f'style="position:absolute;width:0;height:0;overflow:hidden"></span>'
    )

    return "\n".join(html), data, summary_md, referral_file


MOBILE_CSS = """
/* ── Mobile-first base ───────────────────────────────── */
body, .gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}
.gradio-container > .main {
    padding: 8px !important;
}

/* ── Header ──────────────────────────────────────────── */
.vm-header {
    background: linear-gradient(135deg, #0d3b3e 0%, #1a6b6e 100%);
    color: white;
    border-radius: 14px;
    padding: 16px 18px 12px;
    margin-bottom: 12px;
}
.vm-header h1 {
    margin: 0 0 4px;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.vm-header p {
    margin: 0;
    font-size: 0.82rem;
    opacity: 0.85;
}

/* ── Section cards ───────────────────────────────────── */
.vm-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 10px;
    border: 1px solid #2a2a3e;
}
/* ── Ensure result text is always readable on dark card ── */
.vm-card p, .vm-card li, .vm-card b, .vm-card strong, .vm-card label {
    color: #e8f4f5;
}

/* ── Big tap-friendly buttons ────────────────────────── */
#transcribe-btn, #assess-btn {
    min-height: 52px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    width: 100% !important;
    margin-top: 6px !important;
}
#assess-btn {
    background: #1a6b6e !important;
    font-size: 1.1rem !important;
    min-height: 58px !important;
}

/* ── Severity result card ────────────────────────────── */
.vm-result {
    border-radius: 12px;
    padding: 14px;
    margin-top: 4px;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* ── Examples table — horizontal scroll on small screens */
.gr-examples table {
    font-size: 0.78rem;
    display: block;
    overflow-x: auto;
}

/* ── Inputs: larger touch area ───────────────────────── */
input, textarea, select {
    font-size: 16px !important;   /* prevents iOS zoom on focus */
}

/* ── Hide raw JSON on mobile by default (collapsible) ── */
@media (max-width: 600px) {
    .vm-raw { display: none; }
    .vm-raw.open { display: block; }
}

/* ── Compact number fields ───────────────────────────── */
.vm-row-numbers .gr-form { gap: 8px !important; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="VoiceMed — AI Triage") as app:

        # ── Header ────────────────────────────────────────────
        gr.HTML("""
        <div class="vm-header">
          <h1>🩺 VoiceMed</h1>
          <p>AI-powered clinical triage &nbsp;·&nbsp; Voice &amp; Vision &nbsp;·&nbsp; Offline-first inference</p>
        </div>
        """)

        gr.HTML(backend_status_html())

        # ── Language selector ─────────────────────────────────
        with gr.Row():
            language = gr.Dropdown(
                label="🌍 Language / Lugha / Langue",
                choices=list(LANGUAGES.keys()),
                value="English",
                scale=1,
            )
        with gr.Column(elem_classes="vm-card"):
            patient_name = gr.Textbox(
                label="👤 Patient name (optional)",
                lines=1,
                placeholder="e.g. Mary Achieng — appears on the referral letter",
            )
            text = gr.Textbox(
                label="📋 Clinical description",
                lines=4,
                max_lines=8,
                placeholder="Describe the patient's symptoms, e.g.:\nAdult with chest pain and shortness of breath for 1 hour",
            )

            with gr.Row(elem_classes="vm-row-numbers"):
                age = gr.Number(label="Age (years)", precision=0, maximum=120, scale=1, value=None)
                weight = gr.Number(label="Weight (kg)", maximum=300, scale=1, value=None)

        # ── Image + Voice side by side on wide, stacked on mobile
        with gr.Column(elem_classes="vm-card"):
            image_input = gr.Image(
                label="📷 Clinical image (optional)",
                type="filepath",
                height=180,
            )

        with gr.Column(elem_classes="vm-card"):
            audio_input = gr.Audio(
                label="🎤 Voice note (optional) — requires internet for speech recognition",
                sources=["microphone", "upload"],
                type="filepath",
            )
            transcribe_btn = gr.Button(
                "🔊 Transcribe Voice to Text",
                elem_id="transcribe-btn",
                variant="secondary",
            )
            gr.HTML(
                "<p style='font-size:0.78rem;color:#a0c4c6;margin:4px 0 0;line-height:1.4;'>"
                "🌐 Voice transcription uses Google Speech-to-Text <em>(requires internet)</em>. "
                "On-device speech is a planned feature for a future Gemma release."
                "</p>"
            )
            transcribe_status = gr.Markdown("", label="")

        submit = gr.Button("Assess Patient →", variant="primary", elem_id="assess-btn")

        # ── Results ───────────────────────────────────────────
        with gr.Column(elem_classes="vm-card"):
            summary = gr.HTML(label="Assessment Result")
            quick_summary = gr.Markdown(label="")
            referral_file = gr.File(label="📄 Referral Letter (PDF)")
            with gr.Row():
                read_aloud_btn = gr.Button("🔊 Read Aloud", variant="secondary", scale=1)
                stop_btn = gr.Button("⏹ Stop", variant="secondary", scale=1)


        # ── Raw JSON (collapsed on mobile) ────────────────────
        with gr.Accordion("🔧 Raw JSON output", open=False):
            raw = gr.JSON(label="")

        # ── Examples ──────────────────────────────────────────
        gr.Markdown("### Try an example")
        examples = gr.Examples(
            examples=[
                ["Adult with chest pain and shortness of breath for 1 hour", 45, None],
                ["Pregnant woman 32 weeks, severe headache and blurred vision", 29, None],
                ["Child with fever for 2 days and fast breathing", 5, 18],
                ["Small clean cut on finger, bleeding stopped", 28, None],
                ["Adult with mild sore throat for 3 days, no fever, eating well", 34, None],
            ],
            inputs=[text, age, weight],
            label="",
        )

        # ── Events ────────────────────────────────────────────
        transcribe_btn.click(
            fn=transcribe_audio_to_text,
            inputs=[audio_input, text],
            outputs=[text, patient_name, age, weight, transcribe_status],
        )
        submit.click(
            fn=assess_case,
            inputs=[text, patient_name, age, weight, image_input, audio_input, language],
            outputs=[summary, raw, quick_summary, referral_file],
        )
        text.submit(
            fn=assess_case,
            inputs=[text, patient_name, age, weight, image_input, audio_input, language],
            outputs=[summary, raw, quick_summary, referral_file],
        )
        read_aloud_btn.click(
            fn=None,
            inputs=[],
            outputs=[],
            js="""() => {
                var el = document.getElementById('vm-tts-data');
                console.log('[VoiceMed TTS] Clicked Read Aloud');
                if (!el) {
                    alert('Please run an assessment first, then click Read Aloud.');
                    return;
                }
                var text = el.getAttribute('data-text');
                var lang = el.getAttribute('data-lang') || 'en-US';
                console.log('[VoiceMed TTS] text:', text);
                console.log('[VoiceMed TTS] lang:', lang);
                if (!text) {
                    alert('No text to speak!');
                    return;
                }

                function doSpeak() {
                    window.speechSynthesis.cancel();
                    var u = new SpeechSynthesisUtterance(text);
                    u.lang = lang;
                    var voices = window.speechSynthesis.getVoices();
                    var langPrefix = lang.split('-')[0];
                    var voice = voices.find(function(v) { return v.lang === lang; })
                             || voices.find(function(v) { return v.lang.startsWith(langPrefix); });
                    if (voice) {
                        u.voice = voice;
                        console.log('[VoiceMed TTS] Using voice:', voice.name, voice.lang);
                    } else {
                        console.warn('[VoiceMed TTS] No matching voice found for', lang);
                        alert('No matching voice found for ' + lang + '.');
                    }
                    u.onerror = function(e) {
                        console.error('[VoiceMed TTS error]', e.error);
                        alert('TTS error: ' + e.error);
                    };
                    u.onstart = function() {
                        console.log('[VoiceMed TTS] Speech started');
                    };
                    u.onend = function() {
                        console.log('[VoiceMed TTS] Speech ended');
                    };
                    window.speechSynthesis.speak(u);
                }

                var voices = window.speechSynthesis.getVoices();
                if (voices.length > 0) {
                    console.log('[VoiceMed TTS] Voices loaded:', voices.length);
                    doSpeak();
                } else {
                    console.log('[VoiceMed TTS] Waiting for voices...');
                    window.speechSynthesis.onvoiceschanged = function() {
                        window.speechSynthesis.onvoiceschanged = null;
                        console.log('[VoiceMed TTS] voiceschanged event fired');
                        doSpeak();
                    };
                    setTimeout(function() {
                        if (!window.speechSynthesis.onvoiceschanged) return;
                        console.warn('[VoiceMed TTS] voiceschanged event did not fire, trying anyway');
                        doSpeak();
                    }, 700);
                }
            }""",
        )
        stop_btn.click(
            fn=None,
            inputs=[],
            outputs=[],
            js="() => { window.speechSynthesis.cancel(); }",
        )
        _ = examples

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VoiceMed Gradio demo UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=MOBILE_CSS,
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="blue",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
        ),
    )


if __name__ == "__main__":
    main()
