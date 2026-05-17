"""Quick Gradio UI for VoiceMed triage demo."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import speech_recognition as sr
from PIL import Image, ImageStat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from voicemed.engine.model import Gemma4TriageEngine  # noqa: E402
from voicemed.config import settings  # noqa: E402


ENGINE = Gemma4TriageEngine()
SESSION_LOG_PATH = PROJECT_ROOT / "evaluation_results" / "demo_session_log.jsonl"
REFERRAL_DIR = PROJECT_ROOT / "evaluation_results" / "referrals"


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


def _severity_badge(severity: str) -> str:
    color = {
        "EMERGENCY": "#c92a2a",
        "REFER_URGENT": "#d9480f",
        "REFER_ROUTINE": "#e67700",
        "MONITOR_48H": "#2b8a3e",
        "SELF_CARE": "#1971c2",
    }.get(severity, "#495057")
    return (
        "<div style='padding:10px 14px;border-radius:10px;"
        f"background:{color};color:white;font-weight:700;display:inline-block'>"
        f"Severity: {severity}</div>"
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


def _build_referral_file(referral_letter: str | None) -> str | None:
    if not referral_letter:
        return None
    REFERRAL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = REFERRAL_DIR / f"referral-{ts}.txt"
    path.write_text(referral_letter, encoding="utf-8")
    return str(path)


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


def transcribe_audio_to_text(audio_path: str | None, current_text: str | None) -> tuple[str, str]:
    base = (current_text or "").strip()
    if not audio_path:
        return base, "No audio file provided."

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        transcript = recognizer.recognize_google(audio_data)
        merged = f"{base}\n{transcript}".strip() if base else transcript
        return merged, "Voice transcription completed."
    except sr.UnknownValueError:
        return base, "Could not understand the audio."
    except sr.RequestError as exc:
        return base, f"Speech service unavailable: {exc}"
    except Exception as exc:
        return base, f"Transcription failed: {exc}"


def assess_case(
    text: str,
    age: float | None,
    weight: float | None,
    image_path: str | None,
    audio_path: str | None,
) -> tuple[str, dict, str, str | None]:
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

    result = ENGINE.triage(
        text_description=final_text,
        patient_age=parsed_age,
        patient_weight_kg=parsed_weight,
    )
    data = result.to_dict()
    data["_debug"] = ENGINE.get_debug_snapshot()

    try:
        _append_session_log(final_text, parsed_age, parsed_weight, data, image_path, audio_path)
    except Exception as exc:
        data.setdefault("_debug", {})["log_error"] = str(exc)

    referral_file = _build_referral_file(data.get("referral_letter"))
    summary_md = _build_summary(data)

    html = [
        _severity_badge(data["severity"]),
        f"<p><b>Primary concern:</b> {data['primary_concern']}</p>",
        "<p><b>Recommended actions:</b></p><ul>",
    ]
    for action in data.get("recommended_actions", []):
        html.append(f"<li>{action}</li>")
    html.append("</ul>")
    html.append(f"<p><b>Patient advice:</b> {data.get('local_advice', '')}</p>")
    if data.get("referral_letter"):
        html.append("<p><b>Referral letter generated:</b> Yes</p>")

    return "\n".join(html), data, summary_md, referral_file


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="VoiceMed Offline Triage") as app:
        gr.Markdown("# VoiceMed Offline Triage Demo")
        gr.Markdown(
            "Use this quick UI for demo runs. It works with the current offline engine and optional model fallback path."
        )
        gr.Markdown(
            "Set VOICEMED_DEBUG=true to include detailed runtime diagnostics in the Raw JSON output (_debug)."
        )
        gr.HTML(backend_status_html())

        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(
                    label="Clinical description",
                    lines=6,
                    placeholder="Example: Adult with chest pain and shortness of breath for 1 hour",
                )
                image_input = gr.Image(
                    label="Optional clinical image",
                    type="filepath",
                )
                audio_input = gr.Audio(
                    label="Optional voice note",
                    sources=["microphone", "upload"],
                    type="filepath",
                )
                with gr.Row():
                    transcribe_btn = gr.Button("Transcribe Voice to Text")
                    transcribe_status = gr.Markdown("")
                with gr.Row():
                    age = gr.Number(label="Age (years)", precision=0)
                    weight = gr.Number(label="Weight (kg)")
                submit = gr.Button("Assess Patient", variant="primary")
            with gr.Column(scale=2):
                summary = gr.HTML(label="Assessment")
                raw = gr.JSON(label="Raw JSON output")
                quick_summary = gr.Markdown(label="Quick Summary")
                referral_file = gr.File(label="Referral Letter File")

        examples = gr.Examples(
            examples=[
                ["Adult with chest pain and shortness of breath for 1 hour", 45, None],
                ["Small clean cut on finger, bleeding stopped", 28, None],
                ["Child with fever for 2 days and fast breathing", 5, 18],
            ],
            inputs=[text, age, weight],
        )

        transcribe_btn.click(
            fn=transcribe_audio_to_text,
            inputs=[audio_input, text],
            outputs=[text, transcribe_status],
        )

        submit.click(
            fn=assess_case,
            inputs=[text, age, weight, image_input, audio_input],
            outputs=[summary, raw, quick_summary, referral_file],
        )
        text.submit(
            fn=assess_case,
            inputs=[text, age, weight, image_input, audio_input],
            outputs=[summary, raw, quick_summary, referral_file],
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
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
