# VoiceMed — Architecture

> AI-assisted clinical triage for community health workers, powered by Gemma 4 via Ollama.

---

## System Overview

```
Phone / Browser
      │  HTTPS (ngrok tunnel)
      ▼
┌─────────────────────────────────────────────┐
│            Gradio UI  (port 7860)           │
│  scripts/demo_ui.py                         │
│                                             │
│  ┌──────────┐  ┌────────────┐  ┌─────────┐ │
│  │  Voice   │  │  Clinical  │  │  Image  │ │
│  │  Input   │  │   Form     │  │ Upload  │ │
│  └────┬─────┘  └─────┬──────┘  └────┬────┘ │
│       │              │              │       │
│  SpeechRecognition   │         PIL metadata │
│  + NLP field parse   │              │       │
│  (name/age/weight)   │              │       │
└───────┼──────────────┼──────────────┼───────┘
        │              ▼              │
        └──────► assess_case() ◄──────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │   Gemma4TriageEngine     │
        │   src/voicemed/engine/   │
        │   model.py               │
        └──────────┬───────────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
         ▼                    ▼
   Ollama API          Offline heuristic
   (gemma4:latest)     rule engine
   HTTP/JSON           (fallback)
         │
         ▼
   ToolExecutor  →  Tool functions
   tool_executor.py   tools.py
         │
         ▼
   TriageResult
   output/schemas.py
         │
         ▼
   ┌─────────────────────┐
   │  Gradio UI output   │
   │  • HTML result card │
   │  • Quick summary    │
   │  • PDF referral     │
   │  • Raw JSON         │
   └─────────────────────┘
```

---

## Module Reference

### `src/voicemed/config.py`

Runtime configuration via environment variables. Frozen dataclass — read once at startup.

| Variable | Default | Purpose |
|---|---|---|
| `ENABLE_MODEL_INFERENCE` | `false` | Enable AI inference vs heuristic fallback |
| `USE_OLLAMA` | `false` | Route inference through local Ollama server |
| `OLLAMA_MODEL` | `gemma4` | Model tag served by Ollama |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `OLLAMA_TIMEOUT_SEC` | `12` | Per-request timeout |
| `MODEL_ID` | `google/gemma-4-E4B-it` | HuggingFace model ID (direct mode) |
| `MAX_NEW_TOKENS` | `768` | Max tokens generated per response |
| `TEMPERATURE` | `0.1` | Sampling temperature (deterministic-biased) |

---

### `src/voicemed/engine/model.py` — `Gemma4TriageEngine`

Central orchestrator. Two inference backends, selected at runtime:

**Backend A — Ollama (recommended for demo)**
```
triage(text) → POST http://localhost:11434/api/chat
               model: gemma4
               messages: [system_prompt, user_text]
               tools: [assess_triage_severity, lookup_treatment_protocol, ...]
             → parse tool_calls → ToolExecutor.execute()
             → parse final content → TriageResult
```

**Backend B — Direct HuggingFace Transformers**
```
triage(text) → load google/gemma-4-E4B-it (CPU, float32)
             → AutoProcessor + AutoModelForImageTextToText
             → generate() → parse → TriageResult
```

**Fallback — Offline heuristic engine**  
When `ENABLE_MODEL_INFERENCE=false` or on any model/network error, a deterministic rule-based scorer produces a `TriageResult` without any LLM call.

---

### `src/voicemed/engine/tools.py` — Clinical Tool Functions

Five callable tools exposed to the model as a function-calling schema:

| Tool | Inputs | Purpose |
|---|---|---|
| `assess_triage_severity` | symptom, duration_hours, consciousness, breathing_difficulty, bleeding_severity | Score-based severity classification → `SeverityLevel` |
| `lookup_treatment_protocol` | condition, setting | Lookup from `treatment_protocols.json` |
| `check_medication_availability` | medication_name, dose_mg, route | Query `medications.json` stock |
| `calculate_pediatric_dose` | medication_name, weight_kg, age_years | Weight-based dose calculation |
| `generate_referral_letter` | patient_name, finding, severity | Produce plain-text referral letter |

All tools are registered in `TOOL_REGISTRY` dict for safe whitelisted dispatch.

---

### `src/voicemed/engine/tool_executor.py` — `ToolExecutor`

Thin dispatch layer. Only names in `TOOL_REGISTRY` can be called — prevents prompt-injection-driven arbitrary code execution.

```python
ToolExecutor.execute(name, arguments) → {"ok": True, "result": ...}
                                      | {"ok": False, "error": ...}
```

---

### `src/voicemed/engine/system_prompt.py`

Single-file system prompt. Instructs the model to act as a conservative clinical triage assistant for community health workers, with explicit escalation rules for danger signs.

---

### `src/voicemed/output/schemas.py`

**`SeverityLevel`** — ordered enum used throughout the pipeline:

```
SELF_CARE < MONITOR_48H < REFER_ROUTINE < REFER_URGENT < EMERGENCY
```

**`TriageResult`** — structured output dataclass:

| Field | Type | Description |
|---|---|---|
| `severity` | `SeverityLevel` | Triage classification |
| `primary_concern` | `str` | One-line clinical summary |
| `recommended_actions` | `list[str]` | Ordered action steps |
| `red_flags` | `list[str]` | Danger signs requiring escalation |
| `local_advice` | `str` | Context-appropriate local guidance |
| `referral_letter` | `str \| None` | Auto-generated referral text |
| `confidence` | `str` | `LOW` / `MEDIUM` / `HIGH` |
| `generated_at` | `str` | UTC ISO-8601 timestamp |

---

### `scripts/demo_ui.py` — Gradio UI

Mobile-first single-column interface served on port 7860.

**Key functions:**

| Function | Role |
|---|---|
| `transcribe_audio_to_text(audio, current_text)` | Google STT → NLP field extraction (name / age / weight / symptoms) |
| `_parse_voice_fields(transcript)` | Regex-based structured extraction from free speech |
| `assess_case(text, patient_name, age, weight, image, audio)` | Main triage orchestrator → calls engine → builds outputs |
| `_build_referral_pdf(letter, data, image_path, patient_name)` | ReportLab PDF with severity badge, embedded image, letter body |
| `_build_summary(data)` | Markdown quick-summary card |
| `_image_context(image_path)` | PIL image metadata → appended to clinical text |
| `build_ui()` | Gradio layout definition |
| `backend_status_html()` | Live backend mode indicator |

**Voice-to-fields pipeline:**
```
Audio recording
      │
      ▼  Google Speech Recognition (SpeechRecognition lib)
Raw transcript
      │
      ▼  _parse_voice_fields()
      │    • name  ← "patient name is X" / "for patient X" / "called X"
      │    • age   ← "32 years old" / "aged 32" / "35-year-old"
      │    • weight← "65 kg" / "weighs 65" / "weighing 65"
      │    • symptoms ← remainder after stripping above fragments
      ▼
gr.update() per field → auto-populates Patient name, Age, Weight, Clinical description
```

**PDF referral pipeline:**
```
TriageResult + patient_name + clinical_image
      │
      ▼  reportlab SimpleDocTemplate
      │    • Teal header (VoiceMed logo text)
      │    • Patient subtitle with name
      │    • Severity badge table (colour-coded)
      │    • Embedded clinical image (max 12cm × 8cm)
      │    • Referral letter body
      │    • Recommended actions list
      │    • Red flags (red text)
      │    • Disclaimer footer
      ▼
evaluation_results/referrals/referral-YYYYMMDD-HHMMSS.pdf
```

---

## Data Files

| File | Purpose |
|---|---|
| `treatment_protocols.json` | Condition → protocol lookup (offline) |
| `medications.json` | Medication stock / dosing reference (offline) |
| `evaluation_results/demo_session_log.jsonl` | Append-only session log (one JSON object per assessment) |
| `evaluation_results/referrals/` | Generated PDF referral letters |

---

## Deployment

### Local demo (Ollama backend)
```bash
ENABLE_MODEL_INFERENCE=true \
USE_OLLAMA=true \
OLLAMA_MODEL=gemma4 \
OLLAMA_TIMEOUT_SEC=120 \
PYTHONPATH=src \
.venv311/bin/python scripts/demo_ui.py --host 0.0.0.0 --port 7860
```

### Public HTTPS access (phone / mic / camera)
```bash
ngrok http 7860
# Tunnel: https://outgrow-glimmer-galvanize.ngrok-free.dev → localhost:7860
```

### Offline / heuristic mode (no Ollama required)
```bash
PYTHONPATH=src .venv311/bin/python scripts/demo_ui.py
```

---

## Security Notes

- Tool dispatch is whitelisted — only functions in `TOOL_REGISTRY` are callable
- No patient data is persisted beyond the local session log (`demo_session_log.jsonl`)
- All inference is local (Ollama) or rule-based — no data leaves the machine
- ngrok tunnel uses HTTPS; the free-plan URL is fixed per session

---

## Dependencies

| Package | Purpose |
|---|---|
| `gradio >= 5.0` | Web UI framework |
| `SpeechRecognition >= 3.10` | Google STT for voice input |
| `reportlab >= 4.0` | PDF referral generation |
| `Pillow` | Image metadata + PDF embedding |
| `ollama` / `transformers` | LLM inference backends |
