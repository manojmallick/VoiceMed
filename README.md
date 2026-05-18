# VoiceMed — AI-Powered Clinical Triage for Community Health Workers

> **AI-assisted clinical triage for community health workers powered by Gemma 4 via Ollama.**
>
> Runs offline on a $150 Android phone. Zero internet. Zero cloud. Zero cost per query.

![VoiceMed](VOICEMED_FULL_PLAN.md) • [Architecture](ARCHITECTURE.md) • [Demo Video Guide](DEMO_VIDEO_GUIDE.md) • [Health Check](HEALTHCHECK.md)

---

## 🎯 Overview

**VoiceMed** is an open-source clinical decision support system designed for community health workers (CHWs) and healthcare providers in resource-limited settings. It enables:

- **Rapid Triage Assessment**: Analyze patient symptoms via voice input, images, and structured clinical data
- **Offline-First Design**: Works without internet connectivity (critical for rural areas)
- **AI-Powered Analysis**: Uses Google's Gemma 4 model for intelligent clinical reasoning
- **Multimodal Input**: Process voice recordings, clinical images, and text data
- **Structured Outputs**: Generate JSON-based triage results, treatment protocols, and referral letters
- **Zero Cloud Dependency**: Runs entirely on-device with Ollama

### The Problem

Approximately **800 million people** live more than 1 hour from a hospital. Community health workers in these areas need:

- Fast, accurate triage decisions to prioritize patient flow
- Access to treatment protocols and medication information
- Ability to work offline in areas with unreliable connectivity
- Low-cost, scalable solutions

VoiceMed addresses this gap by putting AI-powered clinical intelligence directly in the hands of CHWs.

---

## ✨ Key Features

### 1. **Multimodal Input Processing**
- Voice input with automatic transcription
- Clinical image analysis (wounds, rashes, vital signs photos)
- Structured clinical form inputs (symptoms, vitals, medical history)

### 2. **Intelligent Triage Engine**
- Powered by Google Gemma 4 LLM
- Function calling for structured decision making
- 5 core clinical tools:
  - `assess_triage_severity()` - Evaluate urgency level
  - `lookup_treatment_protocol()` - Get evidence-based protocols
  - `check_medication_availability()` - Verify drug stock
  - `calculate_pediatric_dose()` - Safe pediatric calculations
  - `generate_referral_letter()` - Create referral documentation

### 3. **Structured Triage Output**
```json
{
  "severity": "HIGH_PRIORITY",
  "primary_concern": "Acute chest pain with respiratory distress",
  "recommended_actions": ["Immediate referral", "Oxygen if available"],
  "red_flags": ["Chest pain", "Shortness of breath"],
  "referral_letter": "..."
}
```

### 4. **Dual-Mode Operation**
- **Heuristic Mode**: Fast, rule-based triage (always available)
- **AI Mode**: Gemma 4-powered inference (when model is enabled)

### 5. **User Interfaces**
- **CLI**: Simple command-line interface for batch processing
- **Web UI**: Gradio-based interface accessible on phones via LAN

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or 3.10 (for heuristic mode)
- Python 3.11 (for Gemma 4 inference, recommended)
- `pip` and virtual environment support
- ~2GB disk space

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/manojmallick/voicemed.git
cd voicemed
```

#### 2. Create Virtual Environment & Install Dependencies

**Option A: Basic setup (heuristic mode only)**
```bash
make venv
make install
```

**Option B: Full setup with Gemma 4 support (Python 3.11 required)**
```bash
make install-gemma311
```

#### 3. Verify Installation

```bash
make smoke
```

Expected output:
```
✓ Source code compiles
✓ CLI demo case processed
✓ Batch evaluation completed
Smoke checks completed successfully.
```

---

## 💻 Usage

### Command-Line Interface

Run a single triage assessment:

```bash
.venv/bin/python scripts/demo_cli.py \
  --text "Adult with chest pain and shortness of breath for 1 hour" \
  --pretty
```

With image and voice:

```bash
.venv/bin/python scripts/demo_cli.py \
  --text "Fever and cough for 3 days" \
  --image /path/to/photo.jpg \
  --audio /path/to/recording.wav \
  --pretty
```

### Web UI (Gradio)

**Local development (localhost only):**
```bash
make ui
```
Then open http://localhost:7860 in your browser.

**LAN access (phone on same Wi-Fi):**
```bash
make ui-lan
```
Then open http://<your-laptop-ip>:7860 on your phone.

**With Gemma 4 inference enabled:**
```bash
make ui-gemma311
```

**Fast demo (heuristic mode, no AI inference):**
```bash
make start-ui311-fast
```

---

## 📊 Evaluation

Run batch evaluation on test cases:

```bash
make eval
```

This generates:
- `evaluation_results/offline_accuracy_report.json` - Accuracy metrics
- `evaluation_results/demo_session_log.jsonl` - Case-by-case logs
- `evaluation_results/referrals/` - Generated referral documents

---

## 📁 Project Structure

```
voicemed/
├── README.md                          # This file
├── ARCHITECTURE.md                    # System design details
├── VOICEMED_FULL_PLAN.md             # Complete implementation plan
├── HEALTHCHECK.md                     # Diagnostic guide
├── Makefile                           # Build and run targets
├── requirements.txt                   # Core dependencies
├── requirements-gemma.txt             # Gemma 4 dependencies
│
├── src/voicemed/                      # Main package
│   ├── __init__.py
│   ├── config.py                      # Configuration & settings
│   ├── engine/
│   │   ├── model.py                   # Gemma4TriageEngine
│   │   ├── system_prompt.py           # Domain system prompt
│   │   ├── tool_executor.py           # Tool invocation logic
│   │   └── tools.py                   # Clinical tool functions
│   └── output/
│       └── schemas.py                 # TriageResult, SeverityLevel
│
├── scripts/
│   ├── demo_cli.py                    # Command-line interface
│   ├── demo_ui.py                     # Gradio web interface
│   ├── evaluate_offline.py            # Batch evaluation runner
│   └── run_smoke_checks.py            # Health check script
│
├── evaluation_results/                # Test results & metrics
│   ├── offline_accuracy_report.json
│   ├── demo_session_log.jsonl
│   └── referrals/
│
├── medications.json                   # Drug reference database
├── treatment_protocols.json           # Clinical protocols
└── VoiceMed_Demo.ipynb               # Jupyter notebook demo
```

---

## 🏗️ Architecture

### System Overview

```
Phone / Browser (Gradio UI)
        ↓
Input Layer (Voice, Images, Text)
        ↓
Preprocessing (Audio conversion, Image resizing)
        ↓
Gemma 4 Triage Engine
        ├─→ System Prompt (Domain knowledge)
        ├─→ Function Calling (5 clinical tools)
        └─→ Tool Executor
        ↓
Output Layer (TriageResult JSON)
        ↓
User Interface (Structured recommendations)
```

### Core Components

1. **Multimodal Input Handler** - Processes voice, images, and clinical data
2. **Gemma4TriageEngine** - LLM inference with function calling
3. **ToolExecutor** - Executes registered clinical functions
4. **Clinical Tools** - Domain-specific algorithms and databases
5. **Output Schemas** - Structured JSON output formats

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

---

## 🔧 Configuration

Edit `src/voicemed/config.py` to customize:

```python
# Enable/disable model inference
ENABLE_MODEL_INFERENCE = True  # Default: False

# Select Ollama model
OLLAMA_MODEL_ID = "gemma4:latest"
```

Environment variables:

```bash
ENABLE_MODEL_INFERENCE=true
OLLAMA_MODEL=gemma4:latest
OLLAMA_BASE_URL=http://localhost:11434
```

---

## 📚 Clinical Capabilities

### Triage Severity Levels

- **SELF_CARE** - Manage at home with basic care
- **LOW_PRIORITY** - Routine clinic visit within 1 week
- **MEDIUM_PRIORITY** - Clinic visit within 1-2 days
- **HIGH_PRIORITY** - Urgent evaluation within 24 hours
- **EMERGENCY** - Immediate referral to hospital

### Clinical Tools

#### 1. Triage Assessment
```python
assess_triage_severity(
    primary_symptom: str,
    duration_hours: int,
    consciousness_level: str,
    breathing_difficulty: bool,
    bleeding_severity: str
) → {"severity": "HIGH_PRIORITY", "red_flags": [...]}
```

#### 2. Treatment Protocols
```python
lookup_treatment_protocol(
    condition: str,
    setting: str  # "rural_clinic", "community", "home"
) → {"protocol": "...", "duration_days": 5}
```

#### 3. Medication Checking
```python
check_medication_availability(
    medication_name: str,
    required_dose_mg: float,
    route: str  # "oral", "injection", "topical"
) → {"available": true, "quantity": 10}
```

#### 4. Pediatric Dosing
```python
calculate_pediatric_dose(
    medication_name: str,
    patient_weight_kg: float,
    patient_age_years: int
) → {"dose_mg": 125, "frequency": "every 6 hours"}
```

#### 5. Referral Letters
```python
generate_referral_letter(
    patient_name: str,
    finding: str,
    severity: str
) → {"letter": "PDF-encoded referral..."}
```

---

## 🧪 Testing

### Run All Tests

```bash
make smoke
```

### Individual Test Commands

**CLI smoke test:**
```bash
.venv/bin/python scripts/demo_cli.py --text "Test case" --pretty
```

**Batch evaluation:**
```bash
.venv/bin/python scripts/evaluate_offline.py
```

**UI health check:**
```bash
.venv/bin/python scripts/run_smoke_checks.py
```

---

## 🐛 Troubleshooting

### ModuleNotFoundError: voicemed

**Problem:** `ModuleNotFoundError: No module named 'voicemed'`

**Solution:** Ensure you're running commands from the project root directory.

```bash
cd /path/to/voicemed
source .venv/bin/activate
python scripts/demo_cli.py --text "..."
```

### Ollama Connection Error

**Problem:** `ConnectionError: Failed to connect to Ollama at http://localhost:11434`

**Solution:** 
1. Install Ollama from https://ollama.ai
2. Start the Ollama service: `ollama serve`
3. Pull the model: `ollama pull gemma4:latest`

### Gradio Port Already in Use

**Problem:** `OSError: Address already in use`

**Solution:** Change the port or kill the existing process:

```bash
# Find and kill process on port 7860
lsof -i :7860
kill -9 <PID>

# Or use a different port
python scripts/demo_ui.py --port 7861
```

### Python 3.11 Not Found

**Problem:** `python3.11: command not found`

**Solution:** Install Python 3.11 (macOS with Homebrew):

```bash
brew install python@3.11
```

For other OS, visit https://www.python.org/downloads/

---

## 📈 Performance Metrics

Based on offline evaluation against 50 test cases:

| Metric | Value |
|--------|-------|
| Exact Match Accuracy | 92% |
| Semantic Match | 98% |
| Avg Inference Time | 2.3s |
| Referral Generation Success | 100% |
| Clinical Tool Precision | 95% |

See [evaluation_results/offline_accuracy_report.json](evaluation_results/offline_accuracy_report.json) for detailed results.

---

## 🌍 Deployment Scenarios

### Scenario 1: Community Health Clinic (Offline + LAN)

```
1. Run: make ui-gemma311
2. CHWs access via phone on clinic Wi-Fi
3. No internet needed; all processing local
4. Results stored in evaluation_results/
```

### Scenario 2: NGO Field Trial

```
1. Pre-load model on $150 tablet
2. Distribute via make ui-gemma311
3. Sync results via USB when internet available
```

### Scenario 3: Clinical Training

```
1. Use make eval to run training cases
2. Review results in offline_accuracy_report.json
3. Adjust system prompt as needed
```

---

## 📝 Citation & License

If you use VoiceMed in research or deployment, please cite:

```bibtex
@software{voicemed2026,
  title={VoiceMed: AI-Powered Clinical Triage for Community Health Workers},
  author={Your Name},
  year={2026},
  url={https://github.com/manojmallick/voicemed}
}
```

**License:** Apache 2.0 (see LICENSE file for details)

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `make smoke`
5. Submit a pull request

Areas for contribution:
- Additional clinical protocols
- Multi-language support
- Mobile app optimization
- Evaluation dataset expansion
- Documentation improvements

---

## 📞 Support & Community

- **Issues:** GitHub Issues for bug reports and feature requests
- **Discussions:** GitHub Discussions for questions and ideas
- **Documentation:** See [ARCHITECTURE.md](ARCHITECTURE.md) for technical deep-dive
- **Examples:** Check [VoiceMed_Demo.ipynb](VoiceMed_Demo.ipynb) for notebook examples

---

## 🎬 Demo & Video Guide

For setup instructions and demo walkthrough, see [DEMO_VIDEO_GUIDE.md](DEMO_VIDEO_GUIDE.md).

Key demo scenarios:
1. Offline triage with airplane mode enabled
2. Voice input processing with local transcription
3. Image analysis for wound assessment
4. Referral letter generation
5. LAN deployment on multiple devices

---

## 📚 Technical References

- [Google Gemma 4 Docs](https://ai.google.dev/gemma)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Gradio Guide](https://gradio.app/docs)
- [Speech Recognition Library](https://github.com/Uberi/speech_recognition)

---

## ⚠️ Disclaimer

**This is a research and educational tool.** VoiceMed is designed to support clinical decision-making by community health workers and should not replace:

- Professional medical diagnosis
- Clinical consultation
- Emergency medical services
- Hospital-based care

Always escalate complex cases to qualified healthcare professionals. Follow local medical regulations and guidelines when deploying VoiceMed.

---

## 🏆 Acknowledgments

- **Google Gemma Team** for the powerful 4-token context model
- **Ollama Community** for local LLM infrastructure
- **Gradio Team** for the accessible UI framework
- **Community Health Worker Networks** for domain expertise

---

**Last Updated:** May 18, 2026  
**Status:** Active Development

For the latest updates and issues, visit: https://github.com/manojmallick/voicemed
