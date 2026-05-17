PYTHON ?= python3
VENV := .venv
VENV_PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTHON311 ?= python3.11
OLLAMA_MODEL ?= gemma4
VENV_GEMMA := .venv311
VENV_GEMMA_PY := $(VENV_GEMMA)/bin/python
PIP_GEMMA := $(VENV_GEMMA)/bin/pip

.PHONY: help venv install install-gemma venv-gemma install-gemma311 smoke eval demo ui ui-lan ui-gemma ui-gemma311 start-ui311 start-ui311-fast start-ui311-debug start-ui311-ollama clean

help:
	@echo "Targets:"
	@echo "  make venv     - Create local virtual environment"
	@echo "  make install  - Install dependencies into .venv"
	@echo "  make install-gemma - Install optional model-backed inference dependencies"
	@echo "  make venv-gemma - Create Python 3.11 env at .venv311"
	@echo "  make install-gemma311 - Install base + Gemma deps into .venv311"
	@echo "  make smoke    - Run full project smoke checks"
	@echo "  make eval     - Run offline evaluation"
	@echo "  make demo     - Run demo triage case"
	@echo "  make ui       - Launch Gradio demo UI on localhost:7860"
	@echo "  make ui-lan   - Launch UI on 0.0.0.0:7860 for phone on same Wi-Fi"
	@echo "  make ui-gemma - Launch UI with ENABLE_MODEL_INFERENCE=true"
	@echo "  make ui-gemma311 - Launch UI using .venv311 with ENABLE_MODEL_INFERENCE=true"
	@echo "  make start-ui311 - Alias for ui-gemma311"
	@echo "  make start-ui311-fast - Launch fast demo UI (heuristic mode, low latency)"
	@echo "  make start-ui311-debug - Launch .venv311 UI with debug diagnostics enabled"
	@echo "  make start-ui311-ollama - Launch .venv311 UI using Ollama backend"
	@echo "  make clean    - Remove Python cache files"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-gemma: install
	$(PIP) install -r requirements-gemma.txt

venv-gemma:
	$(PYTHON311) -m venv $(VENV_GEMMA)

install-gemma311: venv-gemma
	$(PIP_GEMMA) install --upgrade pip
	$(PIP_GEMMA) install -r requirements.txt
	$(PIP_GEMMA) install -r requirements-gemma.txt

smoke:
	$(VENV_PY) scripts/run_smoke_checks.py

eval:
	$(VENV_PY) scripts/evaluate_offline.py

demo:
	$(VENV_PY) scripts/demo_cli.py --text "Adult with chest pain and shortness of breath for 1 hour" --pretty

ui:
	$(VENV_PY) scripts/demo_ui.py --host 127.0.0.1 --port 7860

ui-lan:
	$(VENV_PY) scripts/demo_ui.py --host 0.0.0.0 --port 7860

ui-gemma:
	ENABLE_MODEL_INFERENCE=true $(VENV_PY) scripts/demo_ui.py --host 0.0.0.0 --port 7860

ui-gemma311:
	ENABLE_MODEL_INFERENCE=true $(VENV_GEMMA_PY) scripts/demo_ui.py --host 0.0.0.0 --port 7860

start-ui311:
	ENABLE_MODEL_INFERENCE=true $(VENV_GEMMA_PY) scripts/demo_ui.py --host 0.0.0.0 --port 7860

start-ui311-fast:
	ENABLE_MODEL_INFERENCE=false $(VENV_GEMMA_PY) scripts/demo_ui.py --host 0.0.0.0 --port 7860

start-ui311-debug:
	VOICEMED_DEBUG=true ENABLE_MODEL_INFERENCE=true MODEL_INFERENCE_TIMEOUT_SEC=4 $(VENV_GEMMA_PY) scripts/demo_ui.py --host 0.0.0.0 --port 7860

start-ui311-ollama:
	VOICEMED_DEBUG=true ENABLE_MODEL_INFERENCE=true USE_OLLAMA=true OLLAMA_MODEL=$(OLLAMA_MODEL) $(VENV_GEMMA_PY) scripts/demo_ui.py --host 0.0.0.0 --port 7860

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
