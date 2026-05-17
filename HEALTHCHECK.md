# VoiceMed Healthcheck

This document provides a fast way to confirm local VoiceMed readiness.

## Prerequisites

- You are in project root.
- Virtual environment exists at `.venv`.

## One-command check

Run:

```bash
.venv/bin/python scripts/run_smoke_checks.py
```

This executes:

- Python compile checks for key source files
- Two CLI sanity cases
- Feature-flagged model path check with safe fallback
- Offline batch evaluation and metrics summary

## Expected outcomes

- Command exits with code `0`
- You should see `Smoke checks completed successfully.`
- Evaluation metrics are printed from `evaluation_results/offline_accuracy_report.json`

## Manual commands

Single triage run:

```bash
.venv/bin/python scripts/demo_cli.py --text "Adult with chest pain and shortness of breath for 1 hour" --pretty
```

Batch evaluation:

```bash
.venv/bin/python scripts/evaluate_offline.py
```

## Troubleshooting

- `ModuleNotFoundError: voicemed`
  - Run commands from project root.
- CLI exits with JSON error payload
  - Check `error` and `hint` fields printed to stderr.
- Model flag appears slow or unavailable
  - Model-backed path is optional. If load fails, engine falls back to offline heuristics.
- Missing report file
  - Re-run `scripts/evaluate_offline.py` and confirm write permissions for `evaluation_results`.
