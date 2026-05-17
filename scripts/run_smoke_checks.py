"""One-command smoke checks for VoiceMed local readiness."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_step(name: str, cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"\n== {name} ==")
    print("$", " ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"Step failed: {name} (exit={completed.returncode})")


def main() -> None:
    try:
        run_step(
            "Compile check",
            [
                PYTHON,
                "-m",
                "py_compile",
                "scripts/demo_cli.py",
                "scripts/evaluate_offline.py",
                "src/voicemed/config.py",
                "src/voicemed/engine/model.py",
                "src/voicemed/engine/tools.py",
                "src/voicemed/engine/tool_executor.py",
                "src/voicemed/output/schemas.py",
            ],
        )

        run_step(
            "Demo EMERGENCY case",
            [
                PYTHON,
                "scripts/demo_cli.py",
                "--text",
                "Adult with chest pain and shortness of breath for 1 hour",
                "--pretty",
            ],
        )

        run_step(
            "Demo SELF_CARE case",
            [
                PYTHON,
                "scripts/demo_cli.py",
                "--text",
                "Small clean cut on finger, bleeding stopped",
                "--pretty",
            ],
        )

        env = dict(os.environ)
        env["ENABLE_MODEL_INFERENCE"] = "true"
        run_step(
            "Model flag fallback check",
            [
                PYTHON,
                "scripts/demo_cli.py",
                "--text",
                "Child with fever for 2 days and fast breathing",
                "--age",
                "5",
                "--weight",
                "18",
                "--pretty",
            ],
            env=env,
        )

        run_step("Offline batch evaluation", [PYTHON, "scripts/evaluate_offline.py"])

        report_path = PROJECT_ROOT / "evaluation_results" / "offline_accuracy_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print("\n== Metrics ==")
        print(f"total_cases: {report.get('total_cases')}")
        print(f"exact_accuracy_pct: {report.get('exact_accuracy_pct')}")
        print(f"within_one_level_pct: {report.get('within_one_level_pct')}")
        print("\nSmoke checks completed successfully.")
    except Exception as exc:
        print(f"\nSmoke checks failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
