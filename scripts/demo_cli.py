"""Simple CLI for first-pass VoiceMed triage implementation."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from voicemed.engine.model import Gemma4TriageEngine  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VoiceMed triage demo CLI")
    parser.add_argument("--text", required=True, help="Clinical description")
    parser.add_argument("--age", type=int, default=None, help="Patient age in years")
    parser.add_argument("--weight", type=float, default=None, help="Patient weight in kg")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print output JSON",
    )
    return parser.parse_args()


def main() -> None:
    try:
        args = parse_args()

        if args.age is not None and args.age < 0:
            raise ValueError("--age must be >= 0")
        if args.weight is not None and args.weight <= 0:
            raise ValueError("--weight must be > 0")

        engine = Gemma4TriageEngine()
        result = engine.triage(
            text_description=args.text,
            patient_age=args.age,
            patient_weight_kg=args.weight,
        )
        if args.pretty:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(json.dumps(result.to_dict()))
    except Exception as exc:
        error_payload = {
            "ok": False,
            "error": str(exc),
            "hint": (
                "Re-run with --text and optional valid --age/--weight values. "
                "Use scripts/run_smoke_checks.py for a full system check."
            ),
        }
        print(json.dumps(error_payload, indent=2), file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
