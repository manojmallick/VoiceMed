"""Batch evaluation for the offline VoiceMed triage engine.

Reads TEST_CASES from run_evaluation.py, runs the current offline engine,
and writes a quick accuracy report for fast iteration.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from voicemed.engine.model import Gemma4TriageEngine  # noqa: E402


SEVERITY_ORDER = [
    "SELF_CARE",
    "MONITOR_48H",
    "REFER_ROUTINE",
    "REFER_URGENT",
    "EMERGENCY",
]
SEVERITY_INDEX = {name: idx for idx, name in enumerate(SEVERITY_ORDER)}


@dataclass
class CaseResult:
    case_id: str
    expected: str
    predicted: str
    description: str
    exact_match: bool
    distance: int


def load_test_cases(path: Path) -> list[dict]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TEST_CASES":
                    return ast.literal_eval(node.value)
    raise ValueError("Could not find TEST_CASES in run_evaluation.py")


def evaluate(cases: list[dict]) -> dict:
    engine = Gemma4TriageEngine()
    exact = 0
    within_one = 0
    unknown = 0
    confusion: dict[str, Counter] = defaultdict(Counter)
    per_case: list[CaseResult] = []

    for case in cases:
        expected = case["expected_severity"]
        predicted = engine.triage(text_description=case["description"]).severity.value

        if expected not in SEVERITY_INDEX or predicted not in SEVERITY_INDEX:
            unknown += 1
            distance = 99
        else:
            distance = abs(SEVERITY_INDEX[expected] - SEVERITY_INDEX[predicted])

        exact_match = expected == predicted
        if exact_match:
            exact += 1
        if distance <= 1:
            within_one += 1

        confusion[expected][predicted] += 1
        per_case.append(
            CaseResult(
                case_id=case["id"],
                expected=expected,
                predicted=predicted,
                description=case["description"],
                exact_match=exact_match,
                distance=distance,
            )
        )

    total = len(cases)
    by_class = {}
    for expected in SEVERITY_ORDER:
        class_cases = [c for c in per_case if c.expected == expected]
        class_total = len(class_cases)
        class_hits = sum(1 for c in class_cases if c.exact_match)
        by_class[expected] = {
            "total": class_total,
            "exact_accuracy": round((class_hits / class_total) * 100, 2) if class_total else 0.0,
        }

    worst_cases = [c for c in per_case if c.distance >= 2]
    worst_cases.sort(key=lambda c: c.distance, reverse=True)

    return {
        "total_cases": total,
        "exact_accuracy_pct": round((exact / total) * 100, 2) if total else 0.0,
        "within_one_level_pct": round((within_one / total) * 100, 2) if total else 0.0,
        "unknown_label_cases": unknown,
        "per_class": by_class,
        "confusion_matrix": {
            expected: {pred: count for pred, count in predicted_counts.items()}
            for expected, predicted_counts in confusion.items()
        },
        "worst_misclassifications": [
            {
                "id": c.case_id,
                "expected": c.expected,
                "predicted": c.predicted,
                "distance": c.distance,
                "description": c.description,
            }
            for c in worst_cases[:20]
        ],
        "all_results": [
            {
                "id": c.case_id,
                "expected": c.expected,
                "predicted": c.predicted,
                "distance": c.distance,
                "exact_match": c.exact_match,
            }
            for c in per_case
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate offline VoiceMed triage engine")
    parser.add_argument(
        "--cases-file",
        default=str(PROJECT_ROOT / "run_evaluation.py"),
        help="Path to Python file containing TEST_CASES",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "evaluation_results" / "offline_accuracy_report.json"),
        help="Path to output report JSON",
    )
    args = parser.parse_args()

    cases = load_test_cases(Path(args.cases_file))
    report = evaluate(cases)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Total cases: {report['total_cases']}")
    print(f"Exact accuracy: {report['exact_accuracy_pct']}%")
    print(f"Within 1 level: {report['within_one_level_pct']}%")
    print(f"Unknown labels: {report['unknown_label_cases']}")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
