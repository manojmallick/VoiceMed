"""Output schemas for triage responses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SeverityLevel(str, Enum):
    SELF_CARE = "SELF_CARE"
    MONITOR_48H = "MONITOR_48H"
    REFER_ROUTINE = "REFER_ROUTINE"
    REFER_URGENT = "REFER_URGENT"
    EMERGENCY = "EMERGENCY"


@dataclass
class TriageResult:
    severity: SeverityLevel
    primary_concern: str
    recommended_actions: list[str]
    red_flags: list[str] = field(default_factory=list)
    local_advice: str = ""
    referral_letter: str | None = None
    confidence: str = "MEDIUM"
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def safe_default(cls) -> "TriageResult":
        return cls(
            severity=SeverityLevel.REFER_URGENT,
            primary_concern=(
                "Insufficient clinical detail for low-risk triage. "
                "Escalate for clinician review."
            ),
            recommended_actions=[
                "Check airway, breathing, circulation.",
                "Record vitals if available.",
                "Refer to nearest clinic for in-person assessment.",
            ],
            red_flags=["Unclear condition"],
            local_advice="Seek in-person care today.",
            confidence="LOW",
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload
