"""Runtime configuration for VoiceMed."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv("MODEL_ID", "google/gemma-4-E4B-it")
    enable_model_inference: bool = os.getenv("ENABLE_MODEL_INFERENCE", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    debug_enabled: bool = os.getenv("VOICEMED_DEBUG", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    use_ollama: bool = os.getenv("USE_OLLAMA", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4")
    ollama_timeout_sec: float = float(os.getenv("OLLAMA_TIMEOUT_SEC", "12"))
    use_quantization: bool = os.getenv("USE_QUANTIZATION", "true").lower() in {"1", "true", "yes"}
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "768"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    model_inference_timeout_sec: float = float(os.getenv("MODEL_INFERENCE_TIMEOUT_SEC", "8"))

    # Repo-root paths
    project_root: Path = Path(__file__).resolve().parents[2]
    treatment_db_path: Path = project_root / "treatment_protocols.json"
    medication_db_path: Path = project_root / "medications.json"


settings = Settings()
