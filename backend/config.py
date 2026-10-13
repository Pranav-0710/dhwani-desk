from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Runtime settings, read from the environment and the repo-root `.env`."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    gnani_api_key: SecretStr = SecretStr("")
    gnani_stt_url: str = "wss://api.vachana.ai/stt/v3/stream"
    gnani_stt_lang: str = "kn-IN"
    gnani_stt_sample_rate: int = 8000
    gnani_stt_format: Literal["verbatim", "transcribe"] = "transcribe"
    gnani_stt_bias_file: Path = BACKEND_DIR / "gnani" / "stt_bias_words.json"
    gnani_stt_bias_score: float = 1.0

    @field_validator("gnani_stt_sample_rate")
    @classmethod
    def _telephony_rate(cls, value: int) -> int:
        if value not in (8000, 16000):
            raise ValueError("GNANI_STT_SAMPLE_RATE must be 8000 or 16000")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
