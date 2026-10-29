from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from gnani.timbre import DEFAULT_VOICE, SAMPLE_RATES

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

    gnani_tts_url: str = "https://api.vachana.ai/api/v1/tts"
    gnani_tts_lang: str = "kn-IN"
    gnani_tts_voice: str = DEFAULT_VOICE
    gnani_tts_sample_rate: int = 16000
    gnani_tts_speed: float = 1.0

    # SQLite only; a relative path is relative to backend/.
    database_url: str = "sqlite:///./dhwani.db"
    transcript_retention_days: int = 7

    # Who decides on the caller's words: keyword rules, or Laya (decisions/laya.py).
    decider: Literal["keyword", "laya"] = "keyword"
    hf_token: SecretStr = SecretStr("")
    # A Hugging Face repo, or a local directory with the same layout.
    laya_checkpoint_repo: str = "Dhwanidesk/dhwani-desk-laya"
    # A tag or commit of the repo; required unless the checkpoint is a local directory.
    laya_checkpoint_revision: str = ""
    laya_threads: int = 8

    # Shared secret a supervisor console sends before it may watch or take over calls. Empty
    # turns /ws/console off: it can speak to callers.
    console_token: SecretStr = SecretStr("")

    @field_validator("gnani_tts_voice", mode="before")
    @classmethod
    def _default_voice_when_blank(cls, value: object) -> object:
        return DEFAULT_VOICE if value in ("", None) else value

    @field_validator("gnani_tts_sample_rate")
    @classmethod
    def _tts_rate(cls, value: int) -> int:
        if value not in SAMPLE_RATES:
            raise ValueError("GNANI_TTS_SAMPLE_RATE must be a Timbre-supported rate")
        return value

    @field_validator("gnani_tts_speed")
    @classmethod
    def _tts_speed(cls, value: float) -> float:
        if not 0.85 <= value <= 1.15:
            raise ValueError("GNANI_TTS_SPEED must be between 0.85 and 1.15")
        return value

    @field_validator("gnani_stt_sample_rate")
    @classmethod
    def _telephony_rate(cls, value: int) -> int:
        if value not in (8000, 16000):
            raise ValueError("GNANI_STT_SAMPLE_RATE must be 8000 or 16000")
        return value

    @field_validator("database_url")
    @classmethod
    def _sqlite_only(cls, value: str) -> str:
        if not value.startswith("sqlite:///"):
            raise ValueError("DATABASE_URL must be a sqlite:/// URL")
        return value

    @field_validator("transcript_retention_days")
    @classmethod
    def _retention(cls, value: int) -> int:
        if value < 1:
            raise ValueError("TRANSCRIPT_RETENTION_DAYS must be at least 1")
        return value

    @field_validator("laya_threads")
    @classmethod
    def _threads(cls, value: int) -> int:
        if value < 1:
            raise ValueError("LAYA_THREADS must be at least 1")
        return value

    @property
    def database_path(self) -> Path:
        path = Path(self.database_url.removeprefix("sqlite:///"))
        return path if path.is_absolute() else BACKEND_DIR / path

    @field_validator("gnani_stt_bias_score")
    @classmethod
    def _allowed_bias_score(cls, value: float) -> float:
        if value not in (0.0, 0.5, 1.0, 1.5, 2.0):
            raise ValueError("GNANI_STT_BIAS_SCORE must be one of 0.0, 0.5, 1.0, 1.5, 2.0")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
