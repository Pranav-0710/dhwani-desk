from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from gnani.timbre import DEFAULT_VOICE, SAMPLE_RATES

BACKEND_DIR = Path(__file__).resolve().parent
MIN_CONSOLE_TOKEN = 20
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Runtime settings, read from the environment and the repo-root `.env`."""

    # hide_input_in_errors: a rejected CONSOLE_TOKEN must not be printed in the startup log.
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
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
    # A ticket's `location_text` (the caller's own words) is blanked after this long; the rest of
    # the ticket stays.
    ticket_location_retention_days: int = 30

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
    # Refuse /ws/console and /api/* for anything that came through the Cloudflare tunnel, so
    # officers must sit at this machine (http://localhost:8080/supervisor). Callers are not
    # affected. Off by default: the runbook also lets an officer open the console remotely.
    console_local_only: bool = False
    # Consoles signed in at once.
    max_consoles: int = 8
    # Console sockets that have not said hello yet, in all and from one caller (IP address). Apart
    # from `max_consoles`, so strangers holding sockets open cannot lock out an officer.
    max_console_hellos: int = 16
    max_console_hellos_per_client: int = 2
    # Lines an officer may have waiting to be spoken on a call. Each one spends Timbre credits.
    max_officer_lines_queued: int = 5

    # Pages that may open the sockets (/ws/stt, /ws/call, /ws/console), comma-separated, exactly as
    # a browser sends them in its Origin header: scheme, host and port, no path. Browsers apply
    # no CORS to WebSockets, so without this any open web page could stream on our Gnani key.
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # Browser sessions streaming to Prisma at once, across /ws/stt and /ws/call.
    max_stt_sessions: int = 2
    # Of those, sessions one caller (one IP address) may hold at once: a script cannot take
    # every line. 0 turns the per-caller limit off.
    max_sessions_per_client: int = 1
    # The raw transcription relay, /ws/stt. No page uses it: the phone page calls /ws/call. Off, so
    # the tunnel does not offer speech-to-text on our key to anyone who finds it.
    stt_relay_enabled: bool = False
    # Credit ceiling: minutes on the line across all callers in a rolling 24 hours, and calls one
    # caller (IP address) may start in a rolling hour. Once reached, new calls are refused. Counted
    # in memory, per process: restarting the backend starts them afresh. 0 turns either off.
    max_call_minutes_per_day: int = 240
    max_calls_per_client_per_hour: int = 30
    # How much audio, as a share of real time, keeps a session from counting as silent. The phone
    # page streams all the time (1.0), silence included. 0 lets any audio at all count.
    min_audio_share: float = 0.25

    @field_validator("allowed_origins")
    @classmethod
    def _origins(cls, value: str) -> str:
        origins = []
        for entry in value.split(","):
            # As browsers send it: lower case, and no port when it is the scheme's default.
            origin = entry.strip().rstrip("/").lower()
            if not origin:
                continue
            scheme, _, host = origin.partition("://")
            if scheme not in ("http", "https") or not host or "/" in host:
                raise ValueError(
                    f"ALLOWED_ORIGINS: {entry.strip()!r} is not an origin like https://example.com"
                )
            origins.append(origin.removesuffix({"http": ":80", "https": ":443"}[scheme]))
        return ",".join(origins)

    @field_validator("console_token")
    @classmethod
    def _console_token(cls, value: SecretStr) -> SecretStr:
        # The token can speak to callers and read where they are; a short one can be guessed.
        if value.get_secret_value() and len(value.get_secret_value()) < MIN_CONSOLE_TOKEN:
            raise ValueError(
                f"CONSOLE_TOKEN must be at least {MIN_CONSOLE_TOKEN} characters, e.g. "
                '`python -c "import secrets; print(secrets.token_urlsafe(24))"`'
            )
        return value

    @field_validator("max_stt_sessions")
    @classmethod
    def _sessions(cls, value: int) -> int:
        if value < 1:
            raise ValueError("MAX_STT_SESSIONS must be at least 1")
        return value

    @field_validator("max_consoles", "max_console_hellos")
    @classmethod
    def _consoles(cls, value: int) -> int:
        if value < 1:
            raise ValueError("MAX_CONSOLES and MAX_CONSOLE_HELLOS must be at least 1")
        return value

    @field_validator(
        "max_call_minutes_per_day",
        "max_calls_per_client_per_hour",
        "max_console_hellos_per_client",
        "max_officer_lines_queued",
    )
    @classmethod
    def _not_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("A limit cannot be negative")
        return value

    @field_validator("min_audio_share")
    @classmethod
    def _share(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("MIN_AUDIO_SHARE must be between 0 and 1")
        return value

    @property
    def origins(self) -> frozenset[str]:
        return frozenset(self.allowed_origins.split(",")) - {""}

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

    @field_validator("ticket_location_retention_days")
    @classmethod
    def _location_retention(cls, value: int) -> int:
        if value < 1:
            raise ValueError("TICKET_LOCATION_RETENTION_DAYS must be at least 1")
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
