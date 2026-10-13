"""Render the same Kannada helpline lines in several Timbre voices, to pick one by ear.

Usage, from backend/:
    uv run python -m scripts.tts_voices                     # default candidates
    uv run python -m scripts.tts_voices --voices Kavin Saanvi --rate 8000

Writes WAV files to <repo>/samples/tts/ (git-ignored) and prints, per voice, the time to
the first streamed audio chunk and the total synthesis time. Reads GNANI_API_KEY from .env.
"""

import argparse
import asyncio
import contextlib
import sys
import time
import wave
from pathlib import Path

from config import REPO_ROOT, get_settings
from gnani.timbre import SAMPLE_RATES, TimbreClient, TimbreConfig, TimbreError

# Kavin and Saanvi are Timbre's native Kannada voices; the rest are other-language voices
# speaking Kannada, for comparison.
DEFAULT_VOICES = ["Kavin", "Saanvi", "Chitra", "Kaveri", "Lavanya"]

# Greeting, ward + ticket number read back as words, and a code-mixed line as Prisma
# writes it (English words in Kannada script).
SCRIPT = (
    "ನಮಸ್ಕಾರ, ನಗರ ಸೇವಾ ಸಹಾಯವಾಣಿಗೆ ಸ್ವಾಗತ. "
    "ನಿಮ್ಮ ದೂರು ದಾಖಲಾಗಿದೆ. ವಾರ್ಡ್ ಸಂಖ್ಯೆ ನಾಲ್ಕು, ಗಾಂಧಿ ನಗರ. "
    "ನಿಮ್ಮ ಟಿಕೆಟ್ ಸಂಖ್ಯೆ ಎರಡು ಸಾವಿರದ ಮುನ್ನೂರ ನಲವತ್ತೈದು. "
    "ನಿಮ್ಮ ಏರಿಯಾದ ಸ್ಟ್ರೀಟ್ ಲೈಟ್ ರಿಪೇರಿಗೆ ನಾಳೆ ಟೀಮ್ ಬರುತ್ತದೆ."
)
# Same ticket number as digits: does Timbre read numerals itself?
DIGITS = "ನಿಮ್ಮ ಟಿಕೆಟ್ ಸಂಖ್ಯೆ 2345. ವಾರ್ಡ್ 4."


def write_wav(path: Path, pcm: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)


async def render(client: TimbreClient, voice: str, text: str) -> tuple[bytes, float, float]:
    started = time.monotonic()
    first_chunk_s = 0.0
    chunks: list[bytes] = []
    async with contextlib.aclosing(client.stream(text, voice=voice)) as stream:
        async for chunk in stream:
            if not chunks:
                first_chunk_s = time.monotonic() - started
            chunks.append(chunk)
    return b"".join(chunks), first_chunk_s, time.monotonic() - started


async def run(voices: list[str], rate: int, out_dir: Path) -> None:
    settings = get_settings()
    if not settings.gnani_api_key.get_secret_value():
        sys.exit("GNANI_API_KEY is not set in .env")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = TimbreConfig(
        api_key=settings.gnani_api_key.get_secret_value(),
        base_url=settings.gnani_tts_url,
        language=settings.gnani_tts_lang,
        sample_rate=rate,
        speed=settings.gnani_tts_speed,
    )
    async with TimbreClient(config) as client:
        for voice in voices:
            for suffix, text in (("", SCRIPT), ("_digits", DIGITS)):
                try:
                    pcm, first_s, total_s = await render(client, voice, text)
                except TimbreError as exc:
                    print(f"{voice}{suffix}: FAILED {exc}")
                    continue
                path = out_dir / f"{voice}{suffix}_{rate // 1000}k.wav"
                write_wav(path, pcm, rate)
                audio_s = len(pcm) / (2 * rate)
                print(
                    f"{voice}{suffix}: first audio {first_s * 1000:.0f} ms, "
                    f"total {total_s:.2f} s for {audio_s:.1f} s of speech -> {path.name}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voices", nargs="+", default=DEFAULT_VOICES)
    parser.add_argument("--rate", type=int, default=16000, choices=sorted(SAMPLE_RATES))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "samples" / "tts")
    args = parser.parse_args()
    asyncio.run(run(args.voices, args.rate, args.out))


if __name__ == "__main__":
    main()
