"""Stream a WAV file (or silence) to Prisma and print what comes back.

Usage, from backend/:
    uv run python -m scripts.stt_smoke path/to/audio.wav [--format verbatim|transcribe]
    uv run python -m scripts.stt_smoke            # 3 s of silence: checks the key and handshake

The WAV must be 16-bit mono at 8 kHz or 16 kHz. Reads GNANI_API_KEY from the repo-root .env.
"""

import argparse
import asyncio
import sys
import wave
from pathlib import Path

from config import get_settings
from gnani.prisma import PrismaConfig, PrismaStream, load_bias_words


def read_pcm(path: Path | None) -> tuple[bytes, int]:
    if path is None:
        return b"\x00" * 16000 * 2 * 3, 16000
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            sys.exit("WAV must be 16-bit mono")
        if wav.getframerate() not in (8000, 16000):
            sys.exit("WAV must be 8 kHz or 16 kHz")
        return wav.readframes(wav.getnframes()), wav.getframerate()


async def run(path: Path | None, output_format: str, wait_s: float) -> None:
    settings = get_settings()
    if not settings.gnani_api_key.get_secret_value():
        sys.exit("GNANI_API_KEY is not set in .env")
    pcm, rate = read_pcm(path)
    config = PrismaConfig(
        api_key=settings.gnani_api_key.get_secret_value(),
        url=settings.gnani_stt_url,
        lang_code=settings.gnani_stt_lang,
        sample_rate=rate,
        output_format=output_format,  # type: ignore[arg-type]
        bias_words=load_bias_words(settings.gnani_stt_bias_file),
        bias_score=settings.gnani_stt_bias_score,
    )
    async with PrismaStream(config) as stream:

        async def listen() -> None:
            async for event in stream.events():
                print(event.to_client(), flush=True)

        listener = asyncio.create_task(listen())
        # One second of trailing silence lets the server's VAD close the last segment.
        audio = pcm + b"\x00" * rate * 2
        # Feed 100 ms at a time in real time, as a microphone would.
        chunk = rate * 2 // 10
        for start in range(0, len(audio), chunk):
            await stream.send_audio(audio[start : start + chunk])
            await asyncio.sleep(0.1)
        await stream.flush()
        await asyncio.sleep(wait_s)
        listener.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wav", nargs="?", type=Path)
    parser.add_argument("--format", choices=["verbatim", "transcribe"], default="transcribe")
    parser.add_argument("--wait", type=float, default=3.0, help="seconds to wait after audio")
    args = parser.parse_args()
    asyncio.run(run(args.wav, args.format, args.wait))


if __name__ == "__main__":
    main()
