"""Speak the test set with Timbre's voices, in place of human recordings (#13).

Writes `eval/out/audio/<id>.wav` (16 kHz) and `<id>.json` as record.py does, so transcribe.py and
run.py treat them alike; the sidecar also names the voice and speed. Voices take turns, and the
agent's own voice (Saanvi) is left out, so Prisma never hears the agent reading its own words.
Sentences that already have audio for their current text are skipped.

Synthetic speech is cleaner and more regular than a caller's, so a report made from it says so:
it shows what the line and Prisma cost, not how real callers sound.

Usage, from backend/: uv run python ../eval/synthesize.py --test-file ../eval/out/test.jsonl
Costs Timbre credits: one request per sentence.
"""

import argparse
import asyncio
import json
import wave
from pathlib import Path

from common import OUT, fingerprint, read_jsonl

AUDIO = OUT / "audio"
RATE = 16000
# The two Kannada voices and three others that spoke it well (docs/decisions/tts-voice.md).
VOICES = ("Kavin", "Chitra", "Lavanya", "Kaveri")
SPEEDS = (1.0, 0.9, 1.1)


def voice_for(number: int) -> tuple[str, float]:
    """The voice and speed for the `number`th sentence: every voice meets every speed."""
    return VOICES[number % len(VOICES)], SPEEDS[(number // len(VOICES)) % len(SPEEDS)]


def done(row: dict) -> bool:
    sidecar = AUDIO / f"{row['id']}.json"
    if not (AUDIO / f"{row['id']}.wav").exists() or not sidecar.exists():
        return False
    return json.loads(sidecar.read_text("utf-8")).get("sha256_16") == fingerprint(row["text"])


async def run(rows: list[dict]) -> None:
    import httpx2
    from config import get_settings
    from gnani.timbre import TimbreClient, TimbreConfig

    settings = get_settings()
    AUDIO.mkdir(parents=True, exist_ok=True)
    todo = [(n, r) for n, r in enumerate(rows) if not done(r)]
    print(f"{len(rows) - len(todo)} of {len(rows)} done; {len(todo)} to go.")
    for number, row in todo:
        voice, speed = voice_for(number)
        config = TimbreConfig(
            api_key=settings.gnani_api_key.get_secret_value(),
            base_url=settings.gnani_tts_url,
            voice=voice,
            language=settings.gnani_tts_lang,
            sample_rate=RATE,
            speed=speed,
        )
        async with TimbreClient(config, timeout=httpx2.Timeout(30.0)) as tts:
            pcm = await tts.synthesize(row["text"])
        with wave.open(str(AUDIO / f"{row['id']}.wav"), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(RATE)
            out.writeframes(pcm)
        sidecar = {"sha256_16": fingerprint(row["text"]), "voice": voice, "speed": speed}
        (AUDIO / f"{row['id']}.json").write_text(json.dumps(sidecar) + "\n", "utf-8")
        print(f"{row['id']}: {voice} x{speed}, {len(pcm) / 2 / RATE:.1f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-file", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(read_jsonl(args.test_file)))


if __name__ == "__main__":
    main()
