"""Transcribe the recorded test set with Prisma, clean and on a phone line (#13, #19).

For every `eval/out/audio/<id>.wav` (record.py), streams the audio to Prisma in real time, as
the call does, twice: at 16 kHz as recorded, and at 8 kHz with the phone page's heavy line noise
(line_audio.phone_line). Writes `eval/out/transcripts.jsonl`, one row per sentence:
`{id, sha256_16, prisma_16k, prisma_8k_noise}`, the hash naming the version of the sentence that
was read (record.py). Runs again pick up where they stopped, and redo a sentence re-recorded since.
A failed or empty transcript is not saved, so the next run tries it again.

Usage, from backend/: uv run python ../eval/transcribe.py [--only 16k|8k]
Costs Prisma credits: about 150 clips x 2 x their length.
"""

import argparse
import asyncio
import contextlib
import json
from pathlib import Path

from common import OUT, read_jsonl
from line_audio import phone_line, read_wav, to_pcm16, write_wav

AUDIO = OUT / "audio"
TRANSCRIPTS = OUT / "transcripts.jsonl"
CHUNK_S = 0.1
NAMES = {"16k": "prisma_16k", "8k": "prisma_8k_noise"}
TRAILING_SILENCE_S = 1.5  # lets Prisma's VAD close the last segment
WAIT_AFTER_S = 3.0


class NoTranscript(Exception):
    """Prisma reported an error or returned nothing."""


async def transcribe(pcm: bytes, rate: int) -> str:
    """Prisma's transcript of `pcm`, streamed in real time; segments joined in order."""
    from config import get_settings
    from gnani.prisma import PrismaConfig, PrismaStream, load_bias_words

    settings = get_settings()
    config = PrismaConfig(
        api_key=settings.gnani_api_key.get_secret_value(),
        url=settings.gnani_stt_url,
        lang_code=settings.gnani_stt_lang,
        sample_rate=rate,
        output_format=settings.gnani_stt_format,
        bias_words=load_bias_words(settings.gnani_stt_bias_file),
        bias_score=settings.gnani_stt_bias_score,
    )
    segments: list[str] = []
    errors: list[str] = []
    async with PrismaStream(config) as stream:

        async def listen() -> None:
            async for event in stream.events():
                if event.type == "transcript" and event.text:
                    segments.append(event.text.strip())
                elif event.type == "error":
                    errors.append(event.message or "error")

        listener = asyncio.create_task(listen())
        audio = pcm + b"\x00" * int(rate * 2 * TRAILING_SILENCE_S)
        chunk = int(rate * 2 * CHUNK_S)
        for start in range(0, len(audio), chunk):
            await stream.send_audio(audio[start : start + chunk])
            await asyncio.sleep(CHUNK_S)
        await stream.flush()
        await asyncio.sleep(WAIT_AFTER_S)
        listener.cancel()
        # Raises what the listener failed with, if it failed before it was cancelled.
        with contextlib.suppress(asyncio.CancelledError):
            await listener
    if errors:
        raise NoTranscript("; ".join(errors))
    if not (text := " ".join(s for s in segments if s)):
        raise NoTranscript("no transcript")
    return text


def conditions(path: Path, wanted: list[str]) -> dict[str, tuple[bytes, int]]:
    samples, rate = read_wav(path)
    out: dict[str, tuple[bytes, int]] = {}
    if "prisma_16k" in wanted:
        out["prisma_16k"] = (to_pcm16(samples), rate)
    if "prisma_8k_noise" in wanted:
        line = phone_line(samples, rate, "heavy", seed=sum(map(ord, path.stem)))
        write_wav(OUT / "audio_8k_noise" / path.name, line, 8000)  # to listen to what Prisma got
        out["prisma_8k_noise"] = (to_pcm16(line), 8000)
    return out


async def run(only: str | None) -> None:
    clips = sorted(AUDIO.glob("*.wav"))
    if not clips:
        raise SystemExit(f"No recordings in {AUDIO}; run record.py first")
    done = {row["id"]: row for row in read_jsonl(TRANSCRIPTS)} if TRANSCRIPTS.exists() else {}
    wanted = [NAMES[only]] if only else list(NAMES.values())
    failed = 0
    for number, clip in enumerate(clips, 1):
        sidecar = clip.with_suffix(".json")
        if not sidecar.exists():
            raise SystemExit(f"{sidecar} is missing: record {clip.stem} again with record.py")
        text_hash = json.loads(sidecar.read_text("utf-8"))["sha256_16"]
        if done.get(clip.stem, {}).get("sha256_16") != text_hash:
            done[clip.stem] = {"id": clip.stem, "sha256_16": text_hash}
        row = done[clip.stem]
        if not (todo := [c for c in wanted if c not in row]):
            continue
        for condition, (pcm, rate) in conditions(clip, todo).items():
            try:
                row[condition] = await transcribe(pcm, rate)
            except NoTranscript as error:
                failed += 1
                print(f"[{number}/{len(clips)}] {clip.stem} {condition}: FAILED, {error}")
                continue
            print(f"[{number}/{len(clips)}] {clip.stem} {condition}: {row[condition]}", flush=True)
            # Saved after every clip, so a stopped run loses nothing.
            TRANSCRIPTS.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in done.values()), "utf-8"
            )
    if failed:
        raise SystemExit(f"{failed} transcripts failed; run again to retry them")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=list(NAMES), help="one condition only")
    args = parser.parse_args()
    asyncio.run(run(args.only))


if __name__ == "__main__":
    main()
