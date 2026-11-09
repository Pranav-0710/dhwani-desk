"""Transcribe the recorded test set with Prisma, clean and on a phone line (#13, #19).

For every `eval/out/audio/<id>.wav` (record.py), streams the audio to Prisma in real time, as
the call does, twice: at 16 kHz as recorded, and at 8 kHz with the phone page's heavy line noise
(line_audio.phone_line). Writes `eval/out/transcripts.jsonl`, one row per sentence:
`{id, sha256_16, prisma_16k, prisma_8k_noise}`, the hash naming the version of the sentence that
was read (record.py). Runs again pick up where they stopped, and redo a sentence re-recorded since.
A failed or empty transcript is not saved, so the next run tries it again.

Harsher lines (#88), each 8 kHz: `ulaw` through a G.711 mu-law codec with heavy noise, `loss`
with 3% of 20 ms packets lost and heavy noise, and `loud` with noise as loud as the speech.

`phone` is a real mobile call (#100): the clips phone_call.py cut from the far end's recording,
in `out/audio_phone/`. A clip the call did not carry, or cut for another version of the sentence,
is skipped. Each row keeps the cut its phone transcript came from (`prisma_phone_call_cut`), so a
new cut is transcribed again and a sentence the new call did not carry loses the old transcript.

`--model prisma-v3.0` transcribes with Prisma v3.0 into `transcripts-prisma-v3.0.jsonl`, so the
two models are scored on the same audio.

Usage, from backend/: uv run python ../eval/transcribe.py [--only 16k|8k|ulaw|loss|loud|phone]
    [--model prisma-v3.0]
Costs Prisma credits: about 150 clips x each condition x their length.
"""

import argparse
import asyncio
import contextlib
import json
import random
import sys
from pathlib import Path

from common import OUT, read_jsonl
from line_audio import drop_packets, mu_law, phone_line, read_wav, to_pcm16, write_wav

AUDIO = OUT / "audio"
TRANSCRIPTS = OUT / "transcripts.jsonl"
CHUNK_S = 0.1
NAMES = {
    "16k": "prisma_16k",
    "8k": "prisma_8k_noise",
    "ulaw": "prisma_8k_ulaw_noise",
    "loss": "prisma_8k_loss_noise",
    "loud": "prisma_8k_loud_noise",
    "phone": "prisma_phone_call",
}
# Clips cut from a real mobile call's far end by phone_call.py (#100), 8 kHz.
PHONE_AUDIO = OUT / "audio_phone"
PHONE, PHONE_CUT = "prisma_phone_call", "prisma_phone_call_cut"
PACKET_LOSS = 0.03
MODELS = ("prisma-v2.5", "prisma-v3.0")
TRAILING_SILENCE_S = 1.5  # lets Prisma's VAD close the last segment
WAIT_AFTER_S = 3.0


class NoTranscript(Exception):
    """Prisma reported an error or returned nothing."""


def transcripts_file(model: str) -> Path:
    """v2.5, the default, keeps the original file name."""
    return TRANSCRIPTS if model == "prisma-v2.5" else OUT / f"transcripts-{model}.jsonl"


async def transcribe(pcm: bytes, rate: int, model: str = "prisma-v2.5") -> str:
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
        model="" if model == "prisma-v2.5" else model,
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


def conditions(
    path: Path, wanted: list[str], phone: Path | None = None
) -> dict[str, tuple[bytes, int]]:
    """Each wanted version of the clip as (PCM, rate); `phone` is the real call's clip, if any."""
    samples, rate = read_wav(path)
    out: dict[str, tuple[bytes, int]] = {}
    if "prisma_16k" in wanted:
        out["prisma_16k"] = (to_pcm16(samples), rate)
    if "prisma_8k_noise" in wanted:
        line = phone_line(samples, rate, "heavy", seed=sum(map(ord, path.stem)))
        write_wav(OUT / "audio_8k_noise" / path.name, line, 8000)  # to listen to what Prisma got
        out["prisma_8k_noise"] = (to_pcm16(line), 8000)
    seed = sum(map(ord, path.stem))
    if "prisma_8k_ulaw_noise" in wanted:
        coded = mu_law(phone_line(samples, rate, "heavy", seed))
        out["prisma_8k_ulaw_noise"] = (to_pcm16(coded), 8000)
    if "prisma_8k_loss_noise" in wanted:
        line = phone_line(samples, rate, "heavy", seed)
        lost = drop_packets(line, 8000, PACKET_LOSS, random.Random(seed))
        out["prisma_8k_loss_noise"] = (to_pcm16(lost), 8000)
    if "prisma_8k_loud_noise" in wanted:
        out["prisma_8k_loud_noise"] = (to_pcm16(phone_line(samples, rate, "loud", seed)), 8000)
    # Only the clips the call carried: one lost on the line is left out, not retried.
    if PHONE in wanted and phone is not None:
        call, phone_rate = read_wav(phone)
        out[PHONE] = (to_pcm16(call), phone_rate)
    return out


def phone_clip(clip_id: str, text_hash: str, folder: Path = PHONE_AUDIO) -> tuple[Path, str] | None:
    """The real call's clip of this version of the sentence and the cut it came from, or None
    when the call did not carry it or carried another version."""
    wav, sidecar = folder / f"{clip_id}.wav", folder / f"{clip_id}.json"
    if not (wav.exists() and sidecar.exists()):
        return None
    meta = json.loads(sidecar.read_text("utf-8"))
    return (wav, meta["cut"]) if meta.get("sha256_16") == text_hash else None


def forget_stale_phone(row: dict, phone: tuple[Path, str] | None) -> None:
    """Drop a phone transcript from another cut, or one the current cut does not carry."""
    if PHONE in row and (phone is None or row.get(PHONE_CUT) != phone[1]):
        row.pop(PHONE)
        row.pop(PHONE_CUT, None)


async def run(only: str | None, model: str = "prisma-v2.5") -> None:
    from gnani.prisma import PrismaError

    clips = sorted(AUDIO.glob("*.wav"))
    if not clips:
        raise SystemExit(f"No recordings in {AUDIO}; run record.py first")
    target = transcripts_file(model)
    done = {row["id"]: row for row in read_jsonl(target)} if target.exists() else {}
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
        phone = phone_clip(clip.stem, text_hash) if PHONE in wanted else None
        if PHONE in wanted:
            forget_stale_phone(row, phone)
        if not (todo := [c for c in wanted if c not in row]):
            continue
        for condition, (pcm, rate) in conditions(clip, todo, phone and phone[0]).items():
            try:
                row[condition] = await transcribe(pcm, rate, model)
                if condition == PHONE and phone is not None:
                    row[PHONE_CUT] = phone[1]
            except (NoTranscript, PrismaError) as error:
                # A dropped or refused Prisma session fails this clip only; the next run
                # retries it.
                failed += 1
                print(f"[{number}/{len(clips)}] {clip.stem} {condition}: FAILED, {error}")
                continue
            print(f"[{number}/{len(clips)}] {clip.stem} {condition}: {row[condition]}", flush=True)
            # Saved after every clip, so a stopped run loses nothing.
            target.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in done.values()), "utf-8"
            )
    if failed:
        raise SystemExit(f"{failed} transcripts failed; run again to retry them")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=list(NAMES), help="one condition only")
    parser.add_argument("--model", choices=MODELS, default="prisma-v2.5", help="Prisma model")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(run(args.only, args.model))


if __name__ == "__main__":
    main()
