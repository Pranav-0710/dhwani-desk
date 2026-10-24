"""Phone the running backend's /ws/call with synthetic caller speech and print the call.

Each caller line is synthesised with Timbre, then streamed in real time like a browser
microphone: silence while the agent speaks, the caller's audio when it is their turn. Prints
every transcript, dialogue decision and agent line with timings, so the real Prisma
segmentation, turn assembly and decider are exercised together.

Usage, from backend/ with the server running (uv run uvicorn app:app --port 8000):
    uv run python -m scripts.call_smoke
    uv run python -m scripts.call_smoke "ಕಸದ ಗಾಡಿ ಒಂದು ವಾರದಿಂದ ಬಂದಿಲ್ಲ" "ಕುಂಬಾರಗೇರಿ"

Reads GNANI_API_KEY from the repo-root .env. Costs a few seconds of STT and TTS credits.
"""

import argparse
import asyncio
import json
import sys
import time

import websockets

from config import get_settings
from gnani.timbre import TimbreClient, TimbreConfig

# Road damage with no location, then the location: the agent should ask, then file a ticket.
DEFAULT_LINES = [
    "ನಮ್ಮ ರೋಡ್ ತುಂಬಾ ಹಾಳಾಗಿದೆ ದೊಡ್ಡ ಗುಂಡಿ ಬಿದ್ದಿದೆ",
    "ಗಾಂಧಿನಗರ ಸರ್ಕಾರಿ ಆಸ್ಪತ್ರೆ ಹತ್ತಿರ",
]
CALLER_VOICE = "Kavin"  # a different voice from the agent's, so the two are easy to tell apart
TRAILING_SILENCE_S = 1.5  # after each line, so Prisma's VAD closes the segment
REPLY_TIMEOUT_S = 30.0


async def synthesise(lines: list[str], rate: int) -> list[bytes]:
    settings = get_settings()
    config = TimbreConfig(
        api_key=settings.gnani_api_key.get_secret_value(),
        base_url=settings.gnani_tts_url,
        voice=CALLER_VOICE,
        language=settings.gnani_tts_lang,
        sample_rate=rate,
    )
    async with TimbreClient(config) as tts:
        return [await tts.synthesize(line) for line in lines]


async def call(url: str, origin: str, lines: list[str]) -> None:
    settings = get_settings()
    rate = settings.gnani_stt_sample_rate
    speech = await synthesise(lines, rate)
    started = time.monotonic()

    def log(text: str) -> None:
        print(f"{time.monotonic() - started:6.2f}s  {text}", flush=True)

    expected_ends, ends, replies, state, hung_up = 0, 0, 0, "listening", False
    caller_done_at: float | None = None
    talk: asyncio.Queue[bytes] = asyncio.Queue()

    async with websockets.connect(url, origin=origin, max_size=None) as socket:

        async def microphone() -> None:
            chunk = rate * 2 // 10  # 100 ms of 16-bit mono
            pending = b""
            while True:
                if not pending and not talk.empty():
                    pending = talk.get_nowait() + b"\x00" * int(rate * 2 * TRAILING_SILENCE_S)
                frame, pending = (
                    (pending[:chunk], pending[chunk:]) if pending else (bytes(chunk), b"")
                )
                await socket.send(frame)
                await asyncio.sleep(0.1)

        async def listen() -> None:
            nonlocal expected_ends, ends, replies, state, hung_up
            first_audio_logged = True
            async for message in socket:
                if isinstance(message, bytes):
                    if not first_audio_logged and caller_done_at is not None:
                        log(
                            f"first agent audio, {time.monotonic() - caller_done_at:.2f}s "
                            "after the caller stopped"
                        )
                        first_audio_logged = True
                    continue
                event = json.loads(message)
                kind = event["type"]
                if kind == "transcript":
                    log(f"caller (Prisma): {event['text']}")
                elif kind == "dialogue":
                    replies += 1
                    expected_ends += len(event["keys"])
                    state = event["state"]
                    first_audio_logged = False
                    decision = event["decision"] or {}
                    log(
                        f"dialogue: {event['keys']} → {state}"
                        f" slots={event['slots']} ticket={event['ticket']}"
                        f" department={decision.get('department')}"
                        f"@{decision.get('department_confidence')}"
                    )
                elif kind == "agent_text":
                    log(f"agent: {event['text']}")
                elif kind == "agent_audio_end":
                    ends += 1
                elif kind == "call_ended":
                    hung_up = True
                    log(f"the agent hung up ({event['reason']})")
                elif kind in ("tts_error", "error", "closed"):
                    log(f"{kind}: {event}")

        async def until(condition, what: str) -> None:
            try:
                async with asyncio.timeout(REPLY_TIMEOUT_S):
                    while not condition():
                        await asyncio.sleep(0.05)
            except TimeoutError:
                sys.exit(f"timed out waiting for {what}")

        mic = asyncio.create_task(microphone())
        listener = asyncio.create_task(listen())
        await until(lambda: replies >= 1 and ends >= expected_ends, "the greeting")
        for line, pcm in zip(lines, speech, strict=True):
            if state != "listening":
                break
            before = replies
            log(f"caller says: {line}")
            talk.put_nowait(pcm)
            caller_done_at = time.monotonic() + len(pcm) / (rate * 2)
            await until(lambda before=before: replies > before, "the agent's answer")
            await until(lambda: ends >= expected_ends, "the agent to finish speaking")
        if state == "closed":
            await until(lambda: hung_up, "the agent to hang up")
        log(f"call ends in state {state}")
        mic.cancel()
        listener.cancel()
        await asyncio.gather(mic, listener, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lines", nargs="*", help="caller lines in Kannada, one per turn")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/call")
    # The backend only lets pages on ALLOWED_ORIGINS in; call as the first of them.
    first_origin = get_settings().allowed_origins.split(",")[0]
    parser.add_argument("--origin", default=first_origin, help="Origin header to send")
    args = parser.parse_args()
    if not get_settings().gnani_api_key.get_secret_value():
        sys.exit("GNANI_API_KEY is not set in .env")
    asyncio.run(call(args.url, args.origin, args.lines or DEFAULT_LINES))


if __name__ == "__main__":
    main()
