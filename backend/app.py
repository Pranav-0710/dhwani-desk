import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from audio.outbox import Outbox
from audio.speaker import Speaker
from audio.tts_cache import AudioCache, CachedSpeech
from config import Settings, get_settings
from dialogue.decision import Decider
from dialogue.machine import CountingDesk, Dialogue, Reply
from dialogue.standin import KeywordDecider
from dialogue.templates import load_templates
from dialogue.turns import TURN_PAUSE_S, TurnCollector
from gnani.prisma import PrismaConfig, PrismaError, PrismaStream, SttEvent, load_bias_words
from gnani.timbre import TimbreClient, TimbreConfig

logger = logging.getLogger("dhwani.stt")

app = FastAPI(title="Dhwani Desk", version="0.1.0")

EventHandler = Callable[[SttEvent], Awaitable[None]]
FinalMessage = tuple[dict[str, Any], int]


def create_stt_stream(settings: Settings) -> PrismaStream:
    return PrismaStream(
        PrismaConfig(
            api_key=settings.gnani_api_key.get_secret_value(),
            url=settings.gnani_stt_url,
            lang_code=settings.gnani_stt_lang,
            sample_rate=settings.gnani_stt_sample_rate,
            output_format=settings.gnani_stt_format,
            bias_words=load_bias_words(settings.gnani_stt_bias_file),
            bias_score=settings.gnani_stt_bias_score,
        )
    )


def create_tts_client(settings: Settings) -> TimbreClient:
    return TimbreClient(
        TimbreConfig(
            api_key=settings.gnani_api_key.get_secret_value(),
            base_url=settings.gnani_tts_url,
            voice=settings.gnani_tts_voice,
            language=settings.gnani_tts_lang,
            sample_rate=settings.gnani_tts_sample_rate,
            speed=settings.gnani_tts_speed,
        )
    )


stt_stream_factory: Callable[[Settings], PrismaStream] = create_stt_stream
tts_client_factory: Callable[[Settings], TimbreClient] = create_tts_client
# Keyword rules until the fine-tuned Laya checkpoint is served (#15).
decider_factory: Callable[[], Decider] = KeywordDecider
turn_pause_s = TURN_PAUSE_S

TEMPLATES = load_templates()
TTS_CACHE: AudioCache = {}
TICKETS = CountingDesk()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.websocket("/ws/stt")
async def stt_proxy(websocket: WebSocket) -> None:
    """Relay browser audio to Prisma and Prisma's transcripts back to the browser.

    Contract (see backend/README.md): after `config` and `connected` arrive, the browser
    streams binary frames of 16-bit mono PCM at `config.sample_rate`, in real time, and may
    send the text frame "flush". It receives JSON events; `closed` or `error` followed by a
    close ends the session.
    """
    await websocket.accept()
    settings = get_settings()
    if not settings.gnani_api_key.get_secret_value():
        await _fail(websocket, "GNANI_API_KEY is not set")
        return
    outbox = Outbox(websocket)

    async def forward(event: SttEvent) -> None:
        await outbox.json(event.to_client())

    config = {"type": "config", "sample_rate": settings.gnani_stt_sample_rate}
    final = await _stt_session(websocket, settings, outbox, config, forward)
    if final:
        await _send_and_close(websocket, *final)


@app.websocket("/ws/call")
async def call(websocket: WebSocket) -> None:
    """`/ws/stt` plus the agent: it greets, then answers every caller turn (dialogue/).

    Extra server → browser messages: `dialogue` (state, slots, ticket and the decision behind
    each reply), `agent_text`, `agent_audio_start` (`sample_rate`), binary 16-bit mono PCM
    frames, `agent_audio_end`, and `tts_error` (the call continues).
    """
    await websocket.accept()
    settings = get_settings()
    if not settings.gnani_api_key.get_secret_value():
        await _fail(websocket, "GNANI_API_KEY is not set")
        return
    outbox = Outbox(websocket)
    dialogue = Dialogue(TEMPLATES, decider_factory(), TICKETS)
    turns = TurnCollector(turn_pause_s)
    config = {
        "type": "config",
        "sample_rate": settings.gnani_stt_sample_rate,
        "tts_sample_rate": settings.gnani_tts_sample_rate,
    }

    async with tts_client_factory(settings) as tts:
        speech = CachedSpeech(
            tts,
            TTS_CACHE,
            TEMPLATES.fixed_lines(),
            settings.gnani_tts_voice,
            settings.gnani_tts_sample_rate,
        )
        speaker = Speaker(speech, outbox, settings.gnani_tts_sample_rate)

        async def answer(reply: Reply) -> None:
            if not reply.keys:
                return
            await outbox.json(reply.to_client(dialogue.slots))
            for line in reply.lines:
                speaker.say(line)

        async def answer_turns() -> None:
            while True:
                text = await turns.turns.get()
                try:
                    # The decider may be slow (Laya on CPU); keep the event loop free.
                    reply = await asyncio.to_thread(dialogue.turn, text)
                except Exception:
                    logger.exception("Deciding a caller turn failed")
                    continue
                await answer(reply)

        async def converse(event: SttEvent) -> None:
            await outbox.json(event.to_client())
            if event.type == "connected":
                await answer(dialogue.greet())
            elif event.type == "transcript":
                turns.add(event.text)

        answering = asyncio.create_task(answer_turns())
        try:
            final = await _stt_session(websocket, settings, outbox, config, converse)
        finally:
            answering.cancel()
            await asyncio.gather(answering, return_exceptions=True)
            await turns.aclose()
            await speaker.aclose()
    if final:
        await _send_and_close(websocket, *final)


async def _stt_session(
    websocket: WebSocket,
    settings: Settings,
    outbox: Outbox,
    config: dict[str, Any],
    on_event: EventHandler,
) -> FinalMessage | None:
    """Run one Prisma session for this socket. Returns the message to end the session with,
    or None when the browser left (nobody to tell)."""
    try:
        async with stt_stream_factory(settings) as stream:
            await outbox.json(config)
            ended_by = await _relay(websocket, stream, on_event)
            if ended_by == "browser":
                stream.discard_pending()
                return None
        closed = {
            "type": "closed",
            "by": "prisma",
            "code": stream.close_code,
            "reason": stream.close_reason,
        }
        return closed, 1000
    except PrismaError as exc:
        logger.warning("STT session failed: %s", exc)
        return {"type": "error", "message": str(exc)}, 1011
    except WebSocketDisconnect:
        return None


async def _relay(websocket: WebSocket, stream: PrismaStream, on_event: EventHandler) -> str:
    """Pump audio up and events down until one side ends; returns which side ended."""

    async def pump_audio() -> None:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes"):
                await stream.send_audio(message["bytes"])
            elif message.get("text") == "flush":
                await stream.flush()

    async def pump_events() -> None:
        async for event in stream.events():
            await on_event(event)

    audio = asyncio.create_task(pump_audio())
    events = asyncio.create_task(pump_events())
    try:
        done, _ = await asyncio.wait({audio, events}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        return "browser" if audio in done else "prisma"
    finally:
        for task in (audio, events):
            task.cancel()
        await asyncio.gather(audio, events, return_exceptions=True)


async def _send_and_close(websocket: WebSocket, payload: dict[str, Any], code: int) -> None:
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        await websocket.send_json(payload)
        await websocket.close(code=code)


async def _fail(websocket: WebSocket, message: str) -> None:
    await _send_and_close(websocket, {"type": "error", "message": message}, code=1011)
