import asyncio
import contextlib
import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Path, Query, WebSocket, WebSocketDisconnect

from audio.outbox import Outbox
from audio.speaker import Speaker
from audio.tts_cache import AudioCache, CachedSpeech
from config import Settings, get_settings
from dialogue.decision import Decider
from dialogue.machine import Dialogue, Reply, State
from dialogue.standin import KeywordDecider
from dialogue.templates import load_templates
from dialogue.turns import TURN_PAUSE_S, TurnCollector
from gnani.prisma import PrismaConfig, PrismaError, PrismaStream, SttEvent, load_bias_words
from gnani.timbre import TimbreClient, TimbreConfig
from tickets.store import FIRST_NUMBER, TicketStore

logger = logging.getLogger("dhwani.stt")

# How often transcripts past TRANSCRIPT_RETENTION_DAYS are deleted.
RETENTION_SWEEP_S = 3600.0


@functools.cache
def get_ticket_store() -> TicketStore:
    return TicketStore(get_settings().database_path)


async def _sweep_transcripts() -> None:
    while True:
        days = get_settings().transcript_retention_days
        try:
            purged = await asyncio.to_thread(get_ticket_store().purge_transcripts, days)
        except Exception:
            # One failed pass (a locked database) must not end retention for good.
            logger.exception("Deleting old caller turns failed; trying again later")
        else:
            if purged:
                logger.info("Deleted %d caller turns older than %d days", purged, days)
        await asyncio.sleep(RETENTION_SWEEP_S)


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    sweeper = asyncio.create_task(_sweep_transcripts())
    try:
        yield
    finally:
        sweeper.cancel()
        await asyncio.gather(sweeper, return_exceptions=True)


app = FastAPI(title="Dhwani Desk", version="0.1.0", lifespan=lifespan)

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
# Keyword rules until the fine-tuned Laya checkpoint is served (#15). One per process: a model
# behind it should load once, not on every call.
decider: Decider = KeywordDecider()
turn_pause_s = TURN_PAUSE_S
# After the last line of a closed call has played, wait this long before hanging up.
HANG_UP_GRACE_S = 0.5

TEMPLATES = load_templates()
TTS_CACHE: AudioCache = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.get("/api/tickets")
def list_tickets(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    """The newest tickets first, for the supervisor console."""
    return get_ticket_store().tickets(limit)


@app.get("/api/tickets/{number}")
def get_ticket(number: int = Path(ge=FIRST_NUMBER, le=2**31)) -> dict[str, Any]:
    if (ticket := get_ticket_store().ticket(number)) is None:
        raise HTTPException(status_code=404, detail="No such ticket")
    return ticket


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
    frames, `agent_audio_end`, and `tts_error` (the call continues). When the agent closes the
    call, `call_ended` (`reason`) follows its last line and the server hangs up.
    """
    await websocket.accept()
    settings = get_settings()
    if not settings.gnani_api_key.get_secret_value():
        await _fail(websocket, "GNANI_API_KEY is not set")
        return
    outbox = Outbox(websocket)
    store = get_ticket_store()
    dialogue = Dialogue(TEMPLATES, decider, store)
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

        background: list[asyncio.Task[None]] = []
        agent_hung_up = asyncio.Event()

        async def hang_up(reason: str | None) -> None:
            await speaker.wait_until_heard()
            await asyncio.sleep(HANG_UP_GRACE_S)
            await outbox.json({"type": "call_ended", "reason": reason})
            # Stops the relay, which closes Prisma; the socket is closed once that is done, so
            # a late Prisma event is never sent down a closed socket.
            agent_hung_up.set()

        async def answer(reply: Reply) -> None:
            if not reply.keys:
                return
            await outbox.json(reply.to_client(dialogue.slots))
            for line in reply.lines:
                speaker.say(line)
            if reply.state is State.CLOSED:
                background.append(asyncio.create_task(hang_up(reply.reason)))

        async def answer_turns() -> None:
            while True:
                transcript = dialogue.hear(await turns.turns.get())
                if transcript is None:
                    continue
                try:
                    # Only the decision runs in a thread (the decider may be slow); the dialogue
                    # changes here, on the event loop, so a call that ends while a decision is
                    # running files nothing.
                    decision = await asyncio.to_thread(dialogue.decider.decide, transcript)
                except Exception:
                    logger.exception("Deciding a caller turn failed; handing off")
                    reply = dialogue.decider_failed()
                    # No decision to say whether it holds a personal number: mask as if it did.
                    personal_id = 1.0
                else:
                    reply = dialogue.respond(decision)
                    personal_id = decision.contains_personal_id
                # Off the event loop: a disk write must not stall the audio of other calls.
                await asyncio.to_thread(
                    store.record_turn,
                    dialogue.call_id,
                    len(dialogue.turns),
                    dialogue.turns[-1],
                    personal_id,
                )
                await answer(reply)

        async def converse(event: SttEvent) -> None:
            if agent_hung_up.is_set():
                return  # nothing follows call_ended
            await outbox.json(event.to_client())
            if event.type == "connected":
                await answer(dialogue.greet())
            elif event.type == "transcript":
                turns.add(event.text)

        background.append(asyncio.create_task(answer_turns()))
        try:
            final = await _stt_session(
                websocket, settings, outbox, config, converse, stop=agent_hung_up
            )
        finally:
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
            await turns.aclose()
            await speaker.aclose()
    if final:
        await _send_and_close(websocket, *final)
    elif agent_hung_up.is_set():
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close(code=1000)


async def _stt_session(
    websocket: WebSocket,
    settings: Settings,
    outbox: Outbox,
    config: dict[str, Any],
    on_event: EventHandler,
    stop: asyncio.Event | None = None,
) -> FinalMessage | None:
    """Run one Prisma session for this socket. Returns the message to end the session with,
    or None when the browser left or `stop` was set (the caller has nothing more to send)."""
    try:
        async with stt_stream_factory(settings) as stream:
            await outbox.json(config)
            ended_by = await _relay(websocket, stream, on_event, stop)
            if ended_by in ("browser", "agent"):
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


async def _relay(
    websocket: WebSocket,
    stream: PrismaStream,
    on_event: EventHandler,
    stop: asyncio.Event | None = None,
) -> str:
    """Pump audio up and events down until one side ends or `stop` is set; returns "browser",
    "prisma" or "agent"."""

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
    stopped = asyncio.create_task((stop or asyncio.Event()).wait())
    tasks = (audio, events, stopped)
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        if audio in done:
            return "browser"
        return "prisma" if events in done else "agent"
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _send_and_close(websocket: WebSocket, payload: dict[str, Any], code: int) -> None:
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        await websocket.send_json(payload)
        await websocket.close(code=code)


async def _fail(websocket: WebSocket, message: str) -> None:
    await _send_and_close(websocket, {"type": "error", "message": message}, code=1011)
