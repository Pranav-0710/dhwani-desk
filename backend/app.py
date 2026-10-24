import asyncio
import contextlib
import functools
import hmac
import logging
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

from fastapi import FastAPI, HTTPException, Path, Query, WebSocket, WebSocketDisconnect

from audio.outbox import Outbox
from audio.speaker import Speaker
from audio.tts_cache import AudioCache, CachedSpeech
from config import Settings, get_settings
from decisions import thresholds
from dialogue.decision import Decider, Decision, Screening
from dialogue.live import LiveCall, LiveCalls, OfficerCommand
from dialogue.machine import Dialogue, OfficerError, Reply, State
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


@functools.cache
def get_decider() -> Decider:
    """One decider per process, loaded at startup: Laya takes seconds to load."""
    settings = get_settings()
    if settings.decider == "laya":
        # Imported here so the keyword decider never needs Laya installed.
        from decisions.laya import LayaDecider

        return LayaDecider.load(settings)
    return KeywordDecider()


async def _purge_old_transcripts() -> None:
    days = get_settings().transcript_retention_days
    try:
        purged = await asyncio.to_thread(get_ticket_store().purge_transcripts, days)
    except Exception:
        # One failed pass (a locked database) must not end retention for good.
        logger.exception("Deleting old caller turns failed; trying again later")
    else:
        if purged:
            logger.info("Deleted %d caller turns older than %d days", purged, days)


async def _sweep_transcripts() -> None:
    while True:
        await asyncio.sleep(RETENTION_SWEEP_S)
        await _purge_old_transcripts()


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    # Both before the first call is accepted: no caller waits for the model, and turns past
    # their retention are gone before any new ones arrive.
    await asyncio.to_thread(get_decider)
    await _purge_old_transcripts()
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
turn_pause_s = TURN_PAUSE_S
# When the rest of a decision takes longer than this after the safety step, the agent says
# "one moment" (`hold_on`) so the caller is not left in silence.
hold_on_after_s = 0.3
# After the last line of a closed call has played, wait this long before hanging up.
HANG_UP_GRACE_S = 0.5
# A console must say hello with the token within this long.
CONSOLE_HELLO_TIMEOUT_S = 5.0
# Longest a console waits for a call to carry out an officer command (a decision may be
# running on the call).
OFFICER_COMMAND_TIMEOUT_S = 10.0

# The calls in progress, and the consoles watching them.
live_calls = LiveCalls()


class SessionCount:
    """Browser sessions streaming to Prisma right now: each one spends our Gnani credits."""

    def __init__(self) -> None:
        self.open = 0

    @contextlib.contextmanager
    def hold(self, limit: int) -> Iterator[bool]:
        """Count one session while the block runs. Yields False, counting nothing, when
        `limit` are already open."""
        if self.open >= limit:
            yield False
            return
        self.open += 1
        try:
            yield True
        finally:
            self.open -= 1


stt_sessions = SessionCount()
LINE_BUSY = {"type": "error", "message": "The line is busy; please try again in a minute"}
# A session that sends no audio this long is ended, so a socket left open and silent cannot
# hold one of the few sessions. The phone page streams all the time, silence included.
stt_quiet_s = 30.0
WENT_QUIET = {"type": "error", "message": "No audio arrived for too long; call ended"}

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


@app.get("/api/thresholds")
def get_thresholds() -> dict[str, Any]:
    """The escalation thresholds, for the console's decision bars."""
    return thresholds.as_dict()


@app.websocket("/ws/console")
async def console(websocket: WebSocket) -> None:
    """A supervisor console: watch every live call, take one over, speak on it, hand it back.

    Contract (see backend/README.md): the console first sends `hello` with `CONSOLE_TOKEN`.
    It then gets `calls` (every live call, and the thresholds), followed by `call_started`,
    `call_event` (a message the caller's browser got) and `call_closed` as they happen. It may
    send `take_over`, `say` (`text`) and `hand_back`, each with a `call_id` and an optional `id`
    echoed in the answer: `done`, or `error` with a `message`.
    """
    await websocket.accept()
    token = get_settings().console_token.get_secret_value()
    if not token:
        await _fail(websocket, "CONSOLE_TOKEN is not set")
        return
    try:
        hello = await asyncio.wait_for(websocket.receive_json(), CONSOLE_HELLO_TIMEOUT_S)
    except (TimeoutError, WebSocketDisconnect, KeyError, ValueError):
        hello = None
    is_hello = isinstance(hello, dict) and hello.get("type") == "hello"
    offered = hello.get("token") if is_hello else None
    if not isinstance(offered, str) or not hmac.compare_digest(offered.encode(), token.encode()):
        await _send_and_close(websocket, {"type": "error", "message": "Not allowed"}, code=1008)
        return

    outbox = Outbox(websocket)
    with live_calls.watch() as watcher:
        calls = [call.summary() for call in live_calls]
        await outbox.json({"type": "calls", "calls": calls, "thresholds": thresholds.as_dict()})

        async def send_events() -> None:
            while (message := await watcher.messages.get()) is not None:
                await outbox.json(message)
            await outbox.json({"type": "error", "message": "Fell too far behind; reconnect"})

        async def take_commands() -> None:
            while True:
                try:
                    message = await websocket.receive_json()
                except (KeyError, ValueError):
                    await outbox.json({"type": "error", "message": "Send JSON text messages"})
                    continue
                await outbox.json(await _carry_out(message))

        tasks = [asyncio.create_task(send_events()), asyncio.create_task(take_commands())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        await websocket.close(code=1000)


async def _carry_out(message: Any) -> dict[str, Any]:
    """Queue one officer command for its call's route and wait until it is carried out."""
    if not isinstance(message, dict):
        return {"type": "error", "message": "Send a JSON object"}
    kind, call_id = message.get("type"), message.get("call_id")
    answer: dict[str, Any] = {"command": kind, "call_id": call_id}
    if isinstance(message.get("id"), str | int):
        answer["id"] = message["id"]
    if kind not in ("take_over", "say", "hand_back"):
        return {"type": "error", "message": "Unknown command", **answer}
    call = live_calls.get(call_id) if isinstance(call_id, str) else None
    if call is None:
        return {"type": "error", "message": "No such call", **answer}
    text = message.get("text", "")
    if not isinstance(text, str):
        return {"type": "error", "message": "`text` must be a string", **answer}
    command = OfficerCommand(kind, text)
    call.commands.put_nowait(command)
    try:
        await asyncio.wait_for(asyncio.shield(command.done), OFFICER_COMMAND_TIMEOUT_S)
    except (OfficerError, LookupError) as error:
        return {"type": "error", "message": str(error), **answer}
    except TimeoutError:
        return {"type": "error", "message": "The call did not answer in time", **answer}
    finally:
        # Not carried out yet (timed out, or the console left): it never will be, so an
        # officer who tries again is not heard twice.
        command.done.cancel()
    return {"type": "done", **answer}


@app.websocket("/ws/stt")
async def stt_proxy(websocket: WebSocket) -> None:
    """Relay browser audio to Prisma and Prisma's transcripts back to the browser.

    Contract (see backend/README.md): only pages on ALLOWED_ORIGINS may connect, and only
    MAX_STT_SESSIONS at once across /ws/stt and /ws/call. After `config` and `connected`
    arrive, the browser streams binary frames of 16-bit mono PCM at `config.sample_rate`, in
    real time, and may send the text frame "flush". It receives JSON events; `closed` or
    `error` followed by a close ends the session.
    """
    settings = get_settings()
    if not await _admit(websocket, settings):
        return
    with stt_sessions.hold(settings.max_stt_sessions) as held:
        if not held:
            await _line_busy(websocket)
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
    each reply; `screening` instead of `decision` for a hand-off on the safety step, followed
    by a `dialogue` with no `keys` once the full decision is in), `agent_text` (including
    "one moment" while a slow decision runs), `agent_audio_start` (`sample_rate`), binary
    16-bit mono PCM frames, `agent_audio_end`, and `tts_error` (the call continues). When the
    agent closes the call, `call_ended` (`reason`) follows its last line and the server hangs
    up.
    """
    settings = get_settings()
    if not await _admit(websocket, settings):
        return
    with stt_sessions.hold(settings.max_stt_sessions) as held:
        if not held:
            await _line_busy(websocket)
            return
        store = get_ticket_store()
        dialogue = Dialogue(TEMPLATES, get_decider(), store)
        # Every message the caller's browser gets is copied to the supervisor consoles.
        watch = functools.partial(live_calls.publish, dialogue.call_id)
        outbox = Outbox(websocket, watch=watch)
        live = live_calls.open(dialogue)
        try:
            await _converse(websocket, settings, outbox, store, dialogue, live)
        finally:
            live_calls.close(dialogue.call_id)


async def _converse(
    websocket: WebSocket,
    settings: Settings,
    outbox: Outbox,
    store: TicketStore,
    dialogue: Dialogue,
    live: LiveCall,
) -> None:
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

        async def answer(reply: Reply, *, show: bool = False) -> None:
            """Send the reply's `dialogue` event and say its lines. One with no lines and no
            decision is sent only when `show` says the state change matters on its own."""
            if not (reply.keys or reply.decision or show):
                return
            await outbox.json(reply.to_client(dialogue.slots, dialogue.officer))
            for line in reply.lines:
                speaker.say(line)
            if reply.state is State.CLOSED:
                background.append(asyncio.create_task(hang_up(reply.reason)))

        async def decide_holding_on(transcript: str, screening: Screening) -> Decision:
            async def hold_on() -> None:
                await asyncio.sleep(hold_on_after_s)
                speaker.say(TEMPLATES.line("hold_on"))

            timer = asyncio.create_task(hold_on())
            try:
                return await asyncio.to_thread(dialogue.decider.decide, transcript, screening)
            finally:
                timer.cancel()

        async def record_turn(personal_id: float) -> None:
            # Off the event loop: a disk write must not stall the audio of other calls.
            await asyncio.to_thread(
                store.record_turn,
                dialogue.call_id,
                len(dialogue.turns),
                dialogue.turns[-1],
                personal_id,
            )

        async def decide(new_turn: bool) -> None:
            """Decide on the whole call and answer, in the steps `Dialogue.turn` takes in one
            go. `new_turn`: the last turn is new and not stored yet (not so when deciding again
            after a hand-back)."""
            transcript = dialogue.transcript

            async def store_turn(personal_id: float) -> None:
                if new_turn:
                    await record_turn(personal_id)

            async def hand_off_on_failure() -> None:
                logger.exception("Deciding a caller turn failed; handing off")
                # No decision says whether the turn holds a personal number: mask as if it did.
                await store_turn(1.0)
                await answer(dialogue.decider_failed())

            async def screen_whole_call(first: Screening | None) -> Screening:
                # The first step already read the whole call, unless it was handed back.
                if first is not None and escalation_text == transcript:
                    return first
                return await asyncio.to_thread(dialogue.decider.screen, transcript)

            # What can hand the call off: the whole call, or only what was said since an
            # officer handed it back (nothing yet, right after a hand-back).
            escalation_text = dialogue.escalation_text()
            try:
                first = None
                if escalation_text:
                    first = await asyncio.to_thread(dialogue.decider.screen, escalation_text)
            except Exception:
                await hand_off_on_failure()
                return
            if first is not None and (handed_off := dialogue.screened(first)) is not None:
                # Safety first: the turn is stored and the hand-off said before the rest is
                # decided, since a caller told to call 112 may hang up at once. Masked as if it
                # held a personal number, as no decision says otherwise yet. The rest follows
                # for the officer as a `dialogue` event with no lines.
                await store_turn(1.0)
                await answer(handed_off)
                try:
                    screening = await screen_whole_call(first)
                    decision = await asyncio.to_thread(
                        dialogue.decider.decide, transcript, screening
                    )
                except Exception:
                    logger.exception("Completing a decision after a hand-off failed")
                    return
                await answer(dialogue.complete(decision))
                return
            try:
                screening = await screen_whole_call(first)
                decision = await decide_holding_on(transcript, screening)
            except Exception:
                await hand_off_on_failure()
                return
            await store_turn(decision.contains_personal_id)
            await answer(dialogue.respond(decision))

        async def obey(command: OfficerCommand) -> None:
            if command.done.done():
                return  # the console stopped waiting for it: too late to carry it out
            try:
                if command.kind == "take_over":
                    await answer(dialogue.take_over(), show=True)
                elif command.kind == "say":
                    speaker.say(dialogue.officer_says(command.text))
                else:
                    await answer(dialogue.hand_back())
                if not command.done.done():
                    command.done.set_result(None)
            except OfficerError as error:
                if not command.done.done():
                    command.done.set_exception(error)
                return
            finally:
                # Cancelled mid-command, or the caller's socket failed: never leave the
                # console waiting.
                if not command.done.done():
                    command.done.set_exception(LookupError("The call has ended"))
            if command.kind == "hand_back" and dialogue.turns:
                # What the caller told the officer counts too.
                await decide(new_turn=False)

        # Caller turns and officer commands, one at a time: only this task changes the
        # dialogue, so a hand-back never races a decision on a caller turn.
        work: asyncio.Queue[str | OfficerCommand] = asyncio.Queue()

        async def collect(source: asyncio.Queue[Any]) -> None:
            while True:
                await work.put(await source.get())

        async def answer_turns() -> None:
            # Only the decider runs in threads (it may be slow); the dialogue changes here, on
            # the event loop, so a call that ends while a decision is running files nothing.
            while True:
                item = await work.get()
                if isinstance(item, OfficerCommand):
                    await obey(item)
                    continue
                turns_before = len(dialogue.turns)
                if dialogue.hear(item) is not None:
                    await decide(new_turn=True)
                elif len(dialogue.turns) > turns_before:
                    # Said while the call was handed off: kept, and masked as if it held a
                    # personal number, since nothing decides on it now.
                    await record_turn(1.0)

        async def converse(event: SttEvent) -> None:
            if agent_hung_up.is_set():
                return  # nothing follows call_ended
            await outbox.json(event.to_client())
            if event.type == "connected":
                await answer(dialogue.greet())
            elif event.type == "transcript":
                turns.add(event.text)

        background.append(asyncio.create_task(collect(turns.turns)))
        background.append(asyncio.create_task(collect(live.commands)))
        background.append(asyncio.create_task(answer_turns()))
        try:
            final = await _stt_session(
                websocket, settings, outbox, config, converse, stop=agent_hung_up
            )
        finally:
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
            while not work.empty():
                if isinstance(item := work.get_nowait(), OfficerCommand) and not item.done.done():
                    item.done.set_exception(LookupError("The call has ended"))
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
            if ended_by == "quiet":
                stream.discard_pending()
                return WENT_QUIET, 1008
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
    "prisma", "agent" or "quiet" (no audio for `stt_quiet_s`)."""
    went_quiet = False

    async def pump_audio() -> None:
        nonlocal went_quiet
        loop = asyncio.get_running_loop()
        deadline = loop.time() + stt_quiet_s
        while True:
            quiet = asyncio.timeout_at(deadline)
            try:
                async with quiet:
                    message = await websocket.receive()
            except TimeoutError:
                if not quiet.expired():
                    raise
                went_quiet = True
                return
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes"):
                deadline = loop.time() + stt_quiet_s
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
            return "quiet" if went_quiet else "browser"
        return "prisma" if events in done else "agent"
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _admit(websocket: WebSocket, settings: Settings) -> bool:
    """Accept a call socket (/ws/stt, /ws/call), or close it and return False.

    A page not on ALLOWED_ORIGINS is refused before the handshake completes: closing with
    1008 before accepting makes the server answer HTTP 403, so the browser only sees a failed
    connection. Origin stops other web pages, not a script that forges it; the session cap
    bounds what such a script can spend.
    """
    origin = websocket.headers.get("origin")
    if origin not in settings.origins:
        logger.warning("Refused a call socket from origin %r", origin)
        await websocket.close(code=1008)
        return False
    await websocket.accept()
    if not settings.gnani_api_key.get_secret_value():
        await _fail(websocket, "GNANI_API_KEY is not set")
        return False
    return True


async def _line_busy(websocket: WebSocket) -> None:
    logger.warning("Refused a call: all %d STT sessions are in use", stt_sessions.open)
    # 1013: try again later.
    await _send_and_close(websocket, LINE_BUSY, code=1013)


async def _send_and_close(websocket: WebSocket, payload: dict[str, Any], code: int) -> None:
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        await websocket.send_json(payload)
        await websocket.close(code=code)


async def _fail(websocket: WebSocket, message: str) -> None:
    await _send_and_close(websocket, {"type": "error", "message": message}, code=1011)
