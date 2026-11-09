import asyncio
import collections
import contextlib
import functools
import hmac
import ipaddress
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Path,
    Query,
    WebSocket,
    WebSocketDisconnect,
)

from audio.outbox import Outbox
from audio.speaker import Speaker
from audio.tts_cache import AudioCache, CachedSpeech
from cancelling import cancel_and_wait
from config import Settings, get_settings
from decisions import thresholds
from dialogue.decision import Decider, Decision, Screening
from dialogue.live import LiveCall, LiveCalls, OfficerCommand
from dialogue.machine import Dialogue, OfficerError, Reply, State
from dialogue.standin import KeywordDecider
from dialogue.templates import load_templates
from dialogue.turns import TURN_PAUSE_S, TurnCollector
from gnani.prisma import (
    PrismaClosed,
    PrismaConfig,
    PrismaError,
    PrismaStream,
    PrismaUnavailable,
    SttEvent,
    load_bias_words,
)
from gnani.timbre import TimbreClient, TimbreConfig
from tickets.store import FIRST_NUMBER, TicketStore

logger = logging.getLogger("dhwani.stt")

# How often transcripts past TRANSCRIPT_RETENTION_DAYS are deleted, and ticket locations past
# TICKET_LOCATION_RETENTION_DAYS blanked.
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
    settings = get_settings()
    days = settings.transcript_retention_days
    location_days = settings.ticket_location_retention_days
    store = get_ticket_store()
    try:
        purged = await asyncio.to_thread(store.purge_transcripts, days)
        blanked = await asyncio.to_thread(store.purge_ticket_locations, location_days)
    except Exception:
        # One failed pass (a locked database) must not end retention for good.
        logger.exception("Deleting old caller turns failed; trying again later")
    else:
        if purged:
            logger.info("Deleted %d caller turns older than %d days", purged, days)
        if blanked:
            logger.info("Blanked %d ticket locations older than %d days", blanked, location_days)


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
        await cancel_and_wait(sweeper)


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
            model=settings.gnani_stt_model,
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
# "one moment" (`hold_on`) so the caller is not left in silence. Not said when `ack` is on.
hold_on_after_s = 0.3
# Say `ack` ("okay") as soon as a caller's turn ends, before deciding: Laya takes about a second,
# and without it the caller hears nothing for ~2.8 s (docs/decisions/laya-latency.md).
acknowledge_turns = True
# After the last line of a closed call has played, wait this long before hanging up.
HANG_UP_GRACE_S = 0.5
# When Prisma drops a session mid-call (it has closed with 1011 and at its own time limit), the
# call opens a new one this many times before giving up.
STT_RECONNECTS = 3
STT_RECONNECT_DELAY_S = 0.5
# The browser keeps sending through the gap, so a new session starts with a backlog that would
# overflow Prisma's buffer. For this long after reconnecting, only the newest audio is kept.
STT_RESUME_CATCH_UP_S = 2.0
STT_RESUME_BACKLOG_S = 1.0
# A console must say hello with the token within this long. Longer than any real hello, which is
# under 100 characters: the limit is read before the text is parsed, for a stranger's socket.
CONSOLE_HELLO_TIMEOUT_S = 5.0
MAX_HELLO_CHARS = 4096
# Longest a console waits for a call to carry out an officer command (a decision may be
# running on the call).
OFFICER_COMMAND_TIMEOUT_S = 10.0

# The calls in progress, and the consoles watching them.
live_calls = LiveCalls()


class SessionCount:
    """Sockets holding one of a few places right now: browser sessions streaming to Prisma, which
    spend our Gnani credits, and consoles."""

    def __init__(self) -> None:
        self.open = 0
        self.by_client: collections.Counter[str] = collections.Counter()

    @contextlib.contextmanager
    def hold(self, limit: int, client: str = "", per_client: int = 0) -> Iterator[bool]:
        """Count one session while the block runs. Yields False, counting nothing, when
        `limit` are already open, or `per_client` (if set) from this `client`."""
        if self.open >= limit or (per_client and self.by_client[client] >= per_client):
            yield False
            return
        self.open += 1
        self.by_client[client] += 1
        try:
            yield True
        finally:
            self.open -= 1
            self.by_client[client] -= 1
            if not self.by_client[client]:
                del self.by_client[client]


class CallBudget:
    """The ceiling on what the public line may spend, however calls are spread over time: the
    minutes on the line across all callers in a rolling day, and the calls one caller may start
    in a rolling hour. In memory, so restarting the backend starts them afresh.

    Minutes, not calls: credits go by the audio second, and two lines of ten-minute calls need
    only 12 calls an hour to run all day. A call in progress is counted when it ends, so the
    ceiling can be passed by the calls on the line when it is reached, which are never cut.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._calls: collections.deque[tuple[float, str]] = collections.deque()
        self._by_client: collections.Counter[str] = collections.Counter()
        self._line_time: collections.deque[tuple[float, float]] = collections.deque()
        self._line_seconds = 0.0

    def admit(self, client: str, calls_per_hour: int, minutes_per_day: int) -> str | None:
        """Count a new call from `client`, or return why it is refused, counting nothing. A limit
        of 0 is off."""
        now = self._clock()
        while self._calls and now - self._calls[0][0] >= 3600.0:
            _, gone = self._calls.popleft()
            self._by_client[gone] -= 1
            if not self._by_client[gone]:
                del self._by_client[gone]
        while self._line_time and now - self._line_time[0][0] >= 86400.0:
            self._line_seconds -= self._line_time.popleft()[1]
        if minutes_per_day and self._line_seconds >= minutes_per_day * 60:
            return f"{minutes_per_day} minutes on the line in the last 24 hours"
        if calls_per_hour and self._by_client[client] >= calls_per_hour:
            return f"{calls_per_hour} calls from one caller in the last hour"
        self._calls.append((now, client))
        self._by_client[client] += 1
        return None

    def charge(self, seconds: float) -> None:
        """Add the length of a call that has just ended."""
        self._line_time.append((self._clock(), seconds))
        self._line_seconds += seconds


def _client(websocket: WebSocket) -> str:
    """Who is calling: the address nginx passes in X-Real-IP (Cloudflare's client IP behind the
    tunnel), or the socket's peer when there is no nginx in front (local development). All of an
    IPv6 /64 counts as one caller: it is one household or phone, and it has more addresses than
    any limit per address could mean anything against."""
    address = websocket.headers.get("x-real-ip")
    if not address:
        peer = getattr(websocket, "client", None)
        address = peer.host if peer else ""
    with contextlib.suppress(ValueError):
        if ipaddress.ip_address(address).version == 6:
            return str(ipaddress.ip_network((address, 64), strict=False))
    return address


stt_sessions = SessionCount()
call_budget = CallBudget()
console_hellos = SessionCount()
open_consoles = SessionCount()
LINE_BUSY = {"type": "error", "message": "The line is busy; please try again in a minute"}
LINE_RESTING = {
    "type": "error",
    "message": "The helpline has taken all the calls it can for now; please try again later",
}
TOO_MANY_CONSOLES = {"type": "error", "message": "Too many consoles are open"}
# A session that sends too little audio over this long is ended, so a socket left open and silent,
# or sending a byte now and then, cannot hold one of the few sessions. The phone page streams all
# the time, silence included. How little is too little: MIN_AUDIO_SHARE.
stt_quiet_s = 30.0
WENT_QUIET = {"type": "error", "message": "No audio arrived for too long; call ended"}
STT_FAILED = {"type": "error", "message": "Speech recognition is unavailable; call ended"}
# The longest one session may last, so a script streaming silence cannot hold a session (and
# spend credits) for ever. A real complaint call takes a minute or two.
stt_max_session_s = 600.0
TOO_LONG = {"type": "error", "message": "The call reached its time limit; call ended"}

TEMPLATES = load_templates()
TTS_CACHE: AudioCache = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


def _is_console_token(offered: object) -> bool:
    """Whether `offered` is CONSOLE_TOKEN; never true while it is unset."""
    token = get_settings().console_token.get_secret_value()
    if not token or not isinstance(offered, str):
        return False
    return hmac.compare_digest(offered.encode(), token.encode())


def officers_only(
    authorization: str | None = Header(None), cf_connecting_ip: str | None = Header(None)
) -> None:
    """The console's HTTP routes need the console's token, as `Authorization: Bearer <token>`:
    tickets hold where callers live, and the tunnel makes the backend public."""
    settings = get_settings()
    if settings.console_local_only and cf_connecting_ip is not None:
        raise HTTPException(status_code=404, detail="Not Found")
    if not settings.console_token.get_secret_value():
        raise HTTPException(status_code=503, detail="CONSOLE_TOKEN is not set")
    scheme, _, offered = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not _is_console_token(offered.strip()):
        raise HTTPException(
            status_code=401, detail="Not allowed", headers={"WWW-Authenticate": "Bearer"}
        )


@app.get("/api/tickets", dependencies=[Depends(officers_only)])
def list_tickets(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    """The newest tickets first, for the supervisor console."""
    return get_ticket_store().tickets(limit)


@app.get("/api/tickets/{number}", dependencies=[Depends(officers_only)])
def get_ticket(number: int = Path(ge=FIRST_NUMBER, le=2**31)) -> dict[str, Any]:
    if (ticket := get_ticket_store().ticket(number)) is None:
        raise HTTPException(status_code=404, detail="No such ticket")
    return ticket


@app.get("/api/thresholds", dependencies=[Depends(officers_only)])
def get_thresholds() -> dict[str, Any]:
    """The escalation thresholds, for the console's decision bars."""
    return thresholds.as_dict()


@app.websocket("/ws/console")
async def console(websocket: WebSocket) -> None:
    """A supervisor console: watch every live call, take one over, speak on it, hand it back.

    Contract (see backend/README.md): only pages on ALLOWED_ORIGINS may connect. The console first
    sends `hello` with `CONSOLE_TOKEN`. It then gets `calls` (every live call, and the thresholds),
    followed by `call_started`, `call_event` (a message the caller's browser got, with the caller's
    numbers masked) and `call_closed` as they happen. It may send `take_over`, `say` (`text`) and
    `hand_back`, each with a `call_id` and an optional `id` echoed in the answer: `done`, or `error`
    with a `message`.
    """
    settings = get_settings()
    client = _client(websocket)
    if settings.console_local_only and "cf-connecting-ip" in websocket.headers:
        await websocket.close(code=1008)
        return
    if not _page_allowed(websocket, settings):
        await websocket.close(code=1008)
        return
    with console_hellos.hold(
        settings.max_console_hellos, client, settings.max_console_hellos_per_client
    ) as held:
        if not held:
            logger.warning("Refused a console socket: too many are still to say hello")
            await websocket.close(code=1008)
            return
        await websocket.accept()
        if not settings.console_token.get_secret_value():
            await _fail(websocket, "CONSOLE_TOKEN is not set")
            return
        hello = await _read_hello(websocket)
    is_hello = isinstance(hello, dict) and hello.get("type") == "hello"
    if not _is_console_token(hello.get("token") if is_hello else None):
        await _send_and_close(websocket, {"type": "error", "message": "Not allowed"}, code=1008)
        return
    # Warned, not just logged: how an owner notices a token in use that should not be.
    logger.warning("A console signed in from %r", client)
    with open_consoles.hold(settings.max_consoles) as room:
        if not room:
            await _send_and_close(websocket, TOO_MANY_CONSOLES, code=1013)
            return
        await _serve_console(websocket)


async def _read_hello(websocket: WebSocket) -> Any:
    """A console's first message, parsed, or None if it is not there in time, not text, too long
    or not JSON. Not yet known to be from an officer, so its size is checked before parsing it."""
    try:
        message = await asyncio.wait_for(websocket.receive(), CONSOLE_HELLO_TIMEOUT_S)
    except TimeoutError:
        return None
    text = message.get("text")
    if not isinstance(text, str) or len(text) > MAX_HELLO_CHARS:
        return None
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


async def _serve_console(websocket: WebSocket) -> None:
    """The signed-in part of the console's socket."""
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
            await cancel_and_wait(*tasks)
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

    Off unless STT_RELAY_ENABLED is set: the phone page only uses /ws/call. Contract (see
    backend/README.md): only pages on ALLOWED_ORIGINS may connect, and only
    MAX_STT_SESSIONS at once across /ws/stt and /ws/call. After `config` and `connected`
    arrive, the browser streams binary frames of 16-bit mono PCM at `config.sample_rate`, in
    real time, and may send the text frame "flush". It receives JSON events; `closed` or
    `error` followed by a close ends the session.
    """
    settings = get_settings()
    if not settings.stt_relay_enabled:
        logger.warning("Refused a socket on /ws/stt: STT_RELAY_ENABLED is off")
        await websocket.close(code=1008)
        return
    if not await _admit(websocket, settings):
        return
    with _take_line(websocket, settings) as refusal:
        if refusal:
            await _send_and_close(websocket, refusal, code=1013)
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
    up. The browser sends the text frame `barge_in` when the caller speaks over the agent: the
    line being said is cut (its `agent_audio_end` is marked `interrupted`), queued lines are
    dropped, and what the caller says is a new turn.
    """
    settings = get_settings()
    if not await _admit(websocket, settings):
        return
    with _take_line(websocket, settings) as refusal:
        if refusal:
            await _send_and_close(websocket, refusal, code=1013)
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

            # After `ack` the caller has heard the agent already; a second filler is too chatty.
            timer = None if acknowledge_turns else asyncio.create_task(hold_on())
            try:
                return await asyncio.to_thread(dialogue.decider.decide, transcript, screening)
            finally:
                if timer:
                    timer.cancel()

        def write_turn(personal_id: float) -> None:
            store.record_turn(
                dialogue.call_id, len(dialogue.turns), dialogue.turns[-1], personal_id
            )

        async def record_turn(personal_id: float) -> None:
            # Off the event loop: a disk write must not stall the audio of other calls.
            await asyncio.to_thread(write_turn, personal_id)

        def keep(reply: Reply) -> Reply:
            """Store the decision behind a reply, with every probability, for audit. Called in
            the same worker thread that made the reply, so a hang-up that cancels the call
            cannot lose it. A failed write is logged and never changes what the caller hears."""
            answers = reply.decision or reply.screening
            if answers is not None:
                outcome = reply.reason or (reply.keys[-1] if reply.keys else "listening")
                try:
                    store.record_decision(
                        dialogue.call_id,
                        len(dialogue.turns),
                        outcome,
                        answers,
                        dialogue.decider.revision,
                    )
                except Exception:
                    logger.exception("Storing a decision failed")
            return reply

        def hand_off_at_once(first: Screening, new_turn: bool) -> Reply | None:
            """The hand-off the safety step calls for, or None. Run in a thread, with what must
            outlast the call written in the same go: the turn (`new_turn`: not stored yet), and
            for a caller in danger the emergency ticket, which `screened` files. A caller told
            to call 112 may hang up at once, and a write already running here still finishes.
            A failed write is logged, never allowed to silence the hand-off."""
            handed_off = dialogue.screened(first)
            if handed_off is None:
                return None
            if new_turn:
                try:
                    # Masked as if it held a personal number: no decision says otherwise yet.
                    write_turn(1.0)
                except Exception:
                    logger.exception("Storing the turn behind a safety hand-off failed")
            return keep(handed_off)

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
            handed_off = None
            if first is not None:
                handed_off = await asyncio.to_thread(hand_off_at_once, first, new_turn)
            if handed_off is not None:
                # Safety first: the hand-off is said before the rest is decided. The rest
                # follows for the officer as a `dialogue` event with no lines, and fills in the
                # emergency ticket (in a thread: a SQLite write).
                await answer(handed_off)
                try:
                    screening = await screen_whole_call(first)
                    decision = await asyncio.to_thread(
                        dialogue.decider.decide, transcript, screening
                    )
                except Exception:
                    logger.exception("Completing a decision after a hand-off failed")
                    return
                await answer(await asyncio.to_thread(lambda: keep(dialogue.complete(decision))))
                return
            try:
                screening = await screen_whole_call(first)
                decision = await decide_holding_on(transcript, screening)
            except Exception:
                await hand_off_on_failure()
                return
            await store_turn(decision.contains_personal_id)
            # In a thread: a reply that closes the call files the ticket, and every reply keeps
            # its decision: SQLite writes that can wait on the store's lock (a purge, another
            # call's turn) and would stall every call.
            # Only this task changes the dialogue, so nothing else races it meanwhile.
            await answer(await asyncio.to_thread(lambda: keep(dialogue.respond(decision))))

        async def obey(command: OfficerCommand) -> None:
            if command.done.done():
                return  # the console stopped waiting for it: too late to carry it out
            try:
                if command.kind == "take_over":
                    await answer(dialogue.take_over(), show=True)
                elif command.kind == "say":
                    line = dialogue.officer_says(command.text)
                    if speaker.queued >= settings.max_officer_lines_queued:
                        raise OfficerError("Wait for the last line to be spoken")
                    speaker.say(line)
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

        async def answer_one(item: str | OfficerCommand) -> None:
            if isinstance(item, OfficerCommand):
                await obey(item)
                return
            turns_before = len(dialogue.turns)
            if dialogue.hear(item) is not None:
                if acknowledge_turns:
                    speaker.say(TEMPLATES.line("ack"))
                await decide(new_turn=True)
            elif len(dialogue.turns) > turns_before:
                # Said while the call was handed off: kept, and masked as if it held a
                # personal number, since nothing decides on it now.
                await record_turn(1.0)

        async def answer_turns() -> None:
            # The decider runs in threads (it may be slow) and the dialogue changes only once it
            # is back here, so a call that ends while a decision is running files nothing. The
            # closing reply's own ticket write also runs in a thread, which a hang-up cannot
            # stop: a caller who leaves during that write still has the complaint filed.
            while True:
                item = await work.get()
                try:
                    await answer_one(item)
                except Exception:
                    # A failed store or ticket write must not leave the caller in silence with
                    # this task gone: log it and hand the call to an officer.
                    logger.exception("Answering a caller turn failed; handing off")
                    with contextlib.suppress(Exception):
                        await answer(dialogue.decider_failed())

        def barge_in() -> None:
            # Only while the agent is asking: once the call is closed or handed off, the lines
            # left (the ticket number, the safety advice) must be heard in full.
            if dialogue.state is State.LISTENING:
                speaker.interrupt()

        greeted: list[bool] = []

        async def converse(event: SttEvent) -> None:
            if agent_hung_up.is_set():
                return  # nothing follows call_ended
            await outbox.json(event.to_client())
            if event.type == "connected":
                if not greeted:
                    greeted.append(True)
                    await answer(dialogue.greet())
            elif event.type == "transcript":
                turns.add(event.text)

        background.append(asyncio.create_task(collect(turns.turns)))
        background.append(asyncio.create_task(collect(live.commands)))
        background.append(asyncio.create_task(answer_turns()))
        # For the whole call: a new Prisma session does not earn a caller more time.
        ends_at = asyncio.get_running_loop().time() + stt_max_session_s
        try:
            for attempt in range(STT_RECONNECTS + 1):
                failures: list[PrismaError] = []
                final = await _stt_session(
                    websocket,
                    settings,
                    outbox,
                    config,
                    converse,
                    stop=agent_hung_up,
                    on_barge_in=barge_in,
                    failures=failures,
                    ends_at=ends_at,
                    resumed=attempt > 0,
                )
                # Retry only what a new session can fix: Prisma closing on its own, or a failed
                # connect. A rejected key or an audio overflow would fail again the same way.
                recoverable = (final is not None and final[1] == 1000) or any(
                    isinstance(exc, PrismaClosed | PrismaUnavailable) for exc in failures
                )
                if not recoverable or agent_hung_up.is_set() or attempt == STT_RECONNECTS:
                    break
                logger.warning("Prisma ended the session mid-call; reconnecting (%d)", attempt + 1)
                await asyncio.sleep(STT_RECONNECT_DELAY_S)
        finally:
            await cancel_and_wait(*background)
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
    on_barge_in: Callable[[], None] | None = None,
    failures: list[PrismaError] | None = None,
    ends_at: float | None = None,
    resumed: bool = False,
) -> FinalMessage | None:
    """Run one Prisma session for this socket. Returns the message to end the session with,
    or None when the browser left or `stop` was set (the caller has nothing more to send).
    A Prisma failure is also appended to `failures`, for a caller deciding whether to retry.
    A `resumed` session follows a dropped one, so it drops the stale audio it starts with."""
    # 16-bit audio, so two bytes a sample.
    min_audio = int(settings.gnani_stt_sample_rate * 2 * stt_quiet_s * settings.min_audio_share)
    try:
        async with stt_stream_factory(settings) as stream:
            await outbox.json(config)
            catch_up_s = STT_RESUME_CATCH_UP_S if resumed else 0.0
            ended_by = await _relay(
                websocket, stream, on_event, stop, on_barge_in, min_audio, ends_at, catch_up_s
            )
            if ended_by in ("browser", "agent"):
                stream.discard_pending()
                return None
            if ended_by == "quiet":
                stream.discard_pending()
                return WENT_QUIET, 1008
            if ended_by == "too_long":
                stream.discard_pending()
                return TOO_LONG, 1008
        # The reason is Prisma's own text, which may name the account or its quota: log only.
        logger.warning("Prisma closed the session: %s %s", stream.close_code, stream.close_reason)
        return {"type": "closed", "by": "prisma", "code": stream.close_code}, 1000
    except PrismaError as exc:
        logger.warning("STT session failed: %s", exc)
        if failures is not None:
            failures.append(exc)
        # The detail (Prisma's own close reason, account or quota text) stays in the log.
        return STT_FAILED, 1011
    except WebSocketDisconnect:
        return None


async def _relay(
    websocket: WebSocket,
    stream: PrismaStream,
    on_event: EventHandler,
    stop: asyncio.Event | None = None,
    on_barge_in: Callable[[], None] | None = None,
    min_audio: int = 0,
    ends_at: float | None = None,
    catch_up_s: float = 0.0,
) -> str:
    """Pump audio up and events down until one side ends or `stop` is set; returns "browser",
    "prisma", "agent", "quiet" (under `min_audio` bytes of audio in `stt_quiet_s`, or none at all)
    or "too_long" (`ends_at` came, by default `stt_max_session_s` from now). A `barge_in` text
    frame (the caller spoke over the agent) goes to `on_barge_in`. For the first `catch_up_s`,
    audio queued for Prisma is cut to the newest `STT_RESUME_BACKLOG_S`."""
    went_quiet = too_long = False

    async def pump_audio() -> None:
        nonlocal went_quiet, too_long
        loop = asyncio.get_running_loop()
        deadline = loop.time() + stt_quiet_s
        end_of_session = loop.time() + stt_max_session_s if ends_at is None else ends_at
        catching_up_until = loop.time() + catch_up_s
        stale_frames = 0
        # Audio since `deadline` was last set. The first audio sets it afresh: the mic opens only
        # once the caller has answered the browser's permission prompt.
        heard = 0
        any_audio = False
        while True:
            quiet = asyncio.timeout_at(min(deadline, end_of_session))
            try:
                async with quiet:
                    message = await websocket.receive()
            except TimeoutError:
                if not quiet.expired():
                    raise
                # By which limit the timer was set, not the clock: timers can fire a tick early.
                too_long = end_of_session <= deadline
                went_quiet = not too_long and (heard == 0 or heard < min_audio)
                if too_long or went_quiet:
                    return
                heard = 0
                deadline = loop.time() + stt_quiet_s
                continue
            if message["type"] == "websocket.disconnect":
                return
            if loop.time() >= end_of_session:
                too_long = True
                return
            if message.get("bytes"):
                if not any_audio:
                    any_audio = True
                    deadline = loop.time() + stt_quiet_s
                heard += len(message["bytes"])
                await stream.send_audio(message["bytes"])
                if loop.time() < catching_up_until:
                    stale_frames += stream.trim_backlog(STT_RESUME_BACKLOG_S)
                elif stale_frames:
                    logger.info("Dropped %d stale audio frames after reconnecting", stale_frames)
                    stale_frames = 0
            elif message.get("text") == "flush":
                await stream.flush()
            elif message.get("text") == "barge_in" and on_barge_in is not None:
                on_barge_in()

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
            if too_long:
                return "too_long"
            return "quiet" if went_quiet else "browser"
        return "prisma" if events in done else "agent"
    finally:
        await cancel_and_wait(*tasks)


def _page_allowed(websocket: WebSocket, settings: Settings) -> bool:
    """Whether the page opening this socket is on ALLOWED_ORIGINS. Origin stops other web pages,
    not a script that forges it."""
    origin = websocket.headers.get("origin")
    if origin in settings.origins:
        return True
    logger.warning("Refused a socket from origin %r", origin)
    return False


async def _admit(websocket: WebSocket, settings: Settings) -> bool:
    """Accept a call socket (/ws/stt, /ws/call), or close it and return False.

    A page not on ALLOWED_ORIGINS is refused before the handshake completes: closing with
    1008 before accepting makes the server answer HTTP 403, so the browser only sees a failed
    connection. The session cap and the credit ceiling (`_take_line`) bound what a script that
    forges Origin can spend.
    """
    if not _page_allowed(websocket, settings):
        await websocket.close(code=1008)
        return False
    await websocket.accept()
    if not settings.gnani_api_key.get_secret_value():
        await _fail(websocket, "GNANI_API_KEY is not set")
        return False
    return True


@contextlib.contextmanager
def _take_line(websocket: WebSocket, settings: Settings) -> Iterator[dict[str, Any] | None]:
    """Hold one of the lines for this socket, and charge the time to the credit ceiling, while the
    block runs. Yields the message to refuse the call with (close 1013, try again later), before
    Prisma is opened, when there is no line free or the ceiling is reached; otherwise None."""
    client = _client(websocket)
    with stt_sessions.hold(
        settings.max_stt_sessions, client, settings.max_sessions_per_client
    ) as held:
        if not held:
            logger.warning("Refused a call: all %d STT sessions are in use", stt_sessions.open)
            yield LINE_BUSY
            return
        reason = call_budget.admit(
            client, settings.max_calls_per_client_per_hour, settings.max_call_minutes_per_day
        )
        if reason:
            logger.warning("Refused a call: the credit ceiling is reached (%s)", reason)
            yield LINE_RESTING
            return
        started = time.monotonic()
        try:
            yield None
        finally:
            call_budget.charge(time.monotonic() - started)


async def _send_and_close(websocket: WebSocket, payload: dict[str, Any], code: int) -> None:
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        await websocket.send_json(payload)
        await websocket.close(code=code)


async def _fail(websocket: WebSocket, message: str) -> None:
    await _send_and_close(websocket, {"type": "error", "message": message}, code=1011)
