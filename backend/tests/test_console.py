"""The supervisor console's socket: who may connect, what it sees, and an officer taking a call
over, speaking on it and handing it back."""

import asyncio
import dataclasses
import json
import time

import pytest
from fakes import PHONE_ORIGIN, ROADS, ROADS_IN_W04, FakeStream, ScriptedBrowser, ScriptedDecider
from fakes import isolated_settings as settings_with
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from test_call import FakeTts, ended_lines, lines_of

import app as app_module
from decisions import thresholds
from dialogue.live import MAX_BACKLOG, Watcher
from dialogue.machine import CountingDesk
from gnani.prisma import SttEvent
from tickets.store import TicketStore

TOKEN = "console-test-token-0123"


@pytest.fixture
def console_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        app_module, "get_settings", lambda: settings_with(gnani_api_key="t", console_token=TOKEN)
    )


def console_client() -> TestClient:
    return TestClient(app_module.app, headers={"origin": PHONE_ORIGIN})


def test_the_console_is_off_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "get_settings", lambda: settings_with(gnani_api_key="t"))
    with console_client() as client, client.websocket_connect("/ws/console") as ws:
        assert ws.receive_json() == {"type": "error", "message": "CONSOLE_TOKEN is not set"}


@pytest.mark.parametrize(
    "hello",
    [{"type": "hello", "token": "wrong"}, {"type": "hello"}, {"token": TOKEN}, ["hello"]],
)
def test_a_console_without_the_right_token_is_refused(console_settings: None, hello) -> None:
    with console_client() as client, client.websocket_connect("/ws/console") as ws:
        ws.send_json(hello)
        assert ws.receive_json() == {"type": "error", "message": "Not allowed"}
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.code == 1008


@pytest.mark.parametrize("origin", ["https://evil.example", None])
def test_a_page_not_on_the_allowed_list_cannot_open_the_console(
    console_settings: None, origin: str | None
) -> None:
    client = TestClient(app_module.app, headers={} if origin is None else {"origin": origin})

    with pytest.raises(WebSocketDisconnect) as refused, client.websocket_connect("/ws/console"):
        pass

    assert refused.value.code == 1008


@pytest.mark.parametrize(
    "frame",
    [
        "x" * (app_module.MAX_HELLO_CHARS + 1),
        # Valid JSON too deep to parse: it must end this socket quietly, not raise out of the route.
        "[" * app_module.MAX_HELLO_CHARS,
        "not json",
        b"\x00\x01",
    ],
)
def test_a_hello_that_is_too_long_deep_or_garbled_is_refused(console_settings: None, frame) -> None:
    with console_client() as client, client.websocket_connect("/ws/console") as ws:
        ws.send_bytes(frame) if isinstance(frame, bytes) else ws.send_text(frame)
        assert ws.receive_json() == {"type": "error", "message": "Not allowed"}


class HelloConsole:
    """A socket that holds its place, without saying hello, until `leave` is set."""

    def __init__(self, leave: asyncio.Event, client: str = "203.0.113.9") -> None:
        self.headers = {"origin": PHONE_ORIGIN, "x-real-ip": client}
        self.leave = leave
        self.accepted = False
        self.closed_with: int | None = None

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> dict:
        await self.leave.wait()
        return {"type": "websocket.disconnect", "code": 1001}

    async def send_json(self, data: dict) -> None:
        return None

    async def close(self, code: int = 1000) -> None:
        self.closed_with = code


def test_sockets_that_never_say_hello_are_capped_per_caller_and_in_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = settings_with(
        console_token=TOKEN, max_console_hellos=3, max_console_hellos_per_client=2
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)

    async def scenario() -> tuple[HelloConsole, HelloConsole]:
        leave = asyncio.Event()
        holding = [
            asyncio.create_task(app_module.console(HelloConsole(leave, client)))
            for client in ("203.0.113.9", "203.0.113.9", "203.0.113.10")
        ]
        await asyncio.sleep(0.01)
        same_caller = HelloConsole(leave)
        another_caller = HelloConsole(leave, "203.0.113.11")
        await app_module.console(same_caller)
        await app_module.console(another_caller)
        leave.set()
        await asyncio.gather(*holding)
        return same_caller, another_caller

    same_caller, another_caller = asyncio.run(asyncio.wait_for(scenario(), 5))

    for refused in (same_caller, another_caller):
        assert not refused.accepted
        assert refused.closed_with == 1008


def test_a_socket_that_never_says_hello_is_hung_up_and_frees_its_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "get_settings", lambda: settings_with(console_token=TOKEN))
    monkeypatch.setattr(app_module, "CONSOLE_HELLO_TIMEOUT_S", 0.01)
    silent = HelloConsole(asyncio.Event())

    asyncio.run(asyncio.wait_for(app_module.console(silent), 5))

    assert silent.closed_with == 1008
    assert app_module.console_hellos.open == 0


def test_only_so_many_consoles_may_sign_in(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = settings_with(console_token=TOKEN, max_consoles=1)
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(app_module, "live_calls", app_module.LiveCalls())

    async def scenario() -> tuple[ScriptedConsole, ScriptedConsole]:
        second = ScriptedConsole(lambda message: [])
        first = ScriptedConsole(lambda message: [])
        watching = asyncio.create_task(app_module.console(first))
        await asyncio.sleep(0.01)
        await app_module.console(second)
        first.inbox.put_nowait(None)
        await watching
        return first, second

    first, second = asyncio.run(asyncio.wait_for(scenario(), 5))

    assert first.got[0]["type"] == "calls"
    # Not 1008: the page would take that for a wrong token and forget the right one.
    assert second.got == [app_module.TOO_MANY_CONSOLES]
    assert second.closed_with == 1013


def test_a_signed_in_console_is_logged_with_where_it_came_from(
    console_settings: None, caplog: pytest.LogCaptureFixture
) -> None:
    console = ScriptedConsole(lambda message: [None] if message["type"] == "calls" else [])
    console.headers["x-real-ip"] = "203.0.113.5"

    asyncio.run(asyncio.wait_for(app_module.console(console), 5))

    assert "A console signed in from '203.0.113.5'" in caplog.text
    assert TOKEN not in caplog.text


TUNNEL_HEADERS = {"origin": PHONE_ORIGIN, "cf-connecting-ip": "203.0.113.5"}


def test_console_local_only_refuses_the_tunnel_with_the_right_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = settings_with(console_token=TOKEN, console_local_only=True)
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    officer = {"Authorization": f"Bearer {TOKEN}"}

    with TestClient(app_module.app, headers=TUNNEL_HEADERS) as remote:
        assert remote.get("/api/tickets", headers=officer).status_code == 404
        with pytest.raises(WebSocketDisconnect) as refused, remote.websocket_connect("/ws/console"):
            pass
    assert refused.value.code == 1008

    with console_client() as local:
        assert local.get("/api/tickets", headers=officer).status_code == 200
        with local.websocket_connect("/ws/console") as ws:
            ws.send_json({"type": "hello", "token": TOKEN})
            assert ws.receive_json()["type"] == "calls"


def test_the_tunnel_can_reach_the_console_unless_it_is_local_only(console_settings: None) -> None:
    officer = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app_module.app, headers=TUNNEL_HEADERS) as remote:
        assert remote.get("/api/tickets", headers=officer).status_code == 200


def test_consoles_see_numbers_masked_and_the_callers_browser_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "live_calls", app_module.LiveCalls())
    use_call_fakes(monkeypatch, ScriptedDecider(ROADS))
    phone = "ನನ್ನ ನಂಬರ್ 98450 12345"

    class SaysANumber(FakeStream):
        async def send_audio(self, pcm: bytes) -> None:
            self._events.put_nowait(SttEvent(type="transcript", text=phone))

    monkeypatch.setattr(app_module, "stt_stream_factory", lambda settings: SaysANumber())
    console = ScriptedConsole(end_on_close(lambda message: []))
    browser = ScriptedBrowser([bytes(1024)], leave_when=ended_lines(2))

    asyncio.run(call_with_console(browser, console))

    assert {"type": "transcript", "text": phone}.items() <= next(
        m for m in browser.sent if isinstance(m, dict) and m["type"] == "transcript"
    ).items()
    (seen,) = [
        m["event"]["text"]
        for m in console.got
        if m["type"] == "call_event" and m["event"]["type"] == "transcript"
    ]
    assert seen == "ನನ್ನ ನಂಬರ್ XXXXX XXXXX"


def test_a_number_spread_over_segments_is_masked_for_the_console() -> None:
    async def scenario() -> list[str]:
        calls = app_module.LiveCalls()
        dialogue = app_module.Dialogue(app_module.TEMPLATES, ScriptedDecider(ROADS), CountingDesk())
        call = calls.open(dialogue)
        return [call.masked_segment(text) for text in ("ಕಸ", "984", "501", "2345", "ವಾರ್ಡ್ 4")]

    # What was already on screen stays: a three-digit piece cannot be told from a ward yet.
    assert asyncio.run(scenario()) == ["ಕಸ", "984", "XXX", "XXXX", "ವಾರ್ಡ್ 4"]


def test_a_console_joining_mid_call_sees_earlier_turns_masked() -> None:
    async def scenario() -> list[str]:
        calls = app_module.LiveCalls()
        dialogue = app_module.Dialogue(app_module.TEMPLATES, ScriptedDecider(ROADS), CountingDesk())
        call = calls.open(dialogue)
        dialogue.hear("ನನ್ನ ಆಧಾರ್ 1234")
        dialogue.hear("ವಾರ್ಡ್ 4")
        return call.summary()["turns"]

    assert asyncio.run(scenario()) == ["ನನ್ನ ಆಧಾರ್ XXXX", "ವಾರ್ಡ್ 4"]


def test_an_officer_cannot_queue_lines_without_end(monkeypatch: pytest.MonkeyPatch) -> None:
    use_call_fakes(monkeypatch, ScriptedDecider(ROADS))
    settings = settings_with(gnani_api_key="t", console_token=TOKEN, max_officer_lines_queued=2)
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)

    class SlowTts(FakeTts):
        async def stream(self, text, voice=None):
            await asyncio.sleep(0.2)
            async for pcm in super().stream(text, voice):
                yield pcm

    monkeypatch.setattr(app_module, "tts_client_factory", lambda settings: SlowTts())
    answers: list[dict] = []

    def officer(message: dict) -> list[dict | None]:
        event = message.get("event", {})
        if event.get("type") == "dialogue" and event["keys"] == ["greeting"]:
            return [{"type": "take_over", "call_id": message["call_id"], "id": 0}]
        if message["type"] in ("done", "error") and "command" in message:
            answers.append(message)
            if len(answers) <= 4:
                say = {"type": "say", "call_id": message["call_id"], "text": "ಸರಿ"}
                return [say | {"id": len(answers)}]
        return []

    console = ScriptedConsole(end_on_close(officer))
    browser = ScriptedBrowser([], leave_when=lambda sent: len(answers) == 5)

    asyncio.run(call_with_console(browser, console))

    # The greeting is being spoken: two lines may wait behind it, and no more.
    assert [m["type"] for m in answers] == ["done", "done", "done", "error", "error"]
    assert answers[-1]["message"] == "Wait for the last line to be spoken"


def test_a_console_sees_the_live_calls_and_thresholds_and_gets_answers(
    monkeypatch: pytest.MonkeyPatch, console_settings: None
) -> None:
    monkeypatch.setattr(app_module, "live_calls", app_module.LiveCalls())
    script = iter(
        [
            [{"type": "take_over", "call_id": "nobody", "id": 7}],
            [{"type": "dance", "call_id": "nobody"}],
            [None],
        ]
    )
    # Driven directly, as the take-over tests are: TestClient can cancel the app mid-flight
    # when the test leaves the socket.
    console = ScriptedConsole(lambda message: next(script))

    asyncio.run(asyncio.wait_for(app_module.console(console), 5))

    assert console.got == [
        {"type": "calls", "calls": [], "thresholds": thresholds.as_dict()},
        {
            "type": "error",
            "message": "No such call",
            "command": "take_over",
            "call_id": "nobody",
            "id": 7,
        },
        {
            "type": "error",
            "message": "Unknown command",
            "command": "dance",
            "call_id": "nobody",
        },
    ]
    officer = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app_module.app) as client:
        listed = client.get("/api/thresholds", headers=officer).json()
    assert listed["safety_hazard"] == thresholds.SAFETY_HAZARD


class ScriptedConsole:
    """A console socket as uvicorn presents it. It says hello, then `react` answers each
    message it gets with the messages to send back; None ends the session."""

    def __init__(self, react) -> None:
        self.react = react
        self.headers = {"origin": PHONE_ORIGIN}
        self.inbox: asyncio.Queue[dict | None] = asyncio.Queue()
        self.inbox.put_nowait({"type": "hello", "token": TOKEN})
        self.got: list[dict] = []
        self.closed_with: int | None = None

    async def accept(self) -> None:
        return None

    async def receive(self) -> dict:
        return {"type": "websocket.receive", "text": json.dumps(await self.receive_json())}

    async def receive_json(self) -> dict:
        async with asyncio.timeout(5):
            message = await self.inbox.get()
        if message is None:
            raise WebSocketDisconnect(1000)
        return message

    async def send_json(self, data: dict) -> None:
        self.got.append(data)
        for reply in self.react(data):
            self.inbox.put_nowait(reply)

    async def close(self, code: int = 1000) -> None:
        self.closed_with = code
        self.inbox.put_nowait(None)


def test_an_officer_takes_over_speaks_and_hands_a_call_back_to_the_agent(
    monkeypatch: pytest.MonkeyPatch, ticket_store: TicketStore
) -> None:
    # A live wire: handed off on the first turn. The officer takes over, tells the caller to
    # keep away and hands back; the agent then files the ticket without handing off again.
    danger = dataclasses.replace(ROADS, safety_hazard=0.9)
    decider = ScriptedDecider(danger, dataclasses.replace(ROADS_IN_W04, safety_hazard=0.9))
    stream = FakeStream()
    monkeypatch.setattr(
        app_module, "get_settings", lambda: settings_with(gnani_api_key="t", console_token=TOKEN)
    )
    monkeypatch.setattr(app_module, "stt_stream_factory", lambda settings: stream)
    monkeypatch.setattr(app_module, "tts_client_factory", lambda settings: FakeTts())
    monkeypatch.setattr(app_module, "get_decider", lambda: decider)
    monkeypatch.setattr(app_module, "turn_pause_s", 0.02)
    monkeypatch.setattr(app_module, "TTS_CACHE", {})
    monkeypatch.setattr(app_module, "HANG_UP_GRACE_S", 0.01)
    monkeypatch.setattr(app_module, "live_calls", app_module.LiveCalls())

    officer_line = "ದಯವಿಟ್ಟು ಆ ವೈರ್ ಹತ್ತಿರ ಹೋಗಬೇಡಿ"

    def officer(message: dict) -> list[dict | None]:
        if message["type"] == "call_closed":
            return [None]
        if message["type"] == "done":
            follow_up = {1: ("say", {"text": officer_line}), 2: ("hand_back", {})}
            if message["id"] in follow_up:
                kind, extra = follow_up[message["id"]]
                command = {"type": kind, "call_id": message["call_id"], "id": message["id"] + 1}
                return [command | extra]
            return []
        event = message.get("event", {})
        if event.get("type") == "dialogue" and event["keys"] == ["safety"]:
            return [{"type": "take_over", "call_id": message["call_id"], "id": 1}]
        return []

    console = ScriptedConsole(officer)
    browser = ScriptedBrowser([bytes(1024)], leave_when=lambda sent: False)  # agent hangs up

    async def scenario() -> None:
        watching = asyncio.create_task(app_module.console(console))
        await asyncio.sleep(0.05)  # the console is watching before the call starts
        await app_module.call(browser)
        await asyncio.wait_for(watching, 5)

    asyncio.run(scenario())

    assert [m["id"] for m in console.got if m["type"] == "done"] == [1, 2, 3]
    keys = lines_of(browser.sent, "dialogue", "keys")
    # Greeting, the safety line, the rest of the decision (no lines), the take-over (no
    # lines), the hand-back, then the agent's ticket from everything said.
    assert keys == [["greeting"], ["safety"], [], [], ["hand_back"], ["read_back_ward", "closing"]]
    assert officer_line in lines_of(browser.sent, "agent_text", "text")
    # A handed-off call keeps its reason when an officer takes it.
    assert lines_of(browser.sent, "dialogue", "reason")[3] == "safety"
    officer_flags = lines_of(browser.sent, "dialogue", "officer")
    assert officer_flags[3] is True and officer_flags[-1] is False
    assert browser.sent[-1] == {"type": "call_ended", "reason": "ticket"}
    (ticket,) = ticket_store.tickets()
    assert (ticket["department"], ticket["ward"]) == ("roads", "w04")
    # The console saw the call start, everything the browser got, and the call close.
    kinds = [m["type"] for m in console.got]
    assert kinds[0] == "calls" and "call_started" in kinds and kinds[-1] == "call_closed"
    copied = [m["event"] for m in console.got if m["type"] == "call_event"]
    assert copied == [m for m in browser.sent if isinstance(m, dict)]


def use_call_fakes(monkeypatch: pytest.MonkeyPatch, decider: ScriptedDecider) -> None:
    monkeypatch.setattr(
        app_module, "get_settings", lambda: settings_with(gnani_api_key="t", console_token=TOKEN)
    )
    monkeypatch.setattr(app_module, "stt_stream_factory", lambda settings: FakeStream())
    monkeypatch.setattr(app_module, "tts_client_factory", lambda settings: FakeTts())
    monkeypatch.setattr(app_module, "get_decider", lambda: decider)
    monkeypatch.setattr(app_module, "turn_pause_s", 0.02)
    monkeypatch.setattr(app_module, "TTS_CACHE", {})
    monkeypatch.setattr(app_module, "HANG_UP_GRACE_S", 0.01)
    monkeypatch.setattr(app_module, "live_calls", app_module.LiveCalls())


async def call_with_console(browser: ScriptedBrowser, console: ScriptedConsole) -> None:
    watching = asyncio.create_task(app_module.console(console))
    await asyncio.sleep(0.05)  # the console is watching before the call starts
    await app_module.call(browser)
    await asyncio.wait_for(watching, 5)


def end_on_close(react):
    return lambda message: [None] if message["type"] == "call_closed" else react(message)


def test_an_officer_can_take_a_call_the_agent_is_handling_and_hears_the_caller(
    monkeypatch: pytest.MonkeyPatch, ticket_store: TicketStore
) -> None:
    decider = ScriptedDecider(ROADS_IN_W04)
    use_call_fakes(monkeypatch, decider)
    recorded = []
    store_turn = ticket_store.record_turn
    monkeypatch.setattr(
        ticket_store, "record_turn", lambda *turn: (recorded.append(turn), store_turn(*turn))
    )

    def officer(message: dict) -> list[dict]:
        event = message.get("event", {})
        if event.get("type") == "dialogue" and event["keys"] == ["greeting"]:
            return [{"type": "take_over", "call_id": message["call_id"], "id": 1}]
        return []

    console = ScriptedConsole(end_on_close(officer))
    # The caller speaks once the officer has the call, then hangs up once it is stored.
    browser = ScriptedBrowser([bytes(1024)], leave_when=lambda sent: bool(recorded))
    asyncio.run(call_with_console(browser, console))

    assert [m for m in console.got if m["type"] == "done"][0]["id"] == 1
    assert lines_of(browser.sent, "dialogue", "keys") == [["greeting"], []]
    assert lines_of(browser.sent, "dialogue", "reason") == [None, "taken_over"]
    assert decider.transcripts == []  # nothing is decided while the officer has the call
    ((call_id, number, text, personal_id),) = recorded
    assert (number, text, personal_id) == (1, "ರಸ್ತೆ", 1.0)
    assert ticket_store.transcript(call_id) == ["ರಸ್ತೆ"]


def test_a_command_the_console_gave_up_on_is_never_carried_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The take-over waits behind a slow decision, times out, and must not happen afterwards.
    class SlowDecider(ScriptedDecider):
        def decide(self, transcript, screening):
            time.sleep(0.3)
            return super().decide(transcript, screening)

    use_call_fakes(monkeypatch, SlowDecider(ROADS))
    monkeypatch.setattr(app_module, "hold_on_after_s", 0.01)
    monkeypatch.setattr(app_module, "OFFICER_COMMAND_TIMEOUT_S", 0.05)
    hold_on = app_module.TEMPLATES.lines["hold_on"]

    def officer(message: dict) -> list[dict]:
        event = message.get("event", {})
        if event.get("type") == "agent_text" and event["text"] == hold_on:
            return [{"type": "take_over", "call_id": message["call_id"], "id": 1}]
        return []

    console = ScriptedConsole(end_on_close(officer))
    browser = ScriptedBrowser([bytes(1024)], leave_when=ended_lines(3))
    asyncio.run(call_with_console(browser, console))

    (answer,) = [m for m in console.got if m.get("id") == 1]
    assert (answer["type"], answer["message"]) == ("error", "The call did not answer in time")
    assert lines_of(browser.sent, "dialogue", "keys") == [["greeting"], ["ask_location"]]
    assert not any(lines_of(browser.sent, "dialogue", "officer"))


def test_a_console_that_stops_reading_is_cut_off_not_left_to_grow() -> None:
    async def scenario() -> Watcher:
        watcher = Watcher()
        for number in range(MAX_BACKLOG):
            assert watcher.put({"n": number})
        assert not watcher.put({"n": MAX_BACKLOG})
        return watcher

    watcher = asyncio.run(scenario())
    assert watcher.messages.qsize() == 1 and watcher.messages.get_nowait() is None
