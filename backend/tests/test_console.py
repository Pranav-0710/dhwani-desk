"""The supervisor console's socket: who may connect, what it sees, and an officer taking a call
over, speaking on it and handing it back."""

import asyncio
import dataclasses
import time

import pytest
from fakes import ROADS, ROADS_IN_W04, FakeStream, ScriptedBrowser, ScriptedDecider
from fakes import isolated_settings as settings_with
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from test_call import FakeTts, ended_lines, lines_of

import app as app_module
from decisions import thresholds
from dialogue.live import MAX_BACKLOG, Watcher
from tickets.store import TicketStore

TOKEN = "console-test-token"


@pytest.fixture
def console_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        app_module, "get_settings", lambda: settings_with(gnani_api_key="t", console_token=TOKEN)
    )


def test_the_console_is_off_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "get_settings", lambda: settings_with(gnani_api_key="t"))
    with TestClient(app_module.app) as client, client.websocket_connect("/ws/console") as ws:
        assert ws.receive_json() == {"type": "error", "message": "CONSOLE_TOKEN is not set"}


@pytest.mark.parametrize(
    "hello",
    [{"type": "hello", "token": "wrong"}, {"type": "hello"}, {"token": TOKEN}, ["hello"]],
)
def test_a_console_without_the_right_token_is_refused(console_settings: None, hello) -> None:
    with TestClient(app_module.app) as client, client.websocket_connect("/ws/console") as ws:
        ws.send_json(hello)
        assert ws.receive_json() == {"type": "error", "message": "Not allowed"}
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.code == 1008


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
    with TestClient(app_module.app) as client:
        assert client.get("/api/thresholds").json()["safety_hazard"] == thresholds.SAFETY_HAZARD


class ScriptedConsole:
    """A console socket as uvicorn presents it. It says hello, then `react` answers each
    message it gets with the messages to send back; None ends the session."""

    def __init__(self, react) -> None:
        self.react = react
        self.inbox: asyncio.Queue[dict | None] = asyncio.Queue()
        self.inbox.put_nowait({"type": "hello", "token": TOKEN})
        self.got: list[dict] = []

    async def accept(self) -> None:
        return None

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
