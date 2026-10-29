"""Who may open the call sockets: allowed pages only, and a few sessions at a time."""

import asyncio
import time

import pytest
from fakes import PHONE_ORIGIN, FakeStream, ScriptedBrowser, isolated_settings
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

import app as app_module
from gnani.prisma import PrismaRejected

TUNNEL = "https://hosapura-demo.trycloudflare.com"


def use_settings(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> list[FakeStream]:
    """Point the app at a fresh FakeStream per session; returns the streams opened so far."""
    streams: list[FakeStream] = []

    def open_stream(settings: object) -> FakeStream:
        streams.append(FakeStream())
        return streams[-1]

    settings = isolated_settings(gnani_api_key="t", **overrides)
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(app_module, "stt_stream_factory", open_stream)
    return streams


def connected(sent: list) -> bool:
    return {"type": "connected"} in sent


@pytest.mark.parametrize("origin", [PHONE_ORIGIN, "http://127.0.0.1:5173"])
def test_the_phone_page_on_the_dev_server_may_connect(
    monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    use_settings(monkeypatch)
    browser = ScriptedBrowser([], leave_when=connected, origin=origin)

    asyncio.run(app_module.stt_proxy(browser))

    assert browser.sent[:2] == [{"type": "config", "sample_rate": 8000}, {"type": "connected"}]


def test_an_added_origin_such_as_the_tunnel_may_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    use_settings(monkeypatch, allowed_origins=f"{PHONE_ORIGIN}, {TUNNEL}/")
    # A scripted browser, not TestClient: leaving TestClient mid-relay can fail its teardown.
    browser = ScriptedBrowser([], leave_when=connected, origin=TUNNEL)

    asyncio.run(app_module.stt_proxy(browser))

    assert browser.accepted
    assert browser.sent[0]["type"] == "config"


@pytest.mark.parametrize("origin", ["https://evil.example", "http://localhost:5174", None])
def test_any_other_page_is_refused_before_the_handshake(
    monkeypatch: pytest.MonkeyPatch, origin: str | None
) -> None:
    streams = use_settings(monkeypatch)
    client = TestClient(app_module.app, headers={} if origin is None else {"origin": origin})

    with pytest.raises(WebSocketDisconnect) as refused, client.websocket_connect("/ws/stt"):
        pass

    assert refused.value.code == 1008
    assert streams == []  # Prisma never opened: no credits spent


def test_the_call_socket_refuses_other_pages_too(monkeypatch: pytest.MonkeyPatch) -> None:
    streams = use_settings(monkeypatch)
    browser = ScriptedBrowser([], origin="https://evil.example")

    asyncio.run(app_module.call(browser))

    assert not browser.accepted
    assert browser.closed_with == 1008
    assert browser.sent == []
    assert streams == []
    assert list(app_module.live_calls) == []  # no call for the consoles to see


def test_a_session_past_the_cap_hears_the_line_is_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    streams = use_settings(monkeypatch, max_stt_sessions=1)
    busy = ScriptedBrowser([])
    # The first caller stays on the line until the second has been turned away.
    first = ScriptedBrowser([], leave_when=lambda sent: busy.closed_with is not None)

    async def scenario() -> None:
        on_the_line = asyncio.create_task(app_module.stt_proxy(first))
        async with asyncio.timeout(5):
            while not connected(first.sent):
                await asyncio.sleep(0.01)
        await app_module.call(busy)  # /ws/call counts against the same cap
        await on_the_line

    asyncio.run(scenario())

    assert busy.sent == [app_module.LINE_BUSY]
    assert busy.closed_with == 1013
    assert len(streams) == 1  # the busy caller never reached Prisma
    assert first.closed_with is None  # the first caller's session ran on, then they left
    assert app_module.stt_sessions.open == 0


def test_the_cap_defaults_to_two_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    use_settings(monkeypatch)
    third = ScriptedBrowser([])
    callers = [ScriptedBrowser([], leave_when=lambda sent: third.closed_with is not None)]
    callers.append(ScriptedBrowser([], leave_when=lambda sent: third.closed_with is not None))

    async def scenario() -> None:
        on_the_line = [asyncio.create_task(app_module.stt_proxy(c)) for c in callers]
        async with asyncio.timeout(5):
            while not all(connected(c.sent) for c in callers):
                await asyncio.sleep(0.01)
        await app_module.stt_proxy(third)
        await asyncio.gather(*on_the_line)

    asyncio.run(scenario())

    assert third.sent == [app_module.LINE_BUSY]


def test_a_failed_session_frees_its_place(monkeypatch: pytest.MonkeyPatch) -> None:
    use_settings(monkeypatch, max_stt_sessions=1)
    monkeypatch.setattr(
        app_module, "stt_stream_factory", lambda settings: FakeStream(PrismaRejected(401))
    )
    browser = ScriptedBrowser([])

    asyncio.run(app_module.stt_proxy(browser))

    assert browser.sent[0]["type"] == "error"
    assert app_module.stt_sessions.open == 0


def test_a_session_that_is_cancelled_frees_its_place(monkeypatch: pytest.MonkeyPatch) -> None:
    use_settings(monkeypatch, max_stt_sessions=1)
    browser = ScriptedBrowser([], leave_when=lambda sent: False)

    async def scenario() -> None:
        on_the_line = asyncio.create_task(app_module.stt_proxy(browser))
        async with asyncio.timeout(5):
            while not connected(browser.sent):
                await asyncio.sleep(0.01)
        assert app_module.stt_sessions.open == 1
        on_the_line.cancel()
        await asyncio.gather(on_the_line, return_exceptions=True)

    asyncio.run(scenario())

    assert app_module.stt_sessions.open == 0


def test_a_socket_that_sends_no_audio_is_hung_up(monkeypatch: pytest.MonkeyPatch) -> None:
    streams = use_settings(monkeypatch)
    monkeypatch.setattr(app_module, "stt_quiet_s", 0.1)
    # Open and silent: without the limit it would hold a place until Prisma's 15-minute cap.
    browser = ScriptedBrowser([], leave_when=lambda sent: False)

    asyncio.run(app_module.stt_proxy(browser))

    assert browser.sent[-1] == app_module.WENT_QUIET
    assert browser.closed_with == 1008
    assert streams[0].discarded


class PacedBrowser(ScriptedBrowser):
    """Sends a frame every `gap` seconds. ScriptedBrowser's own 10 ms wait can return at once
    on Windows, whose timers tick every 15.6 ms."""

    gap = 0.05

    async def receive(self) -> dict:
        await asyncio.sleep(self.gap)
        return await super().receive()


def test_steady_audio_keeps_the_session_open(monkeypatch: pytest.MonkeyPatch) -> None:
    use_settings(monkeypatch)
    monkeypatch.setattr(app_module, "stt_quiet_s", 0.2)
    # A frame every 50 ms for well past the limit, as the phone page sends, silence or not.
    browser = PacedBrowser([bytes(160)] * 10)
    started = time.monotonic()

    asyncio.run(app_module.stt_proxy(browser))

    assert time.monotonic() - started > 2 * app_module.stt_quiet_s  # the limit was tested
    assert app_module.WENT_QUIET not in browser.sent
    assert browser.closed_with is None  # the browser left; the server never hung up


def test_settings_tidy_the_origin_list() -> None:
    settings = isolated_settings(allowed_origins=f" {TUNNEL}/ ,, {PHONE_ORIGIN} ")

    assert settings.origins == {TUNNEL, PHONE_ORIGIN}
    assert isolated_settings(allowed_origins="").origins == frozenset()


def test_settings_write_origins_as_browsers_send_them() -> None:
    settings = isolated_settings(
        allowed_origins="HTTPS://Hosapura-Demo.trycloudflare.com:443,http://LocalHost:80,"
        "http://localhost:5173"
    )

    assert settings.origins == {TUNNEL, "http://localhost", PHONE_ORIGIN}


@pytest.mark.parametrize(
    "origins", ["localhost:5173", "https://example.com/phone", "ws://localhost:5173", "*"]
)
def test_settings_reject_what_is_not_an_origin(origins: str) -> None:
    with pytest.raises(ValidationError, match="ALLOWED_ORIGINS"):
        isolated_settings(allowed_origins=origins)


def test_settings_reject_a_cap_below_one() -> None:
    with pytest.raises(ValidationError, match="MAX_STT_SESSIONS"):
        isolated_settings(max_stt_sessions=0)
