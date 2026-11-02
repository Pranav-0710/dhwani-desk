"""Who may open the call sockets: allowed pages only, and a few sessions at a time."""

import asyncio
import time

import pytest
from fakes import PHONE_ORIGIN, FakeStream, ScriptedBrowser, isolated_settings
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect
from test_call import FakeTts

import app as app_module
from config import BACKEND_DIR, Settings
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
    # Three different callers, so only the total cap is tested here.
    for number, browser in enumerate([*callers, third]):
        browser.headers["x-real-ip"] = f"198.51.100.{number}"

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
    # A frame of 8 kHz audio every 50 ms for well past the limit, as the phone page streams,
    # silence or not. Each is enough on its own, so a slow timer here cannot fail the test.
    browser = PacedBrowser([bytes(1600)] * 10)
    started = time.monotonic()

    asyncio.run(app_module.stt_proxy(browser))

    assert time.monotonic() - started > 2 * app_module.stt_quiet_s  # the limit was tested
    assert app_module.WENT_QUIET not in browser.sent
    assert browser.closed_with is None  # the browser left; the server never hung up


def test_a_byte_now_and_then_does_not_keep_the_session_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_settings(monkeypatch)
    monkeypatch.setattr(app_module, "stt_quiet_s", 0.2)
    # Often enough to have reset the limit every time while any audio at all did.
    browser = PacedBrowser([b"\x00"] * 40, leave_when=lambda sent: False)

    asyncio.run(app_module.stt_proxy(browser))

    assert browser.sent[-1] == app_module.WENT_QUIET
    assert browser.closed_with == 1008


def test_a_slow_permission_prompt_does_not_end_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    use_settings(monkeypatch)
    monkeypatch.setattr(app_module, "stt_quiet_s", 0.5)

    class AnswersThePromptLate(PacedBrowser):
        waited = False

        async def receive(self) -> dict:
            # The mic opens shortly before the limit, with too little audio by then to count.
            if not self.waited:
                self.waited = True
                await asyncio.sleep(0.3)
            return await super().receive()

    browser = AnswersThePromptLate([bytes(640)] * 12)

    asyncio.run(app_module.stt_proxy(browser))

    assert app_module.WENT_QUIET not in browser.sent
    assert browser.closed_with is None


def test_the_raw_relay_is_off_unless_it_is_turned_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STT_RELAY_ENABLED", raising=False)
    assert Settings(_env_file=None).stt_relay_enabled is False

    streams = use_settings(monkeypatch, stt_relay_enabled=False)
    browser = ScriptedBrowser([])

    asyncio.run(app_module.stt_proxy(browser))

    assert not browser.accepted
    assert browser.closed_with == 1008
    assert browser.sent == []
    assert streams == []  # Prisma never opened: no credits spent


def test_the_phone_page_still_calls_with_the_raw_relay_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streams = use_settings(monkeypatch, stt_relay_enabled=False)
    monkeypatch.setattr(app_module, "tts_client_factory", lambda settings: FakeTts())
    browser = ScriptedBrowser([], leave_when=connected)

    asyncio.run(app_module.call(browser))

    assert browser.accepted
    assert connected(browser.sent)
    assert len(streams) == 1


def test_a_call_is_refused_once_the_days_minutes_are_spent(
    monkeypatch: pytest.MonkeyPatch, call_budget: app_module.CallBudget
) -> None:
    streams = use_settings(monkeypatch, max_call_minutes_per_day=10)
    call_budget.charge(10 * 60)
    browser = ScriptedBrowser([])

    asyncio.run(app_module.stt_proxy(browser))

    assert browser.sent == [app_module.LINE_RESTING]
    assert browser.closed_with == 1013
    assert streams == []  # refused before Prisma opened
    assert app_module.stt_sessions.open == 0


def test_a_call_ending_is_charged_to_the_days_minutes(
    monkeypatch: pytest.MonkeyPatch, call_budget: app_module.CallBudget
) -> None:
    streams = use_settings(monkeypatch)
    charged: list[float] = []
    monkeypatch.setattr(call_budget, "charge", charged.append)

    asyncio.run(app_module.stt_proxy(PacedBrowser([], leave_when=connected)))

    assert len(streams) == 1
    assert len(charged) == 1 and charged[0] >= PacedBrowser.gap / 2


def test_a_refused_call_is_not_charged(
    monkeypatch: pytest.MonkeyPatch, call_budget: app_module.CallBudget
) -> None:
    use_settings(monkeypatch, max_calls_per_client_per_hour=1)
    charged: list[float] = []
    monkeypatch.setattr(call_budget, "charge", charged.append)

    asyncio.run(app_module.stt_proxy(ScriptedBrowser([], leave_when=connected)))
    charged.clear()
    second = ScriptedBrowser([])
    asyncio.run(app_module.stt_proxy(second))

    assert second.sent == [app_module.LINE_RESTING]
    assert charged == []


def test_one_caller_can_only_start_so_many_calls_an_hour(monkeypatch: pytest.MonkeyPatch) -> None:
    streams = use_settings(monkeypatch, max_calls_per_client_per_hour=2)

    def from_address(address: str) -> ScriptedBrowser:
        browser = ScriptedBrowser([], leave_when=connected)
        browser.headers["x-real-ip"] = address
        return browser

    for _ in range(2):
        asyncio.run(app_module.stt_proxy(from_address("203.0.113.7")))
    refused = from_address("203.0.113.7")
    asyncio.run(app_module.stt_proxy(refused))
    someone_else = from_address("203.0.113.8")
    asyncio.run(app_module.stt_proxy(someone_else))

    assert refused.sent == [app_module.LINE_RESTING]
    assert connected(someone_else.sent)
    assert len(streams) == 3


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_the_budget_counts_minutes_over_a_rolling_day() -> None:
    clock = Clock()
    budget = app_module.CallBudget(clock)

    assert budget.admit("a", 0, 10) is None
    budget.charge(9 * 60)
    assert budget.admit("a", 0, 10) is None  # under the ceiling: a call may start
    budget.charge(60)
    assert "10 minutes" in budget.admit("a", 0, 10)

    clock.now += 86400.0 + 1
    assert budget.admit("a", 0, 10) is None  # the day's minutes have rolled off


def test_the_budget_counts_calls_per_caller_over_a_rolling_hour() -> None:
    clock = Clock()
    budget = app_module.CallBudget(clock)

    assert budget.admit("a", 2, 0) is None
    assert budget.admit("a", 2, 0) is None
    assert "2 calls" in budget.admit("a", 2, 0)
    assert budget.admit("b", 2, 0) is None

    clock.now += 3600.0 + 1
    assert budget.admit("a", 2, 0) is None
    assert budget._by_client == {"a": 1}  # nobody's old calls are kept in memory


def test_a_limit_of_zero_is_off() -> None:
    budget = app_module.CallBudget()
    budget.charge(10**9)

    for _ in range(100):
        assert budget.admit("a", 0, 0) is None


def test_all_of_an_ipv6_64_is_one_caller() -> None:
    first, second = ScriptedBrowser([]), ScriptedBrowser([])
    first.headers["x-real-ip"] = "2001:db8:1:2:aaaa::1"
    second.headers["x-real-ip"] = "2001:db8:1:2:bbbb::2"
    other = ScriptedBrowser([])
    other.headers["x-real-ip"] = "2001:db8:1:3::1"

    assert app_module._client(first) == app_module._client(second) == "2001:db8:1:2::/64"
    assert app_module._client(other) == "2001:db8:1:3::/64"


@pytest.mark.parametrize("address", ["203.0.113.7", "not an address"])
def test_any_other_address_is_kept_as_it_came(address: str) -> None:
    browser = ScriptedBrowser([])
    browser.headers["x-real-ip"] = address

    assert app_module._client(browser) == address


def test_a_rejected_console_token_is_not_printed() -> None:
    with pytest.raises(ValidationError) as rejected:
        isolated_settings(console_token="short-secret")

    assert "short-secret" not in str(rejected.value)


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("max_consoles", 0, "MAX_CONSOLES"),
        ("max_console_hellos", 0, "MAX_CONSOLE_HELLOS"),
        ("max_call_minutes_per_day", -1, "cannot be negative"),
        ("max_calls_per_client_per_hour", -1, "cannot be negative"),
        ("max_officer_lines_queued", -1, "cannot be negative"),
        ("min_audio_share", 1.5, "MIN_AUDIO_SHARE"),
        ("ticket_location_retention_days", 0, "TICKET_LOCATION_RETENTION_DAYS"),
    ],
)
def test_settings_reject_limits_that_make_no_sense(
    setting: str, value: object, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        isolated_settings(**{setting: value})


def test_every_limit_is_listed_in_the_example_settings() -> None:
    example = (BACKEND_DIR.parent / ".env.example").read_text(encoding="utf-8")

    for name in (
        "CONSOLE_LOCAL_ONLY",
        "MAX_CONSOLES",
        "MAX_CONSOLE_HELLOS",
        "MAX_CONSOLE_HELLOS_PER_CLIENT",
        "MAX_OFFICER_LINES_QUEUED",
        "STT_RELAY_ENABLED",
        "MAX_CALL_MINUTES_PER_DAY",
        "MAX_CALLS_PER_CLIENT_PER_HOUR",
        "MIN_AUDIO_SHARE",
        "TICKET_LOCATION_RETENTION_DAYS",
        "WS_MAX_FRAME_BYTES",
    ):
        assert f"\n{name}=" in example, name


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


def test_a_short_console_token_is_refused() -> None:
    with pytest.raises(ValidationError, match="at least 20 characters"):
        isolated_settings(console_token="demo")
    assert isolated_settings(console_token="").console_token.get_secret_value() == ""
    assert isolated_settings(console_token="x" * 20).console_token.get_secret_value() == "x" * 20


def test_one_caller_cannot_take_every_line(monkeypatch: pytest.MonkeyPatch) -> None:
    use_settings(monkeypatch)  # two lines in all, one per caller
    second = ScriptedBrowser([])
    first = ScriptedBrowser([], leave_when=lambda sent: second.closed_with is not None)
    for browser in (first, second):
        browser.headers["x-real-ip"] = "203.0.113.7"

    async def scenario() -> None:
        on_the_line = asyncio.create_task(app_module.stt_proxy(first))
        async with asyncio.timeout(5):
            while not connected(first.sent):
                await asyncio.sleep(0.01)
        await app_module.stt_proxy(second)
        await on_the_line

    asyncio.run(scenario())

    assert second.sent == [app_module.LINE_BUSY]


def test_a_session_ends_at_its_time_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    use_settings(monkeypatch)
    monkeypatch.setattr(app_module, "stt_max_session_s", 0.0)
    # Audio keeps arriving, so only the time limit can end it.
    browser = ScriptedBrowser([bytes(320)] * 5, leave_when=lambda sent: False)

    asyncio.run(app_module.stt_proxy(browser))

    assert browser.sent[-1] == app_module.TOO_LONG
    assert browser.closed_with == 1008
