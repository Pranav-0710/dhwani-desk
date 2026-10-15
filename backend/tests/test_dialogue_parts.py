"""The pieces around the state machine: the keyword stand-in, turn assembly, the audio cache."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from audio.tts_cache import AudioCache, CachedSpeech
from dialogue.standin import KeywordDecider
from dialogue.turns import TurnCollector
from gnani.timbre import TimbreError

DECIDER = KeywordDecider()


@pytest.mark.parametrize(
    ("text", "department", "ward"),
    [
        ("ನಮ್ಮ ಏರಿಯಾದಲ್ಲಿ ಮೂರು ದಿನದಿಂದ ನೀರು ಬಂದಿಲ್ಲ ಹೂವಿನಕೇರಿ", "water", "w01"),
        ("ತೋಟದಪಾಳ್ಯ ಮಸೀದಿ ಹತ್ತಿರ ಚರಂಡಿ ಬ್ಲಾಕ್ ಆಗಿದೆ", "sewage-drainage", "w07"),
        ("ವಾರ್ಡ್ ಫೋರ್ ಅಲ್ಲಿ ಸ್ಟ್ರೀಟ್ ಲೈಟ್ ಹತ್ತುತ್ತಿಲ್ಲ", "streetlights", "w04"),
        ("ವಾರ್ಡ್ 10 ಅಲ್ಲಿ ಬೀದಿ ನಾಯಿ ಜಾಸ್ತಿ ಆಗಿದೆ", "stray-animals", "w10"),
    ],
)
def test_keyword_stand_in_routes_plain_complaints(text: str, department: str, ward: str) -> None:
    decision = DECIDER.decide(text)
    assert (decision.department, decision.ward) == (department, ward)
    assert decision.department_confidence > 0.6 and decision.has_location > 0.5


def test_keyword_stand_in_flags_escalations_and_stays_unsure_on_nothing() -> None:
    assert DECIDER.decide("ಕಂಬದಿಂದ ವೈರ್ ಬಿದ್ದು ಸ್ಪಾರ್ಕ್ ಆಗ್ತಿದೆ").safety_hazard > 0.6
    assert DECIDER.decide("ಕಂಬದಿಂದ ವೈರ್ ಬಿದ್ದು ಸ್ಪಾರ್ಕ್ ಆಗ್ತಿದೆ").urgency == "emergency"
    assert DECIDER.decide("ಆಫೀಸರ್ ಜೊತೆ ಮಾತಾಡಬೇಕು").wants_human > 0.7
    chest_pain = DECIDER.decide("ಅಪ್ಪನಿಗೆ ಎದೆ ನೋವು ಬರ್ತಿದೆ ಆಂಬುಲೆನ್ಸ್ ಬೇಕು")
    assert chest_pain.safety_hazard > 0.6 and chest_pain.out_of_scope > 0.7
    redirect = DECIDER.decide("ಕರೆಂಟ್ ಬಿಲ್ ತುಂಬಾ ಜಾಸ್ತಿ ಬಂದಿದೆ")
    assert redirect.out_of_scope > 0.7 and redirect.department == "other"
    nothing = DECIDER.decide("ಹಲೋ")
    assert nothing.department_confidence < 0.6
    assert (nothing.ward, nothing.caller_state) == ("unknown", "calm")
    assert DECIDER.decide("ನನ್ನ ನಂಬರ್ 1234567890").contains_personal_id > 0.5


def collect_turns(segments: list[tuple[float, str]], pause_s: float = 0.05) -> list[str]:
    """Feed (delay before, text) segments to a collector and return the finished turns."""

    async def scenario() -> list[str]:
        collector = TurnCollector(pause_s)
        for delay, text in segments:
            await asyncio.sleep(delay)
            collector.add(text)
        await asyncio.sleep(pause_s * 4)
        await collector.aclose()
        turns = []
        while not collector.turns.empty():
            turns.append(collector.turns.get_nowait())
        return turns

    return asyncio.run(scenario())


def test_segments_close_together_make_one_turn() -> None:
    assert collect_turns([(0, "ವಾರ್ಡ್ ಫೋರ್ ಅಲ್ಲಿ"), (0, " ಲೈಟ್ ಹತ್ತುತ್ತಿಲ್ಲ ")]) == ["ವಾರ್ಡ್ ಫೋರ್ ಅಲ್ಲಿ ಲೈಟ್ ಹತ್ತುತ್ತಿಲ್ಲ"]


def test_a_pause_ends_the_turn_and_blank_segments_are_ignored() -> None:
    assert collect_turns([(0, "ರೋಡ್ ಹಾಳಾಗಿದೆ"), (0, "  "), (0.3, "ಗಾಂಧಿನಗರ")]) == [
        "ರೋಡ್ ಹಾಳಾಗಿದೆ",
        "ಗಾಂಧಿನಗರ",
    ]


def test_closing_drops_a_turn_still_waiting_for_its_pause() -> None:
    async def scenario() -> bool:
        collector = TurnCollector(pause_s=10)
        collector.add("ರೋಡ್")
        await collector.aclose()
        return collector.turns.empty()

    assert asyncio.run(scenario())


class CountingTts:
    def __init__(self, fail: bool = False) -> None:
        self.requests: list[str] = []
        self.fail = fail

    async def stream(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
        self.requests.append(text)
        yield b"\x01\x00"
        if self.fail:
            raise TimbreError(503, "SERVICE_UNAVAILABLE", "busy")
        yield b"\x02\x00"


async def hear(speech: CachedSpeech, text: str) -> list[bytes]:
    return [pcm async for pcm in speech.stream(text)]


def test_a_fixed_line_is_synthesised_once_and_then_replayed() -> None:
    tts, cache = CountingTts(), {}
    speech = CachedSpeech(tts, cache, ["ನಮಸ್ಕಾರ"], voice="Saanvi", sample_rate=16000)

    async def scenario() -> tuple[list[bytes], list[bytes]]:
        return await hear(speech, "ನಮಸ್ಕಾರ"), await hear(speech, "ನಮಸ್ಕಾರ")

    first, second = asyncio.run(scenario())
    assert first == second == [b"\x01\x00", b"\x02\x00"]
    assert tts.requests == ["ನಮಸ್ಕಾರ"]


def test_lines_with_numbers_are_never_cached_and_voices_do_not_share() -> None:
    tts, cache = CountingTts(), {}
    saanvi = CachedSpeech(tts, cache, ["ನಮಸ್ಕಾರ"], voice="Saanvi", sample_rate=16000)
    kavin = CachedSpeech(tts, cache, ["ನಮಸ್ಕಾರ"], voice="Kavin", sample_rate=16000)

    async def scenario() -> None:
        for _ in range(2):
            await hear(saanvi, "ಟಿಕೆಟ್ ಸಂಖ್ಯೆ 1 0 0 4")
        await hear(saanvi, "ನಮಸ್ಕಾರ")
        await hear(kavin, "ನಮಸ್ಕಾರ")

    asyncio.run(scenario())
    assert tts.requests == ["ಟಿಕೆಟ್ ಸಂಖ್ಯೆ 1 0 0 4"] * 2 + ["ನಮಸ್ಕಾರ"] * 2


def test_a_failed_synthesis_is_not_cached() -> None:
    cache: AudioCache = {}
    speech = CachedSpeech(CountingTts(fail=True), cache, ["ನಮಸ್ಕಾರ"], "Saanvi", 16000)

    async def scenario() -> None:
        with pytest.raises(TimbreError):
            await hear(speech, "ನಮಸ್ಕಾರ")

    asyncio.run(scenario())
    assert cache == {}
