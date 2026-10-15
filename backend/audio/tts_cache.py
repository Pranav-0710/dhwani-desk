"""Reuses the audio of the agent's fixed lines across calls.

Timbre's first audio takes ~0.8 s and sometimes over 4 s (docs/decisions/tts-voice.md). The
greeting, the questions and the hand-off lines never change, so each is synthesised once and
then played from memory; lines with a ticket number or a ward name still go to Timbre.
"""

import contextlib
from collections.abc import AsyncIterator, Iterable

from audio.speaker import SpeechStream

# (voice, sample rate, text) -> the line's PCM chunks. Shared by every call in this process.
AudioCache = dict[tuple[str, int, str], list[bytes]]


class CachedSpeech:
    """A `SpeechStream` that serves `cacheable` lines from `cache` once they have been heard.

    Only a line that streamed to the end is stored, so a failed or interrupted synthesis is
    never replayed.
    """

    def __init__(
        self,
        tts: SpeechStream,
        cache: AudioCache,
        cacheable: Iterable[str],
        voice: str,
        sample_rate: int,
    ) -> None:
        self._tts = tts
        self._cache = cache
        self._cacheable = frozenset(cacheable)
        self._voice = voice
        self._sample_rate = sample_rate

    async def stream(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
        key = (voice or self._voice, self._sample_rate, text)
        if (chunks := self._cache.get(key)) is not None:
            for pcm in chunks:
                yield pcm
            return
        heard: list[bytes] = []
        async with contextlib.aclosing(self._tts.stream(text, voice)) as live:
            async for pcm in live:
                heard.append(pcm)
                yield pcm
        if text in self._cacheable:
            self._cache[key] = heard
