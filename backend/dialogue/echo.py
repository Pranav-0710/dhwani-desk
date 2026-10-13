"""Placeholder agent for issue #4: repeats each caller segment back. #9 replaces it with the
dialogue state machine behind the same `reply()` method."""

ECHO_PREFIX = "ನೀವು ಹೇಳಿದ್ದು: "  # "What you said: "


class EchoAgent:
    def reply(self, transcript: str) -> str | None:
        text = transcript.strip()
        return f"{ECHO_PREFIX}{text}" if text else None
