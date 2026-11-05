"""Wordings that always count as a danger or a request for an officer, whatever the model answers.

A net around a decider: it can only raise `safety_hazard` and `wants_human`, never lower them, and
it changes nothing else. A hit answers 1.0, so a ticket shows the rule fired, and `revision` names
this list, so a stored decision says which list was in use.

The hazards come from the labelling rule in data/README.md (live wires, open manholes, sewage in
homes, contaminated drinking water, animal attacks, fire). The list was tried on generated rows
the model never trained on, not on the held-out test sentences, and checked by a non-native reader
(docs/decisions/safety-phrases.md).

Asking for an officer needs the asking, not just the word: "the officer never came" is a complaint.
"""

import dataclasses
import re

from dialogue.decision import Decider, Decision, Screening

RULESET = "safety-phrases-1"

HAZARD_PHRASES = (
    # live wires
    "ವೈರ್ ಬಿದ್ದ",
    "ಸ್ಪಾರ್ಕ್",
    "ಕರೆಂಟ್ ಶಾಕ್",
    "ಶಾಕ್ ಹೊಡೆ",
    # open manholes
    "ಹೋಲ್ ಮುಚ್ಚಳ ಇಲ್ಲ",
    "ಹೋಲ್ಗೆ ಮುಚ್ಚಳ ಇಲ್ಲ",
    "ಮ್ಯಾನ್ ಹೋಲ್ ತೆರೆದ",
    "ಮ್ಯಾನ್ಹೋಲ್ ತೆರೆದ",
    "ಮ್ಯಾನ್ ಹೋಲ್ ಓಪನ್",
    "ಮ್ಯಾನ್ಹೋಲ್ ಓಪನ್",
    # sewage in homes
    "ಕೊಳಚೆ ನೀರು ಮನೆ",
    "ಚರಂಡಿ ನೀರು ಮನೆ",
    # contaminated drinking water
    "ಕುಡಿಯೋ ನೀರಲ್ಲಿ",
    "ಕುಡಿಯುವ ನೀರಲ್ಲಿ",
    # animal attacks
    "ಕಚ್ಚಿದೆ",
    "ಕಚ್ಚಿತು",
    "ಹಾವು",
    # fire
    "ಬೆಂಕಿ",
)
# Matched inside a word, so "ಆಫೀಸರ್ಗೆ" and "ಅಧಿಕಾರಿಯನ್ನ" count.
OFFICER_WORDS = ("ಆಫೀಸರ್", "ಅಧಿಕಾರಿ", "ಮೇಲಧಿಕಾರಿ", "ಸೂಪರ್ವೈಸರ್", "ಮನುಷ್ಯರು")
# Whole words only: "ನೋಡ್ಬೇಕು" ("should look at it") ends in "ಬೇಕು" but asks for no one.
ASKING_WORDS = frozenset(
    {
        "ಬೇಕು",
        "ಕೊಡಿ",
        "ಮಾತಾಡಿ",
        "ಮಾತಾಡಿಸಿ",
        "ಮಾತಾಡ್ಬೇಕು",
        "ಮಾತಾಡಬೇಕು",
        "ಮಾತಾಡಿಸಬೇಕು",
        "ಮಾತನಾಡಿ",
        "ಮಾತನಾಡಿಸಿ",
        "ಮಾತನಾಡಬೇಕು",
    }
)
# How many words after the officer word the asking may come.
ASKING_WITHIN = 3
NOT_A_MACHINE_PHRASES = ("ಮೆಷಿನ್ ಬೇಡ", "ರೋಬೋಟ್ ಜೊತೆ")
_WORD = re.compile(r"[^\s,.;:!?।]+")


def says_hazard(text: str) -> bool:
    return any(phrase in text for phrase in HAZARD_PHRASES)


def asks_for_officer(text: str) -> bool:
    if any(phrase in text for phrase in NOT_A_MACHINE_PHRASES):
        return True
    words = _WORD.findall(text)
    for i, word in enumerate(words):
        is_officer = any(officer in word for officer in OFFICER_WORDS)
        if is_officer and ASKING_WORDS.intersection(words[i : i + ASKING_WITHIN + 1]):
            return True
    return False


class WithSafetyPhrases:
    """A decider with the phrase rules on its safety step. Its other answers are unchanged."""

    def __init__(self, decider: Decider) -> None:
        self._decider = decider
        self.revision = f"{decider.revision}+{RULESET}"

    def screen(self, transcript: str) -> Screening:
        answer = self._decider.screen(transcript)
        if says_hazard(transcript):
            answer = dataclasses.replace(answer, safety_hazard=1.0)
        if asks_for_officer(transcript):
            answer = dataclasses.replace(answer, wants_human=1.0)
        return answer

    def decide(self, transcript: str, screening: Screening) -> Decision:
        return self._decider.decide(transcript, screening)
