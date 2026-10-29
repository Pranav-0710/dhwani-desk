"""The two decisions made by code instead of Laya: which ward, and whether a personal number
was said. Both are lookups, not judgements, and zero-shot Laya got them badly wrong
(docs/decisions/laya-kannada-gate.md): it never answered "unknown" for the ward and flagged a
personal ID in most turns. Code cannot make those mistakes, answers instantly, and leaves Laya
fewer questions per turn.

The ward matcher is the one the training data was labelled with (training/generate_data.py),
so serving and training agree on what counts as naming a ward.
"""

import json
import re
import unicodedata

from config import REPO_ROOT
from tickets.masking import LONG_NUMBER

WARDS = json.loads((REPO_ROOT / "data" / "wards.json").read_text("utf-8"))["wards"]
TEMPLATES = json.loads((REPO_ROOT / "data" / "generation" / "scenarios.json").read_text("utf-8"))

ZERO_WIDTH = re.compile(r"[‌‍]")
VIRAMA = "್"
# Vowel signs Kannada adds when an English loanword ending in a consonant takes a suffix
# (ಚರ್ಚ್ → ಚರ್ಚಿನ, ಟ್ಯಾಂಕ್ → ಟ್ಯಾಂಕು). Any other sign makes another word (ಚರ್ಚೆ, "discussion").
LOANWORD_VOWELS = "ಿು"


def _normalise(text: str) -> str:
    return ZERO_WIDTH.sub("", unicodedata.normalize("NFC", text))


def _squash(text: str) -> str:
    """Without spaces, so "ಗಾಂಧಿ ನಗರ" and "ಗಾಂಧಿನಗರ" compare equal."""
    return re.sub(r"\s", "", _normalise(text))


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """A phrase ending in a virama also matches its inflected forms, but only through the
    virama or a loanword vowel: ಟ್ಯಾಂಕ್ matches ಟ್ಯಾಂಕು, not ಟ್ಯಾಂಕರ್ (tanker)."""
    target = _squash(phrase)
    if target.endswith(VIRAMA):
        return re.compile(re.escape(target[:-1]) + f"[{VIRAMA}{LOANWORD_VOWELS}]")
    return re.compile(re.escape(target))


WARD_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    ward["id"]: [
        _phrase_pattern(phrase)
        for phrase in [
            ward["name"],
            *(landmark["name"] for landmark in ward["landmarks"]),
            *TEMPLATES["ward_aliases"].get(ward["id"], []),
        ]
    ]
    for ward in WARDS
}
WARD_BY_NUMBER = {str(ward["number"]): ward["id"] for ward in WARDS}
# Number words as Prisma may leave them, in Kannada script: English (ಫೋರ್) and Kannada (ಎರಡು).
NUMBER_BY_WORD = {
    word: number
    for key in ("english_number_words", "kannada_number_words")
    for number, word in TEMPLATES[key].items()
}
NUMBER_BY_ORDINAL = {word: number for number, word in TEMPLATES["kannada_ordinals"].items()}


def _alternatives(words: dict[str, str]) -> str:
    return "|".join(sorted(words, key=len, reverse=True))


WARD_NUMBER = re.compile(
    # ವಾರ್ಡ್ 4, ವಾರ್ಡ್ ನಂಬರ್ 4, ವಾರ್ಡ್ ಫೋರ್, ವಾರ್ಡ್ ನಾಲ್ಕು
    r"ವಾರ್ಡ್\s*(?:ನಂಬರ್\s*)?(?P<after>\d+(?!\d)|"
    + _alternatives(NUMBER_BY_WORD)
    + ")"
    # 4ನೇ ವಾರ್ಡ್, 4 ನೇ ವಾರ್ಡ್, ನಾಲ್ಕನೇ ವಾರ್ಡ್
    + r"|(?:(?P<ordinal>\d+)\s*ನೇ|(?P<word>"
    + _alternatives(NUMBER_BY_ORDINAL)
    + r"))\s*ವಾರ್ಡ್"
)


def find_ward(transcript: str) -> str:
    """The ward the caller points to, or "unknown". When several are mentioned (a correction:
    "not Kerekodi, Gandhinagar"), the last one said wins. A ward number that does not exist
    (ward 11) counts as no ward."""
    squashed = _squash(transcript)
    last: dict[str, int] = {}
    for ward_id, patterns in WARD_PATTERNS.items():
        for pattern in patterns:
            for match in pattern.finditer(squashed):
                last[ward_id] = max(last.get(ward_id, -1), match.start())
    normalised = _normalise(transcript)
    for match in WARD_NUMBER.finditer(normalised):
        if match["word"]:
            number = NUMBER_BY_ORDINAL[match["word"]]
        else:
            raw = match["after"] or match["ordinal"]
            number = NUMBER_BY_WORD.get(raw, raw)
        if ward_id := WARD_BY_NUMBER.get(number):
            # In squashed coordinates, so names and numbers compare by position.
            position = len(_squash(normalised[: match.start()]))
            last[ward_id] = max(last.get(ward_id, -1), position)
    return max(last, key=last.__getitem__) if last else "unknown"


def says_personal_number(transcript: str) -> bool:
    """A phone, Aadhaar or account number: the same rule that masks it before storage."""
    return LONG_NUMBER.search(transcript) is not None
