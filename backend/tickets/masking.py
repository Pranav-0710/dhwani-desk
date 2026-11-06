"""Personal numbers never reach storage (docs/PRD.md, "Masking before storage")."""

import re

# A phone, Aadhaar or account number: 10+ digits, which callers say in groups and which
# transcripts may separate with spaces or punctuation ("98450 12345", "1234-5678-9012",
# "(984) 501-2345"). Up to three non-letter characters may sit between digits.
LONG_NUMBER = re.compile(r"\d(?:[^\w]{0,3}\d){9,}")
# Once the decider thinks the caller gave a personal ID, shorter runs go too (a partial
# account number), but not 1–3 digit numbers: ward 4, 3 days, a count of lights.
ID_LIKE_NUMBER = re.compile(r"\d(?:[^\w]{0,3}\d){3,}")
PERSONAL_ID = 0.5
MASK = "X"

# The same numbers said digit by digit in words, as Prisma may write them: Kannada (ಒಂಬತ್ತು
# ಎಂಟು ...) or English in Kannada script (ನೈನ್ ಎಯ್ಟ್ ...), possibly mixed with digits.
DIGIT_WORDS = [
    "ಸೊನ್ನೆ",
    "ಒಂದು",
    "ಎರಡು",
    "ಮೂರು",
    "ನಾಲ್ಕು",
    "ಐದು",
    "ಆರು",
    "ಏಳು",
    "ಎಂಟು",
    "ಒಂಬತ್ತು",
    "ಜೀರೋ",
    "ಝೀರೋ",
    "ಒನ್",
    "ಟೂ",
    "ತ್ರೀ",
    "ಫೋರ್",
    "ಫೈವ್",
    "ಸಿಕ್ಸ್",
    "ಸೆವೆನ್",
    "ಎಯ್ಟ್",
    "ನೈನ್",
]
# English digit words in Latin script, and "double"/"triple" before a digit ("ಡಬಲ್ ಒಂಬತ್ತು" is
# two nines), so a number said that way still counts all its digits.
ENGLISH_DIGIT_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]
REPEATS = {
    "ಡಬಲ್": 2,
    "ಟ್ರಿಪಲ್": 3,
    "double": 2,
    "triple": 3,
}
_KANNADA = r"\u0c80-\u0cff"  # the Kannada block
_LATIN = "A-Za-z"


def _whole_words(words: list[str], script: str) -> str:
    """Any of `words`, not inside a longer word of the same script (ಒಂದು, not ಒಂದುಕಡೆ)."""
    alternatives = "|".join(sorted(words, key=len, reverse=True))
    return rf"(?<![{script}])(?:{alternatives})(?![{script}])"


_DIGIT = (
    rf"(?:\d|{_whole_words(DIGIT_WORDS, _KANNADA)}"
    rf"|(?i:{_whole_words(ENGLISH_DIGIT_WORDS, _LATIN)}))"
)
_REPEAT = (
    rf"(?:{_whole_words(['ಡಬಲ್', 'ಟ್ರಿಪಲ್'], _KANNADA)}"
    rf"|(?i:{_whole_words(['double', 'triple'], _LATIN)}))"
)
_GAP = r"[\s,.\-/()]{0,3}"
# One spoken digit, or "double"/"triple" and a digit; a worded number is a run of them.
_UNIT = rf"(?:{_REPEAT}{_GAP})?{_DIGIT}"
WORDED_RUN = re.compile(rf"{_UNIT}(?:{_GAP}{_UNIT})*")
UNIT = re.compile(_UNIT)
REPEAT_WORD = re.compile(_REPEAT)
LONG_DIGITS, ID_LIKE_DIGITS = 10, 4


def _digits_in(run: str) -> int:
    """How many digits a worded run says: "double" and "triple" count their digit twice or
    three times."""
    extra = sum(REPEATS[word.lower()] - 1 for word in REPEAT_WORD.findall(run))
    return len(UNIT.findall(run)) + extra


def _worded_numbers(text: str, digits: int) -> list[tuple[int, int]]:
    """Where `text` says a number of at least `digits` digits, in digits or digit words."""
    return [m.span() for m in WORDED_RUN.finditer(text) if _digits_in(m.group()) >= digits]


def says_long_number(text: str) -> bool:
    """A phone, Aadhaar or account number, in digits or in digit words."""
    return bool(LONG_NUMBER.search(text) or _worded_numbers(text, LONG_DIGITS))


def mask_personal_ids(text: str, personal_id: float) -> str:
    """`text` with every digit of a personal number, or every letter of a number said in
    digit words, replaced by X.

    `personal_id` is the decider's probability that the caller said one
    (`contains_personal_id`). Long digit runs are masked whatever it says. Every digit
    becomes exactly one X, so the masked text has the same length as the original.
    """
    digits = ID_LIKE_DIGITS if personal_id > PERSONAL_ID else LONG_DIGITS
    # Every letter and digit of a worded number becomes X; the gaps between them stay. Spans
    # are found first and masking keeps lengths, so later spans stay in place.
    for start, end in _worded_numbers(text, digits):
        text = text[:start] + re.sub(r"[^\s,.\-/()]", MASK, text[start:end]) + text[end:]
    pattern = ID_LIKE_NUMBER if personal_id > PERSONAL_ID else LONG_NUMBER
    return pattern.sub(lambda match: re.sub(r"\d", MASK, match.group()), text)


def mask_turns(turns: list[str], personal_id: float) -> list[str]:
    """Mask a call's turns together, so a number spread over two turns ("98450" then
    "12345") is caught as one. Masking keeps lengths, so the result splits back into the
    same turns."""
    joined = mask_personal_ids(" ".join(turns), personal_id)
    masked, start = [], 0
    for turn in turns:
        masked.append(joined[start : start + len(turn)])
        start += len(turn) + 1
    return masked
