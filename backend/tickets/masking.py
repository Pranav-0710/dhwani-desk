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


def mask_personal_ids(text: str, personal_id: float) -> str:
    """`text` with every digit of a personal number replaced by X.

    `personal_id` is the decider's probability that the caller said one
    (`contains_personal_id`). Long digit runs are masked whatever it says. Every digit
    becomes exactly one X, so the masked text has the same length as the original.
    """
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
