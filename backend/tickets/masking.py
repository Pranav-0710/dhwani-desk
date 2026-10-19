"""Personal numbers never reach storage (docs/PRD.md, "Masking before storage")."""

import re

# A phone, Aadhaar or account number: 10+ digits, which callers say in groups
# ("98450 12345", "1234 5678 9012").
LONG_NUMBER = re.compile(r"\d(?:[ -]?\d){9,}")
# Once the decider thinks the caller gave a personal ID, shorter runs go too (a partial
# account number), but not 1–3 digit numbers: ward 4, 3 days, a count of lights.
ID_LIKE_NUMBER = re.compile(r"\d(?:[ -]?\d){3,}")
PERSONAL_ID = 0.5
MASK = "X"


def mask_personal_ids(text: str, personal_id: float) -> str:
    """`text` with every digit of a personal number replaced by X.

    `personal_id` is the decider's probability that the caller said one
    (`contains_personal_id`). Long digit runs are masked whatever it says.
    """
    pattern = ID_LIKE_NUMBER if personal_id > PERSONAL_ID else LONG_NUMBER
    return pattern.sub(lambda match: re.sub(r"\d", MASK, match.group()), text)
