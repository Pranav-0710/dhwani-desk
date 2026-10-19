"""A keyword stand-in for Laya, so the call works end to end before the fine-tuned checkpoint.

#15 replaces it with Laya (plus code for ward and personal-ID detection, per
docs/decisions/laya-kannada-gate.md). The keyword lists come from the week-1 benchmark's rules,
which were written against the seed set: good enough to demo the call flow, not a measure of
anything. A keyword hit answers with high confidence, no hit with low confidence, so the state
machine asks or hands off exactly as it would for an unsure Laya.
"""

import json
import re

from config import REPO_ROOT
from dialogue.decision import Decision

HIT, MISS = 0.9, 0.1

DEPARTMENT_WORDS = {
    "water": ["ನೀರು ಬಂದಿಲ್ಲ", "ನೀರು ಬರ್ತಿಲ್ಲ", "ಕುಡಿಯೋ ನೀರ", "ಕುಡಿಯುವ ನೀರ", "ಪೈಪ್", "ಟ್ಯಾಂಕರ್", "ನಲ್ಲಿ"],
    "sewage-drainage": ["ಚರಂಡಿ", "ಮ್ಯಾನ್ ಹೋಲ್", "ಡ್ರೈನೇಜ್", "ಯುಜಿಡಿ", "ಕೊಳಚೆ"],
    "solid-waste": ["ಕಸ", "ಡಸ್ಟ್ ಬಿನ್", "ಡೆಬ್ರಿಸ್", "ಸತ್ತ"],
    "roads": ["ಗುಂಡಿ", "ರೋಡ್ ರಿಪೇರಿ", "ಫುಟ್ ಪಾತ್", "ಸ್ಪೀಡ್ ಬ್ರೇಕರ್", "ರಸ್ತೆ ಹಾಳ", "ರೋಡ್ ಹಾಳ"],
    "streetlights": ["ಲೈಟ್", "ಬೀದಿ ದೀಪ", "ಕಂಬ"],
    "stray-animals": ["ನಾಯಿ", "ದನ", "ಮಂಗ", "ಹಾವು"],
}
OUT_OF_SCOPE_WORDS = ["ಜಗಳ", "ಹೊಡೆದಾಡ", "ಎದೆ ನೋವು", "ಆಂಬುಲೆನ್ಸ್", "ಕರೆಂಟ್ ಬಿಲ್", "ರೇಷನ್", "ಕಳ್ಳತನ"]
HAZARD_WORDS = [
    "ಮ್ಯಾನ್ ಹೋಲ್",
    "ಸ್ಪಾರ್ಕ್",
    "ವೈರ್",
    "ಕಚ್ಚಿದೆ",
    "ಹಾವು",
    "ಬೆಂಕಿ",
    "ಮನೆ ಒಳಗೆ",
    "ಬೀಳೋ",
    "ಎದೆ ನೋವು",
    "ಆಂಬುಲೆನ್ಸ್",
    "ಹೊಡೆದಾಡ",
]
EMERGENCY_WORDS = ["ಕಚ್ಚಿದೆ", "ಸ್ಪಾರ್ಕ್", "ಮುಚ್ಚಳ ಇಲ್ಲ"]
URGENT_WORDS = ["ಬೇಗ", "ಒಡೆದು", "ಉಕ್ಕಿ"]
REPEAT_WORDS = ["ಕಂಪ್ಲೈಂಟ್ ಕೊಟ್ಟಿದ್ದೆ", "ಮೊದಲೇ", "ಇನ್ನೂ", "ಸಲ"]
HUMAN_WORDS = ["ಆಫೀಸರ್", "ಅಧಿಕಾರಿ"]
ABUSIVE_WORDS = ["ಸೋಮಾರಿ", "ಕೆಲಸಕ್ಕೆ ಬಾರದ"]
DISTRESS_WORDS = ["ಬೇಗ ಬನ್ನಿ", "ಪ್ಲೀಸ್ ಬೇಗ", "ಭಯ"]
FRUSTRATION_WORDS = ["ಇನ್ನೂ", "ಯಾರೂ", "ತಿಂಗಳಾಯ್ತು", "ಎಷ್ಟು ಸಲ"]
PLACE_WORDS = ["ಹತ್ತಿರ", "ಮುಂದೆ", "ಹಿಂದೆ", "ಎದುರು", "ಪಕ್ಕ", "ಕ್ರಾಸ್", "ಬೀದಿಯಲ್ಲಿ"]
ENGLISH_NUMBERS = ["ಒನ್", "ಟೂ", "ತ್ರೀ", "ಫೋರ್", "ಫೈವ್", "ಸಿಕ್ಸ್", "ಸೆವೆನ್", "ಎಯ್ಟ್", "ನೈನ್", "ಟೆನ್"]
WARD_NUMBER = re.compile(r"ವಾರ್ಡ್\s*(?:ನಂಬರ್\s*)?(\d+|" + "|".join(ENGLISH_NUMBERS) + ")")


def _has(text: str, words: list[str]) -> bool:
    return any(word in text for word in words)


class KeywordDecider:
    revision = "keyword-standin-1"

    def __init__(self) -> None:
        wards = json.loads((REPO_ROOT / "data" / "wards.json").read_text("utf-8"))["wards"]
        self._ward_names = {
            ward["id"]: [ward["name"], *(landmark["name"] for landmark in ward["landmarks"])]
            for ward in wards
        }
        self._ward_by_number = {str(ward["number"]): ward["id"] for ward in wards}

    def decide(self, transcript: str) -> Decision:
        text = transcript
        ward = self._ward(text)
        out_of_scope = _has(text, OUT_OF_SCOPE_WORDS)
        department = next(
            (name for name, words in DEPARTMENT_WORDS.items() if _has(text, words)), None
        )
        if out_of_scope:
            department = "other"
        urgency = (
            "emergency"
            if _has(text, EMERGENCY_WORDS)
            else "urgent"
            if _has(text, URGENT_WORDS)
            else "soon"
        )
        caller_state = (
            "abusive"
            if _has(text, ABUSIVE_WORDS)
            else "distressed"
            if _has(text, DISTRESS_WORDS)
            else "frustrated"
            if _has(text, FRUSTRATION_WORDS)
            else "calm"
        )
        digits = re.sub(r"\D", "", text)
        return Decision(
            department=department or "other",
            department_confidence=HIT if department else MISS,
            ward=ward,
            urgency=urgency,
            safety_hazard=HIT if _has(text, HAZARD_WORDS) else MISS,
            has_location=HIT if ward != "unknown" or _has(text, PLACE_WORDS) else MISS,
            repeat_complaint=HIT if _has(text, REPEAT_WORDS) else MISS,
            wants_human=HIT if _has(text, HUMAN_WORDS) else MISS,
            caller_state=caller_state,
            out_of_scope=HIT if out_of_scope else MISS,
            contains_personal_id=HIT if len(digits) >= 10 else MISS,
        )

    def _ward(self, text: str) -> str:
        for ward_id, names in self._ward_names.items():
            if any(name in text for name in names):
                return ward_id
        if match := WARD_NUMBER.search(text):
            value = match.group(1)
            number = str(ENGLISH_NUMBERS.index(value) + 1) if value in ENGLISH_NUMBERS else value
            return self._ward_by_number.get(number, "unknown")
        return "unknown"
