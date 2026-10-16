"""What the dialogue needs to know about the call so far, whoever decides it.

#15 fills a `Decision` from Laya's answers (plus the ward and personal-ID extractors); until then
`dialogue.standin.KeywordDecider` fills it from keyword rules. The dialogue never sees how a
decision was made, so swapping the decider changes nothing here.
"""

from dataclasses import dataclass
from typing import Protocol

URGENCY_LEVELS = ("routine", "soon", "urgent", "emergency")
CALLER_STATES = ("calm", "frustrated", "distressed", "abusive")


@dataclass(frozen=True)
class Decision:
    """One answer per question in `data/decision_schema.json`.

    Yes/no questions are the probability of "yes". Choice questions are the chosen option;
    `department` also carries its probability, because routing waits until it is confident.
    """

    department: str
    department_confidence: float
    ward: str  # a ward id from data/wards.json, or "unknown"
    urgency: str  # one of URGENCY_LEVELS
    safety_hazard: float
    has_location: float
    repeat_complaint: float
    wants_human: float
    caller_state: str  # one of CALLER_STATES
    out_of_scope: float
    contains_personal_id: float


class Decider(Protocol):
    """Decides on the caller's words so far: every caller turn, joined in order.

    One decider serves every call in the process, and `decide` runs in worker threads, so it
    must be safe to call from several threads at once. A decision abandoned when its call ends
    still runs to completion, so keep it bounded (#15: cap the transcript length for Laya).
    """

    def decide(self, transcript: str) -> Decision: ...
