"""What the dialogue needs to know about the call so far, whoever decides it.

`decisions.laya.LayaDecider` fills a `Decision` from Laya's answers plus code for the ward and
personal numbers; `dialogue.standin.KeywordDecider` fills it from keyword rules. The dialogue
never sees how a decision was made, so swapping the decider changes nothing here.

A decision comes in two steps, safety first: `screen` answers the questions that can end a call
at once (danger, an emergency, a request for an officer, not a civic matter), then `decide`
answers the rest. Laya's first step is the fast one (docs/decisions/laya-latency.md), so a
caller in danger is handed off before the slow step even starts.
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


@dataclass(frozen=True)
class Screening:
    """The answers that can end a call before anything else is decided (yes probabilities)."""

    safety_hazard: float
    urgency: str  # one of URGENCY_LEVELS: "emergency" is a safety risk too
    wants_human: float
    out_of_scope: float


class Decider(Protocol):
    """Decides on the caller's words so far: every caller turn, joined in order.

    One decider serves every call in the process, and both steps run in worker threads, so it
    must be safe to call from several threads at once. A decision abandoned when its call ends
    still runs to completion, so keep it bounded (LayaDecider caps the transcript length).
    """

    # Which decider made a decision, stored with every ticket: a rule-set version, or a
    # checkpoint repo and revision for Laya.
    revision: str

    def screen(self, transcript: str) -> Screening: ...

    def decide(self, transcript: str, screening: Screening) -> Decision:
        """The full decision, with `screening`'s answers in it unchanged."""
        ...
