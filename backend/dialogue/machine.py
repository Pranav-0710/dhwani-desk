"""The call as a deterministic state machine: no language model chooses what the agent says.

greet → listen → decide on everything the caller has said so far → one of:
- hand off to an officer (safety risk, emergency, caller asks for a person, caller distressed
  or abusive, the complaint still cannot be routed, or the decider failed)
- redirect (not a civic complaint) and close, with no ticket
- ask for what is missing (the problem, the location, the ward) and listen again
- read back the ticket number and close

Escalation comes first, in the order of the PRD's table: a safety risk wins over everything,
including "not a civic complaint", because a wrong redirect could leave someone hurt. The two
escalations that need nothing but the decider's first step (danger or an emergency, then a
request for an officer) happen on that step alone (`screened`), before the rest of the decision
is made. The thresholds are in decisions/thresholds.py.

A handed-off call waits for an officer, who can take it over (`take_over`), have the agent say
what they type (`officer_says`), and hand it back to the agent (`hand_back`). From then on only
what the caller says after the hand-back can hand the call off again (danger, an emergency or a
request for an officer: `escalation_text`); the officer has dealt with everything before it,
and the decider, rereading the whole call, would otherwise hand it straight back. Routing and
the ticket still come from the whole call.
"""

import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from itertools import count
from typing import Any, Protocol
from uuid import uuid4

from decisions.thresholds import (
    DEPARTMENT_CONFIDENCE,
    HAND_OFF_STATES,
    HAS_LOCATION,
    MAX_LOCATION_ASKS,
    MAX_UNROUTED_TURNS,
    OUT_OF_SCOPE,
    REPEAT_COMPLAINT,
    SAFETY_HAZARD,
    WANTS_HUMAN,
)
from dialogue.decision import Decider, Decision, Screening
from dialogue.templates import Templates
from tickets.store import TicketRequest

logger = logging.getLogger("dhwani.dialogue")

# Longest line an officer may have the agent say, in characters.
MAX_OFFICER_LINE = 300


class State(StrEnum):
    LISTENING = "listening"
    HANDED_OFF = "handed_off"  # for an officer; the agent says only what the officer types
    CLOSED = "closed"


class OfficerError(ValueError):
    """An officer action the call cannot take now (it is over, or not handed to them)."""


@dataclass
class Slots:
    department: str | None = None
    ward: str | None = None  # a ward id
    location_given: bool = False  # a ward, street or landmark was mentioned


@dataclass(frozen=True)
class Ticket:
    number: str
    department: str
    ward: str  # a ward id, or "unknown"
    urgency: str
    repeat_complaint: bool


@dataclass(frozen=True)
class Reply:
    """What the agent does after one caller turn."""

    keys: tuple[str, ...]  # template keys, in speaking order
    lines: tuple[str, ...]  # the same lines as Kannada text
    state: State
    reason: str | None = None  # why the call was handed off or closed
    ticket: Ticket | None = None
    decision: Decision | None = None
    # Set instead of `decision` when the call was handed off on the decider's first step.
    screening: Screening | None = None

    def to_client(self, slots: Slots, officer: bool = False) -> dict[str, Any]:
        """The `dialogue` event the browser and console receive."""
        return {
            "type": "dialogue",
            "keys": list(self.keys),
            "state": self.state.value,
            "reason": self.reason,
            "officer": officer,  # an officer has taken the call
            "slots": asdict(slots),
            "ticket": asdict(self.ticket) if self.ticket else None,
            "decision": asdict(self.decision) if self.decision else None,
            "screening": asdict(self.screening) if self.screening else None,
        }


class TicketDesk(Protocol):
    """Files a ticket and returns its number (tickets.store.TicketStore in the app)."""

    def file(self, request: TicketRequest) -> str: ...


class CountingDesk:
    """Numbers tickets from a counter and keeps them in memory; for tests and scripts."""

    def __init__(self, start: int = 1001) -> None:
        self._numbers = count(start)
        self.filed: list[TicketRequest] = []

    def file(self, request: TicketRequest) -> str:
        self.filed.append(request)
        return str(next(self._numbers))


@dataclass
class Dialogue:
    templates: Templates
    decider: Decider
    desk: TicketDesk
    call_id: str = field(default_factory=lambda: uuid4().hex)
    state: State = State.LISTENING
    reason: str | None = None  # why the call was handed off or closed
    officer: bool = False  # an officer has taken the call
    slots: Slots = field(default_factory=Slots)
    # Every caller turn, including those said while an officer had the call.
    turns: list[str] = field(default_factory=list)
    # The caller turns that gave the location, verbatim: what the field team reads.
    location_turns: list[str] = field(default_factory=list)
    _unrouted_turns: int = 0
    _location_asks: int = 0
    _ward_asked: bool = False
    _apologised: bool = False
    # How many turns had been said when an officer last handed the call back.
    _handed_back_at: int | None = None

    def greet(self) -> Reply:
        return self._reply(("greeting",))

    @property
    def transcript(self) -> str:
        """Everything the caller has said, as the decider reads it."""
        return " ".join(self.turns)

    def escalation_text(self) -> str | None:
        """What can hand the call off: the whole call, or after a hand-back only what the
        caller has said since (None while that is nothing)."""
        if self._handed_back_at is None:
            return self.transcript
        return " ".join(self.turns[self._handed_back_at :]) or None

    def take_over(self) -> Reply:
        """An officer takes the call, handed off or not: the agent now says only what the
        officer types. Nothing is said; the reply carries the new state."""
        if self.state is State.CLOSED:
            raise OfficerError("The call is over")
        if self.state is State.LISTENING:
            self.state, self.reason = State.HANDED_OFF, "taken_over"
        self.officer = True
        return self._reply((), reason=self.reason)

    def officer_says(self, text: str) -> str:
        """The line an officer typed, checked, for the agent to say."""
        if self.state is not State.HANDED_OFF or not self.officer:
            raise OfficerError("Take the call over before speaking on it")
        line = " ".join(text.split())
        if not line:
            raise OfficerError("Nothing to say")
        if len(line) > MAX_OFFICER_LINE:
            raise OfficerError(f"Keep a line under {MAX_OFFICER_LINE} characters")
        return line

    def hand_back(self) -> Reply:
        """The officer returns the call to the agent, which says it is carrying on. Decide on
        the whole call next (`rethink`, or the call route's own steps): what the caller told
        the officer counts too."""
        if self.state is not State.HANDED_OFF:
            raise OfficerError("Only a handed-off call can be handed back")
        self.state, self.reason, self.officer = State.LISTENING, None, False
        self._handed_back_at = len(self.turns)
        # Fresh tries at what the agent could not get before the officer helped.
        self._unrouted_turns = self._location_asks = 0
        self._ward_asked = False
        return self._reply(("hand_back",))

    def rethink(self) -> Reply:
        """Decide on the whole call so far, in one go, as `turn` does for a new turn."""
        return self._answer() if self.turns else self._reply(())

    def turn(self, text: str) -> Reply:
        """Hear and answer one caller turn in one go. The call route takes the same steps one
        at a time (`hear`, `escalation_text`, `screened`, `respond`), so the decider runs off
        the event loop and the caller hears something while the slow step runs."""
        if self.hear(text) is None:
            return self._reply(())
        return self._answer()

    def _answer(self) -> Reply:
        handed_off = None
        try:
            escalation_text = self.escalation_text()
            first = self.decider.screen(escalation_text) if escalation_text else None
            handed_off = self.screened(first) if first else None
            screening = (
                first
                if escalation_text == self.transcript
                else self.decider.screen(self.transcript)
            )
            decision = self.decider.decide(self.transcript, screening)
        except Exception:
            logger.exception("The decider failed; handing the call to an officer")
            return handed_off or self.decider_failed()
        if handed_off is not None:
            self.complete(decision)
            return handed_off
        return self.respond(decision)

    def hear(self, text: str) -> str | None:
        """Add a caller turn. Returns everything the caller has said, for the decider, or None
        when there is nothing to decide: a blank turn, a closed call, or a turn said while the
        call is handed off (kept, so the agent knows it after a hand-back)."""
        if self.state is State.CLOSED or not text.strip():
            return None
        self.turns.append(text.strip())
        return self.transcript if self.state is State.LISTENING else None

    def decider_failed(self) -> Reply:
        """Hand off rather than go quiet: the decider reads the whole call, so if it failed on
        this transcript it would most likely fail on every longer one too."""
        if self.state is not State.LISTENING:
            return self._reply(())  # already handed off on the safety step, or over
        return self._end(State.HANDED_OFF, ("handoff_unclear",), "decider_error", None)

    def screened(self, screening: Screening) -> Reply | None:
        """The hand-off the decider's first step already calls for, or None when the call needs
        the full decision (`respond`). The same rules as `respond`, in the same order, so
        deciding in one step or two never changes what the caller hears."""
        if self.state is not State.LISTENING:
            return self._reply(())
        if (rule := self._first_step_rule(screening)) is not None:
            key, reason = rule
            return self._end(State.HANDED_OFF, (key,), reason, screening=screening)
        return None

    def _first_step_rule(self, answers: Screening | Decision) -> tuple[str, str] | None:
        """(line, reason) of the first escalation that the safety step's answers call for."""
        if answers.safety_hazard > SAFETY_HAZARD or answers.urgency == "emergency":
            return _safety_line(answers.out_of_scope), "safety"
        if answers.wants_human > WANTS_HUMAN:
            return "handoff_human", "wants_human"
        return None

    def complete(self, decision: Decision) -> Reply:
        """The full decision after a hand-off on the safety step, so the officer who takes over
        sees what is already known. Nothing is said: the reply has no lines."""
        if self.state is not State.HANDED_OFF:
            return self._reply(())
        self._remember(decision)
        return self._reply((), reason=self.reason, decision=decision)

    def respond(self, decision: Decision) -> Reply:
        """The agent's answer to a decision on everything the caller has said so far."""
        if self.state is not State.LISTENING:
            return self._reply(())
        # Before any hand-off, so the officer who takes over sees what is already known.
        self._remember(decision)

        # After a hand-back, the full decision reads words the officer has already dealt with:
        # only the first step, on what was said since, can hand the call off (`screened`).
        if self._handed_back_at is None:
            if (rule := self._first_step_rule(decision)) is not None:
                key, reason = rule
                return self._end(State.HANDED_OFF, (key,), reason, decision)
            if decision.caller_state in HAND_OFF_STATES:
                return self._end(
                    State.HANDED_OFF, ("handoff_calm",), decision.caller_state, decision
                )
            if decision.out_of_scope > OUT_OF_SCOPE:
                return self._end(State.CLOSED, ("redirect",), "out_of_scope", decision)

        opening = self._opening(decision)
        if self.slots.department is None:
            self._unrouted_turns += 1
            if self._unrouted_turns >= MAX_UNROUTED_TURNS:
                return self._end(State.HANDED_OFF, ("handoff_unclear",), "unroutable", decision)
            return self._reply((*opening, "ask_problem"), decision=decision)
        if self.slots.ward is None:
            if self.slots.location_given and not self._ward_asked:
                self._ward_asked = True
                return self._reply((*opening, "ask_ward"), decision=decision)
            if not self.slots.location_given:
                if self._location_asks >= MAX_LOCATION_ASKS:
                    return self._end(
                        State.HANDED_OFF, ("handoff_unclear",), "no_location", decision
                    )
                self._location_asks += 1
                return self._reply((*opening, "ask_location"), decision=decision)
        return self._close_with_ticket(opening, decision)

    def _remember(self, decision: Decision) -> None:
        # The decider reads the whole call each turn, so its latest confident answer is its
        # best one; slots are only ever filled, never emptied.
        if decision.department_confidence >= DEPARTMENT_CONFIDENCE:
            self.slots.department = decision.department
        # A turn that first gives the location, or first names the ward, is kept verbatim.
        new_place = not self.slots.location_given and (
            decision.ward != "unknown" or decision.has_location > HAS_LOCATION
        )
        new_ward = self.slots.ward is None and decision.ward != "unknown"
        if (new_place or new_ward) and self.turns:
            self.location_turns.append(self.turns[-1])
        if decision.ward != "unknown":
            self.slots.ward = decision.ward
        if decision.ward != "unknown" or decision.has_location > HAS_LOCATION:
            self.slots.location_given = True

    def _opening(self, decision: Decision) -> tuple[str, ...]:
        """An apology the first time a frustrated caller is answered."""
        if decision.caller_state == "frustrated" and not self._apologised:
            self._apologised = True
            return ("apology",)
        return ()

    def _close_with_ticket(self, opening: tuple[str, ...], decision: Decision) -> Reply:
        department = self.slots.department
        assert department is not None
        repeat = decision.repeat_complaint > REPEAT_COMPLAINT
        request = TicketRequest(
            call_id=self.call_id,
            department=department,
            ward=self.slots.ward or "unknown",
            urgency=decision.urgency,
            repeat_complaint=repeat,
            location_text=" ".join(self.location_turns),
            decision=decision,
            decider_revision=self.decider.revision,
        )
        ticket = Ticket(
            number=self.desk.file(request),
            department=request.department,
            ward=request.ward,
            urgency=request.urgency,
            repeat_complaint=repeat,
        )
        keys = (
            *opening,
            *(("repeat_noted",) if repeat else ()),
            "read_back_ward" if self.slots.ward else "read_back",
            "closing",
        )
        self.state = State.CLOSED
        return self._reply(keys, reason="ticket", ticket=ticket, decision=decision)

    def _end(
        self,
        state: State,
        keys: tuple[str, ...],
        reason: str,
        decision: Decision | None = None,
        screening: Screening | None = None,
    ) -> Reply:
        self.state, self.reason = state, reason
        return self._reply(keys, reason=reason, decision=decision, screening=screening)

    def _reply(
        self,
        keys: tuple[str, ...],
        reason: str | None = None,
        ticket: Ticket | None = None,
        decision: Decision | None = None,
        screening: Screening | None = None,
    ) -> Reply:
        values = {}
        if ticket:
            values = {
                "department": self.templates.departments[ticket.department],
                "ward": self.templates.wards.get(ticket.ward, ""),
                # Digit by digit, so it is read as "1 0 0 4", easy to write down.
                "ticket": " ".join(ticket.number),
            }
        lines = tuple(self.templates.line(key, **values) for key in keys)
        return Reply(keys, lines, self.state, reason, ticket, decision, screening)


def _safety_line(out_of_scope: float) -> str:
    # A risk outside the corporation's remit (chest pain, a fight) still gets an officer, and
    # the caller is told to call 112 at once.
    return "safety_redirect" if out_of_scope > OUT_OF_SCOPE else "safety"
