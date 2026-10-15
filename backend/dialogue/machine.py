"""The call as a deterministic state machine: no language model chooses what the agent says.

greet → listen → decide on everything the caller has said so far → one of:
- hand off to an officer (safety risk, emergency, caller asks for a person, caller distressed
  or abusive, the complaint still cannot be routed, or the decider failed)
- redirect (not a civic complaint) and close, with no ticket
- ask for what is missing (the problem, the location, the ward) and listen again
- read back the ticket number and close

Escalation comes first, in the order of the PRD's table: a safety risk wins over everything,
including "not a civic complaint", because a wrong redirect could leave someone hurt.
"""

import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from itertools import count
from typing import Any, Protocol

from dialogue.decision import Decider, Decision
from dialogue.templates import Templates

logger = logging.getLogger("dhwani.dialogue")

# Thresholds from docs/PRD.md ("Escalation and safety rules", v1); they are tuned on the test
# set later. A shared contract: the console and the eval harness read the same numbers.
OUT_OF_SCOPE = 0.7
SAFETY_HAZARD = 0.6
WANTS_HUMAN = 0.7
DEPARTMENT_CONFIDENCE = 0.6
HAS_LOCATION = 0.5
REPEAT_COMPLAINT = 0.5
HAND_OFF_STATES = ("distressed", "abusive")
# Turns without a confident department before handing off (PRD: "after 2 turns").
MAX_UNROUTED_TURNS = 2
# Times to ask for a location before handing off: a ticket nobody can find helps no one.
MAX_LOCATION_ASKS = 2


class State(StrEnum):
    LISTENING = "listening"
    HANDED_OFF = "handed_off"  # an officer has the call; the agent stays quiet
    CLOSED = "closed"


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

    def to_client(self, slots: Slots) -> dict[str, Any]:
        """The `dialogue` event the browser and console receive."""
        return {
            "type": "dialogue",
            "keys": list(self.keys),
            "state": self.state.value,
            "reason": self.reason,
            "slots": asdict(slots),
            "ticket": asdict(self.ticket) if self.ticket else None,
            "decision": asdict(self.decision) if self.decision else None,
        }


class TicketDesk(Protocol):
    def next_number(self) -> str: ...


class CountingDesk:
    """Ticket numbers from a counter; #17 replaces it with the SQLite ticket store."""

    def __init__(self, start: int = 1001) -> None:
        self._numbers = count(start)

    def next_number(self) -> str:
        return str(next(self._numbers))


@dataclass
class Dialogue:
    templates: Templates
    decider: Decider
    desk: TicketDesk
    state: State = State.LISTENING
    slots: Slots = field(default_factory=Slots)
    turns: list[str] = field(default_factory=list)
    _unrouted_turns: int = 0
    _location_asks: int = 0
    _ward_asked: bool = False
    _apologised: bool = False

    def greet(self) -> Reply:
        return self._reply(("greeting",))

    def turn(self, text: str) -> Reply:
        """Hear and decide one caller turn in one step. The call route does the same in two
        steps (`hear`, then `respond`) so the decision can run off the event loop."""
        transcript = self.hear(text)
        if transcript is None:
            return self._reply(())
        try:
            decision = self.decider.decide(transcript)
        except Exception:
            logger.exception("The decider failed; handing the call to an officer")
            return self.decider_failed()
        return self.respond(decision)

    def hear(self, text: str) -> str | None:
        """Add a caller turn. Returns everything the caller has said, for the decider, or None
        when there is nothing to decide (blank turn, or the call is handed off or closed)."""
        if self.state is not State.LISTENING or not text.strip():
            return None
        self.turns.append(text.strip())
        return " ".join(self.turns)

    def decider_failed(self) -> Reply:
        """Hand off rather than go quiet: the decider reads the whole call, so if it failed on
        this transcript it would most likely fail on every longer one too."""
        return self._end(State.HANDED_OFF, ("handoff_unclear",), "decider_error", None)

    def respond(self, decision: Decision) -> Reply:
        """The agent's answer to a decision on everything the caller has said so far."""
        if self.state is not State.LISTENING:
            return self._reply(())
        # Before any hand-off, so the officer who takes over sees what is already known.
        self._remember(decision)

        if decision.safety_hazard > SAFETY_HAZARD or decision.urgency == "emergency":
            # A risk outside the corporation's remit (chest pain, a fight) still gets an
            # officer, and the caller is told to call 112 at once.
            key = "safety_redirect" if decision.out_of_scope > OUT_OF_SCOPE else "safety"
            return self._end(State.HANDED_OFF, (key,), "safety", decision)
        if decision.wants_human > WANTS_HUMAN:
            return self._end(State.HANDED_OFF, ("handoff_human",), "wants_human", decision)
        if decision.caller_state in HAND_OFF_STATES:
            return self._end(State.HANDED_OFF, ("handoff_calm",), decision.caller_state, decision)
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
        ticket = Ticket(
            number=self.desk.next_number(),
            department=department,
            ward=self.slots.ward or "unknown",
            urgency=decision.urgency,
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
        self, state: State, keys: tuple[str, ...], reason: str, decision: Decision | None
    ) -> Reply:
        self.state = state
        return self._reply(keys, reason=reason, decision=decision)

    def _reply(
        self,
        keys: tuple[str, ...],
        reason: str | None = None,
        ticket: Ticket | None = None,
        decision: Decision | None = None,
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
        return Reply(keys, lines, self.state, reason, ticket, decision)
