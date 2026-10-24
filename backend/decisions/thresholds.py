"""When a decision is acted on: the numbers behind every escalation rule in docs/PRD.md
("Escalation and safety rules", v1). They are tuned on the test set later.

A shared contract: the state machine acts on them, the console draws them on the decision bars
(`GET /api/thresholds`) and the eval harness scores against them. Every comparison is strict, so
an answer exactly at a threshold is not acted on.
"""

from typing import Any

# Yes-probabilities above which a rule fires.
SAFETY_HAZARD = 0.6  # hand off with the safety line (so does urgency "emergency")
WANTS_HUMAN = 0.7  # hand off to an officer
OUT_OF_SCOPE = 0.7  # redirect and close, no ticket
HAS_LOCATION = 0.5  # the caller gave a place: ask for the ward rather than the location
REPEAT_COMPLAINT = 0.5  # flag the ticket as a repeat and say so
# The department is remembered once its probability reaches this.
DEPARTMENT_CONFIDENCE = 0.6
# Caller states handed off with the calm hand-off line.
HAND_OFF_STATES = ("distressed", "abusive")
# Turns without a confident department before handing off (PRD: "after 2 turns").
MAX_UNROUTED_TURNS = 2
# Times to ask for a location before handing off: a ticket nobody can find helps no one.
MAX_LOCATION_ASKS = 2


def as_dict() -> dict[str, Any]:
    return {
        "safety_hazard": SAFETY_HAZARD,
        "emergency_urgency": "emergency",
        "wants_human": WANTS_HUMAN,
        "out_of_scope": OUT_OF_SCOPE,
        "has_location": HAS_LOCATION,
        "repeat_complaint": REPEAT_COMPLAINT,
        "department_confidence": DEPARTMENT_CONFIDENCE,
        "hand_off_caller_states": list(HAND_OFF_STATES),
        "max_unrouted_turns": MAX_UNROUTED_TURNS,
        "max_location_asks": MAX_LOCATION_ASKS,
    }
