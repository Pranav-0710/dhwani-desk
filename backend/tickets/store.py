"""Tickets and caller transcripts in SQLite.

Two tables, kept apart on purpose: `tickets` holds only what a field team needs and stays,
while `transcripts` holds what callers said and is deleted after the retention period. Both
are masked before they are written (tickets/masking.py), so no personal number is ever stored.
A shared contract: the console (#18) and the eval harness (#19) read these fields.
"""

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dialogue.decision import Decision
from tickets.masking import mask_personal_ids

# Ticket numbers start here, so every number the agent reads back has four digits.
FIRST_NUMBER = 1001

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    call_id TEXT NOT NULL,
    department TEXT NOT NULL,
    ward TEXT NOT NULL,
    urgency TEXT NOT NULL,
    repeat_complaint INTEGER NOT NULL,
    location_text TEXT NOT NULL,
    decision TEXT NOT NULL,
    decider_revision TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open'
);
CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    call_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS transcripts_by_age ON transcripts (created_at);
CREATE INDEX IF NOT EXISTS transcripts_by_call ON transcripts (call_id, turn);
"""


@dataclass(frozen=True)
class TicketRequest:
    """What the dialogue files when it closes a call with a ticket."""

    call_id: str
    department: str
    ward: str  # a ward id, or "unknown"
    urgency: str
    repeat_complaint: bool
    location_text: str  # the caller's own words for where the problem is, before masking
    decision: Decision  # every answer behind the ticket, with its probability
    decider_revision: str  # which decider (and checkpoint) made the decision


def now() -> datetime:
    return datetime.now(UTC)


class TicketStore:
    """One SQLite connection shared by every call; writes are short and serialised."""

    def __init__(self, path: str | Path) -> None:
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    def file(self, request: TicketRequest) -> str:
        """Store a ticket and return its number."""
        personal_id = request.decision.contains_personal_id
        decision = asdict(request.decision)
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO tickets (created_at, call_id, department, ward, urgency,"
                " repeat_complaint, location_text, decision, decider_revision)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    now().isoformat(timespec="seconds"),
                    request.call_id,
                    request.department,
                    request.ward,
                    request.urgency,
                    int(request.repeat_complaint),
                    mask_personal_ids(request.location_text, personal_id),
                    json.dumps(decision, ensure_ascii=False),
                    request.decider_revision,
                ),
            )
        return str(cursor.lastrowid + FIRST_NUMBER - 1)

    def record_turn(self, call_id: str, turn: int, text: str, personal_id: float) -> None:
        """Store one caller turn, masked. `personal_id` is the decider's probability that
        the caller said a personal number; pass 1.0 when no decision is available."""
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO transcripts (created_at, call_id, turn, text) VALUES (?, ?, ?, ?)",
                (
                    now().isoformat(timespec="seconds"),
                    call_id,
                    turn,
                    mask_personal_ids(text, personal_id),
                ),
            )

    def tickets(self, limit: int = 50) -> list[dict[str, Any]]:
        """The newest tickets first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM tickets ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_ticket(row) for row in rows]

    def ticket(self, number: str) -> dict[str, Any] | None:
        if not number.isdigit():
            return None
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM tickets WHERE id = ?", (int(number) - FIRST_NUMBER + 1,)
            ).fetchone()
        return _ticket(row) if row else None

    def transcript(self, call_id: str) -> list[str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT text FROM transcripts WHERE call_id = ? ORDER BY turn", (call_id,)
            ).fetchall()
        return [row["text"] for row in rows]

    def purge_transcripts(self, older_than_days: int, at: datetime | None = None) -> int:
        """Delete transcripts older than the retention period; returns how many went."""
        cutoff = ((at or now()) - timedelta(days=older_than_days)).isoformat(timespec="seconds")
        with self._lock, self._db:
            cursor = self._db.execute("DELETE FROM transcripts WHERE created_at < ?", (cutoff,))
        return cursor.rowcount


def _ticket(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "number": str(row["id"] + FIRST_NUMBER - 1),
        "created_at": row["created_at"],
        "call_id": row["call_id"],
        "department": row["department"],
        "ward": row["ward"],
        "urgency": row["urgency"],
        "repeat_complaint": bool(row["repeat_complaint"]),
        "location_text": row["location_text"],
        "status": row["status"],
        "decision": json.loads(row["decision"]),
        "decider_revision": row["decider_revision"],
    }
