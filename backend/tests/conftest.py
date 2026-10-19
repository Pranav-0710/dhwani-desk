"""Every test gets its own in-memory ticket store, so no test writes a database file."""

import pytest

import app as app_module
from tickets.store import TicketStore


@pytest.fixture(autouse=True)
def ticket_store(monkeypatch: pytest.MonkeyPatch) -> TicketStore:
    store = TicketStore(":memory:")
    monkeypatch.setattr(app_module, "get_ticket_store", lambda: store)
    return store
