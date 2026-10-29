"""Every test gets its own in-memory ticket store, so no test writes a database file, and the
keyword decider, so a developer's DECIDER=laya never loads a model in a test."""

import pytest

import app as app_module
from dialogue.standin import KeywordDecider
from tickets.store import TicketStore

KEYWORDS = KeywordDecider()


@pytest.fixture(autouse=True)
def keyword_decider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "get_decider", lambda: KEYWORDS)


@pytest.fixture(autouse=True)
def ticket_store(monkeypatch: pytest.MonkeyPatch) -> TicketStore:
    store = TicketStore(":memory:")
    monkeypatch.setattr(app_module, "get_ticket_store", lambda: store)
    return store
