"""The someone-new card shows what the sender actually asked for.

A long request (a quote with its items, quantity, budget and deadline) used to
be cut at 280 characters, one line, so the card stopped at "a quote for…".
The card now carries the whole message with its paragraphs, and the email
poller hands it over with its line breaks.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from openexecutive.alerts import store as alerts_store
from openexecutive.integrations import email_poller, roster_intake
from openexecutive.memory import episodic
from openexecutive.people import registry as people_registry
from openexecutive.people import roster_requests as rr
from openexecutive.people import store as people_store

CHAIRS = (
    "Dear Sir or Madam,\n\n"
    "My name is Nadia Ferris, and I manage the office at Ferris & Hale Architects. We are refitting our"
    " studio and would like to request a quote for four Contour Mesh ergonomic desk chairs in charcoal grey.\n\n"
    "Could your quote include delivery to our office in Portland, and assembly if you offer it? Our budget"
    " is about $450 per chair.\n\n"
    "We need the chairs by Friday, December 4, so a reply by the end of next week would help us a great"
    " deal.\n\n"
    "Kind regards,\nNadia Ferris\nOffice Manager, Ferris & Hale Architects"
)


@pytest.fixture(autouse=True)
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "roster.db"
    monkeypatch.setattr(people_store, "DB_PATH", path)
    monkeypatch.setattr(episodic, "DB_PATH", path)
    monkeypatch.setattr(alerts_store, "DB_PATH", path)
    people_store.initialize_db()
    alerts_store.initialize_db()
    monkeypatch.setattr("openexecutive.audit.log_event", lambda *a, **kw: None)
    people_registry.invalidate()
    people_store.upsert_person(full_name="Olivia Owner", is_principal=True, email="olivia@acme.com")
    yield path
    people_registry.invalidate()


def test_a_long_request_shows_in_full_with_its_paragraphs() -> None:
    preview = "Request for a Quote: Ergonomic Desk Chairs\n\n" + CHAIRS
    out = rr.hold("email", "nadia@ferrishale.example", external_id="m1", payload={}, preview=preview)
    assert out is not None
    (shown,) = rr.previews(out.request.id)
    assert shown == preview
    for detail in ("four Contour Mesh", "charcoal grey", "delivery", "$450", "Friday, December 4"):
        assert detail in shown


def test_the_message_is_tidied_but_only_a_very_long_one_is_cut() -> None:
    messy = "Hello   there\r\n\r\n\r\n\r\nSecond\tline\x07 here\n"
    out = rr.hold("email", "a@b.example", external_id="m1", payload={}, preview=messy)
    assert out is not None
    assert rr.previews(out.request.id) == ["Hello there\n\nSecond line here"]

    long = "word " * 2000
    out = rr.hold("email", "c@d.example", external_id="m2", payload={}, preview=long)
    assert out is not None
    (shown,) = rr.previews(out.request.id)
    assert len(shown) == rr.PREVIEW_CHARS and shown.endswith("…")


async def test_the_email_poller_hands_over_the_subject_and_the_whole_body(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_intake(channel: str, ref: str, **kw: Any) -> None:
        seen.update(kw)
        return None

    monkeypatch.setattr(roster_intake, "intake", fake_intake)
    raw = (
        "Subject: Request for a Quote: Ergonomic Desk Chairs\n"
        "From: Nadia Ferris <nadia@ferrishale.example>\n"
        "\n--- BODY ---\n" + CHAIRS
    )
    await email_poller._hold_for_roster(None, raw, "Nadia Ferris", "nadia@ferrishale.example", "m1", "t1")
    preview = seen["preview"]
    assert preview.startswith("Request for a Quote: Ergonomic Desk Chairs\n\nDear Sir or Madam,")
    for detail in ("four Contour Mesh", "charcoal grey", "delivery", "$450", "Friday, December 4"):
        assert detail in preview
    assert "\n\nCould your quote include delivery" in preview
