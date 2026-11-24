"""Fixes from the second scenario run: the same email twice gets one answer
(``integrations.email_repeats``), an email turn whose step waits on a card
raises no second card (``alert_tools``), an email to one person can be
allowed from then on, and a card says plainly why it waited and who an
email goes to (``take_the_lead``)."""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

from openexecutive.integrations import email_repeats
from openexecutive.orchestrator import take_the_lead as ttl
from openexecutive.orchestrator.session import Session
from openexecutive.people import store as people_store

from . import test_delegation_inbox as _inbox_tests
from .test_delegation_inbox import DANA

db = _inbox_tests.db
owner = _inbox_tests.owner

SEND = "google_workspace__send_gmail_message"
TEXT = "Hi Sam, do you want the next spools in black or grey? Just reply with the color."


def test_the_same_email_twice_is_answered_once(tmp_path: Path) -> None:
    path = tmp_path / "r.db"
    now = datetime(2026, 11, 13, 16, 0, tzinfo=UTC)
    key = email_repeats.fingerprint("Leo@x.example", "Spool colors", TEXT)
    assert key is not None
    # Nothing answered yet: not a copy. A copy counts only once a turn for a
    # vouched sender ran (``remember``), so a forged first copy can't hide it.
    assert email_repeats.is_copy(key, "m1", now=now, db_path=path) is False
    email_repeats.remember(key, "m1", now=now, db_path=path)
    # The same message read again (a retry) is not a copy.
    assert email_repeats.is_copy(key, "m1", now=now, db_path=path) is False
    # Another message with the same sender, subject and words is.
    again = email_repeats.fingerprint("leo@x.example", "Re: Spool colors", TEXT + "  ")
    assert again == key and email_repeats.is_copy(again, "m2", now=now, db_path=path) is True
    # Someone else, or other words, is new.
    assert email_repeats.fingerprint("ann@x.example", "Spool colors", TEXT) != key
    assert email_repeats.fingerprint("leo@x.example", "Spool colors", TEXT + " Thanks!") != key
    # Short mail is never matched.
    assert email_repeats.fingerprint("leo@x.example", "Re: Q", "Yes, go ahead.") is None
    assert email_repeats.is_copy(None, "m5", now=now, db_path=path) is False
    # A day later it's a new message.
    assert email_repeats.is_copy(key, "m7", now=now + timedelta(hours=25), db_path=path) is False


def test_mail_with_files_is_never_matched_as_a_copy() -> None:
    # The same note can carry a different attachment.
    body = "Please see the attached invoice for the March order, thanks a lot"
    assert email_repeats.fingerprint("dana@x.example", "Invoice", body) is not None
    assert email_repeats.fingerprint("dana@x.example", "Invoice", body, has_attachments=True) is None


def test_copies_rendered_with_their_own_headers_still_match() -> None:
    body = "Could you send over the signed order form for the spring delivery please"
    first = f"Message ID: a1\nDate: Mon\nSubject: Order\n\n--- BODY ---\n{body}"
    second = f"Message ID: b2\nDate: Tue\nSubject: Order\n\n--- BODY ---\n{body}"
    key = email_repeats.fingerprint("dana@x.example", "Order", first)
    assert key is not None and key == email_repeats.fingerprint("dana@x.example", "Re: Order", second)
    # No parsed body: nothing to match.
    assert email_repeats.fingerprint("dana@x.example", "Order", "") is None


def test_an_email_is_noted_as_answered_only_when_the_sender_got_something() -> None:
    from openexecutive.integrations import email_poller

    noted: list[Any] = []

    def note(session: Session, **kw: Any) -> list[Any]:
        noted.clear()
        with patch.object(email_repeats, "remember", new=lambda *a, **_k: noted.append(a)):
            asyncio.run(email_poller._note_if_answered("k", "m1", session, "Dana@X.example", **kw))
        return list(noted)

    session = Session(session_id="email:t1")
    session.email_sender_authenticated = True
    assert note(session) == []  # the turn sent nothing: a resend is handled again
    assert note(session, answered=True) == [("k", "m1")]
    session.mail_sent_to.add("dana@x.example")
    assert note(session) == [("k", "m1")]
    session.mail_sent_to.clear()
    session.steps_held = 1  # its step waits on a card
    assert note(session) == [("k", "m1")]
    session.email_sender_authenticated = False  # nobody vouched for the sender
    assert note(session, answered=True) == []


def test_an_email_is_allowed_next_time_as_its_kind(owner: Any) -> None:
    found = people_store.find_person_by_address(DANA, include_contacts=True)
    assert found is not None
    grant = ttl.allowance(SEND, {"to": DANA, "subject": "Hi", "body": "Hello"}, mcp=True)
    assert grant == (f"{ttl.FEATURE}|message", "Send messages and emails")
    # The same Allow as a message on any channel, and as an email to anyone.
    assert ttl.allowance("message_person", {"person_id": found.id, "text": "Hi"}) == grant
    assert ttl.allowance(SEND, {"to": DANA, "cc": "x@far.example", "body": "Hi"}, mcp=True) == grant
    assert ttl.allowance(SEND, {"to": "x@far.example", "body": "Hi"}, mcp=True) == grant

def test_cards_say_who_an_email_goes_to_and_why_it_waited(owner: Any) -> None:
    found = people_store.find_person_by_address(DANA, include_contacts=True)
    assert found is not None
    step = {"to": DANA, "subject": "Invoice", "body": "Yes, bill $300."}
    assert ttl.summarize(SEND, step, mcp=True).startswith(f"Email {found.full_name}")
    # The activity feed, which the team sees, doesn't name them.
    assert ttl.summarize(SEND, step, mcp=True, quote=False) == "Send an email"
    two = {**step, "to": f"{DANA}, x@far.example"}
    assert ttl.summarize(SEND, two, mcp=True).startswith("Email 2 people")
    ttl.add_rule(ttl.SCOPE_COMPANY, "amount", "$250", created_by="t")
    hit = ttl._rule_hit(ttl.list_rules([ttl.SCOPE_COMPANY]), "Yes, bill $300.", [DANA])
    # It quotes the amount it saw, so the card shows what tripped the rule.
    assert hit is not None and hit.reason == "it says $300, and your rule about amounts of $250 or more says to wait"
    assert f"It waited because {hit.reason}." == (
        "It waited because it says $300, and your rule about amounts of $250 or more says to wait."
    )
    hit = ttl._rule_hit(ttl.list_rules([ttl.SCOPE_COMPANY]), "We'll pay $1,284.50 by Dec 4, not $90.", [DANA])
    assert hit is not None and hit.reason.startswith("it says $1,284.50, and")
    # Quoted as written: another currency is never turned into dollars.
    hit = ttl._rule_hit(ttl.list_rules([ttl.SCOPE_COMPANY]), "Yes, bill $300, thanks.", [DANA])
    assert hit is not None and hit.reason.startswith("it says $300, and")
    hit = ttl._rule_hit(ttl.list_rules([ttl.SCOPE_COMPANY]), "Pay €900.", [DANA])
    assert hit is not None and hit.reason.startswith("it says €900, and")
    hit = ttl._rule_hit(ttl.list_rules([ttl.SCOPE_COMPANY]), "Invoice for 2k EUR", [DANA])
    assert hit is not None and hit.reason.startswith("it says 2k EUR, and")
    assert ttl._money("1500") == "$1,500" and ttl._money("12.5") == "$12.50"
    json.dumps(hit.reason)
