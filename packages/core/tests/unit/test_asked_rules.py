"""Your rules and In training on a step someone asked for
(``take_the_lead.asked_*``), and what the As me look is told
(``workflows.executive_reflection``)."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openexecutive.alerts import store as alerts_store
from openexecutive.orchestrator import take_the_lead as ttl
from openexecutive.orchestrator.schedule_tools import current_session
from openexecutive.people import registry as people_registry
from openexecutive.people import store as people_store
from openexecutive.workflows import executive_reflection

from . import test_delegation_inbox as _inbox_tests
from .test_delegation_inbox import DANA, OWNER

db = _inbox_tests.db
owner = _inbox_tests.owner

SEND = "google_workspace__send_gmail_message"


@pytest.fixture(autouse=True)
def alerts_db(monkeypatch: pytest.MonkeyPatch, db: Path) -> Iterator[None]:
    monkeypatch.setattr(alerts_store, "DB_PATH", db)
    alerts_store.initialize_db(db)
    from openexecutive.workflows import persistence as wf_persistence

    monkeypatch.setattr(wf_persistence, "DB_PATH", db)
    wf_persistence.initialize_runs_db(db)
    # Balanced, unless a test picks Careful: it holds every step on its own.
    from openexecutive.orchestrator import outside_mode

    monkeypatch.setattr(outside_mode, "get_mode", lambda **_k: outside_mode.OUTSIDERS)
    yield


def _calls() -> tuple[list[dict[str, Any]], Any]:
    seen: list[dict[str, Any]] = []

    async def call_tool(tool_input: dict[str, Any]) -> str:
        seen.append(tool_input)
        return "Email sent!"

    return seen, call_tool


def test_a_waits_rule_naming_someone_holds_mail_to_their_address(owner: Any) -> None:
    """A teammate's chat ask to email Dana meets "Always wait for Dana Park",
    though the mail only carries her address."""
    ttl.add_rule(ttl.SCOPE_COMPANY, "person", "Dana Park", created_by="t")
    seen, call_tool = _calls()
    asked = ttl.asked_call_tool(call_tool)
    held = asyncio.run(asked({"name": SEND, "arguments": {"to": DANA, "subject": "Address", "body": "Hi, which address?"}}))
    assert json.loads(held)["status"] == "waiting_for_approval" and seen == []
    # Mail to anyone else goes; so does a read.
    assert asyncio.run(asked({"name": SEND, "arguments": {"to": "x@far.example", "body": "Hi"}})) == "Email sent!"
    assert asyncio.run(asked({"name": "google_workspace__search_gmail_messages", "arguments": {"q": "x"}})) == "Email sent!"
    # Arguments sent as a JSON string are checked the same way.
    held = asyncio.run(asked({"name": SEND, "arguments": json.dumps({"to": DANA, "body": "Hi"})}))
    assert json.loads(held)["status"] == "waiting_for_approval"
    # An unattended run sees the name too (the As me gate).
    hit, _ = ttl._hit(SEND, {"to": DANA, "body": "Hi"}, mcp=True)
    assert hit is not None and hit.kind == "rule"


def test_a_waits_rule_naming_someone_holds_a_meeting_with_them(owner: Any) -> None:
    """Your rules see everyone a booking invites: "Always wait for Dana
    Park" and a rule on her company hold a call booked with her."""
    dana = people_store.find_person_by_address(DANA, include_contacts=True)
    assert dana is not None
    booked: list[dict[str, Any]] = []

    async def create_calendar_event(tool_input: dict[str, Any]) -> str:
        booked.append(tool_input)
        return json.dumps({"status": "booked"})

    book = ttl.asked_handlers({"create_calendar_event": create_calendar_event})["create_calendar_event"]
    step = {"title": "Catch up", "start": "2026-11-20T15:00:00Z", "end": "2026-11-20T15:30:00Z",
            "attendee_person_ids": [dana.id]}
    assert json.loads(asyncio.run(book(step))) == {"status": "booked"}
    for kind, value in (("person", "Dana Park"), ("domain", DANA.split("@")[1])):
        rule = ttl.add_rule(ttl.SCOPE_COMPANY, kind, value, created_by="t")
        held = json.loads(asyncio.run(book(step)))
        assert held["status"] == "waiting_for_approval" and len(booked) == 1
        hit, _ = ttl._hit("create_calendar_event", step, mcp=False)
        assert hit is not None and hit.kind == "rule"
        ttl.delete_rule(rule.id, ttl.SCOPE_COMPANY)


def test_in_training_holds_an_asked_step_until_allowed(owner: Any) -> None:
    seen, call_tool = _calls()
    asked = ttl.asked_call_tool(call_tool)
    step = {"name": SEND, "arguments": {"to": "x@far.example", "body": "Hi"}}
    assert asyncio.run(asked(step)) == "Email sent!"
    ttl.set_(ttl.SCOPE_EXECUTIVE, training=True, updated_by="t")
    held = json.loads(asyncio.run(asked(step)))
    assert held["status"] == "waiting_for_approval" and len(seen) == 1
    # Mail only to the owner never waits.
    assert asyncio.run(asked({"name": SEND, "arguments": {"to": OWNER, "body": "Hi"}})) == "Email sent!"


def test_asked_handlers_gate_acting_skills_only(owner: Any) -> None:
    ttl.add_rule(ttl.SCOPE_COMPANY, "person", "Dana Park", created_by="t")
    dana = people_store.find_person_by_address(DANA, include_contacts=True)
    assert dana is not None
    ran: list[str] = []

    async def message_person(tool_input: dict[str, Any]) -> str:
        ran.append("message_person")
        return json.dumps({"ok": True})

    async def lookup_person(tool_input: dict[str, Any]) -> str:
        ran.append("lookup_person")
        return "{}"

    handlers = ttl.asked_handlers({"message_person": message_person, "lookup_person": lookup_person})
    held = asyncio.run(handlers["message_person"]({"person_id": dana.id, "text": "Hi"}))
    assert json.loads(held)["status"] == "waiting_for_approval"
    asyncio.run(handlers["lookup_person"]({"name": "Dana"}))
    assert ran == ["lookup_person"]
    # To the owner: goes.
    asyncio.run(handlers["message_person"]({"person_id": owner.id, "text": "Done"}))
    assert ran == ["lookup_person", "message_person"]


def test_the_owners_own_ask_meets_the_rules_too(owner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every asked turn is checked, the owner's own included; unattended runs
    meet the gate another way."""
    from openexecutive.orchestrator import content_trust

    assert ttl.asked_checks_apply(None) is True  # no session: fails closed
    assert ttl.asked_checks_apply(SimpleNamespace(unattended=True)) is False
    for owner_speaking in (True, False):
        monkeypatch.setattr(content_trust, "principal_speaking", lambda _s, v=owner_speaking: v)
        assert ttl.asked_checks_apply(SimpleNamespace(unattended=False)) is True


def test_the_owners_own_ask_holds_when_the_rules_cant_be_read(
    owner: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fails closed for the owner as for anyone: unreadable rules hold."""
    def _broken(*_: object) -> list[Any]:
        raise RuntimeError("rules unreadable")

    monkeypatch.setattr(ttl, "list_rules", _broken)
    hit = ttl._asked_hit("message_person", {"person_id": owner.id, "text": "Hi"}, mcp=False)
    assert hit is not None and hit.reason == "its rules couldn't be read"


def test_careful_holds_even_the_owners_own_ask(owner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Careful asks before anything goes: the owner's own chat ask waits on
    a card too, a teammate's as well; on Balanced the owner's meets Your
    rules and In training like anyone's."""
    from openexecutive.orchestrator import content_trust, outside_mode

    monkeypatch.setattr(content_trust, "principal_speaking", lambda _s: True)
    seen, call_tool = _calls()
    asked = ttl.asked_call_tool(call_tool)
    step = {"name": SEND, "arguments": {"to": "x@far.example", "subject": "Prices", "body": "Q4 prices hold."}}
    token = current_session.set(SimpleNamespace(unattended=False))  # type: ignore[arg-type]
    try:
        monkeypatch.setattr(outside_mode, "get_mode", lambda **_k: outside_mode.ASK)
        assert ttl.asked_checks_apply(current_session.get()) is True
        held = json.loads(asyncio.run(asked(step)))
        assert held["status"] == "waiting_for_approval" and "Careful" in held["message"] and seen == []
        # The owner's card, even when the step's words would route it to a
        # delegate (a board range): Careful's hold is never someone else's.
        from openexecutive.memory.decision_ledger import get_decision_instance

        monkeypatch.setattr(ttl, "_team_only", lambda *_a, **_k: True)
        monkeypatch.setattr(ttl, "_approver_for", lambda *_a, **_k: 999)
        board = json.loads(asyncio.run(asked({"name": SEND, "arguments": {"to": OWNER, "cc": "x@far.example", "body": "The board"}})))
        row = get_decision_instance(board["decision_instance_id"])
        assert row is not None and row.approver_person_id == owner.id
        # Mail only to the owner still goes.
        assert asyncio.run(asked({"name": SEND, "arguments": {"to": OWNER, "body": "Hi"}})) == "Email sent!"
        # A teammate's ask waits on Careful too.
        monkeypatch.setattr(content_trust, "principal_speaking", lambda _s: False)
        assert json.loads(asyncio.run(asked(step)))["status"] == "waiting_for_approval"
        # Balanced: the owner's own ask meets their Waits rule like anyone's,
        # and In training holds it too.
        monkeypatch.setattr(content_trust, "principal_speaking", lambda _s: True)
        monkeypatch.setattr(outside_mode, "get_mode", lambda **_k: outside_mode.OUTSIDERS)
        assert ttl._asked_hit(SEND, step["arguments"], mcp=True) is None
        rule = ttl.add_rule(ttl.SCOPE_COMPANY, "words", "Prices", created_by="t")
        assert ttl.asked_checks_apply(current_session.get()) is True
        hit = ttl._asked_hit(SEND, step["arguments"], mcp=True)
        assert hit is not None and hit.kind == "rule"
        ttl.delete_rule(rule.id, ttl.SCOPE_COMPANY)
        ttl.set_(ttl.SCOPE_EXECUTIVE, training=True, ask_first=None, updated_by="t")
        hit = ttl._asked_hit(SEND, step["arguments"], mcp=True)
        assert hit is not None and hit.kind == ttl.TRAINING
    finally:
        current_session.reset(token)


def _card(address: str, headline: str, suffix: str = "") -> int:
    key = f"private:email-from:{address}{suffix}"
    alerts_store.insert_alert(
        source="email", external_id=key, severity="medium", headline=headline, body=headline,
        topic_tags=["private:principal"], dedup_key=key,
    )
    found = [a for a in alerts_store.list_alerts(limit=50) if a.dedup_key == key]
    assert found and found[0].id is not None
    return int(found[0].id)


def test_the_look_is_told_to_take_steps_and_sees_what_came_in(owner: Any) -> None:
    note = executive_reflection._LEAD_NOTE
    assert "never hold one back yourself" in note and "waits for a yes on its own" not in note
    first = _card(DANA, "Dana asks to move the call")
    cards, private = executive_reflection._new_cards_since("2000-01-01T00:00:00+00:00")
    assert any(f"[{first}]" in line for line in cards) and private is True
    assert executive_reflection._new_cards_since("2999-01-01T00:00:00+00:00") == ([], False)
    assert executive_reflection._last_look_at() is None
    people_registry.invalidate()


def test_malformed_arguments_are_refused_not_run(owner: Any) -> None:
    seen, call_tool = _calls()
    asked = ttl.asked_call_tool(call_tool)
    for bad in ("not json", "[1, 2]", ["x"]):
        out = json.loads(asyncio.run(asked({"name": SEND, "arguments": bad})))
        assert "must be an object" in out["error"]
    assert seen == []


def test_the_one_who_asked_never_approves_their_own_held_step(
    owner: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.memory.decision_ledger import get_decision_instance

    mate = people_store.upsert_person(full_name="Bob Mate", email="bob@co.example")
    people_registry.invalidate()
    monkeypatch.setattr(ttl, "_team_only", lambda *_a, **_k: True)
    monkeypatch.setattr(ttl, "_approver_for", lambda *_a, **_k: mate)
    token = current_session.set(SimpleNamespace(caller_person_id=mate))  # type: ignore[arg-type]
    try:
        hit = ttl.Hit("rule", "your rule about Dana Park says to wait")
        chat = ttl.hold("message_person", {"person_id": mate, "text": "board deck"}, hit, source="chat", mcp=False)
        unattended = ttl.hold("message_person", {"person_id": mate, "text": "board"}, hit, source="scheduled", mcp=False)
    finally:
        current_session.reset(token)
    chat_row, other_row = get_decision_instance(chat), get_decision_instance(unattended)
    assert chat_row is not None and chat_row.approver_person_id == owner.id
    assert other_row is not None and other_row.approver_person_id == mate
    # The card says who asked: "Bob Mate asked", on web and phone alike.
    from openexecutive.alerts import store as alerts_store
    from openexecutive.memory.decision_ledger import decision_alert_external_id

    card = alerts_store.get_alert_by_external("decision_scheduling", decision_alert_external_id(chat))
    assert card is not None and card.headline.startswith("Bob Mate asked: ")
    assert json.loads(chat_row.proposed_payload_json)["requested_by"] == "Bob Mate"
    other = alerts_store.get_alert_by_external("decision_scheduling", decision_alert_external_id(unattended))
    # Work nobody asked for says what it does, once, and the button names it.
    assert other is not None and other.headline == "Message Bob Mate"
    assert other.action_label == other.headline


def test_an_asked_send_that_went_on_its_own_is_logged_with_why(owner: Any) -> None:
    """Done for you and the counts read the log: a reply that went without
    a yes is there, with why it went; one that waited is not "done"."""
    from openexecutive.orchestrator import outside_mode

    seen, call_tool = _calls()
    asked = ttl.asked_call_tool(call_tool)
    token = current_session.set(SimpleNamespace(email_from=DANA, unattended=False))
    try:
        sent = asyncio.run(asked({"name": SEND, "arguments": {"to": DANA, "subject": "Hours", "body": "Open 9 to 5."}}))
    finally:
        current_session.reset(token)
    assert sent == "Email sent!"
    [row] = [d for d in ttl.done([ttl.SCOPE_EXECUTIVE], days=1) if d.status == "done"]
    assert row.mail_to == DANA and row.mail_subject == "Hours"
    assert row.why.startswith("Dana Park asked by email. Went on its own")
    assert "no rule of yours said to wait" in row.why
    assert ttl.went_on_its_own_why(outside_mode.OUTSIDERS).startswith("Approvals is Balanced")
    # Mail only to the owner isn't something it did on its own.
    asyncio.run(asked({"name": SEND, "arguments": {"to": OWNER, "body": "Hi"}}))
    assert len([d for d in ttl.done([ttl.SCOPE_EXECUTIVE], days=1) if d.status == "done"]) == 1


def test_who_asked_speaks_to_the_owner(owner: Any) -> None:
    """The owner's own mail reads "You asked", never their name, and only
    when it proved it was theirs."""
    from openexecutive.people.store import find_principal_person

    me = find_principal_person()
    assert me is not None and me.email
    assert ttl.who_asked(SimpleNamespace(email_from=me.email, email_authenticated=True)) == "You asked by email."
    assert ttl.who_asked(SimpleNamespace(email_from=me.email)) == "Mail from your address asked."
    assert ttl.who_asked(SimpleNamespace(email_from=DANA)) == "Dana Park asked by email."
    assert ttl.who_asked(SimpleNamespace(email_from="", caller_person_id=None)) == ""
    # A chat turn names who is in it.
    from openexecutive.people.store import upsert_person

    mate = upsert_person(full_name="Sam Mate", email="sam@example.com")
    assert ttl.who_asked(SimpleNamespace(email_from="", caller_person_id=mate)) == "Sam Mate asked in chat."
    assert ttl.who_asked(SimpleNamespace(email_from="", caller_person_id=me.id)) == "You asked in chat."


def test_a_send_a_later_check_put_on_a_card_is_not_done(owner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gateway held the send on an Approve card: Done for you must not
    say it was sent."""
    async def call_tool(tool_input: dict[str, Any]) -> str:
        return json.dumps({"status": "waiting_for_approval", "decision_instance_id": 9, "message": "Not done yet"})

    from openexecutive.memory import decision_ledger

    asked = ttl.asked_call_tool(call_tool)
    token = current_session.set(SimpleNamespace(email_from=DANA, unattended=False))
    try:
        # A result naming no real card is a step that ran.
        asyncio.run(asked({"name": SEND, "arguments": {"to": DANA, "subject": "Hours", "body": "9 to 5."}}))
        assert len([d for d in ttl.done([ttl.SCOPE_EXECUTIVE], days=1) if d.status == "done"]) == 1
        # A card already answered, or one for another tool, is no card for it.
        for status, tool in (("approved", SEND), ("proposed", "google_workspace__create_event")):
            monkeypatch.setattr(decision_ledger, "get_decision_instance", lambda i, db_path=None, s=status, t=tool: (
                SimpleNamespace(status=s, proposed_payload_json=json.dumps({"tool": t}))))
            asyncio.run(asked({"name": SEND, "arguments": {"to": DANA, "subject": f"Hours {status}", "body": "9 to 5."}}))
        assert len([d for d in ttl.done([ttl.SCOPE_EXECUTIVE], days=1) if d.status == "done"]) == 3
        monkeypatch.setattr(decision_ledger, "get_decision_instance", lambda i, db_path=None: SimpleNamespace(
            status="proposed", proposed_payload_json=json.dumps({"tool": SEND})))
        result = asyncio.run(asked({"name": SEND, "arguments": {"to": DANA, "subject": "Paying early", "body": "Fine."}}))
    finally:
        current_session.reset(token)
    assert "waiting_for_approval" in result
    assert len([d for d in ttl.done([ttl.SCOPE_EXECUTIVE], days=1) if d.status == "done"]) == 3



def test_a_level_that_cant_be_read_holds_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    from openexecutive.orchestrator import levels, outside_mode

    def _broken(*_: object) -> str:
        raise RuntimeError("settings unreadable")

    monkeypatch.setattr(levels, "turn_level", _broken)
    monkeypatch.setattr(outside_mode, "get_mode", _broken)
    assert ttl._careful_turn() == "your Approvals level couldn't be read"
