"""One card per topic, whatever wrote the card (alerts.one_card)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from openexecutive import audit
from openexecutive.alerts import one_card
from openexecutive.alerts import store as alerts_store
from openexecutive.alerts.models import PRIVATE_ALERT_TAG
from openexecutive.audit.context import set_turn
from openexecutive.memory import decision_ledger as ledger
from openexecutive.memory import episodic
from openexecutive.orchestrator.schedule_tools import current_session
from openexecutive.orchestrator.session import Session

OWNER = 1
SAM = 2


@pytest.fixture(autouse=True)
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    path = tmp_path / "episodic.db"
    monkeypatch.setattr(episodic, "DB_PATH", path)
    monkeypatch.setattr(alerts_store, "DB_PATH", path)
    monkeypatch.setattr(audit, "log_event", lambda *_a, **_k: None)
    episodic.initialize_db(path)
    alerts_store.initialize_db(path)
    yield path


@pytest.fixture
def mail_turn() -> Iterator[Session]:
    session = Session(session_id="email:t1", origin_channel="email")
    session.email_from = "dana@northpeak.example"
    session.email_thread_id = "thread-1"
    token = current_session.set(session)
    try:
        yield session
    finally:
        current_session.reset(token)


def _note(
    headline: str, *, reader: int | None = OWNER, private: bool = False, source: str = "email",
    raised: bool = True, joins: int | None = None, severity: str = "medium", tags: tuple[str, ...] = (), **kw: Any,
) -> int:
    with one_card.about(raised=raised, joins=joins):
        alert_id = alerts_store.insert_alert(
            source=source, external_id=f"{headline}-{id(kw)}", severity=severity, headline=headline,
            body=kw.pop("body", headline), routed_to_person_id=reader,
            topic_tags=[*([PRIVATE_ALERT_TAG] if private else []), *tags], **kw,
        )
    assert alert_id is not None
    return alert_id


def _approve(tool: str = "google_workspace__send_gmail_message", tool_input: dict[str, Any] | None = None,
             *, reader: int | None = OWNER) -> tuple[int, int]:
    decision_id = ledger.create_decision_instance(
        decision_class="take_the_lead_action", department="executive", originating_session_id=None,
        proposed_payload={"tool": tool, "input": tool_input or {}}, idempotency_key=None,
        gate_mode="propose", approver_person_id=reader, confidence=None,
    )
    thread, step = one_card.held_topic(tool, [tool_input or {}], mcp=True)
    with one_card.about(thread=thread, step=step):
        alert_id = alerts_store.insert_alert(
            source=ledger.DECISION_ALERT_SOURCE, external_id=ledger.decision_alert_external_id(decision_id),
            severity="medium", headline=f"Waiting for your OK: {tool}", body="Send it.",
            topic_tags=[ledger.decision_instance_tag(decision_id)], routed_to_person_id=reader,
        )
    assert alert_id is not None
    return alert_id, decision_id


def _get(alert_id: int) -> Any:
    alert = alerts_store.get_alert(alert_id)
    assert alert is not None
    return alert


def test_a_note_and_an_approve_card_from_one_turn_are_one_card_either_way_round() -> None:
    with set_turn(session_id="s", turn_id="turn-1"):
        note = _note("Northpeak wants Thursday")
        card, _ = _approve()
    with set_turn(session_id="s", turn_id="turn-2"):
        card_first, _ = _approve()
        late = _note("Leo asked for the order")
    for joined, into in ((note, card), (late, card_first)):
        assert _get(joined).status == "dismissed" and _get(joined).superseded_by_alert_id == into
        assert _get(into).status == "unread"
        assert "Then: " in _get(into).body


def test_two_notes_from_one_turn_are_one_card_asking_the_question() -> None:
    with set_turn(session_id="s", turn_id="turn-1"):
        first = _note("Plan updated")
        second = _note("Pick a slot", needs="choice", question="Which slot?", choices=["Mon", "Tue"])
    kept = _get(first)
    assert _get(second).superseded_by_alert_id == first
    assert kept.question == "Which slot?" and kept.choices == ["Mon", "Tue"] and "Pick a slot" in kept.body


def test_new_text_on_an_opened_card_makes_it_new_again() -> None:
    with set_turn(session_id="s", turn_id="turn-1"):
        first = _note("Dock door 4 on Dec 3?")
        alerts_store.set_status(first, "read")
        _note("Dana needs the answer today")
    kept = _get(first)
    assert kept.status == "unread" and "Dana needs the answer today" in kept.body


def test_a_card_put_away_is_no_card_to_join(mail_turn: Session) -> None:
    with set_turn(session_id="s", turn_id="turn-1"):
        first = _note("Leo asked about Friday")
    assert alerts_store.set_alert_archived(first, True)
    owners = one_card.reader_for(OWNER, private=False)
    assert one_card.open_cards_from("mail:dana@northpeak.example", turn="turn-2", reader=owners) == []
    # The model naming it is its own card too.
    with set_turn(session_id="s", turn_id="turn-2"):
        named = _note("Leo asked again", joins=first)
    assert _get(named).status == "unread" and _get(named).superseded_by_alert_id is None
    # And so is the same person on the same thread.
    alerts_store.set_alert_archived(named, True)
    with set_turn(session_id="s", turn_id="turn-3"):
        again = _note("Leo asked about Friday again")
    assert _get(again).status == "unread" and _get(again).superseded_by_alert_id is None


def test_a_put_away_approve_card_is_still_replaced_by_a_newer_step(mail_turn: Session) -> None:
    reply = {"to": "dana@northpeak.example", "subject": "Re: Labels", "thread_id": "thread-1"}
    with set_turn(session_id="s", turn_id="turn-1"):
        old_card, old = _approve(tool_input={**reply, "body": "x"})
    assert alerts_store.set_alert_archived(old_card, True)
    with set_turn(session_id="s", turn_id="turn-2"):
        new_card, _ = _approve(tool_input={**reply, "body": "y"})
    # Its step never waits on behind the newer one.
    assert ledger.get_decision_instance(old).status == ledger.STATUS_CLOSED_EXTERNALLY  # type: ignore[union-attr]
    assert _get(old_card).superseded_by_alert_id == new_card and _get(new_card).status == "unread"


def test_cards_for_different_readers_or_privacy_stay_apart() -> None:
    with set_turn(session_id="s", turn_id="turn-1"):
        mine = _note("For the owner")
        theirs = _note("For Sam", reader=SAM)
        private = _note("Private to the owner", private=True)
        team = _note("For the team", reader=None)
    assert all(_get(a).status == "unread" for a in (mine, theirs, private, team))


def test_the_same_person_on_another_matter_keeps_its_own_card(mail_turn: Session) -> None:
    with set_turn(session_id="s", turn_id="turn-1"):
        first = _note("Dana asked for Thursday")
    mail_turn.email_thread_id = "thread-2"
    with set_turn(session_id="s", turn_id="turn-2"):
        other = _note("Dana sent a new invoice question")
    assert _get(other).status == "unread"
    # The model said it is the same matter: it joins.
    with set_turn(session_id="s", turn_id="turn-3"):
        same = _note("Dana asked again about Thursday", joins=first)
    assert _get(same).superseded_by_alert_id == first


def test_a_later_mail_on_the_same_thread_from_the_same_person_joins(mail_turn: Session) -> None:
    with set_turn(session_id="s", turn_id="turn-1"):
        first = _note("Dana asked for Thursday")
    with set_turn(session_id="s", turn_id="turn-2"):
        bump = _note("Dana is still waiting")
    assert _get(bump).superseded_by_alert_id == first
    # Someone else writing on that thread is their own card.
    mail_turn.email_from = "leo@northpeak.example"
    with set_turn(session_id="s", turn_id="turn-3"):
        leo = _note("Leo chimed in")
    assert _get(leo).status == "unread"


def test_a_newer_reply_on_the_same_thread_replaces_the_one_waiting() -> None:
    reply = {"to": "dana@northpeak.example", "subject": "Re: Thursday", "body": "x", "thread_id": "thread-1"}
    old_card, old = _approve(tool_input=reply)
    thread, step = one_card.held_topic("google_workspace__send_gmail_message", [reply], mcp=True)
    assert one_card.approve_waiting(thread, reader=one_card.reader_for(OWNER, private=False), step=step)
    new_card, new = _approve(tool_input={**reply, "body": "y"})
    assert ledger.get_decision_instance(old).status == ledger.STATUS_CLOSED_EXTERNALLY  # type: ignore[union-attr]
    assert ledger.get_decision_instance(new).status == ledger.STATUS_PROPOSED  # type: ignore[union-attr]
    assert _get(old_card).status == "dismissed" and _get(new_card).status == "unread"
    # A send to other people on that thread is its own yes.
    other_card, other = _approve(tool_input={**reply, "to": "leo@northpeak.example"})
    assert ledger.get_decision_instance(new).status == ledger.STATUS_PROPOSED  # type: ignore[union-attr]
    assert _get(other_card).status == "unread"


def test_two_different_steps_from_one_turn_are_two_approve_cards() -> None:
    with set_turn(session_id="s", turn_id="turn-1"):
        mail, _ = _approve(tool_input={"to": "dana@northpeak.example", "subject": "Hi", "body": "x"})
        meeting, _ = _approve("create_calendar_event", {"title": "Sync"})
    assert _get(mail).status == "unread" and _get(meeting).status == "unread"


def test_cards_a_feature_writes_keep_to_themselves() -> None:
    """A passed-on message, a meeting's card or a document has its own
    reader and buttons: it never joins or takes in a card."""
    with set_turn(session_id="s", turn_id="turn-1"):
        message = _note("From Sam: the dock is closed Friday", source="message", raised=False)
        meeting = _note("Notes from the 3pm call", source="meetings", raised=False)
        card, _ = _approve()
        note = _note("Northpeak wants Thursday")
    assert _get(message).status == _get(meeting).status == "unread"
    assert _get(message).body == "From Sam: the dock is closed Friday"
    assert _get(note).superseded_by_alert_id == card


def test_a_card_whose_approval_does_its_own_text_keeps_to_itself() -> None:
    with set_turn(session_id="s", turn_id="turn-1"):
        ok = _note("Send the renewal deck to Northpeak?", needs="ok")
        note = _note("Northpeak wants Thursday")
        card, _ = _approve()
    assert _get(ok).status == "unread" and _get(ok).body == "Send the renewal deck to Northpeak?"
    assert _get(note).superseded_by_alert_id == card


def test_another_matter_on_the_same_thread_keeps_its_own_card(mail_turn: Session) -> None:
    with set_turn(session_id="s", turn_id="turn-1"):
        first = _note("Dana asked for Thursday")
    with set_turn(session_id="s", turn_id="turn-2"):
        other = _note("Dana asked for a new price list", joins=0)
    assert _get(first).status == _get(other).status == "unread"


def test_a_grave_card_makes_the_card_it_joins_as_grave() -> None:
    with set_turn(session_id="s", turn_id="turn-1"):
        card, _ = _approve()
        _note("These bank details look changed: possible invoice fraud", severity="high")
    kept = _get(card)
    assert kept.severity == "high" and "possible invoice fraud" in kept.body


def test_a_card_another_joins_no_longer_names_one_step_on_its_button() -> None:
    with set_turn(session_id="s", turn_id="turn-1"):
        card, _ = _approve()
        alerts_store.update_alert_content(card, suggested_action="Send it.", action_label="Email Leo Park")
        assert _get(card).action_label == "Email Leo Park"
        _note("Also email Sam the price list")
    assert _get(card).action_label == ""


def test_a_card_from_no_turn_is_left_alone() -> None:
    first = _note("Weekly digest")
    second = _note("Weekly digest again")
    assert _get(first).status == _get(second).status == "unread"


def test_a_card_that_joined_is_never_sent() -> None:
    import asyncio

    from openexecutive.alerts import dispatcher
    from openexecutive.alerts.models import AlertChannel

    with set_turn(session_id="s", turn_id="turn-1"):
        card, _ = _approve()
        joined = _note("Then this")
    attempted, delivered = asyncio.run(dispatcher.dispatch_all(_get(joined), [AlertChannel.WEB, AlertChannel.EMAIL]))
    assert (attempted, delivered) == ([], [])
    assert _get(card).status == "unread"


def test_the_people_a_turn_is_with() -> None:
    assert one_card.person_key(Session(origin_channel="email", email_from="Dana@X.example")) == "mail:dana@x.example"
    assert one_card.person_key(Session(caller_person_id=4)) == "person:4"
    assert one_card.person_key(Session(unattended=True, run_for_person_id=5)) == "person:5"
    assert one_card.person_key(Session(unattended=True)) == ""


def test_open_cards_from_lists_earlier_turns_only() -> None:
    session = Session(caller_person_id=SAM)
    token = current_session.set(session)
    try:
        with set_turn(session_id="s", turn_id="turn-1"):
            earlier = _note("Sam's first ask")
        with set_turn(session_id="s", turn_id="turn-2"):
            _note("Sam's second ask", reader=SAM)
        owners = one_card.reader_for(OWNER, private=False)
        assert [cid for cid, _, _ in one_card.open_cards_from("person:2", turn="turn-2", reader=owners)] == [earlier]
        # Another reader's cards are never listed.
        assert one_card.open_cards_from("person:2", turn="turn-3", reader=one_card.reader_for(SAM, private=True)) == []
    finally:
        current_session.reset(token)
    assert json.dumps(one_card.open_cards_from("", turn="x", reader=owners)) == "[]"


def test_a_reply_the_mailbox_filed_as_a_new_thread_is_the_same_conversation(mail_turn: Session) -> None:
    one_card.note_mail("thread-1", "dana@northpeak.example", "Pallet labels for January")
    assert one_card.continues("thread-9", "dana@northpeak.example", "Re: Pallet labels for January")
    assert not one_card.continues("thread-7", "dana@northpeak.example", "Pallet labels for January")
    one_card.note_mail("thread-9", "dana@northpeak.example", "Re: Pallet labels for January", proven=True)
    one_card.note_mail("thread-7", "dana@northpeak.example", "Pallet labels for January", proven=True)  # a new ask
    one_card.note_mail("thread-8", "leo@northpeak.example", "Re: Pallet labels for January", proven=True)
    # A From: line the mailbox didn't vouch for starts its own thread.
    one_card.note_mail("thread-6", "dana@northpeak.example", "Re: Pallet labels for January")
    reply = {"to": "dana@northpeak.example", "subject": "Re: Pallet labels for January", "body": "x"}
    with set_turn(session_id="s", turn_id="turn-1"):
        old_card, old = _approve(tool_input={**reply, "thread_id": "thread-1"})
    mail_turn.email_thread_id = "thread-9"
    with set_turn(session_id="s", turn_id="turn-2"):
        bump = _note("Dana is following up on the labels")
        new_card, new = _approve(tool_input={**reply, "thread_id": "thread-9", "body": "y"})
    # The follow-up joined the waiting card, and moved with it to the newer reply.
    assert _get(bump).superseded_by_alert_id == new_card
    assert "Dana is following up on the labels" in _get(new_card).body
    assert ledger.get_decision_instance(old).status == ledger.STATUS_CLOSED_EXTERNALLY  # type: ignore[union-attr]
    assert _get(new_card).status == "unread"
    # The same subject written fresh, or from someone else, is its own matter.
    for thread, sender in (
        ("thread-7", "dana@northpeak.example"), ("thread-8", "leo@northpeak.example"),
        ("thread-6", "dana@northpeak.example"),
    ):
        mail_turn.email_thread_id, mail_turn.email_from = thread, sender
        with set_turn(session_id="s", turn_id=f"turn-{thread}"):
            assert _get(_note(f"From {thread}")).status == "unread"


def test_a_follow_up_joins_the_senders_latest_conversation_on_that_subject() -> None:
    one_card.note_mail("t-1", "dana@northpeak.example", "Labels")
    one_card.note_mail("t-7", "dana@northpeak.example", "Labels")
    one_card.note_mail("t-9", "dana@northpeak.example", "RE: labels", proven=True)
    one_card.note_mail("t-9", "dana@northpeak.example", "Something else", proven=True)  # noted once
    one_card.note_mail("", "dana@northpeak.example", "Re: Labels", proven=True)
    one_card.note_mail("t-5", "", "Re: Labels", proven=True)
    import sqlite3

    with sqlite3.connect(alerts_store.DB_PATH) as conn:
        rows = dict(conn.execute("SELECT thread_id, conversation FROM mail_conversations").fetchall())
    assert rows == {"t-1": "t-1", "t-7": "t-7", "t-9": "t-7"}


def test_a_replaced_card_hands_its_graver_joined_card_to_the_newer_one(mail_turn: Session) -> None:
    reply = {"to": "dana@northpeak.example", "subject": "Re: Thursday", "body": "x", "thread_id": "thread-1"}
    with set_turn(session_id="s", turn_id="turn-1"):
        old_card, _ = _approve(tool_input=reply)
        _note("These bank details look changed", severity="high")
    with set_turn(session_id="s", turn_id="turn-2"):
        new_card, _ = _approve(tool_input={**reply, "body": "y"})
    kept = _get(new_card)
    assert _get(old_card).status == "dismissed"
    assert kept.severity == "high" and "These bank details look changed" in kept.body


def test_the_subject_a_follow_up_is_matched_on_comes_from_the_header() -> None:
    from openexecutive.integrations.email_poller import _header_subject

    assert _header_subject("From: dana@x.example\nSUBJECT: Re: Labels\n\nSubject: not this") == "Re: Labels"
    assert _header_subject("From: dana@x.example\n\nSubject: in the body") == ""


def test_a_forward_on_the_same_subject_starts_its_own_conversation() -> None:
    one_card.note_mail("t-1", "dana@northpeak.example", "Pallet labels")
    assert not one_card.continues("t-2", "dana@northpeak.example", "Fwd: Pallet labels")
    assert one_card.continues("t-2", "dana@northpeak.example", "AW: Pallet labels")


def test_a_reply_waits_on_the_card_of_the_conversation_its_thread_joined(mail_turn: Session) -> None:
    one_card.note_mail("thread-1", "dana@northpeak.example", "Thursday")
    one_card.note_mail("thread-9", "dana@northpeak.example", "Re: Thursday", proven=True)
    reply = {"to": "dana@northpeak.example", "subject": "Re: Thursday", "body": "x", "thread_id": "thread-1"}
    _approve(tool_input=reply)
    thread, step = one_card.held_topic("google_workspace__send_gmail_message", [reply], mcp=True)
    owner = one_card.reader_for(OWNER, private=False)
    assert one_card.approve_waiting("mail:thread-9", reader=owner, step=step)
    assert not one_card.approve_waiting("mail:thread-5", reader=owner, step=step)


def test_each_newer_reply_takes_everything_the_one_it_replaced_held(mail_turn: Session) -> None:
    reply = {"to": "dana@northpeak.example", "subject": "Re: Thursday", "body": "a", "thread_id": "thread-1"}
    with set_turn(session_id="s", turn_id="turn-a"):
        a, _ = _approve(tool_input=reply)
        first = _note("Dana asked for Thursday", severity="high")
    with set_turn(session_id="s", turn_id="turn-b"):
        b, _ = _approve(tool_input={**reply, "body": "b"})
        second = _note("Dana asked again")
    with set_turn(session_id="s", turn_id="turn-c"):
        c, _ = _approve(tool_input={**reply, "body": "c"})
    kept = _get(c)
    assert _get(a).status == _get(b).status == "dismissed"
    assert kept.severity == "high"
    assert "Dana asked for Thursday" in kept.body and "Dana asked again" in kept.body
    assert _get(first).superseded_by_alert_id == _get(second).superseded_by_alert_id == c


def test_the_poller_links_a_follow_up_only_for_a_sender_the_mailbox_vouched_for() -> None:
    import asyncio

    from openexecutive.integrations.email_poller import _note_thread

    calls: list[str] = []

    def prover(result: bool | None) -> Any:
        async def prove() -> bool:
            calls.append("prove")
            if result is None:
                raise RuntimeError("mailbox down")
            return result
        return prove

    dana = "dana@northpeak.example"
    asyncio.run(_note_thread("t-1", dana, "Labels", prover(True)))
    assert calls == []  # a fresh mail is never proven: it links to nothing
    asyncio.run(_note_thread("t-2", dana, "Re: Labels", prover(False)))
    asyncio.run(_note_thread("t-3", dana, "Re: Labels", prover(None)))
    asyncio.run(_note_thread("t-4", dana, "Re: Labels", prover(True)))
    assert calls == ["prove", "prove", "prove"]
    import sqlite3

    with sqlite3.connect(alerts_store.DB_PATH) as conn:
        rows = dict(conn.execute("SELECT thread_id, conversation FROM mail_conversations").fetchall())
    # Unproven (t-2) and unprovable (t-3) follow-ups keep their own thread;
    # the vouched one (t-4) joins the latest conversation on that subject.
    assert rows == {"t-1": "t-1", "t-2": "t-2", "t-3": "t-3", "t-4": "t-3"}


def test_a_newer_reply_takes_each_follow_up_once_and_stays_within_the_cap(mail_turn: Session) -> None:
    from openexecutive.alerts.store import _FOLLOW_UP_MAX

    reply = {"to": "dana@northpeak.example", "subject": "Re: Thursday", "body": "a", "thread_id": "thread-1"}
    with set_turn(session_id="s", turn_id="turn-a"):
        a, _ = _approve(tool_input=reply)
        _note("Dana asked for Thursday")
        _note("x" * _FOLLOW_UP_MAX)
    with set_turn(session_id="s", turn_id="turn-b"):
        b, _ = _approve(tool_input={**reply, "body": "b"})
    body = _get(b).body
    assert len(body) <= _FOLLOW_UP_MAX
    follow_ups = body.split("\n\nThen: ")[1:]
    assert follow_ups and len(follow_ups) == len(set(follow_ups))


def test_a_newer_card_on_the_same_matter_says_where_it_stands_now(mail_turn: Session) -> None:
    """A contact's new request joined their old card, whose title and
    choices stayed the old times. The newer card's title, plan and question
    lead; the old words stay above its own."""
    with set_turn(session_id="s", turn_id="turn-1"):
        first = _note(
            "Sam asks to move Nov 24 to Dec 2 or 3", body="Sam offers Dec 2 or Dec 3.",
            suggested_action="Confirm Dec 2", needs="choice", question="Dec 2 or Dec 3?", choices=["Dec 2", "Dec 3"],
        )
    with set_turn(session_id="s", turn_id="turn-2"):
        newer = _note(
            "Sam now offers Dec 1 or Dec 2", body="Sam can't make Nov 24 and offers Dec 1 or Dec 2.",
            suggested_action="Reply proposing Dec 1", needs="choice", question="Dec 1 or Dec 2?",
            choices=["Dec 1", "Dec 2"],
        )
    assert _get(newer).superseded_by_alert_id == first
    kept = _get(first)
    assert kept.headline == "Sam now offers Dec 1 or Dec 2" and kept.suggested_action == "Reply proposing Dec 1"
    assert kept.question == "Dec 1 or Dec 2?" and kept.choices == ["Dec 1", "Dec 2"]
    assert kept.body.startswith("Sam offers Dec 2 or Dec 3.") and "Then: " in kept.body
    # A card that only tells adds its words and leaves the rest.
    with set_turn(session_id="s", turn_id="turn-3"):
        _note("Sam is still waiting")
    assert _get(first).question == "Dec 1 or Dec 2?" and _get(first).headline == "Sam now offers Dec 1 or Dec 2"


def test_a_team_members_card_joins_only_another_of_theirs() -> None:
    """A teammate's card is read by them and whom it is for: joining a card
    everyone reads would show their ask to the whole team, and joining another teammate's would hide it from them."""
    from openexecutive.alerts.models import TEAM_RAISED_SOURCE

    with set_turn(session_id="s", turn_id="turn-1"):
        team = _note("Owner's own card")
        theirs = _note("Dana asks for the FedEx deadline", source=TEAM_RAISED_SOURCE, owner_person_id=7)
        sams = _note("Sam asks for the UPS deadline", source=TEAM_RAISED_SOURCE, owner_person_id=8)
        again = _note("Dana asks again", source=TEAM_RAISED_SOURCE, owner_person_id=7)
    assert _get(theirs).status == "unread" and _get(theirs).superseded_by_alert_id is None
    assert _get(sams).status == "unread" and _get(team).status == "unread"
    assert _get(again).superseded_by_alert_id == theirs


def test_a_team_member_is_asked_about_their_own_open_card() -> None:
    """The key create_alert asks with (``reader_for``) is the card's own, so
    a teammate's next card on the matter is asked whether it joins theirs."""
    from openexecutive.alerts.models import TEAM_RAISED_SOURCE

    token = current_session.set(Session(caller_person_id=7))
    try:
        with set_turn(session_id="s", turn_id="turn-1"):
            theirs = _note("Dana asks for the FedEx deadline", source=TEAM_RAISED_SOURCE, owner_person_id=7)
        dana = one_card.reader_for(OWNER, private=False, raised_by=7, team_raised=True)
        sam = one_card.reader_for(OWNER, private=False, raised_by=8, team_raised=True)
        assert [cid for cid, _, _ in one_card.open_cards_from("person:7", turn="turn-2", reader=dana)] == [theirs]
        assert one_card.open_cards_from("person:7", turn="turn-2", reader=sam) == []
        owners = one_card.reader_for(OWNER, private=False)
        assert one_card.open_cards_from("person:7", turn="turn-2", reader=owners) == []
    finally:
        current_session.reset(token)


def test_a_team_members_cards_for_two_departments_stay_apart() -> None:
    """Unrouted, a teammate's card is for its department's people: joined
    into another department's card, its ask would reach neither."""
    from openexecutive.alerts.models import TEAM_RAISED_SOURCE

    with set_turn(session_id="s", turn_id="turn-1"):
        finance = _note("Dana asks finance", reader=None, source=TEAM_RAISED_SOURCE, owner_person_id=7,
                        tags=("department:finance",))
        sales = _note("Dana asks sales", reader=None, source=TEAM_RAISED_SOURCE, owner_person_id=7,
                      tags=("department:sales",))
        again = _note("Dana asks finance again", reader=None, source=TEAM_RAISED_SOURCE, owner_person_id=7,
                      tags=("department:finance",))
    assert _get(sales).superseded_by_alert_id is None and _get(sales).status == "unread"
    assert _get(again).superseded_by_alert_id == finance


def test_two_questions_from_one_turn_keep_the_first_ones_question() -> None:
    """Two asks in one turn are two questions, not a newer version of one:
    the card that stays keeps its own question and choices."""
    with set_turn(session_id="s", turn_id="turn-1"):
        budget = _note("Approve the Q1 budget?", needs="choice", question="Approve it?", choices=["Yes", "No"])
        vendor = _note("Which vendor?", needs="choice", question="A or B?", choices=["A", "B"])
    kept = _get(budget)
    assert _get(vendor).superseded_by_alert_id == budget
    assert kept.headline == "Approve the Q1 budget?" and kept.question == "Approve it?"
    assert kept.choices == ["Yes", "No"] and "Which vendor?" in kept.body


def test_create_alert_asks_with_the_key_a_teammates_card_is_written_with() -> None:
    """The key ``create_alert`` looks for open cards with (``_reader_of``)
    is the one the card it raises is stored under, department and all."""
    import sqlite3

    from openexecutive.alerts.models import TEAM_RAISED_SOURCE, AlertEvent
    from openexecutive.orchestrator.alert_tools import _reader_of

    with set_turn(session_id="s", turn_id="turn-1"):
        card = _note("Dana asks finance", reader=None, source=TEAM_RAISED_SOURCE, owner_person_id=7,
                     tags=("department:finance",))
    with sqlite3.connect(str(episodic.DB_PATH)) as conn:
        stored = conn.execute("SELECT reader FROM card_topics WHERE alert_id = ?", (card,)).fetchone()[0]
    event = AlertEvent(source=TEAM_RAISED_SOURCE, external_id="x", raised_by_person_id=7, channel="department:finance")
    assert _reader_of(event) == stored
    sams = AlertEvent(source=TEAM_RAISED_SOURCE, external_id="x", raised_by_person_id=8, channel="department:finance")
    assert _reader_of(sams) != stored


def test_a_question_from_the_same_turn_keeps_the_reports_title() -> None:
    """A report and a question in one turn: the card keeps the report's
    title and plan, and asks the question."""
    with set_turn(session_id="s", turn_id="turn-1"):
        report = _note("Q3 revenue down 12%", suggested_action="Review the forecast")
        _note("Approve the hiring freeze?", needs="choice", question="Freeze?", choices=["Yes", "No"])
    kept = _get(report)
    assert kept.headline == "Q3 revenue down 12%" and kept.suggested_action == "Review the forecast"
    assert kept.question == "Freeze?" and kept.choices == ["Yes", "No"]


def test_an_email_watch_card_joins_the_card_of_its_mail_thread(mail_turn: Session) -> None:
    """The Executive's handling of Robin's email made a card for Kendall; the
    watch Kendall set tells her about the same email: one card, not two,
    whichever comes first. No turn is bound when the watch's card is written:
    the thread and sender it names say what it is about."""
    with set_turn(session_id="s", turn_id="turn-1"):
        turn_card = _note("Robin Hale: December rent")
    token = current_session.set(None)
    try:
        with one_card.about(
            raised=True, thread=one_card.mail_thread("thread-1"),
            person=one_card.mail_person("Dana@Northpeak.example"),
        ):
            watch_card = alerts_store.insert_alert(
                source="mailbox", external_id="watch-1", severity="high", headline="Dana: December rent",
                body="You asked me to watch for this.", routed_to_person_id=OWNER,
            )
        assert watch_card is not None
        assert _get(watch_card).superseded_by_alert_id == turn_card
        # Another thread, or another reader, is a card of its own.
        with one_card.about(raised=True, thread=one_card.mail_thread("thread-2"),
                            person=one_card.mail_person("dana@northpeak.example")):
            other = alerts_store.insert_alert(
                source="mailbox", external_id="watch-2", severity="high", headline="Dana: another thing",
                body="x", routed_to_person_id=OWNER,
            )
        assert other is not None and _get(other).superseded_by_alert_id is None
    finally:
        current_session.reset(token)
