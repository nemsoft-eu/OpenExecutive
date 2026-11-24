"""A teammate's mail with the Executive is theirs: the chat reaches the
Executive's mailbox only around it (orchestrator/teammate_mail.py)."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from openexecutive.memory import episodic
from openexecutive.orchestrator import teammate_mail as tm
from openexecutive.orchestrator.schedule_tools import current_session
from openexecutive.people import registry as people_registry
from openexecutive.people import store as people_store

OWNER = "olivia@co.example"
TEAMMATE = "ben@co.example"
VENDOR = "billing@northwind.example"
EXEC = "exec@co.example"

SEARCH = "google_workspace__search_gmail_messages"
READ = "google_workspace__get_gmail_message_content"
THREAD = "google_workspace__get_gmail_thread_content"
BATCH = "google_workspace__get_gmail_messages_content_batch"
SEND = "google_workspace__send_gmail_message"
OUTLOOK_LIST = "microsoft_365__list-mail-messages"


@pytest.fixture(autouse=True)
def roster(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    path = tmp_path / "people.db"
    monkeypatch.setattr(people_store, "DB_PATH", path)
    monkeypatch.setattr(episodic, "DB_PATH", path)
    episodic.initialize_db(path)
    people_store.initialize_db()
    people_registry.invalidate()
    owner = people_store.upsert_person(full_name="Olivia Owner", is_principal=True, email=OWNER)
    ben = people_store.upsert_person(full_name="Ben Teammate", role="Ops", email=TEAMMATE)
    monkeypatch.setattr(tm, "_exec_address", lambda: EXEC)
    session = SimpleNamespace(unattended=False, mail_ids_seen=set(), caller_person_id=owner)
    token = current_session.set(session)
    yield SimpleNamespace(owner=owner, ben=ben, session=session)
    current_session.reset(token)
    people_registry.invalidate()


def _reader(person_id: int | None):  # type: ignore[no-untyped-def]
    return patch.object(tm, "_reader_id", lambda _s: person_id)


class FakeMailbox:
    """The gateway's call_tool: canned results by tool name, calls recorded."""

    def __init__(self, results: dict[str, str]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, tool_input: dict[str, Any]) -> str:
        self.calls.append(tool_input)
        return self.results.get(tool_input["name"], "ok")


def _run(mailbox: FakeMailbox, name: str, **arguments: Any) -> str:
    return asyncio.run(tm.guard_call_tool(mailbox)({"name": name, "arguments": arguments}))


def _message(sender: str, to: str, body: str = "Numbers attached.", mid: str = "m1") -> str:
    return (
        f"Message ID: {mid}\nSubject: Q3 numbers\nFrom: Someone <{sender}>\n"
        f"To: {to}\nCc: [not present in Gmail response]\n\n--- BODY ---\n{body}\n"
    )


def _listing(*senders: str) -> str:
    lines = [f"Found {len(senders)} messages matching 'q':", "", "📧 MESSAGES:"]
    for i, sender in enumerate(senders, 1):
        lines += [
            f"  {i}. Message ID: m{i}",
            f"     Subject: Subject {i}",
            f"     From: Person {i} <{sender}>",
            "     Date: Mon, 5 Oct 2026",
            f"     Web Link: https://mail.google.com/m{i}",
            f"     Thread ID: t{i}",
            f"     Thread Link: https://mail.google.com/t{i}",
            "",
        ]
    lines += ["💡 USAGE:", "  • Pass the Message IDs as a list"]
    return "\n".join(lines)


def _thread(*messages: tuple[str, str, str]) -> str:
    text = "Thread ID: t1\nSubject: Taxes\nMessages: 2\n\n"
    for i, (sender, to, body) in enumerate(messages, 1):
        text += f"=== Message {i} ===\nFrom: {sender}\nTo: {to}\n\n{body}\n\n"
    return text


# ── searches ──────────────────────────────────────────────────────────────


def test_a_search_leaves_teammates_out_and_asks_for_headers(roster: SimpleNamespace) -> None:
    mailbox = FakeMailbox({SEARCH: _listing(VENDOR)})
    with _reader(roster.owner):
        _run(mailbox, SEARCH, query="from:ben", user_google_email=EXEC)
    sent = mailbox.calls[0]["arguments"]
    assert sent["include_headers"] is True
    assert sent["query"].startswith("(from:ben) ")
    assert sent["query"] == (
        f"(from:ben) {{(-{{from:{TEAMMATE} to:{TEAMMATE} cc:{TEAMMATE}}}) "
        f"from:{OWNER} to:{OWNER} cc:{OWNER}}}"
    )


def test_a_search_with_no_reader_leaves_every_teammate_out(roster: SimpleNamespace) -> None:
    mailbox = FakeMailbox({SEARCH: _listing(VENDOR)})
    with _reader(None):
        _run(mailbox, SEARCH, query="q")
    query = mailbox.calls[0]["arguments"]["query"]
    assert f"-{{from:{TEAMMATE} to:{TEAMMATE} cc:{TEAMMATE}}}" in query
    assert f"-{{from:{OWNER} to:{OWNER} cc:{OWNER}}}" in query
    assert "{(" not in query


def test_a_background_search_leaves_every_other_teammate_out(roster: SimpleNamespace) -> None:
    # A follow-up reads as the principal but may report to a teammate.
    roster.session.unattended = True
    mailbox = FakeMailbox({SEARCH: _listing(VENDOR)})
    _run(mailbox, SEARCH, query="q")
    query = mailbox.calls[0]["arguments"]["query"]
    assert query == f"(q) -{{from:{TEAMMATE} to:{TEAMMATE} cc:{TEAMMATE}}}"


def test_mail_the_executive_sent_to_the_reader_and_a_teammate_is_found_and_read(
    roster: SimpleNamespace,
) -> None:
    # The Executive emailed the owner and Ben together; the owner asks about it.
    listing = _listing(EXEC)
    message = _message(EXEC, f"{OWNER}, {TEAMMATE}", body="Meeting with Andrew at 2:00.")
    mailbox = FakeMailbox({SEARCH: listing, READ: message})
    with _reader(roster.owner):
        found = _run(mailbox, SEARCH, query="in:sent Andrew")
        read = _run(mailbox, READ, message_id="m1")
    assert "Subject 1" in found and "Left out" not in found
    assert "Meeting with Andrew at 2:00." in read


def test_a_listing_drops_the_teammates_entries_and_remembers_the_rest(roster: SimpleNamespace) -> None:
    mailbox = FakeMailbox({SEARCH: _listing(VENDOR, TEAMMATE)})
    with _reader(roster.owner):
        out = _run(mailbox, SEARCH, query="q")
    assert "Subject 1" in out and "Subject 2" not in out and TEAMMATE not in out
    assert "Left out 1 message involving another teammate" in out
    assert {f"{roster.owner}:m1", f"{roster.owner}:t1"} <= roster.session.mail_ids_seen
    assert not any(i.endswith(":m2") for i in roster.session.mail_ids_seen)


def test_a_listing_without_headers_shows_nothing(roster: SimpleNamespace) -> None:
    bare = "Found 1 messages matching 'q':\n\n📧 MESSAGES:\n  1. Message ID: m9\n     Thread ID: t9\n\n💡 USAGE:\n"
    with _reader(roster.owner):
        out = _run(FakeMailbox({SEARCH: bare}), SEARCH, query="q")
    assert "m9" not in out and not roster.session.mail_ids_seen


# ── reaching mail by id ───────────────────────────────────────────────────


def test_an_id_the_conversation_was_not_shown_is_refused(roster: SimpleNamespace) -> None:
    mailbox = FakeMailbox({READ: _message(VENDOR, EXEC)})
    with _reader(roster.owner):
        out = _run(mailbox, READ, message_id="ben-private")
    assert "hasn't found" in json.loads(out)["error"]
    assert mailbox.calls == []


def test_a_reply_into_an_unseen_thread_is_refused(roster: SimpleNamespace) -> None:
    mailbox = FakeMailbox({})
    with _reader(roster.owner):
        out = _run(mailbox, SEND, to=OWNER, thread_id="ben-thread", body="x")
    assert "error" in json.loads(out) and mailbox.calls == []


def test_quoting_a_thread_with_a_teammates_message_is_refused(roster: SimpleNamespace) -> None:
    tm.allow_mail_ids(roster.session, "t1")
    mailbox = FakeMailbox({THREAD: _thread((VENDOR, EXEC, "invoice"), (TEAMMATE, EXEC, "my bank details"))})
    with _reader(roster.owner):
        out = _run(mailbox, SEND, to=OWNER, thread_id="t1", quote_original=True, body="x")
    assert "can't be quoted" in json.loads(out)["error"]
    assert [c["name"] for c in mailbox.calls] == [THREAD]


def test_the_seen_vendor_message_reads(roster: SimpleNamespace) -> None:
    tm.allow_mail_ids(roster.session, "m1")
    text = _message(VENDOR, EXEC)
    with _reader(roster.owner):
        assert _run(FakeMailbox({READ: text}), READ, message_id="m1") == text


def test_the_inbound_email_a_turn_answers_is_reachable(roster: SimpleNamespace) -> None:
    tm.allow_mail_ids(roster.session, "inbound-m", "inbound-t")
    mailbox = FakeMailbox({SEND: "Email sent! Message ID: r1"})
    with _reader(roster.ben):
        out = _run(mailbox, SEND, to=TEAMMATE, thread_id="inbound-t", body="Got it")
    assert out == "Email sent! Message ID: r1"


# ── reads, message by message ─────────────────────────────────────────────


def test_a_teammates_message_is_withheld_from_the_owner(roster: SimpleNamespace) -> None:
    tm.allow_mail_ids(roster.session, "m1")
    with _reader(roster.owner):
        out = _run(FakeMailbox({READ: _message(TEAMMATE, EXEC)}), READ, message_id="m1")
    assert json.loads(out)["withheld"] is True and "Numbers attached" not in out


def test_the_teammate_reads_their_own(roster: SimpleNamespace) -> None:
    tm.allow_mail_ids(roster.session, "m1")
    text = _message(TEAMMATE, EXEC)
    with _reader(roster.ben):
        assert _run(FakeMailbox({READ: text}), READ, message_id="m1") == text


def test_a_teammate_in_the_body_hides_it(roster: SimpleNamespace) -> None:
    tm.allow_mail_ids(roster.session, "m1")
    text = _message(VENDOR, EXEC, body=f"On Monday Ben <{TEAMMATE}> wrote: my bank details")
    with _reader(roster.owner):
        assert json.loads(_run(FakeMailbox({READ: text}), READ, message_id="m1"))["withheld"] is True


def test_a_thread_keeps_only_the_messages_that_are_not_a_teammates(roster: SimpleNamespace) -> None:
    tm.allow_mail_ids(roster.session, "t1")
    text = _thread((VENDOR, EXEC, "invoice due"), (TEAMMATE, EXEC, "my bank details"))
    with _reader(roster.owner):
        out = _run(FakeMailbox({THREAD: text}), THREAD, thread_id="t1")
    assert "invoice due" in out and "bank details" not in out
    assert "Left out 1 message" in out


def test_a_cc_on_one_message_does_not_open_the_rest(roster: SimpleNamespace) -> None:
    tm.allow_mail_ids(roster.session, "t1")
    text = _thread((TEAMMATE, EXEC, "my bank details"), (EXEC, f"{TEAMMATE}, {OWNER}", "Looping in Olivia"))
    with _reader(roster.owner):
        out = _run(FakeMailbox({THREAD: text}), THREAD, thread_id="t1")
    assert "Looping in Olivia" in out and "bank details" not in out


def test_a_batch_with_one_teammates_message_is_withheld(roster: SimpleNamespace) -> None:
    tm.allow_mail_ids(roster.session, "m1", "m2")
    text = "Retrieved 2 messages:\n\n" + _message(OWNER, EXEC) + "\n---\n\n" + _message(TEAMMATE, EXEC, mid="m2")
    with _reader(roster.owner):
        out = json.loads(_run(FakeMailbox({BATCH: text}), BATCH, message_ids=["m1", "m2"]))
    assert out["withheld"] is True and "one at a time" in out["note"]


# ── Outlook ───────────────────────────────────────────────────────────────


def _item(addr: str, subject: str, mid: str) -> dict[str, Any]:
    return {
        "id": mid,
        "subject": subject,
        "from": {"emailAddress": {"address": addr}},
        "toRecipients": [{"emailAddress": {"address": EXEC}}],
    }


def test_an_outlook_list_selects_the_parties_and_drops_the_teammates_items(roster: SimpleNamespace) -> None:
    text = json.dumps({"value": [_item(VENDOR, "Invoice", "o1"), _item(TEAMMATE, "My taxes", "o2")]})
    mailbox = FakeMailbox({OUTLOOK_LIST: text})
    with _reader(roster.owner):
        out = json.loads(_run(mailbox, OUTLOOK_LIST, select="id,subject,bodyPreview"))
    assert "toRecipients" in mailbox.calls[0]["arguments"]["select"]
    assert [i["subject"] for i in out["value"]] == ["Invoice"]
    assert roster.session.mail_ids_seen == {f"{roster.owner}:o1"}


def test_an_outlook_forward_of_an_unseen_message_is_refused(roster: SimpleNamespace) -> None:
    mailbox = FakeMailbox({})
    with _reader(roster.owner):
        out = _run(mailbox, "microsoft_365__create-forward-draft", messageId="ben-private")
    assert "error" in json.loads(out) and mailbox.calls == []


def test_an_outlook_attachment_download_of_an_unseen_message_is_refused(roster: SimpleNamespace) -> None:
    mailbox = FakeMailbox({})
    with _reader(roster.owner):
        out = _run(mailbox, "microsoft_365__download-bytes", target="/me/messages/ben-private/attachments/a1/$value")
    assert "error" in json.loads(out) and mailbox.calls == []


def test_outlook_search_is_refused(roster: SimpleNamespace) -> None:
    with _reader(roster.owner):
        out = _run(FakeMailbox({}), "microsoft_365__search-query", requests=[])
    assert "not available" in json.loads(out)["error"]


# ── who reads, and when nothing changes ──────────────────────────────────


def test_an_unknown_reader_sees_no_teammates_mail(roster: SimpleNamespace) -> None:
    tm.allow_mail_ids(roster.session, "m1")
    with _reader(None):
        out = _run(FakeMailbox({READ: _message(OWNER, EXEC)}), READ, message_id="m1")
    assert json.loads(out)["withheld"] is True


@pytest.mark.parametrize("authenticated", [True, False])
def test_an_email_turn_reads_as_its_sender_only_when_gmail_vouches(
    roster: SimpleNamespace, authenticated: bool,
) -> None:
    session = SimpleNamespace(
        unattended=False, origin_channel="email", caller_person_id=roster.ben,
        from_web_chat=False, email_sender_authenticated=authenticated,
    )
    assert tm._reader_id(session) == (roster.ben if authenticated else None)


def test_ids_shown_to_one_reader_are_not_the_next_readers(roster: SimpleNamespace) -> None:
    mailbox = FakeMailbox({SEARCH: _listing(TEAMMATE), READ: _message(TEAMMATE, EXEC)})
    with _reader(roster.ben):
        _run(mailbox, SEARCH, query="q")
    with _reader(roster.owner):
        out = _run(mailbox, READ, message_id="m1")
    assert "hasn't found" in json.loads(out)["error"]


def test_an_outlook_forward_of_a_teammates_message_is_refused(roster: SimpleNamespace) -> None:
    tm.allow_mail_ids(roster.session, "o2")
    mailbox = FakeMailbox({"microsoft_365__get-mail-message": json.dumps(_item(TEAMMATE, "My taxes", "o2"))})
    with _reader(roster.owner):
        out = _run(mailbox, "microsoft_365__forward-mail-message", messageId="o2",
                   body={"ToRecipients": [{"emailAddress": {"address": OWNER}}]})
    assert "private to them" in json.loads(out)["error"]
    assert [c["name"] for c in mailbox.calls] == ["microsoft_365__get-mail-message"]


def test_an_unbalanced_query_is_refused(roster: SimpleNamespace) -> None:
    mailbox = FakeMailbox({})
    with _reader(roster.owner):
        out = _run(mailbox, SEARCH, query='subject:"taxes OR (')
    assert "bracket" in json.loads(out)["error"] and mailbox.calls == []


def test_a_trimmed_thread_drops_its_subject(roster: SimpleNamespace) -> None:
    tm.allow_mail_ids(roster.session, "t1")
    text = _thread((VENDOR, EXEC, "invoice due"), (TEAMMATE, EXEC, "my bank details"))
    with _reader(roster.owner):
        out = _run(FakeMailbox({THREAD: text}), THREAD, thread_id="t1")
    assert "Taxes" not in out and "Thread ID: t1" in out


def test_a_teammate_shown_their_own_mail_keeps_the_conversation(roster: SimpleNamespace) -> None:
    roster.session.session_id = "s-ben"
    tm.allow_mail_ids(roster.session, "m1")
    with _reader(roster.ben), patch("openexecutive.memory.session_store.mark_mail_private") as mark:
        _run(FakeMailbox({READ: _message(TEAMMATE, EXEC)}), READ, message_id="m1")
    assert mark.call_args.args == ("s-ben", roster.ben)


def test_the_owner_reading_outside_mail_keeps_nothing_private(roster: SimpleNamespace) -> None:
    roster.session.session_id = "s-owner"
    tm.allow_mail_ids(roster.session, "m1")
    with _reader(roster.owner), patch("openexecutive.memory.session_store.mark_mail_private") as mark:
        _run(FakeMailbox({READ: _message(VENDOR, f"{EXEC}, {OWNER}")}), READ, message_id="m1")
    mark.assert_not_called()


def test_a_background_run_reads_as_the_principal(roster: SimpleNamespace) -> None:
    session = SimpleNamespace(unattended=True, caller_person_id=None)
    assert tm._reader_id(session) == roster.owner


def test_a_teammates_scheduled_run_reads_as_that_teammate(roster: SimpleNamespace) -> None:
    """What it writes is sent to them, so the owner's mail stays out."""
    session = SimpleNamespace(unattended=True, caller_person_id=None, run_for_person_id=roster.ben)
    assert tm._reader_id(session) == roster.ben


def test_a_step_outside_any_session_sees_no_teammates_mail(roster: SimpleNamespace) -> None:
    current_session.set(None)
    text = _message(TEAMMATE, EXEC)
    out = asyncio.run(tm.guard_in_turn(FakeMailbox({READ: text}))(
        {"name": READ, "arguments": {"message_id": "m1"}}
    ))
    assert json.loads(out)["withheld"] is True
    vendor = _message(VENDOR, EXEC)
    out = asyncio.run(tm.guard_in_turn(FakeMailbox({READ: vendor}))(
        {"name": READ, "arguments": {"message_id": "m1"}}
    ))
    assert out == vendor


@pytest.mark.parametrize("arguments", [
    {"body": {"conditions": {"fromAddresses": [{"emailAddress": {"address": TEAMMATE}}]}}},
    {"body": {"actions": {"forwardTo": [{"emailAddress": {"address": OWNER}}]}}},
])
def test_an_outlook_rule_that_could_pass_on_a_teammates_mail_is_refused(
    roster: SimpleNamespace, arguments: dict[str, Any],
) -> None:
    mailbox = FakeMailbox({})
    with _reader(roster.owner):
        out = _run(mailbox, "microsoft_365__create-mail-rule", **arguments)
    assert "rule" in json.loads(out)["error"] and mailbox.calls == []


def test_the_rest_of_the_turn_is_kept_to_the_teammate(roster: SimpleNamespace) -> None:
    roster.session.session_id = "s-ben"
    tm.allow_mail_ids(roster.session, "m1")
    pinned = SimpleNamespace(person_id=roster.ben, touched_mail=False)
    with _reader(roster.ben), patch("openexecutive.memory.session_store.mark_mail_private"), \
            patch("openexecutive.delegation.settings.turn_delegation", lambda _s: pinned):
        _run(FakeMailbox({READ: _message(TEAMMATE, EXEC)}), READ, message_id="m1")
    assert pinned.touched_mail is True


def test_other_tools_pass(roster: SimpleNamespace) -> None:
    text = f"Event with {TEAMMATE}"
    with _reader(roster.owner):
        assert _run(FakeMailbox({"google_workspace__get_events": text}), "google_workspace__get_events") == text


def test_a_roster_failure_refuses(roster: SimpleNamespace) -> None:
    mailbox = FakeMailbox({})
    with _reader(roster.owner), patch.object(people_store, "list_people", side_effect=RuntimeError):
        out = _run(mailbox, READ, message_id="m1")
    assert "roster" in json.loads(out)["error"] and mailbox.calls == []


def test_a_quoted_header_line_in_a_body_does_not_count(roster: SimpleNamespace) -> None:
    tm.allow_mail_ids(roster.session, "m1")
    text = _message(TEAMMATE, EXEC, body=f"Forwarding this:\nTo: {OWNER}\nmy bank details")
    with _reader(roster.owner):
        out = _run(FakeMailbox({READ: text}), READ, message_id="m1")
    assert json.loads(out)["withheld"] is True


def test_a_teammate_named_only_in_an_outside_body_keeps_nothing_private(roster: SimpleNamespace) -> None:
    roster.session.session_id = "s-ben"
    tm.allow_mail_ids(roster.session, "m1")
    text = _message(VENDOR, EXEC, body=f"cc'ing {TEAMMATE} later")
    with _reader(roster.ben), patch("openexecutive.memory.session_store.mark_mail_private") as mark:
        _run(FakeMailbox({READ: text}), READ, message_id="m1")
    mark.assert_not_called()


def test_a_query_with_a_bracket_inside_quotes_is_refused() -> None:
    # A search reads a quoted bracket as text; counted as a bracket it would
    # let the query reopen the group the exclusions are appended into.
    for query in ['"(" ) OR x', 'x OR ( ")"', '-( ( ")" ")"', '{ ( ")" "}"', "({)}", "a)(b"]:
        assert not tm._balanced(query), query
    for query in ['subject:"q3 plan" OR {from:a to:b}', "(invoice OR bill) -label:paid", ""]:
        assert tm._balanced(query), query


# ── mail the Executive sent for someone ──────────────────────────────────

DANA = "dana@acme.example"
LATER = "Sat, 1 Jan 2050 09:00:00 +0000"
EARLIER = "Mon, 5 Oct 2020 09:00:00 +0000"


def _sent_by_exec(mid: str, subject: str, to: str, sender: str = EXEC, date: str = LATER) -> str:
    return (
        f"Message ID: {mid}\nSubject: {subject}\nFrom: Executive <{sender}>\n"
        f"To: {to}\nDate: {date}\n\n--- BODY ---\nHello.\n"
    )


def _send_for(person_id: int, mid: str = "s1", subject: str = "Crates on Monday") -> None:
    with _reader(person_id):
        tm.note_sent(subject=subject, recipients={DANA}, result=f"Email sent! Message ID: {mid}")


def test_mail_sent_for_a_teammate_is_left_out_of_the_owners_search(roster: SimpleNamespace) -> None:
    _send_for(roster.ben, mid="m2")
    mailbox = FakeMailbox({SEARCH: _listing(VENDOR, EXEC)})
    with _reader(roster.owner):
        out = _run(mailbox, SEARCH, query=f"to:{DANA}")
    assert "Subject 1" in out and "Subject 2" not in out
    assert "Left out 1 message" in out
    assert not any(i.endswith(":m2") for i in roster.session.mail_ids_seen)


def test_the_teammate_finds_mail_sent_for_them(roster: SimpleNamespace) -> None:
    _send_for(roster.ben, mid="m2")
    with _reader(roster.ben):
        out = _run(FakeMailbox({SEARCH: _listing(VENDOR, EXEC)}), SEARCH, query=f"to:{DANA}")
    assert "Subject 2" in out and "Left out" not in out


def test_mail_sent_for_the_owner_is_left_out_of_a_teammates_search(roster: SimpleNamespace) -> None:
    _send_for(roster.owner, mid="m1")
    with _reader(roster.ben):
        out = _run(FakeMailbox({SEARCH: _listing(EXEC)}), SEARCH, query="q")
    assert "Subject 1" not in out and "Left out 1 message" in out


def test_a_reply_to_mail_sent_for_a_teammate_is_theirs_too(roster: SimpleNamespace) -> None:
    _send_for(roster.ben, mid="s1", subject="Crates on Monday")
    roster.session.mail_ids_seen.add(f"{roster.owner}:r1")
    reply = _sent_by_exec("r1", "Re: Crates on Monday", EXEC, sender=DANA)
    with _reader(roster.owner):
        out = json.loads(_run(FakeMailbox({READ: reply}), READ, message_id="r1"))
    assert out["withheld"] is True


def test_other_outside_mail_with_another_subject_reads(roster: SimpleNamespace) -> None:
    _send_for(roster.ben, mid="s1", subject="Crates on Monday")
    roster.session.mail_ids_seen.add(f"{roster.owner}:r2")
    other = _sent_by_exec("r2", "Invoice 44", EXEC, sender=DANA)
    with _reader(roster.owner):
        out = _run(FakeMailbox({READ: other}), READ, message_id="r2")
    assert "Invoice 44" in out


def _dated_thread(*messages: tuple[str, str, str, str], thread: str = "t1") -> str:
    text = f"Thread ID: {thread}\nSubject: Taxes\nMessages: 2\n\n"
    for i, (sender, to, date, body) in enumerate(messages, 1):
        text += f"=== Message {i} ===\nFrom: {sender}\nTo: {to}\nDate: {date}\n\n{body}\n\n"
    return text


def test_a_thread_of_mail_sent_for_a_teammate_is_withheld(roster: SimpleNamespace) -> None:
    _send_for(roster.ben, mid="s1", subject="Taxes")
    text = _dated_thread((EXEC, DANA, LATER, "Hello."), (DANA, EXEC, LATER, "Thanks."), thread="s1")
    roster.session.mail_ids_seen.add(f"{roster.owner}:s1")
    with _reader(roster.owner):
        out = json.loads(_run(FakeMailbox({THREAD: text}), THREAD, thread_id="s1"))
    assert out["withheld"] is True


def test_an_outlook_item_sent_for_a_teammate_is_dropped(roster: SimpleNamespace) -> None:
    _send_for(roster.ben, subject="Crates on Monday")
    from datetime import UTC, datetime

    sent = {**_item(EXEC, "Crates on Monday", "o2"), "toRecipients": [{"emailAddress": {"address": DANA}}],
            "receivedDateTime": datetime.now(UTC).isoformat()}
    text = json.dumps({"value": [_item(VENDOR, "Invoice", "o1"), sent]})
    with _reader(roster.owner):
        out = json.loads(_run(FakeMailbox({OUTLOOK_LIST: text}), OUTLOOK_LIST))
    assert [i["id"] for i in out["value"]] == ["o1"]


def test_an_approved_send_is_for_whoever_asked(roster: SimpleNamespace) -> None:
    from openexecutive.orchestrator import levels

    with patch.object(levels, "is_approved", lambda: True), \
            patch.object(levels, "approved_requester", lambda: roster.ben):
        assert tm.sent_for() == roster.ben


def test_a_send_no_one_can_be_told_of_is_not_kept(roster: SimpleNamespace) -> None:
    with _reader(None):
        tm.note_sent(subject="Hi", recipients={DANA}, result="Email sent! Message ID: x9")
    assert tm._sent_rows() == []


def test_a_scheduled_step_finds_its_creators_sends(roster: SimpleNamespace) -> None:
    from openexecutive.workflows import chat_tools

    _send_for(roster.ben, mid="m2")
    current_session.set(None)
    with chat_tools.in_workflow("vendor replies", roster.owner):
        tm.note_sent(subject="Crates on Monday", recipients={DANA}, result="Email sent! Message ID: m1")
        out = _run(FakeMailbox({SEARCH: _listing(EXEC, EXEC)}), SEARCH, query=f"to:{DANA}")
    assert "Subject 1" in out and "Subject 2" not in out


def test_an_old_subject_no_longer_hides_new_mail(roster: SimpleNamespace) -> None:
    _send_for(roster.ben, mid="s1", subject="Invoice")
    with tm._open() as conn:
        conn.execute(f"UPDATE {tm.SENT_TABLE} SET at = datetime('now', '-40 days')")
    roster.session.mail_ids_seen.add(f"{roster.owner}:r3")
    later = _sent_by_exec("r3", "Invoice", EXEC, sender=DANA)
    with _reader(roster.owner):
        out = _run(FakeMailbox({READ: later}), READ, message_id="r3")
    assert "Invoice" in out and "withheld" not in out


def test_the_gateway_keeps_whom_a_gmail_send_was_for(roster: SimpleNamespace) -> None:
    from openexecutive.orchestrator import mcp_gateway

    with _reader(roster.ben):
        mcp_gateway._note_sent_for(SEND, SEND, {"to": DANA, "subject": "Re: Crates", "body": "Hi"},
                                   "Email sent! Message ID: g7")
    [row] = tm._sent_rows()
    assert (row.person_id, row.message_id, row.subject, row.recipients) == (roster.ben, "g7", "crates", {DANA})


def test_an_outlook_reply_is_kept_for_whoever_asked(roster: SimpleNamespace) -> None:
    from datetime import UTC, datetime

    from openexecutive.orchestrator import mcp_gateway

    calls: list[dict] = []

    async def fetch(session, tool, arguments, discover=None):  # noqa: ANN001, ANN202
        calls.append(arguments)
        return {"id": "o9", "subject": "Crates on Monday", "from": {"emailAddress": {"address": DANA}}}

    reply = "microsoft_365__reply_mail_message"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mcp_gateway, "_fetch_m365_json", fetch)
        sent_by_id = asyncio.run(mcp_gateway._m365_sent_by_id(None, {"messageId": "o9"}))
    assert calls == [{"messageId": "o9", "select": mcp_gateway._M365_MESSAGE_LOOKUP_SELECT}]
    with _reader(roster.ben):
        mcp_gateway._note_sent_for(reply, reply, {"messageId": "o9", "comment": "Yes"}, "Sent.", sent_by_id)
    [row] = tm._sent_rows()
    assert (row.subject, row.recipients, row.started) == ("crates on monday", {DANA}, False)
    sent = {**_item(EXEC, "RE: Crates on Monday", "o10"), "toRecipients": [{"emailAddress": {"address": DANA}}],
            "receivedDateTime": datetime.now(UTC).isoformat()}
    with _reader(roster.owner):
        out = json.loads(_run(FakeMailbox({OUTLOOK_LIST: json.dumps({"value": [sent]})}), OUTLOOK_LIST))
    assert out["value"] == []


def test_an_outlook_forward_is_kept_with_its_new_recipients(roster: SimpleNamespace) -> None:
    from openexecutive.orchestrator import mcp_gateway

    forward = "microsoft_365__forward_mail_message"
    args = {"messageId": "o9", "body": {"comment": "FYI", "toRecipients": [{"emailAddress": {"address": DANA}}]}}
    with _reader(roster.ben):
        mcp_gateway._note_sent_for(forward, forward, args, "Sent.", {"id": "o9", "subject": "Price list"})
    [row] = tm._sent_rows()
    assert (row.subject, row.recipients, row.started) == ("price list", {DANA}, True)


def test_the_readers_own_send_with_the_same_subject_wins(roster: SimpleNamespace) -> None:
    _send_for(roster.ben, mid="s1", subject="Order for next week")
    _send_for(roster.owner, mid="s2", subject="Order for next week")
    roster.session.mail_ids_seen.add(f"{roster.owner}:r4")
    reply = _sent_by_exec("r4", "Re: Order for next week", EXEC, sender=DANA)
    with _reader(roster.owner):
        out = _run(FakeMailbox({READ: reply}), READ, message_id="r4")
    assert "withheld" not in out


def test_a_send_for_someone_no_longer_in_people_is_nobodys(roster: SimpleNamespace) -> None:
    gone = roster.ben + 1000
    _send_for(roster.ben, mid="m1")
    with tm._open() as conn:
        conn.execute(f"UPDATE {tm.SENT_TABLE} SET person_id = ?", (gone,))
    with _reader(roster.owner):
        out = _run(FakeMailbox({SEARCH: _listing(EXEC)}), SEARCH, query="q")
    assert "Subject 1" in out


def test_an_old_send_keeps_its_id(roster: SimpleNamespace) -> None:
    _send_for(roster.ben, mid="m1", subject="Invoice")
    with tm._open() as conn:
        conn.execute(f"UPDATE {tm.SENT_TABLE} SET at = datetime('now', '-400 days')")
    with _reader(roster.owner):
        out = _run(FakeMailbox({SEARCH: _listing(EXEC)}), SEARCH, query="q")
    assert "Subject 1" not in out


def test_a_teammates_reply_into_a_shared_thread_hides_only_what_came_after(roster: SimpleNamespace) -> None:
    # Dana wrote first; Ben then had the Executive reply in that thread.
    with _reader(roster.ben):
        tm.note_sent(subject="Re: Taxes", recipients={DANA}, thread_id="t1", result="Email sent! Message ID: s9")
    text = _dated_thread(
        (DANA, EXEC, EARLIER, "Original question."), (EXEC, DANA, _now(), "Ben's answer."),
        (DANA, EXEC, LATER, "Dana writes again."),
    )
    roster.session.mail_ids_seen.add(f"{roster.owner}:t1")
    with _reader(roster.owner):
        out = _run(FakeMailbox({THREAD: text}), THREAD, thread_id="t1")
    assert "Original question." in out and "Dana writes again." in out
    assert "Ben's answer." not in out


def _now() -> str:
    from datetime import UTC, datetime
    from email.utils import format_datetime

    return format_datetime(datetime.now(UTC))


def test_an_undated_reply_in_a_thread_the_send_started_is_left_out(roster: SimpleNamespace) -> None:
    # The second entry is a reply in the thread Ben's send started (t2).
    _send_for(roster.ben, mid="t2")
    with _reader(roster.owner):
        out = _run(FakeMailbox({SEARCH: _listing(VENDOR, DANA)}), SEARCH, query="q")
    assert "Subject 1" in out and "Subject 2" not in out
    with _reader(roster.ben):
        assert "Subject 2" in _run(FakeMailbox({SEARCH: _listing(VENDOR, DANA)}), SEARCH, query="q")


def test_a_send_without_an_id_is_left_out_of_a_listing_by_subject_and_time(
    roster: SimpleNamespace,
) -> None:
    # Some mailboxes answer a send with no id; a listing shows only From.
    with _reader(roster.owner):
        tm.note_sent(subject="Crates on Monday", recipients={DANA}, result="Email sent! Message ID: ")
    listing = _listing(EXEC).replace("Subject 1", "Crates on Monday").replace("Mon, 5 Oct 2026", _now())
    with _reader(roster.ben):
        out = _run(FakeMailbox({SEARCH: listing}), SEARCH, query=f"to:{DANA}")
    assert "Crates on Monday" not in out and "Left out 1 message" in out
    with _reader(roster.owner):
        assert "Crates on Monday" in _run(FakeMailbox({SEARCH: listing}), SEARCH, query=f"to:{DANA}")


def test_a_message_that_shows_its_recipients_must_share_one_with_the_send(
    roster: SimpleNamespace,
) -> None:
    # Subject and time alone decide only where no recipient is shown.
    with _reader(roster.owner):
        tm.note_sent(subject="Crates on Monday", recipients={DANA}, result="Email sent! Message ID: ")
    other = _sent_by_exec("x1", "Crates on Monday", VENDOR, date=_now())
    tm.allow_mail_ids(roster.session, "x1")
    with _reader(roster.ben):
        out = _run(FakeMailbox({READ: other}), READ, message_id="x1")
    assert "Crates on Monday" in out and "withheld" not in out


@pytest.mark.parametrize("case", ["other subject", "outside the send's time", "send has an id"])
def test_a_listing_entry_matches_a_send_only_by_its_subject_its_time_and_no_id(
    roster: SimpleNamespace, case: str
) -> None:
    result = "Email sent! Message ID: s9" if case == "send has an id" else "Email sent! Message ID: "
    with _reader(roster.owner):
        tm.note_sent(subject="Crates on Monday", recipients={DANA}, result=result)
    subject = "Pallets on Tuesday" if case == "other subject" else "Crates on Monday"
    if case == "outside the send's time":
        with tm._open() as conn:
            conn.execute(f"UPDATE {tm.SENT_TABLE} SET at = datetime('now', '-1 hour')")
    listing = _listing(EXEC).replace("Subject 1", subject).replace("Mon, 5 Oct 2026", _now())
    with _reader(roster.ben):
        out = _run(FakeMailbox({SEARCH: listing}), SEARCH, query=f"to:{DANA}")
    assert subject in out and "Left out" not in out


def test_a_new_message_with_the_same_subject_is_no_reply(roster: SimpleNamespace) -> None:
    _send_for(roster.ben, mid="s1", subject="Invoice")
    roster.session.mail_ids_seen.update({f"{roster.owner}:r6", f"{roster.owner}:r7"})
    fresh = _sent_by_exec("r6", "Invoice", EXEC, sender=DANA)
    reply = _sent_by_exec("r7", "Re: Invoice", EXEC, sender=DANA)
    with _reader(roster.owner):
        assert "withheld" not in _run(FakeMailbox({READ: fresh}), READ, message_id="r6")
        assert "withheld" in _run(FakeMailbox({READ: reply}), READ, message_id="r7")


def test_a_gmail_send_finds_replies_by_its_thread_not_its_subject(roster: SimpleNamespace) -> None:
    # Ben's "Meeting" started thread s1; Dana's later "Re: Meeting" is in
    # another thread, the principal's own.
    _send_for(roster.ben, mid="s1", subject="Meeting")
    roster.session.mail_ids_seen.add(f"{roster.owner}:r9")
    other = _sent_by_exec("r9", "Re: Meeting", EXEC, sender=DANA) + "Thread ID: t9\n"
    with _reader(roster.owner):
        assert "withheld" not in _run(FakeMailbox({READ: other}), READ, message_id="r9")


def test_the_message_a_quick_reply_answered_stays_shared(roster: SimpleNamespace) -> None:
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    with _reader(roster.ben):
        tm.note_sent(subject="Re: Taxes", recipients={DANA}, thread_id="t1", result="Email sent! Message ID: s9")
    a_minute_ago = format_datetime(datetime.now(UTC) - timedelta(minutes=1))
    text = _dated_thread((DANA, EXEC, a_minute_ago, "Original question."), (EXEC, DANA, _now(), "Ben's answer."))
    roster.session.mail_ids_seen.add(f"{roster.owner}:t1")
    with _reader(roster.owner):
        out = _run(FakeMailbox({THREAD: text}), THREAD, thread_id="t1")
    assert "Original question." in out and "Ben's answer." not in out


def test_a_teammates_reply_into_the_readers_own_thread_stays_theirs(roster: SimpleNamespace) -> None:
    from datetime import UTC, datetime, timedelta
    from email.utils import format_datetime

    _send_for(roster.owner, mid="t1", subject="Taxes")
    with tm._open() as conn:
        conn.execute(f"UPDATE {tm.SENT_TABLE} SET at = datetime('now', '-1 hour')")
    an_hour_ago = format_datetime(datetime.now(UTC) - timedelta(hours=1))
    with _reader(roster.ben):
        tm.note_sent(subject="Re: Taxes", recipients={DANA}, thread_id="t1", result="Email sent! Message ID: s9")
    roster.session.mail_ids_seen.add(f"{roster.owner}:s9")
    bens = _sent_by_exec("s9", "Re: Taxes", DANA) + "Thread ID: t1\n"
    with _reader(roster.owner):
        assert "withheld" in _run(FakeMailbox({READ: bens}), READ, message_id="s9")
    text = _dated_thread((EXEC, DANA, an_hour_ago, "Olivia's note."), (EXEC, DANA, _now(), "Ben's answer."))
    roster.session.mail_ids_seen.add(f"{roster.owner}:t1")
    with _reader(roster.owner):
        out = _run(FakeMailbox({THREAD: text}), THREAD, thread_id="t1")
    assert "Olivia's note." in out and "Ben's answer." not in out


def test_a_reply_sent_into_a_thread_takes_only_itself(roster: SimpleNamespace) -> None:
    with _reader(roster.ben):
        tm.note_sent(subject="Re: Taxes", recipients={DANA}, thread_id="t1", result="Email sent! Message ID: s9")
    [row] = tm._sent_rows()
    assert (row.message_id, row.thread_id, row.started) == ("s9", "", False)
    roster.session.mail_ids_seen.add(f"{roster.owner}:r8")
    reply = _sent_by_exec("r8", "Re: Taxes", EXEC, sender=DANA)
    with _reader(roster.owner):
        assert "withheld" not in _run(FakeMailbox({READ: reply}), READ, message_id="r8")


def test_the_executives_own_address_is_not_kept_as_a_recipient(
    roster: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _reader(roster.ben):
        tm.note_sent(subject="Weekly update", recipients={DANA, EXEC}, result="Email sent! Message ID: w1")
    [row] = tm._sent_rows()
    assert row.recipients == {DANA}


def test_an_outlook_send_is_found_by_its_subject_even_with_a_narrow_select(roster: SimpleNamespace) -> None:
    from datetime import UTC, datetime

    with _reader(roster.ben):
        tm.note_sent(subject="Crates on Monday", recipients={DANA}, result="")
    with tm._open() as conn:
        conn.execute(f"UPDATE {tm.SENT_TABLE} SET at = datetime('now', '-90 days')")
    [row] = tm._sent_rows()
    sent = {**_item(EXEC, "Crates on Monday", "o2"), "toRecipients": [{"emailAddress": {"address": DANA}}],
            "receivedDateTime": row.at.astimezone(UTC).isoformat()}
    assert datetime.fromisoformat(sent["receivedDateTime"]) == row.at
    mailbox = FakeMailbox({OUTLOOK_LIST: json.dumps({"value": [sent]})})
    with _reader(roster.owner):
        out = json.loads(_run(mailbox, OUTLOOK_LIST, select="id,bodyPreview"))
    assert "subject" in mailbox.calls[0]["arguments"]["select"]
    assert out["value"] == []


def test_an_outside_senders_email_turn_reads_the_principals_thread(roster: SimpleNamespace) -> None:
    _send_for(roster.owner, mid="s1", subject="Crates on Monday")
    roster.session.origin_channel = "email"
    tm.allow_mail_ids(roster.session, "r5")
    reply = _sent_by_exec("r5", "Re: Crates on Monday", EXEC, sender=DANA)
    with _reader(None):
        out = _run(FakeMailbox({READ: reply}), READ, message_id="r5")
    assert "withheld" not in out


def test_a_send_on_an_outside_senders_email_turn_is_the_principals(roster: SimpleNamespace) -> None:
    roster.session.origin_channel = "email"
    with _reader(None):
        tm.note_sent(subject="Re: Crates", recipients={DANA}, result="Email sent! Message ID: e1")
    [row] = tm._sent_rows()
    assert row.person_id == roster.owner


def test_an_outside_senders_email_turn_still_leaves_a_teammates_send_out(roster: SimpleNamespace) -> None:
    _send_for(roster.ben, mid="s1", subject="Crates on Monday")
    roster.session.origin_channel = "email"
    tm.allow_mail_ids(roster.session, "s1")
    sent = _sent_by_exec("s1", "Crates on Monday", DANA)
    with _reader(None):
        out = json.loads(_run(FakeMailbox({READ: sent}), READ, message_id="s1"))
    assert out["withheld"] is True
