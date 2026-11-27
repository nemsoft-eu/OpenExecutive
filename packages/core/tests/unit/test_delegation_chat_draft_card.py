"""The card under a chat draft (delegation.chat_draft_card): it shows the
draft as it is in the person's Drafts and changes it there in place, by hand
or by asking, and never touches a draft they changed in their mailbox."""
from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from openexecutive.api.caller import Caller
from openexecutive.delegation import chat_draft_card, drafts
from openexecutive.delegation.reply_send import SendRefused, send_chat_draft
from openexecutive.orchestrator.action_chips import _draft_fields, drop_replaced

from . import test_delegation_chat_draft_send as _send_tests
from .test_delegation_inbox import NOW, FakeInbox

db = _send_tests.db
owner = _send_tests.owner
local_login = _send_tests.local_login
_chat_draft = _send_tests._chat_draft

LOCAL = Caller("open", "")
LATER = NOW + timedelta(minutes=5)


def _state(owner: Any, row: int, mailbox: FakeInbox) -> dict[str, Any]:
    return asyncio.run(chat_draft_card.state(owner, row, caller=LOCAL, gmail=mailbox, now=LATER))


def _refused(coro: Any) -> SendRefused:
    with pytest.raises(SendRefused) as err:
        asyncio.run(coro)
    return err.value


def test_the_card_shows_the_draft_as_it_is_in_their_drafts(db: Path, owner: Any) -> None:
    mailbox, row = _chat_draft(owner)
    shown = _state(owner, row, mailbox)
    assert shown["status"] == chat_draft_card.OPEN
    assert shown["body"] == "Friday at 10 works.\n\nOlivia"
    assert shown["subject"] == "Friday"


def test_their_own_words_replace_the_draft_and_send_sends_them(db: Path, owner: Any) -> None:
    mailbox, row = _chat_draft(owner)
    saved = asyncio.run(chat_draft_card.set_text(
        owner, row, "Friday at 11 is better.\n\nOlivia", caller=LOCAL, gmail=mailbox, now=LATER,
    ))
    assert saved["body"] == "Friday at 11 is better.\n\nOlivia"
    assert mailbox.drafts["d1"].message.text == "Friday at 11 is better.\n\nOlivia"
    # The row holds the new version, so Send under the chat still sends it.
    assert drafts.get_row(row).message_id == mailbox.drafts["d1"].message.id
    assert _state(owner, row, mailbox)["status"] == chat_draft_card.OPEN
    asyncio.run(send_chat_draft(owner, row, caller=LOCAL, gmail=mailbox, now=LATER))
    assert mailbox.sent == ["d1"]


def test_asking_rewrites_it_in_place(db: Path, owner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    mailbox, row = _chat_draft(owner)
    seen: dict[str, Any] = {}

    async def rewrite(person: Any, payload: dict[str, Any], answers: list[Any], **kw: Any) -> Any:
        seen.update(payload=payload, change=kw.get("change"))
        return (
            "Friday at 10 works. Thanks so much!\n\nOlivia",
            {"conversation": [{"kind": "asked", "text": kw["change"]},
                              {"kind": "note", "text": "I made it warmer."}]},
            [],
        )

    monkeypatch.setattr("openexecutive.delegation.answers.rewrite", rewrite)
    saved, new, note = asyncio.run(chat_draft_card.change(
        owner, row, "Make it warmer.", caller=LOCAL, gmail=mailbox, now=LATER,
    ))
    assert seen["change"] == "Make it warmer."
    assert seen["payload"]["draft_body"] == "Friday at 10 works.\n\nOlivia"
    assert note == "I made it warmer."
    assert new == ["Thanks so much!"]
    assert mailbox.drafts["d1"].message.text == saved["body"]
    assert mailbox.sent == []


def test_a_draft_changed_in_the_mailbox_is_never_overwritten(db: Path, owner: Any) -> None:
    mailbox, row = _chat_draft(owner)
    mailbox.edit("d1")
    assert _state(owner, row, mailbox)["status"] == chat_draft_card.CHANGED
    refused = _refused(chat_draft_card.set_text(owner, row, "Mine", caller=LOCAL, gmail=mailbox, now=LATER))
    assert refused.code == "draft_changed"
    assert "update:d1" not in mailbox.calls


def test_only_its_own_signed_person_changes_it(db: Path, owner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    mailbox, row = _chat_draft(owner)
    monkeypatch.delenv("OE_LOCAL_LOGIN")
    refused = _refused(chat_draft_card.set_text(owner, row, "Mine", caller=LOCAL, gmail=mailbox, now=LATER))
    assert refused.code == "caller_signing_required"
    # Reading it needs the same provable caller: it is their own mail.
    assert _refused(chat_draft_card.state(owner, row, caller=LOCAL, gmail=mailbox)).code == "caller_signing_required"
    assert _refused(chat_draft_card.state(owner, row + 99, caller=LOCAL, gmail=mailbox)).code == "draft_not_found"
    assert "update:d1" not in mailbox.calls


def test_a_replaced_version_folds(db: Path, owner: Any) -> None:
    mailbox, first = _chat_draft(owner)
    made = asyncio.run(mailbox.create_draft(mailbox.specs[0]))
    conn_rows = [
        drafts.record(owner.id, source=drafts.SOURCE_CHAT, thread_id=None, draft_id=d, message_id=m,
                      now=NOW, session_id="s1")
        for d, m in (("dx", "mx"), (made.draft_id, made.message_id))
    ]
    drafts.mark_gone(conn_rows[0], LATER)
    # Gone, with a later draft in the same conversation that didn't replace
    # it (another email): just gone.
    assert _state(owner, conn_rows[0], mailbox)["status"] == chat_draft_card.GONE
    # Retired for the newer version: replaced.
    drafts.mark_replaced(conn_rows[0], conn_rows[1])
    assert _state(owner, conn_rows[0], mailbox)["status"] == chat_draft_card.REPLACED
    drafts.mark_gone(first, LATER)
    assert _state(owner, first, mailbox)["status"] == chat_draft_card.GONE


def test_a_draft_written_again_in_one_reply_keeps_one_card() -> None:
    first = {"type": "action_taken", "tool": "ghostwrite_email", **_draft_fields({"draft_row": 68})}
    other = {"type": "action_taken", "tool": "search_my_email"}
    again = {"type": "action_taken", "tool": "ghostwrite_email",
             **_draft_fields({"draft_row": 69, "replaced_rows": [68, 99, "x"]})}
    assert again["draft"]["replaces"] == [68]
    assert drop_replaced([first, other, again]) == [other, again]
    assert drop_replaced([first, other]) == [first, other]


def test_the_routes_read_and_change_the_callers_own_draft(owner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from openexecutive.api.routes import delegation as route
    from openexecutive.delegation import gmail as gmail_module

    mailbox, row = _chat_draft(owner)
    monkeypatch.setattr(gmail_module, "gmail_for", lambda email: mailbox)
    app = FastAPI()
    app.include_router(route.router)
    client = TestClient(app)
    headers = {"x-caller-email": _send_tests.OWNER}

    resp = client.get(f"/delegation/drafts/{row}", headers=headers)
    assert resp.status_code == 200 and resp.json()["status"] == "open"
    assert resp.json()["from_executive"] is False and resp.json()["executive_name"]
    assert resp.json()["body"] == "Friday at 10 works.\n\nOlivia"
    assert client.get(f"/delegation/drafts/{row + 99}", headers=headers).status_code == 404
    resp = client.put(f"/delegation/drafts/{row}/text", json={"text": "Friday at 11.\n\nOlivia"}, headers=headers)
    assert resp.status_code == 200 and resp.json()["body"] == "Friday at 11.\n\nOlivia"
    assert client.post(f"/delegation/drafts/{row}/change", json={"text": ""}, headers=headers).status_code == 422
    # What to change: up to 500 characters.
    assert client.post(f"/delegation/drafts/{row}/change", json={"text": "x" * 501}, headers=headers).status_code == 422

    async def rewrite(person: Any, payload: dict[str, Any], answers: list[Any], **kw: Any) -> Any:
        return "Friday at 11 works.\n\nOlivia", {"conversation": [{"kind": "note", "text": "Done."}]}, []

    monkeypatch.setattr("openexecutive.delegation.answers.rewrite", rewrite)
    resp = client.post(f"/delegation/drafts/{row}/change", json={"text": "x" * 500}, headers=headers)
    assert resp.status_code == 200 and resp.json()["note"] == "Done."
    # Someone else's draft reads the same as none at all.
    theirs = drafts.record(owner.id + 1, source=drafts.SOURCE_CHAT, thread_id=None, draft_id="dz",
                           message_id="mz", now=NOW)
    async def as_executive(model: str, system: str, turn: str) -> dict[str, Any]:
        return {"body": "Hi Dana,\n\nOlivia asked me to say Friday at 11 works."}

    monkeypatch.setattr("openexecutive.delegation.answers._call_model", as_executive)
    resp = client.put(f"/delegation/drafts/{row}/from", json={"executive": True}, headers=headers)
    assert resp.status_code == 200 and resp.json()["from_executive"] is True
    assert resp.json()["note"]
    assert client.put(f"/delegation/drafts/{row}/from", json={}, headers=headers).status_code == 422
    for resp in (
        client.get(f"/delegation/drafts/{theirs}", headers=headers),
        client.put(f"/delegation/drafts/{theirs}/from", json={"executive": True}, headers=headers),
        client.put(f"/delegation/drafts/{theirs}/text", json={"text": "Mine"}, headers=headers),
        client.post(f"/delegation/drafts/{theirs}/change", json={"text": "Shorter"}, headers=headers),
    ):
        assert resp.status_code == 404 and resp.json()["detail"]["code"] == "draft_not_found"
    mailbox.edit("d1")
    resp = client.put(f"/delegation/drafts/{row}/text", json={"text": "Mine"}, headers=headers)
    assert resp.status_code == 409 and resp.json()["detail"]["code"] == "draft_changed"
    assert resp.json()["detail"]["state"] == {"status": "changed"}
    # Without signed sign-ins the API can't tie the caller to the owner: no
    # read, no change.
    monkeypatch.delenv("OE_LOCAL_LOGIN")
    for resp in (
        client.get(f"/delegation/drafts/{row}", headers=headers),
        client.put(f"/delegation/drafts/{row}/text", json={"text": "Mine"}, headers=headers),
        client.post(f"/delegation/drafts/{row}/change", json={"text": "Shorter"}, headers=headers),
    ):
        assert resp.status_code == 409 and resp.json()["detail"]["code"] == "caller_signing_required"
    assert mailbox.sent == []


def test_send_waits_while_the_card_is_changing_it(db: Path, owner: Any) -> None:
    mailbox, row = _chat_draft(owner)
    chat_draft_card._CHANGING.add(row)
    try:
        refused = _refused(send_chat_draft(owner, row, caller=LOCAL, gmail=mailbox, now=LATER))
    finally:
        chat_draft_card._CHANGING.discard(row)
    assert refused.code == "busy" and mailbox.sent == []


def test_a_draft_with_files_is_changed_in_the_mailbox(db: Path, owner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from openexecutive.delegation.gmail import DraftChanged

    mailbox, row = _chat_draft(owner)
    before = drafts.get_row(row).message_id

    async def not_plain(*_a: Any, **_k: Any) -> str:
        raise DraftChanged("not_plain")

    monkeypatch.setattr(mailbox, "update_draft_text", not_plain)
    refused = _refused(chat_draft_card.set_text(owner, row, "Mine", caller=LOCAL, gmail=mailbox, now=LATER))
    assert refused.code == "draft_not_plain"
    # The row keeps the version Send compares with.
    assert drafts.get_row(row).message_id == before
    assert not drafts.set_version(row, "not-the-version", "x")


def test_an_edit_the_mailbox_kept_the_old_version_of_is_never_called_saved(
    db: Path, owner: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mailbox, row = _chat_draft(owner)
    before = drafts.get_row(row).message_id

    async def unchanged(*_a: Any, **_k: Any) -> str:
        return before

    monkeypatch.setattr(mailbox, "update_draft_text", unchanged)
    refused = _refused(chat_draft_card.set_text(owner, row, "Mine", caller=LOCAL, gmail=mailbox, now=LATER))
    assert refused.code == "gmail_error"
    assert drafts.get_row(row).message_id == before


def test_delete_removes_it_from_their_drafts(db: Path, owner: Any) -> None:
    mailbox, row = _chat_draft(owner)
    gone = asyncio.run(chat_draft_card.discard(owner, row, caller=LOCAL, gmail=mailbox, now=LATER))
    assert gone == {"status": chat_draft_card.GONE}
    assert "d1" not in mailbox.drafts and mailbox.sent == []
    # Marked gone, so Home's row for it goes and Send refuses.
    assert drafts.get_row(row).gone_at
    assert _refused(send_chat_draft(owner, row, caller=LOCAL, gmail=mailbox, now=LATER)).code == "draft_gone"
    # Again: already gone, nothing more to do.
    again = asyncio.run(chat_draft_card.discard(owner, row, caller=LOCAL, gmail=mailbox, now=LATER))
    assert again == {"status": chat_draft_card.GONE}


def test_delete_leaves_a_draft_changed_in_the_mailbox(db: Path, owner: Any) -> None:
    mailbox, row = _chat_draft(owner)
    mailbox.edit("d1")
    refused = _refused(chat_draft_card.discard(owner, row, caller=LOCAL, gmail=mailbox, now=LATER))
    assert refused.code == "draft_changed"
    assert "d1" in mailbox.drafts and not drafts.get_row(row).gone_at


def test_delete_never_reports_a_draft_it_left_alone_as_deleted(
    db: Path, owner: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mailbox, row = _chat_draft(owner)

    async def left_alone(draft_id: str) -> bool:
        # What Outlook's delete_draft does when the draft changed between the
        # read and the delete: leaves it and returns False.
        mailbox.edit(draft_id)
        return False

    monkeypatch.setattr(mailbox, "delete_draft", left_alone)
    refused = _refused(chat_draft_card.discard(owner, row, caller=LOCAL, gmail=mailbox, now=LATER))
    assert refused.code == "draft_changed"
    assert "d1" in mailbox.drafts and not drafts.get_row(row).gone_at

def test_the_delete_route_deletes_only_the_callers_own_draft(owner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from openexecutive.api.routes import delegation as route
    from openexecutive.delegation import gmail as gmail_module

    mailbox, row = _chat_draft(owner)
    monkeypatch.setattr(gmail_module, "gmail_for", lambda email: mailbox)
    app = FastAPI()
    app.include_router(route.router)
    client = TestClient(app)
    headers = {"x-caller-email": _send_tests.OWNER}
    theirs = drafts.record(owner.id + 1, source=drafts.SOURCE_CHAT, thread_id=None, draft_id="dz",
                           message_id="mz", now=NOW)
    resp = client.delete(f"/delegation/drafts/{theirs}", headers=headers)
    assert resp.status_code == 404 and resp.json()["detail"]["code"] == "draft_not_found"
    monkeypatch.delenv("OE_LOCAL_LOGIN")
    resp = client.delete(f"/delegation/drafts/{row}", headers=headers)
    assert resp.status_code == 409 and resp.json()["detail"]["code"] == "caller_signing_required"
    assert "d1" in mailbox.drafts
    monkeypatch.setenv("OE_LOCAL_LOGIN", "1")
    resp = client.delete(f"/delegation/drafts/{row}", headers=headers)
    assert resp.status_code == 200 and resp.json()["status"] == "gone"
    assert "d1" not in mailbox.drafts


def test_the_card_names_the_files_on_the_draft(db: Path, owner: Any) -> None:
    from openexecutive.delegation.gmail import MailAttachment

    mailbox, row = _chat_draft(owner)
    assert _state(owner, row, mailbox)["files"] == []
    mailbox.drafts["d1"].message.attachments = [
        MailAttachment(index=1, name="2026 Tax - Lot 1.pdf"), MailAttachment(index=2, name="2026 Tax - Lot 2.pdf"),
    ]
    assert _state(owner, row, mailbox)["files"] == ["2026 Tax - Lot 1.pdf", "2026 Tax - Lot 2.pdf"]


def test_a_question_leaves_the_draft_in_the_mailbox_as_it_is(
    db: Path, owner: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mailbox, row = _chat_draft(owner)

    async def answered(person: Any, payload: dict[str, Any], answers: list[Any], **kw: Any) -> Any:
        turns = [{"kind": "asked", "text": kw["change"]}, {"kind": "note", "text": "You offered Friday at 10."}]
        return payload["draft_body"], {"conversation": turns}, []

    monkeypatch.setattr("openexecutive.delegation.answers.rewrite", answered)
    before = mailbox.drafts["d1"].message.text
    saved, new, note = asyncio.run(chat_draft_card.change(
        owner, row, "What time did I offer?", caller=LOCAL, gmail=mailbox, now=LATER,
    ))
    assert note == "You offered Friday at 10." and new == []
    assert saved["body"] == before and "update:d1" not in mailbox.calls


def test_an_ask_that_needs_looking_up_is_refused_and_leaves_the_draft(
    db: Path, owner: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.delegation.answers import NeedsWork

    mailbox, row = _chat_draft(owner)

    async def needs_tools(*_a: Any, **kw: Any) -> Any:
        raise NeedsWork(kw["change"], question=True)

    monkeypatch.setattr("openexecutive.delegation.answers.rewrite", needs_tools)
    refused = _refused(chat_draft_card.change(
        owner, row, "What did we quote them last quarter?", caller=LOCAL, gmail=mailbox, now=LATER,
    ))
    assert (refused.status, refused.code) == (409, "needs_work")
    assert "Ask me in the chat" in refused.message
    assert "update:d1" not in mailbox.calls
    assert row not in chat_draft_card._CHANGING


def test_send_it_on_a_chat_draft_leaves_its_words(db: Path, owner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from openexecutive.delegation import answers

    async def call(model: str, system: str, turn: str) -> dict[str, Any]:
        main = turn.split("<draft>\n", 1)[1].split("\n</draft>", 1)[0]
        return {"body": main, "send_now": True, "note": "It's ready."}

    monkeypatch.setattr(answers, "_call_model", call)
    mailbox, row = _chat_draft(owner)
    before = mailbox.drafts["d1"].message.text
    # A card with no send confirm of its own: "send it" changes nothing.
    saved, new, _ = asyncio.run(chat_draft_card.change(
        owner, row, "looks good, send it", caller=LOCAL, gmail=mailbox, now=LATER,
    ))
    assert new == [] and mailbox.drafts["d1"].message.text.strip() == before.strip()
    assert mailbox.sent == []


# ── who it goes from ──────────────────────────────────────────────────────────


@pytest.fixture
def as_sender(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The model's rewrite for the new sender: which prompt it was given."""
    from openexecutive.delegation import answers, send_from

    seen: list[str] = []

    async def fake(model: str, system: str, turn: str) -> dict[str, Any]:
        seen.append(system)
        if system == send_from.AS_EXECUTIVE_PROMPT:
            return {"body": "Hi Dana,\n\nOlivia asked me to say Friday at 10 works."}
        return {"body": "Hi Dana,\n\nFriday at 10 works."}

    monkeypatch.setattr(answers, "_call_model", fake)
    return seen


@pytest.fixture
def exec_sends(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    from openexecutive.orchestrator import take_the_lead

    out: list[dict[str, Any]] = []

    async def fake(payload: dict[str, Any], *, by_principal: bool, own_card: bool = False) -> str:
        out.append({**payload, "own_card": own_card})
        return '{"status": "sent"}'

    monkeypatch.setattr(take_the_lead, "carry_out", fake)
    return out


def _switch(owner: Any, row: int, mailbox: FakeInbox, executive: bool) -> dict[str, Any]:
    return asyncio.run(chat_draft_card.switch(
        owner, row, executive=executive, caller=LOCAL, gmail=mailbox, now=LATER,
    ))


def test_switching_to_the_executive_and_back_rewrites_it_in_their_drafts(
    db: Path, owner: Any, as_sender: list[str],
) -> None:
    from openexecutive.delegation import send_from

    mailbox, row = _chat_draft(owner)
    assert _state(owner, row, mailbox)["from_executive"] is False
    saved = _switch(owner, row, mailbox, True)
    assert as_sender == [send_from.AS_EXECUTIVE_PROMPT]
    assert saved["from_executive"] is True
    assert saved["note"] == send_from.NOTE_TO_EXECUTIVE
    assert saved["body"].startswith("Hi Dana,\n\nOlivia asked me")
    # One version: the words in their Drafts are the ones that go.
    assert mailbox.drafts["d1"].message.text == saved["body"]
    assert drafts.get_row(row).from_executive is True
    assert _state(owner, row, mailbox)["from_executive"] is True
    # Switching to where it already is changes nothing.
    assert _switch(owner, row, mailbox, True)["body"] == saved["body"]
    assert len(as_sender) == 1
    back = _switch(owner, row, mailbox, False)
    assert as_sender[-1] == send_from.AS_YOU_PROMPT
    assert back["from_executive"] is False and back["note"] == send_from.NOTE_TO_YOU
    assert drafts.get_row(row).from_executive is False
    assert mailbox.sent == []


def test_asking_for_it_from_the_executive_switches_it_on_the_card(
    db: Path, owner: Any, as_sender: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.delegation import send_from
    from openexecutive.delegation.answers import SendFrom

    mailbox, row = _chat_draft(owner)

    async def only_who(*a: Any, **kw: Any) -> Any:
        raise SendFrom(executive=True)

    monkeypatch.setattr("openexecutive.delegation.answers.rewrite", only_who)
    saved, _new, note = asyncio.run(chat_draft_card.change(
        owner, row, "Send it from the Executive.", caller=LOCAL, gmail=mailbox, now=LATER,
    ))
    assert saved["from_executive"] is True and note == send_from.NOTE_TO_EXECUTIVE
    assert as_sender == [send_from.AS_EXECUTIVE_PROMPT]
    assert mailbox.drafts["d1"].message.text == saved["body"]
    assert mailbox.sent == []


def test_a_switched_draft_goes_from_the_executive_with_them_on_cc(
    db: Path, owner: Any, as_sender: list[str], exec_sends: list[dict[str, Any]],
) -> None:
    from openexecutive.delegation.reply_send import SENT_BY_EXECUTIVE
    from openexecutive.delegation.send_from import executive_step

    mailbox, row = _chat_draft(owner)
    saved = _switch(owner, row, mailbox, True)
    assert asyncio.run(send_chat_draft(owner, row, caller=LOCAL, gmail=mailbox, now=LATER)) == ""
    # Nothing went from their own mailbox, and their draft of it is gone.
    assert mailbox.sent == []
    assert "d1" in mailbox.deleted
    assert len(exec_sends) == 1
    step = exec_sends[0]
    assert step["held"] is True and step["own_card"] is True
    assert step["requested_by_person_id"] == owner.id
    assert step["input"]["cc"] == _send_tests.OWNER
    assert step["input"]["body"] == saved["body"]
    assert drafts.sent_message_ids(owner.id) == {SENT_BY_EXECUTIVE}
    # Sent once: a second tap sends nothing.
    assert _refused(send_chat_draft(owner, row, caller=LOCAL, gmail=mailbox, now=LATER)).code in (
        "already_sent", "draft_gone",
    )
    assert len(exec_sends) == 1
    # The same step a reply card switched to the Executive sends.
    same, _tool = executive_step(owner, to=[_send_tests.DANA], cc=[], subject="Friday", body=saved["body"])
    assert same["input"] == step["input"]


def test_a_failed_executive_send_hands_it_back(
    db: Path, owner: Any, as_sender: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.orchestrator import take_the_lead

    mailbox, row = _chat_draft(owner)
    _switch(owner, row, mailbox, True)

    async def refused(payload: dict[str, Any], **kw: Any) -> str:
        return '{"error": "quota"}'

    monkeypatch.setattr(take_the_lead, "carry_out", refused)
    assert _refused(send_chat_draft(owner, row, caller=LOCAL, gmail=mailbox, now=LATER)).code == "send_failed"
    # Handed back: still in their Drafts, waiting for them.
    assert "d1" in mailbox.drafts
    assert drafts.count_waiting(owner.id, NOW - timedelta(days=1)) == 1


def test_a_table_goes_from_the_executive_as_html(db: Path, owner: Any) -> None:
    from openexecutive.delegation.send_from import executive_step

    table = "Here:\n\n| Item | Cost |\n| --- | --- |\n| Rent | 10 |\n"
    step, _ = executive_step(owner, to=["dana@x.example"], cc=[], subject="S", body=table)
    assert "<table" in str(step["input"])
    plain, _ = executive_step(owner, to=["dana@x.example"], cc=[], subject="S", body="Just words.")
    assert "<table" not in str(plain["input"])


def test_a_bcc_on_the_draft_goes_with_the_executive_send(
    db: Path, owner: Any, as_sender: list[str], exec_sends: list[dict[str, Any]],
) -> None:
    mailbox, row = _chat_draft(owner)
    _switch(owner, row, mailbox, True)
    mailbox.drafts["d1"].message.bcc = ["boss@co.example"]
    asyncio.run(send_chat_draft(owner, row, caller=LOCAL, gmail=mailbox, now=LATER))
    assert exec_sends[0]["input"]["bcc"] == "boss@co.example"


def test_both_mail_backends_carry_a_bcc() -> None:
    from openexecutive.integrations.workspace.google import GoogleMail
    from openexecutive.integrations.workspace.microsoft import MicrosoftMail

    common = {"mailbox": "exec@co.example", "to": "dana@x.example", "subject": "S", "body": "B"}
    assert GoogleMail().build_send_arguments(**common, bcc=["boss@co.example"])["bcc"] == "boss@co.example"
    message = MicrosoftMail().build_send_arguments(**common, bcc=["boss@co.example"])["body"]["Message"]
    assert message["bccRecipients"] == [{"emailAddress": {"address": "boss@co.example"}}]
    assert "bcc" not in GoogleMail().build_send_arguments(**common)


FORWARDED = "---------- Forwarded message ---------\nFrom: Kyle <kyle@x.example>\n\nThe quote is attached."


def _draft_of(owner: Any, subject: str, body: str) -> tuple[FakeInbox, int]:
    from openexecutive.delegation.gmail import DraftSpec

    mailbox = FakeInbox()
    made = asyncio.run(mailbox.create_draft(DraftSpec(to=[_send_tests.DANA], subject=subject, body=body)))
    row = drafts.record(owner.id, source=drafts.SOURCE_CHAT, thread_id=None, draft_id=made.draft_id,
                        message_id=made.message_id, now=NOW)
    return mailbox, row


def test_a_forward_switched_keeps_the_forwarded_message_as_it_was(
    db: Path, owner: Any, as_sender: list[str],
) -> None:
    mailbox, row = _draft_of(owner, "Fwd: Quote", f"See below.\n\n{FORWARDED}")
    saved = _switch(owner, row, mailbox, True)
    assert saved["body"].startswith("Hi Dana,\n\nOlivia asked me")
    assert saved["body"].endswith(FORWARDED) and saved["body"].count(FORWARDED) == 1
    assert mailbox.drafts["d1"].message.text == saved["body"]


def test_an_outlook_forward_switched_keeps_the_forwarded_message_as_it_was(
    db: Path, owner: Any, as_sender: list[str],
) -> None:
    outlook = "From: Dana <dana@example.com>\nSent: Monday\nTo: me\nSubject: Quote\n\n| a | b |\n|---|---|"
    mailbox, row = _draft_of(owner, "FW: Quote", f"See below.\n\n{outlook}")
    saved = _switch(owner, row, mailbox, True)
    assert saved["body"].startswith("Hi Dana,\n\nOlivia asked me")
    assert saved["body"].endswith(outlook) and saved["body"].count(outlook) == 1


def test_a_rewrite_that_changes_nothing_leaves_the_draft_alone(
    db: Path, owner: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = f"See below.\n\n{FORWARDED}"
    mailbox, row = _draft_of(owner, "Fwd: Quote", body)
    seen: dict[str, Any] = {}

    async def same(person: Any, payload: dict[str, Any], answers: list[Any], **kw: Any) -> Any:
        seen.update(payload)
        return payload["draft_body"], {}, []

    monkeypatch.setattr("openexecutive.delegation.answers.rewrite", same)
    saved, new, _ = asyncio.run(chat_draft_card.change(owner, row, "Shorter.", caller=LOCAL, gmail=mailbox, now=LATER))
    assert seen["draft_body"] == "See below."  # only the note is the model's
    assert saved["body"] == body and new == []
    assert "update:d1" not in mailbox.calls


def test_a_reply_quoting_a_forward_is_not_split(db: Path, owner: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    body = f"Thanks.\n\n{FORWARDED}"
    mailbox, row = _draft_of(owner, "Re: Quote", body)
    seen: dict[str, Any] = {}

    async def look(person: Any, payload: dict[str, Any], answers: list[Any], **kw: Any) -> Any:
        seen.update(payload)
        return payload["draft_body"], {}, []

    monkeypatch.setattr("openexecutive.delegation.answers.rewrite", look)
    asyncio.run(chat_draft_card.change(owner, row, "Shorter.", caller=LOCAL, gmail=mailbox, now=LATER))
    assert seen["draft_body"] == body


def test_asking_for_the_sender_it_already_has_says_so(
    db: Path, owner: Any, as_sender: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.delegation.answers import SendFrom

    mailbox, row = _chat_draft(owner)

    async def only_who(*a: Any, **kw: Any) -> Any:
        raise SendFrom(executive=False)

    monkeypatch.setattr("openexecutive.delegation.answers.rewrite", only_who)
    saved, _new, note = asyncio.run(chat_draft_card.change(owner, row, "Send it from me.", caller=LOCAL, gmail=mailbox, now=LATER))
    assert note == "It already goes from you."
    assert saved["from_executive"] is False and as_sender == []


def _with_files(mailbox: FakeInbox, monkeypatch: pytest.MonkeyPatch, *, broken: bool = False) -> None:
    from openexecutive.delegation.gmail import GmailError, MailAttachment

    mailbox.drafts["d1"].message.has_attachments = True

    async def list_attachments(message_id: str) -> list[MailAttachment]:
        return [MailAttachment(index=1, name="quote.pdf", mime_type="application/pdf")]

    async def attachment_bytes(message_id: str, index: int) -> tuple[MailAttachment, bytes]:
        if broken:
            raise GmailError("boom")
        return MailAttachment(index=1, name="quote.pdf", mime_type="application/pdf"), b"%PDF"

    monkeypatch.setattr(mailbox, "list_attachments", list_attachments, raising=False)
    monkeypatch.setattr(mailbox, "attachment_bytes", attachment_bytes, raising=False)


def test_the_executive_sends_the_drafts_files(
    db: Path, owner: Any, as_sender: list[str], exec_sends: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    mailbox, row = _chat_draft(owner)
    _switch(owner, row, mailbox, True)
    _with_files(mailbox, monkeypatch)
    asyncio.run(send_chat_draft(owner, row, caller=LOCAL, gmail=mailbox, now=LATER))
    assert "quote.pdf" in str(exec_sends[0]["input"])


def test_files_it_cant_read_send_nothing_and_hand_it_back(
    db: Path, owner: Any, as_sender: list[str], exec_sends: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    mailbox, row = _chat_draft(owner)
    _switch(owner, row, mailbox, True)
    _with_files(mailbox, monkeypatch, broken=True)
    assert _refused(send_chat_draft(owner, row, caller=LOCAL, gmail=mailbox, now=LATER)).code == "gmail_error"
    assert exec_sends == [] and "d1" in mailbox.drafts
    assert drafts.count_waiting(owner.id, NOW - timedelta(days=1)) == 1


def test_a_send_that_fails_before_it_goes_hands_it_back(
    db: Path, owner: Any, as_sender: list[str], exec_sends: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.delegation import send_from

    mailbox, row = _chat_draft(owner)
    _switch(owner, row, mailbox, True)

    def broken(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("no mail provider")

    monkeypatch.setattr(send_from, "executive_step", broken)
    with pytest.raises(RuntimeError):
        asyncio.run(send_chat_draft(owner, row, caller=LOCAL, gmail=mailbox, now=LATER))
    assert exec_sends == [] and drafts.count_waiting(owner.id, NOW - timedelta(days=1)) == 1


def test_a_draft_with_no_one_in_to_goes_to_them_with_the_rest_as_written(
    db: Path, owner: Any, as_sender: list[str], exec_sends: list[dict[str, Any]],
) -> None:
    mailbox, row = _chat_draft(owner)
    _switch(owner, row, mailbox, True)
    message = mailbox.drafts["d1"].message
    message.cc, message.to = list(message.to), []
    asyncio.run(send_chat_draft(owner, row, caller=LOCAL, gmail=mailbox, now=LATER))
    args = exec_sends[0]["input"]
    assert args["to"] == _send_tests.OWNER and args["cc"] == _send_tests.DANA


@pytest.mark.parametrize("field", ["to", "bcc"])
def test_a_person_already_on_the_draft_is_not_copied_again(
    db: Path, owner: Any, as_sender: list[str], exec_sends: list[dict[str, Any]], field: str,
) -> None:
    mailbox, row = _chat_draft(owner)
    _switch(owner, row, mailbox, True)
    message = mailbox.drafts["d1"].message
    setattr(message, field, [*getattr(message, field), _send_tests.OWNER])
    asyncio.run(send_chat_draft(owner, row, caller=LOCAL, gmail=mailbox, now=LATER))
    assert _send_tests.OWNER not in str(exec_sends[0]["input"].get("cc") or "")


def test_a_card_shown_before_a_switch_sends_nothing(
    db: Path, owner: Any, as_sender: list[str], exec_sends: list[dict[str, Any]],
) -> None:
    mailbox, row = _chat_draft(owner)
    shown = _state(owner, row, mailbox)  # tab A
    _switch(owner, row, mailbox, True)  # tab B
    for seen in ({"seen_version": shown["version"]}, {"seen_from_executive": False}):
        refused = _refused(send_chat_draft(owner, row, caller=LOCAL, confirm=seen, gmail=mailbox, now=LATER))
        assert refused.code == "draft_changed" and "Look at it again" in refused.message
    assert exec_sends == [] and mailbox.sent == []
    # Shown as it is now, it goes.
    now = _state(owner, row, mailbox)
    asyncio.run(send_chat_draft(owner, row, caller=LOCAL, confirm={"seen_version": now["version"],
                                                                   "seen_from_executive": True},
                                gmail=mailbox, now=LATER))
    assert len(exec_sends) == 1


def test_the_card_a_change_returns_sends_as_it_is(
    db: Path, owner: Any, as_sender: list[str], exec_sends: list[dict[str, Any]],
) -> None:
    # The card keeps what a switch or an edit hands back, and sends that.
    mailbox, row = _chat_draft(owner)
    shown = _switch(owner, row, mailbox, True)
    assert shown["version"] == _state(owner, row, mailbox)["version"]
    shown = asyncio.run(chat_draft_card.set_text(owner, row, "Mine", caller=LOCAL, gmail=mailbox, now=LATER))
    assert shown["version"] == _state(owner, row, mailbox)["version"]
    asyncio.run(send_chat_draft(owner, row, caller=LOCAL, confirm={
        "seen_version": shown["version"], "seen_from_executive": shown["from_executive"],
    }, gmail=mailbox, now=LATER))
    assert len(exec_sends) == 1


def test_home_names_who_a_switched_draft_goes_from(db: Path, owner: Any, as_sender: list[str]) -> None:
    mailbox, row = _chat_draft(owner)
    _switch(owner, row, mailbox, True)
    (waiting,) = drafts.waiting_drafts(owner.id, NOW - timedelta(days=1))
    assert waiting.row_id == row and waiting.from_executive is True
