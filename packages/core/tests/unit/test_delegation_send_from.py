"""Who a reply card's email goes from: the person or the Executive
(delegation/send_from.py)."""
from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from openexecutive.delegation import answers, card_drafts, inbox, replies, send_from, training
from openexecutive.delegation.reply_send import SendRefused, send_approved_reply
from openexecutive.integrations.workspace.google import GoogleMail
from openexecutive.integrations.workspace.microsoft import MicrosoftMail

from . import test_delegation_card_drafts as _drafts_tests
from .test_delegation_inbox import DANA, NOW, OWNER, FakeInbox

db = _drafts_tests.db
models = _drafts_tests.models
local_login = _drafts_tests.local_login
owner = _drafts_tests.owner
LOCAL = _drafts_tests.LOCAL
_card = _drafts_tests._card
_fresh = _drafts_tests._fresh


@pytest.fixture
def rewrites(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The model's rewrite for the new sender: which prompt it was given,
    and a body."""
    seen: list[str] = []

    async def fake(model: str, system: str, turn: str) -> dict[str, Any]:
        seen.append(system)
        if system == send_from.AS_EXECUTIVE_PROMPT:
            return {"body": "Hi Dana,\n\nOlivia asked me to say she'll get back to you shortly."}
        if system == send_from.AS_YOU_PROMPT:
            return {"body": "Hi Dana,\n\nI'll get back to you shortly."}
        return {"body": "unused"}

    monkeypatch.setattr(answers, "_call_model", fake)
    return seen


@pytest.fixture
def sends(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """What the Executive's mailbox was asked to send, through the one path
    an approved step takes."""
    from openexecutive.orchestrator import take_the_lead

    out: list[dict[str, Any]] = []

    async def fake(payload: dict[str, Any], *, by_principal: bool, own_card: bool = False) -> str:
        out.append({**payload, "by_principal": by_principal, "own_card": own_card})
        return json.dumps({"status": "sent"})

    monkeypatch.setattr(take_the_lead, "carry_out", fake)
    return out


def _seen(card: Any) -> dict[str, Any]:
    """What a card showing it as it is now sends back with Send."""
    return {"seen_caught_up_at": card_drafts.version(inbox.card_payload(_fresh(card)))}


def _switch(card: Any, mailbox: FakeInbox, owner: Any, executive: bool, **kw: Any) -> dict[str, Any]:
    return asyncio.run(send_from.switch(
        _fresh(card), caller=LOCAL, resolver=owner.id, executive=executive, gmail=mailbox, **kw,
    ))


def test_switch_to_the_executive_and_back(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str],
) -> None:
    mailbox, card = _card(owner)
    before = inbox.card_payload(_fresh(card))["draft_body"]
    saved = _switch(card, mailbox, owner, True)
    assert saved[send_from.FROM_EXECUTIVE] is True
    assert saved["draft_body"].startswith("Hi Dana,\n\nOlivia asked me")
    assert saved["draft_body"].endswith(send_from.sign_off(owner))
    # One version: no hidden copy of their words kept anywhere.
    assert before not in json.dumps(saved)
    assert saved["conversation"][-1]["text"] == send_from.NOTE_TO_EXECUTIVE
    shown = replies.cards(owner)[0]
    assert shown.from_executive is True and shown.executive_name
    # Back to them: the card's words written once more, as them.
    back = _switch(card, mailbox, owner, False)
    assert rewrites[-1] == send_from.AS_YOU_PROMPT
    assert back["draft_body"].startswith("Hi Dana,\n\nI'll get back to you shortly.")
    assert send_from.sign_off(owner) not in back["draft_body"]
    assert back[send_from.FROM_EXECUTIVE] is False
    assert back["conversation"][-1]["text"] == send_from.NOTE_TO_YOU


def test_only_its_person_switches_it(db: Path, owner: Any, models: dict[str, Any], rewrites: list[str]) -> None:
    mailbox, card = _card(owner)
    with pytest.raises(SendRefused) as err:
        asyncio.run(send_from.switch(_fresh(card), caller=LOCAL, resolver=owner.id + 99, executive=True, gmail=mailbox))
    assert err.value.code == "not_yours"


def test_send_goes_from_the_executive_with_them_on_cc(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str], sends: list[dict[str, Any]],
) -> None:
    mailbox, card = _card(owner)
    _switch(card, mailbox, owner, True)
    # A card shown before the switch looks again first; nothing goes.
    with pytest.raises(SendRefused) as err:
        asyncio.run(send_approved_reply(_fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox))
    assert err.value.code == "confirm" and sends == []
    assert asyncio.run(send_approved_reply(
        _fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, now=NOW + timedelta(minutes=30),
        confirm=_seen(card),
    )) == ""
    # Nothing went from their own mailbox.
    assert mailbox.sent == [] and mailbox.specs == []
    assert len(sends) == 1
    step = sends[0]
    assert step["held"] is True and step["mcp"] is True and step["own_card"] is True
    assert step["requested_by_person_id"] == owner.id and step["by_principal"] is True
    args = step["input"]
    assert args["to"] == DANA and args["cc"] == OWNER
    assert args["body"].startswith("Hi Dana,\n\nOlivia asked me")
    # The card closed as sent, once.
    assert _fresh(card).status == "approved_unchanged"
    assert inbox.open_cards(owner.id) == []
    with pytest.raises(SendRefused):
        asyncio.run(send_approved_reply(_fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox))
    assert len(sends) == 1


def test_a_tap_from_the_executive_sends_with_the_inbox_watcher_off(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str], sends: list[dict[str, Any]],
) -> None:
    """A tap is the person's own act whichever mailbox sends: the inbox
    watcher's switch has no say over it, as for their own Send."""
    mailbox, card = _card(owner)
    _switch(card, mailbox, owner, True)
    inbox.set_watch(owner.id, False, updated_by="test")
    assert asyncio.run(send_approved_reply(
        _fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, now=NOW + timedelta(minutes=30),
        confirm=_seen(card),
    )) == ""
    assert len(sends) == 1


def test_a_failed_send_leaves_the_card_waiting(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.orchestrator import take_the_lead

    async def refused(payload: dict[str, Any], *, by_principal: bool, own_card: bool = False) -> str:
        return "Error: HttpError 429 rateLimitExceeded"

    monkeypatch.setattr(take_the_lead, "carry_out", refused)
    mailbox, card = _card(owner)
    _switch(card, mailbox, owner, True)
    with pytest.raises(SendRefused) as err:
        asyncio.run(send_approved_reply(_fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, confirm=_seen(card)))
    assert err.value.code == "send_failed"
    # Plain words, never the app's own error text.
    assert err.value.message == "It wasn't sent. Email was busy, so nothing changed. Try again in a few minutes."
    assert _fresh(card).status == "proposed"


def test_saying_it_on_the_card_switches_the_same_card(
    db: Path, owner: Any, models: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake(model: str, system: str, turn: str) -> dict[str, Any]:
        if system == send_from.AS_EXECUTIVE_PROMPT:
            return {"body": "Hi Dana,\n\nOlivia asked me to reply."}
        # The card's change: only who it goes from, the words as they were.
        main = turn.split("<draft>\n", 1)[1].split("\n</draft>", 1)[0]
        return {"body": main, "send_from": "executive", "note": "ok"}

    monkeypatch.setattr(answers, "_call_model", fake)
    mailbox, card = _card(owner)
    saved, changed, _ = asyncio.run(answers.change(
        _fresh(card), caller=LOCAL, resolver=owner.id, text="send it from you and cc me", gmail=mailbox,
    ))
    assert changed == []
    assert saved[send_from.FROM_EXECUTIVE] is True
    assert saved["draft_body"].startswith("Hi Dana,\n\nOlivia asked me to reply.")
    turns = saved["conversation"]
    assert turns[-2] == {"kind": "asked", "text": "send it from you and cc me"}
    assert turns[-1]["text"] == send_from.NOTE_TO_EXECUTIVE
    # Still one card.
    assert len(inbox.open_cards(owner.id)) == 1


@pytest.mark.parametrize("go", [True, False])
def test_send_it_from_you_switches_the_sender_and_waits(
    db: Path, owner: Any, models: dict[str, Any], monkeypatch: pytest.MonkeyPatch, go: bool,
) -> None:
    from openexecutive.alerts import replan

    async def fake(model: str, system: str, turn: str) -> dict[str, Any]:
        if system == send_from.AS_EXECUTIVE_PROMPT:
            return {"body": "Hi Dana,\n\nOlivia asked me to reply."}
        main = turn.split("<draft>\n", 1)[1].split("\n</draft>", 1)[0]
        return {"body": main, "send_now": True, "send_from": "executive"}

    async def judged(system: str, tool: dict[str, Any], turn: str, **kw: Any) -> dict[str, Any]:
        return {"go": go}

    monkeypatch.setattr(answers, "_call_model", fake)
    monkeypatch.setattr(replan, "_call_tool", judged)
    replan._CALLS.clear()
    replan._CALLER_CALLS.clear()
    mailbox, card = _card(owner)
    saved, changed, send_now = asyncio.run(answers.change(
        _fresh(card), caller=LOCAL, resolver=owner.id, text="send it from you", gmail=mailbox,
    ))
    # Who it goes from switches either way; the send waits, since the new
    # wording is theirs to read first.
    assert changed == [] and send_now is False
    assert saved[send_from.FROM_EXECUTIVE] is True
    assert saved["conversation"][-1]["text"] == send_from.NOTE_TO_EXECUTIVE


def test_a_change_on_a_switched_card_edits_the_words_that_go(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str],
) -> None:
    mailbox, card = _card(owner)
    shown = _switch(card, mailbox, owner, True)["draft_body"]
    given: list[str] = []

    async def write(person: Any, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        given.append(payload["draft_body"])
        return f"Hi Dana,\n\nI'll get back to you Friday.\n\n{send_from.sign_off(person)}", {}

    saved = asyncio.run(card_drafts.edit(_fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, rewrite=write))
    # The change is made to the words on the card, the ones that go.
    assert given == [shown]
    assert saved[send_from.FROM_EXECUTIVE] is True
    assert saved["draft_body"] == f"Hi Dana,\n\nI'll get back to you Friday.\n\n{send_from.sign_off(owner)}"
    # And can be taken back.
    assert saved[card_drafts.UNDO_KEY]["draft_body"] == shown
    undone = asyncio.run(card_drafts.undo(_fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox))
    assert undone["draft_body"] == shown and not card_drafts.can_undo(undone)
    assert undone["conversation"][-1]["text"] == card_drafts.NOTE_UNDONE
    with pytest.raises(SendRefused) as err:
        asyncio.run(card_drafts.undo(_fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox))
    assert err.value.code == "nothing_to_undo"
    # A change by hand is kept as typed, and drops the undo.
    asyncio.run(card_drafts.edit(_fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, rewrite=write))
    kept = asyncio.run(card_drafts.edit(
        _fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, text="Hi Dana,\n\nShort and sweet.",
    ))
    assert kept["draft_body"] == "Hi Dana,\n\nShort and sweet." and not card_drafts.can_undo(kept)


def test_undo_puts_back_the_words_the_answers_and_their_mailbox_draft(
    db: Path, owner: Any, models: dict[str, Any],
) -> None:
    inbox.set_mailbox_drafts(owner.id, True, updated_by="test")
    mailbox, card = _card(owner)
    first = inbox.card_payload(_fresh(card))
    calls: list[int] = []

    async def write(person: Any, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        calls.append(1)
        if len(calls) == 2:
            # A question asked: the words stay as they are.
            return payload["draft_body"], {}
        return "Hi Dana,\n\nFriday at 10 works.", {answers.ANSWERED_KEY: {"0": "Friday"}}

    saved = asyncio.run(card_drafts.edit(_fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, rewrite=write))
    assert saved[answers.ANSWERED_KEY] == {"0": "Friday"} and card_drafts.can_undo(saved)
    undone = asyncio.run(card_drafts.undo(_fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox))
    assert undone["draft_body"] == first["draft_body"]
    assert undone[answers.ANSWERED_KEY] == {} and not card_drafts.can_undo(undone)
    # Its questions are open again, as they were.
    assert undone[card_drafts.QUESTIONS_SETTLED] is False
    # Their mailbox's draft says the same as the card again.
    assert mailbox.drafts[undone["draft_id"]].message.text == first["draft_body"]
    assert replies.cards(owner)[0].can_undo is False
    # A question asked after a change: Undo would show under its answer, so
    # the change before it can't be taken back from there.
    calls.clear()
    changed = asyncio.run(card_drafts.edit(_fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, rewrite=write))
    assert card_drafts.can_undo(changed)
    asked = asyncio.run(card_drafts.edit(_fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, rewrite=write))
    assert not card_drafts.can_undo(asked)


def test_a_switched_card_keeps_its_sign_off_when_its_words_lost_it(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str],
) -> None:
    mailbox, card = _card(owner)
    payload = _switch(card, mailbox, owner, True)
    main, tail = send_from.split(owner, payload, "Hi Dana,\n\nFriday works.\n\nSomeone Else")
    assert main == "Hi Dana,\n\nFriday works.\n\nSomeone Else" and tail == send_from.sign_off(owner)


def test_a_switched_card_tells_the_rewrite_who_i_is(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    mailbox, card = _card(owner)
    _switch(card, mailbox, owner, True)
    turns: list[str] = []

    async def fake(model: str, system: str, turn: str) -> dict[str, Any]:
        turns.append(turn)
        return {"body": "Hi Dana,\n\nI split the taxes 50/50.", "note": "Said I split them."}

    monkeypatch.setattr(answers, "_call_model", fake)
    saved, changed, _ = asyncio.run(answers.change(
        _fresh(card), caller=LOCAL, resolver=owner.id, text="where you say Olivia split taxes, say you", gmail=mailbox,
    ))
    assert "<sender>" in turns[0] and send_from.AS_EXECUTIVE_RULE.split(":")[0] in turns[0]
    # The draft it was given is the Executive's, without its sign-off.
    assert "Olivia asked me" in turns[0] and send_from.sign_off(owner) not in turns[0]
    assert saved["draft_body"] == f"Hi Dana,\n\nI split the taxes 50/50.\n\n{send_from.sign_off(owner)}"
    # The note quotes what changed.
    assert changed == ["I split the taxes 50/50."]
    assert saved["conversation"][-1]["changed"] == changed
    assert replies.cards(owner)[0].conversation[-1]["changed"] == changed
    # Nothing tells a card from them who "I" is: the prompt's default.
    assert send_from.writer_note({}) == ""


def test_the_switch_is_told_what_the_executive_did(
    db: Path, owner: Any, models: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.memory.decision_ledger import update_open_payload

    mailbox, card = _card(owner)
    payload = inbox.card_payload(_fresh(card))
    done = {"kind": "note", "state": "done", "text": "I built the spreadsheet.", "at": "2026-11-17T10:00:00+00:00"}
    assert update_open_payload(card.id, {**payload, "conversation": [done]}, expected_json=_fresh(card).proposed_payload_json)
    turns: list[str] = []

    async def fake(model: str, system: str, turn: str) -> dict[str, Any]:
        turns.append(turn)
        return {"body": "Hi Dana,\n\nI built the spreadsheet."}

    monkeypatch.setattr(answers, "_call_model", fake)
    _switch(card, mailbox, owner, True)
    assert "<work_done>\n- I built the spreadsheet.\n</work_done>" in turns[0]


def test_their_mailbox_draft_waits_while_it_goes_from_the_executive(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str],
) -> None:
    inbox.set_mailbox_drafts(owner.id, True, updated_by="test")
    mailbox, card = _card(owner)
    draft_id = inbox.card_payload(_fresh(card))["draft_id"]
    theirs = mailbox.drafts[draft_id].message.text
    _switch(card, mailbox, owner, True)

    async def write(person: Any, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return "Hi Dana,\n\nOlivia asked me: Friday works.", {}

    asyncio.run(card_drafts.edit(_fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, rewrite=write))
    # It never goes from their mailbox in the Executive's words.
    assert mailbox.drafts[draft_id].message.text == theirs
    # A file the work put on the card meanwhile.
    from openexecutive.delegation import card_work

    card_work._store(card.id, [("sheet.xlsx", "application/octet-stream", b"PK")])
    # From them again: their draft is made again from the card, words and files.
    back = _switch(card, mailbox, owner, False)
    assert back["draft_id"] != draft_id and draft_id not in mailbox.drafts
    made = mailbox.drafts[back["draft_id"]]
    assert made.message.text == back["draft_body"]
    assert "I'll get back to you shortly." in back["draft_body"]
    assert mailbox.specs[-1].attachments == [("sheet.xlsx", "application/octet-stream", b"PK")]
    assert back["draft_message_id"] == made.message.id


def test_a_reply_asked_for_in_chat_goes_from_them_again(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str],
) -> None:
    from openexecutive.delegation import card_work

    inbox.set_mailbox_drafts(owner.id, True, updated_by="test")
    mailbox, card = _card(owner)
    draft_id = inbox.card_payload(_fresh(card))["draft_id"]
    _switch(card, mailbox, owner, True)
    card_work._store(card.id, [("sheet.xlsx", "application/octet-stream", b"PK")])
    saved = asyncio.run(card_drafts.from_chat(
        _fresh(card), owner, mailbox, words="Hi Dana,\n\nMonday works.", to=[DANA], cc=["ben@northpeak.example"],
    ))
    assert saved[send_from.FROM_EXECUTIVE] is False and saved["draft_body"].startswith("Hi Dana,\n\nMonday works.")
    # Said on the card, like a switch there.
    assert saved["conversation"][-1]["text"] == send_from.NOTE_TO_YOU and saved[send_from.SWITCHED_BACK_AT]
    # Their draft made again: the chat's words and recipients, the card's files.
    assert saved["draft_id"] != draft_id and draft_id not in mailbox.drafts
    made = mailbox.drafts[saved["draft_id"]].message
    assert made.text == saved["draft_body"] and made.cc == ["ben@northpeak.example"]
    assert mailbox.specs[-1].attachments == [("sheet.xlsx", "application/octet-stream", b"PK")]


@pytest.mark.parametrize("there", ["changed", "gone"])
def test_back_from_them_never_replaces_a_draft_changed_in_their_mailbox(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str], there: str,
) -> None:
    from dataclasses import replace

    inbox.set_mailbox_drafts(owner.id, True, updated_by="test")
    mailbox, card = _card(owner)
    draft_id = inbox.card_payload(_fresh(card))["draft_id"]
    _switch(card, mailbox, owner, True)
    if there == "gone":
        asyncio.run(mailbox.delete_draft(draft_id))
    else:
        info = mailbox.drafts[draft_id]
        mailbox.drafts[draft_id] = replace(info, message=replace(info.message, id="edited-in-gmail"))
    made = len(mailbox.specs)
    with pytest.raises(SendRefused) as err:
        _switch(card, mailbox, owner, False)
    assert err.value.code == ("draft_changed" if there == "changed" else "draft_gone")
    # No second draft made, and theirs is left as it is.
    assert len(mailbox.specs) == made
    if there == "changed":
        assert draft_id in mailbox.drafts and _fresh(card).status == "proposed"
    else:
        assert _fresh(card).status != "proposed"


def test_a_chat_app_card_shows_who_it_goes_from_when_posted() -> None:
    t1, t2, t3, t4 = (f"2026-11-17T1{h}:00:00+00:00" for h in range(4))
    switched = {send_from.FROM_EXECUTIVE: True, send_from.SWITCHED_AT: t2}
    back = {send_from.FROM_EXECUTIVE: False, send_from.SWITCHED_AT: t2, send_from.SWITCHED_BACK_AT: t3}
    # Posted after it switched to the Executive: it showed what goes.
    assert card_drafts.asks_again(switched, {}, moved_on=False, shown_at=t3) == []
    # Posted before: it showed them as the sender.
    assert card_drafts.asks_again(switched, {}, moved_on=False, shown_at=t1) == ["changed_in_chat"]
    # Posted after it switched back: it shows them, as now.
    assert card_drafts.asks_again(back, {}, moved_on=False, shown_at=t4) == []
    # Any switch since it was posted: look again, however it ended up.
    assert card_drafts.asks_again(back, {}, moved_on=False, shown_at=t1) == ["changed_in_chat"]
    assert card_drafts.asks_again(back, {}, moved_on=False, shown_at=t2) == ["changed_in_chat"]
    # Not known when it was posted: look again.
    assert card_drafts.asks_again(back, {}, moved_on=False) == ["changed_in_chat"]


def test_a_chat_app_tap_sends_from_the_executive(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str], sends: list[dict[str, Any]],
) -> None:
    mailbox, card = _card(owner)
    _switch(card, mailbox, owner, True)
    posted = (NOW + timedelta(days=3650)).isoformat()
    asyncio.run(send_approved_reply(
        _fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, confirm={}, shown_at=posted,
        now=NOW + timedelta(minutes=30),
    ))
    assert len(sends) == 1 and sends[0]["input"]["cc"] == OWNER


def test_newer_mail_asks_again_before_the_executive_sends(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str], sends: list[dict[str, Any]],
) -> None:
    from .test_delegation_inbox import _msg

    mailbox, card = _card(owner)
    _switch(card, mailbox, owner, True)
    mailbox.add(_msg("m9", "t1", sender=DANA, minutes_ago=1, text="Never mind, sorted."))
    later = NOW + timedelta(minutes=30)
    with pytest.raises(SendRefused) as err:
        asyncio.run(send_approved_reply(
            _fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, confirm=_seen(card), now=later,
        ))
    assert err.value.code == "confirm" and err.value.extra["reasons"] == ["thread_moved_on"] and sends == []
    asyncio.run(send_approved_reply(
        _fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, now=later,
        confirm={**_seen(card), "thread_moved_on": True},
    ))
    assert len(sends) == 1


def test_no_on_its_own_next_time_from_the_executive(db: Path, owner: Any) -> None:
    trained = training.Training(person_id=owner.id, training=True, drafts=False)
    payload = {"draft_to": [DANA], "from_email": DANA}
    assert training.can_allow_reply(owner.id, payload, trained) is True
    assert training.can_allow_reply(owner.id, {**payload, "from_executive": True}, trained) is False


def test_send_arguments_carry_cc_and_files() -> None:
    files = [("letter.pdf", "application/pdf", b"%PDF")]
    google = GoogleMail().build_send_arguments(
        mailbox="exec@x.example", to=DANA, subject="Re: hi", body="b", cc=[OWNER], attachments=files,
    )
    assert google["cc"] == OWNER
    assert google["attachments"] == [{"content": "JVBERg==", "filename": "letter.pdf", "mime_type": "application/pdf"}]
    graph = MicrosoftMail().build_send_arguments(
        mailbox="exec@x.example", to=DANA, subject="Re: hi", body="b", cc=[OWNER], attachments=files,
    )["body"]["Message"]
    assert graph["ccRecipients"] == [{"emailAddress": {"address": OWNER}}]
    assert graph["attachments"][0]["contentBytes"] == "JVBERg==" and graph["attachments"][0]["name"] == "letter.pdf"
    # Without them, as before.
    assert "cc" not in GoogleMail().build_send_arguments(mailbox="e", to=DANA, subject="s", body="b")


def test_chat_app_card_names_the_executive(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str],
) -> None:
    from openexecutive.integrations import channel_cards

    def labelled() -> tuple[tuple[str, str], ...]:
        options = channel_cards.card_options(card.id, "")
        return tuple((label, channel_cards.choice_of(data)) for label, data in options)

    mailbox, card = _card(owner)
    assert labelled() == (
        ("Send from me", "a"), ("Send from Open Executive, cc me", "x"), ("Skip", "s"),
    )
    _switch(card, mailbox, owner, True)
    assert labelled() == (
        ("Send from Open Executive, cc me", "a"), ("Send from me", "x"), ("Skip", "s"),
    )


def test_a_chat_tap_switches_the_sender_and_shows_the_new_card(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str], sends: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.integrations import channel_cards

    mailbox, card = _card(owner)
    _in_this_mailbox(monkeypatch, mailbox)
    shown = channel_cards.card_text(card.id) or ""
    tapped = asyncio.run(channel_cards.tap(
        channel_cards.Choice(card.id, channel_cards.SWITCH, channel_cards.version(card.id, shown)), owner,
    ))
    assert tapped.line == "Changed who it goes from. Check it, then send it."
    assert "this reply from Open Executive, with you on cc." in tapped.text
    assert tapped.options[0] == (
        "Send from Open Executive, cc me", channel_cards.callback_data(card.id, "a", tapped.text),
    )
    # The old card's Send from me now only shows the card as it is.
    stale = channel_cards.Choice(card.id, channel_cards.APPROVE, channel_cards.version(card.id, shown))
    assert asyncio.run(channel_cards.tap(stale, owner)).line == channel_cards.CHANGED
    # A switch sends nothing.
    assert sends == [] and _fresh(card).status == "proposed"


def test_a_chat_tap_switches_only_while_as_you_is_on(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.delegation.settings import set_enabled
    from openexecutive.integrations import channel_cards

    mailbox, card = _card(owner)
    _in_this_mailbox(monkeypatch, mailbox)
    set_enabled(owner.id, False, updated_by="test")
    shown = channel_cards.card_text(card.id) or ""
    tapped = asyncio.run(channel_cards.tap(
        channel_cards.Choice(card.id, channel_cards.SWITCH, channel_cards.version(card.id, shown)), owner,
    ))
    # The same answer as the web card's: nothing rewritten.
    assert tapped.line == "Not done: Turn on As you to change this reply here."
    assert rewrites == [] and inbox.card_payload(_fresh(card)).get(send_from.FROM_EXECUTIVE) is not True


def _in_this_mailbox(monkeypatch: pytest.MonkeyPatch, mailbox: Any) -> None:
    switch = send_from.switch

    async def in_this_mailbox(instance: Any, **kwargs: Any) -> dict[str, Any]:
        return await switch(instance, **kwargs, gmail=mailbox)

    monkeypatch.setattr(send_from, "switch", in_this_mailbox)


def test_not_on_a_card_going_on_its_own(db: Path, owner: Any, models: dict[str, Any], rewrites: list[str]) -> None:
    mailbox, card = _card(owner)
    import sqlite3

    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE decision_instances SET gate_mode = 'auto_execute' WHERE id = ?", (card.id,))
    with pytest.raises(SendRefused) as err:
        asyncio.run(send_from.switch_for_work(card.id, owner, executive=True, gmail=mailbox))
    assert err.value.code == "already_handled"
    assert inbox.card_payload(_fresh(card)).get(send_from.FROM_EXECUTIVE) is not True


def test_a_reply_in_their_mailbox_keeps_their_words_there(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str], sends: list[dict[str, Any]],
) -> None:
    inbox.set_mailbox_drafts(owner.id, True, updated_by="test")
    mailbox, card = _card(owner)
    payload = inbox.card_payload(_fresh(card))
    assert payload["draft_id"]
    calls = list(mailbox.calls)
    _switch(card, mailbox, owner, True)
    assert mailbox.calls == calls  # the draft in their mailbox wasn't touched
    asyncio.run(send_approved_reply(_fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, confirm=_seen(card)))
    args = sends[0]["input"]
    assert args["to"] == DANA and args["cc"] == OWNER
    # Their own draft of it is taken out of their mailbox.
    assert mailbox.deleted == [payload["draft_id"]]


def test_not_after_they_replied_themselves(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str], sends: list[dict[str, Any]],
) -> None:
    from .test_delegation_inbox import _msg

    mailbox, card = _card(owner)
    _switch(card, mailbox, owner, True)
    mailbox.add(_msg("m9", "t1", sender=OWNER, minutes_ago=1, labels=("SENT",), text="Done, thanks."))
    with pytest.raises(SendRefused) as err:
        asyncio.run(send_approved_reply(
            _fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, confirm=_seen(card),
            now=NOW + timedelta(minutes=30),
        ))
    assert err.value.code == "you_replied" and sends == []


def test_an_unreadable_mailbox_stops_it(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str], sends: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.delegation.gmail import GmailError

    mailbox, card = _card(owner)
    _switch(card, mailbox, owner, True)

    async def broken(thread_id: str) -> Any:
        raise GmailError("expired")

    monkeypatch.setattr(mailbox, "get_thread", broken)
    with pytest.raises(SendRefused) as err:
        asyncio.run(send_approved_reply(
            _fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, confirm=_seen(card),
            now=NOW + timedelta(minutes=30),
        ))
    assert err.value.code == "gmail_error" and sends == []
    assert _fresh(card).status == "proposed"


def test_new_words_stay_when_the_switch_asked_with_them_fails(
    db: Path, owner: Any, models: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake(model: str, system: str, turn: str) -> dict[str, Any]:
        if system == send_from.AS_EXECUTIVE_PROMPT:
            return {}  # the rewrite as the Executive fails
        return {"body": "Hi Dana,\n\nShorter.", "send_from": "executive", "note": "Shorter."}

    monkeypatch.setattr(answers, "_call_model", fake)
    mailbox, card = _card(owner)
    saved, _, _ = asyncio.run(answers.change(
        _fresh(card), caller=LOCAL, resolver=owner.id, text="shorten it and send it from you", gmail=mailbox,
    ))
    assert "Shorter." in saved["draft_body"] and saved.get(send_from.FROM_EXECUTIVE) is not True
    assert saved["conversation"][-1]["text"] == send_from.NOTE_NOT_SWITCHED


def test_a_change_saved_while_it_checks_is_not_sent(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str], sends: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.memory.decision_ledger import update_open_payload

    mailbox, card = _card(owner)
    _switch(card, mailbox, owner, True)
    tapped = _fresh(card)
    read = mailbox.get_thread

    async def changed_meanwhile(thread_id: str) -> Any:
        # Another tab saves new words while the mailbox is being read.
        now = _fresh(card)
        update_open_payload(
            now.id, {**inbox.card_payload(now), "draft_body": "Newer words."},
            expected_json=now.proposed_payload_json,
        )
        return await read(thread_id)

    monkeypatch.setattr(mailbox, "get_thread", changed_meanwhile)
    with pytest.raises(SendRefused) as err:
        asyncio.run(send_approved_reply(
            tapped, caller=LOCAL, resolver=owner.id, gmail=mailbox, confirm=_seen(card),
            now=NOW + timedelta(minutes=30),
        ))
    assert err.value.code == "already_handled" and sends == []
    assert _fresh(card).status == "proposed"


def test_the_cards_work_switches_it_before_it_hands_back(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.delegation import card_work
    from openexecutive.orchestrator import reply_card_tools
    from openexecutive.orchestrator.schedule_tools import current_session
    from openexecutive.orchestrator.session import Session

    mailbox, card = _card(owner)
    order: list[str] = []
    real_apply, real_switch = card_work.apply, send_from.switch_for_work

    async def apply(*a: Any, **k: Any) -> Any:
        order.append("apply")
        return await real_apply(*a, **k)

    async def switch(*a: Any, **k: Any) -> Any:
        order.append("switch")
        return await real_switch(*a, **k)

    monkeypatch.setattr(card_work, "still_working", lambda c: True)
    monkeypatch.setattr(card_work, "keep", lambda coro: coro.close())
    monkeypatch.setattr(card_work, "apply", apply)
    monkeypatch.setattr(send_from, "switch_for_work", switch)
    writer = type("W", (), {"person": owner, "mailbox": mailbox})()
    monkeypatch.setattr("openexecutive.orchestrator.delegation_tools._writer", lambda name: writer)
    token = current_session.set(Session(reply_card_work=card.id))
    try:
        out = json.loads(asyncio.run(reply_card_tools.handle_update_reply_card(
            {"note": "Sending it from me.", "send_from": "executive"},
        )))
        assert out["status"] == "ok" and order == ["switch", "apply"]
        assert inbox.card_payload(_fresh(card)).get(send_from.FROM_EXECUTIVE) is True

        async def refused(*a: Any, **k: Any) -> None:
            raise SendRefused(409, "gone", "It was sent.")

        monkeypatch.setattr(card_work, "apply", refused)
        out = json.loads(asyncio.run(reply_card_tools.handle_update_reply_card({"send_from": "me", "note": "From you again."})))
        assert out["error"].startswith("Who it goes from was changed, and the reply was written again")
        out = json.loads(asyncio.run(reply_card_tools.handle_update_reply_card({"send_from": "someone"})))
        assert out["error"].startswith("Nothing was changed")
    finally:
        current_session.reset(token)


def test_words_the_work_wrote_with_a_switch_are_written_for_the_new_sender(
    db: Path, owner: Any, models: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.delegation import card_work
    from openexecutive.orchestrator import reply_card_tools
    from openexecutive.orchestrator.schedule_tools import current_session
    from openexecutive.orchestrator.session import Session

    turns: list[str] = []

    async def fake(model: str, system: str, turn: str) -> dict[str, Any]:
        turns.append(turn)
        return {"body": "Hi Dana,\n\nI attached the deck. Olivia will call you Thursday."}

    monkeypatch.setattr(answers, "_call_model", fake)
    mailbox, card = _card(owner)
    monkeypatch.setattr(card_work, "still_working", lambda c: True)
    monkeypatch.setattr(card_work, "keep", lambda coro: coro.close())
    writer = type("W", (), {"person": owner, "mailbox": mailbox})()
    monkeypatch.setattr("openexecutive.orchestrator.delegation_tools._writer", lambda name: writer)
    token = current_session.set(Session(reply_card_work=card.id))
    try:
        out = json.loads(asyncio.run(reply_card_tools.handle_update_reply_card({
            "body": "Hi Dana,\n\nI've attached the deck. I'll call you Thursday.",
            "note": "Attached the deck.", "send_from": "executive",
        })))
    finally:
        current_session.reset(token)
    assert out["status"] == "ok"
    # Written once for the Executive from the words the work wrote, never
    # their words signed as the Executive.
    assert len(turns) == 1 and "I'll call you Thursday." in turns[0]
    # Told what this run did, as the assistant's own work.
    assert "<work_done>\n- Attached the deck.\n</work_done>" in turns[0]
    saved = inbox.card_payload(_fresh(card))
    assert saved[send_from.FROM_EXECUTIVE] is True
    assert saved["draft_body"] == (
        f"Hi Dana,\n\nI attached the deck. Olivia will call you Thursday.\n\n{send_from.sign_off(owner)}"
    )
    assert saved["conversation"][-1]["text"] == "Attached the deck."
    # The run wrote the reply: it goes on its own when Approvals says so,
    # its questions are settled and its note quotes what changed.
    assert saved[card_work.REWROTE_KEY] is True and saved[card_drafts.QUESTIONS_SETTLED] is True
    assert saved["gist"] == "Attached the deck."
    assert saved["conversation"][-1]["changed"] == ["I attached the deck.", "Olivia will call you Thursday."]
    # A switch can't be undone.
    assert not card_drafts.can_undo(saved)


def test_a_send_from_it_already_has_keeps_the_words_the_work_wrote(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.delegation import card_work
    from openexecutive.orchestrator import reply_card_tools
    from openexecutive.orchestrator.schedule_tools import current_session
    from openexecutive.orchestrator.session import Session

    mailbox, card = _card(owner)
    _switch(card, mailbox, owner, True)
    asked = len(rewrites)
    monkeypatch.setattr(card_work, "still_working", lambda c: True)
    monkeypatch.setattr(card_work, "keep", lambda coro: coro.close())
    writer = type("W", (), {"person": owner, "mailbox": mailbox})()
    monkeypatch.setattr("openexecutive.orchestrator.delegation_tools._writer", lambda name: writer)
    token = current_session.set(Session(reply_card_work=card.id))
    try:
        out = json.loads(asyncio.run(reply_card_tools.handle_update_reply_card({
            "body": "Hi Dana,\n\nI split the taxes 50/50.", "note": "Split them.", "send_from": "executive",
        })))
    finally:
        current_session.reset(token)
    assert out["status"] == "ok" and len(rewrites) == asked
    saved = inbox.card_payload(_fresh(card))
    assert saved["draft_body"] == f"Hi Dana,\n\nI split the taxes 50/50.\n\n{send_from.sign_off(owner)}"


def test_a_work_turn_that_would_be_turned_down_switches_nothing(
    db: Path, owner: Any, models: dict[str, Any], rewrites: list[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.delegation import card_work
    from openexecutive.orchestrator import reply_card_tools
    from openexecutive.orchestrator.schedule_tools import current_session
    from openexecutive.orchestrator.session import Session

    mailbox, card = _card(owner)
    before = inbox.card_payload(_fresh(card))
    monkeypatch.setattr(card_work, "still_working", lambda c: True)
    monkeypatch.setattr(card_work, "keep", lambda coro: coro.close())
    writer = type("W", (), {"person": owner, "mailbox": mailbox})()
    monkeypatch.setattr("openexecutive.orchestrator.delegation_tools._writer", lambda name: writer)
    token = current_session.set(Session(reply_card_work=card.id))
    try:
        out = json.loads(asyncio.run(reply_card_tools.handle_update_reply_card({
            "body": "Hi Dana,\n\nI'll send the figures Friday.", "note": "", "send_from": "executive",
        })))
    finally:
        current_session.reset(token)
    # Turned down before the switch: the card is as it was, so a retry
    # with a note switches and writes the reply for the Executive then.
    assert out["error"].startswith("Nothing was changed")
    assert rewrites == [] and inbox.card_payload(_fresh(card)) == before


def test_a_change_of_who_it_goes_to_keeps_the_undo(db: Path, owner: Any, models: dict[str, Any]) -> None:
    mailbox, card = _drafts_tests._group_card(owner)

    async def write(person: Any, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return "Hi Dana,\n\nFriday at 10 works.", {}

    saved = asyncio.run(card_drafts.edit(_fresh(card), caller=LOCAL, resolver=owner.id, gmail=mailbox, rewrite=write))
    moved = _drafts_tests._recipients(card, mailbox, owner, False)
    assert moved["draft_to"] == [DANA] and moved[card_drafts.UNDO_KEY] == saved[card_drafts.UNDO_KEY]


def test_the_executive_signs_once(db: Path, owner: Any, models: dict[str, Any], rewrites: list[str]) -> None:
    mailbox, card = _card(owner)
    payload = _switch(card, mailbox, owner, True)
    mine = send_from.sign_off(owner)
    assert send_from.sign(owner, payload, f"Hi Dana,\n\nDone.\n\n{mine}") == f"Hi Dana,\n\nDone.\n\n{mine}"
