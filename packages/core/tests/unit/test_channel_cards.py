# ruff: noqa: F811 — pytest fixtures imported from test_action_cards share their names with the test parameters.
"""Approval cards in Telegram, with Approve and Skip buttons
(integrations/channel_cards.py, integrations/telegram_bot.py,
api/routes/decisions.resolve_in_chat)."""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from openexecutive.api.routes import decisions as decisions_route
from openexecutive.delegation.settings import set_enabled
from openexecutive.delegation.verified import NOT_YOURS, ChatTap, caller_refusal
from openexecutive.integrations import channel_cards, telegram_bot
from openexecutive.integrations.channel_context import build_channel_context_block
from openexecutive.memory.decision_ledger import STATUS_APPROVED_UNCHANGED, get_decision_instance
from openexecutive.orchestrator.executive import Executive
from openexecutive.orchestrator.session import Session
from openexecutive.people.store import get_person

# Fakes and autouse fixtures (a temp DB, a fresh turn pin, a roster, sends).
from .test_action_cards import _card, _message, _teammate_session, roster, sends  # noqa: F401
from .test_delegation_tools import (  # noqa: F401
    OWNER,
    TEAM,
    FakeMailbox,
    _session,
    db,
    fresh_turn_state,
)

TOKEN = "123456789:AAH" + "x" * 32


@pytest.fixture
def telegram(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OE_LOCAL_LOGIN", "OE_PUBLIC_DEPLOYMENT", "BACKEND_SHARED_SECRET"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "hook-secret")


def _tg_session(chat: str = "4242", person_id: int | None = 1) -> Session:
    session = Session(origin_channel="telegram", origin_channel_ref=chat)
    session.caller_person_id = person_id
    return session


def test_buttons_round_trip_and_junk_is_ignored() -> None:
    shown = channel_cards.version(7, "Meeting to book")
    for choice in (channel_cards.APPROVE, channel_cards.SKIP, channel_cards.ALLOW, channel_cards.SWITCH):
        data = channel_cards.callback_data(7, choice, "Meeting to book")
        assert channel_cards.parse_choice(data) == channel_cards.Choice(7, choice, shown)
    for junk in (
        None, "", "card:7", "card:x:a", "card:7:z", "other:7:a", "card:-7:a", "card:" + "9" * 20 + ":a",
        # Every button carries the version of the card it is on.
        "card:7:a", "card:7:a:", "card:7:a:XYZ", "card:7:a:0000beef:1",
    ):
        assert channel_cards.parse_choice(junk) is None


def test_cards_show_only_in_a_private_telegram_chat_only_telegram_can_reach(
    telegram: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert channel_cards.shows_cards(_tg_session())
    assert not channel_cards.shows_cards(_tg_session(chat="-100123"))  # a group
    assert not channel_cards.shows_cards(_tg_session(person_id=None))
    assert not channel_cards.shows_cards(Session(origin_channel="slack", origin_channel_ref="U1"))
    assert not channel_cards.shows_cards(None)
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "")
    assert not channel_cards.shows_cards(_tg_session())
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "# not one Telegram can send")
    assert not channel_cards.shows_cards(_tg_session())


def test_a_chat_tap_is_only_the_person_it_names() -> None:
    assert caller_refusal(ChatTap("olivia@acme.example"), "olivia@acme.example") is None
    assert caller_refusal(ChatTap("ben@acme.example"), "olivia@acme.example") == NOT_YOURS
    assert caller_refusal(ChatTap(""), "") == NOT_YOURS


def test_only_the_chats_own_person_can_tap() -> None:
    def tap(chat: dict[str, Any], sender: int) -> dict[str, Any]:
        return {"id": "q1", "from": {"id": sender}, "data": "card:3:a:0000beef",
                "message": {"message_id": 9, "chat": chat, "text": "Meeting to book"}}

    found = telegram_bot._card_tap(tap({"id": 4242, "type": "private"}, 4242))
    assert found == {"callback_id": "q1", "chat_id": 4242, "message_id": 9,
                     "card_text": "Meeting to book", "data": "card:3:a:0000beef"}
    assert telegram_bot._card_tap(tap({"id": 4242, "type": "private"}, 999)) is None
    assert telegram_bot._card_tap(tap({"id": -100, "type": "group"}, -100)) is None
    assert telegram_bot._card_tap(tap({"id": 4242, "type": "supergroup"}, 4242)) is None
    assert telegram_bot._card_tap("nope") is None


def test_a_tap_is_taken_only_on_a_webhook_only_telegram_can_call(
    telegram: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.api.main import create_app

    taken: list[dict[str, Any]] = []

    async def handle(**kwargs: Any) -> None:
        taken.append(kwargs)

    monkeypatch.setattr(telegram_bot, "_handle_card_tap", handle)
    update = {"callback_query": {"id": "q1", "from": {"id": 4242}, "data": "card:3:a:0000beef",
                                 "message": {"message_id": 9, "chat": {"id": 4242, "type": "private"}}}}
    client = TestClient(create_app())
    headers = {"X-Telegram-Bot-Api-Secret-Token": "hook-secret"}
    assert client.post("/webhook/telegram", json=update, headers=headers).status_code == 200
    assert [t["chat_id"] for t in taken] == [4242]
    # Someone else's tap in that chat is ignored.
    forged = json.loads(json.dumps(update))
    forged["callback_query"]["from"]["id"] = 7
    assert client.post("/webhook/telegram", json=forged, headers=headers).status_code == 200
    assert len(taken) == 1
    # With no secret, anyone could post a tap: none is taken.
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "")
    assert TestClient(create_app()).post("/webhook/telegram", json=update).status_code == 200
    assert len(taken) == 1


def test_a_card_is_approved_by_its_person_and_skipped_cards_do_nothing(
    roster: SimpleNamespace, sends: list[Any],
) -> None:
    set_enabled(roster.principal, True, updated_by="t")
    card = _card(_session(FakeMailbox()), [_message(roster.teammate)])
    text = channel_cards.card_text(card.id)
    assert text is not None and text.startswith("Waiting for your approval") and "Message Ben Teammate" in text
    # Someone else on the team: not theirs, so it isn't there.
    assert asyncio.run(channel_cards.resolve(card.id, get_person(roster.teammate), approve=True)) == (
        "Not done: This card is no longer there."
    )
    assert sends == []
    assert asyncio.run(channel_cards.resolve(card.id, get_person(roster.principal), approve=True)) == "Approved."
    assert [i.get("text") for _, i, _ in sends] == ["Can you check the pilot dates?"]
    assert get_decision_instance(card.id).status == STATUS_APPROVED_UNCHANGED
    assert channel_cards.card_text(card.id) is None
    assert asyncio.run(channel_cards.resolve(card.id, get_person(roster.principal), approve=True)).startswith("Not done")
    assert len(sends) == 1

    skipped = _card(_session(FakeMailbox()), [_message(roster.teammate, "another")])
    assert asyncio.run(channel_cards.resolve(skipped.id, get_person(roster.principal), approve=False)) == "Skipped."
    assert get_decision_instance(skipped.id).status == "rejected"
    assert len(sends) == 1


def test_a_card_of_actions_shows_what_its_tap_would_do(
    roster: SimpleNamespace, sends: list[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_enabled(roster.principal, True, updated_by="t")
    card = _card(_session(FakeMailbox()), [_message(roster.teammate, "The pilot moves to Friday.")])
    assert channel_cards.card_text(card.id).endswith("- Message Ben Teammate\n> The pilot moves to Friday.")

    # The files and everyone who can open the folder, whole.
    save = {"kind": "save_files", "summary": "Save 1 file to Partner folder",
            "text": "Bill.pdf\nAlso open to: dana@partner.example"}
    stored = SimpleNamespace(
        decision_class="delegation_actions", status="proposed", proposed_payload_json=json.dumps({"actions": [save]}),
    )
    monkeypatch.setattr("openexecutive.memory.decision_ledger.get_decision_instance", lambda _id: stored)
    assert channel_cards.card_text(1) == (
        "Waiting for your approval\n- Save 1 file to Partner folder\n> Bill.pdf\n> Also open to: dana@partner.example"
    )
    # Words can't pass for another action on the card.
    save["text"] = "Thanks!\n- Message Finance"
    stored.proposed_payload_json = json.dumps({"actions": [save]})
    assert channel_cards.card_text(1).endswith("> Thanks!\n> - Message Finance")


def test_a_card_of_actions_too_long_for_a_chat_is_not_approved_there(
    roster: SimpleNamespace, sends: list[Any],
) -> None:
    set_enabled(roster.principal, True, updated_by="t")
    card = _card(_session(FakeMailbox()), [_message(roster.teammate, "x" * channel_cards.CARD_MAX)])
    assert channel_cards.card_text(card.id) is None
    line = asyncio.run(channel_cards.resolve(card.id, get_person(roster.principal), approve=True))
    assert "too long" in line and sends == []
    # Skip still works.
    assert asyncio.run(channel_cards.resolve(card.id, get_person(roster.principal), approve=False)) == "Skipped."


def test_a_team_members_card_is_theirs_alone(
    roster: SimpleNamespace, sends: list[Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_enabled(roster.teammate, True, updated_by="t")
    card = _card(_teammate_session(), [_message(roster.principal)])
    with pytest.raises(HTTPException) as refused:
        asyncio.run(decisions_route.resolve_in_chat(card.id, get_person(roster.principal), approve=False))
    assert refused.value.status_code == 404
    assert get_decision_instance(card.id).status == "proposed"
    asyncio.run(decisions_route.resolve_in_chat(card.id, get_person(roster.teammate), approve=True))
    assert [i.get("text") for _, i, _ in sends] == ["Can you check the pilot dates?"]


def test_the_executive_is_told_where_the_card_is(roster: SimpleNamespace, telegram: None) -> None:
    from .test_action_cards import _propose

    set_enabled(roster.principal, True, updated_by="t")
    web = _session(FakeMailbox())
    assert "in this chat" in _propose(web, {"why": "x", "actions": [_message(roster.teammate)]})["note"]
    slack = _session(FakeMailbox())
    slack.origin_channel = "slack"
    note = _propose(slack, {"why": "x", "actions": [_message(roster.teammate, "two")]})["note"]
    assert "can't show it" in note and "in this chat and" not in note
    # A chat that shows cards, but this one is too long to show whole.
    long = _session(FakeMailbox())
    long.origin_channel, long.origin_channel_ref, long.caller_person_id = "telegram", "4242", roster.principal
    assert "in this chat and" in _propose(long, {"why": "x", "actions": [_message(roster.teammate, "three")]})["note"]
    long = _session(FakeMailbox())
    long.origin_channel, long.origin_channel_ref, long.caller_person_id = "telegram", "4242", roster.principal
    note = _propose(long, {"why": "x", "actions": [_message(roster.teammate, "x" * channel_cards.CARD_MAX)]})["note"]
    assert "too long to show in this chat" in note and "in this chat and" not in note


def test_the_channel_block_says_cards_have_buttons_only_where_they_do() -> None:
    plain = build_channel_context_block("telegram", ui_base_url="")
    assert "can ONLY be approved" in plain and "Approve and Skip" not in plain
    buttons = build_channel_context_block("telegram", ui_base_url="", cards_here=True)
    assert "Approve and Skip buttons" in buttons and "can ONLY be approved" not in buttons


def test_chat_collects_the_cards_a_reply_left() -> None:
    async def stream(**_: Any) -> Any:
        yield "Left you a card."
        yield {"type": "action_taken", "tool": "propose_actions", "decision_id": 5}
        yield {"type": "action_taken", "tool": "propose_actions", "decision_id": 5}
        yield {"type": "action_taken", "tool": "message_person", "decision_id": True}
        yield {"type": "activity", "decision_id": 6}

    executive = Executive.__new__(Executive)
    executive.stream_chat = stream  # type: ignore[method-assign]
    cards: list[int] = []
    reply = asyncio.run(executive.chat(user_message="hi", session=Session(), cards=cards))
    assert reply == "Left you a card." and cards == [5]


def test_a_meeting_is_booked_by_the_owner_or_its_approver(
    roster: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openexecutive.memory.decision_ledger import create_decision_instance

    booked: list[bool] = []

    async def book(instance: Any, payload: dict[str, Any], *, by_principal: bool) -> dict[str, Any]:
        booked.append(by_principal)
        return {"event_id": f"e{len(booked)}"}

    monkeypatch.setattr(decisions_route, "_execute_booking", book)

    def seed(approver: int | None, key: str, private: bool = False) -> int:
        return create_decision_instance(
            decision_class="meeting_scheduling", department="operations", originating_session_id=None,
            proposed_payload={"title": "Pilot review", "start": "2026-06-15T10:00:00", "end": "2026-06-15T11:00:00",
                              "attendee_emails": ["dana@northpeak.example"], "description": "",
                              **({"private": True} if private else {})},
            idempotency_key=key, gate_mode="propose", approver_person_id=approver, confidence=0.8,
        )

    owners = seed(roster.principal, "k1")
    assert channel_cards.card_text(owners) == (
        "Meeting to book\nPilot review\nMon 15 Jun, 10:00-11:00\nWith dana@northpeak.example"
    )
    assert asyncio.run(channel_cards.resolve(owners, get_person(roster.teammate), approve=True)).startswith("Not done")
    assert asyncio.run(channel_cards.resolve(owners, get_person(roster.principal), approve=True)) == "Approved."
    assert booked == [True] and get_decision_instance(owners).external_event_id == "e1"

    theirs = seed(roster.teammate, "k2")
    assert asyncio.run(channel_cards.resolve(theirs, get_person(roster.teammate), approve=True)) == "Approved."
    assert booked == [True, False]
    # The owner's private booking is theirs alone, even when it went to someone.
    private = seed(roster.teammate, "k3", private=True)
    assert asyncio.run(channel_cards.resolve(private, get_person(roster.teammate), approve=False)).startswith("Not done")
    assert get_decision_instance(private).status == "proposed"


class _FakeTelegram:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []

    async def post(self, url: str, json: dict[str, Any]) -> Any:  # noqa: A002 - httpx's keyword
        self.posts.append((url.rsplit("/", 1)[-1], json))
        return SimpleNamespace(is_error=False, status_code=200, text="")


@pytest.fixture
def fake_telegram(monkeypatch: pytest.MonkeyPatch) -> _FakeTelegram:
    fake = _FakeTelegram()
    monkeypatch.setattr(telegram_bot, "_get_http_client", lambda: fake)
    return fake


def test_a_card_goes_out_with_approve_and_skip(fake_telegram: _FakeTelegram) -> None:
    asyncio.run(telegram_bot.send_card(
        TOKEN, 4242, 7, "Meeting to book\nPilot review",
        options=(("Approve", "card:7:a:0000beef"), ("Skip", "card:7:s:0000beef")),
    ))
    [(method, body)] = fake_telegram.posts
    assert method == "sendMessage" and body["chat_id"] == 4242
    assert body["text"] == "Meeting to book\nPilot review" and "parse_mode" not in body
    assert body["reply_markup"] == {"inline_keyboard": [
        [{"text": "Approve", "callback_data": "card:7:a:0000beef"}],
        [{"text": "Skip", "callback_data": "card:7:s:0000beef"}],
    ]}


def test_a_tap_answers_audits_and_leaves_the_card_without_buttons(
    fake_telegram: _FakeTelegram, monkeypatch: pytest.MonkeyPatch,
) -> None:
    person = SimpleNamespace(id=3)
    monkeypatch.setattr("openexecutive.people.store.find_person_by_telegram_chat_id",
                        lambda chat: person if chat == "4242" else None)
    resolved: list[tuple[int, Any, bool]] = []

    async def resolve(decision_id: int, who: Any, *, approve: bool, allow: bool = False, shown_at: str | None = None, shown: str | None = None) -> str:
        resolved.append((decision_id, who, approve))
        return "Approved."

    # The card as its message shows it (no ledger in this test).
    monkeypatch.setattr(channel_cards, "card_text", lambda decision_id: "Meeting to book")
    monkeypatch.setattr(channel_cards, "version", lambda decision_id, text: "0000beef")
    monkeypatch.setattr(channel_cards, "resolve", resolve)
    audited: list[dict[str, Any]] = []
    monkeypatch.setattr("openexecutive.audit.log_event", lambda *a, **k: audited.append(k["details"]))

    def tap(chat_id: int, data: str) -> None:
        asyncio.run(telegram_bot._handle_card_tap(
            callback_id="q1", chat_id=chat_id, message_id=9, card_text="Meeting to book", data=data, token=TOKEN,
        ))

    tap(4242, "card:7:a:0000beef")
    assert resolved == [(7, person, True)]
    assert audited == [{"channel": "telegram", "chat_id": 4242, "decision_id": 7, "choice": "approve", "outcome": "done", "person_id": 3}]
    assert [m for m, _ in fake_telegram.posts] == ["answerCallbackQuery", "editMessageText"]
    edited = fake_telegram.posts[1][1]
    assert edited["text"] == "Meeting to book\n\nApproved." and "reply_markup" not in edited

    # Off the roster, or data that isn't a card's: answered, never resolved.
    fake_telegram.posts.clear()
    tap(999, "card:7:a:0000beef")
    tap(4242, "something:else")
    assert len(resolved) == 1 and len(audited) == 1
    assert [(m, b["text"]) for m, b in fake_telegram.posts] == [
        ("answerCallbackQuery", "This card is no longer there."),
        ("answerCallbackQuery", "This card is no longer there."),
    ]


def test_a_second_tap_while_the_first_is_booking_is_turned_away(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    async def slow(decision_id: int, person: Any, *, approve: bool, allow: bool = False, shown_at: str | None = None,
                   shown: str | None = None) -> None:
        calls.append(decision_id)
        await asyncio.sleep(0.05)

    monkeypatch.setattr(decisions_route, "resolve_in_chat", slow)

    async def both() -> list[str]:
        return list(await asyncio.gather(
            channel_cards.resolve(7, SimpleNamespace(id=1), approve=True),
            channel_cards.resolve(7, SimpleNamespace(id=1), approve=True),
        ))

    assert sorted(asyncio.run(both())) == ["Already on it.", "Approved."]
    assert calls == [7]
    assert asyncio.run(channel_cards.resolve(7, SimpleNamespace(id=1), approve=False)) == "Skipped."


def test_a_busy_tap_only_answers(fake_telegram: _FakeTelegram, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("openexecutive.people.store.find_person_by_telegram_chat_id", lambda chat: SimpleNamespace(id=3))

    async def busy(decision_id: int, who: Any, *, approve: bool, allow: bool = False, shown_at: str | None = None, shown: str | None = None) -> str:
        return channel_cards.BUSY

    # The card as its message shows it (no ledger in this test).
    monkeypatch.setattr(channel_cards, "card_text", lambda decision_id: "Meeting to book")
    monkeypatch.setattr(channel_cards, "version", lambda decision_id, text: "0000beef")
    monkeypatch.setattr(channel_cards, "resolve", busy)
    monkeypatch.setattr("openexecutive.audit.log_event", lambda *a, **k: None)
    asyncio.run(telegram_bot._handle_card_tap(
        callback_id="q1", chat_id=4242, message_id=9, card_text="Meeting to book", data="card:7:a:0000beef", token=TOKEN,
    ))
    assert [(m, b["text"]) for m, b in fake_telegram.posts] == [("answerCallbackQuery", "Already on it.")]


def test_a_taps_outcome_is_recorded_apart_from_its_choice() -> None:
    assert channel_cards.outcome_of("Approved.") == "done"
    assert channel_cards.outcome_of("Skipped.") == "done"
    assert channel_cards.outcome_of("Not done: This card was already handled.") == "refused"
    assert channel_cards.outcome_of(channel_cards.BUSY) == "busy"
    assert channel_cards.outcome_of(channel_cards.CHANGED) == "changed"


def test_a_card_that_can_teach_offers_on_its_own_next_time(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace as NS

    teaching = NS(id=7, decision_class="take_the_lead_action", approver_person_id=1,
                  proposed_payload_json=json.dumps({"kind": "training", "allow": {"key": "k", "label": "Message Sam"}}))
    plain = NS(id=8, decision_class="take_the_lead_action", approver_person_id=1,
               proposed_payload_json=json.dumps({"kind": "level"}))
    cards = {7: teaching, 8: plain}
    monkeypatch.setattr("openexecutive.memory.decision_ledger.get_decision_instance", lambda i, **k: cards.get(i))
    def labelled(card: int) -> tuple[tuple[str, str], ...]:
        return tuple((label, channel_cards.choice_of(data)) for label, data in channel_cards.card_options(card, ""))

    assert labelled(7) == (("Do it", "a"), ("Do it, and do this on its own next time", "n"), ("Skip", "s"))
    assert labelled(8) == (("Do it", "a"), ("Skip", "s"))


def test_a_tap_on_on_its_own_approves_and_allows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, bool, bool]] = []

    async def resolve(decision_id: int, who: Any, *, approve: bool, allow: bool = False, shown_at: str | None = None, shown: str | None = None) -> str:
        calls.append((decision_id, approve, allow))
        return "Approved."

    monkeypatch.setattr(channel_cards, "resolve", resolve)
    monkeypatch.setattr(channel_cards, "card_text", lambda i: "Message Sam")
    shown = channel_cards.version(7, "Message Sam")
    for choice in ("a", "n", "s"):
        tapped = channel_cards.Choice(7, choice, shown)
        assert asyncio.run(channel_cards.tap(tapped, SimpleNamespace(id=3))) == channel_cards.Tapped("Approved.")
    assert calls == [(7, True, False), (7, True, True), (7, False, False)]


def test_a_tap_on_a_card_that_says_something_else_now_shows_it_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """An older copy of a card (switched since, changed in the app) does
    nothing but show the card as it is now: what is done is always what the
    tapped card showed. A Skip is a skip."""
    calls: list[bool] = []

    async def resolve(decision_id: int, who: Any, *, approve: bool, allow: bool = False, shown_at: str | None = None, shown: str | None = None) -> str:
        calls.append(approve)
        return "Skipped."

    monkeypatch.setattr(channel_cards, "resolve", resolve)
    monkeypatch.setattr(channel_cards, "card_text", lambda i: "Reply ready to send\nfrom Ada now")
    monkeypatch.setattr(channel_cards, "card_options", lambda i, shown: (("Send from Ada, cc me", "new"),))
    monkeypatch.setattr(channel_cards, "_theirs", lambda i, who: who.id == 3)
    old = channel_cards.version(7, "Reply ready to send\nfrom you")
    for choice in ("a", "n", "x"):
        assert asyncio.run(channel_cards.tap(channel_cards.Choice(7, choice, old), SimpleNamespace(id=3))) == (
            channel_cards.Tapped(channel_cards.CHANGED, "Reply ready to send\nfrom Ada now", (("Send from Ada, cc me", "new"),))
        )
    assert calls == []
    # Someone else tapping an older copy isn't shown what it says now.
    assert asyncio.run(channel_cards.tap(channel_cards.Choice(7, "a", old), SimpleNamespace(id=4))) == (
        channel_cards.Tapped("Not done: this card is no longer there.")
    )
    assert asyncio.run(channel_cards.tap(channel_cards.Choice(7, "s", old), SimpleNamespace(id=3))).line == "Skipped."
    assert calls == [False]


def test_switching_who_a_reply_goes_from_shows_the_card_again(
    fake_telegram: _FakeTelegram, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("openexecutive.people.store.find_person_by_telegram_chat_id", lambda chat: SimpleNamespace(id=3))
    monkeypatch.setattr("openexecutive.audit.log_event", lambda *a, **k: None)
    again = channel_cards.Tapped(
        "Changed who it goes from. Check it, then send it.", "Reply ready to send\nFrom Ada now",
        (("Send from Ada, cc me", "card:7:a:1234abcd"), ("Send from me", "card:7:x:1234abcd"),
         ("Skip", "card:7:s:1234abcd")),
    )

    async def tap(tapped: channel_cards.Choice, who: Any) -> channel_cards.Tapped:
        assert tapped == channel_cards.Choice(7, "x", "0000beef")
        return again

    monkeypatch.setattr(channel_cards, "tap", tap)
    asyncio.run(telegram_bot._handle_card_tap(
        callback_id="q1", chat_id=4242, message_id=9, card_text="Reply ready to send", data="card:7:x:0000beef", token=TOKEN,
    ))
    edited = fake_telegram.posts[1][1]
    assert edited["text"] == "Reply ready to send\nFrom Ada now\n\nChanged who it goes from. Check it, then send it."
    assert [row[0]["callback_data"] for row in edited["reply_markup"]["inline_keyboard"]] == [
        "card:7:a:1234abcd", "card:7:x:1234abcd", "card:7:s:1234abcd",
    ]


def test_a_card_shown_only_in_part_changes_its_version_where_it_isnt_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    """A card lists the first ten it goes to; an eleventh added after it
    was posted makes it another card, though its words read the same."""
    card = SimpleNamespace(proposed_payload_json=json.dumps({"draft_to": [f"p{i}@x.example" for i in range(10)]}))
    monkeypatch.setattr("openexecutive.memory.decision_ledger.get_decision_instance", lambda i, **k: card)
    before = channel_cards.version(7, "Reply ready to send")
    card.proposed_payload_json = json.dumps({"draft_to": [f"p{i}@x.example" for i in range(11)]})
    assert channel_cards.version(7, "Reply ready to send") != before
    # Who is To and who Cc, which the card doesn't say, likewise.
    spec = {"draft_to": ["a@x.example", "b@x.example"], "draft_spec": {"to": ["a@x.example"], "cc": ["b@x.example"]}}
    card.proposed_payload_json = json.dumps(spec)
    before = channel_cards.version(7, "Reply ready to send")
    card.proposed_payload_json = json.dumps({**spec, "draft_spec": {"to": ["b@x.example"], "cc": ["a@x.example"]}})
    assert channel_cards.version(7, "Reply ready to send") != before


def test_a_tap_is_checked_against_when_its_card_was_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """A matching tap counts its card as shown when the card was read, so a
    change saved after it isn't."""
    seen: list[str | None] = []

    async def resolve(decision_id: int, who: Any, *, approve: bool, allow: bool = False, shown_at: str | None = None, shown: str | None = None) -> str:
        seen.append(shown_at)
        return "Sent."

    reads: list[str] = []

    def card_text(decision_id: int) -> str | None:
        reads.append(datetime.now(UTC).isoformat())
        return texts.pop(0)

    texts: list[str | None] = ["Reply ready to send"]
    monkeypatch.setattr(channel_cards, "resolve", resolve)
    monkeypatch.setattr(channel_cards, "card_text", card_text)
    shown = channel_cards.version(7, "Reply ready to send")
    asyncio.run(channel_cards.tap(channel_cards.Choice(7, "a", shown), SimpleNamespace(id=3)))
    assert seen[0] is not None and seen[0] <= reads[0]
    assert len(seen) == 1


@pytest.mark.parametrize("choice", ["a", "n", "x"])
def test_a_tap_on_a_card_no_chat_shows_whole_does_nothing(monkeypatch: pytest.MonkeyPatch, choice: str) -> None:
    async def done(*a: Any, **k: Any) -> Any:
        raise AssertionError("not done")

    monkeypatch.setattr(channel_cards, "card_text", lambda decision_id: None)
    monkeypatch.setattr(channel_cards, "_waits", lambda decision_id: True)
    monkeypatch.setattr(channel_cards, "_theirs", lambda decision_id, who: who.id == 3)
    monkeypatch.setattr(channel_cards, "resolve", done)
    monkeypatch.setattr(channel_cards, "_switch", done)
    tapped = asyncio.run(channel_cards.tap(channel_cards.Choice(7, choice, "0000beef"), SimpleNamespace(id=3)))
    assert tapped.line.startswith("Not done: this card changed. Open it in the app: ")
    assert tapped.line.endswith("/cards/7") and tapped.options == ()
    # Anyone else isn't told it waits.
    other = asyncio.run(channel_cards.tap(channel_cards.Choice(7, choice, "0000beef"), SimpleNamespace(id=4)))
    assert other.line == "Not done: this card is no longer there."


def test_a_tap_on_a_card_handled_meanwhile_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    async def resolve(decision_id: int, who: Any, *, approve: bool, allow: bool = False, shown_at: str | None = None, shown: str | None = None) -> str:
        return "This card was already handled."

    monkeypatch.setattr(channel_cards, "card_text", lambda decision_id: None)
    monkeypatch.setattr(channel_cards, "_waits", lambda decision_id: False)
    monkeypatch.setattr(channel_cards, "resolve", resolve)
    tapped = asyncio.run(channel_cards.tap(channel_cards.Choice(7, "a", "0000beef"), SimpleNamespace(id=3)))
    assert tapped.line == "This card was already handled."


def test_a_switch_goes_to_the_sender_its_card_showed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever the card says by the time the switch runs (another tab
    switched it meanwhile), the switch does what the tapped button said."""
    switched: list[bool] = []

    async def switch(decision_id: int, who: Any, *, executive: bool) -> channel_cards.Tapped:
        switched.append(executive)
        return channel_cards.Tapped("Changed who it goes from.")

    monkeypatch.setattr(channel_cards, "_look", lambda tapped, who: (None, True, True))
    monkeypatch.setattr(channel_cards, "_switch", switch)
    asyncio.run(channel_cards.tap(channel_cards.Choice(7, "x", "0000beef"), SimpleNamespace(id=3)))
    assert switched == [True]


def test_a_card_never_says_two_outcomes() -> None:
    shown_again = channel_cards.with_line("Reply to Sam\nHi Sam", channel_cards.SWITCHED)
    assert shown_again == f"Reply to Sam\nHi Sam\n\n{channel_cards.SWITCHED}"
    # Tapped again: the outcome takes the earlier line's place.
    assert channel_cards.with_line(shown_again, "Sent.") == "Reply to Sam\nHi Sam\n\nSent."
    assert channel_cards.with_line(channel_cards.CHANGED, "Sent.") == "Sent."
    assert channel_cards.with_line("", "Sent.") == "Sent."


def test_a_switch_holds_its_card_before_it_reads_it(monkeypatch: pytest.MonkeyPatch) -> None:
    held: list[bool] = []

    def read(decision_id: int, **k: Any) -> None:
        held.append(decision_id in channel_cards._resolving)

    monkeypatch.setattr("openexecutive.memory.decision_ledger.get_decision_instance", read)
    tapped = asyncio.run(channel_cards._switch(7, SimpleNamespace(id=3, email=""), executive=True))
    assert held == [True] and tapped.line == "Not done: this card is no longer there."
    assert 7 not in channel_cards._resolving


def test_how_to_say_words_on_a_card_goes_once_it_doesnt_wait_for_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(channel_cards, "takes_words", lambda decision_id: decision_id == 7)
    shown = channel_cards.with_tell_me(7, "Reply to Sam")
    assert shown == f"Reply to Sam\n\n{channel_cards.TELL_ME}"
    assert channel_cards.with_tell_me(8, "Meeting") == "Meeting"
    assert channel_cards.with_line(shown, "Sent.") == "Reply to Sam\n\nSent."
    again = channel_cards.with_tell_me(7, channel_cards.with_line("Reply to Sam", channel_cards.CHANGED))
    assert channel_cards.with_line(again, "Sent.") == "Reply to Sam\n\nSent."


def _reply_card(monkeypatch: pytest.MonkeyPatch, notes: list[str]) -> list[Any]:
    from openexecutive.delegation import answers

    taken: list[Any] = []
    card = SimpleNamespace(id=7, decision_class="delegation_reply", proposed_payload_json="{}")
    monkeypatch.setattr(channel_cards, "_look_said", lambda said, who: (None, True))
    monkeypatch.setattr("openexecutive.memory.decision_ledger.get_decision_instance", lambda i, **k: card)
    monkeypatch.setattr(answers, "conversation", lambda payload: [{"kind": "note", "text": n} for n in notes])
    monkeypatch.setattr(channel_cards, "card_text", lambda i: "Reply to Sam\nThursday works.")
    monkeypatch.setattr(channel_cards, "card_options", lambda i, shown: (("Send from me", "card:7:a:1111beef"),))
    return taken


def test_words_said_on_a_reply_card_change_it_and_show_it_again(monkeypatch: pytest.MonkeyPatch) -> None:
    from openexecutive.delegation import answers

    notes = ["I made it shorter."]
    taken = _reply_card(monkeypatch, notes)

    async def take(instance: Any, *, person: Any, caller: Any, resolver: int | None, text: str,
                   work: bool = False) -> answers.Taken:
        taken.append((caller, resolver, text, 7 in channel_cards._resolving))
        notes.append("I offered Thursday instead.")
        return answers.Taken(["Thursday works."])

    monkeypatch.setattr(answers, "take", take)
    who = SimpleNamespace(id=3, email="Sam@Co.example")
    said = asyncio.run(channel_cards.say(channel_cards.Choice(7, "a", "0000beef"), who, "offer Thursday"))
    assert said == channel_cards.Tapped(
        "I offered Thursday instead.", "Reply to Sam\nThursday works.", (("Send from me", "card:7:a:1111beef"),),
    )
    # As the person, holding the card while it changes.
    assert taken == [(ChatTap("sam@co.example"), 3, "offer Thursday", True)]
    assert 7 not in channel_cards._resolving


def test_words_a_reply_card_wrote_nothing_for_say_it_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The card changed under the words, so nothing was written: an older note isn't their answer."""
    from openexecutive.delegation import answers

    _reply_card(monkeypatch, ["I made it shorter."])

    async def take(instance: Any, **k: Any) -> answers.Taken:
        return answers.Taken([], go=True)

    monkeypatch.setattr(answers, "take", take)
    who = SimpleNamespace(id=3, email="sam@co.example")
    said = asyncio.run(channel_cards.say(channel_cards.Choice(7, "a", "0000beef"), who, "send it"))
    assert said == channel_cards.Tapped(
        "Not done: this card changed meanwhile. Look again.",
        "Reply to Sam\nThursday works.", (("Send from me", "card:7:a:1111beef"),),
    )


def test_words_for_a_reply_cards_work_point_to_it_in_the_app(monkeypatch: pytest.MonkeyPatch) -> None:
    from openexecutive.delegation import answers, card_work

    _reply_card(monkeypatch, [])
    result = [answers.Taken([], worked=True)]

    async def take(instance: Any, **k: Any) -> answers.Taken:
        return result[0]

    monkeypatch.setattr(answers, "take", take)
    who = SimpleNamespace(id=3, email="sam@co.example")
    said = asyncio.run(channel_cards.say(channel_cards.Choice(7, "a", "0000beef"), who, "make a spreadsheet of it"))
    assert said is not None and said.line.startswith("I'm on it. Follow along in the app: ")
    assert said.line.endswith("/cards/7") and said.options
    # The run going now hasn't let go yet: nothing taken, and it says so.
    result[0] = answers.Taken([], worked=True, started=False)
    monkeypatch.setattr(card_work, "running", lambda card_id: True)
    said = asyncio.run(channel_cards.say(channel_cards.Choice(7, "a", "0000beef"), who, "stop"))
    assert said == channel_cards.Tapped("Not done: I'm still finishing the last step. Try again in a moment.")


def test_words_on_a_card_already_being_decided_are_turned_away(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(channel_cards, "_look_said", lambda said, who: (None, True))
    channel_cards._resolving.add(7)
    try:
        said = asyncio.run(channel_cards.say(channel_cards.Choice(7, "a", "0000beef"), SimpleNamespace(id=3), "hi"))
    finally:
        channel_cards._resolving.discard(7)
    assert said == channel_cards.Tapped(channel_cards.BUSY)
