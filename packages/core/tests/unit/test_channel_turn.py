"""The one turn every chat channel takes (``integrations.channel_turn``)."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from openexecutive.integrations import channel_turn
from openexecutive.integrations.channel_turn import ChannelTurn


@pytest.fixture
def turn_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The turn's surroundings, stubbed; ``seen`` records what each was given."""
    seen: dict[str, Any] = {"history": [], "hydrated": [], "retrieve": [], "briefing": []}

    def retrieve(**kw: Any) -> str:
        seen["retrieve"].append(kw)
        return "KNOWLEDGE"

    def briefing(session: Any, *, is_dm: bool, person: Any) -> str:
        seen["briefing"].append((is_dm, person))
        return "BOARD" if is_dm else ""

    def hydrate(*, channel: str, channel_ref: str, user_message: str) -> str:
        seen["hydrated"].append((channel, channel_ref))
        return f"<context/>{user_message}"

    async def chat(**kw: Any) -> str:
        seen["chat"] = kw
        if kw["cards"] is not None:
            kw["cards"].append(7)
        return "the answer"

    monkeypatch.setattr("openexecutive.knowledge.retriever.retrieve", retrieve)
    monkeypatch.setattr("openexecutive.knowledge.retriever.chat_builtin_results", lambda: 4)
    monkeypatch.setattr("openexecutive.memory.episodic.format_for_prompt", lambda **_: "PROJECTS")
    monkeypatch.setattr(
        "openexecutive.memory.session_store.load_messages", lambda sid: list(seen["history"])
    )
    monkeypatch.setattr(
        "openexecutive.onboarding.profile_builder.load_or_create_profile",
        lambda: SimpleNamespace(is_empty=lambda: True),
    )
    monkeypatch.setattr(
        "openexecutive.integrations.channel_context.attach_briefing_context", briefing
    )
    monkeypatch.setattr(
        "openexecutive.integrations.inbound_hydration.hydrate_user_message", hydrate
    )
    monkeypatch.setattr("openexecutive.integrations.channel_cards.shows_cards", lambda s: True)
    monkeypatch.setattr("openexecutive.orchestrator.mcp_gateway.get_active_gateway", lambda: None)
    executive = MagicMock()
    executive.chat = chat
    monkeypatch.setattr(
        "openexecutive.orchestrator.executive.Executive", lambda **_: executive
    )
    return seen


def _turn(**over: Any) -> ChannelTurn:
    base: dict[str, Any] = {
        "channel": "slack",
        "channel_ref": "U1",
        "session_id": "slack:dm:U1",
        "own_text": "what changed?",
        "turn_text": "what changed?",
        "person": SimpleNamespace(id=5),
        "is_dm": True,
    }
    return ChannelTurn(**{**base, **over})


def test_the_turn_gets_the_session_knowledge_and_the_persons_board(turn_env: dict) -> None:
    result = asyncio.run(channel_turn.run(_turn()))

    chat = turn_env["chat"]
    session = chat["session"]
    assert (session.session_id, session.origin_channel, session.origin_channel_ref) == (
        "slack:dm:U1", "slack", "U1",
    )
    assert session.caller_person_id == 5 and chat["person_id"] == 5
    assert chat["retrieved_context"] == "KNOWLEDGE"
    assert chat["episodic_context"] == "PROJECTS"
    assert chat["briefing_context"] == "BOARD"
    assert "Slack" in chat["channel_context_block"]
    assert result.response == "the answer" and result.person_id == 5


def test_knowledge_is_searched_with_the_persons_own_words_and_capped(turn_env: dict) -> None:
    from openexecutive.config import get_settings

    asyncio.run(channel_turn.run(_turn(turn_text="[Ann]: what changed?", store="WARM")))

    [call] = turn_env["retrieve"]
    assert call == {
        "query": "what changed?",
        "max_chars": get_settings().knowledge_chat_max_chars,
        "n_builtin": 4,
        "store": "WARM",
    }


def test_the_conversation_so_far_is_loaded_and_marks_a_first_turn(turn_env: dict) -> None:
    assert asyncio.run(channel_turn.run(_turn())).first_turn is True

    turn_env["history"] = [{"role": "user", "content": "earlier"}]
    result = asyncio.run(channel_turn.run(_turn()))
    assert result.first_turn is False
    assert result.session.conversation_history == [{"role": "user", "content": "earlier"}]


def test_a_history_the_adapter_read_is_used_as_is(turn_env: dict) -> None:
    turn_env["history"] = [{"role": "user", "content": "stored"}]
    result = asyncio.run(channel_turn.run(_turn(history=[])))
    assert result.session.conversation_history == [] and result.first_turn is True


def test_earlier_outbound_context_reaches_the_model_only_with_a_reply_ref(turn_env: dict) -> None:
    asyncio.run(channel_turn.run(_turn()))
    assert turn_env["hydrated"] == []
    assert turn_env["chat"]["user_message"] == "what changed?"

    asyncio.run(channel_turn.run(_turn(reply_ref=("slack_dm", "U1"), model_text="FILE\n\nwhat changed?")))
    assert turn_env["hydrated"] == [("slack_dm", "U1")]
    assert turn_env["chat"]["user_message"] == "<context/>FILE\n\nwhat changed?"


def test_a_shared_conversation_gets_no_board(turn_env: dict) -> None:
    asyncio.run(channel_turn.run(_turn(is_dm=False)))
    assert turn_env["chat"]["briefing_context"] == ""


def test_cards_are_collected_only_where_the_adapter_shows_them(turn_env: dict) -> None:
    result = asyncio.run(channel_turn.run(_turn()))
    assert turn_env["chat"]["cards"] is None and result.cards == []

    result = asyncio.run(channel_turn.run(_turn(offers_cards=True)))
    assert result.cards == [7]
    assert "Approve" in turn_env["chat"]["channel_context_block"]

    with patch("openexecutive.integrations.channel_cards.shows_cards", lambda s: False):
        result = asyncio.run(channel_turn.run(_turn(offers_cards=True)))
    assert turn_env["chat"]["cards"] is None and result.cards == []


def test_what_the_adapter_vouches_for_lands_on_the_session(turn_env: dict) -> None:
    result = asyncio.run(
        channel_turn.run(
            _turn(
                session_facts={"speaker_verified": True, "shared_chat": True},
                co_present_person_ids=[8],
                memory_text="own words",
                image_blocks=[{"type": "image"}],
            )
        )
    )
    assert result.session.speaker_verified is True and result.session.shared_chat is True
    chat = turn_env["chat"]
    assert chat["co_present_person_ids"] == [8]
    assert chat["memory_text"] == "own words"
    assert chat["attachment_blocks"] == [{"type": "image"}]


def test_someone_off_people_takes_a_turn_with_no_caller(turn_env: dict) -> None:
    result = asyncio.run(channel_turn.run(_turn(person=None)))
    assert result.person_id is None and result.session.caller_person_id is None
    assert turn_env["chat"]["person_id"] is None


def test_the_adapters_scope_wraps_the_executives_call_only(turn_env: dict) -> None:
    order: list[str] = []

    @contextmanager
    def scope() -> Any:
        order.append("in")
        yield
        order.append("out")

    real_retrieve = turn_env["retrieve"]

    def retrieve(**kw: Any) -> str:
        order.append("retrieve")
        real_retrieve.append(kw)
        return ""

    with patch("openexecutive.knowledge.retriever.retrieve", retrieve):
        asyncio.run(channel_turn.run(_turn(chat_scope=scope)))
    assert order == ["retrieve", "in", "out"]


def test_save_turn_keeps_the_exchange_and_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(
        "openexecutive.memory.session_store.create_session",
        lambda sid, title, created, caller_person_id=None: calls.append(
            ("create", sid, title, caller_person_id)
        ),
    )
    monkeypatch.setattr(
        "openexecutive.memory.session_store.save_message",
        lambda sid, role, text, sender_person_id=None: calls.append(
            (role, text, sender_person_id)
        ),
    )
    monkeypatch.setattr(
        "openexecutive.memory.session_store.update_session_timestamp",
        lambda sid: calls.append(("touch", sid)),
    )
    kept = channel_turn.save_turn(
        session_id="s", title="Slack Ann", created_at="2026-01-01T00:00:00",
        owner_person_id=5, user_text="hi", assistant_text="hello", sender_person_id=5,
    )
    assert kept is True
    assert calls == [
        ("create", "s", "Slack Ann", 5), ("user", "hi", 5), ("assistant", "hello", None), ("touch", "s"),
    ]

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("disk")

    monkeypatch.setattr("openexecutive.memory.session_store.create_session", boom)
    assert channel_turn.save_turn(
        session_id="s", title="t", created_at="c", owner_person_id=None,
        user_text="u", assistant_text="a",
    ) is False


def test_a_chat_from_someone_off_people_shows_in_the_principals_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owners: list[Any] = []
    monkeypatch.setattr(
        "openexecutive.people.store.find_principal_person", lambda: SimpleNamespace(id=1)
    )
    monkeypatch.setattr(
        "openexecutive.memory.session_store.create_session",
        lambda sid, title, created, caller_person_id=None: owners.append(caller_person_id),
    )
    monkeypatch.setattr(
        "openexecutive.memory.session_store.save_message", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "openexecutive.memory.session_store.update_session_timestamp", lambda sid: None
    )
    for owner in (None, 9):
        assert channel_turn.save_turn(
            session_id="s", title="t", created_at="c", owner_person_id=owner,
            user_text="u", assistant_text="a",
        ) is True
    assert owners == [1, 9]


def test_reads_skip_knowledge_when_the_ask_is_not_the_persons_own(turn_env: dict) -> None:
    session = SimpleNamespace(session_id="one-on-one:1")
    person = SimpleNamespace(id=5)

    assert asyncio.run(
        channel_turn.reads(session, query=None, person=person, is_dm=True)
    ) == ("", "PROJECTS", "BOARD")
    assert turn_env["retrieve"] == []

    assert asyncio.run(
        channel_turn.reads(session, query="what changed?", person=person, is_dm=True)
    ) == ("KNOWLEDGE", "PROJECTS", "BOARD")
    assert turn_env["briefing"] == [(True, person), (True, person)]
