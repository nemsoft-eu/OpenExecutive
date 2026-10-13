"""Every chat channel hands the Executive the person's own words as
``memory_text``, apart from what it packs into the prompt.

The prompt carries more than the person wrote: extracted attachment text, and
on a DM the backstory of an outbound message from another session
(``inbound_hydration``). ``memory_text`` is what peer memory records as their
words, and the only text whose URLs the client ``web_fetch`` may open — so a
channel that leaves it None gets no fetchable URL, and one that passed the
prompt would make a URL inside a document fetchable.
"""
from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

PLANTED = "https://planted.example/collect"
BACKSTORY = f"<outbound_reply_context>see {PLANTED}</outbound_reply_context>"


@pytest.fixture(autouse=True)
def _no_audit_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("openexecutive.audit.log_event", lambda *a, **kw: None, raising=False)


@contextmanager
def _adapter_stubs(captured: dict[str, Any]) -> Iterator[None]:
    """Everything a chat adapter calls on its way to Executive.chat, with
    the DM hydration adding a backstory that holds a URL."""

    class _Exec:
        def __init__(self, **_kw: Any) -> None:
            pass

        async def chat(self, **kwargs: Any) -> str:
            captured.update(kwargs)
            return "ok"

    with (
        patch("openexecutive.orchestrator.executive.Executive", new=_Exec),
        patch("openexecutive.knowledge.retriever.retrieve", return_value=""),
        patch("openexecutive.memory.episodic.format_for_prompt", return_value=""),
        patch("openexecutive.memory.session_store.load_messages", return_value=[]),
        patch("openexecutive.memory.session_store.create_session"),
        patch("openexecutive.memory.session_store.save_message"),
        patch("openexecutive.memory.session_store.update_session_timestamp"),
        patch(
            "openexecutive.onboarding.profile_builder.load_or_create_profile",
            return_value=SimpleNamespace(is_empty=lambda: True),
        ),
        patch("openexecutive.alerts.pipeline.schedule_evaluation"),
        patch("openexecutive.integrations.channel_context.attach_briefing_context", return_value=""),
        patch(
            "openexecutive.integrations.inbound_hydration.hydrate_user_message",
            new=lambda **kw: f"{BACKSTORY}\n\n{kw['user_message']}",
        ),
        patch(
            "openexecutive.workflows.inbound_resolver.resolve_inbound_message",
            new=AsyncMock(return_value=None),
        ),
        patch("openexecutive.orchestrator.mcp_gateway.get_active_gateway", return_value=None),
    ):
        yield


def test_telegram_keeps_attachment_text_and_backstory_out_of_memory_text() -> None:
    from openexecutive.integrations import telegram_bot

    captured: dict[str, Any] = {}
    with (
        _adapter_stubs(captured),
        patch("openexecutive.people.store.find_person_by_telegram_chat_id", return_value=None),
        patch.object(telegram_bot, "send_message", new=AsyncMock(return_value=None)),
        patch.object(telegram_bot, "_get_telegram_file_bytes", new=AsyncMock(return_value=("p", b"x"))),
        patch(
            "openexecutive.integrations.attachments.build_attachment_output",
            new=AsyncMock(return_value=(f"[Attached: notes.pdf]\nSee {PLANTED}", [])),
        ),
    ):
        asyncio.run(telegram_bot._process_and_reply(
            message_text="Summarise this", sender_name="Ben", chat_id=5002, message_id=1,
            token="t", attachment_file_ids=[("f1", "notes.pdf", "application/pdf")],
        ))
    assert PLANTED in captured["user_message"]
    assert captured["memory_text"] == "Summarise this\n\n(Attached files: notes.pdf)"


def test_telegram_without_files_records_the_typed_words() -> None:
    from openexecutive.integrations import telegram_bot

    captured: dict[str, Any] = {}
    with (
        _adapter_stubs(captured),
        patch("openexecutive.people.store.find_person_by_telegram_chat_id", return_value=None),
        patch.object(telegram_bot, "send_message", new=AsyncMock(return_value=None)),
    ):
        asyncio.run(telegram_bot._process_and_reply(
            message_text="Any news?", sender_name="Ben", chat_id=5002, message_id=1, token="t",
        ))
    assert captured["user_message"].startswith(BACKSTORY)
    assert captured["memory_text"] == "Any news?"


@pytest.mark.parametrize("rostered", [True, False])
def test_telegram_opens_pasted_links_only_for_someone_on_the_roster(rostered: bool) -> None:
    """A group chat reaches the Executive from anyone in it, roster or not."""
    from openexecutive.integrations import telegram_bot

    person = SimpleNamespace(id=7) if rostered else None
    captured: dict[str, Any] = {}
    with (
        _adapter_stubs(captured),
        patch("openexecutive.people.store.find_person_by_telegram_chat_id", return_value=person),
        patch.object(telegram_bot, "send_message", new=AsyncMock(return_value=None)),
    ):
        asyncio.run(telegram_bot._process_and_reply(
            message_text=f"Read {PLANTED}", sender_name="Ben", chat_id=5002, message_id=1,
            token="t",
        ))
    assert captured["memory_text"] == f"Read {PLANTED}"
    assert captured["fetch_own_urls"] is rostered


@pytest.mark.parametrize(("own_words", "expected"), [
    ("Summarise this\n\n(Attached files: notes.pdf)", "Summarise this\n\n(Attached files: notes.pdf)"),
    (None, "Summarise this"),  # a slash command or replay: the text is all theirs
])
def test_discord_hands_over_own_words_never_the_hydrated_prompt(
    own_words: str | None, expected: str
) -> None:
    from openexecutive.integrations import discord_bot

    person = SimpleNamespace(
        id=7, full_name="Ben Teammate", is_principal=False, archived=False, kind="teammate",
        email="ben@acme.example", role="Ops",
    )
    captured: dict[str, Any] = {}
    with (
        _adapter_stubs(captured),
        patch("openexecutive.people.store.find_person_by_discord_id", return_value=person),
    ):
        asyncio.run(discord_bot._handle_message(
            text="Summarise this", own_words=own_words, discord_user_id="1002",
            discord_channel="dm-1", message_id="m1", thread_id=None, send_fn=AsyncMock(),
            is_dm=True, session_id="discord:dm:1002", session_title="DM",
        ))
    assert BACKSTORY in captured["user_message"]
    assert captured["memory_text"] == expected
    assert captured["fetch_own_urls"] is True  # past the roster gate
