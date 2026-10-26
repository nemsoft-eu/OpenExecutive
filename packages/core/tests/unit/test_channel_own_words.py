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
import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

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
            token="t", attachment_file_ids=[("f1", "notes.pdf", "application/pdf", False)],
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


_BEN = SimpleNamespace(
    id=7, full_name="Ben Teammate", is_principal=False, archived=False, kind="teammate",
    email="ben@acme.example", role="Ops",
)


def _discord_on_message() -> Any:
    """The real ``on_message`` handler, taken off a bot that never connects.

    ``@bot.event`` sets the handler as an attribute of the bot, so this is the
    function discord.py calls for every message, not a copy of its body."""
    from openexecutive.config import get_settings
    from openexecutive.integrations import discord_bot

    settings = get_settings().model_copy(update={"discord_bot_token": "t"})
    with (
        patch("openexecutive.config.get_settings", return_value=settings),
        # Building the bot registers its replayer in a module-level dict;
        # keep that out of later tests.
        patch.dict("openexecutive.integrations.roster_intake._REPLAYERS"),
    ):
        bot = discord_bot.create_discord_bot()
    bot.process_commands = AsyncMock()
    return bot.on_message


def _discord_dm(content: str, filenames: tuple[str, ...] = ()) -> Any:
    import discord

    channel = MagicMock(spec=discord.DMChannel)
    channel.id = 4001
    channel.send = AsyncMock()
    return SimpleNamespace(
        content=content, id=5001, channel=channel, mentions=[], reply=AsyncMock(),
        author=SimpleNamespace(id=1002, bot=False, display_name="Ben", name="ben"),
        attachments=[
            SimpleNamespace(
                url=f"https://cdn.example/{name}", filename=name,
                content_type="application/pdf", size=10,
            )
            for name in filenames
        ],
    )


def test_discord_on_message_keeps_attachment_text_and_backstory_out_of_memory_text() -> None:
    on_message = _discord_on_message()
    captured: dict[str, Any] = {}
    with (
        _adapter_stubs(captured),
        patch("openexecutive.people.store.find_person_by_discord_id", return_value=_BEN),
        patch(
            "openexecutive.integrations.attachments.process_attachments",
            new=AsyncMock(return_value=(f"[Attached: notes.pdf]\nSee {PLANTED}", [])),
        ),
    ):
        asyncio.run(on_message(_discord_dm("Summarise this", ("notes.pdf",))))
    assert PLANTED in captured["user_message"]
    assert BACKSTORY in captured["user_message"]
    assert captured["memory_text"] == "Summarise this\n\n(Attached files: notes.pdf)"
    assert captured["fetch_own_urls"] is True


@pytest.mark.parametrize("rostered", [True, False])
def test_discord_on_message_opens_pasted_links_only_for_someone_on_the_roster(
    rostered: bool,
) -> None:
    """Discord's roster gate drops an unknown sender before the Executive, so
    their link reaches no ``web_fetch`` at all; the message is held instead."""
    from openexecutive.integrations import discord_bot

    on_message = _discord_on_message()
    captured: dict[str, Any] = {}
    with (
        _adapter_stubs(captured),
        patch(
            "openexecutive.people.store.find_person_by_discord_id",
            return_value=_BEN if rostered else None,
        ),
        patch.object(discord_bot, "_hold_unknown_sender", new=AsyncMock()) as held,
    ):
        asyncio.run(on_message(_discord_dm(f"Read {PLANTED}")))
    if rostered:
        assert captured["memory_text"] == f"Read {PLANTED}"
        assert captured["fetch_own_urls"] is True
        held.assert_not_awaited()
    else:
        assert captured == {}
        held.assert_awaited_once()


def test_discord_on_message_holds_an_unknown_senders_words_without_their_file() -> None:
    """Off the roster, the file is neither downloaded nor read, so the held
    message (replayed as all theirs once the principal confirms them) is only
    what they typed."""
    from openexecutive.integrations import discord_bot

    on_message = _discord_on_message()
    captured: dict[str, Any] = {}
    process = AsyncMock(return_value=(f"[Attached: notes.pdf]\nSee {PLANTED}", []))
    with (
        _adapter_stubs(captured),
        patch("openexecutive.people.store.find_person_by_discord_id", return_value=None),
        patch("openexecutive.integrations.attachments.process_attachments", new=process),
        patch.object(discord_bot, "_hold_unknown_sender", new=AsyncMock()) as held,
    ):
        asyncio.run(on_message(_discord_dm("Summarise this", ("notes.pdf",))))
    process.assert_not_awaited()
    assert captured == {}
    assert held.await_args.kwargs["text"] == "Summarise this"


@contextmanager
def _cli_stubs(calls: list[dict[str, Any]]) -> Iterator[None]:
    """What the CLI calls on its way to ``Executive.stream_chat``. Each call's
    arguments are recorded as the real method would bind them, defaults
    included: the CLI passes no ``fetch_own_urls`` and relies on the default."""
    from openexecutive.orchestrator.executive import Executive

    signature = inspect.signature(Executive.stream_chat)

    class _Exec:
        _THINKING = Executive._THINKING

        def __init__(self, **_kw: Any) -> None:
            pass

        async def stream_chat(self, **kwargs: Any) -> Any:
            bound = signature.bind(self, **kwargs)
            bound.apply_defaults()
            calls.append(dict(bound.arguments))
            yield "ok"

    with (
        patch("openexecutive.orchestrator.executive.Executive", new=_Exec),
        # Non-empty, so a turn that packed the retrieved knowledge into
        # memory_text would show.
        patch("openexecutive.knowledge.retriever.retrieve", return_value=f"See {PLANTED}"),
        patch("openexecutive.memory.episodic.format_for_prompt", return_value=""),
        patch(
            "openexecutive.onboarding.profile_builder.load_or_create_profile",
            return_value=SimpleNamespace(is_empty=lambda: True),
        ),
    ):
        yield


def test_cli_ask_hands_over_the_question_as_own_words_with_its_links_fetchable() -> None:
    from openexecutive.cli import cli

    calls: list[dict[str, Any]] = []
    with _cli_stubs(calls):
        result = CliRunner().invoke(cli, ["ask", f"Read {PLANTED}"])
    assert result.exit_code == 0, result.output
    [call] = calls
    assert call["memory_text"] == f"Read {PLANTED}"
    assert call["fetch_own_urls"] is True  # the operator at the terminal


def test_cli_chat_hands_over_each_typed_line_as_own_words_with_its_links_fetchable() -> None:
    from rich.prompt import Prompt

    from openexecutive.cli import cli

    calls: list[dict[str, Any]] = []
    with (
        _cli_stubs(calls),
        patch.object(Prompt, "ask", side_effect=[f"Read {PLANTED}", "And then?", "exit"]),
    ):
        result = CliRunner().invoke(cli, ["chat"])
    assert result.exit_code == 0, result.output
    assert [c["memory_text"] for c in calls] == [f"Read {PLANTED}", "And then?"]
    assert [c["fetch_own_urls"] for c in calls] == [True, True]
