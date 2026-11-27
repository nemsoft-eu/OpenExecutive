"""One Executive turn for a message that arrived on a chat channel.

Slack, Discord, Telegram and Google Chat each used to put a turn together by
hand: build the session, load the
conversation so far, look up knowledge and the active projects, add the
briefing block and any context of a message the Executive sent this person
earlier, run the Executive, then keep the exchange. The copies drifted (one
had no history, some capped knowledge and some did not, some blocked the event
loop), and a fix in one never reached the rest.

This module is that sequence, once. A channel adapter decides what only it
knows (who sent the message, which conversation it belongs to, how a reply is
posted) and describes the turn with a ``ChannelTurn``; ``run`` takes the turn
and ``save_turn`` keeps it. Nothing here decides whether a step runs or waits:
that stays in ``orchestrator.levels``, which every turn passes through inside
``Executive.chat``.

Everything it calls is imported when ``run`` is called, from the module that
owns it, so a test that patches ``knowledge.retriever.retrieve`` or
``orchestrator.executive.Executive`` patches every channel at once.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ChannelTurn:
    """What a channel adapter knows about one inbound message."""

    # The channel's key ("slack", "discord", "telegram", "google_chat"):
    # names the surface on the session and in the `<channel>` block.
    channel: str
    # Where on that channel the conversation is (a user id, a chat, a space).
    channel_ref: str
    session_id: str
    # The person's own words: what knowledge is searched with.
    own_text: str
    # What the turn reads and the history keeps: the words with the message
    # they answer quoted in front, or the speaker's name in a shared thread.
    turn_text: str
    # The sender on the People list, or None for someone who isn't on it.
    person: Any | None
    # A conversation between the Executive and this person alone.
    is_dm: bool
    # What the model reads when that is more than ``turn_text`` (the text of
    # attached files in front). Never kept in the history.
    model_text: str | None = None
    # The person's own words for the passes that remember the turn, when
    # ``turn_text`` carries more than those (a quote, a file's text).
    memory_text: str | None = None
    # ``web_fetch`` may open the URLs in ``memory_text``: set only for a
    # sender the adapter has checked is on the roster, and never in a chat
    # anyone can write in under another person's row.
    fetch_own_urls: bool = False
    # (channel key, ref) to look up a message the Executive sent this person
    # earlier, so their reply arrives with its backstory. Direct messages
    # only: a shared conversation must never pull one person's context in.
    reply_ref: tuple[str, str] | None = None
    image_blocks: list[dict[str, Any]] | None = None
    co_present_person_ids: list[int] | None = None
    # The adapter can show the cards a reply leaves, with Approve and Skip.
    # They are shown only where ``channel_cards.shows_cards`` also agrees.
    offers_cards: bool = False
    # Session facts only the adapter can vouch for, by attribute name:
    # ``speaker_verified``, ``private_chat``, ``shared_chat``,
    # ``others_spoke_last``.
    session_facts: dict[str, bool] = field(default_factory=dict)
    # The conversation so far, when the adapter has already read it.
    history: list[dict[str, Any]] | None = None
    # A warm knowledge store to search with, when the adapter holds one.
    store: Any | None = None
    # Opens a context the Executive's own call runs inside, when the adapter
    # has a context variable of its own to bind for that call (a channel
    # added outside this package can need one). Who is asking never goes
    # here: that is ``person``, which lands on the session.
    chat_scope: Callable[[], AbstractContextManager[Any]] | None = None


@dataclass
class TurnResult:
    session: Any
    response: str
    # The approval cards the reply left, for the adapter to show under it.
    cards: list[int]
    # Nothing had been said in this conversation before this message.
    first_turn: bool
    person_id: int | None


async def reads(
    session: Any,
    *,
    query: str | None,
    person: Any | None,
    is_dm: bool,
    store: Any | None = None,
) -> tuple[str, str, str]:
    """What a conversation's turn reads before the Executive runs: knowledge
    matched to ``query`` (none for a ``None`` query), the active projects, and
    the person's open cards (in their own direct conversation only, which
    also records on ``session`` which cards the turn may answer).

    Every step blocks (a vector search, SQLite reads), so none runs on the
    event loop every other conversation shares. Knowledge is capped the same
    everywhere: uncapped, a turn carried ~25k characters of excerpts, paid on
    every model call of the turn."""
    from openexecutive.config import get_settings
    from openexecutive.integrations.channel_context import attach_briefing_context
    from openexecutive.knowledge.retriever import chat_builtin_results, retrieve
    from openexecutive.memory.episodic import format_for_prompt

    async def knowledge() -> str:
        if query is None:
            return ""
        kwargs: dict[str, Any] = {
            "query": query,
            "max_chars": get_settings().knowledge_chat_max_chars,
            "n_builtin": chat_builtin_results(),
        }
        if store is not None:
            kwargs["store"] = store
        return await asyncio.to_thread(retrieve, **kwargs)

    retrieved, projects = await asyncio.gather(
        knowledge(),
        asyncio.to_thread(format_for_prompt, session_id=session.session_id),
    )
    board = await asyncio.to_thread(attach_briefing_context, session, is_dm=is_dm, person=person)
    return retrieved, projects, board


async def run(turn: ChannelTurn) -> TurnResult:
    """Take the Executive's turn on ``turn``. Raises what the turn raises:
    the adapter owns the apology, because only it can post one."""
    from openexecutive.integrations.channel_cards import shows_cards
    from openexecutive.integrations.channel_context import build_channel_context_block
    from openexecutive.memory.session_store import load_messages
    from openexecutive.onboarding.profile_builder import load_or_create_profile
    from openexecutive.orchestrator.executive import Executive
    from openexecutive.orchestrator.mcp_gateway import get_active_gateway
    from openexecutive.orchestrator.session import Session

    person_id: int | None = turn.person.id if turn.person is not None else None

    profile = await asyncio.to_thread(load_or_create_profile)
    session = Session(
        session_id=turn.session_id,
        company_profile=profile if not profile.is_empty() else None,
        # Where this conversation is happening, so a step that waits mid-turn
        # can record that the person's next reply here answers it.
        origin_channel=turn.channel,
        origin_channel_ref=turn.channel_ref,
    )
    session.caller_person_id = person_id
    for name, value in turn.session_facts.items():
        setattr(session, name, value)

    history = turn.history
    if history is None:
        history = await asyncio.to_thread(load_messages, turn.session_id)
    if history:
        session.conversation_history = history

    retrieved_context, episodic_context, briefing_context = await reads(
        session, query=turn.own_text, person=turn.person, is_dm=turn.is_dm, store=turn.store
    )

    # Given to the model only: the block is consumed once, so it must not
    # reach the history and replay on every later turn.
    user_message = turn.model_text if turn.model_text is not None else turn.turn_text
    if turn.reply_ref is not None:
        from openexecutive.integrations.inbound_hydration import hydrate_user_message

        user_message = await asyncio.to_thread(
            hydrate_user_message,
            channel=turn.reply_ref[0],
            channel_ref=turn.reply_ref[1],
            user_message=user_message,
        )

    cards_here = turn.offers_cards and shows_cards(session)
    cards: list[int] = []

    with turn.chat_scope() if turn.chat_scope is not None else nullcontext():
        response = await Executive(mcp_gateway=get_active_gateway()).chat(
            user_message=user_message,
            session=session,
            retrieved_context=retrieved_context,
            episodic_context=episodic_context,
            attachment_blocks=turn.image_blocks or None,
            briefing_context=briefing_context,
            channel_context_block=build_channel_context_block(
                turn.channel, cards_here=cards_here
            ),
            cards=cards if cards_here else None,
            person_id=person_id,
            co_present_person_ids=turn.co_present_person_ids or None,
            memory_text=turn.memory_text,
            fetch_own_urls=turn.fetch_own_urls,
        )
    return TurnResult(
        session=session,
        response=response,
        cards=cards,
        first_turn=not history,
        person_id=person_id,
    )


def owner_or_principal(person_id: int | None) -> int | None:
    """Whose chat list a channel conversation shows in: the sender's, or the
    principal's when the sender isn't on People, so it is never left with no
    owner and invisible."""
    if person_id is not None:
        return person_id
    from openexecutive.people.store import find_principal_person

    principal = find_principal_person()
    return principal.id if principal is not None else None


def save_turn(
    *,
    session_id: str,
    title: str,
    created_at: str,
    owner_person_id: int | None,
    user_text: str,
    assistant_text: str,
    sender_person_id: int | None = None,
) -> bool:
    """Keep one exchange in the conversation's history; False when it
    couldn't. The conversation shows in ``owner_person_id``'s chat list, or
    the principal's when the sender isn't on People. Synchronous SQLite. Never raises: the person already has their
    answer, so a failure here must not turn into the error reply."""
    from openexecutive.memory.session_store import (
        create_session,
        save_message,
        update_session_timestamp,
    )

    try:
        create_session(
            session_id, title, created_at,
            caller_person_id=owner_or_principal(owner_person_id),
        )
        save_message(session_id, "user", user_text, sender_person_id=sender_person_id)
        save_message(session_id, "assistant", assistant_text)
        update_session_timestamp(session_id)
    except Exception:
        logger.exception("failed to persist turn for session %s", session_id)
        return False
    return True


__all__ = ["ChannelTurn", "TurnResult", "owner_or_principal", "reads", "run", "save_turn"]
