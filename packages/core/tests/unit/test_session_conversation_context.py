"""Session.render_conversation_context — the specialist's only view of the chat.

Specialists receive no message history and no company profile: `BaseAgent.analyze`
composes department memory, past decisions, failure cases, retrieved knowledge,
this text, and the query. So whatever this renderer drops is invisible to every
specialist in a fan-out.
"""
from __future__ import annotations

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-not-used")

from openexecutive.orchestrator.session import Session  # noqa: E402


def test_tail_carries_the_subject_a_follow_up_omits() -> None:
    """The turn that motivated issue #12, reduced to its essentials.

    Turn 2 asks what to change "if we go ahead with the 30% increase" and never
    names the product or the price — both live in turn 1. Before the tail
    existed, every specialist got only that second message.
    """
    session = Session()
    session.add_user_message(
        "We are considering raising CIAO's subscription price by 30% next quarter."
    )
    session.add_assistant_message("Here is the pricing analysis, at EUR 6/employee/month.")

    rendered = session.render_conversation_context(
        "What would Legal need to change if we go ahead with the 30% increase?"
    )

    assert "CIAO" in rendered
    assert "EUR 6/employee/month" in rendered
    assert "30% increase" in rendered
    assert rendered.startswith("User: We are considering")


def test_question_survives_whether_attachments_are_appended_or_prepended() -> None:
    """The channels disagree about where extracted document text goes.

    `api/routes/chat.py` appends it after the typed question; `telegram_bot.py`
    and `discord_bot.py` prepend it. Truncating from either single end would
    keep the question on some channels and destroy it on others, so both ends
    are kept. Either half of this test fails if that becomes a one-ended slice.
    """
    question = "Should we sign this vendor contract?"
    blob = "X" * 50_000
    session = Session()

    appended = session.render_conversation_context(f"{question}\n\n{blob}")
    prepended = session.render_conversation_context(f"{blob}\n\n{question}")

    assert question in appended, "web-route ordering lost the question"
    assert question in prepended, "telegram/discord ordering lost the question"
    for rendered in (appended, prepended):
        assert rendered.count("X") < 2_100, "document blob was not bounded"


def test_total_cap_sheds_the_oldest_turn_and_keeps_the_current_message() -> None:
    """The cap slices from the tail, so the current message always survives.

    Passed explicitly and small: with the default per-part caps the joined
    result maxes out around 5.6k, so a 6,000-char default cap can never fire
    and asserting against it would pass even with the slice deleted.
    """
    session = Session()
    session.add_user_message("OLDEST_USER_TURN")
    session.add_assistant_message("older assistant reply")

    rendered = session.render_conversation_context(
        "the current question", total_max_chars=30
    )

    assert len(rendered) == 30
    assert rendered.endswith("the current question")
    assert "OLDEST_USER_TURN" not in rendered


def test_window_keeps_two_exchanges_and_drops_older_ones() -> None:
    """Older turns cost prefill on every specialist and are rarely referenced.

    The window is two *exchanges* (four messages), so a session must be three
    exchanges deep before anything is dropped — hence three here, not two.
    """
    session = Session()
    session.add_user_message("ANCIENT_TURN")
    session.add_assistant_message("ancient reply")
    session.add_user_message("MIDDLE_TURN")
    session.add_assistant_message("middle reply")
    session.add_user_message("RECENT_TURN")
    session.add_assistant_message("recent reply")

    rendered = session.render_conversation_context("now")

    assert "RECENT_TURN" in rendered
    assert "MIDDLE_TURN" in rendered
    assert "ANCIENT_TURN" not in rendered


def test_assistant_block_list_content_is_flattened() -> None:
    """A cached penultimate assistant turn is a block list, not a str.

    `_build_messages` wraps it for prompt caching, so the renderer must not
    stringify the raw list into the specialist's context.
    """
    session = Session()
    session.add_user_message("the question")
    session.add_assistant_message(
        [{"type": "text", "text": "THE_ANSWER", "cache_control": {"type": "ephemeral"}}]
    )

    rendered = session.render_conversation_context("follow up")

    assert "THE_ANSWER" in rendered
    assert "cache_control" not in rendered
    assert "'type':" not in rendered


def test_empty_session_renders_just_the_current_message() -> None:
    assert Session().render_conversation_context("first question") == "User: first question"
