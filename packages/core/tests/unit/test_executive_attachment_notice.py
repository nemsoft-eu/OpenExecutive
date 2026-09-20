"""Image attachments reach the Executive alone, so a consult must relay them.

`build_attachment_output` returns NO text for an image — only a vision block —
so nothing about an image lands in `user_message`. A specialist's entire view is
the rendered conversation tail built from that string, which carries text and
never content blocks. Without the `<attachment_notice>` instruction, "have
Finance analyse this forecast" sends the CFO neither the image nor any
description of it, and it answers from priors while the Executive synthesizes
the result as authoritative specialist input (PR #18, Codex P1).
"""
from __future__ import annotations

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-not-used")

_IMAGE_BLOCK = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
}


def _user_turn_texts(messages: list[dict]) -> list[str]:
    """Collect the text blocks of the final (user) turn."""
    user = messages[-1]
    assert user["role"] == "user"
    return [p["text"] for p in user["content"] if p.get("type") == "text"]


def test_notice_present_when_an_image_is_attached() -> None:
    from openexecutive.orchestrator.executive import Executive
    from openexecutive.orchestrator.session import Session

    exec_ = Executive()
    session = Session(session_id="t-att-1")
    messages = exec_._build_messages(
        session,
        "Have Finance analyze this forecast",
        attachment_blocks=[_IMAGE_BLOCK],
    )
    texts = _user_turn_texts(messages)
    assert any("<attachment_notice>" in t for t in texts), texts
    # The instruction must name the mechanism, not merely mention attachments:
    # the model has to know the specialist cannot see the image and that `query`
    # is the channel that reaches it.
    notice = next(t for t in texts if "<attachment_notice>" in t)
    assert "cannot see" in notice
    assert "`query`" in notice


def test_notice_absent_without_attachments() -> None:
    from openexecutive.orchestrator.executive import Executive
    from openexecutive.orchestrator.session import Session

    exec_ = Executive()
    session = Session(session_id="t-att-2")
    messages = exec_._build_messages(session, "plain question")
    texts = _user_turn_texts(messages)
    assert not any("<attachment_notice>" in t for t in texts), texts


def test_block_order_is_image_then_notice_then_question() -> None:
    """Pins the full documented order, not merely "image is not last".

    Ordering is deliberate: the model sees visual context, then the instruction
    about relaying it, then the question. An earlier version of this test only
    asserted the image was not the final block, which a notice inserted BEFORE
    the image would also satisfy.
    """
    from openexecutive.orchestrator.executive import Executive
    from openexecutive.orchestrator.session import Session

    exec_ = Executive()
    session = Session(session_id="t-att-3")
    messages = exec_._build_messages(
        session, "what does this show?", attachment_blocks=[_IMAGE_BLOCK]
    )
    content = messages[-1]["content"]

    image_at = next(i for i, p in enumerate(content) if p.get("type") == "image")
    notice_at = next(
        i
        for i, p in enumerate(content)
        if p.get("type") == "text" and "<attachment_notice>" in p["text"]
    )
    question_at = next(
        i
        for i, p in enumerate(content)
        if p.get("type") == "text" and p["text"] == "what does this show?"
    )

    assert image_at < notice_at < question_at, [p.get("type") for p in content]
    # The question stays last so nothing separates it from the model's turn.
    assert question_at == len(content) - 1


def test_the_notice_is_not_persisted_into_the_specialist_tail() -> None:
    """The notice is an instruction to the Executive, not conversation content.

    `stream_chat` stores the raw `user_message`, so the notice must not reach
    `render_conversation_context` — a specialist receiving "the attached images
    are visible to you alone" would be told about evidence it cannot see, and
    the text would cost fan-out prefill on every later turn.
    """
    from openexecutive.orchestrator.session import Session

    session = Session(session_id="t-att-4")
    session.add_user_message("Have Finance analyze this forecast")
    session.add_assistant_message("Here is the analysis.")

    rendered = session.render_conversation_context("and the Q4 view?")

    assert "attachment_notice" not in rendered
    assert "visible to you alone" not in rendered
