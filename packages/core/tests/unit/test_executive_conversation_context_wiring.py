"""Both chat paths must hand `_stream_agent_loop` the RENDERED conversation tail.

The renderer is tested in test_session_conversation_context.py and the router's
forwarding in test_router.py, but nothing joined them: swapping either call site
back to the raw `user_message`, or dropping the kwarg so the "" default applies,
left the whole suite green. These two tests close that gap — one per call site,
since the committee path is wired separately.
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-not-used")
os.environ.setdefault("EXEC_EMAIL_ADDRESS", "exec@example.test")

from openexecutive.orchestrator.executive import Executive  # noqa: E402
from openexecutive.orchestrator.session import Session  # noqa: E402

_PRIOR_QUESTION = "We are considering raising CIAO's price by 30% next quarter."
_PRIOR_ANSWER = "Here is the pricing analysis at EUR 6/employee/month."
_FOLLOW_UP = "What would Legal change if we go ahead with the 30% increase?"


def _seeded_session() -> Session:
    session = Session()
    session.add_user_message(_PRIOR_QUESTION)
    session.add_assistant_message(_PRIOR_ANSWER)
    return session


def _capture_loop_kwargs(driver: str) -> dict[str, Any]:
    """Drive one chat path with `_stream_agent_loop` stubbed; return its kwargs."""
    captured: dict[str, Any] = {}

    async def _fake_loop(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        captured.update(kwargs)
        yield "ok"

    async def _drive() -> None:
        exec_ = Executive()
        stream = getattr(exec_, driver)(
            user_message=_FOLLOW_UP, session=_seeded_session()
        )
        async for _ in stream:
            pass

    with patch.object(Executive, "_stream_agent_loop", new=_fake_loop):
        try:
            asyncio.run(_drive())
        except Exception:
            # The committee path continues past the stubbed loop into review /
            # revision, which needs a provider, so a failure AFTER the loop ran
            # is expected. A failure BEFORE it means the wiring under test never
            # executed — re-raise that one rather than reporting it as a missing
            # kwarg.
            if not captured:
                raise
    return captured


def test_stream_chat_passes_the_rendered_tail_not_the_raw_message() -> None:
    captured = _capture_loop_kwargs("stream_chat")

    context = captured.get("conversation_context")
    assert context, "stream_chat did not pass conversation_context"
    assert context != _FOLLOW_UP, "passed the raw message instead of the tail"
    # The whole point: the prior turn names the subject the follow-up omits.
    assert "CIAO" in context
    assert _FOLLOW_UP in context


def test_committee_path_passes_the_rendered_tail_exactly_once() -> None:
    captured = _capture_loop_kwargs("stream_chat_with_committee")

    context = captured.get("conversation_context")
    assert context, "committee path did not pass conversation_context"
    assert context != _FOLLOW_UP, "passed the raw message instead of the tail"
    assert "CIAO" in context
    # add_user_message runs after the loop on both paths, so the current message
    # must appear once — a duplicate would mean history already held this turn.
    assert context.count(_FOLLOW_UP) == 1
