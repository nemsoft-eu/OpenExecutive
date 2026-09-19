from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from openexecutive.memory.company_profile import CompanyProfile

_ELISION = "\n[…]\n"


def _keep_both_ends(text: str, limit: int) -> str:
    """Truncate the middle of *text*, keeping its head and tail.

    Used for the current user message, whose typed question may sit at either
    end depending on the channel (see ``render_conversation_context``).
    """
    if len(text) <= limit:
        return text
    budget = limit - len(_ELISION)
    head = budget // 2
    return text[:head] + _ELISION + text[-(budget - head):]


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    company_profile: CompanyProfile | None = None
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    # (channel, channel_ref) pairs the Executive has seen during this session.
    # Used by schedule_followup to refuse scheduling sends to refs the user
    # never actually used — anti-spam guard.
    seen_channel_refs: set[tuple[str, str]] = field(default_factory=set)

    def add_user_message(self, content: str) -> None:
        self.conversation_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str | list[dict[str, Any]]) -> None:
        self.conversation_history.append({"role": "assistant", "content": content})

    def render_conversation_context(
        self,
        current_message: str,
        *,
        # Budgets are sized against what a fan-out costs, not round numbers:
        # this text is re-sent to EVERY specialist, uncached, so each 1k chars
        # is ~250 tokens x N specialists of prefill. Two exchanges covers the
        # "turn 1 named the subject" case without paying for older turns that
        # are rarely referenced. Assistant turns get 2x a user turn because the
        # figures a follow-up refers to are usually in the Executive's answer
        # (the turn-1 reply behind issue #12 was 5,075 chars); the current
        # message gets the most because it may carry attachment text.
        turns: int = 2,
        current_max_chars: int = 2_000,
        user_max_chars: int = 600,
        assistant_max_chars: int = 1_200,
        total_max_chars: int = 6_000,
    ) -> str:
        """Render the recent conversation as plain text for a specialist.

        Specialists see *only* what ``BaseAgent.analyze`` composes — department
        memory, past decisions, failure cases, retrieved knowledge, this text,
        and their query. They get no message history and no company profile, so
        without this they cannot resolve a follow-up: the turn that motivated
        issue #12 asked what to change "if we go ahead with the 30% increase"
        and never named CIAO, the price, or the company.

        This replaces a per-call ``context`` the model used to write itself,
        once per specialist. Rendering it here costs no routing output tokens.

        ``current_message`` keeps its head AND its tail rather than either end,
        because the channels disagree about where extracted attachment text
        goes: ``api/routes/chat.py`` appends it after the typed question, while
        ``integrations/telegram_bot.py`` and ``integrations/discord_bot.py``
        prepend it. Slicing from one end would preserve the question on some
        channels and discard it on others. Keeping both ends preserves it
        wherever it sits, and still drops the middle of a document blob that
        would otherwise reach every specialist in the fan-out as uncached
        input. A typed question longer than the budget is itself cut in the
        middle — accepted, since the alternative is unbounded fan-out cost.
        """
        parts: list[str] = []
        for turn in self.get_recent_history(max_turns=turns):
            content = turn["content"]
            if not isinstance(content, str):
                # Assistant turns can be a cache_control-wrapped block list.
                content = "\n".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if not content:
                continue
            if turn["role"] == "user":
                parts.append(f"User: {content[:user_max_chars]}")
            else:
                parts.append(f"Executive: {content[:assistant_max_chars]}")
        parts.append(f"User: {_keep_both_ends(current_message, current_max_chars)}")
        # Tail slice, so an over-budget history sheds its OLDEST turn and always
        # keeps the current message. With the default per-part caps the joined
        # result cannot exceed ~5.6k, so this only fires on a malformed history
        # (more assistant turns than user turns, e.g. a corrupted restore).
        return "\n\n".join(parts)[-total_max_chars:]

    def get_recent_history(self, max_turns: int = 20) -> list[dict[str, Any]]:
        history = self.conversation_history[-(max_turns * 2):]
        # Anthropic requires messages to start with a user turn.
        # Drop a leading assistant message if history length is odd (can happen on error recovery).
        if history and history[0]["role"] != "user":
            history = history[1:]
        return history
