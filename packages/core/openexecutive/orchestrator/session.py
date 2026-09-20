from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from openexecutive.memory.company_profile import CompanyProfile

_ELISION = "\n[…]\n"

# Budgets for the specialist-facing conversation tail. Named rather than inline
# because they are one fixed policy, not per-caller knobs: the rendered text is
# re-sent to EVERY specialist in a fan-out, uncached, so each 1k chars costs
# ~250 tokens x N specialists of prefill. Two exchanges covers the "turn 1 named
# the subject" case (issue #12) without paying for older turns that are rarely
# referenced. Assistant turns get twice a user turn because the figures a
# follow-up refers to are usually in the Executive's answer — the turn-1 reply
# behind #12 was 5,075 chars. The current message gets the most because it may
# carry attachment text.
_TAIL_TURNS = 2
_TAIL_CURRENT_MAX_CHARS = 2_000
_TAIL_USER_MAX_CHARS = 600
_TAIL_ASSISTANT_MAX_CHARS = 1_200
_TAIL_TOTAL_MAX_CHARS = 6_000
# The Executive gets the company profile in a cached system block; a specialist
# gets no system-level company context at all, so without this a terse
# company-relative question ("can we afford ten more hires?") reaches the CFO
# with no burn, runway, ARR or headcount. The model-written per-call `context`
# used to carry it; nothing else does.
#
# Sized to `CompanyProfile.to_specialist_block`, which measures 3,187-3,440
# chars across the three shipped fixtures — the cap clears the largest by ~25%,
# matching the margin the pre-market-context cap (1_800) had over its own 1,446.
# It is a backstop against an unusually long profile, not a working limit: none
# of the shipped fixtures reaches it.
#
# Enforced by `_fit_lines`, which drops whole LINES, not characters. The
# guarantee that buys is "whole fields are lost, never a cut value, and trailing
# fields go first" — deliberately weaker than the "never a number" this comment
# once claimed, which measurement falsified: an unvalidated free-text `industry`
# could crowd the budget and render `monthly burn $25` for a $250,000 burn.
# Re-derive this whenever `to_specialist_block`'s field order changes; a stale
# claim here is how the earlier 1,200-char head slice of `to_prompt_block` came
# to drop burn and runway while reading as deliberate (PR #18, Bugbot + Codex).
_TAIL_PROFILE_MAX_CHARS = 4_300


def _inert(text: str) -> str:
    """Make *text* unable to open or close any prompt envelope.

    ``BaseAgent.analyze`` wraps this block in ``<conversation_context>`` and
    wraps four sibling blocks in tags of their own. This path is the first to
    carry raw uploaded-document text into one of them, so a document containing
    a closing tag would end the block early and have its remainder read as
    instructions — in every specialist of a fan-out at once.

    Follows the convention `alerts.review._untrusted` established for the
    ``<alert>`` envelope: replace the brackets rather than one known tag, so a
    sibling tag invented later is covered without editing this. It does NOT
    collapse to a single line the way that helper does — the ``User:`` /
    ``Executive:`` structure is the point of this block.
    """
    return text.replace("<", "‹").replace(">", "›")


def _fit_lines(text: str, limit: int) -> str:
    """Trim *text* to *limit* by dropping whole trailing LINES, not characters.

    A raw ``text[:limit]`` cuts mid-character, and on the profile digest that
    produced a corrupted figure rather than a missing one: a profile whose
    free-text ``industry`` crowded the budget rendered
    ``**Financial position**: monthly burn $25`` for a company burning $250,000
    a month, with no elision marker to signal the cut. A specialist reads that
    as fact. Dropping whole lines makes the unit of loss a whole field, so the
    digest's field order is what decides the cost — which is what its docstring
    claims.

    A line that does not fit is SKIPPED rather than ending the scan, so one
    pathological field cannot take every field after it. ``industry`` and the
    other identity strings have no length validation, and stopping at the first
    over-long line meant a 4,300-char ``industry`` discarded the financial line
    behind it — losing the numbers for the opposite reason. Skipping keeps them.

    What this guarantees is therefore: whole fields are lost, never a cut value,
    and trailing fields go before leading ones. It is NOT "a number always
    survives" — a single field longer than the entire budget is dropped, and if
    that field is the identity header its headcount and ARR go with it. Dropping
    a figure is recoverable; showing a specialist ``$25`` for a $250,000 burn is
    not, which is the trade this makes.
    """
    if len(text) <= limit:
        return text
    kept: list[str] = []
    used = 0
    for line in text.split("\n"):
        # +1 for the newline that rejoins this line to the previous one.
        cost = len(line) + (1 if kept else 0)
        if used + cost > limit:
            continue
        kept.append(line)
        used += cost
    return "\n".join(kept)


def _keep_both_ends(text: str, limit: int) -> str:
    """Truncate the middle of *text*, keeping its head and tail.

    Used for any user message, whose typed question may sit at either end
    depending on the channel (see ``render_conversation_context``). Applies to
    replayed history turns too, not just the current one: ``add_user_message``
    stores the attachment-augmented string verbatim, so a one-ended slice here
    would drop next turn exactly the question this preserved last turn.
    """
    if len(text) <= limit:
        return text
    budget = limit - len(_ELISION)
    if budget <= 0:
        # A limit at or below the elision's own length leaves nothing to keep.
        # Returning early also avoids `text[-0:]`, which is the WHOLE string —
        # that would forward an entire attachment blob to every specialist,
        # the exact cost this cap exists to prevent.
        return text[:limit]
    head = budget // 2
    tail = budget - head
    return text[:head] + _ELISION + text[-tail:]


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
        # Defaults are the fixed policy above; the parameters exist so the
        # budget and guard behaviour can be probed at their boundaries without
        # monkeypatching module state. No production caller overrides them.
        turns: int = _TAIL_TURNS,
        current_max_chars: int = _TAIL_CURRENT_MAX_CHARS,
        user_max_chars: int = _TAIL_USER_MAX_CHARS,
        assistant_max_chars: int = _TAIL_ASSISTANT_MAX_CHARS,
        total_max_chars: int = _TAIL_TOTAL_MAX_CHARS,
        profile_max_chars: int = _TAIL_PROFILE_MAX_CHARS,
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
                # Defensive, not a live path: today `add_assistant_message` is
                # only ever called with a plain str, and the cache_control block
                # list is built inside `_build_messages` into a throwaway dict
                # that never reaches conversation_history. Kept because the
                # dataclass signature permits a list and stringifying one would
                # put raw dict repr into a specialist's prompt.
                content = "\n".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if not content:
                continue
            if turn["role"] == "user":
                # Both-ends here too: a stored turn is the same
                # attachment-augmented string, so a head-only slice would drop
                # next turn the question this preserved last turn.
                parts.append(f"User: {_keep_both_ends(content, user_max_chars)}")
            else:
                parts.append(f"Executive: {content[:assistant_max_chars]}")
        parts.append(f"User: {_keep_both_ends(current_message, current_max_chars)}")
        rendered = _inert("\n\n".join(parts))
        # The profile is pinned ahead of the tail slice below, not appended to
        # `parts`: it is the only company context a specialist gets, so an
        # over-budget turn must shed its oldest CONVERSATION, never the profile.
        #
        # It IS passed through _inert. An earlier version of this comment called
        # the profile first-party and skipped the escape; that was false.
        # `POST /clients/generate` feeds uploaded PDF/Word/Excel/CSV text to the
        # engagement-intake agent, which is instructed to copy facts verbatim,
        # and `clients/slots.py` then writes that profile.yaml over the live
        # profile. So a competitor gloss or mission statement can carry text a
        # prospective client wrote. Without the escape, a `</conversation_context>`
        # inside one of those fields closes the envelope early and the remainder
        # is read as instructions — by every specialist in every fan-out, for
        # the life of the engagement.
        profile = ""
        if self.company_profile is not None:
            try:
                profile = _fit_lines(
                    _inert(self.company_profile.to_specialist_block() or ""),
                    profile_max_chars,
                )
            except Exception:
                # A malformed profile degrades to no profile rather than
                # breaking the turn, matching _emit_memory_snapshot's handling.
                profile = ""
        # Tail slice, so an over-budget render loses its OLDEST text and always
        # keeps the current message. It cuts at a character, not a turn
        # boundary, so the first surviving turn can arrive as a headless
        # fragment — acceptable here because with the default per-part caps the
        # joined result cannot exceed ~5.6k, so this only fires on a malformed
        # history (more assistant turns than user turns, e.g. a corrupted
        # restore).
        conversation = rendered[-total_max_chars:]
        return f"{profile}\n\n{conversation}" if profile else conversation

    def get_recent_history(self, max_turns: int = 20) -> list[dict[str, Any]]:
        history = self.conversation_history[-(max_turns * 2):]
        # Anthropic requires messages to start with a user turn.
        # Drop a leading assistant message if history length is odd (can happen on error recovery).
        if history and history[0]["role"] != "user":
            history = history[1:]
        return history
