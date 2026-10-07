from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from openexecutive.memory.company_profile import CompanyProfile

if TYPE_CHECKING:
    from openexecutive.memory.workspace_settings import PrincipalRole

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
    # Which inbound channel this session arrived on ("slack", "discord",
    # "telegram", "google_chat", "email"), and the address on it. Empty for
    # web/CLI turns and runs nobody started from a channel. Used when a
    # workflow raises an approval gate mid-conversation: the gate records
    # where to look for the answer, so a reply on this channel can resolve it
    # (not on email, whose reply is no chat message — see `gate_delivery`).
    # Also how `content_trust.principal_speaking` tells an inbound email from
    # the principal's own surfaces: an email turn that left it empty was read
    # as the web app's. Inbound vocabulary — see `normalize_channel`.
    origin_channel: str = ""
    origin_channel_ref: str = ""
    # True only for a session minted by the web chat route. `origin_channel`
    # cannot stand in for this: alert review, the CLI, the MCP server, the
    # scheduler and the unattended workflows all leave it empty while being
    # nothing like a browser turn. Anything that wants to treat browser turns
    # differently has to ask for them by name.
    from_web_chat: bool = False
    # True only for a session minted by the CLI (`openexecutive ask` / `chat`),
    # which runs on the host itself — the principal's own surface for
    # `content_trust.principal_speaking`, as the web chat is.
    from_cli: bool = False
    # The rostered Person behind this conversation, when one is resolved. The
    # adapters already pass this to `Executive.chat(person_id=...)`; holding it
    # on the session too lets tool handlers running mid-turn tell "the approver
    # is the person I'm already talking to" from "the approver is someone else".
    caller_person_id: int | None = None
    # Set by a channel adapter that verified for itself that `caller_person_id`
    # is who sent this turn's message, on a channel the checks in
    # `people_tools` don't know. Read only by Always in the loop
    # (`memory.history_chat`, `orchestrator.history_tools`), never as a grant
    # for anything else. `private_chat` adds that nobody but that person (and
    # the Executive) can read the conversation, so their notes may be recalled.
    # An adapter sets either only when it resolved `caller_person_id` from the
    # sender identity it verified itself, never from anything the message says.
    speaker_verified: bool = False
    private_chat: bool = False
    # The live alert board as the server derived it this turn, recorded by
    # `briefing.context.render_and_trust`. `ack_alert` refuses anything else on
    # EVERY session, web included, so an id quoted inside an alert's own body —
    # alerts are minted from inbound mail and chat, so that text is
    # attacker-controlled — cannot clear a row that is closed, snoozed or
    # invented. It does not stop the model being argued into acking the wrong
    # LIVE card; see `format_open_alerts_for_prompt` for the limits of this
    # control. Empty means the turn was shown no board and can ack nothing.
    # `find_alerts` is the one thing that adds to it after that, and only on a
    # turn with `principal_board_shown` — so an open row off the board (read,
    # snoozed, past TTL but not yet swept) can be cleared there too.
    trusted_alert_ids: set[int] = field(default_factory=set)
    # Whether `render_and_trust` showed THIS turn the principal's own board:
    # the principal on a verified surface, and on a channel only in their DM
    # (`channel_context.attach_briefing_context`). `find_alerts` answers and
    # widens nothing without it — "principal on a verified surface" alone
    # also passes the principal's turn in a shared Slack or Discord thread.
    principal_board_shown: bool = False
    # Ids `find_alerts` added to `trusted_alert_ids` this turn, capped across
    # calls (`schedule_tools._FIND_ALERTS_MAX_PER_TURN`). Reset per turn with
    # the trusted set.
    found_alert_ids: set[int] = field(default_factory=set)
    # Pending roster requests ("who is this new sender?") the server showed
    # this turn in its <roster_requests> block — the principal's own verified
    # turn only (`briefing.context.render_and_trust`). `resolve_roster_request`
    # answers nothing else, so an id the model invents or reads out of text
    # cannot add anyone.
    trusted_roster_request_ids: set[int] = field(default_factory=set)
    # Per-session override of the install's workspace mode ("solo" / "team";
    # None = use the workspace setting). Evals run scenarios concurrently on
    # one Executive, so they set it here instead of flipping the global. Read
    # it through `memory.workspace_settings.effective_workspace_mode`.
    workspace_mode: str | None = None
    # Per-session override of the principal's role (None = use the
    # workspace's), for the same reason: an eval scenario supplies the role
    # of the principal it plays without writing the install-wide row. Read
    # it through `memory.workspace_settings.effective_principal_role`.
    principal_role: PrincipalRole | None = None
    # Per-session voice persona slug (None = the Executive override's voice,
    # else Direct), so an eval scenario can play a voice without writing the
    # install-wide override that concurrent scenarios share.
    voice_persona_slug: str | None = None
    # The mode resolved for the turn in progress, pinned at its start by
    # `workspace_settings.pin_turn_workspace_mode` (Executive.stream_chat and
    # the committee path) so the tool handlers use the same mode as the
    # persona and tool list. Re-resolved every turn; never an override.
    turn_workspace_mode: str | None = None
    # The principal's role resolved for the turn in progress, pinned with the
    # mode by `workspace_settings.pin_turn_principal_role` (an empty role in
    # team), so the org block, the specialists' <principal_role> tag and any
    # workflow the turn starts agree even if the role is edited mid-turn.
    # Re-resolved every turn; never an override.
    turn_principal_role: PrincipalRole | None = None
    # True for a run nobody is watching that goes through the chat loop (the
    # scheduler's PROACTIVE TRIGGER dispatch). Its prompt quotes stored intent
    # text, so the loop neither offers nor runs the tools in
    # `schedule_tools.UNATTENDED_WITHHELD_TOOLS` — things only the principal
    # decides, such as starting to track a goal.
    unattended: bool = False
    # True for a turn about something private to the principal (mail from one
    # of their contacts, mail they forwarded): an alert raised on it is
    # private to the principal (``alerts.models.PRIVATE_ALERT_TAG``) and it
    # may not draft a team-visible artifact. Set by the email poller.
    private_to_principal: bool = False
    # The bare, lowercased From address of the inbound email this turn is
    # answering, and whether it is the principal's primary address with
    # Gmail's own Authentication-Results reporting dmarc=pass for it
    # (``integrations.fact_confirmation.authenticated_by_gmail``; fails
    # closed). Set by the email poller only; empty / False on every other
    # surface. The fact tools read them to let the principal's own
    # email request a standing fact, held until a token reply confirms it
    # (``orchestrator.fact_tools``) — a From line alone proves nothing.
    email_from: str = ""
    email_authenticated: bool = False
    # Act as me for the turn in progress (``delegation.settings.TurnDelegation``),
    # pinned at its start by ``pin_turn_delegation``: whether the speaker has
    # it on, whether ``ghostwrite_email`` is offered, and whether the turn has
    # touched their mailbox (every audit row it writes after that is private).
    # Re-resolved every turn; never an override.
    turn_delegation: Any = None
    # True when this web request carried a signed-in caller (``x-caller-email``,
    # stamped by the UI proxy from the sign-in), set per turn by the chat
    # route. A header-less request resolves to the principal but is no
    # sign-in: Act as me needs one (or local login).
    web_caller_signed_in: bool = False
    # Whose documents this session reads and publishes, when that is not its
    # own speaker (``artifact_records.current_viewer``): a session a workflow
    # mints inside someone's turn (``executive_research``'s synthesis) is
    # pinned to that turn's viewer, so what it drafts is theirs. An
    # ``artifact_records.Viewer``; None derives it from this session.
    documents_viewer: Any = None
    # Evals and tests only (``delegation.settings.DelegationOverride``): run
    # as if the speaker had Act as me, against a fake mailbox.
    delegation_override: Any = None

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
