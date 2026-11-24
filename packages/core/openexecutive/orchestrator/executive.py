from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from openexecutive.agents.tool_outcome import unwrap_tool_outcome
from openexecutive.audit import bind_turn, clear_turn, private_rows, set_turn
from openexecutive.audit import log_event as audit_log
from openexecutive.audit.redaction import (
    ERROR_DETAIL_LEN,
    audit_tool_input,
    audit_tool_input_full,
    audit_tool_result,
    audit_tool_result_full,
)
from openexecutive.audit.usage import log_model_usage
from openexecutive.config import get_settings
from openexecutive.delegation.lockdown import (
    mail_turn_withholds,
)
from openexecutive.delegation.settings import (
    block0_delegation_on,
    pin_turn_delegation,
    turn_delegation,
)
from openexecutive.memory import replay as chat_replay
from openexecutive.memory.facts import render_facts_for_prompt
from openexecutive.memory.honcho_client import ReasoningLevel as HonchoReasoningLevel
from openexecutive.memory.workspace_settings import (
    effective_principal_role,
    effective_workspace_mode,
    pin_turn_principal_role,
    pin_turn_workspace_mode,
)
from openexecutive.orchestrator import (
    app_control_catalog,  # noqa: F401  (fills the app tools' catalog)
    app_links,
    claim_check,
    content_scan,
    extensions,
    levels,
    outside_content,
    reply_card_tools,
    take_the_lead,
    tool_groups,
    tool_images,
    turn_live,
    turn_progress,
)
from openexecutive.orchestrator import pushback as pushback_mod
from openexecutive.orchestrator import web_search_tool as web_searches
from openexecutive.orchestrator.action_chips import summarize_action
from openexecutive.orchestrator.activity_labels import (
    fallback_activity,
    summarize_activity,
)
from openexecutive.orchestrator.activity_tools import ACTIVITY_TOOL_HANDLERS, ACTIVITY_TOOLS
from openexecutive.orchestrator.alert_tools import (
    CREATE_ALERT_TOOL,
    handle_create_alert,
)
from openexecutive.orchestrator.answer_sources import TurnSources, record_web_sources
from openexecutive.orchestrator.app_control import (
    APP_TOOL_HANDLERS,
    APP_TOOLS,
)
from openexecutive.orchestrator.artifact_tools import (
    DRAFT_ARTIFACT_TOOL_HANDLERS,
    DRAFT_ARTIFACT_TOOLS,
)
from openexecutive.orchestrator.broadcast_tools import (
    BROADCAST_TOOL_HANDLERS,
    BROADCAST_TOOLS,
)
from openexecutive.orchestrator.calendar_tools import (
    CALENDAR_TOOL_HANDLERS,
    CALENDAR_TOOLS,
    typed_booking_only,
)
from openexecutive.orchestrator.chat_search_tools import (
    CHAT_SEARCH_TOOL_HANDLERS,
    CHAT_SEARCH_TOOL_NAMES,
    CHAT_SEARCH_TOOLS,
    recent_chats_block,
    search_person,
)
from openexecutive.orchestrator.content_trust import (
    principal_only_withheld,
    principal_speaking,
)
from openexecutive.orchestrator.correction_tools import (
    CORRECTION_TOOL_HANDLERS,
    CORRECTION_TOOLS,
    pending_followups_block,
)
from openexecutive.orchestrator.debug_events import DebugCollector
from openexecutive.orchestrator.decision_tools import (
    DECISION_TOOL_HANDLERS,
    DECISION_TOOLS,
)
from openexecutive.orchestrator.delegation_tools import (
    DELEGATION_TOOL_HANDLERS,
    DELEGATION_TOOL_NAMES,
    DELEGATION_TOOLS,
    MAILBOX_TOOL_NAMES,
)
from openexecutive.orchestrator.department_tools import (
    DEPARTMENT_TOOL_HANDLERS,
    DEPARTMENT_TOOLS,
)
from openexecutive.orchestrator.document_tools import (
    DOCUMENT_TOOL_HANDLERS,
    DOCUMENT_TOOLS,
)
from openexecutive.orchestrator.fact_tools import (
    FACT_TOOL_HANDLERS,
    FACT_TOOLS,
    own_chat_person,
    private_owner,
)
from openexecutive.orchestrator.figure_tools import (
    WORKED_OUT_TOOL_HANDLERS,
    WORKED_OUT_TOOLS,
)
from openexecutive.orchestrator.file_tools import FILE_TOOL_HANDLERS, FILE_TOOLS
from openexecutive.orchestrator.form_tools import (
    FORM_TOOL_HANDLERS,
    FORM_TOOLS,
    PROPOSE_FORM_VALUES,
    build_form_patch_event,
)
from openexecutive.orchestrator.guide_tools import (
    GUIDE_TOOL_HANDLERS,
    GUIDE_TOOLS,
)
from openexecutive.orchestrator.history_tools import (
    HISTORY_TOOL_HANDLERS,
    HISTORY_TOOL_NAMES,
    HISTORY_TOOLS,
    recall_person,
)
from openexecutive.orchestrator.mcp_gateway import (
    MCP_TOOL_NAMES,
    MCP_TOOLS,
    MCPGateway,
    gateway_reconnect_needed,
    gateway_server_names,
)
from openexecutive.orchestrator.open_loop_tools import (
    OPEN_LOOP_TOOL_HANDLERS,
    OPEN_LOOP_TOOLS,
)
from openexecutive.orchestrator.people_tools import (
    PEOPLE_TOOL_HANDLERS,
    PEOPLE_TOOLS,
    is_principal_on_verified_surface,
    turn_is_private_to_principal,
)
from openexecutive.orchestrator.reading import (
    CUT_OFF_NOTE,
    CUT_OFF_RESULT,
    FULL_NOTE,
    MAX_EMPTY_CUT_ROUNDS,
    MAX_STEPS,
    REPEAT_LIMIT,
    REPEATING_NOTE,
    STEPS_NOTE,
    TurnReadingBudget,
    apply_loop_cache_marker,
    last_written_block,
    says_something,
    with_turn_limits,
)
from openexecutive.orchestrator.research_tools import (
    RESEARCH_TOOL_HANDLERS,
    RESEARCH_TOOLS,
)
from openexecutive.orchestrator.router import (
    NAME_PREVIEW_CHARS,
    SPECIALIST_TOOLS,
    audit_name_resolution,
    partition_specialist_fanout,
    principal_role_context,
    resolve_specialist_name,
    route_parallel,
)
from openexecutive.orchestrator.schedule_tools import (
    PRIVATE_TURN_WITHHELD_TOOLS,
    READ_ONLY_WITHHELD_TOOLS,
    SCHEDULE_TOOL_HANDLERS,
    SCHEDULE_TOOLS,
    SHARED_CHAT_WITHHELD_TOOLS,
    UNATTENDED_WITHHELD_TOOLS,
    current_session,
    filter_tools_for_workspace_mode,
    private_turn_withholds,
    read_only_turn,
    read_only_withholds,
    shared_chat_withholds,
    tools_withheld_in_mode,
    withheld_tool_error,
)
from openexecutive.orchestrator.session import Session
from openexecutive.orchestrator.skills_tools import SKILL_TOOL_HANDLERS, SKILL_TOOLS
from openexecutive.orchestrator.teammate_mail import guard_call_tool
from openexecutive.orchestrator.turn_inbox import (
    HAND_OVER_EMPTY_REPLY,
    HAND_OVER_NOTE,
    TurnInbox,
    render_added_messages,
    with_added,
)
from openexecutive.orchestrator.watchlist_tools import (
    WATCHLIST_TOOL_HANDLERS,
    WATCHLIST_TOOLS,
    speakers_own_row,
)
from openexecutive.orchestrator.web_fetch import SeenUrls, server_search_result_urls
from openexecutive.orchestrator.web_search_tool import (
    WEB_FETCH_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
    WebSearchSelection,
    client_search_handlers,
    select_web_search_tool,
)
from openexecutive.orchestrator.workflow_authoring_tools import (
    WORKFLOW_AUTHORING_TOOL_HANDLERS,
    WORKFLOW_AUTHORING_TOOLS,
)
from openexecutive.orchestrator.workflow_run_tools import (
    WORKFLOW_RUN_TOOL_HANDLERS,
    WORKFLOW_RUN_TOOLS,
)
from openexecutive.prompts.cache_manager import build_system_blocks
from openexecutive.providers import get_provider
from openexecutive.providers.feature_gate import MAX_OUTPUT_TOKENS
from openexecutive.providers.registry import reading_limits_for, sees_images
from openexecutive.providers.translator import (
    OPENROUTER_CITATIONS_BLOCK,
    reasoning_replay_block,
)
from openexecutive.workflows import python_job, step_script, turn_files
from openexecutive.workflows.action_step import looks_like_error
from openexecutive.workflows.python_job import PYTHON_JOB_TOOL_HANDLERS, PYTHON_JOB_TOOLS

logger = logging.getLogger(__name__)


# Bound by ``Executive.chat`` to a list that a round the model refused
# (stop_reason "refusal") appends to, so the collected reply never sends what
# the refused round began writing.
_REFUSED: ContextVar[list[int] | None] = ContextVar("executive_refused", default=None)

# What ``Executive.chat`` answers for a turn the model refused.
REFUSED_REPLY = "I can't help with that one."


def _with_refusal(collected: str, refused_chars: int) -> str:
    """A collected reply whose last round the model refused: the earlier
    rounds' text without the refused round's (its last ``refused_chars``),
    then ``REFUSED_REPLY``."""
    kept = collected[: max(0, len(collected) - refused_chars)].strip()
    return f"{kept}\n\n{REFUSED_REPLY}" if kept else REFUSED_REPLY

# The tools that read the speaker's own notes or earlier chats: private rows,
# withheld from the process log, and checked like a mail read.
_OWN_RECALL_TOOL_NAMES: frozenset[str] = HISTORY_TOOL_NAMES | CHAT_SEARCH_TOOL_NAMES


def _own_tool_call(tu: dict[str, Any], handlers: Any) -> dict[str, Any] | None:
    """A ``call_tool`` that names one of the turn's own built-in tools, as
    that tool's call (same id), or None. Only a plain name the gateway can't
    hold (no server prefix) with object arguments."""
    if tu.get("name") != "call_tool" or not isinstance(tu.get("input"), dict):
        return None
    named = tu["input"].get("name")
    if not isinstance(named, str) or not named or "__" in named or named in MCP_TOOL_NAMES:
        return None
    if named not in handlers:
        return None
    arguments = tu["input"].get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, RecursionError):
            return None
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return None
    return {**tu, "name": named, "input": arguments}


def _private_to_principal(session: Any) -> bool:
    """Whether this turn is about the principal's private mail (set by the
    email poller for mail from a contact and mail the principal forwarded)."""
    return getattr(session, "private_to_principal", False) is True


def _mail_turn_words(session: Any, own_text: str, *, touched_mail: bool, host_words: bool) -> str:
    """What a turn that touched the speaker's own mailbox may still teach:
    the words they typed, never the reply (it quotes mail and a draft built
    from it). Empty for any other turn, and for a replay, a host page's words
    or mail private to the principal from someone else."""
    if not touched_mail or host_words or chat_replay.is_replay(session) or _private_to_principal(session):
        return ""
    return own_text or ""


def _sync_mail_turn_words(words: str, *, person_id: int | None, session_id: str | None) -> None:
    """Peer memory from a mail turn: only the speaker's own words, with no
    reply, no one else in the session and no department mirror, so what it
    learns is what they said about their own mail and stays with them."""
    from openexecutive.memory.honcho_client import sync_turn as _honcho_sync

    _honcho_sync(words, "", person_id=person_id, session_id=session_id, co_present_person_ids=None)


def _contacts_in_prompt(session: Any) -> bool:
    """Whether this turn's system prompt lists the principal's contacts.

    Contacts are private to the principal, so only a turn the principal
    started on a verified surface sees them (the same rule as reaching them:
    ``people_tools.is_principal_on_verified_surface``). Every other turn gets the
    team-only org block. Fails closed.
    """
    try:
        from openexecutive.orchestrator.people_tools import is_principal_on_verified_surface

        return is_principal_on_verified_surface(session)
    except Exception:
        logger.exception("contacts_in_prompt: check failed — contacts left out")
        return False


# A tool name as the process log may show it (see _loggable_tool).
_LOGGABLE_TOOL_RE = re.compile(r"[a-z0-9_]{1,64}")


def _loggable_tool(label: str) -> str:
    """``label`` for the process log. A call_tool's inner name is the model's
    own text: on a turn private to the principal, or one that read their
    mail, it can carry that mail, and the log is not private to anyone. So
    anything not shaped like a tool name is logged as unlisted (the private
    audit row keeps it)."""
    return label if _LOGGABLE_TOOL_RE.fullmatch(label) else "call_tool:<unlisted>"


def _log_value(tool_name: str, value: Any) -> str:
    """A tool's input or result for the process log, which is not private to
    anyone: withheld for the tools that read the speaker's own mailbox or
    notes, and for every tool once the turn has read their mail (the private
    audit row keeps it)."""
    from openexecutive.delegation.settings import turn_touched_delegate_mail

    if (
        tool_name in DELEGATION_TOOL_NAMES or tool_name in _OWN_RECALL_TOOL_NAMES
        or turn_touched_delegate_mail()
    ):
        return "<private>"
    return _trunc(value)


def _trunc(value: Any, limit: int = 200) -> str:
    """Render *value* for a log line, capped at *limit* chars.

    Always uses `repr()` so strings stay quoted (matching the prior `%r`
    behaviour these call sites used) and dict/list payloads stay distinct.
    Appends `…[truncated N chars]` so the reader can tell when content was cut.
    """
    s = repr(value)
    if len(s) <= limit:
        return s
    return f"{s[:limit]}…[truncated {len(s) - limit} chars]"


# The history breakpoint lives an hour: people answer in minutes, not
# seconds, and a 5m entry would be gone before most next messages.
_HISTORY_CACHE_CONTROL: dict[str, str] = {"type": "ephemeral", "ttl": "1h"}


def _apply_history_cache_marker(
    system_blocks: list[dict[str, Any]], messages: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cache the conversation so far, so the next message reads it back.

    Marks the newest non-empty message before the caller's last one (the
    turn being answered): normally the previous reply. History turns are
    stored as plain text and rebuilt identically every turn, so on the next
    message everything up to that point is a cache read, and only what is
    new (the last exchange, this turn's context) is written. Without it the
    loop marker (``apply_loop_cache_marker``) writes the whole history again
    on every turn, because the previous turn's cached prefix ended inside
    that turn's own context and tool calls, which history does not keep.

    The API allows four breakpoints and the other three are spoken for, so
    the history marker takes the slot of the 5m company-profile block's.
    That block still sits in the cached prefix (covered by this marker),
    and dropping its 5m marker keeps the TTLs in the order the API requires:
    tools 1h, persona 1h, history 1h, then the 5m loop marker.

    Returns new lists; the caller's blocks and messages are left as they
    were. With no history (a first message, an unattended run) both come
    back unchanged.
    """
    target = -1
    for i in range(len(messages) - 2, -1, -1):
        content = messages[i].get("content")
        if isinstance(content, str) and content:
            target = i
            break
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "text" and b.get("text") for b in content
        ):
            target = i
            break
    if target < 0:
        return system_blocks, messages

    content = messages[target]["content"]
    if isinstance(content, str):
        marked_content: list[dict[str, Any]] = [
            {"type": "text", "text": content, "cache_control": dict(_HISTORY_CACHE_CONTROL)}
        ]
    else:
        marked_content = [
            {k: v for k, v in b.items() if k != "cache_control"} if isinstance(b, dict) else b
            for b in content
        ]
        last_text = max(
            j for j, b in enumerate(marked_content)
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
        )
        marked_content[last_text] = {
            **marked_content[last_text], "cache_control": dict(_HISTORY_CACHE_CONTROL)
        }
    new_messages = list(messages)
    new_messages[target] = {**messages[target], "content": marked_content}

    new_system = [
        {k: v for k, v in b.items() if k != "cache_control"}
        if isinstance(b, dict)
        and isinstance(b.get("cache_control"), dict)
        and b["cache_control"].get("ttl") != "1h"
        else b
        for b in system_blocks
    ]
    return new_system, new_messages


# The turn's own message (the question plus this turn's context: facts,
# memory, knowledge excerpts) is cached for the loop's first call only. It is
# the slot the loop marker takes from the second call on, so the request
# never carries more than the API's four breakpoints.
_TURN_CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}


def _mark_turn_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Cache the message being answered on the loop's first call, in place.

    Without it the first call sends the turn's context uncached, and the
    second call (after the first tool use, which is most turns) writes the
    same tokens to the cache: the context is paid twice. With it the first
    call writes it once and every later call reads it back.

    Replaces the last message with a marked copy (the caller owns the
    original dict) and returns that copy, so ``_clear_turn_marker`` can take
    the marker off once the loop marker moves past it. Returns None when
    there is nothing to mark: no messages, the last one is not the user's,
    or it carries no text.
    """
    if not messages or messages[-1].get("role") != "user":
        return None
    content = messages[-1].get("content")
    if isinstance(content, str):
        if not content:
            return None
        marked: list[dict[str, Any]] = [
            {"type": "text", "text": content, "cache_control": dict(_TURN_CACHE_CONTROL)}
        ]
    elif isinstance(content, list):
        texts = [
            j for j, b in enumerate(content)
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
        ]
        if not texts:
            return None
        marked = [dict(b) if isinstance(b, dict) else b for b in content]
        marked[texts[-1]]["cache_control"] = dict(_TURN_CACHE_CONTROL)
    else:
        return None
    message = {**messages[-1], "content": marked}
    messages[-1] = message
    return message


def _clear_turn_marker(message: dict[str, Any] | None) -> None:
    """Take ``_mark_turn_message``'s marker off its copy, in place."""
    if message is None:
        return
    for block in message.get("content") or []:
        if isinstance(block, dict):
            block.pop("cache_control", None)


def _with_notice(result: Any, notice: str) -> Any:
    """``result`` with the content check's notice before it
    (``outside_content.FLAG_NOTICE``): text gets it as a prefix, content
    blocks get it as a first text block."""
    if isinstance(result, list):
        return [{"type": "text", "text": notice}, *result]
    if isinstance(result, str):
        return notice + result
    # Anything else as the JSON the model would read without the notice.
    return notice + json.dumps(result, default=str)


def _tool_error_result(tool_name: str, exc: BaseException) -> str:
    """Render a crashed tool handler as a JSON tool_result the model can read.

    Tool handlers are *supposed* to return a JSON error string rather than
    raise (see the `_err` helpers in the orchestrator tool modules), but a bug
    in one of them — or in a library it calls — used to abort the entire turn,
    because `asyncio.gather` propagates the first exception. The adapter then
    showed the user a generic apology with no way to tell which tool failed.
    Converting the exception into a normal error tool_result keeps the turn
    alive and lets the model recover on the next iteration.

    Only the exception's TYPE goes to the model. Its message can carry
    filesystem paths, a validation error echoing the input, or a third-party
    HTTP body — and anything in model context can end up quoted back to the
    user. The full repr stays in the server log and the audit row.
    """
    return json.dumps(
        {
            "error": (
                f"{tool_name} failed with {type(exc).__name__}. The failure is "
                "recorded; do not retry the same call unchanged."
            )
        }
    )


def _drive_memory_block(session: Session, person_id: int | None) -> str:
    """The speaker's ``<drive_memory>`` body for this session, or "": only
    their own reads, none on a turn that may not keep them, and a store
    failure never blocks the turn."""
    from openexecutive.memory.drive_reads import format_drive_memory, may_remember

    try:
        if person_id is None or person_id != session.caller_person_id or not may_remember(session):
            return ""
        return format_drive_memory(session.session_id, person_id)
    except Exception:
        logger.exception("drive_reads: failed to load drive memory")
        return ""


def _current_time_block(now: datetime | None = None) -> str:
    """Render the body of a <current_time> hint: when this turn is, in the
    workspace's timezone (``get_user_timezone``, as the briefs use).

    Without it a chat turn carries no date at all, so the model guesses one,
    for instance from the newest record a tool returned, and "yesterday",
    "this week" or "overdue" resolve against the wrong day.
    """
    from openexecutive.memory.workspace_settings import get_user_timezone

    zone = get_user_timezone()
    local = (now or datetime.now(UTC)).astimezone(zone)
    offset = local.strftime("%z")
    return (
        f"{local:%A} {local.day} {local:%B %Y}, {local:%H:%M} "
        f"({getattr(zone, 'key', None) or local.tzname()}, UTC{offset[:3]}:{offset[3:]}). "
        "Resolve today, yesterday, this week, this month and other relative dates against this."
    )


def _build_current_speaker_block(person_id: int | None) -> str | None:
    """Render the body of a <current_speaker> hint naming who is in the room.

    Without this the Executive has no idea which human it is talking to on
    the web/briefing path, so it routinely offers to "loop in" or DM the very
    person reading the reply — absurd when that person is the principal, the
    audience every brief and nudge is written for. Returns None when the
    speaker can't be resolved (CLI/curl with no caller, or a fresh install
    with no roster), leaving the user turn unchanged.
    """
    if person_id is None:
        return None
    try:
        from openexecutive.people.store import get_person
        person = get_person(person_id)
    except Exception:
        logger.warning("speaker_lookup_failed person_id=%s", person_id, exc_info=True)
        return None
    if person is None:
        return None

    who = f"{person.full_name} ({person.role})" if person.role else person.full_name
    if person.is_principal:
        return (
            f"You are speaking directly with {who} — the principal, the company "
            "owner these morning briefs, end-of-day digests, and proactive nudges "
            "are written for. They are already here in this conversation: address "
            "them directly and never offer to loop them in, DM them, notify them, "
            'escalate to them, or "pull them in."'
        )
    return (
        f"You are speaking directly with {who}. They are already here in this "
        "conversation: address them directly and never offer to loop them in, DM "
        "them, or notify them."
    )


_ALL_SKILL_TOOLS = [
    *SKILL_TOOLS,
    CREATE_ALERT_TOOL,
    *DRAFT_ARTIFACT_TOOLS,
    *SCHEDULE_TOOLS,
    *CALENDAR_TOOLS,
    *PEOPLE_TOOLS,
    *OPEN_LOOP_TOOLS,
    *CORRECTION_TOOLS,
    *DEPARTMENT_TOOLS,
    *DECISION_TOOLS,
    *FACT_TOOLS,
    *WORKED_OUT_TOOLS,
    *DOCUMENT_TOOLS,
    *BROADCAST_TOOLS,
    *WATCHLIST_TOOLS,
    *ACTIVITY_TOOLS,
    *RESEARCH_TOOLS,
    *WORKFLOW_AUTHORING_TOOLS,
    *WORKFLOW_RUN_TOOLS,
    *FORM_TOOLS,
    *GUIDE_TOOLS,
    *PYTHON_JOB_TOOLS,
    *FILE_TOOLS,
    *APP_TOOLS,
]


def _offered_skill_tools() -> list[dict[str, Any]]:
    """_ALL_SKILL_TOOLS with the Python job tool as this instance describes it
    (python_job.offered_definition: a runner's extra libraries)."""
    offered = python_job.offered_definition()
    return [offered if t["name"] == python_job.TOOL_NAME else t for t in _ALL_SKILL_TOOLS]


_ALL_SKILL_HANDLERS = {
    **SKILL_TOOL_HANDLERS,
    "create_alert": handle_create_alert,
    **DRAFT_ARTIFACT_TOOL_HANDLERS,
    **SCHEDULE_TOOL_HANDLERS,
    **CALENDAR_TOOL_HANDLERS,
    **PEOPLE_TOOL_HANDLERS,
    **OPEN_LOOP_TOOL_HANDLERS,
    **CORRECTION_TOOL_HANDLERS,
    **DEPARTMENT_TOOL_HANDLERS,
    **DECISION_TOOL_HANDLERS,
    **FACT_TOOL_HANDLERS,
    **WORKED_OUT_TOOL_HANDLERS,
    **DOCUMENT_TOOL_HANDLERS,
    **BROADCAST_TOOL_HANDLERS,
    **WATCHLIST_TOOL_HANDLERS,
    **ACTIVITY_TOOL_HANDLERS,
    **RESEARCH_TOOL_HANDLERS,
    **WORKFLOW_AUTHORING_TOOL_HANDLERS,
    **WORKFLOW_RUN_TOOL_HANDLERS,
    **FORM_TOOL_HANDLERS,
    **GUIDE_TOOL_HANDLERS,
    **PYTHON_JOB_TOOL_HANDLERS,
    **FILE_TOOL_HANDLERS,
    **APP_TOOL_HANDLERS,
}


def builtin_tool_names() -> frozenset[str]:
    """Every tool name the chat loop knows itself, which an extension's tool
    may not take (orchestrator/extensions.py)."""
    return frozenset({
        *(t["name"] for t in SPECIALIST_TOOLS),
        *_ALL_SKILL_HANDLERS,
        *DELEGATION_TOOL_HANDLERS,
        *MAILBOX_TOOL_NAMES,
        *HISTORY_TOOL_HANDLERS,
        *MCP_TOOL_NAMES,
        step_script.RUN_SCRIPT_TOOL,
        step_script.LIST_SAVED_TOOLS_TOOL,
        tool_groups.OPEN_TOOLS,
        tool_groups.USE_TOOL,
        turn_progress.SHOW_PROGRESS,
        WEB_SEARCH_TOOL_NAME,
        WEB_FETCH_TOOL_NAME,
    })


def _private_tool_row(tool_name: str, tool_input: Any = None, result: Any = None) -> bool:
    """Whether a tool's dispatch audit row is private to the principal: Act as
    me reads the speaker's own mailbox, and the fact tools' input quotes the
    principal verbatim and ties a fact to their chat session and turn, which
    ``GET /memories/facts`` hides from everyone else. A kept figure
    (``keep_figure``) is the speaker's own and may come from their mail."""
    return (
        tool_name in DELEGATION_TOOL_NAMES
        or tool_name in FACT_TOOL_HANDLERS
        or tool_name in WORKED_OUT_TOOL_HANDLERS
        or tool_name in _OWN_RECALL_TOOL_NAMES
        # An installed extension's tool that quotes the speaker's own data
        # (orchestrator/extensions.py, private_row).
        or extensions.private_row(tool_name)
        # A watch on the speaker's own inbox (watchlist_tools.speakers_own_row).
        or speakers_own_row(tool_name, tool_input, result)
    )


def _artifact_row_owner(tool_name: str, tool_input: Any = None, result: Any = None) -> int | None:
    """The person a dispatch row belongs to alone, else None: a document
    tool's input and result quote the speaker's own documents, and a
    watch-list call on their own inbox quotes their mail."""
    if tool_name not in DRAFT_ARTIFACT_TOOL_HANDLERS and not speakers_own_row(
        tool_name, tool_input, result
    ):
        return None
    from openexecutive.orchestrator.artifact_records import current_viewer

    return current_viewer().person_id


def _pushback_for_turn(
    settings: Any, session: Session, text: str, normal_model: str
) -> pushback_mod.Pushback | None:
    """The pushback model for this turn, when the person pushed back on the
    last answer (``orchestrator.pushback``). The Executive's Council pick
    wins over ``PUSHBACK_MODEL``. None runs the turn on ``normal_model``."""
    try:
        from openexecutive.agents.overrides import EXECUTIVE_AGENT_ID, get_override

        ov = get_override(EXECUTIVE_AGENT_ID)
        return pushback_mod.for_turn(
            session_id=session.session_id,
            text=text,
            history=session.conversation_history,
            normal_model=normal_model,
            override_pick=ov.pushback_model if ov is not None else None,
            default=getattr(settings, "pushback_model", "") or "",
        )
    except Exception:
        logger.warning("pushback: couldn't pick the turn's model", exc_info=True)
        return None


def _log_pushback(
    pushback: pushback_mod.Pushback | None, session: Session, turn_id: str
) -> None:
    if pushback is None:
        return
    audit_log(
        "pushback_model",
        f"Pushback on the last answer ({pushback.reason}); this answer on {pushback.model}",
        session_id=session.session_id,
        turn_id=turn_id,
        actor="executive",
        details={"reason": pushback.reason, "model": pushback.model},
    )


def _speaker_text(memory_text: str | None, user_message: str) -> str:
    """The speaker's own words for this turn, which every post-turn pass reads
    instead of the prompt — episodic extraction, open loops and peer memory:
    the caller's ``memory_text`` when given (``""`` included — it means
    "nothing the person wrote"), else the prompt itself."""
    return memory_text if memory_text is not None else user_message


def _turn_lessons(session: Session, person_id: int | None, text: str) -> str:
    """The ``<lessons>`` body for this turn (``memory.lessons``), counted
    under the surface the turn is on: the session's ``memory_surface`` when
    its caller set one, an email for a turn that answers one, else a chat.
    A turn with no single speaker (a shared call) reads the lessons of
    ``session.lessons_person_id``, whoever the call belongs to, and is not
    graded against them: nobody there is that person speaking alone."""
    from openexecutive.memory import lessons as lessons_store

    surface = str(getattr(session, "memory_surface", "") or "")
    if surface not in lessons_store.SURFACES:
        surface = (
            lessons_store.SURFACE_EMAIL
            if getattr(session, "origin_channel", "") == "email"
            else lessons_store.SURFACE_CHAT
        )
    owner_id = getattr(session, "lessons_person_id", None)
    shared = person_id is None and owner_id is not None
    return lessons_store.build_lessons_block(
        owner_id if shared else person_id,
        text,
        count=not chat_replay.is_replay(session),
        surface=surface,
        graded=not shared,
    )


def _lookup_text(memory_text: str | None, user_message: str) -> str:
    """What the memory lookups (lessons, figures, standing facts) rank
    against: the speaker's own words when the caller gave them, so notes put
    before the question (a call's recap, a shared screen) don't steer them;
    the prompt when there are none."""
    return _speaker_text(memory_text, user_message).strip() or user_message


def _sync_consulted_departments_to_honcho(
    consulted_specialists: list[str],
    user_message: str,
    response: str,
    *,
    person_id: int | None,
    session_id: str | None,
    co_present_person_ids: list[int] | None,
) -> None:
    """Fire-and-forget per-dept Honcho sync for every department whose
    specialist contributed to this turn.

    Resolves each consulted specialist key → department slug via the
    departments registry, deduplicates (the same specialist may have been
    consulted in multiple iterations of the tool-use loop), and emits
    one ``sync_department_turn`` per unique dept slug. Person and other
    consulted depts join as co-present peers so Honcho's peer graph can
    cross-pollinate the dept and person representations later.

    Specialists with no owning department (e.g. ``triage``) are skipped.
    """
    if not consulted_specialists:
        return
    from openexecutive.departments.registry import slug_for_specialist
    from openexecutive.memory.honcho_client import sync_department_turn

    dept_slugs: list[str] = []
    seen: set[str] = set()
    for key in consulted_specialists:
        slug = slug_for_specialist(key)
        if slug is None or slug in seen:
            continue
        seen.add(slug)
        dept_slugs.append(slug)
    for slug in dept_slugs:
        sync_department_turn(
            user_message,
            response,
            department_slug=slug,
            session_id=session_id,
            originating_person_id=person_id,
            co_present_person_ids=co_present_person_ids,
            co_present_department_slugs=[s for s in dept_slugs if s != slug],
        )


def _system_block_names(blocks: list[dict[str, Any]]) -> list[str]:
    """Derive stable, human-readable labels for system prompt blocks.

    Used to annotate specialist_consult / tool_invocation full payloads so
    the timeline can show which cached blocks were live during each step.
    Block layout in cache_manager.build_system_blocks() is fixed:
      0 = persona + knowledge_index (1h TTL)
      1 = company_profile + org_context (5m TTL, or unmarked when the
          agent loop's history marker took its slot)
    """
    names: list[str] = []
    for i, b in enumerate(blocks):
        cc = b.get("cache_control")
        # No marker: the block rides inside a later breakpoint's prefix (the
        # history marker takes the company block's slot).
        ttl = cc.get("ttl", "5m") if isinstance(cc, dict) else "none"
        if i == 0:
            names.append(f"persona+knowledge_index (ttl={ttl})")
        elif i == 1:
            names.append(f"company_profile+org_context (ttl={ttl})")
        else:
            names.append(f"system_block_{i} (ttl={ttl})")
    return names


def _emit_memory_snapshot(
    *,
    session_id: str | None,
    turn_id: str,
    user_message: str,
    episodic_context: str,
    retrieved_context: str,
    system_blocks: list[dict[str, Any]],
    history_len: int,
    company_profile: Any,
    model: str,
    committee: bool,
    working_style: str = "",
    standing_facts: str = "",
) -> None:
    """One memory_snapshot per turn: what was in the prompt before the API call.

    Fire-and-forget. Captures both the *summary* (for the audit list view)
    and the *full* episodic + profile text (for the flow-chart detail pane).
    """
    # Only hash the company profile — the rendered prompt block can contain
    # revenue, headcount, hiring plans, and other commercially sensitive
    # facts. The hash is enough to spot "profile changed across turns" in
    # the flow view; the full block is reachable via /company-profile for
    # operators with explicit permission.
    profile_hash: str | None = None
    if company_profile is not None:
        try:
            block = company_profile.to_prompt_block() or ""
            profile_hash = hashlib.sha256(block.encode("utf-8")).hexdigest()[:12]
        except Exception:
            # Profile rendering can fail when company data is malformed;
            # don't let it break the chat turn.
            profile_hash = None
    audit_log(
        "memory_snapshot",
        f"turn entry: model={model} history={history_len} episodic={len(episodic_context)}c rag={len(retrieved_context)}c",
        session_id=session_id,
        turn_id=turn_id,
        actor="executive",
        details={
            "model": model,
            "committee_review": committee,
            "history_len": history_len,
            "episodic_chars": len(episodic_context),
            "retrieved_chars": len(retrieved_context),
            "user_message_preview": user_message[:160],
            "company_profile_hash": profile_hash,
            "system_blocks": _system_block_names(system_blocks),
            "working_style_chars": len(working_style),
            "standing_facts_chars": len(standing_facts),
        },
        # Do NOT duplicate the raw user_message here — chat_turn already
        # persists it in its own row. The episodic_context and
        # retrieved_context are LLM-generated summaries / retrieved chunks
        # already produced for this turn; capturing them here lets the
        # flow chart show "what the agent saw" without re-running RAG.
        full={
            "episodic_context": episodic_context,
            "retrieved_context": retrieved_context,
            "system_blocks": _system_block_names(system_blocks),
            "working_style": working_style,
            "standing_facts": standing_facts,
        },
        # Recall that quotes the speaker's own documents keeps the row theirs.
        **_recalled_documents_owner(retrieved_context),
    )


def _canonical_specialist(raw: Any) -> str:
    """Registry key for a model-emitted specialist name, or the name unchanged.

    Returning the original on failure (rather than "" or None) keeps the
    unresolvable case flowing to ``route_to_specialist``, which answers with an
    error string the model can act on and audits the anomaly there.

    The CORRECTION is audited here, because this is where it happens. Once the
    name is canonical every later layer sees a well-formed value and has
    nothing to report — `route_to_specialist`'s own normalisation branch is
    unreachable from the chat path for exactly that reason, so relying on it
    would leave every corrected name unrecorded.
    """
    resolved = resolve_specialist_name(raw)
    if resolved is None:
        return str(raw)
    if resolved != raw:
        audit_name_resolution(raw, resolved, source="chat_loop")
    return resolved


def _audit_specialist_consults(
    *,
    run_calls: list[dict[str, str]],
    specialist_results: list[str],
    session_id: str | None,
    turn_id: str | None,
    iteration: int,
    phase: str,
    duration_ms: int,
    conversation_context: str,
    system_blocks: list[dict[str, Any]],
) -> None:
    """Write one ``specialist_consult`` row per dispatched consult.

    Shared by the routing pre-pass and the tool-use loop so the row shape is
    defined once: the audit graph and /audit/usage read these rows, so two
    sites drifting apart shows up as a broken flow chart rather than a failing
    test. ``phase`` distinguishes the two ("prepass" / "loop").
    """
    # Per-batch, not per-call: identical for every row here, and it walks the
    # block list to build the names.
    prompt_blocks = _system_block_names(system_blocks)
    context_preview = conversation_context[:200]
    for call, spec_result in zip(run_calls, specialist_results, strict=True):
        audit_log(
            "specialist_consult",
            f"Consulted {call['specialist']}: {str(call['query'])[:160]}",
            session_id=session_id,
            turn_id=turn_id,
            actor=call["specialist"],
            details={
                "iteration": iteration,
                "phase": phase,
                "duration_ms": duration_ms,
                # One shared value per turn, not per call — read it from
                # what route_parallel was given, never from the call dict.
                "context_preview": context_preview,
            },
            full={
                "query": call["query"],
                "context": conversation_context,
                "response": spec_result,
                "active_prompt_blocks": prompt_blocks,
            },
        )

def _recalled_documents_owner(retrieved_context: str) -> dict[str, Any]:
    """`private` / `private_to_person` for a row quoting ``retrieved_context``
    when it recalled a published document (`knowledge.retriever` labels them),
    else nothing."""
    if "[published artifact " not in retrieved_context:
        return {}
    from openexecutive.orchestrator.artifact_records import current_viewer

    return {"private": True, "private_to_person": current_viewer().person_id}


def _emit_cache_event(
    *,
    session_id: str | None,
    turn_id: str,
    iteration: int,
    final_msg: Any,
    model: str,
    actor: str = "executive",
) -> None:
    """Record the token + cache stats of one Executive response as a
    ``cache_event`` row (see ``audit.usage.log_model_usage``, which every
    other model call site uses too). Reads ``final_msg.usage`` after the
    call, never touches the request, never raises."""
    log_model_usage(
        final_msg,
        model=model,
        actor=actor,
        iteration=iteration,
        session_id=session_id,
        turn_id=turn_id,
    )


# What the person sees for a round that was cut off: the step it was on was
# writing something long, not the read before it.
CUT_OFF_LABEL = "Writing it in shorter parts…"
REPEATING_EMPTY_REPLY = (
    "I stopped because the same step kept coming back the same way, and "
    "nothing was ready to show yet."
)
STEPS_EMPTY_REPLY = (
    "I've used all the steps I get for one message. Ask me to carry on and "
    "I'll pick up the rest."
)
# A long web search can pause a round (``pause_turn``); at most this many
# are carried on in one turn.
MAX_PAUSES = 5
FULL_EMPTY_REPLY = (
    "I've read all I can hold for this one question. Ask me to carry on and "
    "I'll pick up the rest."
)
# Said when it ends that way.
EMPTY_CUTS_REPLY = (
    "I kept running out of room before I could finish a step of this. "
    "Ask me to carry on, or to do it in smaller steps."
)


# Said in place of a round's results when nothing came back to put there (a
# round the model ended on a tool call the loop had no call for). The API
# refuses a user message with no content, which would end the whole turn
# mid-task, after its earlier steps already ran.
EMPTY_ROUND_NOTE = (
    "Nothing came back from that step. Carry on from where you are: make the "
    "call again if you still need it, or answer from what you have."
)

async def _kept_if_unwritten(stream: AsyncIterator[Any], thinking: str) -> AsyncIterator[Any]:
    """``stream`` as is, except that a ``rewrite`` whose new reply writes
    nothing (an empty round, a stop or a cut-off before its first word)
    streams the first reply again, so every reader (the saved reply, the web
    and phone chats, an email or chat app) ends with words, never a blank."""
    since: list[str] = []
    before: list[str] = []
    rewritten = False
    async for item in stream:
        if item == {"type": "rewrite"}:
            # A second rewrite keeps the newest words anyone saw.
            if "".join(since).strip():
                before = since
            since, rewritten = [], True
        elif isinstance(item, str) and item != thinking:
            since.append(item)
        yield item
    if rewritten and not "".join(since).strip() and before:
        # Exactly as it streamed, so a reader can tell it from a new reply.
        yield "".join(before)


def _round_separator(tail: str, text: str) -> str:
    """What goes between the text already streamed (its last characters,
    ``tail``) and a new round's first ``text``: a paragraph break, less what
    either already has."""
    if tail.endswith("\n\n") or text.startswith("\n\n"):
        return ""
    if tail.endswith("\n") or text.startswith("\n"):
        return "\n"
    return "\n\n"


# Longest text before a tool call that is still a working note.
WORKING_NOTE_MAX_CHARS = 300
# A sentence that only says what it does next: how a working note reads
# ("Now the live data.", "Let me check the folder.", "Checking the sheet
# now."). Matched whole: a short sentence with no comma, colon or figure,
# so a sentence that carries a finding ("Searching the CRM turned up
# nothing.", "Now the bad news: Acme churned.") is never one.
_NEXT_STEP_RE = re.compile(
    r"^(?:(?:let me|let's) (?:check|look|pull|find|search|try|see|get|grab|open|read|dig)\b"
    r"|now (?:to|for|on to|let me|i'll|i will)\b|next (?:to|let me|i'll|i will)\b"
    r"|(?:now|next) the \w+(?: \w+)?[.!]?$|time to (?:check|look|search|pull|find|dig)\b"
    r"|i'll (?:check|look|search|pull|find|try)\b|i will (?:check|look|search|pull|find|try)\b"
    r"|(?:checking|looking|searching|pulling)\b.*\bnow[.!]?$)",
    re.IGNORECASE,
)
# Longest sentence a working note's step can be.
NEXT_STEP_MAX_CHARS = 80
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
# A call that only looks something up (whole name segments, as in "search_gmail_messages").
_LOOKUP_CALL_RE = re.compile(
    r"(?:^|_)(?:search|read|get|list|find|lookup|query|fetch|consult|retrieve|recall|open)(?:_|$)", re.IGNORECASE
)
# Calls that neither look nor act: they don't change what a round is.
_NEUTRAL_CALLS = frozenset({"show_progress"})


def _only_looks(tool_uses: list[dict[str, Any]]) -> bool:
    """Whether every call a round made only looks something up: a connected
    app's call (call_tool) or a group's (use_tool) by the tool it runs. A
    round that also acts (a send, a save) wrote its text as part of the
    answer ("Runway is 14 months." before the email with it)."""
    names = []
    for tu in tool_uses:
        name = str(tu.get("name", ""))
        given = tu.get("input")
        if name in ("call_tool", "use_tool") and isinstance(given, dict):
            name = str(given.get("name") or name)
        if name not in _NEUTRAL_CALLS:
            names.append(name)
    return bool(names) and all(_LOOKUP_CALL_RE.search(n) for n in names)


def _is_next_step(sentence: str) -> bool:
    return (
        len(sentence) <= NEXT_STEP_MAX_CHARS
        and not any(c.isdigit() or c in ",:;" for c in sentence)
        and _NEXT_STEP_RE.match(sentence) is not None
    )


def _less_notes(text: str) -> str | None:
    """A round's text before its lookups without its working-note
    sentences (``_is_next_step``), when it had any, else None. Only a line
    or two with no list or table is looked at, so "Yes, Acme signed it. Let
    me pull the signed copy." keeps its answer."""
    note = text.strip()
    if not note or len(note) > WORKING_NOTE_MAX_CHARS or "\n" in note or "|" in note:
        return None
    sentences = _SENTENCE_END_RE.split(note)
    kept = [
        s for s in sentences if not _is_next_step(s)
    ]
    if len(kept) == len(sentences):
        return None
    if not kept:
        return ""
    # Its paragraph break before it (the round's separator) stays.
    return text[: len(text) - len(text.lstrip())] + " ".join(kept)


def _without_notes(rounds: list[tuple[str, bool]]) -> str | None:
    """The reply without its working notes, when it had any, else None.
    ``rounds`` is each round's streamed text (its separator first) and
    whether the round went on only to look something up (``_only_looks``,
    and not the rest of a cut round's text). A note stays when no round
    after it wrote anything: then it is all the reply there is."""
    last_text = max((i for i, (text, _) in enumerate(rounds) if text.strip()), default=-1)
    out = ""
    changed = cut = False
    for i, (text, looked) in enumerate(rounds):
        rest = _less_notes(text) if looked and i < last_text else None
        if cut and out.strip() and text and not out[-1:].isspace() and not text[:1].isspace():
            # Text after a web search in the same round follows the note
            # with nothing between; with the note gone it needs its break.
            out += _round_separator(out[-2:], text)
        cut = rest is not None
        if rest is None:
            out += text
        else:
            changed = True
            out += rest
    if not changed:
        return None
    return out.lstrip()


def _round_parts(
    lead: str, text: str, content: list[dict[str, Any]], looked: bool, carries_on: bool
) -> list[tuple[str, bool]]:
    """A round's text for ``_without_notes``, split where it ran a web
    search: a web search runs inside the round, so the text before one was
    written before a lookup, like the text before a lookup call. ``lead`` is
    the round's separator, ``looked`` whether its calls only look
    (``_only_looks``) and ``carries_on`` whether it starts with the rest of
    a cut round's text."""
    parts: list[str] = []
    current = ""
    for block in content:
        if block.get("type") == "text":
            current += str(block.get("text", ""))
        elif block.get("type") == "server_tool_use" and block.get("name") == WEB_SEARCH_TOOL_NAME and current:
            parts.append(current)
            current = ""
    if not parts or "".join(parts) + current != text:
        return [(lead + text, looked)]
    out = [(part, not (i == 0 and carries_on)) for i, part in enumerate(parts)]
    out[0] = (lead + out[0][0], out[0][1])
    out.append((current, looked))
    return out


class Executive:
    """The Executive orchestrator — the single voice the user always interacts with.

    Internally routes to specialist agents via Anthropic tool use, but synthesizes
    everything into one coherent executive response. The sub-agent architecture is
    never exposed to the user.
    """

    def __init__(self, mcp_gateway: MCPGateway | None = None) -> None:
        self._settings = get_settings()
        self._mcp_gateway = mcp_gateway
        self._mcp_tools = MCP_TOOLS if mcp_gateway is not None else []
        # run_script (workflows/step_script.py): one sandboxed script over the
        # gateway tools a conversation has found, each call checked as a
        # call_tool. Only with a gateway, and a constant definition so the
        # cached tool prefix is stable.
        self._script_tools = (
            [step_script.CHAT_TOOL_DEFINITION, step_script.LIST_SAVED_TOOLS_DEFINITION]
            if mcp_gateway is not None
            and self._settings.chat_scripts
            and step_script.available()
            else []
        )

    def _build_messages(
        self,
        session: Session,
        user_message: str,
        retrieved_context: str = "",
        episodic_context: str = "",
        attachment_blocks: list[dict[str, Any]] | None = None,
        peer_memory_context: str = "",
        person_id: int | None = None,
        briefing_context: str = "",
        channel_context_block: str = "",
        page_context_block: str = "",
        working_style: str = "",
        standing_facts: str = "",
        lessons: str = "",
        worked_out: str = "",
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        history = session.get_recent_history()

        for turn in history:
            # No cache_control here. The agent loop marks the newest history
            # turn itself (_apply_history_cache_marker), taking the company
            # block's slot so the request stays at the API's 4 breakpoints;
            # other users of these messages (the committee revision) keep
            # the plain layout. A rolling marker on every history turn used
            # to claim a latent fifth, which Anthropic rejects outright.
            prev = messages[-1] if messages else None
            if (
                prev is not None
                and prev["role"] == turn["role"]
                and isinstance(prev["content"], str)
                and isinstance(turn["content"], str)
            ):
                # Two user rows in a row: a message the person sent while
                # the Executive was working (turn_inbox) is stored as its own
                # row after the turn's message. One turn per role keeps the
                # history alternating.
                prev["content"] = f"{prev['content']}\n\n{turn['content']}"
                continue
            messages.append({"role": turn["role"], "content": turn["content"]})

        user_content_parts: list[dict[str, Any]] = []

        # When this turn is, so relative dates resolve against today rather
        # than a guess. Changes every turn, so the user turn, never a cached
        # system block.
        user_content_parts.append(
            {"type": "text", "text": f"<current_time>\n{_current_time_block()}\n</current_time>"}
        )
        # Name the human in the room first so the model never offers to loop
        # in / DM the person it is replying to. Lives in the user turn (never
        # the cached system block) since the speaker varies per request.
        speaker_block = _build_current_speaker_block(person_id)
        if speaker_block:
            user_content_parts.append(
                {"type": "text", "text": f"<current_speaker>\n{speaker_block}\n</current_speaker>"}
            )
        # How this speaker likes replies (attunement.style) — only ever their
        # own rules, in the user turn, never a cached system block.
        if working_style:
            user_content_parts.append(
                {"type": "text", "text": f"<working_style>\n{working_style}\n</working_style>"}
            )
        # What this workspace asked for differently before (memory.lessons):
        # learned from turns the conversation review graded a miss, in the
        # person's own words. Preferences only, in the user turn.
        if lessons:
            user_content_parts.append({"type": "text", "text": f"<lessons>\n{lessons}\n</lessons>"})
        # Figures it worked out for this person before (memory.worked_out),
        # with source and date: their own, ranked against this message.
        if worked_out:
            user_content_parts.append(
                {"type": "text", "text": f"<worked_out>\n{worked_out}\n</worked_out>"}
            )

        # Facts and corrections the principal asked to keep (memory/facts.py).
        # The same block every unattended prompt reads, so a correction made
        # here holds in the briefs, scheduled runs and alert review too. User
        # turn, never a cached system block: it changes whenever a fact does.
        if standing_facts:
            user_content_parts.append(
                {"type": "text", "text": f"<standing_facts>\n{standing_facts}\n</standing_facts>"}
            )
        if episodic_context:
            user_content_parts.append(
                {"type": "text", "text": f"<past_decisions>\n{episodic_context}\n</past_decisions>"}
            )
        # Current open-alert digest (the items behind the /today "What's going
        # on" narrative + proposal cards) so the Executive can discuss an item
        # the principal clicked or named. Lives in the user turn alongside the
        # other per-request context — never a cached system block.
        if briefing_context:
            user_content_parts.append(
                {"type": "text", "text": f"<briefing>\n{briefing_context}\n</briefing>"}
            )
        # Which surface the principal is talking to us on, and what can and
        # cannot be completed there. Chat adapters (Slack, Discord, Telegram)
        # set this; the web app leaves it empty because the model is already
        # in the app. User turn, never a cached system block — it varies per
        # request and would otherwise invalidate the prompt cache.
        if channel_context_block:
            user_content_parts.append(
                {
                    "type": "text",
                    "text": f"<channel>\n{channel_context_block}\n</channel>",
                }
            )
        # Per-person memory from Honcho (when HONCHO_ENABLED + person_id
        # available). Goes in the user turn alongside episodic / retrieved
        # context so it never lives in the cached system block.
        if peer_memory_context:
            user_content_parts.append(
                {"type": "text", "text": f"<peer_memory>\n{peer_memory_context}\n</peer_memory>"}
            )
        # What the user is looking at right now (Ask OE panel turns only):
        # route, page title, guide excerpt, and — when a form is on screen —
        # its field descriptor. Per-request by nature, so it lives in the
        # user turn alongside the other dynamic context, never a cached
        # system block.
        if page_context_block:
            user_content_parts.append(
                {"type": "text", "text": f"<page_context>\n{page_context_block}\n</page_context>"}
            )
        if retrieved_context:
            user_content_parts.append(
                {
                    "type": "text",
                    "text": f"<retrieved_context>\n{retrieved_context}\n</retrieved_context>",
                }
            )
        # The Drive files this conversation already found or opened, and the
        # searches that matched nothing (memory.drive_reads), so "the file you
        # found earlier" resolves after history has kept only the prose.
        # Per session, so the user turn — never a cached system block.
        drive_memory = _drive_memory_block(session, person_id)
        if drive_memory:
            user_content_parts.append(
                {"type": "text", "text": f"<drive_memory>\n{drive_memory}\n</drive_memory>"}
            )
        # The speaker's own chats from the last two days, titles only
        # (chat_search_tools.recent_chats_block), so "we did it earlier"
        # reaches search_past_chats. Only where that tool is offered; per
        # turn, so the user turn.
        recent_chats = recent_chats_block(session)
        if recent_chats:
            user_content_parts.append(
                {"type": "text", "text": f"<recent_chats>\n{recent_chats}\n</recent_chats>"}
            )
        # The follow-ups this speaker scheduled that haven't gone out
        # (correction_tools.pending_followups_block), so a turn about the
        # thing one waits for closes it instead of leaving it to fire stale.
        # Changes as they are set and sent, so the user turn.
        followups = pending_followups_block(session)
        if followups:
            user_content_parts.append(
                {"type": "text", "text": f"<pending_followups>\n{followups}\n</pending_followups>"}
            )
        # What this speaker asked for in the last day, in any chat or app,
        # that went or waits on its card (take_the_lead.done_lately_block),
        # so the same ask sent again in a new chat is that ask, not a second
        # email or card. Changes with every step, so the user turn.
        from openexecutive.orchestrator.take_the_lead import done_lately_block

        lately = done_lately_block(session)
        if lately:
            user_content_parts.append({"type": "text", "text": f"<done_lately>\n{lately}\n</done_lately>"})

        # Attachment blocks (images) are inserted before the text message so
        # the model sees the visual context first, then the user's question.
        # Text-extracted document content is already prepended to user_message
        # by the caller — no separate block needed for those.
        if attachment_blocks:
            user_content_parts.extend(attachment_blocks)
            # Images reach the Executive ONLY. `build_attachment_output` returns
            # no text for an image, so nothing about it lands in `user_message`
            # — and a specialist's whole view is the rendered conversation tail
            # built from that string (`Session.render_conversation_context` ->
            # `route_parallel(conversation_context=...)`), which carries text,
            # never content blocks. Without this instruction a consult on an
            # image turn sends the specialist neither the image nor any
            # description of it, and it answers from priors while the Executive
            # synthesizes the result as authoritative (PR #18, Codex P1).
            # Forwarding the blocks themselves needs vision gating per
            # specialist model and is tracked in issue #19; relaying the
            # findings through `query` is what restores the evidence path here.
            # Inert on the committee revision pass, which is sent no tools and
            # so has no consult to relay into — harmless, and cheaper than
            # threading a flag through `_build_messages` for one extra turn.
            user_content_parts.append(
                {
                    "type": "text",
                    "text": (
                        "<attachment_notice>\n"
                        "The attached image(s) are visible to you alone. Any "
                        "specialist you consult receives only text, so it "
                        "cannot see them. If you consult a specialist about "
                        "anything shown in an attachment, restate the relevant "
                        "figures, readings, labels or wording inside that "
                        "call's `query` — the specialist has no other way to "
                        "reach them.\n"
                        "</attachment_notice>"
                    ),
                }
            )

        user_content_parts.append({"type": "text", "text": user_message})
        messages.append({"role": "user", "content": user_content_parts})

        return messages

    # Sentinel yielded when specialist calls are in flight — lets callers send keepalives.
    _THINKING = "\x01"

    async def stream_chat(
        self,
        user_message: str,
        session: Session,
        retrieved_context: str = "",
        episodic_context: str = "",
        debug_collector: DebugCollector | None = None,
        attachment_blocks: list[dict[str, Any]] | None = None,
        person_id: int | None = None,
        co_present_person_ids: list[int] | None = None,
        peer_memory_reasoning_level: HonchoReasoningLevel = "minimal",
        peer_memory_context: str | None = None,
        briefing_context: str = "",
        channel_context_block: str = "",
        page_context_block: str = "",
        turn_id: str | None = None,
        memory_text: str | None = None,
        turn_sources: TurnSources | None = None,
        standing_facts: str | None = None,
        inbox: TurnInbox | None = None,
        fetch_own_urls: bool = True,
    ) -> AsyncIterator[str | dict[str, Any]]:
        """Stream a response from the Executive, routing to specialists as needed.

        ``turn_sources`` (the web chat route passes one) collects what the
        answer looked at and which areas it had to leave out; see
        ``orchestrator.answer_sources``. Other callers pass nothing.

        ``person_id`` (when provided) keys the Honcho per-person memory
        prefetch + post-turn sync. The integration adapters (Slack,
        Discord, Telegram, email) resolve the inbound user to a
        :class:`openexecutive.people.models.Person` and pass the id here;
        the web/SSE path leaves it ``None`` and the Honcho layer no-ops.

        ``co_present_person_ids`` (when provided) lists every distinct
        human in the conversation context besides the sender (Discord
        thread participants, Slack channel speakers, email cc'd people).
        They're added to the Honcho session as peers so peer-of-peer
        reasoning (via the ``ask_about_person`` tool) has data.

        ``peer_memory_reasoning_level`` trades latency for synthesis
        depth on Honcho's dialectic prefetch; it applies only when
        ``HONCHO_PREFETCH_MODE=dialectic`` (the default representation
        mode reads derived memory with no LLM call and ignores it). Default ``"minimal"``
        bounds Honcho's synthesis depth — production telemetry showed
        ``"low"`` was hitting the ~5s tail consistently for power-user
        peers as their representations grew. The committee path keeps
        ``"medium"`` since it already pays the deep-review latency cost.

        ``memory_text`` is the person's own words for this turn — what peer
        memory records, and what episodic extraction and the open-loop pass
        must quote a commitment from; ``None`` uses ``user_message``. Pass it
        whenever ``user_message`` carries text the person did not write (an
        inbound email's headers and quoted chain, a briefing card's body, an
        attachment's extracted text): Honcho derives facts about the person
        from everything recorded under their peer, and a quote gate satisfied
        by the Executive's own words stores its recommendation as the
        person's decision or open loop. It is also the only text whose URLs
        the client ``web_fetch`` may open: ``None`` still uses
        ``user_message`` for the passes above, but makes nothing in it
        fetchable. ``fetch_own_urls=False`` makes nothing in ``memory_text``
        fetchable either, while it still feeds those passes: ``chat()``
        leaves it off unless the caller turns it on, which a channel does only
        for a sender on the roster.
        """
        logger.info(
            "chat turn: %s",
            _trunc(user_message, 80),
            extra={"turn_break": True},
        )
        # Expose the current session to tool handlers (e.g. schedule_followup)
        # without threading it through every signature.
        current_session.set(session)
        # The resolved speaker for THIS turn (None for an unrostered sender),
        # so tool handlers can tell who is asking — e.g. close_open_loop only
        # lets the principal or the loop's owner close a loop.
        session.caller_person_id = person_id
        # Persona and model can be overridden via the Agent Council admin UI.
        # Override is admin-set (not per-request dynamic), so placing it in the
        # cached block is fine — cache misses once on change, then hits normally.
        # Wrap in try/except so an override-store outage doesn't block chat.
        persona_override: str | None = None
        persona_instructions: str | None = None
        voice_persona_body: str | None = None
        effective_model = self._settings.default_model
        try:
            from openexecutive.agents.overrides import (
                EXECUTIVE_AGENT_ID,
            )
            from openexecutive.agents.overrides import (
                get_override as _get_override,
            )
            from openexecutive.personas.loader import get_active_body as _get_voice_body
            _ov = _get_override(EXECUTIVE_AGENT_ID)
            # Use falsy check so an empty-string override is treated as absent
            # (an empty persona sent to the API would produce garbled output).
            if _ov is not None and _ov.prompt:
                persona_override = _ov.prompt
            if _ov is not None:
                persona_instructions = _ov.instructions
            if _ov is not None and _ov.model:
                effective_model = _ov.model
            voice_persona_body = _get_voice_body(
                session.voice_persona_slug
                or (_ov.voice_persona_slug if _ov else None)
            )
        except Exception:
            logger.exception("Failed to load executive override; using defaults")
        # Pushback on the last answer: this one answer on the pushback model.
        pushback = _pushback_for_turn(
            self._settings, session, _speaker_text(memory_text, user_message), effective_model
        )
        if pushback is not None:
            effective_model = pushback.model
        # Resolved once and handed to both the prompt and the tool loop, so
        # the persona can never promise a search tool the request lacks.
        search = select_web_search_tool(effective_model)
        # Solo / team, resolved once per turn and pinned on the session, so the
        # persona, the org block, the toolkit the loop offers and every tool
        # handler agree even if the setting flips mid-turn (Session override →
        # workspace).
        workspace_mode = pin_turn_workspace_mode(session)
        # Solo: the principal's role (the session's override, else the
        # workspace's), pinned for the turn with the mode so the org block,
        # every specialist's <principal_role> tag and any workflow the turn
        # starts describe the same role. Team pins an empty role and renders
        # none.
        principal_role = pin_turn_principal_role(session, workspace_mode)
        # Act as me, pinned with the mode so the tool list, the handler and
        # the turn's audit privacy agree even if the setting flips mid-turn.
        delegation_pin = pin_turn_delegation(session, _speaker_text(memory_text, user_message))
        system_blocks = build_system_blocks(
            session.company_profile,
            mcp_enabled=self._mcp_gateway is not None,
            persona_override=persona_override,
            voice_persona_body=voice_persona_body,
            web_search_available=search is not None,
            workspace_mode=workspace_mode,
            principal_role=principal_role if workspace_mode == "solo" else None,
            include_contacts=_contacts_in_prompt(session),
            delegation=block0_delegation_on(session),
            mcp_servers=gateway_server_names(self._mcp_gateway),
            needs_reconnect=gateway_reconnect_needed(self._mcp_gateway),
            persona_instructions=persona_instructions,
        )
        # turn_id ties every downstream audit row (knowledge_retrieval,
        # specialist_consult, tool_invocation, cache_event, peer_memory)
        # to this exchange. set_turn binds the audit ContextVars scoped
        # to the current asyncio task — restored automatically on exit.
        # Bound BEFORE prefetch so the Honcho audit row inherits the
        # turn link (otherwise it lands with session_id=NULL and is
        # invisible in the session-grouped audit view).
        # A caller-supplied id wins: the SSE route binds the turn at route
        # level and must agree with us, or one turn splits across two ids.
        # Every other entry point passes nothing and keeps its own id.
        # `is None`, not `or`: an explicitly-passed empty string would
        # otherwise silently mint a second id and split one turn in two.
        turn_id = f"t-{uuid.uuid4().hex[:12]}" if turn_id is None else turn_id

        t0 = time.monotonic()
        full_response = ""
        with set_turn(session_id=session.session_id, turn_id=turn_id):
            _log_pushback(pushback, session, turn_id)
            # Per-person prefetch from Honcho. Runs once per turn (NOT inside
            # the tool-call loop) so a long multi-tool turn doesn't accrue
            # multiple round trips. The wrapper enforces a hard timeout and
            # returns "" on any error, so a Honcho outage degrades silently.
            #
            # Callers can pre-resolve this in parallel with RAG and episodic
            # context (see api/routes/chat.py) and pass the result through —
            # in that case we skip the await here entirely.
            if peer_memory_context is None:
                from openexecutive.memory.honcho_client import prefetch as _honcho_prefetch
                peer_memory_context = await _honcho_prefetch(
                    user_message,
                    person_id=person_id,
                    session_id=session.session_id,
                    reasoning_level=peer_memory_reasoning_level,
                )
            from openexecutive.attunement.style import build_style_block

            working_style = build_style_block(person_id)
            # What this workspace asked for differently before (memory.lessons):
            # a SQLite read, off the event loop like the facts.
            lessons = await asyncio.to_thread(
                _turn_lessons, session, person_id, _lookup_text(memory_text, user_message)
            )
            from openexecutive.memory.worked_out import build_worked_out_block

            worked_out = await asyncio.to_thread(
                build_worked_out_block, person_id, _lookup_text(memory_text, user_message), session=session
            )
            if standing_facts is None:
                # A SQLite read: off the event loop, like the other context.
                standing_facts = await asyncio.to_thread(
                    render_facts_for_prompt, query=_lookup_text(memory_text, user_message),
                    # Their own private decisions, in their own chat only.
                    reader=own_chat_person(session),
                )
            messages = self._build_messages(
                session,
                user_message,
                retrieved_context,
                episodic_context,
                attachment_blocks,
                peer_memory_context=peer_memory_context,
                person_id=person_id,
                briefing_context=briefing_context,
                channel_context_block=channel_context_block,
                page_context_block=page_context_block,
                working_style=working_style,
                standing_facts=standing_facts,
                lessons=lessons,
                worked_out=worked_out,
            )

            _emit_memory_snapshot(
                session_id=session.session_id,
                turn_id=turn_id,
                user_message=user_message,
                episodic_context=episodic_context,
                retrieved_context=retrieved_context,
                system_blocks=system_blocks,
                history_len=len(session.get_recent_history()),
                company_profile=session.company_profile,
                model=effective_model,
                committee=False,
                working_style=working_style,
                standing_facts=standing_facts,
            )
            # What this turn reads from outside the team, starting with the
            # message's own attachments and any synced excerpt retrieved for it
            # (orchestrator.outside_content). Closed when the loop ends; nothing
            # between the two can leave it open.
            outside_turn = outside_content.begin_turn(
                session,
                user_message=user_message,
                attachment_blocks=attachment_blocks,
                retrieved_context=retrieved_context,
                own_words=memory_text,
            )
            consulted: list[str] = []
            turn_calls: list[tuple[str, str]] = []
            try:
                async for item in _kept_if_unwritten(self._stream_agent_loop(
                    system_blocks,
                    messages,
                    model=effective_model,
                    search=search,
                    episodic_context=episodic_context,
                    debug_collector=debug_collector,
                    consulted_out=consulted,
                    calls_out=turn_calls,
                    turn_id=turn_id,
                    conversation_context=session.render_conversation_context(user_message),
                    turn_sources=turn_sources,
                    workspace_mode=workspace_mode,
                    principal_role_tag=principal_role_context(principal_role),
                    inbox=inbox,
                    # Never the _speaker_text fallback: a caller that passes no
                    # memory_text may have packed attachment text, mail or another
                    # session's backstory into user_message.
                    seen_url_text=(memory_text or "") if fetch_own_urls else "",
                ), self._THINKING):
                    if isinstance(item, str) and item != self._THINKING:
                        full_response += item
                    elif item == {"type": "rewrite"}:
                        # Written again whole: the history keeps only the new reply.
                        full_response = ""
                    yield item
            finally:
                outside_content.end_turn(outside_turn)

        if debug_collector:
            evt = debug_collector.emit("synthesis_done", {
                "total_duration_ms": round((time.monotonic() - t0) * 1000),
                "response_length": len(full_response),
            })
            yield debug_collector.to_sse_dict(evt)

        session.add_user_message(user_message)
        for added_text in inbox.taken_texts() if inbox is not None else []:
            session.add_user_message(added_text)
        session.add_assistant_message(full_response)
        # A turn that read or drafted in the speaker's own mailbox (Act as me)
        # stays private to them from here on, and teaches only their own
        # private memory, from what they typed (its reply quotes a draft built
        # from other people's mail). Read from this turn's own pin (a
        # concurrent turn on the session has its own).
        touched_mail = delegation_pin.touched_mail

        # Audit the Executive's outbound response. Without this, the audit
        # log only records the *inputs* to a turn (inbound message, memory
        # snapshot, retrievals, specialist consults, tool calls) but never
        # the response itself — making sessions hard to follow. Mirrors the
        # web /api/chat route's pattern, but at the unified place that every
        # entry point (web stream, .chat() wrapper used by Discord / Slack
        # / Telegram / Email / Google Chat) flows through.
        # Only emit on real text — an empty response means the agent loop
        # produced only tool_use blocks and no text,
        # and an audit row for "" adds noise without signal.
        if full_response.strip():
            audit_log(
                "chat_turn",
                f"Executive: {full_response[:200]}",
                session_id=session.session_id,
                turn_id=turn_id,
                actor="executive",
                details={
                    "direction": "out",
                    "response_len": len(full_response),
                    "duration_s": round(time.monotonic() - t0, 3),
                    "model": effective_model,
                    "committee": False,
                },
                full={"response": full_response},
                private=touched_mail,
            )

        from openexecutive.memory.episodic import (
            schedule_extraction,
            should_extract,
        )

        # Everything below reads the speaker's own words, not the prompt: the
        # extraction and open-loop passes accept an item only with a verbatim
        # quote from this text, and peer memory records it as what the person
        # said. A quoted Executive email or a briefing card's body in the
        # prompt would otherwise satisfy that quote gate with the Executive's
        # own words. A turn that touched the speaker's own mailbox (Act as me)
        # teaches only what is theirs alone, from what they typed
        # (``mail_words``): their private notes and their own peer memory,
        # without the reply (it quotes a draft built from other people's
        # mail). Its lessons come from the review (``record_turn``'s
        # ``mail``). Extraction, open loops and style skip it.
        own_text = with_added(_speaker_text(memory_text, user_message), inbox)
        # A host-surface turn (Session.from_host) is kept nowhere either: its
        # words may be a script's on the host's page, never the person's own.
        host_words = getattr(session, "from_host", False) is True
        speaker_text = "" if touched_mail or host_words else own_text
        # A replay of a graded turn (memory.replay) is kept nowhere: no open
        # loops, style, review or mirror of it (should_extract and
        # schedule_chat_notes refuse a replay session themselves).
        kept = not touched_mail and not host_words and not chat_replay.is_replay(session)
        repeat_offer: dict[str, Any] | None = None
        mail_words = _mail_turn_words(session, own_text, touched_mail=touched_mail, host_words=host_words)

        # Re-bind the audit ContextVars for the duration of these calls so
        # the fire-and-forget tasks they schedule can snapshot the right
        # session_id/turn_id (the main `with set_turn(...)` block above
        # exited at the end of the `async for` so the ContextVars are back
        # to None by now). Without this wrapper, every extraction /
        # sync_turn / sync_department_turn audit row would land with
        # session_id=NULL and be invisible in the per-session audit view.
        # private_rows: the passes scheduled here copy the context, so a turn
        # that touched the speaker's mailbox keeps their rows private too.
        with set_turn(session_id=session.session_id, turn_id=turn_id), private_rows(touched_mail):
            # Inside the wrapper: schedule_extraction snapshots the vars at
            # call time, so scheduling it out here would snapshot (None,
            # None) and the memory_extractor's model call would record
            # unattributed however correct the snapshot itself was.
            if should_extract(speaker_text, session=session):
                schedule_extraction(
                    speaker_text, full_response, session_id=session.session_id
                )

            # Open loops from ANY rostered speaker (not just the principal):
            # "I'll send the quote Thursday" becomes a loop the nudge engine
            # chases once due. `person_id` is the resolved speaker, so an
            # unrostered sender (None) records nothing.
            from openexecutive.attunement.open_loops import schedule_open_loop_pass

            if kept:
                schedule_open_loop_pass(
                    speaker_text, full_response, person_id=person_id,
                    session_id=session.session_id,
                    # The turn's pinned mode: solo opens the principal's own
                    # dated commitments, team does not — and only when this
                    # surface verified the speaker is the principal.
                    workspace_mode=workspace_mode,
                    principal_verified=is_principal_on_verified_surface(session),
                    private_to_person_id=private_owner(session),
                )
            # Re-learn this speaker's working style once enough new
            # messages have arrived (paced and budgeted inside).
            from openexecutive.attunement.style import schedule_style_pass

            if kept:
                schedule_style_pass(person_id, session_id=session.session_id)

            # Always in the loop: private notes of what the speaker said, when
            # they turned it on and this surface verified it is them
            # (history_chat). Their own words only, never the reply.
            from openexecutive.memory.history_chat import schedule_chat_notes

            # A turn that touched their mailbox notes their own words too:
            # notes are theirs alone (memory.history), and own_text is what
            # they typed, never the mail or the draft.
            schedule_chat_notes(speaker_text or mail_words, session=session, person_id=person_id)

            # Kept for the conversation review once the chat goes quiet
            # (memory.chat_review): a verified speaker's own words, the reply
            # and the tools the turn ran. A turn that touched their mailbox is
            # kept too, for their own review only (record_turn's ``mail``).
            from openexecutive.memory.chat_review import record_turn

            if not chat_replay.is_replay(session):
                record_turn(own_text, full_response, session=session,
                            person_id=person_id, calls=turn_calls, mail=touched_mail)

            # The same ask on a few different days: offer to do it on a
            # schedule (workflows.repeat_asks), under this reply and on
            # Workflows. Only words they typed: not a turn private to their
            # mailbox, a host page's or a replay (``kept``), nor one whose
            # words were composed for them (``memory_text``: a card's
            # Approve line, a hand-off, a shared chat's or attachment's).
            if kept and memory_text is None:
                from openexecutive.workflows.repeat_asks import note_ask

                repeat_offer = await asyncio.to_thread(
                    note_ask, own_text, session=session, person_id=person_id, calls=turn_calls,
                )

            # What it worked out (memory.worked_out): the figures the
            # Executive chose to keep this turn (keep_figure), checked against
            # the reply now that it is written and kept for this person alone
            # (private when the turn touched their mailbox).
            from openexecutive.memory.worked_out import KEEP_TOOL, keep_turn_figures

            # Not from a turn the person wrote nothing in (``memory_text=""``,
            # e.g. an app's own ask): its numbers repeat what it was shown.
            # Most turns keep nothing: only a turn that called keep_figure
            # opens the database.
            if own_text.strip() and any(name == KEEP_TOOL for name, _ in turn_calls):
                await asyncio.to_thread(keep_turn_figures, full_response, turn_calls, session=session,
                                        person_id=person_id, private=touched_mail)

            # Mirror the completed exchange into Honcho so its server-side
            # extraction can update the peer card. Fire-and-forget; the
            # wrapper no-ops when person_id is None or Honcho is disabled.
            # Never for a turn private to the principal (mail from one of
            # their contacts, mail they forwarded): the reply summarises that
            # mail, and peer and department memory are read on other people's
            # turns. Nor for a turn that touched the speaker's own mailbox.
            if not _private_to_principal(session) and kept and own_text.strip():
                from openexecutive.memory.honcho_client import sync_turn as _honcho_sync
                _honcho_sync(
                    speaker_text,
                    full_response,
                    person_id=person_id,
                    session_id=session.session_id,
                    co_present_person_ids=co_present_person_ids,
                )
                # Per-dept mirror for every department whose specialist
                # contributed this turn. Runs after the person-side sync so
                # dept and person syncs are visible in the audit log as a
                # related pair.
                _sync_consulted_departments_to_honcho(
                    consulted,
                    speaker_text,
                    full_response,
                    person_id=person_id,
                    session_id=session.session_id,
                    co_present_person_ids=co_present_person_ids,
                )
            elif mail_words:
                _sync_mail_turn_words(mail_words, person_id=person_id, session_id=session.session_id)

        # Outside the turn's context block: the next step may run on another task.
        if repeat_offer is not None:
            yield {"type": "repeat_offer", "offer": repeat_offer, "session_id": session.session_id}

    async def stream_chat_with_committee(
        self,
        user_message: str,
        session: Session,
        retrieved_context: str = "",
        episodic_context: str = "",
        debug_collector: DebugCollector | None = None,
        attachment_blocks: list[dict[str, Any]] | None = None,
        person_id: int | None = None,
        co_present_person_ids: list[int] | None = None,
        peer_memory_reasoning_level: HonchoReasoningLevel = "medium",
        peer_memory_context: str | None = None,
        briefing_context: str = "",
        channel_context_block: str = "",
        page_context_block: str = "",
        turn_id: str | None = None,
        memory_text: str | None = None,
        turn_sources: TurnSources | None = None,
        standing_facts: str | None = None,
        inbox: TurnInbox | None = None,
        fetch_own_urls: bool = True,
    ) -> AsyncIterator[str | dict[str, Any]]:
        """Committee-reviewed variant of stream_chat.

        ``turn_sources`` is as in ``stream_chat``: the draft's searches and
        specialists record into it.

        Flow: drafting (silent — text chunks accumulated, debug events
        passed through) → reviewing (3 parallel critique calls) →
        finalizing (revision streams to caller as text chunks).

        Only the revised response is yielded as text and persisted to
        history; the draft and critiques are recorded via audit_log so
        cache continuity is preserved against the polished version.

        Defaults ``peer_memory_reasoning_level="medium"`` (vs
        ``"minimal"`` on the non-committee path) — committee turns
        already pay deep-review latency, so a richer Honcho synthesis
        is worth the extra second + Sonnet-tier OpenRouter cost.
        """
        logger.info(
            "chat turn (committee): %s",
            _trunc(user_message, 80),
            extra={"turn_break": True},
        )
        current_session.set(session)
        # The resolved speaker for THIS turn (None for an unrostered sender),
        # so tool handlers can tell who is asking — e.g. close_open_loop only
        # lets the principal or the loop's owner close a loop.
        session.caller_person_id = person_id

        persona_override: str | None = None
        persona_instructions: str | None = None
        voice_persona_body: str | None = None
        effective_model = self._settings.default_model
        try:
            from openexecutive.agents.overrides import (
                EXECUTIVE_AGENT_ID,
            )
            from openexecutive.agents.overrides import (
                get_override as _get_override,
            )
            from openexecutive.personas.loader import get_active_body as _get_voice_body
            _ov = _get_override(EXECUTIVE_AGENT_ID)
            if _ov is not None and _ov.prompt:
                persona_override = _ov.prompt
            if _ov is not None:
                persona_instructions = _ov.instructions
            if _ov is not None and _ov.model:
                effective_model = _ov.model
            voice_persona_body = _get_voice_body(
                session.voice_persona_slug
                or (_ov.voice_persona_slug if _ov else None)
            )
        except Exception:
            logger.exception("Failed to load executive override; using defaults")
        # Pushback on the last answer: this one answer on the pushback model.
        pushback = _pushback_for_turn(
            self._settings, session, _speaker_text(memory_text, user_message), effective_model
        )
        if pushback is not None:
            effective_model = pushback.model

        # Gated on the DRAFT pass, which is the one that carries tools. The
        # revision pass below reuses these exact blocks on purpose — a second
        # variant would miss the draft's cached prefix on every committee
        # turn — so the addendum is as accurate there as it has always been.
        search = select_web_search_tool(effective_model)
        workspace_mode = pin_turn_workspace_mode(session)
        principal_role = pin_turn_principal_role(session, workspace_mode)
        delegation_pin = pin_turn_delegation(session, _speaker_text(memory_text, user_message))
        system_blocks = build_system_blocks(
            session.company_profile,
            mcp_enabled=self._mcp_gateway is not None,
            persona_override=persona_override,
            voice_persona_body=voice_persona_body,
            web_search_available=search is not None,
            workspace_mode=workspace_mode,
            principal_role=principal_role if workspace_mode == "solo" else None,
            include_contacts=_contacts_in_prompt(session),
            delegation=block0_delegation_on(session),
            mcp_servers=gateway_server_names(self._mcp_gateway),
            needs_reconnect=gateway_reconnect_needed(self._mcp_gateway),
            persona_instructions=persona_instructions,
        )
        # turn_id covers both the draft and (later) the revision pass so a
        # committee-reviewed turn renders as one flow chart, not two.
        # Generated and bound BEFORE the Honcho prefetch so the peer_memory
        # audit row inherits the turn link (else it lands with
        # session_id=NULL and is invisible in the session-grouped view).
        # A caller-supplied id wins: the SSE route binds the turn at route
        # level and must agree with us, or one turn splits across two ids.
        # Every other entry point passes nothing and keeps its own id.
        # `is None`, not `or`: an explicitly-passed empty string would
        # otherwise silently mint a second id and split one turn in two.
        turn_id = f"t-{uuid.uuid4().hex[:12]}" if turn_id is None else turn_id
        # Stash on the function frame so the closing audit_log("committee_review")
        # at the end of this function can carry the same turn_id.
        _committee_turn_id = turn_id

        # Bind ContextVars for the full turn (draft + review + revision).
        # bind_turn (no reset) matches the existing `current_session.set()`
        # pattern used for scheduling — fine for one-task-per-turn handlers.
        # A `with set_turn(...)` would force wrapping the entire committee
        # body in an extra indent level for negligible safety gain.
        #
        # Tradeoff note: binding before prefetch widens the window between
        # bind_turn and the eventual clear_turn at function end. If
        # something between here and the first clear_turn raises uncaught,
        # the binding leaks to the next turn on this asyncio task. Prefetch
        # swallows its own exceptions; _build_messages is pure. If a future
        # refactor adds a raising step in this window, wrap that step in
        # try/finally with clear_turn() or migrate to `with set_turn`.
        bind_turn(session_id=session.session_id, turn_id=turn_id)
        _log_pushback(pushback, session, turn_id)

        # Per-person prefetch from Honcho (see stream_chat for rationale).
        # Skip when caller pre-resolved (e.g. parallel context fetch in route).
        if peer_memory_context is None:
            from openexecutive.memory.honcho_client import prefetch as _honcho_prefetch
            peer_memory_context = await _honcho_prefetch(
                user_message,
                person_id=person_id,
                session_id=session.session_id,
                reasoning_level=peer_memory_reasoning_level,
            )
        from openexecutive.attunement.style import build_style_block

        working_style = build_style_block(person_id)
        lessons = await asyncio.to_thread(
            _turn_lessons, session, person_id, _lookup_text(memory_text, user_message)
        )
        from openexecutive.memory.worked_out import build_worked_out_block

        worked_out = await asyncio.to_thread(
            build_worked_out_block, person_id, _lookup_text(memory_text, user_message), session=session
        )
        if standing_facts is None:
            standing_facts = await asyncio.to_thread(
                render_facts_for_prompt, query=_lookup_text(memory_text, user_message),
                reader=own_chat_person(session),
            )
        messages = self._build_messages(
            session,
            user_message,
            retrieved_context,
            episodic_context,
            attachment_blocks,
            peer_memory_context=peer_memory_context,
            person_id=person_id,
            briefing_context=briefing_context,
            channel_context_block=channel_context_block,
            page_context_block=page_context_block,
            working_style=working_style,
            standing_facts=standing_facts,
            lessons=lessons,
            worked_out=worked_out,
        )

        # ----- Phase 1: drafting -----------------------------------------
        yield {
            "type": "phase",
            "phase": "drafting",
            "session_id": session.session_id,
        }

        t0 = time.monotonic()
        draft = ""
        consulted: list[str] = []
        specialist_outputs: dict[str, str] = {}
        _emit_memory_snapshot(
            session_id=session.session_id,
            turn_id=turn_id,
            user_message=user_message,
            episodic_context=episodic_context,
            retrieved_context=retrieved_context,
            system_blocks=system_blocks,
            history_len=len(session.get_recent_history()),
            company_profile=session.company_profile,
            model=effective_model,
            committee=True,
            working_style=working_style,
            standing_facts=standing_facts,
        )

        # What this turn reads from outside the team, starting with the
        # message's own attachments and any synced excerpt retrieved for it
        # (orchestrator.outside_content). Closed when the loop ends; nothing
        # between the two can leave it open.
        outside_turn = outside_content.begin_turn(
            session,
            user_message=user_message,
            attachment_blocks=attachment_blocks,
            retrieved_context=retrieved_context,
            own_words=memory_text,
        )
        turn_calls: list[tuple[str, str]] = []
        # A refused draft goes to no review or revision: they could rewrite
        # it into an answer.
        draft_refused: list[int] = []
        refusing = _REFUSED.set(draft_refused)
        try:
            async for item in _kept_if_unwritten(self._stream_agent_loop(
                system_blocks,
                messages,
                model=effective_model,
                search=search,
                episodic_context=episodic_context,
                debug_collector=debug_collector,
                consulted_out=consulted,
                specialist_outputs_out=specialist_outputs,
                calls_out=turn_calls,
                turn_id=turn_id,
                conversation_context=session.render_conversation_context(user_message),
                turn_sources=turn_sources,
                workspace_mode=workspace_mode,
                principal_role_tag=principal_role_context(principal_role),
                inbox=inbox,
                seen_url_text=(memory_text or "") if fetch_own_urls else "",  # as in stream_chat
            ), self._THINKING):
                # Swallow draft text and the THINKING sentinel — the user sees
                # only the revised stream. Pass debug-event dicts through so the
                # UI still gets specialist routing visibility.
                if isinstance(item, str):
                    if item != self._THINKING:
                        draft += item
                elif item == {"type": "rewrite"}:
                    # The draft is written again whole: reviewers see only that.
                    draft = ""
                else:
                    yield item
        finally:
            # Closed from another context (a client that went away), the
            # token can't be reset there; the turn still ends.
            with contextlib.suppress(ValueError):
                _REFUSED.reset(refusing)
            outside_content.end_turn(outside_turn)

        draft_ms = round((time.monotonic() - t0) * 1000)
        logger.info(
            "committee.draft_done duration_ms=%d consulted=%s draft_chars=%d",
            draft_ms,
            consulted,
            len(draft),
        )

        # If the draft came back empty (a timeout, a turn that wrote nothing) skip
        # review — there is nothing to critique. Still emit the remaining
        # phase events so the UI's stepper completes; otherwise it stays
        # frozen on "drafting" until the connection closes.
        if draft_refused or not draft.strip():
            yield {
                "type": "phase",
                "phase": "reviewing",
                "session_id": session.session_id,
            }
            yield {
                "type": "phase",
                "phase": "finalizing",
                "session_id": session.session_id,
            }
            fallback = (
                REFUSED_REPLY if draft_refused
                else "I was unable to complete the analysis. Please try again."
            )
            yield fallback
            session.add_user_message(user_message)
            for added_text in inbox.taken_texts() if inbox is not None else []:
                session.add_user_message(added_text)
            session.add_assistant_message(fallback)
            # Reset audit ContextVars so this task doesn't leak the turn_id
            # to a follow-up turn that runs on the same task.
            clear_turn()
            return

        # ----- Phase 2: reviewing ----------------------------------------
        yield {
            "type": "phase",
            "phase": "reviewing",
            "session_id": session.session_id,
        }

        from openexecutive.orchestrator.committee import Committee
        committee = Committee(
            reviewer_model=self._settings.default_model,
        )
        # Mirror the upcoming review onto the debug stream so the Agent
        # Activity panel shows committee progress alongside the inline
        # phase events that drive the in-chat stepper.
        planned_reviewers = [r.name for r in committee.select_reviewers(consulted)]
        if debug_collector:
            evt = debug_collector.emit("committee_review_start", {
                "reviewers": planned_reviewers,
                "consulted": consulted,
                "draft_length": len(draft),
            })
            yield debug_collector.to_sse_dict(evt)

        review_t0 = time.monotonic()
        critiques = await committee.review(
            user_message=user_message,
            draft=draft,
            consulted=consulted,
            specialist_outputs=specialist_outputs,
        )
        review_ms = round((time.monotonic() - review_t0) * 1000)
        logger.info(
            "committee.review_done duration_ms=%d critiques=%s",
            review_ms,
            [(c.reviewer_name, c.severity) for c in critiques],
        )

        if debug_collector:
            evt = debug_collector.emit("committee_review_done", {
                "review_ms": review_ms,
                "critiques": [
                    {
                        "reviewer": c.reviewer_name,
                        "severity": c.severity,
                        # Short preview only — the debug panel surfaces
                        # this on expand, full text lives in audit_log.
                        "critique_preview": c.critique[:200],
                    }
                    for c in critiques
                ],
            })
            yield debug_collector.to_sse_dict(evt)

        # Surface critique severities (not full text) to the UI for debug.
        # Full critiques stay server-side via audit_log below.
        for c in critiques:
            yield {
                "type": "committee_critique",
                "reviewer": c.reviewer_name,
                "severity": c.severity,
                "session_id": session.session_id,
            }

        # ----- Phase 3: finalizing (revision streams to caller) ----------
        yield {
            "type": "phase",
            "phase": "finalizing",
            "session_id": session.session_id,
        }

        if debug_collector:
            evt = debug_collector.emit("committee_revision_start", {
                "critique_count": len(critiques),
            })
            yield debug_collector.to_sse_dict(evt)

        from openexecutive.prompts.committee_prompts import build_revision_user_turn
        revision_turn = build_revision_user_turn(
            [c.as_dict() for c in critiques]
        )
        revision_messages = [
            *messages,
            {"role": "assistant", "content": draft},
            {"role": "user", "content": revision_turn},
        ]

        revision_t0 = time.monotonic()
        final_response = ""
        revision_final_msg: Any = None
        async with get_provider(effective_model).messages_stream(
            model=effective_model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system_blocks,  # type: ignore[arg-type]
            messages=revision_messages,  # type: ignore[arg-type]
        ) as stream:
            async for event in stream:
                if (
                    hasattr(event, "type")
                    and event.type == "content_block_delta"
                    and hasattr(event, "delta")
                    and hasattr(event.delta, "type")
                    and event.delta.type == "text_delta"
                ):
                    final_response += event.delta.text
                    yield event.delta.text
            try:
                revision_final_msg = await stream.get_final_message()
            except Exception:
                # Provider variance — some adapters may not implement
                # get_final_message after we've already iterated. Don't
                # let an audit-only branch break the revision response.
                revision_final_msg = None
        revision_ms = round((time.monotonic() - revision_t0) * 1000)
        if revision_final_msg is not None:
            _emit_cache_event(
                session_id=session.session_id,
                turn_id=_committee_turn_id,
                iteration=0,  # 0 = revision pass (post-draft, post-review)
                final_msg=revision_final_msg,
                model=effective_model,
                actor="committee_revision",
            )

        if debug_collector:
            evt = debug_collector.emit("synthesis_done", {
                "total_duration_ms": round((time.monotonic() - t0) * 1000),
                "response_length": len(final_response),
                "committee": True,
                "draft_length": len(draft),
                "review_ms": review_ms,
                "revision_ms": revision_ms,
            })
            yield debug_collector.to_sse_dict(evt)

        session.add_user_message(user_message)
        for added_text in inbox.taken_texts() if inbox is not None else []:
            session.add_user_message(added_text)
        # Act as me: a turn that touched the speaker's mailbox stays private
        # and teaches only their own private memory (see stream_chat).
        touched_mail = delegation_pin.touched_mail
        # Guard against a revision pass that produced no text (only tool_use
        # blocks, model_stop, etc.). Persisting an empty assistant turn
        # corrupts the in-memory history with a phantom turn that future
        # cache hits will key off of.
        if final_response.strip():
            session.add_assistant_message(final_response)

            # Audit the committee's final response. Same rationale as the
            # non-committee path in stream_chat: callers should always be
            # able to see the Executive's outbound text in the audit log,
            # not just the inputs. committee=True so the UI can distinguish
            # a committee-revised response from a normal one.
            audit_log(
                "chat_turn",
                f"Executive: {final_response[:200]}",
                session_id=session.session_id,
                turn_id=_committee_turn_id,
                actor="executive",
                details={
                    "direction": "out",
                    "response_len": len(final_response),
                    "duration_s": round(time.monotonic() - t0, 3),
                    "model": effective_model,
                    "committee": True,
                    "draft_length": len(draft),
                },
                full={"response": final_response, "draft": draft},
                private=touched_mail,
            )

        audit_log(
            "committee_review",
            f"Committee revised draft (consulted={consulted})",
            session_id=session.session_id,
            turn_id=_committee_turn_id,
            actor="committee",
            private=touched_mail,
            details={
                "draft_length": len(draft),
                "final_length": len(final_response),
                "consulted": consulted,
                "review_ms": review_ms,
                "revision_ms": revision_ms,
                "critiques": [
                    {
                        "reviewer": c.reviewer_name,
                        "severity": c.severity,
                        "critique": c.critique[:500],
                        "suggested_edits": c.suggested_edits[:500],
                    }
                    for c in critiques
                ],
            },
        )

        from openexecutive.memory.episodic import (
            schedule_extraction,
            should_extract,
        )
        # The speaker's own words, for extraction, open loops and peer
        # memory alike — none for a turn that touched their mailbox; see
        # stream_chat.
        own_text = with_added(_speaker_text(memory_text, user_message), inbox)
        # A host-surface turn (Session.from_host) is kept nowhere either: its
        # words may be a script's on the host's page, never the person's own.
        host_words = getattr(session, "from_host", False) is True
        speaker_text = "" if touched_mail or host_words else own_text
        # private_rows: see stream_chat — the scheduled passes copy it.
        # A replay (memory.replay) is kept nowhere — see stream_chat.
        kept = not touched_mail and not host_words and not chat_replay.is_replay(session)
        repeat_offer: dict[str, Any] | None = None
        mail_words = _mail_turn_words(session, own_text, touched_mail=touched_mail, host_words=host_words)
        with private_rows(touched_mail):
            if should_extract(speaker_text, session=session):
                schedule_extraction(
                    speaker_text, final_response, session_id=session.session_id
                )

            # Open loops — see stream_chat.
            from openexecutive.attunement.open_loops import schedule_open_loop_pass

            if kept:
                schedule_open_loop_pass(
                    speaker_text, final_response, person_id=person_id,
                    session_id=session.session_id, workspace_mode=workspace_mode,
                    principal_verified=is_principal_on_verified_surface(session),
                    private_to_person_id=private_owner(session),
                )
            from openexecutive.attunement.style import schedule_style_pass

            if kept:
                schedule_style_pass(person_id, session_id=session.session_id)

            # Always in the loop — see stream_chat.
            from openexecutive.memory.history_chat import schedule_chat_notes

            schedule_chat_notes(speaker_text or mail_words, session=session, person_id=person_id)

            # Conversation review — see stream_chat.
            from openexecutive.memory.chat_review import record_turn

            if not chat_replay.is_replay(session):
                record_turn(own_text, final_response, session=session,
                            person_id=person_id, calls=turn_calls, mail=touched_mail)

            # Repeated asks — see stream_chat.
            if kept and memory_text is None:
                from openexecutive.workflows.repeat_asks import note_ask

                repeat_offer = await asyncio.to_thread(
                    note_ask, own_text, session=session, person_id=person_id, calls=turn_calls,
                )

            # What it worked out — see stream_chat.
            from openexecutive.memory.worked_out import KEEP_TOOL, keep_turn_figures

            if own_text.strip() and any(name == KEEP_TOOL for name, _ in turn_calls):
                await asyncio.to_thread(keep_turn_figures, final_response, turn_calls, session=session,
                                        person_id=person_id, private=touched_mail)

        # Mirror the completed exchange into Honcho (see stream_chat for
        # rationale, and for why a turn private to the principal — or one
        # that touched the speaker's own mailbox — is not).
        if not _private_to_principal(session) and kept and own_text.strip():
            from openexecutive.memory.honcho_client import sync_turn as _honcho_sync
            _honcho_sync(
                speaker_text,
                final_response,
                person_id=person_id,
                session_id=session.session_id,
                co_present_person_ids=co_present_person_ids,
            )
            _sync_consulted_departments_to_honcho(
                consulted,
                speaker_text,
                final_response,
                person_id=person_id,
                session_id=session.session_id,
                co_present_person_ids=co_present_person_ids,
            )
        elif mail_words:
            with private_rows(True):
                _sync_mail_turn_words(mail_words, person_id=person_id, session_id=session.session_id)

        if repeat_offer is not None:
            yield {"type": "repeat_offer", "offer": repeat_offer, "session_id": session.session_id}

        # Reset audit ContextVars at normal completion. The abandoned-stream
        # case (SSE client drop mid-yield) doesn't reach here — accept that
        # narrow leak window; concurrent FastAPI requests run on their own
        # tasks and don't share the ContextVar value.
        clear_turn()

    async def _routing_prepass(
        self,
        system_blocks: list[dict[str, Any]],
        current_messages: list[dict[str, Any]],
        *,
        model: str,
        episodic_context: str,
        debug_collector: DebugCollector | None,
        consulted_out: list[str] | None,
        specialist_outputs_out: dict[str, str] | None,
        turn_id: str | None,
        conversation_context: str,
        specialists_consulted: list[str],
    ) -> AsyncIterator[str | dict[str, Any]]:
        """Decide and dispatch specialist consults before the main turn.

        One model call offering `consult_specialist` and nothing else. On
        the full surface that tool competes with the rest of the client
        tool surface and a smaller model does not reach for it; the
        measurements are recorded on ``Settings.routing_prepass_enabled``.

        Appends the decision and its results to ``current_messages`` as a
        well-formed tool_use/tool_result pair, so the main loop — which is
        otherwise untouched and still sees every tool — opens with the
        specialist findings already in hand. A pre-pass that picks nobody
        appends nothing, which is why an action turn ("schedule X") cannot
        regress: it simply falls through to the loop as before.
        """
        prepass_tools = [
            {
                **SPECIALIST_TOOLS[0],
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]
        try:
            decision = await get_provider(model).messages_create(
                model=model,
                max_tokens=1024,
                system=system_blocks,
                tools=prepass_tools,
                messages=current_messages,
            )
        except Exception:
            # Never fail the turn on the pre-pass: the main loop still offers
            # consult_specialist, so a failure here costs routing quality, not
            # the answer.
            logger.exception("routing pre-pass failed; continuing without it")
            return

        session_id = getattr(current_session.get(), "session_id", None)
        # The pre-pass is a real billed call against the full system prefix.
        # Without this row /audit/usage and the per-session CostSummary — both
        # of which aggregate cache_event only — would under-report every turn
        # by one call. iteration=0 marks it as pre-loop.
        _emit_cache_event(
            session_id=session_id,
            turn_id=turn_id or "",
            iteration=0,
            final_msg=decision,
            model=model,
            actor="routing_prepass",
        )

        run_tool_uses: list[dict[str, Any]] = []
        run_calls: list[dict[str, str]] = []
        reasoning_blocks: list[dict[str, Any]] = []
        consult_blocks = [
            b
            for b in decision.content
            if getattr(b, "type", None) == "tool_use"
            and b.name == "consult_specialist"
        ]
        # What the model asked for, counted before anything is dropped, so the
        # debug event can report the difference. Counting survivors plus
        # per-reason counters would make every future `continue` responsible
        # for keeping the arithmetic true.
        requested_count = len(consult_blocks)
        for block in decision.content:
            # Carry the OpenRouter reasoning array into the replayed assistant
            # turn. Every other block-by-block replay in this repo does; the
            # translator's contract is that a tool_use turn whose reasoning was
            # dropped can be 400'd by a thinking-enabled model, and OpenRouter
            # loses continuity without it.
            if (replay := reasoning_replay_block(block)) is not None:
                reasoning_blocks.append(replay)
                continue
            if getattr(block, "type", None) != "tool_use":
                continue
            if block.name != "consult_specialist":
                continue
            # A tool_use's `input` is always a dict: Anthropic validates it
            # against the schema, and translator._parse_tool_arguments coerces
            # a non-object payload to {} for every OpenAI-compatible provider.
            block_input = block.input
            # One rule, not three: a consult carrying no question is dropped,
            # whatever produced it. Dispatching one bills a specialist call that
            # can only answer generically and files a consult in the audit graph
            # for a turn that reached nobody — the looks-routed-but-isn't
            # failure this pre-pass exists to fix.
            #
            # Truncation is the usual source: the pre-pass runs on a 1024-token
            # budget, and a tool_use cut mid-arguments deserializes with `query`
            # missing or blank. But `stop_reason == "max_tokens"` is NOT
            # evidence that a given call was the casualty — a reasoning model
            # routinely finishes a complete consult and then spends the rest of
            # the budget thinking — and a blank query is worth dropping even
            # when nothing was truncated. Keying the drop on stop_reason plus
            # "is the last tool_use" got both directions wrong at once: it
            # discarded a complete final consult (making the pre-pass a no-op)
            # while still dispatching a blank one that happened to be emitted
            # first. The query is the thing actually being tested, so test it
            # alone. stop_reason stays in the log line, where it is diagnosis
            # rather than control flow; getattr because not every adapter sets
            # it.
            #
            # The main tool loop deliberately has no equivalent drop: there the
            # model has already been shown a tool surface, so a blank consult
            # can be answered with a tool_result it can re-ask from. The
            # pre-pass has no such recovery — it either dispatches or discards.
            if not str(block_input.get("query") or "").strip():
                # Name the block and the requested specialist: several blank
                # consults in one decision are otherwise indistinguishable in
                # the log, and this drop writes no audit row to correlate with.
                # The name is model-emitted, so it gets the same preview cap
                # the router applies before echoing one.
                logger.warning(
                    "routing pre-pass: dropping consult with no query "
                    "(id=%s specialist=%r stop_reason=%s)",
                    getattr(block, "id", None),
                    str(block_input.get("specialist", ""))[:NAME_PREVIEW_CHARS],
                    getattr(decision, "stop_reason", None),
                )
                continue
            raw_name = block_input.get("specialist", "")
            resolved = resolve_specialist_name(raw_name)
            if resolved is None:
                # Dropped rather than dispatched: route_to_specialist would
                # only hand back an error string, and replaying the tool_use
                # would oblige us to carry a useless tool_result into the
                # main turn.
                audit_name_resolution(
                    raw_name,
                    None,
                    source="routing_prepass",
                    session_id=session_id,
                    turn_id=turn_id,
                )
                continue
            if resolved != raw_name:
                audit_name_resolution(
                    raw_name,
                    resolved,
                    source="routing_prepass",
                    session_id=session_id,
                    turn_id=turn_id,
                )
            run_tool_uses.append(
                {"id": block.id, "name": block.name, "input": block_input}
            )
            run_calls.append(
                {
                    "specialist": resolved,
                    "query": str(block_input.get("query", "")),
                }
            )

        run_tool_uses, run_calls, _skipped_results, fanout_cap = (
            partition_specialist_fanout(
                run_tool_uses, run_calls, self._settings.max_parallel_specialists
            )
        )
        # Over-cap picks are dropped outright rather than carried as skip
        # results (hence discarding `_skipped_results`, which the loop uses to
        # give each over-cap tool_use a tool_result). The loop's skip message
        # exists so the model can re-ask; here the model has not yet seen a
        # tool surface to re-ask with, and the main turn can still consult
        # whoever was left out.
        #
        # Emitted before the empty check so "asked for specialists, reached
        # nobody" is still reported: it is the looks-routed-but-isn't case, and
        # the counts are what carry it. Note the Agent Activity panel renders
        # this event from `specialists` alone, so an all-dropped decision shows
        # there as an empty routing line — the detail lives in the event stream
        # and, for a name that failed to resolve, the routing_anomaly audit
        # rows. A consult dropped for a blank query writes no audit row: there
        # is no name anomaly to record, and the drop happens before the name is
        # read. That class is visible only here and in the log line above, so
        # `skipped_count` exceeding the number of routing_anomaly rows for the
        # turn is expected, not a gap.
        if debug_collector and requested_count:
            evt = debug_collector.emit("routing_decision", {
                "iteration": 0,
                "phase": "prepass",
                "requested_count": requested_count,
                "cap": fanout_cap,
                "dispatched_count": len(run_calls),
                "skipped_count": requested_count - len(run_calls),
                "specialists": [
                    {"specialist": c["specialist"], "query": c["query"]}
                    for c in run_calls
                ],
            })
            yield debug_collector.to_sse_dict(evt)

        if not run_calls:
            return

        yield self._THINKING

        # route_parallel appends specialist_start/specialist_done to the
        # collector but does not yield them; the caller drains them. The loop
        # does that from its own cursor — and takes that cursor AFTER the
        # pre-pass runs, so anything emitted here would never be yielded by
        # either. Without this drain the live Agent Activity panel shows
        # "Routing to CSO, CFO" and then nothing until synthesis, on what is
        # now the primary routing path. /debug/last-turn snapshots _events
        # wholesale afterwards, so a post-hoc check would not catch it.
        event_cursor = len(debug_collector._events) if debug_collector else 0
        spec_t0 = time.monotonic()
        specialist_results = await route_parallel(
            run_calls,
            episodic_context=episodic_context,
            session_id=session_id,
            debug_collector=debug_collector,
            conversation_context=conversation_context,
        )
        spec_ms = round((time.monotonic() - spec_t0) * 1000)
        if debug_collector:
            for evt in debug_collector._events[event_cursor:]:
                yield debug_collector.to_sse_dict(evt)

        specialists_consulted.extend(c["specialist"] for c in run_calls)
        if consulted_out is not None:
            consulted_out.extend(c["specialist"] for c in run_calls)
        if specialist_outputs_out is not None:
            for call, result in zip(run_calls, specialist_results, strict=True):
                specialist_outputs_out[call["specialist"]] = result

        _audit_specialist_consults(
            run_calls=run_calls,
            specialist_results=specialist_results,
            session_id=session_id,
            turn_id=turn_id,
            iteration=0,
            phase="prepass",
            duration_ms=spec_ms,
            conversation_context=conversation_context,
            system_blocks=system_blocks,
        )

        # Replay only the dispatched tool_uses, so every tool_use in the
        # assistant message has exactly the one tool_result the API requires.
        # Reasoning blocks lead, matching the loop's block-by-block replay.
        current_messages.append(
            {
                "role": "assistant",
                "content": [
                    *reasoning_blocks,
                    *(
                        {
                            "type": "tool_use",
                            "id": tu["id"],
                            "name": tu["name"],
                            "input": tu["input"],
                        }
                        for tu in run_tool_uses
                    ),
                ],
            }
        )
        current_messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": result,
                    }
                    for tu, result in zip(
                        run_tool_uses, specialist_results, strict=True
                    )
                ],
            }
        )

    async def _stream_agent_loop(
        self,
        system_blocks: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        model: str = "",
        *,
        # Defaults to None — "no web search this turn" — which every use site
        # below already guards for. Production callers always pass an explicit
        # selection from `select_web_search_tool`; leaving the default here
        # keeps the loop callable without one, which is how tests that exercise
        # unrelated parts of it (cache markers, tool-error isolation) drive it.
        search: WebSearchSelection | None = None,
        episodic_context: str = "",
        debug_collector: DebugCollector | None = None,
        consulted_out: list[str] | None = None,
        specialist_outputs_out: dict[str, str] | None = None,
        turn_id: str | None = None,
        conversation_context: str = "",
        turn_sources: TurnSources | None = None,
        workspace_mode: str | None = None,
        principal_role_tag: str | None = None,
        inbox: TurnInbox | None = None,
        seen_url_text: str = "",
        calls_out: list[tuple[str, str]] | None = None,
    ) -> AsyncIterator[str | dict[str, Any]]:
        """Tool-use loop that yields text deltas as they arrive.

        Yields _THINKING before each specialist call round so callers can send
        keepalive/progress events while the blocking specialist calls run.
        Also yields debug event dicts when a debug_collector is provided.

        ``turn_sources`` collects the documents and web pages the turn looked
        at, and which specialists failed or answered; its owner (the web chat
        route) sends it once the reply is over.

        ``workspace_mode`` is the turn's solo/team mode (the caller resolves it
        once, for the system blocks too); None resolves it from the current
        session. Solo withholds the team-only tools and refuses a call to one.

        ``principal_role_tag`` is the specialists' ``<principal_role>`` body
        for the turn, resolved by the caller with the mode ("" for none);
        None resolves it here from the current session — solo only.

        ``inbox`` holds messages the person sent while this turn runs (the
        web chat's POST /chat/add). Whatever has arrived is added after each
        round's tool results, and a ``message_added`` event names it.

        ``seen_url_text`` is the speaker's own words for the turn (the
        caller's ``memory_text``): the client ``web_fetch`` may open a URL
        in it, in a message the inbox adds, or in a result of this turn's
        ``web_search``, and nothing else. Never pass the prompt: callers
        pack attachment text and inbound mail into it, and a URL there must
        stay unfetchable.
        """
        if workspace_mode is None:
            workspace_mode = effective_workspace_mode(current_session.get())
        if principal_role_tag is None:
            principal_role_tag = (
                principal_role_context(effective_principal_role(current_session.get()))
                if workspace_mode == "solo"
                else ""
            )
        # An unattended run (the scheduler's proactive trigger) is also not
        # offered the principal-only tools; its tool list is a stable variant
        # of its own, like each mode's.
        unattended_withheld = (
            UNATTENDED_WITHHELD_TOOLS
            if bool(getattr(current_session.get(), "unattended", False))
            else frozenset()
        )
        # A turn private to the principal (mail from one of their contacts,
        # mail they forwarded) may reach the principal and nobody else: it is
        # not offered the tools that post to other people, publish where they
        # read, or start work outside the turn — again a stable list of its
        # own — and of the MCP tools, only Google Workspace reads and the
        # recipient-gated Gmail send (`PRIVATE_TURN_MCP_TOOLS`).
        private_turn = turn_is_private_to_principal()
        private_withheld = PRIVATE_TURN_WITHHELD_TOOLS if private_turn else frozenset()
        # The untrusted-content policy: a tool that changes the install for
        # every later turn (load_mcp_server) is offered only while the
        # principal is speaking on a verified, interactive surface — never on
        # an inbound email, a teammate's turn or Google Chat
        # (`content_trust.principal_only_withheld`).
        principal_withheld = principal_only_withheld(current_session.get())
        # Python jobs need the sandbox the API image installs: without it the
        # tool is not offered (fixed per process, so the prefix stays stable).
        sandbox_missing = frozenset() if python_job.available() else frozenset({python_job.TOOL_NAME})
        # A turn that may only read and advise (`read_only_turn`: a page in
        # another app, or a scheduled run a teammate queued, whose message the
        # runner sends): nothing that sends, posts, books, schedules, runs or
        # changes stored state runs without the owner's card
        # (`READ_ONLY_WITHHELD_TOOLS`, `levels.READ_ONLY`). Another AI app's
        # turn is the signed-in person's own: Approvals decides it like the
        # web chat.
        mcp_client = read_only_turn(current_session.get())
        mcp_client_withheld = READ_ONLY_WITHHELD_TOOLS if mcp_client else frozenset()
        # A shared web chat someone other than its owner has written into
        # since the owner last did (`Session.others_spoke_last`): nothing
        # that sends to people or changes contacts (`SHARED_CHAT_WITHHELD_TOOLS`,
        # and the gateway's mail writes through `shared_chat_withholds`).
        others_spoke = (
            getattr(current_session.get(), "shared_chat", False) is True
            and getattr(current_session.get(), "others_spoke_last", False) is True
        )
        shared_withheld = SHARED_CHAT_WITHHELD_TOOLS if others_spoke else frozenset()
        # Someone on no People entry (an unmatched Google Chat sender, a web
        # sign-in on nobody's entry) has no level of their own: everything
        # they ask to send or change waits on the owner's card at every level.
        unrostered = levels.unrostered_speaker(current_session.get())
        # A replay of a graded turn (memory.replay): nothing that runs a
        # script or a job, or changes the install; its acting tools are
        # faked below, after the gates.
        replaying = chat_replay.is_replay(current_session.get())
        replay_withheld = chat_replay.WITHHELD_TOOLS if replaying else frozenset()
        # The gates' tools are offered at every level: the one check
        # (orchestrator.levels) decides at dispatch whether a call they
        # notice runs (Hands-off) or waits on an Approve card.
        not_offered = sandbox_missing | replay_withheld
        withheld_tools = tools_withheld_in_mode(workspace_mode) | not_offered
        # Act as me: ghostwrite_email joins the toolkit only on a turn
        # pin_turn_delegation offered it to. Its own registry, never
        # _ALL_SKILL_TOOLS, and a per-turn handler map built from it — so on
        # any other turn a call to it is an unknown tool, not a refusal to argue
        # with.
        pinned_delegation = turn_delegation(current_session.get())
        # Nor on a replay (memory.replay): Act as me's tools save drafts,
        # reminders and files for real, and its reads mark a chat mail-private.
        ghostwrite_offered = (
            not mcp_client and not replaying
            and pinned_delegation is not None and pinned_delegation.offered
        )
        delegation_tools = DELEGATION_TOOLS if ghostwrite_offered else []
        # The URLs web_fetch may open this turn; search results join through
        # the client search handler, or as the server search's blocks arrive.
        seen_urls = SeenUrls()
        seen_urls.add_from_text(seen_url_text)
        # The client search and fetch handlers join the canonical dispatch map
        # rather than sitting in a dict of their own: built once per turn (see
        # client_search_handlers on why the budget must not be rebuilt per
        # iteration), and folded in HERE so they inherit every later
        # transformation below — in particular take_the_lead.gated_handlers,
        # which a separate dict would silently bypass.
        search_handlers = client_search_handlers(search, seen_urls)
        turn_handlers = (
            {**_ALL_SKILL_HANDLERS, **DELEGATION_TOOL_HANDLERS, **search_handlers}
            if ghostwrite_offered
            else {**_ALL_SKILL_HANDLERS, **search_handlers}
        )
        # Always in the loop: recall_history joins the same way, only on a
        # turn whose speaker may read their own notes here (history_tools).
        history_offered = (
            not mcp_client and not replaying and recall_person(current_session.get()) is not None
        )
        if history_offered:
            delegation_tools = [*delegation_tools, *HISTORY_TOOLS]
            turn_handlers = {**turn_handlers, **HISTORY_TOOL_HANDLERS}
        # Earlier chats: search_past_chats joins the same way, on a turn whose
        # speaker is verified in a chat nobody else reads (chat_search_tools).
        if not mcp_client and not replaying and search_person(current_session.get()) is not None:
            delegation_tools = [*delegation_tools, *CHAT_SEARCH_TOOLS]
            turn_handlers = {**turn_handlers, **CHAT_SEARCH_TOOL_HANDLERS}
        # The inbox's work turn on a reply card (delegation.card_work):
        # update_reply_card hands back what it did, with Act as me's tools.
        if ghostwrite_offered and reply_card_tools.offered(current_session.get()):
            delegation_tools = [*delegation_tools, *reply_card_tools.REPLY_CARD_TOOLS]
            turn_handlers = {**turn_handlers, **reply_card_tools.REPLY_CARD_TOOL_HANDLERS}
        # Tools an installed extension adds (orchestrator/extensions.py): only
        # on a turn someone is in, never an unattended run or a turn private
        # to the principal, and only where each tool says it is offered.
        if not unattended_withheld and not private_turn and not mcp_client and not replaying:
            extra_tools = extensions.offered_tools(current_session.get())
            if extra_tools:
                delegation_tools = [*delegation_tools, *(t.definition for t in extra_tools)]
                turn_handlers = {**turn_handlers, **{t.name: t.handler for t in extra_tools}}
        # Take the lead as the Executive: an unattended run's acting tools,
        # its own and the MCP ones, go through the gate (take_the_lead).
        leading = bool(unattended_withheld) and take_the_lead.executive_on()
        if leading:
            turn_handlers = take_the_lead.gated_handlers(turn_handlers, source="scheduled")
        # A step someone asked for on a turn that isn't the owner's own ask
        # (a teammate, a contact's mail) also meets the owner's Waits rules,
        # then In training, after Approvals.
        asked_checks = not leading and not replaying and take_the_lead.asked_checks_apply(current_session.get())
        if asked_checks:
            turn_handlers = take_the_lead.asked_handlers(turn_handlers)
        # A team member's turn runs like the owner's: Approvals decides each
        # step (orchestrator.levels), and a held one is their own card.
        # A replay (memory.replay): outermost, so a tool that would send,
        # post, book, schedule, run or change stored state is recorded on the
        # session and answered "done" before any gate or handler sees it;
        # reads still go through every gate above.
        if replaying:
            turn_handlers = chat_replay.faked_handlers(turn_handlers, current_session.get())
        # Something new came in on a channel (mail, Slack, Telegram, …): with
        # Take the lead on, the reflection looks at it soon, batched.
        origin = str(getattr(current_session.get(), "origin_channel", "") or "")
        if origin and not unattended_withheld and not replaying:
            take_the_lead.wake(take_the_lead.arrival_reason(origin))
        current_messages = list(messages)
        if self._settings.enable_caching:
            system_blocks, current_messages = _apply_history_cache_marker(
                system_blocks, current_messages
            )
        turn_marked = (
            _mark_turn_message(current_messages) if self._settings.enable_caching else None
        )
        # Tool calls this turn's scripts have made, in all and per own tool
        # (settings.chat_script_max_calls, step_script.CHAT_OWN_TOOL_CAPS).
        script_counts: dict[str, int] = {}
        # The fan-out hint (step_script.FANOUT_HINT) goes out once a turn.
        fanout_hinted = False
        # Shallow copy — the caller owns every dict up to this index.
        caller_message_count = len(current_messages)
        # The end of the text this turn has streamed so far, across rounds:
        # a round's text after an earlier round's (the model's words before a
        # tool call, then its answer) starts a new paragraph, so the two
        # never run together ("...stock.Yes. ITEM-1...").
        streamed_tail = ""
        # Each round's streamed text (its separator first) and whether it went
        # on only to look something up, since the reply was last written
        # again: the working notes among them leave the reply at the end
        # (_without_notes).
        rounds: list[tuple[str, bool]] = []
        # Web searches this turn has run: one budget for the whole turn
        # (web_search_tool.turn_search_tool); spent_at is the count when the
        # turn was told it had spent it.
        searches_used = 0
        spent_at: int | None = None
        # Rounds a long web search paused (pause_turn), carried on.
        pauses = 0
        # Set by a round cut off mid-text: the next round's text carries it
        # on, so that round alone gets no paragraph break before its text.
        run_on = False
        specialists_consulted: list[str] = []
        synthesis_roster_size = 0
        # What this turn's tool results have put in the prompt so far: every
        # one is re-sent on each later call.
        reading_budget = TurnReadingBudget(reading_limits_for(model))

        # Every tool call this turn with the start of its result, for the
        # end-of-turn check that the reply claims nothing it didn't do
        # (claim_check); checked once a turn.
        # Every tool the turn ran with its result: for the claim check, and
        # (``calls_out``) for the conversation review's record of the turn.
        turn_calls: list[tuple[str, str]] = calls_out if calls_out is not None else []
        # What the after-reply check reads (claim_check): each call with what
        # it was asked to do, since a result's first line rarely says it all.
        claim_calls: list[tuple[str, str]] = []
        # Reply text of cut rounds the next round carries on: the claim
        # check reads it with the round that finishes the reply.
        cut_text = ""
        # Cut rounds in a row that wrote nothing (MAX_EMPTY_CUT_ROUNDS).
        empty_cuts = 0
        claim_checked = False
        # Set once the person asks to finish the work themselves: the next
        # round writes up what the turn has, and runs no tool.
        handing_over = False
        # Set once the same call came back the same REPEAT_LIMIT times: the next
        # round writes up what the turn has, and runs no tool, as on a hand-over.
        repeating = False
        # What a write-up that wrote nothing says: why it was asked for.
        write_up_empty = REPEATING_EMPTY_REPLY
        # What the turn last showed the person (show_progress), so each call
        # adds to it, and the reads it made, counted once each for the card.
        shown_progress: turn_progress.Progress | None = None
        turn_reads = turn_progress.Reads()

        stream_model = model or self._settings.default_model
        if self._settings.routing_prepass_enabled:
            async for item in self._routing_prepass(
                system_blocks,
                current_messages,
                model=stream_model,
                episodic_context=episodic_context,
                debug_collector=debug_collector,
                consulted_out=consulted_out,
                specialist_outputs_out=specialist_outputs_out,
                turn_id=turn_id,
                conversation_context=conversation_context,
                specialists_consulted=specialists_consulted,
            ):
                yield item

        for iteration in itertools.count(1):
            logger.info(
                "iter %d", iteration,
                extra={"iter_marker": True},
            )
            full_text = ""
            # The separator streamed before this round's text (_round_separator).
            round_sep = 0
            round_sep_text = ""
            carries_on, run_on = run_on, False
            tool_uses: list[dict[str, Any]] = []
            response_content: list[dict[str, Any]] = []
            # The content check's notice on each result this round, for every
            # part of it (orchestrator.reading).
            read_notices: dict[str, str] = {}

            # If specialists were already consulted, this is the synthesis pass.
            # Emit synthesis_start before text chunks begin streaming.
            #
            # Re-emitted whenever the consulted roster GROWS, rather than once.
            # This event could always fire on several iterations of one turn
            # (consult at iteration 1, use a tool at iteration 2, and it fired
            # at the top of both 2 and 3) — the pre-pass only makes iteration 1
            # eligible too. Firing once would be the regression: the panel
            # would report the pre-pass's roster and never mention a specialist
            # the loop went on to consult.
            # Names are de-duplicated because the same specialist can be
            # consulted by both the pre-pass and the loop; the count is a
            # roster for the UI ("Synthesizing N specialist responses"), not a
            # call tally.
            unique_consulted = list(dict.fromkeys(specialists_consulted))
            if debug_collector and len(unique_consulted) > synthesis_roster_size:
                evt = debug_collector.emit("synthesis_start", {
                    "specialist_count": len(unique_consulted),
                    "specialists_consulted": unique_consulted,
                })
                synthesis_roster_size = len(unique_consulted)
                yield debug_collector.to_sse_dict(evt)

            # Client-side tools are sorted by name for cache stability; the
            # last one carries the cache_control marker. Anthropic server-side
            # tools (e.g. web_search) are appended after — they use a `type`
            # field instead of input_schema and cannot accept cache_control.
            # The SearXNG search tool and web_fetch (always the client tool,
            # web_search_tool explains why) are ordinary client tools and join
            # the sorted list; only the server search is appended (below).
            extra_client_tools = (
                [
                    t for t in (
                        search.tool if search.kind == "client" else None,
                        search.fetch_tool,
                    )
                    if t is not None
                ]
                if search is not None
                else []
            )
            # Solo withholds the team-only tools before the sort, so each mode
            # has its own stable, sorted tool prefix (as do an unattended run
            # and a turn private to the principal).
            # The less common tools are offered through open_tools/use_tool
            # rather than one by one (tool_groups), so the list is short and
            # stays the same whichever groups the turn opens. The fork's
            # client-side SearxNG `web_search` goes in here with the rest:
            # it is not in tool_groups.DEFERRED, so split() puts it straight
            # into direct_tools, exactly where it sat before.
            direct_tools, deferred_tools = tool_groups.split(
                t for t in filter_tools_for_workspace_mode(
                    [
                        *SPECIALIST_TOOLS, *_offered_skill_tools(), *self._mcp_tools,
                        *self._script_tools, *delegation_tools, *extra_client_tools,
                        turn_progress.SHOW_PROGRESS_TOOL,
                    ],
                    workspace_mode,
                )
                if t["name"] not in not_offered
            )
            client_tools = sorted(direct_tools, key=lambda t: t["name"])
            booking_offered = any(
                isinstance(t, dict) and t.get("name") == "create_calendar_event"
                for t in [*direct_tools, *deferred_tools]
            )
            tools_with_cache: list[dict[str, Any]] = [
                *client_tools[:-1],
                {**client_tools[-1], "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            ]
            # Server-side search only where the turn's selection says the
            # provider runs it (select_web_search_tool), with upstream's one
            # budget for the whole turn (turn_search_tool); the client-side
            # SearXNG tool is already in the sorted client list above.
            web_search_tool = (
                web_searches.turn_search_tool()
                if search is not None and search.kind == "server"
                else None
            )
            if web_search_tool is not None:
                tools_with_cache.append(web_search_tool)
            async with get_provider(stream_model).messages_stream(
                model=stream_model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=system_blocks,  # type: ignore[arg-type]
                tools=tools_with_cache,  # type: ignore[arg-type,list-item]
                messages=current_messages,  # type: ignore[arg-type]
            ) as stream:
                async for event in stream:
                    if (
                        hasattr(event, "type")
                        and event.type == "content_block_delta"
                        and hasattr(event, "delta")
                        and hasattr(event.delta, "type")
                        and event.delta.type == "text_delta"
                    ):
                        if not full_text and streamed_tail and event.delta.text and not carries_on:
                            separator = _round_separator(streamed_tail, event.delta.text)
                            if separator:
                                yield separator
                                streamed_tail = separator
                                round_sep = len(separator)
                                round_sep_text = separator
                        full_text += event.delta.text
                        streamed_tail = (streamed_tail + event.delta.text)[-2:]
                        yield event.delta.text

                final_msg = await stream.get_final_message()

            # cache_event: capture per-iteration token + cache stats so the
            # flow chart can show "this turn used N cache hits at iter K".
            # Always after the API call, never in the request path — does
            # not touch system_blocks / messages, so caching is unaffected.
            _emit_cache_event(
                session_id=getattr(current_session.get(), "session_id", None),
                turn_id=turn_id or "",
                iteration=iteration,
                final_msg=final_msg,
                model=stream_model,
            )

            web_search_queries: list[str] = []
            for block in final_msg.content:
                if block.type == "text":
                    response_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    tool_uses.append({"id": block.id, "name": block.name, "input": block.input})
                    response_content.append(
                        {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                    )
                elif block.type == "server_tool_use":
                    # Anthropic resolves server tools (web_search) within the
                    # same generation; we just echo the block back on the next
                    # iteration so the model retains continuity. No handler runs.
                    response_content.append(block.model_dump(exclude_none=True))
                    if block.name == WEB_SEARCH_TOOL_NAME:
                        query = (block.input or {}).get("query", "")
                        if isinstance(query, str) and query:
                            web_search_queries.append(query)
                elif block.type == "web_search_tool_result":
                    response_content.append(block.model_dump(exclude_none=True))
                    # Results the model read before writing this round's
                    # tool calls (outside_content).
                    outside_content.mark(outside_content.WEB)
                    # Its result URLs may be opened, as the client search's are
                    # (searxng_search's on_result_url).
                    for url in server_search_result_urls(block):
                        seen_urls.add(url)
                elif block.type == OPENROUTER_CITATIONS_BLOCK:
                    # OpenRouter's search returns its results as citations,
                    # which may be opened likewise. Not added to
                    # response_content: it holds nothing the model needs on
                    # the next round, and the translator would drop it from
                    # that request anyway. The model read them, so the turn
                    # has read the web, as for web_search_tool_result above.
                    outside_content.mark(outside_content.WEB)
                    for url in server_search_result_urls(block):
                        seen_urls.add(url)
                elif (replay := reasoning_replay_block(block)) is not None:
                    # OpenRouter reasoning continuity across tool iterations
                    # (replayed as ``reasoning_details`` by the translator).
                    response_content.append(replay)
            if turn_sources is not None:
                record_web_sources(turn_sources, final_msg.content)
            rounds.extend(_round_parts(
                round_sep_text,
                full_text,
                response_content,
                final_msg.stop_reason == "tool_use" and not carries_on and _only_looks(tool_uses),
                carries_on,
            ))
            # A web search is a look, like any read: the after-reply check
            # sees it with the turn's other calls.
            claim_calls.extend(web_searches.searched(response_content))
            searches_used += web_searches.searches_run(response_content)

            if web_search_queries:
                session_id = getattr(current_session.get(), "session_id", None)
                if debug_collector:
                    evt = debug_collector.emit("web_search_invocation", {
                        "iteration": iteration,
                        "queries": web_search_queries,
                    })
                    yield debug_collector.to_sse_dict(evt)
                for q in web_search_queries:
                    audit_log(
                        "tool_invocation",
                        f"web_search: {q[:200]}",
                        session_id=session_id,
                        turn_id=turn_id,
                        actor="executive",
                        details={
                            "tool": WEB_SEARCH_TOOL_NAME,
                            "kind": "server_tool",
                            "iteration": iteration,
                            "query": q[:500],
                        },
                        full={
                            "query": q,
                            "active_prompt_blocks": _system_block_names(system_blocks),
                        },
                    )

            # Cut off at MAX_OUTPUT_TOKENS mid-answer: carry on rather than
            # end the turn on half a step. A cut tool call is answered with
            # CUT_OFF_RESULT below and never runs; cut text gets CUT_OFF_NOTE.
            reading_budget.saw(getattr(final_msg, "usage", None))
            cut_off = final_msg.stop_reason == "max_tokens"
            # The call the round was still writing when it was cut off is the
            # last one; its input is unfinished, so nothing reads or runs it.
            # Only text and calls count as written: a reasoning block that a
            # translated provider (OpenRouter, OpenAI) puts last is skipped.
            # A call followed by text or by a call the provider dropped closed
            # before the cut and runs. A translated provider lists text first,
            # so its last call always counts as cut and is asked for again.
            last_written = last_written_block(response_content)
            # A cut round got somewhere when its cut fell in text (shown, and
            # carried on) or a call closed before the cut (it runs). A round
            # cut while writing its only call got nowhere, however much it
            # wrote before it: that call never runs.
            cut_in_call = last_written is not None and last_written.get("type") == "tool_use"
            closed_calls = sum(b.get("type") == "tool_use" for b in response_content) - int(cut_in_call)
            got_somewhere = closed_calls > 0 or (
                last_written is not None and not cut_in_call and says_something(str(last_written.get("text", "")))
            )
            empty_cuts = empty_cuts + 1 if cut_off and not got_somewhere else 0
            if empty_cuts > MAX_EMPTY_CUT_ROUNDS:
                logger.warning("round %d: %d cut rounds in a row wrote nothing; ending the turn", iteration, empty_cuts)
                yield ("\n\n" if streamed_tail else "") + EMPTY_CUTS_REPLY
                return
            if cut_off:
                logger.warning(
                    "round %d cut off at max_tokens=%d; carrying on", iteration, MAX_OUTPUT_TOKENS,
                )
                if not tool_uses:
                    if not any(b.get("type") == "text" and str(b.get("text", "")).strip() for b in response_content):
                        # Nothing was said: the step was writing something
                        # long (a call the provider dropped whole).
                        yield {"type": "activity", "label": CUT_OFF_LABEL, "tool": "", "iteration": iteration}
                    if response_content:
                        current_messages.append({"role": "assistant", "content": response_content})
                    cut_text += "".join(
                        str(b.get("text", "")) for b in response_content if b.get("type") == "text"
                    )
                    if reading_budget.window_full:
                        # Another cut round would not fit the window: what
                        # it wrote stays shown and the turn ends plainly.
                        logger.warning("round %d cut off with the window full; ending the turn", iteration)
                        yield ("\n\n" if streamed_tail else "") + FULL_EMPTY_REPLY
                        return
                    note: list[dict[str, Any]] = [{"type": "text", "text": CUT_OFF_NOTE}]
                    current_messages.append({"role": "user", "content": note})
                    # The next round carries on the cut text where it
                    # stopped: no paragraph break between them
                    # (_round_separator). A cut that streamed nothing keeps
                    # the break after earlier rounds' text, and any round
                    # after the next gets the break as usual.
                    run_on = bool(full_text)
                    continue
            if (
                final_msg.stop_reason == "pause_turn" and response_content and pauses < MAX_PAUSES
                and not repeating and not handing_over
            ):
                # A long web search paused the round (the API's pause_turn):
                # sent back as it is, it carries on where it stopped. It
                # counts as a step; at most MAX_PAUSES in a turn. Past the
                # turn's searches, it is told to write up instead.
                pauses += 1
                logger.info("round %d paused by a long web search; carrying on (%d)", iteration, pauses)
                current_messages.append({"role": "assistant", "content": response_content})
                if web_search_tool is not None and web_searches.spent(searches_used):
                    current_messages.append(
                        {"role": "user", "content": [{"type": "text", "text": web_searches.SEARCHES_OVER_NOTE}]},
                    )
                    repeating = True
                    write_up_empty = web_searches.SEARCHES_EMPTY_REPLY
                run_on = bool(full_text)
                continue
            if final_msg.stop_reason != "tool_use" and not cut_off:
                # A reply with no tool call still gets the queued files and
                # synced excerpts checked, so what they tried is flagged to
                # the person (outside_content.settle).
                await outside_content.settle()
                if final_msg.stop_reason == "refusal":
                    # The model refused: no claim check or retry round, and
                    # a caller collecting the reply learns of it (_REFUSED).
                    logger.warning("round %d refused by the model", iteration)
                    refused = _REFUSED.get()
                    if refused is not None:
                        # How much of the collected text the refused round
                        # wrote, its separator included.
                        refused.append(round_sep + len(full_text))
                    else:
                        # A streamed reply (the web chat) ends on a plain
                        # refusal, not on what the refused round began.
                        yield ("\n\n" if streamed_tail else "") + REFUSED_REPLY
                    return
                # Do what it says: a reply that claims a change no call this
                # turn made gets one more round to make it or say it can't.
                claim: str | None = None
                if not (claim_checked or handing_over or repeating):
                    claim_checked = True
                    reply_text = cut_text + "\n".join(
                        str(b.get("text", "")) for b in response_content if b.get("type") == "text"
                    )
                    claim = await claim_check.unbacked_claim(
                        claim_check.last_person_message(messages), reply_text, claim_calls,
                    )
                if claim is None:
                    # The reply is the answer alone: a working note written
                    # before a lookup ("Now the live data.") leaves it, on
                    # every page and channel showing it (``rewrite``).
                    clean = _without_notes(rounds)
                    if clean is not None:
                        yield {"type": "rewrite"}
                        yield clean
                    return
                audit_log(
                    "claim_check",
                    "Reply claimed a change no tool made; one more round",
                    session_id=getattr(current_session.get(), "session_id", None),
                    turn_id=turn_id,
                    actor="executive",
                    details={"iteration": iteration, "calls": len(turn_calls)},
                    full={"claim": claim},
                    private=private_turn,
                )
                current_messages.append({"role": "assistant", "content": response_content})
                # The retry writes the whole reply again and it replaces the
                # first, never trailing it as a correction: a collected reply
                # (an email, a chat app) drops it before it is sent, and the
                # web chat swaps it out on ``rewrite``.
                yield {"type": "rewrite"}
                streamed_tail, rounds = "", []
                nudge: list[dict[str, Any]] = [
                    {"type": "text", "text": claim_check.nudge(claim)}
                ]
                current_messages.append({"role": "user", "content": nudge})
                # The retry's text gets its paragraph break from
                # _round_separator, like any other round's.
                continue

            # Handed over (or repeating itself): the write-up asked for a tool
            # anyway. Run nothing; what it wrote (if anything) is the reply.
            if handing_over or repeating:
                if not any(
                    b.get("type") == "text" and str(b.get("text", "")).strip()
                    for b in response_content
                ):
                    yield HAND_OVER_EMPTY_REPLY if handing_over else write_up_empty
                return

            cut_text = ""
            # open_tools is answered here; a use_tool call becomes the call it
            # stands for (same id), so every guard, handler, chip and audit
            # row below sees the tool's own name, as for a direct call.
            # show_progress is answered here too: nothing runs, and the chat
            # gets it before this round's activity label.
            group_results: dict[str, str] = {}
            cut_id = (
                tool_uses[-1]["id"]
                if cut_off and tool_uses and last_written is not None
                and last_written.get("type") == "tool_use"
                and last_written.get("id") == tool_uses[-1]["id"]
                else None
            )
            # The cut call's half-written input is never read: drop it from
            # the history, which every later round sends again.
            if cut_id is not None and last_written is not None:
                last_written["input"] = {}
            # Nothing below sees the cut call (no hand-over ask, no count of
            # calls made): it only gets CUT_OFF_RESULT, after the others.
            cut_results: list[dict[str, Any]] = []
            if cut_id is not None:
                tool_uses = [tu for tu in tool_uses if tu["id"] != cut_id]
                cut_results.append({"type": "tool_result", "tool_use_id": cut_id, "content": CUT_OFF_RESULT})
            for i, tu in enumerate(tool_uses):
                if tu["name"] == turn_progress.SHOW_PROGRESS:
                    result, progress_event, shown_progress = turn_progress.apply(
                        tu["input"],
                        shown_progress,
                        alone=all(
                            other["name"] == turn_progress.SHOW_PROGRESS for other in tool_uses
                        ),
                    )
                    group_results[tu["id"]] = result
                    if progress_event is not None:
                        yield progress_event
                elif tu["name"] == tool_groups.OPEN_TOOLS:
                    group_results[tu["id"]] = tool_groups.open_result(tu["input"], deferred_tools)
                elif tu["name"] == tool_groups.USE_TOOL:
                    call, error = tool_groups.unwrap(tu)
                    if call is None:
                        group_results[tu["id"]] = error or ""
                    else:
                        tool_uses[i] = call
                else:
                    # A built-in tool asked for through the gateway
                    # (call_tool name="create_alert") is that tool's own call,
                    # checked by its own name: the gateway doesn't have it.
                    own = _own_tool_call(tu, turn_handlers)
                    if own is not None:
                        tool_uses[i] = own
            # A repeat of a call whose result came back in parts is answered
            # with its next part, never by running the tool again (it may
            # have been a write).
            for tu in tool_uses:
                if tu["id"] not in group_results:
                    replayed = reading_budget.replay(tu)
                    if replayed is not None:
                        group_results[tu["id"]] = replayed
            # A read of the company's own Drive item is not outside content:
            # look those up first, so source_of (and defers_read) pass them.
            await outside_content.vet_reads(
                [tu for tu in tool_uses if tu["id"] not in group_results], self._mcp_gateway
            )
            # The person just said go to what the last turn held: the first
            # round that reads runs its other calls without those reads, so a
            # re-read doesn't hold the approved steps again
            # (orchestrator.outside_content.defers_read).
            deferred = [
                tu for tu in tool_uses
                if tu["id"] not in group_results and outside_content.defers_read(tu["name"], tu["input"])
            ]
            if deferred:
                for tu in deferred:
                    named = tu["input"].get("name") if isinstance(tu["input"], dict) else None
                    label = named[:200] if tu["name"] == "call_tool" and isinstance(named, str) and named else tu["name"]
                    group_results[tu["id"]] = outside_content.go_first_error(label)
                outside_content.go_first_round_done()
                logger.info("go_first: deferred %d read(s) until the held steps ran", len(deferred))
            dispatch_uses = [tu for tu in tool_uses if tu["id"] not in group_results]

            specialist_tool_uses = [tu for tu in dispatch_uses if tu["name"] == "consult_specialist"]
            skill_tool_uses = [tu for tu in dispatch_uses if tu["name"] in turn_handlers]
            # Dispatch guard: a tool this mode does not offer never runs, even
            # if the model emits it anyway — it gets an error tool_result.
            withheld_uses = [tu for tu in skill_tool_uses if tu["name"] in withheld_tools]
            if withheld_uses:
                skill_tool_uses = [
                    tu for tu in skill_tool_uses if tu["name"] not in withheld_tools
                ]
            mcp_tool_uses = [tu for tu in dispatch_uses if tu["name"] in MCP_TOOL_NAMES]
            script_tool_uses = [
                tu for tu in dispatch_uses
                if self._script_tools
                and tu["name"] in (step_script.RUN_SCRIPT_TOOL, step_script.LIST_SAVED_TOOLS_TOOL)
            ]
            # What the gates notice (they decide nothing): a step the
            # owner's level then runs or puts on an Approve card
            # (orchestrator.levels.decide, below).
            script_ids = {tu["id"] for tu in script_tool_uses}

            def _noticed(tu: dict[str, Any], script_ids: set[str] = script_ids) -> str | None:
                name, given = tu["name"], tu["input"]
                is_mcp = name in MCP_TOOL_NAMES
                is_script = tu["id"] in script_ids
                if name in unattended_withheld:
                    return levels.UNATTENDED
                if unrostered and (is_script or levels.acts(name, given)):
                    return levels.STRANGER
                if name in principal_withheld:
                    return levels.NOT_OWNER
                if private_turn and (
                    name in private_withheld or is_script or (is_mcp and private_turn_withholds(name, given))
                ):
                    return levels.PRIVATE
                if mcp_client and (
                    name in mcp_client_withheld or is_script
                    or (is_mcp and read_only_withholds(name, given))
                ):
                    return levels.READ_ONLY
                if others_spoke and (name in shared_withheld or (is_mcp and shared_chat_withholds(name, given))):
                    return levels.SHARED
                return None

            gate_reasons: dict[str, str] = {
                tu["id"]: reason
                for tu in [*skill_tool_uses, *mcp_tool_uses, *script_tool_uses]
                if (reason := _noticed(tu)) is not None
            }
            # Act as me: once the turn has read the principal's own mail (in
            # an earlier round, or with a ghostwrite_email, a mailbox read or
            # recall_history in this one — a round's tools run together),
            # nothing that opens a link, runs a script or workflow, or posts
            # to everyone runs for the rest of the turn (delegation.lockdown).
            # A later turn of that conversation stays private (touched_mail)
            # but is not locked: it is the person asking again. The offered
            # list stays as it is, so the cached prefix never changes
            # mid-turn.
            # How much of it holds is the safety level's, like any read: every
            # mail read is checked once its round is done (check_mail_read
            # below), and Balanced then holds it all only if one was flagged
            # or too big to check; a round that reads mail holds it all for
            # that round, since its reads aren't checked yet; Hands-off holds
            # none of it (delegation.lockdown.mail_turn_withholds).
            mail_touched_uses: list[dict[str, Any]] = []
            reads_mail_now = any(
                tu["name"] in MAILBOX_TOOL_NAMES or tu["name"] in _OWN_RECALL_TOOL_NAMES for tu in dispatch_uses
            )
            if pinned_delegation is not None and (pinned_delegation.read_mail or reads_mail_now):
                mail_touched_uses = [
                    tu for tu in [*skill_tool_uses, *mcp_tool_uses, *script_tool_uses]
                    if mail_turn_withholds(
                        tu["name"], tu["input"], reading_now=reads_mail_now, pinned=pinned_delegation
                    )
                ]
            # Outside content: once the turn has read anything from outside
            # the team (an earlier round, or a web search in this response),
            # what the Act as me lockdown withholds, contact writes and every
            # gateway write stay shut for the rest of the turn
            # (orchestrator.outside_content). A read in this round shuts the
            # round's other such calls too, since a round's tools run
            # together; the reads themselves run, and mark the turn once the
            # round is done. The offered list stays as it is, as above.
            # Reads queued for their check (an attachment, a synced excerpt)
            # are checked first, so the owner's safety level can decide
            # whether they hold this round (outside_content.settle). Under
            # Hands-off a read in this round holds nothing either.
            await outside_content.settle()
            read_before_round = outside_content.read_outside()
            round_sources = [] if outside_content.trusted_turn() else [
                (tu, source) for tu in dispatch_uses
                if (source := outside_content.source_of(tu["name"], tu["input"])) is not None
            ]
            round_source_ids = {tu["id"] for tu, _ in round_sources}
            outside_uses: list[dict[str, Any]] = []
            if read_before_round or round_sources:
                outside_uses = [
                    tu for tu in [*skill_tool_uses, *mcp_tool_uses, *script_tool_uses]
                    if (read_before_round or tu["id"] not in round_source_ids)
                    and outside_content.outside_withholds(
                        tu["name"], tu["input"], pending=[source for _, source in round_sources]
                    )
                ]
            # The one check: each step a gate noticed runs or waits on an
            # Approve card, by the owner's level (orchestrator.levels).
            stranger = outside_content.answers_off_team_mail()
            noticed: dict[str, tuple[dict[str, Any], str]] = {
                tu["id"]: (tu, gate_reasons[tu["id"]])
                for tu in [*skill_tool_uses, *mcp_tool_uses, *script_tool_uses]
                if tu["id"] in gate_reasons
            }
            mail_why = levels.mail_reason() if mail_touched_uses else levels.OWN_MAIL
            for tu in mail_touched_uses:
                noticed.setdefault(tu["id"], (tu, mail_why))
            for tu in outside_uses:
                noticed.setdefault(tu["id"], (tu, levels.STRANGER if stranger else levels.outside_reason()))
            held_uses = [
                (tu, reason) for tu, reason in noticed.values()
                # The level now, not the turn's first: a flagged read drops
                # Hands-off to Balanced for the rest of the turn
                # (outside_content._flag).
                if levels.decide(reason) == levels.CARD
                and not levels.only_to_owner(tu["name"], tu["input"])
            ]
            held_ids = {tu["id"] for tu, _ in held_uses}
            if held_ids:
                skill_tool_uses = [tu for tu in skill_tool_uses if tu["id"] not in held_ids]
                mcp_tool_uses = [tu for tu in mcp_tool_uses if tu["id"] not in held_ids]
                script_tool_uses = [tu for tu in script_tool_uses if tu["id"] not in held_ids]

            # No per-call "context": the tool no longer advertises one, and the
            # orchestrator forwards a rendered conversation tail to every
            # specialist via route_parallel(conversation_context=...). A model
            # that still emits the field is ignored rather than paid for.
            # Normalise the name HERE as well as inside route_parallel. The
            # router rebinds its own local `calls`, so without this the loop
            # would dispatch to the canonical specialist while recording the
            # raw token in specialists_consulted / consulted_out and as the
            # audit actor — committee reviewer selection and the department
            # sync both drop an unrecognised key, so a truncated `cs` would
            # consult the CSO and then be invisible to everything downstream.
            # An unresolvable name is left as-is so route_to_specialist returns
            # its error string and audits the anomaly exactly once.
            specialist_calls = [
                {
                    "specialist": _canonical_specialist(
                        tu["input"].get("specialist", "")
                    ),
                    "query": str(tu["input"].get("query", "")),
                }
                for tu in specialist_tool_uses
            ]

            # Cap fan-out width per turn (inert by default — cap == roster
            # size). Dispatch the first `cap`; the rest get a skip tool_result
            # the model can react to. See router.partition_specialist_fanout.
            run_tool_uses, run_calls, skipped_results, fanout_cap = (
                partition_specialist_fanout(
                    specialist_tool_uses,
                    specialist_calls,
                    self._settings.max_parallel_specialists,
                )
            )

            if debug_collector and specialist_calls:
                evt = debug_collector.emit("routing_decision", {
                    "iteration": iteration,
                    # Set on both emitters or neither: a field present on some
                    # routing_decision events and absent on others is worse
                    # than no field at all for anything reading the stream.
                    "phase": "loop",
                    "requested_count": len(specialist_calls),
                    "cap": fanout_cap,
                    "dispatched_count": len(run_calls),
                    "skipped_count": len(skipped_results),
                    "specialists": [
                        {
                            "specialist": c["specialist"],
                            "query": c["query"],
                        }
                        for c in run_calls
                    ],
                })
                yield debug_collector.to_sse_dict(evt)

            if debug_collector and skill_tool_uses:
                evt = debug_collector.emit("skill_invocation", {
                    "iteration": iteration,
                    "calls": [
                        {"tool": tu["name"], "input_preview": str(tu["input"])[:200]}
                        for tu in skill_tool_uses
                    ],
                })
                yield debug_collector.to_sse_dict(evt)

            if debug_collector and mcp_tool_uses:
                evt = debug_collector.emit("mcp_invocation", {
                    "iteration": iteration,
                    "calls": [
                        {"tool": tu["name"], "input_preview": str(tu["input"])[:200]}
                        for tu in mcp_tool_uses
                    ],
                })
                yield debug_collector.to_sse_dict(evt)

            # Name the round before the sentinel, so an ordered consumer has
            # the label in hand by the time it switches its in-flight
            # indicator on. Never fatal: a labelling bug must not take the
            # turn down with it.
            try:
                # The progress call is not what the round is doing.
                round_work = [
                    tu for tu in tool_uses
                    if tu["name"] != turn_progress.SHOW_PROGRESS
                ]
                activity = summarize_activity(round_work, iteration=iteration)
                # What the round read, by kind, for the card when the turn
                # names nothing itself.
                reads = turn_reads.add(round_work)
                if activity is not None and reads:
                    activity["reads"] = reads
                if cut_id is not None:
                    # The step was writing something long, not the call's own
                    # kind of work (which never ran).
                    activity = {**(activity or fallback_activity(iteration=iteration)), "label": CUT_OFF_LABEL}
            except Exception:
                logger.warning(
                    "activity_label_failed iteration=%d", iteration, exc_info=True
                )
                activity = None
            # Exactly one activity per sentinel, unconditionally. Skipping it
            # on the None/raise paths would leave a client that keeps the last
            # label it saw captioning this round with the previous round's
            # work; an honest "Working…" beats a stale name.
            yield activity or fallback_activity(iteration=iteration)
            # A step in their browser only they can do is asked in the chat
            # they're typing in, with Done and Stop, before it runs
            # (browser.tools.hand_over_asks).
            try:
                from openexecutive.browser import tools as browser_tools

                for ask in browser_tools.hand_over_asks(
                    tool_uses, getattr(current_session.get(), "session_id", None)
                ):
                    yield ask
            except Exception:
                logger.warning("hand_over_ask_failed iteration=%d", iteration, exc_info=True)

            # Signal that tool calls are in flight so the client can show progress.
            yield self._THINKING

            results_by_id: dict[str, str] = dict(group_results)
            # tool_use ids whose handler reported a failure; their tool_result
            # carries is_error (see ToolOutcome for what that does and does
            # not reach on the local path). open_tools/use_tool results from
            # group_results are seeded above and never error-flagged: a bad
            # group name comes back as a plain error payload, not a failure.
            error_tool_use_ids: set[str] = set()

            event_cursor = len(debug_collector._events) if debug_collector else 0
            session_id = getattr(current_session.get(), "session_id", None)
            to_card: dict[tuple[str, bool, str], list[tuple[dict[str, Any], str, dict[str, Any]]]] = {}
            for tu, reason in held_uses:
                # Never dropped: the exact step goes on an Approve card for
                # the owner, or the model is told why it didn't run.
                kind = "mcp" if tu["name"] in MCP_TOOL_NAMES else "skill"
                label = tu["name"]
                held_tool, held_input, held_mcp = tu["name"], tu["input"], False
                if label == "call_tool" and isinstance(tu["input"], dict):
                    named = tu["input"].get("name")
                    label = named[:200] if isinstance(named, str) and named else label
                    arguments = tu["input"].get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (ValueError, RecursionError):
                            arguments = None
                    if isinstance(named, str) and named and isinstance(arguments, dict | None):
                        held_tool, held_input, held_mcp = named, dict(arguments or {}), True
                can_card = (
                    tu["id"] not in script_ids
                    and (held_mcp or tu["name"] not in MCP_TOOL_NAMES)
                    and isinstance(held_input, dict)
                    and levels.carriable(held_tool, mcp=held_mcp)
                )
                logger.info("%s:%s held for approval (%s)", kind, _loggable_tool(label), reason)
                audit_log(
                    "tool_invocation",
                    f"{kind}:{label} waiting for approval: {reason}",
                    session_id=session_id,
                    turn_id=turn_id,
                    actor="executive",
                    details={
                        "tool": label,
                        "kind": kind,
                        "iteration": iteration,
                        "ok": False,
                        "held": reason,
                        "card": can_card,
                        "sources": sorted(getattr(outside_content.current(), "sources", ())),
                    },
                    private=private_turn or bool(
                        pinned_delegation is not None and pinned_delegation.touched_mail
                    ) or reason in (levels.OWN_MAIL, levels.MAIL_FLAGGED, levels.PRIVATE),
                )
                if not can_card:
                    results_by_id[tu["id"]] = levels.not_carriable_error(label, reason)
                    continue
                to_card.setdefault((held_tool, held_mcp, reason), []).append((tu, label, held_input))
            # Steps of one kind held together (fourteen renames in a round)
            # go on one card, so one Approve does them all.
            for (held_tool, held_mcp, reason), group in to_card.items():
                held_result = await asyncio.to_thread(
                    levels.hold, held_tool, group[0][2],
                    mcp=held_mcp, reason=reason, session=current_session.get(),
                ) if len(group) == 1 else await asyncio.to_thread(
                    levels.hold_many, held_tool, [held_input for _, _, held_input in group],
                    mcp=held_mcp, reason=reason, session=current_session.get(),
                )
                for tu, _, _ in group:
                    results_by_id[tu["id"]] = held_result
                # The card shows under this reply (the web chat) and goes to
                # the chat app the turn came from (Executive.chat's cards).
                held_chip = levels.waiting_chip(held_result, label=group[0][1], iteration=iteration)
                if held_chip is not None:
                    yield held_chip
            for tu in withheld_uses:
                logger.warning(
                    "skill:%s refused — not offered in %s mode", tu["name"], workspace_mode
                )
                results_by_id[tu["id"]] = withheld_tool_error(tu["name"], workspace_mode)
            if specialist_calls:
                spec_t0 = time.monotonic()
                failed_calls: list[int] = []
                # Specialists get the stage from the same profile the
                # Executive reasons over; no session profile → the router
                # reads it from disk.
                session_stage = getattr(
                    getattr(current_session.get(), "company_profile", None), "stage", None
                )
                specialist_results = await route_parallel(
                    run_calls,
                    episodic_context=episodic_context,
                    session_id=session_id,
                    debug_collector=debug_collector,
                    conversation_context=conversation_context,
                    company_stage=(
                        session_stage if isinstance(session_stage, str) else None
                    ),
                    # Solo: what the principal does, so a specialist advises
                    # a VP inside a large company differently from an owner.
                    # Resolved with the turn's mode; team sends no tag.
                    principal_role=principal_role_tag,
                    record_source=turn_sources.add if turn_sources is not None else None,
                    failed_calls_out=failed_calls,
                    # The web chat shows a missing area under the reply;
                    # everywhere else the reply itself has to say so.
                    tell_user_when_unavailable=not getattr(
                        current_session.get(), "from_web_chat", False
                    ),
                )
                # Only a specialist that actually answered counts as
                # consulted: the committee picks its critics from that list,
                # and each consulted department's memory records the turn.
                answered = [
                    (call, result)
                    for i, (call, result) in enumerate(
                        zip(run_calls, specialist_results, strict=True)
                    )
                    if i not in failed_calls
                ]
                if turn_sources is not None:
                    for i in failed_calls:
                        turn_sources.mark_unavailable(run_calls[i]["specialist"])
                    for call, _ in answered:
                        turn_sources.mark_answered(call["specialist"])
                spec_ms = round((time.monotonic() - spec_t0) * 1000)
                for tu, result in zip(
                    run_tool_uses, specialist_results, strict=True
                ):
                    results_by_id[tu["id"]] = result
                # Over-budget specialist calls get an explicit skip result
                # rather than being silently fanned out.
                results_by_id.update(skipped_results)
                if skipped_results:
                    logger.info(
                        "specialist fan-out capped: requested=%d cap=%d skipped=%d",
                        len(specialist_calls),
                        fanout_cap,
                        len(skipped_results),
                    )
                # Two independent filters, both of which must hold before a
                # call counts as a real consult. `answered` (above) drops the
                # ones that FAILED; this drops the ones that reached NOBODY —
                # an unresolvable name still gets its tool_result above (the
                # model needs the error to recover, and the API needs one
                # result per tool_use), but it consulted no one, so letting it
                # through would file a specialist_consult row for a consult
                # that never ran (the audit graph then draws a specialist node
                # and attributes later tool calls to it) and seed committee
                # reviewer selection and the department sync with a token they
                # all silently no-op on. The pre-pass drops such names
                # outright; this is the same rule on the path that cannot.
                reached = [
                    (call, result)
                    for call, result in answered
                    if resolve_specialist_name(call["specialist"]) is not None
                ]
                reached_calls = [c for c, _ in reached]
                reached_results = [r for _, r in reached]
                # Every dispatched call, failed ones too: this list only marks
                # the next round as the synthesis pass in the debug panel, so
                # it deliberately does NOT use the filtered list.
                specialists_consulted.extend(c["specialist"] for c in run_calls)
                if consulted_out is not None:
                    consulted_out.extend(c["specialist"] for c in reached_calls)
                if specialist_outputs_out is not None:
                    for call, result in reached:
                        # Last-write-wins if the same specialist is consulted
                        # in multiple iterations — committee only needs a
                        # representative excerpt per domain.
                        specialist_outputs_out[call["specialist"]] = result
                _audit_specialist_consults(
                    run_calls=reached_calls,
                    specialist_results=reached_results,
                    session_id=session_id,
                    turn_id=turn_id,
                    iteration=iteration,
                    phase="loop",
                    duration_ms=spec_ms,
                    conversation_context=conversation_context,
                    system_blocks=system_blocks,
                )

            if skill_tool_uses:
                for tu in skill_tool_uses:
                    logger.info("→ skill:%s  input=%s", tu["name"], _log_value(tu["name"], tu["input"]))
                # return_exceptions=True: one crashing handler must not abort
                # the whole turn. See `_tool_error_result`.
                # What a running tool says meanwhile (a workflow's steps)
                # streams as it happens (orchestrator.turn_live).
                # Bound only while the task is made (it copies the context),
                # never across a yield.
                feed = turn_live.Feed()
                with feed.bound():
                    skill_task = asyncio.ensure_future(asyncio.gather(
                        *(
                            with_turn_limits(reading_budget.limits, turn_handlers[tu["name"]](tu["input"]))
                            for tu in skill_tool_uses
                        ),
                        return_exceptions=True,
                    ))
                # Closed here, so a turn that stops at a yield cancels the
                # tools at once, not when the generator is collected.
                async with contextlib.aclosing(feed.until(skill_task)) as live_events:
                    async for live in live_events:
                        yield live
                raw_skill_results = skill_task.result()
                for tu, raw in zip(skill_tool_uses, raw_skill_results, strict=True):
                    if isinstance(raw, BaseException):
                        # Cancellation is not a tool failure — `gather` captures
                        # it like any other exception, so re-raise it or the
                        # turn-timeout / client-disconnect paths in
                        # api/routes/chat.py silently stop working.
                        if isinstance(raw, asyncio.CancelledError):
                            raise raw
                        # Computed, not hardcoded: turn_handlers also
                        # carries the client-side web_search overlay, which
                        # audits as client_tool on the success path below. A
                        # crash must not reclassify it as a skill.
                        tool_kind = (
                            "skill" if tu["name"] in _ALL_SKILL_HANDLERS else "client_tool"
                        )
                        logger.exception(
                            "%s:%s raised — session=%s turn=%s iteration=%d",
                            tool_kind, tu["name"], session_id, turn_id, iteration,
                            exc_info=raw,
                        )
                        audit_log(
                            "tool_invocation",
                            f"{tool_kind}:{tu['name']} FAILED: {type(raw).__name__}",
                            session_id=session_id,
                            turn_id=turn_id,
                            actor="executive",
                            details={
                                "tool": tu["name"],
                                "kind": tool_kind,
                                "iteration": iteration,
                                "ok": False,
                                "error": repr(raw)[:ERROR_DETAIL_LEN],
                            },
                            # Act as me reads the speaker's own mailbox; the
                            # fact tools carry the principal's own words.
                            private=_private_tool_row(tu["name"], tu["input"])
                            or tu["name"] in DRAFT_ARTIFACT_TOOL_HANDLERS,
                            private_to_person=_artifact_row_owner(tu["name"], tu["input"]),
                        )
                        # Hand the model an error tool_result and move on. No
                        # chip: summarize_action must never see an exception.
                        results_by_id[tu["id"]] = _tool_error_result(tu["name"], raw)
                        # Same flag a handler-reported ToolOutcome failure gets
                        # below: a crash is an error too, and without this the
                        # API sees a successful tool_result carrying an error
                        # payload.
                        error_tool_use_ids.add(tu["id"])
                        continue
                    # Any handler may return a ToolOutcome to report failure;
                    # everything downstream of this line sees a plain string.
                    result, is_error = unwrap_tool_outcome(raw)
                    if is_error:
                        error_tool_use_ids.add(tu["id"])
                    logger.info("← skill:%s  result=%s", tu["name"], _log_value(tu["name"], result))
                    results_by_id[tu["id"]] = result
                    # Inline action chip for side-effecting tools. None
                    # when the tool is read-only (search_skills, load_skill,
                    # list_people, ask_about_person, lookup_person) or
                    # when the handler reported an error.
                    chip = summarize_action(
                        tool_name=tu["name"],
                        tool_input=tu["input"],
                        tool_result=result,
                        iteration=iteration,
                        workspace_mode=workspace_mode,
                    )
                    if chip is not None:
                        yield chip
                    # Form proposals reach the Ask OE panel as a dedicated
                    # SSE event (not an action chip — nothing was mutated).
                    # Only deliver what the handler accepted; a shape error
                    # already went back to the model as the tool_result.
                    if tu["name"] == PROPOSE_FORM_VALUES:
                        try:
                            handler_ok = "error" not in json.loads(result)
                        except (json.JSONDecodeError, TypeError):
                            # The handler always returns JSON today; if a
                            # future change breaks that, drop the event
                            # rather than killing the whole SSE stream.
                            logger.warning(
                                "propose_form_values returned non-JSON; "
                                "suppressing form_patch event"
                            )
                            handler_ok = False
                        if handler_ok:
                            yield build_form_patch_event(tu["input"], iteration)
                    audit_log(
                        "tool_invocation",
                        f"skill:{tu['name']} input={audit_tool_input(tu['name'], tu['input'])}",
                        session_id=session_id,
                        turn_id=turn_id,
                        actor="executive",
                        details={
                            "tool": tu["name"],
                            # Handlers overlaid for this turn only (today the
                            # client-side web_search) audit as client_tool, so
                            # the view can tell them from the module-wide
                            # skills and from the server-side web_search,
                            # which audits separately as kind="server_tool".
                            "kind": (
                                "skill"
                                if tu["name"] in _ALL_SKILL_HANDLERS
                                else "client_tool"
                            ),
                            "iteration": iteration,
                            "result_preview": audit_tool_result(tu["name"], result),
                        },
                        full={
                            "input": audit_tool_input_full(tu["name"], tu["input"]),
                            "result": audit_tool_result_full(tu["name"], result),
                            "active_prompt_blocks": _system_block_names(system_blocks),
                        },
                        # A document tool quotes the speaker's own
                        # documents: the row is theirs alone.
                        private=_private_tool_row(tu["name"], tu["input"], result)
                        or tu["name"] in DRAFT_ARTIFACT_TOOL_HANDLERS,
                        private_to_person=_artifact_row_owner(tu["name"], tu["input"], result),
                    )

            if mcp_tool_uses and self._mcp_gateway is not None:
                # A teammate's mail with the Executive stays theirs: every
                # mailbox read the model makes is filtered (teammate_mail).
                guarded_call_tool = typed_booking_only(booking_offered, guard_call_tool(
                    take_the_lead.gated_call_tool(self._mcp_gateway.call_tool, source="scheduled")
                    if leading
                    else take_the_lead.asked_call_tool(self._mcp_gateway.call_tool)
                    if asked_checks
                    else self._mcp_gateway.call_tool
                ))
                _mcp_dispatch = {
                    # An empty search names the groups open_tools opens.
                    "search_tools": tool_groups.searching(self._mcp_gateway.search_tools, deferred_tools),
                    # A replay (memory.replay) makes the reads and records the rest.
                    "call_tool": (
                        chat_replay.faked_call_tool(guarded_call_tool, current_session.get())
                        if replaying
                        else guarded_call_tool
                    ),
                    "load_mcp_server": self._mcp_gateway.load_mcp_server,
                }
                for tu in mcp_tool_uses:
                    if tu["name"] == "call_tool":
                        logger.info(
                            "→ %s  args=%s",
                            tu["input"].get("name", "call_tool"),
                            _log_value(tu["name"], tu["input"].get("arguments", "")),
                        )
                    else:
                        logger.info("→ %s  input=%s", tu["name"], _log_value(tu["name"], tu["input"]))
                # Same isolation as the skill gather above: a gateway crash on
                # one tool must not take the turn down with it.
                try:
                    mcp_results = await asyncio.gather(
                        *(_mcp_dispatch[tu["name"]](tu["input"]) for tu in mcp_tool_uses),
                        return_exceptions=True,
                    )
                except asyncio.CancelledError:
                    # The turn was cut off (a deadline, a stop) with these
                    # calls in flight. A request already sent may still run
                    # on the server, a send included, so each gets its row,
                    # as run_script's calls do.
                    for tu in mcp_tool_uses:
                        tool_label = tu["input"].get("name", tu["name"]) if tu["name"] == "call_tool" else tu["name"]
                        audit_log(
                            "tool_invocation",
                            f"mcp:{tool_label} CANCELLED",
                            session_id=session_id,
                            turn_id=turn_id,
                            actor="executive",
                            details={
                                "tool": tool_label,
                                "kind": "mcp",
                                "iteration": iteration,
                                "ok": False,
                                "cancelled": True,
                            },
                        )
                    raise
                for tu, raw in zip(mcp_tool_uses, mcp_results, strict=True):
                    tool_label = tu["input"].get("name", tu["name"]) if tu["name"] == "call_tool" else tu["name"]
                    if isinstance(raw, BaseException):
                        if isinstance(raw, asyncio.CancelledError):
                            raise raw
                        logger.exception(
                            "mcp:%s raised — session=%s turn=%s iteration=%d",
                            tool_label, session_id, turn_id, iteration,
                            exc_info=raw,
                        )
                        audit_log(
                            "tool_invocation",
                            f"mcp:{tool_label} FAILED: {type(raw).__name__}",
                            session_id=session_id,
                            turn_id=turn_id,
                            actor="executive",
                            details={
                                "tool": tool_label,
                                "kind": "mcp",
                                "iteration": iteration,
                                "ok": False,
                                "error": repr(raw)[:ERROR_DETAIL_LEN],
                            },
                        )
                        results_by_id[tu["id"]] = _tool_error_result(tool_label, raw)
                        # Same flag the skill crash path sets: without it the
                        # API sees a successful tool_result whose payload
                        # happens to be an error string.
                        error_tool_use_ids.add(tu["id"])
                        continue
                    result = raw
                    logger.info("← %s  result=%s", tool_label, _log_value(tool_label, result))
                    results_by_id[tu["id"]] = result
                    # MCP chip emission. search_tools is read-only (gets
                    # filtered out by summarize_action's allowlist);
                    # call_tool and load_mcp_server both surface a chip.
                    chip = summarize_action(
                        tool_name=tu["name"],
                        tool_input=tu["input"],
                        tool_result=result,
                        iteration=iteration,
                    )
                    if chip is not None:
                        yield chip
                    audit_log(
                        "tool_invocation",
                        f"mcp:{tool_label} input={audit_tool_input(tool_label, tu['input'])}",
                        session_id=session_id,
                        turn_id=turn_id,
                        actor="executive",
                        details={
                            "tool": tool_label,
                            "kind": "mcp",
                            "iteration": iteration,
                            "result_preview": audit_tool_result(tool_label, result),
                        },
                        full={
                            "input": audit_tool_input_full(tool_label, tu["input"]),
                            "result": audit_tool_result_full(tool_label, result),
                            "active_prompt_blocks": _system_block_names(system_blocks),
                        },
                    )

            if script_tool_uses and self._mcp_gateway is not None:
                # Each call a script makes is a call_tool in every way that
                # matters: the gateway's own gates (discovery, deny-list,
                # recipients), Take the lead's gate on an unattended run, and
                # the same chip and audit row, as each call happens.
                script_call = typed_booking_only(booking_offered, guard_call_tool(
                    take_the_lead.gated_call_tool(self._mcp_gateway.call_tool, source="scheduled")
                    if leading
                    else take_the_lead.asked_call_tool(self._mcp_gateway.call_tool)
                    if asked_checks
                    else self._mcp_gateway.call_tool
                ))
                # The Executive's own tools a script may call this turn: the
                # fixed list, less any this turn is not offered. Each runs
                # through the turn's handler (Take the lead's gate included).
                script_own = frozenset(
                    n for n in step_script.CHAT_OWN_TOOLS
                    if n in turn_handlers and n not in withheld_tools
                )
                # (call_tool input, result text, the exception if the call
                # raised); an own tool's input is {"name", "arguments", "own": True}.
                made: list[tuple[dict[str, Any], str, BaseException | None]] = []
                # The safety check's notices on the script's reads: shown to
                # the model above the script's result, never put into the
                # text the script parses.
                notices: list[str] = []

                async def _script_call(
                    tool: str,
                    arguments: dict[str, Any],
                    _call: Any = script_call,
                    _made: list[tuple[dict[str, Any], str, BaseException | None]] = made,
                    _notices: list[str] = notices,
                    _own: frozenset[str] = script_own,
                    _handlers: dict[str, Any] = turn_handlers,
                    _counts: dict[str, int] = script_counts,
                    _iteration: int = iteration,
                    _session_id: str | None = session_id,
                    _turn_id: str | None = turn_id,
                ) -> tuple[str, bool]:
                    own = tool in _own
                    call_input: dict[str, Any] = {"name": tool, "arguments": arguments}
                    if own:
                        call_input["own"] = True
                    # Budgets for the whole turn: a direct call needs a
                    # tool_use each, a script could otherwise make thousands.
                    cap = (
                        step_script.CHAT_OWN_TOOL_CAPS.get(tool)
                        if own and not principal_speaking(current_session.get())
                        else None
                    )
                    if _counts.get("", 0) >= self._settings.chat_script_max_calls or (
                        cap is not None and _counts.get(tool, 0) >= cap
                    ):
                        limit = cap if cap is not None and _counts.get(tool, 0) >= cap else (
                            self._settings.chat_script_max_calls
                        )
                        text = json.dumps({
                            "error": f"{tool} was not run: this turn's tools may make at most "
                            f"{limit} such calls. Say what is left and offer to continue."
                        })
                        return text, True
                    _counts[""] = _counts.get("", 0) + 1
                    _counts[tool] = _counts.get(tool, 0) + 1
                    # A script's calls run one after another: once one has
                    # read outside content, the rest are held as a direct
                    # call would be (orchestrator.outside_content).
                    probe_name, probe_input = (
                        (tool, arguments) if own else ("call_tool", {"name": tool, "arguments": arguments})
                    )
                    await outside_content.settle()
                    # A script's call is checked as a direct one would be:
                    # what a gate notices runs or waits on an Approve card,
                    # by the owner's level (orchestrator.levels).
                    script_reason: str | None = None
                    if outside_content.read_outside() and outside_content.outside_withholds(
                        probe_name, probe_input
                    ):
                        script_reason = (
                            levels.STRANGER if outside_content.answers_off_team_mail() else levels.outside_reason()
                        )
                    elif pinned_delegation is not None and mail_turn_withholds(
                        probe_name, probe_input, pinned=pinned_delegation
                    ):
                        script_reason = levels.mail_reason()
                    if (
                        script_reason is not None
                        and levels.decide(script_reason) == levels.CARD
                        and not levels.only_to_owner(probe_name, probe_input)
                    ):
                        if isinstance(arguments, dict) and levels.carriable(tool, mcp=not own):
                            held = await asyncio.to_thread(
                                levels.hold, tool, dict(arguments), mcp=not own,
                                reason=script_reason, session=current_session.get(),
                            )
                            return held, False
                        return levels.not_carriable_error(tool[:200], script_reason), True
                    if not own:
                        await outside_content.vet_reads(
                            [{"name": probe_name, "input": probe_input}], self._mcp_gateway
                        )
                    source = outside_content.source_of(probe_name, probe_input)
                    try:
                        if own:
                            text = str(await with_turn_limits(reading_budget.limits, _handlers[tool](arguments)))
                        else:
                            text = str(await _call({"name": tool, "arguments": arguments}))
                    except asyncio.CancelledError:
                        # Stopped mid-call (the script's clock, or the turn):
                        # it may have run, and the drain below won't see it.
                        audit_log(
                            "tool_invocation",
                            f"{'skill' if own else 'mcp'}:{_loggable_tool(tool)} CANCELLED (run_script)",
                            session_id=_session_id,
                            turn_id=_turn_id,
                            actor="executive",
                            details={
                                "tool": tool[:200],
                                "kind": "skill" if own else "mcp",
                                "via": "run_script",
                                "iteration": _iteration,
                                "ok": False,
                                "cancelled": True,
                            },
                        )
                        raise
                    except Exception as exc:
                        logger.warning(
                            "script call_tool:%s raised %s", _loggable_tool(tool), type(exc).__name__
                        )
                        text = _tool_error_result(tool, exc)
                        _made.append((call_input, text, exc))
                        if source is not None:
                            # The error can carry what the tool read: checked
                            # like any read.
                            notice = await outside_content.mark_read(source, probe_input, text, self._mcp_gateway)
                            if notice:
                                _notices.append(notice)
                        return text, True
                    if source is not None:
                        notice = await outside_content.mark_read(source, probe_input, text, self._mcp_gateway)
                        if notice:
                            _notices.append(notice)
                    _made.append((call_input, text, None))
                    return text, looks_like_error(text)

                # Well inside the turn's own deadline, so a long script ends
                # here, with the list of calls it already made, rather than
                # being cut off with the turn and leaving the model no record.
                script_clock = max(30.0, min(600.0, self._settings.chat_stream_timeout_s / 2))
                for tu in script_tool_uses:
                    if tu["name"] == step_script.LIST_SAVED_TOOLS_TOOL:
                        results_by_id[tu["id"]] = step_script.list_saved_tools_result()
                        continue
                    script_args = tu["input"] if isinstance(tu["input"], dict) else {}
                    script = script_args.get("script")
                    script_text = str(script or "")
                    logger.info(
                        "→ run_script  chars=%d saved_tool=%s",
                        len(script_text), _loggable_tool(str(script_args.get("tool") or "-")),
                    )
                    script_result = json.dumps({"error": "the script did not finish"})
                    script_failed = True
                    script_stats: dict[str, Any] = {}
                    notices.clear()
                    # aclosing: a stopped turn closes the script, and with it
                    # the Monty worker, instead of leaving it to the GC.
                    async with contextlib.aclosing(
                        step_script.run_script_tool(
                            script_args,
                            tools=None,
                            call=_script_call,
                            origin="chat",
                            # Saving and running while someone on the team
                            # (the principal or a teammate) speaks on a
                            # verified, interactive surface: each call a
                            # saved tool makes is checked on the turn that
                            # runs it, by that turn's level.
                            may_save=not principal_withheld and not unattended_withheld,
                            may_run_saved=not principal_withheld and not unattended_withheld,
                            wall_clock_s=script_clock,
                            own_tools=script_own,
                        )
                    ) as script_steps:
                        async for kind, payload in script_steps:
                            for call_input, text, raised in made:
                                label = str(call_input["name"])[:200]
                                if call_input.get("own"):
                                    # One of the Executive's own tools: the
                                    # same chip and row as a direct call.
                                    own_input = call_input["arguments"]
                                    if raised is None:
                                        chip = summarize_action(
                                            tool_name=label,
                                            tool_input=own_input,
                                            tool_result=text,
                                            iteration=iteration,
                                            workspace_mode=workspace_mode,
                                        )
                                        if chip is not None:
                                            yield chip
                                    audit_log(
                                        "tool_invocation",
                                        (
                                            f"skill:{label} input={audit_tool_input(label, own_input)} (run_script)"
                                            if raised is None
                                            else f"skill:{label} FAILED: {type(raised).__name__} (run_script)"
                                        ),
                                        session_id=session_id,
                                        turn_id=turn_id,
                                        actor="executive",
                                        details={
                                            "tool": label,
                                            "kind": "skill",
                                            "via": "run_script",
                                            "iteration": iteration,
                                            **(
                                                {"result_preview": audit_tool_result(label, text)}
                                                if raised is None
                                                else {"ok": False, "error": repr(raised)[:ERROR_DETAIL_LEN]}
                                            ),
                                        },
                                        full=(
                                            {
                                                "input": audit_tool_input_full(label, own_input),
                                                "result": audit_tool_result_full(label, text),
                                                "active_prompt_blocks": _system_block_names(system_blocks),
                                            }
                                            if raised is None
                                            else None
                                        ),
                                        private=_private_tool_row(label),
                                        private_to_person=_artifact_row_owner(label),
                                    )
                                    continue
                                if raised is None:
                                    chip = summarize_action(
                                        tool_name="call_tool",
                                        tool_input=call_input,
                                        tool_result=text,
                                        iteration=iteration,
                                    )
                                    if chip is not None:
                                        yield chip
                                    audit_log(
                                        "tool_invocation",
                                        f"mcp:{label} input={audit_tool_input(label, call_input)} (run_script)",
                                        session_id=session_id,
                                        turn_id=turn_id,
                                        actor="executive",
                                        details={
                                            "tool": label,
                                            "kind": "mcp",
                                            "via": "run_script",
                                            "iteration": iteration,
                                            "result_preview": audit_tool_result(label, text),
                                        },
                                        full={
                                            "input": audit_tool_input_full(label, call_input),
                                            "result": audit_tool_result_full(label, text),
                                            "active_prompt_blocks": _system_block_names(system_blocks),
                                        },
                                    )
                                else:
                                    # Same shape as a direct call_tool that raised.
                                    audit_log(
                                        "tool_invocation",
                                        f"mcp:{label} FAILED: {type(raised).__name__} (run_script)",
                                        session_id=session_id,
                                        turn_id=turn_id,
                                        actor="executive",
                                        details={
                                            "tool": label,
                                            "kind": "mcp",
                                            "via": "run_script",
                                            "iteration": iteration,
                                            "ok": False,
                                            "error": repr(raised)[:ERROR_DETAIL_LEN],
                                        },
                                    )
                            made.clear()
                            if kind == "stats":
                                script_stats = payload
                            elif kind == "done":
                                script_result, script_failed = payload
                    logger.info("← run_script  result=%s", _trunc(script_result))
                    read_notices[tu["id"]] = "".join(dict.fromkeys(notices))
                    results_by_id[tu["id"]] = read_notices[tu["id"]] + script_result
                    # Agent Activity: the built tool as one card, with the
                    # calls it made (the result lists them, capped). Sent by
                    # the round's event replay below, like specialist events.
                    if debug_collector:
                        try:
                            listed = json.loads(script_result).get("calls") or []
                        except (ValueError, AttributeError):
                            listed = []
                        debug_collector.emit("script_run", {
                            "iteration": iteration,
                            "ok": not script_failed,
                            "saved_tool": script_args.get("tool"),
                            "kept_as": script_args.get("save_as"),
                            "calls": [
                                {"tool": str(c.get("tool", ""))[:200], "ok": bool(c.get("ok"))}
                                for c in listed if isinstance(c, dict)
                            ],
                            "calls_made": script_stats.get("calls", len(listed)),
                            "duration_ms": script_stats.get("duration_ms"),
                        })
                    # The script itself: its source and what it returned, so
                    # the per-call rows above can be traced back to it.
                    audit_log(
                        "tool_invocation",
                        f"script:run_script ({'failed' if script_failed else 'ok'})",
                        session_id=session_id,
                        turn_id=turn_id,
                        actor="executive",
                        details={
                            "tool": step_script.RUN_SCRIPT_TOOL,
                            "kind": "script",
                            "iteration": iteration,
                            "ok": not script_failed,
                            # Calls made and time taken (the usage summary
                            # adds these up: audit.logger.script_summary).
                            **script_stats,
                            "result_preview": audit_tool_result(step_script.RUN_SCRIPT_TOOL, script_result),
                            **({"saved_tool": str(script_args["tool"])[:60]} if script_args.get("tool") else {}),
                            **({"save_as": str(script_args["save_as"])[:60]} if script_args.get("save_as") else {}),
                        },
                        full={
                            "input": {
                                k: script_args.get(k)
                                for k in ("script", "tool", "inputs", "save_as", "description")
                                if script_args.get(k) is not None
                            },
                            "result": audit_tool_result_full(step_script.RUN_SCRIPT_TOOL, script_result),
                            "active_prompt_blocks": _system_block_names(system_blocks),
                        },
                    )

            if debug_collector:
                for evt in debug_collector._events[event_cursor:]:
                    yield debug_collector.to_sse_dict(evt)

            # A picture a tool held (the browser's screenshot) comes off its
            # answer first. The model sees it only when it can see images and
            # the check below can read it too (text in a picture is outside
            # content like any other); otherwise it gets the text alone.
            images_by_id: dict[str, dict[str, Any]] = {}
            for tu in tool_uses:
                bare, image = tool_images.take(results_by_id.get(tu["id"]))
                if image is not None:
                    results_by_id[tu["id"]] = bare
                    images_by_id[tu["id"]] = image
            sees = bool(images_by_id) and sees_images(stream_model) and sees_images(content_scan.scan_model())
            # What this round read from outside reaches the model with its
            # results: from the next round on the turn is held.
            # The check sees each read's result first (orchestrator.content_scan):
            # a flagged one reaches the model behind a notice not to follow it.
            for tu, source in [
                (tu, source) for tu in dispatch_uses
                if (source := outside_content.source_of(tu["name"], tu["input"])) is not None
            ]:
                if tu["id"] not in held_ids and tu not in withheld_uses:
                    notice = await outside_content.mark_read(
                        source,
                        tu["input"],
                        tool_images.content(
                            results_by_id.get(tu["id"], ""), images_by_id.get(tu["id"]), sees=sees
                        ),
                        self._mcp_gateway,
                    )
                    if notice:
                        results_by_id[tu["id"]] = _with_notice(results_by_id.get(tu["id"], ""), notice)
                        read_notices[tu["id"]] = notice
            # The owner's own mail this round read is checked the same way,
            # and decides how much of the Act as me lockdown holds from the
            # next round on (delegation.lockdown.mail_lock_holds).
            if pinned_delegation is not None and reads_mail_now:
                for tu in dispatch_uses:
                    if tu["name"] not in MAILBOX_TOOL_NAMES and tu["name"] not in _OWN_RECALL_TOOL_NAMES:
                        continue
                    if tu["id"] not in results_by_id:
                        continue  # never ran (not offered): no mail was read
                    checked, notice = await outside_content.check_mail_read(
                        pinned_delegation, results_by_id.get(tu["id"], "")
                    )
                    results_by_id[tu["id"]] = checked
                    if notice:
                        results_by_id[tu["id"]] = _with_notice(results_by_id.get(tu["id"], ""), notice)
                        # Before the content check's notice, as _with_notice
                        # put them, so every later part carries both.
                        read_notices[tu["id"]] = notice + read_notices.get(tu["id"], "")
                # A script that ran beside the read may have read mail the
                # loop never checks: the turn stays unvetted.
                ran_script = any(
                    tu["name"] in (step_script.RUN_SCRIPT_TOOL, "run_python_job")
                    for tu in [*skill_tool_uses, *mcp_tool_uses, *script_tool_uses]
                )
                if not ran_script:
                    outside_content.check_mail_reads_done(pinned_delegation)

            current_messages.append({"role": "assistant", "content": response_content})
            # Cap here, at the single point every result reaches the model,
            # rather than at each producer: this also covers specialist
            # output, tool errors and the unknown-tool fallback, and it
            # leaves non-model consumers (the propose_form_values JSON
            # parse above, the audit trail) reading the full text.
            #
            # The cap is the turn-wide reading budget, which has to see every
            # result in order, so this stays a loop. `is_error` is set inside
            # it rather than in the dict literal because only the tool_uses
            # whose handler reported or raised a failure carry the key.
            tool_results: list[dict[str, Any]] = []
            for tu in tool_uses:
                turn_calls.append((tu["name"], str(results_by_id.get(tu["id"], ""))))
                claim_calls.append(
                    (claim_check.call_label(tu["name"], tu["input"]), str(results_by_id.get(tu["id"], "")))
                )
            # A call that may have changed something drops the parts held
            # from before it (orchestrator.reading).
            reading_budget.ran(tool_uses)
            for tu in tool_uses:
                result_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": tool_images.content(
                        reading_budget.cap(
                            tu["name"],
                            tu["input"],
                            app_links.for_model(results_by_id.get(tu["id"], f"Unknown tool: {tu['name']}")),
                            tu["id"],
                            read_notices.get(tu["id"], ""),
                        ),
                        images_by_id.get(tu["id"]),
                        sees=sees,
                    ),
                }
                if tu["id"] in error_tool_use_ids:
                    result_block["is_error"] = True
                tool_results.append(result_block)
            tool_results.extend(cut_results)
            # Cut in text written after a call that closed: the call ran, and the
            # next round carries the text on where it stopped.
            if cut_off and last_written is not None and last_written.get("type") == "text":
                tool_results.append({"type": "text", "text": CUT_OFF_NOTE})
                run_on = True
                # The claim check reads the cut text with the round that ends it.
                last_call = max(n for n, blk in enumerate(response_content) if blk.get("type") == "tool_use")
                cut_text = "".join(
                    str(blk.get("text", "")) for blk in response_content[last_call + 1:] if blk.get("type") == "text"
                )
            # Messages the person sent since the last round ride in this
            # round's user message, after the tool results (which must come
            # first). Only this loop's own, newest message changes, so the
            # cached prefix is untouched.
            # A tool came back with a list and this turn may build a tool:
            # nudge toward one run_script for the per-item work, once a
            # turn, in this user message (never a cached block).
            if (
                not fanout_hinted
                and self._script_tools
                and step_script.RUN_SCRIPT_TOOL not in not_offered
                and not (pinned_delegation is not None and pinned_delegation.read_mail)
                and not any(tu["name"] == step_script.RUN_SCRIPT_TOOL for tu in tool_uses)
                and any(step_script.lists_many(str(results_by_id.get(tu["id"], ""))) for tu in tool_uses)
            ):
                tool_results.append({"type": "text", "text": step_script.FANOUT_HINT})
                fanout_hinted = True
            added = inbox.take() if inbox is not None else []
            if added:
                tool_results.append({"type": "text", "text": render_added_messages(added)})
                # The person's own words, as much as the turn's message.
                for message in added:
                    seen_urls.add_from_text(message.text)
                # Audited by the inbox's owner (the web chat route), under
                # the same privacy as the turn's own message.
                yield {"type": "message_added", "ids": [m.id for m in added]}
            if inbox is not None and inbox.take_hand_over():
                tool_results.append({"type": "text", "text": HAND_OVER_NOTE})
                handing_over = True
                yield {"type": "handing_over"}
            for tu in tool_uses:
                reading_budget.came_back(tu["name"], tu["input"], results_by_id.get(tu["id"], ""))
            if not handing_over and reading_budget.repeating:
                logger.warning("round %d: a call came back the same %d times; writing up", iteration, REPEAT_LIMIT)
                tool_results.append({"type": "text", "text": REPEATING_NOTE})
                repeating = True
            elif not handing_over and not repeating and reading_budget.full:
                logger.info("round %d: the turn has read all its model holds; writing up", iteration)
                tool_results.append({"type": "text", "text": FULL_NOTE})
                repeating = True
                write_up_empty = FULL_EMPTY_REPLY
            elif not handing_over and not repeating and iteration >= MAX_STEPS:
                logger.info("round %d: the turn has run its %d steps; writing up", iteration, MAX_STEPS)
                tool_results.append({"type": "text", "text": STEPS_NOTE})
                repeating = True
                write_up_empty = STEPS_EMPTY_REPLY
            if web_search_tool is not None and web_searches.spent(searches_used) and not repeating:
                if spent_at is None:
                    # Spent: told once, and its other steps still run.
                    tool_results.append({"type": "text", "text": web_searches.SEARCHES_SPENT_NOTE})
                    spent_at = searches_used
                elif searches_used > spent_at and not handing_over:
                    # Searched again after being told: the next round writes
                    # up with no more calls, so a turn's searches stay bounded
                    # (the tool can't shrink without missing the cache).
                    logger.info("round %d: searched past the turn's budget; writing up", iteration)
                    tool_results.append({"type": "text", "text": web_searches.SEARCHES_OVER_NOTE})
                    repeating = True
                    write_up_empty = web_searches.SEARCHES_EMPTY_REPLY
            if sees:
                # Only the newest screenshots ride along: older ones would be
                # sent again every round.
                tool_images.drop_older(current_messages)
            if not tool_results:
                logger.warning(
                    "round %d ended on %s with no tool call to answer; blocks=%s",
                    iteration, final_msg.stop_reason, [b.get("type") for b in response_content],
                )
                tool_results.append({"type": "text", "text": EMPTY_ROUND_NOTE})
            current_messages.append({"role": "user", "content": tool_results})
            # Bounded to the messages this loop appended: current_messages
            # is a shallow copy, so anything at a lower index is still owned
            # by the caller and must not be mutated.
            if self._settings.enable_caching and apply_loop_cache_marker(
                current_messages, caller_message_count
            ):
                _clear_turn_marker(turn_marked)
                turn_marked = None

    async def chat(
        self,
        user_message: str,
        session: Session,
        retrieved_context: str = "",
        episodic_context: str = "",
        committee_review: bool = False,
        debug_collector: DebugCollector | None = None,
        attachment_blocks: list[dict[str, Any]] | None = None,
        person_id: int | None = None,
        co_present_person_ids: list[int] | None = None,
        peer_memory_reasoning_level: HonchoReasoningLevel | None = None,
        peer_memory_context: str | None = None,
        briefing_context: str = "",
        channel_context_block: str = "",
        memory_text: str | None = None,
        standing_facts: str | None = None,
        fetch_own_urls: bool = False,
        cards: list[int] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        """Non-streaming chat — collects and returns the full response.
        A turn the model refused ends on ``REFUSED_REPLY``.

        ``on_progress``, when given, sees each ``activity`` and ``progress``
        event as the turn runs (what the web chat's progress card shows),
        for a caller with its own way of showing it, such as the steps on a
        reply card while its work runs (``delegation.card_work``).

        ``cards``, when given, collects the ids of the approval cards the
        reply left (the ``decision_id`` on its ``action_taken`` events, as the
        web chat shows under the reply), for a chat app that shows them
        (``integrations.channel_cards``).

        When ``committee_review=True`` the call routes through
        ``stream_chat_with_committee`` so the answer is the revised
        response, not the raw draft. Lets non-HTTP callers get
        committee output without holding an SSE stream open; the
        channel adapters (Slack, Discord, email) all use the default.

        ``attachment_blocks`` is a list of Anthropic content blocks (image
        type) assembled by the caller from inbound file attachments.  Text
        document content is expected to be prepended to ``user_message``
        directly by the integration layer.

        ``person_id`` / ``co_present_person_ids`` / ``peer_memory_reasoning_level``
        thread through to the Honcho memory layer — see ``stream_chat``
        for details. ``peer_memory_reasoning_level=None`` (default)
        keeps each underlying entry point's own default (``"minimal"``
        for the streaming path, ``"medium"`` for the committee path).
        ``memory_text`` is the person's own words for the post-turn passes
        (see ``stream_chat``).

        ``fetch_own_urls`` lets ``web_fetch`` open the URLs in
        ``memory_text``. It is off unless the caller turns it on, and a
        channel turns it on only for a sender it matched to someone on the
        roster: anyone else who can message the bot gets no page opened
        from a link they pasted (search results stay openable).
        """
        # Build the kwargs dict so we can conditionally include the
        # reasoning_level only when caller specified one — otherwise
        # the inner method's own default ("low" or "medium") wins.
        common_kwargs: dict[str, Any] = {
            "user_message": user_message,
            "session": session,
            "retrieved_context": retrieved_context,
            "episodic_context": episodic_context,
            "debug_collector": debug_collector,
            "attachment_blocks": attachment_blocks,
            "person_id": person_id,
            "co_present_person_ids": co_present_person_ids,
            "briefing_context": briefing_context,
            "channel_context_block": channel_context_block,
            "memory_text": memory_text,
            "fetch_own_urls": fetch_own_urls,
        }
        if peer_memory_reasoning_level is not None:
            common_kwargs["peer_memory_reasoning_level"] = peer_memory_reasoning_level
        if peer_memory_context is not None:
            common_kwargs["peer_memory_context"] = peer_memory_context
        if standing_facts is not None:
            common_kwargs["standing_facts"] = standing_facts
        # A chat app turn from the person's own account: messages to them go
        # here next, unless they picked a channel (people.reach).
        if session.origin_channel:
            from openexecutive.people.reach import note_talked

            await asyncio.to_thread(note_talked, session)
        # The files a chat app read for this message (turn_files.offer): bound
        # for the turn and kept for the conversation, as a web upload is.
        offered = turn_files.take_offered()
        if offered:
            from openexecutive.orchestrator import conversation_files

            await asyncio.to_thread(conversation_files.keep, session.session_id, offered)
        stream = (
            self.stream_chat_with_committee(**common_kwargs)
            if committee_review
            else self.stream_chat(**common_kwargs)
        )
        result = ""
        rewritten = False
        refused: list[int] = []
        refusing = _REFUSED.set(refused)
        try:
            with turn_files.bind(offered or None) if offered else contextlib.nullcontext():
                async for chunk in stream:
                    if isinstance(chunk, str) and chunk != self._THINKING:
                        result += chunk
                    elif isinstance(chunk, dict) and chunk.get("type") == "rewrite":
                        # The next round writes the reply again.
                        result, rewritten = "", True
                    elif isinstance(chunk, dict) and chunk.get("type") in ("activity", "progress"):
                        if on_progress is not None:
                            on_progress(chunk)
                    elif cards is not None and isinstance(chunk, dict) and chunk.get("type") == "action_taken":
                        decision_id = chunk.get("decision_id")
                        if isinstance(decision_id, int) and not isinstance(decision_id, bool) and decision_id not in cards:
                            cards.append(decision_id)
        finally:
            _REFUSED.reset(refusing)
        if rewritten:
            # A retry that wrote nothing streams the first reply again
            # (_kept_if_unwritten).
            result = result.lstrip()
        if refused:
            # Nothing the refused round began ("You're") is sent. What the
            # earlier rounds said stays (what their tools did), then a plain
            # refusal.
            result = _with_refusal(result, refused[-1])
        # A card named by its number ("card 220") tells the reader nothing:
        # name it by its headline, linked to it on Home (alerts.card_refs).
        # Private cards only when the principal reads it alone.
        from openexecutive.alerts.card_refs import name_cards_in_reply

        return await asyncio.to_thread(
            name_cards_in_reply, result, session, person_id, co_present_person_ids,
        )
