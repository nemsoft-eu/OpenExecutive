"""Do what it says: a reply may not claim a change this turn didn't make.

The Executive sometimes answers "Done", "Removed it" or "I'll treat it as
closed" without calling the tool that would make it so, and the person only
finds out later, when the brief still lists the thing. So before a turn ends,
its last reply is checked against the tool calls the turn made:

1. A cheap word check (``CLAIM_RE``): a reply that claims no change ("Here are
   the numbers…") ends the turn as before, with no model call.
2. Otherwise one small forced-tool call on ``ROUTING_MODEL`` reads the
   person's message, the reply and the turn's tool calls (name, what each was
   given, and the start of each result) and says whether the reply claims a change this turn that
   none of those calls made. A claim about an earlier turn, a plan for later
   that the reply doesn't present as set up, or a change a call did make (or
   put up for approval) is not one.
3. When it finds one, the claim must appear word for word in the reply (code
   check), or it is ignored. The loop then gets one more round with
   ``nudge(claim)``: when the person asked for it, do it now with the right
   tool or say plainly it couldn't be done; when they didn't, offer it; in
   a whole new reply that replaces the first. Once a turn, so a
   loop can't form.

Never raises and never blocks a reply: a failure, a timeout or a missing
setting ends the turn as before. ``CLAIM_CHECK_ENABLED`` turns it off.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Words a reply uses to say something changed. Lexical on purpose: it only
# decides whether the model call runs, so a miss costs a check, not a reply.
CLAIM_RE = re.compile(
    r"\b(?:done|did it|removed|deleted|dropped|closed|cancel+ed|marked|"
    r"updated|changed|saved|noted|added|set (?:up|it|a)|scheduled|booked|"
    r"sent|moved|archived|cleared|fixed|taken care of|handled|treat(?:ed|ing)? "
    r"(?:it|this|that|the)|i['’]?ve|i have|i['’]?ll remind|will remind|reminder (?:is )?set|"
    r"won'?t (?:show|appear|include)|no longer|from now on|going forward|"
    r"(?:every|each)(?: other| \d+| two| few)? "
    r"(?:minute|hour|day|week|month|quarter|morning|evening|weekday|monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?|"
    r"hourly|daily|weekly|monthly|quarterly|keep(?:ing)? an eye|watching|"
    r"(?:i['’]?ll|i will) (?:flag|ping|let you know|tell you|schedule|watch|send)|"
    # Saying a message went to someone ("I just messaged her", "emailed
    # Pat", "let Kendall know") and anything the reply says it "just" or
    # "already" did.
    r"messaged|emailed|e-mailed|texted|pinged|dm['’]?(?:e)?d|told|let (?!me\b|us\b)\w+ know|"
    r"reached out|wrote (?:to|back)|replied|forwarded|notified|invited|shared|"
    r"posted|followed up|nudged|(?:i|we) (?:just|already|also|went ahead and) \w+)\b",
    re.IGNORECASE,
)

MAX_REPLY_CHARS = 3000
MAX_MESSAGE_CHARS = 2000
# Every call the turn made is shown, never only the last few: a save early
# in a long turn backs "I saved it" as much as one at the end. A long turn's
# lines get shorter instead, down to CALL_MIN_CHARS each.
CALLS_BUDGET_CHARS = 16000
CALL_MIN_CHARS = 160
CALL_RESULT_CHARS = 400
# What each call was asked to do, shown next to its name: "moved the total
# to row 10" is in what a sheet write was given, not in its one-line result.
CALL_INPUT_CHARS = 600
CLAIM_MAX_CHARS = 200
_MAX_TOKENS = 300
_TIMEOUT_S = 20.0

_SYSTEM = (
    "You check an assistant's reply against what it actually did. You get the "
    "person's message, the assistant's reply, and every tool call the "
    "assistant made while writing that reply, with what each call was given "
    "and the start of each result. What a call wrote (cells, rows, formulas, "
    "text) is a change it made.\n\n"
    "Report an unbacked claim when the reply says or implies that it made a "
    "change during this reply (removed, closed, saved, scheduled, sent, "
    "updated, cancelled, set a reminder, will stop doing something, will "
    "treat something as done) and no tool call in the list made that change "
    "or put it up for approval. Messaging, emailing, texting or telling "
    "someone is a change: only a call that sent that message to that person "
    "backs \"I messaged her\" or \"I let Pat know\". A step put up for approval "
    "backs \"it's waiting for your OK\", never that it went. A tool call that "
    "failed or was refused does not back a claim.\n\n"
    "A call backs only the kind of thing it made. A follow-up or reminder "
    "scheduled once does not back a claim that something is sent or run "
    "again on a timetable (every Friday, each morning); only a workflow "
    "saved switched on with a schedule does, and only the timetable that "
    "schedule says (a weekly one doesn't back \"every other Friday\"). A standing change (a fact kept, "
    "something left out of the brief) still backs \"from now on\". A workflow that was only drafted, or saved off or "
    "waiting for review, backs \"drafted\" or \"saved, turn it on to start\", "
    "never that it will go out. A fact or note kept in memory does not back "
    "a claim that something is being watched, or that the person will be "
    "told or flagged when something arrives; a call that set up a watch "
    "does (add_watchlist_entry, or propose_watch when its "
    "result says it was added), checking as often as that watch says. A "
    "watch left as a suggestion backs only \"suggested\". A promise that a one-time "
    "step will set up the next one is an unbacked claim.\n\n"
    "Not unbacked: a change one of the calls made; something it says was "
    "done earlier or by someone else; an offer or a question ('want me to "
    "remove it?'); advice; a plain answer to a question; saying it can't do "
    "something; a promise about how it will write or talk (shorter, more "
    "formal), which needs no tool.\n\n"
    "When there is one, copy the claim word for word from the reply (one "
    "short phrase). Record your verdict with the tool."
)

# A claim about work before this reply is about another turn, which this
# one's calls can't back or disprove: the judge says which it is.
WHEN_THIS_REPLY = "this reply"
WHEN_BEFORE = "before this reply"

_TOOL: dict[str, Any] = {
    "name": "record_claim_check",
    "description": "Record whether the reply claims a change no tool call made.",
    "input_schema": {
        "type": "object",
        "properties": {
            "unbacked": {
                "type": "boolean",
                "description": "True when the reply claims a change this turn that no listed call made.",
            },
            "claim": {
                "type": "string",
                "description": "The claim, copied word for word from the reply. Empty when unbacked is false.",
            },
            "when": {
                "type": "string",
                "enum": [WHEN_THIS_REPLY, WHEN_BEFORE],
                "description": (
                    "When the claim says the change happened: during this reply, or before it "
                    "(an earlier message, earlier today, yesterday). A time the change set "
                    "('moved it to this morning') or a time in what it names ('the notes from "
                    "yesterday's call') is not when it happened."
                ),
            },
        },
        "required": ["unbacked", "claim", "when"],
    },
}


def enabled() -> bool:
    try:
        from openexecutive.config import get_settings

        return bool(get_settings().claim_check_enabled)
    except Exception:
        return False


def may_claim(reply: str) -> bool:
    """Whether the reply uses a word that can claim a change."""
    return bool(reply.strip()) and CLAIM_RE.search(reply) is not None


def call_label(name: str, tool_input: Any) -> str:
    """``name`` with what the call was given, for ``render_calls``. A
    connected app's call is named by the tool it ran."""
    import json

    if name == "call_tool" and isinstance(tool_input, dict):
        name = str(tool_input.get("name") or name)
        tool_input = tool_input.get("arguments")
    try:
        given = json.dumps(tool_input, default=str, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        given = str(tool_input)
    given = " ".join(given.split())
    if len(given) > CALL_INPUT_CHARS:
        given = given[:CALL_INPUT_CHARS] + "…"
    return f"{name} {given}" if given not in ("", "null", "{}") else name


def _reads(label: str) -> bool:
    """Whether a call only read: its tool name starts with a read verb, by
    the same test the gate uses for connected tools. Anything else may have
    changed something and backs a claim."""
    from openexecutive.orchestrator.take_the_lead import mcp_tool_acts

    try:
        return not mcp_tool_acts(label.split(" ", 1)[0])
    except Exception:
        return False


def _fit(calls: list[tuple[str, str]], most: int) -> list[tuple[str, str]]:
    """At most ``most`` lines for a very long turn. Reads go first, from the
    middle out, so every call that may have changed something stays while
    those fit in ``most - 1`` lines; past that the middle ones go too, and a
    true claim about one of them can come back unbacked. One line says how
    many were left out."""
    room = most - 1
    keep = [i for i, (label, _) in enumerate(calls) if not _reads(label)]
    if len(keep) > room:
        keep = [*keep[: room // 2], *keep[len(keep) - (room - room // 2):]]
    else:
        changes = set(keep)
        reads = [i for i in range(len(calls)) if i not in changes]
        spare = room - len(keep)
        keep += [*reads[: spare // 2], *reads[len(reads) - (spare - spare // 2):]] if spare else []
    kept = sorted(set(keep))
    left_out = len(calls) - len(kept)
    gap = next((n for n, i in enumerate(kept) if i != n), len(kept))
    out = [calls[i] for i in kept]
    out.insert(gap, ("…", f"{left_out} more calls left out"))
    return out


def render_calls(calls: list[tuple[str, str]]) -> str:
    """Every call, one line each: what it was given and the start of its
    result, each line cut to its share of ``CALLS_BUDGET_CHARS``."""
    if not calls:
        return "(no tool calls)"
    most = CALLS_BUDGET_CHARS // CALL_MIN_CHARS
    if len(calls) > most:
        calls = _fit(calls, most)
    width = max(CALL_MIN_CHARS, CALLS_BUDGET_CHARS // len(calls))
    lines = []
    for name, result in calls:
        preview = " ".join(str(result or "").split())[:CALL_RESULT_CHARS]
        line = f"- {name}: {preview}"
        lines.append(line if len(line) <= width else line[: width - 1] + "…")
    return "\n".join(lines)


def _render_turn(message: str, reply: str, calls: list[tuple[str, str]]) -> str:
    return (
        f"<person_message>\n{message[-MAX_MESSAGE_CHARS:]}\n</person_message>\n\n"
        f"<assistant_reply>\n{reply[:MAX_REPLY_CHARS]}\n</assistant_reply>\n\n"
        f"<tool_calls>\n{render_calls(calls)}\n</tool_calls>"
    )


def _normalize(text: str) -> str:
    return " ".join(text.replace("’", "'").split()).lower()


def accept_claim(payload: dict[str, Any], reply: str) -> str | None:
    """The model's claim when it reported one, it is in the reply word for
    word (whitespace, case and curly quotes aside) and it doesn't place
    itself before this reply (``when``); None otherwise."""
    if payload.get("unbacked") is not True:
        return None
    claim = " ".join(str(payload.get("claim") or "").split())
    if not claim or len(claim) > CLAIM_MAX_CHARS:
        return None
    if _normalize(claim) not in _normalize(reply):
        return None
    if payload.get("when") == WHEN_BEFORE:
        return None
    return claim



async def _call_model(model: str, turn: str) -> dict[str, Any]:
    from openexecutive.audit.usage import log_model_usage
    from openexecutive.providers import get_provider

    response = await get_provider(model).messages_create(
        model=model,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": _TOOL["name"]},
        messages=[{"role": "user", "content": turn}],
    )
    log_model_usage(response, model=model, actor="claim_check")
    for block in response.content:
        if getattr(block, "type", "") == "tool_use" and getattr(block, "name", "") == _TOOL["name"]:
            return block.input if isinstance(block.input, dict) else {}
    return {}


async def unbacked_claim(message: str, reply: str, calls: list[tuple[str, str]]) -> str | None:
    """The reply's claim of a change no call in ``calls`` made, or None.
    Never raises."""
    if not enabled() or not may_claim(reply):
        return None
    try:
        from openexecutive.config import get_settings

        model = get_settings().routing_model
        payload = await asyncio.wait_for(
            _call_model(model, _render_turn(message, reply, calls)), timeout=_TIMEOUT_S
        )
    except Exception:
        logger.warning("claim_check: check failed, reply goes out as is", exc_info=True)
        return None
    return accept_claim(payload, reply)


def nudge(claim: str) -> str:
    """The user-turn note for the extra round (never a cached block). The
    round writes the reply again in full, correct, and it replaces the first
    wherever the person reads it: on a chat app or in an email the first is
    never sent, and the web and phone chats swap it out (``rewrite``)."""
    return (
        "[Check before you finish] Your reply says “" + claim + "”, but no "
        "tool call in this reply backs that. If a call already put it up for "
        "approval, don't make it again: say it's waiting for their OK. "
        "Otherwise, if the person asked for it, make it now with the right "
        "tool, or say plainly you couldn't; if they didn't, offer it instead "
        "(\"want me to message her?\"). Then write your whole reply again, start to finish, "
        "saying only what is true: it replaces the one above, which the person "
        "will never see. No note about a correction."
    )


def message_text(content: Any) -> str:
    """The text of a user message's content (a string, or text blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", "")) for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def last_person_message(messages: list[dict[str, Any]]) -> str:
    """The last user message that the person wrote (not a tool-results turn)."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            continue
        text = message_text(content)
        if text.strip():
            return text
    return ""
